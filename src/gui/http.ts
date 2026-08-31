import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { extname, join, resolve } from "node:path";
import { stringify as stringifyYaml } from "yaml";
import { canonicalizeDirectory, isPathInside } from "../allowlist.js";
import { parseConfigYaml, packageRoot, patchBackendModelYaml, patchBackendNicknameYaml, readConfigYaml, validateConfigYaml, writeConfigYaml } from "../config.js";
import { isGeminiOpenAiConfig, normalizeGeminiConfigModel } from "../providers/gemini.js";
import { extractBearerToken, tokensEqual } from "../gui-auth.js";
import type { ChatService } from "../chat/service.js";
import type { ChatSuggestedAction } from "../chat/types.js";
import type { Orchestrator } from "../orchestrator.js";
import { decodeLogoDataUrl, hasLogo, parseModelId, parseNickname, readLogo, removeLogo, saveLogo } from "../identity.js";
import { envNamesForBackend, isEnvVarName } from "../providers/keys.js";
import { DEFAULT_LLAMACPP_BASE, DEFAULT_OLLAMA_BASE, normalizeLoopbackOpenAiUrl } from "../local-servers/loopback.js";
import { llamaServerOnPath, ollamaOnPath, probeLlamaCpp, probeOllama } from "../local-servers/status.js";
import { startLlamaServer, startOllama, stopLocalServer } from "../local-servers/spawn.js";
import {
  DEFAULT_OLLAMA_BACKEND_ID,
  DEFAULT_OLLAMA_SPECIALIST_ID,
  assertLocalBackendPatch,
  ollamaSpecialistDescription,
  patchLocalOrchestratorYaml,
} from "../local-servers/upsert.js";
import { redactConfigValue, redactSecretText, restoreMaskedSecrets } from "../redact.js";
import { KNOWN_SECRET_NAMES, deleteSecrets, refreshRuntimeEnv, secretStatus, upsertSecrets } from "../secrets.js";
import { toRunView } from "../views.js";
import type { OrchestratedRun } from "../types.js";
import { handleTempAnalyzeApi, TEMP_ANALYZE_PATH } from "../temp-analyze-http.js";
import type { McpHttpHandler } from "@modelcontextprotocol/server";
import { tryHandleMcpRequest } from "../mcp/attach.js";
import type { McpAuth } from "../mcp/auth/index.js";
import { loopbackMcpUrl } from "../mcp/bind.js";
import { isMcpPath } from "../mcp/paths.js";
import { applyConfirmedUpdates, checkInstalledReleases, parseUpdateChoice } from "../update-apply.js";

const HOST = "127.0.0.1";
const MAX_BODY = 1_000_000;
const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".json": "application/json; charset=utf-8",
};

export interface GuiListen {
  host: typeof HOST;
  port: number;
  url: string;
  mcpUrl: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function responseAlive(res: ServerResponse): boolean {
  return !res.writableEnded && !res.destroyed && res.writable;
}

function send(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  if (!responseAlive(res)) return;
  try {
    if (res.headersSent) {
      res.end();
      return;
    }
    const payload = typeof body === "string" ? body : JSON.stringify(body);
    const type = typeof body === "string" && !extra["content-type"]
      ? "text/plain; charset=utf-8"
      : extra["content-type"] ?? "application/json; charset=utf-8";
    res.writeHead(status, {
      "content-type": type,
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
      "x-frame-options": "DENY",
      "referrer-policy": "no-referrer",
      "content-security-policy":
        "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
      ...extra,
    });
    res.end(payload);
  } catch {
    try {
      res.destroy();
    } catch {
      /* client already gone */
    }
  }
}

function sseAlive(res: ServerResponse): boolean {
  return responseAlive(res);
}

function sseWrite(client: ServerResponse, chunk: string, clients: Set<ServerResponse>): void {
  if (!sseAlive(client)) {
    clients.delete(client);
    return;
  }
  try {
    client.write(chunk);
  } catch {
    clients.delete(client);
    try {
      client.destroy();
    } catch {
      /* ignore */
    }
  }
}

function sseBroadcast(clients: Set<ServerResponse>, event: string, data: unknown): void {
  let payload: string;
  try {
    payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  } catch (error) {
    console.error("gui sse stringify failed", error instanceof Error ? error.message : error);
    return;
  }
  for (const client of [...clients]) sseWrite(client, payload, clients);
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolveBody, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY) {
        req.destroy();
        reject(new Error("Request body too large"));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolveBody(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

export function hostAllowed(hostHeader: string | undefined, port: number): boolean {
  if (!hostHeader) return false;
  const host = hostHeader.trim().toLowerCase();
  return (
    host === `${HOST}:${port}` ||
    host === `localhost:${port}` ||
    host === HOST ||
    host === "localhost"
  );
}

/** GUI /api Origin must match this server's port. Missing Origin is allowed. */
export function originAllowed(origin: string | undefined, port: number): boolean {
  if (!origin) return true;
  try {
    const url = new URL(origin);
    if (url.protocol !== "http:") return false;
    const hostOk = url.hostname === HOST || url.hostname === "localhost";
    const portOk = url.port === String(port) || (url.port === "" && port === 80);
    return hostOk && portOk;
  } catch {
    return false;
  }
}

/** Streamable HTTP /mcp: any loopback Origin (Late UI :5173 / sidecar :7430) or none. */
export function mcpOriginAllowed(origin: string | undefined): boolean {
  if (!origin) return true;
  try {
    const url = new URL(origin);
    if (url.protocol !== "http:") return false;
    return url.hostname === HOST || url.hostname === "localhost" || url.hostname === "::1";
  } catch {
    return false;
  }
}

function publicDir(): string {
  return join(packageRoot(), "gui", "public");
}

function safePublicFile(urlPath: string): string | undefined {
  const root = resolve(publicDir());
  const trimmed = urlPath === "/" ? "/index.html" : urlPath;
  const candidate = resolve(root, `.${trimmed}`);
  if (!isPathInside(candidate, root)) return undefined;
  if (!existsSync(candidate) || !statSync(candidate).isFile()) return undefined;
  return candidate;
}

export function startGuiServer(options: {
  orchestrator: Orchestrator;
  chat: ChatService;
  token: string;
  port: number;
  mcpHandler?: McpHttpHandler;
  mcpAuth?: McpAuth;
}): { server: ReturnType<typeof createServer>; listen: GuiListen } {
  const { orchestrator, chat, token, port, mcpHandler, mcpAuth } = options;
  const listen: GuiListen = { host: HOST, port, url: `http://${HOST}:${port}`, mcpUrl: loopbackMcpUrl(port) };
  const sseClients = new Set<ServerResponse>();

  const broadcastRun = (run: OrchestratedRun) => {
    sseBroadcast(sseClients, "run", toRunView(run, true));
  };
  const broadcastCatalog = () => {
    void orchestrator.catalog().then(
      (catalog) => sseBroadcast(sseClients, "catalog", catalog),
      (error) => console.error("gui catalog broadcast failed", error instanceof Error ? error.message : error),
    );
  };

  const broadcastLocalModels = () => {
    try {
      sseBroadcast(sseClients, "local-models", orchestrator.localModels.snapshot());
    } catch (error) {
      console.error("gui local-models broadcast failed", error instanceof Error ? error.message : error);
    }
  };
  const broadcastVllm = (status: unknown) => {
    sseBroadcast(sseClients, "vllm", status);
  };

  const broadcastChat = (thread: unknown) => {
    sseBroadcast(sseClients, "chat", thread);
  };
  const broadcastChats = (list: unknown) => {
    sseBroadcast(sseClients, "chats", list);
  };
  const broadcastHeartbeat = (payload: unknown) => {
    sseBroadcast(sseClients, "chat-heartbeat", payload);
  };

  orchestrator.events.on("run", broadcastRun);
  orchestrator.events.on("catalog", broadcastCatalog);
  orchestrator.events.on("local-models", broadcastLocalModels);
  orchestrator.events.on("vllm", broadcastVllm);
  orchestrator.events.on("chat", broadcastChat);
  orchestrator.events.on("chats", broadcastChats);
  orchestrator.events.on("chat-heartbeat", broadcastHeartbeat);

  const dropSse = (res: ServerResponse, ping?: ReturnType<typeof setInterval>): void => {
    if (ping) clearInterval(ping);
    sseClients.delete(res);
  };

  const server = createServer((req, res) => {
    req.on("error", () => {
      dropSse(res);
    });
    res.on("error", () => {
      dropSse(res);
    });
    void handle(req, res).catch((error) => {
      try {
        if (res.headersSent) {
          if (responseAlive(res)) res.end();
          return;
        }
        send(res, 500, { error: redactSecretText(error instanceof Error ? error.message : String(error)) });
      } catch {
        /* never take down the GUI */
      }
    });
  });

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const method = req.method ?? "GET";
    const host = req.headers.host;
    if (!hostAllowed(host, port)) {
      send(res, 403, { error: "Invalid Host header" });
      return;
    }

    const url = new URL(req.url ?? "/", `http://${HOST}:${port}`);
    const origin = typeof req.headers.origin === "string" ? req.headers.origin : undefined;
    if (mcpHandler && mcpAuth && isMcpPath(url.pathname)) {
      if (!mcpOriginAllowed(origin)) {
        send(res, 403, { error: "Origin not allowed" });
        return;
      }
      if (await tryHandleMcpRequest({ req, res, url, auth: mcpAuth, handler: mcpHandler })) {
        return;
      }
    } else if (!originAllowed(origin, port)) {
      send(res, 403, { error: "Origin not allowed" });
      return;
    }

    if (url.pathname === "/health" && method === "GET") {
      send(res, 200, { ok: true, bind: `${HOST}:${port}`, pid: process.pid });
      return;
    }

    if (method === "GET" && !url.pathname.startsWith("/api/")) {
      const file = safePublicFile(url.pathname);
      if (!file) {
        send(res, 404, { error: "Not found" });
        return;
      }
      const type = MIME[extname(file)] ?? "application/octet-stream";
      res.writeHead(200, {
        "content-type": type,
        "x-content-type-options": "nosniff",
        "x-frame-options": "DENY",
        "referrer-policy": "no-referrer",
        "content-security-policy":
          "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        "cache-control": extname(file) === ".html" ? "no-store" : "public, max-age=300",
      });
      const stream = createReadStream(file);
      stream.on("error", () => {
        if (!res.headersSent) send(res, 500, { error: "Failed to read file" });
        else if (responseAlive(res)) {
          try {
            res.end();
          } catch {
            /* ignore */
          }
        }
      });
      res.on("error", () => stream.destroy());
      stream.pipe(res);
      return;
    }

    const provided =
      extractBearerToken(
        typeof req.headers.authorization === "string" ? req.headers.authorization : undefined,
        url.searchParams.get("token") ??
          (typeof req.headers["x-orchestrator-token"] === "string" ? req.headers["x-orchestrator-token"] : null),
      ) ?? "";
    if (!tokensEqual(provided, token)) {
      send(res, 401, { error: "unauthorized" });
      return;
    }

    if (url.pathname === "/api/events" && method === "GET") {
      res.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-store",
        connection: "keep-alive",
        "x-accel-buffering": "no",
        "x-content-type-options": "nosniff",
        "x-frame-options": "DENY",
      });
      sseWrite(res, `event: hello\ndata: ${JSON.stringify({ ok: true })}\n\n`, sseClients);
      try {
        sseWrite(res, `event: catalog\ndata: ${JSON.stringify(await orchestrator.catalog())}\n\n`, sseClients);
        sseWrite(res, `event: local-models\ndata: ${JSON.stringify(orchestrator.localModels.snapshot())}\n\n`, sseClients);
        sseWrite(res, `event: chats\ndata: ${JSON.stringify(chat.list())}\n\n`, sseClients);
      } catch (error) {
        console.error("gui sse hello failed", error instanceof Error ? error.message : error);
      }
      sseClients.add(res);
      const ping = setInterval(() => {
        sseWrite(res, `event: ping\ndata: {}\n\n`, sseClients);
        if (!sseClients.has(res)) clearInterval(ping);
      }, 15_000);
      ping.unref?.();
      const onGone = (): void => dropSse(res, ping);
      req.on("close", onGone);
      res.on("close", onGone);
      return;
    }

    let body: unknown = undefined;
    if (method !== "GET" && method !== "HEAD") {
      const raw = await readBody(req);
      if (raw) {
        try {
          body = JSON.parse(raw);
        } catch {
          send(res, 400, { error: "Invalid JSON" });
          return;
        }
      } else {
        body = {};
      }
    }

    try {
      await routeApi(method, url, body, res);
    } catch (error) {
      send(res, 400, { error: redactSecretText(error instanceof Error ? error.message : String(error)) });
    }
  }

  async function routeApi(method: string, url: URL, body: unknown, res: ServerResponse): Promise<void> {
    const path = url.pathname;

    if (path === "/api/session" && method === "GET") {
      send(res, 200, { ok: true, bind: `${HOST}:${port}`, mcpUrl: listen.mcpUrl });
      return;
    }

    if (path === "/api/updates" && method === "GET") {
      try {
        send(res, 200, await checkInstalledReleases());
      } catch (error) {
        send(res, 500, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === "/api/updates/apply" && method === "POST") {
      const confirmed = isRecord(body) && (body.confirm === true || body.confirmed === true);
      if (!confirmed) {
        send(res, 400, {
          error: "Say yes first. I will not change files on your computer until you confirm.",
        });
        return;
      }
      const rawChoice = isRecord(body) ? (body.choice ?? body.which) : undefined;
      try {
        const which = parseUpdateChoice(rawChoice);
        send(res, 200, await applyConfirmedUpdates({ which, confirmed: true }));
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === TEMP_ANALYZE_PATH) {
      handleTempAnalyzeApi({
        method,
        pathname: path,
        body,
        res,
        allowlist: orchestrator.tempAnalyze,
      });
      return;
    }

    if (path === "/api/chats" && method === "GET") {
      send(res, 200, chat.list());
      return;
    }

    if (path === "/api/chats" && method === "POST") {
      const pin = isRecord(body) && typeof body.pin === "string" ? body.pin : "auto";
      send(res, 200, chat.create(pin));
      return;
    }

    const chatMatch = /^\/api\/chats\/([^/]+)$/.exec(path);
    if (chatMatch && method === "GET") {
      try {
        send(res, 200, chat.get(decodeURIComponent(chatMatch[1] ?? "")));
      } catch (error) {
        send(res, 404, { error: error instanceof Error ? error.message : "Unknown chat" });
      }
      return;
    }
    if (chatMatch && method === "DELETE") {
      const id = decodeURIComponent(chatMatch[1] ?? "");
      if (!chat.delete(id)) {
        send(res, 404, { error: "Unknown chat" });
        return;
      }
      send(res, 200, { ok: true, id });
      return;
    }

    const chatPinMatch = /^\/api\/chats\/([^/]+)\/pin$/.exec(path);
    if (chatPinMatch && method === "POST") {
      if (!isRecord(body) || typeof body.pin !== "string") {
        send(res, 400, { error: "pin string required" });
        return;
      }
      send(res, 200, chat.setPin(decodeURIComponent(chatPinMatch[1] ?? ""), body.pin));
      return;
    }

    const chatWorkspaceMatch = /^\/api\/chats\/([^/]+)\/workspace$/.exec(path);
    if (chatWorkspaceMatch && method === "POST") {
      if (!isRecord(body) || typeof body.path !== "string") {
        send(res, 400, { error: "path string required" });
        return;
      }
      try {
        const granted = orchestrator.allowlist.assertCwd(body.path);
        send(res, 200, chat.setWorkspaceDir(decodeURIComponent(chatWorkspaceMatch[1] ?? ""), granted));
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : "Invalid workspace path" });
      }
      return;
    }

    const chatMsgMatch = /^\/api\/chats\/([^/]+)\/messages$/.exec(path);
    if (chatMsgMatch && method === "POST") {
      if (!isRecord(body) || typeof body.message !== "string") {
        send(res, 400, { error: "message string required" });
        return;
      }
      const thread = await chat.send({
        threadId: decodeURIComponent(chatMsgMatch[1] ?? ""),
        message: body.message,
        pin: typeof body.pin === "string" ? body.pin : undefined,
        cwd: typeof body.cwd === "string" ? body.cwd : undefined,
        prUrl: typeof body.prUrl === "string" ? body.prUrl : undefined,
        repoUrl: typeof body.repoUrl === "string" ? body.repoUrl : undefined,
        branch: typeof body.branch === "string" ? body.branch : undefined,
        extraContext: typeof body.extraContext === "string" ? body.extraContext : undefined,
        rounds: typeof body.rounds === "number" ? body.rounds : undefined,
        wait: typeof body.wait === "boolean" ? body.wait : false,
      });
      send(res, 200, thread);
      return;
    }

    const chatApprovalMatch = /^\/api\/chats\/([^/]+)\/approval$/.exec(path);
    if (chatApprovalMatch && method === "POST") {
      if (!isRecord(body) || (body.decision !== "approve" && body.decision !== "reject")) {
        send(res, 400, { error: "decision must be approve or reject" });
        return;
      }
      const thread = await chat.resolveApproval({
        threadId: decodeURIComponent(chatApprovalMatch[1] ?? ""),
        decision: body.decision,
        comment: typeof body.comment === "string" ? body.comment : undefined,
      });
      send(res, 200, thread);
      return;
    }

    if (path === "/api/chat/actions" && method === "POST") {
      if (!isRecord(body) || typeof body.action !== "string") {
        send(res, 400, { error: "action string required" });
        return;
      }
      const result = await chat.runAction({
        threadId: typeof body.threadId === "string" ? body.threadId : undefined,
        action: body.action as ChatSuggestedAction["action"],
        payload: isRecord(body.payload) ? body.payload : undefined,
      });
      send(res, 200, result);
      return;
    }

    if (path === "/api/catalog" && method === "GET") {
      send(res, 200, await orchestrator.catalog());
      return;
    }

    if (path === "/api/runs" && method === "GET") {
      const limit = Number(url.searchParams.get("limit") ?? "50");
      send(res, 200, orchestrator.store.list(Number.isFinite(limit) ? limit : 50).map((run) => toRunView(run)));
      return;
    }

    const runMatch = /^\/api\/runs\/([^/]+)$/.exec(path);
    if (runMatch && method === "GET") {
      const run = orchestrator.store.get(decodeURIComponent(runMatch[1] ?? ""));
      if (!run) {
        send(res, 404, { error: "Unknown run" });
        return;
      }
      send(res, 200, toRunView(run, true));
      return;
    }

    if (path === "/api/dispatch" && method === "POST") {
      if (!isRecord(body)) {
        send(res, 400, { error: "JSON object required" });
        return;
      }
      const run = await orchestrator.dispatch({
        specialist: String(body.specialist ?? ""),
        task: String(body.task ?? ""),
        backend: typeof body.backend === "string" ? body.backend : undefined,
        cwd: typeof body.cwd === "string" ? body.cwd : undefined,
        model: typeof body.model === "string" ? body.model : undefined,
        wait: typeof body.wait === "boolean" ? body.wait : undefined,
        prUrl: typeof body.prUrl === "string" ? body.prUrl : undefined,
        repoUrl: typeof body.repoUrl === "string" ? body.repoUrl : undefined,
        branch: typeof body.branch === "string" ? body.branch : undefined,
        extraContext: typeof body.extraContext === "string" ? body.extraContext : undefined,
        cloudAutoCreatePr: typeof body.cloudAutoCreatePr === "boolean" ? body.cloudAutoCreatePr : undefined,
      });
      send(res, 200, toRunView(run, true));
      return;
    }

    if (path === "/api/follow-up" && method === "POST") {
      if (!isRecord(body)) {
        send(res, 400, { error: "JSON object required" });
        return;
      }
      const run = await orchestrator.followUp({
        runId: String(body.runId ?? body.run_id ?? ""),
        message: String(body.message ?? ""),
        wait: typeof body.wait === "boolean" ? body.wait : undefined,
      });
      send(res, 200, toRunView(run, true));
      return;
    }

    if (path === "/api/workflows" && method === "POST") {
      if (!isRecord(body)) {
        send(res, 400, { error: "JSON object required" });
        return;
      }
      const result = await orchestrator.runWorkflow({
        workflow: String(body.workflow ?? ""),
        task: String(body.task ?? ""),
        cwd: typeof body.cwd === "string" ? body.cwd : undefined,
        prUrl: typeof body.prUrl === "string" ? body.prUrl : undefined,
        repoUrl: typeof body.repoUrl === "string" ? body.repoUrl : undefined,
        branch: typeof body.branch === "string" ? body.branch : undefined,
        extraContext: typeof body.extraContext === "string" ? body.extraContext : undefined,
        stopOnError: typeof body.stopOnError === "boolean" ? body.stopOnError : undefined,
      });
      send(res, 200, {
        workflow: result.workflow,
        status: result.status,
        summary: result.summary,
        runs: result.runs.map((run) => toRunView(run)),
      });
      return;
    }

    if (path === "/api/config" && method === "GET") {
      const yaml = readConfigYaml(orchestrator.configPath);
      send(res, 200, {
        path: orchestrator.configPath,
        yaml,
        parsed: redactConfigValue(parseConfigYaml(yaml, false)),
      });
      return;
    }

    if (path === "/api/config/validate" && method === "POST") {
      if (!isRecord(body) || typeof body.yaml !== "string") {
        send(res, 400, { error: "yaml string required" });
        return;
      }
      const parsed = validateConfigYaml(body.yaml);
      send(res, 200, { ok: true, specialists: Object.keys(parsed.specialists), backends: Object.keys(parsed.backends) });
      return;
    }

    if (path === "/api/config" && method === "PUT") {
      if (!isRecord(body) || typeof body.yaml !== "string") {
        send(res, 400, { error: "yaml string required" });
        return;
      }
      const parsed = writeConfigYaml(body.yaml, orchestrator.configPath);
      orchestrator.reloadConfig(parsed);
      send(res, 200, { ok: true, path: orchestrator.configPath });
      return;
    }

    if (path === "/api/config" && method === "PATCH") {
      if (!isRecord(body)) {
        send(res, 400, { error: "JSON object required" });
        return;
      }
      const currentYaml = readConfigYaml(orchestrator.configPath);
      const currentParsed = parseConfigYaml(currentYaml, false);
      const incoming = body.config ?? body;
      const restored = restoreMaskedSecrets(currentParsed, incoming);
      const yaml = stringifyYaml(restored, { indent: 2, lineWidth: 0 });
      const parsed = writeConfigYaml(yaml, orchestrator.configPath);
      orchestrator.reloadConfig(parsed);
      send(res, 200, { ok: true, yaml, parsed: redactConfigValue(parseConfigYaml(yaml, false)) });
      return;
    }

    if (path === "/api/allowlist" && method === "GET") {
      send(res, 200, {
        allowedDirectories: orchestrator.allowlist.list(),
        defaultCwd: orchestrator.defaultCwd(),
        fileWrites: "cursor-local-only",
      });
      return;
    }

    if (path === "/api/allowlist" && method === "POST") {
      if (!isRecord(body) || typeof body.path !== "string") {
        send(res, 400, { error: "path string required" });
        return;
      }
      const dirPath = body.path;
      try {
        const allowedDirectories = orchestrator.allowlist.add(dirPath);
        const granted =
          allowedDirectories.find((d) => d === canonicalizeDirectory(dirPath)) ?? canonicalizeDirectory(dirPath);
        void orchestrator.catalog().then(
          (catalog) => orchestrator.events.emit("catalog", catalog),
          () => undefined,
        );
        send(res, 200, { allowedDirectories, defaultCwd: orchestrator.defaultCwd(), granted });
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : "Invalid path" });
      }
      return;
    }

    if (path === "/api/allowlist" && method === "DELETE") {
      const fromQuery = url.searchParams.get("path");
      const fromBody = isRecord(body) && typeof body.path === "string" ? body.path : undefined;
      const dir = fromBody ?? fromQuery;
      if (!dir) {
        send(res, 400, { error: "path string required" });
        return;
      }
      const allowedDirectories = orchestrator.allowlist.remove(dir);
      send(res, 200, { allowedDirectories, defaultCwd: orchestrator.defaultCwd() });
      return;
    }

    if (path === "/api/env/reload" && method === "POST") {
      refreshRuntimeEnv();
      send(res, 200, { ok: true, catalog: await orchestrator.catalog() });
      return;
    }

    if (path === "/api/secrets" && method === "GET") {
      send(res, 200, { secrets: secretStatus(secretNamesForConfig(orchestrator)) });
      return;
    }

    if (path === "/api/secrets" && method === "DELETE") {
      if (!isRecord(body)) {
        send(res, 400, { error: "JSON object required" });
        return;
      }
      const names: string[] = [];
      if (typeof body.name === "string") names.push(body.name);
      if (Array.isArray(body.names)) {
        for (const name of body.names) {
          if (typeof name === "string") names.push(name);
        }
      }
      if (!names.length) {
        send(res, 400, { error: "name string required" });
        return;
      }
      const allowed = new Set(secretNamesForConfig(orchestrator));
      for (const name of names) {
        if (!allowed.has(name)) {
          send(res, 400, { error: `Unknown secret "${name}". Use a backend env name such as GEMINI_API_KEY.` });
          return;
        }
      }
      const changed = deleteSecrets(names);
      send(res, 200, {
        ok: true,
        cleared: changed,
        secrets: secretStatus(secretNamesForConfig(orchestrator)),
        catalog: await orchestrator.catalog(),
      });
      return;
    }

    if (path === "/api/secrets" && (method === "PUT" || method === "POST")) {
      if (!isRecord(body)) {
        send(res, 400, { error: "JSON object required" });
        return;
      }
      const { updates, clears } = collectSecretMutations(body);
      const allowed = new Set(secretNamesForConfig(orchestrator));
      for (const name of [...Object.keys(updates), ...clears]) {
        if (!allowed.has(name)) {
          send(res, 400, { error: `Unknown secret "${name}". Use a backend env name such as GEMINI_API_KEY.` });
          return;
        }
      }
      const changed = [
        ...(clears.length ? deleteSecrets(clears) : []),
        ...(Object.keys(updates).length ? upsertSecrets(updates) : []),
      ];
      send(res, 200, {
        ok: true,
        updated: changed,
        secrets: secretStatus(secretNamesForConfig(orchestrator)),
        catalog: await orchestrator.catalog(),
      });
      return;
    }

    const backendPatch = /^\/api\/backends\/([^/]+)$/.exec(path);
    if (backendPatch && method === "PATCH") {
      if (!isRecord(body)) {
        send(res, 400, { error: "JSON object required" });
        return;
      }
      const id = decodeURIComponent(backendPatch[1] ?? "").trim();
      const currentYaml = readConfigYaml(orchestrator.configPath);
      const currentParsed = parseConfigYaml(currentYaml, false);
      if (!isRecord(currentParsed) || !isRecord(currentParsed.backends) || !isRecord(currentParsed.backends[id])) {
        send(res, 404, { error: `Unknown backend "${id}"` });
        return;
      }
      const hasModel = typeof body.model === "string";
      const hasNick = "nickname" in body;
      if (!hasModel && !hasNick) {
        send(res, 400, { error: "model or nickname required" });
        return;
      }
      let yaml = currentYaml;
      let nickname: string | undefined;
      if (hasNick) {
        nickname =
          body.nickname === null || body.nickname === ""
            ? undefined
            : parseNickname(body.nickname);
        yaml = patchBackendNicknameYaml(yaml, id, nickname);
      }
      let model: string | undefined;
      if (hasModel) {
        const existing = currentParsed.backends[id];
        const type = typeof existing.type === "string" ? existing.type : "";
        const baseUrl = typeof existing.baseUrl === "string" ? existing.baseUrl : undefined;
        const apiKeyEnv = typeof existing.apiKeyEnv === "string" ? existing.apiKeyEnv : undefined;
        model = parseModelId(body.model);
        if (isGeminiOpenAiConfig(id, { type, baseUrl, apiKeyEnv, model })) {
          try {
            model = normalizeGeminiConfigModel(model);
          } catch (error) {
            send(res, 400, { error: error instanceof Error ? error.message : String(error) });
            return;
          }
        }
        yaml = patchBackendModelYaml(yaml, id, model);
        if (type === "ollama" || type === "llamacpp") {
          assertLocalBackendPatch(yaml, {
            backendId: id,
            type,
            baseUrl:
              typeof baseUrl === "string" && baseUrl.trim()
                ? baseUrl
                : type === "ollama"
                  ? DEFAULT_OLLAMA_BASE
                  : DEFAULT_LLAMACPP_BASE,
            model,
          });
        }
      }
      const parsed = writeConfigYaml(yaml, orchestrator.configPath);
      orchestrator.reloadConfig(parsed);
      send(res, 200, {
        ok: true,
        id,
        ...(hasModel ? { model } : {}),
        ...(hasNick ? { nickname: nickname ?? "" } : {}),
        catalog: await orchestrator.catalog(),
      });
      return;
    }

    const logoMatch = /^\/api\/backends\/([^/]+)\/logo$/.exec(path);
    if (logoMatch) {
      const id = decodeURIComponent(logoMatch[1] ?? "").trim();
      if (method === "GET") {
        const logo = readLogo(id);
        if (!logo) {
          send(res, 404, { error: "No logo" });
          return;
        }
        res.writeHead(200, {
          "content-type": logo.mime,
          "cache-control": "no-store",
          "x-content-type-options": "nosniff",
          "x-frame-options": "DENY",
          "referrer-policy": "no-referrer",
          "content-security-policy":
            "default-src 'none'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'",
          "content-length": String(logo.bytes.length),
        });
        res.end(logo.bytes);
        return;
      }
      if (method === "DELETE") {
        removeLogo(id);
        send(res, 200, { ok: true, id, hasLogo: false, catalog: await orchestrator.catalog() });
        return;
      }
      if (method === "PUT" || method === "POST") {
        if (!isRecord(body) || typeof body.data !== "string") {
          send(res, 400, { error: "data URL required (PNG, JPEG, or WebP)" });
          return;
        }
        const decoded = decodeLogoDataUrl(body.data);
        saveLogo(id, decoded.buffer, decoded.mime);
        send(res, 200, { ok: true, id, hasLogo: hasLogo(id), catalog: await orchestrator.catalog() });
        return;
      }
    }

    if (path === "/api/backends" && method === "POST") {
      if (!isRecord(body) || typeof body.id !== "string") {
        send(res, 400, { error: "id string required" });
        return;
      }
      const id = body.id.trim();
      if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(id)) {
        send(res, 400, { error: "Backend id must match [a-zA-Z][a-zA-Z0-9_-]*" });
        return;
      }
      const type = typeof body.type === "string" ? body.type : "vllm";
      if (type !== "vllm" && type !== "openai" && type !== "ollama" && type !== "llamacpp") {
        send(res, 400, { error: "GUI add-backend supports type vllm, openai, ollama, or llamacpp" });
        return;
      }
      let model = parseModelId(typeof body.model === "string" ? body.model : "");
      const defaultBase =
        type === "vllm"
          ? "http://127.0.0.1:8000/v1"
          : type === "ollama"
            ? DEFAULT_OLLAMA_BASE
            : type === "llamacpp"
              ? DEFAULT_LLAMACPP_BASE
              : "https://api.openai.com/v1";
      let baseUrl =
        typeof body.baseUrl === "string" && body.baseUrl.trim() ? body.baseUrl.trim() : defaultBase;
      if (type === "vllm" || type === "ollama" || type === "llamacpp") {
        const label = type === "vllm" ? "vLLM" : type === "ollama" ? "Ollama" : "llama.cpp";
        try {
          baseUrl = normalizeLoopbackOpenAiUrl(baseUrl, label);
        } catch (error) {
          send(res, 400, { error: error instanceof Error ? error.message : String(error) });
          return;
        }
      }
      const apiKeyEnv =
        typeof body.apiKeyEnv === "string" && body.apiKeyEnv.trim()
          ? body.apiKeyEnv.trim()
          : type === "vllm"
            ? undefined
            : undefined;
      if (apiKeyEnv && !isEnvVarName(apiKeyEnv)) {
        send(res, 400, { error: "apiKeyEnv must be an environment variable name, not a secret value" });
        return;
      }
      const currentYaml = readConfigYaml(orchestrator.configPath);
      const currentParsed = parseConfigYaml(currentYaml, false);
      if (!isRecord(currentParsed)) {
        send(res, 400, { error: "Config root must be a mapping" });
        return;
      }
      if (isGeminiOpenAiConfig(id, { type, baseUrl, apiKeyEnv, model })) {
        try {
          model = normalizeGeminiConfigModel(model);
        } catch (error) {
          send(res, 400, { error: error instanceof Error ? error.message : String(error) });
          return;
        }
      }
      const backends = isRecord(currentParsed.backends) ? { ...currentParsed.backends } : {};
      const record: Record<string, unknown> = {
        type,
        baseUrl,
        model,
      };
      if (apiKeyEnv) record.apiKeyEnv = apiKeyEnv;
      if ((type === "vllm" || type === "ollama" || type === "llamacpp") && body.probe === false) {
        record.probe = false;
      }
      if (type === "ollama") record.apiKey = "ollama";
      if ("nickname" in body) {
        const nickname =
          body.nickname === null || body.nickname === ""
            ? undefined
            : parseNickname(body.nickname);
        if (nickname) record.nickname = nickname;
      }
      backends[id] = record;
      currentParsed.backends = backends;
      if (isRecord(body.specialist) && typeof body.specialist.id === "string") {
        const specId = body.specialist.id.trim();
        if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(specId)) {
          send(res, 400, { error: "Specialist id must match [a-zA-Z][a-zA-Z0-9_-]*" });
          return;
        }
        const specialists = isRecord(currentParsed.specialists) ? { ...currentParsed.specialists } : {};
        specialists[specId] = {
          description:
            typeof body.specialist.description === "string" && body.specialist.description.trim()
              ? body.specialist.description.trim()
              : `Text-only specialist on ${id}`,
          backend: id,
          fallback:
            typeof body.specialist.fallback === "string" && body.specialist.fallback.trim()
              ? body.specialist.fallback.trim()
              : "cursor-local",
        };
        currentParsed.specialists = specialists;
      }
      const yaml = stringifyYaml(currentParsed, { indent: 2, lineWidth: 0 });
      if (type === "ollama" || type === "llamacpp") {
        assertLocalBackendPatch(yaml, { backendId: id, type, baseUrl, model });
      }
      const parsed = writeConfigYaml(yaml, orchestrator.configPath);
      if (type === "ollama" || type === "llamacpp") {
        const written = parsed.backends[id];
        if (!written || written.type !== type) {
          throw new Error(`Refusing config write: backend "${id}" type must remain ${type}`);
        }
        const label = type === "ollama" ? "Ollama" : "llama.cpp";
        normalizeLoopbackOpenAiUrl(written.baseUrl ?? "", label);
      }
      orchestrator.reloadConfig(parsed);
      const keyName =
        typeof body.apiKeyEnv === "string" && isEnvVarName(body.apiKeyEnv.trim())
          ? body.apiKeyEnv.trim()
          : type === "vllm"
            ? "VLLM_API_KEY"
            : undefined;
      if (typeof body.apiKey === "string" && body.apiKey.trim() && keyName) {
        upsertSecrets({ [keyName]: body.apiKey });
      }
      send(res, 200, {
        ok: true,
        id,
        yaml,
        catalog: await orchestrator.catalog(),
      });
      return;
    }

    if (path === "/api/hardware" && method === "GET") {
      send(res, 200, orchestrator.localModels.listHardware());
      return;
    }

    if (path === "/api/local-models" && method === "GET") {
      send(res, 200, orchestrator.localModels.listModels());
      return;
    }

    if (path === "/api/local-models/recommend" && method === "GET") {
      send(res, 200, orchestrator.localModels.recommend());
      return;
    }

    if (path === "/api/local-models/download" && method === "POST") {
      if (!isRecord(body) || typeof body.modelId !== "string") {
        send(res, 400, { error: "modelId string required" });
        return;
      }
      refreshRuntimeEnv();
      const job = orchestrator.localModels.download({
        modelId: body.modelId,
        hfRepo: typeof body.hfRepo === "string" ? body.hfRepo : undefined,
        dest: typeof body.dest === "string" ? body.dest : undefined,
        dryRun: body.dryRun === true,
      });
      send(res, 200, job);
      return;
    }

    if (path === "/api/vllm" && method === "GET") {
      send(res, 200, orchestrator.localModels.vllmStatus());
      return;
    }

    if (path === "/api/vllm/start" && method === "POST") {
      if (!isRecord(body) || typeof body.modelId !== "string") {
        send(res, 400, { error: "modelId string required" });
        return;
      }
      const started = orchestrator.localModels.startVllmAsync({
        modelId: body.modelId,
        port: typeof body.port === "number" ? body.port : undefined,
        quantization: typeof body.quantization === "string" ? body.quantization : undefined,
        host: typeof body.host === "string" ? body.host : undefined,
        timeoutMs: typeof body.timeoutMs === "number" ? body.timeoutMs : undefined,
        image: typeof body.image === "string" ? body.image : undefined,
        runtime: body.runtime === "docker" || body.runtime === "host" ? body.runtime : undefined,
        replace: body.replace === true,
        useAllGpus: body.useAllGpus !== false && body.use_all_gpus !== false,
      });
      send(res, started.status === "starting" ? 202 : 200, {
        status: started.status,
        jobId: started.jobId,
        vllm: started.vllm,
      });
      return;
    }

    if (path === "/api/vllm/stop" && method === "POST") {
      const modelId = isRecord(body) && typeof body.modelId === "string" ? body.modelId : undefined;
      const backendId = isRecord(body) && typeof body.backendId === "string" ? body.backendId : undefined;
      const all = isRecord(body) && body.all === true;
      try {
        const status = orchestrator.localModels.stopVllm({ modelId, backendId, all });
        send(res, 200, { ok: true, vllm: status, catalog: await orchestrator.catalog() });
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === "/api/vllm/remove" && method === "POST") {
      const modelId = isRecord(body) && typeof body.modelId === "string" ? body.modelId : undefined;
      const backendId = isRecord(body) && typeof body.backendId === "string" ? body.backendId : undefined;
      if (!modelId && !backendId) {
        send(res, 400, { error: "modelId or backendId is required" });
        return;
      }
      try {
        const status = orchestrator.localModels.removeVllm({ modelId, backendId });
        send(res, 200, { ok: true, vllm: status, catalog: await orchestrator.catalog() });
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === "/api/local-models/delete" && method === "POST") {
      if (!isRecord(body) || typeof body.modelId !== "string") {
        send(res, 400, { error: "modelId string required" });
        return;
      }
      try {
        const result = orchestrator.localModels.deleteLocalModel({
          modelId: body.modelId,
          confirm: body.confirm === true,
        });
        send(res, 200, { ...result, catalog: await orchestrator.catalog() });
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === "/api/local-servers" && method === "GET") {
      try {
        send(res, 200, await localServersSnapshot(orchestrator));
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === "/api/local-servers/start" && method === "POST") {
      try {
        const kind = isRecord(body) && body.kind === "llamacpp" ? "llamacpp" : "ollama";
        if (kind === "llamacpp") {
          const modelPath = isRecord(body) && typeof body.modelPath === "string" ? body.modelPath.trim() : "";
          send(res, 200, await startLlamaServer(modelPath));
        } else {
          send(res, 200, await startOllama());
        }
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === "/api/local-servers/stop" && method === "POST") {
      try {
        const kind = isRecord(body) && body.kind === "llamacpp" ? "llamacpp" : "ollama";
        send(res, 200, stopLocalServer(kind));
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === "/api/ollama" && method === "GET") {
      try {
        const configured = Object.values(orchestrator.config.backends).find((b) => b.type === "ollama");
        const baseUrl = configured?.type === "ollama" ? configured.baseUrl : DEFAULT_OLLAMA_BASE;
        const apiKey = configured?.type === "ollama" ? configured.apiKey : undefined;
        send(res, 200, await probeOllama({ baseUrl, apiKey }));
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === "/api/ollama/connect" && method === "POST") {
      try {
        const configured = Object.values(orchestrator.config.backends).find((b) => b.type === "ollama");
        const requestedUrl =
          isRecord(body) && typeof body.baseUrl === "string" && body.baseUrl.trim()
            ? normalizeLoopbackOpenAiUrl(body.baseUrl.trim(), "Ollama")
            : configured?.type === "ollama"
              ? configured.baseUrl
              : DEFAULT_OLLAMA_BASE;
        const probe = await probeOllama({
          baseUrl: requestedUrl,
          apiKey: configured?.type === "ollama" ? configured.apiKey : undefined,
        });
        if (!probe.running) {
          send(res, 400, { error: probe.reason, ollama: probe });
          return;
        }
        const backendId =
          isRecord(body) && typeof body.id === "string" && body.id.trim()
            ? body.id.trim()
            : DEFAULT_OLLAMA_BACKEND_ID;
        if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(backendId)) {
          send(res, 400, { error: "Backend id must match [a-zA-Z][a-zA-Z0-9_-]*" });
          return;
        }
        const model = parseModelId(
          isRecord(body) && typeof body.model === "string" && body.model.trim()
            ? body.model
            : probe.models[0] ||
              (configured?.type === "ollama" ? configured.model : undefined) ||
              "llama3.1",
        );
        const yaml = readConfigYaml(orchestrator.configPath);
        const next = patchLocalOrchestratorYaml(yaml, {
          backendId,
          type: "ollama",
          baseUrl: probe.baseUrl,
          model,
          apiKey: "ollama",
          specialistId: DEFAULT_OLLAMA_SPECIALIST_ID,
          description: ollamaSpecialistDescription(),
        });
        assertLocalBackendPatch(next, {
          backendId,
          type: "ollama",
          baseUrl: probe.baseUrl,
          model,
        });
        const parsed = writeConfigYaml(next, orchestrator.configPath);
        const written = parsed.backends[backendId];
        if (!written || written.type !== "ollama") {
          throw new Error(`Refusing config write: backend "${backendId}" type must remain ollama`);
        }
        normalizeLoopbackOpenAiUrl(written.baseUrl ?? DEFAULT_OLLAMA_BASE, "Ollama");
        orchestrator.reloadConfig(parsed);
        send(res, 200, {
          ok: true,
          id: backendId,
          model,
          ollama: probe,
          catalog: await orchestrator.catalog(),
        });
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    if (path === "/api/llamacpp" && method === "GET") {
      try {
        const q = url.searchParams.get("baseUrl");
        const configured = Object.entries(orchestrator.config.backends).filter(([, b]) => b.type === "llamacpp");
        if (q) {
          send(res, 200, await probeLlamaCpp({ baseUrl: normalizeLoopbackOpenAiUrl(q, "llama.cpp") }));
          return;
        }
        if (configured.length === 0) {
          send(res, 200, { endpoints: [] });
          return;
        }
        const endpoints = await Promise.all(
          configured.map(([, b]) =>
            probeLlamaCpp({
              baseUrl: b.type === "llamacpp" ? b.baseUrl : DEFAULT_LLAMACPP_BASE,
              apiKey: b.type === "llamacpp" ? b.apiKey : undefined,
            }),
          ),
        );
        send(res, 200, { endpoints });
      } catch (error) {
        send(res, 400, { error: error instanceof Error ? error.message : String(error) });
      }
      return;
    }

    send(res, 404, { error: "Not found" });
  }

  server.on("close", () => {
    orchestrator.events.off("run", broadcastRun);
    orchestrator.events.off("catalog", broadcastCatalog);
    orchestrator.events.off("local-models", broadcastLocalModels);
    orchestrator.events.off("vllm", broadcastVllm);
    orchestrator.events.off("chat", broadcastChat);
    orchestrator.events.off("chats", broadcastChats);
    orchestrator.events.off("chat-heartbeat", broadcastHeartbeat);
    for (const client of sseClients) {
      try {
        client.end();
      } catch {
        /* ignore */
      }
    }
    sseClients.clear();
  });

  return { server, listen };
}

export function bindLoopbackOnly(requestedHost: string | undefined): typeof HOST {
  if (requestedHost && requestedHost !== HOST && requestedHost !== "localhost") {
    throw new Error(
      `Refusing to bind GUI to "${requestedHost}". Only ${HOST} is allowed so the control plane stays on this machine.`,
    );
  }
  return HOST;
}

async function localServersSnapshot(orchestrator: Orchestrator) {
  const ollamaCfg = Object.values(orchestrator.config.backends).find((b) => b.type === "ollama");
  const llamaCfgs = Object.entries(orchestrator.config.backends).filter(([, b]) => b.type === "llamacpp");
  const ollama = await probeOllama({
    baseUrl: ollamaCfg?.type === "ollama" ? ollamaCfg.baseUrl : DEFAULT_OLLAMA_BASE,
    apiKey: ollamaCfg?.type === "ollama" ? ollamaCfg.apiKey : undefined,
  });
  const llamacpp =
    llamaCfgs.length > 0
      ? await Promise.all(
          llamaCfgs.map(([id, b]) =>
            probeLlamaCpp({
              baseUrl: b.type === "llamacpp" ? b.baseUrl : DEFAULT_LLAMACPP_BASE,
              apiKey: b.type === "llamacpp" ? b.apiKey : undefined,
            }).then((status) => ({ id, ...status })),
          ),
        )
      : [];
  return {
    ollama,
    llamacpp,
    llamaServerBinary: llamaServerOnPath() ?? null,
    ollamaBinary: ollamaOnPath() ?? null,
  };
}

function collectSecretMutations(body: Record<string, unknown>): {
  updates: Record<string, string>;
  clears: string[];
} {
  const updates: Record<string, string> = {};
  const clears: string[] = [];
  const apply = (name: string, value: string) => {
    if (!value.trim()) clears.push(name);
    else updates[name] = value;
  };
  if (typeof body.name === "string" && typeof body.value === "string") {
    apply(body.name, body.value);
  } else if (isRecord(body.secrets)) {
    for (const [name, value] of Object.entries(body.secrets)) {
      if (typeof value === "string") apply(name, value);
    }
  } else {
    for (const [name, value] of Object.entries(body)) {
      if (typeof value === "string" && isEnvVarName(name)) apply(name, value);
    }
  }
  return { updates, clears };
}

function secretNamesForConfig(orchestrator: Orchestrator): string[] {
  const names = new Set<string>(KNOWN_SECRET_NAMES);
  for (const [id, config] of Object.entries(orchestrator.config.backends)) {
    for (const name of envNamesForBackend(id, config)) names.add(name);
  }
  return [...names];
}
