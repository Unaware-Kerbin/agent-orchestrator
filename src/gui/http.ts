import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { extname, join, relative, resolve, sep } from "node:path";
import { stringify as stringifyYaml } from "yaml";
import { parseConfigYaml, packageRoot, patchBackendModelYaml, readConfigYaml, validateConfigYaml, writeConfigYaml } from "../config.js";
import { isGeminiOpenAiConfig, normalizeGeminiConfigModel } from "../providers/gemini.js";
import { extractBearerToken, tokensEqual } from "../gui-auth.js";
import type { ChatService } from "../chat/service.js";
import type { ChatSuggestedAction } from "../chat/types.js";
import type { Orchestrator } from "../orchestrator.js";
import { envNamesForBackend, isEnvVarName } from "../providers/keys.js";
import { redactConfigValue, restoreMaskedSecrets } from "../redact.js";
import { KNOWN_SECRET_NAMES, refreshRuntimeEnv, secretStatus, upsertSecrets } from "../secrets.js";
import { toRunView } from "../views.js";
import type { OrchestratedRun } from "../types.js";

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
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function send(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
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

function publicDir(): string {
  return join(packageRoot(), "gui", "public");
}

function safePublicFile(urlPath: string): string | undefined {
  const root = resolve(publicDir());
  const trimmed = urlPath === "/" ? "/index.html" : urlPath;
  const candidate = resolve(root, `.${trimmed}`);
  const rel = relative(root, candidate);
  if (rel.startsWith(`..${sep}`) || rel === ".." || rel.startsWith("..") || rel.includes(`..${sep}`)) {
    return undefined;
  }
  if (!existsSync(candidate) || !statSync(candidate).isFile()) return undefined;
  return candidate;
}

export function startGuiServer(options: {
  orchestrator: Orchestrator;
  chat: ChatService;
  token: string;
  port: number;
}): { server: ReturnType<typeof createServer>; listen: GuiListen } {
  const { orchestrator, chat, token, port } = options;
  const listen: GuiListen = { host: HOST, port, url: `http://${HOST}:${port}` };
  const sseClients = new Set<ServerResponse>();

  const broadcastRun = (run: OrchestratedRun) => {
    const payload = `event: run\ndata: ${JSON.stringify(toRunView(run, true))}\n\n`;
    for (const client of sseClients) client.write(payload);
  };
  const broadcastCatalog = () => {
    void orchestrator.catalog().then((catalog) => {
      const payload = `event: catalog\ndata: ${JSON.stringify(catalog)}\n\n`;
      for (const client of sseClients) client.write(payload);
    });
  };

  const broadcastLocalModels = () => {
    const payload = `event: local-models\ndata: ${JSON.stringify(orchestrator.localModels.snapshot())}\n\n`;
    for (const client of sseClients) client.write(payload);
  };
  const broadcastVllm = (status: unknown) => {
    const payload = `event: vllm\ndata: ${JSON.stringify(status)}\n\n`;
    for (const client of sseClients) client.write(payload);
  };

  const broadcastChat = (thread: unknown) => {
    const payload = `event: chat\ndata: ${JSON.stringify(thread)}\n\n`;
    for (const client of sseClients) client.write(payload);
  };
  const broadcastChats = (list: unknown) => {
    const payload = `event: chats\ndata: ${JSON.stringify(list)}\n\n`;
    for (const client of sseClients) client.write(payload);
  };
  const broadcastHeartbeat = (payload: unknown) => {
    const frame = `event: chat-heartbeat\ndata: ${JSON.stringify(payload)}\n\n`;
    for (const client of sseClients) client.write(frame);
  };

  orchestrator.events.on("run", broadcastRun);
  orchestrator.events.on("catalog", broadcastCatalog);
  orchestrator.events.on("local-models", broadcastLocalModels);
  orchestrator.events.on("vllm", broadcastVllm);
  orchestrator.events.on("chat", broadcastChat);
  orchestrator.events.on("chats", broadcastChats);
  orchestrator.events.on("chat-heartbeat", broadcastHeartbeat);

  const server = createServer(async (req, res) => {
    try {
      await handle(req, res);
    } catch (error) {
      if (res.headersSent) {
        res.end();
        return;
      }
      send(res, 500, { error: error instanceof Error ? error.message : String(error) });
    }
  });

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const method = req.method ?? "GET";
    const host = req.headers.host;
    if (!hostAllowed(host, port)) {
      send(res, 403, { error: "Invalid Host header" });
      return;
    }
    const origin = typeof req.headers.origin === "string" ? req.headers.origin : undefined;
    if (!originAllowed(origin, port)) {
      send(res, 403, { error: "Origin not allowed" });
      return;
    }

    const url = new URL(req.url ?? "/", `http://${HOST}:${port}`);

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
      createReadStream(file).pipe(res);
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
      res.write(`event: hello\ndata: ${JSON.stringify({ ok: true })}\n\n`);
      res.write(`event: catalog\ndata: ${JSON.stringify(await orchestrator.catalog())}\n\n`);
      res.write(`event: local-models\ndata: ${JSON.stringify(orchestrator.localModels.snapshot())}\n\n`);
      res.write(`event: chats\ndata: ${JSON.stringify(chat.list())}\n\n`);
      sseClients.add(res);
      const ping = setInterval(() => {
        res.write(`event: ping\ndata: {}\n\n`);
      }, 15_000);
      req.on("close", () => {
        clearInterval(ping);
        sseClients.delete(res);
      });
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
      send(res, 400, { error: error instanceof Error ? error.message : String(error) });
    }
  }

  async function routeApi(method: string, url: URL, body: unknown, res: ServerResponse): Promise<void> {
    const path = url.pathname;

    if (path === "/api/session" && method === "GET") {
      send(res, 200, { ok: true, bind: `${HOST}:${port}` });
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
      const allowedDirectories = orchestrator.allowlist.add(body.path);
      void orchestrator.catalog().then((catalog) => orchestrator.events.emit("catalog", catalog));
      send(res, 200, { allowedDirectories, defaultCwd: orchestrator.defaultCwd() });
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

    if (path === "/api/secrets" && (method === "PUT" || method === "POST")) {
      if (!isRecord(body)) {
        send(res, 400, { error: "JSON object required" });
        return;
      }
      const updates: Record<string, string> = {};
      if (typeof body.name === "string" && typeof body.value === "string") {
        updates[body.name] = body.value;
      } else if (isRecord(body.secrets)) {
        for (const [name, value] of Object.entries(body.secrets)) {
          if (typeof value === "string") updates[name] = value;
        }
      } else {
        for (const [name, value] of Object.entries(body)) {
          if (typeof value === "string" && isEnvVarName(name)) updates[name] = value;
        }
      }
      const allowed = new Set(secretNamesForConfig(orchestrator));
      for (const name of Object.keys(updates)) {
        if (!allowed.has(name)) {
          send(res, 400, { error: `Unknown secret "${name}". Use a backend env name such as GEMINI_API_KEY.` });
          return;
        }
      }
      const changed = upsertSecrets(updates);
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
      if (!isRecord(body) || typeof body.model !== "string") {
        send(res, 400, { error: "model string required" });
        return;
      }
      const id = decodeURIComponent(backendPatch[1] ?? "").trim();
      const currentYaml = readConfigYaml(orchestrator.configPath);
      const currentParsed = parseConfigYaml(currentYaml, false);
      if (!isRecord(currentParsed) || !isRecord(currentParsed.backends) || !isRecord(currentParsed.backends[id])) {
        send(res, 404, { error: `Unknown backend "${id}"` });
        return;
      }
      const existing = currentParsed.backends[id];
      const type = typeof existing.type === "string" ? existing.type : "";
      const baseUrl = typeof existing.baseUrl === "string" ? existing.baseUrl : undefined;
      const apiKeyEnv = typeof existing.apiKeyEnv === "string" ? existing.apiKeyEnv : undefined;
      let model = body.model.trim();
      if (isGeminiOpenAiConfig(id, { type, baseUrl, apiKeyEnv, model })) {
        try {
          model = normalizeGeminiConfigModel(model);
        } catch (error) {
          send(res, 400, { error: error instanceof Error ? error.message : String(error) });
          return;
        }
      }
      const yaml = patchBackendModelYaml(currentYaml, id, model);
      const parsed = writeConfigYaml(yaml, orchestrator.configPath);
      orchestrator.reloadConfig(parsed);
      send(res, 200, { ok: true, id, model, catalog: await orchestrator.catalog() });
      return;
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
      if (type !== "vllm" && type !== "openai") {
        send(res, 400, { error: "GUI add-backend supports type vllm or openai" });
        return;
      }
      let model = typeof body.model === "string" ? body.model.trim() : "";
      if (!model) {
        send(res, 400, { error: "model string required" });
        return;
      }
      const baseUrl =
        typeof body.baseUrl === "string" && body.baseUrl.trim()
          ? body.baseUrl.trim()
          : type === "vllm"
            ? "http://127.0.0.1:8000/v1"
            : "https://api.openai.com/v1";
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
      if (type === "vllm" && body.probe === false) record.probe = false;
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
      const parsed = writeConfigYaml(yaml, orchestrator.configPath);
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

    send(res, 404, { error: "Not found" });
  }

  server.on("close", () => {
    orchestrator.events.off("run", broadcastRun);
    orchestrator.events.off("catalog", broadcastCatalog);
    orchestrator.events.off("local-models", broadcastLocalModels);
    orchestrator.events.off("vllm", broadcastVllm);
    orchestrator.events.off("chat", broadcastChat);
    orchestrator.events.off("chats", broadcastChats);
    for (const client of sseClients) client.end();
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

function secretNamesForConfig(orchestrator: Orchestrator): string[] {
  const names = new Set<string>(KNOWN_SECRET_NAMES);
  for (const [id, config] of Object.entries(orchestrator.config.backends)) {
    for (const name of envNamesForBackend(id, config)) names.add(name);
  }
  return [...names];
}
