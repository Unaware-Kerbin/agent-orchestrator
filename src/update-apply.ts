/**
 * Live GitHub I/O + confirm-to-download for the loopback GUI.
 * Semver / "newer" meaning lives in update-sync.ts (same file Late copies).
 */
import { createWriteStream, existsSync, readFileSync, createReadStream } from "node:fs";
import { mkdir, rename, unlink } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { isPathInside } from "./allowlist.js";
import { packageRoot } from "./config.js";
import { ensureSecureDir } from "./platform.js";
import { stateDir } from "./state.js";
import {
  UNSIGNED_MAC_WIN,
  allowedCdnRedirectUrl,
  allowedDownloadHost,
  allowedFirstDownloadUrl,
  applyTargetFromChoice,
  checkBothReleases,
  contentDispositionFileName,
  digestHex,
  githubReleaseUrl,
  hostArch,
  hostPlatform,
  localVersionOverride,
  sha256HexMatches,
  type AppId,
  type GithubFetch,
  type UpdateCheck,
} from "./update-sync.js";

export const UPDATE_SYNC_SCHEMA = "update-sync/v1" as const;

export type UpdateChoice = "late" | "orchestrator" | "both";

export type GuiUpdateCheck = UpdateCheck & {
  schema: typeof UPDATE_SYNC_SCHEMA;
  cloudAiRequired: false;
  loopbackOnly: true;
  updateLate: boolean;
  updateOrchestrator: boolean;
  updateBoth: boolean;
  choices: UpdateChoice[];
  message: string;
};

export type UpdateApplyResult = {
  schema: typeof UPDATE_SYNC_SCHEMA;
  cloudAiRequired: false;
  choice: UpdateChoice;
  applied: AppId[];
  skipped: Array<{ id: AppId; reason: string }>;
  saved: Array<{ id: AppId; path: string; note: string }>;
  unsigned: boolean;
  message: string;
};

const UA = "agent-orchestrator-update-sync";
const MAX_ASSET_BYTES = 400 * 1024 * 1024;
const MAX_RELEASE_JSON = 1_500_000;

export function parseUpdateChoice(value: unknown): UpdateChoice {
  if (value === "late" || value === "orchestrator" || value === "both") return value;
  throw new Error("Pick this app, Late, or both.");
}

export function readPackageVersion(file: string): string | null {
  try {
    const raw = JSON.parse(readFileSync(file, "utf8")) as { version?: unknown };
    return typeof raw.version === "string" && raw.version.trim() ? raw.version.trim() : null;
  } catch {
    return null;
  }
}

export function findLateLocalVersion(opts?: { env?: NodeJS.ProcessEnv; packageRoot?: string }): string {
  const env = opts?.env ?? process.env;
  const checkout = env.LATE_CHECKOUT?.trim();
  if (checkout) {
    const fromCheckout =
      readPackageVersion(join(checkout, "apps", "desktop", "package.json")) ??
      readPackageVersion(join(checkout, "package.json"));
    if (fromCheckout) return localVersionOverride("late", fromCheckout, env);
  }
  const root = opts?.packageRoot ?? packageRoot();
  const parent = dirname(root);
  for (const name of ["Local_AI_Terminal_Emulator", "late"]) {
    const hit = readPackageVersion(join(parent, name, "apps", "desktop", "package.json"));
    if (hit) return localVersionOverride("late", hit, env);
  }
  return localVersionOverride("late", "0.0.0", env);
}

export function findOrchLocalVersion(opts?: { env?: NodeJS.ProcessEnv; packageRoot?: string }): string {
  const env = opts?.env ?? process.env;
  const fallback = readPackageVersion(join(opts?.packageRoot ?? packageRoot(), "package.json")) ?? "0.0.0";
  return localVersionOverride("orchestrator", fallback, env);
}

export function guiMessage(check: UpdateCheck): string {
  if (check.bothNewer) {
    return "GitHub has a newer Late and a newer Agent Orchestrator. You can update Late, Orchestrator, or both.";
  }
  if (check.lateOnly) return "GitHub has a newer Late. Orchestrator on your computer is already current.";
  if (check.orchestratorOnly) {
    return "GitHub has a newer Agent Orchestrator. Late on your computer is already current.";
  }
  if (check.late.error || check.orchestrator.error) {
    return "Checked GitHub. One repo did not answer — see the card. This check does not need Cloud AI.";
  }
  return "Late and Orchestrator on your computer match GitHub. You can still pick Late, Orchestrator, or both.";
}

export function toGuiCheck(check: UpdateCheck): GuiUpdateCheck {
  return {
    ...check,
    schema: UPDATE_SYNC_SCHEMA,
    cloudAiRequired: false,
    loopbackOnly: true,
    updateLate: check.late.newer,
    updateOrchestrator: check.orchestrator.newer,
    updateBoth: check.bothNewer,
    choices: ["orchestrator", "late", "both"],
    message: guiMessage(check),
  };
}

export async function defaultGithubFetch(url: string): Promise<import("./update-sync.js").FetchResult> {
  if (url !== githubReleaseUrl("late") && url !== githubReleaseUrl("orchestrator")) {
    return { ok: false, status: 0, error: "Update check only reads Unaware-Kerbin/late or Unaware-Kerbin/agent-orchestrator latest." };
  }
  if (!allowedDownloadHost(url)) {
    return { ok: false, status: 0, error: "Not a GitHub address." };
  }
  try {
    const response = await fetch(url, {
      method: "GET",
      redirect: "error",
      headers: {
        accept: "application/vnd.github+json",
        "user-agent": UA,
        "x-github-api-version": "2022-11-28",
        // Never Authorization / GITHUB_TOKEN / CURSOR_API_KEY / MCP tokens.
      },
    });
    if (response.status !== 200) {
      const hint =
        response.status === 404 ? "No GitHub release found (404)." : `GitHub returned HTTP ${response.status}.`;
      return { ok: false, status: response.status, error: hint };
    }
    const text = await response.text();
    if (text.length > MAX_RELEASE_JSON) {
      return { ok: false, status: 0, error: "GitHub reply was larger than I will read." };
    }
    return { ok: true, status: response.status, release: JSON.parse(text) as import("./update-sync.js").GithubRelease };
  } catch (error) {
    return { ok: false, status: 0, error: error instanceof Error ? error.message : String(error) };
  }
}

export async function checkUpdates(opts?: { fetchImpl?: GithubFetch }): Promise<GuiUpdateCheck> {
  const check = await checkBothReleases({
    lateLocal: findLateLocalVersion(),
    orchLocal: findOrchLocalVersion(),
    platform: hostPlatform(),
    arch: hostArch(),
    fetchImpl: opts?.fetchImpl ?? defaultGithubFetch,
  });
  return toGuiCheck(check);
}

export function assertGithubDownloadUrl(url: string, expectedFileName?: string): void {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error("Not a GitHub download address.");
  }
  if (parsed.protocol !== "https:") throw new Error("Download must be https.");
  if (parsed.username || parsed.password) {
    throw new Error("Download address must not include a login.");
  }
  const host = parsed.hostname.toLowerCase();
  if (!allowedFirstDownloadUrl(url, expectedFileName || lastPathSegment(parsed))) {
    if (host === "github.com" || host === "api.github.com" || host.endsWith(".githubusercontent.com")) {
      throw new Error("Download must be a Unaware-Kerbin late or agent-orchestrator release file.");
    }
    throw new Error("Download host is not GitHub.");
  }
}

function lastPathSegment(parsed: URL): string {
  const raw = parsed.pathname.split("/").filter(Boolean).pop() ?? "";
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

export function safeAssetFileName(name: string): string {
  const base = basename(String(name).replaceAll("\\", "/"));
  if (!base || base === "." || base === ".." || base.includes("\0")) {
    throw new Error("Release file name is not safe.");
  }
  return base;
}

/** Stem used as the unpack folder. `..tar.gz` must not become `..`. */
export function safeExtractStem(assetName: string): string {
  const file = safeAssetFileName(assetName);
  const stem = file.replace(/\.(tar\.gz|tgz|zip)$/i, "");
  if (!stem || stem === "." || stem === ".." || stem.includes("..")) {
    throw new Error("Release file name is not safe.");
  }
  return stem;
}

export function archiveHasUnsafeMember(listing: string): boolean {
  for (const line of listing.split(/\r?\n/)) {
    const n = line.trim().replaceAll("\\", "/");
    if (!n) continue;
    if (n.includes("\0")) return true;
    if (n.startsWith("/") || n.startsWith("//") || /^[a-zA-Z]:(\/|$)/.test(n)) return true;
    if (n === ".." || n.startsWith("../") || n.includes("/../") || n.endsWith("/..")) return true;
    if (n.split("/").some((part) => part === "..")) return true;
  }
  return false;
}

/** Resolved member must stay inside destDir (Windows drive letters and `..` included). */
export function archiveMemberEscapesDest(member: string, destDir: string): boolean {
  if (archiveHasUnsafeMember(member)) return true;
  const n = member.trim().replaceAll("\\", "/");
  if (!n) return true;
  return !isPathInside(resolve(destDir, n), resolve(destDir));
}

function listingEscapesDest(listing: string, destDir: string): boolean {
  return listing.split(/\r?\n/).some((line) => {
    const n = line.trim();
    return Boolean(n) && archiveMemberEscapesDest(n, destDir);
  });
}

/** Verbose `tar -tv` / `unzip -Z` lines. Do not use on name-only listings (`linux-…` is a normal path). */
export function archiveListingHasLink(listing: string): boolean {
  for (const line of listing.split(/\r?\n/)) {
    const n = line.trim();
    if (!n) continue;
    if (/^[lh][-rwxsStT]{9}\b/.test(n) || /(^|\s)->\s/.test(n)) return true;
  }
  return false;
}

export async function verifyFileDigest(filePath: string, pin: string): Promise<void> {
  const expected = digestHex(pin);
  if (!expected) throw new Error("Release file digest is not a SHA-256 pin. I will not keep it.");
  const hash = createHash("sha256");
  await pipeline(createReadStream(filePath), hash);
  const actual = hash.digest("hex");
  if (!sha256HexMatches(actual, expected)) {
    await unlink(filePath).catch(() => undefined);
    throw new Error("Downloaded file did not match the GitHub SHA-256 digest. I will not keep it.");
  }
}

export async function downloadAsset(
  url: string,
  destFile: string,
  fetchFn: typeof fetch = fetch,
  expected?: { fileName: string; digest?: string | null },
): Promise<void> {
  const fileName = expected?.fileName ?? safeAssetFileName(lastPathSegment(new URL(url)));
  const digestPinned = Boolean(digestHex(expected?.digest ?? null));
  const first = new URL(url);
  if (first.hostname.toLowerCase() !== "github.com") {
    throw new Error("Download must start at github.com/Unaware-Kerbin/…/releases/download/.");
  }
  assertGithubDownloadUrl(url, fileName);
  await mkdir(dirname(destFile), { recursive: true, mode: 0o700 });
  let current = url;
  for (let hop = 0; hop < 5; hop++) {
    if (hop === 0) {
      if (!allowedFirstDownloadUrl(current, fileName)) {
        throw new Error("Download must start at github.com/Unaware-Kerbin/…/releases/download/.");
      }
    } else if (!allowedCdnRedirectUrl(current, fileName, digestPinned)) {
      throw new Error("Refusing redirect off GitHub https.");
    }
    const response = await fetchFn(current, {
      method: "GET",
      redirect: "manual",
      headers: { "user-agent": UA, accept: "application/octet-stream" },
    });
    if (response.status >= 300 && response.status < 400) {
      const loc = response.headers.get("location");
      if (!loc) throw new Error("GitHub redirect had no next address.");
      const next = new URL(loc, current);
      if (next.protocol !== "https:") throw new Error("Refusing redirect off GitHub https.");
      current = next.href;
      continue;
    }
    if (!response.ok || !response.body) throw new Error(`Download failed (${response.status}).`);
    const headerName = contentDispositionFileName(response.headers.get("content-disposition") ?? "");
    if (headerName && headerName !== fileName) {
      throw new Error("Download was not the chosen GitHub release file.");
    }
    const tmp = `${destFile}.part`;
    const file = createWriteStream(tmp, { mode: 0o600 });
    let size = 0;
    const source = Readable.fromWeb(response.body as never);
    source.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_ASSET_BYTES) {
        source.destroy(new Error("That file is larger than I will save on your computer."));
      }
    });
    try {
      await pipeline(source, file);
      await rename(tmp, destFile);
    } catch (error) {
      file.destroy();
      await unlink(tmp).catch(() => undefined);
      throw error;
    }
    const pin = expected?.digest ?? null;
    if (!pin || !digestHex(pin)) {
      await unlink(destFile).catch(() => undefined);
      throw new Error("GitHub did not publish a SHA-256 digest. I will not keep it.");
    }
    await verifyFileDigest(destFile, pin);
    return;
  }
  throw new Error("Too many GitHub redirects.");
}

function extractOrchestratorArchive(archivePath: string, destDir: string): void {
  ensureSecureDir(destDir);
  if (archivePath.endsWith(".tar.gz") || archivePath.endsWith(".tgz")) {
    const listing = spawnSync("tar", ["-tzf", archivePath], { encoding: "utf8" });
    if (listing.status !== 0) throw new Error(listing.stderr?.trim() || "Could not read the archive.");
    if (listingEscapesDest(listing.stdout ?? "", destDir)) {
      throw new Error("Archive has an unsafe path. I will not unpack it.");
    }
    const verbose = spawnSync("tar", ["-tvzf", archivePath], { encoding: "utf8" });
    if (verbose.status !== 0) throw new Error(verbose.stderr?.trim() || "Could not inspect the archive.");
    if (archiveListingHasLink(verbose.stdout ?? "")) {
      throw new Error("Archive has an unsafe path. I will not unpack it.");
    }
    const result = spawnSync(
      "tar",
      ["--no-absolute-filenames", "--no-same-owner", "-xzf", archivePath, "-C", destDir],
      { encoding: "utf8" },
    );
    if (result.status !== 0) throw new Error(result.stderr?.trim() || "Could not unpack the archive.");
    return;
  }
  if (archivePath.endsWith(".zip")) {
    const listing = spawnSync("unzip", ["-Z1", archivePath], { encoding: "utf8" });
    if (listing.status !== 0) throw new Error(listing.stderr?.trim() || "Could not read the zip.");
    if (listingEscapesDest(listing.stdout ?? "", destDir)) {
      throw new Error("Archive has an unsafe path. I will not unpack it.");
    }
    const verbose = spawnSync("unzip", ["-Z", archivePath], { encoding: "utf8" });
    if (verbose.status !== 0) throw new Error(verbose.stderr?.trim() || "Could not inspect the zip.");
    if (archiveListingHasLink(verbose.stdout ?? "")) {
      throw new Error("Archive has an unsafe path. I will not unpack it.");
    }
    const result = spawnSync("unzip", ["-q", archivePath, "-d", destDir], { encoding: "utf8" });
    if (result.status !== 0) throw new Error(result.stderr?.trim() || "Could not unpack the zip.");
    return;
  }
  throw new Error("That file is not the portable archive.");
}

function portableInstall(root = packageRoot()): boolean {
  return existsSync(join(root, "runtime", "bin", "node")) || existsSync(join(root, "runtime", "node.exe"));
}

export const checkInstalledReleases = checkUpdates;

export async function applyConfirmedUpdates(input: {
  which: UpdateChoice;
  confirmed: boolean;
  fetchImpl?: GithubFetch;
  download?: (url: string, destFile: string) => Promise<void>;
}): Promise<UpdateApplyResult> {
  return applyUpdates({
    choice: input.which,
    confirm: input.confirmed,
    fetchImpl: input.fetchImpl,
    download: input.download,
  });
}

export async function applyUpdates(input: {
  choice: UpdateChoice;
  confirm: boolean;
  fetchImpl?: GithubFetch;
  download?: (url: string, destFile: string) => Promise<void>;
}): Promise<UpdateApplyResult> {
  if (input.confirm !== true) {
    throw new Error("Say yes first. I will not change files on your computer until you confirm.");
  }
  const check = await checkBothReleases({
    lateLocal: findLateLocalVersion(),
    orchLocal: findOrchLocalVersion(),
    platform: hostPlatform(),
    arch: hostArch(),
    fetchImpl: input.fetchImpl ?? defaultGithubFetch,
  });
  const destRoot = join(stateDir(), "updates");
  ensureSecureDir(destRoot);
  const applied: AppId[] = [];
  const skipped: Array<{ id: AppId; reason: string }> = [];
  const saved: Array<{ id: AppId; path: string; note: string }> = [];
  const wanted = input.choice === "both" ? (["late", "orchestrator"] as const) : ([input.choice] as const);
  const newerIds = new Set(applyTargetFromChoice(input.choice, check));

  for (const id of wanted) {
    const client = check[id];
    if (!newerIds.has(id)) {
      skipped.push({
        id,
        reason: client.error
          ? client.error
          : client.newer
            ? "GitHub has a newer tag but no matching file for this computer. I will not build from the tag."
            : "That copy on your computer is already current. Nothing was downloaded.",
      });
      continue;
    }
    if (!client.asset?.url || !client.asset.name) {
      skipped.push({
        id,
        reason: "GitHub has a newer tag but no matching file for this computer. I will not build from the tag. Open the release page.",
      });
      continue;
    }
    if (!digestHex(client.asset.digest)) {
      skipped.push({
        id,
        reason: "GitHub did not publish a SHA-256 digest. I will not download it.",
      });
      continue;
    }
    assertGithubDownloadUrl(client.asset.url, client.asset.name);
    const destFile = join(destRoot, safeAssetFileName(client.asset.name));
    try {
      if (input.download) await input.download(client.asset.url, destFile);
      else {
        await downloadAsset(client.asset.url, destFile, fetch, {
          fileName: client.asset.name,
          digest: client.asset.digest,
        });
      }
      const pin = client.asset.digest;
      if (!pin) throw new Error("GitHub did not publish a SHA-256 digest. I will not keep it.");
      await verifyFileDigest(destFile, pin);
    } catch (error) {
      await unlink(destFile).catch(() => undefined);
      skipped.push({
        id,
        reason: error instanceof Error ? error.message : String(error),
      });
      continue;
    }
    let path = destFile;
    let note =
      id === "late"
        ? "Saved the Late file on your computer. I did not run the installer or use root."
        : "Saved the portable archive on your computer. I did not overwrite this running copy.";
    if (id === "orchestrator" && /\.(tar\.gz|tgz|zip)$/i.test(client.asset.name)) {
      const extractTo = resolve(destRoot, safeExtractStem(client.asset.name));
      if (!isPathInside(extractTo, resolve(destRoot))) {
        throw new Error("Release file name is not safe.");
      }
      try {
        extractOrchestratorArchive(destFile, extractTo);
        path = extractTo;
        note = portableInstall()
          ? "Unpacked next to the download. Restart this GUI from that folder when you are ready."
          : "This copy is an npm folder. The new archive is unpacked beside it — I did not check out a git tag or overwrite your files. Restart from the unpacked folder.";
      } catch {
        note = "Saved the portable archive. I could not unpack it here — open that file yourself. I did not overwrite this running copy.";
      }
    }
    applied.push(id);
    saved.push({ id, path, note });
  }

  const who =
    input.choice === "both" ? "Late and Agent Orchestrator" : input.choice === "late" ? "Late" : "Agent Orchestrator";
  const unsigned = Boolean(check.late.unsigned || check.orchestrator.unsigned);
  const already = skipped.length > 0 && skipped.every((row) => /already current/i.test(row.reason));
  const message =
    applied.length === 0 && already
      ? "That copy on your computer is already current. Nothing was downloaded."
      : applied.length === 0
        ? skipped.map((row) => `${row.id === "late" ? "Late" : "Agent Orchestrator"}: ${row.reason}`).join(" ")
        : `Saved ${who} on your computer. ${unsigned ? UNSIGNED_MAC_WIN : "Linux files from GitHub are not code-signed either — you chose this."} Keys stayed here. Cloud AI was not used.`;

  return {
    schema: UPDATE_SYNC_SCHEMA,
    cloudAiRequired: false,
    choice: input.choice,
    applied,
    skipped,
    saved,
    unsigned,
    message,
  };
}
