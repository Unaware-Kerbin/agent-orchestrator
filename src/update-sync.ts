/** Shared GitHub release check. Late copies this file to `apps/desktop/src/lib/updateSync.ts`. Keep the JSON shape in sync. */

export type AppId = "late" | "orchestrator";
export type HostPlatform = "linux" | "darwin" | "win32";
export type HostArch = "x64" | "arm64";
export type LatePrefer = "appimage" | "deb" | "auto";

export const UPDATE_REPOS = {
  late: { owner: "Unaware-Kerbin", repo: "late", label: "Late" },
  orchestrator: { owner: "Unaware-Kerbin", repo: "agent-orchestrator", label: "Orchestrator" },
} as const;

export const UNSIGNED_MAC_WIN =
  "macOS and Windows installers are unsigned (no Apple notarization or Windows Authenticode). On a Mac, right-click Late → Open the first time. On Windows, if SmartScreen appears, More info → Run anyway.";

export interface GithubAsset {
  name: string;
  browser_download_url: string;
  size?: number;
  digest?: string;
  sha256?: string;
}

export interface GithubRelease {
  tag_name?: unknown;
  html_url?: unknown;
  name?: unknown;
  assets?: unknown;
}

export interface ChosenAsset {
  name: string;
  url: string;
  size: number;
  /** Pinned `sha256:` hex from the GitHub release API, or null if GitHub omitted it. */
  digest: string | null;
}

export interface AppUpdateStatus {
  id: AppId;
  label: string;
  localVersion: string;
  remoteVersion: string | null;
  remoteTag: string | null;
  releaseUrl: string | null;
  newer: boolean;
  asset: ChosenAsset | null;
  error: string | null;
  unsigned: boolean;
}

export interface UpdateCheck {
  late: AppUpdateStatus;
  orchestrator: AppUpdateStatus;
  anyNewer: boolean;
  bothNewer: boolean;
  lateOnly: boolean;
  orchestratorOnly: boolean;
  unsignedNote: string | null;
}

export interface FetchResult {
  ok: boolean;
  status: number;
  release?: GithubRelease;
  error?: string;
}

export type GithubFetch = (url: string) => Promise<FetchResult>;

export function stripTag(tag: string): string {
  return tag.trim().replace(/^v/i, "");
}

export function parseSemver(tag: string): { major: number; minor: number; patch: number } | null {
  const core = stripTag(tag).split(/[-+]/)[0] ?? "";
  const m = /^(\d+)\.(\d+)\.(\d+)$/.exec(core);
  if (!m) return null;
  return { major: Number(m[1]), minor: Number(m[2]), patch: Number(m[3]) };
}

/** Negative if a < b, 0 if equal, positive if a > b. Null if either tag is not semver. */
export function compareSemver(a: string, b: string): number | null {
  const left = parseSemver(a);
  const right = parseSemver(b);
  if (!left || !right) return null;
  if (left.major !== right.major) return left.major - right.major;
  if (left.minor !== right.minor) return left.minor - right.minor;
  return left.patch - right.patch;
}

export function isNewerThan(remote: string, local: string): boolean {
  const localP = parseSemver(local);
  if (!localP || localP.major >= 20) return false;
  const n = compareSemver(remote, local);
  return n !== null && n > 0;
}

/** Electron reports 28–44+ when Late is unpackaged. Late product versions are 0.x. */
export function isImplausibleProductVersion(version: string): boolean {
  const p = parseSemver(version);
  return !p || p.major >= 20;
}

export function usableProductVersion(version: string | null | undefined): string | null {
  if (typeof version !== "string" || !version.trim()) return null;
  const t = stripTag(version.trim());
  if (isImplausibleProductVersion(t)) return null;
  return t;
}

export function hostPlatform(value = process.platform): HostPlatform {
  if (value === "darwin" || value === "win32") return value;
  return "linux";
}

export function hostArch(value = process.arch): HostArch {
  return value === "arm64" ? "arm64" : "x64";
}

const SHA256_HEX = /^[0-9a-f]{64}$/;
const GITHUB_CDN_PATH = /^\/github-production-release-asset(?:-[0-9a-f]+)?\//i;

/** SHA-256 hex, or null if GitHub omitted digest/sha256. Throws if a digest field is present but not SHA-256. */
export function parseAssetDigest(digest: unknown, sha256?: unknown): string | null {
  const fromField = (value: unknown): string | null | undefined => {
    if (value == null || value === "") return null;
    if (typeof value !== "string") return undefined;
    const trimmed = value.trim();
    if (!trimmed) return null;
    const m = /^(?:sha256:)?([0-9a-fA-F]{64})$/.exec(trimmed);
    const captured = m?.[1];
    if (!captured) return undefined;
    const hex = captured.toLowerCase();
    return SHA256_HEX.test(hex) ? hex : undefined;
  };
  const primary = fromField(digest);
  if (primary) return primary;
  if (primary === undefined) throw new Error("Release file digest is not a SHA-256 pin.");
  const secondary = fromField(sha256);
  if (secondary) return secondary;
  if (secondary === undefined) throw new Error("Release file digest is not a SHA-256 pin.");
  return null;
}

export function pinDigestString(hex: string | null): string | null {
  return hex ? `sha256:${hex}` : null;
}

export function digestHex(pin: string | null | undefined): string | null {
  if (!pin) return null;
  const m = /^sha256:([0-9a-f]{64})$/.exec(pin);
  return m?.[1] ?? null;
}

export function sha256HexMatches(actualHex: string, expectedHex: string): boolean {
  if (!SHA256_HEX.test(actualHex) || !SHA256_HEX.test(expectedHex)) return false;
  if (actualHex.length !== expectedHex.length) return false;
  let diff = 0;
  for (let i = 0; i < actualHex.length; i++) {
    diff |= actualHex.charCodeAt(i) ^ expectedHex.charCodeAt(i);
  }
  return diff === 0;
}

export function contentDispositionFileName(value: string): string | null {
  const text = String(value);
  const star = /filename\*\s*=\s*(?:UTF-8''|utf-8'')([^;]+)/i.exec(text);
  let raw: string | null = null;
  if (star) {
    try {
      raw = decodeURIComponent((star[1] ?? "").trim().replace(/^"+|"+$/g, ""));
    } catch {
      raw = null;
    }
  } else {
    const quoted = /filename\s*=\s*"([^"]+)"/i.exec(text);
    const bare = /filename\s*=\s*([^;]+)/i.exec(text);
    raw = (quoted?.[1] ?? bare?.[1] ?? "").trim() || null;
  }
  if (!raw) return null;
  const base = raw.replaceAll("\\", "/").split("/").pop() ?? "";
  if (!base || base === "." || base === ".." || base.includes("\0")) return null;
  return base;
}

function pathSegments(pathname: string): string[] {
  return pathname.split("/").filter((s) => s !== "");
}

export function githubAssetDownloadUrl(app: AppId, tag: string, fileName: string): string | null {
  const { owner, repo } = UPDATE_REPOS[app];
  const tagSeg = tag.trim();
  const file = fileName.trim();
  if (!tagSeg || tagSeg === "." || tagSeg === ".." || tagSeg.includes("/") || tagSeg.includes("\\") || tagSeg.includes("\0") || tagSeg.includes("..")) {
    return null;
  }
  if (!file || file === "." || file === ".." || file.includes("/") || file.includes("\\") || file.includes("\0")) return null;
  return `https://github.com/${owner}/${repo}/releases/download/${tagSeg}/${file}`;
}

export function urlsPointAtSameGithubDownload(apiUrl: string, official: string): boolean {
  try {
    const a = new URL(apiUrl);
    const b = new URL(official);
    if (a.protocol !== "https:" || b.protocol !== "https:") return false;
    if (a.username || a.password || b.username || b.password) return false;
    if (a.hostname.toLowerCase() !== "github.com" || b.hostname.toLowerCase() !== "github.com") return false;
    return decodeURIComponent(a.pathname) === decodeURIComponent(b.pathname);
  } catch {
    return false;
  }
}

function cdnDeclaredFileName(u: URL): string | null {
  for (const key of ["response-content-disposition", "rscd", "filename"]) {
    const v = u.searchParams.get(key);
    if (!v) continue;
    const n = key === "filename" ? contentDispositionFileName(`filename=${v}`) ?? v.replaceAll("\\", "/").split("/").pop() ?? null : contentDispositionFileName(v);
    if (n) return n;
  }
  return null;
}

function asAssets(raw: unknown): GithubAsset[] {
  if (!Array.isArray(raw)) return [];
  const out: GithubAsset[] = [];
  for (const row of raw) {
    if (!row || typeof row !== "object") continue;
    const rec = row as Record<string, unknown>;
    if (typeof rec.name !== "string" || typeof rec.browser_download_url !== "string") continue;
    let digest: string | undefined;
    try {
      const hex = parseAssetDigest(rec.digest, rec.sha256);
      digest = pinDigestString(hex) ?? undefined;
    } catch {
      continue;
    }
    out.push({
      name: rec.name,
      browser_download_url: rec.browser_download_url,
      size: typeof rec.size === "number" ? rec.size : undefined,
      digest,
    });
  }
  return out;
}

function matchesOs(name: string, platform: HostPlatform): boolean {
  const n = name.toLowerCase();
  if (platform === "linux") return n.includes("linux");
  if (platform === "darwin") return n.includes("mac") || n.includes("darwin");
  return n.includes("win");
}

function matchesArch(name: string, arch: HostArch): boolean {
  const n = name.toLowerCase();
  if (arch === "arm64") return /arm64|aarch64/.test(n);
  return /(?:x64|x86_64|amd64)/.test(n) && !/arm64|aarch64/.test(n);
}

function lateExtScore(name: string, prefer: LatePrefer, platform: HostPlatform): number {
  const n = name.toLowerCase();
  if (platform === "linux") {
    if (prefer === "appimage") return n.endsWith(".appimage") ? 4 : n.endsWith(".deb") ? 1 : 0;
    if (prefer === "deb") return n.endsWith(".deb") ? 4 : n.endsWith(".appimage") ? 1 : 0;
    return n.endsWith(".appimage") ? 3 : n.endsWith(".deb") ? 2 : 0;
  }
  if (platform === "darwin") return n.endsWith(".dmg") ? 4 : n.endsWith(".zip") ? 2 : 0;
  return n.endsWith(".exe") ? 4 : n.endsWith(".msi") ? 2 : 0;
}

function orchExtScore(name: string, platform: HostPlatform): number {
  const n = name.toLowerCase();
  if (platform === "win32") return n.endsWith(".zip") ? 4 : 0;
  return n.endsWith(".tar.gz") || n.endsWith(".tgz") ? 4 : n.endsWith(".zip") ? 1 : 0;
}

export function pickReleaseAsset(opts: {
  app: AppId;
  assets: GithubAsset[];
  platform: HostPlatform;
  arch: HostArch;
  prefer?: LatePrefer;
}): GithubAsset | null {
  const prefer = opts.prefer ?? "auto";
  let best: GithubAsset | null = null;
  let bestScore = 0;
  for (const asset of opts.assets) {
    const name = asset.name;
    if (!matchesOs(name, opts.platform) || !matchesArch(name, opts.arch)) continue;
    const ext =
      opts.app === "late"
        ? lateExtScore(name, prefer, opts.platform)
        : orchExtScore(name, opts.platform);
    if (ext <= 0) continue;
    const score = 10 + ext;
    if (score > bestScore) {
      best = asset;
      bestScore = score;
    }
  }
  return best;
}

export function localVersionOverride(app: AppId, fallback: string, env: NodeJS.ProcessEnv = process.env): string {
  const key = app === "late" ? "UPDATE_SYNC_LATE_LOCAL" : "UPDATE_SYNC_ORCH_LOCAL";
  const raw = env[key];
  if (typeof raw === "string" && raw.trim()) {
    const ok = usableProductVersion(raw);
    if (ok) return ok;
  }
  return usableProductVersion(fallback) ?? "";
}

export function buildAppStatus(opts: {
  id: AppId;
  localVersion: string;
  release: GithubRelease | null;
  error: string | null;
  platform: HostPlatform;
  arch: HostArch;
  prefer?: LatePrefer;
}): AppUpdateStatus {
  const meta = UPDATE_REPOS[opts.id];
  const unsigned = opts.platform === "darwin" || opts.platform === "win32";
  const tag = typeof opts.release?.tag_name === "string" ? opts.release.tag_name : null;
  const remoteVersion = tag ? stripTag(tag) : null;
  const parsed = tag ? parseSemver(tag) : null;
  let error = opts.error;
  if (!error && tag && !parsed) error = `Malformed release tag "${tag}"`;
  const localUsable = usableProductVersion(opts.localVersion);
  if (!error && !localUsable) {
    error =
      "Could not read this app's version on your computer. Electron's version is not Late — I will not skip or force an update from that.";
  }
  const assets = asAssets(opts.release?.assets);
  const picked =
    parsed && remoteVersion
      ? pickReleaseAsset({
          app: opts.id,
          assets,
          platform: opts.platform,
          arch: opts.arch,
          prefer: opts.prefer,
        })
      : null;
  const newer = Boolean(parsed && remoteVersion && localUsable && isNewerThan(remoteVersion, localUsable));
  let asset: ChosenAsset | null = null;
  if (picked && tag) {
    const official = githubAssetDownloadUrl(opts.id, tag, picked.name);
    if (official && urlsPointAtSameGithubDownload(picked.browser_download_url, official)) {
      asset = {
        name: picked.name,
        url: official,
        size: picked.size ?? 0,
        digest: picked.digest ?? null,
      };
    }
  }
  return {
    id: opts.id,
    label: meta.label,
    localVersion: localUsable ?? (stripTag(opts.localVersion) || opts.localVersion),
    remoteVersion,
    remoteTag: tag,
    releaseUrl: typeof opts.release?.html_url === "string" ? opts.release.html_url : null,
    newer,
    asset,
    error,
    unsigned,
  };
}

export function buildUpdateCheck(late: AppUpdateStatus, orchestrator: AppUpdateStatus): UpdateCheck {
  const unsignedNote = late.unsigned || orchestrator.unsigned ? UNSIGNED_MAC_WIN : null;
  return {
    late,
    orchestrator,
    anyNewer: late.newer || orchestrator.newer,
    bothNewer: late.newer && orchestrator.newer,
    lateOnly: late.newer && !orchestrator.newer,
    orchestratorOnly: orchestrator.newer && !late.newer,
    unsignedNote,
  };
}

export function githubReleaseUrl(app: AppId): string {
  const { owner, repo } = UPDATE_REPOS[app];
  return `https://api.github.com/repos/${owner}/${repo}/releases/latest`;
}

function httpsNoUserinfo(u: URL): boolean {
  if (u.protocol !== "https:") return false;
  if (u.username || u.password) return false;
  if (u.port && u.port !== "443") return false;
  return true;
}

function githubDownloadSegments(u: URL): { repo: "late" | "agent-orchestrator"; tag: string; file: string } | null {
  if (u.hostname.toLowerCase() !== "github.com") return null;
  const parts = pathSegments(u.pathname).map((s) => {
    try {
      return decodeURIComponent(s);
    } catch {
      return s;
    }
  });
  if (parts.length !== 6) return null;
  if (parts[0] !== "Unaware-Kerbin") return null;
  if (parts[1] !== "late" && parts[1] !== "agent-orchestrator") return null;
  if (parts[2] !== "releases" || parts[3] !== "download") return null;
  const tag = parts[4];
  const file = parts[5];
  if (!tag || tag === "." || tag === ".." || tag.includes("..")) return null;
  if (!file || file === "." || file === ".." || file.includes("\0")) return null;
  return { repo: parts[1], tag, file };
}

function isGithubReleaseCdn(u: URL): boolean {
  const host = u.hostname.toLowerCase();
  if (host !== "objects.githubusercontent.com" && host !== "release-assets.githubusercontent.com") return false;
  return GITHUB_CDN_PATH.test(u.pathname);
}

/** API latest JSON or a github.com Unaware-Kerbin release file. CDN hops need expectedFileName (or digestPinned). */
export function allowedDownloadHost(
  url: string,
  expectedFileName?: string,
  opts?: { digestPinned?: boolean; hop?: "first" | "cdn" | "any" },
): boolean {
  try {
    const u = new URL(url);
    if (!httpsNoUserinfo(u)) return false;
    const host = u.hostname.toLowerCase();
    const hop = opts?.hop ?? "any";
    if (host === "api.github.com") {
      if (hop === "cdn" || hop === "first") return false;
      return /^\/repos\/Unaware-Kerbin\/(late|agent-orchestrator)\/releases\/latest$/.test(u.pathname);
    }
    if (host === "github.com") {
      if (hop === "cdn") return false;
      const segs = githubDownloadSegments(u);
      if (!segs) return false;
      if (expectedFileName && segs.file !== expectedFileName) return false;
      return true;
    }
    if (hop === "first") return false;
    if (!isGithubReleaseCdn(u)) return false;
    if (!expectedFileName) return hop !== "cdn" || Boolean(opts?.digestPinned);
    const declared = cdnDeclaredFileName(u);
    if (declared) return declared === expectedFileName;
    return hop === "cdn" ? Boolean(opts?.digestPinned) : true;
  } catch {
    return false;
  }
}

export function allowedFirstDownloadUrl(url: string, expectedFileName: string): boolean {
  return allowedDownloadHost(url, expectedFileName, { hop: "first" });
}

export function allowedCdnRedirectUrl(url: string, expectedFileName: string, digestPinned: boolean): boolean {
  return allowedDownloadHost(url, expectedFileName, { hop: "cdn", digestPinned });
}

export async function fetchGithubRelease(app: AppId, fetchImpl: GithubFetch): Promise<FetchResult> {
  try {
    const result = await fetchImpl(githubReleaseUrl(app));
    if (!result.ok) {
      const hint = result.status === 404 ? "No GitHub release found (404)." : `GitHub returned HTTP ${result.status}.`;
      return { ok: false, status: result.status, error: result.error ?? hint };
    }
    return { ok: true, status: result.status, release: result.release };
  } catch (err) {
    return { ok: false, status: 0, error: err instanceof Error ? err.message : String(err) };
  }
}

export async function checkBothReleases(opts: {
  lateLocal: string;
  orchLocal: string;
  platform: HostPlatform;
  arch: HostArch;
  prefer?: LatePrefer;
  fetchImpl: GithubFetch;
}): Promise<UpdateCheck> {
  const [lateRes, orchRes] = await Promise.all([
    fetchGithubRelease("late", opts.fetchImpl),
    fetchGithubRelease("orchestrator", opts.fetchImpl),
  ]);
  const late = buildAppStatus({
    id: "late",
    localVersion: opts.lateLocal,
    release: lateRes.release ?? null,
    error: lateRes.error ?? null,
    platform: opts.platform,
    arch: opts.arch,
    prefer: opts.prefer,
  });
  const orchestrator = buildAppStatus({
    id: "orchestrator",
    localVersion: opts.orchLocal,
    release: orchRes.release ?? null,
    error: orchRes.error ?? null,
    platform: opts.platform,
    arch: opts.arch,
    prefer: opts.prefer,
  });
  return buildUpdateCheck(late, orchestrator);
}

export function applyTargetFromChoice(
  which: "late" | "orchestrator" | "both",
  check: UpdateCheck,
): AppId[] {
  if (which === "both") return (["late", "orchestrator"] as const).filter((id) => check[id].newer);
  if (which === "late") return check.late.newer ? ["late"] : [];
  return check.orchestrator.newer ? ["orchestrator"] : [];
}

export function checkFromFetchBoth(raw: {
  lateLocal: string;
  orchLocal: string;
  platform: HostPlatform;
  arch: HostArch;
  prefer?: LatePrefer;
  lateRelease: GithubRelease | null;
  orchRelease: GithubRelease | null;
  lateError?: string | null;
  orchError?: string | null;
}): UpdateCheck {
  return buildUpdateCheck(
    buildAppStatus({
      id: "late",
      localVersion: raw.lateLocal,
      release: raw.lateRelease,
      error: raw.lateError ?? null,
      platform: raw.platform,
      arch: raw.arch,
      prefer: raw.prefer,
    }),
    buildAppStatus({
      id: "orchestrator",
      localVersion: raw.orchLocal,
      release: raw.orchRelease,
      error: raw.orchError ?? null,
      platform: raw.platform,
      arch: raw.arch,
      prefer: raw.prefer,
    }),
  );
}
