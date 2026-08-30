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

function asAssets(raw: unknown): GithubAsset[] {
  if (!Array.isArray(raw)) return [];
  const out: GithubAsset[] = [];
  for (const row of raw) {
    if (!row || typeof row !== "object") continue;
    const rec = row as Record<string, unknown>;
    if (typeof rec.name !== "string" || typeof rec.browser_download_url !== "string") continue;
    out.push({
      name: rec.name,
      browser_download_url: rec.browser_download_url,
      size: typeof rec.size === "number" ? rec.size : undefined,
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
  return {
    id: opts.id,
    label: meta.label,
    localVersion: localUsable ?? (stripTag(opts.localVersion) || opts.localVersion),
    remoteVersion,
    remoteTag: tag,
    releaseUrl: typeof opts.release?.html_url === "string" ? opts.release.html_url : null,
    newer,
    asset: picked
      ? { name: picked.name, url: picked.browser_download_url, size: picked.size ?? 0 }
      : null,
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

export function allowedDownloadHost(url: string): boolean {
  try {
    const u = new URL(url);
    if (u.protocol !== "https:") return false;
    if (u.username || u.password) return false;
    const host = u.hostname.toLowerCase();
    if (host === "api.github.com") {
      return /^\/repos\/Unaware-Kerbin\/(late|agent-orchestrator)\/releases\/latest$/.test(u.pathname);
    }
    if (host === "github.com") {
      return /^\/Unaware-Kerbin\/(late|agent-orchestrator)\/releases\/download\//.test(u.pathname);
    }
    return host === "objects.githubusercontent.com" || host === "release-assets.githubusercontent.com";
  } catch {
    return false;
  }
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
