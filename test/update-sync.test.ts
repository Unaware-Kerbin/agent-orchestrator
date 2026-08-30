import assert from "node:assert/strict";
import test from "node:test";
import {
  allowedCdnRedirectUrl,
  allowedDownloadHost,
  allowedFirstDownloadUrl,
  applyTargetFromChoice,
  buildAppStatus,
  checkBothReleases,
  compareSemver,
  isNewerThan,
  localVersionOverride,
  parseAssetDigest,
  parseSemver,
  pickReleaseAsset,
  pinDigestString,
  sha256HexMatches,
  stripTag,
  type GithubAsset,
  type GithubFetch,
} from "../src/update-sync.js";

function asset(name: string, tag = "v0.2.0"): GithubAsset {
  return { name, browser_download_url: `https://github.com/Unaware-Kerbin/late/releases/download/${tag}/${name}`, size: 10 };
}

function lateAssets(tag = "v0.2.0"): GithubAsset[] {
  return [
    asset("Late-0.2.0-linux-amd64.deb", tag),
    asset("Late-0.2.0-linux-x86_64.AppImage", tag),
    asset("Late-0.2.0-mac-arm64.dmg", tag),
    asset("Late-0.2.0-win-x64.exe", tag),
  ];
}

function orchAssets(tag = "v0.2.0"): GithubAsset[] {
  return [
    {
      name: "agent-orchestrator-0.2.0-linux-x64.tar.gz",
      browser_download_url: `https://github.com/Unaware-Kerbin/agent-orchestrator/releases/download/${tag}/agent-orchestrator-0.2.0-linux-x64.tar.gz`,
    },
    {
      name: "agent-orchestrator-0.2.0-win-x64.zip",
      browser_download_url: `https://github.com/Unaware-Kerbin/agent-orchestrator/releases/download/${tag}/agent-orchestrator-0.2.0-win-x64.zip`,
    },
  ];
}

function mockFetch(map: Record<string, { status: number; tag?: string; assets?: GithubAsset[]; body?: unknown }>): GithubFetch {
  return async (url) => {
    const key = url.includes("/late/") ? "late" : url.includes("/agent-orchestrator/") ? "orch" : "other";
    const row = map[key];
    if (!row) return { ok: false, status: 0, error: "unexpected URL" };
    if (row.status === 404) return { ok: false, status: 404, error: "No GitHub release found (404)." };
    if (row.status >= 400) return { ok: false, status: row.status, error: `GitHub returned HTTP ${row.status}.` };
    return {
      ok: true,
      status: row.status,
      release: {
        tag_name: row.tag,
        html_url: `https://github.com/example/${key}/releases/tag/${row.tag}`,
        assets: row.assets ?? [],
        ...(typeof row.body === "object" && row.body ? row.body : {}),
      },
    };
  };
}

test("semver compare treats v-prefix and equal tags", () => {
  assert.deepEqual(parseSemver("v0.1.9"), { major: 0, minor: 1, patch: 9 });
  assert.equal(stripTag("v0.1.9"), "0.1.9");
  assert.equal(compareSemver("0.1.9", "v0.1.9"), 0);
  assert.ok((compareSemver("0.1.8", "0.1.9") ?? 0) < 0);
  assert.ok((compareSemver("0.2.0", "0.1.9") ?? 0) > 0);
  assert.equal(isNewerThan("0.1.9", "0.1.9"), false);
  assert.equal(isNewerThan("v0.2.0", "0.1.9"), true);
  assert.equal(isNewerThan("0.1.8", "0.1.9"), false);
});

test("malformed tag is not newer", () => {
  assert.equal(parseSemver("nightly"), null);
  assert.equal(parseSemver("v1.2"), null);
  assert.equal(compareSemver("nightly", "0.1.9"), null);
  assert.equal(isNewerThan("nightly", "0.1.9"), false);
  const status = buildAppStatus({
    id: "late",
    localVersion: "0.1.9",
    release: { tag_name: "nightly", assets: lateAssets() },
    error: null,
    platform: "linux",
    arch: "x64",
  });
  assert.equal(status.newer, false);
  assert.match(status.error ?? "", /Malformed release tag/);
});

test("pick Late AppImage vs deb and orchestrator archive", () => {
  const img = pickReleaseAsset({ app: "late", assets: lateAssets(), platform: "linux", arch: "x64", prefer: "appimage" });
  assert.equal(img?.name, "Late-0.2.0-linux-x86_64.AppImage");
  const deb = pickReleaseAsset({ app: "late", assets: lateAssets(), platform: "linux", arch: "x64", prefer: "deb" });
  assert.equal(deb?.name, "Late-0.2.0-linux-amd64.deb");
  const mac = pickReleaseAsset({ app: "late", assets: lateAssets(), platform: "darwin", arch: "arm64" });
  assert.equal(mac?.name, "Late-0.2.0-mac-arm64.dmg");
  const orch = pickReleaseAsset({ app: "orchestrator", assets: orchAssets(), platform: "linux", arch: "x64" });
  assert.equal(orch?.name, "agent-orchestrator-0.2.0-linux-x64.tar.gz");
});

test("no update when both remotes match local", async () => {
  const check = await checkBothReleases({
    lateLocal: "0.1.9",
    orchLocal: "0.1.1",
    platform: "linux",
    arch: "x64",
    fetchImpl: mockFetch({
      late: { status: 200, tag: "v0.1.9", assets: lateAssets("v0.1.9") },
      orch: { status: 200, tag: "v0.1.1", assets: orchAssets("v0.1.1") },
    }),
  });
  assert.equal(check.anyNewer, false);
  assert.equal(check.bothNewer, false);
  assert.equal(check.lateOnly, false);
  assert.equal(check.orchestratorOnly, false);
  assert.equal(check.late.newer, false);
  assert.equal(check.orchestrator.newer, false);
});

test("Late only", async () => {
  const check = await checkBothReleases({
    lateLocal: "0.1.0",
    orchLocal: "0.1.1",
    platform: "linux",
    arch: "x64",
    prefer: "appimage",
    fetchImpl: mockFetch({
      late: { status: 200, tag: "v0.1.9", assets: lateAssets("v0.1.9") },
      orch: { status: 200, tag: "v0.1.1", assets: orchAssets("v0.1.1") },
    }),
  });
  assert.equal(check.lateOnly, true);
  assert.equal(check.bothNewer, false);
  assert.equal(check.late.newer, true);
  assert.equal(check.orchestrator.newer, false);
  assert.deepEqual(applyTargetFromChoice("late", check), ["late"]);
  assert.deepEqual(applyTargetFromChoice("both", check), ["late"]);
  assert.deepEqual(applyTargetFromChoice("orchestrator", check), []);
});

test("orchestrator only", async () => {
  const check = await checkBothReleases({
    lateLocal: "0.1.9",
    orchLocal: "0.1.0",
    platform: "linux",
    arch: "x64",
    fetchImpl: mockFetch({
      late: { status: 200, tag: "v0.1.9", assets: lateAssets("v0.1.9") },
      orch: { status: 200, tag: "v0.1.1", assets: orchAssets("v0.1.1") },
    }),
  });
  assert.equal(check.orchestratorOnly, true);
  assert.equal(check.late.newer, false);
  assert.equal(check.orchestrator.newer, true);
  assert.deepEqual(applyTargetFromChoice("orchestrator", check), ["orchestrator"]);
  assert.deepEqual(applyTargetFromChoice("both", check), ["orchestrator"]);
});

test("both", async () => {
  const check = await checkBothReleases({
    lateLocal: "0.1.0",
    orchLocal: "0.1.0",
    platform: "linux",
    arch: "x64",
    fetchImpl: mockFetch({
      late: { status: 200, tag: "v0.2.0", assets: lateAssets() },
      orch: { status: 200, tag: "v0.2.0", assets: orchAssets() },
    }),
  });
  assert.equal(check.bothNewer, true);
  assert.equal(check.anyNewer, true);
  assert.deepEqual(applyTargetFromChoice("both", check), ["late", "orchestrator"]);
  assert.equal(check.late.asset?.name.includes("AppImage"), true);
});

test("asset URL that does not match the release tag is not downloadable", () => {
  const status = buildAppStatus({
    id: "late",
    localVersion: "0.1.0",
    release: { tag_name: "v0.1.9", assets: lateAssets() },
    error: null,
    platform: "linux",
    arch: "x64",
  });
  assert.equal(status.newer, true);
  assert.equal(status.asset, null);
});

test("404 does not throw and is not an update", async () => {
  const check = await checkBothReleases({
    lateLocal: "0.1.9",
    orchLocal: "0.1.1",
    platform: "linux",
    arch: "x64",
    fetchImpl: mockFetch({
      late: { status: 404 },
      orch: { status: 404 },
    }),
  });
  assert.equal(check.anyNewer, false);
  assert.match(check.late.error ?? "", /404/);
  assert.match(check.orchestrator.error ?? "", /404/);
  assert.equal(check.late.remoteVersion, null);
});

test("allowedDownloadHost refuses custom GitHub URLs, userinfo, and raw CDN", () => {
  assert.equal(
    allowedDownloadHost("https://github.com/Unaware-Kerbin/late/releases/download/v0.1.9/Late-0.1.9-linux-x64.AppImage"),
    true,
  );
  assert.equal(allowedDownloadHost("https://api.github.com/repos/Unaware-Kerbin/late/releases/latest"), true);
  assert.equal(allowedDownloadHost("https://objects.githubusercontent.com/github-production-release-asset-2e65be/x"), true);
  assert.equal(allowedDownloadHost("https://raw.githubusercontent.com/evil/malware/main/pwn"), false);
  assert.equal(allowedDownloadHost("https://github.com/evil/late/releases/download/v1/x.bin"), false);
  assert.equal(
    allowedDownloadHost("https://user:token@github.com/Unaware-Kerbin/late/releases/download/v0.1.9/x.bin"),
    false,
  );
  assert.equal(allowedDownloadHost("https://127.0.0.1/secret"), false);
  assert.equal(allowedDownloadHost("http://github.com/Unaware-Kerbin/late/releases/download/v0.1.9/x.bin"), false);
  assert.equal(allowedDownloadHost("https://objects.githubusercontent.com/evil/pwn"), false);
  assert.equal(allowedDownloadHost("https://objects.githubusercontent.com/github-production-user-asset/x"), false);
});

test("first hop and CDN redirect stay on the chosen asset", () => {
  const file = "Late-0.1.9-linux-x64.AppImage";
  const first = `https://github.com/Unaware-Kerbin/late/releases/download/v0.1.9/${file}`;
  assert.equal(allowedFirstDownloadUrl(first, file), true);
  assert.equal(allowedFirstDownloadUrl(first, "other.bin"), false);
  assert.equal(
    allowedFirstDownloadUrl("https://objects.githubusercontent.com/github-production-release-asset/1/2", file),
    false,
  );
  const cdn = `https://release-assets.githubusercontent.com/github-production-release-asset/1/2?rscd=attachment%3B%20filename%3D${file}`;
  assert.equal(allowedCdnRedirectUrl(cdn, file, false), true);
  assert.equal(allowedCdnRedirectUrl(cdn, "other.bin", true), false);
  assert.equal(
    allowedCdnRedirectUrl("https://release-assets.githubusercontent.com/github-production-release-asset/1/2", file, false),
    false,
  );
  assert.equal(
    allowedCdnRedirectUrl("https://release-assets.githubusercontent.com/github-production-release-asset/1/2", file, true),
    true,
  );
  assert.equal(
    allowedCdnRedirectUrl("http://release-assets.githubusercontent.com/github-production-release-asset/1/2", file, true),
    false,
  );
});

test("parseAssetDigest pins sha256 and sha256HexMatches fails closed", () => {
  assert.equal(parseAssetDigest("sha256:277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9"), "277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9");
  assert.equal(parseAssetDigest(null, "277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9"), "277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9");
  assert.equal(parseAssetDigest(null), null);
  assert.throws(() => parseAssetDigest("md5:abc"), /SHA-256/);
  assert.equal(pinDigestString("277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9"), "sha256:277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9");
  assert.equal(
    sha256HexMatches(
      "277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9",
      "277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9",
    ),
    true,
  );
  assert.equal(
    sha256HexMatches(
      "277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9",
      "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    false,
  );
});

test("Electron 44 is not Late: do not skip or force a GitHub update", () => {
  const lie = buildAppStatus({
    id: "late",
    localVersion: "44.0.0",
    release: { tag_name: "v0.1.9", assets: lateAssets() },
    error: null,
    platform: "linux",
    arch: "x64",
  });
  assert.equal(lie.newer, false);
  assert.match(lie.error ?? "", /Electron/);
  assert.equal(isNewerThan("0.1.9", "44.0.0"), false);

  const real = buildAppStatus({
    id: "late",
    localVersion: "0.1.8",
    release: { tag_name: "v0.1.9", assets: lateAssets() },
    error: null,
    platform: "linux",
    arch: "x64",
  });
  assert.equal(real.newer, true);
  assert.equal(real.error, null);

  const current = buildAppStatus({
    id: "late",
    localVersion: "0.1.9",
    release: { tag_name: "v0.1.9", assets: lateAssets() },
    error: null,
    platform: "linux",
    arch: "x64",
  });
  assert.equal(current.newer, false);
  assert.equal(current.error, null);

  assert.equal(localVersionOverride("late", "0.1.9", { UPDATE_SYNC_LATE_LOCAL: "44.0.0" }), "0.1.9");
  assert.equal(localVersionOverride("late", "0.1.8", { UPDATE_SYNC_LATE_LOCAL: "0.1.8" }), "0.1.8");
});

test("unsigned note on mac/win only", () => {
  const mac = buildAppStatus({
    id: "late",
    localVersion: "0.1.0",
    release: { tag_name: "v0.1.9", assets: lateAssets() },
    error: null,
    platform: "darwin",
    arch: "arm64",
  });
  assert.equal(mac.unsigned, true);
  assert.equal(mac.newer, true);
  const linux = buildAppStatus({
    id: "late",
    localVersion: "0.1.0",
    release: { tag_name: "v0.1.9", assets: lateAssets() },
    error: null,
    platform: "linux",
    arch: "x64",
  });
  assert.equal(linux.unsigned, false);
});
