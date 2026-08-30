import assert from "node:assert/strict";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  applyUpdates,
  archiveHasUnsafeMember,
  archiveListingHasLink,
  archiveMemberEscapesDest,
  assertGithubDownloadUrl,
  downloadAsset,
  parseUpdateChoice,
  safeAssetFileName,
  safeExtractStem,
  toGuiCheck,
  verifyFileDigest,
} from "../src/update-apply.js";
import {
  allowedCdnRedirectUrl,
  allowedDownloadHost,
  checkBothReleases,
  parseAssetDigest,
  type GithubAsset,
  type GithubFetch,
} from "../src/update-sync.js";

const BYTES_DIGEST = "sha256:277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9";
const BYTES_SHA = "277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9";

function asset(name: string, repo: "late" | "agent-orchestrator", digest?: string): GithubAsset {
  return {
    name,
    browser_download_url: `https://github.com/Unaware-Kerbin/${repo}/releases/download/v0.2.0/${name}`,
    size: 10,
    digest,
  };
}

function mockFetch(map: Record<string, { status: number; tag?: string; assets?: GithubAsset[] }>): GithubFetch {
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
        html_url: `https://github.com/Unaware-Kerbin/${key === "late" ? "late" : "agent-orchestrator"}/releases/tag/${row.tag}`,
        assets: row.assets ?? [],
      },
    };
  };
}

test("parseUpdateChoice and download URL stay on the two repos", () => {
  assert.equal(parseUpdateChoice("both"), "both");
  assert.throws(() => parseUpdateChoice("all"), /this app, Late, or both/);
  assert.doesNotThrow(() =>
    assertGithubDownloadUrl("https://github.com/Unaware-Kerbin/late/releases/download/v0.1.9/Late-0.1.9-linux-x64.AppImage"),
  );
  assert.throws(() => assertGithubDownloadUrl("https://127.0.0.1/secret"), /GitHub|Unaware-Kerbin/);
  assert.throws(() => assertGithubDownloadUrl("https://github.com/evil/late/releases/download/v1/x.bin"), /Unaware-Kerbin|GitHub/);
  assert.throws(() => assertGithubDownloadUrl("https://raw.githubusercontent.com/evil/malware/main/pwn"), /Unaware-Kerbin|GitHub/);
  assert.throws(() => assertGithubDownloadUrl("https://user:token@github.com/Unaware-Kerbin/late/releases/download/v1/x"), /Unaware-Kerbin|GitHub|login/);
  assert.equal(safeAssetFileName("../../etc/passwd"), "passwd");
  assert.throws(() => safeAssetFileName(".."), /not safe/);
  assert.equal(safeExtractStem("agent-orchestrator-0.2.0-linux-x64.tar.gz"), "agent-orchestrator-0.2.0-linux-x64");
  assert.throws(() => safeExtractStem("..tar.gz"), /not safe/);
  assert.throws(() => safeExtractStem("..zip"), /not safe/);
  assert.equal(archiveHasUnsafeMember("foo/bar\n"), false);
  assert.equal(archiveHasUnsafeMember("linux-x64/bin/node\n"), false);
  assert.equal(archiveHasUnsafeMember("../etc/passwd\n"), true);
  assert.equal(archiveHasUnsafeMember("C:/Windows/system32/evil\n"), true);
  assert.equal(archiveMemberEscapesDest("foo/bar", "/tmp/updates"), false);
  assert.equal(archiveMemberEscapesDest("../etc/passwd", "/tmp/updates"), true);
  assert.equal(archiveListingHasLink("linux-x64/bin/node\n"), false);
  assert.equal(archiveListingHasLink("lrwxrwxrwx 0 0 0 0 Jan 1 foo -> /etc\n"), true);
  assert.equal(parseAssetDigest("sha256:277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9"), "277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9");
  assert.throws(() => parseAssetDigest("md5:abc"), /SHA-256/);
  assert.equal(
    allowedCdnRedirectUrl("https://evil.example/github-production-release-asset/x", "Late.AppImage", true),
    false,
  );
  assert.equal(
    allowedDownloadHost("https://github.com/Unaware-Kerbin/late/releases/download/v0.2.0/other.bin", "Late.AppImage", {
      hop: "first",
    }),
    false,
  );
});

test("toGuiCheck always offers Late, this app, and both", async () => {
  const check = await checkBothReleases({
    lateLocal: "0.1.9",
    orchLocal: "0.1.1",
    platform: "linux",
    arch: "x64",
    fetchImpl: mockFetch({
      late: { status: 200, tag: "v0.1.9", assets: [asset("Late-0.1.9-linux-x64.AppImage", "late")] },
      orch: { status: 200, tag: "v0.1.1", assets: [asset("agent-orchestrator-0.1.1-linux-x64.tar.gz", "agent-orchestrator")] },
    }),
  });
  const gui = toGuiCheck(check);
  assert.equal(gui.cloudAiRequired, false);
  assert.deepEqual(gui.choices, ["orchestrator", "late", "both"]);
  assert.equal(gui.updateBoth, false);
  assert.match(gui.message, /already newest|match GitHub/i);
});

test("applyUpdates refuses without confirm", async () => {
  await assert.rejects(() => applyUpdates({ choice: "both", confirm: false }), /Say yes first/);
});

function withLocalVersions<T>(late: string, orch: string, fn: () => Promise<T>): Promise<T> {
  const prevLate = process.env.UPDATE_SYNC_LATE_LOCAL;
  const prevOrch = process.env.UPDATE_SYNC_ORCH_LOCAL;
  process.env.UPDATE_SYNC_LATE_LOCAL = late;
  process.env.UPDATE_SYNC_ORCH_LOCAL = orch;
  return fn().finally(() => {
    if (prevLate === undefined) delete process.env.UPDATE_SYNC_LATE_LOCAL;
    else process.env.UPDATE_SYNC_LATE_LOCAL = prevLate;
    if (prevOrch === undefined) delete process.env.UPDATE_SYNC_ORCH_LOCAL;
    else process.env.UPDATE_SYNC_ORCH_LOCAL = prevOrch;
  });
}

test("applyUpdates skips already-current without downloading", async () => {
  await withLocalVersions("0.1.9", "0.1.1", async () => {
    const downloaded: string[] = [];
    const fetchImpl = mockFetch({
      late: { status: 200, tag: "v0.1.9", assets: [asset("Late-0.1.9-linux-x64.AppImage", "late")] },
      orch: { status: 200, tag: "v0.1.1", assets: [asset("agent-orchestrator-0.1.1-linux-x64.tar.gz", "agent-orchestrator")] },
    });
    const result = await applyUpdates({
      choice: "both",
      confirm: true,
      fetchImpl,
      download: async (url) => {
        downloaded.push(url);
      },
    });
    assert.deepEqual(result.applied, []);
    assert.equal(result.skipped.length, 2);
    assert.equal(downloaded.length, 0);
    assert.match(result.message, /already current/i);
  });
});

test("applyUpdates does not treat a failed GitHub check as already current", async () => {
  await withLocalVersions("0.1.9", "0.1.1", async () => {
    const downloaded: string[] = [];
    const result = await applyUpdates({
      choice: "both",
      confirm: true,
      fetchImpl: mockFetch({ late: { status: 404 }, orch: { status: 500 } }),
      download: async (url) => {
        downloaded.push(url);
      },
    });
    assert.equal(downloaded.length, 0);
    assert.equal(result.applied.length, 0);
    assert.ok(result.skipped.every((row) => !/already current/i.test(row.reason)));
    assert.match(result.skipped.map((row) => row.reason).join(" "), /404|HTTP 500/);
  });
});

test("applyUpdates downloads both when newer", async () => {
  await withLocalVersions("0.1.0", "0.1.0", async () => {
    const dir = mkdtempSync(join(tmpdir(), "orch-apply-"));
    const prev = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
    const fetchImpl = mockFetch({
      late: { status: 200, tag: "v0.2.0", assets: [asset("Late-0.2.0-linux-x64.AppImage", "late", BYTES_DIGEST)] },
      orch: {
        status: 200,
        tag: "v0.2.0",
        assets: [asset("agent-orchestrator-0.2.0-linux-x64.tar.gz", "agent-orchestrator", BYTES_DIGEST)],
      },
    });
    const downloaded: string[] = [];
    try {
      const result = await applyUpdates({
        choice: "both",
        confirm: true,
        fetchImpl,
        download: async (url, dest) => {
          downloaded.push(url);
          writeFileSync(dest, "bytes");
        },
      });
      assert.equal(result.cloudAiRequired, false);
      assert.deepEqual(result.applied.slice().sort(), ["late", "orchestrator"]);
      assert.equal(downloaded.length, 2);
      assert.ok(result.saved.every((row) => row.path.startsWith(dir)));
    } finally {
      if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
      else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prev;
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

function assetWithDigest(name: string, repo: "late" | "agent-orchestrator", digest: string): GithubAsset {
  return { ...asset(name, repo), digest };
}

test("applyUpdates does not download when GitHub omitted the digest", async () => {
  await withLocalVersions("0.1.0", "0.1.0", async () => {
    const downloaded: string[] = [];
    const result = await applyUpdates({
      choice: "late",
      confirm: true,
      fetchImpl: mockFetch({
        late: { status: 200, tag: "v0.2.0", assets: [asset("Late-0.2.0-linux-x64.AppImage", "late")] },
        orch: { status: 200, tag: "v0.1.1", assets: [asset("agent-orchestrator-0.1.1-linux-x64.tar.gz", "agent-orchestrator")] },
      }),
      download: async (url) => {
        downloaded.push(url);
      },
    });
    assert.equal(downloaded.length, 0);
    assert.equal(result.applied.length, 0);
    assert.match(result.skipped.map((row) => row.reason).join(" "), /digest/i);
  });
});

test("applyUpdates fails closed when the pinned digest does not match", async () => {
  await withLocalVersions("0.1.0", "0.1.1", async () => {
    const dir = mkdtempSync(join(tmpdir(), "orch-digest-"));
    const prev = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
    try {
      const result = await applyUpdates({
        choice: "late",
        confirm: true,
        fetchImpl: mockFetch({
          late: {
            status: 200,
            tag: "v0.2.0",
            assets: [assetWithDigest("Late-0.2.0-linux-x64.AppImage", "late", "sha256:0000000000000000000000000000000000000000000000000000000000000000")],
          },
          orch: { status: 200, tag: "v0.1.1", assets: [asset("agent-orchestrator-0.1.1-linux-x64.tar.gz", "agent-orchestrator")] },
        }),
        download: async (_url, dest) => {
          writeFileSync(dest, "bytes");
        },
      });
      assert.equal(result.applied.length, 0);
      assert.match(result.skipped.map((row) => row.reason).join(" "), /did not match|digest/i);
      assert.equal(existsSync(join(dir, "updates", "Late-0.2.0-linux-x64.AppImage")), false);
    } finally {
      if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
      else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prev;
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

test("applyUpdates keeps a file when the pinned digest matches", async () => {
  await withLocalVersions("0.1.0", "0.1.1", async () => {
    const dir = mkdtempSync(join(tmpdir(), "orch-digest-ok-"));
    const prev = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
    try {
      const result = await applyUpdates({
        choice: "late",
        confirm: true,
        fetchImpl: mockFetch({
          late: {
            status: 200,
            tag: "v0.2.0",
            assets: [assetWithDigest("Late-0.2.0-linux-x64.AppImage", "late", `sha256:${BYTES_SHA}`)],
          },
          orch: { status: 200, tag: "v0.1.1", assets: [asset("agent-orchestrator-0.1.1-linux-x64.tar.gz", "agent-orchestrator")] },
        }),
        download: async (_url, dest) => {
          writeFileSync(dest, "bytes");
        },
      });
      assert.deepEqual(result.applied, ["late"]);
      assert.match(result.saved[0]?.note ?? "", /did not run the installer or use root/i);
    } finally {
      if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
      else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prev;
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

test("downloadAsset refuses http and off-asset CDN redirects", async () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-dl-"));
  const dest = join(dir, "Late-0.2.0-linux-x64.AppImage");
  const first = "https://github.com/Unaware-Kerbin/late/releases/download/v0.2.0/Late-0.2.0-linux-x64.AppImage";
  const mockHttp = (async () =>
    new Response(null, { status: 302, headers: { location: "http://evil.example/x" } })) as typeof fetch;
  await assert.rejects(
    () => downloadAsset(first, dest, mockHttp, { fileName: "Late-0.2.0-linux-x64.AppImage" }),
    /https|redirect/i,
  );
  const mockWrong = (async (_url: string | URL | Request) => {
    const href = String(_url);
    if (href.startsWith("https://github.com/")) {
      return new Response(null, {
        status: 302,
        headers: {
          location:
            "https://release-assets.githubusercontent.com/github-production-release-asset/1/2?rscd=attachment%3B%20filename%3Dother.bin",
        },
      });
    }
    return new Response(new Uint8Array([1, 2, 3]), { status: 200 });
  }) as typeof fetch;
  await assert.rejects(
    () => downloadAsset(first, dest, mockWrong, { fileName: "Late-0.2.0-linux-x64.AppImage" }),
    /redirect|chosen|GitHub/i,
  );
  rmSync(dir, { recursive: true, force: true });
});

test("verifyFileDigest fails closed on mismatch and unlinks", async () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-hash-"));
  const file = join(dir, "x.bin");
  writeFileSync(file, "bytes");
  await assert.rejects(
    () => verifyFileDigest(file, "sha256:0000000000000000000000000000000000000000000000000000000000000000"),
    /digest/i,
  );
  await verifyFileDigest(file, `sha256:${BYTES_SHA}`).then(
    () => {
      throw new Error("mismatch should have unlinked the file");
    },
    () => undefined,
  );
  writeFileSync(file, "bytes");
  await verifyFileDigest(file, `sha256:${BYTES_SHA}`);
  rmSync(dir, { recursive: true, force: true });
});
