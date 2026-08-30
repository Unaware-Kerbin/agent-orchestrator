import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  applyUpdates,
  archiveHasUnsafeMember,
  archiveListingHasLink,
  assertGithubDownloadUrl,
  parseUpdateChoice,
  safeAssetFileName,
  safeExtractStem,
  toGuiCheck,
} from "../src/update-apply.js";
import { checkBothReleases, type GithubAsset, type GithubFetch } from "../src/update-sync.js";

function asset(name: string, repo: "late" | "agent-orchestrator"): GithubAsset {
  return {
    name,
    browser_download_url: `https://github.com/Unaware-Kerbin/${repo}/releases/download/v0.2.0/${name}`,
    size: 10,
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
  assert.equal(archiveListingHasLink("linux-x64/bin/node\n"), false);
  assert.equal(archiveListingHasLink("lrwxrwxrwx 0 0 0 0 Jan 1 foo -> /etc\n"), true);
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
      late: { status: 200, tag: "v0.2.0", assets: [asset("Late-0.2.0-linux-x64.AppImage", "late")] },
      orch: {
        status: 200,
        tag: "v0.2.0",
        assets: [asset("agent-orchestrator-0.2.0-linux-x64.tar.gz", "agent-orchestrator")],
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
