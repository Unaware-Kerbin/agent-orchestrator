import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { packageRoot } from "../src/config.js";

const helper = join(packageRoot(), "scripts", "hf_download.py");

test("hf_download.py never passes tqdm_class / JsonTqdm into snapshot_download", () => {
  const source = readFileSync(helper, "utf8");
  assert.equal(source.includes("tqdm_class="), false);
  assert.equal(source.includes("tqdm_class:"), false);
  assert.match(source, /def snapshot_kwargs/);
  assert.match(source, /get_lock/);
});

test("download_snapshot wrapper mocks snapshot_download without JsonTqdm", () => {
  const script = `
import importlib.util, json, os, sys, tempfile
spec = importlib.util.spec_from_file_location("hf_download", ${JSON.stringify(helper)})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
captured = {}
def fake_snapshot(**kwargs):
    captured.update(kwargs)
    dest = kwargs["local_dir"]
    os.makedirs(dest, exist_ok=True)
    open(os.path.join(dest, "config.json"), "w").write("{}")
    return dest
dest = tempfile.mkdtemp(prefix="hf-wrap-")
path = mod.download_snapshot(
    "hf-internal-testing/tiny-random-gpt2",
    dest,
    snapshot_download=fake_snapshot,
    estimate_bytes=lambda repo, rev: 0,
    watch=False,
)
assert path == dest
assert "tqdm_class" not in captured
assert captured["repo_id"] == "hf-internal-testing/tiny-random-gpt2"
assert captured["local_dir"] == dest
kwargs = mod.snapshot_kwargs("org/name", "/tmp/out")
assert "tqdm_class" not in kwargs
print(json.dumps({"ok": True, "keys": sorted(captured.keys())}))
`;
  const result = spawnSync("python3", ["-c", script], { encoding: "utf8", timeout: 10_000 });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  const payload = JSON.parse((result.stdout ?? "").trim().split(/\n/).pop() ?? "{}") as {
    ok?: boolean;
    keys?: string[];
  };
  assert.equal(payload.ok, true);
  assert.equal((payload.keys ?? []).includes("tqdm_class"), false);
});
