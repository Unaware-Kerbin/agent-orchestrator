import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import {
  ensureLateInfer,
  findShippedLateInfer,
  needsLateInferBuild,
  writeLateInferStamps,
} from "../scripts/ensure-late-infer.js";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

function withTmp(fn: (root: string) => void): void {
  const root = mkdtempSync(join(tmpdir(), "orch-ensure-infer-"));
  try {
    fn(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

function writeBinary(root: string, where: "bin" | "runtime", name = "late-infer"): string {
  const dir = where === "bin" ? join(root, "bin") : join(root, "runtime", "bin");
  mkdirSync(dir, { recursive: true });
  const path = join(dir, name);
  writeFileSync(path, "fake-bin");
  return path;
}

test("needsLateInferBuild skips when stamp matches version and binary exists", () => {
  withTmp((root) => {
    writeBinary(root, "bin");
    writeLateInferStamps(root, "0.1.4");
    const decision = needsLateInferBuild(root, "0.1.4");
    assert.equal(decision.needed, false);
    assert.equal(decision.reason, "ok");
  });
});

test("needsLateInferBuild rebuilds when binary is missing", () => {
  withTmp((root) => {
    writeLateInferStamps(root, "0.1.4");
    const decision = needsLateInferBuild(root, "0.1.4");
    assert.equal(decision.needed, true);
    assert.equal(decision.reason, "first launch");
  });
});

test("needsLateInferBuild rebuilds when package version changed", () => {
  withTmp((root) => {
    writeBinary(root, "bin");
    writeLateInferStamps(root, "0.1.4");
    const decision = needsLateInferBuild(root, "0.1.5");
    assert.equal(decision.needed, true);
    assert.equal(decision.reason, "after upgrade");
  });
});

test("needsLateInferBuild rebuilds when packed binary exists but stamp is stale", () => {
  withTmp((root) => {
    writeBinary(root, "runtime");
    writeFileSync(join(root, "runtime", "bin", ".late-infer-built-for"), "0.1.3\n");
    const decision = needsLateInferBuild(root, "0.1.4");
    assert.equal(decision.needed, true);
    assert.equal(decision.reason, "after upgrade");
    assert.equal(findShippedLateInfer(root)?.kind, "runtime");
  });
});

test("ensureLateInfer skips spawn when stamp matches version and binary exists", () => {
  withTmp((root) => {
    writeBinary(root, "bin");
    writeLateInferStamps(root, "0.1.4");
    let spawned = 0;
    const logs: string[] = [];
    const result = ensureLateInfer({
      root,
      version: "0.1.4",
      spawnBuild: () => {
        spawned += 1;
        throw new Error("spawn should not run when stamp matches");
      },
      log: (line) => logs.push(line),
    });
    assert.equal(result.skipped, true);
    assert.equal(result.built, false);
    assert.equal(spawned, 0);
    assert.deepEqual(logs, []);
  });
});

test("ensureLateInfer rebuilds when binary missing and writes stamp without cargo", () => {
  withTmp((root) => {
    let spawned = 0;
    const logs: string[] = [];
    const result = ensureLateInfer({
      root,
      version: "0.1.4",
      spawnBuild: () => {
        spawned += 1;
        writeBinary(root, "bin");
        writeBinary(root, "runtime");
        return { status: 0 };
      },
      log: (line) => logs.push(line),
    });
    assert.equal(spawned, 1);
    assert.equal(result.built, true);
    assert.equal(result.failed, false);
    assert.equal(readFileSync(join(root, "bin", "late-infer.stamp"), "utf8").trim(), "0.1.4");
    assert.equal(readFileSync(join(root, "runtime", "bin", ".late-infer-built-for"), "utf8").trim(), "0.1.4");
    assert.equal(logs[0], "agent-orchestrator: building late-infer on your computer (first launch)…");
    assert.equal(logs[1], "agent-orchestrator: late-infer ready");
  });
});

test("ensureLateInfer rebuilds after version bump with mocked spawn", () => {
  withTmp((root) => {
    writeBinary(root, "bin");
    writeLateInferStamps(root, "0.1.3");
    let spawned = 0;
    const logs: string[] = [];
    const result = ensureLateInfer({
      root,
      version: "0.1.4",
      spawnBuild: () => {
        spawned += 1;
        return { status: 0 };
      },
      log: (line) => logs.push(line),
    });
    assert.equal(spawned, 1);
    assert.equal(result.built, true);
    assert.equal(result.reason, "after upgrade");
    assert.equal(readFileSync(join(root, "bin", "late-infer.stamp"), "utf8").trim(), "0.1.4");
    assert.match(logs[0] ?? "", /after upgrade/);
    assert.equal(logs[1], "agent-orchestrator: late-infer ready");
  });
});

test("ensureLateInfer does not write stamp when infer:build fails", () => {
  withTmp((root) => {
    const logs: string[] = [];
    const result = ensureLateInfer({
      root,
      version: "0.1.4",
      spawnBuild: () => ({ status: 1 }),
      log: (line) => logs.push(line),
    });
    assert.equal(result.failed, true);
    assert.equal(result.status, 1);
    assert.equal(result.built, false);
    assert.equal(findShippedLateInfer(root), undefined);
    try {
      readFileSync(join(root, "bin", "late-infer.stamp"), "utf8");
      assert.fail("stamp must not exist after a failed build");
    } catch {
      // expected
    }
    assert.match(logs[0] ?? "", /first launch/);
    assert.equal(logs.some((line) => /ready/.test(line)), false);
  });
});

test("npm run gui ensures late-infer; mcp:http / start / test do not", () => {
  const pkg = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8")) as {
    scripts: Record<string, string>;
  };
  assert.match(pkg.scripts.gui ?? "", /scripts\/ensure-late-infer\.js/);
  assert.match(pkg.scripts.gui ?? "", /tsx src\/gui\.ts/);
  assert.doesNotMatch(pkg.scripts["mcp:http"] ?? "", /ensure-late-infer/);
  assert.doesNotMatch(pkg.scripts["mcp:http"] ?? "", /infer:build/);
  assert.doesNotMatch(pkg.scripts.start ?? "", /ensure-late-infer/);
  assert.doesNotMatch(pkg.scripts.start ?? "", /infer:build/);
  assert.doesNotMatch(pkg.scripts.test ?? "", /ensure-late-infer/);
  assert.doesNotMatch(pkg.scripts.test ?? "", /infer:build/);
  assert.doesNotMatch(pkg.scripts["gui:stop"] ?? "", /ensure-late-infer/);
  assert.equal(pkg.scripts["infer:build"], "bash scripts/build-late-infer.sh");
});

test("MCP entrypoints and bind never spawn cargo or ensure-late-infer", () => {
  const files = [
    "src/mcp-http.ts",
    "src/index.ts",
    "src/mcp/bind.ts",
    "src/gui.ts",
    "src/server.ts",
    "src/mcp-http-handler.ts",
  ];
  for (const rel of files) {
    const src = readFileSync(join(repoRoot, rel), "utf8");
    assert.doesNotMatch(src, /ensure-late-infer/);
    assert.doesNotMatch(src, /infer:build/);
    assert.doesNotMatch(src, /cargo build/);
    assert.doesNotMatch(src, /build-late-infer/);
  }
});

test("README says first npm run gui builds the engine and MCP start does not", () => {
  const readme = readFileSync(join(repoRoot, "README.md"), "utf8");
  assert.match(readme, /first `npm run gui` builds the engine on your computer/);
  assert.match(readme, /later launches skip/);
  assert.match(readme, /app version bump rebuilds/);
  assert.match(readme, /MCP start \(`npm start` \/ `npm run mcp:http`\) does not/);
});

test("ensure-late-infer copy stays on your computer and does not send people to Late desktop", () => {
  const src = readFileSync(join(repoRoot, "scripts", "ensure-late-infer.js"), "utf8");
  assert.match(src, /your computer/);
  assert.match(src, /first launch/);
  assert.match(src, /after upgrade/);
  assert.match(src, /npm", \["run", "infer:build"\]/);
  assert.doesNotMatch(src, /open Late/i);
  assert.doesNotMatch(src, /Late desktop/);
  assert.doesNotMatch(src, /install Late/i);
});
