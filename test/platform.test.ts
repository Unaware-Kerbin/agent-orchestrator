import assert from "node:assert/strict";
import { mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import {
  candidateFileNames,
  extraBinaryDirs,
  parseNetstatListenPids,
  parseSsListenPids,
  pythonDashArgs,
  pythonInterpreterNames,
  SECURE_FILE_MODE,
  which,
  writeSecureFile,
} from "../src/platform.js";

test("candidateFileNames adds PATHEXT on Windows and leaves POSIX names alone", () => {
  assert.deepEqual(candidateFileNames("nvidia-smi", "linux"), ["nvidia-smi"]);
  const win = candidateFileNames("ollama", "win32");
  assert.ok(win.includes("ollama"));
  assert.ok(win.some((name) => name.toLowerCase() === "ollama.exe"));
  assert.deepEqual(candidateFileNames("llama-server.exe", "win32"), ["llama-server.exe"]);
});

test("which finds .exe on a mocked Windows PATH plus typical install dirs", () => {
  const found = which("ollama", {
    platform: "win32",
    pathImpl: path.win32,
    pathEnv: "C:\\Windows\\System32",
    extraDirs: ["C:\\Users\\me\\AppData\\Local\\Programs\\Ollama"],
    exists: (candidate) => /Programs\\Ollama\\ollama\.exe$/i.test(candidate.replace(/\//g, "\\")),
  });
  assert.ok(found);
  assert.match(found.replace(/\//g, "\\"), /Programs\\Ollama\\ollama\.exe$/i);
});

test("which finds nvidia-smi.exe in NVSMI extra dirs", () => {
  const dirs = extraBinaryDirs("win32", path.win32);
  assert.ok(dirs.some((dir) => /NVIDIA Corporation\\NVSMI/i.test(dir)));
  const found = which("nvidia-smi", {
    platform: "win32",
    pathImpl: path.win32,
    pathEnv: "",
    extraDirs: ["C:\\Program Files\\NVIDIA Corporation\\NVSMI"],
    exists: (candidate) => /nvidia-smi\.exe$/i.test(candidate.replace(/\//g, "\\")),
  });
  assert.ok(found);
});

test("python interpreter names prefer py on Windows and python3 on POSIX", () => {
  assert.deepEqual(pythonInterpreterNames("win32")[0], "py");
  assert.deepEqual(pythonInterpreterNames("linux")[0], "python3");
  assert.deepEqual(pythonDashArgs("py"), ["-3"]);
  assert.deepEqual(pythonDashArgs("C:\\\\Windows\\\\py.exe"), ["-3"]);
  assert.deepEqual(pythonDashArgs("python3"), []);
});

test("parseNetstatListenPids extracts LISTENING pid for a TCP port", () => {
  const text = [
    "TCP    127.0.0.1:8787         0.0.0.0:0              LISTENING       4242",
    "TCP    127.0.0.1:18787        0.0.0.0:0              LISTENING       99",
    "TCP    [::1]:8787             [::]:0                 LISTENING       4242",
  ].join("\n");
  assert.deepEqual(parseNetstatListenPids(text, 8787), [4242]);
  assert.deepEqual(parseSsListenPids("users:((\"node\",pid=1234,fd=23))"), [1234]);
});

test("writeSecureFile chmod 0600 on POSIX after overwrite", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "orch-secure-"));
  const file = path.join(dir, "secret.env");
  try {
    writeSecureFile(file, "first\n");
    writeSecureFile(file, "second\n");
    const mode = statSync(file).mode & 0o777;
    if (process.platform !== "win32") {
      assert.equal(mode, SECURE_FILE_MODE);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
