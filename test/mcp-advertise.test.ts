import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { MCP_GUI_URL_BASENAME, MCP_HTTP_URL_BASENAME, MCP_URL_BASENAME, writeAdvertisedMcpUrl, xdgOrchestratorDir } from "../src/mcp/advertise.js";

test("xdgOrchestratorDir uses XDG or ~/.config, not a hardcoded home", () => {
  assert.equal(
    xdgOrchestratorDir({ XDG_CONFIG_HOME: "/tmp/xdg-cfg" }, "/unused-home"),
    join("/tmp/xdg-cfg", "agent-orchestrator"),
  );
  assert.equal(
    xdgOrchestratorDir({}, "/tmp/fake-home"),
    join("/tmp/fake-home", ".config", "agent-orchestrator"),
  );
});

test("writeAdvertisedMcpUrl writes the bound URL into state and XDG", () => {
  const root = mkdtempSync(join(tmpdir(), "orch-mcp-adv-"));
  const state = join(root, "state");
  const xdg = join(root, "xdg");
  mkdirSync(state, { recursive: true });
  try {
    writeAdvertisedMcpUrl("http://127.0.0.1:8120/mcp", {
      stateDir: state,
      env: { XDG_CONFIG_HOME: xdg },
      home: join(root, "unused-home"),
      kind: "gui",
    });
    assert.equal(readFileSync(join(state, MCP_URL_BASENAME), "utf8").trim(), "http://127.0.0.1:8120/mcp");
    assert.equal(readFileSync(join(state, MCP_GUI_URL_BASENAME), "utf8").trim(), "http://127.0.0.1:8120/mcp");
    assert.equal(
      readFileSync(join(xdg, "agent-orchestrator", MCP_URL_BASENAME), "utf8").trim(),
      "http://127.0.0.1:8120/mcp",
    );
    writeAdvertisedMcpUrl("http://127.0.0.1:8790/mcp", {
      stateDir: state,
      env: { XDG_CONFIG_HOME: xdg },
      home: join(root, "unused-home"),
      kind: "http",
    });
    assert.equal(readFileSync(join(state, MCP_URL_BASENAME), "utf8").trim(), "http://127.0.0.1:8790/mcp");
    assert.equal(readFileSync(join(state, MCP_GUI_URL_BASENAME), "utf8").trim(), "http://127.0.0.1:8120/mcp");
    assert.equal(readFileSync(join(state, MCP_HTTP_URL_BASENAME), "utf8").trim(), "http://127.0.0.1:8790/mcp");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
