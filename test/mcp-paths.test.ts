import assert from "node:assert/strict";
import { test } from "node:test";
import { bindMcpLoopbackOnly, lateMcpCopyLines, loopbackMcpUrl } from "../src/mcp/bind.js";
import { isMcpHealthPath, isMcpLivenessGet, isMcpLoginPath, isMcpPath, normalizeMcpPath } from "../src/mcp/paths.js";
import { assertNoMachineHome } from "./machine-paths.js";

test("MCP HTTP bind refuses 0.0.0.0 and non-loopback", () => {
  assert.equal(bindMcpLoopbackOnly(undefined), "127.0.0.1");
  assert.equal(bindMcpLoopbackOnly("127.0.0.1"), "127.0.0.1");
  assert.equal(bindMcpLoopbackOnly("localhost"), "127.0.0.1");
  assert.throws(() => bindMcpLoopbackOnly("0.0.0.0"), /127\.0\.0\.1/);
  assert.throws(() => bindMcpLoopbackOnly("::"), /127\.0\.0\.1/);
  assert.throws(() => bindMcpLoopbackOnly("192.168.1.5"), /127\.0\.0\.1/);
});

test("/mcp and /MCP are MCP paths; other GUI routes are not", () => {
  assert.equal(normalizeMcpPath("/MCP"), "/mcp");
  assert.equal(isMcpPath("/mcp"), true);
  assert.equal(isMcpPath("/MCP"), true);
  assert.equal(isMcpPath("/mcp/login"), true);
  assert.equal(isMcpLoginPath("/mcp/login"), true);
  assert.equal(isMcpLoginPath("/MCP/login"), true);
  assert.equal(isMcpPath("/api/chats"), false);
  assert.equal(isMcpPath("/"), false);
  assert.equal(isMcpHealthPath("/mcp/health"), true);
  assert.equal(isMcpHealthPath("/MCP/health"), true);
  assert.equal(isMcpHealthPath("/mcp"), false);
});

test("GET /mcp/health and GET /mcp without SSE are liveness, not tools/list", () => {
  assert.equal(isMcpLivenessGet("GET", "/mcp/health", undefined), true);
  assert.equal(isMcpLivenessGet("GET", "/MCP/health", "application/json"), true);
  assert.equal(isMcpLivenessGet("GET", "/mcp", "application/json"), true);
  assert.equal(isMcpLivenessGet("GET", "/mcp", undefined), true);
  assert.equal(isMcpLivenessGet("GET", "/mcp", "text/event-stream"), false);
  assert.equal(isMcpLivenessGet("POST", "/mcp", undefined), false);
  assert.equal(isMcpLivenessGet("GET", "/", undefined), false);
  assert.equal(isMcpLivenessGet("GET", "/", undefined, { standalone: true }), true);
  assert.equal(isMcpLivenessGet("GET", "/", "text/event-stream", { standalone: true }), false);
});

test("loopback MCP URL is 127.0.0.1 plus the bound port", () => {
  assert.equal(loopbackMcpUrl(8107), "http://127.0.0.1:8107/mcp");
  assert.equal(loopbackMcpUrl(8787), "http://127.0.0.1:8787/mcp");
  assert.equal(loopbackMcpUrl(8790), "http://127.0.0.1:8790/mcp");
  const lines = lateMcpCopyLines(loopbackMcpUrl(8107));
  assert.ok(lines.some((l) => l.includes("http://127.0.0.1:8107/mcp")));
  assert.ok(lines.some((l) => /Late works with MCP off/i.test(l)));
  for (const line of lines) assertNoMachineHome(line, "lateMcpCopyLines");
  assert.throws(() => loopbackMcpUrl(0), /1–65535/);
});

