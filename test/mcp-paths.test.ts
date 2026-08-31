import assert from "node:assert/strict";
import { test } from "node:test";
import {
  bindMcpListenHost,
  bindMcpLoopbackOnly,
  isAutoListenHost,
  lateMcpCopyLines,
  listenHostHeaderOk,
  listenMcpUrl,
  listenOriginOk,
  loopbackMcpUrl,
  primaryPrivateIpv4,
} from "../src/mcp/bind.js";
import { isMcpHealthPath, isMcpLivenessGet, isMcpLoginPath, isMcpPath, normalizeMcpPath } from "../src/mcp/paths.js";
import { assertNoMachineHome } from "./machine-paths.js";

test("MCP HTTP bind allows loopback and one RFC1918 IP; refuses 0.0.0.0 and public", () => {
  assert.equal(bindMcpListenHost(undefined), "127.0.0.1");
  assert.equal(bindMcpListenHost("127.0.0.1"), "127.0.0.1");
  assert.equal(bindMcpListenHost("localhost"), "127.0.0.1");
  assert.equal(bindMcpListenHost("192.168.2.139"), "192.168.2.139");
  assert.equal(bindMcpListenHost("192.168.1.5"), "192.168.1.5");
  assert.equal(bindMcpListenHost("10.1.2.3"), "10.1.2.3");
  assert.equal(bindMcpListenHost("172.16.0.9"), "172.16.0.9");
  assert.equal(bindMcpListenHost("fd12:3456:789a::1"), "fd12:3456:789a::1");
  assert.equal(bindMcpLoopbackOnly("192.168.2.139"), "192.168.2.139");
  assert.throws(() => bindMcpListenHost("0.0.0.0"), /0\.0\.0\.0|private/);
  assert.throws(() => bindMcpListenHost("::"), /private|::/);
  assert.throws(() => bindMcpListenHost("8.8.8.8"), /8\.8\.8\.8|private/);
  assert.throws(() => bindMcpListenHost("1.1.1.1"), /private/);
  assert.throws(() => bindMcpListenHost("169.254.1.1"), /private/);
  assert.throws(() => bindMcpListenHost("255.255.255.255"), /private/);
  assert.throws(() => bindMcpListenHost("example.com"), /private/);
});

test("auto picks the primary RFC1918 IPv4; empty bind stays loopback", () => {
  assert.equal(isAutoListenHost("auto"), true);
  assert.equal(isAutoListenHost("AUTO"), true);
  assert.equal(isAutoListenHost(""), false);
  const picked = primaryPrivateIpv4(
    {
      lo: [{ address: "127.0.0.1", family: "IPv4", internal: true, mac: "", netmask: "255.0.0.0", cidr: null }],
      docker0: [{ address: "172.17.0.1", family: "IPv4", internal: false, mac: "", netmask: "255.255.0.0", cidr: null }],
      virbr0: [{ address: "192.168.122.1", family: "IPv4", internal: false, mac: "", netmask: "255.255.255.0", cidr: null }],
      eno2: [{ address: "192.168.2.139", family: "IPv4", internal: false, mac: "", netmask: "255.255.255.0", cidr: null }],
    },
    "eno2",
  );
  assert.equal(picked, "192.168.2.139");
  const noDefault = primaryPrivateIpv4(
    {
      docker0: [{ address: "172.17.0.1", family: "IPv4", internal: false, mac: "", netmask: "255.255.0.0", cidr: null }],
      eno2: [{ address: "192.168.2.139", family: "IPv4", internal: false, mac: "", netmask: "255.255.255.0", cidr: null }],
    },
    undefined,
  );
  assert.equal(noDefault, "192.168.2.139");
  assert.equal(bindMcpListenHost(undefined), "127.0.0.1");
  assert.equal(bindMcpListenHost(""), "127.0.0.1");
  const lan = primaryPrivateIpv4();
  if (lan) {
    assert.equal(bindMcpListenHost("auto"), lan);
    assert.equal(listenMcpUrl(8790, "auto"), `http://${lan}:8790/mcp`);
  } else {
    assert.throws(() => bindMcpListenHost("auto"), /auto/);
  }
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

test("MCP URL uses the bound listen host", () => {
  assert.equal(loopbackMcpUrl(8107), "http://127.0.0.1:8107/mcp");
  assert.equal(loopbackMcpUrl(8787), "http://127.0.0.1:8787/mcp");
  assert.equal(loopbackMcpUrl(8790), "http://127.0.0.1:8790/mcp");
  assert.equal(listenMcpUrl(8790, "192.168.2.139"), "http://192.168.2.139:8790/mcp");
  assert.equal(listenMcpUrl(8787, "192.168.2.139"), "http://192.168.2.139:8787/mcp");
  assert.equal(listenMcpUrl(8790, "fd12:3456:789a::1"), "http://[fd12:3456:789a::1]:8790/mcp");
  const lines = lateMcpCopyLines(loopbackMcpUrl(8107));
  assert.ok(lines.some((l) => l.includes("http://127.0.0.1:8107/mcp")));
  assert.ok(lines.some((l) => /Late works with MCP off/i.test(l)));
  const lan = lateMcpCopyLines(listenMcpUrl(8790, "192.168.2.139"));
  assert.ok(lan.some((l) => l.includes("http://192.168.2.139:8790/mcp")));
  assert.ok(lan.some((l) => /trusted LAN/i.test(l)));
  for (const line of lines) assertNoMachineHome(line, "lateMcpCopyLines");
  assert.throws(() => loopbackMcpUrl(0), /1–65535/);
});

test("Host and Origin allow the bound private IP; evil.com is rejected", () => {
  assert.equal(listenHostHeaderOk("192.168.2.139:8790", 8790, "192.168.2.139"), true);
  assert.equal(listenHostHeaderOk("127.0.0.1:8790", 8790, "192.168.2.139"), true);
  assert.equal(listenHostHeaderOk("evil.com:8790", 8790, "192.168.2.139"), false);
  assert.equal(listenHostHeaderOk("0.0.0.0:8790", 8790, "192.168.2.139"), false);
  assert.equal(listenOriginOk(undefined, "192.168.2.139"), true);
  assert.equal(listenOriginOk("http://192.168.2.139:8790", "192.168.2.139"), true);
  assert.equal(listenOriginOk("http://127.0.0.1:5173", "192.168.2.139"), true);
  assert.equal(listenOriginOk("http://evil.com", "192.168.2.139"), false);
  assert.equal(listenOriginOk("https://evil.com", "192.168.2.139"), false);
  assert.equal(listenOriginOk("null", "192.168.2.139"), false);
});
