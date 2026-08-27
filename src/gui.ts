#!/usr/bin/env node
import { spawn } from "node:child_process";
import { createOrchestrator } from "./bootstrap.js";
import { loadOrCreateGuiToken } from "./gui-auth.js";
import { bindLoopbackOnly, startGuiServer } from "./gui/http.js";
import { clearGuiPid, guiAddrInUseMessage, guiPidPath, stopOurGui, writeGuiPid } from "./gui-process.js";

const port = Number(process.env.AGENT_ORCHESTRATOR_GUI_PORT ?? "8787");
if (!Number.isInteger(port) || port < 1 || port > 65535) {
  console.error("AGENT_ORCHESTRATOR_GUI_PORT must be an integer 1–65535");
  process.exit(1);
}

if (process.argv.includes("--stop")) {
  const result = await stopOurGui({ port });
  console.error(result.message);
  process.exit(result.foreign.length > 0 ? 1 : 0);
}

try {
  bindLoopbackOnly(process.env.AGENT_ORCHESTRATOR_GUI_HOST);
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}

const { orchestrator, chat } = createOrchestrator();
const secret = loadOrCreateGuiToken();
const { server, listen } = startGuiServer({ orchestrator, chat, token: secret.token, port });

server.on("error", (error) => {
  const err = error as NodeJS.ErrnoException;
  if (err.code === "EADDRINUSE") {
    console.error(guiAddrInUseMessage(port));
    process.exit(1);
  }
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});

const shutdown = (): void => {
  clearGuiPid(process.pid);
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1500).unref();
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

server.listen(listen.port, listen.host, () => {
  writeGuiPid(process.pid);
  const openUrl = `${listen.url}/?token=${encodeURIComponent(secret.token)}`;
  console.error(`Agent Orchestrator GUI`);
  console.error(`  bind:   ${listen.host}:${listen.port} (loopback only)`);
  console.error(`  pid:    ${process.pid} (${guiPidPath()})`);
  console.error(`  open:   ${openUrl}`);
  console.error(`  token:  ${secret.path}${secret.created ? " (created)" : ""}`);
  console.error(`  notes:  MCP stdio is unchanged. This UI is a local extra surface.`);
  console.error(`          Do not tunnel, proxy, or bind this process off localhost.`);
  console.error(`          Stop: npm run gui:stop   Restart: npm run gui:restart`);
  console.error(`          Docker stop of an orch-vllm-* container does not stop this GUI.`);

  if (process.argv.includes("--open")) {
    const opener = process.platform === "darwin" ? "open" : process.platform === "win32" ? "cmd" : "xdg-open";
    const args = process.platform === "win32" ? ["/c", "start", "", openUrl] : [openUrl];
    spawn(opener, args, { stdio: "ignore", detached: true }).unref();
  }
});
