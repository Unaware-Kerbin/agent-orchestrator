#!/usr/bin/env node
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { createOrchestrator } from "./bootstrap.js";
import { createServer } from "./server.js";

const { orchestrator, chat } = createOrchestrator();
void serveStdio(() => createServer(orchestrator, chat));
console.error("agent-orchestrator MCP server running on stdio");
