import { loadConfig, packageRoot, resolveConfigPath, defaultWorkspaceCwd } from "./config.js";
import { ChatService } from "./chat/service.js";
import { Orchestrator } from "./orchestrator.js";
import { WriteAllowlist } from "./allowlist.js";
import { refreshRuntimeEnv } from "./secrets.js";

export function createOrchestrator(): {
  orchestrator: Orchestrator;
  allowlist: WriteAllowlist;
  configPath: string;
  chat: ChatService;
} {
  refreshRuntimeEnv();
  const configPath = resolveConfigPath();
  const config = loadConfig(configPath);
  const allowlist = WriteAllowlist.load([defaultWorkspaceCwd(config), process.cwd(), packageRoot()]);
  const orchestrator = new Orchestrator(config, allowlist, configPath);
  const chat = new ChatService(orchestrator);
  return { orchestrator, allowlist, configPath, chat };
}
