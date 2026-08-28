/** Log-only. Cursor/Gemini/OpenAI failures must not exit `npm run mcp:http`. */
export const MCP_PROCESS_GUARD_REJECTION = "mcp:http unhandledRejection (kept running)";
export const MCP_PROCESS_GUARD_EXCEPTION = "mcp:http uncaughtException (kept running)";

export function installMcpProcessGuards(
  log: (msg: string, detail?: unknown) => void = console.error,
): () => void {
  const onRejection = (reason: unknown): void => {
    log(MCP_PROCESS_GUARD_REJECTION, reason instanceof Error ? (reason.stack ?? reason.message) : reason);
  };
  const onException = (error: Error): void => {
    log(MCP_PROCESS_GUARD_EXCEPTION, error.stack ?? error.message);
  };
  process.on("unhandledRejection", onRejection);
  process.on("uncaughtException", onException);
  return () => {
    process.off("unhandledRejection", onRejection);
    process.off("uncaughtException", onException);
  };
}
