/** Log-only last resort. Prefer fixing throw sites; this must not swallow security checks. */
export const MCP_PROCESS_GUARD_REJECTION = "orchestrator unhandledRejection (kept running)";
export const MCP_PROCESS_GUARD_EXCEPTION = "orchestrator uncaughtException (kept running)";

export function processGuardRejection(label: string): string {
  return `${label} unhandledRejection (kept running)`;
}

export function processGuardException(label: string): string {
  return `${label} uncaughtException (kept running)`;
}

/** Keep the Node process alive after stray rejections/exceptions (GUI HTTP, SSE, chat). */
export function installProcessGuards(
  label = "orchestrator",
  log: (msg: string, detail?: unknown) => void = console.error,
): () => void {
  const rejectionMsg = processGuardRejection(label);
  const exceptionMsg = processGuardException(label);
  const onRejection = (reason: unknown): void => {
    log(rejectionMsg, reason instanceof Error ? (reason.stack ?? reason.message) : reason);
  };
  const onException = (error: Error): void => {
    log(exceptionMsg, error.stack ?? error.message);
  };
  process.on("unhandledRejection", onRejection);
  process.on("uncaughtException", onException);
  return () => {
    process.off("unhandledRejection", onRejection);
    process.off("uncaughtException", onException);
  };
}

export function installMcpProcessGuards(
  log: (msg: string, detail?: unknown) => void = console.error,
): () => void {
  return installProcessGuards("orchestrator", log);
}
