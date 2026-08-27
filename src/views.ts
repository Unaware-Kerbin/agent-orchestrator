import type { OrchestratedRun } from "./types.js";

export function toRunView(run: OrchestratedRun, includeHistory = false) {
  return {
    id: run.id,
    specialist: run.specialist,
    backend: run.backend,
    status: run.status,
    agentId: run.agentId,
    providerRunId: run.providerRunId,
    workflowId: run.workflowId,
    stepIndex: run.stepIndex,
    cwd: run.cwd,
    durationMs: run.durationMs,
    error: run.error,
    text: run.text,
    createdAt: new Date(run.createdAt).toISOString(),
    updatedAt: new Date(run.updatedAt).toISOString(),
    ...(includeHistory ? { history: run.history } : {}),
  };
}
