import type { OrchestratedRun } from "./types.js";

const MAX_RUNS = 200;

export class RunStore {
  private readonly runs = new Map<string, OrchestratedRun>();

  create(partial: Omit<OrchestratedRun, "id" | "createdAt" | "updatedAt" | "history"> & {
    id?: string;
    history?: OrchestratedRun["history"];
  }): OrchestratedRun {
    const now = Date.now();
    const run: OrchestratedRun = {
      id: partial.id ?? crypto.randomUUID(),
      createdAt: now,
      updatedAt: now,
      history: partial.history ?? [],
      ...partial,
    };
    this.runs.set(run.id, run);
    this.prune();
    return run;
  }

  get(id: string): OrchestratedRun | undefined {
    return this.runs.get(id);
  }

  update(id: string, patch: Partial<OrchestratedRun>): OrchestratedRun {
    const current = this.runs.get(id);
    if (!current) throw new Error(`Unknown run ${id}`);
    const next = { ...current, ...patch, id, updatedAt: Date.now() };
    this.runs.set(id, next);
    return next;
  }

  list(limit = 50): OrchestratedRun[] {
    return [...this.runs.values()]
      .sort((a, b) => b.createdAt - a.createdAt)
      .slice(0, limit);
  }

  private prune(): void {
    if (this.runs.size <= MAX_RUNS) return;
    const oldest = [...this.runs.values()].sort((a, b) => a.createdAt - b.createdAt);
    const extra = this.runs.size - MAX_RUNS;
    for (const run of oldest.slice(0, extra)) this.runs.delete(run.id);
  }
}
