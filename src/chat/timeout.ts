/** Per-speaker debate cap. Hung Cursor/Gemini must not block the round-table. */
export const DEFAULT_SPEAKER_TIMEOUT_MS = 25_000;
/** After a Late `propose_command` JSON lands, wait this long for other speakers then return. */
export const DEFAULT_EARLY_FLUSH_GRACE_MS = 2_500;

const LATE_TOOL_RE =
  /"tool"\s*:\s*"(propose_command|propose_api_get|propose_staged_artifact|list_open_sessions|read_scrollback|query_pcap|ask_user)"/;

export function speakerTimeoutMs(): number {
  return readBoundedMs("AGENT_ORCHESTRATOR_SPEAKER_TIMEOUT_MS", DEFAULT_SPEAKER_TIMEOUT_MS, 50, 120_000);
}

export function earlyFlushGraceMs(): number {
  return readBoundedMs("AGENT_ORCHESTRATOR_EARLY_FLUSH_GRACE_MS", DEFAULT_EARLY_FLUSH_GRACE_MS, 0, 15_000);
}

function readBoundedMs(envName: string, fallback: number, min: number, max: number): number {
  const raw = process.env[envName];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(n)));
}

export function looksLikeLateToolJson(text: string): boolean {
  return LATE_TOOL_RE.test(text);
}

export function timeoutErrorMessage(label: string, timeoutMs: number): string {
  const secs = Math.max(1, Math.round(timeoutMs / 1000));
  return `${label} timed out after ${secs}s — skipped so other speakers can finish.`;
}

/** Timeouts, 429s, and quota errors skip one speaker — they are not a thread/tool failure. */
export function isSpeakerSkipError(text: string): boolean {
  const t = text.trim();
  if (!t) return false;
  if (/timed out after \d+s/i.test(t)) return true;
  if (/\b429\b/.test(t)) return true;
  if (/rate[- ]limit|RESOURCE_EXHAUSTED|quota exceeded/i.test(t)) return true;
  return false;
}

/** One-line chip for a skipped debate speaker (never a JSON blob). */
export function speakerSkipLine(label: string, text: string): string {
  const t = text.trim();
  const secs = t.match(/timed out after (\d+)s/i)?.[1];
  if (secs) return `${label}: timed out after ${secs}s — skipped`;
  if (/\b429\b/.test(t) || /rate[- ]limit|RESOURCE_EXHAUSTED|quota exceeded/i.test(t)) {
    return `${label}: rate-limited (429) — skipped`;
  }
  const first = (t.split(/\n/)[0] ?? "skipped").trim().slice(0, 100);
  return `${label}: ${first || "skipped"}`;
}

/** Attach a no-op catch so a later rejection cannot become `unhandledRejection`. */
export function ignoreLater(promise: Promise<unknown>): void {
  void promise.then(
    () => undefined,
    () => undefined,
  );
}

/**
 * Await `promise` or fail after `ms`. The original work may keep running;
 * its late rejection is swallowed so Node 22 cannot exit the process.
 */
export async function awaitOrTimeout<T>(promise: Promise<T>, ms: number, timeoutMessage: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(timeoutMessage)), ms);
    timer.unref?.();
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timer) clearTimeout(timer);
    ignoreLater(promise);
    ignoreLater(timeout);
  }
}

/** Resolve with `fallback` if `promise` has not settled in `ms`. The original work may keep running. */
export function raceTimeout<T>(promise: Promise<T>, ms: number, fallback: () => T): Promise<T> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (fn: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn();
    };
    const timer = setTimeout(() => {
      finish(() => resolve(fallback()));
      ignoreLater(promise);
    }, ms);
    timer.unref?.();
    promise.then(
      (value) => finish(() => resolve(value)),
      (err) => finish(() => reject(err)),
    );
  });
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    timer.unref?.();
  });
}
