import type { Orchestrator } from "../orchestrator.js";
import { canonicalizeDirectory } from "../allowlist.js";
import type { OrchestratedRun } from "../types.js";
import { summarizePcapFile } from "../pcap-summary.js";
import { buildPendingApproval, pendingCardText } from "./approval.js";
import { extractFilesystemPaths, expandUserPath, extractRoutableMessage, isLateDeviceWrap, routeChat, speakerLabel } from "./router.js";
import { ChatStore } from "./store.js";
import {
  earlyFlushGraceMs,
  isSpeakerSkipError,
  looksLikeLateToolJson,
  raceTimeout,
  sleep,
  speakerSkipLine,
  speakerTimeoutMs,
  timeoutErrorMessage,
} from "./timeout.js";
import type {
  ChatHeartbeatPayload,
  ChatMessage,
  ChatSuggestedAction,
  ChatThread,
  PendingApproval,
  RouteDecision,
  RouteSpeaker,
  RouterBackend,
  ThinkingPhase,
  WorkspaceHint,
} from "./types.js";

export interface ChatSendInput {
  threadId?: string;
  message: string;
  pin?: string;
  cwd?: string;
  prUrl?: string;
  repoUrl?: string;
  branch?: string;
  extraContext?: string;
  rounds?: number;
  wait?: boolean;
}

const DEBATE_SYSTEM = `You are one specialist in a round-table. Other models will see your reply and bounce off it.
Be concrete. Critique, improve, or dissent when others have already spoken.
Do not modify files in this turn — discussion only. If you cannot tell, say what you would check.`;

const SYNTHESIS_SYSTEM = `You are the closer for a multi-agent round-table.
Merge the debate into one recommended answer for the user. Resolve disagreements briefly.
If you are a Cursor builder with workspace access, apply the agreed plan by creating or editing files in the allowlisted cwd. Do not only outline steps.
If you cannot write files, return the merged recommendation as text and do not claim to have edited the repo.`;

const SYNTHESIS_PLAN_ONLY = `You are the closer for a multi-agent round-table.
Merge the debate into one recommended plan for the user. List proposed cwd, specialist, and commands (file writes, apt, Unity Hub, etc.).
Do not modify files, run shell installs, or claim to have installed anything. A human must click Approve before writes or host packages.`;

export class ChatService {
  readonly store = new ChatStore();
  private readonly queues = new Map<string, Promise<void>>();
  private readonly heartbeats = new Map<string, ReturnType<typeof setInterval>>();
  private readonly busyCount = new Map<string, number>();

  constructor(private readonly orchestrator: Orchestrator) {}

  isBusy(id: string): boolean {
    return (this.busyCount.get(id) ?? 0) > 0;
  }

  /** Thread plus in-flight flag for MCP/Late polling. `busy` is not persisted. */
  view(id: string): ChatThread {
    const thread = this.store.require(id);
    return { ...thread, busy: this.isBusy(id) };
  }

  list() {
    return this.store.list();
  }

  get(id: string): ChatThread {
    return this.store.require(id);
  }

  create(pin = "auto"): ChatThread {
    const thread = this.store.create(pin);
    this.emit(thread);
    return thread;
  }

  delete(id: string): boolean {
    const ok = this.store.delete(id);
    this.orchestrator.events.emit("chats", this.store.list());
    return ok;
  }

  setPin(id: string, pin: string): ChatThread {
    const thread = this.store.setPin(id, pin);
    this.emit(thread);
    return thread;
  }

  async send(input: ChatSendInput): Promise<ChatThread> {
    const message = input.message.trim();
    if (!message) throw new Error("message is required");
    const thread = input.threadId ? this.store.require(input.threadId) : this.store.create(input.pin ?? "auto");
    if (input.pin) this.store.setPin(thread.id, input.pin);
    this.store.append(thread.id, {
      role: "user",
      speaker: "user",
      label: "You",
      content: message,
      status: "finished",
    });
    this.emit(this.store.require(thread.id));

    this.markBusy(thread.id);
    const work = this.enqueue(thread.id, async () => {
      try {
        await this.respond(thread.id, { ...input, message });
      } finally {
        this.unmarkBusy(thread.id);
      }
    });
    if (input.wait === false) {
      void work.catch((error) => {
        this.fail(thread.id, error);
      });
      return this.view(thread.id);
    }
    await work;
    return this.view(thread.id);
  }

  async runAction(input: {
    threadId?: string;
    action: ChatSuggestedAction["action"];
    payload?: Record<string, unknown>;
  }): Promise<ChatThread | { ok: true; detail: unknown }> {
    if (input.action === "start_vllm") {
      const snap = this.orchestrator.localModels.snapshot();
      const requested = typeof input.payload?.modelId === "string" ? input.payload.modelId : undefined;
      const model =
        snap.models.find((m) => m.id === requested) ??
        snap.recommended.find((m) => m.downloaded && m.fits && m.newest) ??
        snap.recommended.find((m) => m.downloaded && m.fits) ??
        snap.recommended.find((m) => m.fits && m.newest && !m.cpuFeasible) ??
        snap.recommended.find((m) => m.fits) ??
        snap.models.find((m) => m.downloaded);
      if (!model) {
        throw new Error("No catalog model is available to start. Open Settings → Local models.");
      }
      if (!model.downloaded) {
        const job = this.orchestrator.localModels.download({ modelId: model.id });
        const thread = this.note(
          input.threadId,
          `Starting download of ${model.name} (${model.id}). Start vLLM after it finishes.`,
          {
            suggestedAction: { label: "Start recommended local model", action: "start_vllm", payload: { modelId: model.id } },
          },
        );
        return thread ?? { ok: true, detail: job };
      }
      const started = this.orchestrator.localModels.startVllmAsync({ modelId: model.id });
      const thread = this.note(
        input.threadId,
        started.status === "starting"
          ? `Starting local vLLM (${model.name}) on 127.0.0.1. Intel Docker often takes several minutes — stay on Settings → Local models. Chat Auto will use it when it is ready.`
          : `Local vLLM is running (${model.name} on 127.0.0.1:${started.vllm.port ?? ""}). Resend your question to use it.`,
      );
      return thread ?? { ok: true, detail: started };
    }
    if (input.action === "download_model") {
      const modelId = typeof input.payload?.modelId === "string" ? input.payload.modelId : "";
      if (!modelId) throw new Error("modelId required");
      const job = this.orchestrator.localModels.download({ modelId });
      const thread = this.note(input.threadId, `Download started for ${modelId}.`);
      return thread ?? { ok: true, detail: job };
    }
    if (input.action === "add_allowed_dir") {
      const path = typeof input.payload?.path === "string" ? input.payload.path.trim() : "";
      if (!path) throw new Error("path required");
      const dirs = this.orchestrator.allowlist.add(path);
      this.orchestrator.events.emit("catalog", await this.orchestrator.catalog());
      const granted = dirs.find((d) => d === canonicalizeDirectory(path)) ?? canonicalizeDirectory(path);
      const threadId = input.threadId;
      this.note(
        threadId,
        `Added ${granted} to the write allowlist. Cursor local can use it as cwd. Re-running your last request.`,
      );
      if (threadId) {
        const thread = this.store.require(threadId);
        const lastUser = [...thread.messages].reverse().find((m) => m.role === "user");
        if (lastUser?.content.trim()) {
          await this.respond(threadId, { message: lastUser.content, pin: thread.pin });
        }
        return this.store.require(threadId);
      }
      return { ok: true, detail: { allowedDirectories: dirs } };
    }
    throw new Error(`Unknown action "${input.action}"`);
  }

  async resolveApproval(input: {
    threadId: string;
    decision: "approve" | "reject";
    comment?: string;
  }): Promise<ChatThread> {
    const thread = this.store.require(input.threadId);
    const pending = thread.pendingApproval;
    if (!pending || pending.status !== "pending") {
      throw new Error("No pending actions to approve. Send an implement/install request first.");
    }
    const comment = input.comment?.trim() || undefined;
    if (input.decision === "reject") {
      this.store.setPendingApproval(thread.id, undefined);
      this.note(
        thread.id,
        `Rejected.${comment ? ` ${comment}` : ""} Nothing was written or installed. Cursor stayed plan-only.`,
      );
      return this.store.require(thread.id);
    }
    this.store.setPendingApproval(thread.id, { ...pending, status: "approved", comment });
    this.note(
      thread.id,
      `Approved.${comment ? ` ${comment}` : ""} Closer may write only inside the allowlisted cwd. Host packages (Unity Hub, apt, sudo) run only because you approved.`,
    );
    await this.runApprovedCloser(thread.id, pending);
    return this.store.require(thread.id);
  }

  private enqueue(threadId: string, fn: () => Promise<void>): Promise<void> {
    const prev = this.queues.get(threadId) ?? Promise.resolve();
    const next = prev.then(fn, fn);
    this.queues.set(threadId, next.catch(() => undefined));
    return next;
  }

  private async respond(threadId: string, input: ChatSendInput): Promise<void> {
    const thread = this.store.require(threadId);
    const catalog = await this.orchestrator.catalog();
    const backends: RouterBackend[] = (catalog.backends ?? []).map((b) => ({
      id: b.id,
      type: b.type,
      ready: b.ready,
      writesLocalFiles: b.writesLocalFiles,
      runtime: b.runtime,
      model: b.model,
      reason: b.reason,
      nickname: b.nickname,
      hasLogo: b.hasLogo,
    }));
    const vllm = catalog.localRuntime?.vllm;
    const workspace = this.resolveWorkspace(input.message, input.cwd);
    const pcap = await this.pcapAnalyzeContext(input.message, input.extraContext);
    const decision = routeChat({
      message: input.message,
      pin: input.pin ?? thread.pin,
      backends,
      specialists: (catalog.specialists ?? []).map((s) => ({ id: s.id, backend: s.backend })),
      vllmRunning: Boolean(vllm?.running),
      vllmModelId: vllm?.modelId,
      prior:
        thread.lastRunId && thread.lastBackend
          ? { runId: thread.lastRunId, agentId: thread.lastAgentId, backend: thread.lastBackend }
          : undefined,
      followUp: thread.messages.filter((m) => m.role === "assistant").length > 0,
      allowedDirectories: this.orchestrator.allowlist.list(),
      workspace,
      skipBackendIds: skippedBackendIds(thread),
    });
    const routedInput = { ...input, extraContext: pcap.extraContext };

    try {
    if (decision.kind === "control") {
      await this.handleControl(threadId, decision, input.message);
      return;
    }
    if (decision.kind === "error") {
      this.store.append(threadId, {
        role: "assistant",
        speaker: "orchestrator",
        label: "Orchestrator",
        content: decision.error ?? "Unable to route this message.",
        status: "error",
        phase: "control",
        chip: decision.chip,
        error: decision.error,
        suggestedAction: decision.suggestedAction,
      });
      this.emit(this.store.require(threadId));
      return;
    }
    if (decision.kind === "single") {
      await this.runSingle(threadId, decision, routedInput, Boolean(decision.needsApproval));
      return;
    }
    await this.runDebate(threadId, decision, routedInput, Boolean(decision.needsApproval));
    } finally {
      this.releaseTempPcaps(pcap.granted);
    }
  }

  private async handleControl(threadId: string, decision: RouteDecision, message: string): Promise<void> {
    const kind = decision.control ?? "hardware";
    let content = "";
    let suggestedAction: ChatSuggestedAction | undefined;
    try {
      if (kind === "hardware" || kind === "models") {
        const snap = this.orchestrator.localModels.snapshot();
        const hw = snap.hardware;
        const acc = (hw.accelerators ?? []).map((g) => `${g.name} · ${g.vramMiB} MiB`).join("; ") || "no discrete GPU";
        const rec = snap.recommended
          .map((m) => `${m.name} (${m.id}, ~${m.vramNeededMiB} MiB${m.parallel && m.parallel > 1 ? `, TP ${m.parallel}` : ""})`)
          .join("; ") || "none fit";
        content =
          `Hardware: ${acc}\nBackend: ${hw.primaryBackend} · ${hw.deviceCount ?? 0} device(s) · ${hw.totalVramMiB ?? hw.vramMiB} MiB total · RAM ${hw.ramMiB} MiB\n` +
          `vLLM: ${snap.vllm.running ? `running ${snap.vllm.modelId} on 127.0.0.1:${snap.vllm.port}` : "stopped"}\n` +
          `Recommended: ${rec}`;
        if (!snap.vllm.running && snap.recommended.find((m) => m.fits && m.newest)) {
          const start = snap.recommended.find((m) => m.fits && m.newest && !m.cpuFeasible) ?? snap.recommended.find((m) => m.fits);
          if (start) {
            suggestedAction = {
              label: "Start recommended local model",
              action: "start_vllm",
              payload: { modelId: start.id },
            };
          }
        }
      } else if (kind === "vllm_status") {
        const status = this.orchestrator.localModels.vllmStatus();
        const rows = status.instances ?? [];
        content = rows.length
          ? rows
              .map(
                (row) =>
                  `${row.backendId} · ${row.healthy ? "ready" : row.phase} · 127.0.0.1:${row.port ?? "?"} · ${row.modelId ?? ""}${row.image ? ` · ${row.image}` : ""}`,
              )
              .join("\n")
          : status.running
            ? `vLLM running pid ${status.pid} · 127.0.0.1:${status.port} · ${status.modelId}`
            : `vLLM stopped.${status.installHint ? ` ${status.installHint}` : ""}`;
        if (!status.running && !rows.some((row) => row.healthy || row.running)) {
          suggestedAction = { label: "Start recommended local model", action: "start_vllm" };
        }
      } else if (kind === "start_vllm") {
        await this.runAction({ threadId, action: "start_vllm" });
        return;
      } else if (kind === "stop_vllm") {
        const status = this.orchestrator.localModels.stopVllm();
        content = status.running ? "Stop requested but process still listed as running." : "vLLM stopped.";
      } else if (kind === "allowlist") {
        const dirs = this.orchestrator.allowlist.list();
        content = `Write allowlist (${dirs.length}):\n${dirs.map((d) => `• ${d}`).join("\n") || "(empty)"}\nDefault cwd: ${this.orchestrator.defaultCwd()}`;
      } else {
        content = `Unhandled control "${kind}" for: ${message}`;
      }
    } catch (error) {
      content = error instanceof Error ? error.message : String(error);
      suggestedAction = { label: "Start recommended local model", action: "start_vllm" };
    }
    this.store.append(threadId, {
      role: "assistant",
      speaker: "orchestrator",
      label: "Orchestrator",
      content,
      status: "finished",
      phase: "control",
      chip: "orchestrator",
      suggestedAction,
    });
    this.emit(this.store.require(threadId));
  }

  private async runSingle(
    threadId: string,
    decision: RouteDecision,
    input: ChatSendInput,
    holdWrites: boolean,
  ): Promise<void> {
    const speaker = decision.speakers?.[0];
    if (!speaker) throw new Error("Router returned a single turn without a speaker");
    const placeholder = this.beginSpeaker(threadId, speaker, decision, "single", "waiting");
    try {
      const history = this.historyFor(threadId, placeholder.id);
      const onDelta = this.deltaHandler(threadId, placeholder.id);
      const timeoutMs = speakerTimeoutMs();
      const work =
        decision.followUpRunId && !holdWrites
          ? this.orchestrator.followUp({
              runId: decision.followUpRunId,
              message: input.message,
              wait: true,
              timeoutMs,
              onDelta,
            })
          : this.orchestrator.dispatch({
              specialist: speaker.specialist,
              backend: speaker.backendId,
              task: holdWrites
                ? `${input.message}\n\nPlan only: do not write files or install host packages (Unity Hub, apt, sudo). List proposed cwd, specialist, and commands.`
                : input.message,
              cwd: this.cwdForDispatch(decision, speaker, input.cwd),
              prUrl: input.prUrl,
              repoUrl: input.repoUrl,
              branch: input.branch,
              extraContext: this.mergeContext(input.extraContext, history),
              wait: true,
              timeoutMs,
              mode: this.dispatchMode(decision, speaker, holdWrites, false),
              onDelta,
            });
      const run = await this.dispatchTimed(threadId, work, timeoutMs, speaker.label, placeholder.id);
      if (!run) {
        if (holdWrites) this.holdForApproval(threadId, decision, input, input.message);
        return;
      }
      this.applyRun(threadId, placeholder.id, run, decision);
      if (holdWrites) {
        this.holdForApproval(threadId, decision, input, run.text?.trim() || run.error || input.message);
      }
    } catch (error) {
      this.patchError(threadId, placeholder.id, error, decision.suggestedAction);
      if (holdWrites) {
        this.holdForApproval(threadId, decision, input, error instanceof Error ? error.message : String(error));
      }
    }
  }

  private async runDebate(
    threadId: string,
    decision: RouteDecision,
    input: ChatSendInput,
    holdWrites: boolean,
  ): Promise<void> {
    const speakers = decision.speakers ?? [];
    const closer = decision.closer ?? speakers[speakers.length - 1];
    if (speakers.length === 0 || !closer) throw new Error("Router returned a debate without speakers");
    const lateWrap = isLateDeviceWrap(input.message);
    const rounds = lateWrap ? 1 : Math.min(Math.max(input.rounds ?? decision.rounds ?? 2, 1), 3);
    const transcript: Array<{ label: string; speaker: string; text: string }> = [];
    const history = this.historyFor(threadId);
    let lateJson = false;

    for (let round = 1; round <= rounds; round++) {
      const snapshot = [...transcript];
      const turns = speakers.map((speaker) =>
        this.debateTurn({
          threadId,
          speaker,
          decision,
          input,
          history,
          round,
          rounds,
          transcript: snapshot,
        }).then((text) => {
          if (text) transcript.push({ label: speaker.label, speaker: speaker.backendId, text });
          if (looksLikeLateToolJson(text)) lateJson = true;
          return text;
        }),
      );
      const all = Promise.all(turns);
      if (lateWrap) {
        while (true) {
          const winner = await Promise.race([all.then(() => "all" as const), sleep(80).then(() => "tick" as const)]);
          if (winner === "all") break;
          if (lateJson) {
            const grace = earlyFlushGraceMs();
            if (grace > 0) await sleep(grace);
            this.skipLeftoverThinking(threadId, timeoutErrorMessage("speaker", speakerTimeoutMs()));
            break;
          }
        }
      } else {
        await all;
      }
      if (lateWrap && lateJson) break;
    }

    if (lateWrap) {
      this.skipLeftoverThinking(threadId, timeoutErrorMessage("speaker", speakerTimeoutMs()));
      if (holdWrites) {
        this.holdForApproval(threadId, decision, input, transcript.map((t) => t.text).join("\n\n") || input.message);
      }
      return;
    }

    const closerFailed = this.store.require(threadId).messages.some(
      (m) => m.role === "assistant" && m.speaker === closer.backendId && m.status === "error",
    );
    if (closerFailed) {
      if (holdWrites) {
        this.holdForApproval(threadId, decision, input, transcript.map((t) => t.text).join("\n\n") || input.message);
      }
      return;
    }

    const placeholder = this.beginSpeaker(threadId, closer, decision, "synthesis", "waiting");
    try {
      const timeoutMs = speakerTimeoutMs();
      const run = await this.dispatchTimed(
        threadId,
        this.orchestrator.dispatch({
          specialist: closer.specialist,
          backend: closer.backendId,
          task: synthesisPrompt(input.message, transcript, closer, holdWrites),
          cwd: this.cwdForDispatch(decision, closer, input.cwd),
          prUrl: input.prUrl,
          repoUrl: input.repoUrl,
          branch: input.branch,
          extraContext: this.mergeContext(input.extraContext, history),
          wait: true,
          timeoutMs,
          mode: this.dispatchMode(decision, closer, holdWrites, true),
          onDelta: this.deltaHandler(threadId, placeholder.id),
        }),
        timeoutMs,
        closer.label,
        placeholder.id,
      );
      if (run) this.applyRun(threadId, placeholder.id, run, decision);
      if (holdWrites) {
        this.holdForApproval(threadId, decision, input, run?.text?.trim() || transcript.map((t) => t.text).join("\n\n") || input.message);
      }
    } catch (error) {
      this.patchError(threadId, placeholder.id, error, decision.suggestedAction);
      if (holdWrites) {
        this.holdForApproval(threadId, decision, input, transcript.map((t) => t.text).join("\n\n") || input.message);
      }
    }
  }

  private async debateTurn(opts: {
    threadId: string;
    speaker: RouteSpeaker;
    decision: RouteDecision;
    input: ChatSendInput;
    history: Array<{ role: "user" | "assistant"; content: string }>;
    round: number;
    rounds: number;
    transcript: Array<{ label: string; speaker: string; text: string }>;
  }): Promise<string> {
    const { threadId, speaker, decision, input, history, round, rounds, transcript } = opts;
    const placeholder = this.beginSpeaker(threadId, speaker, decision, "debate", "debating", round);
    const task = debateTurnPrompt({
      user: input.message,
      round,
      rounds,
      speaker,
      transcript,
    });
    const timeoutMs = speakerTimeoutMs();
    try {
      const run = await this.dispatchTimed(
        threadId,
        this.orchestrator.dispatch({
          specialist: speaker.specialist,
          backend: speaker.backendId,
          task,
          cwd: this.cwdForDispatch(decision, speaker, input.cwd),
          prUrl: input.prUrl,
          repoUrl: input.repoUrl,
          branch: input.branch,
          extraContext: this.mergeContext(input.extraContext, history),
          wait: true,
          timeoutMs,
          mode: speaker.writesLocalFiles ? "plan" : undefined,
          onDelta: this.deltaHandler(threadId, placeholder.id),
        }),
        timeoutMs,
        speaker.label,
        placeholder.id,
      );
      if (!run) return "";
      this.applyRun(threadId, placeholder.id, run, decision, { systemNote: DEBATE_SYSTEM });
      if (run.status === "error") return "";
      return run.text?.trim() || "";
    } catch (error) {
      this.patchError(threadId, placeholder.id, error);
      return "";
    }
  }

  /** Wait for a dispatch, or skip this speaker when it exceeds the per-speaker cap. */
  private async dispatchTimed(
    threadId: string,
    work: Promise<OrchestratedRun>,
    timeoutMs: number,
    label: string,
    messageId: string,
  ): Promise<OrchestratedRun | undefined> {
    const timedOut = { current: false };
    const run = await raceTimeout(work, timeoutMs, () => {
      timedOut.current = true;
      return {
        id: "",
        specialist: "",
        backend: "",
        status: "error" as const,
        prompt: "",
        text: "",
        error: timeoutErrorMessage(label, timeoutMs),
        createdAt: Date.now(),
        updatedAt: Date.now(),
        history: [],
      };
    });
    if (timedOut.current) {
      this.patchError(threadId, messageId, new Error(run.error ?? timeoutErrorMessage(label, timeoutMs)));
      return undefined;
    }
    return run;
  }

  private markBusy(threadId: string): void {
    this.busyCount.set(threadId, (this.busyCount.get(threadId) ?? 0) + 1);
  }

  private unmarkBusy(threadId: string): void {
    const n = (this.busyCount.get(threadId) ?? 1) - 1;
    if (n <= 0) this.busyCount.delete(threadId);
    else this.busyCount.set(threadId, n);
    try {
      this.emit(this.store.require(threadId));
    } catch {
      /* thread deleted */
    }
  }

  private applyRun(
    threadId: string,
    messageId: string,
    run: OrchestratedRun,
    decision: RouteDecision,
    _opts?: { systemNote?: string },
  ): void {
    const error = run.status === "error" ? run.error : undefined;
    const current = this.store.require(threadId).messages.find((m) => m.id === messageId);
    const label = current?.label || current?.nickname || current?.speaker || "Speaker";
    const content = error
      ? isSpeakerSkipError(error)
        ? speakerSkipLine(label, error)
        : error
      : run.text?.trim() || "";
    this.store.patchMessage(threadId, messageId, {
      content,
      status: error ? "error" : "finished",
      runId: error ? undefined : run.id,
      agentId: error ? undefined : run.agentId,
      error: error ? content : undefined,
      chip: decision.chip,
      thinkingPhase: undefined,
      thinkingStartedAt: undefined,
      suggestedAction: suggestedForRunError(run.error, decision),
    });
    this.stopHeartbeatIfIdle(threadId);
    this.emit(this.store.require(threadId));
  }

  private skipLeftoverThinking(threadId: string, reason: string): void {
    const thread = this.store.get(threadId);
    if (!thread) return;
    for (const msg of thread.messages) {
      if (msg.status !== "thinking" && msg.status !== "streaming") continue;
      this.patchError(threadId, msg.id, new Error(reason));
    }
  }

  private patchError(
    threadId: string,
    messageId: string,
    error: unknown,
    suggestedAction?: ChatSuggestedAction,
  ): void {
    const raw = error instanceof Error ? error.message : String(error);
    const current = this.store.require(threadId).messages.find((m) => m.id === messageId);
    const label = current?.label || current?.nickname || current?.speaker || "Speaker";
    const text = isSpeakerSkipError(raw) || /skipped/i.test(raw) ? speakerSkipLine(label, raw) : raw;
    this.store.patchMessage(threadId, messageId, {
      content: text,
      status: "error",
      error: text,
      thinkingPhase: undefined,
      thinkingStartedAt: undefined,
      suggestedAction: suggestedAction ?? suggestedForRunError(raw),
    });
    this.stopHeartbeatIfIdle(threadId);
    this.emit(this.store.require(threadId));
  }

  private fail(threadId: string, error: unknown): void {
    const text = error instanceof Error ? error.message : String(error);
    this.store.append(threadId, {
      role: "assistant",
      speaker: "orchestrator",
      label: "Orchestrator",
      content: text,
      status: "error",
      error: text,
      phase: "control",
      suggestedAction: suggestedForRunError(text),
    });
    this.emit(this.store.require(threadId));
  }

  private note(
    threadId: string | undefined,
    content: string,
    extra: Partial<ChatMessage> = {},
  ): ChatThread | undefined {
    if (!threadId) return undefined;
    this.store.append(threadId, {
      role: "assistant",
      speaker: "orchestrator",
      label: "Orchestrator",
      content,
      status: "finished",
      phase: "control",
      chip: "orchestrator",
      ...extra,
    });
    const thread = this.store.require(threadId);
    this.emit(thread);
    return thread;
  }

  private dispatchMode(
    decision: RouteDecision,
    speaker: RouteSpeaker,
    holdWrites: boolean,
    isCloser: boolean,
  ): "plan" | "agent" | undefined {
    if (!speaker.writesLocalFiles) return undefined;
    if (holdWrites) return "plan";
    if (isCloser && decision.needsWrites) return "agent";
    if (!isCloser) return "plan";
    return decision.needsWrites ? "agent" : undefined;
  }

  private writerForApproval(decision: RouteDecision): RouteSpeaker {
    if (decision.closer?.writesLocalFiles) return decision.closer;
    const fromSpeakers = decision.speakers?.find((s) => s.writesLocalFiles);
    if (fromSpeakers) return fromSpeakers;
    return {
      backendId: "cursor-local",
      specialist: "builder",
      label: "Cursor local",
      writesLocalFiles: true,
    };
  }

  private holdForApproval(
    threadId: string,
    decision: RouteDecision,
    input: ChatSendInput,
    planText: string,
  ): void {
    const writer = this.writerForApproval(decision);
    let cwd = input.cwd ?? decision.cwd ?? this.orchestrator.defaultCwd();
    try {
      cwd = this.orchestrator.allowlist.assertCwd(cwd);
    } catch {
      try {
        cwd = this.orchestrator.allowlist.assertCwd(this.orchestrator.defaultCwd());
      } catch {
        cwd = this.orchestrator.defaultCwd();
      }
    }
    const pending = buildPendingApproval({
      decision: { ...decision, closer: writer, cwd },
      userMessage: input.message,
      planText,
      cwd,
      extraContext: input.extraContext,
      prUrl: input.prUrl,
      repoUrl: input.repoUrl,
      branch: input.branch,
      pin: input.pin,
    });
    this.store.setPendingApproval(threadId, pending);
    this.store.append(threadId, {
      role: "assistant",
      speaker: "orchestrator",
      label: "Pending actions",
      content: pendingCardText(pending),
      status: "finished",
      phase: "approval",
      chip: "approval",
    });
    this.emit(this.store.require(threadId));
  }

  private async runApprovedCloser(threadId: string, pending: PendingApproval): Promise<void> {
    const cwd = this.orchestrator.allowlist.assertCwd(pending.cwd ?? this.orchestrator.defaultCwd());
    const placeholder = this.beginSpeaker(
      threadId,
      {
        backendId: pending.backendId,
        specialist: pending.specialist,
        label: pending.label,
        writesLocalFiles: true,
      },
      { kind: "single", pin: pending.pin ?? "auto", intent: "code", chip: pending.label, needsWrites: true },
      "single",
      "waiting",
    );
    const hostNote = pending.systemWideInstall
      ? `\n\nThe user approved host installs (Unity Hub, apt, sudo) in addition to writes inside ${cwd}. Do not install anything outside that approval.`
      : `\n\nWrites only inside ${cwd}. Do not install host packages (Unity Hub, apt, sudo).`;
    try {
      const run = await this.orchestrator.dispatch({
        specialist: pending.specialist,
        backend: pending.backendId,
        task: `${pending.userMessage}\n\nApproved plan:\n${pending.summary}${hostNote}`,
        cwd,
        prUrl: pending.prUrl,
        repoUrl: pending.repoUrl,
        branch: pending.branch,
        extraContext: pending.extraContext,
        wait: true,
        mode: "agent",
        onDelta: this.deltaHandler(threadId, placeholder.id),
      });
      this.applyRun(
        threadId,
        placeholder.id,
        run,
        {
          kind: "single",
          pin: pending.pin ?? "auto",
          intent: "code",
          chip: pending.label,
          needsWrites: true,
        },
      );
    } catch (error) {
      this.patchError(threadId, placeholder.id, error);
    } finally {
      const thread = this.store.require(threadId);
      if (thread.pendingApproval?.id === pending.id) {
        this.store.setPendingApproval(threadId, undefined);
      }
    }
  }

  private cwdForDispatch(decision: RouteDecision, speaker: RouteSpeaker, fallback?: string): string | undefined {
    if (!speaker.writesLocalFiles) return undefined;
    const candidate = decision.cwd ?? fallback;
    if (!candidate) return fallback;
    return this.orchestrator.allowlist.assertCwd(candidate);
  }

  private async pcapAnalyzeContext(
    message: string,
    extra?: string,
  ): Promise<{ extraContext?: string; granted: string[] }> {
    const temp = this.orchestrator.tempAnalyze;
    if (!temp) return { extraContext: extra, granted: [] };
    const granted: string[] = [];
    const summaries: string[] = [];
    const turn = extractRoutableMessage(message);
    const candidates = new Set<string>([...extractFilesystemPaths(turn), ...temp.list().map((g) => g.path)]);
    for (const raw of candidates) {
      const expanded = expandUserPath(raw);
      if (!temp.has(expanded) && !temp.has(raw)) continue;
      if (!turn.includes(raw) && !turn.includes(expanded)) continue;
      try {
        const path = temp.list().find((g) => g.path === expanded || g.path.endsWith(raw) || expanded.endsWith(g.path))?.path ?? expanded;
        if (granted.includes(path)) continue;
        granted.push(path);
        summaries.push(await summarizePcapFile(path));
      } catch {
        /* skip unreadable grants */
      }
    }
    if (summaries.length === 0) return { extraContext: extra, granted };
    const block = `Temporary pcap analyze grant (read-only, payloads omitted):\n${summaries.join("\n\n")}`;
    const extraContext = extra?.trim() ? `${extra.trim()}\n\n${block}` : block;
    return { extraContext, granted };
  }

  private releaseTempPcaps(granted: string[]): void {
    const temp = this.orchestrator.tempAnalyze;
    if (!temp) return;
    for (const path of granted) {
      temp.remove(path);
    }
  }

  private resolveWorkspace(message: string, explicitCwd?: string): WorkspaceHint | undefined {
    const raw = explicitCwd?.trim() || extractFilesystemPaths(extractRoutableMessage(message))[0];
    if (!raw) return undefined;
    const expanded = expandUserPath(raw);
    try {
      const real = canonicalizeDirectory(expanded);
      const cwd = this.orchestrator.allowlist.tryCwd(real);
      return { path: raw, allowed: Boolean(cwd), cwd };
    } catch {
      return { path: raw, allowed: false, missing: true };
    }
  }

  private historyFor(threadId: string, excludeMessageId?: string): Array<{ role: "user" | "assistant"; content: string }> {
    const thread = this.store.require(threadId);
    const turns: Array<{ role: "user" | "assistant"; content: string }> = [];
    for (const msg of thread.messages) {
      if (msg.id === excludeMessageId) continue;
      if (!msg.content.trim() || msg.status === "streaming" || msg.status === "thinking") continue;
      if (msg.role === "user") turns.push({ role: "user", content: msg.content });
      else turns.push({ role: "assistant", content: `[${msg.label}]\n${msg.content}` });
    }
    return turns.slice(-16);
  }

  private mergeContext(extra: string | undefined, history: Array<{ role: string; content: string }>): string | undefined {
    const prior = history
      .map((t) => `${t.role === "user" ? "User" : "Assistant"}: ${t.content}`)
      .join("\n\n");
    const parts = [extra?.trim(), prior ? `Prior chat:\n${prior}` : ""].filter(Boolean);
    return parts.length ? parts.join("\n\n") : undefined;
  }

  private emit(thread: ChatThread): void {
    this.orchestrator.events.emit("chat", this.view(thread.id));
    this.orchestrator.events.emit("chats", this.store.list());
  }

  private beginSpeaker(
    threadId: string,
    speaker: RouteSpeaker,
    decision: RouteDecision,
    phase: ChatMessage["phase"],
    thinkingPhase: ThinkingPhase,
    round?: number,
  ): ChatMessage {
    const placeholder = this.store.append(threadId, {
      role: "assistant",
      speaker: speaker.backendId,
      label: speaker.label,
      content: "",
      status: "thinking",
      phase,
      round,
      chip: decision.chip,
      thinkingPhase,
      thinkingStartedAt: Date.now(),
      nickname: speaker.nickname,
      hasLogo: speaker.hasLogo,
      logoUrl: speaker.logoUrl,
    });
    this.emit(this.store.require(threadId));
    this.startHeartbeat(threadId);
    return placeholder;
  }

  private deltaHandler(threadId: string, messageId: string): (delta: string) => void {
    let assembled = "";
    let lastEmit = 0;
    return (delta: string) => {
      assembled += delta;
      const now = Date.now();
      this.store.patchMessage(
        threadId,
        messageId,
        {
          content: assembled,
          status: "streaming",
          thinkingPhase: "streaming",
        },
        false,
      );
      if (now - lastEmit >= 80) {
        lastEmit = now;
        this.emit(this.store.require(threadId));
      }
    };
  }

  private thinkingRows(thread: ChatThread): ChatHeartbeatPayload["thinking"] {
    return thread.messages
      .filter((m) => m.status === "thinking" || m.status === "streaming")
      .map((m) => ({
        id: m.id,
        label: m.label,
        speaker: m.speaker,
        status: m.status ?? "thinking",
        thinkingPhase: m.thinkingPhase ?? (m.status === "streaming" ? "streaming" : "waiting"),
        thinkingStartedAt: m.thinkingStartedAt ?? m.createdAt,
      }));
  }

  private startHeartbeat(threadId: string): void {
    if (this.heartbeats.has(threadId)) return;
    const tick = () => {
      const thread = this.store.get(threadId);
      if (!thread) {
        this.stopHeartbeat(threadId);
        return;
      }
      const thinking = this.thinkingRows(thread);
      if (thinking.length === 0) {
        this.stopHeartbeat(threadId);
        return;
      }
      const payload: ChatHeartbeatPayload = { threadId, now: Date.now(), thinking };
      this.orchestrator.events.emit("chat-heartbeat", payload);
    };
    tick();
    const timer = setInterval(tick, 1000);
    timer.unref?.();
    this.heartbeats.set(threadId, timer);
  }

  private stopHeartbeat(threadId: string): void {
    const timer = this.heartbeats.get(threadId);
    if (!timer) return;
    clearInterval(timer);
    this.heartbeats.delete(threadId);
  }

  private stopHeartbeatIfIdle(threadId: string): void {
    const thread = this.store.get(threadId);
    if (!thread || this.thinkingRows(thread).length === 0) this.stopHeartbeat(threadId);
  }
}

function debateTurnPrompt(input: {
  user: string;
  round: number;
  rounds: number;
  speaker: RouteSpeaker;
  transcript: Array<{ label: string; text: string }>;
}): string {
  const prior =
    input.transcript.length === 0
      ? "You speak first this round. Give an independent take."
      : `Round-table so far:\n${input.transcript.map((t) => `### ${t.label}\n${t.text}`).join("\n\n")}\n\nCritique, improve, or dissent. Quote disagreements clearly.`;
  return `${DEBATE_SYSTEM}

You are "${input.speaker.label}" (${input.speaker.backendId}). Round ${input.round} of ${input.rounds}.

User request:
${input.user}

${prior}`;
}

function synthesisPrompt(
  user: string,
  transcript: Array<{ label: string; text: string }>,
  closer: RouteSpeaker,
  planOnly = false,
): string {
  const body = transcript.map((t) => `### ${t.label}\n${t.text}`).join("\n\n") || "(no prior turns)";
  const system = planOnly ? SYNTHESIS_PLAN_ONLY : SYNTHESIS_SYSTEM;
  const closerHint = planOnly
    ? "Write the merged plan only. Do not write files or install host packages."
    : closer.writesLocalFiles
      ? "Then apply the agreed plan by writing files in the workspace cwd. Do not only outline steps."
      : "Text only — do not claim to have edited files.";
  return `${system}

You are "${closer.label}" (${closer.backendId}), the closer.

User request:
${user}

Round-table transcript:
${body}

Write the merged recommendation. ${closerHint}`;
}

function skippedBackendIds(thread: ChatThread): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const msg of thread.messages) {
    if (msg.role !== "assistant" || msg.status !== "error" || !msg.speaker) continue;
    if (msg.speaker === "orchestrator" || msg.speaker === "user") continue;
    const text = `${msg.error ?? ""} ${msg.content ?? ""}`;
    if (!isSpeakerSkipError(text) && !/skipped/i.test(text)) continue;
    if (seen.has(msg.speaker)) continue;
    seen.add(msg.speaker);
    ids.push(msg.speaker);
  }
  return ids;
}

function suggestedForRunError(error?: string, decision?: RouteDecision): ChatSuggestedAction | undefined {
  if (decision?.suggestedAction) return decision.suggestedAction;
  if (!error) return undefined;
  if (/vllm not running|vllm not reachable|not running at/i.test(error)) {
    return { label: "Start recommended local model", action: "start_vllm" };
  }
  if (/CURSOR_API_KEY/i.test(error)) {
    return { label: "Open backends", action: "open_settings", payload: { page: "backends" } };
  }
  if (/not inside an allowed directory/i.test(error)) {
    return { label: "Open allowlist", action: "open_settings", payload: { page: "allowlist" } };
  }
  if (/no longer available|not found for API version|models\/gemini-/i.test(error)) {
    return { label: "Open backends", action: "open_settings", payload: { page: "backends" } };
  }
  if (/GEMINI_API_KEY|GOOGLE_API_KEY|not ready/i.test(error)) {
    return { label: "Open backends", action: "open_settings", payload: { page: "backends" } };
  }
  return undefined;
}

export function backendChip(backends: RouterBackend[], vllmModelId?: string): string {
  return backends.map((b) => speakerLabel(b, vllmModelId)).join(" · ");
}
