import type { Orchestrator } from "../orchestrator.js";
import { canonicalizeDirectory } from "../allowlist.js";
import type { OrchestratedRun } from "../types.js";
import { summarizePcapFile } from "../pcap-summary.js";
import { applyParsedFiles, APPLY_PATCH_INSTRUCTIONS, parseOrchestratorFiles } from "./apply-patch.js";
import { buildPendingApproval, pendingCardText } from "./approval.js";
import { latePlaybookPatchFiles, LATE_JSON_SYSTEM, withOrchestratorFilesFence } from "./late-wrap.js";
import { extractFilesystemPaths, expandUserPath, extractRoutableMessage, isLateDeviceWrap, routeChat, speakerLabel } from "./router.js";
import { ChatStore } from "./store.js";
import {
    earlyFlushGraceMs,
    isCursorSpeaker,
    isSpeakerSkipError,
    LEFTOVER_SPEAKER_SKIP,
    looksLikeLateToolJson,
    raceTimeout,
    sleep,
    speakerErrorText,
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
    this.stopHeartbeat(id);
    this.busyCount.delete(id);
    const ok = this.store.delete(id);
    this.orchestrator.events.emit("chats", this.store.list());
    return ok;
  }

  private stillOpen(threadId: string): boolean {
    return Boolean(this.store.get(threadId));
  }

  private tryView(id: string): ChatThread | undefined {
    if (!this.stillOpen(id)) return undefined;
    try {
      return this.view(id);
    } catch {
      return undefined;
    }
  }

  setPin(id: string, pin: string): ChatThread {
    const thread = this.store.setPin(id, pin);
    this.emit(thread);
    return thread;
  }

  setWorkspaceDir(id: string, path: string): ChatThread {
    const thread = this.store.setWorkspaceDir(id, path);
    this.emit(thread);
    return thread;
  }

  async send(input: ChatSendInput): Promise<ChatThread> {
    const message = input.message.trim();
    if (!message) throw new Error("message is required");
    const thread = input.threadId ? this.store.require(input.threadId) : this.store.create(input.pin ?? "auto");
    if (input.pin) this.store.setPin(thread.id, input.pin);
    if (input.cwd) {
      const granted = this.orchestrator.allowlist.tryCwd(input.cwd);
      if (granted) this.store.setWorkspaceDir(thread.id, granted);
    }
    this.store.append(thread.id, {
      role: "user",
      speaker: "user",
      label: "You",
      content: message,
      status: "finished",
    });
    this.emit(this.store.get(thread.id));

    this.markBusy(thread.id);
    const work = this.enqueue(thread.id, async () => {
      try {
        if (!this.stillOpen(thread.id)) return;
        const latest = this.store.get(thread.id);
        if (!latest) return;
        await this.respond(thread.id, { ...input, message, cwd: input.cwd ?? latest.workspaceDir });
      } finally {
        this.unmarkBusy(thread.id);
      }
    });
    if (input.wait === false) {
      void work.catch((error) => {
        try {
          this.fail(thread.id, error);
        } catch {
          /* thread gone */
        }
      });
      return this.tryView(thread.id) ?? { ...thread, busy: this.isBusy(thread.id) };
    }
    await work;
    return this.tryView(thread.id) ?? { ...thread, busy: false };
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
      void this.orchestrator.catalog().then(
        (catalog) => this.orchestrator.events.emit("catalog", catalog),
        () => undefined,
      );
      const granted = dirs.find((d) => d === canonicalizeDirectory(path)) ?? canonicalizeDirectory(path);
      const threadId = input.threadId;
      this.note(
        threadId,
        `Added ${granted} to the write allowlist. Cursor local can use it as cwd. Re-running your last request.`,
      );
      if (threadId && this.stillOpen(threadId)) {
        const thread = this.store.get(threadId);
        const lastUser = thread ? [...thread.messages].reverse().find((m) => m.role === "user") : undefined;
        if (lastUser?.content.trim()) {
          await this.respond(threadId, { message: lastUser.content, pin: thread!.pin });
        }
        return this.tryView(threadId) ?? { ok: true, detail: { allowedDirectories: dirs } };
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
    return this.tryView(thread.id) ?? thread;
  }

  private enqueue(threadId: string, fn: () => Promise<void>): Promise<void> {
    const prev = this.queues.get(threadId) ?? Promise.resolve();
    const next = prev.then(fn, fn);
    this.queues.set(threadId, next.catch(() => undefined));
    return next;
  }

  private async respond(threadId: string, input: ChatSendInput): Promise<void> {
    const thread = this.store.get(threadId);
    if (!thread) return;
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
    const vllmBackendIds = [
      ...new Set(
        (vllm?.instances ?? [])
          .filter((row: { healthy?: boolean; running?: boolean; backendId?: string }) => row.healthy || row.running)
          .map((row: { backendId?: string }) => row.backendId)
          .filter((id: string | undefined): id is string => Boolean(id)),
      ),
    ];
    if (!vllmBackendIds.length && vllm?.backendId && (vllm.healthy || vllm.running)) {
      vllmBackendIds.push(vllm.backendId);
    }
    const decision = routeChat({
      message: input.message,
      pin: input.pin ?? thread.pin,
      backends,
      specialists: (catalog.specialists ?? []).map((s) => ({ id: s.id, backend: s.backend })),
      vllmRunning: Boolean(vllm?.running),
      vllmModelId: vllm?.modelId,
      vllmBackendIds,
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
      if (!this.stillOpen(threadId)) return;
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
      this.emit(this.store.get(threadId));
      return;
    }
    if (decision.kind === "single") {
      await this.runSingle(threadId, decision, routedInput, Boolean(decision.needsApproval));
      return;
    }
    await this.runDebate(threadId, decision, routedInput, Boolean(decision.needsApproval));
    } catch (error) {
      if (!this.stillOpen(threadId)) return;
      throw error;
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
    if (!this.stillOpen(threadId)) return;
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
    this.emit(this.store.get(threadId));
  }

  private async runSingle(
    threadId: string,
    decision: RouteDecision,
    input: ChatSendInput,
    holdWrites: boolean,
  ): Promise<void> {
    if (!this.stillOpen(threadId)) return;
    const speaker = decision.speakers?.[0];
    if (!speaker) throw new Error("Router returned a single turn without a speaker");
    const placeholder = this.beginSpeaker(threadId, speaker, decision, "single", "waiting");
    try {
      const history = this.historyFor(threadId, placeholder.id);
      const onDelta = this.deltaHandler(threadId, placeholder.id);
      const timeoutMs = speakerTimeoutMs(speaker.backendId);
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
                ? `${input.message}\n\nPlan only: do not write files or install host packages (Unity Hub, apt, sudo). List proposed cwd, specialist, and commands.${decision.applyPatch ? `\n\n${APPLY_PATCH_INSTRUCTIONS}` : ""}`
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
    if (!this.stillOpen(threadId)) return;
    const speakers = decision.speakers ?? [];
    const closer = decision.closer ?? speakers[speakers.length - 1];
    if (speakers.length === 0 || !closer) throw new Error("Router returned a debate without speakers");
    const lateWrap = isLateDeviceWrap(input.message);
    const rounds = lateWrap ? 1 : Math.min(Math.max(input.rounds ?? decision.rounds ?? 2, 1), 3);
    const transcript: Array<{ label: string; speaker: string; text: string }> = [];
    const history = this.historyFor(threadId);
    let lateJson = false;

    const runWave = async (wave: RouteSpeaker[], round: number): Promise<void> => {
      if (wave.length === 0) return;
      const snapshot = [...transcript];
      const turns = wave.map((speaker) =>
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
            this.skipLeftoverThinking(threadId, LEFTOVER_SPEAKER_SKIP);
            break;
          }
        }
      } else {
        await all;
      }
    };

    for (let round = 1; round <= rounds; round++) {
      if (lateWrap) {
        const firstWave = speakers.filter((s) => !isCursorSpeaker(s.backendId));
        const laterWave = speakers.filter((s) => isCursorSpeaker(s.backendId));
        await runWave(firstWave.length ? firstWave : speakers, round);
        if (!lateJson) await runWave(firstWave.length ? laterWave : [], round);
      } else {
        await runWave(speakers, round);
      }
      if (lateWrap && lateJson) break;
    }

    if (lateWrap) {
      this.skipLeftoverThinking(threadId, LEFTOVER_SPEAKER_SKIP);
      if (holdWrites) {
        const turn = extractRoutableMessage(input.message);
        const files = latePlaybookPatchFiles(
          turn,
          transcript.map((t) => t.text),
        );
        const plan = withOrchestratorFilesFence(
          transcript.map((t) => t.text).join("\n\n") || input.message,
          files,
        );
        this.holdForApproval(
          threadId,
          { ...decision, applyPatch: Boolean(decision.applyPatch || files.length) },
          input,
          plan,
        );
      }
      return;
    }

    const debateThread = this.store.get(threadId);
    if (!debateThread) return;
    const closerFailed = debateThread.messages.some(
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
      const timeoutMs = speakerTimeoutMs(closer.backendId);
      const run = await this.dispatchTimed(
        threadId,
        this.orchestrator.dispatch({
          specialist: closer.specialist,
          backend: closer.backendId,
          task: synthesisPrompt(input.message, transcript, closer, holdWrites, Boolean(decision.applyPatch)),
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
    if (!this.stillOpen(threadId)) return "";
    const placeholder = this.beginSpeaker(threadId, speaker, decision, "debate", "debating", round);
    const task = debateTurnPrompt({
      user: input.message,
      round,
      rounds,
      speaker,
      transcript,
    });
    const timeoutMs = speakerTimeoutMs(speaker.backendId);
    try {
      const run = await this.dispatchTimed(
        threadId,
        this.orchestrator.dispatch({
          specialist: speaker.specialist,
          backend: speaker.backendId,
          task,
          cwd: this.cwdForDispatch(decision, speaker, input.cwd, isLateDeviceWrap(input.message)),
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
      this.applyRun(threadId, placeholder.id, run, decision, {
        systemNote: isLateDeviceWrap(input.message) ? LATE_JSON_SYSTEM : DEBATE_SYSTEM,
      });
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
    this.emit(this.store.get(threadId));
  }

  private applyRun(
    threadId: string,
    messageId: string,
    run: OrchestratedRun,
    decision: RouteDecision,
    _opts?: { systemNote?: string },
  ): void {
    if (!this.stillOpen(threadId)) return;
    const current = this.store.get(threadId)?.messages.find((m) => m.id === messageId);
    const label = current?.label || current?.nickname || current?.speaker || "Speaker";
    const error = run.status === "error" ? speakerErrorText(label, run.error) : undefined;
    const content = error ?? run.text?.trim() ?? "";
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
    this.emit(this.store.get(threadId));
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
    if (!this.stillOpen(threadId)) return;
    try {
      const raw = error instanceof Error ? error.message : String(error);
      const current = this.store.get(threadId)?.messages.find((m) => m.id === messageId);
      const label = current?.label || current?.nickname || current?.speaker || "Speaker";
      const text = speakerErrorText(label, raw);
      this.store.patchMessage(threadId, messageId, {
        content: text,
        status: "error",
        error: text,
        thinkingPhase: undefined,
        thinkingStartedAt: undefined,
        suggestedAction: suggestedAction ?? suggestedForRunError(raw),
      });
      this.stopHeartbeatIfIdle(threadId);
      this.emit(this.store.get(threadId));
    } catch {
      this.fail(threadId, error);
    }
  }

  private fail(threadId: string, error: unknown): void {
    if (!this.stillOpen(threadId)) return;
    try {
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
      this.emit(this.store.get(threadId));
    } catch {
      /* thread gone between stillOpen and append, or SSE emit failed */
    }
  }

  private note(
    threadId: string | undefined,
    content: string,
    extra: Partial<ChatMessage> = {},
  ): ChatThread | undefined {
    if (!threadId || !this.stillOpen(threadId)) return undefined;
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
    const thread = this.store.get(threadId);
    if (!thread) return undefined;
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
    if (decision.applyPatch) {
      return {
        backendId: "orchestrator",
        specialist: "apply-patch",
        label: "Apply patch",
        writesLocalFiles: true,
      };
    }
    if (decision.closer?.writesLocalFiles) return decision.closer;
    if (decision.closer && /cursor/i.test(decision.closer.backendId)) return decision.closer;
    const fromSpeakers = decision.speakers?.find((s) => s.writesLocalFiles);
    if (fromSpeakers) return fromSpeakers;
    const cursorSpeaker = decision.speakers?.find((s) => /cursor/i.test(s.backendId));
    if (cursorSpeaker) return cursorSpeaker;
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
    if (!this.stillOpen(threadId)) return;
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
    this.emit(this.store.get(threadId));
  }

  private async runApplyPatch(threadId: string, pending: PendingApproval): Promise<void> {
    let placeholderId: string | undefined;
    try {
      if (!this.stillOpen(threadId)) return;
      const cwd = this.orchestrator.allowlist.assertCwd(pending.cwd ?? this.orchestrator.defaultCwd());
      const speaker: RouteSpeaker = {
        backendId: "orchestrator",
        specialist: "apply-patch",
        label: "Apply patch",
        writesLocalFiles: true,
      };
      const decision: RouteDecision = {
        kind: "single",
        pin: pending.pin ?? "auto",
        intent: "code",
        chip: speaker.label,
        needsWrites: true,
        applyPatch: true,
      };
      const placeholder = this.beginSpeaker(threadId, speaker, decision, "single", "waiting");
      placeholderId = placeholder.id;
      if (!this.stillOpen(threadId)) return;
      const hostNote = pending.systemWideInstall
        ? "\n\nHost package commands were listed only — not executed (no apt, sudo, or Unity installer)."
        : "";
      const files = parseOrchestratorFiles(pending.summary);
      if (!files.length && !pending.systemWideInstall) {
        throw new Error(
          'No files to apply. Put a fenced orchestrator-files JSON block in the plan: ```orchestrator-files\n{"files":[{"path":"README.md","content":"..."}]}\n```',
        );
      }
      const written = files.length
        ? applyParsedFiles({ cwd, files, allowlist: this.orchestrator.allowlist }).written
        : [];
      if (!this.stillOpen(threadId)) return;
      const list = written.length ? written.map((p) => `• ${p}`).join("\n") : "(no files)";
      this.store.patchMessage(threadId, placeholder.id, {
        content: `Wrote ${written.length} file(s) inside ${cwd}:\n${list}${hostNote}`,
        status: "finished",
        chip: speaker.label,
        thinkingPhase: undefined,
        thinkingStartedAt: undefined,
      });
      this.stopHeartbeatIfIdle(threadId);
      this.emit(this.store.get(threadId));
    } catch (error) {
      if (placeholderId) this.patchError(threadId, placeholderId, error);
      else this.fail(threadId, error);
    } finally {
      const thread = this.store.get(threadId);
      if (thread?.pendingApproval?.id === pending.id) {
        this.store.setPendingApproval(threadId, undefined);
      }
    }
  }

  private async runApprovedCloser(threadId: string, pending: PendingApproval): Promise<void> {
    if (pending.applyPatch || pending.backendId === "orchestrator" || pending.specialist === "apply-patch") {
      await this.runApplyPatch(threadId, pending);
      return;
    }
    let placeholderId: string | undefined;
    try {
      if (!this.stillOpen(threadId)) return;
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
      placeholderId = placeholder.id;
      if (!this.stillOpen(threadId)) return;
      const hostNote = pending.systemWideInstall
        ? `\n\nThe user approved host installs (Unity Hub, apt, sudo) in addition to writes inside ${cwd}. Do not install anything outside that approval.`
        : `\n\nWrites only inside ${cwd}. Do not install host packages (Unity Hub, apt, sudo).`;
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
      if (!this.stillOpen(threadId)) return;
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
      if (placeholderId) this.patchError(threadId, placeholderId, error);
      else this.fail(threadId, error);
    } finally {
      const thread = this.store.get(threadId);
      if (thread?.pendingApproval?.id === pending.id) {
        this.store.setPendingApproval(threadId, undefined);
      }
    }
  }

  private cwdForDispatch(
    decision: RouteDecision,
    speaker: RouteSpeaker,
    fallback?: string,
    lateWrap = false,
  ): string | undefined {
    if (!speaker.writesLocalFiles) return undefined;
    if (lateWrap) {
      const granted = decision.cwd ?? fallback;
      if (!granted) return undefined;
      return this.orchestrator.allowlist.tryCwd(granted) ?? undefined;
    }
    const candidate = decision.cwd ?? fallback ?? this.orchestrator.defaultCwd();
    const allowed = this.orchestrator.allowlist.tryCwd(candidate);
    if (allowed) return allowed;
    return this.orchestrator.allowlist.list()[0] ?? candidate;
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
    const thread = this.store.get(threadId);
    if (!thread) return [];
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

  private emit(thread: ChatThread | undefined): void {
    if (!thread || !this.stillOpen(thread.id)) return;
    try {
      this.orchestrator.events.emit("chat", this.view(thread.id));
      this.orchestrator.events.emit("chats", this.store.list());
    } catch {
      /* SSE client gone — chat state is already saved */
    }
  }

  private beginSpeaker(
    threadId: string,
    speaker: RouteSpeaker,
    decision: RouteDecision,
    phase: ChatMessage["phase"],
    thinkingPhase: ThinkingPhase,
    round?: number,
  ): ChatMessage {
    if (!this.stillOpen(threadId)) {
      return {
        id: crypto.randomUUID(),
        createdAt: Date.now(),
        role: "assistant",
        speaker: speaker.backendId,
        label: speaker.label,
        content: "",
        status: "error",
        error: "chat deleted",
        phase,
        round,
        chip: decision.chip,
        nickname: speaker.nickname,
        hasLogo: speaker.hasLogo,
        logoUrl: speaker.logoUrl,
      };
    }
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
    this.emit(this.store.get(threadId));
    this.startHeartbeat(threadId);
    return placeholder;
  }

  private deltaHandler(threadId: string, messageId: string): (delta: string) => void {
    let assembled = "";
    let lastEmit = 0;
    return (delta: string) => {
      if (!this.stillOpen(threadId)) return;
      try {
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
          this.emit(this.store.get(threadId));
        }
      } catch {
        /* thread gone */
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
      try {
        this.orchestrator.events.emit("chat-heartbeat", payload);
      } catch {
        /* SSE client gone */
      }
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
  const late = isLateDeviceWrap(input.user);
  const system = late ? LATE_JSON_SYSTEM : DEBATE_SYSTEM;
  const prior =
    input.transcript.length === 0
      ? late
        ? "You speak first. Reply with one Late JSON object only."
        : "You speak first this round. Give an independent take."
      : late
        ? `A prior speaker said:\n${input.transcript.map((t) => `### ${t.label}\n${t.text}`).join("\n\n")}\n\nIf they already emitted valid Late JSON, repeat that same JSON object with no extra prose. Do not lecture about the repo.`
        : `Round-table so far:\n${input.transcript.map((t) => `### ${t.label}\n${t.text}`).join("\n\n")}\n\nCritique, improve, or dissent. Quote disagreements clearly.`;
  return `${system}

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
  applyPatch = false,
): string {
  const body = transcript.map((t) => `### ${t.label}\n${t.text}`).join("\n\n") || "(no prior turns)";
  const system = planOnly
    ? applyPatch
      ? `${SYNTHESIS_PLAN_ONLY}\n${APPLY_PATCH_INSTRUCTIONS}`
      : SYNTHESIS_PLAN_ONLY
    : SYNTHESIS_SYSTEM;
  const closerHint = planOnly
    ? applyPatch
      ? "Write the merged plan only, ending with an orchestrator-files fence. Do not write files or install host packages."
      : "Write the merged plan only. Do not write files or install host packages."
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
  if (/CURSOR_API_KEY|Cursor not configured/i.test(error)) {
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
