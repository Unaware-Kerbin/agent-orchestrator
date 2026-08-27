import type { Orchestrator } from "../orchestrator.js";
import { canonicalizeDirectory } from "../allowlist.js";
import type { OrchestratedRun } from "../types.js";
import { buildPendingApproval, pendingCardText } from "./approval.js";
import { extractFilesystemPaths, expandUserPath, routeChat, speakerLabel } from "./router.js";
import { ChatStore } from "./store.js";
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

  constructor(private readonly orchestrator: Orchestrator) {}

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

    const work = this.enqueue(thread.id, () => this.respond(thread.id, { ...input, message }));
    if (input.wait === false) {
      void work.catch((error) => {
        this.fail(thread.id, error);
      });
      return this.store.require(thread.id);
    }
    await work;
    return this.store.require(thread.id);
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
        snap.recommended.find((m) => m.downloaded) ??
        snap.recommended[0] ??
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
    }));
    const vllm = catalog.localRuntime?.vllm;
    const workspace = this.resolveWorkspace(input.message, input.cwd);
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
    });

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
      await this.runSingle(threadId, decision, input, Boolean(decision.needsApproval));
      return;
    }
    await this.runDebate(threadId, decision, input, Boolean(decision.needsApproval));
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
        const rec = snap.recommended.map((m) => `${m.name} (${m.id}, ~${m.vramNeededMiB} MiB)`).join("; ") || "none fit";
        content =
          `Hardware: ${acc}\nBackend: ${hw.primaryBackend} · ${hw.totalVramMiB ?? hw.vramMiB} MiB total · RAM ${hw.ramMiB} MiB\n` +
          `vLLM: ${snap.vllm.running ? `running ${snap.vllm.modelId} on 127.0.0.1:${snap.vllm.port}` : "stopped"}\n` +
          `Recommended: ${rec}`;
        if (!snap.vllm.running && snap.recommended[0]) {
          suggestedAction = {
            label: "Start recommended local model",
            action: "start_vllm",
            payload: { modelId: snap.recommended[0].id },
          };
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
      const run = decision.followUpRunId && !holdWrites
        ? await this.orchestrator.followUp({
            runId: decision.followUpRunId,
            message: input.message,
            wait: true,
            onDelta,
          })
        : await this.orchestrator.dispatch({
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
            mode: this.dispatchMode(decision, speaker, holdWrites, false),
            onDelta,
          });
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
    const rounds = Math.min(Math.max(input.rounds ?? decision.rounds ?? 2, 1), 3);
    const transcript: Array<{ label: string; speaker: string; text: string }> = [];
    const history = this.historyFor(threadId);

    for (let round = 1; round <= rounds; round++) {
      for (const speaker of speakers) {
        const placeholder = this.beginSpeaker(threadId, speaker, decision, "debate", "debating", round);
        const task = debateTurnPrompt({
          user: input.message,
          round,
          rounds,
          speaker,
          transcript,
        });
        try {
          const run = await this.orchestrator.dispatch({
            specialist: speaker.specialist,
            backend: speaker.backendId,
            task,
            cwd: this.cwdForDispatch(decision, speaker, input.cwd),
            prUrl: input.prUrl,
            repoUrl: input.repoUrl,
            branch: input.branch,
            extraContext: this.mergeContext(input.extraContext, history),
            wait: true,
            mode: speaker.writesLocalFiles ? "plan" : undefined,
            onDelta: this.deltaHandler(threadId, placeholder.id),
          });
          this.applyRun(threadId, placeholder.id, run, decision, { systemNote: DEBATE_SYSTEM });
          const text = run.text?.trim() || run.error || "";
          if (text) transcript.push({ label: speaker.label, speaker: speaker.backendId, text });
        } catch (error) {
          this.patchError(threadId, placeholder.id, error);
        }
      }
    }

    const placeholder = this.beginSpeaker(threadId, closer, decision, "synthesis", "waiting");
    try {
      const run = await this.orchestrator.dispatch({
        specialist: closer.specialist,
        backend: closer.backendId,
        task: synthesisPrompt(input.message, transcript, closer, holdWrites),
        cwd: this.cwdForDispatch(decision, closer, input.cwd),
        prUrl: input.prUrl,
        repoUrl: input.repoUrl,
        branch: input.branch,
        extraContext: this.mergeContext(input.extraContext, history),
        wait: true,
        mode: this.dispatchMode(decision, closer, holdWrites, true),
        onDelta: this.deltaHandler(threadId, placeholder.id),
      });
      this.applyRun(threadId, placeholder.id, run, decision);
      if (holdWrites) {
        this.holdForApproval(threadId, decision, input, run.text?.trim() || transcript.map((t) => t.text).join("\n\n") || input.message);
      }
    } catch (error) {
      this.patchError(threadId, placeholder.id, error, decision.suggestedAction);
      if (holdWrites) {
        this.holdForApproval(threadId, decision, input, transcript.map((t) => t.text).join("\n\n") || input.message);
      }
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
    this.store.patchMessage(threadId, messageId, {
      content: run.text?.trim() || run.error || "",
      status: error ? "error" : "finished",
      runId: run.id,
      agentId: run.agentId,
      error,
      chip: decision.chip,
      thinkingPhase: undefined,
      thinkingStartedAt: undefined,
      suggestedAction: suggestedForRunError(run.error, decision),
    });
    this.stopHeartbeatIfIdle(threadId);
    this.emit(this.store.require(threadId));
  }

  private patchError(
    threadId: string,
    messageId: string,
    error: unknown,
    suggestedAction?: ChatSuggestedAction,
  ): void {
    const text = error instanceof Error ? error.message : String(error);
    this.store.patchMessage(threadId, messageId, {
      content: text,
      status: "error",
      error: text,
      thinkingPhase: undefined,
      thinkingStartedAt: undefined,
      suggestedAction: suggestedAction ?? suggestedForRunError(text),
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

  private resolveWorkspace(message: string, explicitCwd?: string): WorkspaceHint | undefined {
    const raw = explicitCwd?.trim() || extractFilesystemPaths(message)[0];
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
    this.orchestrator.events.emit("chat", thread);
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
