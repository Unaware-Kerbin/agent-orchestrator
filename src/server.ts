import { McpServer } from "@modelcontextprotocol/server";
import * as z from "zod/v4";
import { compactChatToolError } from "./chat/mcp-error.js";
import type { ChatService } from "./chat/service.js";
import type { Orchestrator } from "./orchestrator.js";
import { probeLlamaCpp, probeOllama } from "./local-servers/status.js";
import { toRunView } from "./views.js";

function json(data: unknown): string {
  return JSON.stringify(data, null, 2);
}

function textResult(data: unknown, isError = false) {
  return { content: [{ type: "text" as const, text: json(data) }], isError };
}

/**
 * Speaker skip/429 is thread content (status=error on that chip), not an MCP tool failure.
 * Late treats isError as “Agent stopped” and would dump this whole JSON.
 */
export function mcpChatToolIsError(_thread: unknown): boolean {
  void _thread;
  return false;
}

export function createServer(orchestrator: Orchestrator, chat: ChatService): McpServer {
  const server = new McpServer({
    name: "agent-orchestrator",
    version: "0.1.0",
  });

  server.registerTool(
    "list_agents",
    {
      description:
        "List specialist agents, backends (Cursor + external), workflows, write-allowlist directories, default cwd, and local-runtime status (hardware summary, vLLM running vs stopped, Cursor cloud / CURSOR_API_KEY). Call this before dispatching work so you pick a ready backend. Re-reads .env and GUI secrets on each call so newly added keys take effect without restarting Cursor. Local Cursor agents may only write inside allowed directories. Cloud Cursor agents cannot reach local vLLM; use local-and-cloud or cloud-with-local-draft workflows.",
      inputSchema: z.object({}),
    },
    async () => textResult(await orchestrator.catalog()),
  );

  server.registerTool(
    "dispatch",
    {
      description:
        "Send a task to one specialist. Prefer chat_send for natural language so the orchestrator auto-routes (including round-table debate). Cursor local backends can edit files only inside the write allowlist.",
      inputSchema: z.object({
        specialist: z
          .string()
          .describe("Specialist id, e.g. planner, builder, reviewer, pr-triage, or a custom external agent id"),
        task: z.string().describe("What the specialist should do"),
        backend: z
          .string()
          .optional()
          .describe("Override backend id from config, e.g. cursor-local, anthropic, openai, openrouter, gemini, vllm-local"),
        cwd: z
          .string()
          .optional()
          .describe("Workspace path for local Cursor agents. Must be inside an allowed directory (see list_agents writePolicy)."),
        model: z.string().optional().describe("Per-run model override"),
        wait: z
          .boolean()
          .optional()
          .describe("Wait for completion (default true). If false, poll with get_run"),
        pr_url: z.string().optional().describe("Pull request URL for triage/review"),
        repo_url: z.string().optional().describe("Git remote URL for Cursor cloud agents"),
        branch: z.string().optional().describe("Branch or starting ref"),
        extra_context: z.string().optional().describe("Additional context to include in the prompt"),
        cloud_auto_create_pr: z.boolean().optional().describe("Cloud Cursor agents only: open a PR when done"),
      }),
    },
    async (args) => {
      try {
        const run = await orchestrator.dispatch({
          specialist: args.specialist,
          task: args.task,
          backend: args.backend,
          cwd: args.cwd,
          model: args.model,
          wait: args.wait,
          prUrl: args.pr_url,
          repoUrl: args.repo_url,
          branch: args.branch,
          extraContext: args.extra_context,
          cloudAutoCreatePr: args.cloud_auto_create_pr,
        });
        return textResult(toRunView(run), run.status === "error");
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "chat_send",
    {
      description:
        "Send a natural-language message through the same auto-router the GUI uses. Default pin is auto: hardware/vLLM/allowlist questions hit control tools; build/fix/review/plan with two or more ready backends (including Cursor) runs a round-table. pin=debate forces every ready local server plus ready Cursor/Gemini/cloud specialists to answer (Late MCP uses this). Implement/install stays plan-only until the user Approves (pendingApproval on the thread). Q&A and debate text are not blocked. pin=single or a backend id skips debate. File writes never go to vLLM. Assistant messages include speaker id, nickname, and a loopback logoUrl when a logo is set.",
      inputSchema: z.object({
        message: z.string().describe("User message, as you would type in the GUI chat"),
        thread_id: z.string().optional().describe("Existing chat id; omit to start a new thread"),
        pin: z
          .string()
          .optional()
          .describe(
            "auto (default: debate on build/fix/review/plan when two backends including Cursor are ready, or when two local servers are ready), debate (force round-table: every ready local vLLM/Ollama/llama.cpp plus ready Gemini/Cursor/cloud), single (one speaker), or a backend id (local, cloud, gemini, …). Pinning a backend skips debate. Naming a backend still pins that one speaker.",
          ),
        cwd: z.string().optional(),
        pr_url: z.string().optional(),
        repo_url: z.string().optional(),
        branch: z.string().optional(),
        extra_context: z.string().optional(),
        wait: z.boolean().optional().describe("Wait for the full reply (default true)"),
      }),
    },
    async (args) => {
      try {
      const thread = await chat.send({
        threadId: args.thread_id,
        message: args.message,
        pin: args.pin,
        cwd: args.cwd,
        prUrl: args.pr_url,
        repoUrl: args.repo_url,
        branch: args.branch,
        extraContext: args.extra_context,
        wait: args.wait,
      });
      // chat.send already attaches `busy` for wait:false polling.
      const pending = thread.pendingApproval?.status === "pending" ? thread.pendingApproval : undefined;
      return textResult(
        {
          ...thread,
          pendingApproval: thread.pendingApproval,
          approvalRequired: Boolean(pending),
          note: pending
            ? "Implement/install is waiting for human Approve. Debate/Q&A already ran plan-only. Call chat_approve to confirm or reject. Unity/apt/sudo will not run until Approve."
            : undefined,
        },
        mcpChatToolIsError(thread),
      );
      } catch (error) {
        return textResult({ error: compactChatToolError(error) }, true);
      }
    },
  );

  server.registerTool(
    "chat_approve",
    {
      description:
        "Approve or reject pending implement/install actions from chat_send. Until Approve, Cursor stays plan-only (no writes, no Unity/apt/sudo). After Approve, the closer may write only inside the allowlisted cwd. Optional comment is stored on the thread.",
      inputSchema: z.object({
        thread_id: z.string().describe("Chat thread id that has pendingApproval"),
        decision: z.enum(["approve", "reject"]),
        comment: z.string().optional().describe("Optional note stored with the decision"),
      }),
    },
    async (args) => {
      try {
        const thread = await chat.resolveApproval({
          threadId: args.thread_id,
          decision: args.decision,
          comment: args.comment,
        });
        return textResult(thread, mcpChatToolIsError(thread));
      } catch (error) {
        return textResult({ error: compactChatToolError(error) }, true);
      }
    },
  );

  server.registerTool(
    "chat_list",
    {
      description: "List persisted GUI/MCP chat threads (id, title, agents, updatedAt).",
      inputSchema: z.object({}),
    },
    async () => textResult(chat.list()),
  );

  server.registerTool(
    "chat_get",
    {
      description:
        "Read one chat thread (messages, speakers, pending approval). Includes busy=true while debate/single is still running so clients can poll instead of blocking on chat_send wait.",
      inputSchema: z.object({
        thread_id: z.string().describe("Chat thread id from chat_send"),
      }),
    },
    async (args) => {
      try {
        return textResult(chat.view(args.thread_id), false);
      } catch (error) {
        return textResult({ error: compactChatToolError(error) }, true);
      }
    },
  );

  server.registerTool(
    "run_workflow",
    {
      description:
        "Run a named multi-agent pipeline. Built-in: ship-feature (planner → builder → reviewer), troubleshoot-pr (pr-triage → builder → reviewer), local-and-cloud (vLLM + Cursor cloud in parallel), cloud-with-local-draft (vLLM draft then cloud builder). Each sequential step sees prior output. Parallel recipes collect both results. Local file writes still require an allowed cwd. Cloud agents never call localhost vLLM.",
      inputSchema: z.object({
        workflow: z.string().describe("Workflow id from list_agents"),
        task: z.string().describe("User goal or PR description"),
        cwd: z.string().optional(),
        pr_url: z.string().optional(),
        repo_url: z.string().optional(),
        branch: z.string().optional(),
        extra_context: z.string().optional(),
        stop_on_error: z.boolean().optional().describe("Default true"),
      }),
    },
    async (args) => {
      try {
        const result = await orchestrator.runWorkflow({
          workflow: args.workflow,
          task: args.task,
          cwd: args.cwd,
          prUrl: args.pr_url,
          repoUrl: args.repo_url,
          branch: args.branch,
          extraContext: args.extra_context,
          stopOnError: args.stop_on_error,
        });
        return textResult(
          {
            workflow: result.workflow,
            status: result.status,
            summary: result.summary,
            runs: result.runs.map((run) => toRunView(run)),
          },
          result.status === "error",
        );
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "follow_up",
    {
      description:
        "Continue an existing specialist run. Cursor backends resume the same agent (full conversation + workspace). External backends continue with stored chat history.",
      inputSchema: z.object({
        run_id: z.string().describe("Run id returned by dispatch or run_workflow"),
        message: z.string(),
        wait: z.boolean().optional(),
      }),
    },
    async (args) => {
      try {
        const run = await orchestrator.followUp({
          runId: args.run_id,
          message: args.message,
          wait: args.wait,
        });
        return textResult(toRunView(run), run.status === "error");
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "get_run",
    {
      description: "Fetch status and output for a dispatch or workflow step by run id.",
      inputSchema: z.object({ run_id: z.string() }),
    },
    async ({ run_id }) => {
      const run = orchestrator.store.get(run_id);
      if (!run) return textResult({ error: `Unknown run ${run_id}` }, true);
      return textResult(toRunView(run, true));
    },
  );

  server.registerTool(
    "list_runs",
    {
      description: "List recent orchestrated runs, newest first.",
      inputSchema: z.object({
        limit: z.number().int().min(1).max(200).optional(),
      }),
    },
    async ({ limit }) => textResult(orchestrator.store.list(limit ?? 50).map((run) => toRunView(run))),
  );

  server.registerTool(
    "list_allowed_dirs",
    {
      description:
        "List directories local Cursor agents may write to, plus the default cwd. The workspace is granted by default until you add more.",
      inputSchema: z.object({}),
    },
    async () =>
      textResult({
        allowedDirectories: orchestrator.allowlist.list(),
        defaultCwd: orchestrator.defaultCwd(),
        fileWrites: "cursor-local-only",
      }),
  );

  server.registerTool(
    "add_allowed_dir",
    {
      description:
        "Grant a directory for local Cursor agent file writes. The path is resolved with realpath (symlink escapes are rejected) and must already exist.",
      inputSchema: z.object({
        path: z.string().describe("Absolute or relative directory path to allow"),
      }),
    },
    async ({ path }) => {
      try {
        return textResult({ allowedDirectories: orchestrator.allowlist.add(path) });
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "remove_allowed_dir",
    {
      description: "Revoke a previously granted write directory. Local Cursor dispatch will fail if cwd is no longer allowed.",
      inputSchema: z.object({
        path: z.string().describe("Directory path to remove from the allowlist"),
      }),
    },
    async ({ path }) => {
      try {
        return textResult({ allowedDirectories: orchestrator.allowlist.remove(path) });
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "list_hardware",
    {
      description:
        "Summarize discrete GPUs for local model serving (NVIDIA, AMD, Intel, or CPU if none), with VRAM, primaryBackend, and any local vendor Docker images for serving.",
      inputSchema: z.object({}),
    },
    async () => textResult(orchestrator.localModels.listHardware()),
  );

  server.registerTool(
    "list_local_models",
    {
      description:
        "Every curated open-weight chat model for local vLLM (full catalog, not a short slice): fit flags for this GPU (weights + ~20% KV headroom), newest Hub id when a family has several names, downloaded under the allowlisted models dir, and every running loopback vLLM instance (backend id, port, image).",
      inputSchema: z.object({}),
    },
    async () => textResult(orchestrator.localModels.listModels()),
  );

  server.registerTool(
    "recommend_local_models",
    {
      description:
        "Every catalog model for local vLLM, with fit flags for the GPUs on this computer (fits / needs tensor parallel / too big). Newest Hub id is marked when a family has several names (Qwen3.8 over Qwen2.5, Gemma 4 over Gemma 2/3, Llama 4/3.3 over 3.1). Nothing is hidden. Without an accelerator, only tiny CPU-feasible entries fit.",
      inputSchema: z.object({}),
    },
    async () => textResult(orchestrator.localModels.recommend()),
  );

  server.registerTool(
    "download_local_model",
    {
      description:
        "Download a catalog model (or Hugging Face org/name repo) into the allowlisted models directory (default .orchestrator/models). Large; requires an explicit model_id. Honors HF_TOKEN / HUGGING_FACE_HUB_TOKEN for gated repos. dry_run only checks the destination path.",
      inputSchema: z.object({
        model_id: z.string().describe("Catalog id from list_local_models, or a Hugging Face repo id"),
        dest: z.string().optional().describe("Optional destination directory; must be inside the write allowlist"),
        dry_run: z.boolean().optional().describe("Validate allowlisted path only; do not download"),
      }),
    },
    async (args) => {
      try {
        return textResult(
          orchestrator.localModels.download({
            modelId: args.model_id,
            dest: args.dest,
            dryRun: args.dry_run,
          }),
        );
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "start_vllm",
    {
      description:
        "Launch a local OpenAI-compatible server bound to 127.0.0.1 only (never 0.0.0.0). Each catalog model gets its own Docker container, port in 8000–8099, backend id (vllm-<slug> from the catalog id, e.g. vllm-qwen25-7b-instruct), and specialist. Tensor-parallels across every GPU only when the catalog fit needs more than one card; models that fit a single GPU stay at --tensor-parallel-size 1. Pass use_all_gpus=false to pin to one GPU even when the model is larger. Does not stop other orchestrator vLLM containers unless replace=true (restarts this model only). On intel-xpu, if intel/llm-scaler-vllm or intel/vllm:*xpu is local, starts that container (API published as 127.0.0.1:port:8000). Model must already be downloaded. Waits until GET /v1/models is healthy, upserts that backend + specialist, and stores a dummy loopback Bearer in gitignored GUI secrets if needed (never copy a key from vLLM). Cloud agents still cannot reach this server.",
      inputSchema: z.object({
        model_id: z.string().describe("Catalog id or Hugging Face repo already downloaded"),
        port: z.number().int().min(8000).max(8099).optional(),
        quantization: z.enum(["awq", "gptq"]).optional(),
        host: z
          .string()
          .optional()
          .describe("Must be 127.0.0.1 or localhost; any other value is rejected"),
        timeout_ms: z.number().int().min(5000).max(600_000).optional(),
        image: z
          .string()
          .optional()
          .describe("Local Docker image override, e.g. intel/vllm:0.17.0-xpu or intel/llm-scaler-vllm:0.21.0-b3"),
        runtime: z
          .enum(["docker", "host"])
          .optional()
          .describe("Force Docker (Intel images) or host vllm. Default on intel-xpu is docker when a matching image exists."),
        replace: z
          .boolean()
          .optional()
          .describe("If true, stop the existing instance of this model before starting. Other running vLLM models are left alone."),
        use_all_gpus: z
          .boolean()
          .optional()
          .describe(
            "Allow using every GPU when the catalog fit needs tensor parallel. False: pin to one GPU. Models that already fit one card stay on one GPU even when this is true.",
          ),
      }),
    },
    async (args) => {
      try {
        const status = await orchestrator.localModels.startVllm({
          modelId: args.model_id,
          port: args.port,
          quantization: args.quantization,
          host: args.host,
          timeoutMs: args.timeout_ms,
          image: args.image,
          runtime: args.runtime,
          replace: args.replace,
          useAllGpus: args.use_all_gpus,
        });
        return textResult(status);
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "stop_vllm",
    {
      description:
        "Stop one orchestrator-managed vLLM instance (container or host process group). Pass model_id or backend_id to leave other running models up. If several are running and neither id is set, the call fails unless all=true. Does not stop unrelated Docker containers (including a leftover orch-vllm you did not start this way).",
      inputSchema: z.object({
        model_id: z.string().optional().describe("Catalog id of the instance to stop"),
        backend_id: z.string().optional().describe("Backend id, e.g. vllm-qwen25-7b-instruct"),
        all: z.boolean().optional().describe("Stop every orchestrator-tracked vLLM instance"),
      }),
    },
    async (args) => {
      try {
        return textResult(
          orchestrator.localModels.stopVllm({
            modelId: args.model_id,
            backendId: args.backend_id,
            all: args.all,
          }),
        );
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "remove_vllm",
    {
      description:
        "Stop one vLLM instance and unregister its backend and specialist from agents.config.yaml. Other running models stay up. Does not delete downloaded weights.",
      inputSchema: z.object({
        model_id: z.string().optional().describe("Catalog id of the instance to remove"),
        backend_id: z.string().optional().describe("Backend id, e.g. vllm-qwen25-7b-instruct"),
      }),
    },
    async (args) => {
      try {
        return textResult(
          orchestrator.localModels.removeVllm({
            modelId: args.model_id,
            backendId: args.backend_id,
          }),
        );
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "delete_local_model",
    {
      description:
        "Permanently delete a downloaded snapshot from the allowlisted models directory. Stops that model if it is running. Requires confirm=true. Does not unregister other backends.",
      inputSchema: z.object({
        model_id: z.string().describe("Catalog id whose weights to delete"),
        confirm: z.boolean().describe("Must be true; refuses otherwise"),
      }),
    },
    async (args) => {
      try {
        return textResult(
          orchestrator.localModels.deleteLocalModel({
            modelId: args.model_id,
            confirm: args.confirm,
          }),
        );
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "vllm_status",
    {
      description:
        "All orchestrator-managed vLLM instances (image, port, model, backend id, container) plus whether a serving stack is installed (including local Intel Docker images).",
      inputSchema: z.object({}),
    },
    async () => textResult(orchestrator.localModels.vllmStatus()),
  );

  server.registerTool(
    "ollama_status",
    {
      description:
        "Probe a loopback Ollama daemon (default http://127.0.0.1:11434). Lists tags when it is running. Does not install Ollama. Non-loopback URLs are rejected.",
      inputSchema: z.object({
        base_url: z
          .string()
          .optional()
          .describe("OpenAI-compat base such as http://127.0.0.1:11434/v1; must be 127.0.0.1/localhost"),
      }),
    },
    async (args) => {
      try {
        return textResult(await probeOllama({ baseUrl: args.base_url }));
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerTool(
    "llamacpp_status",
    {
      description:
        "Probe a loopback llama.cpp llama-server OpenAI API (default http://127.0.0.1:8080/v1). Does not download GGUF files or start a process. Non-loopback URLs are rejected.",
      inputSchema: z.object({
        base_url: z
          .string()
          .optional()
          .describe("OpenAI-compat base such as http://127.0.0.1:8080/v1; must be 127.0.0.1/localhost"),
      }),
    },
    async (args) => {
      try {
        return textResult(await probeLlamaCpp({ baseUrl: args.base_url }));
      } catch (error) {
        return textResult({ error: error instanceof Error ? error.message : String(error) }, true);
      }
    },
  );

  server.registerResource(
    "agent-catalog",
    "agents://catalog",
    {
      description: "Current specialists, backends, workflows, and write allowlist",
      mimeType: "application/json",
    },
    async () => ({
      contents: [{ uri: "agents://catalog", text: json(await orchestrator.catalog()), mimeType: "application/json" }],
    }),
  );

  server.registerPrompt(
    "ship-feature",
    {
      description: "Plan, implement, and review a feature with the orchestrated agent team",
      argsSchema: {
        goal: z.string().describe("What to build"),
      },
    },
    async ({ goal }) => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Use the agent-orchestrator MCP. Call run_workflow with workflow "ship-feature" and this task:\n\n${goal}`,
          },
        },
      ],
    }),
  );

  server.registerPrompt(
    "troubleshoot-pr",
    {
      description: "Diagnose a PR, patch it, then review",
      argsSchema: {
        pr_url: z.string().describe("Pull request URL"),
        notes: z.string().optional().describe("Extra failure context"),
      },
    },
    async ({ pr_url, notes }) => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Use the agent-orchestrator MCP. Call run_workflow with workflow "troubleshoot-pr", pr_url "${pr_url}", and task: Troubleshoot and fix this pull request.${notes ? `\n\n${notes}` : ""}`,
          },
        },
      ],
    }),
  );

  return server;
}
