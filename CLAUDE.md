# CLAUDE.md - Nanobot Development Guide

## Build & Dev Commands

```bash
# Install (editable dev mode)
pip install -e ".[dev]"

# Run tests
pytest                    # all tests (async auto-mode)
pytest tests/test_foo.py  # single file

# Lint & format
ruff check .              # lint
ruff format .             # format

# Docker
docker build -t nanobot .
```

## Architecture

### Single-Agent Mode (default)

**Message flow:** Channel → InboundMessage → MessageBus → AgentLoop → LLM Provider → OutboundMessage → Channel

### Multi-Agent Mode (Hive Mind)

**Message flow:** Channel → InboundMessage → MessageBus → OrchestratorLoop → TaskRouter → Specialist AgentLoop(s) → OutboundMessage → Channel

Enabled via `agents.orchestrator.enabled: true` in config. The orchestrator is a lightweight routing layer that classifies messages and dispatches them to specialist agents, each with isolated context and memory.

```
User Message
    │
    ▼
┌──────────────────┐
│   MessageBus     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────────┐
│  Orchestrator    │────▶│  AgentRegistry          │
│  (router model)  │     │  - tracks live agents   │
│                  │     │  - lifecycle management  │
│  Responsibilities│     └─────────────────────────┘
│  - classify task │
│  - route/create  │            │
│  - memory extract│            ▼
│  - model select  │    ┌───────────────┐
└──────┬───────────┘    │ SpecialistAgent│ ◀──inter-agent──▶ ┌───────────────┐
       │ dispatch       │ (reminder)     │                   │ SpecialistAgent│
       ▼                │ Own session    │                   │ (research)     │
┌──────────────────┐    │ Own memory     │                   │ Own session    │
│ SpecialistAgent  │    │ Own model      │                   │ Own memory     │
│ (coding)         │    └───────────────┘                   │ Own model      │
│ Own session      │                                        └───────────────┘
│ Own memory       │
│ Own model        │
└──────────────────┘
```

### Key Components

- `nanobot/agent/loop.py` — AgentLoop: orchestrates message processing, tool execution (max 40 iterations), token usage tracking
- `nanobot/agent/orchestrator.py` — OrchestratorLoop: lightweight router that classifies messages, manages specialist lifecycle, handles parallel dispatch
- `nanobot/agent/router.py` — TaskRouter: hybrid routing (deterministic regex rules + LLM classification) with proactive memory extraction and model tier selection
- `nanobot/agent/registry.py` — AgentRegistry: tracks agent profiles, lifecycle (suspend/resume/destroy), persists to `agents/registry.json`
- `nanobot/agent/context.py` — ContextBuilder: assembles system prompt from bootstrap files + memory + skills + agent identity
- `nanobot/agent/memory.py` — MemoryConsolidator: archives old messages to MEMORY.md/HISTORY.md via LLM summarization. MemoryStore: two-layer memory with shared global + agent-private scoping
- `nanobot/agent/tools/inter_agent.py` — SendToAgentTool + AgentCallbackTool: async fire-and-forget inter-agent messaging with callback IDs
- `nanobot/agent/tools/registry.py` — ToolRegistry: dynamic tool registration and execution
- `nanobot/session/manager.py` — SessionManager: JSONL-based conversation persistence, configurable session directory for per-agent scoping
- `nanobot/bus/queue.py` — MessageBus: async queue with agent-scoped routing support
- `nanobot/bus/events.py` — InboundMessage/OutboundMessage with `source_agent`, `target_agent`, `type` fields
- `nanobot/channels/base.py` — BaseChannel: abstract base for platform integrations
- `nanobot/providers/base.py` — LLMProvider: abstract base for LLM integrations
- `nanobot/config/schema.py` — OrchestratorConfig, RoutingRule, model tiers
- `nanobot/config/paths.py` — Agent workspace path helpers

**Channels:** telegram, discord, slack, feishu, dingtalk, qq, email, matrix, wecom, whatsapp, mochat

**Providers:** litellm, azure_openai, openai_codex, custom

## Multi-Agent System (Hive Mind)

### Router Signals

The LLM-based router classifies messages and returns structured signals:

| Signal | Format | Purpose |
|--------|--------|---------|
| Basic route | `agent_name` | Route to existing agent |
| New agent | `__new__:name:description` | Create specialist |
| Multi-task | `__multi__\nagent1: task1\nagent2: task2` | Parallel dispatch |
| Correction | `__correct__:target:cancel` | Cancel misrouted agent |
| Memory | `__memory__:fact` | Save to shared memory |
| Model select | `__model__:index_or_name` | Pick model tier |
| Upgrade | `__upgrade__:agent_name` | Escalate to stronger model |

### Memory Architecture

```
~/.nanobot/workspace/
├── memory/                    # Shared global (all agents read)
│   ├── MEMORY.md
│   └── HISTORY.md
├── agents/                    # Per-agent isolated state
│   ├── coding/
│   │   ├── memory/            # Agent-private MEMORY.md + HISTORY.md
│   │   └── sessions/          # Agent-private conversation history
│   └── reminder/
│       ├── memory/
│       └── sessions/
├── sessions/                  # Orchestrator/single-agent sessions
└── skills/
```

- Specialists read both shared global memory AND their private memory
- Proactive memory: router extracts user facts from every message automatically
- Shared memory is visible to the router for informed routing decisions

### Model Selection

The orchestrator supports multiple models sorted by cost/capability:

```json
{
  "orchestrator": {
    "routerModel": "cheap/fast-model",
    "models": [
      "cheapest-model",
      "medium-model",
      "most-capable-model"
    ]
  }
}
```

- Router uses `routerModel` (cheap) for classification only
- Router picks model tier per task via `__model__` signal
- On user dissatisfaction, `__upgrade__` escalates agent to next tier
- Upgrade preserves full session history

### User Commands (Orchestrator Mode)

| Command | Action |
|---------|--------|
| `/agents` | Show active agents, models, and token usage |
| `/stop` | Stop all running tasks |
| `/stop <agent>` | Stop a specific agent |
| `/new` | Clear routing history, start fresh |
| `@agent_name message` | Manual routing to specific agent |

### Parallel Dispatch

- Multiple user messages are routed concurrently (routing serialized, execution parallel)
- Single message with multiple tasks splits via `__multi__` and runs in parallel
- Heartbeat and cron jobs route through orchestrator in multi-agent mode

### Cross-Model Compatibility

- Session `get_history()` normalizes `content: null` → `content: ""` for providers that reject null
- Models must support tool call messages in conversation history to work with nanobot
- Known incompatible via OpenRouter: `stepfun/step-3.5-flash`, `google/gemini-*` (tool history rejected)

## Testing

- pytest with `asyncio_mode = "auto"`, test path: `tests/`
- 30+ test files, async-first
- Pattern: one test file per feature (e.g., `test_feishu_channel.py`, `test_cron_service.py`)

## Code Style (ruff)

- Line length: 100
- Target: Python 3.11+
- Rules: E, F, I, N, W (E501 ignored — formatter handles line length)

## Key Conventions

- **Skills**: Markdown files at `{workspace}/skills/{name}/SKILL.md` with YAML frontmatter (`name`, `description`, `always`, `metadata`). Skills with `always: true` auto-load into context.
- **Sessions**: JSONL at `{workspace}/sessions/{key}.jsonl`. Append-only; consolidation updates `last_consolidated` offset. Per-agent sessions at `{workspace}/agents/{name}/sessions/`.
- **Config**: `~/.nanobot/config.json`
- **Workspace**: `~/.nanobot/workspace/` (sessions, memory, skills, agents)
- **Memory**: Two-layer — `MEMORY.md` (long-term facts, always loaded) + `HISTORY.md` (append-only event log, searchable). Shared global + agent-private scoping in multi-agent mode.
- **Bootstrap templates**: `nanobot/templates/{AGENTS.md, SOUL.md, USER.md, TOOLS.md, HEARTBEAT.md}` — auto-synced to workspace on startup
- **Tool results**: truncated to 16,000 chars before storage
- **Consolidation**: triggers when prompt tokens exceed half context window; uses LLM to summarize → save_memory
- **Token tracking**: each AgentLoop accumulates `prompt_tokens`, `completion_tokens`, `total_tokens`; router overhead tracked separately
