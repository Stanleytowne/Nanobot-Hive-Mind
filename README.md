<div align="center">
  <img src="nanobot_logo.png" alt="nanobot hive mind" width="500">
  <h1>Nanobot Hive Mind</h1>
  <p><strong>Multi-agent orchestration for personal AI assistants</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/architecture-multi--agent-purple" alt="Multi-Agent">
  </p>
</div>

Nanobot Hive Mind extends [nanobot](https://github.com/HKUDS/nanobot) with a **multi-agent orchestration layer** — a lightweight router that dynamically creates, manages, and coordinates specialist agents, each with their own model, memory, and session context.

Instead of one monolithic agent trying to do everything, Hive Mind routes each task to the right specialist with the right model at the right cost.

## Why Multi-Agent?

A single AI agent handling all your tasks is simple — but fundamentally wasteful. Here's why a multi-agent architecture is more effective:

### Context Isolation

A typical user juggles many different tasks: coding, research, scheduling, Q&A. These tasks are often concurrent and unrelated. Routing all of them through a single agent **contaminates the context window** — a coding conversation polluted with scheduling history degrades performance on both. Hive Mind gives each specialist its own session, so every agent operates with a clean, focused context.

### Intelligent Memory

Because every message passes through the orchestrator first, it acts as a **triage layer for knowledge**. The router proactively extracts personal facts (timezone, preferences, project context) from every message and commits them to shared long-term memory — something a single agent busy with "real work" does inconsistently. Each specialist also maintains private memory relevant to its domain, so a coding agent remembers your codebase while a reminder agent knows your schedule.

### Cost Efficiency

Not every task needs the most powerful model. A simple greeting doesn't require the same compute as a complex coding task. Hive Mind assigns **model tiers per task** — cheap/fast models for classification and chat, expensive models for analysis and coding. If a cheaper model can't handle a task (hits iteration limits), the system **auto-upgrades** to the next tier and retries, preserving the full session. You get strong-model quality only when you need it.

### Fault Isolation

When a specialist gets stuck or produces poor output, it doesn't affect other agents' work. The system detects failures (max iterations reached) and can auto-upgrade that one agent — while other specialists continue uninterrupted. You can also manually stop, restart, or upgrade individual agents without losing progress elsewhere.

### True Parallelism

A single agent is inherently sequential. Hive Mind can **split a multi-part request** and dispatch tasks to different specialists concurrently. A coding agent and a research agent can work on different parts of the same request simultaneously, then return results independently.

### Specialist Continuity

Each agent accumulates domain-specific context over time. A "coding" agent builds up understanding of your codebase across conversations; a "research" agent retains your research threads. A single generalist would dilute this accumulated expertise with unrelated context from other tasks.

### Graceful Scaling

Adding a new capability means creating a new specialist — not bloating a single agent's system prompt. Agents are created and destroyed dynamically based on demand, with a configurable cap on active specialists.

## Architecture

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
└──────┬───────────┘    │ Specialist    │ ◀──inter-agent──▶ ┌───────────────┐
       │ dispatch       │ (reminder)    │                   │ Specialist    │
       ▼                │ Own session   │                   │ (research)    │
┌──────────────────┐    │ Own memory    │                   │ Own session   │
│ Specialist       │    │ Own model     │                   │ Own memory    │
│ (coding)         │    └───────────────┘                   │ Own model     │
│ Own session      │                                        └───────────────┘
│ Own memory       │
│ Own model        │
└──────────────────┘
```

**Message flow:** Channel → MessageBus → Orchestrator (classify + route) → Specialist Agent(s) → Response → Channel

The orchestrator uses a **cheap/fast model** purely for routing — it never generates end-user responses. Specialist agents use independently assigned models matched to their task complexity.

### Memory Architecture

```
~/.nanobot/workspace/
├── memory/                    # Shared global (all agents read)
│   ├── MEMORY.md              # Long-term facts, always loaded
│   └── HISTORY.md             # Append-only event log
├── agents/                    # Per-agent isolated state
│   ├── coding/
│   │   ├── memory/            # Agent-private memory
│   │   └── sessions/          # Agent-private conversation history
│   └── reminder/
│       ├── memory/
│       └── sessions/
└── sessions/                  # Orchestrator sessions
```

## Install

```bash
git clone https://github.com/Stanleytowne/Nanobot-Hive-Mind.git
cd Nanobot-Hive-Mind
pip install -e .
```

## Quick Start

**1. Initialize**

```bash
nanobot onboard
```

**2. Configure** (`~/.nanobot/config.json`)

Set your API key and enable the orchestrator:

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-sonnet-4-20250514"
    },
    "orchestrator": {
      "enabled": true,
      "routerModel": "google/gemini-2.0-flash-001",
      "models": [
        "google/gemini-2.0-flash-001",
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-opus-4-20250514"
      ],
      "maxSpecialists": 10
    }
  }
}
```

The `models` array is sorted cheapest → most expensive. The router assigns model tiers per task and auto-upgrades when needed.

**3. Connect a channel and run**

```bash
nanobot gateway
```

See [Chat Apps](#chat-apps) for channel setup (Telegram, Discord, Slack, etc.).

## Hive Mind Commands

| Command | Action |
|---------|--------|
| `/agents` | Show active agents, their models, and token usage |
| `/model` | List available models and current assignments |
| `/model <agent> <model>` | Set a specific agent's model |
| `@agent message` | Route directly to a specific agent |
| `@agent:model message` | Route to agent with a model override |
| `/stop` | Stop all running tasks |
| `/stop <agent>` | Stop a specific agent |
| `/new` | Clear routing history, start fresh |

## Router Signals

The orchestrator's router is an LLM that classifies messages and emits structured signals:

| Signal | Format | Purpose |
|--------|--------|---------|
| Route | `agent_name` | Send to existing specialist |
| Create | `__new__:name:description` | Spin up a new specialist |
| Multi-task | `__multi__\nagent1: task1\nagent2: task2` | Parallel dispatch |
| Memory | `__memory__:fact` | Save to shared long-term memory |
| Model | `__model__:index_or_name` | Select model tier for task |
| Upgrade | `__upgrade__:agent_name` | Escalate to stronger model |
| Correct | `__correct__:target:cancel` | Cancel a misrouted agent |

## Chat Apps

Connect to your preferred chat platform. **Note:** We have only tested Telegram — other channels are inherited from upstream nanobot and may not work correctly with the Hive Mind orchestrator.

| Channel | What you need |
|---------|---------------|
| **Telegram** | Bot token from @BotFather |
| **Discord** | Bot token + Message Content intent |
| **WhatsApp** | QR code scan |
| **Feishu** | App ID + App Secret |
| **DingTalk** | App Key + App Secret |
| **Slack** | Bot token + App-Level token |
| **Email** | IMAP/SMTP credentials |
| **QQ** | App ID + App Secret |
| **Wecom** | Bot ID + Bot Secret |
| **Matrix** | Homeserver URL + credentials |

<details>
<summary><b>Telegram</b> (Recommended)</summary>

1. Open Telegram → search `@BotFather` → `/newbot` → copy the token
2. Add to config:

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

3. Run `nanobot gateway`

</details>

<details>
<summary><b>Discord</b></summary>

1. Create app at [Discord Developer Portal](https://discord.com/developers/applications)
2. Enable **Message Content Intent** under Bot settings
3. Add to config:

```json
{
  "channels": {
    "discord": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

4. Run `nanobot gateway`

</details>

<details>
<summary><b>Slack</b></summary>

1. Create a Slack app with Socket Mode enabled
2. Add Bot Token Scopes: `chat:write`, `app_mentions:read`, `im:history`, `im:read`
3. Add to config:

```json
{
  "channels": {
    "slack": {
      "enabled": true,
      "botToken": "xoxb-xxx",
      "appToken": "xapp-xxx"
    }
  }
}
```

4. Run `nanobot gateway`

</details>

## Providers

Nanobot supports multiple LLM providers:

| Provider | Config key | Notes |
|----------|-----------|-------|
| **OpenRouter** | `openrouter` | Recommended — access to 100+ models |
| **LiteLLM** | `litellm` | Generic gateway for any LiteLLM-compatible endpoint |
| **Azure OpenAI** | `azure_openai` | Azure-hosted OpenAI models |
| **OpenAI** | `openai` | Direct OpenAI API |

## Docker

```bash
docker build -t nanobot-hive-mind .
docker run -v ~/.nanobot:/root/.nanobot nanobot-hive-mind gateway
```

## Development

```bash
pip install -e ".[dev]"
pytest                    # run tests
ruff check .              # lint
ruff format .             # format
```

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<sub>Built on [nanobot](https://github.com/HKUDS/nanobot) by [HKUDS](https://github.com/HKUDS). The Hive Mind orchestration layer is developed independently.</sub>
