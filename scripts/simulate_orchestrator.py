#!/usr/bin/env python3
"""Comprehensive orchestrator simulation.

Simulates a multi-day user workflow with Stanley — a senior software engineer
in Hong Kong — sending ~40 messages across 9 phases through the full orchestrator
pipeline with real LLM calls.

Usage:
    python scripts/simulate_orchestrator.py [--context-window 8192] [--delay 5]

Requires:
    - ~/.nanobot/config.json with orchestrator.enabled=true and valid API keys
    - All nanobot dependencies installed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure nanobot is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanobot.agent.loop import AgentLoop
from nanobot.agent.orchestrator import OrchestratorLoop
from nanobot.agent.registry import AgentRegistry
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.cli.commands import _load_runtime_config, _make_provider
from nanobot.config.paths import get_cron_dir
from nanobot.cron.service import CronService
from nanobot.session.manager import SessionManager


# ---------------------------------------------------------------------------
# Colored output
# ---------------------------------------------------------------------------

def _c(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m"

def green(t: str) -> str: return _c(t, 32)
def red(t: str) -> str: return _c(t, 31)
def yellow(t: str) -> str: return _c(t, 33)
def cyan(t: str) -> str: return _c(t, 36)
def dim(t: str) -> str: return _c(t, 2)
def bold(t: str) -> str: return _c(t, 1)


# ---------------------------------------------------------------------------
# Simulation messages per phase
# ---------------------------------------------------------------------------

PHASES: list[dict] = [
    # Phase 1: Personal Setup & Casual Chat
    {
        "name": "Phase 1: Personal Setup & Casual Chat",
        "goal": "Test agent creation, router memory extraction, shared memory",
        "messages": [
            "Hi! I'm Stanley, a senior software engineer based in Hong Kong. I prefer concise, code-focused responses. My timezone is UTC+8.",
            "What's the weather usually like in Hong Kong in March?",
            "Any good dim sum spots near Central?",
        ],
        "checks": [
            ("shared_memory_contains", "Stanley"),
            ("shared_memory_contains", "Hong Kong"),
            ("agent_exists", None),  # at least one agent
        ],
    },
    # Phase 2: Coding Work Session
    {
        "name": "Phase 2: Coding Work Session",
        "goal": "Test coding agent with multi-turn tasks",
        "messages": [
            "Design a FastAPI middleware for request logging. Include request method, path, status code, and response time.",
            "Add correlation IDs using UUID4 — inject as X-Request-ID header and include in all log lines.",
            "Now add retry logic for transient errors (5xx). Use exponential backoff with max 3 retries.",
            "Write pytest tests for the middleware — cover success, error, retry, and correlation ID propagation.",
        ],
        "checks": [
            ("agent_exists_by_type", "coding"),
        ],
    },
    # Phase 3: Context Switch
    {
        "name": "Phase 3: Context Switch",
        "goal": "Test routing between agents, multi-task",
        "messages": [
            "Remind me to review the PR tomorrow at 3pm HKT.",
            "I'm thinking of a weekend trip to Macau. Any must-visit spots for food?",
            "Back to coding — where were we with the middleware? Give me a quick summary of what we've built so far.",
        ],
        "checks": [
            ("min_agents", 2),
        ],
    },
    # Phase 4: Deep Coding Session (Fill Context)
    {
        "name": "Phase 4: Deep Coding Session (Fill Context)",
        "goal": "Force compaction by generating lots of coding tokens",
        "messages": [
            "Build a full CRUD API for a user management system with FastAPI. Include models for User (id, email, name, role, created_at), endpoints for create/read/update/delete, and proper error handling.",
            "Add JWT authentication — include login endpoint, token generation, password hashing with bcrypt, and a dependency for protecting routes.",
            "Add database migrations with Alembic. Show the migration script for creating the users table and the alembic configuration.",
            "Add comprehensive input validation with Pydantic models — request schemas, response schemas, and custom validators for email and password strength.",
            "Generate OpenAPI documentation configuration with custom tags, descriptions, and example requests/responses for every endpoint.",
        ],
        "checks": [
            ("check_compaction", None),
        ],
    },
    # Phase 5: Evening — Memory Recall
    {
        "name": "Phase 5: Evening — Memory Recall",
        "goal": "Test memory recall after compaction",
        "messages": [
            "What have we built together today? Give me a high-level summary.",
            "By the way, I'm going hiking this weekend at Dragon's Back trail in Hong Kong.",
            "What do you know about me so far? List everything.",
        ],
        "checks": [
            ("shared_memory_contains", "Dragon"),
        ],
    },
    # Phase 6: Morning Day 2 — Fresh Start
    {
        "name": "Phase 6: Morning Day 2 — Fresh Start",
        "goal": "Test persistence across session boundaries",
        "messages": [
            "Good morning! Ready for another productive day.",
            "Remind me — what were we working on yesterday with the user API?",
            "I need to add rate limiting to the user API. Use a token bucket algorithm, 100 requests per minute per user.",
        ],
        "checks": [],
    },
    # Phase 7: Cross-Agent Knowledge Sharing
    {
        "name": "Phase 7: Cross-Agent Knowledge Sharing",
        "goal": "Test shared memory between agents",
        "messages": [
            "Important deployment constraint: we deploy on Kubernetes with 512MB memory pods. Keep this in mind for all code suggestions.",
            "What's a good lunch spot near Causeway Bay? Something quick.",
            "Back to the user API — considering our k8s memory constraint, should we use connection pooling for the database?",
        ],
        "checks": [
            ("shared_memory_contains", "k8s"),
        ],
    },
    # Phase 8: Second Compaction Cycle
    {
        "name": "Phase 8: Second Compaction Cycle",
        "goal": "Test compaction-on-compaction",
        "messages": [
            "Add a Redis caching layer to the user API. Cache user lookups for 5 minutes, invalidate on update/delete.",
            "Add WebSocket support for real-time user status updates. Include connection management, heartbeat, and broadcast.",
            "Add background task processing with Celery. Set up a task for sending welcome emails and another for periodic user data cleanup.",
            "Refactor the entire project into a clean architecture with separate layers: routes, services, repositories, and schemas.",
        ],
        "checks": [
            ("check_compaction", None),
        ],
    },
    # Phase 9: Final Recall & Wrap-up
    {
        "name": "Phase 9: Final Recall & Wrap-up",
        "goal": "Comprehensive memory and knowledge test",
        "messages": [
            "Give me a complete summary of everything we've built for the user management system.",
            "What personal things do you know about me? List them all.",
        ],
        "checks": [],
    },
]


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

class SimulationReport:
    """Collects results for final report."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.phases: list[dict] = []
        self.messages_sent = 0
        self.messages_received = 0
        self.errors: list[str] = []
        self.start_time = time.time()

    def add_phase(self, name: str, results: dict) -> None:
        self.phases.append({"name": name, **results})

    def check_shared_memory_contains(self, keyword: str) -> tuple[bool, str]:
        mem_file = self.workspace / "memory" / "MEMORY.md"
        if not mem_file.exists():
            return False, f"MEMORY.md does not exist"
        content = mem_file.read_text(encoding="utf-8")
        if keyword.lower() in content.lower():
            return True, f"Found '{keyword}' in shared memory"
        return False, f"'{keyword}' NOT found in shared memory"

    def check_agent_exists(self) -> tuple[bool, str]:
        registry_file = self.workspace / "agents" / "registry.json"
        if not registry_file.exists():
            return False, "No registry.json"
        data = json.loads(registry_file.read_text(encoding="utf-8"))
        agents = data.get("agents", [])
        if agents:
            names = [a["name"] for a in agents]
            return True, f"Agents: {', '.join(names)}"
        return False, "No agents in registry"

    def check_agent_exists_by_type(self, agent_type: str) -> tuple[bool, str]:
        registry_file = self.workspace / "agents" / "registry.json"
        if not registry_file.exists():
            return False, "No registry.json"
        data = json.loads(registry_file.read_text(encoding="utf-8"))
        for a in data.get("agents", []):
            if agent_type.lower() in a["name"].lower() or agent_type.lower() in a.get("description", "").lower():
                return True, f"Found agent matching '{agent_type}': {a['name']}"
        names = [a["name"] for a in data.get("agents", [])]
        return False, f"No agent matching '{agent_type}' (have: {', '.join(names)})"

    def check_min_agents(self, n: int) -> tuple[bool, str]:
        registry_file = self.workspace / "agents" / "registry.json"
        if not registry_file.exists():
            return False, "No registry.json"
        data = json.loads(registry_file.read_text(encoding="utf-8"))
        count = len(data.get("agents", []))
        if count >= n:
            return True, f"{count} agents (>= {n})"
        return False, f"Only {count} agents (need >= {n})"

    def check_compaction(self, registry: AgentRegistry) -> tuple[bool, str]:
        """Check if any agent session has a compaction_summary."""
        for profile in registry.list_agents():
            session_dir = self.workspace / "agents" / profile.name / "sessions"
            if not session_dir.exists():
                continue
            for f in session_dir.glob("*.jsonl"):
                try:
                    first_line = f.read_text(encoding="utf-8").split("\n")[0]
                    meta = json.loads(first_line)
                    if meta.get("compaction_summary"):
                        return True, f"Compaction found in {profile.name}: {meta['compaction_summary'][:80]}..."
                except Exception:
                    continue
        return False, "No compaction summary found in any agent session"

    def print_report(self, registry: AgentRegistry) -> None:
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print(bold("SIMULATION REPORT"))
        print("=" * 70)
        print(f"Duration: {elapsed:.1f}s")
        print(f"Messages sent: {self.messages_sent}")
        print(f"Messages received: {self.messages_received}")
        print()

        # Agent summary
        agents = registry.list_agents()
        print(bold("Agents Created:"))
        for a in agents:
            loop = registry.get_loop(a.name)
            tokens = loop.token_usage if loop else {}
            model = a.model or "default"
            print(f"  {green('●')} {a.name} [{a.status}] model={model} tokens={tokens.get('total_tokens', 0):,}")
        print()

        # Shared memory
        mem_file = self.workspace / "memory" / "MEMORY.md"
        if mem_file.exists():
            content = mem_file.read_text(encoding="utf-8").strip()
            print(bold("Shared Memory (MEMORY.md):"))
            for line in content.split("\n")[:20]:
                print(f"  {line}")
            if content.count("\n") > 20:
                print(f"  ... ({content.count(chr(10)) - 20} more lines)")
        print()

        # Phase results
        total_checks = 0
        passed_checks = 0
        print(bold("Phase Results:"))
        for phase in self.phases:
            name = phase["name"]
            checks = phase.get("checks", [])
            msgs = phase.get("message_count", 0)
            errs = phase.get("errors", [])

            check_pass = sum(1 for c in checks if c[0])
            check_total = len(checks)
            total_checks += check_total
            passed_checks += check_pass

            status = green("PASS") if check_pass == check_total and not errs else red("FAIL")
            print(f"\n  {status} {name} ({msgs} messages, {check_total} checks)")
            for ok, detail in checks:
                icon = green("✓") if ok else red("✗")
                print(f"    {icon} {detail}")
            for err in errs:
                print(f"    {red('✗')} Error: {err}")

        print()
        # Compaction check
        ok, detail = self.check_compaction(registry)
        total_checks += 1
        if ok:
            passed_checks += 1
        icon = green("✓") if ok else red("✗")
        print(f"  {icon} Compaction: {detail}")

        # History files
        for agent in agents:
            hist_file = self.workspace / "agents" / agent.name / "memory" / "HISTORY.md"
            if hist_file.exists():
                lines = hist_file.read_text(encoding="utf-8").strip().split("\n")
                print(f"  {green('✓')} {agent.name}/HISTORY.md: {len(lines)} lines")

        print()
        print("=" * 70)
        final = green("ALL CHECKS PASSED") if passed_checks == total_checks else red(f"{passed_checks}/{total_checks} CHECKS PASSED")
        print(f"  {final}")
        if self.errors:
            print(f"  {red(f'{len(self.errors)} errors encountered')}")
            for e in self.errors[:5]:
                print(f"    - {e[:100]}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

async def run_simulation(
    context_window: int = 8192,
    delay: float = 5.0,
    workspace_override: str | None = None,
) -> None:
    """Run the full simulation."""

    # Load config
    config = _load_runtime_config(None, workspace_override)
    workspace = config.workspace_path

    # Override context window for faster compaction
    original_cw = config.agents.defaults.context_window_tokens
    config.agents.defaults.context_window_tokens = context_window

    print(bold(f"\n{'='*70}"))
    print(bold("NANOBOT ORCHESTRATOR SIMULATION"))
    print(bold(f"{'='*70}"))
    print(f"Workspace: {workspace}")
    print(f"Model: {config.agents.defaults.model}")
    print(f"Router model: {config.agents.orchestrator.router_model}")
    print(f"Available models: {config.agents.orchestrator.models}")
    print(f"Context window: {context_window} (original: {original_cw})")
    print(f"Message delay: {delay}s")
    print()

    # Reset state for clean simulation
    print(dim("Resetting workspace state..."))
    for subdir in ["agents", "memory"]:
        d = workspace / subdir
        if d.exists():
            import shutil
            shutil.rmtree(d)
            d.mkdir(parents=True)
    # Clear session files (but keep sessions dir)
    sessions_dir = workspace / "sessions"
    if sessions_dir.exists():
        for f in sessions_dir.glob("*.jsonl"):
            f.unlink()
    print(dim("State reset complete."))

    # Wire up components
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(workspace)
    cron_store = get_cron_dir() / "jobs.json"
    cron = CronService(cron_store)

    registry = AgentRegistry(workspace)
    report = SimulationReport(workspace)

    # Specialist loop factory (mirrors gateway startup)
    orch_cfg = config.agents.orchestrator

    async def _create_specialist_loop(profile):
        agent_model = profile.model or config.agents.defaults.model
        ctx_window = context_window  # Use simulation override
        if orch_cfg and orch_cfg.model_context_windows:
            for key, val in orch_cfg.model_context_windows.items():
                if key == agent_model or key.split("/")[-1] == agent_model.split("/")[-1] or key in agent_model:
                    ctx_window = val
                    break
        # Force small context window for simulation
        ctx_window = min(ctx_window, context_window)

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=workspace,
            model=agent_model,
            max_iterations=config.agents.defaults.max_tool_iterations,
            context_window_tokens=ctx_window,
            web_search_config=config.tools.web.search,
            web_proxy=config.tools.web.proxy or None,
            exec_config=config.tools.exec,
            cron_service=cron,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            mcp_servers=config.tools.mcp_servers,
            channels_config=config.channels,
            agent_name=profile.name,
            agent_profile=profile,
        )
        return loop

    orchestrator = OrchestratorLoop(
        config=config,
        bus=bus,
        registry=registry,
        provider=provider,
        create_specialist_loop=_create_specialist_loop,
    )

    # Collect outbound messages
    outbound_messages: list[OutboundMessage] = []
    original_publish = bus.publish_outbound

    async def capture_and_print(msg: OutboundMessage) -> None:
        outbound_messages.append(msg)
        report.messages_received += 1
        is_notification = msg.metadata.get("_notification") or msg.metadata.get("_progress")
        if is_notification:
            print(dim(f"  📡 {msg.content[:120]}"))
        else:
            preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            print(cyan(f"  🤖 {preview}"))

    bus.publish_outbound = capture_and_print

    # Run phases
    session_key = "sim:test"
    chat_id = "sim_test"

    for phase_idx, phase in enumerate(PHASES):
        print(f"\n{bold(f'── {phase['name']} ──')}")
        print(dim(f"Goal: {phase['goal']}"))

        phase_errors = []
        phase_message_count = 0

        for msg_idx, content in enumerate(phase["messages"]):
            report.messages_sent += 1
            phase_message_count += 1
            print(f"\n  {yellow(f'[{phase_idx+1}.{msg_idx+1}]')} {bold('Stanley:')} {content[:100]}{'...' if len(content) > 100 else ''}")

            msg = InboundMessage(
                channel="cli",
                sender_id="stanley",
                chat_id=chat_id,
                content=content,
            )

            try:
                await orchestrator._dispatch(msg)
            except Exception as e:
                error_msg = f"Phase {phase_idx+1} msg {msg_idx+1}: {type(e).__name__}: {e}"
                phase_errors.append(error_msg)
                report.errors.append(error_msg)
                print(red(f"  ❌ Error: {e}"))

            if delay > 0 and msg_idx < len(phase["messages"]) - 1:
                await asyncio.sleep(delay)

        # Run checks
        check_results = []
        for check_type, check_arg in phase.get("checks", []):
            if check_type == "shared_memory_contains":
                ok, detail = report.check_shared_memory_contains(check_arg)
            elif check_type == "agent_exists":
                ok, detail = report.check_agent_exists()
            elif check_type == "agent_exists_by_type":
                ok, detail = report.check_agent_exists_by_type(check_arg)
            elif check_type == "min_agents":
                ok, detail = report.check_min_agents(check_arg)
            elif check_type == "check_compaction":
                ok, detail = report.check_compaction(registry)
            else:
                ok, detail = False, f"Unknown check: {check_type}"
            check_results.append((ok, detail))
            icon = green("✓") if ok else red("✗")
            print(f"\n  {icon} Check: {detail}")

        report.add_phase(phase["name"], {
            "message_count": phase_message_count,
            "checks": check_results,
            "errors": phase_errors,
        })

        # Inter-phase delay
        if delay > 0 and phase_idx < len(PHASES) - 1:
            print(dim(f"\n  (waiting {delay}s between phases...)"))
            await asyncio.sleep(delay)

    # Print final report
    report.print_report(registry)

    # Save report to file
    report_path = workspace / "simulation_report.txt"
    # Redirect print to file
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    report.print_report(registry)
    sys.stdout = old_stdout
    report_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Nanobot orchestrator simulation")
    parser.add_argument(
        "--context-window", type=int, default=8192,
        help="Context window size (small = faster compaction, default: 8192)"
    )
    parser.add_argument(
        "--delay", type=float, default=5.0,
        help="Delay between messages in seconds (default: 5)"
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Override workspace directory"
    )
    args = parser.parse_args()
    asyncio.run(run_simulation(
        context_window=args.context_window,
        delay=args.delay,
        workspace_override=args.workspace,
    ))


if __name__ == "__main__":
    main()
