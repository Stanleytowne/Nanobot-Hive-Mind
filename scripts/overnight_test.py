#!/usr/bin/env python3
"""Overnight continuous testing for nanobot orchestrator.

Runs diverse simulation scenarios in a loop, writes reports after each run,
and accumulates an overnight summary.

Usage:
    python3 scripts/overnight_test.py [--runs 20] [--delay 2] [--context-window 8192]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

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
# Scenario definitions — each is a distinct user persona / workflow
# ---------------------------------------------------------------------------

SCENARIOS: list[dict] = [
    # Scenario 1: Quick coding task with context switch (lightweight)
    {
        "name": "Quick Code + Chat",
        "description": "Test basic routing, agent creation, and a short coding task",
        "context_window": 8192,
        "messages": [
            "Hey, I'm Alex, a backend dev in Singapore. I work with Python and Go.",
            "Write me a Python decorator that retries a function on exception, with configurable max_retries and delay.",
            "Now make it support async functions too.",
            "What's the best laksa in Singapore?",
            "Back to the decorator — add exponential backoff support.",
            "What do you remember about me?",
        ],
        "checks": [
            ("shared_memory_contains", "Alex"),
            ("shared_memory_contains", "Singapore"),
            ("min_agents", 1),
        ],
    },
    # Scenario 2: Heavy coding session to stress compaction
    {
        "name": "Heavy Coding — Compaction Stress",
        "description": "Generate lots of code to force multiple compaction cycles",
        "context_window": 8192,
        "messages": [
            "I'm a DevOps engineer building microservices. I use Python, Docker, and Kubernetes.",
            "Build a complete health check microservice in Python with FastAPI. Include /health, /ready, /live endpoints, dependency checks (DB, Redis, external API), and structured JSON responses.",
            "Add a Dockerfile for this service. Use multi-stage build, non-root user, and health check instruction.",
            "Write a Kubernetes deployment manifest with liveness and readiness probes pointing to our health endpoints. Include resource limits, rolling update strategy, and pod disruption budget.",
            "Now add Prometheus metrics integration — expose /metrics endpoint with request count, latency histogram, and health check status gauges.",
            "Write a Helm chart for deploying this service. Include values.yaml with configurable replicas, resource limits, and probe intervals.",
            "Add comprehensive integration tests using pytest and httpx. Test all endpoints, dependency failure scenarios, and metric exposition.",
            "What have we built so far? Give me a complete summary.",
        ],
        "checks": [
            ("min_agents", 1),
            ("check_compaction", None),
        ],
    },
    # Scenario 3: Multi-agent parallel dispatch
    {
        "name": "Multi-Agent Parallel",
        "description": "Test routing across many different agent types",
        "context_window": 16384,
        "messages": [
            "I'm Maria, a tech lead in Berlin. I manage a team of 5 engineers.",
            "Write a Python script to parse our nginx access logs and find the top 10 IP addresses by request count.",
            "Remind me to do the sprint retrospective on Friday at 2pm CET.",
            "What are some good team-building activities for a remote engineering team?",
            "Back to the log parser — also add detection for suspicious patterns: more than 100 requests per minute from a single IP, or requests to /admin from non-whitelisted IPs.",
            "I need to write a performance review for one of my engineers. Can you give me a template?",
            "Also, can you explain the CAP theorem in simple terms? I need to brief my team on distributed systems.",
            "What do you know about me so far?",
        ],
        "checks": [
            ("shared_memory_contains", "Maria"),
            ("shared_memory_contains", "Berlin"),
            ("min_agents", 2),
        ],
    },
    # Scenario 4: Rapid-fire context switching
    {
        "name": "Rapid Context Switching",
        "description": "Test router's ability to handle fast topic switches",
        "context_window": 8192,
        "messages": [
            "I'm Tom, a full-stack developer in Tokyo. I speak English and Japanese.",
            "Write a React hook for debounced search input.",
            "What's the best ramen shop in Shibuya?",
            "Now convert that React hook to Vue 3 composition API.",
            "Remind me to call the dentist tomorrow at 10am JST.",
            "Write a SQL query to find users who haven't logged in for 30 days but have active subscriptions.",
            "What's a good weekend hike near Tokyo?",
            "Back to the SQL — add an index recommendation for that query.",
            "How do you say 'thank you for your hard work' in Japanese? (I know it's otsukaresama but want the kanji)",
            "List everything you know about me.",
        ],
        "checks": [
            ("shared_memory_contains", "Tom"),
            ("shared_memory_contains", "Tokyo"),
            ("min_agents", 2),
        ],
    },
    # Scenario 5: Long conversation with single agent (compaction focus)
    {
        "name": "Deep Dive Single Agent",
        "description": "Extended single-topic conversation to test deep compaction",
        "context_window": 8192,
        "messages": [
            "I'm building a real-time chat application. Let's design the architecture together.",
            "Start with the WebSocket server in Python using FastAPI and websockets library. Handle connection, disconnection, and message broadcast.",
            "Add user authentication — JWT tokens verified on WebSocket handshake. Include a middleware approach.",
            "Now add chat rooms — users can join/leave rooms, and messages are scoped to rooms. Use an in-memory room manager.",
            "Add message persistence with PostgreSQL. Store messages with sender, room, timestamp, and content. Include a message history endpoint.",
            "Add typing indicators — when a user starts typing, broadcast a typing event to other users in the same room.",
            "Add rate limiting — max 10 messages per second per user, with a backpressure mechanism.",
            "Now add file attachments support — users can share images and files up to 5MB. Store in S3, send URL via WebSocket.",
            "Add end-to-end encryption using the Signal protocol. Key exchange during handshake, encrypt all messages.",
            "Write a comprehensive README.md for the project with setup instructions, API documentation, and architecture diagram in mermaid.",
        ],
        "checks": [
            ("check_compaction", None),
        ],
    },
    # Scenario 6: Error handling and edge cases
    {
        "name": "Edge Cases & Recovery",
        "description": "Test error handling, empty responses, and unusual inputs",
        "context_window": 8192,
        "messages": [
            "Hi there!",
            "",  # empty message
            "Write a one-liner Python function to check if a string is a palindrome.",
            "好的，用中文回复我。我是一个在上海的开发者。",  # Chinese input
            "Now switch back to English. Write a bash script to find and delete all .pyc files in a directory tree.",
            "What's 2+2?",
            "@coding write a simple hello world in Rust",  # manual routing
            "Tell me everything you know about me.",
        ],
        "checks": [
            ("agent_exists", None),
        ],
    },
    # Scenario 7: Memory extraction stress test
    {
        "name": "Memory Extraction Stress",
        "description": "Pack many personal facts into messages to test proactive memory",
        "context_window": 16384,
        "messages": [
            "Hey! I'm Sarah, 28 years old, working as a data scientist at a fintech startup in London. I'm originally from Canada. I have a golden retriever named Max. My timezone is GMT. I prefer dark mode and vim keybindings.",
            "I'm allergic to shellfish, so no seafood restaurant suggestions please. Also, I'm vegetarian on weekdays.",
            "My team uses Python, Spark, and dbt for data pipelines. We deploy on AWS with EKS. Our main DB is Redshift.",
            "Write me a PySpark job to deduplicate events from a Kafka topic. The events have an event_id and timestamp. Keep the latest version of each event.",
            "I have a meeting with the VP of Engineering next Tuesday at 3pm about our data quality initiative. Can you help me prepare talking points?",
            "What do you know about me? List EVERYTHING.",
        ],
        "checks": [
            ("shared_memory_contains", "Sarah"),
            ("shared_memory_contains", "London"),
            ("shared_memory_contains", "golden retriever"),
            ("shared_memory_contains", "vegetarian"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class RunResult:
    """Result from a single simulation run."""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.start_time = time.time()
        self.end_time: float = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.agents_created: list[str] = []
        self.compaction_count = 0
        self.errors: list[str] = []
        self.check_results: list[tuple[bool, str]] = []
        self.shared_memory_facts: int = 0
        self.total_tokens: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.check_results) and not self.errors

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "duration_s": round(self.duration, 1),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "agents": self.agents_created,
            "compactions": self.compaction_count,
            "errors": self.errors,
            "checks_passed": sum(1 for ok, _ in self.check_results if ok),
            "checks_total": len(self.check_results),
            "shared_memory_facts": self.shared_memory_facts,
            "total_tokens": self.total_tokens,
            "passed": self.passed,
        }


async def run_single_scenario(
    scenario: dict,
    delay: float = 2.0,
    context_window_override: int | None = None,
) -> RunResult:
    """Run a single scenario and return results."""
    result = RunResult(scenario["name"])
    context_window = context_window_override or scenario.get("context_window", 8192)

    # Load config fresh each run
    config = _load_runtime_config(None, None)
    workspace = config.workspace_path
    config.agents.defaults.context_window_tokens = context_window

    # Clean workspace
    for subdir in ["agents", "memory"]:
        d = workspace / subdir
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    sessions_dir = workspace / "sessions"
    if sessions_dir.exists():
        for f in sessions_dir.glob("*.jsonl"):
            f.unlink()
    # Clean workspace-generated files from previous coding tasks
    for ext in ("*.py", "*.ini", "*.yaml", "*.yml", "*.md", "*.json"):
        for f in workspace.glob(ext):
            if f.name not in ("MEMORY.md", "HISTORY.md"):
                try:
                    f.unlink()
                except Exception:
                    pass

    # Wire up components
    bus = MessageBus()
    provider = _make_provider(config)
    cron = CronService(get_cron_dir() / "jobs.json")
    registry = AgentRegistry(workspace)

    orch_cfg = config.agents.orchestrator

    async def _create_specialist_loop(profile):
        agent_model = profile.model or config.agents.defaults.model
        ctx = context_window
        if orch_cfg and orch_cfg.model_context_windows:
            for key, val in orch_cfg.model_context_windows.items():
                if key == agent_model or key.split("/")[-1] == agent_model.split("/")[-1] or key in agent_model:
                    ctx = val
                    break
        ctx = min(ctx, context_window) if context_window_override else ctx
        return AgentLoop(
            bus=bus, provider=provider, workspace=workspace,
            model=agent_model,
            max_iterations=config.agents.defaults.max_tool_iterations,
            context_window_tokens=ctx,
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

    orchestrator = OrchestratorLoop(
        config=config, bus=bus, registry=registry, provider=provider,
        create_specialist_loop=_create_specialist_loop,
    )

    # Capture outbound
    async def capture_outbound(msg: OutboundMessage) -> None:
        result.messages_received += 1
        is_notif = msg.metadata.get("_notification") or msg.metadata.get("_progress")
        prefix = "📡" if is_notif else "🤖"
        preview = msg.content[:150].replace("\n", " ")
        print(f"    {prefix} {preview}{'...' if len(msg.content) > 150 else ''}")

    bus.publish_outbound = capture_outbound

    # Send messages
    for i, content in enumerate(scenario["messages"]):
        if not content.strip():
            continue  # skip empty
        result.messages_sent += 1
        short = content[:80].replace("\n", " ")
        print(f"  [{i+1}/{len(scenario['messages'])}] User: {short}{'...' if len(content) > 80 else ''}")

        msg = InboundMessage(
            channel="cli", sender_id="user", chat_id="sim_test",
            content=content,
        )
        try:
            await orchestrator._dispatch(msg)
        except Exception as e:
            err = f"msg {i+1}: {type(e).__name__}: {e}"
            result.errors.append(err)
            print(f"    ❌ {err}")
            traceback.print_exc()

        if delay > 0 and i < len(scenario["messages"]) - 1:
            await asyncio.sleep(delay)

    # Collect results
    result.end_time = time.time()
    result.agents_created = [a.name for a in registry.list_agents()]

    # Count compactions
    for profile in registry.list_agents():
        sdir = workspace / "agents" / profile.name / "sessions"
        if not sdir.exists():
            continue
        for f in sdir.glob("*.jsonl"):
            try:
                meta = json.loads(f.read_text(encoding="utf-8").split("\n")[0])
                if meta.get("compaction_summary"):
                    result.compaction_count += 1
            except Exception:
                pass

    # Count shared memory facts
    mem_file = workspace / "memory" / "MEMORY.md"
    if mem_file.exists():
        lines = [l for l in mem_file.read_text(encoding="utf-8").split("\n") if l.strip().startswith("-")]
        result.shared_memory_facts = len(lines)

    # Token usage
    for profile in registry.list_agents():
        loop = registry.get_loop(profile.name)
        if loop:
            result.total_tokens += loop.token_usage.get("total_tokens", 0)
    result.total_tokens += orchestrator._router_token_usage.get("total_tokens", 0)

    # Run checks
    for check_type, check_arg in scenario.get("checks", []):
        if check_type == "shared_memory_contains":
            found = mem_file.exists() and check_arg.lower() in mem_file.read_text(encoding="utf-8").lower()
            result.check_results.append(
                (found, f"Memory contains '{check_arg}'" if found else f"'{check_arg}' NOT in memory")
            )
        elif check_type == "agent_exists":
            ok = len(result.agents_created) > 0
            result.check_results.append((ok, f"Agents: {result.agents_created}"))
        elif check_type == "agent_exists_by_type":
            found = any(check_arg.lower() in a.lower() for a in result.agents_created)
            result.check_results.append(
                (found, f"Agent type '{check_arg}'" if found else f"No '{check_arg}' agent")
            )
        elif check_type == "min_agents":
            ok = len(result.agents_created) >= check_arg
            result.check_results.append((ok, f"{len(result.agents_created)} agents (need {check_arg})"))
        elif check_type == "check_compaction":
            ok = result.compaction_count > 0
            result.check_results.append((ok, f"{result.compaction_count} compactions"))

    return result


# ---------------------------------------------------------------------------
# Overnight loop
# ---------------------------------------------------------------------------

async def overnight_loop(
    total_runs: int = 20,
    delay: float = 2.0,
    context_window: int | None = None,
    report_dir: str = "/tmp/nanobot_overnight",
) -> None:
    """Run scenarios in rotation, write reports."""
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    run_number = 0
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"OVERNIGHT TESTING — {total_runs} runs")
    print(f"Report dir: {report_path}")
    print(f"Context window override: {context_window or 'per-scenario'}")
    print(f"Delay: {delay}s")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    for run_number in range(total_runs):
        scenario = SCENARIOS[run_number % len(SCENARIOS)]
        timestamp = datetime.now().strftime("%H:%M:%S")

        print(f"\n{'─'*70}")
        print(f"RUN {run_number+1}/{total_runs} — {scenario['name']} — {timestamp}")
        print(f"  {scenario['description']}")
        print(f"{'─'*70}")

        try:
            result = await run_single_scenario(
                scenario, delay=delay, context_window_override=context_window,
            )
        except Exception as e:
            result = RunResult(scenario["name"])
            result.end_time = time.time()
            result.errors.append(f"FATAL: {type(e).__name__}: {e}")
            traceback.print_exc()

        # Print summary
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n  {status} | {result.duration:.0f}s | {result.messages_sent} msgs | "
              f"{len(result.agents_created)} agents | {result.compaction_count} compactions | "
              f"{result.total_tokens:,} tokens | {result.shared_memory_facts} facts | "
              f"{len(result.errors)} errors")
        for ok, detail in result.check_results:
            icon = "✓" if ok else "✗"
            print(f"    {icon} {detail}")
        for err in result.errors:
            print(f"    ✗ {err[:120]}")

        all_results.append(result.to_dict())

        # Save incremental report every run
        report_file = report_path / "overnight_results.json"
        report_file.write_text(json.dumps({
            "started": datetime.fromtimestamp(start_time).isoformat(),
            "updated": datetime.now().isoformat(),
            "total_runs": run_number + 1,
            "results": all_results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        # Write human-readable summary every 3 runs
        if (run_number + 1) % 3 == 0 or run_number == total_runs - 1:
            _write_summary(report_path, all_results, start_time)

        # Brief cooldown between runs
        if run_number < total_runs - 1:
            await asyncio.sleep(5)

    # Final summary
    _write_summary(report_path, all_results, start_time)
    print(f"\n{'='*70}")
    print(f"OVERNIGHT TESTING COMPLETE — {len(all_results)} runs")
    elapsed = time.time() - start_time
    passed = sum(1 for r in all_results if r["passed"])
    print(f"Passed: {passed}/{len(all_results)}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Reports: {report_path}")
    print(f"{'='*70}")


def _write_summary(report_path: Path, results: list[dict], start_time: float) -> None:
    """Write a human-readable summary report."""
    lines = []
    lines.append(f"# Nanobot Overnight Test Report")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Duration: {(time.time() - start_time)/60:.1f} minutes")
    lines.append(f"Total runs: {len(results)}")
    lines.append("")

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    lines.append(f"## Summary: {passed} passed, {failed} failed")
    lines.append("")

    total_tokens = sum(r["total_tokens"] for r in results)
    total_msgs = sum(r["messages_sent"] for r in results)
    total_compactions = sum(r["compactions"] for r in results)
    lines.append(f"- Total messages: {total_msgs}")
    lines.append(f"- Total tokens: {total_tokens:,}")
    lines.append(f"- Total compactions: {total_compactions}")
    lines.append("")

    lines.append("## Run Details")
    lines.append("")
    for i, r in enumerate(results):
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"### Run {i+1}: {r['scenario']} [{status}]")
        lines.append(f"- Duration: {r['duration_s']}s")
        lines.append(f"- Agents: {', '.join(r['agents']) if r['agents'] else 'none'}")
        lines.append(f"- Compactions: {r['compactions']}")
        lines.append(f"- Tokens: {r['total_tokens']:,}")
        lines.append(f"- Memory facts: {r['shared_memory_facts']}")
        lines.append(f"- Checks: {r['checks_passed']}/{r['checks_total']}")
        if r["errors"]:
            lines.append(f"- Errors:")
            for e in r["errors"]:
                lines.append(f"  - {e[:200]}")
        lines.append("")

    # Common issues
    all_errors = []
    for r in results:
        all_errors.extend(r["errors"])
    if all_errors:
        lines.append("## Errors Summary")
        from collections import Counter
        error_types = Counter()
        for e in all_errors:
            # Extract error type
            if ":" in e:
                etype = e.split(":")[0].strip().split()[-1]
                error_types[etype] += 1
            else:
                error_types[e[:50]] += 1
        for etype, count in error_types.most_common():
            lines.append(f"- {etype}: {count} occurrences")
        lines.append("")

    summary_file = report_path / "overnight_summary.md"
    summary_file.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Nanobot overnight testing")
    parser.add_argument("--runs", type=int, default=20, help="Number of simulation runs")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between messages (seconds)")
    parser.add_argument("--context-window", type=int, default=None,
                        help="Override context window (default: per-scenario)")
    parser.add_argument("--report-dir", type=str, default="/tmp/nanobot_overnight",
                        help="Directory for reports")
    args = parser.parse_args()
    asyncio.run(overnight_loop(
        total_runs=args.runs,
        delay=args.delay,
        context_window=args.context_window,
        report_dir=args.report_dir,
    ))


if __name__ == "__main__":
    main()
