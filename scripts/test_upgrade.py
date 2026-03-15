#!/usr/bin/env python3
"""Test model upgrade flow via __upgrade__ signal.

Sends messages that should trigger the router to upgrade an agent's model
when the user expresses dissatisfaction with quality.
"""

from __future__ import annotations

import asyncio
import json
import sys
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


async def run():
    config = _load_runtime_config(None, None)
    workspace = config.workspace_path

    # Clean state
    import shutil
    for subdir in ["agents", "memory"]:
        d = workspace / subdir
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    for f in (workspace / "sessions").glob("*.jsonl"):
        f.unlink()

    bus = MessageBus()
    provider = _make_provider(config)
    cron = CronService(get_cron_dir() / "jobs.json")
    registry = AgentRegistry(workspace)
    orch_cfg = config.agents.orchestrator

    async def _create_specialist_loop(profile):
        agent_model = profile.model or config.agents.defaults.model
        ctx = config.agents.defaults.context_window_tokens
        if orch_cfg and orch_cfg.model_context_windows:
            for key, val in orch_cfg.model_context_windows.items():
                if key == agent_model or key.split("/")[-1] == agent_model.split("/")[-1] or key in agent_model:
                    ctx = val
                    break
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

    notifications = []

    async def capture(msg: OutboundMessage):
        is_notif = msg.metadata.get("_notification") or msg.metadata.get("_progress")
        prefix = "  📡" if is_notif else "  🤖"
        preview = msg.content[:200].replace("\n", " ")
        print(f"{prefix} {preview}")
        notifications.append(msg.content)

    bus.publish_outbound = capture

    messages = [
        # Step 1: Create coding agent with a simple task
        "Write a Python function that checks if a number is prime.",
        # Step 2: Express dissatisfaction to trigger upgrade
        "That's too simple and not good enough. Can you write a much better version with proper optimization?",
        # Step 3: Another dissatisfaction (should upgrade again)
        "The quality is still poor. Rewrite it properly with Miller-Rabin primality test.",
        # Step 4: Check what model we're on
        "What model are you using? Also, summarize what we've done.",
    ]

    models_seen = []

    for i, content in enumerate(messages):
        print(f"\n{'─'*60}")
        print(f"[{i+1}] User: {content}")
        print(f"{'─'*60}")

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="upgrade_test", content=content)
        await orchestrator._dispatch(msg)

        # Check registry for model changes
        for profile in registry.list_agents():
            model = profile.model or config.agents.defaults.model
            short = model.split("/")[-1]
            if not models_seen or models_seen[-1] != f"{profile.name}:{short}":
                models_seen.append(f"{profile.name}:{short}")
                print(f"\n  📊 Agent '{profile.name}' model: {short}")

        await asyncio.sleep(3)

    # Final report
    print(f"\n{'='*60}")
    print("UPGRADE TEST REPORT")
    print(f"{'='*60}")

    print("\nAgents:")
    for profile in registry.list_agents():
        model = profile.model or "default"
        loop = registry.get_loop(profile.name)
        tokens = loop.token_usage.get("total_tokens", 0) if loop else 0
        print(f"  {profile.name}: model={model.split('/')[-1]}, tokens={tokens:,}")

    print(f"\nModel progression: {' → '.join(models_seen)}")

    # Check if upgrades happened
    upgrade_notifs = [n for n in notifications if "Upgraded" in n or "⬆️" in n]
    if upgrade_notifs:
        print(f"\nUpgrade notifications ({len(upgrade_notifs)}):")
        for n in upgrade_notifs:
            print(f"  {n}")
    else:
        print("\n⚠️  No upgrade notifications detected")

    # Check router upgrade signals
    print(f"\nRouter last_upgrades: {orchestrator.router.last_upgrades}")
    print(f"Available models: {[m.split('/')[-1] for m in orchestrator.available_models]}")


if __name__ == "__main__":
    asyncio.run(run())
