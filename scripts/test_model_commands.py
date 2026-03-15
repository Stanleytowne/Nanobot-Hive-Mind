#!/usr/bin/env python3
"""Test /model command and @agent:model syntax."""

from __future__ import annotations

import asyncio
import shutil
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
            model=agent_model, max_iterations=config.agents.defaults.max_tool_iterations,
            context_window_tokens=ctx,
            web_search_config=config.tools.web.search,
            web_proxy=config.tools.web.proxy or None,
            exec_config=config.tools.exec, cron_service=cron,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            mcp_servers=config.tools.mcp_servers,
            channels_config=config.channels,
            agent_name=profile.name, agent_profile=profile,
        )

    orchestrator = OrchestratorLoop(
        config=config, bus=bus, registry=registry, provider=provider,
        create_specialist_loop=_create_specialist_loop,
    )

    responses = []

    async def capture(msg: OutboundMessage):
        responses.append(msg.content)
        preview = msg.content[:120].replace("\n", " ")
        print(f"  -> {preview}")

    bus.publish_outbound = capture

    async def send(content):
        print(f"\n>> {content}")
        responses.clear()
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="model_test", content=content)
        cmd_base = content.strip().lower().split()[0] if content.strip() else ""
        if cmd_base in ("/model", "/help", "/agents"):
            await orchestrator._handle_passthrough(msg)
        else:
            await orchestrator._dispatch(msg)
        return list(responses)

    # Test 1: Create a coding agent
    print("=" * 60)
    print("TEST 1: Create coding agent with default model")
    print("=" * 60)
    await send("Write a hello world in Python")
    for p in registry.list_agents():
        print(f"  Agent: {p.name}, model: {p.model or 'default'}")

    # Test 2: /model (list models)
    print("\n" + "=" * 60)
    print("TEST 2: /model command — list models")
    print("=" * 60)
    await send("/model")

    # Test 3: /model coding 0 (set to cheapest)
    print("\n" + "=" * 60)
    print("TEST 3: /model coding 0 — set to cheapest model")
    print("=" * 60)
    await send("/model coding 0")
    for p in registry.list_agents():
        if p.name == "coding":
            print(f"  Agent: {p.name}, model: {p.model}")

    # Test 4: @coding:heavy (inline model override)
    print("\n" + "=" * 60)
    print("TEST 4: @coding:heavy — inline model override to most expensive")
    print("=" * 60)
    await send("@coding:heavy Write a more advanced hello world with logging")
    for p in registry.list_agents():
        if p.name == "coding":
            print(f"  Agent: {p.name}, model: {p.model}")

    # Test 5: @coding:1 (by index)
    print("\n" + "=" * 60)
    print("TEST 5: @coding:1 — set model by index")
    print("=" * 60)
    await send("@coding:1 Show me what model you're using")
    for p in registry.list_agents():
        if p.name == "coding":
            print(f"  Agent: {p.name}, model: {p.model}")

    # Test 6: /model with invalid
    print("\n" + "=" * 60)
    print("TEST 6: /model with invalid model name")
    print("=" * 60)
    await send("/model coding nonexistent")

    # Test 7: /agents to see final state
    print("\n" + "=" * 60)
    print("TEST 7: /agents — final state")
    print("=" * 60)
    await send("/agents")


if __name__ == "__main__":
    asyncio.run(run())
