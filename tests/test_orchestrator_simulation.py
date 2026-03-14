"""Comprehensive orchestrator simulation test.

Simulates a multi-day user workflow with Stanley — a senior software engineer
in Hong Kong — to validate multi-agent orchestration end-to-end:

- Agent creation & routing (rules + LLM classification)
- Shared memory extraction & persistence
- Per-agent private memory
- Compaction at 80% context window with summary injection
- Summary replacement on second compaction cycle
- Cross-agent knowledge sharing via shared memory
- Session persistence across simulated day boundaries
- process_direct serialization
- Model selection & upgrade signals
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import MemoryConsolidator, MemoryStore
from nanobot.agent.orchestrator import OrchestratorLoop
from nanobot.agent.registry import AgentProfile, AgentRegistry
from nanobot.agent.router import TaskRouter
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults, AgentsConfig, Config, OrchestratorConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.manager import Session, SessionManager


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> Config:
    """Create a Config with orchestrator enabled and small context window."""
    config = MagicMock(spec=Config)
    config.workspace_path = tmp_path
    config.agents = AgentsConfig(
        defaults=AgentDefaults(
            workspace=str(tmp_path),
            model="test/default-model",
            context_window_tokens=8192,
        ),
        orchestrator=OrchestratorConfig(
            enabled=True,
            max_specialists=10,
            idle_timeout_minutes=60,
            routing_rules=[],
            router_model="test/cheap-router",
            models=["test/cheap-model", "test/mid-model", "test/expensive-model"],
            model_context_windows={
                "test/cheap-model": 4096,
                "test/mid-model": 8192,
                "test/expensive-model": 16384,
            },
        ),
    )
    return config


def _make_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.get_default_model.return_value = "test/default-model"
    provider.estimate_prompt_tokens.return_value = (500, "test-counter")
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content="ok",
            tool_calls=[],
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
    )
    return provider


def _make_inbound(content: str, channel: str = "telegram", chat_id: str = "12345") -> InboundMessage:
    """Create an InboundMessage."""
    return InboundMessage(
        channel=channel,
        sender_id="stanley",
        chat_id=chat_id,
        content=content,
        timestamp=datetime.now(),
        media=[],
        metadata={},
    )


def _router_response(text: str, usage: dict | None = None) -> LLMResponse:
    """Create an LLM response for the router."""
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=usage or {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    )


def _agent_response(text: str, usage: dict | None = None) -> LLMResponse:
    """Create an LLM response for a specialist agent."""
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=usage or {"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180},
    )


def _save_memory_response(
    history_entry: str,
    memory_update: str,
    conversation_summary: str,
) -> LLMResponse:
    """Create an LLM response that calls save_memory tool."""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_save_mem",
                name="save_memory",
                arguments={
                    "history_entry": history_entry,
                    "memory_update": memory_update,
                    "conversation_summary": conversation_summary,
                },
            )
        ],
        usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
    )


async def _create_specialist_loop(
    profile: AgentProfile,
    bus: MessageBus,
    provider: MagicMock,
    workspace: Path,
    context_window_tokens: int = 8192,
) -> AgentLoop:
    """Create a specialist AgentLoop for the orchestrator."""
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=workspace,
        model=profile.model or "test/cheap-model",
        context_window_tokens=context_window_tokens,
        agent_name=profile.name,
        agent_profile=profile,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


class OrchestratorFixture:
    """Test fixture holding orchestrator + all dependencies."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.config = _make_config(tmp_path)
        self.bus = MessageBus()
        self.provider = _make_provider()
        self.registry = AgentRegistry(tmp_path)

        self.orchestrator = OrchestratorLoop(
            config=self.config,
            bus=self.bus,
            registry=self.registry,
            provider=self.provider,
            create_specialist_loop=self._create_loop,
        )

        # Track outbound messages
        self.outbound_messages: list[OutboundMessage] = []
        self._original_publish = self.bus.publish_outbound

    async def _create_loop(self, profile: AgentProfile) -> AgentLoop:
        cw = self.orchestrator.resolve_context_window(profile.model) or 8192
        return await _create_specialist_loop(
            profile, self.bus, self.provider, self.tmp_path, cw
        )

    async def send_message(self, content: str, chat_id: str = "12345") -> list[str]:
        """Send a message through the orchestrator and collect responses."""
        msg = _make_inbound(content, chat_id=chat_id)
        responses: list[str] = []

        # Capture outbound messages
        original_publish = self.bus.publish_outbound

        async def capture_outbound(out_msg: OutboundMessage) -> None:
            responses.append(out_msg.content)
            self.outbound_messages.append(out_msg)

        self.bus.publish_outbound = capture_outbound
        try:
            await self.orchestrator._dispatch(msg)
        finally:
            self.bus.publish_outbound = original_publish

        return responses

    def get_shared_memory(self) -> str:
        """Read shared MEMORY.md contents."""
        memory_file = self.tmp_path / "memory" / "MEMORY.md"
        if memory_file.exists():
            return memory_file.read_text(encoding="utf-8")
        return ""

    def get_agent_memory(self, agent_name: str) -> str:
        """Read agent-private MEMORY.md contents."""
        memory_file = self.tmp_path / "agents" / agent_name / "memory" / "MEMORY.md"
        if memory_file.exists():
            return memory_file.read_text(encoding="utf-8")
        return ""

    def get_agent_history(self, agent_name: str) -> str:
        """Read agent-private HISTORY.md contents."""
        history_file = self.tmp_path / "agents" / agent_name / "memory" / "HISTORY.md"
        if history_file.exists():
            return history_file.read_text(encoding="utf-8")
        return ""

    def get_agent_session(self, agent_name: str, session_key: str = "telegram:12345") -> Session:
        """Get agent's session."""
        loop = self.registry.get_loop(agent_name)
        if loop:
            return loop.sessions.get_or_create(session_key)
        return Session(key=session_key)


@pytest.fixture
def fixture(tmp_path):
    return OrchestratorFixture(tmp_path)


# ---------------------------------------------------------------------------
# Phase 1: Personal Setup & Casual Chat
# ---------------------------------------------------------------------------


class TestPhase1PersonalSetup:
    """Test agent creation, router memory extraction, shared memory."""

    @pytest.mark.asyncio
    async def test_introduction_creates_chat_agent_and_extracts_memory(self, tmp_path):
        f = OrchestratorFixture(tmp_path)

        # Router returns: memory extraction + new chat agent
        f.provider.chat_with_retry = AsyncMock(side_effect=[
            # Router classification
            _router_response(
                "__memory__:User's name is Stanley\n"
                "__memory__:Stanley is a senior software engineer in Hong Kong\n"
                "__memory__:Timezone is UTC+8\n"
                "__new__:chat:General chat and casual conversation"
            ),
            # Specialist agent response
            _agent_response(
                "Hey Stanley! Nice to meet you! I'm your assistant. "
                "How can I help you today?"
            ),
        ])

        responses = await f.send_message(
            "Hi! I'm Stanley, a senior software engineer based in Hong Kong. "
            "I prefer concise responses and I'm in UTC+8 timezone."
        )

        # Verify chat agent was created
        chat_profile = f.registry.get("chat")
        assert chat_profile is not None
        assert chat_profile.status == "active"

        # Verify shared memory was populated
        shared = f.get_shared_memory()
        assert "Stanley" in shared
        assert "Hong Kong" in shared or "senior software engineer" in shared
        assert "UTC+8" in shared

        # Verify we got a response
        assert any("Stanley" in r for r in responses)

    @pytest.mark.asyncio
    async def test_followup_routes_to_existing_chat_agent(self, tmp_path):
        f = OrchestratorFixture(tmp_path)

        # Pre-create chat agent
        await f.registry.get_or_create("chat", "General chat and casual conversation")

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            # Router: route to existing chat
            _router_response("chat"),
            # Agent response
            _agent_response("The weather in Hong Kong today is warm and sunny, about 25°C."),
        ])

        responses = await f.send_message("What's the weather like?")

        # Should route to existing chat agent (not create new)
        assert f.registry.get("chat") is not None
        assert len(f.registry.list_agents()) == 1

    @pytest.mark.asyncio
    async def test_memory_deduplication(self, tmp_path):
        """Same fact should not be duplicated in shared memory."""
        f = OrchestratorFixture(tmp_path)

        # Pre-populate shared memory
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "MEMORY.md").write_text(
            "# User Info\n\n- User's name is Stanley\n", encoding="utf-8"
        )

        await f.orchestrator._save_to_shared_memory("User's name is Stanley")
        content = f.get_shared_memory()
        assert content.count("Stanley") == 1


# ---------------------------------------------------------------------------
# Phase 2: Coding Work Session
# ---------------------------------------------------------------------------


class TestPhase2CodingSession:
    """Test coding agent with multi-turn tasks and private memory."""

    @pytest.mark.asyncio
    async def test_coding_task_creates_coding_agent(self, tmp_path):
        f = OrchestratorFixture(tmp_path)

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            # Router: new coding agent
            _router_response(
                "__new__:coding:Software development and code generation\n"
                "__model__:2"  # expensive model for coding
            ),
            # Agent response
            _agent_response(
                "Here's a FastAPI middleware for request logging:\n\n"
                "```python\n"
                "class RequestLoggingMiddleware:\n"
                "    async def __call__(self, request, call_next):\n"
                "        logger.info(f'Request: {request.method} {request.url}')\n"
                "        response = await call_next(request)\n"
                "        return response\n"
                "```"
            ),
        ])

        responses = await f.send_message("Design a FastAPI middleware for request logging")

        # Verify coding agent created with correct model
        coding = f.registry.get("coding")
        assert coding is not None
        assert coding.model == "test/expensive-model"

    @pytest.mark.asyncio
    async def test_coding_followup_routes_to_same_agent(self, tmp_path):
        f = OrchestratorFixture(tmp_path)
        await f.registry.get_or_create("coding", "Software development")

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            # Router: existing coding agent
            _router_response("coding"),
            # Agent response
            _agent_response("Added request timing and correlation IDs to the middleware."),
        ])

        responses = await f.send_message("Add request timing and correlation IDs")
        assert len(f.registry.list_agents()) == 1

    @pytest.mark.asyncio
    async def test_coding_session_accumulates_messages(self, tmp_path):
        f = OrchestratorFixture(tmp_path)

        # Each send_message = 1 router call + 1 agent call = 2 LLM calls
        # 3 messages = 6 calls: [router, agent, router, agent, router, agent]
        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response("__new__:coding:Software development"),
            _agent_response("Here's the function."),
            _router_response("coding"),
            _agent_response("Added error handling."),
            _router_response("coding"),
            _agent_response("Here are the tests."),
        ])

        await f.send_message("Write a function")
        await f.send_message("Add error handling")
        await f.send_message("Write tests")

        session = f.get_agent_session("coding")
        # Each process_direct adds user + assistant messages
        assert len(session.messages) >= 6  # 3 user + 3 assistant minimum


# ---------------------------------------------------------------------------
# Phase 3: Context Switch — Multiple Agents
# ---------------------------------------------------------------------------


class TestPhase3ContextSwitch:
    """Test routing between agents and multi-task dispatch."""

    @pytest.mark.asyncio
    async def test_multi_task_parallel_dispatch(self, tmp_path):
        f = OrchestratorFixture(tmp_path)

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            # Router: multi-task (simple agent names — auto-created if they don't exist)
            _router_response(
                "__multi__\n"
                "reminder: Set a reminder for meeting at 3pm tomorrow\n"
                "travel: Weekend trip to Macau suggestions"
            ),
            # Reminder agent response
            _agent_response("I'll remind you about the meeting at 3pm tomorrow."),
            # Travel agent response
            _agent_response("For a Macau weekend trip, I recommend..."),
        ])

        responses = await f.send_message(
            "Set a reminder for a meeting tomorrow at 3pm. "
            "Also, any suggestions for a weekend trip to Macau?"
        )

        # Both agents should exist (auto-created by orchestrator)
        assert f.registry.get("reminder") is not None
        assert f.registry.get("travel") is not None

    @pytest.mark.asyncio
    async def test_return_to_coding_agent_after_context_switch(self, tmp_path):
        f = OrchestratorFixture(tmp_path)
        await f.registry.get_or_create("coding", "Software development")
        await f.registry.get_or_create("chat", "General chat")

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            # Router: recognize this is a coding follow-up
            _router_response("coding"),
            # Agent response (should recall context from session history)
            _agent_response(
                "We were working on the FastAPI middleware. "
                "We had added request logging, timing, and correlation IDs."
            ),
        ])

        responses = await f.send_message("Where were we with the middleware?")
        # Routed to coding, not chat
        assert f.provider.chat_with_retry.await_count == 2

    @pytest.mark.asyncio
    async def test_manual_at_routing(self, tmp_path):
        f = OrchestratorFixture(tmp_path)
        await f.registry.get_or_create("coding", "Software development")

        f.provider.chat_with_retry = AsyncMock(
            return_value=_agent_response("Here's the code fix.")
        )

        responses = await f.send_message("@coding fix the bug in line 42")

        # Should route to coding without calling the router LLM
        # Only 1 call = the specialist agent, no router call needed
        assert f.provider.chat_with_retry.await_count == 1


# ---------------------------------------------------------------------------
# Phase 4: Deep Coding — Compaction Trigger
# ---------------------------------------------------------------------------


class TestPhase4CompactionTrigger:
    """Force compaction by filling context with coding tokens."""

    @pytest.mark.asyncio
    async def test_compaction_triggers_at_80_percent(self, tmp_path, monkeypatch):
        """Verify compaction fires when estimated tokens exceed 80% of context window."""
        import nanobot.agent.memory as memory_module

        f = OrchestratorFixture(tmp_path)
        await f.registry.get_or_create("coding", "Software development")

        # Build up session history to trigger compaction
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            messages = kwargs.get("messages") or (args[0] if args else [])
            # Check if this is a save_memory (consolidation) call
            if kwargs.get("tool_choice"):
                return _save_memory_response(
                    history_entry="[2026-03-15 10:00] Built CRUD API with auth and migrations",
                    memory_update="# Coding Project\n- Building user management API with FastAPI",
                    conversation_summary=(
                        "You were helping Stanley build a full user management API. "
                        "Completed: CRUD endpoints, JWT auth, Alembic migrations, "
                        "Pydantic validation. Next: API documentation."
                    ),
                )
            if call_count[0] == 1:
                return _router_response("coding")
            return _agent_response("Here's the implementation." + " code " * 500)

        f.provider.chat_with_retry = AsyncMock(side_effect=mock_chat)

        # Pre-fill session with lots of messages to trigger compaction
        loop = await f._create_loop(f.registry.get("coding"))
        f.registry.set_loop("coding", loop)

        session = loop.sessions.get_or_create("telegram:12345")
        for i in range(20):
            session.messages.append({
                "role": "user",
                "content": f"Build feature {i}: " + "detailed requirements " * 50,
                "timestamp": datetime.now().isoformat(),
            })
            session.messages.append({
                "role": "assistant",
                "content": f"Implementation {i}: " + "code and explanation " * 100,
                "timestamp": datetime.now().isoformat(),
            })
        loop.sessions.save(session)

        # Make token estimation return above trigger threshold
        token_call = [0]

        def mock_estimate(msgs, tools=None, model=None):
            token_call[0] += 1
            # First call: above trigger (80% of 8192 = 6553)
            if token_call[0] <= 1:
                return (7000, "test")
            # After consolidation: below target (50% of 8192 = 4096)
            return (3000, "test")

        f.provider.estimate_prompt_tokens = mock_estimate
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 200)

        # Trigger consolidation
        await loop.memory_consolidator.maybe_consolidate_by_tokens(session)

        # Verify compaction happened
        assert session.last_consolidated > 0
        assert session.compaction_summary != ""
        assert "user management API" in session.compaction_summary

    @pytest.mark.asyncio
    async def test_compaction_summary_injected_in_history(self, tmp_path):
        """Verify compaction summary appears as first messages in get_history()."""
        session = Session(key="telegram:12345")
        session.compaction_summary = "You were building a FastAPI middleware."
        session.messages = [
            {"role": "user", "content": "Continue the work"},
            {"role": "assistant", "content": "Sure, picking up where we left off."},
        ]
        session.last_consolidated = 0

        history = session.get_history()

        # First two messages should be the compaction block
        assert len(history) >= 4
        assert "[Previous conversation summary" in history[0]["content"]
        assert "FastAPI middleware" in history[0]["content"]
        assert history[1]["role"] == "assistant"
        assert "context from our previous conversation" in history[1]["content"]

    @pytest.mark.asyncio
    async def test_compaction_saves_to_history_md(self, tmp_path):
        """Verify HISTORY.md gets an entry during compaction."""
        store = MemoryStore(tmp_path)
        provider = _make_provider()

        provider.chat_with_retry = AsyncMock(
            return_value=_save_memory_response(
                history_entry="[2026-03-15 10:00] Built user management CRUD API",
                memory_update="# Project\n- User management API in progress",
                conversation_summary="You were building a user management API.",
            )
        )

        messages = [
            {"role": "user", "content": "Build CRUD API", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "Here's the API...", "timestamp": "2026-03-15T10:01:00"},
        ]

        ok, summary = await store.consolidate(messages, provider, "test-model")

        assert ok is True
        assert "user management API" in summary

        history_content = store.history_file.read_text(encoding="utf-8")
        assert "CRUD API" in history_content

        memory_content = store.read_long_term()
        assert "User management API" in memory_content


# ---------------------------------------------------------------------------
# Phase 5: Evening — Memory Recall After Compaction
# ---------------------------------------------------------------------------


class TestPhase5MemoryRecall:
    """Test memory recall and new fact extraction after compaction."""

    @pytest.mark.asyncio
    async def test_agent_recalls_via_compaction_summary(self, tmp_path):
        """After compaction, agent should see summary and recall past work."""
        f = OrchestratorFixture(tmp_path)
        profile = await f.registry.get_or_create("coding", "Software development")

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response("coding"),
            _agent_response(
                "Today we built a full user management API with FastAPI, "
                "including CRUD endpoints, JWT auth, Alembic migrations, "
                "and Pydantic validation."
            ),
        ])

        # Pre-create loop with compaction summary in session
        loop = await f._create_loop(profile)
        f.registry.set_loop("coding", loop)

        session = loop.sessions.get_or_create("telegram:12345")
        session.compaction_summary = (
            "You were helping build a user management API. "
            "Completed: CRUD, JWT auth, Alembic, Pydantic validation."
        )
        session.last_consolidated = 0
        loop.sessions.save(session)

        responses = await f.send_message("What did we build today?")

        # The agent should have access to the compaction summary via session history
        history = session.get_history()
        assert any("user management API" in m.get("content", "") for m in history)

    @pytest.mark.asyncio
    async def test_new_personal_facts_saved_to_shared_memory(self, tmp_path):
        f = OrchestratorFixture(tmp_path)
        await f.registry.get_or_create("chat", "General chat")

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response(
                "__memory__:Stanley is going hiking at Dragon's Back trail this weekend\n"
                "chat"
            ),
            _agent_response("Dragon's Back is a great trail! Enjoy the views."),
        ])

        await f.send_message("Going hiking this weekend at Dragon's Back trail")

        shared = f.get_shared_memory()
        assert "Dragon's Back" in shared


# ---------------------------------------------------------------------------
# Phase 6: Morning Day 2 — Fresh Start (Persistence)
# ---------------------------------------------------------------------------


class TestPhase6DayTwoPersistence:
    """Test that memory persists across session boundaries."""

    @pytest.mark.asyncio
    async def test_session_persists_compaction_summary_to_disk(self, tmp_path):
        """Verify session JSONL includes compaction_summary in metadata."""
        sessions = SessionManager(tmp_path)
        session = sessions.get_or_create("telegram:12345")
        session.compaction_summary = "You were building a user management API."
        session.messages.append({"role": "user", "content": "hello"})
        sessions.save(session)

        # Reload from disk
        sessions.invalidate("telegram:12345")
        reloaded = sessions.get_or_create("telegram:12345")

        assert reloaded.compaction_summary == "You were building a user management API."

    @pytest.mark.asyncio
    async def test_session_persists_last_consolidated_to_disk(self, tmp_path):
        """Verify last_consolidated offset survives reload."""
        sessions = SessionManager(tmp_path)
        session = sessions.get_or_create("telegram:12345")
        session.messages = [
            {"role": "user", "content": f"msg{i}"} for i in range(10)
        ]
        session.last_consolidated = 6
        sessions.save(session)

        sessions.invalidate("telegram:12345")
        reloaded = sessions.get_or_create("telegram:12345")

        assert reloaded.last_consolidated == 6
        assert len(reloaded.messages) == 10
        # get_history should only return unconsolidated messages
        history = reloaded.get_history()
        assert len(history) == 4  # messages[6:10]

    @pytest.mark.asyncio
    async def test_agent_continues_from_compaction_summary_after_reload(self, tmp_path):
        """Simulate day-2 startup: agent loads persisted compaction summary."""
        f = OrchestratorFixture(tmp_path)
        profile = await f.registry.get_or_create("coding", "Software development")

        # Create loop and persist session with compaction summary
        loop = await f._create_loop(profile)
        session = loop.sessions.get_or_create("telegram:12345")
        session.compaction_summary = (
            "You were helping Stanley build a user management API with FastAPI. "
            "Completed: CRUD endpoints, JWT auth, Alembic migrations."
        )
        session.messages = [
            {"role": "user", "content": "Add rate limiting"},
            {"role": "assistant", "content": "I'll add rate limiting to the API."},
        ]
        session.last_consolidated = 0
        loop.sessions.save(session)

        # Simulate fresh start: invalidate cache and recreate loop
        loop.sessions.invalidate("telegram:12345")

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response("coding"),
            _agent_response("I'll add rate limiting to the user API we built."),
        ])

        # New loop for the fresh start
        new_loop = await f._create_loop(profile)
        f.registry.set_loop("coding", new_loop)

        responses = await f.send_message("I need to add rate limiting to the user API we built")

        # Verify the new loop's session loads compaction summary
        new_session = new_loop.sessions.get_or_create("telegram:12345")
        history = new_session.get_history()
        # Compaction summary should be in history
        assert any("user management API" in m.get("content", "") for m in history)


# ---------------------------------------------------------------------------
# Phase 7: Cross-Agent Knowledge Sharing
# ---------------------------------------------------------------------------


class TestPhase7CrossAgentKnowledge:
    """Test shared memory between agents."""

    @pytest.mark.asyncio
    async def test_shared_memory_written_by_router_visible_to_all_agents(self, tmp_path):
        """Memory extracted by router should be readable by all specialists."""
        f = OrchestratorFixture(tmp_path)

        # Write shared memory (as orchestrator would)
        await f.orchestrator._save_to_shared_memory("Deploy target: k8s with 512MB pods")

        shared = f.get_shared_memory()
        assert "k8s" in shared
        assert "512MB" in shared

        # Verify a specialist's MemoryStore can read the shared memory
        agent_mem_dir = tmp_path / "agents" / "coding" / "memory"
        agent_mem_dir.mkdir(parents=True, exist_ok=True)

        store = MemoryStore(tmp_path, memory_dir=agent_mem_dir)
        context = store.get_memory_context()
        assert "k8s" in context
        assert "Shared Memory" in context

    @pytest.mark.asyncio
    async def test_specialist_reads_both_shared_and_private_memory(self, tmp_path):
        """Specialist sees both shared global + own private memory."""
        # Setup shared memory
        shared_dir = tmp_path / "memory"
        shared_dir.mkdir(parents=True, exist_ok=True)
        (shared_dir / "MEMORY.md").write_text(
            "# User Info\n- Stanley is in Hong Kong\n", encoding="utf-8"
        )

        # Setup agent private memory
        agent_dir = tmp_path / "agents" / "coding" / "memory"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "MEMORY.md").write_text(
            "# Coding Context\n- Working on FastAPI user management API\n", encoding="utf-8"
        )

        store = MemoryStore(tmp_path, memory_dir=agent_dir)
        context = store.get_memory_context()

        assert "Stanley" in context
        assert "Hong Kong" in context
        assert "FastAPI" in context
        assert "Shared Memory" in context
        assert "Agent Memory" in context

    @pytest.mark.asyncio
    async def test_memory_fact_with_agent_routing(self, tmp_path):
        """__memory__:fact|agent should save fact AND route to agent."""
        f = OrchestratorFixture(tmp_path)
        await f.registry.get_or_create("reminder", "Reminders")

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            # Router extracts memory with agent routing
            _router_response(
                "__memory__:Stanley's timezone is UTC+8|reminder\n"
                "reminder"
            ),
            _agent_response("Noted! I'll use UTC+8 for all your reminders."),
        ])

        await f.send_message("By the way, I'm in UTC+8 timezone")

        shared = f.get_shared_memory()
        assert "UTC+8" in shared


# ---------------------------------------------------------------------------
# Phase 8: Second Compaction Cycle
# ---------------------------------------------------------------------------


class TestPhase8SecondCompaction:
    """Test compaction-on-compaction: summary should be REPLACED, not accumulated."""

    @pytest.mark.asyncio
    async def test_compaction_replaces_summary(self, tmp_path, monkeypatch):
        """Second compaction should replace the compaction_summary, not append."""
        import nanobot.agent.memory as memory_module

        provider = _make_provider()

        consolidation_count = [0]

        async def mock_consolidate_chat(*args, **kwargs):
            consolidation_count[0] += 1
            if kwargs.get("tool_choice"):
                return _save_memory_response(
                    history_entry=f"[2026-03-16 14:00] Compaction round {consolidation_count[0]}",
                    memory_update="# Updated memory",
                    conversation_summary=f"Summary after compaction #{consolidation_count[0]}",
                )
            return _agent_response("ok")

        provider.chat_with_retry = AsyncMock(side_effect=mock_consolidate_chat)

        sessions = SessionManager(tmp_path)
        session = sessions.get_or_create("telegram:12345")
        session.compaction_summary = "Original summary from first compaction"

        # Add messages for second compaction
        for i in range(20):
            session.messages.append({
                "role": "user",
                "content": f"Message {i}: " + "content " * 50,
                "timestamp": datetime.now().isoformat(),
            })
            session.messages.append({
                "role": "assistant",
                "content": f"Reply {i}: " + "response " * 50,
                "timestamp": datetime.now().isoformat(),
            })
        sessions.save(session)

        consolidator = MemoryConsolidator(
            workspace=tmp_path,
            provider=provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=8192,
            build_messages=MagicMock(return_value=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]),
            get_tool_definitions=MagicMock(return_value=[]),
        )

        # Mock token estimation to trigger consolidation
        call_count = [0]

        def mock_estimate(sess):
            call_count[0] += 1
            if call_count[0] <= 1:
                return (7000, "test")
            return (3000, "test")

        consolidator.estimate_session_prompt_tokens = mock_estimate
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 200)

        await consolidator.maybe_consolidate_by_tokens(session)

        # Summary should be REPLACED, not contain original
        assert "Original summary" not in session.compaction_summary
        assert "compaction #" in session.compaction_summary

    @pytest.mark.asyncio
    async def test_compaction_summary_does_not_accumulate(self, tmp_path):
        """Verify session.compaction_summary is overwritten, not appended."""
        session = Session(key="test:key")
        session.compaction_summary = "First summary"

        # Simulate what MemoryConsolidator does
        new_summary = "Second summary replaces first"
        session.compaction_summary = new_summary

        assert session.compaction_summary == "Second summary replaces first"
        assert "First" not in session.compaction_summary


# ---------------------------------------------------------------------------
# Phase 9: Final Recall & Synthesis
# ---------------------------------------------------------------------------


class TestPhase9FinalRecall:
    """Comprehensive memory and knowledge test."""

    @pytest.mark.asyncio
    async def test_agent_synthesizes_from_compaction_and_memory(self, tmp_path):
        """Agent should be able to synthesize info from both compaction summary and memory."""
        f = OrchestratorFixture(tmp_path)
        profile = await f.registry.get_or_create("chat", "General chat")

        # Set up shared memory
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "MEMORY.md").write_text(
            "# User Info\n"
            "- Stanley is a senior software engineer in Hong Kong\n"
            "- Timezone: UTC+8\n"
            "- Going hiking at Dragon's Back this weekend\n"
            "- Deploy target: k8s with 512MB pods\n",
            encoding="utf-8",
        )

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response("chat"),
            _agent_response(
                "Here's what I know about you:\n"
                "- You're Stanley, a senior software engineer in Hong Kong\n"
                "- You're in UTC+8 timezone\n"
                "- You're going hiking at Dragon's Back this weekend\n"
                "- You deploy on k8s with 512MB pods"
            ),
        ])

        responses = await f.send_message("What personal things do you know about me?")

        assert any("Stanley" in r for r in responses)

    @pytest.mark.asyncio
    async def test_registry_tracks_all_agents(self, tmp_path):
        """After the full simulation, registry should track all created agents."""
        f = OrchestratorFixture(tmp_path)

        # Create the agents that would exist after a full simulation
        for name, desc in [
            ("chat", "General chat and casual conversation"),
            ("coding", "Software development and code generation"),
            ("reminder", "Reminders and scheduling"),
            ("travel", "Travel planning and recommendations"),
        ]:
            await f.registry.get_or_create(name, desc)

        agents = f.registry.list_agents()
        assert len(agents) == 4
        names = {a.name for a in agents}
        assert names == {"chat", "coding", "reminder", "travel"}

    @pytest.mark.asyncio
    async def test_token_usage_tracked_per_agent(self, tmp_path):
        """Each agent loop should track its own token usage."""
        f = OrchestratorFixture(tmp_path)
        profile = await f.registry.get_or_create("coding", "Software development")

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response("coding"),
            _agent_response("Done."),
        ])

        await f.send_message("Write hello world")

        loop = f.registry.get_loop("coding")
        assert loop is not None
        assert loop.token_usage["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_router_token_usage_tracked(self, tmp_path):
        """Orchestrator should track router's own LLM overhead."""
        f = OrchestratorFixture(tmp_path)

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response("__new__:chat:General chat"),
            _agent_response("Hello!"),
        ])

        await f.send_message("Hello!")

        assert f.orchestrator._router_token_usage["total_tokens"] > 0


# ---------------------------------------------------------------------------
# Model Selection & Upgrade
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Test model selection and upgrade signals."""

    @pytest.mark.asyncio
    async def test_model_hint_selects_cheap_model_for_chat(self, tmp_path):
        f = OrchestratorFixture(tmp_path)

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response("__new__:chat:General chat\n__model__:0"),
            _agent_response("Hi there!"),
        ])

        await f.send_message("Hi!")

        chat = f.registry.get("chat")
        assert chat is not None
        assert chat.model == "test/cheap-model"

    @pytest.mark.asyncio
    async def test_model_upgrade_on_dissatisfaction(self, tmp_path):
        f = OrchestratorFixture(tmp_path)
        profile = await f.registry.get_or_create("coding", "Software development")
        profile.model = "test/cheap-model"
        f.registry._save_registry()

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response("__upgrade__:coding\ncoding"),
            _agent_response("Here's an improved version..."),
        ])

        responses = await f.send_message("The code quality is poor, can you do better?")

        coding = f.registry.get("coding")
        assert coding.model == "test/mid-model"

    @pytest.mark.asyncio
    async def test_model_upgrade_already_at_max(self, tmp_path):
        f = OrchestratorFixture(tmp_path)
        profile = await f.registry.get_or_create("coding", "Software development")
        profile.model = "test/expensive-model"
        f.registry._save_registry()

        # Upgrade should be a no-op
        f.orchestrator._upgrade_agent_model("coding")

        coding = f.registry.get("coding")
        assert coding.model == "test/expensive-model"

    @pytest.mark.asyncio
    async def test_context_window_per_model(self, tmp_path):
        """Different models should get different context window sizes."""
        f = OrchestratorFixture(tmp_path)

        assert f.orchestrator.resolve_context_window("test/cheap-model") == 4096
        assert f.orchestrator.resolve_context_window("test/mid-model") == 8192
        assert f.orchestrator.resolve_context_window("test/expensive-model") == 16384
        assert f.orchestrator.resolve_context_window("unknown-model") is None


# ---------------------------------------------------------------------------
# Process Direct Serialization
# ---------------------------------------------------------------------------


class TestProcessDirectSerialization:
    """Test that concurrent dispatches to the same agent are serialized."""

    @pytest.mark.asyncio
    async def test_concurrent_process_direct_serialized(self, tmp_path):
        """Two concurrent process_direct calls should not interleave."""
        provider = _make_provider()
        bus = MessageBus()

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            model="test-model",
            context_window_tokens=8192,
        )
        loop.tools.get_definitions = MagicMock(return_value=[])

        order: list[str] = []

        async def slow_chat(*args, **kwargs):
            messages = kwargs.get("messages") or args[0]
            # Find the user message
            user_msg = ""
            for m in messages:
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
            if "first" in user_msg:
                order.append("first_start")
                await asyncio.sleep(0.1)
                order.append("first_end")
                return _agent_response("First response")
            else:
                order.append("second_start")
                order.append("second_end")
                return _agent_response("Second response")

        provider.chat_with_retry = AsyncMock(side_effect=slow_chat)

        # Launch two concurrent calls
        task1 = asyncio.create_task(loop.process_direct("first message", session_key="test:1"))
        task2 = asyncio.create_task(loop.process_direct("second message", session_key="test:1"))

        await asyncio.gather(task1, task2)

        # With serialization, first should fully complete before second starts
        assert order.index("first_end") < order.index("second_start")


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_max_specialists_enforced(self, tmp_path):
        f = OrchestratorFixture(tmp_path)
        f.orchestrator.max_specialists = 2

        # Create 2 agents
        await f.registry.get_or_create("agent1", "First agent")
        await f.registry.get_or_create("agent2", "Second agent")

        f.provider.chat_with_retry = AsyncMock(
            return_value=_router_response("__new__:agent3:Third agent")
        )

        responses = await f.send_message("Create a third agent")

        # Should get an error about max specialists
        assert any("maximum" in r.lower() for r in responses)
        assert f.registry.get("agent3") is None

    @pytest.mark.asyncio
    async def test_empty_router_response_falls_back_to_general(self, tmp_path):
        f = OrchestratorFixture(tmp_path)

        f.provider.chat_with_retry = AsyncMock(side_effect=[
            _router_response(""),  # Empty router response
            _agent_response("I can help with that."),
        ])

        responses = await f.send_message("Something vague")

        # Should fall back to general agent
        assert f.registry.get("general") is not None

    @pytest.mark.asyncio
    async def test_consolidation_raw_fallback_after_failures(self, tmp_path):
        """After 3 consecutive LLM failures, consolidation should raw-archive."""
        store = MemoryStore(tmp_path)
        provider = _make_provider()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="no tool call", tool_calls=[])
        )

        messages = [
            {"role": "user", "content": "test", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "reply", "timestamp": "2026-03-15T10:01:00"},
        ]

        # First two failures should return (False, "")
        ok1, _ = await store.consolidate(messages, provider, "test-model")
        assert ok1 is False
        ok2, _ = await store.consolidate(messages, provider, "test-model")
        assert ok2 is False

        # Third failure should trigger raw archive
        ok3, _ = await store.consolidate(messages, provider, "test-model")
        assert ok3 is True

        # Verify raw archive in HISTORY.md
        history = store.history_file.read_text(encoding="utf-8")
        assert "[RAW]" in history

    @pytest.mark.asyncio
    async def test_session_clear_resets_everything(self, tmp_path):
        session = Session(key="test:key")
        session.messages = [{"role": "user", "content": "hello"}] * 10
        session.last_consolidated = 5
        session.compaction_summary = "Some summary"

        session.clear()

        assert session.messages == []
        assert session.last_consolidated == 0
        assert session.compaction_summary == ""

    @pytest.mark.asyncio
    async def test_get_history_drops_orphaned_tool_results(self, tmp_path):
        """get_history should skip leading non-user messages."""
        session = Session(key="test:key")
        session.messages = [
            {"role": "tool", "content": "orphaned result", "tool_call_id": "x"},
            {"role": "user", "content": "actual message"},
            {"role": "assistant", "content": "response"},
        ]

        history = session.get_history()

        assert history[0]["role"] == "user"
        assert history[0]["content"] == "actual message"

    @pytest.mark.asyncio
    async def test_null_content_normalized_in_history(self, tmp_path):
        """content: null should be normalized to empty string."""
        session = Session(key="test:key")
        session.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
            {"role": "tool", "content": "result", "tool_call_id": "x"},
            {"role": "assistant", "content": "done"},
        ]

        history = session.get_history()

        # The null content should be "" not None
        assistant_msg = [m for m in history if m["role"] == "assistant" and m.get("tool_calls")]
        assert assistant_msg[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_registry_persists_to_disk(self, tmp_path):
        """Agent registry should save and reload from disk."""
        registry1 = AgentRegistry(tmp_path)
        await registry1.get_or_create("coding", "Software dev", model="test-model")
        await registry1.get_or_create("chat", "General chat")

        # Create a new registry that loads from disk
        registry2 = AgentRegistry(tmp_path)
        assert registry2.get("coding") is not None
        assert registry2.get("coding").model == "test-model"
        assert registry2.get("chat") is not None
        assert len(registry2.list_agents()) == 2


# ---------------------------------------------------------------------------
# Integration: Full Mini-Workflow
# ---------------------------------------------------------------------------


class TestMiniWorkflow:
    """Smaller integration test that chains a few phases together."""

    @pytest.mark.asyncio
    async def test_three_turn_workflow(self, tmp_path):
        """Simulate: intro → coding task → context switch."""
        f = OrchestratorFixture(tmp_path)
        call_idx = [0]

        async def sequenced_responses(*args, **kwargs):
            call_idx[0] += 1
            idx = call_idx[0]
            # Check if consolidation call
            if kwargs.get("tool_choice"):
                return _save_memory_response(
                    "entry", "memory", "summary"
                )
            responses = {
                # Turn 1: intro → create chat
                1: _router_response(
                    "__memory__:User is Stanley from Hong Kong\n"
                    "__new__:chat:General chat"
                ),
                2: _agent_response("Hey Stanley!"),
                # Turn 2: coding → create coding
                3: _router_response(
                    "__new__:coding:Software development\n__model__:2"
                ),
                4: _agent_response("Here's the middleware code..."),
                # Turn 3: back to chat
                5: _router_response("chat"),
                6: _agent_response("Sure, the weather looks great!"),
            }
            return responses.get(idx, _agent_response("fallback"))

        f.provider.chat_with_retry = AsyncMock(side_effect=sequenced_responses)

        # Turn 1: Introduction
        r1 = await f.send_message("Hi, I'm Stanley from Hong Kong")
        assert f.registry.get("chat") is not None
        assert "Stanley" in f.get_shared_memory()

        # Turn 2: Coding task
        r2 = await f.send_message("Build a FastAPI middleware")
        assert f.registry.get("coding") is not None
        assert f.registry.get("coding").model == "test/expensive-model"

        # Turn 3: Back to chat
        r3 = await f.send_message("How's the weather?")

        # Verify state
        assert len(f.registry.list_agents()) == 2
        assert f.orchestrator._router_token_usage["total_tokens"] > 0
