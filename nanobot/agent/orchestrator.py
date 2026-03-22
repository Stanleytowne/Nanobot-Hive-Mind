"""Orchestrator agent for multi-agent task routing."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.registry import AgentRegistry
from nanobot.agent.router import TaskRouter
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import Config
    from nanobot.providers.base import LLMProvider


# ---------------------------------------------------------------------------
# Orchestrator-only tools
# ---------------------------------------------------------------------------


class RouteToAgentTool(Tool):
    """Route a user task to an existing specialist agent."""

    name = "route_to_agent"
    description = (
        "Route the current user message to an existing specialist agent for processing. "
        "The agent will handle the task and reply to the user."
    )
    parameters = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Name of the specialist agent to route to.",
            },
            "task": {
                "type": "string",
                "description": "The task or message to send to the agent.",
            },
        },
        "required": ["agent_name", "task"],
    }

    def __init__(self, orchestrator: OrchestratorLoop):
        self._orchestrator = orchestrator

    async def execute(self, *, agent_name: str, task: str) -> str:
        result = await self._orchestrator.dispatch_to_agent(agent_name, task)
        return result or f"Task routed to agent '{agent_name}'."


class CreateAgentTool(Tool):
    """Create a new specialist agent."""

    name = "create_agent"
    description = (
        "Create a new specialist agent with a given name and description. "
        "Use this when no existing agent matches the user's task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Short lowercase slug for the agent (e.g. 'coding', 'research').",
            },
            "description": {
                "type": "string",
                "description": "One-sentence description of the agent's purpose.",
            },
            "task": {
                "type": "string",
                "description": "The initial task to send to the newly created agent.",
            },
        },
        "required": ["agent_name", "description", "task"],
    }

    def __init__(self, orchestrator: OrchestratorLoop):
        self._orchestrator = orchestrator

    async def execute(self, *, agent_name: str, description: str, task: str) -> str:
        await self._orchestrator.registry.get_or_create(agent_name, description)
        result = await self._orchestrator.dispatch_to_agent(agent_name, task)
        return result or f"Created agent '{agent_name}' and routed task."


class ListAgentsTool(Tool):
    """List all specialist agents and their status."""

    name = "list_agents"
    description = "List all registered specialist agents with their status and description."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, registry: AgentRegistry):
        self._registry = registry

    async def execute(self) -> str:
        agents = self._registry.list_agents()
        if not agents:
            return "No specialist agents registered."
        lines = []
        for a in agents:
            lines.append(f"- {a.name} [{a.status}]: {a.description}")
        return "\n".join(lines)


class SuspendAgentTool(Tool):
    """Manually suspend a specialist agent."""

    name = "suspend_agent"
    description = "Suspend a specialist agent, persisting its session and freeing resources."
    parameters = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Name of the agent to suspend.",
            },
        },
        "required": ["agent_name"],
    }

    def __init__(self, registry: AgentRegistry):
        self._registry = registry

    async def execute(self, *, agent_name: str) -> str:
        profile = self._registry.get(agent_name)
        if not profile:
            return f"Agent '{agent_name}' not found."
        await self._registry.suspend(agent_name)
        return f"Agent '{agent_name}' suspended."


# ---------------------------------------------------------------------------
# OrchestratorLoop
# ---------------------------------------------------------------------------


class OrchestratorLoop:
    """Lightweight agent that routes tasks to specialist agents.

    The orchestrator:
    - Classifies incoming messages (rules first, then LLM)
    - Creates or resumes specialist agents as needed
    - Dispatches tasks and relays results
    - Sends user notifications on agent creation / handoff
    """

    def __init__(
        self,
        config: Config,
        bus: MessageBus,
        registry: AgentRegistry,
        provider: LLMProvider,
        *,
        create_specialist_loop: Any = None,
    ):
        self.config = config
        self.bus = bus
        self.registry = registry
        self.provider = provider
        self.model = config.agents.defaults.model

        orch_cfg = config.agents.orchestrator
        self.router = TaskRouter(orch_cfg.routing_rules if orch_cfg else [])
        self.max_specialists = orch_cfg.max_specialists if orch_cfg else 10
        self.idle_timeout_minutes = orch_cfg.idle_timeout_minutes if orch_cfg else 60
        self.router_model = (orch_cfg.router_model if orch_cfg else "") or self.model
        # Models sorted cheapest → most expensive; router picks based on task complexity
        self.available_models = (orch_cfg.models if orch_cfg else []) or [self.model]
        # Per-model context window overrides
        self.model_context_windows = (orch_cfg.model_context_windows if orch_cfg else {}) or {}

        self._create_specialist_loop = create_specialist_loop

        # Orchestrator tools
        self.tools = ToolRegistry()
        self.tools.register(RouteToAgentTool(self))
        self.tools.register(CreateAgentTool(self))
        self.tools.register(ListAgentsTool(self.registry))
        self.tools.register(SuspendAgentTool(self.registry))

        # Orchestrator's own session (routing decisions only)
        self.sessions = SessionManager(config.workspace_path)
        self.context = ContextBuilder(config.workspace_path)

        self._running = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._processing_lock = asyncio.Lock()

        # Pending dispatch info set by _process_message for tool callbacks
        self._current_msg: InboundMessage | None = None

        # Routing history: session_key -> list of (user_message, agent_name, summary)
        # Persists across messages so the LLM classifier can see conversation context
        self._routing_history: dict[str, list[dict[str, str]]] = {}

        # Track running specialist tasks per session+agent so they can be cancelled
        # session_key -> {agent_name: asyncio.Task}
        self._running_dispatches: dict[str, dict[str, asyncio.Task]] = {}

        # Track router's own LLM token usage (classification calls)
        self._router_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the orchestrator loop, consuming messages from the bus."""
        self._running = True
        logger.info("Orchestrator loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Check for inter-agent messages routed to a specialist
            if msg.channel == "agent" and msg.metadata.get("target_agent"):
                target = msg.metadata["target_agent"]
                asyncio.create_task(self._forward_to_agent(target, msg))
                continue

            cmd = msg.content.strip().lower()
            cmd_base = cmd.split()[0] if cmd else ""
            if cmd_base in ("/stop", "/restart", "/new", "/help", "/agents", "/threads", "/model"):
                if cmd_base == "/new":
                    self._routing_history.pop(msg.session_key, None)
                task = asyncio.create_task(self._handle_passthrough(msg))
            else:
                task = asyncio.create_task(self._dispatch(msg))

            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(
                lambda t, k=msg.session_key: (
                    self._active_tasks.get(k, []).remove(t)
                    if t in self._active_tasks.get(k, [])
                    else None
                )
            )

    def stop(self) -> None:
        self._running = False
        logger.info("Orchestrator loop stopping")

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Route a user message to specialist(s).

        Routing is serialized (fast LLM call), but specialist execution runs
        concurrently so multiple tasks can be processed in parallel.
        """
        try:
            # Phase 1: Route (serialized — keeps routing history consistent)
            routes = await self._route_message(msg)
            if not routes:
                return

            # Phase 2: Handle correction — if routing detected a user correction
            correction_target = getattr(msg, "_correction_cancel_agent", None)
            if correction_target:
                running = self._running_dispatches.get(msg.session_key, {})
                prev_task = running.get(correction_target)
                if prev_task and not prev_task.done():
                    prev_task.cancel()
                    first_agent = routes[0][0]
                    logger.info(
                        "Cancelled running [{}] task — user corrected to [{}]",
                        correction_target,
                        first_agent,
                    )
                    await self._notify_user(
                        msg, f"Interrupted [{correction_target}] — redirecting to [{first_agent}]"
                    )
                    history = self._routing_history.get(msg.session_key, [])
                    if history and history[-1]["agent"] == correction_target:
                        history.pop()
                    self._rollback_agent_session(correction_target, msg.session_key)

            # Phase 3: Execute — dispatch all routes in parallel
            if len(routes) == 1:
                await self._execute_single(routes[0], msg)
            else:
                await self._execute_multi(routes, msg)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Orchestrator error processing message")
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Sorry, I encountered an error routing your request.",
                )
            )

    async def _execute_single(self, route: tuple[str, str], msg: InboundMessage) -> None:
        """Execute a single agent dispatch."""
        agent_name, task_content = route

        dispatch_task = asyncio.current_task()
        session_dispatches = self._running_dispatches.setdefault(msg.session_key, {})
        session_dispatches[agent_name] = dispatch_task

        try:
            response_content = await self.dispatch_to_agent(agent_name, task_content, msg=msg)
        except asyncio.CancelledError:
            logger.debug("Dispatch to [{}] cancelled for {}", agent_name, msg.session_key)
            return
        finally:
            running = self._running_dispatches.get(msg.session_key, {})
            if running.get(agent_name) is dispatch_task:
                running.pop(agent_name, None)

        # Auto-upgrade: if agent hit max iterations, upgrade model and retry once
        response_content = await self._maybe_auto_upgrade(
            agent_name, task_content, response_content, msg
        )

        self._routing_history.setdefault(msg.session_key, []).append(
            {
                "user_message": task_content,
                "agent": agent_name,
            }
        )

        # Fire-and-forget turn summary update
        asyncio.create_task(
            self._update_turn_summary(agent_name, task_content, response_content)
        )

        if response_content:
            labeled = f"[{agent_name}] {response_content}"
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=labeled,
                    metadata=msg.metadata or {},
                )
            )

    async def _execute_multi(self, routes: list[tuple[str, str]], msg: InboundMessage) -> None:
        """Execute multiple agent dispatches in parallel."""

        async def _run_one(agent_name: str, task_content: str) -> tuple[str, str | None]:
            try:
                response = await self.dispatch_to_agent(agent_name, task_content, msg=msg)
                response = await self._maybe_auto_upgrade(agent_name, task_content, response, msg)
                self._routing_history.setdefault(msg.session_key, []).append(
                    {
                        "user_message": task_content,
                        "agent": agent_name,
                    }
                )
                asyncio.create_task(
                    self._update_turn_summary(agent_name, task_content, response)
                )
                return agent_name, response
            except asyncio.CancelledError:
                return agent_name, None
            except Exception:
                logger.exception("Parallel dispatch to [{}] failed", agent_name)
                return agent_name, f"Agent '{agent_name}' encountered an error."

        tasks = [_run_one(name, task) for name, task in routes]
        results = await asyncio.gather(*tasks)

        for agent_name, response_content in results:
            if response_content:
                labeled = f"[{agent_name}] {response_content}"
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=labeled,
                        metadata=msg.metadata or {},
                    )
                )

    def _rollback_agent_session(self, agent_name: str, session_key: str) -> None:
        """Remove the last user+assistant turn from a specialist's session after cancellation."""
        loop = self.registry.get_loop(agent_name)
        if not loop:
            return
        session = loop.sessions.get_or_create(session_key)
        # Remove trailing messages from the cancelled turn (user msg + any partial response)
        while session.messages and session.messages[-1].get("role") != "user":
            session.messages.pop()
        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.pop()
        loop.sessions.save(session)

    def _parse_manual_routing(self, content: str) -> tuple[str | None, str, str | None]:
        """Parse @agent_name[:model] prefix for manual routing.

        Supported formats:
            @coding fix this bug         → ("coding", "fix this bug", None)
            @coding:heavy fix this bug   → ("coding", "fix this bug", "heavy")
            @coding:2 fix this bug       → ("coding", "fix this bug", "2")
            normal message               → (None, "normal message", None)
        """
        stripped = content.strip()
        if stripped.startswith("@"):
            parts = stripped[1:].split(None, 1)
            if parts:
                agent_part = parts[0].lower()
                model_hint: str | None = None
                if ":" in agent_part:
                    candidate, model_hint = agent_part.split(":", 1)
                else:
                    candidate = agent_part
                # Verify it's a known agent or a plausible slug
                if self.registry.get(candidate) or re.match(r"^[a-z][a-z0-9_-]{0,30}$", candidate):
                    task = parts[1] if len(parts) > 1 else ""
                    return candidate, task, model_hint
        return None, content, None

    async def _route_message(self, msg: InboundMessage) -> list[tuple[str, str]]:
        """Determine which agent(s) should handle this message.

        Returns a list of (agent_name, task_content) tuples.
        Serialized via _processing_lock to keep routing history consistent.
        """
        async with self._processing_lock:
            self._current_msg = msg
            try:
                return await self._classify_and_prepare(msg)
            finally:
                self._current_msg = None

    def resolve_context_window(self, model: str | None) -> int | None:
        """Return the context window override for a model, or None to use default."""
        if not model or not self.model_context_windows:
            return None
        # Try exact match first
        if model in self.model_context_windows:
            return self.model_context_windows[model]
        # Try matching by short name (after last /)
        short = model.split("/")[-1]
        for key, val in self.model_context_windows.items():
            if key.split("/")[-1] == short or key in model:
                return val
        return None

    def _resolve_model_hint(self, hint: str | None) -> str | None:
        """Resolve a model hint (index or name) to an actual model name."""
        if not hint or len(self.available_models) <= 1:
            return None
        # Try as index
        try:
            idx = int(hint)
            if 0 <= idx < len(self.available_models):
                return self.available_models[idx]
        except ValueError:
            pass
        # Try as name (exact or partial match)
        for m in self.available_models:
            if hint.lower() in m.lower():
                return m
        # "light" / "heavy" shortcuts
        if hint == "light":
            return self.available_models[0]
        if hint == "heavy":
            return self.available_models[-1]
        return None

    def _upgrade_agent_model(self, agent_name: str) -> None:
        """Upgrade an agent to the next more powerful model in the available list."""
        profile = self.registry.get(agent_name)
        if not profile:
            return

        current = profile.model or self.available_models[0]
        # Find current model's index
        try:
            idx = self.available_models.index(current)
        except ValueError:
            idx = 0

        # Move to next tier
        if idx < len(self.available_models) - 1:
            new_model = self.available_models[idx + 1]
            old_model = current.split("/")[-1]
            new_model_short = new_model.split("/")[-1]
            profile.model = new_model
            self.registry._save_registry()

            # Tear down existing loop so it's recreated with the new model.
            # Session history is preserved — get_history() sanitizes it for
            # cross-model compatibility (see Session.get_history).
            loop = self.registry.get_loop(agent_name)
            if loop:
                loop.stop()
                self.registry.remove_loop(agent_name)

            logger.info(
                "Upgraded [{}] model: {} → {}",
                agent_name,
                old_model,
                new_model_short,
            )
        else:
            logger.info("[{}] already on most powerful model", agent_name)

    async def _maybe_auto_upgrade(
        self,
        agent_name: str,
        task_content: str,
        response_content: str | None,
        msg: InboundMessage,
    ) -> str | None:
        """Auto-upgrade agent model if it hit max iterations, then retry once."""
        loop = self.registry.get_loop(agent_name)
        if not loop or loop.last_iteration_count < loop.max_iterations:
            return response_content

        profile = self.registry.get(agent_name)
        if not profile:
            return response_content

        old_model = (profile.model or self.available_models[0]).split("/")[-1]
        self._upgrade_agent_model(agent_name)
        profile = self.registry.get(agent_name)
        new_model = (profile.model or "?").split("/")[-1] if profile else "?"

        if old_model == new_model:
            # Already on most powerful model, nothing to do
            return response_content

        await self._notify_user(
            msg,
            f"[{agent_name}] hit iteration limit — upgrading: {old_model} → {new_model}",
        )
        # Re-dispatch with upgraded model
        retry_response = await self.dispatch_to_agent(agent_name, task_content, msg=msg)
        return retry_response if retry_response else response_content

    async def _save_to_shared_memory(self, fact: str) -> None:
        """Append a user fact/preference to the shared global MEMORY.md."""
        from nanobot.utils.helpers import ensure_dir

        memory_dir = ensure_dir(self.config.workspace_path / "memory")
        memory_file = memory_dir / "MEMORY.md"

        existing = ""
        if memory_file.exists():
            existing = memory_file.read_text(encoding="utf-8").strip()

        # Avoid duplicates
        if fact.strip() in existing:
            return

        updated = (
            f"{existing}\n- {fact.strip()}\n" if existing else f"# User Info\n\n- {fact.strip()}\n"
        )
        memory_file.write_text(updated, encoding="utf-8")
        logger.info("Saved to shared memory: {}", fact.strip())

    async def _classify_and_prepare(self, msg: InboundMessage) -> list[tuple[str, str]]:
        """Classify the message and ensure the target agent(s) exist.

        Returns a list of (agent_name, task_content) tuples.
        """
        content = msg.content
        session_key = msg.session_key
        history = self._routing_history.get(session_key, [])

        # 1. Check for manual @agent[:model] routing
        manual_agent, task_content, model_hint = self._parse_manual_routing(content)
        if manual_agent:
            agent_name = manual_agent
            profile = self.registry.get(agent_name)
            if not profile:
                profile = await self.registry.get_or_create(
                    agent_name, f"Specialist agent for {agent_name} tasks"
                )
                await self._notify_user(
                    msg,
                    f"Created new agent: [{agent_name}] — {profile.description}\n"
                    f"  Router: {self._router_model_display()} | Agent: {self._agent_model_display(profile)}",
                )
            # Apply manual model selection if provided
            if model_hint:
                resolved = self._resolve_model_hint(model_hint)
                if resolved and resolved != profile.model:
                    old = (profile.model or self.available_models[0]).split("/")[-1]
                    profile.model = resolved
                    self.registry._save_registry()
                    loop = self.registry.get_loop(agent_name)
                    if loop:
                        loop.stop()
                        self.registry.remove_loop(agent_name)
                    await self._notify_user(
                        msg, f"Set [{agent_name}] model: {old} → {resolved.split('/')[-1]}"
                    )
            await self._notify_user(msg, f"→ {agent_name}")
            return [(agent_name, task_content)]

        # 2. Try deterministic rule matching
        agent_name = self.router.match_rules(content)

        # 3. If no rule match, try LLM classification (with conversation history)
        if not agent_name:
            available = self.registry.list_agents()
            # Load shared memory so the router knows user context
            memory_file = self.config.workspace_path / "memory" / "MEMORY.md"
            shared_mem = ""
            if memory_file.exists():
                shared_mem = memory_file.read_text(encoding="utf-8").strip()
            result = await self.router.llm_classify(
                content,
                available,
                self.provider,
                self.router_model,
                routing_history=history,
                shared_memory=shared_mem or None,
                available_models=self.available_models if len(self.available_models) > 1 else None,
            )
            # Accumulate router token usage
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self._router_token_usage[k] += self.router.last_usage.get(k, 0)

            # Process proactively extracted memories
            known_agents = {a.name.lower() for a in available}
            for mem_line in self.router.last_memories:
                payload = mem_line[len("__memory__:") :]
                if "|" in payload:
                    fact, _ = payload.rsplit("|", 1)
                else:
                    fact = payload
                fact = fact.strip()
                # Strip any agent name that got concatenated without separator
                for aname in known_agents:
                    if fact.lower().endswith(aname) and len(fact) > len(aname):
                        fact = fact[: -len(aname)].strip()
                        break
                if fact:
                    await self._save_to_shared_memory(fact)

            # Process model upgrades (user dissatisfied with quality)
            for upgrade_agent in self.router.last_upgrades:
                profile = self.registry.get(upgrade_agent)
                old_model = (
                    (profile.model or self.available_models[0]).split("/")[-1] if profile else "?"
                )
                self._upgrade_agent_model(upgrade_agent)
                profile = self.registry.get(upgrade_agent)
                new_model = (profile.model or "?").split("/")[-1] if profile else "?"
                if old_model != new_model:
                    await self._notify_user(
                        msg, f"⬆️ Upgraded [{upgrade_agent}]: {old_model} → {new_model}"
                    )

            # Inject cross-thread references into the message content
            ref_context = self._build_ref_context(msg.session_key)
            if ref_context:
                content = f"{ref_context}\n\n{content}"

            if result and result.startswith("__multi__"):
                # Multiple tasks for different agents
                routes = []
                for line in result.split("\n")[1:]:  # skip "__multi__" line
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    agent_part, task_part = line.split(":", 1)
                    agent_part = agent_part.strip().lower()
                    task_part = task_part.strip()
                    if not agent_part or not task_part:
                        continue

                    # Handle __new__ inline: __new__:name:desc
                    if agent_part.startswith("__new__"):
                        new_parts = agent_part.split("__new__", 1)[1].lstrip(":")
                        if ":" in new_parts:
                            a_name, a_desc = new_parts.split(":", 1)
                        else:
                            a_name = new_parts or "general"
                            a_desc = f"Handles {a_name} tasks"
                        a_name = a_name.strip()
                        profile = await self.registry.get_or_create(a_name, a_desc.strip())
                        await self._notify_user(
                            msg,
                            f"Created new agent: [{a_name}] — {a_desc.strip()}\n"
                            f"  Router: {self._router_model_display()} | Agent: {self._agent_model_display(profile)}",
                        )
                        agent_part = a_name
                    else:
                        # Ensure agent exists
                        if not self.registry.get(agent_part):
                            profile = await self.registry.get_or_create(
                                agent_part, f"Specialist agent for {agent_part} tasks"
                            )
                            await self._notify_user(
                                msg,
                                f"Created new agent: [{agent_part}]\n"
                                f"  Router: {self._router_model_display()} | Agent: {self._agent_model_display(profile)}",
                            )

                    await self._notify_user(msg, f"→ {agent_part}")
                    routes.append((agent_part, task_part))

                if routes:
                    return routes
                # Fallback if parsing failed
                agent_name = "general"

            elif result and result.startswith("__memory__:"):
                # Format: __memory__:fact|agent  or  __memory__:fact
                payload = result[len("__memory__:") :]
                # Use last pipe as separator between fact and agent name
                if "|" in payload:
                    fact, route_to = payload.rsplit("|", 1)
                    route_to = route_to.strip() or None
                else:
                    fact = payload
                    route_to = None

                if fact:
                    await self._save_to_shared_memory(fact)
                    await self._notify_user(msg, f"📝 Remembered: {fact}")

                if route_to:
                    agent_name = route_to
                else:
                    # No agent action needed, route to general to acknowledge
                    agent_name = "general"
                    await self.registry.get_or_create(
                        "general", "General-purpose assistant for miscellaneous tasks"
                    )

            elif result and result.startswith("__correct__:"):
                # User is correcting a misroute: __correct__:target:cancel
                parts = result.split(":", 2)
                agent_name = parts[1] if len(parts) > 1 else None
                cancel_agent = parts[2] if len(parts) > 2 else None
                if agent_name and cancel_agent:
                    # Signal _dispatch to cancel the misrouted agent
                    msg._correction_cancel_agent = cancel_agent
                    logger.info(
                        "Correction detected: route to [{}], cancel [{}]",
                        agent_name,
                        cancel_agent,
                    )
            elif result and result.startswith("__new__:"):
                parts = result.split(":", 2)
                agent_name = parts[1] if len(parts) > 1 else "general"
                description = parts[2] if len(parts) > 2 else f"Handles {agent_name} tasks"

                # Enforce max specialists
                if len(self.registry.list_agents()) >= self.max_specialists:
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=(
                                f"Cannot create new agent '{agent_name}': "
                                f"maximum of {self.max_specialists} specialists reached."
                            ),
                        )
                    )
                    return []

                profile = await self.registry.get_or_create(agent_name, description)
                await self._notify_user(
                    msg,
                    f"Created new agent: [{agent_name}] — {description}\n"
                    f"  Router: {self._router_model_display()} | Agent: {self._agent_model_display(profile)}",
                )
            elif result:
                agent_name = result
            else:
                agent_name = "general"
                await self.registry.get_or_create(
                    "general", "General-purpose assistant for miscellaneous tasks"
                )

        # 4. Ensure agent exists
        profile = self.registry.get(agent_name)
        if not profile:
            profile = await self.registry.get_or_create(
                agent_name, f"Specialist agent for {agent_name} tasks"
            )
            await self._notify_user(
                msg,
                f"Created new agent: [{agent_name}] — {profile.description}\n"
                f"  Router: {self._router_model_display()} | Agent: {self._agent_model_display(profile)}",
            )

        await self._notify_user(msg, f"→ {agent_name}")
        return [(agent_name, content)]

    async def dispatch_to_agent(
        self,
        agent_name: str,
        task: str,
        *,
        msg: InboundMessage | None = None,
    ) -> str | None:
        """Send a task to a specialist agent and return its response."""
        msg = msg or self._current_msg
        if not msg:
            return "Error: No message context for dispatch."

        profile = self.registry.get(agent_name)
        if not profile:
            return f"Error: Agent '{agent_name}' not found."

        # Resume if suspended
        if profile.status == "suspended":
            await self.registry.resume(agent_name)

        # Resolve model: use router's hint if available, then profile override, then default
        resolved_model = self._resolve_model_hint(self.router.last_model_hint)
        if resolved_model and not profile.model:
            profile.model = resolved_model

        # Get or create the specialist loop
        loop = self.registry.get_loop(agent_name)
        if not loop and self._create_specialist_loop:
            loop = await self._create_specialist_loop(profile)
            self.registry.set_loop(agent_name, loop)

        if not loop:
            return f"Error: Could not create loop for agent '{agent_name}'."

        # Process the message directly through the specialist
        profile.touch()
        self.registry._save_registry()

        try:
            response = await loop.process_direct(
                task,
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )
            return response
        except Exception:
            logger.exception("Specialist agent '{}' failed", agent_name)
            return f"Agent '{agent_name}' encountered an error processing the task."

    # ------------------------------------------------------------------
    # Turn summaries & cross-thread references
    # ------------------------------------------------------------------

    async def _update_turn_summary(
        self, agent_name: str, user_message: str, response: str | None
    ) -> None:
        """Generate a 1-2 sentence state summary for a thread (fire-and-forget)."""
        profile = self.registry.get(agent_name)
        if not profile:
            return

        profile.message_count += 1

        truncated_response = (response or "")[:2000]
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the current state of this thread in 1-2 sentences. "
                    "Focus on what was accomplished and what's pending. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Thread: {agent_name}\n"
                    f"Previous state: {profile.turn_summary or '(new thread)'}\n"
                    f"User message: {user_message}\n"
                    f"Bot response: {truncated_response}"
                ),
            },
        ]

        try:
            resp = await self.provider.chat_with_retry(
                messages=messages, tools=[], model=self.router_model
            )
            summary = (resp.content or "").strip()
            if summary:
                profile.turn_summary = summary
            # Accumulate router token usage for summary calls
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self._router_token_usage[k] += (resp.usage or {}).get(k, 0)
        except Exception:
            logger.debug("Failed to update turn summary for [{}]", agent_name)

        self.registry._save_registry()

    def _build_ref_context(self, session_key: str) -> str:
        """Build context string from cross-thread references (__ref__ signals)."""
        refs = self.router.last_refs
        if not refs:
            return ""

        parts: list[str] = []
        for ref_thread in refs:
            loop = self.registry.get_loop(ref_thread)
            if loop:
                # Get last assistant message from the thread's session
                session = loop.sessions.get_or_create(session_key)
                last_content = ""
                for m in reversed(session.messages):
                    if m.get("role") == "assistant" and m.get("content"):
                        last_content = m["content"]
                        break
                if last_content:
                    parts.append(
                        f"[Context from thread '{ref_thread}']\n{last_content[:3000]}"
                    )
                    continue

            # Fallback: use turn_summary
            profile = self.registry.get(ref_thread)
            if profile and profile.turn_summary:
                parts.append(
                    f"[Context from thread '{ref_thread}']\n{profile.turn_summary}"
                )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # User notifications
    # ------------------------------------------------------------------

    def _agent_model_display(self, profile=None) -> str:
        """Return the short model name a specialist will use."""
        model = None
        if profile and profile.model:
            model = profile.model
        if not model:
            model = self._resolve_model_hint(self.router.last_model_hint)
        if not model:
            model = self.available_models[0] if self.available_models else self.model
        return model.split("/")[-1]

    def _router_model_display(self) -> str:
        """Return the short model name used by the router."""
        return self.router_model.split("/")[-1]

    async def _notify_user(self, msg: InboundMessage, text: str) -> None:
        """Send a lightweight notification to the user."""
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=text,
                metadata={"_notification": True, "_progress": True},
            )
        )

    # ------------------------------------------------------------------
    # Passthrough & inter-agent forwarding
    # ------------------------------------------------------------------

    async def _handle_passthrough(self, msg: InboundMessage) -> None:
        """Handle slash commands by passing them to the first active agent or echoing."""
        cmd = msg.content.strip().lower()
        if cmd == "/help":
            lines = [
                "nanobot orchestrator commands:",
                "/new — Start a new conversation",
                "/stop — Stop all running tasks",
                "/stop <thread> — Stop a specific thread",
                "/threads — Show active threads and token usage",
                "/agents — Alias for /threads",
                "/model — List available models",
                "/model <thread> <model> — Set thread model",
                "@thread:model <msg> — Route with model override",
                "/restart — Restart the bot",
                "/help — Show available commands",
            ]
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="\n".join(lines),
                )
            )
        elif cmd in ("/agents", "/threads"):
            await self._handle_threads_command(msg)
        elif cmd.startswith("/model"):
            await self._handle_model_command(msg)
        elif cmd.startswith("/stop"):
            parts = cmd.split(None, 1)
            target_agent = parts[1].strip() if len(parts) > 1 else None

            if target_agent:
                # Stop a specific agent
                running = self._running_dispatches.get(msg.session_key, {})
                task = running.get(target_agent)
                if task and not task.done():
                    task.cancel()
                    running.pop(target_agent, None)
                    content = f"Stopped [{target_agent}] agent."
                else:
                    content = f"Agent [{target_agent}] is not running."
            else:
                # Stop all
                all_tasks = self._active_tasks.pop(msg.session_key, [])
                running = self._running_dispatches.pop(msg.session_key, {})
                cancelled = sum(1 for t in all_tasks if not t.done() and t.cancel())
                cancelled += sum(1 for t in running.values() if not t.done() and t.cancel())
                content = f"Stopped {cancelled} task(s)." if cancelled else "No active tasks."

            await self.bus.publish_outbound(
                OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            )

    async def _handle_threads_command(self, msg: InboundMessage) -> None:
        """Show active threads and token usage."""
        agents = self.registry.list_agents()
        if not agents:
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="No threads registered.",
                )
            )
            return

        total_prompt = 0
        total_completion = 0
        total_all = 0
        lines = ["📊 **Active Threads**\n"]

        for a in agents:
            loop = self.registry.get_loop(a.name)
            if loop:
                u = loop.token_usage
                prompt = u["prompt_tokens"]
                completion = u["completion_tokens"]
                total = u["total_tokens"]
                total_prompt += prompt
                total_completion += completion
                total_all += total
                status = "🟢"
                token_info = f"  Tokens: {total:,} (in: {prompt:,} / out: {completion:,})"
            else:
                status = "💤"
                token_info = "  Tokens: —"

            summary = a.turn_summary or a.description
            model = (a.model or self.available_models[0]).split("/")[-1] if a.model or self.available_models else "?"
            lines.append(f"{status} **{a.name}** ({model}) — {summary}")
            lines.append(f"  Messages: {a.message_count} | Last active: {a.last_active[:16]}")
            lines.append(token_info)

        lines.append(
            f"\n**Total: {total_all:,} tokens** (in: {total_prompt:,} / out: {total_completion:,})"
        )

        # Also count orchestrator's own routing LLM calls
        router_usage = getattr(self, "_router_token_usage", None)
        if router_usage and router_usage["total_tokens"] > 0:
            lines.append(f"Router overhead: {router_usage['total_tokens']:,} tokens")

        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="\n".join(lines),
            )
        )

    async def _handle_model_command(self, msg: InboundMessage) -> None:
        """Handle /model [agent] [model_hint] — list or set agent model."""
        parts = msg.content.strip().split(None, 2)

        if len(parts) == 1:
            # /model — list available models and current assignments
            lines = ["**Available models:**"]
            for i, m in enumerate(self.available_models):
                lines.append(f"  {i}: {m.split('/')[-1]}  (`{m}`)")
            agents = self.registry.list_agents()
            if agents:
                lines.append("\n**Agent models:**")
                for a in agents:
                    model = a.model or self.available_models[0]
                    lines.append(f"  [{a.name}] → {model.split('/')[-1]}")
            lines.append("\nUsage: `/model <agent> <model_name_or_index>`")
            lines.append("Inline: `@agent:model_hint message`")
            await self.bus.publish_outbound(
                OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines))
            )
            return

        if len(parts) < 3:
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: `/model <agent> <model_name_or_index>`",
                )
            )
            return

        agent_name = parts[1].lower()
        model_hint = parts[2]
        resolved = self._resolve_model_hint(model_hint)
        if not resolved:
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Unknown model: `{model_hint}`",
                )
            )
            return

        profile = self.registry.get(agent_name)
        if not profile:
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Agent `{agent_name}` not found.",
                )
            )
            return

        old_model = (profile.model or self.available_models[0]).split("/")[-1]
        profile.model = resolved
        self.registry._save_registry()
        # Tear down loop so it recreates with the new model
        loop = self.registry.get_loop(agent_name)
        if loop:
            loop.stop()
            self.registry.remove_loop(agent_name)

        new_short = resolved.split("/")[-1]
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Set [{agent_name}] model: {old_model} → {new_short}",
            )
        )

    async def _forward_to_agent(self, agent_name: str, msg: InboundMessage) -> None:
        """Forward an inter-agent message to a specialist's bus."""
        loop = self.registry.get_loop(agent_name)
        if loop:
            try:
                response = await loop.process_direct(
                    msg.content,
                    session_key=f"agent:{msg.metadata.get('source_agent', 'unknown')}",
                    channel="agent",
                    chat_id=msg.chat_id,
                )
                # If there is a callback_id, store the response
                callback_id = msg.metadata.get("callback_id")
                if callback_id and hasattr(self, "_callback_store"):
                    self._callback_store[callback_id] = response
            except Exception:
                logger.exception("Failed to forward to agent '{}'", agent_name)
