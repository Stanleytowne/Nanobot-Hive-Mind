"""Task router for multi-agent orchestration.

Hybrid routing: deterministic regex rules first, then LLM classification.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.registry import AgentProfile
    from nanobot.config.schema import RoutingRule
    from nanobot.providers.base import LLMProvider


class TaskRouter:
    """Routes incoming messages to the appropriate specialist agent."""

    def __init__(self, rules: list[RoutingRule] | None = None):
        self._rules = rules or []
        self.last_usage: dict[str, int] = {}
        self.last_memories: list[str] = []  # __memory__ lines from last classification
        self.last_model_hint: str | None = None  # __model__ hint from last classification
        self.last_upgrades: list[str] = []  # agent names to upgrade model
        # Pre-compile regex patterns
        self._compiled: list[tuple[re.Pattern, str]] = []
        for rule in self._rules:
            try:
                self._compiled.append((re.compile(rule.pattern, re.IGNORECASE), rule.agent_name))
            except re.error:
                logger.warning("Invalid routing rule pattern: {}", rule.pattern)

    def match_rules(self, content: str) -> str | None:
        """Check deterministic regex/keyword rules. Returns agent name or None."""
        for pattern, agent_name in self._compiled:
            if pattern.search(content):
                logger.debug("Rule match: '{}' -> agent '{}'", pattern.pattern, agent_name)
                return agent_name
        return None

    async def llm_classify(
        self,
        content: str,
        available_agents: list[AgentProfile],
        provider: LLMProvider,
        model: str,
        *,
        routing_history: list[dict[str, str]] | None = None,
        shared_memory: str | None = None,
        available_models: list[str] | None = None,
    ) -> str | None:
        """Ask LLM to classify which agent should handle the message.

        Returns:
            - An existing agent name, or
            - ``"__new__:category_name:description"`` if a new agent should be created, or
            - ``None`` if classification fails.
        """
        agent_list = (
            "\n".join(f"- {a.name}: {a.description}" for a in available_agents)
            if available_agents
            else "(no existing agents)"
        )

        # Build conversation history section for context
        history_section = ""
        if routing_history:
            lines = []
            for entry in routing_history[-30:]:  # last 30 entries to stay within limits
                preview = entry.get("response_preview", "")
                preview_text = f" → {preview}..." if preview else ""
                lines.append(
                    f"  User: {entry['user_message']}\n"
                    f"  → Handled by: [{entry['agent']}]{preview_text}"
                )
            history_section = (
                "\n\nConversation history (previous messages and which agent handled them):\n"
                + "\n\n".join(lines)
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task router. Given a user message, decide which specialist agent "
                    "should handle it.\n\n"
                    "## Response formats\n\n"
                    "Basic routing (just the agent name):\n"
                    "  agent_name\n\n"
                    "New agent needed:\n"
                    "  __new__:category_name:short description\n\n"
                    "User correcting a misroute:\n"
                    "  __correct__:target_agent:cancel_agent\n\n"
                    "Multiple tasks for different agents:\n"
                    "  __multi__\n"
                    "  agent_name_1: task description 1\n"
                    "  agent_name_2: task description 2\n\n"
                    "## Memory extraction\n\n"
                    "IMPORTANT: Be proactive about detecting memorable information. If the user's "
                    "message contains ANY personal facts, preferences, or context — even if "
                    "mentioned casually — prepend a __memory__ line BEFORE your routing decision.\n\n"
                    "Examples of things to remember:\n"
                    "- Personal info: name, location, timezone, job, age\n"
                    "- Preferences: dietary, language, communication style\n"
                    "- Context: travel plans, health conditions, ongoing projects\n"
                    "- Relationships: mentions of family, colleagues, pets\n\n"
                    "Format (one or more __memory__ lines, then the routing decision):\n"
                    "  __memory__:fact 1\n"
                    "  __memory__:fact 2\n"
                    "  agent_name\n"
                    "Write facts in the SAME language the user used. Do not translate.\n\n"
                    "If a memory fact requires an agent to ACT on it, append |agent_name:\n"
                    "  __memory__:user timezone is UTC+8|reminder\n"
                    "  reminder\n\n"
                    "## Model selection\n\n"
                    "After the routing decision, optionally specify which model to use:\n"
                    "  __model__:model_name_or_index\n"
                    "Use the index (0 = cheapest, last = most expensive) from the available "
                    "models list, or the model name. Pick cheaper models for simple tasks "
                    "(chat, Q&A, greetings) and expensive ones for complex tasks (coding, "
                    "analysis, research). If omitted, uses the most capable model.\n\n"
                    "## Model upgrade on dissatisfaction\n\n"
                    "If the user expresses dissatisfaction with quality (e.g. '写得不好', "
                    "'not good enough', '太简单了', 'can you do better', '重新写', "
                    "'quality is poor'), reply with:\n"
                    "  __upgrade__:agent_name\n"
                    "  agent_name\n"
                    "This upgrades the agent to the next more powerful model and retries. "
                    "The routing decision should still point to the same agent.\n\n"
                    "## Rules\n\n"
                    "- category_name: short lowercase slug (e.g. 'coding', 'research', 'reminder')\n"
                    "- ALWAYS create specialized agents for distinct task types. Do NOT route "
                    "coding tasks to a 'chat' or 'general' agent — create a 'coding' agent. "
                    "Do NOT route reminders to 'chat' — create a 'reminder' agent. Each task "
                    "domain should have its own specialist.\n"
                    "- Follow-ups go to the SAME agent that handled the earlier related task\n"
                    "- Only use __correct__ for explicit misroute corrections, not topic switches\n"
                    "- Only use __multi__ for clearly independent tasks for different agents\n"
                    "- Do not explain your reasoning, just output the decision"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Available agents:\n{agent_list}"
                    + (f"\n\nAvailable models (0=cheapest, {len(available_models)-1}=most capable):\n"
                       + "\n".join(f"  {i}: {m}" for i, m in enumerate(available_models))
                       if available_models and len(available_models) > 1 else "")
                    + f"{history_section}"
                    + (f"\n\nUser profile/preferences (shared memory):\n{shared_memory}"
                       if shared_memory else "")
                    + f"\n\nNew user message:\n{content}"
                ),
            },
        ]

        try:
            response = await provider.chat_with_retry(messages=messages, tools=[], model=model)
            self.last_usage = response.usage or {}
            raw = (response.content or "").strip()
            if not raw:
                return None

            # Parse multi-line response: extract __memory__ lines, __tier__, and routing
            lines = raw.split("\n")
            memory_lines: list[str] = []
            upgrade_lines: list[str] = []
            model_hint: str | None = None
            routing_lines: list[str] = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("__memory__:"):
                    memory_lines.append(stripped)
                elif stripped.startswith("__model__:"):
                    model_hint = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("__tier__:"):
                    model_hint = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("__upgrade__:"):
                    upgrade_lines.append(stripped)
                else:
                    routing_lines.append(stripped)

            # Store extracted metadata for the orchestrator to use
            self.last_memories = memory_lines
            self.last_model_hint = model_hint
            self.last_upgrades = [u.split(":", 1)[1].strip() for u in upgrade_lines if ":" in u]

            # Reconstruct the routing decision (non-memory, non-tier lines)
            result = "\n".join(routing_lines).strip()
            if not result:
                # Only memory lines, no routing — route to general
                return "general" if not memory_lines else None

            # Check for __multi__ pattern
            if result.startswith("__multi__"):
                return result

            # Check for __correct__ pattern
            if result.startswith("__correct__:"):
                return result

            # Check if it matches an existing agent
            for agent in available_agents:
                if result.lower() == agent.name.lower():
                    return agent.name

            # Check for __new__ pattern
            if result.startswith("__new__:"):
                return result

            # Fuzzy match: LLM might return something close
            for agent in available_agents:
                if agent.name.lower() in result.lower():
                    return agent.name

            # Treat as new agent creation if we can parse it
            if ":" in result:
                return f"__new__:{result}"

            return f"__new__:{result}:General tasks related to {result}"

        except Exception:
            logger.exception("LLM classification failed")
            return None
