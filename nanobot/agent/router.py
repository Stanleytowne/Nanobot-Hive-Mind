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
        self.last_refs: list[str] = []  # __ref__ thread names from last classification
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
            "\n".join(
                f"- {a.name}: {a.turn_summary or a.description}"
                for a in available_agents
            )
            if available_agents
            else "(no active threads)"
        )

        # Build conversation history section for context
        history_section = ""
        if routing_history:
            lines = []
            for entry in routing_history[-30:]:  # last 30 entries to stay within limits
                lines.append(
                    f"  User: {entry['user_message']}\n"
                    f"  → Handled by: [{entry['agent']}]"
                )
            history_section = (
                "\n\nConversation history (previous messages and which agent handled them):\n"
                + "\n\n".join(lines)
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task router. Given a user message, decide which thread "
                    "should handle it. Each thread is an isolated conversation about a "
                    "specific task or topic.\n\n"
                    "## Response formats\n\n"
                    "Basic routing (just the thread name):\n"
                    "  thread_name\n\n"
                    "New thread needed:\n"
                    "  __new__:task-slug:short description\n\n"
                    "User correcting a misroute:\n"
                    "  __correct__:target_thread:cancel_thread\n\n"
                    "Multiple tasks for different threads (executed IN PARALLEL — no "
                    "dependencies allowed between tasks):\n"
                    "  __multi__\n"
                    "  thread_name_1: task description 1\n"
                    "  thread_name_2: task description 2\n\n"
                    "Cross-thread reference (include context from another thread):\n"
                    "  __ref__:thread_name\n"
                    "  target_thread\n\n"
                    "## Thread naming\n\n"
                    "Thread names must be **specific task slugs** describing the actual work, "
                    "NOT generic categories. Examples:\n"
                    "  Good: fix-auth-timeout, vacation-planning-june, crawler-bug-403\n"
                    "  Bad: coding, research, chat, general\n\n"
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
                    "  thread_name\n"
                    "Write facts in the SAME language the user used. Do not translate.\n\n"
                    "If a memory fact requires a thread to ACT on it, append |thread_name:\n"
                    "  __memory__:user timezone is UTC+8|daily-standup\n"
                    "  daily-standup\n\n"
                    "## Model selection\n\n"
                    "After the routing decision, optionally specify which model to use:\n"
                    "  __model__:model_name_or_index\n"
                    "Use the index (0 = cheapest, last = most expensive) from the available "
                    "models list, or the model name. Pick cheaper models for simple tasks "
                    "(chat, Q&A, greetings) and expensive ones for complex tasks (coding, "
                    "analysis, research). If omitted, uses the most capable model.\n\n"
                    "## Model upgrade on dissatisfaction\n\n"
                    "If the user shows signs that the current thread quality is insufficient, "
                    "reply with __upgrade__:thread_name BEFORE the routing decision.\n\n"
                    "Upgrade triggers:\n"
                    "- Explicit dissatisfaction: 'not good enough', 'quality is poor', "
                    "'写得不好', '太简单了', 'can you do better', '重新写'\n"
                    "- Repeated debugging: user has been going back and forth fixing the "
                    "same issue (e.g. 'still broken', 'that didn't work either', "
                    "'same error again', 'try again')\n"
                    "- Escalation language: 'use a better model', 'give me something "
                    "more advanced', 'this needs a smarter approach'\n\n"
                    "Format:\n"
                    "  __upgrade__:thread_name\n"
                    "  thread_name\n"
                    "This upgrades the thread to the next more powerful model and retries. "
                    "The routing decision should still point to the same thread.\n\n"
                    "## Rules\n\n"
                    "- Thread names: short lowercase slugs describing the specific task "
                    "(e.g. 'fix-auth-timeout', 'plan-vacation-june', 'debug-404-api')\n"
                    "- ALWAYS create a new thread for each distinct task or topic. Do NOT "
                    "reuse threads for unrelated work.\n"
                    "- Each thread's state summary tells you what it's currently working on "
                    "— use it to decide routing\n"
                    "- Follow-ups go to the SAME thread that handled the earlier related task\n"
                    "- Feedback or comments about a thread's response (e.g. 'too long', "
                    "'reformat this', 'send as PDF') are follow-ups — route to that thread\n"
                    "- Only use __correct__ for explicit misroute corrections, not topic switches\n"
                    "- __multi__ tasks run in parallel, so they MUST NOT depend on each other's "
                    "results. If task B needs output from task A, route both to a single thread\n"
                    "- Use __ref__ when the user's message references results from another thread "
                    "(e.g. 'use the data from the crawler' while talking to a different thread)\n"
                    "- Do not explain your reasoning, just output the decision"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Active threads:\n{agent_list}"
                    + (
                        f"\n\nAvailable models (0=cheapest, {len(available_models) - 1}=most capable):\n"
                        + "\n".join(f"  {i}: {m}" for i, m in enumerate(available_models))
                        if available_models and len(available_models) > 1
                        else ""
                    )
                    + f"{history_section}"
                    + (
                        f"\n\nUser profile/preferences (shared memory):\n{shared_memory}"
                        if shared_memory
                        else ""
                    )
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

            # Parse multi-line response: extract signals and routing
            lines = raw.split("\n")
            memory_lines: list[str] = []
            upgrade_lines: list[str] = []
            ref_lines: list[str] = []
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
                elif stripped.startswith("__ref__:"):
                    ref_lines.append(stripped.split(":", 1)[1].strip())
                else:
                    routing_lines.append(stripped)

            # Store extracted metadata for the orchestrator to use
            self.last_memories = memory_lines
            self.last_model_hint = model_hint
            self.last_upgrades = [u.split(":", 1)[1].strip() for u in upgrade_lines if ":" in u]
            self.last_refs = ref_lines

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
