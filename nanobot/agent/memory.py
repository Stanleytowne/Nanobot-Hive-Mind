"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import json
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                    "conversation_summary": {
                        "type": "string",
                        "description": "A concise summary of the conversation state for continuity. "
                        "Include: what tasks were being worked on, key decisions made, current "
                        "progress, what was discussed, and any pending next steps. Write in "
                        "second person ('You were working on...'). This will be shown to the "
                        "agent as context when the conversation continues.",
                    },
                },
                "required": ["history_entry", "memory_update", "conversation_summary"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path, memory_dir: Path | None = None):
        self.memory_dir = ensure_dir(memory_dir or (workspace / "memory"))
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._consecutive_failures = 0

        # For specialist agents: shared global memory is readable alongside private memory
        self._shared_memory_dir: Path | None = None
        if memory_dir and memory_dir != workspace / "memory":
            shared = workspace / "memory"
            if shared.exists():
                self._shared_memory_dir = shared

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        parts = []

        # Shared global memory (read-only for specialists)
        if self._shared_memory_dir:
            shared_file = self._shared_memory_dir / "MEMORY.md"
            if shared_file.exists():
                shared = shared_file.read_text(encoding="utf-8").strip()
                if shared:
                    parts.append(f"## Shared Memory (global)\n{shared}")

        # Agent-private memory (or global memory in single-agent mode)
        long_term = self.read_long_term()
        if long_term:
            label = "Agent Memory (private)" if self._shared_memory_dir else "Long-term Memory"
            parts.append(f"## {label}\n{long_term}")

        return "\n\n".join(parts)

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = (
                f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            )
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
    ) -> tuple[bool, str]:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md.

        Returns:
            (success, conversation_summary) — the summary is injected into the
            session as a compaction block so the agent retains conversation state.
        """
        if not messages:
            return True, ""

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {
                "role": "system",
                "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                ok = self._fail_or_raw_archive(messages)
                return ok, ""

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                ok = self._fail_or_raw_archive(messages)
                return ok, ""

            if "history_entry" not in args or "memory_update" not in args:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                ok = self._fail_or_raw_archive(messages)
                return ok, ""

            entry = args["history_entry"]
            update = args["memory_update"]

            if entry is None or update is None:
                logger.warning(
                    "Memory consolidation: save_memory payload contains null required fields"
                )
                ok = self._fail_or_raw_archive(messages)
                return ok, ""

            entry = _ensure_text(entry).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                ok = self._fail_or_raw_archive(messages)
                return ok, ""

            self.append_history(entry)
            update = _ensure_text(update)
            if update != current_memory:
                self.write_long_term(update)

            # Extract conversation summary (new field, optional for backward compat)
            summary = _ensure_text(args.get("conversation_summary", "")).strip()

            self._consecutive_failures = 0
            logger.info(
                "Memory consolidation done for {} messages (summary={}chars)",
                len(messages),
                len(summary),
            )
            return True, summary
        except Exception:
            logger.exception("Memory consolidation failed")
            ok = self._fail_or_raw_archive(messages)
            return ok, ""

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n{self._format_messages(messages)}"
        )
        logger.warning("Memory consolidation degraded: raw-archived {} messages", len(messages))


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        memory_dir: Path | None = None,
    ):
        self.store = MemoryStore(workspace, memory_dir=memory_dir)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(
        self, messages: list[dict[str, object]]
    ) -> tuple[bool, str]:
        """Archive a selected message chunk into persistent memory.

        Returns (success, conversation_summary).
        """
        return await self.store.consolidate(messages, self.provider, self.model)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = session.key.split(":", 1) if ":" in session.key else (None, None)
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_unconsolidated(self, session: Session) -> bool:
        """Archive the full unconsolidated tail for /new-style session rollover."""
        lock = self.get_lock(session.key)
        async with lock:
            snapshot = session.messages[session.last_consolidated :]
            if not snapshot:
                return True
            ok, _ = await self.consolidate_messages(snapshot)
            return ok

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within half the context window."""
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            target = self.context_window_tokens // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < self.context_window_tokens:
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated : end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                ok, summary = await self.consolidate_messages(chunk)
                if not ok:
                    return
                session.last_consolidated = end_idx
                if summary:
                    session.compaction_summary = summary
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
