"""Agent registry for multi-agent orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


@dataclass
class AgentProfile:
    """Profile describing a specialist agent."""

    name: str  # e.g. "coding", "scheduling"
    description: str  # LLM-readable purpose description
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # "active", "suspended", "idle"
    model: str | None = None  # override model (or use default)
    tools: list[str] | None = None  # override tool list (or use all)
    system_prompt_extra: str = ""  # agent-specific instructions
    turn_summary: str = ""  # current state summary (updated after each turn)
    message_count: int = 0  # number of messages handled

    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentProfile:
        """Deserialize from a dict."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class AgentRegistry:
    """Tracks all specialist agents and their state.

    Persists agent profiles to ``agents/registry.json`` under the workspace.
    Running ``AgentLoop`` instances are held in-memory only.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.agents_dir = ensure_dir(workspace / "agents")
        self._state_path = self.agents_dir / "registry.json"
        self._agents: dict[str, AgentProfile] = {}
        self._loops: dict[str, AgentLoop] = {}
        self._load_registry()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_registry(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            for entry in data.get("agents", []):
                profile = AgentProfile.from_dict(entry)
                self._agents[profile.name] = profile
            logger.debug("Loaded {} agent profiles from registry", len(self._agents))
        except Exception:
            logger.exception("Failed to load agent registry")

    def _save_registry(self) -> None:
        payload = {"agents": [p.to_dict() for p in self._agents.values()]}
        self._state_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Agent workspace helpers
    # ------------------------------------------------------------------

    def agent_workspace(self, name: str) -> Path:
        """Return and ensure the workspace directory for a specialist agent."""
        return ensure_dir(self.agents_dir / name)

    def agent_sessions_dir(self, name: str) -> Path:
        return ensure_dir(self.agent_workspace(name) / "sessions")

    def agent_memory_dir(self, name: str) -> Path:
        return ensure_dir(self.agent_workspace(name) / "memory")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, name: str) -> AgentProfile | None:
        return self._agents.get(name)

    def get_loop(self, name: str) -> AgentLoop | None:
        return self._loops.get(name)

    def set_loop(self, name: str, loop: AgentLoop) -> None:
        self._loops[name] = loop

    def remove_loop(self, name: str) -> None:
        self._loops.pop(name, None)

    async def get_or_create(
        self,
        name: str,
        description: str,
        *,
        model: str | None = None,
        tools: list[str] | None = None,
        system_prompt_extra: str = "",
    ) -> AgentProfile:
        """Return an existing agent profile or create a new one."""
        if name in self._agents:
            profile = self._agents[name]
            profile.touch()
            self._save_registry()
            return profile

        profile = AgentProfile(
            name=name,
            description=description,
            model=model,
            tools=tools,
            system_prompt_extra=system_prompt_extra,
        )
        self._agents[name] = profile
        # Ensure agent workspace directories exist
        self.agent_sessions_dir(name)
        self.agent_memory_dir(name)
        self._save_registry()
        logger.info("Created agent profile: {} — {}", name, description)
        return profile

    def list_agents(self) -> list[AgentProfile]:
        """Return all registered agent profiles."""
        return list(self._agents.values())

    def find_by_name(self, name: str) -> AgentProfile | None:
        """Find an agent by exact name."""
        return self._agents.get(name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def suspend(self, name: str) -> None:
        """Persist session and tear down loop (lazy — no LLM summarization)."""
        profile = self._agents.get(name)
        if not profile:
            return

        loop = self._loops.pop(name, None)
        if loop:
            loop.stop()
            await loop.close_mcp()

        profile.status = "suspended"
        profile.touch()
        self._save_registry()
        logger.info("Suspended agent: {}", name)

    async def resume(self, name: str) -> AgentProfile:
        """Mark agent as active for resumption. Caller creates the loop."""
        profile = self._agents.get(name)
        if not profile:
            raise ValueError(f"Agent '{name}' not found in registry")

        profile.status = "active"
        profile.touch()
        self._save_registry()
        logger.info("Resumed agent: {}", name)
        return profile

    async def destroy(self, name: str) -> None:
        """Remove agent profile and tear down loop if running."""
        await self.suspend(name)
        self._agents.pop(name, None)
        self._save_registry()
        logger.info("Destroyed agent: {}", name)

    def active_agent_names(self) -> list[str]:
        """Return names of agents that have running loops."""
        return list(self._loops.keys())

    def agent_summary(self) -> str:
        """Return a concise text summary of all agents for LLM context."""
        if not self._agents:
            return "No specialist agents registered."
        lines = []
        for p in self._agents.values():
            status_icon = {"active": "[active]", "suspended": "[suspended]", "idle": "[idle]"}.get(
                p.status, f"[{p.status}]"
            )
            summary = p.turn_summary or p.description
            lines.append(f"- {p.name} {status_icon}: {summary}")
        return "\n".join(lines)
