"""Inter-agent communication tools."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import InboundMessage

if TYPE_CHECKING:
    from nanobot.agent.registry import AgentRegistry
    from nanobot.bus.queue import MessageBus


class SendToAgentTool(Tool):
    """Send a message to another specialist agent (async, fire-and-forget with callback)."""

    name = "send_to_agent"
    description = (
        "Send a message or request to another specialist agent. "
        "The message is delivered asynchronously. Use check_agent_callback to check for a response."
    )
    parameters = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Name of the target specialist agent.",
            },
            "message": {
                "type": "string",
                "description": "The message or request to send.",
            },
            "callback_id": {
                "type": "string",
                "description": "Optional unique ID for tracking the response.",
            },
        },
        "required": ["agent_name", "message"],
    }

    def __init__(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        current_agent_name: str,
        callback_store: dict[str, str],
    ):
        self._registry = registry
        self._bus = bus
        self._current_agent_name = current_agent_name
        self._callback_store = callback_store

    async def execute(
        self,
        *,
        agent_name: str,
        message: str,
        callback_id: str | None = None,
    ) -> str:
        profile = self._registry.get(agent_name)
        if not profile:
            return f"Error: Agent '{agent_name}' not found."

        callback_id = callback_id or f"{self._current_agent_name}:{uuid4().hex[:8]}"

        msg = InboundMessage(
            channel="agent",
            sender_id=self._current_agent_name,
            chat_id=f"agent:{agent_name}",
            content=message,
            metadata={
                "source_agent": self._current_agent_name,
                "target_agent": agent_name,
                "callback_id": callback_id,
            },
        )
        await self._bus.publish_inbound(msg)
        return f"Message sent to {agent_name}. Callback ID: {callback_id}"


class AgentCallbackTool(Tool):
    """Check for async responses from other agents."""

    name = "check_agent_callback"
    description = (
        "Check if a response has been received from another agent for a given callback ID."
    )
    parameters = {
        "type": "object",
        "properties": {
            "callback_id": {
                "type": "string",
                "description": "The callback ID returned by send_to_agent.",
            },
        },
        "required": ["callback_id"],
    }

    def __init__(self, callback_store: dict[str, str]):
        self._callback_store = callback_store

    async def execute(self, *, callback_id: str) -> str:
        result = self._callback_store.get(callback_id)
        if result is None:
            return "No response yet."
        return result
