"""Async message queue for decoupled channel-agent communication."""

import asyncio

from nanobot.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.

    When the orchestrator is enabled, agent-scoped queues allow messages
    to be routed directly to individual specialist agents.
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._agent_queues: dict[str, asyncio.Queue[InboundMessage]] = {}

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent.

        If the message has a ``target_agent`` (or metadata ``target_agent``),
        it is placed on that agent's private queue instead of the global one.
        """
        target = msg.target_agent or (msg.metadata or {}).get("target_agent")
        if target and target in self._agent_queues:
            await self._agent_queues[target].put(msg)
        else:
            await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    # ------------------------------------------------------------------
    # Agent-scoped queues
    # ------------------------------------------------------------------

    def register_agent_queue(self, agent_name: str) -> asyncio.Queue[InboundMessage]:
        """Create and return a dedicated inbound queue for a specialist agent."""
        if agent_name not in self._agent_queues:
            self._agent_queues[agent_name] = asyncio.Queue()
        return self._agent_queues[agent_name]

    def unregister_agent_queue(self, agent_name: str) -> None:
        """Remove the agent's dedicated queue."""
        self._agent_queues.pop(agent_name, None)

    async def consume_agent_inbound(self, agent_name: str) -> InboundMessage:
        """Consume the next inbound message for a specific agent."""
        q = self._agent_queues.get(agent_name)
        if q is None:
            raise ValueError(f"No queue registered for agent '{agent_name}'")
        return await q.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
