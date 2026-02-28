"""TCP-based transport for gossip protocol.

Provides:
- TcpTransport: asyncio TCP transport implementing the Transport ABC
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections import deque
from typing import Optional

import numpy as np

from aimemory.online.gossip import Transport

logger = logging.getLogger(__name__)

# Message types
MSG_PARAMS_DELTA = 0x01
MSG_RULE_HASH = 0x02
MSG_REJECT = 0x03


class TcpTransport(Transport):
    """TCP transport for gossip communication.

    Each node runs a TCP server to receive messages and connects
    as a client to send messages to peers.
    """

    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = 9400,
    ) -> None:
        self._node_id = node_id
        self._host = host
        self._port = port
        self._inbox: deque[tuple[str, np.ndarray]] = deque()
        self._rule_hash_inbox: deque[tuple[str, bytes]] = deque()
        self._peer_addresses: dict[str, tuple[str, int]] = {}
        self._server: Optional[asyncio.Server] = None
        self._running = False

    def register_peer(self, peer_id: str, host: str, port: int) -> None:
        """Register a peer's address for sending."""
        self._peer_addresses[peer_id] = (host, port)

    async def start(self) -> None:
        """Start the TCP server to receive incoming messages."""
        self._server = await asyncio.start_server(self._handle_connection, self._host, self._port)
        self._running = True
        logger.info("Transport server started on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the TCP server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def send(self, peer_id: str, data: np.ndarray) -> None:
        """Send parameter delta to a peer (schedules async send)."""
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._async_send(peer_id, MSG_PARAMS_DELTA, data.tobytes()))
        except RuntimeError:
            logger.warning("No running event loop; cannot send to %s", peer_id)

    async def async_send_delta(self, peer_id: str, data: np.ndarray) -> None:
        """Async version of send for use in async contexts."""
        await self._async_send(peer_id, MSG_PARAMS_DELTA, data.tobytes())

    async def send_rule_hash(self, peer_id: str, rule_hash: bytes) -> None:
        """Send rule hash to a peer for verification."""
        await self._async_send(peer_id, MSG_RULE_HASH, rule_hash)

    async def send_reject(self, peer_id: str, reason: bytes) -> None:
        """Send rejection notification to a peer."""
        await self._async_send(peer_id, MSG_REJECT, reason)

    def receive(self) -> list[tuple[str, np.ndarray]]:
        """Drain and return all pending parameter delta messages."""
        messages = list(self._inbox)
        self._inbox.clear()
        return messages

    def receive_rule_hashes(self) -> list[tuple[str, bytes]]:
        """Drain and return all pending rule hash messages."""
        hashes = list(self._rule_hash_inbox)
        self._rule_hash_inbox.clear()
        return hashes

    async def _async_send(self, peer_id: str, msg_type: int, payload: bytes) -> None:
        """Send a framed message to a peer."""
        if peer_id not in self._peer_addresses:
            logger.warning("Unknown peer: %s", peer_id)
            return

        host, port = self._peer_addresses[peer_id]
        sender_bytes = self._node_id.encode("utf-8")

        # Frame: [msg_type:4][sender_len:4][sender][payload_len:4][payload]
        frame = (
            struct.pack("!I", msg_type)
            + struct.pack("!I", len(sender_bytes))
            + sender_bytes
            + struct.pack("!I", len(payload))
            + payload
        )

        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(frame)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except (ConnectionRefusedError, OSError) as e:
            logger.warning("Failed to send to %s (%s:%d): %s", peer_id, host, port, e)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an incoming TCP connection."""
        try:
            # Read message type
            msg_type_bytes = await reader.readexactly(4)
            msg_type = struct.unpack("!I", msg_type_bytes)[0]

            # Read sender ID
            sender_len_bytes = await reader.readexactly(4)
            sender_len = struct.unpack("!I", sender_len_bytes)[0]
            sender_id = (await reader.readexactly(sender_len)).decode("utf-8")

            # Read payload
            payload_len_bytes = await reader.readexactly(4)
            payload_len = struct.unpack("!I", payload_len_bytes)[0]
            payload = await reader.readexactly(payload_len)

            if msg_type == MSG_PARAMS_DELTA:
                delta = np.frombuffer(payload, dtype=np.float64).copy()
                self._inbox.append((sender_id, delta))
            elif msg_type == MSG_RULE_HASH:
                self._rule_hash_inbox.append((sender_id, payload))
            elif msg_type == MSG_REJECT:
                logger.warning("Received rejection from %s: %s", sender_id, payload.decode())

        except (asyncio.IncompleteReadError, struct.error) as e:
            logger.warning("Malformed message: %s", e)
        finally:
            writer.close()
            await writer.wait_closed()
