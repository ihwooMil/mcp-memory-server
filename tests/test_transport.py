"""Tests for TcpTransport: asyncio TCP transport for gossip protocol."""

from __future__ import annotations

import asyncio

import numpy as np

from aimemory.online.transport import TcpTransport


def _get_free_port() -> int:
    """Get a free localhost port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestTcpTransport:
    def test_send_receive_loopback(self):
        """Start server, send delta to self (loopback), verify received data matches."""

        async def _run():
            port = _get_free_port()
            transport = TcpTransport("node_a", host="127.0.0.1", port=port)
            transport.register_peer("node_a", "127.0.0.1", port)
            await transport.start()
            try:
                delta = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
                await transport._async_send("node_a", 0x01, delta.tobytes())
                await asyncio.sleep(0.05)
                messages = transport.receive()
                assert len(messages) == 1
                sender_id, received = messages[0]
                assert sender_id == "node_a"
                np.testing.assert_array_equal(received, delta)
            finally:
                await transport.stop()

        asyncio.run(_run())

    def test_two_nodes_exchange(self):
        """Two TcpTransport instances exchange deltas over localhost."""

        async def _run():
            port_a = _get_free_port()
            port_b = _get_free_port()

            transport_a = TcpTransport("node_a", host="127.0.0.1", port=port_a)
            transport_b = TcpTransport("node_b", host="127.0.0.1", port=port_b)

            transport_a.register_peer("node_b", "127.0.0.1", port_b)
            transport_b.register_peer("node_a", "127.0.0.1", port_a)

            await transport_a.start()
            await transport_b.start()
            try:
                delta_a = np.array([1.1, 2.2, 3.3], dtype=np.float64)
                delta_b = np.array([4.4, 5.5, 6.6], dtype=np.float64)

                await transport_a._async_send("node_b", 0x01, delta_a.tobytes())
                await transport_b._async_send("node_a", 0x01, delta_b.tobytes())
                await asyncio.sleep(0.1)

                msgs_b = transport_b.receive()
                msgs_a = transport_a.receive()

                assert len(msgs_b) == 1
                sender_id, received = msgs_b[0]
                assert sender_id == "node_a"
                np.testing.assert_array_equal(received, delta_a)

                assert len(msgs_a) == 1
                sender_id, received = msgs_a[0]
                assert sender_id == "node_b"
                np.testing.assert_array_equal(received, delta_b)
            finally:
                await transport_a.stop()
                await transport_b.stop()

        asyncio.run(_run())

    def test_connection_refused_graceful(self):
        """Sending to unreachable peer logs warning, does not crash."""

        async def _run():
            transport = TcpTransport("node_x", host="127.0.0.1", port=_get_free_port())
            transport.register_peer("unreachable", "127.0.0.1", _get_free_port())
            # Should not raise even when connection is refused
            await transport._async_send("unreachable", 0x01, b"some data")

        asyncio.run(_run())
