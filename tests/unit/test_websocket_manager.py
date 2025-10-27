"""
Test suite for app/websockets/manager.py

Coverage targets:
- Connection lifecycle management
- Connection acceptance and registration
- Connection disconnection and cleanup
- Library subscriptions
- Message sending to specific connections
- Broadcasting to library subscribers
- Event broadcasting
- Connection statistics
- Error handling (send failures, unknown connections)
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.websockets.manager import ConnectionManager, get_connection_manager


class TestConnectionManager:
    """Test ConnectionManager initialization."""

    def test_connection_manager_creation(self):
        """Test creating a ConnectionManager instance."""
        manager = ConnectionManager()

        assert len(manager._connections) == 0
        assert len(manager._library_subscriptions) == 0
        assert len(manager._connection_metadata) == 0


class TestWebSocketConnection:
    """Test WebSocket connection management."""

    @pytest.fixture
    def manager(self):
        """Provide fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        ws = Mock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect_websocket(self, manager, mock_websocket):
        """Test connecting a WebSocket."""
        library_id = uuid4()

        connection_id = await manager.connect(mock_websocket, library_id)

        assert connection_id is not None
        assert connection_id in manager._connections
        assert manager._connections[connection_id] == mock_websocket
        assert library_id in manager._library_subscriptions
        assert connection_id in manager._library_subscriptions[library_id]

        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_multiple_to_same_library(self, manager):
        """Test multiple connections to the same library."""
        library_id = uuid4()

        ws1 = Mock()
        ws1.accept = AsyncMock()
        ws2 = Mock()
        ws2.accept = AsyncMock()

        conn_id1 = await manager.connect(ws1, library_id)
        conn_id2 = await manager.connect(ws2, library_id)

        assert conn_id1 != conn_id2
        assert len(manager._library_subscriptions[library_id]) == 2
        assert conn_id1 in manager._library_subscriptions[library_id]
        assert conn_id2 in manager._library_subscriptions[library_id]

    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, manager, mock_websocket):
        """Test disconnecting a WebSocket."""
        library_id = uuid4()

        connection_id = await manager.connect(mock_websocket, library_id)
        assert connection_id in manager._connections

        await manager.disconnect(connection_id)

        assert connection_id not in manager._connections
        assert connection_id not in manager._connection_metadata
        # Library should have no more subscriptions
        if library_id in manager._library_subscriptions:
            assert connection_id not in manager._library_subscriptions[library_id]

    @pytest.mark.asyncio
    async def test_disconnect_removes_empty_library_subscription(self, manager):
        """Test that library subscription is removed when last connection disconnects."""
        library_id = uuid4()

        ws = Mock()
        ws.accept = AsyncMock()

        connection_id = await manager.connect(ws, library_id)
        assert library_id in manager._library_subscriptions

        await manager.disconnect(connection_id)

        # Library subscription should be completely removed
        assert library_id not in manager._library_subscriptions

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_connection(self, manager):
        """Test disconnecting a nonexistent connection (should not error)."""
        # Should not raise exception
        await manager.disconnect("nonexistent-id")

    @pytest.mark.asyncio
    async def test_disconnect_one_keeps_others(self, manager):
        """Test disconnecting one connection keeps others active."""
        library_id = uuid4()

        ws1 = Mock()
        ws1.accept = AsyncMock()
        ws2 = Mock()
        ws2.accept = AsyncMock()

        conn_id1 = await manager.connect(ws1, library_id)
        conn_id2 = await manager.connect(ws2, library_id)

        await manager.disconnect(conn_id1)

        # conn_id2 should still be active
        assert conn_id2 in manager._connections
        assert conn_id1 not in manager._connections
        assert library_id in manager._library_subscriptions
        assert conn_id2 in manager._library_subscriptions[library_id]


class TestMessageSending:
    """Test message sending functionality."""

    @pytest.fixture
    def manager(self):
        """Provide fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_send_message_success(self, manager):
        """Test sending a message to a connection."""
        library_id = uuid4()
        ws = Mock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()

        connection_id = await manager.connect(ws, library_id)

        message = {"type": "test", "data": "hello"}
        await manager.send_message(connection_id, message)

        ws.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_message_to_unknown_connection(self, manager):
        """Test sending to unknown connection (should not error)."""
        message = {"type": "test"}

        # Should not raise exception, just log warning
        await manager.send_message("unknown-id", message)

    @pytest.mark.asyncio
    async def test_send_message_failure_disconnects(self, manager):
        """Test that send failure triggers disconnect."""
        library_id = uuid4()
        ws = Mock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock(side_effect=Exception("Connection lost"))

        connection_id = await manager.connect(ws, library_id)

        message = {"type": "test"}
        await manager.send_message(connection_id, message)

        # Connection should be removed after failure
        assert connection_id not in manager._connections


class TestBroadcasting:
    """Test message broadcasting functionality."""

    @pytest.fixture
    def manager(self):
        """Provide fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_broadcast_to_library(self, manager):
        """Test broadcasting to all connections in a library."""
        library_id = uuid4()

        ws1 = Mock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        ws2 = Mock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1, library_id)
        await manager.connect(ws2, library_id)

        message = {"type": "broadcast", "data": "test"}
        await manager.broadcast_to_library(library_id, message)

        # Both connections should receive the message
        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_library_without_subscribers(self, manager):
        """Test broadcasting to library with no subscribers."""
        library_id = uuid4()

        message = {"type": "broadcast"}

        # Should not raise exception
        await manager.broadcast_to_library(library_id, message)

    @pytest.mark.asyncio
    async def test_broadcast_to_correct_library_only(self, manager):
        """Test that broadcast only goes to correct library."""
        lib_id1 = uuid4()
        lib_id2 = uuid4()

        ws1 = Mock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        ws2 = Mock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1, lib_id1)
        await manager.connect(ws2, lib_id2)

        message = {"type": "broadcast", "data": "test"}
        await manager.broadcast_to_library(lib_id1, message)

        # Only ws1 should receive the message
        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_event(self, manager):
        """Test broadcasting an event notification."""
        library_id = uuid4()

        ws = Mock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()

        await manager.connect(ws, library_id)

        event_type = "document.added"
        data = {"document_id": "doc123"}

        await manager.broadcast_event(event_type, library_id, data)

        # Verify message format
        ws.send_json.assert_called_once()
        sent_message = ws.send_json.call_args[0][0]

        assert sent_message["type"] == "event"
        assert sent_message["event_type"] == event_type
        assert sent_message["library_id"] == str(library_id)
        assert sent_message["data"] == data


class TestConnectionStatistics:
    """Test connection statistics."""

    @pytest.fixture
    def manager(self):
        """Provide fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_stats_empty(self, manager):
        """Test statistics with no connections."""
        stats = manager.get_stats()

        assert stats["total_connections"] == 0
        assert stats["libraries_with_subscribers"] == 0
        assert stats["subscriptions_by_library"] == {}

    @pytest.mark.asyncio
    async def test_stats_with_connections(self, manager):
        """Test statistics with active connections."""
        lib_id1 = uuid4()
        lib_id2 = uuid4()

        ws1 = Mock()
        ws1.accept = AsyncMock()
        ws2 = Mock()
        ws2.accept = AsyncMock()
        ws3 = Mock()
        ws3.accept = AsyncMock()

        await manager.connect(ws1, lib_id1)
        await manager.connect(ws2, lib_id1)
        await manager.connect(ws3, lib_id2)

        stats = manager.get_stats()

        assert stats["total_connections"] == 3
        assert stats["libraries_with_subscribers"] == 2
        assert stats["subscriptions_by_library"][str(lib_id1)] == 2
        assert stats["subscriptions_by_library"][str(lib_id2)] == 1

    @pytest.mark.asyncio
    async def test_stats_after_disconnect(self, manager):
        """Test that statistics update after disconnect."""
        library_id = uuid4()

        ws1 = Mock()
        ws1.accept = AsyncMock()
        ws2 = Mock()
        ws2.accept = AsyncMock()

        conn_id1 = await manager.connect(ws1, library_id)
        await manager.connect(ws2, library_id)

        stats_before = manager.get_stats()
        assert stats_before["total_connections"] == 2

        await manager.disconnect(conn_id1)

        stats_after = manager.get_stats()
        assert stats_after["total_connections"] == 1
        assert stats_after["subscriptions_by_library"][str(library_id)] == 1


class TestConnectionManagerSingleton:
    """Test singleton pattern for ConnectionManager."""

    def test_get_connection_manager_singleton(self):
        """Test that get_connection_manager returns same instance."""
        manager1 = get_connection_manager()
        manager2 = get_connection_manager()

        assert manager1 is manager2


class TestConcurrentConnections:
    """Test concurrent connection handling."""

    @pytest.fixture
    def manager(self):
        """Provide fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_multiple_libraries(self, manager):
        """Test managing connections across multiple libraries."""
        lib_ids = [uuid4() for _ in range(5)]

        # Connect 2 clients per library
        for lib_id in lib_ids:
            for _ in range(2):
                ws = Mock()
                ws.accept = AsyncMock()
                await manager.connect(ws, lib_id)

        stats = manager.get_stats()
        assert stats["total_connections"] == 10
        assert stats["libraries_with_subscribers"] == 5

    @pytest.mark.asyncio
    async def test_broadcast_with_partial_failures(self, manager):
        """Test broadcasting when some connections fail."""
        library_id = uuid4()

        # Working connection
        ws1 = Mock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        # Failing connection
        ws2 = Mock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock(side_effect=Exception("Connection error"))

        # Another working connection
        ws3 = Mock()
        ws3.accept = AsyncMock()
        ws3.send_json = AsyncMock()

        await manager.connect(ws1, library_id)
        conn_id2 = await manager.connect(ws2, library_id)
        await manager.connect(ws3, library_id)

        message = {"type": "test"}
        await manager.broadcast_to_library(library_id, message)

        # Working connections should receive message
        ws1.send_json.assert_called_once()
        ws3.send_json.assert_called_once()

        # Failed connection should be removed
        assert conn_id2 not in manager._connections


class TestConnectionMetadata:
    """Test connection metadata tracking."""

    @pytest.fixture
    def manager(self):
        """Provide fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_metadata_stored(self, manager):
        """Test that connection metadata is stored."""
        library_id = uuid4()

        ws = Mock()
        ws.accept = AsyncMock()

        connection_id = await manager.connect(ws, library_id)

        assert connection_id in manager._connection_metadata
        metadata = manager._connection_metadata[connection_id]
        assert metadata["library_id"] == library_id

    @pytest.mark.asyncio
    async def test_metadata_removed_on_disconnect(self, manager):
        """Test that metadata is cleaned up on disconnect."""
        library_id = uuid4()

        ws = Mock()
        ws.accept = AsyncMock()

        connection_id = await manager.connect(ws, library_id)
        assert connection_id in manager._connection_metadata

        await manager.disconnect(connection_id)

        assert connection_id not in manager._connection_metadata


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def manager(self):
        """Provide fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_disconnect_twice(self, manager):
        """Test disconnecting same connection twice."""
        library_id = uuid4()

        ws = Mock()
        ws.accept = AsyncMock()

        connection_id = await manager.connect(ws, library_id)

        await manager.disconnect(connection_id)
        # Second disconnect should not error
        await manager.disconnect(connection_id)

    @pytest.mark.asyncio
    async def test_send_to_disconnected_connection(self, manager):
        """Test sending to a connection after it's disconnected."""
        library_id = uuid4()

        ws = Mock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()

        connection_id = await manager.connect(ws, library_id)
        await manager.disconnect(connection_id)

        message = {"type": "test"}
        # Should not error, just log warning
        await manager.send_message(connection_id, message)

        # Message should not be sent
        ws.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_broadcast(self, manager):
        """Test broadcasting empty message."""
        library_id = uuid4()

        ws = Mock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()

        await manager.connect(ws, library_id)

        await manager.broadcast_to_library(library_id, {})

        ws.send_json.assert_called_once_with({})
