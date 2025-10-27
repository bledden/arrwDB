"""
WebSocket connection manager for real-time bidirectional communication.

Manages WebSocket connections, message routing, and event broadcasting.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections and message routing.

    Features:
    - Connection lifecycle management
    - Per-library subscriptions
    - Message broadcasting
    - Event notification integration
    """

    def __init__(self):
        # Active connections by connection_id
        self._connections: Dict[str, WebSocket] = {}

        # Library subscriptions: library_id -> set of connection_ids
        self._library_subscriptions: Dict[UUID, Set[str]] = {}

        # Connection metadata: connection_id -> {library_id, etc}
        self._connection_metadata: Dict[str, dict] = {}

        logger.info("ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, library_id: UUID) -> str:
        """
        Accept a new WebSocket connection and subscribe to library events.

        Args:
            websocket: The WebSocket connection
            library_id: Library to subscribe to

        Returns:
            connection_id: Unique identifier for this connection
        """
        await websocket.accept()

        connection_id = str(uuid4())
        self._connections[connection_id] = websocket
        self._connection_metadata[connection_id] = {
            "library_id": library_id,
        }

        # Subscribe to library events
        if library_id not in self._library_subscriptions:
            self._library_subscriptions[library_id] = set()
        self._library_subscriptions[library_id].add(connection_id)

        logger.info(
            f"WebSocket connected: {connection_id} (library: {library_id}), "
            f"total connections: {len(self._connections)}"
        )

        return connection_id

    async def disconnect(self, connection_id: str):
        """
        Remove a WebSocket connection and clean up subscriptions.

        Args:
            connection_id: Connection to remove
        """
        if connection_id not in self._connections:
            return

        # Get metadata before removing
        metadata = self._connection_metadata.get(connection_id, {})
        library_id = metadata.get("library_id")

        # Remove from subscriptions
        if library_id and library_id in self._library_subscriptions:
            self._library_subscriptions[library_id].discard(connection_id)
            if not self._library_subscriptions[library_id]:
                del self._library_subscriptions[library_id]

        # Remove connection
        del self._connections[connection_id]
        if connection_id in self._connection_metadata:
            del self._connection_metadata[connection_id]

        logger.info(
            f"WebSocket disconnected: {connection_id}, "
            f"remaining connections: {len(self._connections)}"
        )

    async def send_message(self, connection_id: str, message: dict):
        """
        Send a message to a specific connection.

        Args:
            connection_id: Target connection
            message: Message to send
        """
        if connection_id not in self._connections:
            logger.warning(f"Cannot send to unknown connection: {connection_id}")
            return

        websocket = self._connections[connection_id]
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            await self.disconnect(connection_id)

    async def broadcast_to_library(self, library_id: UUID, message: dict):
        """
        Broadcast a message to all connections subscribed to a library.

        Args:
            library_id: Target library
            message: Message to broadcast
        """
        if library_id not in self._library_subscriptions:
            return

        connection_ids = list(self._library_subscriptions[library_id])

        logger.debug(
            f"Broadcasting to library {library_id}: "
            f"{len(connection_ids)} connections"
        )

        # Send to all connections in parallel
        tasks = [
            self.send_message(conn_id, message)
            for conn_id in connection_ids
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_event(self, event_type: str, library_id: UUID, data: dict):
        """
        Broadcast an event notification to subscribed connections.

        Args:
            event_type: Type of event (e.g., "library.created")
            library_id: Library that event relates to
            data: Event payload
        """
        message = {
            "type": "event",
            "event_type": event_type,
            "library_id": str(library_id),
            "data": data,
        }

        await self.broadcast_to_library(library_id, message)

    def get_stats(self) -> dict:
        """
        Get connection statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_connections": len(self._connections),
            "libraries_with_subscribers": len(self._library_subscriptions),
            "subscriptions_by_library": {
                str(lib_id): len(conn_ids)
                for lib_id, conn_ids in self._library_subscriptions.items()
            },
        }


# Global singleton
_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """
    Get or create the global ConnectionManager singleton.

    Returns:
        The connection manager instance
    """
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager
