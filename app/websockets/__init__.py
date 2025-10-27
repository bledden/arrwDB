"""
WebSocket support for real-time bidirectional communication.
"""

from app.websockets.manager import ConnectionManager, get_connection_manager

__all__ = ["ConnectionManager", "get_connection_manager"]
