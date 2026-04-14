"""
Leader-Follower Replication via WAL Streaming.

Architecture:
  Leader: Accepts writes, appends to WAL, streams entries to followers.
  Follower: Consumes WAL stream, applies entries locally, serves reads.

Protocol:
  1. Follower connects to leader's replication endpoint
  2. Sends its last known WAL position (sequence number)
  3. Leader streams all entries after that position
  4. Follower applies entries and acknowledges

Failover:
  If leader goes down, a follower can be promoted by setting
  REPLICATION_ROLE=leader. It starts accepting writes and the
  remaining followers reconnect to it.

Configuration:
  REPLICATION_ENABLED=true
  REPLICATION_ROLE=leader|follower
  REPLICATION_LEADER_URL=http://leader:8000  (for followers)
  REPLICATION_PORT=8001  (streaming port)
"""

import asyncio
import logging
import time
from enum import Enum
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


class ReplicationRole(str, Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    STANDALONE = "standalone"  # Default: no replication


class WALEntry:
    """A single WAL entry to be replicated."""
    def __init__(self, sequence: int, operation: str, data: dict, timestamp: float):
        self.sequence = sequence
        self.operation = operation  # "add_vector", "remove_vector", "upsert_vector", etc.
        self.data = data
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "seq": self.sequence,
            "op": self.operation,
            "data": self.data,
            "ts": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WALEntry":
        return cls(
            sequence=d["seq"],
            operation=d["op"],
            data=d["data"],
            timestamp=d["ts"],
        )


class ReplicationLeader:
    """Manages WAL streaming to followers."""

    def __init__(self):
        self._sequence = 0
        self._wal_buffer: list[WALEntry] = []
        self._max_buffer_size = 100_000
        self._followers: dict[str, int] = {}  # follower_id -> last_acked_seq

    def append(self, operation: str, data: dict) -> int:
        """Append a write operation to the replication WAL."""
        self._sequence += 1
        entry = WALEntry(
            sequence=self._sequence,
            operation=operation,
            data=data,
            timestamp=time.time(),
        )
        self._wal_buffer.append(entry)

        # Trim buffer if too large
        if len(self._wal_buffer) > self._max_buffer_size:
            self._wal_buffer = self._wal_buffer[-self._max_buffer_size:]

        return self._sequence

    async def stream_from(self, after_sequence: int) -> AsyncIterator[WALEntry]:
        """Stream WAL entries after the given sequence number."""
        # First, send buffered entries
        for entry in self._wal_buffer:
            if entry.sequence > after_sequence:
                yield entry

        # Then wait for new entries (long-poll style)
        last_sent = self._sequence
        while True:
            await asyncio.sleep(0.1)  # Poll interval
            for entry in self._wal_buffer:
                if entry.sequence > last_sent:
                    last_sent = entry.sequence
                    yield entry

    def register_follower(self, follower_id: str, last_seq: int):
        self._followers[follower_id] = last_seq
        logger.info(f"Follower {follower_id} registered at seq={last_seq}")

    def ack(self, follower_id: str, sequence: int):
        self._followers[follower_id] = sequence

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def follower_count(self) -> int:
        return len(self._followers)


class ReplicationFollower:
    """Consumes WAL stream from leader and applies entries locally."""

    def __init__(self, leader_url: str, apply_fn=None):
        self._leader_url = leader_url
        self._last_sequence = 0
        self._apply_fn = apply_fn  # Callback to apply a WALEntry locally

    @property
    def last_sequence(self) -> int:
        return self._last_sequence

    async def start_streaming(self):
        """Connect to leader and consume WAL stream."""
        import aiohttp

        url = f"{self._leader_url}/v1/replication/stream?after={self._last_sequence}"
        logger.info(f"Connecting to leader at {url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                async for line in response.content:
                    line = line.decode().strip()
                    if not line:
                        continue
                    try:
                        import json
                        entry = WALEntry.from_dict(json.loads(line))
                        if self._apply_fn:
                            self._apply_fn(entry)
                        self._last_sequence = entry.sequence
                    except Exception as e:
                        logger.error(f"Failed to apply WAL entry: {e}")

    def apply_entry(self, entry: WALEntry):
        """Apply a single WAL entry to the local index."""
        if self._apply_fn:
            self._apply_fn(entry)
        self._last_sequence = entry.sequence
