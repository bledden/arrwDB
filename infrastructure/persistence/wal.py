"""
Write-Ahead Log (WAL) for durability.

This module provides a Write-Ahead Log that records all operations
before they're applied, ensuring durability and enabling crash recovery.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Types of operations that can be logged."""

    CREATE_LIBRARY = "create_library"
    DELETE_LIBRARY = "delete_library"
    ADD_DOCUMENT = "add_document"
    DELETE_DOCUMENT = "delete_document"
    ADD_CHUNK = "add_chunk"
    DELETE_CHUNK = "delete_chunk"


class WALEntry:
    """
    A single entry in the Write-Ahead Log.

    Each entry represents one operation that was performed on the database.
    """

    def __init__(
        self,
        operation_type: OperationType,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ):
        """
        Initialize a WAL entry.

        Args:
            operation_type: The type of operation.
            data: The operation data (must be JSON-serializable).
            timestamp: Optional timestamp (defaults to now).
        """
        self.operation_type = operation_type
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_type": self.operation_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WALEntry":
        """Create from dictionary."""
        return cls(
            operation_type=OperationType(d["operation_type"]),
            data=d["data"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "WALEntry":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class WriteAheadLog:
    """
    Write-Ahead Log for operation durability.

    The WAL records all operations in append-only files before they're
    applied to the database. This ensures that:
    1. Operations are durable (survive crashes)
    2. Operations can be replayed for recovery
    3. Operations can be replicated to followers

    Thread-Safety: All methods are thread-safe using a lock.
    """

    def __init__(
        self,
        wal_dir: Path,
        max_file_size: int = 100 * 1024 * 1024,  # 100 MB
        sync_on_write: bool = True,
    ):
        """
        Initialize the Write-Ahead Log.

        Args:
            wal_dir: Directory to store WAL files.
            max_file_size: Maximum size of a single WAL file before rotation.
            sync_on_write: Whether to call fsync after each write for durability.
                Set to False for better performance at risk of data loss.
        """
        self._wal_dir = wal_dir
        self._wal_dir.mkdir(parents=True, exist_ok=True)
        self._max_file_size = max_file_size
        self._sync_on_write = sync_on_write
        self._lock = threading.Lock()

        # Current WAL file
        self._current_file: Optional[Path] = None
        self._current_file_handle = None
        self._current_file_size = 0
        self._sequence_number = 0

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the WAL by finding the latest file or creating a new one."""
        # Find existing WAL files
        wal_files = sorted(self._wal_dir.glob("wal_*.log"))

        if wal_files:
            # Get the latest WAL file
            latest_file = wal_files[-1]
            self._current_file = latest_file

            # Parse sequence number from filename
            filename = latest_file.stem  # e.g., "wal_00000042"
            self._sequence_number = int(filename.split("_")[1])

            # Open in append mode
            self._current_file_handle = open(latest_file, "a", encoding="utf-8")
            self._current_file_size = latest_file.stat().st_size

            logger.info(
                f"Opened existing WAL file: {latest_file} "
                f"(size: {self._current_file_size} bytes)"
            )
        else:
            # Create first WAL file
            self._rotate_file()

    def _rotate_file(self) -> None:
        """Rotate to a new WAL file."""
        # Close current file if open
        if self._current_file_handle is not None:
            self._current_file_handle.close()

        # Increment sequence number
        self._sequence_number += 1

        # Create new file with zero-padded sequence number
        filename = f"wal_{self._sequence_number:08d}.log"
        self._current_file = self._wal_dir / filename

        # Open new file
        self._current_file_handle = open(
            self._current_file, "w", encoding="utf-8"
        )
        self._current_file_size = 0

        logger.info(f"Rotated to new WAL file: {self._current_file}")

    def append(self, entry: WALEntry) -> None:
        """
        Append an entry to the WAL.

        Args:
            entry: The WAL entry to append.

        Raises:
            IOError: If write fails.
        """
        with self._lock:
            # Check if we need to rotate
            if self._current_file_size >= self._max_file_size:
                self._rotate_file()

            # Serialize entry
            json_str = entry.to_json()
            line = json_str + "\n"

            # Write to file
            try:
                self._current_file_handle.write(line)

                # Sync to disk if configured
                if self._sync_on_write:
                    self._current_file_handle.flush()
                    import os

                    os.fsync(self._current_file_handle.fileno())

                self._current_file_size += len(line.encode("utf-8"))

            except IOError as e:
                logger.error(f"Failed to write to WAL: {e}")
                raise

    def append_operation(
        self, operation_type: OperationType, data: Dict[str, Any]
    ) -> None:
        """
        Convenience method to append an operation.

        Args:
            operation_type: The type of operation.
            data: The operation data.
        """
        entry = WALEntry(operation_type, data)
        self.append(entry)

    def read_all(self) -> List[WALEntry]:
        """
        Read all entries from all WAL files.

        Returns:
            List of all WAL entries in order.

        Raises:
            IOError: If read fails.
        """
        with self._lock:
            entries = []

            # Read all WAL files in order
            wal_files = sorted(self._wal_dir.glob("wal_*.log"))

            for wal_file in wal_files:
                try:
                    with open(wal_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    entry = WALEntry.from_json(line)
                                    entries.append(entry)
                                except json.JSONDecodeError as e:
                                    logger.error(
                                        f"Corrupted WAL entry in {wal_file}: {e}"
                                    )
                                    # Skip corrupted entries
                                    continue

                except IOError as e:
                    logger.error(f"Failed to read WAL file {wal_file}: {e}")
                    raise

            logger.info(f"Read {len(entries)} entries from WAL")
            return entries

    def truncate_before(self, timestamp: datetime) -> int:
        """
        Truncate WAL entries before a given timestamp.

        This is typically called after a successful snapshot, as older
        entries are no longer needed for recovery.

        Args:
            timestamp: Entries before this time will be removed.

        Returns:
            Number of entries removed.

        Raises:
            IOError: If file operations fail.
        """
        with self._lock:
            # Close current file
            if self._current_file_handle is not None:
                self._current_file_handle.close()
                self._current_file_handle = None

            removed_count = 0
            wal_files = sorted(self._wal_dir.glob("wal_*.log"))

            for wal_file in wal_files:
                # Read entries from this file
                keep_entries = []

                with open(wal_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            entry = WALEntry.from_json(line)
                            if entry.timestamp >= timestamp:
                                keep_entries.append(entry)
                            else:
                                removed_count += 1
                        except json.JSONDecodeError:
                            # Keep corrupted entries to be safe
                            keep_entries.append(None)
                            continue

                # If file has entries to keep, rewrite it
                if keep_entries:
                    with open(wal_file, "w", encoding="utf-8") as f:
                        for entry in keep_entries:
                            if entry is not None:
                                f.write(entry.to_json() + "\n")
                else:
                    # Delete empty file
                    wal_file.unlink()
                    logger.info(f"Deleted empty WAL file: {wal_file}")

            # Re-open current file
            self._initialize()

            logger.info(f"Truncated {removed_count} WAL entries before {timestamp}")
            return removed_count

    def close(self) -> None:
        """Close the WAL."""
        with self._lock:
            if self._current_file_handle is not None:
                self._current_file_handle.close()
                self._current_file_handle = None
                logger.info("Closed WAL")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WriteAheadLog(dir={self._wal_dir}, "
            f"current_file={self._current_file}, "
            f"size={self._current_file_size})"
        )
