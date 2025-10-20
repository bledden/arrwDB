"""
Snapshot management for periodic state persistence.

This module provides functionality to create and load full snapshots
of the database state, enabling fast recovery without replaying the
entire WAL.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import logging
import threading

logger = logging.getLogger(__name__)


class Snapshot:
    """
    A point-in-time snapshot of the database state.

    Snapshots contain the full state of all libraries, documents, and chunks.
    They're used for:
    1. Fast recovery (avoiding WAL replay)
    2. Backup and restore
    3. Replication to new followers
    """

    def __init__(self, timestamp: datetime, data: Dict[str, Any]):
        """
        Initialize a snapshot.

        Args:
            timestamp: When this snapshot was created.
            data: The full database state.
        """
        self.timestamp = timestamp
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"timestamp": self.timestamp.isoformat(), "data": self.data}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Snapshot":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(d["timestamp"]), data=d["data"]
        )


class SnapshotManager:
    """
    Manager for creating and loading snapshots.

    The manager:
    - Creates periodic snapshots of the database state
    - Stores snapshots in compressed format
    - Provides snapshot loading for recovery
    - Manages snapshot retention (keeping only N recent snapshots)

    Thread-Safety: All methods are thread-safe using a lock.
    """

    def __init__(
        self,
        snapshot_dir: Path,
        max_snapshots: int = 5,
        use_compression: bool = True,
    ):
        """
        Initialize the snapshot manager.

        Args:
            snapshot_dir: Directory to store snapshot files.
            max_snapshots: Maximum number of snapshots to retain.
            use_compression: Whether to use pickle compression.
        """
        self._snapshot_dir = snapshot_dir
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._max_snapshots = max_snapshots
        self._use_compression = use_compression
        self._lock = threading.Lock()

    def create_snapshot(
        self, data: Dict[str, Any], timestamp: Optional[datetime] = None
    ) -> Path:
        """
        Create a new snapshot.

        Args:
            data: The full database state to snapshot.
            timestamp: Optional timestamp (defaults to now).

        Returns:
            Path to the created snapshot file.

        Raises:
            IOError: If snapshot creation fails.
        """
        with self._lock:
            timestamp = timestamp or datetime.utcnow()
            snapshot = Snapshot(timestamp, data)

            # Create filename with timestamp
            filename = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.pkl"
            filepath = self._snapshot_dir / filename

            logger.info(f"Creating snapshot: {filepath}")

            try:
                # Write snapshot
                with open(filepath, "wb") as f:
                    if self._use_compression:
                        # Use highest protocol for better compression
                        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        pickle.dump(snapshot, f)

                # Get file size
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(
                    f"Created snapshot {filepath.name} ({size_mb:.2f} MB)"
                )

                # Clean up old snapshots
                self._cleanup_old_snapshots()

                return filepath

            except Exception as e:
                logger.error(f"Failed to create snapshot: {e}")
                # Clean up partial file
                if filepath.exists():
                    filepath.unlink()
                raise

    def load_latest_snapshot(self) -> Optional[Snapshot]:
        """
        Load the most recent snapshot.

        Returns:
            The latest snapshot, or None if no snapshots exist.

        Raises:
            IOError: If snapshot loading fails.
        """
        with self._lock:
            snapshot_files = sorted(self._snapshot_dir.glob("snapshot_*.pkl"))

            if not snapshot_files:
                logger.info("No snapshots found")
                return None

            latest_file = snapshot_files[-1]
            logger.info(f"Loading snapshot: {latest_file}")

            try:
                with open(latest_file, "rb") as f:
                    snapshot = pickle.load(f)

                logger.info(
                    f"Loaded snapshot from {snapshot.timestamp.isoformat()}"
                )
                return snapshot

            except Exception as e:
                logger.error(f"Failed to load snapshot {latest_file}: {e}")
                raise

    def load_snapshot(self, filename: str) -> Snapshot:
        """
        Load a specific snapshot by filename.

        Args:
            filename: Name of the snapshot file.

        Returns:
            The loaded snapshot.

        Raises:
            FileNotFoundError: If snapshot doesn't exist.
            IOError: If snapshot loading fails.
        """
        with self._lock:
            filepath = self._snapshot_dir / filename

            if not filepath.exists():
                raise FileNotFoundError(f"Snapshot not found: {filename}")

            logger.info(f"Loading snapshot: {filepath}")

            with open(filepath, "rb") as f:
                snapshot = pickle.load(f)

            logger.info(f"Loaded snapshot from {snapshot.timestamp.isoformat()}")
            return snapshot

    def list_snapshots(self) -> list[Dict[str, Any]]:
        """
        List all available snapshots.

        Returns:
            List of snapshot info dictionaries with:
            - filename: Snapshot filename
            - timestamp: When snapshot was created
            - size_mb: Size in megabytes
        """
        with self._lock:
            snapshot_files = sorted(self._snapshot_dir.glob("snapshot_*.pkl"))

            snapshots = []
            for filepath in snapshot_files:
                size_mb = filepath.stat().st_size / (1024 * 1024)

                # Parse timestamp from filename
                # Format: snapshot_YYYYMMDD_HHMMSS_ffffff.pkl
                try:
                    timestamp_str = filepath.stem.replace("snapshot_", "")
                    timestamp = datetime.strptime(
                        timestamp_str, "%Y%m%d_%H%M%S_%f"
                    )
                except ValueError:
                    timestamp = None

                snapshots.append(
                    {
                        "filename": filepath.name,
                        "timestamp": timestamp.isoformat() if timestamp else None,
                        "size_mb": round(size_mb, 2),
                    }
                )

            return snapshots

    def delete_snapshot(self, filename: str) -> bool:
        """
        Delete a specific snapshot.

        Args:
            filename: Name of the snapshot file.

        Returns:
            True if deleted, False if didn't exist.
        """
        with self._lock:
            filepath = self._snapshot_dir / filename

            if not filepath.exists():
                return False

            filepath.unlink()
            logger.info(f"Deleted snapshot: {filename}")
            return True

    def _cleanup_old_snapshots(self) -> None:
        """
        Remove old snapshots beyond the retention limit.

        Keeps only the most recent max_snapshots snapshots.
        """
        snapshot_files = sorted(self._snapshot_dir.glob("snapshot_*.pkl"))

        if len(snapshot_files) <= self._max_snapshots:
            return

        # Delete oldest snapshots
        to_delete = snapshot_files[: -self._max_snapshots]
        for filepath in to_delete:
            filepath.unlink()
            logger.info(f"Deleted old snapshot: {filepath.name}")

        logger.info(
            f"Cleaned up {len(to_delete)} old snapshots "
            f"(keeping {self._max_snapshots})"
        )

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            num_snapshots = len(list(self._snapshot_dir.glob("snapshot_*.pkl")))
            return (
                f"SnapshotManager(dir={self._snapshot_dir}, "
                f"snapshots={num_snapshots}, "
                f"max={self._max_snapshots})"
            )
