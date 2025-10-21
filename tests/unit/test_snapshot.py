"""
Comprehensive tests for Snapshot functionality.

Tests snapshot creation, loading, and management for database recovery.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from infrastructure.persistence.snapshot import Snapshot, SnapshotManager


class TestSnapshot:
    """Test Snapshot class."""

    def test_snapshot_creation(self):
        """Test creating a snapshot."""
        timestamp = datetime.utcnow()
        data = {"libraries": [], "documents": []}

        snapshot = Snapshot(timestamp=timestamp, data=data)

        assert snapshot.timestamp == timestamp
        assert snapshot.data == data

    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        data = {"count": 42, "items": ["a", "b"]}

        snapshot = Snapshot(timestamp=timestamp, data=data)
        snapshot_dict = snapshot.to_dict()

        assert "timestamp" in snapshot_dict
        assert snapshot_dict["data"] == data

    def test_snapshot_from_dict(self):
        """Test creating snapshot from dictionary."""
        snapshot_dict = {
            "timestamp": "2024-01-01T12:00:00",
            "data": {"libraries": [{"id": "123"}]}
        }

        snapshot = Snapshot.from_dict(snapshot_dict)

        assert snapshot.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert snapshot.data["libraries"][0]["id"] == "123"

    def test_snapshot_roundtrip(self):
        """Test full snapshot serialization roundtrip."""
        original = Snapshot(
            timestamp=datetime(2024, 6, 15, 10, 30, 0),
            data={"vectors": 100, "name": "test_db"}
        )

        # to_dict -> from_dict roundtrip
        reconstructed = Snapshot.from_dict(original.to_dict())

        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.data == original.data


class TestSnapshotManager:
    """Test SnapshotManager implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for snapshots."""
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create SnapshotManager instance."""
        return SnapshotManager(snapshot_dir=temp_dir)

    def test_manager_initialization_creates_directory(self, temp_dir):
        """Test that manager creates snapshot directory."""
        snap_dir = temp_dir / "snapshots"
        assert not snap_dir.exists()

        manager = SnapshotManager(snapshot_dir=snap_dir)

        assert snap_dir.exists()
        assert snap_dir.is_dir()

    def test_create_snapshot(self, manager):
        """Test creating a snapshot."""
        data = {"libraries": [{"id": str(uuid4()), "name": "test"}]}

        filename = manager.create_snapshot(data)

        assert filename is not None
        assert str(filename).endswith(".pkl")

    def test_create_multiple_snapshots(self, manager):
        """Test creating multiple snapshots."""
        for i in range(3):
            data = {"iteration": i, "count": i * 10}
            filename = manager.create_snapshot(data)
            assert filename is not None

    def test_load_latest_snapshot(self, manager):
        """Test loading the most recent snapshot."""
        # Create snapshots
        data1 = {"version": 1}
        data2 = {"version": 2}

        manager.create_snapshot(data1)
        import time
        time.sleep(0.01)  # Ensure different timestamps
        manager.create_snapshot(data2)

        # Load latest
        latest = manager.load_latest_snapshot()

        assert latest is not None
        assert latest.data["version"] == 2

    def test_load_latest_when_no_snapshots(self, manager):
        """Test loading latest when no snapshots exist."""
        latest = manager.load_latest_snapshot()
        assert latest is None

    def test_load_specific_snapshot(self, manager):
        """Test loading a specific snapshot by filename."""
        data = {"test": "data", "id": str(uuid4())}
        filename = manager.create_snapshot(data)

        loaded = manager.load_snapshot(filename)

        assert loaded is not None
        assert loaded.data == data

    def test_list_snapshots(self, manager):
        """Test listing all snapshots."""
        # Create snapshots
        for i in range(3):
            manager.create_snapshot({"index": i})

        snapshots = manager.list_snapshots()

        assert len(snapshots) == 3
        assert all("filename" in s for s in snapshots)
        assert all("timestamp" in s for s in snapshots)

    def test_list_snapshots_when_empty(self, manager):
        """Test listing snapshots when none exist."""
        snapshots = manager.list_snapshots()
        assert snapshots == []

    def test_delete_snapshot(self, manager):
        """Test deleting a specific snapshot."""
        data = {"to_delete": True}
        filename = manager.create_snapshot(data)

        # Verify it exists
        assert manager.load_snapshot(filename) is not None

        # Delete it
        result = manager.delete_snapshot(filename)

        assert result is True

        # Verify it's gone
        with pytest.raises(FileNotFoundError):
            manager.load_snapshot(filename)

    def test_delete_nonexistent_snapshot(self, manager):
        """Test deleting a snapshot that doesn't exist."""
        result = manager.delete_snapshot("nonexistent.pkl")
        assert result is False

    def test_max_snapshots_retention(self, temp_dir):
        """Test that old snapshots are cleaned up."""
        # Create manager with max 3 snapshots
        manager = SnapshotManager(snapshot_dir=temp_dir, max_snapshots=3)

        # Create 5 snapshots
        for i in range(5):
            manager.create_snapshot({"index": i})
            import time
            time.sleep(0.01)

        # Should only have 3 newest
        snapshots = manager.list_snapshots()
        assert len(snapshots) <= 3

    def test_snapshot_with_compression_enabled(self, temp_dir):
        """Test snapshots with compression."""
        manager = SnapshotManager(
            snapshot_dir=temp_dir,
            use_compression=True
        )

        data = {"large_data": "x" * 1000}
        filename = manager.create_snapshot(data)

        loaded = manager.load_snapshot(filename)
        assert loaded.data == data

    def test_snapshot_with_compression_disabled(self, temp_dir):
        """Test snapshots without compression."""
        manager = SnapshotManager(
            snapshot_dir=temp_dir,
            use_compression=False
        )

        data = {"uncompressed": "data"}
        filename = manager.create_snapshot(data)

        loaded = manager.load_snapshot(filename)
        assert loaded.data == data

    def test_snapshot_with_complex_data(self, manager):
        """Test snapshot with complex nested data."""
        data = {
            "libraries": [
                {"id": str(uuid4()), "vectors": [1.0, 2.0, 3.0]},
                {"id": str(uuid4()), "vectors": [4.0, 5.0, 6.0]}
            ],
            "metadata": {
                "version": "1.0",
                "timestamp": datetime.utcnow().isoformat()
            },
            "counts": {"documents": 42, "chunks": 1337}
        }

        filename = manager.create_snapshot(data)
        loaded = manager.load_snapshot(filename)

        assert loaded.data["libraries"] == data["libraries"]
        assert loaded.data["counts"] == data["counts"]

    def test_concurrent_snapshot_creation(self, manager):
        """Test thread-safe snapshot creation."""
        import threading

        created_files = []
        lock = threading.Lock()

        def create_snap(thread_id):
            filename = manager.create_snapshot({"thread": thread_id})
            with lock:
                created_files.append(filename)

        # Create snapshots concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_snap, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed
        assert len(created_files) == 5
        assert len(set(created_files)) == 5  # All unique

    def test_manager_repr(self, manager, temp_dir):
        """Test SnapshotManager __repr__."""
        repr_str = repr(manager)
        assert "SnapshotManager" in repr_str
        assert str(temp_dir) in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
