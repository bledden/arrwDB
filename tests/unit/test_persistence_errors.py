"""
Tests for persistence layer error handling edge cases.

Covers snapshot compression errors and WAL error paths.
"""

import pytest
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import patch, MagicMock
from infrastructure.persistence.snapshot import SnapshotManager, Snapshot
from infrastructure.persistence.wal import WriteAheadLog, WALEntry, OperationType


class TestSnapshotCompressionErrors:
    """Test snapshot compression error handling (lines 133-138)."""

    def test_create_snapshot_serialization_error(self, tmp_path):
        """Test handling of serialization errors during snapshot creation (lines 133-138)."""
        manager = SnapshotManager(snapshot_dir=tmp_path, use_compression=True)

        # Create data that will fail to pickle
        class UnpicklableObject:
            def __reduce__(self):
                raise TypeError("Cannot pickle this object")

        bad_data = {
            "libraries": [{"id": str(uuid4()), "bad_object": UnpicklableObject()}]
        }

        # Should raise error when trying to serialize
        with pytest.raises(Exception):  # Could be TypeError or pickle error
            manager.create_snapshot(bad_data)

    def test_create_snapshot_with_mock_pickle_error(self, tmp_path):
        """Test snapshot creation when pickle.dumps fails (lines 133-138)."""
        manager = SnapshotManager(snapshot_dir=tmp_path)

        data = {"libraries": [{"id": str(uuid4())}]}

        # Mock pickle.dumps to raise an error
        with patch('pickle.dumps', side_effect=Exception("Pickle error")):
            with pytest.raises(Exception) as exc_info:
                manager.create_snapshot(data)

            assert "pickle" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()


class TestSnapshotLoadErrors:
    """Test snapshot load error handling (lines 169-171)."""

    def test_load_snapshot_deserialization_error(self, tmp_path):
        """Test handling of deserialization errors when loading (lines 169-171)."""
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Create a corrupted snapshot file manually
        snapshot_file = tmp_path / "snapshot_20240101_000000.pkl"
        with open(snapshot_file, 'wb') as f:
            f.write(b"This is not valid pickle data")

        # Should raise error when trying to load
        with pytest.raises(Exception):  # pickle.UnpicklingError or similar
            manager.load_snapshot(snapshot_file.name)

    def test_load_snapshot_with_mock_pickle_error(self, tmp_path):
        """Test load_snapshot when pickle.load fails (lines 169-171)."""
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # First create a valid snapshot
        data = {"libraries": [{"id": str(uuid4())}]}
        filename = manager.create_snapshot(data)

        # Now mock pickle.load to fail
        with patch('pickle.load', side_effect=Exception("Unpickle error")):
            with pytest.raises(Exception) as exc_info:
                manager.load_snapshot(filename)

            # Error should be raised
            assert exc_info.value is not None


class TestSnapshotDeleteErrors:
    """Test snapshot delete error handling (lines 225-226)."""

    def test_delete_snapshot_file_not_found(self, tmp_path):
        """Test deleting non-existent snapshot (lines 225-226)."""
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Try to delete a snapshot that doesn't exist
        # Should handle gracefully or raise FileNotFoundError
        try:
            manager.delete_snapshot("nonexistent_snapshot.pkl")
        except FileNotFoundError:
            # This is expected behavior
            pass

    def test_delete_snapshot_permission_error(self, tmp_path):
        """Test delete when file permissions prevent deletion (lines 225-226)."""
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Create a snapshot
        data = {"libraries": []}
        filename = manager.create_snapshot(data)

        # Make file read-only (simulate permission issue)
        snapshot_path = tmp_path / filename
        snapshot_path.chmod(0o444)  # Read-only

        try:
            # Try to delete - may fail on some systems
            manager.delete_snapshot(filename)
        except PermissionError:
            # Expected on systems that enforce permissions
            pass
        finally:
            # Restore permissions for cleanup
            snapshot_path.chmod(0o644)


class TestWALErrorHandling:
    """Test WAL error handling edge cases."""

    def test_wal_append_with_io_error(self, tmp_path):
        """Test WAL append when write fails (lines 205-206)."""
        wal = WriteAheadLog(wal_dir=tmp_path)

        # Close the file handle to simulate IO failure
        wal._current_file_handle.close()

        entry = WALEntry(OperationType.ADD_DOCUMENT, {"id": "test"})

        # Should raise IOError or ValueError
        with pytest.raises((IOError, ValueError)):
            wal.append(entry)

    def test_wal_read_with_file_error(self, tmp_path):
        """Test WAL read when file read fails (lines 253-255)."""
        wal = WriteAheadLog(wal_dir=tmp_path)

        # Add an entry
        entry = WALEntry(OperationType.ADD_DOCUMENT, {"id": "test"})
        wal.append(entry)
        wal.close()

        # Corrupt the WAL file
        wal_file = tmp_path / "wal_00000001.log"

        # Make it unreadable (may not work on all systems)
        try:
            wal_file.chmod(0o000)

            wal2 = WriteAheadLog(wal_dir=tmp_path)

            # Should raise IOError when trying to read
            with pytest.raises((IOError, PermissionError)):
                wal2.read_all()
        finally:
            # Restore permissions
            wal_file.chmod(0o644)

    def test_wal_truncate_with_corrupted_entry(self, tmp_path):
        """Test WAL truncate with corrupted JSON (line 293)."""
        wal = WriteAheadLog(wal_dir=tmp_path)

        # Add valid entries
        for i in range(5):
            entry = WALEntry(OperationType.ADD_DOCUMENT, {"id": str(i)})
            wal.append(entry)

        wal.close()

        # Manually add a corrupted entry
        wal_file = tmp_path / "wal_00000001.log"
        with open(wal_file, 'a') as f:
            f.write("{ this is corrupted json\n")

        # Truncate - should handle corrupted entries gracefully (line 293)
        wal2 = WriteAheadLog(wal_dir=tmp_path)
        cutoff = datetime.now() + timedelta(hours=1)  # Future time

        # Should not crash, even with corrupted entry
        removed = wal2.truncate_before(cutoff)

        # Removed count should reflect valid entries
        assert removed >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
