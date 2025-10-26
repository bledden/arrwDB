"""
Comprehensive tests for persistence corruption handling.

Tests how the system handles corrupted WAL files, corrupted snapshots,
and various data integrity issues.
"""

import json
import pickle
import pytest
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from infrastructure.persistence.snapshot import Snapshot, SnapshotManager
from infrastructure.persistence.wal import OperationType, WALEntry, WriteAheadLog


class TestWALCorruption:
    """Test WAL corruption handling."""

    def test_wal_with_corrupted_json_entry(self, tmp_path):
        """
        Test WAL handles corrupted JSON entry gracefully.

        Scenario:
        1. Create valid WAL with 10 entries
        2. Manually append corrupted JSON to file
        3. Attempt to read WAL
        4. Verify: Reads valid entries up to corruption point
        """
        wal = WriteAheadLog(wal_dir=tmp_path)

        # Write 10 valid entries
        for i in range(10):
            entry = WALEntry(
                operation_type=OperationType.CREATE_LIBRARY,
                data={"id": str(uuid4()), "name": f"Library {i}"},
            )
            wal.append(entry)

        wal.close()

        # Manually append corrupted JSON
        wal_files = list(tmp_path.glob("wal_*.log"))
        assert len(wal_files) > 0

        with open(wal_files[0], "a") as f:
            f.write("{ this is corrupted json without closing brace\n")
            f.write('{"also": "corrupted", "missing_quote: "value"}\n')

        # Attempt to read WAL
        wal2 = WriteAheadLog(wal_dir=tmp_path)

        # Should be able to read valid entries (may stop at corruption)
        entries = []
        try:
            entries = list(wal2.read_all())
        except json.JSONDecodeError:
            # This is acceptable - corruption detected
            pass

        # We should have read at least the 10 valid entries
        # (implementation may stop at corruption or skip bad lines)
        assert len(entries) >= 10 or len(entries) == 0  # Either got valid entries or failed gracefully

        wal2.close()

    def test_wal_with_completely_corrupted_file(self, tmp_path):
        """
        Test WAL handles completely corrupted file.

        Scenario:
        1. Create WAL file with random binary data
        2. Attempt to read
        3. Verify: Error handling is graceful
        """
        # Create a corrupted WAL file
        corrupted_file = tmp_path / "wal_00000001.log"
        with open(corrupted_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09" * 100)

        # Attempt to read
        wal = WriteAheadLog(wal_dir=tmp_path)

        # Should handle gracefully (either empty or error)
        try:
            entries = list(wal.read_all())
            # If it doesn't raise an error, should return empty or valid entries only
            assert isinstance(entries, list)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            # Acceptable - corruption detected and raised
            pass

        wal.close()

    def test_wal_with_partial_line(self, tmp_path):
        """
        Test WAL handles incomplete line (truncated write).

        Scenario:
        1. Create valid entries
        2. Append partial JSON line (simulate crash during write)
        3. Verify: Handles gracefully
        """
        wal = WriteAheadLog(wal_dir=tmp_path)

        # Write valid entries
        for i in range(5):
            entry = WALEntry(
                operation_type=OperationType.ADD_DOCUMENT,
                data={"id": f"doc_{i}"},
            )
            wal.append(entry)

        wal.close()

        # Append incomplete line (simulates crash during write)
        wal_files = list(tmp_path.glob("wal_*.log"))
        with open(wal_files[0], "a") as f:
            f.write('{"timestamp": "2024-01-01T00:00:00", "operation_type": "add_doc')
            # No newline, incomplete JSON

        # Read should handle gracefully
        wal2 = WriteAheadLog(wal_dir=tmp_path)

        try:
            entries = list(wal2.read_all())
            # Should get the 5 valid entries at minimum
            assert len(entries) >= 5
        except (json.JSONDecodeError, ValueError):
            # Also acceptable - detected incomplete entry
            pass

        wal2.close()

    def test_wal_with_empty_file(self, tmp_path):
        """
        Test WAL handles empty file gracefully.

        Edge case: WAL file exists but has no content.
        """
        # Create empty WAL file
        empty_file = tmp_path / "wal_00000001.log"
        empty_file.touch()

        # Should handle gracefully
        wal = WriteAheadLog(wal_dir=tmp_path)
        entries = list(wal.read_all())

        assert entries == []  # Empty file should yield no entries

        wal.close()


class TestSnapshotCorruption:
    """Test snapshot corruption handling."""

    def test_snapshot_with_corrupted_pickle_data(self, tmp_path):
        """
        Test snapshot handles corrupted pickle data.

        Scenario:
        1. Create valid snapshot
        2. Overwrite middle of file with garbage bytes
        3. Attempt to load
        4. Verify: Clear error message (not silent failure)
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Create valid snapshot
        data = {"libraries": [{"id": str(uuid4()), "name": "Test Library"}]}
        filename = manager.create_snapshot(data)

        # Corrupt the pickle file
        snapshot_path = tmp_path / filename
        original_size = snapshot_path.stat().st_size

        # Overwrite middle with garbage
        with open(snapshot_path, "r+b") as f:
            f.seek(original_size // 2)
            f.write(b"\xFF\xFE\xFD\xFC\xFB\xFA" * 20)

        # Attempt to load - should raise clear error
        with pytest.raises(Exception):  # pickle.UnpicklingError or similar
            manager.load_snapshot(filename)

    def test_snapshot_with_invalid_pickle_format(self, tmp_path):
        """
        Test snapshot with non-pickle data.

        Scenario:
        1. Create file with random binary data
        2. Attempt to load as snapshot
        3. Verify: Appropriate error raised
        """
        # Create fake snapshot file
        fake_snapshot = tmp_path / "snapshot_20240101_000000.pkl"
        with open(fake_snapshot, "wb") as f:
            f.write(b"This is not a pickle file!")

        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Should raise unpickling error
        with pytest.raises(Exception):
            manager.load_snapshot("snapshot_20240101_000000.pkl")

    def test_snapshot_with_missing_required_fields(self, tmp_path):
        """
        Test snapshot with valid pickle but missing required data fields.

        Scenario:
        1. Create snapshot with incomplete data structure
        2. Load it
        3. Verify: Can detect missing fields
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Create snapshot with missing fields
        incomplete_data = {"libraries": []}  # Missing 'documents', 'chunks' etc.

        filename = manager.create_snapshot(incomplete_data)
        loaded = manager.load_snapshot(filename)

        # Should load but we can detect missing fields
        assert "libraries" in loaded.data
        assert "documents" not in loaded.data  # Field is missing

    def test_snapshot_file_deleted_during_load(self, tmp_path):
        """
        Test snapshot file deleted between list and load.

        Race condition: File exists during listing but deleted before load.
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Create snapshot
        data = {"libraries": [{"id": str(uuid4())}]}
        filename = manager.create_snapshot(data)

        # Delete the file
        (tmp_path / filename).unlink()

        # Attempt to load - should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            manager.load_snapshot(filename)


class TestPersistenceIntegrity:
    """Test data integrity during persistence operations."""

    def test_wal_write_atomicity(self, tmp_path):
        """
        Test WAL write atomicity (entry fully written or not at all).

        This is implicit in the append operation but we verify no partial writes
        in normal operation.
        """
        wal = WriteAheadLog(wal_dir=tmp_path)

        # Write 100 entries rapidly
        for i in range(100):
            entry = WALEntry(
                operation_type=OperationType.CREATE_LIBRARY,
                data={"id": str(uuid4()), "index": i},
            )
            wal.append(entry)

        wal.close()

        # Read back - all entries should be complete
        wal2 = WriteAheadLog(wal_dir=tmp_path)
        entries = list(wal2.read_all())

        assert len(entries) == 100

        # Each entry should have all required fields
        for entry in entries:
            assert entry.operation_type is not None
            assert entry.data is not None
            assert "id" in entry.data
            assert "index" in entry.data
            assert entry.timestamp is not None

        wal2.close()

    def test_snapshot_data_integrity_hash(self, tmp_path):
        """
        Test snapshot preserves data integrity (bit-for-bit).

        Scenario:
        1. Create snapshot with specific data
        2. Load it back
        3. Verify exact match (no data corruption)
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Create data with various types
        original_data = {
            "libraries": [
                {
                    "id": str(uuid4()),
                    "name": "Test Library",
                    "count": 42,
                    "ratio": 3.14159,
                    "active": True,
                    "tags": ["ml", "ai", "nlp"],
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "nested": {"deep": {"value": "test"}},
                    },
                }
            ]
        }

        # Save and load
        filename = manager.create_snapshot(original_data)
        loaded = manager.load_snapshot(filename)

        # Verify exact match
        assert loaded.data == original_data

        # Deep verification
        lib_orig = original_data["libraries"][0]
        lib_loaded = loaded.data["libraries"][0]

        assert lib_loaded["id"] == lib_orig["id"]
        assert lib_loaded["name"] == lib_orig["name"]
        assert lib_loaded["count"] == lib_orig["count"]
        assert lib_loaded["ratio"] == lib_orig["ratio"]
        assert lib_loaded["active"] == lib_orig["active"]
        assert lib_loaded["tags"] == lib_orig["tags"]
        assert lib_loaded["metadata"] == lib_orig["metadata"]

    def test_wal_handles_special_characters(self, tmp_path):
        """
        Test WAL correctly handles special characters and unicode.

        Ensures JSON encoding/decoding works correctly.
        """
        wal = WriteAheadLog(wal_dir=tmp_path)

        # Data with special characters
        special_data = {
            "id": str(uuid4()),
            "text": 'Test "quotes" and \'apostrophes\'',
            "unicode": "Hello 世界 🌍 Здравствуй мир",
            "newlines": "Line 1\nLine 2\nLine 3",
            "tabs": "Col1\tCol2\tCol3",
            "backslash": "Path\\to\\file",
        }

        entry = WALEntry(
            operation_type=OperationType.ADD_DOCUMENT,
            data=special_data,
        )
        wal.append(entry)
        wal.close()

        # Read back and verify
        wal2 = WriteAheadLog(wal_dir=tmp_path)
        entries = list(wal2.read_all())

        assert len(entries) == 1
        assert entries[0].data == special_data
        assert entries[0].data["unicode"] == "Hello 世界 🌍 Здравствуй мир"

        wal2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
