"""
Comprehensive tests for persistence recovery scenarios.

Tests WAL recovery, snapshot recovery, and combined recovery after
simulated crashes and failures.
"""

import pickle
import pytest
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from infrastructure.persistence.snapshot import Snapshot, SnapshotManager
from infrastructure.persistence.wal import OperationType, WALEntry, WriteAheadLog


class TestWALRecovery:
    """Test Write-Ahead Log recovery after simulated crashes."""

    def test_wal_recovery_after_crash(self, tmp_path):
        """
        Test WAL recovery after simulated crash.

        Scenario:
        1. Write 100 operations to WAL
        2. Close WAL without cleanup (simulate crash)
        3. Create new WAL instance
        4. Verify all 100 operations can be read back
        """
        # Phase 1: Write operations
        wal = WriteAheadLog(wal_dir=tmp_path)

        library_ids = []
        for i in range(100):
            lib_id = str(uuid4())
            library_ids.append(lib_id)
            entry = WALEntry(
                operation_type=OperationType.CREATE_LIBRARY,
                data={"id": lib_id, "name": f"Library {i}"},
            )
            wal.append(entry)

        # Phase 2: Simulate crash (close without cleanup)
        wal.close()

        # Phase 3: Recovery - create new WAL instance
        wal2 = WriteAheadLog(wal_dir=tmp_path)

        # Phase 4: Verify all entries can be read
        entries = list(wal2.read_all())

        assert len(entries) == 100, f"Expected 100 entries, got {len(entries)}"

        # Verify each entry
        for i, entry in enumerate(entries):
            assert entry.operation_type == OperationType.CREATE_LIBRARY
            assert entry.data["id"] == library_ids[i]
            assert entry.data["name"] == f"Library {i}"

        wal2.close()

    def test_wal_recovery_preserves_order(self, tmp_path):
        """
        Test that WAL recovery preserves operation order.

        Critical for maintaining data consistency.
        """
        wal = WriteAheadLog(wal_dir=tmp_path)

        # Create operations in specific order
        operations = [
            (OperationType.CREATE_LIBRARY, {"id": "lib1", "name": "First"}),
            (OperationType.ADD_DOCUMENT, {"lib_id": "lib1", "doc_id": "doc1"}),
            (OperationType.ADD_CHUNK, {"doc_id": "doc1", "chunk_id": "chunk1"}),
            (OperationType.DELETE_CHUNK, {"chunk_id": "chunk1"}),
            (OperationType.DELETE_DOCUMENT, {"doc_id": "doc1"}),
            (OperationType.DELETE_LIBRARY, {"id": "lib1"}),
        ]

        for op_type, data in operations:
            wal.append(WALEntry(operation_type=op_type, data=data))

        wal.close()

        # Recovery
        wal2 = WriteAheadLog(wal_dir=tmp_path)
        recovered = list(wal2.read_all())

        # Verify order preserved
        assert len(recovered) == len(operations)
        for i, (expected_type, expected_data) in enumerate(operations):
            assert recovered[i].operation_type == expected_type
            assert recovered[i].data == expected_data

        wal2.close()

    def test_wal_recovery_with_rotation(self, tmp_path):
        """
        Test WAL recovery across multiple rotated files.

        Simulates a long-running system with file rotation.
        """
        wal = WriteAheadLog(wal_dir=tmp_path, max_file_size=1024)  # Small size to force rotation

        # Write enough data to force rotation
        all_entries = []
        for i in range(200):
            entry = WALEntry(
                operation_type=OperationType.ADD_DOCUMENT,
                data={"doc_id": f"doc_{i}", "content": f"Content {i}" * 10},  # Make it bigger
            )
            wal.append(entry)
            all_entries.append(entry)

        wal.close()

        # Verify multiple WAL files were created
        wal_files = list(tmp_path.glob("wal_*.log"))
        assert len(wal_files) > 1, "Expected multiple WAL files due to rotation"

        # Recovery should read all files
        wal2 = WriteAheadLog(wal_dir=tmp_path)
        recovered = list(wal2.read_all())

        assert len(recovered) == 200
        for i in range(200):
            assert recovered[i].data["doc_id"] == f"doc_{i}"

        wal2.close()


class TestSnapshotRecovery:
    """Test snapshot recovery after simulated crashes."""

    def test_snapshot_recovery_after_crash(self, tmp_path):
        """
        Test snapshot recovery after simulated crash.

        Scenario:
        1. Create snapshot with 10 libraries
        2. Don't call cleanup (simulate crash)
        3. Load snapshot
        4. Verify all 10 libraries restored correctly
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Phase 1: Create snapshot data
        libraries = []
        for i in range(10):
            lib = {
                "id": str(uuid4()),
                "name": f"Library {i}",
                "documents": [
                    {"id": str(uuid4()), "title": f"Doc {j}"}
                    for j in range(5)
                ],
            }
            libraries.append(lib)

        snapshot_data = {"libraries": libraries}

        # Phase 2: Create snapshot
        filename = manager.create_snapshot(snapshot_data)

        # Phase 3: Simulate crash (don't cleanup, just load)
        loaded_snapshot = manager.load_snapshot(filename)

        # Phase 4: Verify data integrity
        assert loaded_snapshot is not None
        assert "libraries" in loaded_snapshot.data
        assert len(loaded_snapshot.data["libraries"]) == 10

        for i in range(10):
            original = libraries[i]
            recovered = loaded_snapshot.data["libraries"][i]

            assert recovered["id"] == original["id"]
            assert recovered["name"] == original["name"]
            assert len(recovered["documents"]) == 5

    def test_snapshot_recovery_empty_state(self, tmp_path):
        """
        Test snapshot recovery with empty state.

        Edge case: System started with no data.
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Create empty snapshot
        empty_data = {"libraries": [], "documents": [], "chunks": []}
        filename = manager.create_snapshot(empty_data)

        # Load it back
        loaded = manager.load_snapshot(filename)

        assert loaded is not None
        assert loaded.data["libraries"] == []
        assert loaded.data["documents"] == []
        assert loaded.data["chunks"] == []

    def test_snapshot_preserves_complex_metadata(self, tmp_path):
        """
        Test that snapshots preserve complex nested metadata.
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Complex data structure
        complex_data = {
            "libraries": [
                {
                    "id": str(uuid4()),
                    "name": "Complex Library",
                    "metadata": {
                        "tags": ["ml", "ai", "nlp"],
                        "created_at": datetime.now().isoformat(),
                        "config": {
                            "index_type": "hnsw",
                            "dimension": 1024,
                            "params": {"M": 16, "ef_construction": 200},
                        },
                    },
                    "documents": [
                        {
                            "id": str(uuid4()),
                            "chunks": [
                                {"id": str(uuid4()), "text": "Chunk 1", "metadata": {"page": 1}},
                                {"id": str(uuid4()), "text": "Chunk 2", "metadata": {"page": 2}},
                            ],
                        }
                    ],
                }
            ]
        }

        filename = manager.create_snapshot(complex_data)
        loaded = manager.load_snapshot(filename)

        # Deep comparison
        assert loaded.data == complex_data


class TestCombinedWALAndSnapshotRecovery:
    """Test combined WAL + Snapshot recovery scenarios."""

    def test_combined_recovery_wal_after_snapshot(self, tmp_path):
        """
        Test recovery with snapshot + subsequent WAL entries.

        Scenario:
        1. Create snapshot at operation 100
        2. Add 50 more WAL entries
        3. Crash
        4. Recovery should load snapshot + replay 50 WAL entries
        5. Verify all 150 operations accounted for
        """
        # Setup
        snapshot_dir = tmp_path / "snapshots"
        wal_dir = tmp_path / "wal"
        snapshot_dir.mkdir()
        wal_dir.mkdir()

        manager = SnapshotManager(snapshot_dir=snapshot_dir)
        wal = WriteAheadLog(wal_dir=wal_dir)

        # Phase 1: Write 100 operations and create snapshot
        libraries = []
        for i in range(100):
            lib_id = str(uuid4())
            libraries.append({"id": lib_id, "name": f"Library {i}"})
            wal.append(
                WALEntry(
                    operation_type=OperationType.CREATE_LIBRARY,
                    data={"id": lib_id, "name": f"Library {i}"},
                )
            )

        # Create snapshot of first 100
        snapshot_data = {"libraries": libraries}
        snapshot_filename = manager.create_snapshot(snapshot_data)
        snapshot_time = manager.load_snapshot(snapshot_filename).timestamp

        # Phase 2: Add 50 more WAL entries AFTER snapshot
        additional_libs = []
        for i in range(100, 150):
            lib_id = str(uuid4())
            additional_libs.append({"id": lib_id, "name": f"Library {i}"})
            wal.append(
                WALEntry(
                    operation_type=OperationType.CREATE_LIBRARY,
                    data={"id": lib_id, "name": f"Library {i}"},
                )
            )

        wal.close()

        # Phase 3: Simulate recovery
        # Load latest snapshot
        latest_snapshot = manager.load_latest_snapshot()
        assert latest_snapshot is not None

        # Recover snapshot data
        recovered_libraries = latest_snapshot.data["libraries"].copy()
        assert len(recovered_libraries) == 100

        # Replay WAL entries after snapshot
        wal2 = WriteAheadLog(wal_dir=wal_dir)
        wal_entries = []

        for entry in wal2.read_all():
            # Only replay entries after snapshot time
            if entry.timestamp > snapshot_time:
                wal_entries.append(entry)
                if entry.operation_type == OperationType.CREATE_LIBRARY:
                    recovered_libraries.append(entry.data)

        # Phase 4: Verification
        assert len(wal_entries) == 50, f"Expected 50 WAL entries after snapshot, got {len(wal_entries)}"
        assert len(recovered_libraries) == 150, f"Expected 150 total libraries, got {len(recovered_libraries)}"

        # Verify all libraries present
        recovered_ids = {lib["id"] for lib in recovered_libraries}
        expected_ids = {lib["id"] for lib in libraries} | {lib["id"] for lib in additional_libs}
        assert recovered_ids == expected_ids

        wal2.close()

    def test_recovery_with_multiple_snapshots(self, tmp_path):
        """
        Test recovery chooses the latest snapshot.

        Scenario:
        1. Create snapshot 1 with 50 libraries
        2. Add 25 more, create snapshot 2 (75 total)
        3. Add 25 more WAL entries (100 total)
        4. Recovery should use snapshot 2 + 25 WAL entries
        """
        snapshot_dir = tmp_path / "snapshots"
        wal_dir = tmp_path / "wal"
        snapshot_dir.mkdir()
        wal_dir.mkdir()

        manager = SnapshotManager(snapshot_dir=snapshot_dir)
        wal = WriteAheadLog(wal_dir=wal_dir)

        # Create first snapshot (50 libraries)
        libs_snapshot1 = [{"id": str(uuid4()), "name": f"Lib {i}"} for i in range(50)]
        manager.create_snapshot({"libraries": libs_snapshot1})

        # Add 25 more and create second snapshot (75 total)
        libs_snapshot2 = libs_snapshot1 + [{"id": str(uuid4()), "name": f"Lib {i}"} for i in range(50, 75)]
        snapshot2_filename = manager.create_snapshot({"libraries": libs_snapshot2})
        snapshot2_time = manager.load_snapshot(snapshot2_filename).timestamp

        # Add 25 more WAL entries after snapshot 2
        for i in range(75, 100):
            wal.append(
                WALEntry(
                    operation_type=OperationType.CREATE_LIBRARY,
                    data={"id": str(uuid4()), "name": f"Lib {i}"},
                )
            )

        wal.close()

        # Recovery should use latest snapshot (snapshot 2)
        latest = manager.load_latest_snapshot()
        assert len(latest.data["libraries"]) == 75

        # Plus 25 WAL entries
        wal2 = WriteAheadLog(wal_dir=wal_dir)
        wal_after_snapshot = [e for e in wal2.read_all() if e.timestamp > snapshot2_time]
        assert len(wal_after_snapshot) == 25

        wal2.close()

    def test_recovery_handles_wal_gaps(self, tmp_path):
        """
        Test recovery when WAL has been truncated.

        Scenario:
        1. Create snapshot
        2. Add WAL entries
        3. Truncate WAL (simulate cleanup)
        4. Add more WAL entries
        5. Recovery should work correctly
        """
        snapshot_dir = tmp_path / "snapshots"
        wal_dir = tmp_path / "wal"
        snapshot_dir.mkdir()
        wal_dir.mkdir()

        manager = SnapshotManager(snapshot_dir=snapshot_dir)
        wal = WriteAheadLog(wal_dir=wal_dir)

        # Create initial snapshot
        initial_data = {"libraries": [{"id": str(uuid4()), "name": "Initial"}]}
        snapshot_filename = manager.create_snapshot(initial_data)
        snapshot_time = manager.load_snapshot(snapshot_filename).timestamp

        # Add entries and truncate
        for i in range(10):
            wal.append(
                WALEntry(
                    operation_type=OperationType.ADD_DOCUMENT,
                    data={"id": str(uuid4())},
                )
            )

        # Truncate old entries (before snapshot)
        removed = wal.truncate_before(snapshot_time)

        # Add new entries after truncation
        for i in range(5):
            wal.append(
                WALEntry(
                    operation_type=OperationType.ADD_CHUNK,
                    data={"id": str(uuid4())},
                )
            )

        wal.close()

        # Recovery
        latest_snapshot = manager.load_latest_snapshot()
        wal2 = WriteAheadLog(wal_dir=wal_dir)

        # Should be able to read remaining entries
        remaining = list(wal2.read_all())

        # All ADD_CHUNK entries should be present
        chunk_entries = [e for e in remaining if e.operation_type == OperationType.ADD_CHUNK]
        assert len(chunk_entries) == 5

        wal2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
