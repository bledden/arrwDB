"""
Comprehensive tests for persistence concurrency and thread safety.

Tests concurrent WAL writes, concurrent snapshot operations, and
concurrent read/write scenarios to ensure thread safety.
"""

import pytest
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from infrastructure.persistence.snapshot import Snapshot, SnapshotManager
from infrastructure.persistence.wal import OperationType, WALEntry, WriteAheadLog


class TestConcurrentWALWrites:
    """Test concurrent WAL write operations."""

    def test_concurrent_wal_writes_from_multiple_threads(self, tmp_path):
        """
        Test concurrent WAL writes from 10 threads.

        Scenario:
        1. Spawn 10 threads
        2. Each thread writes 100 entries
        3. Join all threads
        4. Verify: All 1000 entries present and uncorrupted
        """
        wal = WriteAheadLog(wal_dir=tmp_path)
        num_threads = 10
        entries_per_thread = 100
        results = []
        errors = []

        def writer_thread(thread_id: int):
            """Write entries from a single thread."""
            try:
                thread_entries = []
                for i in range(entries_per_thread):
                    lib_id = str(uuid4())
                    entry = WALEntry(
                        operation_type=OperationType.CREATE_LIBRARY,
                        data={
                            "id": lib_id,
                            "thread": thread_id,
                            "index": i,
                            "name": f"Thread-{thread_id}-Lib-{i}",
                        },
                    )
                    wal.append(entry)
                    thread_entries.append((thread_id, i, lib_id))
                results.extend(thread_entries)
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Spawn threads
        threads = []
        for thread_id in range(num_threads):
            t = threading.Thread(target=writer_thread, args=(thread_id,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        wal.close()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all entries written
        assert len(results) == num_threads * entries_per_thread

        # Read back and verify
        wal2 = WriteAheadLog(wal_dir=tmp_path)
        all_entries = list(wal2.read_all())

        assert len(all_entries) == num_threads * entries_per_thread, (
            f"Expected {num_threads * entries_per_thread} entries, "
            f"got {len(all_entries)}"
        )

        # Verify each entry is valid
        for entry in all_entries:
            assert entry.operation_type == OperationType.CREATE_LIBRARY
            assert "thread" in entry.data
            assert "index" in entry.data
            assert "id" in entry.data
            assert entry.timestamp is not None

        # Verify all thread IDs present
        thread_ids = {entry.data["thread"] for entry in all_entries}
        assert thread_ids == set(range(num_threads))

        wal2.close()

    def test_concurrent_wal_write_no_data_loss(self, tmp_path):
        """
        Test that concurrent WAL writes don't lose data.

        Verifies that thread-safety mechanisms prevent data races.
        """
        wal = WriteAheadLog(wal_dir=tmp_path)
        num_threads = 5
        entries_per_thread = 50
        written_ids = set()
        lock = threading.Lock()

        def writer_thread(thread_id: int):
            """Write entries and track IDs."""
            for i in range(entries_per_thread):
                entry_id = f"thread{thread_id}_entry{i}"
                entry = WALEntry(
                    operation_type=OperationType.ADD_DOCUMENT,
                    data={"id": entry_id, "thread": thread_id},
                )
                wal.append(entry)

                with lock:
                    written_ids.add(entry_id)

        # Execute concurrent writes
        threads = [
            threading.Thread(target=writer_thread, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        wal.close()

        # Read back and verify all IDs present
        wal2 = WriteAheadLog(wal_dir=tmp_path)
        entries = list(wal2.read_all())

        read_ids = {entry.data["id"] for entry in entries}

        # No data loss
        assert len(read_ids) == len(written_ids)
        assert read_ids == written_ids

        wal2.close()

    def test_concurrent_wal_writes_maintain_per_thread_order(self, tmp_path):
        """
        Test that entries from same thread maintain their order.

        While global order may be interleaved, each thread's entries
        should maintain their order relative to each other.
        """
        wal = WriteAheadLog(wal_dir=tmp_path)
        num_threads = 5
        entries_per_thread = 20

        def writer_thread(thread_id: int):
            """Write sequentially numbered entries."""
            for i in range(entries_per_thread):
                entry = WALEntry(
                    operation_type=OperationType.ADD_CHUNK,
                    data={
                        "thread": thread_id,
                        "sequence": i,  # Sequential within thread
                    },
                )
                wal.append(entry)
                # Small delay to increase likelihood of interleaving
                time.sleep(0.001)

        threads = [
            threading.Thread(target=writer_thread, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        wal.close()

        # Read and verify per-thread ordering
        wal2 = WriteAheadLog(wal_dir=tmp_path)
        all_entries = list(wal2.read_all())

        # Group by thread
        thread_sequences = {i: [] for i in range(num_threads)}
        for entry in all_entries:
            thread_id = entry.data["thread"]
            sequence = entry.data["sequence"]
            thread_sequences[thread_id].append(sequence)

        # Verify each thread's sequence is in order
        for thread_id, sequences in thread_sequences.items():
            assert sequences == sorted(sequences), (
                f"Thread {thread_id} entries out of order: {sequences}"
            )
            assert sequences == list(range(entries_per_thread)), (
                f"Thread {thread_id} missing entries: {sequences}"
            )

        wal2.close()


class TestConcurrentSnapshotOperations:
    """Test concurrent snapshot operations."""

    def test_concurrent_snapshot_creation(self, tmp_path):
        """
        Test concurrent snapshot creation from 3 threads.

        Scenario:
        1. Spawn 3 threads trying to create snapshots simultaneously
        2. Verify: All snapshots created successfully
        3. Verify: All snapshots are valid
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)
        num_threads = 3
        created_snapshots = []
        errors = []
        lock = threading.Lock()

        def create_snapshot_thread(thread_id: int):
            """Create a snapshot from a thread."""
            try:
                data = {
                    "libraries": [
                        {
                            "id": str(uuid4()),
                            "thread": thread_id,
                            "name": f"Thread {thread_id} Library",
                        }
                    ]
                }
                filename = manager.create_snapshot(data)
                with lock:
                    created_snapshots.append((thread_id, filename))
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))

        # Spawn threads
        threads = [
            threading.Thread(target=create_snapshot_thread, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all snapshots created
        assert len(created_snapshots) == num_threads

        # Verify each snapshot is valid
        for thread_id, filename in created_snapshots:
            snapshot = manager.load_snapshot(filename)
            assert snapshot is not None
            assert "libraries" in snapshot.data
            assert snapshot.data["libraries"][0]["thread"] == thread_id

    def test_concurrent_snapshot_read(self, tmp_path):
        """
        Test concurrent snapshot reads are thread-safe.

        Scenario:
        1. Create one snapshot
        2. Read it from 10 threads simultaneously
        3. Verify: All reads successful and data consistent
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)

        # Create snapshot
        original_data = {
            "libraries": [
                {"id": str(uuid4()), "name": "Test Library", "count": 100}
            ]
        }
        filename = manager.create_snapshot(original_data)

        # Concurrent reads
        num_readers = 10
        read_results = []
        errors = []
        lock = threading.Lock()

        def reader_thread(thread_id: int):
            """Read snapshot from a thread."""
            try:
                snapshot = manager.load_snapshot(filename)
                with lock:
                    read_results.append((thread_id, snapshot.data))
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))

        threads = [
            threading.Thread(target=reader_thread, args=(i,))
            for i in range(num_readers)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all reads successful
        assert len(read_results) == num_readers

        # Verify all read same data
        for thread_id, data in read_results:
            assert data == original_data


class TestConcurrentReadWrite:
    """Test concurrent read and write operations."""

    def test_concurrent_wal_read_and_write(self, tmp_path):
        """
        Test concurrent WAL read + write operations.

        Scenario:
        1. Thread 1: Continuously writing entries
        2. Thread 2: Continuously reading entries
        3. Run for 2 seconds
        4. Verify: No deadlocks, no corrupted reads
        """
        wal = WriteAheadLog(wal_dir=tmp_path)
        stop_flag = threading.Event()
        write_count = [0]
        read_count = [0]
        errors = []
        lock = threading.Lock()

        def writer_thread():
            """Continuously write entries."""
            try:
                while not stop_flag.is_set():
                    entry = WALEntry(
                        operation_type=OperationType.ADD_DOCUMENT,
                        data={"id": str(uuid4()), "timestamp": time.time()},
                    )
                    wal.append(entry)
                    with lock:
                        write_count[0] += 1
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(("writer", str(e)))

        def reader_thread():
            """Continuously read entries."""
            try:
                while not stop_flag.is_set():
                    # Close and reopen to read latest
                    entries = list(wal.read_all())
                    with lock:
                        read_count[0] = len(entries)
                    time.sleep(0.02)  # Small delay
            except Exception as e:
                errors.append(("reader", str(e)))

        # Start threads
        writer = threading.Thread(target=writer_thread)
        reader = threading.Thread(target=reader_thread)

        writer.start()
        reader.start()

        # Run for 2 seconds
        time.sleep(2)
        stop_flag.set()

        writer.join()
        reader.join()

        wal.close()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify some work was done
        assert write_count[0] > 0, "No writes occurred"
        assert read_count[0] > 0, "No reads occurred"

        # Final verification - all writes persisted
        wal2 = WriteAheadLog(wal_dir=tmp_path)
        final_entries = list(wal2.read_all())
        assert len(final_entries) == write_count[0]

        wal2.close()

    def test_concurrent_snapshot_create_and_read(self, tmp_path):
        """
        Test concurrent snapshot creation and reading.

        Scenario:
        1. Thread 1: Creates snapshots periodically
        2. Thread 2: Reads latest snapshot periodically
        3. Verify: No race conditions, no corruption
        """
        manager = SnapshotManager(snapshot_dir=tmp_path)
        stop_flag = threading.Event()
        snapshots_created = []
        snapshots_read = []
        errors = []
        lock = threading.Lock()

        def creator_thread():
            """Periodically create snapshots."""
            try:
                count = 0
                while not stop_flag.is_set():
                    data = {
                        "libraries": [
                            {"id": str(uuid4()), "count": count}
                        ]
                    }
                    filename = manager.create_snapshot(data)
                    with lock:
                        snapshots_created.append(filename)
                    count += 1
                    time.sleep(0.1)
            except Exception as e:
                errors.append(("creator", str(e)))

        def reader_thread():
            """Periodically read latest snapshot."""
            try:
                while not stop_flag.is_set():
                    try:
                        snapshot = manager.load_latest_snapshot()
                        if snapshot:
                            with lock:
                                snapshots_read.append(snapshot.data)
                    except FileNotFoundError:
                        # Acceptable - no snapshots yet
                        pass
                    time.sleep(0.05)
            except Exception as e:
                errors.append(("reader", str(e)))

        # Start threads
        creator = threading.Thread(target=creator_thread)
        reader = threading.Thread(target=reader_thread)

        creator.start()
        time.sleep(0.05)  # Let creator create first snapshot
        reader.start()

        # Run for 1 second
        time.sleep(1)
        stop_flag.set()

        creator.join()
        reader.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify work was done
        assert len(snapshots_created) > 0, "No snapshots created"
        assert len(snapshots_read) > 0, "No snapshots read"

        # All read snapshots should be valid
        for data in snapshots_read:
            assert "libraries" in data
            assert "count" in data["libraries"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
