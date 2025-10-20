"""
Unit tests for ReaderWriterLock thread safety.
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from infrastructure.concurrency.rw_lock import ReaderWriterLock


@pytest.mark.unit
@pytest.mark.thread_safety
class TestReaderWriterLockBasics:
    """Basic tests for ReaderWriterLock."""

    def test_single_reader(self):
        """Test that a single reader can acquire the lock."""
        lock = ReaderWriterLock()

        with lock.read():
            pass  # Should not block

    def test_single_writer(self):
        """Test that a single writer can acquire the lock."""
        lock = ReaderWriterLock()

        with lock.write():
            pass  # Should not block

    def test_multiple_concurrent_readers(self):
        """Test that multiple readers can hold the lock simultaneously."""
        lock = ReaderWriterLock()
        results = []

        def reader(reader_id: int):
            with lock.read():
                results.append(f"reader_{reader_id}_start")
                time.sleep(0.01)
                results.append(f"reader_{reader_id}_end")

        threads = []
        for i in range(5):
            t = threading.Thread(target=reader, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All readers should have overlapped
        assert len(results) == 10
        # Should have multiple starts before any ends
        first_end_index = next(i for i, r in enumerate(results) if "end" in r)
        starts_before_first_end = sum(1 for r in results[:first_end_index] if "start" in r)
        assert starts_before_first_end > 1  # Multiple readers entered

    def test_writer_excludes_readers(self):
        """Test that a writer blocks readers."""
        lock = ReaderWriterLock()
        results = []

        def writer():
            with lock.write():
                results.append("writer_start")
                time.sleep(0.05)
                results.append("writer_end")

        def reader():
            time.sleep(0.01)  # Let writer start first
            with lock.read():
                results.append("reader_start")

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # Writer should complete before reader starts
        assert results == ["writer_start", "writer_end", "reader_start"]

    def test_writer_excludes_writer(self):
        """Test that writers are mutually exclusive."""
        lock = ReaderWriterLock()
        results = []

        def writer(writer_id: int):
            with lock.write():
                results.append(f"writer_{writer_id}_start")
                time.sleep(0.01)
                results.append(f"writer_{writer_id}_end")

        threads = []
        for i in range(3):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Writers should be sequential: start, end, start, end, start, end
        assert len(results) == 6
        for i in range(0, 6, 2):
            assert "start" in results[i]
            assert "end" in results[i + 1]


@pytest.mark.unit
@pytest.mark.thread_safety
class TestReaderWriterLockWriterPriority:
    """Tests for writer priority."""

    def test_writer_priority_over_new_readers(self):
        """Test that waiting writers have priority over new readers."""
        lock = ReaderWriterLock()
        results = []

        def initial_reader():
            with lock.read():
                results.append("initial_reader_start")
                time.sleep(0.1)  # Hold read lock
                results.append("initial_reader_end")

        def writer():
            time.sleep(0.02)  # Let initial reader start
            with lock.write():
                results.append("writer_start")
                time.sleep(0.01)
                results.append("writer_end")

        def new_reader():
            time.sleep(0.03)  # Start after writer is waiting
            with lock.read():
                results.append("new_reader_start")

        initial_reader_thread = threading.Thread(target=initial_reader)
        writer_thread = threading.Thread(target=writer)
        new_reader_thread = threading.Thread(target=new_reader)

        initial_reader_thread.start()
        writer_thread.start()
        new_reader_thread.start()

        initial_reader_thread.join()
        writer_thread.join()
        new_reader_thread.join()

        # Writer should go before new reader (writer priority)
        assert results.index("writer_start") < results.index("new_reader_start")


@pytest.mark.unit
@pytest.mark.thread_safety
class TestReaderWriterLockTimeout:
    """Tests for lock timeout functionality."""

    def test_read_timeout_success(self):
        """Test that read lock acquires within timeout."""
        lock = ReaderWriterLock()

        acquired = False
        with lock.read(timeout=1.0):
            acquired = True

        assert acquired

    def test_read_timeout_failure(self):
        """Test that read lock times out when writer holds lock."""
        lock = ReaderWriterLock()

        def writer():
            with lock.write():
                time.sleep(0.5)

        writer_thread = threading.Thread(target=writer)
        writer_thread.start()

        time.sleep(0.05)  # Let writer acquire lock

        # Try to acquire read lock with short timeout
        with pytest.raises(TimeoutError):
            with lock.read(timeout=0.1):
                pass

        writer_thread.join()

    def test_write_timeout_failure(self):
        """Test that write lock times out when another writer holds lock."""
        lock = ReaderWriterLock()

        def first_writer():
            with lock.write():
                time.sleep(0.5)

        writer_thread = threading.Thread(target=first_writer)
        writer_thread.start()

        time.sleep(0.05)  # Let first writer acquire lock

        # Try to acquire write lock with short timeout
        with pytest.raises(TimeoutError):
            with lock.write(timeout=0.1):
                pass

        writer_thread.join()


@pytest.mark.unit
@pytest.mark.thread_safety
class TestReaderWriterLockStress:
    """Stress tests for ReaderWriterLock."""

    def test_high_concurrency_mix(self):
        """Test with high concurrency mix of readers and writers."""
        lock = ReaderWriterLock()
        shared_data = {"value": 0, "reads": 0, "writes": 0}

        def reader(reader_id: int):
            for _ in range(10):
                with lock.read():
                    # Read the value
                    _ = shared_data["value"]
                    shared_data["reads"] += 1
                    time.sleep(0.0001)  # Tiny sleep

        def writer(writer_id: int):
            for _ in range(5):
                with lock.write():
                    # Write the value
                    shared_data["value"] += 1
                    shared_data["writes"] += 1
                    time.sleep(0.0001)  # Tiny sleep

        # 10 readers, 5 writers
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []

            # Submit readers
            for i in range(10):
                futures.append(executor.submit(reader, i))

            # Submit writers
            for i in range(5):
                futures.append(executor.submit(writer, i))

            # Wait for all to complete
            for future in as_completed(futures):
                future.result()

        # Verify integrity
        assert shared_data["writes"] == 25  # 5 writers * 5 iterations
        assert shared_data["value"] == 25  # Should match writes
        assert shared_data["reads"] == 100  # 10 readers * 10 iterations

    def test_no_deadlock_with_many_threads(self):
        """Test that many threads don't deadlock."""
        lock = ReaderWriterLock()
        completed = []

        def worker(worker_id: int):
            for i in range(5):
                if i % 2 == 0:
                    with lock.read():
                        time.sleep(0.0001)
                else:
                    with lock.write():
                        time.sleep(0.0001)
            completed.append(worker_id)

        # 50 threads
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(worker, i) for i in range(50)]

            # Should complete within reasonable time (no deadlock)
            for future in as_completed(futures, timeout=10.0):
                future.result()

        # All threads should complete
        assert len(completed) == 50


@pytest.mark.unit
@pytest.mark.thread_safety
class TestReaderWriterLockEdgeCases:
    """Edge case tests for ReaderWriterLock."""

    def test_reentrant_read_from_same_thread(self):
        """Test that same thread can acquire read lock multiple times."""
        lock = ReaderWriterLock()

        with lock.read():
            with lock.read():
                with lock.read():
                    pass  # Nested read locks should work

    def test_rapid_acquire_release(self):
        """Test rapid acquisition and release."""
        lock = ReaderWriterLock()

        def rapid_reader():
            for _ in range(100):
                with lock.read():
                    pass

        def rapid_writer():
            for _ in range(50):
                with lock.write():
                    pass

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=rapid_reader))
            threads.append(threading.Thread(target=rapid_writer))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
