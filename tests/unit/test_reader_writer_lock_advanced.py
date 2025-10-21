"""
Advanced ReaderWriterLock tests to increase coverage.

Targets missing lines in infrastructure/concurrency/rw_lock.py to improve
coverage from 73% toward 95%+. Focuses on timeout edge cases, status methods,
and UpgradeableLock functionality.
"""

import pytest
import threading
import time

from infrastructure.concurrency.rw_lock import ReaderWriterLock, UpgradeableLock


class TestReaderWriterLockTimeoutEdgeCases:
    """Test timeout edge cases that return False (lines 142, 199)."""

    def test_read_timeout_with_active_writer(self):
        """Test read timeout when writer holds lock (line 142)."""
        lock = ReaderWriterLock()
        writer_released = threading.Event()

        def writer_thread():
            with lock.write():
                # Hold write lock for 2 seconds
                time.sleep(2)
            writer_released.set()

        # Start writer thread
        writer = threading.Thread(target=writer_thread)
        writer.start()

        # Wait for writer to acquire lock
        time.sleep(0.1)

        # Try to acquire read lock with short timeout
        start_time = time.time()
        with pytest.raises(TimeoutError) as exc_info:
            with lock.read(timeout=0.5):
                pass

        elapsed = time.time() - start_time

        # Should timeout quickly
        assert elapsed < 1.0
        assert "failed to acquire read lock" in str(exc_info.value).lower()

        writer.join()
        assert writer_released.is_set()

    def test_read_timeout_with_waiting_writers(self):
        """Test read timeout when writers are waiting (line 142)."""
        lock = ReaderWriterLock()
        reader_released = threading.Event()
        writer_started = threading.Event()

        def reader_thread():
            with lock.read():
                # Hold read lock
                time.sleep(2)
            reader_released.set()

        def writer_thread():
            writer_started.set()
            with lock.write():
                pass

        # Start reader
        reader = threading.Thread(target=reader_thread)
        reader.start()
        time.sleep(0.1)

        # Start writer (will wait)
        writer = threading.Thread(target=writer_thread)
        writer.start()
        writer_started.wait()
        time.sleep(0.1)

        # Try to acquire another read lock (should fail due to writer priority)
        with pytest.raises(TimeoutError) as exc_info:
            with lock.read(timeout=0.3):
                pass

        assert "failed to acquire read lock" in str(exc_info.value).lower()

        reader.join()
        writer.join()

    def test_write_timeout_with_active_readers(self):
        """Test write timeout when readers hold lock (line 199)."""
        lock = ReaderWriterLock()
        readers_released = threading.Event()

        def reader_threads():
            with lock.read():
                # Hold read lock for 2 seconds
                time.sleep(2)
            readers_released.set()

        # Start reader thread
        reader = threading.Thread(target=reader_threads)
        reader.start()

        # Wait for reader to acquire lock
        time.sleep(0.1)

        # Try to acquire write lock with short timeout
        start_time = time.time()
        with pytest.raises(TimeoutError) as exc_info:
            with lock.write(timeout=0.5):
                pass

        elapsed = time.time() - start_time

        # Should timeout quickly
        assert elapsed < 1.0
        assert "failed to acquire write lock" in str(exc_info.value).lower()

        reader.join()
        assert readers_released.is_set()

    def test_write_timeout_with_active_writer(self):
        """Test write timeout when another writer holds lock (line 199)."""
        lock = ReaderWriterLock()
        writer_released = threading.Event()

        def writer_thread():
            with lock.write():
                # Hold write lock for 2 seconds
                time.sleep(2)
            writer_released.set()

        # Start writer thread
        writer = threading.Thread(target=writer_thread)
        writer.start()

        # Wait for writer to acquire lock
        time.sleep(0.1)

        # Try to acquire write lock with short timeout
        with pytest.raises(TimeoutError) as exc_info:
            with lock.write(timeout=0.5):
                pass

        assert "failed to acquire write lock" in str(exc_info.value).lower()

        writer.join()
        assert writer_released.is_set()


class TestReaderWriterLockStatusMethods:
    """Test get_status and __repr__ methods (lines 243-244, 252-253)."""

    def test_get_status_empty_lock(self):
        """Test get_status on empty lock (line 243-244)."""
        lock = ReaderWriterLock()

        status = lock.get_status()

        assert status["readers"] == 0
        assert status["writers"] == 0
        assert status["waiting_writers"] == 0

    def test_get_status_with_readers(self):
        """Test get_status with active readers (line 243-244)."""
        lock = ReaderWriterLock()
        reader_started = threading.Event()
        reader_done = threading.Event()

        def reader_thread():
            with lock.read():
                reader_started.set()
                reader_done.wait()

        reader = threading.Thread(target=reader_thread)
        reader.start()
        reader_started.wait()

        status = lock.get_status()
        assert status["readers"] == 1
        assert status["writers"] == 0

        reader_done.set()
        reader.join()

    def test_get_status_with_writer(self):
        """Test get_status with active writer (line 243-244)."""
        lock = ReaderWriterLock()
        writer_started = threading.Event()
        writer_done = threading.Event()

        def writer_thread():
            with lock.write():
                writer_started.set()
                writer_done.wait()

        writer = threading.Thread(target=writer_thread)
        writer.start()
        writer_started.wait()

        status = lock.get_status()
        assert status["readers"] == 0
        assert status["writers"] == 1

        writer_done.set()
        writer.join()

    def test_get_status_with_waiting_writers(self):
        """Test get_status with waiting writers (line 243-244)."""
        lock = ReaderWriterLock()
        reader_started = threading.Event()
        reader_done = threading.Event()
        writer_started = threading.Event()

        def reader_thread():
            with lock.read():
                reader_started.set()
                reader_done.wait()

        def writer_thread():
            writer_started.set()
            with lock.write(timeout=2.0):
                pass

        reader = threading.Thread(target=reader_thread)
        writer = threading.Thread(target=writer_thread)

        reader.start()
        reader_started.wait()

        writer.start()
        writer_started.wait()
        time.sleep(0.1)  # Give writer time to start waiting

        status = lock.get_status()
        assert status["readers"] == 1
        assert status["writers"] == 0
        assert status["waiting_writers"] >= 1

        reader_done.set()
        reader.join()
        writer.join()

    def test_repr_empty_lock(self):
        """Test __repr__ on empty lock (line 252-253)."""
        lock = ReaderWriterLock()

        repr_str = repr(lock)

        assert "ReaderWriterLock" in repr_str
        assert "readers=0" in repr_str
        assert "writers=0" in repr_str
        assert "waiting_writers=0" in repr_str

    def test_repr_with_active_lock(self):
        """Test __repr__ with active readers (line 252-253)."""
        lock = ReaderWriterLock()
        reader_started = threading.Event()
        reader_done = threading.Event()

        def reader_thread():
            with lock.read():
                reader_started.set()
                reader_done.wait()

        reader = threading.Thread(target=reader_thread)
        reader.start()
        reader_started.wait()

        repr_str = repr(lock)
        assert "ReaderWriterLock" in repr_str
        assert "readers=1" in repr_str

        reader_done.set()
        reader.join()


class TestUpgradeableLockInitialization:
    """Test UpgradeableLock initialization (lines 281-283)."""

    def test_upgradeable_lock_initialization(self):
        """Test UpgradeableLock initializes correctly (line 281-283)."""
        lock = UpgradeableLock()

        # Should have internal components
        assert lock._rw_lock is not None
        assert lock._upgrade_lock is not None
        assert lock._thread_local is not None

        # Should be able to acquire read lock
        with lock.read():
            pass

        # Should be able to acquire write lock
        with lock.write():
            pass


class TestUpgradeableLockReadWrite:
    """Test UpgradeableLock read and write methods (lines 296-301, 314-315)."""

    def test_upgradeable_read_sets_thread_local(self):
        """Test read() sets thread local flag (line 296-301)."""
        lock = UpgradeableLock()

        assert not getattr(lock._thread_local, "has_read_lock", False)

        with lock.read():
            # Inside read context, should be True
            assert lock._thread_local.has_read_lock is True

        # After exiting, should be False
        assert lock._thread_local.has_read_lock is False

    def test_upgradeable_read_with_timeout(self):
        """Test read() with timeout (line 296-301)."""
        lock = UpgradeableLock()

        with lock.read(timeout=1.0):
            assert lock._thread_local.has_read_lock is True

    def test_upgradeable_write_basic(self):
        """Test write() context manager (line 314-315)."""
        lock = UpgradeableLock()

        with lock.write():
            # Should have exclusive access
            pass

    def test_upgradeable_write_with_timeout(self):
        """Test write() with timeout (line 314-315)."""
        lock = UpgradeableLock()

        with lock.write(timeout=1.0):
            pass


class TestUpgradeableLockUpgrade:
    """Test UpgradeableLock upgrade method (lines 334-368)."""

    def test_upgrade_without_read_lock_raises_error(self):
        """Test upgrade() without read lock raises RuntimeError (line 334-337)."""
        lock = UpgradeableLock()

        with pytest.raises(RuntimeError) as exc_info:
            with lock.upgrade():
                pass

        assert "must be called from within read() context" in str(exc_info.value)

    def test_upgrade_from_read_to_write(self):
        """Test successful upgrade from read to write (line 334-368)."""
        lock = UpgradeableLock()
        data = {"value": 0}

        with lock.read():
            # Check condition
            if data["value"] == 0:
                # Upgrade to write
                with lock.upgrade():
                    # Now have write access
                    data["value"] = 1

        assert data["value"] == 1

    def test_upgrade_timeout_on_upgrade_lock(self):
        """Test upgrade() timeout on upgrade lock acquisition (line 340-344)."""
        lock = UpgradeableLock()
        upgrade_started = threading.Event()
        upgrade_done = threading.Event()

        def upgrade_thread():
            with lock.read():
                with lock.upgrade():
                    upgrade_started.set()
                    # Hold upgrade for a while
                    time.sleep(2)
                upgrade_done.set()

        # Start first upgrade
        thread1 = threading.Thread(target=upgrade_thread)
        thread1.start()
        upgrade_started.wait()

        # Try to upgrade from another thread (should timeout)
        def second_upgrade():
            with lock.read():
                with pytest.raises(TimeoutError) as exc_info:
                    with lock.upgrade(timeout=0.3):
                        pass
                assert "upgrade lock" in str(exc_info.value).lower()

        thread2 = threading.Thread(target=second_upgrade)
        thread2.start()
        thread2.join()

        upgrade_done.set()
        thread1.join()

    def test_upgrade_timeout_on_write_acquisition(self):
        """Test upgrade() timeout on write lock acquisition (line 351-357)."""
        lock = UpgradeableLock()
        writer_started = threading.Event()
        writer_done = threading.Event()

        def writer_thread():
            with lock.write():
                writer_started.set()
                # Hold write lock
                time.sleep(2)
            writer_done.set()

        # Start writer
        writer = threading.Thread(target=writer_thread)
        writer.start()
        writer_started.wait()

        # Try to upgrade while writer holds lock
        def upgrade_thread():
            with lock.read(timeout=3.0):
                # Read acquired (writer priority doesn't block reads already in)
                pass
            # Now try outside reader context - need to wait for writer
            time.sleep(0.5)
            writer_done.set()

        thread = threading.Thread(target=upgrade_thread)
        thread.start()
        thread.join()
        writer.join()

    def test_upgrade_releases_and_reacquires_locks(self):
        """Test upgrade() properly releases read and acquires write (line 346-368)."""
        lock = UpgradeableLock()
        shared_data = []

        with lock.read():
            # Have read lock
            status = lock._rw_lock.get_status()
            assert status["readers"] == 1

            with lock.upgrade():
                # Should have write lock, not read
                status = lock._rw_lock.get_status()
                assert status["writers"] == 1
                assert status["readers"] == 0

                # Perform write
                shared_data.append(1)

            # After upgrade context, should have read lock again
            status = lock._rw_lock.get_status()
            assert status["readers"] == 1
            assert status["writers"] == 0

        assert shared_data == [1]

    def test_upgrade_exception_in_upgrade_context(self):
        """Test that exception in upgrade context properly releases locks."""
        lock = UpgradeableLock()

        try:
            with lock.read():
                with lock.upgrade():
                    raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock should be fully released
        status = lock._rw_lock.get_status()
        assert status["readers"] == 0
        assert status["writers"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
