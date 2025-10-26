"""
Reader-Writer Lock with writer priority.

This module provides a reader-writer lock that allows multiple concurrent
readers but exclusive access for writers. Writers have priority to prevent
writer starvation.
"""

import threading
from contextlib import contextmanager
from typing import Optional


class ReaderWriterLock:
    """
    Reader-Writer lock with writer priority.

    This lock allows:
    - Multiple readers can hold the lock simultaneously
    - Only one writer can hold the lock at a time
    - Writers have priority over readers to prevent starvation

    When a writer is waiting, new readers will block until the writer
    completes, ensuring writers don't starve.

    Usage:
        lock = ReaderWriterLock()

        # For read operations
        with lock.read():
            # Read data

        # For write operations
        with lock.write():
            # Modify data
    """

    def __init__(self):
        """Initialize the reader-writer lock."""
        # Core lock protecting the state
        self._lock = threading.Lock()

        # Condition variables for coordination
        self._read_ready = threading.Condition(self._lock)
        self._write_ready = threading.Condition(self._lock)

        # State counters
        self._readers = 0  # Number of active readers
        self._writers = 0  # Number of active writers (0 or 1)
        self._waiting_writers = 0  # Number of writers waiting

    @contextmanager
    def read(self, timeout: Optional[float] = None):
        """
        Context manager for acquiring read lock.

        Args:
            timeout: Optional timeout in seconds. If specified and lock
                cannot be acquired within timeout, raises TimeoutError.

        Yields:
            None

        Raises:
            TimeoutError: If timeout is specified and exceeded.

        Example:
            with lock.read():
                # Perform read operations
                data = read_from_storage()
        """
        acquired = self._acquire_read(timeout)
        if not acquired:
            raise TimeoutError(
                f"Failed to acquire read lock within {timeout} seconds"
            )

        try:
            yield
        finally:
            self._release_read()

    @contextmanager
    def write(self, timeout: Optional[float] = None):
        """
        Context manager for acquiring write lock.

        Args:
            timeout: Optional timeout in seconds. If specified and lock
                cannot be acquired within timeout, raises TimeoutError.

        Yields:
            None

        Raises:
            TimeoutError: If timeout is specified and exceeded.

        Example:
            with lock.write():
                # Perform write operations
                write_to_storage(data)
        """
        acquired = self._acquire_write(timeout)
        if not acquired:
            raise TimeoutError(
                f"Failed to acquire write lock within {timeout} seconds"
            )

        try:
            yield
        finally:
            self._release_write()

    def _acquire_read(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the read lock.

        Blocks if there are active writers or waiting writers (writer priority).

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            True if lock was acquired, False if timeout occurred.
        """
        self._read_ready.acquire()
        try:
            # Wait while there are active writers or waiting writers
            # (writer priority: don't let new readers in if writers are waiting)
            end_time = None
            if timeout is not None:
                import time

                end_time = time.time() + timeout

            while self._writers > 0 or self._waiting_writers > 0:
                if timeout is not None:
                    import time

                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return False
                    if not self._read_ready.wait(remaining):
                        return False
                else:
                    self._read_ready.wait()

            # Acquired: increment reader count
            self._readers += 1
            return True

        finally:
            self._read_ready.release()

    def _release_read(self) -> None:
        """
        Release the read lock.

        Notifies waiting writers if this was the last reader.
        """
        with self._lock:
            self._readers -= 1

            # If no more readers, wake up waiting writers
            if self._readers == 0:
                self._write_ready.notify()

    def _acquire_write(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the write lock.

        Blocks if there are active readers or active writers.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            True if lock was acquired, False if timeout occurred.
        """
        self._write_ready.acquire()
        try:
            # Indicate we're waiting for write
            self._waiting_writers += 1

            end_time = None
            if timeout is not None:
                import time

                end_time = time.time() + timeout

            try:
                # Wait while there are active readers or writers
                while self._readers > 0 or self._writers > 0:
                    if timeout is not None:
                        import time

                        remaining = end_time - time.time()
                        if remaining <= 0:
                            return False
                        if not self._write_ready.wait(remaining):
                            return False
                    else:
                        self._write_ready.wait()

                # Acquired: mark as active writer
                self._writers = 1
                return True

            finally:
                # No longer waiting
                self._waiting_writers -= 1

        finally:
            self._write_ready.release()

    def _release_write(self) -> None:
        """
        Release the write lock.

        Wakes up all waiting readers and one waiting writer.
        """
        with self._lock:
            self._writers = 0

            # Wake up waiting writers first (writer priority)
            self._write_ready.notify()

            # Then wake up all waiting readers
            self._read_ready.notify_all()

    def get_status(self) -> dict:
        """
        Get the current status of the lock.

        Returns:
            Dictionary with:
            - readers: Number of active readers
            - writers: Number of active writers (0 or 1)
            - waiting_writers: Number of writers waiting for lock

        Note: This is a snapshot and may be outdated immediately after return.
        """
        with self._lock:
            return {
                "readers": self._readers,
                "writers": self._writers,
                "waiting_writers": self._waiting_writers,
            }

    def __repr__(self) -> str:
        """String representation showing current state."""
        status = self.get_status()
        return (
            f"ReaderWriterLock(readers={status['readers']}, "
            f"writers={status['writers']}, "
            f"waiting_writers={status['waiting_writers']})"
        )


class UpgradeableLock:
    """
    Lock that allows upgrading from read to write access.

    This is useful for read-modify-write operations where you want to:
    1. Acquire read lock and check a condition
    2. Upgrade to write lock if modification is needed
    3. Perform modification

    Usage:
        lock = UpgradeableLock()

        with lock.read():
            if needs_update():
                with lock.upgrade():
                    # Now have write access
                    perform_update()
    """

    def __init__(self):
        """Initialize the upgradeable lock."""
        self._rw_lock = ReaderWriterLock()
        self._upgrade_lock = threading.Lock()
        self._thread_local = threading.local()

    @contextmanager
    def read(self, timeout: Optional[float] = None):
        """
        Acquire read lock.

        Args:
            timeout: Optional timeout in seconds.

        Yields:
            None
        """
        with self._rw_lock.read(timeout=timeout):
            self._thread_local.has_read_lock = True
            try:
                yield
            finally:
                self._thread_local.has_read_lock = False

    @contextmanager
    def write(self, timeout: Optional[float] = None):
        """
        Acquire write lock directly.

        Args:
            timeout: Optional timeout in seconds.

        Yields:
            None
        """
        with self._rw_lock.write(timeout=timeout):
            yield

    @contextmanager
    def upgrade(self, timeout: Optional[float] = None):
        """
        Upgrade from read to write lock.

        Must be called from within a read() context.

        Args:
            timeout: Optional timeout in seconds.

        Yields:
            None

        Raises:
            RuntimeError: If called without holding read lock.
            TimeoutError: If timeout is exceeded.
        """
        if not getattr(self._thread_local, "has_read_lock", False):
            raise RuntimeError(
                "upgrade() must be called from within read() context"
            )

        # Only one thread can upgrade at a time
        acquired = self._upgrade_lock.acquire(timeout=timeout or -1)
        if not acquired:
            raise TimeoutError(
                f"Failed to acquire upgrade lock within {timeout} seconds"
            )

        try:
            # Release read lock (we still hold it through the context manager)
            self._rw_lock._release_read()

            # Acquire write lock
            write_acquired = self._rw_lock._acquire_write(timeout)
            if not write_acquired:
                # Re-acquire read lock before raising
                self._rw_lock._acquire_read(timeout)
                raise TimeoutError(
                    f"Failed to upgrade to write lock within {timeout} seconds"
                )

            try:
                yield
            finally:
                # Release write lock
                self._rw_lock._release_write()

        finally:
            # Re-acquire read lock
            self._rw_lock._acquire_read(timeout)
            self._upgrade_lock.release()
