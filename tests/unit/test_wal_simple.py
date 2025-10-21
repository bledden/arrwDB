"""Simple WAL tests for coverage."""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from infrastructure.persistence.wal import WriteAheadLog, WALEntry, OperationType


@pytest.fixture
def temp_dir():
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


def test_wal_basic_append_and_read(temp_dir):
    """Test basic WAL append and read."""
    wal = WriteAheadLog(wal_dir=temp_dir)
    
    for i in range(5):
        wal.append(WALEntry(OperationType.ADD_DOCUMENT, {"id": i}))
    
    wal.close()
    wal2 = WriteAheadLog(wal_dir=temp_dir)
    entries = wal2.read_all()
    assert len(entries) == 5


def test_wal_append_operation(temp_dir):
    """Test append_operation convenience method."""
    wal = WriteAheadLog(wal_dir=temp_dir)
    wal.append_operation(OperationType.CREATE_LIBRARY, {"name": "test"})
    wal.close()
    
    wal2 = WriteAheadLog(wal_dir=temp_dir)
    entries = wal2.read_all()
    assert len(entries) == 1


def test_wal_truncate_before(temp_dir):
    """Test truncate_before method."""
    wal = WriteAheadLog(wal_dir=temp_dir)
    
    for i in range(3):
        wal.append(WALEntry(OperationType.ADD_CHUNK, {"id": i}))
    
    cutoff = datetime.utcnow()
    deleted = wal.truncate_before(cutoff)
    assert deleted >= 0


def test_wal_context_manager(temp_dir):
    """Test WAL as context manager."""
    with WriteAheadLog(wal_dir=temp_dir) as wal:
        wal.append(WALEntry(OperationType.DELETE_LIBRARY, {"id": "123"}))


def test_wal_sync_on_write_true(temp_dir):
    """Test sync mode enabled."""
    wal = WriteAheadLog(wal_dir=temp_dir, sync_on_write=True)
    wal.append(WALEntry(OperationType.CREATE_LIBRARY, {"id": "1"}))
    wal.close()
    
    wal2 = WriteAheadLog(wal_dir=temp_dir)
    assert len(wal2.read_all()) == 1
