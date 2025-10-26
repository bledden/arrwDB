"""
Benchmark comparison: Python WAL vs Rust WAL

This benchmark tests the core WAL operations:
1. Appending operations (with fsync)
2. Reading all entries
3. Truncating old entries
4. File rotation
"""

import time
import json
import tempfile
import shutil
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.persistence.wal import WriteAheadLog, OperationType as PyOperationType
import rust_wal

def generate_operation_data(index):
    """Generate operation data for testing."""
    return {
        "id": f"item_{index}",
        "index": index,
        "data": f"operation_data_{index}" * 10,  # Make it bigger
    }

def benchmark_python(n_operations, temp_dir, sync_on_write=True):
    """Benchmark Python WAL."""
    print(f"\n{'='*60}")
    print("PYTHON WAL")
    print(f"{'='*60}")

    from pathlib import Path
    wal_dir = Path(temp_dir) / "python_wal"
    wal_dir.mkdir(exist_ok=True)

    # Create WAL
    wal = WriteAheadLog(
        wal_dir=wal_dir,
        max_file_size=10 * 1024,  # 10KB for rotation testing
        sync_on_write=sync_on_write
    )

    # Benchmark: Append operations
    start = time.time()
    for i in range(n_operations):
        data = generate_operation_data(i)
        wal.append_operation(PyOperationType.ADD_CHUNK, data)
    append_time = time.time() - start
    print(f"Append {n_operations} operations: {append_time:.4f}s ({n_operations/append_time:.0f} ops/sec)")

    # Benchmark: Read all
    start = time.time()
    entries = wal.read_all()
    read_time = time.time() - start
    print(f"Read all {len(entries)} entries: {read_time:.4f}s ({len(entries)/read_time:.0f} ops/sec)")

    # Benchmark: Truncate
    cutoff_time = datetime.utcnow() - timedelta(seconds=1)
    start = time.time()
    removed = wal.truncate_before(cutoff_time)
    truncate_time = time.time() - start
    print(f"Truncate (removed {removed}): {truncate_time:.4f}s")

    # Close
    wal.close()

    # Count WAL files
    wal_files = list(wal_dir.glob("wal_*.log"))
    print(f"WAL files created: {len(wal_files)}")

    return {
        'append': append_time,
        'read': read_time,
        'truncate': truncate_time,
        'files': len(wal_files),
    }

def benchmark_rust(n_operations, temp_dir, sync_on_write=True):
    """Benchmark Rust WAL."""
    print(f"\n{'='*60}")
    print("RUST WAL")
    print(f"{'='*60}")

    wal_dir = os.path.join(temp_dir, "rust_wal")
    os.makedirs(wal_dir, exist_ok=True)

    # Create WAL
    wal = rust_wal.RustWriteAheadLog(
        wal_dir=wal_dir,
        max_file_size=10 * 1024,  # 10KB for rotation testing
        sync_on_write=sync_on_write
    )

    # Benchmark: Append operations
    start = time.time()
    for i in range(n_operations):
        data = generate_operation_data(i)
        data_json = json.dumps(data)
        wal.append("add_chunk", data_json)
    append_time = time.time() - start
    print(f"Append {n_operations} operations: {append_time:.4f}s ({n_operations/append_time:.0f} ops/sec)")

    # Benchmark: Read all
    start = time.time()
    entries = wal.read_all()
    read_time = time.time() - start
    print(f"Read all {len(entries)} entries: {read_time:.4f}s ({len(entries)/read_time:.0f} ops/sec)")

    # Benchmark: Truncate
    cutoff_time = (datetime.utcnow() - timedelta(seconds=1)).isoformat() + 'Z'
    start = time.time()
    removed = wal.truncate_before(cutoff_time)
    truncate_time = time.time() - start
    print(f"Truncate (removed {removed}): {truncate_time:.4f}s")

    # Close
    wal.close()

    # Count WAL files
    wal_files = [f for f in os.listdir(wal_dir) if f.startswith("wal_") and f.endswith(".log")]
    print(f"WAL files created: {len(wal_files)}")

    return {
        'append': append_time,
        'read': read_time,
        'truncate': truncate_time,
        'files': len(wal_files),
    }

def main():
    print("="*60)
    print("WAL Benchmark: Python vs Rust")
    print("="*60)

    n_operations = 1000

    # Test with fsync enabled (durable mode)
    print(f"\nConfiguration: DURABLE MODE (fsync enabled)")
    print(f"  Operations: {n_operations}")
    print(f"  Max file size: 10KB (to test rotation)")

    temp_dir = tempfile.mkdtemp()

    try:
        # Run benchmarks
        py_results = benchmark_python(n_operations, temp_dir, sync_on_write=True)
        rust_results = benchmark_rust(n_operations, temp_dir, sync_on_write=True)

        # Print comparison
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON (Durable Mode)")
        print(f"{'='*60}")
        print(f"{'Operation':<30} {'Python':>12} {'Rust':>12} {'Speedup':>10}")
        print(f"{'-'*60}")

        ops = [
            ('Append 1K operations', 'append'),
            ('Read all entries', 'read'),
            ('Truncate old entries', 'truncate'),
        ]

        for op_name, key in ops:
            py_time = py_results[key]
            rust_time = rust_results[key]
            speedup = py_time / rust_time if rust_time > 0 else 0
            print(f"{op_name:<30} {py_time:>10.4f}s {rust_time:>10.4f}s {speedup:>9.2f}x")

        total_py = sum(py_results[k] for k in ['append', 'read', 'truncate'])
        total_rust = sum(rust_results[k] for k in ['append', 'read', 'truncate'])
        overall_speedup = total_py / total_rust

        print(f"{'-'*60}")
        print(f"{'TOTAL':<30} {total_py:>10.4f}s {total_rust:>10.4f}s {overall_speedup:>9.2f}x")

        # Now test without fsync (fast mode)
        print(f"\n\nConfiguration: FAST MODE (fsync disabled)")
        print(f"  Operations: {n_operations}")
        print(f"  Max file size: 10KB (to test rotation)")

        py_results_fast = benchmark_python(n_operations, temp_dir, sync_on_write=False)
        rust_results_fast = benchmark_rust(n_operations, temp_dir, sync_on_write=False)

        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON (Fast Mode)")
        print(f"{'='*60}")
        print(f"{'Operation':<30} {'Python':>12} {'Rust':>12} {'Speedup':>10}")
        print(f"{'-'*60}")

        for op_name, key in ops:
            py_time = py_results_fast[key]
            rust_time = rust_results_fast[key]
            speedup = py_time / rust_time if rust_time > 0 else 0
            print(f"{op_name:<30} {py_time:>10.4f}s {rust_time:>10.4f}s {speedup:>9.2f}x")

        total_py_fast = sum(py_results_fast[k] for k in ['append', 'read', 'truncate'])
        total_rust_fast = sum(rust_results_fast[k] for k in ['append', 'read', 'truncate'])
        overall_speedup_fast = total_py_fast / total_rust_fast

        print(f"{'-'*60}")
        print(f"{'TOTAL':<30} {total_py_fast:>10.4f}s {total_rust_fast:>10.4f}s {overall_speedup_fast:>9.2f}x")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Durable mode (fsync): Rust is {overall_speedup:.2f}x faster")
        print(f"Fast mode (no fsync): Rust is {overall_speedup_fast:.2f}x faster")
        print(f"\nWAL files created:")
        print(f"  Python: {py_results['files']} files")
        print(f"  Rust: {rust_results['files']} files")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
