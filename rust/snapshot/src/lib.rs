use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError, PyFileNotFoundError};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::{PathBuf};
use std::sync::Arc;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

/// A point-in-time snapshot of the database state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotData {
    timestamp: DateTime<Utc>,
    data: String,  // Store as JSON string for bincode compatibility
}

#[pyclass]
pub struct Snapshot {
    inner: SnapshotData,
}

#[pymethods]
impl Snapshot {
    #[new]
    fn new(timestamp_str: String, data_json: String) -> PyResult<Self> {
        let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid timestamp: {}", e)))?
            .with_timezone(&Utc);

        // Validate JSON but store as string
        let _: serde_json::Value = serde_json::from_str(&data_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON data: {}", e)))?;

        Ok(Self {
            inner: SnapshotData { timestamp, data: data_json },
        })
    }

    fn get_timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    fn get_data(&self) -> String {
        self.inner.data.clone()
    }

    fn to_dict(&self) -> PyResult<String> {
        let data_value: serde_json::Value = serde_json::from_str(&self.inner.data)
            .unwrap_or(serde_json::json!({}));

        let dict = serde_json::json!({
            "timestamp": self.inner.timestamp.to_rfc3339(),
            "data": data_value,
        });
        serde_json::to_string(&dict)
            .map_err(|e| PyIOError::new_err(format!("JSON serialization failed: {}", e)))
    }
}

/// Internal state of the SnapshotManager
struct SnapshotManagerState {
    snapshot_dir: PathBuf,
    max_snapshots: usize,
    use_compression: bool,
}

impl SnapshotManagerState {
    fn new(snapshot_dir: PathBuf, max_snapshots: usize, use_compression: bool) -> PyResult<Self> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&snapshot_dir)
            .map_err(|e| PyIOError::new_err(format!("Failed to create snapshot directory: {}", e)))?;

        Ok(Self {
            snapshot_dir,
            max_snapshots,
            use_compression,
        })
    }

    fn create_snapshot(&self, data_json: String, timestamp: Option<DateTime<Utc>>) -> PyResult<String> {
        let timestamp = timestamp.unwrap_or_else(Utc::now);

        // Validate JSON
        let _: serde_json::Value = serde_json::from_str(&data_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON data: {}", e)))?;

        let snapshot = SnapshotData { timestamp, data: data_json };

        // Create filename with timestamp
        let filename = format!(
            "snapshot_{}.snap",
            timestamp.format("%Y%m%d_%H%M%S_%6f")
        );
        let filepath = self.snapshot_dir.join(&filename);

        // Write snapshot
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&filepath)
            .map_err(|e| PyIOError::new_err(format!("Failed to create snapshot file: {}", e)))?;

        if self.use_compression {
            // Use gzip compression
            let mut encoder = GzEncoder::new(BufWriter::new(file), Compression::best());
            bincode::serialize_into(&mut encoder, &snapshot)
                .map_err(|e| PyIOError::new_err(format!("Failed to serialize snapshot: {}", e)))?;
            encoder.finish()
                .map_err(|e| PyIOError::new_err(format!("Failed to compress snapshot: {}", e)))?;
        } else {
            // No compression
            bincode::serialize_into(BufWriter::new(file), &snapshot)
                .map_err(|e| PyIOError::new_err(format!("Failed to serialize snapshot: {}", e)))?;
        }

        // Get file size
        let size_bytes = std::fs::metadata(&filepath)
            .map_err(|e| PyIOError::new_err(format!("Failed to get file size: {}", e)))?
            .len();
        let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

        // Return info as JSON
        let info = serde_json::json!({
            "filename": filename,
            "path": filepath.to_str(),
            "size_mb": size_mb,
            "timestamp": timestamp.to_rfc3339(),
        });

        Ok(serde_json::to_string(&info).unwrap())
    }

    fn load_latest_snapshot(&self) -> PyResult<Option<SnapshotData>> {
        // Find all snapshot files
        let mut snapshot_files: Vec<PathBuf> = std::fs::read_dir(&self.snapshot_dir)
            .map_err(|e| PyIOError::new_err(format!("Failed to read snapshot directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("snapshot_") && n.ends_with(".snap"))
                    .unwrap_or(false)
            })
            .collect();

        if snapshot_files.is_empty() {
            return Ok(None);
        }

        snapshot_files.sort();
        let latest_file = &snapshot_files[snapshot_files.len() - 1];

        self.load_snapshot_from_path(latest_file).map(Some)
    }

    fn load_snapshot(&self, filename: &str) -> PyResult<SnapshotData> {
        let filepath = self.snapshot_dir.join(filename);

        if !filepath.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "Snapshot not found: {}",
                filename
            )));
        }

        self.load_snapshot_from_path(&filepath)
    }

    fn load_snapshot_from_path(&self, filepath: &PathBuf) -> PyResult<SnapshotData> {
        let file = File::open(filepath)
            .map_err(|e| PyIOError::new_err(format!("Failed to open snapshot file: {}", e)))?;

        // Try to detect if compressed (check for gzip magic bytes)
        let mut buf_reader = BufReader::new(file);
        let mut magic = [0u8; 2];
        buf_reader.read_exact(&mut magic)
            .map_err(|e| PyIOError::new_err(format!("Failed to read snapshot: {}", e)))?;

        // Reopen file for actual reading
        let file = File::open(filepath)
            .map_err(|e| PyIOError::new_err(format!("Failed to open snapshot file: {}", e)))?;

        let snapshot = if magic[0] == 0x1f && magic[1] == 0x8b {
            // Gzip compressed
            let decoder = GzDecoder::new(BufReader::new(file));
            bincode::deserialize_from(decoder)
                .map_err(|e| PyIOError::new_err(format!("Failed to deserialize snapshot: {}", e)))?
        } else {
            // Not compressed
            bincode::deserialize_from(BufReader::new(file))
                .map_err(|e| PyIOError::new_err(format!("Failed to deserialize snapshot: {}", e)))?
        };

        Ok(snapshot)
    }

    fn list_snapshots(&self) -> PyResult<Vec<String>> {
        let mut snapshot_files: Vec<PathBuf> = std::fs::read_dir(&self.snapshot_dir)
            .map_err(|e| PyIOError::new_err(format!("Failed to read snapshot directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("snapshot_") && n.ends_with(".snap"))
                    .unwrap_or(false)
            })
            .collect();

        snapshot_files.sort();

        let mut snapshots = Vec::new();
        for filepath in snapshot_files {
            let filename = filepath.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            let size_bytes = std::fs::metadata(&filepath)
                .map(|m| m.len())
                .unwrap_or(0);
            let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

            // Parse timestamp from filename
            // Format: snapshot_YYYYMMDD_HHMMSS_ffffff.snap
            let timestamp_str = filename
                .strip_prefix("snapshot_")
                .and_then(|s| s.strip_suffix(".snap"));

            let info = if let Some(ts_str) = timestamp_str {
                // Try to parse timestamp
                match DateTime::parse_from_str(&format!("{} +0000", ts_str), "%Y%m%d_%H%M%S_%6f %z") {
                    Ok(dt) => serde_json::json!({
                        "filename": filename,
                        "timestamp": dt.to_rfc3339(),
                        "size_mb": (size_mb * 100.0).round() / 100.0,
                    }),
                    Err(_) => serde_json::json!({
                        "filename": filename,
                        "timestamp": null,
                        "size_mb": (size_mb * 100.0).round() / 100.0,
                    }),
                }
            } else {
                serde_json::json!({
                    "filename": filename,
                    "timestamp": null,
                    "size_mb": (size_mb * 100.0).round() / 100.0,
                })
            };

            snapshots.push(serde_json::to_string(&info).unwrap());
        }

        Ok(snapshots)
    }

    fn delete_snapshot(&self, filename: &str) -> PyResult<bool> {
        let filepath = self.snapshot_dir.join(filename);

        if !filepath.exists() {
            return Ok(false);
        }

        std::fs::remove_file(&filepath)
            .map_err(|e| PyIOError::new_err(format!("Failed to delete snapshot: {}", e)))?;

        Ok(true)
    }

    fn cleanup_old_snapshots(&self) -> PyResult<usize> {
        let mut snapshot_files: Vec<PathBuf> = std::fs::read_dir(&self.snapshot_dir)
            .map_err(|e| PyIOError::new_err(format!("Failed to read snapshot directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("snapshot_") && n.ends_with(".snap"))
                    .unwrap_or(false)
            })
            .collect();

        if snapshot_files.len() <= self.max_snapshots {
            return Ok(0);
        }

        snapshot_files.sort();

        // Delete oldest snapshots
        let to_delete = &snapshot_files[..snapshot_files.len() - self.max_snapshots];
        let mut deleted_count = 0;

        for filepath in to_delete {
            if std::fs::remove_file(filepath).is_ok() {
                deleted_count += 1;
            }
        }

        Ok(deleted_count)
    }
}

/// Snapshot manager for creating and loading snapshots
#[pyclass]
pub struct RustSnapshotManager {
    state: Arc<Mutex<SnapshotManagerState>>,
}

#[pymethods]
impl RustSnapshotManager {
    #[new]
    #[pyo3(signature = (snapshot_dir, max_snapshots=5, use_compression=true))]
    fn new(snapshot_dir: String, max_snapshots: usize, use_compression: bool) -> PyResult<Self> {
        let dir_path = PathBuf::from(snapshot_dir);
        let state = SnapshotManagerState::new(dir_path, max_snapshots, use_compression)?;

        Ok(Self {
            state: Arc::new(Mutex::new(state)),
        })
    }

    fn create_snapshot(&self, data_json: String, timestamp_str: Option<String>) -> PyResult<String> {
        let timestamp = if let Some(ts_str) = timestamp_str {
            Some(DateTime::parse_from_rfc3339(&ts_str)
                .map_err(|e| PyValueError::new_err(format!("Invalid timestamp: {}", e)))?
                .with_timezone(&Utc))
        } else {
            None
        };

        let info = self.state.lock().create_snapshot(data_json, timestamp)?;

        // Cleanup old snapshots
        self.state.lock().cleanup_old_snapshots()?;

        Ok(info)
    }

    fn load_latest_snapshot(&self) -> PyResult<Option<String>> {
        let state = self.state.lock();
        match state.load_latest_snapshot()? {
            Some(snapshot) => {
                let data_value: serde_json::Value = serde_json::from_str(&snapshot.data)
                    .unwrap_or(serde_json::json!({}));

                let result = serde_json::json!({
                    "timestamp": snapshot.timestamp.to_rfc3339(),
                    "data": data_value,
                });
                Ok(Some(serde_json::to_string(&result).unwrap()))
            }
            None => Ok(None),
        }
    }

    fn load_snapshot(&self, filename: String) -> PyResult<String> {
        let snapshot = self.state.lock().load_snapshot(&filename)?;

        let data_value: serde_json::Value = serde_json::from_str(&snapshot.data)
            .unwrap_or(serde_json::json!({}));

        let result = serde_json::json!({
            "timestamp": snapshot.timestamp.to_rfc3339(),
            "data": data_value,
        });

        serde_json::to_string(&result)
            .map_err(|e| PyIOError::new_err(format!("JSON serialization failed: {}", e)))
    }

    fn list_snapshots(&self) -> PyResult<Vec<String>> {
        self.state.lock().list_snapshots()
    }

    fn delete_snapshot(&self, filename: String) -> PyResult<bool> {
        self.state.lock().delete_snapshot(&filename)
    }

    fn __repr__(&self) -> String {
        let state = self.state.lock();
        let num_snapshots = std::fs::read_dir(&state.snapshot_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.file_name()
                            .to_str()
                            .map(|n| n.starts_with("snapshot_") && n.ends_with(".snap"))
                            .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0);

        format!(
            "RustSnapshotManager(dir={:?}, snapshots={}, max={})",
            state.snapshot_dir, num_snapshots, state.max_snapshots
        )
    }
}

#[pymodule]
fn rust_snapshot(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustSnapshotManager>()?;
    m.add_class::<Snapshot>()?;
    Ok(())
}
