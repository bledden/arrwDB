use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Operation types that can be logged
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub enum OperationType {
    CreateLibrary,
    DeleteLibrary,
    AddDocument,
    DeleteDocument,
    AddChunk,
    DeleteChunk,
}

#[pymethods]
impl OperationType {
    #[new]
    fn new(op_type: &str) -> PyResult<Self> {
        match op_type {
            "create_library" => Ok(OperationType::CreateLibrary),
            "delete_library" => Ok(OperationType::DeleteLibrary),
            "add_document" => Ok(OperationType::AddDocument),
            "delete_document" => Ok(OperationType::DeleteDocument),
            "add_chunk" => Ok(OperationType::AddChunk),
            "delete_chunk" => Ok(OperationType::DeleteChunk),
            _ => Err(PyValueError::new_err(format!("Unknown operation type: {}", op_type))),
        }
    }

    fn __str__(&self) -> &str {
        match self {
            OperationType::CreateLibrary => "create_library",
            OperationType::DeleteLibrary => "delete_library",
            OperationType::AddDocument => "add_document",
            OperationType::DeleteDocument => "delete_document",
            OperationType::AddChunk => "add_chunk",
            OperationType::DeleteChunk => "delete_chunk",
        }
    }
}

/// A single entry in the Write-Ahead Log
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WALEntryData {
    operation_type: String,
    data: serde_json::Value,
    timestamp: DateTime<Utc>,
}

#[pyclass]
pub struct WALEntry {
    inner: WALEntryData,
}

#[pymethods]
impl WALEntry {
    #[new]
    fn new(operation_type: String, data: String) -> PyResult<Self> {
        let data_json: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON data: {}", e)))?;

        Ok(Self {
            inner: WALEntryData {
                operation_type,
                data: data_json,
                timestamp: Utc::now(),
            },
        })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyIOError::new_err(format!("JSON serialization failed: {}", e)))
    }

    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let inner: WALEntryData = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(format!("JSON deserialization failed: {}", e)))?;

        Ok(Self { inner })
    }

    fn get_operation_type(&self) -> String {
        self.inner.operation_type.clone()
    }

    fn get_data(&self) -> String {
        self.inner.data.to_string()
    }

    fn get_timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
}

/// Internal state of the WAL
struct WALState {
    wal_dir: PathBuf,
    max_file_size: usize,
    sync_on_write: bool,
    current_file: Option<PathBuf>,
    current_file_handle: Option<File>,
    current_file_size: usize,
    sequence_number: u64,
}

impl WALState {
    fn new(wal_dir: PathBuf, max_file_size: usize, sync_on_write: bool) -> PyResult<Self> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&wal_dir)
            .map_err(|e| PyIOError::new_err(format!("Failed to create WAL directory: {}", e)))?;

        let mut state = Self {
            wal_dir,
            max_file_size,
            sync_on_write,
            current_file: None,
            current_file_handle: None,
            current_file_size: 0,
            sequence_number: 0,
        };

        state.initialize()?;
        Ok(state)
    }

    fn initialize(&mut self) -> PyResult<()> {
        // Find existing WAL files
        let mut wal_files: Vec<PathBuf> = std::fs::read_dir(&self.wal_dir)
            .map_err(|e| PyIOError::new_err(format!("Failed to read WAL directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("wal_") && n.ends_with(".log"))
                    .unwrap_or(false)
            })
            .collect();

        wal_files.sort();

        if let Some(latest_file) = wal_files.last() {
            // Parse sequence number from filename
            let filename = latest_file.file_stem().and_then(|s| s.to_str()).unwrap();
            let parts: Vec<&str> = filename.split('_').collect();
            if parts.len() >= 2 {
                self.sequence_number = parts[1].parse().unwrap_or(0);
            }

            // Get file size
            self.current_file_size = std::fs::metadata(latest_file)
                .map_err(|e| PyIOError::new_err(format!("Failed to get file size: {}", e)))?
                .len() as usize;

            // Open in append mode
            let file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(latest_file)
                .map_err(|e| PyIOError::new_err(format!("Failed to open WAL file: {}", e)))?;

            self.current_file = Some(latest_file.clone());
            self.current_file_handle = Some(file);
        } else {
            // Create first WAL file
            self.rotate_file()?;
        }

        Ok(())
    }

    fn rotate_file(&mut self) -> PyResult<()> {
        // Close current file if open
        if let Some(handle) = self.current_file_handle.take() {
            // File will be closed automatically when dropped
            drop(handle);
        }

        // Increment sequence number
        self.sequence_number += 1;

        // Create new filename
        let filename = format!("wal_{:08}.log", self.sequence_number);
        let file_path = self.wal_dir.join(filename);

        // Open new file
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to create WAL file: {}", e)))?;

        self.current_file = Some(file_path);
        self.current_file_handle = Some(file);
        self.current_file_size = 0;

        Ok(())
    }

    fn append(&mut self, entry: &WALEntryData) -> PyResult<()> {
        // Check if we need to rotate
        if self.current_file_size >= self.max_file_size {
            self.rotate_file()?;
        }

        // Serialize entry
        let json_str = serde_json::to_string(entry)
            .map_err(|e| PyIOError::new_err(format!("JSON serialization failed: {}", e)))?;
        let line = format!("{}\n", json_str);

        // Write to file
        if let Some(handle) = &mut self.current_file_handle {
            handle.write_all(line.as_bytes())
                .map_err(|e| PyIOError::new_err(format!("Failed to write to WAL: {}", e)))?;

            // Sync to disk if configured
            if self.sync_on_write {
                handle.sync_all()
                    .map_err(|e| PyIOError::new_err(format!("Failed to sync WAL: {}", e)))?;
            }

            self.current_file_size += line.len();
            Ok(())
        } else {
            Err(PyIOError::new_err("No current WAL file open"))
        }
    }

    fn read_all(&self) -> PyResult<Vec<WALEntryData>> {
        let mut entries = Vec::new();

        // Find all WAL files
        let mut wal_files: Vec<PathBuf> = std::fs::read_dir(&self.wal_dir)
            .map_err(|e| PyIOError::new_err(format!("Failed to read WAL directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("wal_") && n.ends_with(".log"))
                    .unwrap_or(false)
            })
            .collect();

        wal_files.sort();

        // Read each file
        for wal_file in wal_files {
            let file = File::open(&wal_file)
                .map_err(|e| PyIOError::new_err(format!("Failed to open WAL file: {}", e)))?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line.map_err(|e| PyIOError::new_err(format!("Failed to read line: {}", e)))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                match serde_json::from_str::<WALEntryData>(line) {
                    Ok(entry) => entries.push(entry),
                    Err(_) => {
                        // Skip corrupted entries but log them
                        continue;
                    }
                }
            }
        }

        Ok(entries)
    }

    fn truncate_before(&mut self, timestamp: DateTime<Utc>) -> PyResult<usize> {
        // Close current file
        if let Some(handle) = self.current_file_handle.take() {
            drop(handle);
        }

        let mut removed_count = 0;

        // Find all WAL files
        let mut wal_files: Vec<PathBuf> = std::fs::read_dir(&self.wal_dir)
            .map_err(|e| PyIOError::new_err(format!("Failed to read WAL directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("wal_") && n.ends_with(".log"))
                    .unwrap_or(false)
            })
            .collect();

        wal_files.sort();

        // Process each file
        for wal_file in wal_files {
            let mut keep_entries = Vec::new();

            // Read entries
            let file = File::open(&wal_file)
                .map_err(|e| PyIOError::new_err(format!("Failed to open WAL file: {}", e)))?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line.map_err(|e| PyIOError::new_err(format!("Failed to read line: {}", e)))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                match serde_json::from_str::<WALEntryData>(line) {
                    Ok(entry) => {
                        if entry.timestamp >= timestamp {
                            keep_entries.push(entry);
                        } else {
                            removed_count += 1;
                        }
                    }
                    Err(_) => {
                        // Keep corrupted entries to be safe
                        continue;
                    }
                }
            }

            // Rewrite file or delete if empty
            if !keep_entries.is_empty() {
                let mut file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&wal_file)
                    .map_err(|e| PyIOError::new_err(format!("Failed to open WAL file for writing: {}", e)))?;

                for entry in keep_entries {
                    let json_str = serde_json::to_string(&entry)
                        .map_err(|e| PyIOError::new_err(format!("JSON serialization failed: {}", e)))?;
                    let line = format!("{}\n", json_str);
                    file.write_all(line.as_bytes())
                        .map_err(|e| PyIOError::new_err(format!("Failed to write to WAL: {}", e)))?;
                }

                file.sync_all()
                    .map_err(|e| PyIOError::new_err(format!("Failed to sync WAL: {}", e)))?;
            } else {
                // Delete empty file
                std::fs::remove_file(&wal_file)
                    .map_err(|e| PyIOError::new_err(format!("Failed to delete WAL file: {}", e)))?;
            }
        }

        // Re-initialize
        self.initialize()?;

        Ok(removed_count)
    }

    fn close(&mut self) -> PyResult<()> {
        if let Some(handle) = self.current_file_handle.take() {
            // Ensure all data is synced before closing
            if self.sync_on_write {
                handle.sync_all()
                    .map_err(|e| PyIOError::new_err(format!("Failed to sync WAL: {}", e)))?;
            }
            drop(handle);
        }
        Ok(())
    }
}

/// Write-Ahead Log for operation durability
#[pyclass]
pub struct RustWriteAheadLog {
    state: Arc<Mutex<WALState>>,
}

#[pymethods]
impl RustWriteAheadLog {
    #[new]
    #[pyo3(signature = (wal_dir, max_file_size=100*1024*1024, sync_on_write=true))]
    fn new(wal_dir: String, max_file_size: usize, sync_on_write: bool) -> PyResult<Self> {
        let wal_path = PathBuf::from(wal_dir);
        let state = WALState::new(wal_path, max_file_size, sync_on_write)?;

        Ok(Self {
            state: Arc::new(Mutex::new(state)),
        })
    }

    fn append(&self, operation_type: String, data: String) -> PyResult<()> {
        let data_json: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON data: {}", e)))?;

        let entry = WALEntryData {
            operation_type,
            data: data_json,
            timestamp: Utc::now(),
        };

        self.state.lock().append(&entry)
    }

    fn read_all(&self) -> PyResult<Vec<String>> {
        let entries = self.state.lock().read_all()?;

        entries.iter()
            .map(|entry| {
                serde_json::to_string(entry)
                    .map_err(|e| PyIOError::new_err(format!("JSON serialization failed: {}", e)))
            })
            .collect()
    }

    fn truncate_before(&self, timestamp_str: String) -> PyResult<usize> {
        let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid timestamp: {}", e)))?
            .with_timezone(&Utc);

        self.state.lock().truncate_before(timestamp)
    }

    fn close(&self) -> PyResult<()> {
        self.state.lock().close()
    }

    fn __repr__(&self) -> String {
        let state = self.state.lock();
        format!(
            "RustWriteAheadLog(dir={:?}, current_file={:?}, size={})",
            state.wal_dir,
            state.current_file,
            state.current_file_size
        )
    }
}

#[pymodule]
fn rust_wal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustWriteAheadLog>()?;
    m.add_class::<WALEntry>()?;
    m.add_class::<OperationType>()?;
    Ok(())
}
