"""
Structured logging configuration for production deployments.

Outputs logs in JSON format for easy parsing by log aggregators
(ELK stack, Datadog, CloudWatch, etc.)
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.

    Each log entry includes:
    - timestamp (ISO 8601)
    - level (INFO, WARNING, ERROR, etc.)
    - logger name
    - message
    - Additional fields (if provided)
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if provided
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add common fields from record
        if hasattr(record, "library_id"):
            log_data["library_id"] = record.library_id
        if hasattr(record, "document_id"):
            log_data["document_id"] = record.document_id
        if hasattr(record, "operation"):
            log_data["operation"] = record.operation
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        # Add file and line info for debugging
        log_data["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_data)


def configure_structured_logging(
    level: str = "INFO",
    enable_json: bool = False,
) -> None:
    """
    Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: If True, use JSON format. If False, use standard format.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if enable_json:
        formatter = JSONFormatter()
    else:
        # Standard format with more detail
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Configure uvicorn access logs
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.setLevel(logging.INFO)

    # Configure uvicorn error logs
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.setLevel(logging.INFO)


def get_structured_logger(name: str) -> logging.Logger:
    """
    Get a logger with structured logging support.

    Usage:
        logger = get_structured_logger(__name__)
        logger.info(
            "Document added",
            extra={
                "extra_fields": {
                    "library_id": "abc-123",
                    "document_id": "doc-456",
                    "chunks_count": 10
                }
            }
        )
    """
    return logging.getLogger(name)


# Example usage with context
class LogContext:
    """
    Context manager for adding context to log messages.

    Usage:
        with LogContext(library_id="abc-123"):
            logger.info("Processing document")  # Includes library_id
    """

    def __init__(self, **kwargs):
        self.context = kwargs
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)
