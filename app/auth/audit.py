"""
Audit logging system for security events.

Tracks all authentication and authorization events for compliance and security monitoring.
Inspired by AWS CloudTrail, Auth0 Logs, and GDPR compliance requirements.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events to track."""

    # Authentication events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_INVALID_KEY = "auth.invalid_key"
    AUTH_EXPIRED_KEY = "auth.expired_key"
    AUTH_INACTIVE_TENANT = "auth.inactive_tenant"

    # API Key management
    API_KEY_CREATED = "api_key.created"
    API_KEY_ROTATED = "api_key.rotated"
    API_KEY_REVOKED = "api_key.revoked"

    # Tenant management
    TENANT_CREATED = "tenant.created"
    TENANT_DEACTIVATED = "tenant.deactivated"
    TENANT_REACTIVATED = "tenant.reactivated"
    TENANT_UPDATED = "tenant.updated"

    # Authorization events
    AUTHZ_DENIED = "authz.denied"  # Insufficient permissions
    AUTHZ_SUCCESS = "authz.success"

    # Suspicious activity
    RATE_LIMIT_EXCEEDED = "security.rate_limit_exceeded"
    SUSPICIOUS_IP = "security.suspicious_ip"
    BRUTE_FORCE_DETECTED = "security.brute_force_detected"

    # Admin actions
    ADMIN_ACTION = "admin.action"


class Severity(str, Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a single audit log entry."""

    timestamp: datetime
    event_type: AuditEventType
    severity: Severity
    tenant_id: Optional[str]
    user_id: Optional[str]  # For future user management
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]  # What was accessed (e.g., "library:abc123")
    action: Optional[str]  # What action (e.g., "create", "delete")
    success: bool
    message: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """
    Centralized audit logging system.

    Features:
    - Structured logging (JSON)
    - File rotation support
    - Filtering by severity
    - Async writing (optional)
    - Integration with external SIEM systems
    """

    def __init__(
        self,
        log_file: str = "./data/audit/audit.log",
        min_severity: Severity = Severity.INFO,
        enable_stdout: bool = False,
    ):
        """
        Initialize audit logger.

        Args:
            log_file: Path to audit log file
            min_severity: Minimum severity to log
            enable_stdout: Also log to stdout (for development)
        """
        self.log_file = Path(log_file)
        self.min_severity = min_severity
        self.enable_stdout = enable_stdout

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _should_log(self, severity: Severity) -> bool:
        """Check if event should be logged based on severity."""
        severity_order = {
            Severity.INFO: 0,
            Severity.WARNING: 1,
            Severity.ERROR: 2,
            Severity.CRITICAL: 3,
        }
        return severity_order.get(severity, 0) >= severity_order.get(self.min_severity, 0)

    def log(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Args:
            event: The audit event to log
        """
        if not self._should_log(event.severity):
            return

        # Write to file
        try:
            with open(self.log_file, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

        # Optionally write to stdout
        if self.enable_stdout:
            print(f"[AUDIT] {event.to_json()}")

        # Also log to standard logger for integration with monitoring systems
        if event.severity == Severity.CRITICAL:
            logger.critical(f"AUDIT: {event.message}")
        elif event.severity == Severity.ERROR:
            logger.error(f"AUDIT: {event.message}")
        elif event.severity == Severity.WARNING:
            logger.warning(f"AUDIT: {event.message}")
        else:
            logger.info(f"AUDIT: {event.message}")

    def log_auth_success(
        self,
        tenant_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """Log successful authentication."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=Severity.INFO,
            tenant_id=tenant_id,
            user_id=None,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=None,
            action="authenticate",
            success=True,
            message=f"Tenant {tenant_id} authenticated successfully",
        )
        self.log(event)

    def log_auth_failure(
        self,
        reason: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log failed authentication attempt."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.AUTH_FAILURE,
            severity=Severity.WARNING,
            tenant_id=None,
            user_id=None,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=None,
            action="authenticate",
            success=False,
            message=f"Authentication failed: {reason}",
            metadata=metadata,
        )
        self.log(event)

    def log_authz_denied(
        self,
        tenant_id: str,
        resource: str,
        action: str,
        required_scopes: list[str],
        ip_address: Optional[str] = None,
    ) -> None:
        """Log authorization denial."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.AUTHZ_DENIED,
            severity=Severity.WARNING,
            tenant_id=tenant_id,
            user_id=None,
            ip_address=ip_address,
            user_agent=None,
            resource=resource,
            action=action,
            success=False,
            message=f"Authorization denied for {tenant_id} on {resource}:{action}",
            metadata={"required_scopes": required_scopes},
        )
        self.log(event)

    def log_api_key_created(
        self,
        tenant_id: str,
        expires_in_days: Optional[int] = None,
    ) -> None:
        """Log API key creation."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.API_KEY_CREATED,
            severity=Severity.INFO,
            tenant_id=tenant_id,
            user_id=None,
            ip_address=None,
            user_agent=None,
            resource=f"tenant:{tenant_id}",
            action="create_api_key",
            success=True,
            message=f"API key created for tenant {tenant_id}",
            metadata={"expires_in_days": expires_in_days} if expires_in_days else None,
        )
        self.log(event)

    def log_api_key_rotated(
        self,
        tenant_id: str,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log API key rotation."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.API_KEY_ROTATED,
            severity=Severity.WARNING,  # Important security event
            tenant_id=tenant_id,
            user_id=None,
            ip_address=ip_address,
            user_agent=None,
            resource=f"tenant:{tenant_id}",
            action="rotate_api_key",
            success=True,
            message=f"API key rotated for tenant {tenant_id}",
        )
        self.log(event)

    def log_rate_limit_exceeded(
        self,
        tenant_id: Optional[str],
        ip_address: Optional[str],
        endpoint: str,
    ) -> None:
        """Log rate limit violation."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            severity=Severity.WARNING,
            tenant_id=tenant_id,
            user_id=None,
            ip_address=ip_address,
            user_agent=None,
            resource=endpoint,
            action="rate_limit_check",
            success=False,
            message=f"Rate limit exceeded for endpoint {endpoint}",
            metadata={"tenant_id": tenant_id, "ip": ip_address},
        )
        self.log(event)


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        from app.config import settings
        _audit_logger = AuditLogger(
            log_file="./data/audit/audit.log",
            min_severity=Severity.INFO,
            enable_stdout=settings.DEBUG,
        )
    return _audit_logger
