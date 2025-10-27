"""
Unit tests for the audit logging system.

This module tests the comprehensive audit logging functionality
implemented in app/auth/audit.py for GDPR and SOC 2 compliance.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from app.auth.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    Severity,
)


class TestSeverityEnum:
    """Test the Severity enum."""

    def test_all_severities_defined(self):
        """Test that all expected severity levels are defined."""
        expected_severities = {"info", "warning", "error", "critical"}
        actual_severities = {severity.value for severity in Severity}
        assert actual_severities == expected_severities


class TestAuditEventTypeEnum:
    """Test the AuditEventType enum."""

    def test_required_event_types_defined(self):
        """Test that all expected event types are defined."""
        required_event_types = {
            "auth.success",
            "auth.failure",
            "api_key.created",
            "api_key.rotated",
            "authz.denied",
            "security.rate_limit_exceeded",
        }

        actual_event_types = {event_type.value for event_type in AuditEventType}
        assert required_event_types.issubset(actual_event_types)


class TestAuditEvent:
    """Test the AuditEvent dataclass."""

    def test_create_audit_event(self):
        """Test creating an audit event."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=Severity.INFO,
            tenant_id="test-tenant",
            user_id=None,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            resource=None,
            action=None,
            success=True,
            message="User authenticated successfully",
        )

        assert event.event_type == AuditEventType.AUTH_SUCCESS
        assert event.severity == Severity.INFO
        assert event.tenant_id == "test-tenant"
        assert event.success is True

    def test_audit_event_to_dict(self):
        """Test converting an audit event to a dictionary."""
        timestamp = datetime.utcnow()
        event = AuditEvent(
            timestamp=timestamp,
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=Severity.INFO,
            tenant_id="tenant-123",
            user_id=None,
            ip_address="172.16.0.1",
            user_agent="curl/7.68.0",
            resource="library/test-lib",
            action="create",
            success=True,
            message="Library created successfully",
            metadata={"library_id": "lib-456"},
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "auth.success"
        assert event_dict["severity"] == "info"
        assert event_dict["tenant_id"] == "tenant-123"
        assert event_dict["success"] is True
        assert event_dict["metadata"]["library_id"] == "lib-456"

    def test_audit_event_to_json(self):
        """Test converting an audit event to JSON."""
        timestamp = datetime.utcnow()
        event = AuditEvent(
            timestamp=timestamp,
            event_type=AuditEventType.AUTH_FAILURE,
            severity=Severity.WARNING,
            tenant_id="tenant-789",
            user_id=None,
            ip_address="203.0.113.1",
            user_agent="Python/3.9",
            resource=None,
            action="authenticate",
            success=False,
            message="Authentication failed: invalid credentials",
            metadata={"attempt_count": 3},
        )

        event_json = event.to_json()

        # Should be valid JSON
        parsed = json.loads(event_json)
        assert parsed["event_type"] == "auth.failure"
        assert parsed["severity"] == "warning"
        assert parsed["success"] is False
        assert parsed["metadata"]["attempt_count"] == 3


class TestAuditLogger:
    """Test the AuditLogger class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for audit logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def audit_logger(self, temp_log_dir):
        """Create an AuditLogger instance with temporary log directory."""
        log_file = Path(temp_log_dir) / "audit.log"
        return AuditLogger(log_file=str(log_file))

    def test_create_audit_logger(self, temp_log_dir):
        """Test creating an AuditLogger instance."""
        log_file = Path(temp_log_dir) / "audit.log"
        logger = AuditLogger(log_file=str(log_file))

        assert str(logger.log_file) == str(log_file)
        assert logger.min_severity == Severity.INFO

    def test_log_audit_event(self, audit_logger, temp_log_dir):
        """Test logging an audit event to file."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=Severity.INFO,
            tenant_id="test-tenant",
            user_id=None,
            ip_address="192.168.1.1",
            user_agent="test-agent",
            resource=None,
            action=None,
            success=True,
            message="Test event",
        )

        audit_logger.log(event)

        # Verify event was written to file
        log_file = Path(audit_logger.log_file)
        assert log_file.exists()

        with open(log_file, "r") as f:
            log_content = f.read()
            assert "auth.success" in log_content
            assert "test-tenant" in log_content

    def test_log_multiple_events(self, audit_logger):
        """Test logging multiple audit events."""
        events = [
            AuditEvent(
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.AUTH_SUCCESS,
                severity=Severity.INFO,
                tenant_id="tenant-1",
                user_id=None,
                ip_address="10.0.0.1",
                user_agent="agent1",
                resource=None,
                action=None,
                success=True,
                message="Event 1",
            ),
            AuditEvent(
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.API_KEY_CREATED,
                severity=Severity.INFO,
                tenant_id="tenant-1",
                user_id=None,
                ip_address="10.0.0.1",
                user_agent="agent1",
                resource="tenant:tenant-1",
                action="create_api_key",
                success=True,
                message="Event 2",
            ),
        ]

        for event in events:
            audit_logger.log(event)

        # Verify both events were written
        log_file = Path(audit_logger.log_file)
        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert "auth.success" in lines[0]
            assert "api_key.created" in lines[1]

    def test_log_auth_success(self, audit_logger):
        """Test logging an authentication success event."""
        audit_logger.log_auth_success(
            tenant_id="tenant-123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        log_file = Path(audit_logger.log_file)
        with open(log_file, "r") as f:
            log_content = f.read()
            log_data = json.loads(log_content)

            assert log_data["event_type"] == "auth.success"
            assert log_data["severity"] == "info"
            assert log_data["success"] is True
            assert log_data["tenant_id"] == "tenant-123"

    def test_log_auth_failure(self, audit_logger):
        """Test logging an authentication failure event."""
        audit_logger.log_auth_failure(
            reason="Invalid credentials",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            metadata={"attempt_count": 3},
        )

        log_file = Path(audit_logger.log_file)
        with open(log_file, "r") as f:
            log_content = f.read()
            log_data = json.loads(log_content)

            assert log_data["event_type"] == "auth.failure"
            assert log_data["severity"] == "warning"
            assert log_data["success"] is False
            assert "Invalid credentials" in log_data["message"]

    def test_log_authz_denied(self, audit_logger):
        """Test logging a permission denied event."""
        audit_logger.log_authz_denied(
            tenant_id="tenant-123",
            resource="library/test",
            action="delete",
            required_scopes=["libraries:delete"],
            ip_address="192.168.1.1",
        )

        log_file = Path(audit_logger.log_file)
        with open(log_file, "r") as f:
            log_content = f.read()
            log_data = json.loads(log_content)

            assert log_data["event_type"] == "authz.denied"
            assert log_data["severity"] == "warning"
            assert log_data["success"] is False
            assert log_data["resource"] == "library/test"
            assert log_data["action"] == "delete"

    def test_log_api_key_created(self, audit_logger):
        """Test logging an API key creation event."""
        audit_logger.log_api_key_created(
            tenant_id="tenant-123",
            expires_in_days=90,
        )

        log_file = Path(audit_logger.log_file)
        with open(log_file, "r") as f:
            log_content = f.read()
            log_data = json.loads(log_content)

            assert log_data["event_type"] == "api_key.created"
            assert log_data["severity"] == "info"
            assert log_data["metadata"]["expires_in_days"] == 90

    def test_log_api_key_rotated(self, audit_logger):
        """Test logging an API key rotation event."""
        audit_logger.log_api_key_rotated(
            tenant_id="tenant-123",
            ip_address="192.168.1.1",
        )

        log_file = Path(audit_logger.log_file)
        with open(log_file, "r") as f:
            log_content = f.read()
            log_data = json.loads(log_content)

            assert log_data["event_type"] == "api_key.rotated"
            assert log_data["severity"] == "warning"  # Important security event

    def test_log_rate_limit_exceeded(self, audit_logger):
        """Test logging a rate limit exceeded event."""
        audit_logger.log_rate_limit_exceeded(
            tenant_id="tenant-123",
            ip_address="192.168.1.1",
            endpoint="/api/search",
        )

        log_file = Path(audit_logger.log_file)
        with open(log_file, "r") as f:
            log_content = f.read()
            log_data = json.loads(log_content)

            assert log_data["event_type"] == "security.rate_limit_exceeded"
            assert log_data["severity"] == "warning"
            assert log_data["resource"] == "/api/search"

    def test_json_format_for_siem_integration(self, audit_logger):
        """Test that logs are in JSON format suitable for SIEM ingestion."""
        audit_logger.log_auth_success(
            tenant_id="tenant-123",
            ip_address="10.0.0.1",
            user_agent="test",
        )

        log_file = Path(audit_logger.log_file)
        with open(log_file, "r") as f:
            log_content = f.read()

            # Should be valid JSON
            log_data = json.loads(log_content)

            # Should have all required SIEM fields
            required_fields = [
                "timestamp",
                "event_type",
                "severity",
                "tenant_id",
                "ip_address",
                "success",
                "message",
            ]

            for field in required_fields:
                assert field in log_data, f"Missing required field: {field}"

    def test_severity_filtering(self, temp_log_dir):
        """Test that severity filtering works correctly."""
        log_file = Path(temp_log_dir) / "audit.log"
        logger = AuditLogger(log_file=str(log_file), min_severity=Severity.WARNING)

        # Log INFO event (should be filtered)
        event_info = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=Severity.INFO,
            tenant_id="test",
            user_id=None,
            ip_address="10.0.0.1",
            user_agent="test",
            resource=None,
            action=None,
            success=True,
            message="Info event",
        )
        logger.log(event_info)

        # Log WARNING event (should be logged)
        event_warning = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.AUTH_FAILURE,
            severity=Severity.WARNING,
            tenant_id="test",
            user_id=None,
            ip_address="10.0.0.1",
            user_agent="test",
            resource=None,
            action=None,
            success=False,
            message="Warning event",
        )
        logger.log(event_warning)

        # Check that only WARNING was logged
        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            assert "warning" in lines[0]
            assert "auth.failure" in lines[0]
