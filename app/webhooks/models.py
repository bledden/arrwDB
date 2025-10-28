"""
Webhook data models for arrwDB.

Provides Pydantic models for webhooks, events, and delivery tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class WebhookEventType(str, Enum):
    """Supported webhook event types."""

    # Document events
    DOCUMENT_CREATED = "document.created"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"

    # Library events
    LIBRARY_CREATED = "library.created"
    LIBRARY_UPDATED = "library.updated"
    LIBRARY_DELETED = "library.deleted"

    # Job events (for compression/training workflows)
    JOB_STARTED = "job.started"
    JOB_PROGRESS = "job.progress"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"

    # Index events
    INDEX_REBUILT = "index.rebuilt"
    INDEX_OPTIMIZED = "index.optimized"

    # Cost events (for budget tracking)
    COST_THRESHOLD_REACHED = "cost.threshold_reached"
    COST_BUDGET_EXCEEDED = "cost.budget_exceeded"

    # Search events
    SEARCH_COMPLETED = "search.completed"

    # Wildcard (subscribe to all events)
    ALL = "*"


class WebhookStatus(str, Enum):
    """Webhook status."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class Webhook(BaseModel):
    """
    Webhook configuration.

    Webhooks allow external services to receive real-time notifications
    about events happening in arrwDB/Facilitair.
    """

    id: UUID = Field(default_factory=uuid4)
    tenant_id: Optional[str] = None  # For multi-tenancy
    url: HttpUrl = Field(..., description="The URL to send webhook payloads")
    events: List[WebhookEventType] = Field(
        ..., description="List of event types to subscribe to"
    )
    secret: str = Field(
        ...,
        description="Secret for HMAC signature verification",
        min_length=32,
    )
    status: WebhookStatus = Field(default=WebhookStatus.ACTIVE)
    description: Optional[str] = Field(None, max_length=500)

    # Delivery settings
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=60, ge=1, le=3600)
    timeout_seconds: int = Field(default=30, ge=1, le=120)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_triggered_at: Optional[datetime] = None

    # Statistics
    total_deliveries: int = Field(default=0, ge=0)
    successful_deliveries: int = Field(default=0, ge=0)
    failed_deliveries: int = Field(default=0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/webhooks/arrwdb",
                "events": ["job.completed", "job.failed"],
                "secret": "whsec_" + "a" * 40,
                "description": "Notify on compression job completion",
                "max_retries": 3,
                "timeout_seconds": 30,
            }
        }


class WebhookEvent(BaseModel):
    """
    Webhook event payload.

    This is the actual payload sent to webhook URLs.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique event ID")
    type: WebhookEventType = Field(..., description="Event type")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Event timestamp"
    )
    data: Dict[str, Any] = Field(
        ..., description="Event-specific data payload"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant ID (if multi-tenant)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "evt_123abc",
                "type": "job.completed",
                "created_at": "2024-01-01T12:00:00Z",
                "data": {
                    "job_id": "job_xyz789",
                    "job_type": "compression",
                    "status": "completed",
                    "result": {
                        "compression_rate": 0.65,
                        "model_size_before": 1024000,
                        "model_size_after": 665600,
                        "accuracy_loss": 0.02,
                    },
                    "cost": 2.45,
                    "duration_seconds": 720,
                },
                "tenant_id": "tenant_abc123",
            }
        }


class WebhookDelivery(BaseModel):
    """
    Webhook delivery attempt record.

    Tracks individual delivery attempts for auditing and debugging.
    """

    id: UUID = Field(default_factory=uuid4)
    webhook_id: UUID = Field(..., description="Webhook that received the event")
    event_id: UUID = Field(..., description="Event that was delivered")
    event_type: WebhookEventType

    # Delivery details
    status: DeliveryStatus = Field(default=DeliveryStatus.PENDING)
    attempt_number: int = Field(default=1, ge=1)
    attempted_at: datetime = Field(default_factory=datetime.utcnow)

    # Response details
    status_code: Optional[int] = None
    response_body: Optional[str] = Field(None, max_length=1000)
    error_message: Optional[str] = Field(None, max_length=500)
    duration_ms: Optional[int] = Field(None, ge=0)

    # Next retry
    next_retry_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "webhook_id": "webhook_abc123",
                "event_id": "evt_xyz789",
                "event_type": "job.completed",
                "status": "success",
                "attempt_number": 1,
                "status_code": 200,
                "response_body": '{"received": true}',
                "duration_ms": 125,
            }
        }


class CreateWebhookRequest(BaseModel):
    """Request model for creating a webhook."""

    url: HttpUrl
    events: List[WebhookEventType]
    description: Optional[str] = None
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=60, ge=1, le=3600)
    timeout_seconds: int = Field(default=30, ge=1, le=120)


class UpdateWebhookRequest(BaseModel):
    """Request model for updating a webhook."""

    url: Optional[HttpUrl] = None
    events: Optional[List[WebhookEventType]] = None
    description: Optional[str] = None
    status: Optional[WebhookStatus] = None
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    retry_delay_seconds: Optional[int] = Field(None, ge=1, le=3600)
    timeout_seconds: Optional[int] = Field(None, ge=1, le=120)


class WebhookListResponse(BaseModel):
    """Response model for listing webhooks."""

    webhooks: List[Webhook]
    total: int
    page: int
    page_size: int


class WebhookDeliveryListResponse(BaseModel):
    """Response model for listing webhook deliveries."""

    deliveries: List[WebhookDelivery]
    total: int
    page: int
    page_size: int
