"""
Webhook API routes for arrwDB.

Provides REST API endpoints for managing webhooks and viewing delivery history.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from app.webhooks.manager import get_webhook_manager
from app.webhooks.models import (
    CreateWebhookRequest,
    UpdateWebhookRequest,
    Webhook,
    WebhookListResponse,
    WebhookDeliveryListResponse,
    WebhookStatus,
    DeliveryStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


@router.post(
    "",
    response_model=Webhook,
    status_code=status.HTTP_201_CREATED,
    summary="Create webhook",
    description="Register a new webhook to receive event notifications",
)
async def create_webhook(request: CreateWebhookRequest) -> Webhook:
    """
    Create a new webhook.

    The webhook will receive POST requests with event payloads
    whenever subscribed events occur.

    Returns:
        Created webhook with generated secret
    """
    manager = get_webhook_manager()

    webhook = manager.create_webhook(
        url=str(request.url),
        events=request.events,
        description=request.description,
        max_retries=request.max_retries,
        retry_delay_seconds=request.retry_delay_seconds,
        timeout_seconds=request.timeout_seconds,
    )

    logger.info(f"Created webhook {webhook.id} for URL: {webhook.url}")

    return webhook


@router.get(
    "",
    response_model=WebhookListResponse,
    summary="List webhooks",
    description="List all registered webhooks with optional filtering",
)
async def list_webhooks(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    status: Optional[WebhookStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
) -> WebhookListResponse:
    """
    List all webhooks.

    Args:
        tenant_id: Optional tenant filter
        status: Optional status filter
        page: Page number
        page_size: Items per page

    Returns:
        Paginated list of webhooks
    """
    manager = get_webhook_manager()

    webhooks = manager.list_webhooks(tenant_id=tenant_id, status=status)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    paginated_webhooks = webhooks[start:end]

    return WebhookListResponse(
        webhooks=paginated_webhooks,
        total=len(webhooks),
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{webhook_id}",
    response_model=Webhook,
    summary="Get webhook",
    description="Get webhook details by ID",
)
async def get_webhook(webhook_id: UUID) -> Webhook:
    """
    Get webhook by ID.

    Args:
        webhook_id: Webhook ID

    Returns:
        Webhook details

    Raises:
        404: Webhook not found
    """
    manager = get_webhook_manager()

    webhook = manager.get_webhook(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    return webhook


@router.patch(
    "/{webhook_id}",
    response_model=Webhook,
    summary="Update webhook",
    description="Update webhook configuration",
)
async def update_webhook(
    webhook_id: UUID, request: UpdateWebhookRequest
) -> Webhook:
    """
    Update webhook configuration.

    Args:
        webhook_id: Webhook ID
        request: Update request with fields to modify

    Returns:
        Updated webhook

    Raises:
        404: Webhook not found
    """
    manager = get_webhook_manager()

    webhook = manager.update_webhook(
        webhook_id=webhook_id,
        url=str(request.url) if request.url else None,
        events=request.events,
        description=request.description,
        status=request.status,
        max_retries=request.max_retries,
        retry_delay_seconds=request.retry_delay_seconds,
        timeout_seconds=request.timeout_seconds,
    )

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    logger.info(f"Updated webhook {webhook_id}")

    return webhook


@router.delete(
    "/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete webhook",
    description="Delete a webhook registration",
)
async def delete_webhook(webhook_id: UUID):
    """
    Delete a webhook.

    Args:
        webhook_id: Webhook ID

    Raises:
        404: Webhook not found
    """
    manager = get_webhook_manager()

    if not manager.delete_webhook(webhook_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    logger.info(f"Deleted webhook {webhook_id}")


@router.get(
    "/{webhook_id}/deliveries",
    response_model=WebhookDeliveryListResponse,
    summary="Get webhook deliveries",
    description="Get delivery history for a webhook",
)
async def get_webhook_deliveries(
    webhook_id: UUID,
    status: Optional[DeliveryStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
) -> WebhookDeliveryListResponse:
    """
    Get webhook delivery history.

    Args:
        webhook_id: Webhook ID
        status: Optional status filter
        page: Page number
        page_size: Items per page

    Returns:
        Paginated list of deliveries
    """
    manager = get_webhook_manager()

    # Verify webhook exists
    webhook = manager.get_webhook(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Get deliveries
    deliveries = manager.get_deliveries(
        webhook_id=webhook_id, status=status, limit=1000
    )

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    paginated_deliveries = deliveries[start:end]

    return WebhookDeliveryListResponse(
        deliveries=paginated_deliveries,
        total=len(deliveries),
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{webhook_id}/stats",
    summary="Get webhook statistics",
    description="Get delivery statistics for a webhook",
)
async def get_webhook_stats(webhook_id: UUID):
    """
    Get webhook statistics.

    Args:
        webhook_id: Webhook ID

    Returns:
        Statistics including success rate, total deliveries, etc.

    Raises:
        404: Webhook not found
    """
    manager = get_webhook_manager()

    stats = manager.get_webhook_stats(webhook_id)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    return stats


@router.post(
    "/{webhook_id}/test",
    summary="Test webhook",
    description="Send a test event to verify webhook configuration",
)
async def test_webhook(webhook_id: UUID):
    """
    Test webhook by sending a test event.

    Args:
        webhook_id: Webhook ID

    Returns:
        Test delivery result

    Raises:
        404: Webhook not found
    """
    manager = get_webhook_manager()

    webhook = manager.get_webhook(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Trigger test event
    from app.webhooks.models import WebhookEvent, WebhookEventType

    test_event = WebhookEvent(
        type=WebhookEventType.JOB_COMPLETED,
        data={
            "test": True,
            "message": "This is a test webhook delivery",
            "webhook_id": str(webhook_id),
        },
    )

    delivery = await manager.deliver_event(webhook, test_event)

    logger.info(f"Sent test event to webhook {webhook_id}")

    return {
        "success": delivery.status == DeliveryStatus.SUCCESS,
        "delivery": delivery,
    }
