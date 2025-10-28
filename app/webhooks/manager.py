"""
Webhook Manager for arrwDB.

Handles webhook registration, event delivery, retries, and HMAC signature generation.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.webhooks.models import (
    Webhook,
    WebhookEvent,
    WebhookDelivery,
    WebhookEventType,
    WebhookStatus,
    DeliveryStatus,
)

logger = logging.getLogger(__name__)


class WebhookManager:
    """
    Manages webhooks and event delivery.

    Features:
    - HMAC signature generation for security
    - Automatic retries with exponential backoff
    - Delivery tracking and statistics
    - Event filtering by subscription
    """

    def __init__(self):
        """Initialize webhook manager."""
        self.webhooks: Dict[UUID, Webhook] = {}
        self.deliveries: List[WebhookDelivery] = []
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=100),
            )
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def create_webhook(
        self,
        url: str,
        events: List[WebhookEventType],
        tenant_id: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_seconds: int = 60,
        timeout_seconds: int = 30,
    ) -> Webhook:
        """
        Register a new webhook.

        Args:
            url: Webhook endpoint URL
            events: List of event types to subscribe to
            tenant_id: Optional tenant ID for multi-tenancy
            description: Optional webhook description
            max_retries: Maximum number of retry attempts
            retry_delay_seconds: Delay between retries
            timeout_seconds: Request timeout

        Returns:
            Created Webhook
        """
        # Generate secure secret (64 hex characters = 256 bits)
        secret = "whsec_" + secrets.token_hex(32)

        webhook = Webhook(
            url=url,
            events=events,
            secret=secret,
            tenant_id=tenant_id,
            description=description,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
        )

        self.webhooks[webhook.id] = webhook
        logger.info(f"Created webhook {webhook.id} for events: {events}")

        return webhook

    def get_webhook(self, webhook_id: UUID) -> Optional[Webhook]:
        """Get webhook by ID."""
        return self.webhooks.get(webhook_id)

    def list_webhooks(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[WebhookStatus] = None,
    ) -> List[Webhook]:
        """
        List all webhooks with optional filtering.

        Args:
            tenant_id: Filter by tenant ID
            status: Filter by status

        Returns:
            List of webhooks
        """
        webhooks = list(self.webhooks.values())

        if tenant_id is not None:
            webhooks = [w for w in webhooks if w.tenant_id == tenant_id]

        if status is not None:
            webhooks = [w for w in webhooks if w.status == status]

        return webhooks

    def update_webhook(
        self,
        webhook_id: UUID,
        url: Optional[str] = None,
        events: Optional[List[WebhookEventType]] = None,
        description: Optional[str] = None,
        status: Optional[WebhookStatus] = None,
        max_retries: Optional[int] = None,
        retry_delay_seconds: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Optional[Webhook]:
        """
        Update an existing webhook.

        Args:
            webhook_id: Webhook ID
            url: New URL
            events: New event subscriptions
            description: New description
            status: New status
            max_retries: New max retries
            retry_delay_seconds: New retry delay
            timeout_seconds: New timeout

        Returns:
            Updated Webhook or None if not found
        """
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return None

        if url is not None:
            webhook.url = url
        if events is not None:
            webhook.events = events
        if description is not None:
            webhook.description = description
        if status is not None:
            webhook.status = status
        if max_retries is not None:
            webhook.max_retries = max_retries
        if retry_delay_seconds is not None:
            webhook.retry_delay_seconds = retry_delay_seconds
        if timeout_seconds is not None:
            webhook.timeout_seconds = timeout_seconds

        webhook.updated_at = datetime.utcnow()
        logger.info(f"Updated webhook {webhook_id}")

        return webhook

    def delete_webhook(self, webhook_id: UUID) -> bool:
        """
        Delete a webhook.

        Args:
            webhook_id: Webhook ID

        Returns:
            True if deleted, False if not found
        """
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Deleted webhook {webhook_id}")
            return True
        return False

    def generate_signature(self, payload: bytes, secret: str) -> str:
        """
        Generate HMAC SHA-256 signature for payload.

        Args:
            payload: JSON payload as bytes
            secret: Webhook secret

        Returns:
            Hex-encoded HMAC signature
        """
        signature = hmac.new(
            secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    async def trigger_event(
        self,
        event_type: WebhookEventType,
        data: Dict,
        tenant_id: Optional[str] = None,
    ):
        """
        Trigger webhooks for a specific event type.

        This method finds all webhooks subscribed to the event type
        and delivers the event to them asynchronously.

        Args:
            event_type: Type of event
            data: Event data payload
            tenant_id: Optional tenant ID
        """
        # Create event
        event = WebhookEvent(
            type=event_type, data=data, tenant_id=tenant_id
        )

        # Find matching webhooks
        matching_webhooks = []
        for webhook in self.webhooks.values():
            # Check if webhook is active
            if webhook.status != WebhookStatus.ACTIVE:
                continue

            # Check tenant match (if multi-tenant)
            if tenant_id is not None and webhook.tenant_id != tenant_id:
                continue

            # Check event subscription
            if WebhookEventType.ALL in webhook.events or event_type in webhook.events:
                matching_webhooks.append(webhook)

        logger.info(
            f"Triggering {len(matching_webhooks)} webhooks for event {event_type}"
        )

        # Deliver to all matching webhooks concurrently
        tasks = [
            self.deliver_event(webhook, event) for webhook in matching_webhooks
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def deliver_event(
        self, webhook: Webhook, event: WebhookEvent, attempt: int = 1
    ) -> WebhookDelivery:
        """
        Deliver event to a single webhook.

        Args:
            webhook: Webhook to deliver to
            event: Event to deliver
            attempt: Attempt number (for retries)

        Returns:
            WebhookDelivery record
        """
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_id=event.id,
            event_type=event.type,
            attempt_number=attempt,
        )

        try:
            # Prepare payload
            payload = event.model_dump_json().encode()

            # Generate signature
            signature = self.generate_signature(payload, webhook.secret)

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature,
                "X-Webhook-ID": str(webhook.id),
                "X-Event-ID": str(event.id),
                "X-Event-Type": event.type.value,
                "User-Agent": "arrwDB-Webhooks/1.0",
            }

            # Send request with timeout
            start_time = datetime.utcnow()
            response = await self.http_client.post(
                str(webhook.url),
                content=payload,
                headers=headers,
                timeout=webhook.timeout_seconds,
            )
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Update delivery record
            delivery.status_code = response.status_code
            delivery.response_body = response.text[:1000]  # Limit to 1000 chars
            delivery.duration_ms = duration_ms

            if 200 <= response.status_code < 300:
                delivery.status = DeliveryStatus.SUCCESS
                webhook.successful_deliveries += 1
                logger.info(
                    f"Successfully delivered event {event.id} to webhook {webhook.id} "
                    f"(attempt {attempt}, {duration_ms}ms)"
                )
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

        except Exception as e:
            logger.warning(
                f"Failed to deliver event {event.id} to webhook {webhook.id} "
                f"(attempt {attempt}): {e}"
            )

            delivery.status = DeliveryStatus.FAILED
            delivery.error_message = str(e)[:500]
            webhook.failed_deliveries += 1

            # Schedule retry if attempts remaining
            if attempt < webhook.max_retries:
                delivery.status = DeliveryStatus.RETRYING
                delivery.next_retry_at = datetime.utcnow() + timedelta(
                    seconds=webhook.retry_delay_seconds * attempt
                )
                logger.info(
                    f"Will retry delivery at {delivery.next_retry_at} "
                    f"(attempt {attempt + 1}/{webhook.max_retries})"
                )

                # Schedule retry
                asyncio.create_task(
                    self._retry_delivery(webhook, event, attempt + 1, delivery.next_retry_at)
                )

        # Update webhook statistics
        webhook.total_deliveries += 1
        webhook.last_triggered_at = datetime.utcnow()

        # Store delivery record
        self.deliveries.append(delivery)

        return delivery

    async def _retry_delivery(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        attempt: int,
        retry_at: datetime,
    ):
        """
        Retry failed delivery after delay.

        Args:
            webhook: Webhook to retry
            event: Event to retry
            attempt: Attempt number
            retry_at: When to retry
        """
        # Wait until retry time
        delay = (retry_at - datetime.utcnow()).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)

        # Retry delivery
        await self.deliver_event(webhook, event, attempt)

    def get_deliveries(
        self,
        webhook_id: Optional[UUID] = None,
        event_id: Optional[UUID] = None,
        status: Optional[DeliveryStatus] = None,
        limit: int = 100,
    ) -> List[WebhookDelivery]:
        """
        Get webhook deliveries with filtering.

        Args:
            webhook_id: Filter by webhook ID
            event_id: Filter by event ID
            status: Filter by delivery status
            limit: Maximum number of results

        Returns:
            List of deliveries
        """
        deliveries = self.deliveries

        if webhook_id is not None:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        if event_id is not None:
            deliveries = [d for d in deliveries if d.event_id == event_id]

        if status is not None:
            deliveries = [d for d in deliveries if d.status == status]

        # Sort by most recent first
        deliveries = sorted(deliveries, key=lambda d: d.attempted_at, reverse=True)

        return deliveries[:limit]

    def get_webhook_stats(self, webhook_id: UUID) -> Optional[Dict]:
        """
        Get statistics for a webhook.

        Args:
            webhook_id: Webhook ID

        Returns:
            Statistics dictionary or None if not found
        """
        webhook = self.get_webhook(webhook_id)
        if not webhook:
            return None

        success_rate = (
            (webhook.successful_deliveries / webhook.total_deliveries * 100)
            if webhook.total_deliveries > 0
            else 0.0
        )

        return {
            "webhook_id": str(webhook_id),
            "total_deliveries": webhook.total_deliveries,
            "successful_deliveries": webhook.successful_deliveries,
            "failed_deliveries": webhook.failed_deliveries,
            "success_rate": round(success_rate, 2),
            "last_triggered_at": webhook.last_triggered_at,
        }


# Global webhook manager instance
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get global webhook manager instance."""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager
