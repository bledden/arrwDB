"""
Authentication middleware for multi-tenancy.

Validates API keys and injects tenant context into requests.
"""

from fastapi import Header, HTTPException, status
from typing import Optional

from .api_keys import Tenant, get_api_key_manager


async def get_current_tenant(
    x_api_key: Optional[str] = Header(None, description="API key for authentication"),
    authorization: Optional[str] = Header(None, description="Bearer token (alternative to X-API-Key)"),
) -> Optional[Tenant]:
    """
    Dependency to get the current tenant from API key.

    Supports two authentication methods:
    1. X-API-Key header: X-API-Key: arrw_abc123...
    2. Authorization header: Authorization: Bearer arrw_abc123...

    Args:
        x_api_key: API key from X-API-Key header
        authorization: API key from Authorization header

    Returns:
        Tenant if authenticated, None if multi-tenancy is disabled

    Raises:
        HTTPException 401: If authentication is required but fails
    """
    # Extract API key from either header
    api_key = None
    if x_api_key:
        api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "").strip()

    # If no API key provided
    if not api_key:
        # Check if multi-tenancy is enabled
        from app.config import settings
        if getattr(settings, "MULTI_TENANCY_ENABLED", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required. Provide X-API-Key header or Authorization: Bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Multi-tenancy disabled, allow anonymous access
        return None

    # Validate API key
    api_key_manager = get_api_key_manager()
    tenant = api_key_manager.validate_api_key(api_key)

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return tenant


async def require_tenant(
    tenant: Optional[Tenant] = None,
) -> Tenant:
    """
    Dependency to require authentication (even if multi-tenancy is disabled).

    Use this for admin endpoints that always require authentication.

    Args:
        tenant: Current tenant from get_current_tenant

    Returns:
        Tenant

    Raises:
        HTTPException 401: If not authenticated
    """
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return tenant
