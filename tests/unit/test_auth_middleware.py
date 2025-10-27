"""
Test suite for app/auth/middleware.py

Coverage targets:
- API key authentication via X-API-Key header
- API key authentication via Authorization Bearer header
- Multi-tenancy enabled/disabled scenarios
- Valid API key authentication
- Invalid/expired API key rejection
- Missing API key handling
- Tenant context injection
- Require tenant dependency
"""

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.auth.api_keys import Tenant
from app.auth.middleware import get_current_tenant, require_tenant


class TestGetCurrentTenant:
    """Test get_current_tenant dependency."""

    @pytest.mark.asyncio
    async def test_valid_api_key_x_api_key_header(self):
        """Test authentication with valid X-API-Key header."""
        tenant_id = uuid4()
        tenant = Tenant(
            tenant_id=tenant_id,
            name="Test Tenant",
            api_key="arrw_test123",
            is_active=True
        )

        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_manager.return_value.validate_api_key.return_value = tenant

            result = await get_current_tenant(x_api_key="arrw_test123")

            assert result == tenant
            mock_manager.return_value.validate_api_key.assert_called_once_with("arrw_test123")

    @pytest.mark.asyncio
    async def test_valid_api_key_authorization_header(self):
        """Test authentication with valid Authorization Bearer header."""
        tenant_id = uuid4()
        tenant = Tenant(
            tenant_id=tenant_id,
            name="Test Tenant",
            api_key="arrw_test456",
            is_active=True
        )

        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_manager.return_value.validate_api_key.return_value = tenant

            result = await get_current_tenant(
                x_api_key=None,
                authorization="Bearer arrw_test456"
            )

            assert result == tenant
            mock_manager.return_value.validate_api_key.assert_called_once_with("arrw_test456")

    @pytest.mark.asyncio
    async def test_authorization_header_with_extra_whitespace(self):
        """Test that extra whitespace in Bearer token is handled."""
        tenant_id = uuid4()
        tenant = Tenant(
            tenant_id=tenant_id,
            name="Test Tenant",
            api_key="arrw_test789",
            is_active=True
        )

        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_manager.return_value.validate_api_key.return_value = tenant

            result = await get_current_tenant(
                authorization="Bearer   arrw_test789   "
            )

            assert result == tenant
            mock_manager.return_value.validate_api_key.assert_called_once_with("arrw_test789")

    @pytest.mark.asyncio
    async def test_x_api_key_takes_precedence(self):
        """Test that X-API-Key header takes precedence over Authorization."""
        tenant_id = uuid4()
        tenant = Tenant(
            tenant_id=tenant_id,
            name="Test Tenant",
            api_key="arrw_primary",
            is_active=True
        )

        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_manager.return_value.validate_api_key.return_value = tenant

            result = await get_current_tenant(
                x_api_key="arrw_primary",
                authorization="Bearer arrw_secondary"
            )

            # Should use X-API-Key
            mock_manager.return_value.validate_api_key.assert_called_once_with("arrw_primary")

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_401(self):
        """Test that invalid API key raises 401 Unauthorized."""
        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_manager.return_value.validate_api_key.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(x_api_key="arrw_invalid")

            assert exc_info.value.status_code == 401
            assert "Invalid or inactive" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_expired_api_key_raises_401(self):
        """Test that expired API key is rejected."""
        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            # validate_api_key returns None for expired keys
            mock_manager.return_value.validate_api_key.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(x_api_key="arrw_expired")

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_api_key_with_multi_tenancy_enabled(self):
        """Test that missing API key raises 401 when multi-tenancy is enabled."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = True

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant()

            assert exc_info.value.status_code == 401
            assert "API key required" in exc_info.value.detail
            assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"

    @pytest.mark.asyncio
    async def test_missing_api_key_with_multi_tenancy_disabled(self):
        """Test that missing API key returns None when multi-tenancy is disabled."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = False

            result = await get_current_tenant()

            assert result is None

    @pytest.mark.asyncio
    async def test_authorization_header_without_bearer_prefix(self):
        """Test that Authorization header without 'Bearer ' prefix is ignored."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = True

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(authorization="arrw_test123")

            # Should be treated as missing API key
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_api_key_headers(self):
        """Test behavior with empty header values."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = True

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(x_api_key="", authorization="")

            assert exc_info.value.status_code == 401


class TestRequireTenant:
    """Test require_tenant dependency."""

    @pytest.mark.asyncio
    async def test_require_tenant_with_valid_tenant(self):
        """Test that valid tenant passes through."""
        tenant_id = uuid4()
        tenant = Tenant(
            tenant_id=tenant_id,
            name="Test Tenant",
            api_key="arrw_test",
            is_active=True
        )

        result = await require_tenant(tenant=tenant)

        assert result == tenant

    @pytest.mark.asyncio
    async def test_require_tenant_with_none_raises_401(self):
        """Test that None tenant raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await require_tenant(tenant=None)

        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail
        assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"


class TestAuthenticationFlow:
    """Integration tests for authentication flow."""

    @pytest.mark.asyncio
    async def test_full_authentication_flow_success(self):
        """Test complete authentication flow from API key to tenant."""
        tenant_id = uuid4()
        tenant = Tenant(
            tenant_id=tenant_id,
            name="Production Tenant",
            api_key="arrw_prod_key",
            is_active=True
        )

        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_manager.return_value.validate_api_key.return_value = tenant

            # Step 1: Authenticate
            authenticated_tenant = await get_current_tenant(x_api_key="arrw_prod_key")
            assert authenticated_tenant == tenant

            # Step 2: Require authentication (should pass)
            required_tenant = await require_tenant(tenant=authenticated_tenant)
            assert required_tenant == tenant

    @pytest.mark.asyncio
    async def test_full_authentication_flow_failure(self):
        """Test complete authentication flow with invalid key."""
        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_manager.return_value.validate_api_key.return_value = None

            # Should fail at authentication step
            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(x_api_key="arrw_invalid")

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_anonymous_access_flow(self):
        """Test anonymous access when multi-tenancy is disabled."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = False

            # Step 1: Authenticate (returns None for anonymous)
            tenant = await get_current_tenant()
            assert tenant is None

            # Step 2: Require tenant (should fail for admin endpoints)
            with pytest.raises(HTTPException) as exc_info:
                await require_tenant(tenant=tenant)

            assert exc_info.value.status_code == 401


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_malformed_bearer_token(self):
        """Test handling of malformed Bearer token."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = True

            # "Bearer" without space
            with pytest.raises(HTTPException):
                await get_current_tenant(authorization="Bearerarrw_test")

    @pytest.mark.asyncio
    async def test_case_sensitive_bearer_prefix(self):
        """Test that Bearer prefix is case-sensitive."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = True

            # lowercase "bearer" should not work
            with pytest.raises(HTTPException):
                await get_current_tenant(authorization="bearer arrw_test")

    @pytest.mark.asyncio
    async def test_api_key_validation_exception(self):
        """Test handling of exceptions during API key validation."""
        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_manager.return_value.validate_api_key.side_effect = Exception("DB error")

            with pytest.raises(Exception, match="DB error"):
                await get_current_tenant(x_api_key="arrw_test")

    @pytest.mark.asyncio
    async def test_tenant_with_inactive_status(self):
        """Test that inactive tenant is rejected."""
        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            # validate_api_key returns None for inactive tenants
            mock_manager.return_value.validate_api_key.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(x_api_key="arrw_inactive")

            assert exc_info.value.status_code == 401


class TestMultiTenancyScenarios:
    """Test multi-tenancy enabled/disabled scenarios."""

    @pytest.mark.asyncio
    async def test_multi_tenancy_enabled_requires_api_key(self):
        """Test that multi-tenancy enabled requires API key."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = True

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant()

            assert exc_info.value.status_code == 401
            assert "required" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_multi_tenancy_disabled_allows_anonymous(self):
        """Test that multi-tenancy disabled allows anonymous access."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = False

            result = await get_current_tenant()

            assert result is None  # Anonymous access allowed

    @pytest.mark.asyncio
    async def test_multi_tenancy_disabled_with_valid_key(self):
        """Test that valid API key still works when multi-tenancy is disabled."""
        tenant_id = uuid4()
        tenant = Tenant(
            tenant_id=tenant_id,
            name="Test Tenant",
            api_key="arrw_test",
            is_active=True
        )

        with patch('app.auth.middleware.settings') as mock_settings, \
             patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_settings.MULTI_TENANCY_ENABLED = False
            mock_manager.return_value.validate_api_key.return_value = tenant

            result = await get_current_tenant(x_api_key="arrw_test")

            # Should authenticate even with multi-tenancy disabled
            assert result == tenant


class TestSecurityHeaders:
    """Test security-related headers in responses."""

    @pytest.mark.asyncio
    async def test_www_authenticate_header_on_401(self):
        """Test that WWW-Authenticate header is present on 401 responses."""
        with patch('app.auth.middleware.settings') as mock_settings:
            mock_settings.MULTI_TENANCY_ENABLED = True

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant()

            assert "WWW-Authenticate" in exc_info.value.headers
            assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"

    @pytest.mark.asyncio
    async def test_www_authenticate_header_on_invalid_key(self):
        """Test WWW-Authenticate header for invalid API key."""
        with patch('app.auth.middleware.get_api_key_manager') as mock_manager:
            mock_manager.return_value.validate_api_key.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(x_api_key="arrw_invalid")

            assert "WWW-Authenticate" in exc_info.value.headers
