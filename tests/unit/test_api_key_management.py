"""
Test suite for app/auth/api_keys.py

Coverage targets:
- API key generation and format validation
- Tenant creation with API keys
- API key validation (valid/invalid/expired)
- API key hashing for security
- Tenant activation/deactivation
- API key rotation
- Usage tracking (last_used_at, request_count)
- Tenant persistence (save/load)
- Tenant lookup and listing
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.auth.api_keys import APIKeyManager, Tenant


class TestTenantDataclass:
    """Test Tenant dataclass."""

    def test_tenant_creation(self):
        """Test creating a Tenant."""
        tenant = Tenant(
            tenant_id="tenant_123",
            name="Test Corp",
            api_key_hash="hash_abc123",
            created_at=datetime.utcnow(),
            is_active=True
        )

        assert tenant.tenant_id == "tenant_123"
        assert tenant.name == "Test Corp"
        assert tenant.is_active is True
        assert tenant.request_count == 0

    def test_tenant_to_dict(self):
        """Test converting Tenant to dictionary."""
        now = datetime.utcnow()
        tenant = Tenant(
            tenant_id="tenant_456",
            name="Test Corp",
            api_key_hash="hash_def456",
            created_at=now,
            is_active=True,
            metadata={"plan": "enterprise"}
        )

        tenant_dict = tenant.to_dict()

        assert tenant_dict["tenant_id"] == "tenant_456"
        assert tenant_dict["name"] == "Test Corp"
        assert tenant_dict["is_active"] is True
        assert tenant_dict["metadata"] == {"plan": "enterprise"}
        assert "created_at" in tenant_dict

    def test_tenant_from_dict(self):
        """Test creating Tenant from dictionary."""
        now = datetime.utcnow()
        tenant_dict = {
            "tenant_id": "tenant_789",
            "name": "Test Corp",
            "api_key_hash": "hash_ghi789",
            "created_at": now.isoformat(),
            "is_active": True,
            "metadata": {"region": "us-west"},
            "request_count": 42
        }

        tenant = Tenant.from_dict(tenant_dict)

        assert tenant.tenant_id == "tenant_789"
        assert tenant.name == "Test Corp"
        assert tenant.request_count == 42
        assert tenant.metadata == {"region": "us-west"}

    def test_tenant_is_expired_never_expires(self):
        """Test that tenant without expiration never expires."""
        tenant = Tenant(
            tenant_id="tenant_001",
            name="Test",
            api_key_hash="hash",
            created_at=datetime.utcnow(),
            key_expires_at=None
        )

        assert tenant.is_expired() is False

    def test_tenant_is_expired_future_expiration(self):
        """Test tenant with future expiration date."""
        tenant = Tenant(
            tenant_id="tenant_002",
            name="Test",
            api_key_hash="hash",
            created_at=datetime.utcnow(),
            key_expires_at=datetime.utcnow() + timedelta(days=30)
        )

        assert tenant.is_expired() is False

    def test_tenant_is_expired_past_expiration(self):
        """Test tenant with past expiration date."""
        tenant = Tenant(
            tenant_id="tenant_003",
            name="Test",
            api_key_hash="hash",
            created_at=datetime.utcnow(),
            key_expires_at=datetime.utcnow() - timedelta(days=1)
        )

        assert tenant.is_expired() is True


class TestAPIKeyGeneration:
    """Test API key generation."""

    def test_generate_api_key_format(self):
        """Test that generated API keys have correct format."""
        api_key = APIKeyManager.generate_api_key()

        assert api_key.startswith("arrw_")
        assert len(api_key) == 37  # "arrw_" (5) + 32 hex chars

    def test_generate_api_key_uniqueness(self):
        """Test that generated API keys are unique."""
        keys = [APIKeyManager.generate_api_key() for _ in range(100)]

        # All keys should be unique
        assert len(set(keys)) == 100

    def test_hash_api_key(self):
        """Test API key hashing."""
        api_key = "arrw_test123456789"
        hash1 = APIKeyManager._hash_api_key(api_key)
        hash2 = APIKeyManager._hash_api_key(api_key)

        # Same key should produce same hash
        assert hash1 == hash2
        # Hash should be different from original
        assert hash1 != api_key
        # Hash should be 64 chars (SHA256 hex)
        assert len(hash1) == 64


class TestAPIKeyManager:
    """Test API Key Manager functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Provide temporary storage for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "tenants.json"

    @pytest.fixture
    def manager(self, temp_storage):
        """Provide fresh APIKeyManager for each test."""
        return APIKeyManager(storage_path=str(temp_storage))

    def test_create_tenant(self, manager):
        """Test creating a new tenant."""
        tenant_id, api_key = manager.create_tenant(
            name="Acme Corp",
            metadata={"plan": "pro"}
        )

        assert tenant_id.startswith("tenant_")
        assert api_key.startswith("arrw_")

        # Verify tenant is stored
        tenant = manager.get_tenant(tenant_id)
        assert tenant is not None
        assert tenant.name == "Acme Corp"
        assert tenant.metadata == {"plan": "pro"}
        assert tenant.is_active is True

    def test_create_tenant_with_expiration(self, manager):
        """Test creating tenant with key expiration."""
        tenant_id, api_key = manager.create_tenant(
            name="Temp Corp",
            expires_in_days=30
        )

        tenant = manager.get_tenant(tenant_id)
        assert tenant.key_expires_at is not None
        assert tenant.key_expires_at > datetime.utcnow()

    def test_validate_api_key_valid(self, manager):
        """Test validating a valid API key."""
        tenant_id, api_key = manager.create_tenant(name="Test Corp")

        validated_tenant = manager.validate_api_key(api_key)

        assert validated_tenant is not None
        assert validated_tenant.tenant_id == tenant_id
        assert validated_tenant.name == "Test Corp"

    def test_validate_api_key_invalid_format(self, manager):
        """Test validating API key with invalid format."""
        # Missing prefix
        result = manager.validate_api_key("invalid_key")
        assert result is None

        # Wrong prefix
        result = manager.validate_api_key("wrong_abc123")
        assert result is None

        # Empty string
        result = manager.validate_api_key("")
        assert result is None

    def test_validate_api_key_nonexistent(self, manager):
        """Test validating nonexistent API key."""
        fake_key = "arrw_nonexistent12345678901234567890"
        result = manager.validate_api_key(fake_key)

        assert result is None

    def test_validate_api_key_inactive_tenant(self, manager):
        """Test that inactive tenant's key is rejected."""
        tenant_id, api_key = manager.create_tenant(name="Test Corp")

        # Deactivate tenant
        manager.deactivate_tenant(tenant_id)

        # Key should be invalid
        result = manager.validate_api_key(api_key)
        assert result is None

    def test_validate_api_key_expired(self, manager):
        """Test that expired API key is rejected."""
        tenant_id, api_key = manager.create_tenant(
            name="Expired Corp",
            expires_in_days=0  # Expires immediately
        )

        # Manually set expiration to past
        tenant = manager.get_tenant(tenant_id)
        tenant.key_expires_at = datetime.utcnow() - timedelta(days=1)

        # Key should be invalid
        result = manager.validate_api_key(api_key)
        assert result is None

    def test_validate_api_key_tracks_usage(self, manager):
        """Test that validation tracks usage."""
        tenant_id, api_key = manager.create_tenant(name="Test Corp")

        # Validate multiple times
        for _ in range(5):
            manager.validate_api_key(api_key, track_usage=True)

        tenant = manager.get_tenant(tenant_id)
        assert tenant.request_count == 5
        assert tenant.last_used_at is not None

    def test_validate_api_key_no_tracking(self, manager):
        """Test validation without usage tracking."""
        tenant_id, api_key = manager.create_tenant(name="Test Corp")

        manager.validate_api_key(api_key, track_usage=False)

        tenant = manager.get_tenant(tenant_id)
        assert tenant.request_count == 0
        assert tenant.last_used_at is None

    def test_deactivate_tenant(self, manager):
        """Test deactivating a tenant."""
        tenant_id, api_key = manager.create_tenant(name="Test Corp")

        # Deactivate
        success = manager.deactivate_tenant(tenant_id)
        assert success is True

        tenant = manager.get_tenant(tenant_id)
        assert tenant.is_active is False

        # API key should no longer work
        result = manager.validate_api_key(api_key)
        assert result is None

    def test_deactivate_nonexistent_tenant(self, manager):
        """Test deactivating nonexistent tenant."""
        success = manager.deactivate_tenant("nonexistent_id")
        assert success is False

    def test_rotate_api_key(self, manager):
        """Test rotating an API key."""
        tenant_id, old_api_key = manager.create_tenant(name="Test Corp")

        # Rotate key
        new_api_key = manager.rotate_api_key(tenant_id)

        assert new_api_key is not None
        assert new_api_key != old_api_key
        assert new_api_key.startswith("arrw_")

        # Old key should no longer work
        result = manager.validate_api_key(old_api_key)
        assert result is None

        # New key should work
        result = manager.validate_api_key(new_api_key)
        assert result is not None
        assert result.tenant_id == tenant_id

    def test_rotate_api_key_nonexistent_tenant(self, manager):
        """Test rotating key for nonexistent tenant."""
        new_key = manager.rotate_api_key("nonexistent_id")
        assert new_key is None

    def test_list_tenants(self, manager):
        """Test listing all tenants."""
        # Create multiple tenants
        manager.create_tenant(name="Corp A")
        manager.create_tenant(name="Corp B")
        manager.create_tenant(name="Corp C")

        tenants = manager.list_tenants()

        assert len(tenants) == 3
        names = [t.name for t in tenants]
        assert "Corp A" in names
        assert "Corp B" in names
        assert "Corp C" in names

    def test_list_tenants_empty(self, manager):
        """Test listing tenants when none exist."""
        tenants = manager.list_tenants()
        assert len(tenants) == 0


class TestAPIKeyManagerPersistence:
    """Test persistence (save/load) functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Provide temporary storage for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "tenants.json"

    def test_save_and_load_tenants(self, temp_storage):
        """Test saving and loading tenants."""
        # Create manager and add tenant
        manager1 = APIKeyManager(storage_path=str(temp_storage))
        tenant_id, api_key = manager1.create_tenant(
            name="Persistent Corp",
            metadata={"plan": "enterprise"}
        )

        # Create new manager instance (should load from storage)
        manager2 = APIKeyManager(storage_path=str(temp_storage))

        # Verify tenant was loaded
        tenant = manager2.get_tenant(tenant_id)
        assert tenant is not None
        assert tenant.name == "Persistent Corp"
        assert tenant.metadata == {"plan": "enterprise"}

        # Verify API key still works
        validated = manager2.validate_api_key(api_key)
        assert validated is not None

    def test_load_nonexistent_storage(self, temp_storage):
        """Test loading when storage file doesn't exist."""
        # Should not error, just create empty manager
        manager = APIKeyManager(storage_path=str(temp_storage))
        assert len(manager.list_tenants()) == 0

    def test_persistence_after_deactivation(self, temp_storage):
        """Test that deactivation is persisted."""
        manager1 = APIKeyManager(storage_path=str(temp_storage))
        tenant_id, api_key = manager1.create_tenant(name="Test Corp")
        manager1.deactivate_tenant(tenant_id)

        # Reload
        manager2 = APIKeyManager(storage_path=str(temp_storage))
        tenant = manager2.get_tenant(tenant_id)
        assert tenant.is_active is False

    def test_persistence_after_rotation(self, temp_storage):
        """Test that key rotation is persisted."""
        manager1 = APIKeyManager(storage_path=str(temp_storage))
        tenant_id, old_key = manager1.create_tenant(name="Test Corp")
        new_key = manager1.rotate_api_key(tenant_id)

        # Reload
        manager2 = APIKeyManager(storage_path=str(temp_storage))

        # Old key shouldn't work
        assert manager2.validate_api_key(old_key) is None

        # New key should work
        result = manager2.validate_api_key(new_key)
        assert result is not None


class TestUsageTracking:
    """Test usage tracking features."""

    @pytest.fixture
    def manager(self):
        """Provide fresh APIKeyManager for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "tenants.json"
            yield APIKeyManager(storage_path=str(storage))

    def test_request_count_increments(self, manager):
        """Test that request count increments."""
        tenant_id, api_key = manager.create_tenant(name="Test Corp")

        for i in range(10):
            manager.validate_api_key(api_key, track_usage=True)

        tenant = manager.get_tenant(tenant_id)
        assert tenant.request_count == 10

    def test_last_used_at_updates(self, manager):
        """Test that last_used_at timestamp updates."""
        tenant_id, api_key = manager.create_tenant(name="Test Corp")

        # First usage
        manager.validate_api_key(api_key, track_usage=True)
        tenant = manager.get_tenant(tenant_id)
        first_used = tenant.last_used_at

        # Wait a tiny bit and use again
        import time
        time.sleep(0.01)

        manager.validate_api_key(api_key, track_usage=True)
        tenant = manager.get_tenant(tenant_id)
        second_used = tenant.last_used_at

        assert second_used > first_used


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def manager(self):
        """Provide fresh APIKeyManager for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "tenants.json"
            yield APIKeyManager(storage_path=str(storage))

    def test_multiple_tenants_different_keys(self, manager):
        """Test multiple tenants have different keys."""
        _, key1 = manager.create_tenant(name="Corp 1")
        _, key2 = manager.create_tenant(name="Corp 2")
        _, key3 = manager.create_tenant(name="Corp 3")

        assert key1 != key2 != key3

    def test_get_tenant_by_id(self, manager):
        """Test getting tenant directly by ID."""
        tenant_id, _ = manager.create_tenant(name="Test Corp")

        tenant = manager.get_tenant(tenant_id)
        assert tenant is not None
        assert tenant.tenant_id == tenant_id

    def test_get_nonexistent_tenant(self, manager):
        """Test getting nonexistent tenant."""
        tenant = manager.get_tenant("nonexistent_id")
        assert tenant is None
