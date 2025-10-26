"""
API Key management for multi-tenancy.

API keys are used to:
1. Authenticate requests
2. Identify the tenant
3. Scope data access to tenant's resources
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json


@dataclass
class Tenant:
    """Represents a tenant in the system."""

    tenant_id: str
    name: str
    api_key_hash: str
    created_at: datetime
    is_active: bool = True
    metadata: Optional[Dict] = None
    key_expires_at: Optional[datetime] = None  # API key expiration (None = never expires)
    last_used_at: Optional[datetime] = None    # Track last usage for audit
    request_count: int = 0                      # Track total requests for monitoring

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "api_key_hash": self.api_key_hash,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata or {},
            "key_expires_at": self.key_expires_at.isoformat() if self.key_expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "request_count": self.request_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Tenant":
        """Create from dictionary."""
        return cls(
            tenant_id=data["tenant_id"],
            name=data["name"],
            api_key_hash=data["api_key_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata"),
            key_expires_at=datetime.fromisoformat(data["key_expires_at"]) if data.get("key_expires_at") else None,
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None,
            request_count=data.get("request_count", 0),
        )

    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.key_expires_at is None:
            return False
        return datetime.utcnow() > self.key_expires_at


class APIKeyManager:
    """
    Manages API keys for multi-tenant authentication.

    Features:
    - Generate API keys
    - Validate API keys
    - Tenant lookup
    - Key rotation
    """

    def __init__(self, storage_path: str = "./data/tenants.json"):
        """
        Initialize API key manager.

        Args:
            storage_path: Path to store tenant/key mappings
        """
        self.storage_path = Path(storage_path)
        self.tenants: Dict[str, Tenant] = {}  # tenant_id -> Tenant
        self.key_hash_to_tenant: Dict[str, str] = {}  # api_key_hash -> tenant_id
        self._load_tenants()

    def _load_tenants(self) -> None:
        """Load tenants from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for tenant_data in data.get("tenants", []):
                        tenant = Tenant.from_dict(tenant_data)
                        self.tenants[tenant.tenant_id] = tenant
                        self.key_hash_to_tenant[tenant.api_key_hash] = tenant.tenant_id
            except Exception as e:
                print(f"Warning: Failed to load tenants: {e}")

    def _save_tenants(self) -> None:
        """Save tenants to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tenants": [tenant.to_dict() for tenant in self.tenants.values()]
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _hash_api_key(api_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def generate_api_key() -> str:
        """
        Generate a cryptographically secure API key.

        Format: arrw_<32 random hex chars>
        Example: arrw_a1b2c3d4e5f67890abcdef1234567890
        """
        random_part = secrets.token_hex(16)  # 32 chars
        return f"arrw_{random_part}"

    def create_tenant(
        self,
        name: str,
        metadata: Optional[Dict] = None,
        expires_in_days: Optional[int] = None
    ) -> tuple[str, str]:
        """
        Create a new tenant with API key.

        Args:
            name: Tenant name (e.g., "Acme Corp")
            metadata: Optional metadata
            expires_in_days: Optional key expiration (None = never expires)

        Returns:
            Tuple of (tenant_id, api_key)
            IMPORTANT: api_key is only returned once! Store it securely.
        """
        # Generate tenant ID and API key
        tenant_id = f"tenant_{secrets.token_hex(8)}"
        api_key = self.generate_api_key()
        api_key_hash = self._hash_api_key(api_key)

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create tenant
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            api_key_hash=api_key_hash,
            created_at=datetime.utcnow(),
            is_active=True,
            metadata=metadata,
            key_expires_at=expires_at,
        )

        # Store tenant
        self.tenants[tenant_id] = tenant
        self.key_hash_to_tenant[api_key_hash] = tenant_id
        self._save_tenants()

        # Return tenant_id and PLAINTEXT api_key (only time it's visible)
        return tenant_id, api_key

    def validate_api_key(self, api_key: str, track_usage: bool = True) -> Optional[Tenant]:
        """
        Validate an API key and return the tenant.

        Args:
            api_key: The API key to validate
            track_usage: Whether to track usage (last_used_at, request_count)

        Returns:
            Tenant if valid, active, and not expired. None otherwise.
        """
        if not api_key or not api_key.startswith("arrw_"):
            return None

        api_key_hash = self._hash_api_key(api_key)
        tenant_id = self.key_hash_to_tenant.get(api_key_hash)

        if not tenant_id:
            return None

        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None

        # Check if tenant is active
        if not tenant.is_active:
            return None

        # Check if key is expired
        if tenant.is_expired():
            return None

        # Track usage for audit/monitoring
        if track_usage:
            tenant.last_used_at = datetime.utcnow()
            tenant.request_count += 1
            # Save periodically (every 100 requests to avoid too many writes)
            if tenant.request_count % 100 == 0:
                self._save_tenants()

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """
        Deactivate a tenant (disable their API key).

        Args:
            tenant_id: The tenant ID

        Returns:
            True if deactivated, False if not found
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.is_active = False
        self._save_tenants()
        return True

    def rotate_api_key(self, tenant_id: str) -> Optional[str]:
        """
        Rotate an API key for a tenant.

        Args:
            tenant_id: The tenant ID

        Returns:
            New API key if successful, None if tenant not found
            IMPORTANT: New key is only returned once!
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None

        # Generate new key
        new_api_key = self.generate_api_key()
        new_api_key_hash = self._hash_api_key(new_api_key)

        # Remove old key hash mapping
        old_hash = tenant.api_key_hash
        if old_hash in self.key_hash_to_tenant:
            del self.key_hash_to_tenant[old_hash]

        # Update tenant
        tenant.api_key_hash = new_api_key_hash
        self.key_hash_to_tenant[new_api_key_hash] = tenant_id
        self._save_tenants()

        return new_api_key

    def list_tenants(self) -> list[Tenant]:
        """List all tenants."""
        return list(self.tenants.values())


# Global API key manager instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
