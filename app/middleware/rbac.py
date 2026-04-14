"""
Role-Based Access Control (RBAC) middleware.

Roles:
  admin  - Full access: create/delete libraries, manage users, all operations
  editor - Read/write: add documents, search, upsert, but cannot delete libraries
  viewer - Read-only: search and read operations only

Permissions are checked per-request based on the API key's assigned role.
When RBAC is disabled (default), all requests are allowed.

Configuration:
  RBAC_ENABLED=true  # Enable RBAC
  RBAC_KEYS_PATH=./data/rbac_keys.json  # Key-role mappings
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class Role(str, Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


# Path suffix patterns -> minimum required role per HTTP method.
# Checked in order; first match wins. More specific patterns first.
PERMISSION_RULES = [
    # Health (always allowed, any method)
    (None, "/health", None),
    (None, "/ready", None),
    (None, "/ping", None),
    (None, "/metrics", None),

    # Search endpoints (read-level, POST because they take a body)
    ("POST", "/search", Role.VIEWER),
    ("POST", "/search/hybrid", Role.VIEWER),
    ("POST", "/search/stream", Role.VIEWER),

    # Index operations
    ("POST", "/rebuild", Role.EDITOR),
    ("POST", "/optimize", Role.EDITOR),

    # Document writes
    ("POST", "/documents", Role.EDITOR),
    ("DELETE", "/documents", Role.ADMIN),

    # Library management (exact path /v1/libraries with no trailing segments)
    ("POST", "/v1/libraries", Role.ADMIN),
    ("DELETE", "/v1/libraries", Role.ADMIN),

    # Library reads
    ("GET", "/v1/libraries", Role.VIEWER),
]

# Role hierarchy: admin > editor > viewer
ROLE_HIERARCHY = {
    Role.ADMIN: 3,
    Role.EDITOR: 2,
    Role.VIEWER: 1,
}


class RBACKeyStore:
    """Manages API key -> role mappings."""

    def __init__(self, keys_path: str = "./data/rbac_keys.json"):
        self._path = Path(keys_path)
        self._keys: dict[str, dict] = {}  # api_key -> {"role": Role, "name": str}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                for key, info in data.items():
                    self._keys[key] = {
                        "role": Role(info["role"]),
                        "name": info.get("name", "unknown"),
                    }
                logger.info(f"Loaded {len(self._keys)} RBAC keys")
            except Exception as e:
                logger.warning(f"Failed to load RBAC keys: {e}")

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for key, info in self._keys.items():
            data[key] = {"role": info["role"].value, "name": info["name"]}
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def get_role(self, api_key: str) -> Optional[Role]:
        info = self._keys.get(api_key)
        return info["role"] if info else None

    def add_key(self, api_key: str, role: Role, name: str = ""):
        self._keys[api_key] = {"role": role, "name": name}
        self._save()

    def remove_key(self, api_key: str) -> bool:
        if api_key in self._keys:
            del self._keys[api_key]
            self._save()
            return True
        return False

    def list_keys(self) -> list[dict]:
        return [
            {"key_prefix": k[:8] + "...", "role": v["role"].value, "name": v["name"]}
            for k, v in self._keys.items()
        ]


def _match_permission(method: str, path: str) -> Optional[Role]:
    """Find the minimum required role for a request."""
    for rule_method, pattern, required_role in PERMISSION_RULES:
        # Check method (None = any method)
        if rule_method is not None and method != rule_method:
            continue
        # Check if path ends with or contains the pattern
        if path.endswith(pattern) or pattern in path.split("?")[0]:
            return required_role

    # Default: require editor for writes, viewer for reads
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return Role.EDITOR
    return Role.VIEWER


def _has_permission(user_role: Role, required_role: Optional[Role]) -> bool:
    """Check if a user's role meets the required permission level."""
    if required_role is None:
        return True  # No restriction
    return ROLE_HIERARCHY.get(user_role, 0) >= ROLE_HIERARCHY.get(required_role, 0)


class RBACMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces role-based access control."""

    def __init__(self, app, key_store: RBACKeyStore, enabled: bool = False):
        super().__init__(app)
        self.key_store = key_store
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # Extract API key from header
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")

        if not api_key:
            # Allow health endpoints without auth
            if request.url.path in ("/health", "/ready", "/ping", "/metrics"):
                return await call_next(request)
            raise HTTPException(status_code=401, detail="API key required")

        # Look up role
        role = self.key_store.get_role(api_key)
        if role is None:
            raise HTTPException(status_code=403, detail="Invalid API key")

        # Check permission
        required_role = _match_permission(request.method, request.url.path)
        if not _has_permission(role, required_role):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {required_role.value}, have: {role.value}",
            )

        # Attach role to request state for downstream use
        request.state.user_role = role
        return await call_next(request)
