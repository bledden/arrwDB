"""
Permission and scope system for granular access control.

Implements Auth0/AWS Cognito-style scopes and permissions.
"""

from enum import Enum
from typing import Set, List


class Scope(str, Enum):
    """
    OAuth 2.0-style scopes for API access control.

    Format: resource:action
    Inspired by Auth0, AWS IAM, and GitHub's permission model.
    """

    # Library permissions
    LIBRARIES_READ = "libraries:read"
    LIBRARIES_WRITE = "libraries:write"
    LIBRARIES_DELETE = "libraries:delete"
    LIBRARIES_ADMIN = "libraries:admin"  # All library operations

    # Document permissions
    DOCUMENTS_READ = "documents:read"
    DOCUMENTS_WRITE = "documents:write"
    DOCUMENTS_DELETE = "documents:delete"

    # Search permissions
    SEARCH_READ = "search:read"
    SEARCH_ADVANCED = "search:advanced"  # Hybrid search, reranking

    # Index management permissions
    INDEX_READ = "index:read"
    INDEX_WRITE = "index:write"  # Rebuild, optimize
    INDEX_ADMIN = "index:admin"  # All index operations

    # Tenant/Admin permissions
    TENANT_READ = "tenant:read"  # View own tenant info
    TENANT_ADMIN = "tenant:admin"  # Manage tenant settings
    API_KEYS_MANAGE = "api_keys:manage"  # Rotate keys

    # System admin (super user)
    ADMIN_ALL = "admin:all"  # Full system access

    # Analytics/Monitoring
    METRICS_READ = "metrics:read"
    LOGS_READ = "logs:read"


class Role(str, Enum):
    """
    Predefined roles with scope bundles.

    Makes it easier to assign common permission sets.
    """

    # Read-only access (analytics, monitoring)
    VIEWER = "viewer"

    # Standard user (read + write documents & search)
    USER = "user"

    # Developer (user + index management)
    DEVELOPER = "developer"

    # Admin (all permissions for tenant)
    ADMIN = "admin"

    # System admin (cross-tenant access)
    SUPERUSER = "superuser"


# Role to scopes mapping
ROLE_SCOPES: dict[Role, Set[Scope]] = {
    Role.VIEWER: {
        Scope.LIBRARIES_READ,
        Scope.DOCUMENTS_READ,
        Scope.SEARCH_READ,
        Scope.INDEX_READ,
        Scope.TENANT_READ,
        Scope.METRICS_READ,
    },
    Role.USER: {
        Scope.LIBRARIES_READ,
        Scope.LIBRARIES_WRITE,
        Scope.DOCUMENTS_READ,
        Scope.DOCUMENTS_WRITE,
        Scope.SEARCH_READ,
        Scope.SEARCH_ADVANCED,
        Scope.INDEX_READ,
        Scope.TENANT_READ,
    },
    Role.DEVELOPER: {
        Scope.LIBRARIES_READ,
        Scope.LIBRARIES_WRITE,
        Scope.DOCUMENTS_READ,
        Scope.DOCUMENTS_WRITE,
        Scope.DOCUMENTS_DELETE,
        Scope.SEARCH_READ,
        Scope.SEARCH_ADVANCED,
        Scope.INDEX_READ,
        Scope.INDEX_WRITE,
        Scope.INDEX_ADMIN,
        Scope.TENANT_READ,
        Scope.METRICS_READ,
        Scope.LOGS_READ,
    },
    Role.ADMIN: {
        Scope.LIBRARIES_READ,
        Scope.LIBRARIES_WRITE,
        Scope.LIBRARIES_DELETE,
        Scope.LIBRARIES_ADMIN,
        Scope.DOCUMENTS_READ,
        Scope.DOCUMENTS_WRITE,
        Scope.DOCUMENTS_DELETE,
        Scope.SEARCH_READ,
        Scope.SEARCH_ADVANCED,
        Scope.INDEX_READ,
        Scope.INDEX_WRITE,
        Scope.INDEX_ADMIN,
        Scope.TENANT_READ,
        Scope.TENANT_ADMIN,
        Scope.API_KEYS_MANAGE,
        Scope.METRICS_READ,
        Scope.LOGS_READ,
    },
    Role.SUPERUSER: {
        Scope.ADMIN_ALL,  # Implies all scopes
    },
}


def get_scopes_for_role(role: Role) -> Set[Scope]:
    """Get all scopes for a given role."""
    return ROLE_SCOPES.get(role, set())


def get_scopes_for_roles(roles: List[Role]) -> Set[Scope]:
    """Get combined scopes for multiple roles."""
    scopes = set()
    for role in roles:
        scopes.update(get_scopes_for_role(role))
    return scopes


def has_scope(required_scope: Scope, user_scopes: Set[Scope]) -> bool:
    """
    Check if user has required scope.

    Special handling:
    - admin:all grants all scopes
    - libraries:admin grants all library scopes
    - index:admin grants all index scopes
    """
    # Superuser has everything
    if Scope.ADMIN_ALL in user_scopes:
        return True

    # Check for admin scopes that grant multiple permissions
    if required_scope.value.startswith("libraries:"):
        if Scope.LIBRARIES_ADMIN in user_scopes:
            return True

    if required_scope.value.startswith("index:"):
        if Scope.INDEX_ADMIN in user_scopes:
            return True

    # Direct scope check
    return required_scope in user_scopes


def validate_scopes(required_scopes: List[Scope], user_scopes: Set[Scope]) -> bool:
    """
    Check if user has ALL required scopes.

    Args:
        required_scopes: List of scopes required for the operation
        user_scopes: Set of scopes the user has

    Returns:
        True if user has all required scopes
    """
    return all(has_scope(scope, user_scopes) for scope in required_scopes)


def scope_list_to_set(scope_strings: List[str]) -> Set[Scope]:
    """
    Convert list of scope strings to Set of Scope enums.

    Invalid scopes are silently ignored.
    """
    scopes = set()
    for scope_str in scope_strings:
        try:
            scopes.add(Scope(scope_str))
        except ValueError:
            # Invalid scope, skip it
            continue
    return scopes
