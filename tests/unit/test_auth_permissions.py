"""
Unit tests for the authentication permissions and RBAC system.

This module tests the OAuth 2.0-style scopes and role-based access control
implemented in app/auth/permissions.py.
"""

import pytest
from app.auth.permissions import (
    Scope,
    Role,
    ROLE_SCOPES,
    has_scope,
    validate_scopes,
    get_scopes_for_role,
    get_scopes_for_roles,
    scope_list_to_set,
)


class TestScopeEnum:
    """Test the Scope enum definitions."""

    def test_all_required_scopes_defined(self):
        """Test that all expected scopes are defined."""
        required_scopes = {
            "libraries:read",
            "libraries:write",
            "libraries:delete",
            "libraries:admin",
            "documents:read",
            "documents:write",
            "documents:delete",
            "search:read",
            "search:advanced",
            "index:read",
            "index:write",
            "index:admin",
            "tenant:read",
            "tenant:admin",
            "api_keys:manage",
            "metrics:read",
            "logs:read",
            "admin:all",
        }

        actual_scopes = {scope.value for scope in Scope}
        assert required_scopes.issubset(actual_scopes)

    def test_scope_values_follow_resource_action_format(self):
        """Test that all scopes follow the resource:action format."""
        for scope in Scope:
            assert ":" in scope.value or "_" in scope.value
            if ":" in scope.value:
                parts = scope.value.split(":")
                assert len(parts) == 2
                assert parts[0]  # resource should not be empty
                assert parts[1]  # action should not be empty


class TestRoleEnum:
    """Test the Role enum definitions."""

    def test_all_roles_defined(self):
        """Test that all expected roles are defined."""
        expected_roles = {"viewer", "user", "developer", "admin", "superuser"}
        actual_roles = {role.value for role in Role}
        assert actual_roles == expected_roles


class TestRoleScopesMapping:
    """Test the ROLE_SCOPES mapping dictionary."""

    def test_all_roles_have_scope_mappings(self):
        """Test that all roles have scope mappings defined."""
        for role in Role:
            assert role in ROLE_SCOPES
            assert isinstance(ROLE_SCOPES[role], set)
            assert len(ROLE_SCOPES[role]) > 0

    def test_viewer_role_has_read_scopes(self):
        """Test that viewer role has only read scopes."""
        viewer_scopes = ROLE_SCOPES[Role.VIEWER]
        assert Scope.LIBRARIES_READ in viewer_scopes
        assert Scope.DOCUMENTS_READ in viewer_scopes
        assert Scope.SEARCH_READ in viewer_scopes
        # Should not have write scopes
        assert Scope.LIBRARIES_WRITE not in viewer_scopes

    def test_user_role_includes_write_access(self):
        """Test that user role includes write access."""
        user_scopes = ROLE_SCOPES[Role.USER]
        assert Scope.LIBRARIES_READ in user_scopes
        assert Scope.LIBRARIES_WRITE in user_scopes
        assert Scope.DOCUMENTS_WRITE in user_scopes

    def test_developer_role_includes_index_access(self):
        """Test that developer role includes index management."""
        developer_scopes = ROLE_SCOPES[Role.DEVELOPER]
        assert Scope.SEARCH_ADVANCED in developer_scopes
        assert Scope.INDEX_READ in developer_scopes
        assert Scope.INDEX_WRITE in developer_scopes

    def test_admin_role_has_all_admin_scopes(self):
        """Test that admin role has all admin-level scopes."""
        admin_scopes = ROLE_SCOPES[Role.ADMIN]
        assert Scope.LIBRARIES_ADMIN in admin_scopes
        assert Scope.INDEX_ADMIN in admin_scopes
        assert Scope.TENANT_ADMIN in admin_scopes

    def test_superuser_role_has_admin_all(self):
        """Test that superuser role has the admin:all scope."""
        superuser_scopes = ROLE_SCOPES[Role.SUPERUSER]
        assert Scope.ADMIN_ALL in superuser_scopes


class TestHasScope:
    """Test the has_scope permission checking function."""

    def test_has_scope_with_exact_match(self):
        """Test that has_scope returns True when user has the exact scope."""
        user_scopes = {Scope.LIBRARIES_READ, Scope.DOCUMENTS_READ}
        assert has_scope(Scope.LIBRARIES_READ, user_scopes) is True
        assert has_scope(Scope.DOCUMENTS_READ, user_scopes) is True

    def test_has_scope_without_required_scope(self):
        """Test that has_scope returns False when user lacks the required scope."""
        user_scopes = {Scope.LIBRARIES_READ}
        assert has_scope(Scope.LIBRARIES_WRITE, user_scopes) is False
        assert has_scope(Scope.DOCUMENTS_READ, user_scopes) is False

    def test_has_scope_with_admin_all(self):
        """Test that admin:all grants access to all scopes."""
        user_scopes = {Scope.ADMIN_ALL}

        # Should have access to everything
        assert has_scope(Scope.LIBRARIES_READ, user_scopes) is True
        assert has_scope(Scope.LIBRARIES_WRITE, user_scopes) is True
        assert has_scope(Scope.DOCUMENTS_DELETE, user_scopes) is True
        assert has_scope(Scope.INDEX_ADMIN, user_scopes) is True
        assert has_scope(Scope.METRICS_READ, user_scopes) is True

    def test_has_scope_with_libraries_admin(self):
        """Test that libraries:admin grants all library-related scopes."""
        user_scopes = {Scope.LIBRARIES_ADMIN}

        # Should have all library scopes
        assert has_scope(Scope.LIBRARIES_READ, user_scopes) is True
        assert has_scope(Scope.LIBRARIES_WRITE, user_scopes) is True
        assert has_scope(Scope.LIBRARIES_DELETE, user_scopes) is True

        # Should NOT have non-library scopes
        assert has_scope(Scope.DOCUMENTS_WRITE, user_scopes) is False

    def test_has_scope_with_index_admin(self):
        """Test that index:admin grants all index-related scopes."""
        user_scopes = {Scope.INDEX_ADMIN}

        # Should have all index scopes
        assert has_scope(Scope.INDEX_READ, user_scopes) is True
        assert has_scope(Scope.INDEX_WRITE, user_scopes) is True

        # Should NOT have non-index scopes
        assert has_scope(Scope.SEARCH_ADVANCED, user_scopes) is False

    def test_has_scope_with_empty_user_scopes(self):
        """Test that has_scope returns False when user has no scopes."""
        user_scopes = set()
        assert has_scope(Scope.LIBRARIES_READ, user_scopes) is False

    def test_has_scope_with_multiple_scopes(self):
        """Test has_scope with users having multiple scopes."""
        user_scopes = {
            Scope.LIBRARIES_READ,
            Scope.LIBRARIES_WRITE,
            Scope.DOCUMENTS_READ,
        }

        assert has_scope(Scope.LIBRARIES_READ, user_scopes) is True
        assert has_scope(Scope.LIBRARIES_WRITE, user_scopes) is True
        assert has_scope(Scope.DOCUMENTS_READ, user_scopes) is True
        assert has_scope(Scope.DOCUMENTS_WRITE, user_scopes) is False


class TestValidateScopes:
    """Test the validate_scopes function."""

    def test_validate_scopes_with_all_required(self):
        """Test validate_scopes when user has all required scopes."""
        user_scopes = {Scope.LIBRARIES_READ, Scope.DOCUMENTS_READ, Scope.SEARCH_READ}
        required = [Scope.LIBRARIES_READ, Scope.DOCUMENTS_READ]

        assert validate_scopes(required, user_scopes) is True

    def test_validate_scopes_missing_one(self):
        """Test validate_scopes when user is missing one required scope."""
        user_scopes = {Scope.LIBRARIES_READ}
        required = [Scope.LIBRARIES_READ, Scope.DOCUMENTS_WRITE]

        assert validate_scopes(required, user_scopes) is False

    def test_validate_scopes_with_admin_all(self):
        """Test validate_scopes with admin:all scope."""
        user_scopes = {Scope.ADMIN_ALL}
        required = [Scope.LIBRARIES_WRITE, Scope.DOCUMENTS_DELETE, Scope.INDEX_ADMIN]

        assert validate_scopes(required, user_scopes) is True

    def test_validate_scopes_with_hierarchical_admin(self):
        """Test validate_scopes with hierarchical admin scopes."""
        user_scopes = {Scope.LIBRARIES_ADMIN}
        required = [Scope.LIBRARIES_READ, Scope.LIBRARIES_WRITE]

        assert validate_scopes(required, user_scopes) is True

    def test_validate_scopes_empty_required(self):
        """Test validate_scopes with no required scopes."""
        user_scopes = {Scope.LIBRARIES_READ}
        required = []

        assert validate_scopes(required, user_scopes) is True


class TestGetScopesForRole:
    """Test the get_scopes_for_role helper function."""

    def test_get_scopes_for_viewer(self):
        """Test getting scopes for viewer role."""
        scopes = get_scopes_for_role(Role.VIEWER)
        assert Scope.LIBRARIES_READ in scopes
        assert Scope.DOCUMENTS_READ in scopes
        assert Scope.SEARCH_READ in scopes
        assert len(scopes) >= 3

    def test_get_scopes_for_user(self):
        """Test getting scopes for user role."""
        scopes = get_scopes_for_role(Role.USER)
        assert Scope.LIBRARIES_READ in scopes
        assert Scope.DOCUMENTS_WRITE in scopes
        assert len(scopes) >= 4

    def test_get_scopes_for_developer(self):
        """Test getting scopes for developer role."""
        scopes = get_scopes_for_role(Role.DEVELOPER)
        assert Scope.SEARCH_ADVANCED in scopes
        assert Scope.INDEX_READ in scopes
        assert Scope.INDEX_WRITE in scopes
        assert len(scopes) >= 6

    def test_get_scopes_for_admin(self):
        """Test getting scopes for admin role."""
        scopes = get_scopes_for_role(Role.ADMIN)
        assert Scope.LIBRARIES_ADMIN in scopes
        assert Scope.INDEX_ADMIN in scopes
        assert Scope.TENANT_ADMIN in scopes

    def test_get_scopes_for_superuser(self):
        """Test getting scopes for superuser role."""
        scopes = get_scopes_for_role(Role.SUPERUSER)
        assert Scope.ADMIN_ALL in scopes


class TestGetScopesForRoles:
    """Test the get_scopes_for_roles helper function."""

    def test_get_scopes_for_multiple_roles(self):
        """Test getting combined scopes for multiple roles."""
        roles = [Role.VIEWER, Role.USER]
        scopes = get_scopes_for_roles(roles)

        # Should have scopes from both roles
        assert Scope.LIBRARIES_READ in scopes
        assert Scope.DOCUMENTS_WRITE in scopes

    def test_get_scopes_for_single_role(self):
        """Test getting scopes for a single role in list."""
        roles = [Role.DEVELOPER]
        scopes = get_scopes_for_roles(roles)

        assert Scope.INDEX_WRITE in scopes
        assert Scope.SEARCH_ADVANCED in scopes

    def test_get_scopes_for_empty_roles(self):
        """Test getting scopes for empty role list."""
        roles = []
        scopes = get_scopes_for_roles(roles)

        assert len(scopes) == 0


class TestScopeListToSet:
    """Test the scope_list_to_set helper function."""

    def test_scope_list_to_set_valid_scopes(self):
        """Test converting valid scope strings to Scope enum set."""
        scope_strings = ["libraries:read", "documents:write", "search:read"]
        scopes = scope_list_to_set(scope_strings)

        assert Scope.LIBRARIES_READ in scopes
        assert Scope.DOCUMENTS_WRITE in scopes
        assert Scope.SEARCH_READ in scopes
        assert len(scopes) == 3

    def test_scope_list_to_set_invalid_scopes(self):
        """Test that invalid scopes are silently ignored."""
        scope_strings = ["libraries:read", "invalid:scope", "documents:write"]
        scopes = scope_list_to_set(scope_strings)

        # Should only include valid scopes
        assert Scope.LIBRARIES_READ in scopes
        assert Scope.DOCUMENTS_WRITE in scopes
        assert len(scopes) == 2

    def test_scope_list_to_set_empty_list(self):
        """Test converting empty list."""
        scope_strings = []
        scopes = scope_list_to_set(scope_strings)

        assert len(scopes) == 0

    def test_scope_list_to_set_all_invalid(self):
        """Test converting list with all invalid scopes."""
        scope_strings = ["invalid:scope", "another:invalid"]
        scopes = scope_list_to_set(scope_strings)

        assert len(scopes) == 0


class TestPermissionHierarchy:
    """Test the hierarchical permission system."""

    def test_admin_scopes_grant_sub_permissions(self):
        """Test that admin scopes grant their respective sub-permissions."""
        # Test libraries:admin
        assert has_scope(Scope.LIBRARIES_READ, {Scope.LIBRARIES_ADMIN})
        assert has_scope(Scope.LIBRARIES_WRITE, {Scope.LIBRARIES_ADMIN})
        assert has_scope(Scope.LIBRARIES_DELETE, {Scope.LIBRARIES_ADMIN})

        # Test index:admin
        assert has_scope(Scope.INDEX_READ, {Scope.INDEX_ADMIN})
        assert has_scope(Scope.INDEX_WRITE, {Scope.INDEX_ADMIN})

    def test_admin_all_grants_everything(self):
        """Test that admin:all grants all possible permissions."""
        admin_all_scopes = {Scope.ADMIN_ALL}

        # Should grant access to every single scope
        test_scopes = [
            Scope.LIBRARIES_READ,
            Scope.DOCUMENTS_WRITE,
            Scope.SEARCH_ADVANCED,
            Scope.INDEX_ADMIN,
            Scope.TENANT_ADMIN,
        ]

        for scope in test_scopes:
            assert has_scope(scope, admin_all_scopes), f"admin:all should grant {scope.value}"
