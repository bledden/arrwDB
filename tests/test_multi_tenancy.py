"""
Test multi-tenancy and admin endpoints.

This script tests:
1. Creating tenants
2. Listing tenants
3. Getting tenant details
4. Deactivating tenants
5. Rotating API keys
6. API key validation and expiration
"""

import time
from datetime import datetime, timedelta
import requests

BASE_URL = "http://localhost:8000/v1"


def test_create_tenant():
    """Test creating a new tenant."""
    print("\n" + "=" * 60)
    print("TEST: Create Tenant")
    print("=" * 60)

    payload = {
        "name": "Test Corporation",
        "metadata": {"plan": "enterprise", "contact": "admin@test.com"},
        "expires_in_days": 365,
    }

    response = requests.post(f"{BASE_URL}/admin/tenants", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 201:
        data = response.json()
        print(f"✓ Tenant created successfully")
        print(f"  - Tenant ID: {data['tenant_id']}")
        print(f"  - Name: {data['name']}")
        print(f"  - API Key: {data['api_key']}")
        print(f"  - Created: {data['created_at']}")
        print(f"  - Expires: {data['expires_at']}")
        return data["tenant_id"], data["api_key"]
    else:
        print(f"✗ Failed: {response.text}")
        return None, None


def test_create_tenant_without_expiration():
    """Test creating a tenant without expiration."""
    print("\n" + "=" * 60)
    print("TEST: Create Tenant Without Expiration")
    print("=" * 60)

    payload = {
        "name": "Permanent Corp",
        "metadata": {"plan": "starter"},
    }

    response = requests.post(f"{BASE_URL}/admin/tenants", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 201:
        data = response.json()
        print(f"✓ Tenant created successfully")
        print(f"  - Tenant ID: {data['tenant_id']}")
        print(f"  - API Key: {data['api_key']}")
        print(f"  - Expires: {data['expires_at']} (None = never expires)")
        return data["tenant_id"], data["api_key"]
    else:
        print(f"✗ Failed: {response.text}")
        return None, None


def test_list_tenants():
    """Test listing all tenants."""
    print("\n" + "=" * 60)
    print("TEST: List All Tenants")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/admin/tenants")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        tenants = response.json()
        print(f"✓ Found {len(tenants)} tenants")
        for i, tenant in enumerate(tenants, 1):
            print(f"\n  Tenant {i}:")
            print(f"    - ID: {tenant['tenant_id']}")
            print(f"    - Name: {tenant['name']}")
            print(f"    - Active: {tenant['is_active']}")
            print(f"    - Created: {tenant['created_at']}")
            print(f"    - Expires: {tenant.get('expires_at', 'Never')}")
            print(f"    - Last Used: {tenant.get('last_used_at', 'Never')}")
            print(f"    - Request Count: {tenant['request_count']}")
            print(f"    - Metadata: {tenant.get('metadata', {})}")
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False


def test_get_tenant(tenant_id):
    """Test getting details for a specific tenant."""
    print("\n" + "=" * 60)
    print("TEST: Get Tenant Details")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/admin/tenants/{tenant_id}")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        tenant = response.json()
        print(f"✓ Tenant found")
        print(f"  - ID: {tenant['tenant_id']}")
        print(f"  - Name: {tenant['name']}")
        print(f"  - Active: {tenant['is_active']}")
        print(f"  - Request Count: {tenant['request_count']}")
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False


def test_api_key_usage(api_key):
    """Test using API key to access protected endpoints."""
    print("\n" + "=" * 60)
    print("TEST: API Key Usage")
    print("=" * 60)

    # Test with X-API-Key header
    print("\n1. Testing with X-API-Key header...")
    headers = {"X-API-Key": api_key}
    response = requests.get(f"{BASE_URL}/libraries", headers=headers)
    print(f"   Status: {response.status_code}")

    if response.status_code in [200, 404]:  # 404 is ok if no libraries yet
        print(f"   ✓ API key accepted (X-API-Key header)")
    else:
        print(f"   ✗ Failed: {response.text}")

    # Test with Authorization: Bearer header
    print("\n2. Testing with Authorization: Bearer header...")
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{BASE_URL}/libraries", headers=headers)
    print(f"   Status: {response.status_code}")

    if response.status_code in [200, 404]:
        print(f"   ✓ API key accepted (Authorization: Bearer header)")
    else:
        print(f"   ✗ Failed: {response.text}")

    # Test with invalid key
    print("\n3. Testing with invalid API key...")
    headers = {"X-API-Key": "arrw_invalid_key_1234567890abcdef"}
    response = requests.get(f"{BASE_URL}/libraries", headers=headers)
    print(f"   Status: {response.status_code}")

    if response.status_code == 401:
        print(f"   ✓ Invalid key correctly rejected")
    else:
        print(f"   ✗ Expected 401, got {response.status_code}")


def test_rotate_api_key(tenant_id, old_api_key):
    """Test rotating a tenant's API key."""
    print("\n" + "=" * 60)
    print("TEST: Rotate API Key")
    print("=" * 60)

    response = requests.post(f"{BASE_URL}/admin/tenants/{tenant_id}/rotate")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        new_api_key = data["new_api_key"]
        print(f"✓ API key rotated successfully")
        print(f"  - Old Key: {old_api_key[:20]}...")
        print(f"  - New Key: {new_api_key[:20]}...")
        print(f"  - Rotated At: {data['rotated_at']}")

        # Test that old key no longer works
        print("\n  Testing old key (should fail)...")
        headers = {"X-API-Key": old_api_key}
        response = requests.get(f"{BASE_URL}/libraries", headers=headers)
        if response.status_code == 401:
            print(f"  ✓ Old key correctly invalidated")
        else:
            print(f"  ✗ Old key still works (unexpected)")

        # Test that new key works
        print("\n  Testing new key (should work)...")
        headers = {"X-API-Key": new_api_key}
        response = requests.get(f"{BASE_URL}/libraries", headers=headers)
        if response.status_code in [200, 404]:
            print(f"  ✓ New key works correctly")
        else:
            print(f"  ✗ New key doesn't work")

        return new_api_key
    else:
        print(f"✗ Failed: {response.text}")
        return None


def test_deactivate_tenant(tenant_id, api_key):
    """Test deactivating a tenant."""
    print("\n" + "=" * 60)
    print("TEST: Deactivate Tenant")
    print("=" * 60)

    response = requests.delete(f"{BASE_URL}/admin/tenants/{tenant_id}")
    print(f"Status: {response.status_code}")

    if response.status_code == 204:
        print(f"✓ Tenant deactivated successfully")

        # Test that API key no longer works
        print("\n  Testing API key after deactivation...")
        headers = {"X-API-Key": api_key}
        response = requests.get(f"{BASE_URL}/libraries", headers=headers)
        if response.status_code == 401:
            print(f"  ✓ API key correctly rejected after deactivation")
        else:
            print(f"  ✗ API key still works (unexpected)")

        # Verify tenant still exists but is inactive
        print("\n  Checking tenant status...")
        response = requests.get(f"{BASE_URL}/admin/tenants/{tenant_id}")
        if response.status_code == 200:
            tenant = response.json()
            if not tenant['is_active']:
                print(f"  ✓ Tenant marked as inactive")
            else:
                print(f"  ✗ Tenant still marked as active")
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False


def test_usage_tracking(tenant_id, api_key):
    """Test that usage is tracked correctly."""
    print("\n" + "=" * 60)
    print("TEST: Usage Tracking")
    print("=" * 60)

    # Get initial stats
    response = requests.get(f"{BASE_URL}/admin/tenants/{tenant_id}")
    if response.status_code != 200:
        print(f"✗ Failed to get tenant: {response.text}")
        return False

    initial_tenant = response.json()
    initial_count = initial_tenant['request_count']
    print(f"Initial request count: {initial_count}")

    # Make some API calls
    print("\nMaking 5 API calls...")
    headers = {"X-API-Key": api_key}
    for i in range(5):
        requests.get(f"{BASE_URL}/libraries", headers=headers)
        print(f"  Call {i+1}/5")

    # Check updated stats
    time.sleep(0.5)  # Give it a moment to update
    response = requests.get(f"{BASE_URL}/admin/tenants/{tenant_id}")
    if response.status_code != 200:
        print(f"✗ Failed to get updated tenant: {response.text}")
        return False

    updated_tenant = response.json()
    updated_count = updated_tenant['request_count']
    last_used = updated_tenant['last_used_at']

    print(f"\nUpdated request count: {updated_count}")
    print(f"Last used at: {last_used}")

    if updated_count >= initial_count + 5:
        print(f"✓ Usage tracking working correctly (count increased by {updated_count - initial_count})")
    else:
        print(f"⚠ Usage count only increased by {updated_count - initial_count} (expected 5+)")

    if last_used:
        print(f"✓ Last used timestamp recorded")
    else:
        print(f"✗ Last used timestamp not recorded")

    return True


def main():
    """Run all multi-tenancy tests."""
    print("\n" + "=" * 60)
    print("MULTI-TENANCY & ADMIN ENDPOINT TESTS")
    print("=" * 60)
    print("\nMake sure the API server is running:")
    print("  python -m app.api.cli run-server")
    print("\nNote: Multi-tenancy must be enabled in config:")
    print("  MULTI_TENANCY_ENABLED=True")
    print("\n" + "=" * 60)

    # Wait for user confirmation
    input("\nPress Enter to start tests...")

    # Test 1: Create tenant with expiration
    tenant1_id, tenant1_key = test_create_tenant()
    if not tenant1_id:
        print("\n✗ Failed to create tenant 1, stopping tests")
        return

    # Test 2: Create tenant without expiration
    tenant2_id, tenant2_key = test_create_tenant_without_expiration()
    if not tenant2_id:
        print("\n✗ Failed to create tenant 2, stopping tests")
        return

    # Test 3: List all tenants
    test_list_tenants()

    # Test 4: Get specific tenant
    test_get_tenant(tenant1_id)

    # Test 5: API key usage
    test_api_key_usage(tenant1_key)

    # Test 6: Usage tracking
    test_usage_tracking(tenant1_id, tenant1_key)

    # Test 7: Rotate API key
    new_key = test_rotate_api_key(tenant1_id, tenant1_key)

    # Test 8: Deactivate tenant
    test_deactivate_tenant(tenant2_id, tenant2_key)

    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print("\nVerify in API docs: http://localhost:8000/docs")
    print("Check 'Admin' and 'Multi-Tenancy' tags for endpoints")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
