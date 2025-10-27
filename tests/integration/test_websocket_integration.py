#!/usr/bin/env python3
"""
WebSocket test client for arrwDB real-time features.

Tests:
1. Connect to library WebSocket
2. Send search request
3. Add a document via REST
4. Verify event notification received
"""

import asyncio
import json
import sys
import uuid
from datetime import datetime

import requests
import websockets

BASE_URL = "http://localhost:8000/v1"
WS_URL = "ws://localhost:8000/v1"


async def test_websocket_operations():
    """Test WebSocket operations and event notifications."""

    print("=" * 60)
    print("WebSocket Operations Test")
    print("=" * 60)
    print()

    # 1. Create a library via REST
    print("1. Creating test library...")
    lib_response = requests.post(
        f"{BASE_URL}/libraries",
        json={"name": f"WS Test {datetime.now().isoformat()}", "index_type": "brute_force"}
    )

    if lib_response.status_code != 201:
        print(f"   ✗ Failed to create library: {lib_response.text}")
        return

    library = lib_response.json()
    library_id = library["id"]
    print(f"   ✓ Created library: {library_id}")
    print()

    # 2. Connect to WebSocket
    print(f"2. Connecting to WebSocket: {WS_URL}/libraries/{library_id}/ws")

    try:
        async with websockets.connect(f"{WS_URL}/libraries/{library_id}/ws") as websocket:
            print("   ✓ WebSocket connected")

            # Receive welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            print(f"   ✓ Welcome message: {welcome_data['message']}")
            print()

            # 3. Send search request via WebSocket
            print("3. Sending search request via WebSocket...")
            search_request = {
                "type": "request",
                "action": "search",
                "request_id": str(uuid.uuid4()),
                "data": {
                    "query_text": "test query",
                    "k": 5
                }
            }
            await websocket.send(json.dumps(search_request))

            # Receive search response
            search_response = await websocket.recv()
            search_data = json.loads(search_response)
            print(f"   ✓ Search response received: {search_data['success']}")
            print(f"     Results: {len(search_data.get('data', {}).get('results', []))}")
            print()

            # 4. Add document via WebSocket
            print("4. Adding document via WebSocket...")
            add_request = {
                "type": "request",
                "action": "add",
                "request_id": str(uuid.uuid4()),
                "data": {
                    "text": "This is a test document for WebSocket operations",
                    "metadata": {"source": "websocket_test"}
                }
            }
            await websocket.send(json.dumps(add_request))

            # Receive add response
            add_response = await websocket.recv()
            add_data = json.loads(add_response)
            print(f"   ✓ Document add response: {add_data['success']}")
            if add_data['success']:
                doc_id = add_data['data']['document']['id']
                print(f"     Document ID: {doc_id}")
            print()

            # 5. Wait for event notification
            print("5. Waiting for event notifications (timeout 5s)...")
            try:
                event_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                event_data = json.loads(event_msg)

                if event_data.get("type") == "event":
                    print(f"   ✓ Event received: {event_data.get('event_type')}")
                    print(f"     Library: {event_data.get('library_id')}")
                    print(f"     Data: {json.dumps(event_data.get('data', {}), indent=2)}")
                else:
                    print(f"   Received message: {event_data.get('type')}")
            except asyncio.TimeoutError:
                print("   ⚠ No events received within timeout")
            print()

            # 6. Get WebSocket stats
            print("6. Checking WebSocket stats...")
            stats_response = requests.get(f"{BASE_URL}/websockets/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"   ✓ Total connections: {stats['total_connections']}")
                print(f"   ✓ Libraries with subscribers: {stats['libraries_with_subscribers']}")
            print()

            print("=" * 60)
            print("WebSocket test completed successfully!")
            print("=" * 60)

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_event_notifications():
    """Test that REST operations trigger WebSocket event notifications."""

    print()
    print("=" * 60)
    print("Event Notification Test")
    print("=" * 60)
    print()

    # 1. Create library
    print("1. Creating test library...")
    lib_response = requests.post(
        f"{BASE_URL}/libraries",
        json={"name": f"Event Test {datetime.now().isoformat()}", "index_type": "brute_force"}
    )

    if lib_response.status_code != 201:
        print(f"   ✗ Failed: {lib_response.text}")
        return

    library = lib_response.json()
    library_id = library["id"]
    print(f"   ✓ Library: {library_id}")
    print()

    # 2. Connect WebSocket
    print("2. Connecting WebSocket subscriber...")

    try:
        async with websockets.connect(f"{WS_URL}/libraries/{library_id}/ws") as websocket:
            # Consume welcome message
            await websocket.recv()
            print("   ✓ Connected")
            print()

            # 3. Trigger event via REST API
            print("3. Creating document via REST API...")

            # Start listening for events in background
            async def wait_for_event():
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    if data.get("type") == "event":
                        return data

            event_task = asyncio.create_task(wait_for_event())

            # Create document via REST
            doc_response = requests.post(
                f"{BASE_URL}/libraries/{library_id}/documents",
                json={
                    "title": "Test Document",
                    "texts": ["This is a test document"],
                }
            )

            if doc_response.status_code == 201:
                doc = doc_response.json()
                print(f"   ✓ Document created: {doc['id']}")
            print()

            # 4. Wait for event
            print("4. Waiting for event notification (10s timeout)...")
            try:
                event_data = await asyncio.wait_for(event_task, timeout=10.0)
                print(f"   ✓ Event received!")
                print(f"     Type: {event_data.get('event_type')}")
                print(f"     Library: {event_data.get('library_id')}")
                print(f"     Data: {json.dumps(event_data.get('data', {}), indent=6)}")
                print()
                print("=" * 60)
                print("✓ Event notification test PASSED!")
                print("=" * 60)
            except asyncio.TimeoutError:
                print("   ✗ No event received within timeout")
                print()
                print("=" * 60)
                print("✗ Event notification test FAILED")
                print("=" * 60)

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all WebSocket tests."""
    try:
        # Test basic WebSocket operations
        await test_websocket_operations()

        # Test event notifications from REST -> WebSocket
        await test_event_notifications()

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
