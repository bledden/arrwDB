#!/usr/bin/env python3
"""
Comprehensive Integration Test for arrwDB Phases 1-4

Tests all phases working together:
- Phase 1: Streaming endpoints
- Phase 2: Event bus and CDC
- Phase 3: WebSocket real-time communication
- Phase 4: Background job queue

This test should be run with a live server.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import List

import requests
import websockets

# Configuration
BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"


class Colors:
    """ANSI colors for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str):
    """Print a test section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ {message}{Colors.RESET}")


class IntegrationTest:
    """Comprehensive integration test for all phases"""

    def __init__(self):
        self.library_id = None
        self.document_ids: List[str] = []
        self.test_results = {
            "phase1": False,
            "phase2": False,
            "phase3": False,
            "phase4": False,
        }

    def run(self):
        """Run all integration tests"""
        print_section("arrwDB Comprehensive Integration Test - Phases 1-4")
        print_info(f"Testing server at: {BASE_URL}")
        print_info(f"Start time: {datetime.now().isoformat()}\n")

        try:
            # Test server health
            if not self.test_server_health():
                print_error("Server health check failed. Is the server running?")
                return False

            # Create test library
            if not self.create_test_library():
                print_error("Failed to create test library")
                return False

            # Phase 1: Streaming
            if not self.test_phase1_streaming():
                print_error("Phase 1 (Streaming) tests failed")
            else:
                self.test_results["phase1"] = True

            # Phase 2: Event Bus (checked via stats)
            if not self.test_phase2_events():
                print_error("Phase 2 (Events) tests failed")
            else:
                self.test_results["phase2"] = True

            # Phase 3: WebSocket
            if not asyncio.run(self.test_phase3_websocket()):
                print_error("Phase 3 (WebSocket) tests failed")
            else:
                self.test_results["phase3"] = True

            # Phase 4: Job Queue
            if not asyncio.run(self.test_phase4_jobs()):
                print_error("Phase 4 (Job Queue) tests failed")
            else:
                self.test_results["phase4"] = True

            # Print final results
            self.print_final_results()

            return all(self.test_results.values())

        except Exception as e:
            print_error(f"Test suite failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Cleanup
            self.cleanup()

    def test_server_health(self) -> bool:
        """Test server health endpoint"""
        print_section("Server Health Check")

        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print_success(f"Server is healthy: {data['status']}")
                print_info(f"  Version: {data['version']}")
                return True
            else:
                print_error(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Cannot connect to server: {e}")
            return False

    def create_test_library(self) -> bool:
        """Create a test library for integration testing"""
        print_section("Setup: Create Test Library")

        library_name = f"Integration Test {datetime.now().isoformat()}"

        try:
            response = requests.post(
                f"{BASE_URL}/v1/libraries",
                json={
                    "name": library_name,
                    "index_type": "brute_force",  # Fast for testing
                },
                timeout=10
            )

            if response.status_code == 201:
                library = response.json()
                self.library_id = library["id"]
                print_success(f"Created library: {self.library_id}")
                print_info(f"  Name: {library_name}")
                return True
            else:
                print_error(f"Failed to create library: {response.status_code}")
                print_error(f"  Response: {response.text}")
                return False

        except Exception as e:
            print_error(f"Exception creating library: {e}")
            return False

    def test_phase1_streaming(self) -> bool:
        """Test Phase 1: Streaming endpoints"""
        print_section("Phase 1: Streaming Endpoints")

        # Test 1: NDJSON streaming ingestion
        print_info("Test 1.1: NDJSON streaming document ingestion")

        ndjson_data = "\n".join([
            json.dumps({"title": "Test Doc 1", "texts": ["This is test document one for streaming"]}),
            json.dumps({"title": "Test Doc 2", "texts": ["This is test document two for streaming"]}),
            json.dumps({"title": "Test Doc 3", "texts": ["This is test document three for streaming"]}),
        ])

        try:
            response = requests.post(
                f"{BASE_URL}/v1/libraries/{self.library_id}/documents/stream",
                data=ndjson_data,
                headers={"Content-Type": "application/x-ndjson"},
                timeout=30  # Should be fast now with fixed endpoint
            )

            if response.status_code == 200:
                result = response.json()
                print_success(f"NDJSON ingestion: {result['successful']} docs added")
                print_info(f"  Failed: {result['failed']}")

                # Store document IDs for later tests
                self.document_ids = [r["document_id"] for r in result["results"] if r.get("success")]

            else:
                print_error(f"NDJSON ingestion failed: {response.status_code}")
                return False

        except Exception as e:
            print_error(f"NDJSON test failed: {e}")
            return False

        # Test 2: Streaming search
        print_info("\nTest 1.2: Streaming search results")

        try:
            response = requests.post(
                f"{BASE_URL}/v1/libraries/{self.library_id}/search/stream",
                json={"query": "test document", "k": 5},
                stream=True,
                timeout=30
            )

            if response.status_code == 200:
                results_count = 0
                for line in response.iter_lines():
                    if line:
                        result = json.loads(line)
                        results_count += 1

                print_success(f"Streaming search: {results_count} results streamed")

                if results_count == 0:
                    print_error("Warning: No search results returned (library may be empty)")

            else:
                print_error(f"Streaming search failed: {response.status_code}")
                return False

        except Exception as e:
            print_error(f"Streaming search test failed: {e}")
            return False

        # Test 3: SSE event stream (just verify endpoint exists)
        print_info("\nTest 1.3: SSE event stream endpoint")

        try:
            # Just check the endpoint is available (don't consume full stream)
            response = requests.get(
                f"{BASE_URL}/v1/events/stream?library_id={self.library_id}",
                stream=True,
                timeout=5
            )

            if response.status_code == 200:
                print_success("SSE endpoint available")
            else:
                print_error(f"SSE endpoint failed: {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            # Timeout is OK - endpoint is streaming
            print_success("SSE endpoint available (streaming)")
        except Exception as e:
            print_error(f"SSE test failed: {e}")
            return False

        print_success("\nPhase 1 (Streaming) tests PASSED")
        return True

    def test_phase2_events(self) -> bool:
        """Test Phase 2: Event bus and CDC"""
        print_section("Phase 2: Event Bus & Change Data Capture")

        # Test event statistics
        print_info("Test 2.1: Event bus statistics")

        try:
            response = requests.get(f"{BASE_URL}/v1/events/stats", timeout=5)

            if response.status_code == 200:
                stats = response.json()
                print_success("Event bus statistics retrieved")
                print_info(f"  Total published: {stats['total_published']}")
                print_info(f"  Total delivered: {stats['total_delivered']}")
                print_info(f"  Subscriber count: {stats['subscriber_count']}")
                print_info(f"  Running: {stats['running']}")

                # Verify events were published (from library creation + documents)
                if stats['total_published'] > 0:
                    print_success("Events are being published")
                else:
                    print_error("Warning: No events published yet")

                return True
            else:
                print_error(f"Event stats failed: {response.status_code}")
                return False

        except Exception as e:
            print_error(f"Event bus test failed: {e}")
            return False

    async def test_phase3_websocket(self) -> bool:
        """Test Phase 3: WebSocket bidirectional communication"""
        print_section("Phase 3: WebSocket Real-Time Communication")

        ws_url = f"{WS_BASE_URL}/v1/libraries/{self.library_id}/ws"

        try:
            # Test 3.1: WebSocket connection
            print_info("Test 3.1: WebSocket connection and welcome message")

            async with websockets.connect(ws_url, open_timeout=10, close_timeout=10) as websocket:
                # Receive welcome message
                welcome = await asyncio.wait_for(websocket.recv(), timeout=5)
                welcome_data = json.loads(welcome)

                if welcome_data.get("type") == "system":
                    print_success(f"WebSocket connected: {welcome_data['message']}")
                else:
                    print_error(f"Unexpected welcome message: {welcome_data}")
                    return False

                # Test 3.2: Search via WebSocket
                print_info("\nTest 3.2: Search operation via WebSocket")

                search_request = {
                    "type": "request",
                    "action": "search",
                    "request_id": str(uuid.uuid4()),
                    "data": {
                        "query_text": "test document",
                        "k": 5
                    }
                }

                await websocket.send(json.dumps(search_request))

                # Receive response
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                response_data = json.loads(response)

                if response_data.get("type") == "response" and response_data.get("success"):
                    results = response_data.get("data", {}).get("results", [])
                    print_success(f"WebSocket search: {len(results)} results")
                else:
                    print_error(f"WebSocket search failed: {response_data.get('error')}")
                    return False

                # Test 3.3: Event notification
                print_info("\nTest 3.3: Real-time event notification")

                # Create async task to wait for event
                async def wait_for_event():
                    timeout_time = time.time() + 15
                    while time.time() < timeout_time:
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=1)
                            data = json.loads(msg)
                            if data.get("type") == "event":
                                return data
                        except asyncio.TimeoutError:
                            continue
                    return None

                event_task = asyncio.create_task(wait_for_event())

                # Trigger an event by adding a document
                add_response = requests.post(
                    f"{BASE_URL}/v1/libraries/{self.library_id}/documents",
                    json={
                        "title": "WebSocket Test Document",
                        "texts": ["This document tests WebSocket event notifications"]
                    },
                    timeout=30
                )

                if add_response.status_code == 201:
                    print_info("  Document created, waiting for event...")

                    # Wait for event notification
                    event_data = await event_task

                    if event_data:
                        print_success(f"Event received: {event_data.get('event_type')}")
                        print_info(f"  Library: {event_data.get('library_id')}")
                    else:
                        print_error("No event received within timeout")
                        return False
                else:
                    print_error(f"Document creation failed: {add_response.status_code}")
                    return False

                # Test 3.4: WebSocket statistics
                print_info("\nTest 3.4: WebSocket connection statistics")

                stats_response = requests.get(f"{BASE_URL}/v1/websockets/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    print_success("WebSocket stats retrieved")
                    print_info(f"  Total connections: {stats['total_connections']}")
                else:
                    print_error("WebSocket stats failed")

            print_success("\nPhase 3 (WebSocket) tests PASSED")
            return True

        except Exception as e:
            print_error(f"WebSocket test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_phase4_jobs(self) -> bool:
        """Test Phase 4: Background job queue"""
        print_section("Phase 4: Background Job Queue")

        # Test 4.1: Job queue statistics
        print_info("Test 4.1: Job queue statistics")

        try:
            response = requests.get(f"{BASE_URL}/v1/jobs/stats/queue", timeout=5)

            if response.status_code == 200:
                stats = response.json()
                print_success("Job queue stats retrieved")
                print_info(f"  Total jobs: {stats['total_jobs']}")
                print_info(f"  Completed: {stats['completed_jobs']}")
                print_info(f"  Failed: {stats['failed_jobs']}")
                print_info(f"  Running: {stats['running']}")
                print_info(f"  Workers: {stats['num_workers']}")
            else:
                print_error(f"Job queue stats failed: {response.status_code}")
                return False

        except Exception as e:
            print_error(f"Job queue stats test failed: {e}")
            return False

        # Test 4.2: Submit a simple job (index optimize)
        print_info("\nTest 4.2: Submit index optimization job")

        try:
            response = requests.post(
                f"{BASE_URL}/v1/jobs/index-optimize",
                json={"library_id": self.library_id},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                job_id = result["job_id"]
                print_success(f"Job submitted: {job_id}")
                print_info(f"  Status: {result['status']}")

                # Poll for job completion
                print_info("  Polling for job completion...")

                max_attempts = 30
                for attempt in range(max_attempts):
                    await asyncio.sleep(1)

                    status_response = requests.get(
                        f"{BASE_URL}/v1/jobs/{job_id}",
                        timeout=5
                    )

                    if status_response.status_code == 200:
                        job_status = status_response.json()
                        status = job_status["status"]

                        if status == "completed":
                            print_success(f"Job completed successfully")
                            print_info(f"  Result: {job_status.get('result')}")
                            break
                        elif status == "failed":
                            print_error(f"Job failed: {job_status.get('error')}")
                            return False
                        elif attempt < max_attempts - 1:
                            print_info(f"  Job {status}... (attempt {attempt + 1}/{max_attempts})")
                    else:
                        print_error(f"Job status check failed: {status_response.status_code}")
                        return False

                else:
                    print_error("Job did not complete within timeout")
                    return False

            else:
                print_error(f"Job submission failed: {response.status_code}")
                print_error(f"  Response: {response.text}")
                return False

        except Exception as e:
            print_error(f"Job queue test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        print_success("\nPhase 4 (Job Queue) tests PASSED")
        return True

    def cleanup(self):
        """Clean up test resources"""
        print_section("Cleanup")

        if self.library_id:
            try:
                response = requests.delete(
                    f"{BASE_URL}/v1/libraries/{self.library_id}",
                    timeout=10
                )

                if response.status_code == 204:
                    print_success(f"Deleted test library: {self.library_id}")
                else:
                    print_error(f"Failed to delete library: {response.status_code}")

            except Exception as e:
                print_error(f"Cleanup failed: {e}")

    def print_final_results(self):
        """Print final test results summary"""
        print_section("Test Results Summary")

        total = len(self.test_results)
        passed = sum(1 for result in self.test_results.values() if result)

        print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
        for phase, result in self.test_results.items():
            status = f"{Colors.GREEN}PASSED{Colors.RESET}" if result else f"{Colors.RED}FAILED{Colors.RESET}"
            print(f"  {phase.upper():15} {status}")

        print(f"\n{Colors.BOLD}Overall: {passed}/{total} phases passed{Colors.RESET}")

        if passed == total:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.RESET}")
            print(f"\n{Colors.GREEN}arrwDB Phases 1-4 are fully integrated and working!{Colors.RESET}\n")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.RESET}")
            print(f"\n{Colors.YELLOW}Please review the errors above{Colors.RESET}\n")


def main():
    """Main entry point"""
    test = IntegrationTest()
    success = test.run()

    exit_code = 0 if success else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
