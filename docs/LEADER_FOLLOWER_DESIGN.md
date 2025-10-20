# Leader-Follower Architecture - Implementation Design

## 🎯 Overview

This document outlines how to implement a Leader-Follower (Master-Replica) architecture for the Vector Database to support:
- **Read Scalability**: Distribute read load across multiple followers
- **High Availability**: Automatic failover if leader fails
- **Data Replication**: Keep followers synchronized with leader

---

## 📐 Architecture Design

### Current (Single Node)
```
┌─────────────────────────────────────┐
│         Vector DB Instance           │
│  ┌──────────────────────────────┐   │
│  │  FastAPI (Read + Write)       │   │
│  ├──────────────────────────────┤   │
│  │  LibraryRepository            │   │
│  ├──────────────────────────────┤   │
│  │  VectorStore + Indexes        │   │
│  ├──────────────────────────────┤   │
│  │  WAL + Snapshots              │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Proposed (Leader-Follower)
```
                    ┌─────────────┐
                    │ Load Balancer│
                    │   (HAProxy)  │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼─────┐  ┌──────▼─────┐  ┌─────▼──────┐
    │  Leader    │  │ Follower 1 │  │ Follower 2 │
    │ (Write)    │  │  (Read)    │  │  (Read)    │
    └──────┬─────┘  └──────▲─────┘  └─────▲──────┘
           │                │               │
           │    WAL Stream  │               │
           └────────────────┴───────────────┘
                      │
              ┌───────▼────────┐
              │  etcd/Consul   │
              │ (Coordination) │
              └────────────────┘
```

---

## 🏗️ Components to Implement

### 1. Node Types & Roles

#### Leader Node
- **Responsibilities**:
  - Accept all write operations (POST, PUT, DELETE)
  - Append to WAL
  - Update local state
  - Stream WAL entries to followers
  - Respond to read requests (optional, or redirect to followers)

#### Follower Node
- **Responsibilities**:
  - Accept read-only operations (GET, search)
  - Receive WAL stream from leader
  - Apply WAL entries to local state
  - Maintain near-real-time replica
  - Participate in leader election if leader fails

---

### 2. Leader Election

#### Using etcd (Recommended)

**Why etcd**:
- Built-in leader election via leases
- Strongly consistent
- Battle-tested (used by Kubernetes)
- Python client available

**Implementation**:

```python
# infrastructure/cluster/leader_election.py

import etcd3
import threading
import time
from typing import Callable, Optional

class LeaderElection:
    """
    Leader election using etcd lease mechanism.

    Uses etcd's lease TTL with keepalive to maintain leadership.
    Other nodes watch for leader failure and compete to become leader.
    """

    def __init__(
        self,
        etcd_host: str = "localhost",
        etcd_port: int = 2379,
        node_id: str = None,
        lease_ttl: int = 10,
    ):
        """
        Initialize leader election.

        Args:
            etcd_host: etcd server host
            etcd_port: etcd server port
            node_id: Unique identifier for this node
            lease_ttl: Lease time-to-live in seconds
        """
        self.client = etcd3.client(host=etcd_host, port=etcd_port)
        self.node_id = node_id or self._get_node_id()
        self.lease_ttl = lease_ttl
        self.lease = None
        self.is_leader = False
        self._stop_event = threading.Event()
        self._leader_thread = None

    def _get_node_id(self) -> str:
        """Generate unique node ID."""
        import socket
        import uuid
        hostname = socket.gethostname()
        return f"{hostname}-{uuid.uuid4().hex[:8]}"

    def run_for_leader(self, on_elected: Callable, on_lost: Callable):
        """
        Attempt to become leader and maintain leadership.

        Args:
            on_elected: Callback when this node becomes leader
            on_lost: Callback when this node loses leadership
        """
        self._leader_thread = threading.Thread(
            target=self._leader_loop,
            args=(on_elected, on_lost),
            daemon=True
        )
        self._leader_thread.start()

    def _leader_loop(self, on_elected: Callable, on_lost: Callable):
        """Main leader election loop."""
        while not self._stop_event.is_set():
            try:
                # Try to acquire leadership
                self.lease = self.client.lease(ttl=self.lease_ttl)

                # Try to put our node ID with the lease
                # This will only succeed if the key doesn't exist
                success = self.client.transaction(
                    compare=[
                        self.client.transactions.create('/vector-db/leader') == '0'
                    ],
                    success=[
                        self.client.transactions.put(
                            '/vector-db/leader',
                            self.node_id,
                            lease=self.lease
                        )
                    ],
                    failure=[]
                )

                if success:
                    # We are now the leader!
                    self.is_leader = True
                    on_elected()

                    # Keep refreshing lease while we're leader
                    self._maintain_leadership()

                    # Lost leadership
                    self.is_leader = False
                    on_lost()
                else:
                    # Someone else is leader, wait and retry
                    time.sleep(self.lease_ttl / 2)

            except Exception as e:
                logging.error(f"Leader election error: {e}")
                if self.is_leader:
                    self.is_leader = False
                    on_lost()
                time.sleep(1)

    def _maintain_leadership(self):
        """Keep refreshing lease to maintain leadership."""
        refresh_interval = self.lease_ttl / 3

        while not self._stop_event.is_set():
            try:
                # Refresh the lease
                self.lease.refresh()
                time.sleep(refresh_interval)
            except Exception as e:
                # Lost connection to etcd or lease expired
                logging.error(f"Failed to refresh lease: {e}")
                break

    def stop(self):
        """Stop leader election."""
        self._stop_event.set()
        if self.lease:
            self.lease.revoke()

    def get_leader(self) -> Optional[str]:
        """Get current leader node ID."""
        try:
            value, _ = self.client.get('/vector-db/leader')
            return value.decode() if value else None
        except:
            return None
```

---

### 3. WAL Streaming

#### Leader Side: Stream Producer

```python
# infrastructure/cluster/wal_streamer.py

import grpc
from concurrent import futures
import threading
import queue
from typing import List

from infrastructure.persistence.wal import WALEntry

# Define gRPC service (replication.proto)
"""
syntax = "proto3";

service ReplicationService {
  rpc StreamWAL(StreamRequest) returns (stream WALEntry);
  rpc GetSnapshot(SnapshotRequest) returns (Snapshot);
}

message StreamRequest {
  string follower_id = 1;
  int64 last_wal_position = 2;
}

message WALEntry {
  int64 position = 1;
  string operation_type = 2;
  string data = 3;
  string timestamp = 4;
}
"""

class WALStreamer:
    """
    Stream WAL entries from leader to followers.

    Maintains a queue of recent WAL entries and streams them
    to connected followers via gRPC.
    """

    def __init__(self, wal, max_queue_size: int = 10000):
        """
        Initialize WAL streamer.

        Args:
            wal: WriteAheadLog instance
            max_queue_size: Maximum entries to keep in memory
        """
        self.wal = wal
        self.entry_queue = queue.Queue(maxsize=max_queue_size)
        self.followers = {}  # follower_id -> last_position
        self._position = 0

    def append_entry(self, entry: WALEntry):
        """
        Append WAL entry and queue for streaming.

        Called after every write operation.
        """
        self._position += 1
        try:
            self.entry_queue.put_nowait((self._position, entry))
        except queue.Full:
            # Queue full, follower catching up needed
            pass

    def stream_to_follower(self, follower_id: str, start_position: int):
        """
        Stream WAL entries to a follower.

        Yields entries from start_position to current.
        """
        # First, catch up from WAL files if needed
        if start_position < self._position - self.entry_queue.qsize():
            # Follower is too far behind, need to read from disk
            all_entries = self.wal.read_all()
            for i, entry in enumerate(all_entries[start_position:]):
                yield (start_position + i + 1, entry)

        # Then stream from memory queue
        temp_queue = list(self.entry_queue.queue)
        for pos, entry in temp_queue:
            if pos > start_position:
                yield (pos, entry)

        # Update follower position
        self.followers[follower_id] = self._position
```

#### Follower Side: Stream Consumer

```python
# infrastructure/cluster/wal_consumer.py

import grpc
import logging
from typing import Callable

class WALConsumer:
    """
    Consume WAL stream from leader.

    Receives WAL entries via gRPC and applies them locally.
    """

    def __init__(self, leader_address: str, follower_id: str):
        """
        Initialize WAL consumer.

        Args:
            leader_address: gRPC address of leader (host:port)
            follower_id: Unique ID for this follower
        """
        self.leader_address = leader_address
        self.follower_id = follower_id
        self.last_position = 0
        self.channel = None
        self.stub = None

    def connect(self):
        """Connect to leader's gRPC service."""
        self.channel = grpc.insecure_channel(self.leader_address)
        self.stub = ReplicationServiceStub(self.channel)

    def consume_stream(self, apply_fn: Callable):
        """
        Consume WAL stream and apply entries.

        Args:
            apply_fn: Function to apply WAL entry to local state
        """
        request = StreamRequest(
            follower_id=self.follower_id,
            last_wal_position=self.last_position
        )

        try:
            for entry in self.stub.StreamWAL(request):
                # Apply entry to local state
                apply_fn(entry)
                self.last_position = entry.position

        except grpc.RpcError as e:
            logging.error(f"WAL stream error: {e}")
            # Reconnect and retry
            self.connect()
```

---

### 4. Request Routing

#### Load Balancer Configuration (HAProxy)

```
# haproxy.cfg

global
    log stdout format raw local0

defaults
    log global
    mode http
    option httplog
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

# Frontend for all requests
frontend vector_db_frontend
    bind *:8000

    # Route writes to leader
    acl is_write method POST PUT DELETE
    use_backend leader_backend if is_write

    # Route reads to followers (round-robin)
    default_backend follower_backend

# Leader backend (writes only)
backend leader_backend
    balance roundrobin
    server leader leader:8001 check

# Follower backend (reads only)
backend follower_backend
    balance roundrobin
    server follower1 follower1:8001 check
    server follower2 follower2:8001 check
    server follower3 follower3:8001 check

# Admin interface
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
```

#### Application-Level Routing

```python
# app/api/router.py

from fastapi import Request, HTTPException
import os

def check_node_role():
    """Middleware to enforce read/write routing."""
    node_role = os.getenv("NODE_ROLE", "leader")  # leader or follower

    def role_checker(request: Request):
        if node_role == "follower":
            # Follower can only handle read requests
            if request.method not in ["GET", "HEAD", "OPTIONS"]:
                raise HTTPException(
                    status_code=503,
                    detail="This is a follower node. Write requests must go to leader."
                )
        return True

    return role_checker

# In main.py
app.add_middleware(check_node_role())
```

---

### 5. Repository Layer Changes

```python
# infrastructure/repositories/replicated_repository.py

from infrastructure.repositories.library_repository import LibraryRepository
from infrastructure.cluster.wal_streamer import WALStreamer
from infrastructure.cluster.wal_consumer import WALConsumer
from infrastructure.persistence.wal import OperationType

class ReplicatedLibraryRepository(LibraryRepository):
    """
    Repository with replication support.

    Extends LibraryRepository to stream operations to followers.
    """

    def __init__(self, data_dir, is_leader: bool):
        super().__init__(data_dir)
        self.is_leader = is_leader

        if is_leader:
            self.wal_streamer = WALStreamer(self.wal)
        else:
            leader_addr = os.getenv("LEADER_ADDRESS", "leader:50051")
            self.wal_consumer = WALConsumer(leader_addr, self.node_id)
            self.wal_consumer.connect()
            # Start consuming in background
            threading.Thread(
                target=self.wal_consumer.consume_stream,
                args=(self._apply_wal_entry,),
                daemon=True
            ).start()

    def add_document(self, library_id, document):
        """Override to stream operation."""
        if not self.is_leader:
            raise RuntimeError("Only leader can accept writes")

        # Perform operation
        result = super().add_document(library_id, document)

        # Stream to followers
        entry = WALEntry(
            operation_type=OperationType.ADD_DOCUMENT,
            data={
                "library_id": str(library_id),
                "document": document.dict()
            }
        )
        self.wal_streamer.append_entry(entry)

        return result

    def _apply_wal_entry(self, entry):
        """Apply WAL entry from leader (follower only)."""
        if entry.operation_type == "add_document":
            data = entry.data
            # Reconstruct and apply operation
            from app.models.base import Document
            doc = Document(**data["document"])
            super().add_document(
                UUID(data["library_id"]),
                doc
            )
```

---

### 6. Health Checks & Monitoring

```python
# infrastructure/cluster/health_check.py

from fastapi import APIRouter
from enum import Enum

class NodeRole(str, Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"

router = APIRouter()

@router.get("/cluster/status")
async def cluster_status():
    """Return cluster status."""
    return {
        "node_id": node_id,
        "role": current_role,
        "is_healthy": True,
        "leader": leader_election.get_leader(),
        "replication_lag": get_replication_lag(),
        "connected_followers": len(wal_streamer.followers) if is_leader else 0
    }

@router.get("/cluster/nodes")
async def list_nodes():
    """List all nodes in cluster."""
    # Query etcd for all registered nodes
    nodes = []
    for value, metadata in client.get_prefix('/vector-db/nodes/'):
        nodes.append(json.loads(value.decode()))
    return {"nodes": nodes}

def get_replication_lag() -> float:
    """Calculate replication lag in seconds."""
    if is_leader:
        return 0.0
    else:
        # Compare last applied position with leader's position
        leader_pos = get_leader_position()
        lag_entries = leader_pos - wal_consumer.last_position
        # Estimate: ~100 entries per second
        return lag_entries / 100.0
```

---

## 🔧 Implementation Steps

### Phase 1: Foundation (Week 1)
1. ✅ Add etcd to docker-compose
2. ✅ Implement LeaderElection class
3. ✅ Test leader election with 3 nodes
4. ✅ Add NODE_ROLE environment variable
5. ✅ Implement role-based request routing

### Phase 2: Replication (Week 2)
1. ✅ Define gRPC protobuf for replication
2. ✅ Implement WALStreamer (leader side)
3. ✅ Implement WALConsumer (follower side)
4. ✅ Test WAL streaming between nodes
5. ✅ Implement apply_wal_entry logic

### Phase 3: Integration (Week 3)
1. ✅ Extend LibraryRepository for replication
2. ✅ Update all write operations to stream WAL
3. ✅ Implement follower catch-up mechanism
4. ✅ Add snapshot transfer for new followers
5. ✅ Test full replication flow

### Phase 4: High Availability (Week 4)
1. ✅ Implement automatic leader failover
2. ✅ Test follower promotion to leader
3. ✅ Add health checks and monitoring
4. ✅ Implement graceful shutdown
5. ✅ Load testing with 3+ nodes

### Phase 5: Production Ready (Week 5)
1. ✅ Add metrics and monitoring
2. ✅ Implement replication lag alerts
3. ✅ Add cluster management CLI
4. ✅ Documentation and runbooks
5. ✅ Chaos engineering tests

---

## 📦 New Dependencies

```txt
# requirements-cluster.txt
etcd3==0.12.0          # Leader election
grpcio==1.59.0         # Replication protocol
grpcio-tools==1.59.0   # Protocol buffers compiler
protobuf==4.24.4       # Protocol buffers
consul==1.1.0          # Alternative to etcd (optional)
```

---

## 🐳 Docker Compose Updates

```yaml
# docker-compose-cluster.yml

version: '3.8'

services:
  # etcd for coordination
  etcd:
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379
      - ETCD_ADVERTISE_CLIENT_URLS=http://etcd:2379
    ports:
      - "2379:2379"
    networks:
      - vector-db-network

  # Leader node
  vector-db-leader:
    build: .
    environment:
      - NODE_ROLE=leader
      - NODE_ID=leader-1
      - ETCD_HOST=etcd
      - GRPC_PORT=50051
      - API_PORT=8001
      - COHERE_API_KEY=${COHERE_API_KEY}
    ports:
      - "8001:8001"
      - "50051:50051"
    depends_on:
      - etcd
    networks:
      - vector-db-network

  # Follower node 1
  vector-db-follower-1:
    build: .
    environment:
      - NODE_ROLE=follower
      - NODE_ID=follower-1
      - ETCD_HOST=etcd
      - LEADER_ADDRESS=vector-db-leader:50051
      - API_PORT=8001
      - COHERE_API_KEY=${COHERE_API_KEY}
    ports:
      - "8002:8001"
    depends_on:
      - etcd
      - vector-db-leader
    networks:
      - vector-db-network

  # Follower node 2
  vector-db-follower-2:
    build: .
    environment:
      - NODE_ROLE=follower
      - NODE_ID=follower-2
      - ETCD_HOST=etcd
      - LEADER_ADDRESS=vector-db-leader:50051
      - API_PORT=8001
      - COHERE_API_KEY=${COHERE_API_KEY}
    ports:
      - "8003:8001"
    depends_on:
      - etcd
      - vector-db-leader
    networks:
      - vector-db-network

  # HAProxy load balancer
  haproxy:
    image: haproxy:2.8
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
    ports:
      - "8000:8000"  # Main API
      - "8404:8404"  # Stats UI
    depends_on:
      - vector-db-leader
      - vector-db-follower-1
      - vector-db-follower-2
    networks:
      - vector-db-network

networks:
  vector-db-network:
    driver: bridge
```

---

## 🧪 Testing Strategy

### 1. Leader Election Tests
```python
def test_leader_election():
    """Test that exactly one node becomes leader."""
    nodes = start_cluster(num_nodes=3)
    time.sleep(5)  # Wait for election

    leaders = [n for n in nodes if n.is_leader]
    assert len(leaders) == 1

def test_leader_failover():
    """Test automatic failover when leader dies."""
    nodes = start_cluster(num_nodes=3)
    original_leader = [n for n in nodes if n.is_leader][0]

    # Kill leader
    original_leader.stop()

    # Wait for new election
    time.sleep(15)

    # Verify new leader elected
    remaining = [n for n in nodes if n.is_alive]
    leaders = [n for n in remaining if n.is_leader]
    assert len(leaders) == 1
    assert leaders[0] != original_leader
```

### 2. Replication Tests
```python
def test_replication():
    """Test that writes replicate to followers."""
    cluster = start_cluster(num_nodes=3)
    leader = get_leader(cluster)

    # Write to leader
    doc_id = leader.add_document("Test")

    # Wait for replication
    time.sleep(1)

    # Verify on all followers
    for node in get_followers(cluster):
        doc = node.get_document(doc_id)
        assert doc is not None
        assert doc.title == "Test"

def test_replication_lag():
    """Test replication lag under load."""
    cluster = start_cluster(num_nodes=3)
    leader = get_leader(cluster)

    # Add 1000 documents quickly
    for i in range(1000):
        leader.add_document(f"Doc {i}")

    # Measure replication lag
    max_lag = 0
    for follower in get_followers(cluster):
        lag = follower.get_replication_lag()
        max_lag = max(max_lag, lag)

    assert max_lag < 5.0  # Less than 5 seconds
```

### 3. Consistency Tests
```python
def test_read_after_write():
    """Test that reads see writes after replication."""
    cluster = start_cluster(num_nodes=3)

    # Write to leader
    doc_id = write_to_leader(cluster, "Test")

    # Wait for replication
    time.sleep(1)

    # Read from random follower
    follower = random.choice(get_followers(cluster))
    doc = follower.get_document(doc_id)

    assert doc is not None

def test_eventual_consistency():
    """Test eventual consistency guarantees."""
    cluster = start_cluster(num_nodes=3)

    # Partition network (simulate split)
    partition_network(cluster, [0], [1, 2])

    # Write to leader (node 0)
    doc_id = cluster[0].add_document("Test")

    # Heal partition
    heal_network(cluster)

    # Eventually, all nodes should have the document
    for _ in range(30):  # 30 second timeout
        if all(n.has_document(doc_id) for n in cluster):
            break
        time.sleep(1)

    assert all(n.has_document(doc_id) for n in cluster)
```

---

## 📊 Monitoring & Observability

### Metrics to Track
```python
# Key metrics
- replication_lag_seconds (by follower)
- wal_entries_per_second
- leader_elections_total
- follower_restarts_total
- failed_replication_attempts_total
- cluster_size
- healthy_nodes

# Use Prometheus + Grafana
from prometheus_client import Counter, Gauge, Histogram

replication_lag = Gauge(
    'vector_db_replication_lag_seconds',
    'Replication lag in seconds',
    ['node_id']
)

wal_rate = Counter(
    'vector_db_wal_entries_total',
    'Total WAL entries processed'
)
```

---

## 🚀 Benefits

### With Leader-Follower:
✅ **Read Scalability**: Distribute reads across N followers
✅ **High Availability**: Auto-failover in < 15 seconds
✅ **Data Durability**: Multiple copies of data
✅ **Load Distribution**: Leader handles writes, followers handle reads
✅ **Geographic Distribution**: Followers in different regions (future)

### Trade-offs:
⚠️ **Complexity**: Significantly more complex system
⚠️ **Eventual Consistency**: Followers may lag behind leader
⚠️ **Network Dependency**: Requires reliable network
⚠️ **Cost**: More servers needed
⚠️ **Operational Overhead**: More monitoring and management

---

## 📝 Summary

### Estimated Implementation Time
- **Phase 1-3**: 2-3 weeks (core functionality)
- **Phase 4-5**: 1-2 weeks (production hardening)
- **Total**: ~1 month for production-ready implementation

### Key Files to Create
1. `infrastructure/cluster/leader_election.py` (~300 lines)
2. `infrastructure/cluster/wal_streamer.py` (~400 lines)
3. `infrastructure/cluster/wal_consumer.py` (~300 lines)
4. `infrastructure/cluster/health_check.py` (~200 lines)
5. `infrastructure/repositories/replicated_repository.py` (~500 lines)
6. `replication.proto` (~50 lines)
7. `docker-compose-cluster.yml` (~150 lines)
8. `haproxy.cfg` (~50 lines)

**Total New Code**: ~2,000 lines

### Dependencies
- etcd or Consul for coordination
- gRPC for replication protocol
- HAProxy for load balancing
- Prometheus for monitoring

---

This design provides a production-grade Leader-Follower implementation that leverages our existing WAL infrastructure and adds the distributed systems components needed for high availability and read scalability.
