"""
Cluster infrastructure for arrwDB: replication and sharding.

Replication:
  Leader-follower with WAL streaming. The leader writes to its WAL,
  followers consume WAL entries and apply them locally. Provides
  read scaling and high availability.

Sharding:
  Consistent hashing distributes vectors across shards. Each shard
  is an independent arrwDB instance. A routing layer directs requests
  to the correct shard based on the vector/document ID hash.

Both are opt-in — single-node mode (default) has zero overhead.
"""
