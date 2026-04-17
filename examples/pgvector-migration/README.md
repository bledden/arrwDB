# pgvector → arrwDB migration example

Run pgvector and arrwDB side by side, sync embeddings over, and compare
latency on the same query.

## Stack

- `pgvector/pgvector:pg16` with a `documents` table preloaded with
  two 1024-d rows (random embeddings for demo purposes).
- `bledden/arrwdb:latest` on port 8000.
- A Python script (`migrate.py`) that copies rows into arrwDB using the
  `sync_from_postgres` helper.
- A Python script (`search.py`) that runs the same nearest-neighbor
  query through both systems and prints the timing.

## Run it

```bash
docker compose up -d
pip install "arrwdb[postgres]" psycopg[binary]

# one-shot migration — prints the new arrwDB library ID
python migrate.py
# Synced 2 rows into arrwDB library 6fd3...

# compare latency (paste the library ID from the previous output)
ARRWDB_LIBRARY_ID=6fd3... python search.py
```

## Incremental mode

`sync_from_postgres` accepts an `updated_at_column` and `since=` so the
same script can run on a cron and only pull new/changed rows. See the
docstring in `arrwdb.integrations.postgres` for details.

## Production pattern

The recommended architecture:

```
App → Postgres   ← source of truth, full SQL, ACID
         ↓ trigger / cron / logical replication
       arrwDB    ← hot-path vector search (~170x faster than pgvector)
         ↑
App ←─┘ (vector queries only)
```

arrwDB is *not* a Postgres replacement. It's a specialized side-car
that takes the one workload pgvector struggles with — high-QPS
vector search — off your primary database.
