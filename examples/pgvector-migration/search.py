"""Side-by-side pgvector vs arrwDB search timing.

Runs the same query against both systems and prints latency + results.
Requires the companion docker-compose stack and a prior ``migrate.py`` run.
"""

import os
import random
import time

import psycopg
from arrwdb import ArrwDBClient

PG_URL = os.environ.get("PG_URL", "postgresql://postgres:example@localhost:5432/ragdb")
ARRWDB_URL = os.environ.get("ARRWDB_URL", "http://localhost:8000")
LIBRARY_ID = os.environ["ARRWDB_LIBRARY_ID"]  # from migrate.py output

K = 5


def random_embedding(dim: int = 1024) -> list[float]:
    return [random.random() for _ in range(dim)]


def main() -> None:
    query_vec = random_embedding()

    # --- pgvector ---
    with psycopg.connect(PG_URL) as conn:
        with conn.cursor() as cur:
            vec_str = "[" + ",".join(f"{x:.6f}" for x in query_vec) + "]"
            t0 = time.perf_counter()
            cur.execute(
                f"SELECT id, title FROM documents "
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                (vec_str, K),
            )
            pg_rows = cur.fetchall()
            pg_ms = (time.perf_counter() - t0) * 1000

    # --- arrwDB ---
    client = ArrwDBClient(ARRWDB_URL)
    t0 = time.perf_counter()
    arrw_resp = client.search_with_embedding(
        library_id=LIBRARY_ID, embedding=query_vec, k=K
    )
    arrw_ms = (time.perf_counter() - t0) * 1000

    print(f"pgvector:  {pg_ms:7.2f} ms  {len(pg_rows)} rows")
    print(f"arrwDB:    {arrw_ms:7.2f} ms  {len(arrw_resp['results'])} rows")
    print(f"speedup:   {pg_ms / max(arrw_ms, 0.001):.1f}x")


if __name__ == "__main__":
    main()
