"""One-shot migration from pgvector to arrwDB.

Reads every row from the ``documents`` table in the companion Postgres
container and loads it into a fresh arrwDB library. Run after the
docker-compose stack is up:

    python migrate.py

Follow-up searches compare the two systems side by side — see search.py.
"""

import os

from arrwdb.integrations.postgres import sync_from_postgres

PG_URL = os.environ.get(
    "PG_URL", "postgresql://postgres:example@localhost:5432/ragdb"
)
ARRWDB_URL = os.environ.get("ARRWDB_URL", "http://localhost:8000")


def main() -> None:
    library_id, count = sync_from_postgres(
        pg_url=PG_URL,
        table="documents",
        id_column="id",
        text_column="content",
        embedding_column="embedding",
        title_column="title",
        library_name="pgvector-migration-demo",
        base_url=ARRWDB_URL,
        updated_at_column="updated_at",
    )
    print(f"Synced {count} rows into arrwDB library {library_id}")


if __name__ == "__main__":
    main()
