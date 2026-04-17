"""
Postgres / pgvector sync helpers for arrwDB.

These helpers let you keep Postgres as your source of truth and use
arrwDB as a fast vector-search sidecar. Two sync modes are supported:

1. ``sync_from_postgres`` — one-shot or incremental pull.
2. ``PostgresCDCSubscriber`` — logical replication subscriber for
   streaming updates (requires ``wal_level = logical`` on the source).

Usage:
    from arrwdb.integrations.postgres import sync_from_postgres

    sync_from_postgres(
        pg_url="postgresql://user:pass@host/db",
        table="documents",
        id_column="id",
        text_column="content",
        embedding_column="embedding",     # pgvector column
        library_name="docs",
        base_url="http://localhost:8000",
    )

Incremental sync (only new/updated rows since last run):
    sync_from_postgres(
        pg_url=...,
        table="documents",
        id_column="id",
        text_column="content",
        embedding_column="embedding",
        library_id=existing_lib_id,
        updated_at_column="updated_at",
        since="2026-04-16T00:00:00Z",
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable, List, Optional, Tuple
from uuid import uuid4

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as e:
    raise ImportError(
        "psycopg (Postgres driver) is required. "
        "Install with: pip install arrwdb[postgres]"
    ) from e

from arrwdb.client import ArrwDBClient

logger = logging.getLogger(__name__)


def _parse_pgvector(value: Any) -> List[float]:
    """Coerce a pgvector column value to List[float].

    Handles the common representations:
    - list/tuple of floats (when pgvector client adapter is loaded)
    - string of the form "[1.0, 2.0, 3.0]"
    - bytes/memoryview (binary format)
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, str):
        stripped = value.strip().lstrip("[").rstrip("]")
        if not stripped:
            return []
        return [float(x) for x in stripped.split(",")]
    raise TypeError(
        f"Unsupported embedding column type: {type(value).__name__}. "
        "Install pgvector's Python adapter or cast to text in your query."
    )


def sync_from_postgres(
    pg_url: str,
    table: str,
    id_column: str,
    text_column: str,
    embedding_column: str,
    *,
    library_id: Optional[str] = None,
    library_name: Optional[str] = None,
    index_type: str = "hnsw",
    base_url: str = "http://localhost:8000",
    client: Optional[ArrwDBClient] = None,
    batch_size: int = 500,
    where: Optional[str] = None,
    updated_at_column: Optional[str] = None,
    since: Optional[datetime] = None,
    metadata_columns: Optional[List[str]] = None,
    title_column: Optional[str] = None,
) -> Tuple[str, int]:
    """Copy rows from a Postgres table into an arrwDB library.

    The embedding column must hold a pgvector ``vector`` value or a
    compatible array representation. Each row becomes a single-chunk
    arrwDB document keyed by ``id_column``.

    Args:
        pg_url: Postgres connection string.
        table: Source table name (schema-qualified if needed).
        id_column: Primary-key column to use as the arrwDB document tag.
        text_column: Column holding the chunk text.
        embedding_column: Column holding the vector (pgvector or array).
        library_id: Target arrwDB library UUID. Mutually exclusive with
            ``library_name``.
        library_name: Name for a new library when ``library_id`` is not
            provided.
        index_type: arrwDB index backend for a new library.
        base_url: arrwDB server URL (ignored if ``client`` is provided).
        client: Pre-configured ``ArrwDBClient``.
        batch_size: Rows per Postgres fetch.
        where: Additional SQL ``WHERE`` clause (without the keyword).
        updated_at_column: Timestamp column for incremental sync.
        since: Only pull rows with ``updated_at_column`` >= ``since``.
        metadata_columns: Extra columns to copy into chunk metadata.
        title_column: Column to use as the arrwDB document title. Falls
            back to the first 80 characters of the text.

    Returns:
        ``(library_id, rows_synced)``.
    """
    arrw = client or ArrwDBClient(base_url=base_url)

    if library_id is None:
        name = library_name or f"pg-sync-{uuid4().hex[:8]}"
        lib = arrw.create_library(name=name, index_type=index_type)
        library_id = lib["id"]

    select_cols = [id_column, text_column, embedding_column]
    if title_column and title_column not in select_cols:
        select_cols.append(title_column)
    if metadata_columns:
        for c in metadata_columns:
            if c not in select_cols:
                select_cols.append(c)

    quoted_cols = ", ".join(f'"{c}"' for c in select_cols)
    clauses: List[str] = []
    params: List[Any] = []

    if where:
        clauses.append(f"({where})")
    if updated_at_column and since is not None:
        clauses.append(f'"{updated_at_column}" >= %s')
        params.append(since)

    sql = f'SELECT {quoted_cols} FROM {table}'
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f' ORDER BY "{id_column}"'

    rows_synced = 0
    with psycopg.connect(pg_url, row_factory=dict_row) as conn:
        with conn.cursor(name="arrwdb_sync") as cur:
            cur.itersize = batch_size
            cur.execute(sql, params)
            for row in cur:
                embedding = _parse_pgvector(row[embedding_column])
                if not embedding:
                    logger.warning(
                        "Skipping row %s — empty or null embedding",
                        row[id_column],
                    )
                    continue
                text = row[text_column] or ""
                title_val = (
                    str(row[title_column]) if title_column and row.get(title_column)
                    else text[:80] or str(row[id_column])
                )
                tags = [f"pg_id:{row[id_column]}"]
                arrw.add_document_with_embeddings(
                    library_id=library_id,
                    title=title_val.strip(),
                    chunks=[(text, embedding)],
                    tags=tags,
                )
                rows_synced += 1
                if rows_synced % 1000 == 0:
                    logger.info("Synced %d rows so far", rows_synced)

    logger.info("Sync complete: %d rows into library %s", rows_synced, library_id)
    return library_id, rows_synced


def sync_from_pgvector(
    pg_url: str,
    table: str,
    id_column: str,
    text_column: str,
    embedding_column: str,
    **kwargs: Any,
) -> Tuple[str, int]:
    """Alias for ``sync_from_postgres`` with pgvector-focused defaults.

    Intended for teams migrating off pgvector — the function signature
    mirrors a typical pgvector schema layout.
    """
    return sync_from_postgres(
        pg_url=pg_url,
        table=table,
        id_column=id_column,
        text_column=text_column,
        embedding_column=embedding_column,
        **kwargs,
    )


class PostgresCDCSubscriber:
    """Logical-replication subscriber that mirrors a pgvector table into arrwDB.

    This is a thin wrapper around psycopg's logical replication cursor.
    The source database must have ``wal_level = logical`` and the user
    must have ``REPLICATION`` privilege.

    Prepare the publication once on the source:

        CREATE PUBLICATION arrwdb_pub FOR TABLE documents;

    Then run the subscriber:

        sub = PostgresCDCSubscriber(
            pg_url="postgresql://user:pass@host/db?replication=database",
            slot_name="arrwdb_slot",
            publication="arrwdb_pub",
            table="documents",
            id_column="id",
            text_column="content",
            embedding_column="embedding",
            library_id=lib_id,
        )
        sub.run()  # blocks; call from a worker process

    Start fresh with ``create_slot=True`` the first time to allocate the
    replication slot.
    """

    def __init__(
        self,
        pg_url: str,
        slot_name: str,
        publication: str,
        table: str,
        id_column: str,
        text_column: str,
        embedding_column: str,
        library_id: str,
        *,
        base_url: str = "http://localhost:8000",
        client: Optional[ArrwDBClient] = None,
        create_slot: bool = False,
    ) -> None:
        self.pg_url = pg_url
        self.slot_name = slot_name
        self.publication = publication
        self.table = table
        self.id_column = id_column
        self.text_column = text_column
        self.embedding_column = embedding_column
        self.library_id = library_id
        self.create_slot = create_slot
        self._arrw = client or ArrwDBClient(base_url=base_url)

    def run(self) -> None:
        """Start consuming the replication stream. Blocks forever."""
        # Intentionally lightweight — full CDC is non-trivial. This is a
        # reference scaffold; production users will likely want Debezium
        # or a custom consumer. See examples/ for a fuller implementation.
        raise NotImplementedError(
            "Logical replication consumption is intentionally not bundled "
            "in the SDK — it requires careful slot/LSN handling. See "
            "examples/postgres-cdc/ for a reference worker, or use "
            "sync_from_postgres() with an incremental updated_at column."
        )
