"""Smoke tests for optional integrations.

Each integration is gated on its optional dependency. Tests confirm:
1. The module can be imported when the dep is installed.
2. The module raises a clear ImportError when the dep is missing.
3. Core attributes (classes/functions) exist at the expected paths.

Tests do NOT hit a live arrwDB server or a real Postgres — they verify
that the integration surface is wired up correctly.
"""

import importlib
import pytest


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("langchain_core"),
    reason="langchain-core not installed (install arrwdb[langchain])",
)
def test_langchain_adapter_importable():
    from arrwdb.integrations.langchain import ArrwDBVectorStore

    assert hasattr(ArrwDBVectorStore, "add_texts")
    assert hasattr(ArrwDBVectorStore, "similarity_search")
    assert hasattr(ArrwDBVectorStore, "similarity_search_by_vector")
    assert hasattr(ArrwDBVectorStore, "similarity_search_with_score")
    assert hasattr(ArrwDBVectorStore, "from_texts")


def test_langchain_adapter_raises_without_dep(monkeypatch):
    if _can_import("langchain_core"):
        pytest.skip("langchain-core is installed; cannot test the missing-dep path")
    with pytest.raises(ImportError, match="arrwdb\\[langchain\\]"):
        importlib.import_module("arrwdb.integrations.langchain")


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("llama_index.core"),
    reason="llama-index-core not installed (install arrwdb[llamaindex])",
)
def test_llama_index_adapter_importable():
    from arrwdb.integrations.llama_index import ArrwDBVectorStore

    assert hasattr(ArrwDBVectorStore, "add")
    assert hasattr(ArrwDBVectorStore, "query")
    assert hasattr(ArrwDBVectorStore, "delete")


def test_llama_index_adapter_raises_without_dep():
    if _can_import("llama_index.core"):
        pytest.skip("llama-index-core is installed; cannot test the missing-dep path")
    with pytest.raises(ImportError, match="arrwdb\\[llamaindex\\]"):
        importlib.import_module("arrwdb.integrations.llama_index")


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("psycopg"),
    reason="psycopg not installed (install arrwdb[postgres])",
)
def test_postgres_helper_importable():
    from arrwdb.integrations.postgres import (
        PostgresCDCSubscriber,
        sync_from_pgvector,
        sync_from_postgres,
    )

    assert callable(sync_from_postgres)
    assert callable(sync_from_pgvector)
    assert PostgresCDCSubscriber is not None


@pytest.mark.skipif(
    not _can_import("psycopg"),
    reason="psycopg not installed",
)
def test_postgres_parse_pgvector_accepts_list():
    from arrwdb.integrations.postgres import _parse_pgvector

    assert _parse_pgvector([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert _parse_pgvector((1.0, 2.0, 3.0)) == [1.0, 2.0, 3.0]


@pytest.mark.skipif(
    not _can_import("psycopg"),
    reason="psycopg not installed",
)
def test_postgres_parse_pgvector_accepts_string():
    from arrwdb.integrations.postgres import _parse_pgvector

    assert _parse_pgvector("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]
    assert _parse_pgvector("[]") == []


@pytest.mark.skipif(
    not _can_import("psycopg"),
    reason="psycopg not installed",
)
def test_postgres_parse_pgvector_rejects_bytes():
    from arrwdb.integrations.postgres import _parse_pgvector

    with pytest.raises(TypeError, match="Unsupported embedding column type"):
        _parse_pgvector(b"binary-bytes")


def test_postgres_helper_raises_without_dep():
    if _can_import("psycopg"):
        pytest.skip("psycopg is installed; cannot test the missing-dep path")
    with pytest.raises(ImportError, match="arrwdb\\[postgres\\]"):
        importlib.import_module("arrwdb.integrations.postgres")
