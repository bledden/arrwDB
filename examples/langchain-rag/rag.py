"""Minimal LangChain RAG pipeline backed by arrwDB.

Swap ``ArrwDBVectorStore`` for ``PGVector`` / ``Pinecone`` / ``Qdrant``
to compare. Interface is identical.

Run:
    pip install "arrwdb[langchain]" langchain-openai
    export OPENAI_API_KEY=...
    python rag.py
"""

import os

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from arrwdb.integrations.langchain import ArrwDBVectorStore


def main() -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    corpus = [
        Document(
            page_content="arrwDB is a Rust vector database with AVX-512 HNSW search.",
            metadata={"title": "arrwDB Overview"},
        ),
        Document(
            page_content="pgvector runs inside Postgres and is convenient "
                         "but caps out around 20 QPS at 0.99 recall on SIFT-1M.",
            metadata={"title": "pgvector Limitations"},
        ),
        Document(
            page_content="HNSW builds a multi-layer navigable small-world graph.",
            metadata={"title": "HNSW Primer"},
        ),
    ]

    store = ArrwDBVectorStore.from_texts(
        texts=[d.page_content for d in corpus],
        embedding=embeddings,
        metadatas=[d.metadata for d in corpus],
        library_name="langchain-rag-demo",
        base_url=os.environ.get("ARRWDB_URL", "http://localhost:8000"),
    )

    for hit, score in store.similarity_search_with_score(
        "how fast is arrwDB compared to pgvector?", k=3
    ):
        print(f"[{score:.4f}] {hit.metadata.get('document_title')}: {hit.page_content}")


if __name__ == "__main__":
    main()
