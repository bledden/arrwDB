"""
arrwDB integrations for popular data/ML frameworks.

Available integrations:
    - langchain: VectorStore adapter for LangChain
    - llama_index: VectorStore adapter for LlamaIndex
    - postgres: sync helpers for Postgres / pgvector

Each integration is imported lazily and requires its optional dependency:
    pip install arrwdb[langchain]
    pip install arrwdb[llamaindex]
    pip install arrwdb[postgres]
"""
