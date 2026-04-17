# LangChain RAG with arrwDB

Drop-in replacement for pgvector / Pinecone / Qdrant / Weaviate in any
LangChain pipeline.

```python
from arrwdb.integrations.langchain import ArrwDBVectorStore

store = ArrwDBVectorStore.from_texts(
    texts=[...],
    embedding=your_embedding,
    base_url="http://localhost:8000",
    library_name="my-library",
)

results = store.similarity_search("query", k=5)
```

## Install

```bash
pip install "arrwdb[langchain]" langchain-openai
```

## Migration diff

```diff
- from langchain_community.vectorstores import PGVector
- store = PGVector.from_texts(texts, embedding, collection_name="docs",
-                             connection_string="postgresql://...")
+ from arrwdb.integrations.langchain import ArrwDBVectorStore
+ store = ArrwDBVectorStore.from_texts(texts, embedding,
+                                      library_name="docs",
+                                      base_url="http://localhost:8000")
```

Everything else in your LangChain pipeline — retrievers, chains,
query engines — keeps working unchanged.
