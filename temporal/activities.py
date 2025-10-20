"""
Temporal activities for the RAG (Retrieval-Augmented Generation) workflow.

This module defines the 5 activities in the RAG pipeline:
1. Preprocess: Clean and prepare the query
2. Embed: Generate vector embedding for the query
3. Retrieve: Search the vector database
4. Rerank: Rerank results for better relevance
5. Generate Answer: Use LLM to generate final answer
"""

from temporalio import activity
from typing import List, Dict, Any, Tuple
import logging
import re
from uuid import UUID

from app.services.library_service import LibraryService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@activity.defn(name="preprocess_query")
async def preprocess_query(query: str) -> str:
    """
    Activity 1: Preprocess the user query.

    This activity:
    - Removes extra whitespace
    - Normalizes punctuation
    - Expands common abbreviations
    - Removes stop words (optional)

    Args:
        query: Raw user query.

    Returns:
        Preprocessed query.
    """
    logger.info(f"Preprocessing query: {query[:50]}...")

    # Remove extra whitespace
    processed = " ".join(query.split())

    # Normalize punctuation
    processed = re.sub(r"[^\w\s\?\.\!]", "", processed)

    # Common abbreviations expansion
    abbreviations = {
        " ML ": " machine learning ",
        " AI ": " artificial intelligence ",
        " NLP ": " natural language processing ",
        " DL ": " deep learning ",
        " CNN ": " convolutional neural network ",
        " RNN ": " recurrent neural network ",
    }

    for abbr, expansion in abbreviations.items():
        processed = processed.replace(abbr, expansion)

    logger.info(f"Preprocessed query: {processed}")
    return processed


@activity.defn(name="embed_query")
async def embed_query(
    query: str, embedding_service_config: Dict[str, Any]
) -> List[float]:
    """
    Activity 2: Generate embedding for the query.

    Args:
        query: Preprocessed query.
        embedding_service_config: Configuration for embedding service.

    Returns:
        Query embedding vector.
    """
    logger.info("Generating query embedding...")

    # Create embedding service
    embedding_service = EmbeddingService(
        api_key=embedding_service_config["api_key"],
        model=embedding_service_config.get("model", "embed-english-v3.0"),
        input_type="search_query",  # Important: use search_query for queries
    )

    # Generate embedding
    embedding = embedding_service.embed_text(query)

    logger.info(f"Generated embedding with dimension {len(embedding)}")
    return embedding.tolist()


@activity.defn(name="retrieve_chunks")
async def retrieve_chunks(
    library_id: str,
    embedding: List[float],
    k: int,
    service_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Activity 3: Retrieve relevant chunks from the vector database.

    Args:
        library_id: ID of the library to search.
        embedding: Query embedding.
        k: Number of results to retrieve.
        service_config: Configuration for services.

    Returns:
        List of retrieved chunks with metadata.
    """
    logger.info(f"Retrieving top {k} chunks from library {library_id}...")

    # Create services
    from pathlib import Path
    from infrastructure.repositories.library_repository import LibraryRepository

    repository = LibraryRepository(Path(service_config["data_dir"]))

    embedding_service = EmbeddingService(
        api_key=service_config["api_key"],
        model=service_config.get("model", "embed-english-v3.0"),
    )

    service = LibraryService(repository, embedding_service)

    # Search
    results = service.search_with_embedding(
        library_id=UUID(library_id), query_embedding=embedding, k=k
    )

    # Convert to serializable format
    chunks = []
    for chunk, distance in results:
        doc = service.get_document(chunk.metadata.source_document_id)

        chunks.append(
            {
                "chunk_id": str(chunk.id),
                "text": chunk.text,
                "distance": distance,
                "document_id": str(doc.id),
                "document_title": doc.metadata.title,
                "chunk_index": chunk.metadata.chunk_index,
            }
        )

    logger.info(f"Retrieved {len(chunks)} chunks")
    return chunks


@activity.defn(name="rerank_results")
async def rerank_results(
    query: str, chunks: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Activity 4: Rerank retrieved chunks for better relevance.

    This activity uses a simple relevance scoring algorithm:
    1. Keyword overlap with query
    2. Distance score from vector search
    3. Position in document (earlier chunks ranked higher)

    A production system might use a cross-encoder model here.

    Args:
        query: Original query.
        chunks: Retrieved chunks.
        top_k: Number of top results to return after reranking.

    Returns:
        Reranked chunks.
    """
    logger.info(f"Reranking {len(chunks)} chunks...")

    # Extract query keywords (simple tokenization)
    query_keywords = set(query.lower().split())

    # Score each chunk
    scored_chunks = []
    for chunk in chunks:
        text = chunk["text"].lower()
        text_words = set(text.split())

        # Calculate keyword overlap
        overlap = len(query_keywords & text_words) / max(
            len(query_keywords), 1
        )

        # Combined score (lower distance is better, higher overlap is better)
        # Normalize distance to 0-1 range (assuming cosine distance 0-2)
        distance_score = 1.0 - (chunk["distance"] / 2.0)

        # Position score (earlier chunks are slightly preferred)
        position_score = 1.0 / (chunk["chunk_index"] + 1)

        # Weighted combination
        final_score = (
            0.4 * distance_score + 0.4 * overlap + 0.2 * position_score
        )

        scored_chunks.append((final_score, chunk))

    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Take top_k
    reranked = [chunk for score, chunk in scored_chunks[:top_k]]

    logger.info(f"Reranked to top {len(reranked)} chunks")
    return reranked


@activity.defn(name="generate_answer")
async def generate_answer(
    query: str, context_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Activity 5: Generate final answer using LLM.

    This activity would typically call an LLM like GPT-4, Claude, or Cohere
    to generate a natural language answer based on the retrieved context.

    For now, this is a simplified implementation that returns a structured
    response with the context. In production, you'd integrate with an LLM API.

    Args:
        query: Original query.
        context_chunks: Reranked context chunks.

    Returns:
        Generated answer with sources.
    """
    logger.info("Generating answer...")

    # In a production system, you would:
    # 1. Construct a prompt with query and context
    # 2. Call LLM API (e.g., Cohere.generate(), OpenAI.chat_completions())
    # 3. Parse and structure the response

    # Simplified implementation:
    # Build context from chunks
    context_texts = [chunk["text"] for chunk in context_chunks]
    combined_context = "\n\n".join(context_texts)

    # Build sources list
    sources = []
    for chunk in context_chunks:
        sources.append(
            {
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "chunk_index": chunk["chunk_index"],
                "relevance_score": 1.0 - chunk["distance"],
            }
        )

    # In production, here you would call the LLM:
    # prompt = f"Given the context:\n{combined_context}\n\nAnswer: {query}"
    # answer = llm.generate(prompt)

    # For now, return a structured response
    answer = {
        "query": query,
        "answer": f"Based on the retrieved context, I found {len(context_chunks)} relevant passages. "
        f"[In production, this would be an LLM-generated answer using the context.]",
        "context": combined_context[:500] + "..."
        if len(combined_context) > 500
        else combined_context,
        "sources": sources,
        "num_sources": len(sources),
    }

    logger.info("Generated answer with {num_sources} sources")
    return answer
