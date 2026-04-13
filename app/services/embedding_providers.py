"""
Pluggable Embedding Providers.

Supports multiple embedding APIs through a common interface:
- Voyage AI voyage-4-large (1024d, $0.12/1M tokens, 200M free) — best retrieval quality
- NVIDIA NV-Embed-v2 (4096d, free via NGC API)
- Google Gemini Embedding 2 (3072d, free tier) — #1 MTEB overall
- Cohere embed-v4 (1536d, $0.12/1M tokens) — multimodal
- Cohere embed-english-v3.0 (1024d, $0.10/1M tokens) — legacy

Usage:
    provider = get_embedding_provider("voyage")
    embeddings = provider.embed_texts(["hello world"])

Configuration via environment variables:
    EMBEDDING_PROVIDER=voyage|nvidia|gemini|cohere
    VOYAGE_API_KEY=...
    NGC_API_KEY=...
    GEMINI_API_KEY=...
    COHERE_API_KEY=...
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import requests
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text."""
        ...

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Generate embeddings for multiple texts in batch."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'cohere', 'nvidia', 'voyage')."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output embedding dimension."""
        ...

    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Maximum texts per API call."""
        ...

    def _normalize(self, vec: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2-normalize a vector."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def _embed_texts_chunked(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Process large batches in chunks."""
        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            chunk = texts[i : i + self.max_batch_size]
            chunk_embeddings = self.embed_texts(chunk)
            all_embeddings.extend(chunk_embeddings)
        return all_embeddings

    def _retry_request(self, fn, max_retries: int = 5, base_delay: float = 2.0):
        """Retry a request with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return fn()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                time.sleep(delay)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (429, 500, 502, 503, 504):
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"HTTP {e.response.status_code} (attempt {attempt + 1}), retrying in {delay}s")
                    time.sleep(delay)
                else:
                    raise


class NvidiaEmbeddingProvider(EmbeddingProvider):
    """NVIDIA NV-Embed-v2 via NGC API.

    Model: nvidia/nv-embedqa-e5-v5 (NV-Embed-v2)
    Dimension: 4096 (or truncated)
    Cost: Free with NGC API key
    MTEB: 72.3 (highest as of 2025)
    """

    API_URL = "https://integrate.api.nvidia.com/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/nv-embedqa-e5-v5",
        truncate_dim: Optional[int] = None,
        input_type: str = "passage",
    ):
        if not api_key:
            raise ValueError("NGC API key cannot be empty")
        self._api_key = api_key
        self._model = model
        self._truncate_dim = truncate_dim
        self._input_type = input_type  # "passage" for docs, "query" for search
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        logger.info(f"Initialized NvidiaEmbeddingProvider: model={model}, truncate_dim={truncate_dim}")

    @property
    def name(self) -> str:
        return "nvidia"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._truncate_dim or 4096

    @property
    def max_batch_size(self) -> int:
        return 50  # NGC API limit

    def embed_text(self, text: str) -> NDArray[np.float32]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[NDArray[np.float32]]:
        if len(texts) > self.max_batch_size:
            return self._embed_texts_chunked(texts)

        payload = {
            "input": texts,
            "model": self._model,
            "input_type": self._input_type,
            "encoding_format": "float",
        }
        if self._truncate_dim:
            payload["truncate"] = "END"

        def do_request():
            resp = self._session.post(self.API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()

        data = self._retry_request(do_request)

        embeddings = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            vec = np.array(item["embedding"], dtype=np.float32)
            if self._truncate_dim:
                vec = vec[: self._truncate_dim]
            embeddings.append(self._normalize(vec))

        return embeddings

    def change_input_type(self, input_type: str) -> None:
        """Switch between 'passage' and 'query' modes."""
        self._input_type = input_type


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI voyage-4-large embedding provider.

    Model: voyage-4-large (default)
    Dimension: 1024 (default, supports 256-2048 via Matryoshka)
    Cost: $0.12/1M tokens (200M tokens free per account)
    RTEB: 71.41 NDCG@10 (best retrieval quality, April 2026)
    Context: 32K tokens
    Note: All Voyage 4 models share embedding space — can mix model sizes.
    """

    API_URL = "https://api.voyageai.com/v1/embeddings"

    # Voyage 4 supported dimensions (Matryoshka)
    SUPPORTED_DIMS = {256, 512, 1024, 2048}

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-4-large",
        input_type: str = "document",
        output_dimension: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("Voyage AI API key cannot be empty")
        self._api_key = api_key
        self._model = model
        self._input_type = input_type  # "document" or "query"
        self._output_dimension = output_dimension
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        logger.info(f"Initialized VoyageEmbeddingProvider: model={model}, dim={output_dimension or 1024}")

    @property
    def name(self) -> str:
        return "voyage"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._output_dimension or 1024

    @property
    def max_batch_size(self) -> int:
        return 128  # Voyage API limit

    def embed_text(self, text: str) -> NDArray[np.float32]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[NDArray[np.float32]]:
        if len(texts) > self.max_batch_size:
            return self._embed_texts_chunked(texts)

        payload = {
            "input": texts,
            "model": self._model,
            "input_type": self._input_type,
        }
        if self._output_dimension:
            payload["output_dimension"] = self._output_dimension

        def do_request():
            resp = self._session.post(self.API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()

        data = self._retry_request(do_request)

        embeddings = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            vec = np.array(item["embedding"], dtype=np.float32)
            embeddings.append(self._normalize(vec))

        return embeddings

    def change_input_type(self, input_type: str) -> None:
        """Switch between 'document' and 'query' modes."""
        self._input_type = input_type


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini Embedding 2 provider.

    Model: gemini-embedding-exp-03-07 (latest preview)
    Dimension: 3072 (default, Matryoshka: 128-3072)
    Cost: Free tier available, $0.15/1M tokens paid
    MTEB: #1 overall (68.32 multilingual v2)
    Context: 8K tokens
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    SUPPORTED_DIMS = {128, 256, 512, 768, 1536, 3072}

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-exp-03-07",
        output_dimension: Optional[int] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        if not api_key:
            raise ValueError("Gemini API key cannot be empty")
        self._api_key = api_key
        self._model = model
        self._output_dimension = output_dimension
        self._task_type = task_type
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        logger.info(f"Initialized GeminiEmbeddingProvider: model={model}, dim={output_dimension or 3072}")

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._output_dimension or 3072

    @property
    def max_batch_size(self) -> int:
        return 100

    def embed_text(self, text: str) -> NDArray[np.float32]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[NDArray[np.float32]]:
        if len(texts) > self.max_batch_size:
            return self._embed_texts_chunked(texts)

        # Gemini uses batchEmbedContents for multiple texts
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:batchEmbedContents?key={self._api_key}"

        requests_payload = []
        for text in texts:
            req = {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                "taskType": self._task_type,
            }
            if self._output_dimension:
                req["outputDimensionality"] = self._output_dimension
            requests_payload.append(req)

        payload = {"requests": requests_payload}

        def do_request():
            resp = self._session.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()

        data = self._retry_request(do_request)

        embeddings = []
        for item in data["embeddings"]:
            vec = np.array(item["values"], dtype=np.float32)
            embeddings.append(self._normalize(vec))

        return embeddings

    def change_input_type(self, input_type: str) -> None:
        """Switch between document and query task types."""
        type_map = {
            "document": "RETRIEVAL_DOCUMENT",
            "query": "RETRIEVAL_QUERY",
        }
        self._task_type = type_map.get(input_type, input_type)


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embed-english-v3.0 wrapper implementing the provider interface.

    Wraps the existing EmbeddingService for backward compatibility.
    """

    def __init__(self, embedding_service):
        """Wrap an existing EmbeddingService instance."""
        self._service = embedding_service

    @property
    def name(self) -> str:
        return "cohere"

    @property
    def model(self) -> str:
        return self._service.model

    @property
    def dimension(self) -> int:
        return self._service.embedding_dimension

    @property
    def max_batch_size(self) -> int:
        return 96

    def embed_text(self, text: str) -> NDArray[np.float32]:
        return self._service.embed_text(text)

    def embed_texts(self, texts: List[str]) -> List[NDArray[np.float32]]:
        return self._service.embed_texts(texts)

    def change_input_type(self, input_type: str) -> None:
        self._service.change_input_type(input_type)


def get_embedding_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> EmbeddingProvider:
    """Factory function to create an embedding provider.

    Args:
        provider_name: One of 'voyage', 'nvidia', 'gemini', 'cohere'.
        api_key: API key for the provider.
        **kwargs: Additional provider-specific arguments.

    Returns:
        An EmbeddingProvider instance.
    """
    import os

    if provider_name == "voyage":
        key = api_key or os.getenv("VOYAGE_API_KEY")
        if not key:
            raise ValueError("VOYAGE_API_KEY required for Voyage AI provider")
        return VoyageEmbeddingProvider(api_key=key, **kwargs)

    elif provider_name == "nvidia":
        key = api_key or os.getenv("NGC_API_KEY")
        if not key:
            raise ValueError("NGC_API_KEY required for NVIDIA provider")
        return NvidiaEmbeddingProvider(api_key=key, **kwargs)

    elif provider_name == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY required for Gemini provider")
        return GeminiEmbeddingProvider(api_key=key, **kwargs)

    elif provider_name == "cohere":
        from app.services.embedding_service import EmbeddingService

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY required for Cohere provider")
        model = kwargs.get("model", os.getenv("EMBEDDING_MODEL", "embed-english-v3.0"))
        service = EmbeddingService(
            api_key=key,
            model=model,
            input_type=kwargs.get("input_type", "search_document"),
        )
        return CohereEmbeddingProvider(service)

    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported: voyage, nvidia, gemini, cohere"
        )
