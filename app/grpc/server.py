"""
gRPC server for arrwDB.

Provides low-latency binary protocol access for high-throughput search workloads.
Runs alongside the REST API on a separate port.

Usage:
    python -m app.grpc.server --port 50051

    # Or start both REST + gRPC:
    python run_api.py  # REST on 8000
    python -m app.grpc.server  # gRPC on 50051
"""

import argparse
import asyncio
import logging
import time
from concurrent import futures

import grpc
import numpy as np

from app.grpc import arrwdb_pb2, arrwdb_pb2_grpc

logger = logging.getLogger(__name__)

# Start time for uptime reporting
_start_time = time.time()


class ArrwDBServicer(arrwdb_pb2_grpc.ArrwDBServicer):
    """gRPC service implementation backed by arrwDB's library service."""

    def __init__(self, library_service=None, embedding_service=None):
        self._library_service = library_service
        self._embedding_service = embedding_service
        self._bm25_indexes = {}  # library_id -> RustBM25Index

    def _get_services(self):
        """Lazy-init services if not injected."""
        if self._library_service is None:
            from app.api.dependencies import get_library_repository, get_embedding_service
            from app.services.library_service import LibraryService
            repo = get_library_repository()
            emb = get_embedding_service()
            self._library_service = LibraryService(repo, emb)
            self._embedding_service = emb
        return self._library_service, self._embedding_service

    def HealthCheck(self, request, context):
        return arrwdb_pb2.HealthCheckResponse(
            status="ok",
            uptime_seconds=time.time() - _start_time,
        )

    def Search(self, request, context):
        t0 = time.time()
        lib_service, _ = self._get_services()

        try:
            embedding = np.array(request.embedding, dtype=np.float32)
            from uuid import UUID
            library_id = UUID(request.library_id)

            results = lib_service._repository.search(
                library_id=library_id,
                query_embedding=embedding.tolist(),
                k=request.k,
                distance_threshold=request.distance_threshold if request.HasField("distance_threshold") else None,
            )

            response = arrwdb_pb2.SearchResponse(
                latency_ms=(time.time() - t0) * 1000,
            )
            for chunk, distance in results:
                result = arrwdb_pb2.SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.metadata.source_document_id) if chunk.metadata else "",
                    text=chunk.text or "",
                    distance=distance,
                )
                response.results.append(result)
            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return arrwdb_pb2.SearchResponse()

    def SearchByText(self, request, context):
        t0 = time.time()
        lib_service, emb_service = self._get_services()

        try:
            # Generate embedding from text
            emb_service.change_input_type("search_query")
            embedding = emb_service.embed_text(request.query)

            from uuid import UUID
            library_id = UUID(request.library_id)

            results = lib_service._repository.search(
                library_id=library_id,
                query_embedding=embedding.tolist(),
                k=request.k,
                distance_threshold=request.distance_threshold if request.HasField("distance_threshold") else None,
            )

            response = arrwdb_pb2.SearchResponse(
                latency_ms=(time.time() - t0) * 1000,
            )
            for chunk, distance in results:
                response.results.append(arrwdb_pb2.SearchResult(
                    chunk_id=str(chunk.id),
                    text=chunk.text or "",
                    distance=distance,
                ))
            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return arrwdb_pb2.SearchResponse()

    def AddDocument(self, request, context):
        lib_service, _ = self._get_services()

        try:
            from uuid import UUID
            library_id = UUID(request.library_id)

            doc = lib_service.add_document_with_text(
                corpus_id=library_id,
                title=request.title,
                texts=list(request.texts),
                metadata=dict(request.metadata) if request.metadata else None,
            )

            return arrwdb_pb2.AddDocumentResponse(
                document_id=str(doc.id),
                chunk_count=len(doc.chunks) if hasattr(doc, "chunks") else 0,
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return arrwdb_pb2.AddDocumentResponse()

    def AddDocumentWithEmbeddings(self, request, context):
        lib_service, _ = self._get_services()

        try:
            from uuid import UUID
            library_id = UUID(request.library_id)

            chunks = [(pair.text, list(pair.embedding)) for pair in request.chunks]

            doc = lib_service.add_document_with_embeddings(
                corpus_id=library_id,
                title=request.title,
                text_embedding_pairs=chunks,
            )

            return arrwdb_pb2.AddDocumentResponse(
                document_id=str(doc.id),
                chunk_count=len(chunks),
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return arrwdb_pb2.AddDocumentResponse()

    def UpsertVector(self, request, context):
        # Direct index-level upsert via the Rust FastHNSW
        try:
            lib_service, _ = self._get_services()
            from uuid import UUID

            library_id = UUID(request.library_id)
            embedding = np.array(request.embedding, dtype=np.float32)

            # Get the library's index and upsert
            library = lib_service._repository.get_library(library_id)
            if library is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Library not found")
                return arrwdb_pb2.UpsertVectorResponse()

            # Use the index wrapper's upsert if available
            index = lib_service._repository._get_index(library_id)
            if hasattr(index, "upsert_vector"):
                vec_id = UUID(request.vector_id)
                vector_store = lib_service._repository._get_vector_store(library_id)
                vec_idx = vector_store.add_vector(vec_id, embedding)
                was_update = index.upsert_vector(vec_id, vec_idx)
                return arrwdb_pb2.UpsertVectorResponse(was_update=was_update)

            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Upsert not supported on this index type")
            return arrwdb_pb2.UpsertVectorResponse()

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return arrwdb_pb2.UpsertVectorResponse()

    def HybridSearch(self, request, context):
        t0 = time.time()

        try:
            lib_service, _ = self._get_services()
            from uuid import UUID
            library_id = UUID(request.library_id)

            # Vector search
            embedding = np.array(request.embedding, dtype=np.float32)
            vector_results = lib_service._repository.search(
                library_id=library_id,
                query_embedding=embedding.tolist(),
                k=request.k * 2,
            )

            # BM25 search (if index exists for this library)
            lib_id_str = request.library_id
            if lib_id_str in self._bm25_indexes:
                bm25 = self._bm25_indexes[lib_id_str]
                bm25_results = bm25.search(request.query_text, k=request.k * 2)
            else:
                bm25_results = []

            # RRF fusion
            rrf_k = request.rrf_k if request.HasField("rrf_k") else 60.0
            fused = self._rrf_fuse(vector_results, bm25_results, rrf_k, request.k)

            response = arrwdb_pb2.SearchResponse(
                latency_ms=(time.time() - t0) * 1000,
            )
            for chunk, score in fused:
                response.results.append(arrwdb_pb2.SearchResult(
                    chunk_id=str(chunk.id) if hasattr(chunk, "id") else str(chunk),
                    text=chunk.text if hasattr(chunk, "text") else "",
                    distance=score,
                ))
            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return arrwdb_pb2.SearchResponse()

    def _rrf_fuse(self, vector_results, bm25_results, rrf_k, limit):
        """Reciprocal Rank Fusion."""
        scores = {}
        for rank, (chunk, dist) in enumerate(vector_results):
            key = str(chunk.id) if hasattr(chunk, "id") else str(chunk)
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)

        for rank, (doc_id, score) in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

        sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:limit]
        return [(k, v) for k, v in sorted_results]

    def DeleteDocument(self, request, context):
        try:
            lib_service, _ = self._get_services()
            from uuid import UUID
            doc_id = UUID(request.document_id)
            lib_service._repository.delete_document(doc_id)
            return arrwdb_pb2.DeleteDocumentResponse(deleted=True)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return arrwdb_pb2.DeleteDocumentResponse(deleted=False)


def serve(port: int = 50051, max_workers: int = 10):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    arrwdb_pb2_grpc.add_ArrwDBServicer_to_server(ArrwDBServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"gRPC server started on port {port}")
    print(f"gRPC server listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="arrwDB gRPC Server")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.workers)
