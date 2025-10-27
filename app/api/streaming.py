"""
FIXED: Simplified NDJSON streaming endpoint
"""
import asyncio
import json
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import get_library_service
from app.services.library_service import LibraryService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/libraries/{library_id}/documents/stream")
async def stream_documents_fixed(
    library_id: UUID,
    request: Request,
    service: LibraryService = Depends(get_library_service),
):
    """
    Simplified NDJSON document ingestion - returns standard JSON response.
    """
    logger.info(f"Starting NDJSON ingestion for library {library_id}")
    
    # Read entire body
    body = await request.body()
    body_text = body.decode("utf-8").strip()
    
    # Process each line
    results = []
    for line in body_text.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        try:
            doc_data = json.loads(line)
            title = doc_data.get("title", "Untitled")
            texts = doc_data.get("texts", [])
            
            # Add document
            result = await asyncio.to_thread(
                service.add_document_with_text,
                library_id,
                title=title,
                texts=texts,
                author=doc_data.get("author"),
                document_type=doc_data.get("document_type", "text"),
                source_url=doc_data.get("source_url"),
                tags=doc_data.get("tags"),
            )
            
            results.append({
                "success": True,
                "document_id": str(result.id),
                "title": title,
                "num_chunks": len(result.chunks)
            })
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            results.append({
                "success": False,
                "error": str(e)
            })
    
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful
    
    logger.info(f"NDJSON ingestion complete: {successful}/{len(results)} succeeded")
    
    return JSONResponse({
        "successful": successful,
        "failed": failed,
        "results": results,
        "processing_time_ms": 0  # Placeholder
    })

@router.post("/libraries/{library_id}/search/stream")
async def stream_search_results(
    library_id: UUID,
    request: Request,
    service: LibraryService = Depends(get_library_service),
):
    """
    Stream search results - simplified version returns JSON.
    """
    body = await request.body()
    search_params = json.loads(body.decode("utf-8"))
    
    query = search_params.get("query")
    k = search_params.get("k", 10)
    distance_threshold = search_params.get("distance_threshold")
    
    logger.info(f"Streaming search for library {library_id}: query='{query}', k={k}")
    
    # Execute search
    results = await asyncio.to_thread(
        service.search_with_text,
        library_id=library_id,
        query_text=query,
        k=k,
        distance_threshold=distance_threshold,
    )
    
    # Format results
    formatted_results = []
    for rank, (chunk, distance) in enumerate(results, start=1):
        # Serialize metadata - use mode='json' to automatically convert datetime/UUID to strings
        metadata_dict = chunk.metadata.model_dump(mode='json') if hasattr(chunk.metadata, 'model_dump') else chunk.metadata

        formatted_results.append({
            "rank": rank,
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.metadata.source_document_id),
            "distance": distance,
            "text": chunk.text,
            "metadata": metadata_dict,
        })
    
    return JSONResponse({
        "results": formatted_results,
        "total": len(formatted_results)
    })


# SSE endpoint for real-time event streaming
from app.events.bus import EventBus, get_event_bus
from sse_starlette.sse import EventSourceResponse

@router.get("/events/stream")
async def stream_events(
    library_id: Optional[UUID] = None,
    event_bus: EventBus = Depends(get_event_bus),
):
    """
    Server-Sent Events (SSE) endpoint for real-time event notifications.
    
    Subscribe to all events or filter by library_id.
    """
    logger.info(f"SSE client connected (library: {library_id or 'all'})")
    
    async def event_generator():
        """Generate SSE events from event bus."""
        import asyncio
        queue = asyncio.Queue()
        
        # Subscribe to events
        async def handler(event):
            # Filter by library if specified
            if library_id is None or event.library_id == library_id:
                await queue.put(event)
        
        event_bus.subscribe(handler)
        
        try:
            while True:
                event = await queue.get()
                yield {
                    "event": event.type.value,
                    "data": json.dumps({
                        "library_id": str(event.library_id),
                        "data": event.data,
                        "timestamp": event.timestamp.isoformat(),
                    })
                }
        except asyncio.CancelledError:
            logger.info("SSE client disconnected")
            raise
    
    return EventSourceResponse(event_generator())
