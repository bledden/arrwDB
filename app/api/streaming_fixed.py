"""
FIXED: Simplified NDJSON streaming endpoint
"""
import asyncio
import json
import logging
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
