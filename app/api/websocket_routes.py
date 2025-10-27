"""
WebSocket endpoints for real-time bidirectional communication.

Provides WebSocket connections for:
- Real-time search with streaming results
- Live document operations
- Event notifications
- Interactive query sessions
"""

import asyncio
import json
import logging
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, ValidationError

from app.api.dependencies import get_library_service
from app.services.library_service import LibraryService
from app.websockets.manager import get_connection_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["websocket"])


# Message models
class WSRequest(BaseModel):
    """WebSocket request message."""

    type: str = "request"
    action: str  # search, add, delete, get, subscribe
    request_id: str
    data: dict


class WSResponse(BaseModel):
    """WebSocket response message."""

    type: str = "response"
    request_id: str
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class WSEvent(BaseModel):
    """WebSocket event notification."""

    type: str = "event"
    event_type: str
    library_id: str
    data: dict


@router.websocket("/libraries/{library_id}/ws")
async def websocket_library_endpoint(
    websocket: WebSocket,
    library_id: UUID,
    library_service: LibraryService = Depends(get_library_service),
):
    """
    WebSocket endpoint for real-time library operations.

    **Supported Actions**:
    - `search`: Real-time semantic search with streaming results
    - `add`: Add documents and get immediate confirmation
    - `delete`: Delete documents
    - `get`: Retrieve documents
    - `subscribe`: Subscribe to library events (automatic on connect)

    **Message Format**:
    ```json
    {
      "type": "request",
      "action": "search",
      "request_id": "unique-id",
      "data": {
        "query_text": "search query",
        "k": 10
      }
    }
    ```

    **Response Format**:
    ```json
    {
      "type": "response",
      "request_id": "unique-id",
      "success": true,
      "data": {...}
    }
    ```

    **Event Format**:
    ```json
    {
      "type": "event",
      "event_type": "library.updated",
      "library_id": "uuid",
      "data": {...}
    }
    ```
    """
    manager = get_connection_manager()

    # Verify library exists
    library = library_service.get_library(library_id)
    if not library:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=f"Library {library_id} not found",
        )
        return

    # Accept connection and subscribe to library events
    connection_id = await manager.connect(websocket, library_id)

    # Send welcome message
    await manager.send_message(
        connection_id,
        {
            "type": "system",
            "message": f"Connected to library {library_id}",
            "connection_id": connection_id,
        },
    )

    try:
        while True:
            # Receive message from client
            raw_message = await websocket.receive_text()

            try:
                # Parse and validate request
                message_data = json.loads(raw_message)
                request = WSRequest(**message_data)

                # Handle request based on action
                response = await handle_request(
                    request, library_id, library_service
                )

                # Send response
                await manager.send_message(connection_id, response.dict())

            except ValidationError as e:
                # Invalid message format
                await manager.send_message(
                    connection_id,
                    {
                        "type": "error",
                        "error": "Invalid message format",
                        "details": str(e),
                    },
                )

            except Exception as e:
                # General error handling
                logger.error(f"Error handling WebSocket message: {e}", exc_info=True)
                await manager.send_message(
                    connection_id,
                    {
                        "type": "error",
                        "request_id": message_data.get("request_id"),
                        "error": str(e),
                    },
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")

    finally:
        await manager.disconnect(connection_id)


async def handle_request(
    request: WSRequest,
    library_id: UUID,
    library_service: LibraryService,
) -> WSResponse:
    """
    Handle a WebSocket request and return response.

    Args:
        request: Validated WebSocket request
        library_id: Target library ID
        library_service: Library service instance

    Returns:
        WebSocket response
    """
    try:
        if request.action == "search":
            return await handle_search(request, library_id, library_service)

        elif request.action == "add":
            return await handle_add(request, library_id, library_service)

        elif request.action == "delete":
            return await handle_delete(request, library_id, library_service)

        elif request.action == "get":
            return await handle_get(request, library_id, library_service)

        elif request.action == "subscribe":
            # Already subscribed on connection
            return WSResponse(
                request_id=request.request_id,
                success=True,
                data={"message": "Already subscribed to library events"},
            )

        else:
            return WSResponse(
                request_id=request.request_id,
                success=False,
                error=f"Unknown action: {request.action}",
            )

    except Exception as e:
        logger.error(f"Error handling {request.action}: {e}", exc_info=True)
        return WSResponse(
            request_id=request.request_id,
            success=False,
            error=str(e),
        )


async def handle_search(
    request: WSRequest,
    library_id: UUID,
    library_service: LibraryService,
) -> WSResponse:
    """Handle search request."""
    data = request.data

    query_text = data.get("query_text")
    if not query_text:
        return WSResponse(
            request_id=request.request_id,
            success=False,
            error="query_text is required",
        )

    k = data.get("k", 10)
    threshold = data.get("threshold")
    metadata_filter = data.get("metadata_filter")

    # Perform search (async - will use embedding service)
    results = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: library_service.search_with_text(
            library_id=library_id,
            query_text=query_text,
            k=k,
            distance_threshold=threshold,
        ),
    )

    # Serialize results - use mode='json' to automatically convert datetime/UUID to strings
    serialized_results = []
    for chunk, distance in results:
        metadata_dict = chunk.metadata.model_dump(mode='json') if hasattr(chunk.metadata, 'model_dump') else chunk.metadata

        serialized_results.append({
            "id": str(chunk.id),
            "text": chunk.text,
            "metadata": metadata_dict,
            "distance": distance,
        })

    return WSResponse(
        request_id=request.request_id,
        success=True,
        data={"results": serialized_results},
    )


async def handle_add(
    request: WSRequest,
    library_id: UUID,
    library_service: LibraryService,
) -> WSResponse:
    """Handle add document request."""
    data = request.data

    text = data.get("text")
    if not text:
        return WSResponse(
            request_id=request.request_id,
            success=False,
            error="text is required",
        )

    title = data.get("title", "Untitled")
    metadata = data.get("metadata", {})

    # Add document (async - will use embedding service)
    document = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: library_service.add_document_with_text(
            library_id=library_id,
            title=title,
            texts=[text],  # Single chunk
        ),
    )

    return WSResponse(
        request_id=request.request_id,
        success=True,
        data={
            "document": {
                "id": str(document.id),
                "title": document.metadata.title,
                "num_chunks": len(document.chunks),
            }
        },
    )


async def handle_delete(
    request: WSRequest,
    library_id: UUID,
    library_service: LibraryService,
) -> WSResponse:
    """Handle delete document request."""
    data = request.data

    doc_id_str = data.get("document_id")
    if not doc_id_str:
        return WSResponse(
            request_id=request.request_id,
            success=False,
            error="document_id is required",
        )

    try:
        doc_id = UUID(doc_id_str)
    except ValueError:
        return WSResponse(
            request_id=request.request_id,
            success=False,
            error="Invalid document_id format",
        )

    # Delete document (sync operation)
    success = library_service.delete_document(
        library_id=library_id,
        document_id=doc_id,
    )

    if not success:
        return WSResponse(
            request_id=request.request_id,
            success=False,
            error="Document not found",
        )

    return WSResponse(
        request_id=request.request_id,
        success=True,
        data={"message": "Document deleted"},
    )


async def handle_get(
    request: WSRequest,
    library_id: UUID,
    library_service: LibraryService,
) -> WSResponse:
    """Handle get document request."""
    data = request.data

    doc_id_str = data.get("document_id")
    if not doc_id_str:
        return WSResponse(
            request_id=request.request_id,
            success=False,
            error="document_id is required",
        )

    try:
        doc_id = UUID(doc_id_str)
    except ValueError:
        return WSResponse(
            request_id=request.request_id,
            success=False,
            error="Invalid document_id format",
        )

    # Get document (sync operation)
    document = library_service.get_document(
        library_id=library_id,
        document_id=doc_id,
    )

    if not document:
        return WSResponse(
            request_id=request.request_id,
            success=False,
            error="Document not found",
        )

    return WSResponse(
        request_id=request.request_id,
        success=True,
        data={
            "document": {
                "id": str(document.id),
                "text": document.text,
                "metadata": document.metadata,
            }
        },
    )


@router.get("/websockets/stats")
async def websocket_stats():
    """
    Get WebSocket connection statistics.

    Returns statistics about active WebSocket connections.
    """
    manager = get_connection_manager()
    return manager.get_stats()
