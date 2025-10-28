"""
API endpoints for SearchReplay debugging feature.

NOVEL FEATURE: Expose search path transparency to API consumers.
No other vector DB API provides this level of debugging insight.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.search_replay import get_recorder

router = APIRouter(prefix="/search-replay", tags=["SearchReplay"])


# ============================================================================
# Request/Response Models
# ============================================================================


class SearchReplayConfig(BaseModel):
    """Configuration for enabling/disabling search replay recording."""

    enabled: bool = Field(..., description="Enable or disable search path recording")


class SearchReplayStatus(BaseModel):
    """Status of search replay recording."""

    enabled: bool
    total_paths_recorded: int
    max_stored_paths: int


class SearchPathSummary(BaseModel):
    """Summary of a recorded search path."""

    replay_id: str
    corpus_id: str
    k: int
    index_type: str
    duration_ms: float
    nodes_visited: int
    nodes_skipped: int
    total_distance_computations: int
    result_count: int


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/enable", response_model=SearchReplayStatus)
async def enable_search_replay():
    """
    Enable search path recording.

    WARNING: Adds ~10% overhead to search operations.
    Recommended for debugging/development only, not production.

    Returns:
        Current status after enabling.
    """
    recorder = get_recorder()
    recorder.enable()

    return SearchReplayStatus(
        enabled=recorder.is_enabled(),
        total_paths_recorded=len(recorder._paths),
        max_stored_paths=recorder._max_stored_paths,
    )


@router.post("/disable", response_model=SearchReplayStatus)
async def disable_search_replay():
    """
    Disable search path recording.

    Returns to zero overhead. Previously recorded paths remain accessible
    until cleared.

    Returns:
        Current status after disabling.
    """
    recorder = get_recorder()
    recorder.disable()

    return SearchReplayStatus(
        enabled=recorder.is_enabled(),
        total_paths_recorded=len(recorder._paths),
        max_stored_paths=recorder._max_stored_paths,
    )


@router.get("/status", response_model=SearchReplayStatus)
async def get_search_replay_status():
    """
    Get current search replay recording status.

    Returns:
        Whether recording is enabled and how many paths are stored.
    """
    recorder = get_recorder()

    return SearchReplayStatus(
        enabled=recorder.is_enabled(),
        total_paths_recorded=len(recorder._paths),
        max_stored_paths=recorder._max_stored_paths,
    )


@router.get("/paths", response_model=list[SearchPathSummary])
async def list_search_paths(
    corpus_id: Optional[UUID] = Query(
        None, description="Filter paths by corpus ID"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum paths to return"),
):
    """
    List recorded search paths.

    Args:
        corpus_id: Optional filter to only show paths for specific corpus.
        limit: Maximum number of paths to return (default 100, max 1000).

    Returns:
        List of search path summaries, sorted by most recent first.
    """
    recorder = get_recorder()
    paths = recorder.list_paths(corpus_id=corpus_id, limit=limit)

    return [
        SearchPathSummary(
            replay_id=path.replay_id,
            corpus_id=str(path.corpus_id),
            k=path.k,
            index_type=path.index_type,
            duration_ms=path.duration_ms,
            nodes_visited=path.nodes_visited,
            nodes_skipped=path.nodes_skipped,
            total_distance_computations=path.total_distance_computations,
            result_count=len(path.result_ids),
        )
        for path in paths
    ]


@router.get("/paths/{replay_id}")
async def get_search_path(replay_id: str):
    """
    Get detailed search path by replay ID.

    Returns the complete execution trace including:
    - Every step taken during search
    - Distance computations at each step
    - Actions (visited, skipped, backtracked, pruned)
    - Reasons for each decision
    - Performance metrics

    This is the full debug output - use for deep dive analysis.

    Args:
        replay_id: Unique identifier for the search path.

    Returns:
        Complete search path with all steps and metrics.

    Raises:
        404: If replay_id not found.
    """
    recorder = get_recorder()
    path = recorder.get_path(replay_id)

    if not path:
        raise HTTPException(
            status_code=404,
            detail=f"Search path {replay_id} not found. "
            "It may have been evicted (max 1000 paths stored) or recording was disabled.",
        )

    return path.to_dict()


@router.delete("/paths")
async def clear_search_paths():
    """
    Clear all recorded search paths.

    Use to free memory when you've collected enough debugging data.

    Returns:
        Number of paths cleared.
    """
    recorder = get_recorder()
    count = len(recorder._paths)
    recorder.clear()

    return {"cleared": count, "message": f"Cleared {count} search paths"}


@router.get("/paths/{replay_id}/summary")
async def get_search_path_summary(replay_id: str):
    """
    Get human-readable summary of a search path.

    Lighter weight than full path details - useful for quick inspection.

    Args:
        replay_id: Unique identifier for the search path.

    Returns:
        Human-readable summary string.

    Raises:
        404: If replay_id not found.
    """
    recorder = get_recorder()
    path = recorder.get_path(replay_id)

    if not path:
        raise HTTPException(
            status_code=404,
            detail=f"Search path {replay_id} not found.",
        )

    return {"replay_id": replay_id, "summary": path.summary()}
