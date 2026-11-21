from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Response
from sqlalchemy.orm import Session

from app.db.models import UserSession
from app.db.session import get_db
from app.dependencies.auth import get_current_session, require_session_csrf
from app.dependencies.swipes import get_swipe_service
from app.schemas.swipes import (
    SwipeItem,
    SwipeListsResponse,
    SwipeMutationResponse,
    SwipeRequest,
    SwipeUndoResponse,
)


router = APIRouter(prefix="/api/v1/swipes", tags=["swipes"])


@router.get("", response_model=SwipeListsResponse)
def list_swipes(
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    service=Depends(get_swipe_service),
):
    rows = service.list_swipes(db, record.user_id)
    interested = [
        SwipeItem(**service.to_item(row))
        for row in rows
        if row.action == "interested"
    ]
    not_interested = [
        SwipeItem(**service.to_item(row))
        for row in rows
        if row.action == "not_interested"
    ]
    return SwipeListsResponse(
        interested=interested,
        not_interested=not_interested,
        counts={
            "interested": len(interested),
            "not_interested": len(not_interested),
        },
    )


@router.put(
    "/{internal_movie_id}",
    response_model=SwipeMutationResponse,
    dependencies=[Depends(require_session_csrf)],
)
def save_swipe(
    payload: SwipeRequest,
    internal_movie_id: int = Path(..., ge=0),
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    service=Depends(get_swipe_service),
):
    swipe, event = service.upsert_swipe(
        db,
        record.user,
        internal_movie_id,
        payload.action,
    )
    return SwipeMutationResponse(
        swipe=SwipeItem(**service.to_item(swipe)),
        event_id=event.id,
    )


@router.post(
    "/{internal_movie_id}/undo",
    response_model=SwipeUndoResponse,
    dependencies=[Depends(require_session_csrf)],
)
def undo_swipe(
    internal_movie_id: int = Path(..., ge=0),
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    service=Depends(get_swipe_service),
):
    swipe = service.undo_last_swipe(
        db,
        record.user,
        internal_movie_id,
    )
    return SwipeUndoResponse(
        swipe=(
            SwipeItem(**service.to_item(swipe))
            if swipe is not None
            else None
        )
    )


@router.delete(
    "/{internal_movie_id}",
    status_code=204,
    dependencies=[Depends(require_session_csrf)],
)
def delete_swipe(
    internal_movie_id: int = Path(..., ge=0),
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    service=Depends(get_swipe_service),
):
    service.remove_swipe(db, record.user, internal_movie_id)
    return Response(status_code=204)
