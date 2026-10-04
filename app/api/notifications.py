"""In-app notifications (B5): what the platform wants the operator to know."""

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Notification
from app.notifications import mark_read, recent
from app.schemas import NotificationRead, NotificationSummary

router = APIRouter(prefix="/notifications", tags=["notifications"])


async def _summary(
    session: AsyncSession, limit: int = 50, unread_only: bool = False, category: str | None = None
) -> NotificationSummary:
    """Shared by both endpoints: calling a route function directly would pass its
    query defaults as objects rather than values."""
    items = await recent(session, limit=limit, unread_only=unread_only, category=category)
    unread = await session.scalar(
        select(func.count()).select_from(Notification).where(Notification.read_at.is_(None))
    )
    return NotificationSummary(
        unread=unread or 0,
        items=[NotificationRead.model_validate(item) for item in items],
    )


@router.get("", response_model=NotificationSummary)
async def list_notifications(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    unread_only: bool = Query(False),
    category: str | None = Query(None, description="pipeline, value or accuracy"),
) -> NotificationSummary:
    """Newest first, with the unread count for a badge."""
    return await _summary(session, limit=limit, unread_only=unread_only, category=category)


@router.post("/read", response_model=NotificationSummary)
async def read_notifications(
    session: AsyncSession = Depends(get_session),
    notification_ids: list[int] | None = Body(
        None, embed=True, description="omit to mark every unread notification read"
    ),
) -> NotificationSummary:
    await mark_read(session, notification_ids)
    return await _summary(session)
