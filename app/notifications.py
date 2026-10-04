"""Raising and reading in-app notifications.

The platform runs unattended twice a week. Anything the operator would want to
know is recorded here, so the app can show it, instead of living only in a log
file on the machine that happened to run the job.

Kept deliberately small: a notification is a level, a category, a one-line title
and optional detail. Nothing here sends email or messages anywhere.
"""

import logging
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Notification

logger = logging.getLogger(__name__)

LEVELS = ("info", "warning", "error")


async def notify(
    session: AsyncSession,
    level: str,
    category: str,
    title: str,
    detail: str | None = None,
    context: dict | None = None,
) -> Notification:
    """Record one notification. Commits, so a failing run still leaves its trace."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    notification = Notification(
        level=level, category=category, title=title[:200], detail=detail, context=context
    )
    session.add(notification)
    await session.commit()
    logger.info("notification: [%s] %s", level, title)
    return notification


async def recent(
    session: AsyncSession, limit: int = 50, unread_only: bool = False, category: str | None = None
) -> list[Notification]:
    stmt = select(Notification).order_by(Notification.created_at.desc()).limit(limit)
    if unread_only:
        stmt = stmt.where(Notification.read_at.is_(None))
    if category is not None:
        stmt = stmt.where(Notification.category == category)
    return list((await session.execute(stmt)).scalars().all())


async def mark_read(session: AsyncSession, notification_ids: list[int] | None = None) -> int:
    """Mark the given notifications read, or every unread one when none are named."""
    stmt = update(Notification).where(Notification.read_at.is_(None)).values(read_at=datetime.now(UTC))
    if notification_ids:
        stmt = stmt.where(Notification.notification_id.in_(notification_ids))
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount
