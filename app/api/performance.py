"""How the live models are actually doing (B7.5).

Every model's test season has been spent, so these numbers -- forecasts made
before kickoff, scored against what happened -- are the only ongoing evidence.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.evaluation.track import report
from app.schemas import MarketPerformance

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("", response_model=list[MarketPerformance])
async def market_performance(
    session: AsyncSession = Depends(get_session),
    days: int = Query(90, ge=1, le=3650, description="fixtures played in the last N days"),
    all_lines: bool = Query(False, description="every line, not just each market's main one"),
) -> list[MarketPerformance]:
    rows = await report(session, days=days, reference_only=not all_lines)
    return [MarketPerformance(**row) for row in rows]
