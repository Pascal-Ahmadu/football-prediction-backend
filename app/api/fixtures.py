"""Fixtures endpoints (FR-WEB-03, FR-PRED-01, B5)."""

from collections import defaultdict
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Distribution, Match, Prediction
from app.schemas import (
    CountProbability,
    FixtureDistributions,
    FixturePredictions,
    FixtureSummary,
    LinePrediction,
    MarketDistribution,
    MarketPrediction,
    TailProbability,
    TeamSummary,
)

router = APIRouter(prefix="/fixtures", tags=["fixtures"])


@router.get("", response_model=list[FixtureSummary])
async def list_fixtures(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
) -> list[Match]:
    """Upcoming fixtures, earliest kickoff first."""
    stmt = (
        select(Match)
        .where(Match.kickoff_utc >= datetime.now(UTC))
        .order_by(Match.kickoff_utc)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


EntityType = Literal["match", "player", "all"]


def _entity_filter(table, entity_type: EntityType):
    """A fixture has a handful of match markets and dozens of player rows, so the
    match markets are returned on their own unless player rows are asked for."""
    return () if entity_type == "all" else (table.entity_type == entity_type,)


def _latest_per_market(table, match_id: int, entity_type: EntityType = "all"):
    """(market, computed_at) of the newest forecast run for each market of a fixture.

    Each market is forecast in its own run with its own timestamp, so "the newest
    run overall" would return only whichever market happened to be forecast last.
    """
    return (
        select(table.market, func.max(table.computed_at).label("computed_at"))
        .where(table.match_id == match_id, *_entity_filter(table, entity_type))
        .group_by(table.market)
        .subquery()
    )


async def _fixture(session: AsyncSession, match_id: int) -> Match:
    match = await session.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Fixture not found")
    return match


@router.get("/{match_id}/predictions", response_model=FixturePredictions)
async def fixture_predictions(
    match_id: int,
    entity_type: EntityType = Query("match", description="match markets, player props, or all"),
    session: AsyncSession = Depends(get_session),
) -> FixturePredictions:
    """The most recent forecast for one fixture, for every market."""
    match = await _fixture(session, match_id)

    latest = _latest_per_market(Prediction, match_id, entity_type)
    predictions = (
        await session.execute(
            select(Prediction)
            .join(
                latest,
                (latest.c.market == Prediction.market)
                & (latest.c.computed_at == Prediction.computed_at),
            )
            .where(Prediction.match_id == match_id, *_entity_filter(Prediction, entity_type))
            .order_by(Prediction.market, Prediction.entity_id, Prediction.line)
        )
    ).scalars().all()
    if not predictions:
        raise HTTPException(status_code=404, detail="No forecast for this fixture yet")

    latest_distribution = _latest_per_market(Distribution, match_id, entity_type)
    distributions = {
        (d.market, d.model_version, d.entity_id): d
        for d in (
            await session.execute(
                select(Distribution)
                .join(
                    latest_distribution,
                    (latest_distribution.c.market == Distribution.market)
                    & (latest_distribution.c.computed_at == Distribution.computed_at),
                )
                .where(Distribution.match_id == match_id, *_entity_filter(Distribution, entity_type))
            )
        ).scalars()
    }

    grouped: dict[tuple, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        key = (
            prediction.market,
            prediction.entity_type,
            prediction.entity_id,
            prediction.model_version,
        )
        grouped[key].append(prediction)

    markets = []
    for (market, market_entity_type, entity_id, model_version), lines in grouped.items():
        distribution = distributions.get((market, model_version, entity_id))
        markets.append(
            MarketPrediction(
                market=market,
                entity_type=market_entity_type,
                entity_id=entity_id,
                model_version=model_version,
                computed_at=lines[0].computed_at,
                expected_value=distribution.expected_value if distribution else None,
                lines=[LinePrediction.model_validate(line) for line in lines],
            )
        )

    return FixturePredictions(
        match_id=match.match_id,
        kickoff_utc=match.kickoff_utc,
        home_team=TeamSummary.model_validate(match.home_team),
        away_team=TeamSummary.model_validate(match.away_team),
        markets=markets,
    )


@router.get("/{match_id}/distributions", response_model=FixtureDistributions)
async def fixture_distributions(
    match_id: int,
    market: str | None = Query(None, description="Only this market, e.g. total_corners"),
    entity_type: EntityType = Query("match", description="match markets, player props, or all"),
    session: AsyncSession = Depends(get_session),
) -> FixtureDistributions:
    """The full predicted distribution for one fixture: the chance of every count.

    Lines only say "over or under 9.5"; this says "exactly 8: 11%, exactly 9: 12%...",
    which a chart or any custom line can be built from.
    """
    match = await _fixture(session, match_id)

    latest = _latest_per_market(Distribution, match_id, entity_type)
    stmt = (
        select(Distribution)
        .join(
            latest,
            (latest.c.market == Distribution.market)
            & (latest.c.computed_at == Distribution.computed_at),
        )
        .where(Distribution.match_id == match_id, *_entity_filter(Distribution, entity_type))
        .order_by(Distribution.market, Distribution.entity_id)
    )
    if market is not None:
        stmt = stmt.where(Distribution.market == market)
    rows = (await session.execute(stmt)).scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No forecast for this fixture yet")

    markets = []
    for row in rows:
        counts = sorted((int(k), p) for k, p in row.pmf.items() if k.isdigit())
        tail_key = next(k for k in row.pmf if not k.isdigit())  # e.g. "30+"
        markets.append(
            MarketDistribution(
                market=row.market,
                entity_type=row.entity_type,
                entity_id=row.entity_id,
                model_version=row.model_version,
                computed_at=row.computed_at,
                expected_value=row.expected_value,
                probabilities=[CountProbability(count=c, probability=p) for c, p in counts],
                tail=TailProbability(at_least=int(tail_key.rstrip("+")), probability=row.pmf[tail_key]),
            )
        )

    return FixtureDistributions(
        match_id=match.match_id,
        kickoff_utc=match.kickoff_utc,
        home_team=TeamSummary.model_validate(match.home_team),
        away_team=TeamSummary.model_validate(match.away_team),
        markets=markets,
    )
