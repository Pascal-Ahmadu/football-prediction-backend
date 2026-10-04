"""Confident forecasts for upcoming fixtures (FR-WEB-05, B5 GET /picks).

Both final evaluations found the edge concentrated where the model is confident.
But "confident" has to be measured against the right yardstick. Only 44% of
matches go over 4.5 cards, and leagues range from 20% to 63%, so an ordinary
match in a low-card league already looks "confident" on the under.

On 2024/25 cards at 4.5, scored against "always the more common side in that
league", the top 10% of picks ranked by:
    distance from 50/50                 added  +0.0 points
    distance from the global base rate  added  +2.3 points
    distance from the league base rate  added +15.4 points
So picks are ranked, by default, by how unusual a forecast is for its league.
"""

from datetime import UTC, datetime
from math import floor
from typing import Literal

from scipy.stats import poisson

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.modelling.player_predict import FORECASTS as PLAYER_FORECASTS
from app.modelling.predict import FORECASTS
from app.models import (
    Competition,
    Player,
    Distribution,
    Match,
    ModelRegistry,
    Prediction,
    Season,
)
from app.schemas import Pick, TeamSummary

router = APIRouter(prefix="/picks", tags=["picks"])


def _base_rate(metrics: dict, competition_id: int, line_key: str, scope: str) -> float | None:
    """The base rate a forecast is judged against: its league's, else the global one."""
    if scope == "league":
        league = metrics.get("base_rates_by_competition", {}).get(str(competition_id), {})
        if line_key in league:
            return league[line_key]
    return metrics.get("base_rates", {}).get(line_key)


@router.get("", response_model=list[Pick])
async def list_picks(
    session: AsyncSession = Depends(get_session),
    market: Literal[
        "total_corners", "total_cards", "total_fouls",
        "player_shots_on_target", "player_fouls_committed",
    ] = Query("total_corners"),
    line: float | None = Query(
        None,
        description=(
            "Line; defaults to the market's validated line "
            "(corners 9.5, cards 4.5, fouls 24.5, player markets 0.5)"
        ),
    ),
    min_confidence: float = Query(
        0.05, ge=0.0, le=0.5, description="Minimum distance of P(over) from 0.5"
    ),
    competition_id: int | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    rank_by: Literal["league", "global", "even"] = Query(
        "league",
        description=(
            "league: most unusual for its league first (default); "
            "global: most unusual against all leagues; even: furthest from 50/50"
        ),
    ),
) -> list[Pick]:
    """Upcoming fixtures where the latest forecast leans clearly one way."""
    player_market = market in PLAYER_FORECASTS
    if line is None:
        line = PLAYER_FORECASTS[market].lines[0] if player_market else FORECASTS[market].reference_line
    line_key = str(float(line))

    latest = (
        select(
            Prediction.match_id,
            func.max(Prediction.computed_at).label("computed_at"),
        )
        .where(Prediction.market == market)
        .group_by(Prediction.match_id)
        .subquery()
    )
    stmt = (
        select(
            Prediction, Match, Competition,
            Distribution.expected_value, Distribution.baseline_expected_value, Player.name,
        )
        .join(
            latest,
            (latest.c.match_id == Prediction.match_id)
            & (latest.c.computed_at == Prediction.computed_at),
        )
        .join(Match, Match.match_id == Prediction.match_id)
        .join(Season, Season.season_id == Match.season_id)
        .join(Competition, Competition.competition_id == Season.competition_id)
        .outerjoin(
            Distribution,
            (Distribution.match_id == Prediction.match_id)
            & (Distribution.market == Prediction.market)
            & (Distribution.model_version == Prediction.model_version)
            & (Distribution.computed_at == Prediction.computed_at)
            & (Distribution.entity_id.is_not_distinct_from(Prediction.entity_id)),
        )
        .outerjoin(Player, Player.player_id == Prediction.entity_id)
        .where(
            Prediction.market == market,
            Prediction.line == line,
            Match.kickoff_utc > datetime.now(UTC),
            func.abs(Prediction.prob_over - 0.5) >= min_confidence,
        )
    )
    if competition_id is not None:
        stmt = stmt.where(Competition.competition_id == competition_id)
    rows = (await session.execute(stmt)).all()

    versions = {prediction.model_version for prediction, *_ in rows}
    metrics = {
        version: validation_metrics
        for version, validation_metrics in await session.execute(
            select(ModelRegistry.model_version, ModelRegistry.validation_metrics).where(
                ModelRegistry.model_version.in_(versions)
            )
        )
    } if versions else {}

    picks = []
    for prediction, match, competition, expected_value, baseline_value, player_name in rows:
        if player_market:
            # A player pick is unusual when it departs from what this player's own
            # rate and usual minutes would suggest -- his personal base rate.
            base = (
                None
                if rank_by == "even" or baseline_value is None
                else float(poisson.sf(floor(line), baseline_value))
            )
        else:
            base = (
                None
                if rank_by == "even"
                else _base_rate(
                    metrics.get(prediction.model_version, {}),
                    competition.competition_id,
                    line_key,
                    rank_by,
                )
            )
        # Which side to back is a question of disagreement, not of which side is
        # likelier: a striker the model gives a 43% chance of a foul, where his own
        # rate implies 18%, is an OVER opportunity even though under is likelier.
        # Without a base rate to compare against, fall back to the likelier side.
        over = prediction.prob_over >= (0.5 if base is None else base)
        picks.append(
            Pick(
                match_id=match.match_id,
                kickoff_utc=match.kickoff_utc,
                competition=competition.name,
                country=competition.country,
                home_team=TeamSummary.model_validate(match.home_team),
                away_team=TeamSummary.model_validate(match.away_team),
                market=prediction.market,
                entity_type=prediction.entity_type,
                entity_id=prediction.entity_id,
                entity_name=player_name,
                line=prediction.line,
                selection="over" if over else "under",
                probability=prediction.prob_over if over else prediction.prob_under,
                confidence=abs(prediction.prob_over - 0.5),
                base_rate=base,
                edge_over_base=None if base is None else abs(prediction.prob_over - base),
                expected_value=expected_value,
                model_version=prediction.model_version,
                computed_at=prediction.computed_at,
            )
        )

    def ranking(pick: Pick) -> tuple[float, datetime]:
        score = pick.confidence if pick.edge_over_base is None else pick.edge_over_base
        return (-score, pick.kickoff_utc)

    picks.sort(key=ranking)
    if player_market:
        # One entry per player: the same player appears in every fixture his team
        # has scheduled, and six rows for one man is not a list of picks.
        seen: set[int] = set()
        picks = [p for p in picks if not (p.entity_id in seen or seen.add(p.entity_id))]
    return picks[:limit]
