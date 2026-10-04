"""
Build training sets from point-in-time snapshots, one market at a time.

Each market declares what it predicts (the target statistics, summed over both
teams) and which snapshot features it reads. Training and forecasting both build
rows through _vector, so the two can never disagree about a feature.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings


@dataclass(frozen=True)
class Market:
    """What one market predicts and which features it reads."""

    name: str
    target_columns: tuple[str, ...]
    team_features: tuple[str, ...]
    referee_features: tuple[str, ...] = ()
    # A side below this is treated as incomplete statistics, not a real count.
    min_per_side: int = 0


MARKETS = {
    "total_corners": Market(
        name="total_corners",
        target_columns=("corners",),
        team_features=(
            "corners_for",
            "corners_against",
            "corners_for_venue",
            "corners_against_venue",
            "corners_attack",
            "corners_defence",
        ),
    ),
    "total_cards": Market(
        name="total_cards",
        target_columns=("yellow", "red"),
        team_features=(
            "yellow_for",
            "yellow_against",
            "yellow_for_venue",
            "yellow_against_venue",
            "yellow_attack",
            "yellow_defence",
            "fouls_for",
            "fouls_against",
        ),
        referee_features=(
            "ref_matches_seen",
            "ref_cards_pm",
            "ref_yellow_pm",
            "ref_red_pm",
            "ref_fouls_pm",
            "ref_cards_per_foul",
            "ref_home_card_share",
        ),
    ),
    "total_fouls": Market(
        name="total_fouls",
        target_columns=("fouls",),
        team_features=(
            "fouls_for",
            "fouls_against",
            "fouls_for_venue",
            "fouls_against_venue",
            "fouls_attack",
            "fouls_defence",
        ),
        # ref_cards_pm and ref_cards_per_foul added nothing on 2024/25.
        referee_features=("ref_matches_seen", "ref_fouls_pm"),
        # A team credited with 0 or 1 fouls averages under 14 shots a match
        # too: those are partial statistics from the provider.
        min_per_side=2,
    ),
}

MATCHES = text("""
    select m.match_id, m.kickoff_utc, se.start_date as season_start,
           se.competition_id, m.home_team_id, m.away_team_id, m.referee_id,
           h.corners as h_corners, a.corners as a_corners,
           h.yellow as h_yellow, a.yellow as a_yellow,
           h.red as h_red, a.red as a_red,
           h.fouls as h_fouls, a.fouls as a_fouls
    from core.matches m
    join core.seasons se on se.season_id = m.season_id
    join core.team_match_stats h
      on h.match_id = m.match_id and h.team_id = m.home_team_id
    join core.team_match_stats a
      on a.match_id = m.match_id and a.team_id = m.away_team_id
    where m.status = 'Finished'
    order by m.kickoff_utc, m.match_id
""")

UPCOMING = text("""
    select m.match_id, m.kickoff_utc, se.competition_id,
           m.home_team_id, m.away_team_id, m.referee_id
    from core.matches m
    join core.seasons se on se.season_id = m.season_id
    where m.kickoff_utc > now()
    order by m.kickoff_utc, m.match_id
""")

SNAPSHOTS = text("""
    select match_id, entity_type, entity_id, features
    from features.feature_snapshots
    where feature_set_version = :version
""")


def feature_names(market: Market) -> list[str]:
    return (
        [f"home_{n}" for n in market.team_features]
        + [f"away_{n}" for n in market.team_features]
        + list(market.referee_features)
        + ["competition_id"]
    )


def _num(value) -> float:
    """None becomes NaN, which LightGBM treats as missing rather than zero."""
    return float("nan") if value is None else float(value)


async def _snapshots(session: AsyncSession) -> dict[tuple[int, str, int], dict]:
    return {
        (row.match_id, row.entity_type, row.entity_id): row.features
        for row in await session.execute(
            SNAPSHOTS, {"version": settings.feature_set_version}
        )
    }


def _vector(
    market: Market,
    home: dict,
    away: dict,
    referee: dict | None,
    competition_id: int,
) -> list[float]:
    """One model input row. Training and prediction both build rows here.

    A match with no referee yet still gets a row: its referee features are NaN,
    which is exactly the situation a forecast made days before kickoff is in.
    """
    referee = referee or {}
    return (
        [_num(home.get(n)) for n in market.team_features]
        + [_num(away.get(n)) for n in market.team_features]
        + [_num(referee.get(n)) for n in market.referee_features]
        + [float(competition_id)]
    )


def _snapshot_triplet(snaps, row) -> tuple[dict | None, dict | None, dict | None]:
    home = snaps.get((row.match_id, "team", row.home_team_id))
    away = snaps.get((row.match_id, "team", row.away_team_id))
    referee = (
        snaps.get((row.match_id, "referee", row.referee_id))
        if row.referee_id is not None
        else None
    )
    return home, away, referee


async def load(session: AsyncSession, market_name: str = "total_corners"):
    """Return (X, y, seasons, kickoffs, feature_names) for one market."""
    market = MARKETS[market_name]
    snaps = await _snapshots(session)

    X: list[list[float]] = []
    y: list[float] = []
    seasons: list[int] = []
    kickoffs: list = []

    for row in (await session.execute(MATCHES)).all():
        values = [
            getattr(row, f"{side}_{column}")
            for column in market.target_columns
            for side in ("h", "a")
        ]
        if any(value is None or value < market.min_per_side for value in values):
            continue
        home, away, referee = _snapshot_triplet(snaps, row)
        if home is None or away is None:
            continue
        X.append(_vector(market, home, away, referee, row.competition_id))
        y.append(sum(values))
        seasons.append(row.season_start.year)
        kickoffs.append(row.kickoff_utc.replace(tzinfo=None))

    return (
        np.array(X, dtype=float),
        np.array(y, dtype=float),
        np.array(seasons),
        np.array(kickoffs, dtype="datetime64[s]"),
        feature_names(market),
    )


async def load_upcoming(session: AsyncSession, market_name: str = "total_corners"):
    """Return (X, match_ids) for fixtures not yet played that have snapshots."""
    market = MARKETS[market_name]
    snaps = await _snapshots(session)

    X: list[list[float]] = []
    match_ids: list[int] = []
    for row in (await session.execute(UPCOMING)).all():
        home, away, referee = _snapshot_triplet(snaps, row)
        if home is None or away is None:
            continue
        X.append(_vector(market, home, away, referee, row.competition_id))
        match_ids.append(row.match_id)

    return np.array(X, dtype=float), match_ids
