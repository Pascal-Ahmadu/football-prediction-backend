"""Training sets for player props, one market at a time.

One row per player appearance (minutes > 0): bookmakers void a player prop when
the player does not play, so the model predicts the count GIVEN that he plays.
Whether and how long he plays is still unknown before kickoff, which is what the
involvement features (minutes_recent, starts_recent, ...) are for.

Each row joins four point-in-time snapshots taken before the match: the player's,
his team's, the opponent's and the referee's. Values are extracted in SQL, which
is far faster than decoding ~1M JSON documents in Python.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

PLAYER_COMMON = (
    "position",
    "appearances_seen",
    "minutes_pa",
    "team_matches_recent",
    "minutes_recent",
    "starts_recent",
    "apps_recent",
    "team_matches_since_played",
)


@dataclass(frozen=True)
class PlayerMarket:
    """What one player market predicts and which snapshot features it reads."""

    name: str
    target: str  # a core.player_match_stats column
    player_features: tuple[str, ...]
    team_features: tuple[str, ...] = ()
    opponent_features: tuple[str, ...] = ()
    referee_features: tuple[str, ...] = ()


PLAYER_MARKETS = {
    "player_shots_on_target": PlayerMarket(
        name="player_shots_on_target",
        target="shots_on_target",
        player_features=PLAYER_COMMON + ("shots_p90", "shots_on_target_p90"),
        team_features=("is_home", "shots_for", "shots_on_target_for", "possession_for"),
        opponent_features=("shots_against", "shots_on_target_against"),
    ),
    "player_fouls_committed": PlayerMarket(
        name="player_fouls_committed",
        target="fouls_committed",
        player_features=PLAYER_COMMON + ("fouls_committed_p90",),
        team_features=("is_home", "fouls_for", "possession_for"),
        opponent_features=("fouls_against",),
        # No referee features: on 2024/25 they changed log loss by 0.0001, and
        # a model trained with them risks degrading when the referee is not yet
        # named (as the cards model did).
    ),
}


def feature_names(market: PlayerMarket) -> list[str]:
    return (
        list(market.player_features)
        + [f"team_{n}" for n in market.team_features]
        + [f"opp_{n}" for n in market.opponent_features]
        + list(market.referee_features)
        + ["competition_id"]
    )


def _json_float(alias: str, name: str) -> str:
    return f"({alias}.features->>'{name}')::float"


def _columns(market: PlayerMarket) -> list[str]:
    return (
        [_json_float("pf", n) for n in market.player_features]
        + [_json_float("tf", n) for n in market.team_features]
        + [_json_float("opf", n) for n in market.opponent_features]
        + [_json_float("rf", n) for n in market.referee_features]
    )


def _query(market: PlayerMarket) -> str:
    columns = _columns(market)
    return f"""
        select p.match_id, m.kickoff_utc, se.start_date as season_start, se.competition_id,
               p.player_id, p.minutes, p.{market.target} as target, m.referee_id,
               {", ".join(f"{c} as f{i}" for i, c in enumerate(columns))}
        from core.player_match_stats p
        join core.matches m on m.match_id = p.match_id and m.status = 'Finished'
        join core.seasons se on se.season_id = m.season_id
        join features.feature_snapshots pf
          on pf.match_id = p.match_id and pf.entity_type = 'player'
         and pf.entity_id = p.player_id and pf.feature_set_version = :player_version
        left join features.feature_snapshots tf
          on tf.match_id = p.match_id and tf.entity_type = 'team'
         and tf.entity_id = p.team_id and tf.feature_set_version = :team_version
        left join features.feature_snapshots opf
          on opf.match_id = p.match_id and opf.entity_type = 'team'
         and opf.entity_id = case when p.team_id = m.home_team_id
                                  then m.away_team_id else m.home_team_id end
         and opf.feature_set_version = :team_version
        left join features.feature_snapshots rf
          on rf.match_id = p.match_id and rf.entity_type = 'referee'
         and rf.entity_id = m.referee_id and rf.feature_set_version = :team_version
        where p.minutes > 0 and p.{market.target} is not null
        order by m.kickoff_utc, p.match_id, p.player_id
    """


async def load(session: AsyncSession, market_name: str):
    """Return (X, y, seasons, kickoffs, names, info) for one player market.

    info holds per-row match_id, player_id, minutes and referee_known, for
    evaluation only -- minutes played is NOT a feature (it is unknown before kickoff).
    """
    market = PLAYER_MARKETS[market_name]
    names = feature_names(market)
    n_features = len(names) - 1

    rows = (
        await session.execute(
            text(_query(market)),
            {
                "player_version": settings.player_feature_set_version,
                "team_version": settings.feature_set_version,
            },
        )
    ).all()

    X = np.full((len(rows), len(names)), np.nan)
    y = np.empty(len(rows))
    seasons = np.empty(len(rows), dtype=int)
    kickoffs = np.empty(len(rows), dtype="datetime64[s]")
    info = {
        "match_id": np.empty(len(rows), dtype=int),
        "player_id": np.empty(len(rows), dtype=int),
        "minutes": np.empty(len(rows), dtype=int),
        "referee_known": np.empty(len(rows), dtype=bool),
    }
    first_feature = 8  # columns before f0 in the query
    for i, row in enumerate(rows):
        for j in range(n_features):
            value = row[first_feature + j]
            if value is not None:
                X[i, j] = value
        X[i, -1] = row.competition_id
        y[i] = row.target
        seasons[i] = row.season_start.year
        kickoffs[i] = np.datetime64(row.kickoff_utc.replace(tzinfo=None), "s")
        info["match_id"][i] = row.match_id
        info["player_id"][i] = row.player_id
        info["minutes"][i] = row.minutes
        info["referee_known"][i] = row.referee_id is not None

    return X, y, seasons, kickoffs, names, info


UPCOMING = """
    select pf.match_id, pf.entity_id as player_id, pl.name as player_name,
           (pf.features->>'team_id')::int as team_id, se.competition_id,
           m.kickoff_utc,
           {columns}
    from features.feature_snapshots pf
    join core.matches m on m.match_id = pf.match_id
    join core.seasons se on se.season_id = m.season_id
    join core.players pl on pl.player_id = pf.entity_id
    left join features.feature_snapshots tf
      on tf.match_id = pf.match_id and tf.entity_type = 'team'
     and tf.entity_id = (pf.features->>'team_id')::int
     and tf.feature_set_version = :team_version
    left join features.feature_snapshots opf
      on opf.match_id = pf.match_id and opf.entity_type = 'team'
     and opf.entity_id = case when (pf.features->>'team_id')::int = m.home_team_id
                              then m.away_team_id else m.home_team_id end
     and opf.feature_set_version = :team_version
    left join features.feature_snapshots rf
      on rf.match_id = pf.match_id and rf.entity_type = 'referee'
     and rf.entity_id = m.referee_id and rf.feature_set_version = :team_version
    where pf.entity_type = 'player' and pf.feature_set_version = :player_version
      and m.kickoff_utc > now()
    order by m.kickoff_utc, pf.match_id, pf.entity_id
"""


async def load_upcoming(session: AsyncSession, market_name: str):
    """Return (X, rows) for players expected to feature in fixtures not yet played.

    rows carries match_id, player_id, player_name and the baseline expectation a
    naive forecast would give him, which is what player picks are ranked against.
    """
    market = PLAYER_MARKETS[market_name]
    names = feature_names(market)
    columns = ", ".join(f"{c} as f{i}" for i, c in enumerate(_columns(market)))
    statement = UPCOMING.format(columns=columns)

    result = (
        await session.execute(
            text(statement),
            {
                "player_version": settings.player_feature_set_version,
                "team_version": settings.feature_set_version,
            },
        )
    ).all()

    first_feature = 6
    X = np.full((len(result), len(names)), np.nan)
    rows = []
    for i, row in enumerate(result):
        for j in range(len(names) - 1):
            value = row[first_feature + j]
            if value is not None:
                X[i, j] = value
        X[i, -1] = row.competition_id
        rows.append({
            "match_id": row.match_id,
            "player_id": row.player_id,
            "player_name": row.player_name,
            "team_id": row.team_id,
            "kickoff_utc": row.kickoff_utc,
        })
    return X, rows, names
