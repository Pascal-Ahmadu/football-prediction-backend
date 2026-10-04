"""Player features for player props (FR-FEAT-04).

Same point-in-time rule as the team features (risk R-02): a player's snapshot for
a match is WRITTEN before that match is folded into anything, so every feature
comes from strictly earlier matches.

Two kinds of feature:

  rates       how much a player does when he plays -- shots, shots on target and
              fouls per 90 minutes, as exponentially weighted averages over his
              appearances (half-life settings.player_half_life appearances)
  involvement how likely he is to play, and for how long -- minutes, starts and
              appearances over his TEAM's last 5 matches, and how many team
              matches have passed since he last played. These are counted per
              team match, so a player who was injured or dropped (and therefore
              not listed at all) shows up as absent rather than as missing data.

Snapshots are stored as entity_type 'player' under settings.player_feature_set_version,
for every listed player (unused substitutes included) and, for upcoming fixtures,
for every player who played for the team in any of its last 5 matches.

Run:
    python -m app.features.players
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import UTC, datetime

from sqlalchemy import delete, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import SessionLocal, dispose_engine
from app.features.rates import decay_factor
from app.models import FeatureSnapshot, Match, Season

logger = logging.getLogger(__name__)

RATE_STATS = ("shots", "shots_on_target", "fouls_committed")
RECENT_TEAM_MATCHES = 5
POSITIONS = {"Goalkeepers": 0, "Defenders": 1, "Midfielders": 2, "Forwards": 3}
INSERT_CHUNK = 5_000

PLAYER_ROWS = text("""
    select p.match_id, m.kickoff_utc, p.player_id, p.team_id, p.is_starter, p.position,
           p.minutes, p.shots, p.shots_on_target, p.fouls_committed
    from core.player_match_stats p
    join core.matches m on m.match_id = p.match_id
    where m.status = 'Finished'
    order by m.kickoff_utc, p.match_id, p.team_id, p.player_id
""")


class PlayerState:
    """Everything known about every player at one moment in the walk through time."""

    def __init__(self, lam: float) -> None:
        self.lam = lam
        # Exponentially weighted SUMS, with the total weight behind them, rather
        # than a running mean. A running mean starts at the first observation and
        # moves only (1 - lam) per match, so after three appearances it still
        # mostly reports the first one: a player whose debut was a 4-minute cameo
        # looked like a 4-minute player ever after. Dividing sum by weight gives
        # the correct weighted average from the very first appearance, and rates
        # divide two sums weighted alike.
        self.totals: dict[tuple[int, str], float] = defaultdict(float)
        self.weight: dict[int, float] = defaultdict(float)
        self.appearances: dict[int, int] = defaultdict(int)
        self.minutes_total: dict[int, int] = defaultdict(int)
        self.position: dict[int, int] = {}
        self.last_team: dict[int, int] = {}
        # For each team, its last few matches: {player_id: (minutes, is_starter)}.
        self.recent: dict[int, deque] = defaultdict(lambda: deque(maxlen=RECENT_TEAM_MATCHES))
        self.team_matches: dict[int, int] = defaultdict(int)
        self.last_played_at: dict[tuple[int, int], int] = {}

    def features(self, player_id: int, team_id: int) -> dict:
        """The one place player features are built -- for training AND for prediction."""
        weight = self.weight[player_id]
        minutes = self.totals[(player_id, "minutes")]
        features: dict[str, float | int | None] = {
            "team_id": team_id,
            "position": self.position.get(player_id),
            "appearances_seen": self.appearances[player_id],
            "minutes_seen": self.minutes_total[player_id],
            "minutes_pa": minutes / weight if weight else None,
        }
        for name in RATE_STATS:
            features[f"{name}_p90"] = (
                90.0 * self.totals[(player_id, name)] / minutes if minutes else None
            )

        recent = self.recent[team_id]
        features["team_matches_recent"] = len(recent)
        features["minutes_recent"] = sum(match.get(player_id, (0, False))[0] for match in recent)
        features["starts_recent"] = sum(1 for match in recent if match.get(player_id, (0, False))[1])
        features["apps_recent"] = sum(1 for match in recent if match.get(player_id, (0, False))[0] > 0)
        last = self.last_played_at.get((team_id, player_id))
        features["team_matches_since_played"] = (
            None if last is None else self.team_matches[team_id] - last
        )
        return features

    def fold_team_match(self, team_id: int, rows: list) -> None:
        """Absorb one finished match from one team's side: every listed player."""
        self.team_matches[team_id] += 1
        involvement = {}
        for row in rows:
            minutes = row.minutes or 0
            involvement[row.player_id] = (minutes, bool(row.is_starter))
            if row.position in POSITIONS:
                self.position[row.player_id] = POSITIONS[row.position]
            if minutes <= 0:
                continue
            self.last_played_at[(team_id, row.player_id)] = self.team_matches[team_id]
            self.last_team[row.player_id] = team_id
            self.appearances[row.player_id] += 1
            self.minutes_total[row.player_id] += minutes
            self.weight[row.player_id] = 1.0 + self.lam * self.weight[row.player_id]
            self.totals[(row.player_id, "minutes")] = (
                minutes + self.lam * self.totals[(row.player_id, "minutes")]
            )
            for name in RATE_STATS:
                value = getattr(row, name) or 0
                self.totals[(row.player_id, name)] = (
                    value + self.lam * self.totals[(row.player_id, name)]
                )
        self.recent[team_id].append(involvement)

    def candidates(self, team_id: int) -> list[int]:
        """Players likely available to a team: played for it recently, not moved on since."""
        played = {
            player_id
            for match in self.recent[team_id]
            for player_id, (minutes, _starter) in match.items()
            if minutes > 0
        }
        return sorted(p for p in played if self.last_team.get(p) == team_id)


async def _insert(session: AsyncSession, rows: list[dict]) -> None:
    for start in range(0, len(rows), INSERT_CHUNK):
        await session.execute(insert(FeatureSnapshot), rows[start:start + INSERT_CHUNK])


async def build(session: AsyncSession) -> tuple[int, int]:
    """Rebuild player snapshots. Returns (historical, upcoming) counts."""
    version = settings.player_feature_set_version
    state = PlayerState(decay_factor(settings.player_half_life))

    await session.execute(
        delete(FeatureSnapshot).where(
            FeatureSnapshot.feature_set_version == version,
            FeatureSnapshot.entity_type == "player",
        )
    )

    rows = (await session.execute(PLAYER_ROWS)).all()
    pending: list[dict] = []
    historical = 0

    def team_groups(match_rows):
        by_team = defaultdict(list)
        for row in match_rows:
            by_team[row.team_id].append(row)
        return by_team

    start = 0
    while start < len(rows):
        end = start
        while end < len(rows) and rows[end].match_id == rows[start].match_id:
            end += 1
        match_rows = rows[start:end]
        by_team = team_groups(match_rows)

        for team_id, team_rows in by_team.items():
            for row in team_rows:
                pending.append({
                    "match_id": row.match_id,
                    "entity_type": "player",
                    "entity_id": row.player_id,
                    "feature_set_version": version,
                    "computed_as_of": row.kickoff_utc,
                    "features": state.features(row.player_id, team_id),
                })
        historical += len(match_rows)

        # Only now may this match influence anything.
        for team_id, team_rows in by_team.items():
            state.fold_team_match(team_id, team_rows)

        if len(pending) >= INSERT_CHUNK:
            await _insert(session, pending)
            pending = []
        start = end

    now = datetime.now(UTC)
    fixtures = await session.execute(
        select(Match.match_id, Match.home_team_id, Match.away_team_id)
        .join(Season, Season.season_id == Match.season_id)
        .where(Match.kickoff_utc > now)
    )
    upcoming = 0
    for fixture in fixtures:
        for team_id in (fixture.home_team_id, fixture.away_team_id):
            for player_id in state.candidates(team_id):
                pending.append({
                    "match_id": fixture.match_id,
                    "entity_type": "player",
                    "entity_id": player_id,
                    "feature_set_version": version,
                    "computed_as_of": now,
                    "features": state.features(player_id, team_id),
                })
                upcoming += 1

    await _insert(session, pending)
    await session.commit()
    return historical, upcoming


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        async with SessionLocal() as session:
            historical, upcoming = await build(session)
        print(
            f"wrote {historical} historical and {upcoming} upcoming player snapshots "
            f"({settings.player_feature_set_version})"
        )
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
