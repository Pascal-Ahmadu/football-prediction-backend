"""Team rates and opponent-adjusted ratings (FR-FEAT-01, FR-FEAT-02).

Point-in-time correctness (B1.1, risk R-02): a finished match's snapshot contains
only information from matches played strictly before its kickoff. Two things
enforce that -- the snapshot is WRITTEN before the match is folded into the
running averages, and ratings are refit only from matches already folded in.

Upcoming fixtures get a snapshot of each team's state as of the moment this runs
(computed_as_of = now). Re-run before each matchweek so those snapshots include
the latest results.

Run:
    python -m app.features.rates
"""

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import SessionLocal, dispose_engine
from app.models import FeatureSnapshot, Match, Season, TeamMatchStats

logger = logging.getLogger(__name__)

STATS = (
    "corners",
    "fouls",
    "yellow",
    "red",
    "shots",
    "shots_on_target",
    "possession",
)

RATED = ("corners", "fouls", "yellow")


def decay_factor(half_life: float) -> float:
    """Weight multiplier per match into the past."""
    return 0.5 ** (1.0 / half_life)


def _fold(
    ewma: dict[tuple[int, str], float],
    team_id: int,
    key: str,
    value: float | None,
    lam: float,
) -> None:
    """Fold one observation into the running exponentially weighted mean."""
    if value is None:
        return
    previous = ewma.get((team_id, key))
    ewma[(team_id, key)] = (
        float(value) if previous is None else (1 - lam) * float(value) + lam * previous
    )


def fit_ratings(rows: list[tuple[int, int, int, float]], lam: float) -> dict[int, tuple[float, float]]:
    """Ridge-fit attack and defence ratings from (team, opponent, is_home, value).

    Solves  value = mean + attack[team] + defence[opponent] + home * is_home
    for every team simultaneously. Ridge (the lam * I term) both keeps thin
    samples from producing wild ratings and resolves the fact that adding a
    constant to every attack and subtracting it from every defence would
    otherwise fit equally well.
    """
    teams = sorted({r[0] for r in rows} | {r[1] for r in rows})
    index = {team: i for i, team in enumerate(teams)}
    k = len(teams)

    X = np.zeros((len(rows), 2 * k + 1))
    y = np.zeros(len(rows))
    for i, (team, opponent, is_home, value) in enumerate(rows):
        X[i, index[team]] = 1.0
        X[i, k + index[opponent]] = 1.0
        X[i, 2 * k] = float(is_home)
        y[i] = value

    centred = y - y.mean()
    normal = X.T @ X + lam * np.eye(2 * k + 1)
    beta = np.linalg.solve(normal, X.T @ centred)
    return {team: (float(beta[index[team]]), float(beta[k + index[team]])) for team in teams}


class TeamState:
    """Everything known about every team at one moment in the walk through time."""

    def __init__(self) -> None:
        self.ewma: dict[tuple[int, str], float] = {}
        self.played: dict[int, int] = defaultdict(int)
        self.played_venue: dict[tuple[int, str], int] = defaultdict(int)
        self.history: dict[tuple[int, str], list] = defaultdict(list)
        self.ratings: dict[tuple[int, str], dict[int, tuple[float, float]]] = {}

    def refit_ratings(self) -> None:
        for key, observations in self.history.items():
            if len(observations) >= settings.rating_min_rows:
                self.ratings[key] = fit_ratings(observations, settings.rating_lambda)

    def features(self, team_id: int, venue: str, competition_id: int) -> dict:
        """The one place features are built -- for training AND for prediction."""
        features: dict[str, float | int | None] = {
            "matches_seen": self.played[team_id],
            "matches_seen_venue": self.played_venue[(team_id, venue)],
            "is_home": int(venue == "home"),
        }
        for name in STATS:
            features[f"{name}_for"] = self.ewma.get((team_id, f"{name}_for"))
            features[f"{name}_against"] = self.ewma.get((team_id, f"{name}_against"))
            features[f"{name}_for_venue"] = self.ewma.get((team_id, f"{name}_for_{venue}"))
            features[f"{name}_against_venue"] = self.ewma.get(
                (team_id, f"{name}_against_{venue}")
            )
        for name in RATED:
            table = self.ratings.get((competition_id, name), {})
            attack, defence = table.get(team_id, (None, None))
            features[f"{name}_attack"] = attack
            features[f"{name}_defence"] = defence
        return features

    def fold(self, row, opponent, lam: float) -> None:
        """Absorb one finished match from one team's side."""
        venue = "home" if row.team_id == row.home_team_id else "away"
        for name in STATS:
            value, conceded = getattr(row, name), getattr(opponent, name)
            _fold(self.ewma, row.team_id, f"{name}_for", value, lam)
            _fold(self.ewma, row.team_id, f"{name}_against", conceded, lam)
            _fold(self.ewma, row.team_id, f"{name}_for_{venue}", value, lam)
            _fold(self.ewma, row.team_id, f"{name}_against_{venue}", conceded, lam)
        for name in RATED:
            value = getattr(row, name)
            if value is not None:
                self.history[(row.competition_id, name)].append(
                    (row.team_id, opponent.team_id, int(venue == "home"), float(value))
                )
        self.played[row.team_id] += 1
        self.played_venue[(row.team_id, venue)] += 1


class RefereeState:
    """What each referee's recent matches say about how they officiate (FR-FEAT-03)."""

    def __init__(self) -> None:
        self.ewma: dict[tuple[int, str], float] = {}
        self.matches: dict[int, int] = defaultdict(int)

    def features(self, referee_id: int) -> dict:
        cards = self.ewma.get((referee_id, "cards"))
        fouls = self.ewma.get((referee_id, "fouls"))
        return {
            "ref_matches_seen": self.matches[referee_id],
            "ref_cards_pm": cards,
            "ref_yellow_pm": self.ewma.get((referee_id, "yellow")),
            "ref_red_pm": self.ewma.get((referee_id, "red")),
            "ref_fouls_pm": fouls,
            "ref_cards_per_foul": cards / fouls if cards is not None and fouls else None,
            "ref_home_card_share": self.ewma.get((referee_id, "home_share")),
        }

    def fold(self, referee_id: int, home, away, lam: float) -> None:
        """Absorb one finished match this referee officiated."""

        def total(name: str) -> int | None:
            h, a = getattr(home, name), getattr(away, name)
            return None if h is None or a is None else h + a

        yellow, red, fouls = total("yellow"), total("red"), total("fouls")
        cards = None if yellow is None or red is None else yellow + red
        _fold(self.ewma, referee_id, "yellow", yellow, lam)
        _fold(self.ewma, referee_id, "red", red, lam)
        _fold(self.ewma, referee_id, "cards", cards, lam)
        _fold(self.ewma, referee_id, "fouls", fouls, lam)
        if cards:  # the home share of a card-free match is undefined, not zero
            home_cards = (home.yellow or 0) + (home.red or 0)
            _fold(self.ewma, referee_id, "home_share", home_cards / cards, lam)
        self.matches[referee_id] += 1


async def build(session: AsyncSession) -> tuple[int, int]:
    """Rebuild this feature version. Returns (historical, upcoming) snapshot counts,
    team and referee snapshots together."""
    lam = decay_factor(settings.feature_half_life)
    referee_lam = decay_factor(settings.referee_half_life)
    version = settings.feature_set_version

    await session.execute(
        delete(FeatureSnapshot).where(FeatureSnapshot.feature_set_version == version)
    )

    stmt = (
        select(
            Match.match_id,
            Match.kickoff_utc,
            Match.home_team_id,
            Match.referee_id,
            Season.competition_id,
            TeamMatchStats.team_id,
            *(getattr(TeamMatchStats, name) for name in STATS),
        )
        .join(TeamMatchStats, TeamMatchStats.match_id == Match.match_id)
        .join(Season, Season.season_id == Match.season_id)
        .where(Match.status == "Finished")
        .order_by(Match.kickoff_utc, Match.match_id, TeamMatchStats.team_id)
    )
    rows = (await session.execute(stmt)).all()

    sides: dict[int, list] = defaultdict(list)
    order: list[int] = []
    for row in rows:
        if row.match_id not in sides:
            order.append(row.match_id)
        sides[row.match_id].append(row)

    state = TeamState()
    referees = RefereeState()
    current_week: tuple[int, int] | None = None
    historical = 0

    for match_id in order:
        pair = sides[match_id]
        if len(pair) != 2:
            continue

        week = pair[0].kickoff_utc.isocalendar()[:2]
        if week != current_week:
            current_week = week
            state.refit_ratings()

        for row in pair:
            venue = "home" if row.team_id == row.home_team_id else "away"
            session.add(
                FeatureSnapshot(
                    match_id=match_id,
                    entity_type="team",
                    entity_id=row.team_id,
                    feature_set_version=version,
                    computed_as_of=row.kickoff_utc,
                    features=state.features(row.team_id, venue, row.competition_id),
                )
            )
            historical += 1

        referee_id = pair[0].referee_id
        if referee_id is not None:
            session.add(
                FeatureSnapshot(
                    match_id=match_id,
                    entity_type="referee",
                    entity_id=referee_id,
                    feature_set_version=version,
                    computed_as_of=pair[0].kickoff_utc,
                    features=referees.features(referee_id),
                )
            )
            historical += 1

        # Only now may this match influence anything.
        state.fold(pair[0], pair[1], lam)
        state.fold(pair[1], pair[0], lam)
        if referee_id is not None:
            home, away = pair if pair[0].team_id == pair[0].home_team_id else pair[::-1]
            referees.fold(referee_id, home, away, referee_lam)

    # Upcoming fixtures: every team's state as of right now.
    now = datetime.now(UTC)
    state.refit_ratings()
    fixtures = await session.execute(
        select(
            Match.match_id,
            Match.home_team_id,
            Match.away_team_id,
            Match.referee_id,
            Season.competition_id,
        )
        .join(Season, Season.season_id == Match.season_id)
        .where(Match.kickoff_utc > now)
    )
    upcoming = 0
    for fixture in fixtures:
        for team_id, venue in (
            (fixture.home_team_id, "home"),
            (fixture.away_team_id, "away"),
        ):
            session.add(
                FeatureSnapshot(
                    match_id=fixture.match_id,
                    entity_type="team",
                    entity_id=team_id,
                    feature_set_version=version,
                    computed_as_of=now,
                    features=state.features(team_id, venue, fixture.competition_id),
                )
            )
            upcoming += 1
        if fixture.referee_id is not None:
            session.add(
                FeatureSnapshot(
                    match_id=fixture.match_id,
                    entity_type="referee",
                    entity_id=fixture.referee_id,
                    feature_set_version=version,
                    computed_as_of=now,
                    features=referees.features(fixture.referee_id),
                )
            )
            upcoming += 1

    await session.commit()
    return historical, upcoming


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        async with SessionLocal() as session:
            historical, upcoming = await build(session)
        print(
            f"wrote {historical} historical and {upcoming} upcoming feature snapshots "
            f"({settings.feature_set_version})"
        )
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
