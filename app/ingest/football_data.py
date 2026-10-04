"""Results and fixtures from football-data.co.uk: free, no key, no quota.

Our paid provider's trial lapsed, so this keeps the three match markets alive at
no cost. It publishes one CSV per division per season with exactly the statistics
we model -- corners, fouls, yellow and red cards, shots, shots on target -- plus
referee names for England and Scotland, and a fixtures file for the week ahead.

What it does not give: player-level data (so player props cannot be refreshed
from here) and Austria (not published).

A seam to remember: providers count fouls differently. On 2025/26 Premier League
matches, their corners agreed with our stored figures on 98% of matches and their
fouls on 88%. Matches already carrying statistics are therefore left alone --
this source fills gaps, it does not overwrite history.

    python -m app.ingest.football_data --season 2627          # results + fixtures
    python -m app.ingest.football_data --season 2627 --dry-run
"""

import argparse
import asyncio
import csv
import io
import logging
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal, dispose_engine
from app.ingest.market_odds import similarity
from app.models import Competition, EntityMap, Match, Referee, Season, Team, TeamMatchStats

logger = logging.getLogger(__name__)

PROVIDER = "footballdata"
BASE_URL = "https://www.football-data.co.uk"
# Their kickoff times are UK local.
SITE_TZ = ZoneInfo("Europe/London")

# Their division code -> the apifootball league id our competitions are keyed by.
DIVISIONS = {
    "E0": "152", "E1": "153", "SP1": "302", "SP2": "301", "I1": "207", "I2": "206",
    "D1": "175", "D2": "171", "F1": "168", "F2": "164", "N1": "244", "B1": "63",
    "P1": "266", "T1": "322", "G1": "178", "SC0": "279",
}

# Names letters alone cannot connect (checked against our stored teams).
ALIASES = {
    "qpr": "Queens Park Rangers",
    "wolves": "Wolverhampton Wanderers",
    "m'gladbach": "B. Monchengladbach",
    "st etienne": "Saint-Etienne",
    "nijmegen": "NEC",
    "oud-heverlee leuven": "OH Leuven",
    "st. gilloise": "Union Saint-Gilloise",
    "sp braga": "Sporting Braga",
    "sp lisbon": "Sporting CP",
    "sp gijon": "Sporting Gijón",
    "buyuksehyr": "Istanbul Basaksehir",
    # Both score 1.00 against "Manchester Utd" on letters alone, so they must be named.
    "man city": "Manchester City",
    "man united": "Manchester Utd",
    "andorra": "FC Andorra",
    "la coruna": "Dep. A Coruna",
}

# Their column -> our core.team_match_stats column, per side.
STAT_COLUMNS = {
    "corners": ("HC", "AC"),
    "fouls": ("HF", "AF"),
    "yellow": ("HY", "AY"),
    "red": ("HR", "AR"),
    "shots": ("HS", "AS"),
    "shots_on_target": ("HST", "AST"),
}
MIN_SIMILARITY = 0.75
MIN_LEAD = 0.05  # the best name must beat the next best by this much


def _int(value: str | None) -> int | None:
    value = (value or "").strip()
    return int(value) if value.lstrip("-").isdigit() else None


async def fetch_csv(url: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"{url}: HTTP {response.status_code}")
    text_body = response.content.decode("utf-8-sig", errors="replace")
    return [row for row in csv.DictReader(io.StringIO(text_body)) if row.get("Date")]


def parse_kickoff(row: dict) -> datetime:
    """Their Date is dd/mm/yyyy and Time, when present, is UK local."""
    day = datetime.strptime(row["Date"].strip(), "%d/%m/%Y").date()
    clock = (row.get("Time") or "").strip()
    try:
        at = time.fromisoformat(clock) if clock else time(15, 0)
    except ValueError:
        at = time(15, 0)
    return datetime.combine(day, at).replace(tzinfo=SITE_TZ).astimezone(UTC)


class TeamResolver:
    """Their team names to our team ids, league by league."""

    def __init__(self) -> None:
        self.by_league: dict[str, dict[str, int]] = {}
        self.misses: set[str] = set()

    async def load(self, session: AsyncSession) -> None:
        rows = (
            await session.execute(
                text("""
                    select distinct e.provider_id as league, t.team_id, t.name
                    from core.matches m
                    join core.seasons se using (season_id)
                    join core.entity_map e on e.canonical_id = se.competition_id
                     and e.provider_entity_type = 'competition'
                    join core.teams t on t.team_id in (m.home_team_id, m.away_team_id)
                    where m.kickoff_utc >= now() - interval '2 years'
                """)
            )
        ).all()
        for row in rows:
            self.by_league.setdefault(row.league, {})[row.name] = row.team_id

    def resolve(self, league_id: str, name: str) -> int | None:
        """Exact name, then a known alias, then similarity -- but only when one
        team clearly wins. Without that margin two different clubs can land on
        the same team, and a match against itself is worse than no match."""
        ours = self.by_league.get(league_id, {})
        if not ours:
            return None
        target = ALIASES.get(name.strip().casefold(), name)
        if target in ours:
            return ours[target]

        ranked = sorted(((similarity(target, our), our) for our in ours), reverse=True)
        best_score, best = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < MIN_SIMILARITY or best_score - runner_up < MIN_LEAD:
            self.misses.add(f"{league_id}:{name} (best {best} {best_score:.2f})")
            return None
        return ours[best]


async def find_match(
    session: AsyncSession, season_id: int, home_id: int, away_id: int, kickoff: datetime
) -> Match | None:
    """The same fixture in our data, allowing for kickoff times that disagree.

    Two providers rarely agree to the minute, and a fixture can be moved a day.
    Matching on the pairing within a short window avoids creating a duplicate of
    a match we already hold.
    """
    stmt = (
        select(Match)
        .where(
            Match.season_id == season_id,
            Match.home_team_id == home_id,
            Match.away_team_id == away_id,
            Match.kickoff_utc >= kickoff - timedelta(days=3),
            Match.kickoff_utc <= kickoff + timedelta(days=3),
        )
        .order_by(Match.kickoff_utc)
    )
    return (await session.execute(stmt)).scalars().first()


async def season_id_for(session: AsyncSession, league_id: str, on: date) -> int | None:
    competition_id = await session.scalar(
        select(EntityMap.canonical_id).where(
            EntityMap.provider_entity_type == "competition", EntityMap.provider_id == league_id
        )
    )
    if competition_id is None:
        return None
    return await session.scalar(
        select(Season.season_id).where(
            Season.competition_id == competition_id,
            Season.start_date <= on,
            Season.end_date >= on,
        )
    )


async def resolve_referee(session: AsyncSession, name: str) -> int | None:
    """Their referees are 'N Hair'; ours are full names. Only an existing referee
    is used -- inventing one from an initial would split a history we just merged."""
    from app.ingest.fixtures import referee_key

    name = " ".join((name or "").split())
    if not name:
        return None
    key = referee_key(name)
    found = await session.scalar(
        select(EntityMap.canonical_id).where(
            EntityMap.provider_entity_type == "referee", EntityMap.provider_id == key
        )
    )
    if found is not None:
        return found
    # 'N Hair' -> look for a stored referee whose surname and initial agree.
    parts = key.split()
    if len(parts) < 2:
        return None
    initial, surname = parts[0][0], parts[-1]
    candidates = (
        await session.execute(
            select(Referee.referee_id, Referee.name).where(Referee.name.ilike(f"%{surname}"))
        )
    ).all()
    exact = [c for c in candidates if c.name.casefold().startswith(initial)]
    return exact[0].referee_id if len(exact) == 1 else None


async def ingest_results(
    session: AsyncSession, season_code: str, resolver: TeamResolver, dry_run: bool
) -> dict:
    """One division CSV per league: fill in results and statistics we are missing."""
    summary = {"rows": 0, "updated": 0, "created": 0, "stats_added": 0,
               "skipped_existing": 0, "unmatched": 0}
    for code, league_id in DIVISIONS.items():
        try:
            rows = await fetch_csv(f"{BASE_URL}/mmz4281/{season_code}/{code}.csv")
        except Exception as exc:  # noqa: BLE001 -- one division must not stop the rest
            print(f"  {code}: FAILED -- {exc}")
            continue

        changed = stats_added = skipped = unmatched = created = 0
        for row in rows:
            summary["rows"] += 1
            kickoff = parse_kickoff(row)
            season_id = await season_id_for(session, league_id, kickoff.date())
            home_id = resolver.resolve(league_id, row.get("HomeTeam", ""))
            away_id = resolver.resolve(league_id, row.get("AwayTeam", ""))
            if season_id is None or home_id is None or away_id is None or home_id == away_id:
                unmatched += 1
                continue

            home_goals, away_goals = _int(row.get("FTHG")), _int(row.get("FTAG"))
            if home_goals is None or away_goals is None:
                continue  # not played yet

            match = await find_match(session, season_id, home_id, away_id, kickoff)
            if match is None:
                # A match we never received -- the paid feed was cut off while these
                # were played. Create it, rather than leaving a hole in the history.
                created += 1
                if dry_run:
                    continue
                match = Match(
                    season_id=season_id, kickoff_utc=kickoff, home_team_id=home_id,
                    away_team_id=away_id, status="Finished",
                    home_goals=home_goals, away_goals=away_goals,
                    referee_id=await resolve_referee(session, row.get("Referee", "")),
                )
                session.add(match)
                await session.flush()

            if match.status != "Finished" and not dry_run:
                match.status = "Finished"
                match.home_goals, match.away_goals = home_goals, away_goals
                if not match.referee_id and row.get("Referee"):
                    match.referee_id = await resolve_referee(session, row["Referee"])
                changed += 1
            elif match.status != "Finished":
                changed += 1

            for team_id, side in ((home_id, 0), (away_id, 1)):
                existing = await session.get(TeamMatchStats, (match.match_id, team_id))
                if existing is not None and existing.corners is not None:
                    skipped += 1
                    continue  # ours came from the other provider; leave history alone
                values = {
                    column: _int(row.get(columns[side])) for column, columns in STAT_COLUMNS.items()
                }
                if all(value is None for value in values.values()):
                    continue
                if dry_run:
                    stats_added += 1
                    continue
                if existing is None:
                    session.add(TeamMatchStats(match_id=match.match_id, team_id=team_id, **values))
                else:
                    for column, value in values.items():
                        setattr(existing, column, value)
                stats_added += 1

        if not dry_run:
            await session.commit()
        print(f"  {code:4} {len(rows):4} rows -> {changed} results filled, {created} matches added, "
              f"{stats_added} stat rows{f', {unmatched} unmatched' if unmatched else ''}")
        summary["updated"] += changed
        summary["created"] += created
        summary["stats_added"] += stats_added
        summary["skipped_existing"] += skipped
        summary["unmatched"] += unmatched
    return summary


async def ingest_fixtures(session: AsyncSession, resolver: TeamResolver, dry_run: bool) -> dict:
    """Their one fixtures file covers the week ahead across every division."""
    rows = await fetch_csv(f"{BASE_URL}/fixtures.csv")
    summary = {"rows": len(rows), "created": 0, "known": 0, "unmatched": 0}
    for row in rows:
        league_id = DIVISIONS.get((row.get("Div") or "").strip())
        if league_id is None:
            continue
        kickoff = parse_kickoff(row)
        season_id = await season_id_for(session, league_id, kickoff.date())
        home_id = resolver.resolve(league_id, row.get("HomeTeam", ""))
        away_id = resolver.resolve(league_id, row.get("AwayTeam", ""))
        if season_id is None or home_id is None or away_id is None or home_id == away_id:
            summary["unmatched"] += 1
            continue

        match = await find_match(session, season_id, home_id, away_id, kickoff)
        if match is not None:
            summary["known"] += 1
            if row.get("Referee") and not match.referee_id and not dry_run:
                match.referee_id = await resolve_referee(session, row["Referee"])
            continue
        summary["created"] += 1
        if dry_run:
            continue
        session.add(
            Match(
                season_id=season_id, kickoff_utc=kickoff, home_team_id=home_id,
                away_team_id=away_id, status="Not Started",
                referee_id=await resolve_referee(session, row.get("Referee", "")),
            )
        )
    if not dry_run:
        await session.commit()
    return summary


def current_season_code(today: date | None = None) -> str:
    """Their season code: 2627 for 2026/27. Seasons run July to June."""
    today = today or datetime.now(UTC).date()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


async def loaded_resolver(session: AsyncSession) -> TeamResolver:
    resolver = TeamResolver()
    await resolver.load(session)
    return resolver


async def run(season_code: str, dry_run: bool, fixtures_only: bool) -> None:
    try:
        async with SessionLocal() as session:
            resolver = TeamResolver()
            await resolver.load(session)

            if not fixtures_only:
                print(f"results, season {season_code}:")
                results = await ingest_results(session, season_code, resolver, dry_run)
                print(
                    f"  total: {results['rows']} rows, {results['updated']} results filled, "
                    f"{results['created']} matches added, "
                    f"{results['stats_added']} statistic rows written, "
                    f"{results['skipped_existing']} left as they were, "
                    f"{results['unmatched']} unmatched"
                )

            fixtures = await ingest_fixtures(session, resolver, dry_run)
            print(
                f"fixtures: {fixtures['rows']} rows in their file, {fixtures['created']} new, "
                f"{fixtures['known']} already known, {fixtures['unmatched']} unmatched"
            )
            if resolver.misses:
                print(f"  names not resolved: {sorted(resolver.misses)[:10]}")
            if dry_run:
                print("\nDRY RUN -- nothing written")
    finally:
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest free results and fixtures.")
    parser.add_argument("--season", default="2627", help="their season code, e.g. 2627 for 2026/27")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args.season, args.dry_run, args.fixtures_only))


if __name__ == "__main__":
    main()
