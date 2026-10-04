"""Injest fixtures"""

from app.ingest.schemas import ProviderEvent
import argparse
import asyncio
import hashlib
import json
import logging
import unicodedata
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import Base, SessionLocal, dispose_engine
from app.ingest.client import PROVIDER, ApiFootballClient, ProviderError
from app.models import (
    Competition,
    EntityMap,
    Match,
    Player,
    PlayerMatchStats,
    RawPayloadRef,
    Referee,
    RejectedPayload,
    Season,
    Team,
    TeamMatchStats,
    Venue,
)

RAW_DIR = Path("data")/ "raw" / PROVIDER

COUNT_STATS = {
    "corners": "Corners",
    "fouls": "Fouls",
    "shots": "Shots Total",
    "shots_on_target": "Shots On Goal",
    "offsides": "Offsides",
    "saves": "Saves",
    "attacks": "Attacks",
    "dangerous_attacks": "Dangerous Attacks",
    "passes": "Passes Total",
    "passes_accurate": "Passes Accurate",
}

# Cards are resolved by ProviderEvent.card_count, which falls back to the
# booking events when the statistic is missing: (statistic name, booking label).
CARD_STATS = {
    "yellow": ("Yellow Cards", "yellow card"),
    "red": ("Red Cards", "red card"),
}

# The provider omits these rows entirely when the count is zero, so on a
# finished match "absent" means 0, not "unknown".
ZERO_WHEN_ABSENT = {"offsides"}

# core.player_match_stats column -> apifootball player_stats field (spelling theirs).
PLAYER_COUNT_STATS = {
    "minutes": "player_minutes_played",
    "shots": "player_total_shots",
    "shots_on_target": "player_shots_on_goal",
    "fouls_committed": "player_fouls_commited",
    "yellow": "player_yellow_cards",
    "red": "player_red_cards",
    "goals": "player_goals",
    "assists": "player_assists",
    "offsides": "player_offsides",
    "tackles": "player_tackles",
    "passes": "player_passes",
}


def competition_name(league_name: str) -> str:
    """'Championship - Promotion Play-offs - Final' -> 'Championship'.

    The provider appends the stage of each match to the league name, so the
    stored name would otherwise depend on which match happened to be loaded first.
    """
    return league_name.split(" - ")[0].strip()


def referee_base_name(name: str) -> str:
    """'David Webb, England' -> 'David Webb'. The provider sometimes appends a country."""
    name = " ".join(name.split())
    return name.rsplit(",", 1)[0].strip() if "," in name else name


def fold_accents(text_value: str) -> str:
    """'Badstübner' -> 'Badstubner'. The provider dropped accents from 2026."""
    decomposed = unicodedata.normalize("NFKD", text_value)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def referee_key(name: str) -> str:
    """Entity-map key for a referee.

    Ignores case, accents and a trailing ', Country', so 'Carlos Del Cerro',
    'Carlos del Cerro' and 'Carlos del Cerro, Spain' are one referee. Abbreviated
    forms ('C. del Cerro') are linked by app.ingest.merge_referees instead, since
    an initial alone could belong to two different people.
    """
    return fold_accents(referee_base_name(name)).casefold()


def save_raw(payload: Any) -> tuple[str, str]:
    """Write the untouch response to disk, named by SHA-256"""
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    path = RAW_DIR / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return digest, path.as_posix()

async def resolve(
    session: AsyncSession,
    entity_type: str,
    provider_id: str,
    make: Callable[[], Base],
    id_attr: str,
) -> int:
    """Given or create: our id for a provider, create and store if absent"""
    mapping = await session.get(
        EntityMap,
        (PROVIDER,
        entity_type,
        provider_id,
        )
        )
    if mapping is not None:
        return mapping.canonical_id

    obj = make()
    session.add(obj)
    await session.flush()
    canonical_id = getattr(obj, id_attr)
    session.add(
        EntityMap(
            provider=PROVIDER,
            provider_entity_type=entity_type,
            provider_id=provider_id,
            canonical_id=canonical_id,
        )
    )
    await session.flush()
    return canonical_id

async def resolve_season(session: AsyncSession, competition_id: int, on: date) -> int:
    """The seasion containing a date. Sessions run 1st July to 30th June"""
    stmt = select(Season).where(
        Season.competition_id == competition_id,
        Season.start_date <= on,
        Season.end_date >= on,
    )

    season = (await session.execute(stmt)).scalar_one_or_none()
    if season is not None:
        return season.season_id

    start_year = on.year if on.month >= 7 else on.year -1
    season = Season(
        competition_id = competition_id,
        start_date=date(start_year, 7, 1),
        end_date = date(start_year + 1, 6, 30),
    )
    session.add(season)
    await session.flush()
    return season.season_id

async def upsert_team_stats(
    session: AsyncSession, ev: ProviderEvent, match_id: int, home_id: int, away_id: int
) -> None:
    """Write one core.team_match_stats row per side (B4)."""
    if not ev.statistics:
        return

    home: dict[str, float | None] = {}
    away: dict[str, float | None] = {}

    for column, stat_name in COUNT_STATS.items():
        pair = ev.stat(stat_name)
        if pair is None:
            default = 0 if (ev.is_finished and column in ZERO_WHEN_ABSENT) else None
            home[column] = away[column] = default
        else:
            home[column], away[column] = pair

    for column, (stat_name, label) in CARD_STATS.items():
        pair = ev.card_count(stat_name, label) if ev.is_finished else ev.stat(stat_name)
        home[column], away[column] = pair if pair is not None else (None, None)

    possession = ev.percentage("Ball Possession")
    if possession is not None and not 99 <= possession[0] + possession[1] <= 101:
        possession = None
    home["possession"], away["possession"] = possession or (None, None)

    for team_id, values in ((home_id, home), (away_id, away)):
        row = await session.get(TeamMatchStats, (match_id, team_id))
        if row is None:
            session.add(TeamMatchStats(match_id=match_id, team_id=team_id, **values))
        else:
            for column, value in values.items():
                setattr(row, column, value)

    await session.flush()


def _count(value: Any) -> int | None:
    text_value = str(value).strip()
    return int(text_value) if text_value.isdigit() else None


def _rating(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except ValueError:
        return None


async def upsert_player_stats(
    session: AsyncSession, ev: ProviderEvent, match_id: int, home_id: int, away_id: int
) -> int:
    """Write one core.player_match_stats row per listed player. Returns rows written.

    Batched per match: one lookup for all ~40 player ids, one insert for new
    players, one upsert for the stats -- instead of several round trips per player.
    """
    listed: dict[str, tuple[dict, int]] = {}
    for side, team_id in (("home", home_id), ("away", away_id)):
        for player in ev.players(side):
            key = str(player.get("player_key", "")).strip()
            if key and key not in listed:
                listed[key] = (player, team_id)
    if not listed:
        return 0

    known = dict(
        (
            await session.execute(
                select(EntityMap.provider_id, EntityMap.canonical_id).where(
                    EntityMap.provider == PROVIDER,
                    EntityMap.provider_entity_type == "player",
                    EntityMap.provider_id.in_(listed),
                )
            )
        ).all()
    )
    new_keys = [key for key in listed if key not in known]
    if new_keys:
        new_players = [
            Player(name=str(listed[key][0].get("player_name", "")).strip()[:120] or key)
            for key in new_keys
        ]
        session.add_all(new_players)
        await session.flush()
        for key, player in zip(new_keys, new_players):
            known[key] = player.player_id
            session.add(
                EntityMap(
                    provider=PROVIDER,
                    provider_entity_type="player",
                    provider_id=key,
                    canonical_id=player.player_id,
                )
            )
        await session.flush()

    rows = []
    for key, (player, team_id) in listed.items():
        rows.append({
            "match_id": match_id,
            "player_id": known[key],
            "team_id": team_id,
            "is_starter": str(player.get("player_isSubst", "")).strip() != "True",
            "position": str(player.get("player_position", "")).strip()[:30] or None,
            "rating": _rating(player.get("player_rating", "")),
            **{column: _count(player.get(field, "")) for column, field in PLAYER_COUNT_STATS.items()},
        })
    statement = insert(PlayerMatchStats).values(rows)
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=["match_id", "player_id"],
            set_={name: statement.excluded[name] for name in rows[0] if name not in ("match_id", "player_id")},
        )
    )
    return len(rows)


async def upsert_match(session: AsyncSession, ev: ProviderEvent) -> bool:
    """Insert or update on match. Returns True if it was new"""
    competition_id = await resolve(
       session, "competition", ev.league_id, 
        lambda: Competition(
           name=competition_name(ev.league_name),
           country=ev.country_name
        ),
        "competition_id",
    )
    home_id = await resolve(
        session, "team", ev.match_hometeam_id,
        lambda: Team(
            name=ev.match_hometeam_name,
        ),
        "team_id",
    )
    away_id = await resolve(
        session, "team", ev.match_awayteam_id,
        lambda: Team(name=ev.match_awayteam_name),
        "team_id",
    )

    referee_id = None

    referee = " ".join(ev.match_referee.split())
    if referee:
        referee_id = await resolve(
            session, "referee", referee_key(referee),
            lambda: Referee(name=referee),
            "referee_id",
        )
    
    venue_id = None
    stadium = ev.match_stadium.strip()
    if stadium:
        venue_id = await resolve(
            session, "venue", stadium, lambda: Venue(name=stadium),
            "venue_id",
        )

    fields = {
        "season_id": await resolve_season(session, competition_id,ev.match_date),
        "kickoff_utc": ev.kickoff_utc,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "referee_id": referee_id,
        "venue_id": venue_id,
        "status": ev.match_status,
        "home_goals": ev.home_goals,
        "away_goals": ev.away_goals,        
    }

    mapping = await session.get(EntityMap, (PROVIDER, "match", ev.match_id))

    if mapping is not None:
        match = await session.get(Match, mapping.canonical_id)
        created = False
    else:
        # The provider sometimes lists one fixture under two match ids, so look
        # for an existing row by its natural key before inserting (R-03).
        stmt = select(Match).where(
            Match.season_id == fields["season_id"],
            Match.home_team_id == fields["home_team_id"],
            Match.away_team_id == fields["away_team_id"],
            Match.kickoff_utc == fields["kickoff_utc"],
        )
        match = (await session.execute(stmt)).scalar_one_or_none()
        created = match is None
        if match is None:
            match = Match(**fields)
            session.add(match)
            await session.flush()
        session.add(
            EntityMap(
                provider=PROVIDER,
                provider_entity_type="match",
                provider_id=ev.match_id,
                canonical_id=match.match_id,
            )
        )
        await session.flush()

    for name, value in fields.items():
        setattr(match, name, value)

    await upsert_team_stats(session, ev, match.match_id, home_id, away_id)
    if ev.is_finished:
        await upsert_player_stats(session, ev, match.match_id, home_id, away_id)
    return created



    match = await session.get(Match, mapping.canonical_id)
    for name , value in fields.items():
        setattr(match, name, value)
    return False


async def ingest_league(start: date, end: date, league_id: str) -> dict:
    """Fetch one league's matches for a date range and upsert them.

    Returns what happened, so the caller can tell a quiet week ("no matches in
    this window") from a dead feed ("the provider refuses every league"). A run
    that silently ingests nothing is worse than one that fails: the forecasts
    still appear, built on stale form.
    """

    params= {
        "league_id": league_id,
        "from": start,
        "to": end,
        "withPlayerStats": 1,
    }
    endpoint = f"get_events?league_id={league_id}&from={start}&to={end}&withPlayerStats=1"

    try:
        payload = await ApiFootballClient().get("get_events", **params)
    except ProviderError as exc:
        print(f"league {league_id}: skipped -- {exc}")
        return {"league_id": league_id, "events": 0, "created": 0, "updated": 0,
                "rejected": 0, "error": str(exc)}

    digest, object_key = save_raw(payload)

    created = updated = rejected = 0

    async with SessionLocal() as session:
        session.add(
            RawPayloadRef(
                provider=PROVIDER,
                endpoint=endpoint,
                content_hash=digest,
                object_key=object_key
            )
        )

        for item in payload:
            try:
                ev = ProviderEvent.model_validate(item)
                _=ev.kickoff_utc
            except (ValidationError, ValueError) as exc:
                session.add(
                    RejectedPayload(
                        provider=PROVIDER,
                        endpoint=endpoint,
                       payload=item,
                       validation_error=str(exc)
                    )
                )
                rejected +=1
                continue
                
            if await upsert_match(session, ev):
                created += 1
            else:
                updated += 1

        await session.commit()

    league_name = payload[0].get("league_name", "?") if payload else "?"
    print(
        f"league {league_id} ({league_name}): {len(payload)} events -- "
        f"{created} created, {updated} updated, {rejected} rejected"
    )
    return {"league_id": league_id, "events": len(payload), "created": created,
            "updated": updated, "rejected": rejected, "error": None}


async def run(start: date, end: date, league_ids: list[str]) -> None:
    try:
        for league_id in league_ids:
            try:
                await ingest_league(start, end, league_id)
            except Exception as exc:  # noqa: BLE001 -- isolate one league's failure
                print(f"league {league_id}: FAILED -- {type(exc).__name__}: {exc}")
    finally:
        await dispose_engine()



def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest apifootball fixtures into core.")
    parser.add_argument("--from", dest="start", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--league",
        dest="leagues",
        action="append",
        help="apifootball league id; repeat for several. Default: settings.league_ids",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args.start, args.end, args.leagues or settings.league_ids))


if __name__ == "__main__":
    main()
    

    


