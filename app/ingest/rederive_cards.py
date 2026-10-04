"""Re-derive stored card counts from the raw payload archive (FR-DATA-06, R-03).

The original ingestion treated a missing card statistic as zero cards. Checking
36,184 archived matches showed that is wrong for several hundred of them:
bookings were recorded, the statistic just wasn't sent. This script re-applies
the corrected rule (ProviderEvent.card_count) to every archived match and
updates core.team_match_stats wherever the answer differs. It uses no API
requests.

It also migrates referee identities to case-insensitive keys, merging referees
that were stored twice ('Carlos Del Cerro' / 'Carlos del Cerro').

    python -m app.ingest.rederive_cards --dry-run
    python -m app.ingest.rederive_cards
"""

import argparse
import asyncio
import collections
import json
import sys

from pydantic import ValidationError
from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal, dispose_engine
from app.ingest.client import PROVIDER
from app.ingest.fixtures import CARD_STATS, RAW_DIR, referee_key
from app.ingest.schemas import ProviderEvent
from app.models import EntityMap, Match, Referee, TeamMatchStats


def best_archived_events() -> dict[str, ProviderEvent]:
    """The most complete finished record of each match across all raw files.

    Some archived payloads are degraded (a whole-season request returned stubs),
    so when a match appears more than once, keep the version with the most
    statistics, then the most bookings.
    """
    best: dict[str, ProviderEvent] = {}
    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            try:
                event = ProviderEvent.model_validate(item)
            except ValidationError:
                continue
            if not event.is_finished:
                continue
            current = best.get(event.match_id)
            rank = (len(event.statistics), len(event.cards))
            if current is None or rank > (len(current.statistics), len(current.cards)):
                best[event.match_id] = event
    return best


async def merge_referees(session: AsyncSession, dry_run: bool) -> tuple[int, int]:
    """Re-key referee mappings case-insensitively. Returns (groups, merged)."""
    mappings = (
        await session.execute(
            select(EntityMap.provider_id, EntityMap.canonical_id).where(
                EntityMap.provider == PROVIDER,
                EntityMap.provider_entity_type == "referee",
            )
        )
    ).all()

    groups: dict[str, set[int]] = collections.defaultdict(set)
    for provider_id, canonical_id in mappings:
        groups[referee_key(provider_id)].add(canonical_id)

    duplicates = {key: ids for key, ids in groups.items() if len(ids) > 1}
    if dry_run:
        return len(groups), sum(len(ids) - 1 for ids in duplicates.values())

    for ids in duplicates.values():
        keep, *others = sorted(ids)
        await session.execute(
            update(Match).where(Match.referee_id.in_(others)).values(referee_id=keep)
        )
        await session.execute(delete(Referee).where(Referee.referee_id.in_(others)))

    await session.execute(
        delete(EntityMap).where(
            EntityMap.provider == PROVIDER, EntityMap.provider_entity_type == "referee"
        )
    )
    await session.execute(
        insert(EntityMap),
        [
            {
                "provider": PROVIDER,
                "provider_entity_type": "referee",
                "provider_id": key,
                "canonical_id": min(ids),
            }
            for key, ids in groups.items()
        ],
    )
    return len(groups), sum(len(ids) - 1 for ids in duplicates.values())


async def rederive(session: AsyncSession, dry_run: bool) -> collections.Counter:
    events = best_archived_events()

    match_of = {
        provider_id: (match_id, home_id, away_id)
        for provider_id, match_id, home_id, away_id in await session.execute(
            select(
                EntityMap.provider_id, Match.match_id, Match.home_team_id, Match.away_team_id
            )
            .join(Match, Match.match_id == EntityMap.canonical_id)
            .where(EntityMap.provider == PROVIDER, EntityMap.provider_entity_type == "match")
        )
    }
    stored = {
        (row.match_id, row.team_id): (row.yellow, row.red)
        for row in await session.execute(
            select(TeamMatchStats.match_id, TeamMatchStats.team_id,
                   TeamMatchStats.yellow, TeamMatchStats.red)
        )
    }

    # The provider sometimes lists one fixture under two match ids, both mapped to
    # the same stored match -- one complete, one an empty stub. Choose the most
    # complete record per STORED match, or the stub would erase good counts.
    per_match: dict[int, tuple[ProviderEvent, int, int]] = {}
    for provider_id, event in events.items():
        if provider_id not in match_of:
            continue
        match_id, home_id, away_id = match_of[provider_id]
        current = per_match.get(match_id)
        rank = (len(event.statistics), len(event.cards))
        if current is None or rank > (len(current[0].statistics), len(current[0].cards)):
            per_match[match_id] = (event, home_id, away_id)

    tally: collections.Counter = collections.Counter()
    updates: list[dict] = []
    for match_id, (event, home_id, away_id) in per_match.items():
        derived = {
            column: event.card_count(stat_name, label)
            for column, (stat_name, label) in CARD_STATS.items()
        }
        for side, team_id in ((0, home_id), (1, away_id)):
            key = (match_id, team_id)
            if key not in stored:
                continue
            new = tuple(
                None if derived[column] is None else derived[column][side]
                for column in ("yellow", "red")
            )
            old = stored[key]
            if new == old:
                tally["unchanged"] += 1
                continue
            if new[0] is None:
                tally["now unknown (no source)"] += 1
            elif event.stat("Yellow Cards") is None and event.cards:
                tally["filled from bookings"] += 1
            else:
                tally["other correction"] += 1
            updates.append({"match_id": match_id, "team_id": team_id,
                            "yellow": new[0], "red": new[1]})

    tally["archived matches"] = len(events)
    if updates and not dry_run:
        await session.execute(update(TeamMatchStats), updates)
    return tally


async def main(dry_run: bool) -> None:
    try:
        async with SessionLocal() as session:
            groups, merged = await merge_referees(session, dry_run)
            tally = await rederive(session, dry_run)
            if not dry_run:
                await session.commit()
    finally:
        await dispose_engine()

    mode = "DRY RUN -- nothing written" if dry_run else "applied"
    print(f"{mode}")
    print(f"referees: {groups} distinct after case-insensitive matching, {merged} duplicates merged")
    for label, count in tally.most_common():
        print(f"  {count:6}  {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-derive card counts from the raw archive.")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main(parser.parse_args().dry_run))
