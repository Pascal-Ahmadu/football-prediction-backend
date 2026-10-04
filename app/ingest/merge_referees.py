"""Merge referees stored under more than one name format.

apifootball writes the same referee several ways over time: 'Michael Oliver',
'Michael Oliver, England', 'M. Oliver', and from 2026 with accents dropped
('F. Badstübner' -> 'F. Badstubner'). Each form became a separate referee, so
most referee histories were split and every profile restarted at each format
change. This script joins them.

Two passes, from safest to least safe:
  1. Names identical apart from case, accents and a trailing ', Country' are
     always the same referee.
  2. An abbreviated form ('M. Oliver') joins a full form ('Michael Oliver') only
     if they share initial, surname and country, and no second full forename
     exists for that initial -- 'J. Smith' stays apart when both a 'Josh' and a
     'John' are present.

A backup of every affected row is written first, so the merge can be undone.

    python -m app.ingest.merge_referees --dry-run
    python -m app.ingest.merge_referees
"""

import argparse
import asyncio
import collections
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal, dispose_engine
from app.ingest.client import PROVIDER
from app.ingest.fixtures import fold_accents, referee_base_name, referee_key
from app.models import EntityMap, Match, Referee

BACKUP_DIR = Path("data") / "backups"


class Components:
    """Union-find: which referee ids have been declared the same person."""

    def __init__(self, ids) -> None:
        self.parent = {i: i for i in ids}

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = collections.defaultdict(list)
        for i in self.parent:
            out[self.find(i)].append(i)
        return out


def _words(base: str) -> list[str]:
    return [w for w in re.split(r"[\s.]+", fold_accents(base).casefold()) if w]


def _is_abbreviated(base: str) -> bool:
    return bool(re.match(r"^\w\.\s", base))


async def plan(session: AsyncSession):
    referees = (
        await session.execute(text("""
            select r.referee_id, r.name, count(m.match_id) as matches,
                   mode() within group (order by c.country) as country
            from core.referees r
            left join core.matches m on m.referee_id = r.referee_id
            left join core.seasons se on se.season_id = m.season_id
            left join core.competitions c on c.competition_id = se.competition_id
            group by 1, 2
        """))
    ).all()
    info = {r.referee_id: r for r in referees}
    components = Components(info)

    # Pass 1: identical apart from case, accents and a country suffix.
    by_key: dict[str, list[int]] = collections.defaultdict(list)
    for r in referees:
        by_key[referee_key(r.name)].append(r.referee_id)
    for ids in by_key.values():
        for other in ids[1:]:
            components.join(other, ids[0])

    # Pass 2: abbreviated and full forms, same initial + surname + country.
    by_initial: dict[tuple, list[int]] = collections.defaultdict(list)
    for root, members in components.groups().items():
        lead = max(members, key=lambda i: info[i].matches)
        words = _words(referee_base_name(info[lead].name))
        if len(words) < 2:
            continue
        by_initial[(words[0][0], " ".join(words[1:]), info[lead].country)].append(root)

    ambiguous = 0
    for roots in by_initial.values():
        if len(roots) < 2:
            continue
        forenames = {
            _words(referee_base_name(info[i].name))[0]
            for root in roots
            for i in components.groups()[components.find(root)]
            if not _is_abbreviated(referee_base_name(info[i].name))
        }
        if len(forenames) > 1:
            ambiguous += 1
            continue
        for other in roots[1:]:
            components.join(other, roots[0])

    merges = []
    for members in components.groups().values():
        if len(members) < 2:
            continue
        survivor = max(members, key=lambda i: (info[i].matches, -i))
        full_forms = [i for i in members if not _is_abbreviated(referee_base_name(info[i].name))]
        name_source = max(full_forms or members, key=lambda i: info[i].matches)
        merges.append({
            "survivor": survivor,
            "name": referee_base_name(info[name_source].name),
            "absorbed": sorted(i for i in members if i != survivor),
            "names": sorted({info[i].name for i in members}),
            "matches": sum(info[i].matches for i in members),
        })
    return info, merges, ambiguous


async def backup(session: AsyncSession, path: Path) -> None:
    referees = [
        {"referee_id": r.referee_id, "name": r.name}
        for r in (await session.execute(select(Referee))).scalars()
    ]
    mappings = [
        {"provider_id": p, "canonical_id": c}
        for p, c in await session.execute(
            select(EntityMap.provider_id, EntityMap.canonical_id).where(
                EntityMap.provider == PROVIDER, EntityMap.provider_entity_type == "referee"
            )
        )
    ]
    assignments = [
        {"match_id": m, "referee_id": r}
        for m, r in await session.execute(
            select(Match.match_id, Match.referee_id).where(Match.referee_id.is_not(None))
        )
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"referees": referees, "entity_map": mappings, "match_referees": assignments},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


async def apply(session: AsyncSession, info, merges) -> int:
    survivor_of = {i: m["survivor"] for m in merges for i in m["absorbed"]}

    for m in merges:
        await session.execute(
            update(Match).where(Match.referee_id.in_(m["absorbed"])).values(referee_id=m["survivor"])
        )
        await session.execute(
            update(Referee).where(Referee.referee_id == m["survivor"]).values(name=m["name"])
        )
        await session.execute(delete(Referee).where(Referee.referee_id.in_(m["absorbed"])))

    old = (
        await session.execute(
            select(EntityMap.provider_id, EntityMap.canonical_id).where(
                EntityMap.provider == PROVIDER, EntityMap.provider_entity_type == "referee"
            )
        )
    ).all()
    keys: dict[str, int] = {}
    for provider_id, canonical_id in old:
        keys[referee_key(provider_id)] = survivor_of.get(canonical_id, canonical_id)
    for referee_id, r in info.items():  # every written form now points at its survivor
        keys[referee_key(r.name)] = survivor_of.get(referee_id, referee_id)

    await session.execute(
        delete(EntityMap).where(
            EntityMap.provider == PROVIDER, EntityMap.provider_entity_type == "referee"
        )
    )
    await session.execute(
        insert(EntityMap),
        [
            {"provider": PROVIDER, "provider_entity_type": "referee",
             "provider_id": key, "canonical_id": canonical}
            for key, canonical in keys.items()
        ],
    )
    return len(keys)


async def main(dry_run: bool) -> None:
    try:
        async with SessionLocal() as session:
            info, merges, ambiguous = await plan(session)
            absorbed = sum(len(m["absorbed"]) for m in merges)
            print(f"referees: {len(info)} -> {len(info) - absorbed}")
            print(f"merge groups: {len(merges)}, identities absorbed: {absorbed}, "
                  f"matches covered: {sum(m['matches'] for m in merges)}")
            print(f"groups left apart as ambiguous: {ambiguous}")
            for m in sorted(merges, key=lambda m: -m["matches"])[:8]:
                print(f"  {m['name']:26} <- {m['names']}  ({m['matches']} matches)")

            if dry_run:
                print("\nDRY RUN -- nothing written")
                return

            path = BACKUP_DIR / f"referees-before-merge-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
            await backup(session, path)
            print(f"\nbackup written: {path}")
            mappings = await apply(session, info, merges)
            await session.commit()
            print(f"applied; {mappings} referee name keys now map to {len(info) - absorbed} referees")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge referees stored under several names.")
    parser.add_argument("--dry-run", action="store_true")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main(parser.parse_args().dry_run))
