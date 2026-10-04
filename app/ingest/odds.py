"""Poll bookmaker odds and store every price change (FR-DATA-02).

Each poll fetches one day of odds per request, keeps only matches the Platform
holds, ignores any match that has already kicked off, and writes a row only when
a price differs from the last one stored for that exact selection.

Run once:     python -m app.ingest.odds --days 2
Run forever:  python -m app.ingest.odds --loop
"""

import argparse
import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import SessionLocal, dispose_engine
from app.ingest.client import PROVIDER, ApiFootballClient, ProviderError
from app.models import Bookmaker, EntityMap, Match, OddsSnapshot

logger = logging.getLogger(__name__)

SHARP_BOOKS = {"Pncl"}  # Pinnacle
TOTAL_KEY = re.compile(r"^o\+(\d+(?:\.\d+)?)$")

LATEST_PRICES = text("""
    select distinct on (bookmaker_id, match_id, market, line, selection)
           bookmaker_id, match_id, market, line, selection, decimal_price
    from core.odds_snapshots
    where match_id = any(:match_ids) and entity_type = 'match'
    order by bookmaker_id, match_id, market, line, selection, captured_at desc
""")


@dataclass
class PollResult:
    matches_priced: int = 0
    prices_seen: int = 0
    prices_stored: int = 0
    skipped_started: int = 0


def _price(raw: str | None) -> float | None:
    try:
        value = float(raw) if raw else None
    except ValueError:
        return None
    return value if value is not None and value > 1.0 else None


def parse_markets(row: dict) -> list[tuple[str, float | None, str, float]]:
    """(market, line, selection, price) for every COMPLETE market in one row.

    A market is kept only when every outcome has a price -- half a market cannot
    be de-vigged, so storing it would only mislead.
    """
    out: list[tuple[str, float | None, str, float]] = []

    home = _price(row.get("odd_1"))
    draw = _price(row.get("odd_x"))
    away = _price(row.get("odd_2"))
    if home and draw and away:
        out += [
            ("match_result", None, "home", home),
            ("match_result", None, "draw", draw),
            ("match_result", None, "away", away),
        ]

    yes, no = _price(row.get("bts_yes")), _price(row.get("bts_no"))
    if yes and no:
        out += [("btts", None, "yes", yes), ("btts", None, "no", no)]

    for key in row:
        found = TOTAL_KEY.match(key)
        if not found:
            continue
        over = _price(row.get(key))
        under = _price(row.get(f"u+{found.group(1)}"))
        if over and under:
            line = float(found.group(1))
            out += [
                ("total_goals", line, "over", over),
                ("total_goals", line, "under", under),
            ]
    return out


async def _bookmaker_ids(session: AsyncSession, names: set[str]) -> dict[str, int]:
    """Look up bookmakers by name, creating any not seen before."""
    rows = await session.execute(select(Bookmaker).where(Bookmaker.name.in_(names)))
    ids = {book.name: book.bookmaker_id for book in rows.scalars()}
    for name in sorted(names - ids.keys()):
        book = Bookmaker(name=name, is_sharp=name in SHARP_BOOKS)
        session.add(book)
        await session.flush()
        ids[name] = book.bookmaker_id
    return ids


async def poll(session: AsyncSession, client: ApiFootballClient, days: int) -> PollResult:
    """One pass over today plus `days` more. Flushes but does not commit."""
    captured_at = datetime.now(UTC)
    feed_tz = ZoneInfo(settings.api_football_timezone)
    first_day = captured_at.astimezone(feed_tz).date()

    rows: list[dict] = []
    for offset in range(days + 1):
        day = (first_day + timedelta(days=offset)).isoformat()
        try:
            rows += await client.get("get_odds", **{"from": day, "to": day})
        except ProviderError as exc:
            logger.warning("odds for %s unavailable: %s", day, exc)

    provider_ids = {row["match_id"] for row in rows}
    known = {
        found.provider_id: (found.match_id, found.kickoff_utc)
        for found in await session.execute(
            select(EntityMap.provider_id, Match.match_id, Match.kickoff_utc)
            .join(Match, Match.match_id == EntityMap.canonical_id)
            .where(
                EntityMap.provider == PROVIDER,
                EntityMap.provider_entity_type == "match",
                EntityMap.provider_id.in_(provider_ids),
            )
        )
    }

    result = PollResult()
    books = await _bookmaker_ids(
        session, {row["odd_bookmakers"] for row in rows if row["match_id"] in known}
    )

    match_ids = [match_id for match_id, _ in known.values()]
    latest: dict[tuple, float] = {}
    if match_ids:
        for prior in await session.execute(LATEST_PRICES, {"match_ids": match_ids}):
            line = None if prior.line is None else round(float(prior.line), 2)
            key = (prior.bookmaker_id, prior.match_id, prior.market, line, prior.selection)
            latest[key] = round(float(prior.decimal_price), 3)

    new_rows: list[dict] = []
    priced: set[int] = set()
    for row in rows:
        if row["match_id"] not in known:
            continue
        match_id, kickoff = known[row["match_id"]]
        if kickoff <= captured_at:
            result.skipped_started += 1
            continue

        provider_time = None
        if row.get("odd_date"):
            provider_time = (
                datetime.fromisoformat(row["odd_date"])
                .replace(tzinfo=feed_tz)
                .astimezone(UTC)
            )
        bookmaker_id = books[row["odd_bookmakers"]]

        for market, line, selection, price in parse_markets(row):
            result.prices_seen += 1
            key = (
                bookmaker_id, match_id, market,
                None if line is None else round(line, 2), selection,
            )
            if latest.get(key) == round(price, 3):
                continue
            latest[key] = round(price, 3)
            priced.add(match_id)
            new_rows.append({
                "bookmaker_id": bookmaker_id,
                "match_id": match_id,
                "market": market,
                "entity_type": "match",
                "entity_id": None,
                "line": line,
                "selection": selection,
                "decimal_price": price,
                "captured_at": captured_at,
                "provider_updated_at": provider_time,
            })

    if new_rows:
        await session.execute(insert(OddsSnapshot), new_rows)
    result.prices_stored = len(new_rows)
    result.matches_priced = len(priced)
    return result


async def _kickoff_within(hours: float) -> bool:
    now = datetime.now(UTC)
    async with SessionLocal() as session:
        found = await session.scalar(
            select(Match.match_id)
            .where(Match.kickoff_utc > now, Match.kickoff_utc <= now + timedelta(hours=hours))
            .limit(1)
        )
    return found is not None


async def run_once(days: int) -> None:
    async with SessionLocal() as session:
        result = await poll(session, ApiFootballClient(), days)
        await session.commit()
    print(
        f"{datetime.now(UTC):%Y-%m-%d %H:%M:%S}Z  stored {result.prices_stored} "
        f"of {result.prices_seen} prices across {result.matches_priced} matches"
    )


async def run_forever() -> None:
    """FR-DATA-02 cadence: every 5 minutes near kickoff, hourly otherwise."""
    last_wide: datetime | None = None
    while True:
        now = datetime.now(UTC)
        wide = last_wide is None or now - last_wide >= timedelta(
            seconds=settings.odds_far_interval_seconds
        )
        try:
            await run_once(settings.odds_days_ahead if wide else 1)
            if wide:
                last_wide = now
        except Exception:  # noqa: BLE001 -- one failed poll must not stop the worker
            logger.exception("poll failed; retrying next cycle")

        near = await _kickoff_within(settings.odds_near_kickoff_hours)
        await asyncio.sleep(
            settings.odds_near_interval_seconds if near else settings.odds_far_interval_seconds
        )


async def _main(args: argparse.Namespace) -> None:
    try:
        if args.loop:
            await run_forever()
        else:
            await run_once(args.days)
    finally:
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll bookmaker odds into core.odds_snapshots.")
    parser.add_argument("--days", type=int, default=1, help="days ahead for a single run")
    parser.add_argument("--loop", action="store_true", help="run forever on the FR-DATA-02 cadence")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
