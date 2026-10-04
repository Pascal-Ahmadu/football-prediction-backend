"""Market prices from The Odds API, for the markets we actually predict (FR-DATA-02).

Being accurate is not the same as being profitable: a forecast only makes money
if the price is better than the forecast deserves. This fetches real corners and
cards prices, stores them, and fills in each prediction's best price and edge.

The free plan is 500 credits a month and one credit buys ONE market for ONE
fixture in ONE region, so fixtures are sampled rather than swept: listing events
is free, and credits are spent only on the fixtures our own picks care about,
most confident first.

    python -m app.ingest.market_odds --dry-run   # match fixtures, spend nothing
    python -m app.ingest.market_odds             # price settings.odds_sample_size fixtures
    python -m app.ingest.market_odds --report    # compare stored prices with the model
"""

import argparse
import asyncio
import logging
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher

import httpx
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import SessionLocal, dispose_engine
from app.models import Bookmaker, Competition, EntityMap, Match, OddsSnapshot, Prediction, Season, Team
from app.pricing.devig import devig

logger = logging.getLogger(__name__)

PROVIDER = "theoddsapi"

# Our apifootball league id -> The Odds API sport key.
SPORT_KEYS = {
    "152": "soccer_epl",
    "302": "soccer_spain_la_liga",
    "207": "soccer_italy_serie_a",
    "175": "soccer_germany_bundesliga",
    "168": "soccer_france_ligue_one",
    "244": "soccer_netherlands_eredivisie",
    "266": "soccer_portugal_primeira_liga",
    "63": "soccer_belgium_first_div",
    "322": "soccer_turkey_super_league",
    "279": "soccer_spl",
    "178": "soccer_greece_super_league",
    "56": "soccer_austria_bundesliga",
    "153": "soccer_efl_champ",
    "301": "soccer_spain_segunda_division",
    "206": "soccer_italy_serie_b",
    "171": "soccer_germany_bundesliga2",
    "164": "soccer_france_ligue_two",
}

# Their market key -> (our market, the line we validated)
MARKETS = {
    "alternate_totals_corners": "total_corners",
    "alternate_totals_cards": "total_cards",
}

NAME_NOISE = {"fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "cd", "ud", "sv", "vfl",
              "vfb", "tsg", "fsv", "bsc", "if", "ik", "sk", "club", "de", "the"}

# Clubs the two providers spell differently enough that letters alone cannot
# connect them. English exonyms, mostly.
ALIASES = {
    "koln": "cologne", "munchen": "munich", "wien": "vienna",
    "sevilla": "seville", "milano": "milan", "napoli": "naples",
    "torino": "turin", "roma": "rome", "genoa": "genoa",
    "sporting cp": "sporting lisbon", "psv": "psv eindhoven",
}

# A fixture must clear these before a credit is spent on it: a wrong pairing
# would price the wrong match, which is worse than pricing none.
MIN_SIDE = 0.55        # weaker of the two team names
MIN_AVERAGE = 0.75     # both names together
MIN_LEAD = 0.05        # over the next best candidate, or it is too close to call


def _normalise(name: str) -> str:
    """'Brighton and Hove Albion' and 'Brighton' must look alike enough to match."""
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", name.casefold()) if not unicodedata.combining(c)
    )
    for source, target in ALIASES.items():
        folded = folded.replace(source, target)
    words = [w for w in "".join(c if c.isalnum() else " " for c in folded).split()
             if w not in NAME_NOISE]
    return " ".join(words)


def _word_match(a: str, b: str) -> float:
    """'ath' and 'athletic' are the same word abbreviated; 'utd' and 'city' are not."""
    if a == b or (len(a) >= 3 and (b.startswith(a) or a.startswith(b))):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def similarity(ours: str, theirs: str) -> float:
    """How alike two club names are, judged on what distinguishes them.

    'Manchester Utd' and 'Manchester City' share the word that carries most of the
    letters, so comparing the whole strings calls them 83% alike. What separates
    the clubs is 'utd' against 'city', and that is what decides it here.
    """
    a, b = _normalise(ours), _normalise(theirs)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    a_words, b_words = a.split(), b.split()
    shared = set(a_words) & set(b_words)
    rest_a = [w for w in a_words if w not in shared]
    rest_b = [w for w in b_words if w not in shared]
    if shared and (not rest_a or not rest_b):
        return 1.0  # one name is the other plus extra words: Brighton (and Hove Albion)
    if not rest_a or not rest_b:
        return SequenceMatcher(None, a, b).ratio()

    best = max(_word_match(x, y) for x in rest_a for y in rest_b)
    return best if not shared else max(best, min(best + 0.1, 0.99))


def pair_score(fixture_home: str, fixture_away: str, event: dict) -> float:
    """How well one of their events matches one of our fixtures.

    Both names must look right on their own: 'Manchester Utd' against
    'Manchester City' scores well on the average alone, and pricing the wrong
    derby would be worse than pricing nothing.
    """
    home = similarity(fixture_home, event["home_team"])
    away = similarity(fixture_away, event["away_team"])
    if min(home, away) < MIN_SIDE:
        return 0.0
    return (home + away) / 2


@dataclass
class Fixture:
    match_id: int
    kickoff: datetime
    home: str
    away: str
    sport_key: str


class OddsApiClient:
    """Thin client. Every response reports the credits left; nothing else spends."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.odds_api_key
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY is not set (put it in .env, never in the repo)")
        self.remaining: int | None = None
        self.used_here = 0

    async def _get(self, path: str, **params) -> list | dict:
        url = f"{settings.odds_api_base_url}{path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params={"apiKey": self.api_key, **params})
        if response.status_code != 200:
            raise RuntimeError(f"odds api {path}: {response.status_code} {response.text[:200]}")
        self.remaining = int(response.headers.get("x-requests-remaining", -1))
        self.used_here += int(response.headers.get("x-requests-last", 0))
        return response.json()

    async def events(self, sport_key: str) -> list[dict]:
        """Upcoming fixtures for one competition. Free: costs no credits."""
        return await self._get(f"sports/{sport_key}/events")

    async def event_odds(self, sport_key: str, event_id: str, markets: list[str]) -> dict:
        """Prices for one fixture. Costs one credit per market per region."""
        return await self._get(
            f"sports/{sport_key}/events/{event_id}/odds",
            regions=settings.odds_api_regions,
            markets=",".join(markets),
            oddsFormat="decimal",
        )


async def _candidates(session: AsyncSession, days: int, limit: int) -> list[Fixture]:
    """Our upcoming fixtures, most confidently forecast first -- those are the ones
    worth a credit, because they are the ones we would act on."""
    now = datetime.now(UTC)
    rows = (
        await session.execute(
            text("""
                select m.match_id, m.kickoff_utc, th.name as home, ta.name as away,
                       e.provider_id as league_id,
                       max(abs(p.prob_over - 0.5)) as confidence
                from core.matches m
                join core.teams th on th.team_id = m.home_team_id
                join core.teams ta on ta.team_id = m.away_team_id
                join core.seasons se on se.season_id = m.season_id
                join core.entity_map e on e.canonical_id = se.competition_id
                 and e.provider = 'apifootball' and e.provider_entity_type = 'competition'
                join models.predictions p on p.match_id = m.match_id
                 and p.entity_type = 'match' and p.market in ('total_corners', 'total_cards')
                 and p.line in (9.5, 4.5)
                where m.kickoff_utc between :now and :until
                group by 1, 2, 3, 4, 5
                order by confidence desc
            """),
            {"now": now, "until": now + timedelta(days=days)},
        )
    ).all()
    fixtures = []
    for row in rows:
        sport_key = SPORT_KEYS.get(row.league_id)
        if sport_key is not None:
            fixtures.append(Fixture(row.match_id, row.kickoff_utc, row.home, row.away, sport_key))
        if len(fixtures) >= limit:
            break
    return fixtures


async def _match_event(session: AsyncSession, fixture: Fixture, events: list[dict]) -> str | None:
    """Their event id for one of our fixtures, by kickoff and team names.

    Once matched the pairing is remembered, so a fixture can be priced again later
    -- prices move, and the price near kickoff is the one that matters -- without
    matching names a second time.
    """
    known = await session.scalar(
        select(EntityMap.provider_id).where(
            EntityMap.provider == PROVIDER,
            EntityMap.provider_entity_type == "match",
            EntityMap.canonical_id == fixture.match_id,
        )
    )
    if known is not None:
        return known

    scored = []
    for event in events:
        kickoff = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if abs((kickoff - fixture.kickoff).total_seconds()) > 6 * 3600:
            continue
        score = pair_score(fixture.home, fixture.away, event)
        if score > 0:
            scored.append((score, event["id"]))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_score, best_id = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < MIN_AVERAGE or best_score - runner_up < MIN_LEAD:
        return None
    return best_id


async def _bookmaker_id(session: AsyncSession, name: str, cache: dict[str, int]) -> int:
    if name in cache:
        return cache[name]
    bookmaker = (
        await session.execute(select(Bookmaker).where(Bookmaker.name == name[:60]))
    ).scalar_one_or_none()
    if bookmaker is None:
        bookmaker = Bookmaker(name=name[:60], is_sharp=name.lower() in {"pinnacle", "betfair"})
        session.add(bookmaker)
        await session.flush()
    cache[name] = bookmaker.bookmaker_id
    return bookmaker.bookmaker_id


async def fetch(session: AsyncSession, dry_run: bool = False) -> dict:
    """Price the most confidently forecast fixtures. Returns a summary."""
    client = None if dry_run else OddsApiClient()
    fixtures = await _candidates(session, settings.odds_sample_days, settings.odds_sample_size)
    by_sport: dict[str, list[Fixture]] = {}
    for fixture in fixtures:
        by_sport.setdefault(fixture.sport_key, []).append(fixture)

    lister = client or OddsApiClient()  # listing events is free, so a dry run may do it
    matched, priced, rows_written, unmatched = 0, 0, 0, []
    cache: dict[str, int] = {}
    now = datetime.now(UTC)

    for sport_key, group in by_sport.items():
        events = await lister.events(sport_key)
        for fixture in group:
            event_id = await _match_event(session, fixture, events)
            if event_id is None:
                unmatched.append(f"{fixture.home} v {fixture.away}")
                continue
            matched += 1
            if dry_run:
                continue

            payload = await client.event_odds(sport_key, event_id, list(MARKETS))
            priced += 1
            for bookmaker in payload.get("bookmakers", []):
                bookmaker_id = await _bookmaker_id(session, bookmaker["title"], cache)
                for market in bookmaker["markets"]:
                    our_market = MARKETS.get(market["key"])
                    if our_market is None:
                        continue
                    # Additional markets carry their own timestamp; some bookmakers
                    # give one per market and none at the bookmaker level.
                    stamp = market.get("last_update") or bookmaker.get("last_update")
                    for outcome in market["outcomes"]:
                        session.add(
                            OddsSnapshot(
                                bookmaker_id=bookmaker_id,
                                match_id=fixture.match_id,
                                market=our_market,
                                entity_type="match",
                                entity_id=None,
                                line=outcome["point"],
                                selection=outcome["name"].lower(),
                                decimal_price=outcome["price"],
                                captured_at=now,
                                provider_updated_at=(
                                    datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                                    if stamp else None
                                ),
                            )
                        )
                        rows_written += 1
            if not await session.scalar(
                select(EntityMap.provider_id).where(
                    EntityMap.provider == PROVIDER,
                    EntityMap.provider_entity_type == "match",
                    EntityMap.provider_id == event_id,
                )
            ):
                session.add(
                    EntityMap(
                        provider=PROVIDER,
                        provider_entity_type="match",
                        provider_id=event_id,
                        canonical_id=fixture.match_id,
                    )
                )
            await session.flush()

    await session.commit()
    return {
        "candidates": len(fixtures), "matched": matched, "priced": priced,
        "prices": rows_written, "unmatched": unmatched,
        "credits_used": 0 if dry_run else client.used_here,
        "credits_left": lister.remaining,
    }


async def apply_edges(session: AsyncSession) -> int:
    """Fill in each current prediction's best price, implied probability and edge."""
    result = await session.execute(text("""
        with best as (
            select o.match_id, o.market, o.line, o.selection,
                   max(o.decimal_price) as price,
                   (array_agg(o.bookmaker_id order by o.decimal_price desc))[1] as bookmaker_id
            from core.odds_snapshots o
            join core.matches m on m.match_id = o.match_id
            where m.kickoff_utc > now() and o.entity_type = 'match'
            group by 1, 2, 3, 4
        ),
        two_sided as (
            select match_id, market, line,
                   max(price) filter (where selection = 'over') as over_price,
                   max(price) filter (where selection = 'under') as under_price,
                   max(bookmaker_id) filter (where selection = 'over') as over_book,
                   max(bookmaker_id) filter (where selection = 'under') as under_book
            from best group by 1, 2, 3
        ),
        latest as (
            select p.prediction_id, p.match_id, p.market, p.line, p.prob_over, p.prob_under,
                   row_number() over (partition by p.match_id, p.market, p.line
                                      order by p.computed_at desc) as rn
            from models.predictions p where p.entity_type = 'match'
        )
        update models.predictions p
        set best_odds = case when l.prob_over >= l.prob_under then t.over_price else t.under_price end,
            best_bookmaker_id = case when l.prob_over >= l.prob_under then t.over_book else t.under_book end,
            implied_prob = case when l.prob_over >= l.prob_under
                                then 1.0 / t.over_price else 1.0 / t.under_price end,
            edge = case when l.prob_over >= l.prob_under
                        then l.prob_over * t.over_price - 1
                        else l.prob_under * t.under_price - 1 end,
            flagged = case when l.prob_over >= l.prob_under
                           then l.prob_over * t.over_price - 1
                           else l.prob_under * t.under_price - 1 end > 0.02
        from latest l
        join two_sided t
          on t.match_id = l.match_id and t.market = l.market and t.line = l.line
        where p.prediction_id = l.prediction_id and l.rn = 1
          and t.over_price is not null and t.under_price is not null
    """))
    await session.commit()
    return result.rowcount


REPORT = text("""
    with best as (
        select o.match_id, o.market, o.line, o.selection, max(o.decimal_price) as price
        from core.odds_snapshots o
        join core.matches m on m.match_id = o.match_id
        where m.kickoff_utc > now() and o.entity_type = 'match'
        group by 1, 2, 3, 4
    ),
    sides as (
        select match_id, market, line,
               max(price) filter (where selection = 'over') as over_price,
               max(price) filter (where selection = 'under') as under_price
        from best group by 1, 2, 3
    ),
    latest as (
        select distinct on (p.match_id, p.market, p.line)
               p.match_id, p.market, p.line, p.prob_over
        from models.predictions p where p.entity_type = 'match'
        order by p.match_id, p.market, p.line, p.computed_at desc
    )
    select m.kickoff_utc, th.name as home, ta.name as away, c.name as competition,
           s.market, s.line, s.over_price, s.under_price, l.prob_over
    from sides s
    join latest l on l.match_id = s.match_id and l.market = s.market and l.line = s.line
    join core.matches m on m.match_id = s.match_id
    join core.teams th on th.team_id = m.home_team_id
    join core.teams ta on ta.team_id = m.away_team_id
    join core.seasons se on se.season_id = m.season_id
    join core.competitions c on c.competition_id = se.competition_id
    where s.over_price is not null and s.under_price is not null
    order by m.kickoff_utc
""")


async def report(session: AsyncSession) -> list[dict]:
    """Model against market, with the bookmaker's margin removed."""
    out = []
    for row in (await session.execute(REPORT)).all():
        # Prices come back as Decimal; probabilities are floats.
        over_price, under_price = float(row.over_price), float(row.under_price)
        fair_over, fair_under = devig([over_price, under_price])
        margin = 1 / over_price + 1 / under_price - 1
        for side, ours, market_p, price in (
            ("over", row.prob_over, fair_over, over_price),
            ("under", 1 - row.prob_over, fair_under, under_price),
        ):
            out.append({
                "kickoff": row.kickoff_utc, "fixture": f"{row.home} v {row.away}",
                "competition": row.competition, "market": row.market, "line": float(row.line),
                "side": side, "ours": ours, "market_prob": market_p, "price": price,
                "margin": margin, "edge": ours * float(price) - 1,
            })
    return out


async def main(args: argparse.Namespace) -> None:
    try:
        async with SessionLocal() as session:
            if not args.report_only:
                summary = await fetch(session, dry_run=args.dry_run)
                print(
                    f"{summary['matched']} of {summary['candidates']} fixtures matched to the "
                    f"odds provider; {summary['priced']} priced, {summary['prices']} prices stored"
                )
                print(f"credits used {summary['credits_used']}, left {summary['credits_left']}")
                if summary["unmatched"]:
                    print("  not found there: " + ", ".join(summary["unmatched"][:6]))
                if not args.dry_run:
                    updated = await apply_edges(session)
                    print(f"{updated} predictions updated with best price and edge")

            lines = await report(session)

        if not lines:
            print("\nno stored prices yet")
            return
        value = sorted([r for r in lines if r["edge"] > 0], key=lambda r: -r["edge"])
        print(f"\n{len(lines)} priced sides; margin averages "
              f"{sum(r['margin'] for r in lines) / len(lines):.1%}")
        print(f"sides where our probability beats the price: {len(value)}")
        for r in value[:15]:
            print(
                f"  {r['kickoff']:%a %d %b}  {r['fixture'][:34]:36} {r['market'][6:]:8}"
                f"{r['side']:6}{r['line']:>6}  price {r['price']:>5.2f}  "
                f"ours {r['ours']:.1%} vs market {r['market_prob']:.1%}  edge {r['edge']:+.1%}"
            )
    finally:
        await dispose_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch market prices and compare with the model.")
    parser.add_argument("--dry-run", action="store_true", help="match fixtures, spend no credits")
    parser.add_argument("--report", dest="report_only", action="store_true",
                        help="only compare prices already stored")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(parser.parse_args()))
