"""Weekly refresh: fixtures -> features -> forecasts, in the only safe order.

Run before each matchweek:
    python -m app.pipeline.weekly

Order matters. Features must be built after the latest results are ingested, and
forecasts after the features. Run them out of order and nothing fails -- you
just get forecasts built on last week's form.

Referees are usually named only a few days before kickoff, and cards forecasts
are sharper once the referee is known. So run a short refresh again two days
before the weekend's matches:
    python -m app.pipeline.weekly --days-back 3 --days-ahead 4
"""

import argparse
import asyncio
import calendar
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func, select, text

from app.api.picks import list_picks
from app.config import settings
from app.db import SessionLocal, dispose_engine
from app.evaluation import track
from app.features import players, rates
from app.ingest import football_data, market_odds
from app.ingest.fixtures import ingest_league
from app.modelling import player_predict, predict
from app.notifications import notify
from app.models import Match


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range at month boundaries.

    apifootball silently returns thinner payloads for long ranges, so fixtures
    are always fetched at most one calendar month at a time.
    """
    windows = []
    current = start
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)
        windows.append((current, min(month_end, end)))
        current = month_end + timedelta(days=1)
    return windows


async def check_database() -> None:
    """Fail in seconds, not minutes, when Postgres is down."""
    try:
        async with SessionLocal() as session:
            await asyncio.wait_for(session.execute(text("select 1")), timeout=15)
    except Exception as exc:  # noqa: BLE001 -- any failure means the same thing here
        raise SystemExit(
            f"Postgres is not reachable ({type(exc).__name__}). "
            "Is Docker running? Try: docker start fmep-pg"
        ) from exc


def feed_verdict(results: list[dict]) -> str | None:
    """Why the fixture feed should be treated as broken, or None if it is healthy.

    A subscription lapsing looks exactly like a quiet week to each individual
    league call: the provider answers, politely, with nothing.
    """
    if not results:
        return "no leagues were requested"
    errors = [r for r in results if r["error"]]
    if len(errors) == len(results):
        return f"every request failed; first: {errors[0]['error']}"
    if sum(r["events"] for r in results) == 0:
        return "the provider returned no matches at all, for any league or date"
    plan = [r for r in errors if "plan" in r["error"].lower()]
    if plan:
        leagues = ", ".join(sorted({r["league_id"] for r in plan})[:6])
        return f"{len(plan)} leagues refused for subscription reasons (e.g. {leagues})"
    return None


async def refresh_fixtures(days_back: int, days_ahead: int) -> list[dict]:
    today = datetime.now(UTC).date()
    windows = month_windows(today - timedelta(days=days_back), today + timedelta(days=days_ahead))
    results: list[dict] = []
    for start, end in windows:
        print(f"  {start} .. {end}")
        for league_id in settings.league_ids:
            try:
                results.append(await ingest_league(start, end, league_id))
            except Exception as exc:  # noqa: BLE001 -- one league must not stop the rest
                print(f"league {league_id}: FAILED -- {type(exc).__name__}: {exc}")
                results.append({"league_id": league_id, "events": 0, "created": 0,
                                "updated": 0, "rejected": 0,
                                "error": f"{type(exc).__name__}: {exc}"})
    return results


async def run(args: argparse.Namespace) -> None:
    await check_database()
    started = time.monotonic()

    if args.skip_fixtures:
        print("1/3 fixtures: skipped")
    else:
        step = time.monotonic()
        print(f"1/3 fixtures: last {args.days_back} days and next {args.days_ahead} days")
        results = await refresh_fixtures(args.days_back, args.days_ahead)
        print(
            f"    {sum(r['events'] for r in results)} events, "
            f"{sum(r['created'] for r in results)} created, "
            f"{sum(r['updated'] for r in results)} updated "
            f"in {time.monotonic() - step:.0f}s"
        )
        broken = feed_verdict(results)
        if broken:
            # The paid feed is unavailable (a lapsed subscription looks like a quiet
            # week). Fall back to the free source, which covers 16 of our 17 leagues
            # for the three match markets but has no player data.
            print(f"    paid feed unusable: {broken}")
            print("    falling back to football-data.co.uk (free)")
            async with SessionLocal() as session:
                free = await football_data.ingest_results(
                    session, football_data.current_season_code(), 
                    await football_data.loaded_resolver(session), dry_run=False,
                )
                fixtures = await football_data.ingest_fixtures(
                    session, await football_data.loaded_resolver(session), dry_run=False,
                )
                recovered = free["updated"] + free["created"]
                # Nothing NEW is fine -- it means we are already up to date. What
                # matters is whether the source answered at all.
                reachable = free["rows"] > 0 or fixtures["rows"] > 0
                print(
                    f"    free source: {free['rows']} rows read, {recovered} results, "
                    f"{fixtures['created']} new fixtures"
                )
                await notify(
                    session,
                    "warning" if reachable else "error",
                    "pipeline", "Running on the free data source",
                    detail=(
                        f"The paid feed is unusable ({broken}). football-data.co.uk read "
                        f"{free['rows']} rows and supplied {recovered} results and "
                        f"{fixtures['created']} fixtures. Player props cannot be refreshed "
                        "from it, and Austria is not covered."
                    ),
                    context={"rows": free["rows"], "recovered": recovered,
                             "fixtures": fixtures["created"]},
                )
                if not reachable:
                    raise SystemExit("neither the paid nor the free source could be reached")

    step = time.monotonic()
    print("2/3 features")
    async with SessionLocal() as session:
        historical, upcoming = await rates.build(session)
    print(
        f"    teams and referees: {historical} historical + {upcoming} upcoming "
        f"({settings.feature_set_version}) in {time.monotonic() - step:.0f}s"
    )
    step = time.monotonic()
    async with SessionLocal() as session:
        historical, upcoming = await players.build(session)
    print(
        f"    players: {historical} historical + {upcoming} upcoming "
        f"({settings.player_feature_set_version}) in {time.monotonic() - step:.0f}s"
    )

    # Results arriving is the one thing nothing else can compensate for.
    async with SessionLocal() as session:
        newest_result = await session.scalar(
            select(func.max(Match.kickoff_utc)).where(Match.status == "Finished")
        )
        stale_days = (datetime.now(UTC) - newest_result).days if newest_result else None
        if stale_days is not None and stale_days > 4:
            await notify(
                session, "warning", "pipeline",
                f"No match results for {stale_days} days",
                detail=(
                    f"The most recent finished match is {newest_result:%d %b %Y}. Features and "
                    "forecasts are being built on form that old."
                ),
                context={"stale_days": stale_days},
            )
            print(f"    WARNING: newest result is {stale_days} days old")

    # Settle before forecasting: the forecast step prunes old rows.
    async with SessionLocal() as session:
        settled = await track.settle(session)
    if sum(settled.values()):
        print("    settled " + ", ".join(f"{m} {n}" for m, n in settled.items() if n))

    step = time.monotonic()
    print("3/3 forecasts")
    for market_name in predict.FORECASTS:
        async with SessionLocal() as session:
            summary = await predict.predict(session, market_name)
        for model in summary["models"]:
            print(
                f"    {model['model_version']}: {model['fixtures']} fixtures, "
                f"recent hit {model['hit_rate']:.1%} at {model['line']}"
            )
    for market_name in player_predict.FORECASTS:
        async with SessionLocal() as session:
            summary = await player_predict.predict(session, market_name)
        print(
            f"    {summary['model_version']}: {summary['forecast']} player forecasts, "
            f"held-back log loss {summary['log_loss']:.5f} at {summary['line']}"
        )
    print(f"    done in {time.monotonic() - step:.0f}s")

    if args.with_odds and settings.odds_api_key:
        try:
            async with SessionLocal() as session:
                summary = await market_odds.fetch(session)
                updated = await market_odds.apply_edges(session)
            print(
                f"    market prices: {summary['priced']} fixtures, {summary['prices']} prices, "
                f"{updated} predictions priced ({summary['credits_used']} credits used, "
                f"{summary['credits_left']} left)"
            )
        except Exception as exc:  # noqa: BLE001 -- odds are a bonus, never the point
            print(f"    market prices: FAILED -- {type(exc).__name__}: {exc}")

    now = datetime.now(UTC)
    week_ahead = now + timedelta(days=7)
    async with SessionLocal() as session:
        upcoming_fixtures = await session.scalar(
            select(func.count()).select_from(Match).where(Match.kickoff_utc > now)
        )
        referees_named = await session.scalar(
            select(func.count())
            .select_from(Match)
            .where(
                Match.kickoff_utc > now,
                Match.kickoff_utc <= week_ahead,
                Match.referee_id.is_not(None),
            )
        )
        fixtures_this_week = await session.scalar(
            select(func.count())
            .select_from(Match)
            .where(Match.kickoff_utc > now, Match.kickoff_utc <= week_ahead)
        )
        picks_by_market = {
            market_name: await list_picks(
                session=session, market=market_name, line=None,
                min_confidence=0.05, competition_id=None, limit=200,
                rank_by="league",
            )
            for market_name in (*predict.FORECASTS, *player_predict.FORECASTS)
        }

    print(
        f"\ndone in {time.monotonic() - started:.0f}s -- {upcoming_fixtures} upcoming fixtures; "
        f"referees named for {referees_named} of {fixtures_this_week} in the next 7 days"
    )

    async with SessionLocal() as session:
        live = await track.report(session, days=90)
    if live:
        print("\nlive accuracy over the last 90 days (settled forecasts)")
        for row in live:
            print(
                f"  {row['market']:24} line {row['line']:<6} n={row['settled']:<6} "
                f"hit {row['hit_rate']:.1%}  log loss {row['log_loss']:.5f} "
                f"({row['baseline_log_loss'] - row['log_loss']:+.5f} vs base)"
            )
    async with SessionLocal() as session:
        await notify(
            session, "info", "pipeline",
            f"Forecasts ready for {upcoming_fixtures} fixtures",
            detail=(
                f"{sum(len(p) for p in picks_by_market.values())} confident picks across "
                f"{len(picks_by_market)} markets; referees named for {referees_named} of "
                f"{fixtures_this_week} matches in the next 7 days."
            ),
            context={"fixtures": upcoming_fixtures, "minutes": round((time.monotonic() - started) / 60, 1)},
        )

        # A pick that beats the price on offer is the only kind worth acting on.
        value = (await session.execute(text("""
            select count(*) as n, max(edge) as best
            from models.predictions p
            join core.matches m on m.match_id = p.match_id
            where p.edge is not null and p.edge > 0.02 and m.kickoff_utc > now()
        """))).one()
        if value.n:
            await notify(
                session, "info", "value",
                f"{value.n} selections beat the market price",
                detail=f"Best edge {value.best:+.1%} at the prices collected before kickoff.",
                context={"selections": value.n, "best_edge": float(value.best)},
            )

        # Live accuracy slipping well below the tested figures means a stale model.
        for row in live:
            if row["settled"] >= 100 and row["log_loss"] > row["baseline_log_loss"]:
                await notify(
                    session, "warning", "accuracy",
                    f"{row['market']} is not beating its baseline",
                    detail=(
                        f"Over {row['settled']} settled forecasts at line {row['line']}, "
                        f"log loss {row['log_loss']:.4f} against a baseline of "
                        f"{row['baseline_log_loss']:.4f}."
                    ),
                    context={"market": row["market"], "line": row["line"], "settled": row["settled"]},
                )

    # Forecasts further out are built on today's form and will be redone before
    # then, so the report only shows the coming week.
    for market_name, picks in picks_by_market.items():
        soon = [p for p in picks if p.kickoff_utc <= week_ahead]
        print(f"\n{market_name}: {len(soon)} confident picks in the next 7 days, most unusual first")
        for pick in soon[:10]:
            typical = (
                f"typical {pick.base_rate if pick.selection == 'over' else 1 - pick.base_rate:.0%}"
                if pick.base_rate is not None else "typical n/a"
            )
            who = f"{pick.entity_name}: " if pick.entity_name else ""
            print(
                f"  {pick.kickoff_utc:%a %d %b}  {who}{pick.home_team.name} v {pick.away_team.name}  "
                f"{pick.selection.upper()} {pick.line}  {pick.probability:.1%} ({typical})  "
                f"({pick.competition}, {pick.country})"
            )


async def _main(args: argparse.Namespace) -> None:
    try:
        await run(args)
    except Exception as exc:  # noqa: BLE001 -- the run is unattended; record why it stopped
        try:
            async with SessionLocal() as session:
                await notify(
                    session, "error", "pipeline", "Weekly run failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
        except Exception:  # noqa: BLE001 -- the database may be what failed
            logging.exception("could not record the failure notification")
        raise
    finally:
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly refresh: fixtures, features, forecasts.")
    parser.add_argument(
        "--days-back", type=int, default=14,
        help="re-ingest this many past days, to pick up final scores and statistics",
    )
    parser.add_argument(
        "--days-ahead", type=int, default=45, help="load fixtures this many days ahead",
    )
    parser.add_argument(
        "--skip-fixtures", action="store_true", help="only rebuild features and forecasts",
    )
    parser.add_argument(
        "--with-odds", action="store_true",
        help="also fetch market prices for the most confident fixtures (spends API credits)",
    )
    args = parser.parse_args()

    # Team names include characters such as Turkish "ı" that the Windows console's
    # default encoding cannot print; replace rather than crash.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
