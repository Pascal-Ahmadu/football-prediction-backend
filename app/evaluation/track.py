"""Settle past forecasts against what happened, and report live accuracy (B7.5).

Every market's test season has been spent, so the only honest evidence left is
how the live models do from here. This records, for each played fixture, the last
forecast made before kickoff -- the one anybody acting on the model would have
had -- together with the actual count.

    python -m app.evaluation.track              # settle, then report
    python -m app.evaluation.track --days 90    # report window
"""

import argparse
import asyncio
import logging
import math
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal, dispose_engine
from app.modelling.player_predict import FORECASTS as PLAYER_FORECASTS
from app.modelling.predict import FORECASTS as MATCH_FORECASTS

logger = logging.getLogger(__name__)

# How each market's actual count is read back out of core.
MATCH_ACTUALS = {
    "total_corners": "h.corners + a.corners",
    "total_cards": "h.yellow + a.yellow + h.red + a.red",
    "total_fouls": "h.fouls + a.fouls",
}
PLAYER_ACTUALS = {
    "player_shots_on_target": "s.shots_on_target",
    "player_fouls_committed": "s.fouls_committed",
}

# The last forecast made strictly before kickoff, per market, entity and line.
SETTLE_MATCH = """
with priced as (
    select o.match_id, o.market, o.line,
           max(o.decimal_price) filter (where o.selection = 'over') as over_price,
           max(o.decimal_price) filter (where o.selection = 'under') as under_price,
           max(o.captured_at) as priced_at
    from core.odds_snapshots o
    join core.matches m2 on m2.match_id = o.match_id
    where o.entity_type = 'match' and o.captured_at < m2.kickoff_utc
    group by 1, 2, 3
)
insert into models.prediction_outcomes (
    match_id, market, entity_type, entity_id, line, model_version, kickoff_utc,
    computed_at, hours_before_kickoff, prob_over, expected_value,
    baseline_expected_value, actual, went_over, over_price, under_price, priced_at)
select distinct on (p.match_id, p.market, p.line)
       p.match_id, p.market, p.entity_type, p.entity_id, p.line, p.model_version,
       m.kickoff_utc, p.computed_at,
       extract(epoch from (m.kickoff_utc - p.computed_at)) / 3600.0,
       p.prob_over, d.expected_value, d.baseline_expected_value,
       ({actual})::float as actual, ({actual}) > p.line as went_over,
       pr.over_price, pr.under_price, pr.priced_at
from models.predictions p
join core.matches m on m.match_id = p.match_id
join core.team_match_stats h on h.match_id = m.match_id and h.team_id = m.home_team_id
join core.team_match_stats a on a.match_id = m.match_id and a.team_id = m.away_team_id
left join priced pr
  on pr.match_id = p.match_id and pr.market = p.market and pr.line = p.line
left join models.distributions d
  on d.match_id = p.match_id and d.market = p.market
 and d.model_version = p.model_version and d.computed_at = p.computed_at
 and d.entity_id is not distinct from p.entity_id
where p.market = :market and p.entity_type = 'match'
  and m.status = 'Finished' and m.kickoff_utc < now()
  and p.computed_at < m.kickoff_utc
  and ({actual}) is not null
order by p.match_id, p.market, p.line, p.computed_at desc
on conflict on constraint uq_prediction_outcome do nothing
"""

SETTLE_PLAYER = """
with priced as (
    select o.match_id, o.market, o.line,
           max(o.decimal_price) filter (where o.selection = 'over') as over_price,
           max(o.decimal_price) filter (where o.selection = 'under') as under_price,
           max(o.captured_at) as priced_at
    from core.odds_snapshots o
    join core.matches m2 on m2.match_id = o.match_id
    where o.entity_type = 'match' and o.captured_at < m2.kickoff_utc
    group by 1, 2, 3
)
insert into models.prediction_outcomes (
    match_id, market, entity_type, entity_id, line, model_version, kickoff_utc,
    computed_at, hours_before_kickoff, prob_over, expected_value,
    baseline_expected_value, actual, went_over, over_price, under_price, priced_at)
select distinct on (p.match_id, p.market, p.entity_id, p.line)
       p.match_id, p.market, p.entity_type, p.entity_id, p.line, p.model_version,
       m.kickoff_utc, p.computed_at,
       extract(epoch from (m.kickoff_utc - p.computed_at)) / 3600.0,
       p.prob_over, d.expected_value, d.baseline_expected_value,
       ({actual})::float as actual, ({actual}) > p.line as went_over,
       pr.over_price, pr.under_price, pr.priced_at
from models.predictions p
join core.matches m on m.match_id = p.match_id
join core.player_match_stats s
  on s.match_id = p.match_id and s.player_id = p.entity_id
left join priced pr
  on pr.match_id = p.match_id and pr.market = p.market and pr.line = p.line
left join models.distributions d
  on d.match_id = p.match_id and d.market = p.market
 and d.model_version = p.model_version and d.computed_at = p.computed_at
 and d.entity_id is not distinct from p.entity_id
where p.market = :market and p.entity_type = 'player'
  and m.status = 'Finished' and m.kickoff_utc < now()
  and p.computed_at < m.kickoff_utc
  -- a player prop is void if he did not play, so those are not settled at all
  and s.minutes > 0 and ({actual}) is not null
order by p.match_id, p.market, p.entity_id, p.line, p.computed_at desc
on conflict on constraint uq_prediction_outcome do nothing
"""

REPORT = text("""
    select market, line, count(*) as n,
           avg(case when (prob_over > 0.5) = went_over then 1.0 else 0.0 end) as hit_rate,
           avg(-(case when went_over then ln(greatest(prob_over, 1e-9))
                      else ln(greatest(1 - prob_over, 1e-9)) end)) as log_loss,
           avg(case when went_over then 1.0 else 0.0 end) as over_rate,
           avg(prob_over) as mean_prob,
           avg(abs(actual - expected_value)) as mean_abs_error,
           min(kickoff_utc) as first_kickoff, max(kickoff_utc) as last_kickoff,
           avg(hours_before_kickoff) as hours_before,
           count(*) filter (where over_price is not null) as priced,
           -- flat one-unit stake on whichever side the model prices as value
           sum(case
               when over_price is null then null
               when prob_over * over_price > (1 - prob_over) * under_price
                    and prob_over * over_price > 1
                   then case when went_over then over_price - 1 else -1 end
               when (1 - prob_over) * under_price > 1
                   then case when went_over then -1 else under_price - 1 end
               else 0 end) as profit,
           count(*) filter (where over_price is not null and greatest(
               prob_over * over_price, (1 - prob_over) * under_price) > 1) as bets
    from models.prediction_outcomes
    where kickoff_utc >= :since
    group by market, line
    order by market, line
""")


async def settle(session: AsyncSession) -> dict[str, int]:
    """Record outcomes for forecasts whose fixtures have now been played."""
    written = {}
    for market, actual in MATCH_ACTUALS.items():
        if market not in MATCH_FORECASTS:
            continue
        result = await session.execute(text(SETTLE_MATCH.format(actual=actual)), {"market": market})
        written[market] = result.rowcount
    for market, actual in PLAYER_ACTUALS.items():
        if market not in PLAYER_FORECASTS:
            continue
        result = await session.execute(text(SETTLE_PLAYER.format(actual=actual)), {"market": market})
        written[market] = result.rowcount
    await session.commit()
    return written


def _baseline_log_loss(over_rate: float) -> float:
    """Log loss of always predicting the observed frequency -- the yardstick."""
    p = min(max(over_rate, 1e-9), 1 - 1e-9)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


async def report(session: AsyncSession, days: int, reference_only: bool = True) -> list[dict]:
    since = datetime.now(UTC) - timedelta(days=days)
    rows = (await session.execute(REPORT, {"since": since})).all()
    reference = {
        **{m: spec.reference_line for m, spec in MATCH_FORECASTS.items()},
        **{m: spec.lines[0] for m, spec in PLAYER_FORECASTS.items()},
    }
    out = []
    for row in rows:
        if reference_only and float(row.line) != reference.get(row.market):
            continue
        bets = int(row.bets or 0)
        out.append({
            "market": row.market, "line": float(row.line), "settled": row.n,
            "priced": int(row.priced or 0), "bets": bets,
            "profit": None if row.profit is None else float(row.profit),
            "roi": None if not bets or row.profit is None else float(row.profit) / bets,
            "hit_rate": float(row.hit_rate), "log_loss": float(row.log_loss),
            "baseline_log_loss": _baseline_log_loss(float(row.over_rate)),
            "over_rate": float(row.over_rate), "mean_prob": float(row.mean_prob),
            "mean_abs_error": None if row.mean_abs_error is None else float(row.mean_abs_error),
            "first_kickoff": row.first_kickoff, "last_kickoff": row.last_kickoff,
            "hours_before": float(row.hours_before),
        })
    return out


async def main(days: int, settle_first: bool) -> None:
    try:
        async with SessionLocal() as session:
            if settle_first:
                written = await settle(session)
                total = sum(written.values())
                print(f"settled {total} forecasts: " + ", ".join(
                    f"{market} {count}" for market, count in written.items() if count
                ) or "settled nothing new")
            lines = await report(session, days)

        if not lines:
            print(f"\nno settled forecasts in the last {days} days yet")
            return
        print(f"\nlive accuracy, fixtures played in the last {days} days")
        print(f"{'market':24}{'line':>6}{'n':>8}{'hit':>8}{'log loss':>10}{'vs base':>10}"
              f"{'calls over':>12}{'actual over':>12}{'lead time':>11}")
        for r in lines:
            print(
                f"{r['market']:24}{r['line']:>6}{r['settled']:>8}{r['hit_rate']:>8.1%}"
                f"{r['log_loss']:>10.5f}{r['baseline_log_loss'] - r['log_loss']:>+10.5f}"
                f"{r['mean_prob']:>12.1%}{r['over_rate']:>12.1%}{r['hours_before']:>9.0f}h"
            )
        print("\n'vs base' is how much better than always predicting the observed rate;"
              " positive is good.")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settle forecasts and report live accuracy.")
    parser.add_argument("--days", type=int, default=90, help="report window")
    parser.add_argument("--no-settle", action="store_true", help="report only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(args.days, not args.no_settle))
