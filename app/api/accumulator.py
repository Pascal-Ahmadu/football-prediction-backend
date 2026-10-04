"""Build the best accumulator available (FR-WEB-05).

An accumulator's chance is capped by its payout: odds of 1000 mean about one
chance in a thousand, whoever picks the legs. What a model can change is which
legs get picked -- and every leg without genuine value multiplies the bookmaker's
margin instead of your return.

So this returns the best combination it can find, states the true chance plainly,
and says when nothing is worth backing.

Legs are taken from fixtures whose price we have collected, one leg per match:
cards and fouls in the same game move together, and combining them would overstate
the odds of success. (Measured on 2024/25: combining cards-under with fouls-under
lands 6% LESS often than multiplying the two probabilities suggests.)
"""

import math
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas import Accumulator, AccumulatorLeg

router = APIRouter(prefix="/accumulator", tags=["picks"])

CANDIDATES = text("""
    select distinct on (p.match_id, p.market, p.line)
           p.match_id, p.market, p.line, p.prob_over, p.prob_under,
           p.best_odds, p.edge, m.kickoff_utc,
           th.name as home, ta.name as away, c.name as competition
    from models.predictions p
    join core.matches m on m.match_id = p.match_id
    join core.teams th on th.team_id = m.home_team_id
    join core.teams ta on ta.team_id = m.away_team_id
    join core.seasons se on se.season_id = m.season_id
    join core.competitions c on c.competition_id = se.competition_id
    where p.entity_type = 'match' and p.best_odds is not null
      and m.kickoff_utc between now() and :until
    order by p.match_id, p.market, p.line, p.computed_at desc
""")


@router.get("", response_model=Accumulator)
async def accumulator(
    session: AsyncSession = Depends(get_session),
    legs: int = Query(3, ge=1, le=20, description="how many selections"),
    target_odds: float | None = Query(
        None, ge=1.1, description="combined odds to reach; legs are added until it is met"
    ),
    min_edge: float = Query(
        0.0, description="require this much value per leg, e.g. 0.02 for 2%"
    ),
    days: int = Query(7, ge=1, le=30),
) -> Accumulator:
    """The best combination of priced selections, with its true chance."""
    rows = (
        await session.execute(
            CANDIDATES, {"until": datetime.now(UTC) + timedelta(days=days)}
        )
    ).all()

    # One candidate per match: the side we back, at the price on offer.
    per_match: dict[int, AccumulatorLeg] = {}
    for row in rows:
        over = row.prob_over >= row.prob_under
        probability = row.prob_over if over else row.prob_under
        price = float(row.best_odds)
        edge = probability * price - 1
        if edge < min_edge:
            continue
        leg = AccumulatorLeg(
            match_id=row.match_id,
            kickoff_utc=row.kickoff_utc,
            fixture=f"{row.home} v {row.away}",
            competition=row.competition,
            market=row.market,
            line=float(row.line),
            selection="over" if over else "under",
            probability=probability,
            price=price,
            edge=edge,
        )
        if row.match_id not in per_match or edge > per_match[row.match_id].edge:
            per_match[row.match_id] = leg

    candidates = list(per_match.values())
    if not candidates:
        return Accumulator(
            legs=[], combined_odds=1.0, probability=0.0, expected_return=0.0,
            note=(
                "No upcoming fixture has a collected price offering value, so there is "
                "nothing to back. If this persists, check that prices are being fetched."
            ),
        )

    if target_odds is None:
        # Fixed number of legs: value multiplies, so take the best-priced ones.
        chosen = sorted(candidates, key=lambda leg: -leg.edge)[:legs]
    else:
        # Reaching a payout costs probability. Spend it where each unit of odds
        # costs the least chance: the highest log(probability) / log(price).
        ranked = sorted(
            candidates,
            key=lambda leg: math.log(max(leg.probability, 1e-9)) / math.log(max(leg.price, 1.0001)),
            reverse=True,
        )
        chosen, odds = [], 1.0
        for leg in ranked:
            if odds >= target_odds or len(chosen) >= legs:
                break
            chosen.append(leg)
            odds *= leg.price

    combined_odds = math.prod(leg.price for leg in chosen)
    probability = math.prod(leg.probability for leg in chosen)
    note = None
    if target_odds is not None and combined_odds < target_odds:
        note = (
            f"Only reached {combined_odds:.1f}x with {len(chosen)} legs; "
            f"{target_odds:.0f}x would need more priced fixtures than are available."
        )
    elif probability < 0.01:
        note = (
            f"This lands about once in {round(1 / probability):,} attempts. "
            "Each extra leg multiplies the bookmaker's margin as well as the odds."
        )

    return Accumulator(
        legs=chosen,
        combined_odds=combined_odds,
        probability=probability,
        expected_return=probability * combined_odds,
        note=note,
    )
