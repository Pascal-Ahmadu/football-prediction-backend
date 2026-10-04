"""Forecast player props for upcoming fixtures (FR-PRED-01, FR-PRED-02).

Trains each player market on every appearance in history -- with the same
fit_model the walk-forward evaluations use -- then, for each upcoming fixture,
forecasts every player likely to feature.

A player prop is settled only if the player plays, so these are probabilities
GIVEN that he plays. Whether he plays is a separate question; only players who
featured in at least MIN_APPS_RECENT of their team's last five matches are
forecast at all.

Player rows for fixtures that have already kicked off are deleted on each run, so
the tables do not grow without limit.

Run after app.features.players:
    python -m app.modelling.player_predict
    python -m app.modelling.player_predict --market player_fouls_committed
"""

import argparse
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from scipy.stats import nbinom
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import SessionLocal, dispose_engine
from app.modelling.corners import INNER_VALID_WEEKS, PARAMS, fit_model, log_loss, prob_over
from app.modelling.player_dataset import PLAYER_MARKETS, load, load_upcoming
from app.models import Distribution, Match, ModelRegistry, Prediction

PMF_MAX = 10  # counts 0..9 individually, the rest as "10+"
MIN_APPS_RECENT = 2   # appearances in the team's last five matches
MIN_APPEARANCES = 5   # a player's rates mean little before this many appearances
PRIOR_MATCHES = 5     # how far a player's own rate is shrunk toward his position's
KEEP_PLAYED_DAYS = 14 # played fixtures' rows are kept this long, for settlement


def baseline_expectation(X, names, target: str, position_rate: dict[float, float],
                         global_rate: float, typical_minutes: float) -> np.ndarray:
    """What a naive forecast would expect of each player: his own rate per minute,
    shrunk toward his position's rate, times his usual minutes.

    This is what the evaluations scored the model against and what picks are
    ranked by, so it is computed identically here.
    """
    position = X[:, names.index("position")]
    minutes_pa = X[:, names.index("minutes_pa")]
    minutes_seen = X[:, names.index("appearances_seen")] * np.nan_to_num(minutes_pa)
    prior = np.array([position_rate.get(p, global_rate) for p in position])
    own = np.nan_to_num(X[:, names.index(f"{target}_p90")] / 90.0)
    shrunk = (own * minutes_seen + prior * 90 * PRIOR_MATCHES) / (minutes_seen + 90 * PRIOR_MATCHES)
    return shrunk * np.where(np.isnan(minutes_pa), typical_minutes, minutes_pa)


@dataclass(frozen=True)
class PlayerForecastSpec:
    version_prefix: str
    lines: tuple[float, ...]


FORECASTS = {
    "player_shots_on_target": PlayerForecastSpec("player-sot", (0.5, 1.5, 2.5)),
    "player_fouls_committed": PlayerForecastSpec("player-fouls", (0.5, 1.5, 2.5, 3.5)),
}

CALIBRATION_NOTE = {
    "player_shots_on_target": "raw ECE 0.0048 at 0.5 in the 2025/26 final evaluation",
    "player_fouls_committed": "raw negative binomial; see the 2025/26 final evaluation",
}


def _utc(value: np.datetime64) -> datetime:
    return value.astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)


async def predict(session: AsyncSession, market_name: str) -> dict:
    """Train, register and forecast one player market. Commits."""
    spec = FORECASTS[market_name]
    market = PLAYER_MARKETS[market_name]
    now = datetime.now(UTC)
    version = f"{spec.version_prefix}-nb-lgbm-{now:%Y%m%d-%H%M%S}"

    X, y, _seasons, kickoffs, names, info = await load(session, market_name)
    X_up, rows, _names_up = await load_upcoming(session, market_name)
    weeks = kickoffs.astype("datetime64[W]").astype(int)

    model, r, rounds, inner_valid, mu_valid = fit_model(
        X, y, weeks, names, int(weeks.max()) + 1
    )

    line = spec.lines[0]
    actual = (y[inner_valid] > line).astype(float)
    earlier = ~inner_valid
    base_rate = float((y[earlier] > line).mean()) if earlier.any() else float(actual.mean())
    prob = prob_over(mu_valid, r, line)
    metrics = {
        "window": f"last {INNER_VALID_WEEKS} weeks of history",
        "appearances": int(inner_valid.sum()),
        "line": line,
        "log_loss": log_loss(actual, prob),
        "baseline_log_loss": log_loss(actual, np.full_like(prob, base_rate)),
        "mean_abs_error": float(np.abs(y[inner_valid] - mu_valid).mean()),
        "dispersion_r": r,
        "base_rates": {str(float(ln)): float((y > ln).mean()) for ln in spec.lines},
    }

    session.add(
        ModelRegistry(
            model_version=version,
            market=market_name,
            algorithm="lightgbm-poisson + negative-binomial",
            feature_set_version=f"{settings.player_feature_set_version}+{settings.feature_set_version}",
            train_start=_utc(kickoffs.min()),
            train_end=_utc(kickoffs.max()),
            params={**PARAMS, "num_boost_round": rounds, "features": names,
                    "target": market.target, "min_apps_recent": MIN_APPS_RECENT},
            validation_metrics=metrics,
            calibration={"method": "none", "reason": CALIBRATION_NOTE[market_name]},
        )
    )
    await session.flush()

    # Forecast only players likely to feature, and only once their own history
    # says something: the weighted rates are dominated by a player's first few
    # appearances until he has a handful of them.
    apps_recent = X_up[:, names.index("apps_recent")]
    appearances = X_up[:, names.index("appearances_seen")]
    likely = np.where(
        (np.nan_to_num(apps_recent) >= MIN_APPS_RECENT)
        & (np.nan_to_num(appearances) >= MIN_APPEARANCES)
    )[0]
    mu = model.predict(X_up[likely]) if likely.size else np.array([])
    p = r / (r + mu)

    position = X[:, names.index("position")]
    known_position = ~np.isnan(position)
    position_rate = {
        float(value): float(y[position == value].sum() / info["minutes"][position == value].sum())
        for value in np.unique(position[known_position])
    }
    baseline = baseline_expectation(
        X_up[likely], names, market.target, position_rate,
        global_rate=float(y.sum() / info["minutes"].sum()),
        typical_minutes=float(np.median(info["minutes"])),
    )

    # Old player rows are dropped once they are well past -- but not before
    # app.evaluation.track has had the chance to settle them against the result.
    played = select(Match.match_id).where(
        Match.kickoff_utc <= now - timedelta(days=KEEP_PLAYED_DAYS)
    ).scalar_subquery()
    for table in (Distribution, Prediction):
        await session.execute(
            delete(table).where(
                table.market == market_name,
                table.entity_type == "player",
                table.match_id.in_(played),
            )
        )

    counts = np.arange(PMF_MAX)
    distributions, predictions = [], []
    for k, row_index in enumerate(likely):
        row = rows[row_index]
        pmf = {str(c): round(float(v), 6) for c, v in zip(counts, nbinom.pmf(counts, r, p[k]))}
        pmf[f"{PMF_MAX}+"] = round(float(nbinom.sf(PMF_MAX - 1, r, p[k])), 6)
        distributions.append({
            "match_id": row["match_id"], "market": market_name, "entity_type": "player",
            "entity_id": row["player_id"], "model_version": version, "computed_at": now,
            "expected_value": float(mu[k]),
            "baseline_expected_value": float(baseline[k]),
            "pmf": pmf,
        })
        for ln in spec.lines:
            over = float(nbinom.sf(np.floor(ln), r, p[k]))
            predictions.append({
                "match_id": row["match_id"], "market": market_name, "entity_type": "player",
                "entity_id": row["player_id"], "line": ln, "model_version": version,
                "prob_over": over, "prob_under": 1.0 - over,
                "fair_odds_over": 1.0 / max(over, 1e-9),
                "fair_odds_under": 1.0 / max(1.0 - over, 1e-9),
                "flagged": False, "computed_at": now,
            })

    if distributions:
        await session.execute(insert(Distribution), distributions)
        await session.execute(insert(Prediction), predictions)
    await session.commit()
    return {
        "market": market_name, "model_version": version,
        "candidates": len(rows), "forecast": len(distributions), **metrics,
    }


async def main(markets: list[str]) -> None:
    try:
        for market_name in markets:
            async with SessionLocal() as session:
                summary = await predict(session, market_name)
            print(
                f"{summary['market']}: {summary['forecast']} of {summary['candidates']} "
                f"candidate players forecast ({summary['model_version']}); held-back "
                f"log loss {summary['log_loss']:.5f} vs {summary['baseline_log_loss']:.5f} "
                f"at {summary['line']}"
            )
    finally:
        await dispose_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast player props for upcoming fixtures.")
    parser.add_argument("--market", choices=[*FORECASTS, "all"], default="all")
    chosen = parser.parse_args().market
    asyncio.run(main(list(FORECASTS) if chosen == "all" else [chosen]))
