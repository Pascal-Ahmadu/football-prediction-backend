"""Forecast upcoming fixtures, market by market (FR-PRED-01, FR-PRED-02).

For each market, trains on every finished match -- with the same fit_model the
walk-forward evaluations use -- then writes, for each upcoming fixture:

  models.distributions  the probability of every count
  models.predictions    P(over), P(under) and fair odds at each line

and registers every model, so each forecast traces back to the exact model and
feature version that produced it (FR-ML-06, BR-06).

Markets with referee features train TWO models: one with the referee and one
without. The 2024/25 evaluation showed a referee-trained model does badly when
the referee is missing, so each fixture uses the referee model only if its
referee is already known, and the forecast row records which model that was.

Run after app.features.rates, so upcoming snapshots are fresh:
    python -m app.modelling.predict                       # every market
    python -m app.modelling.predict --market total_cards
"""

import argparse
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from scipy.stats import nbinom
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import SessionLocal, dispose_engine
from app.modelling.corners import (
    INNER_VALID_WEEKS,
    PARAMS,
    fit_model,
    log_loss,
    prob_over,
)
from app.modelling.dataset import load, load_upcoming
from app.models import Distribution, ModelRegistry, Prediction

PMF_MAX = 30  # counts 0..29 stored individually, the rest as "30+"
MIN_LEAGUE_MATCHES = 100  # fewer than this and a league's base rate is too noisy


@dataclass(frozen=True)
class ForecastSpec:
    version_prefix: str
    lines: tuple[float, ...]
    reference_line: float


FORECASTS = {
    "total_corners": ForecastSpec("corners", (7.5, 8.5, 9.5, 10.5, 11.5, 12.5), 9.5),
    "total_cards": ForecastSpec("cards", (2.5, 3.5, 4.5, 5.5, 6.5), 4.5),
    "total_fouls": ForecastSpec(
        "fouls", (20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5), 24.5
    ),
}

CALIBRATION_NOTE = {
    "total_corners": "raw ECE 0.0123 beat isotonic 0.0139 in 2024/25 walk-forward",
    "total_cards": "raw ECE 0.0116 in 2025/26 final evaluation; no adjustment needed",
    "total_fouls": "raw ECE 0.0174 in 2025/26 final evaluation; no adjustment needed",
}


def _utc(value: np.datetime64) -> datetime:
    return value.astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)


async def predict(session: AsyncSession, market_name: str = "total_corners") -> dict:
    """Train, register and forecast one market. Commits."""
    spec = FORECASTS[market_name]
    now = datetime.now(UTC)
    stamp = f"{now:%Y%m%d-%H%M%S}"

    X, y, _seasons, kickoffs, names = await load(session, market_name)
    X_up, match_ids = await load_upcoming(session, market_name)
    weeks = kickoffs.astype("datetime64[W]").astype(int)
    competitions = X[:, names.index("competition_id")].astype(int)

    referee_cols =[i for i, n in enumerate(names) if n.startswith("ref_")]
    if referee_cols:
        variants = {
            "referee": list(range(len(names))),
            "team": [i for i in range(len(names)) if i not in referee_cols],
        }
        if match_ids:
            referee_known = ~np.isnan(X_up[:, names.index("ref_matches_seen")])
            chosen = np.where(referee_known, "referee", "team")
        else:
            chosen = np.array([], dtype=str)
    else:
        variants = {"": list(range(len(names)))}
        chosen = np.full(len(match_ids), "", dtype=object)

    summary: dict = {"market": market_name, "fixtures": len(match_ids), "models": []}
    fitted: dict[str, tuple] = {}

    for variant, columns in variants.items():
        version = "-".join(p for p in (spec.version_prefix, variant, "nb-lgbm", stamp) if p)
        variant_names = [names[i] for i in columns]
        model, r, rounds, inner_valid, mu_valid = fit_model(
            X[:, columns], y, weeks, variant_names, int(weeks.max()) + 1
        )

        line = spec.reference_line
        actual = (y[inner_valid] > line).astype(float)
        earlier = ~inner_valid
        base_rate = float((y[earlier] > line).mean()) if earlier.any() else float(actual.mean())
        prob = prob_over(mu_valid, r, line)
        metrics = {
            "window": f"last {INNER_VALID_WEEKS} weeks of history",
            "matches": int(inner_valid.sum()),
            "line": line,
            "log_loss": log_loss(actual, prob),
            "baseline_log_loss": log_loss(actual, np.full_like(prob, base_rate)),
            "hit_rate": float(np.where(prob > 0.5, actual, 1 - actual).mean()),
            "mean_abs_error": float(np.abs(y[inner_valid] - mu_valid).mean()),
            "dispersion_r": r,
            # How often a typical match goes over each line, across all history and
            # per league. Picks are ranked by how far a forecast departs from its
            # LEAGUE's rate: distance from 50/50, or from the global rate, mostly
            # rediscovers which leagues are high- or low-scoring (see picks.py).
            "base_rates": {str(float(ln)): float((y > ln).mean()) for ln in spec.lines},
            "base_rates_by_competition": {
                str(int(c)): {
                    str(float(ln)): float((y[competitions == c] > ln).mean())
                    for ln in spec.lines
                }
                for c in np.unique(competitions)
                if (competitions == c).sum() >= MIN_LEAGUE_MATCHES
            },
        }

        session.add(
            ModelRegistry(
                model_version=version,
                market=market_name,
                algorithm="lightgbm-poisson + negative-binomial",
                feature_set_version=settings.feature_set_version,
                train_start=_utc(kickoffs.min()),
                train_end=_utc(kickoffs.max()),
                params={
                    **PARAMS,
                    "num_boost_round": rounds,
                    "features": variant_names,
                    "used_for": {
                        "referee": "fixtures whose referee is known",
                        "team": "fixtures whose referee is not yet known",
                        "": "all fixtures",
                    }[variant],
                },
                validation_metrics=metrics,
                calibration={"method": "none", "reason": CALIBRATION_NOTE[market_name]},
            )
        )
        fitted[variant] = (version, model, r, columns)
        summary["models"].append(
            {"model_version": version, "fixtures": int((chosen == variant).sum()), **metrics}
        )
    await session.flush()

    distributions: list[dict] = []
    predictions: list[dict] = []
    counts = np.arange(PMF_MAX)
    for variant, (version, model, r, columns) in fitted.items():
        rows = np.where(chosen == variant)[0]
        if rows.size == 0:
            continue
        mu = model.predict(X_up[rows][:, columns])
        p = r / (r + mu)
        for k, row in enumerate(rows):
            match_id = match_ids[row]
            pmf = {str(c): round(float(v), 6) for c, v in zip(counts, nbinom.pmf(counts, r, p[k]))}
            pmf[f"{PMF_MAX}+"] = round(float(nbinom.sf(PMF_MAX - 1, r, p[k])), 6)
            distributions.append({
                "match_id": match_id,
                "market": market_name,
                "entity_type": "match",
                "entity_id": None,
                "model_version": version,
                "computed_at": now,
                "expected_value": float(mu[k]),
                "pmf": pmf,
            })
            for line in spec.lines:
                over = float(nbinom.sf(np.floor(line), r, p[k]))
                under = 1.0 - over
                predictions.append({
                    "match_id": match_id,
                    "market": market_name,
                    "entity_type": "match",
                    "entity_id": None,
                    "line": line,
                    "model_version": version,
                    "prob_over": over,
                    "prob_under": under,
                    "fair_odds_over": 1.0 / max(over, 1e-9),
                    "fair_odds_under": 1.0 / max(under, 1e-9),
                    "flagged": False,
                    "computed_at": now,
                })

    if distributions:
        await session.execute(insert(Distribution), distributions)
        await session.execute(insert(Prediction), predictions)
    await session.commit()
    return summary


async def main(markets: list[str]) -> None:
    try:
        for market_name in markets:
            async with SessionLocal() as session:
                summary = await predict(session, market_name)
            print(f"{market_name}: forecast {summary['fixtures']} fixtures")
            for model in summary["models"]:
                print(
                    f"  {model['model_version']}: {model['fixtures']} fixtures; last "
                    f"{INNER_VALID_WEEKS} weeks held back ({model['matches']} matches): "
                    f"log loss {model['log_loss']:.5f} vs {model['baseline_log_loss']:.5f}, "
                    f"hit {model['hit_rate']:.1%} at {model['line']}"
                )
    finally:
        await dispose_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast upcoming fixtures.")
    parser.add_argument("--market", choices=[*FORECASTS, "all"], default="all")
    chosen_market = parser.parse_args().market
    asyncio.run(main(list(FORECASTS) if chosen_market == "all" else [chosen_market]))
