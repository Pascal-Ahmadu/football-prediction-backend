"""Walk-forward total-corners model (FR-ML-01, FR-ML-08).

Expanding window: at each matchweek, train on every match played before that
week and predict that week only. No model ever sees a match from its own week
or later, so every prediction is genuinely out of sample.

LightGBM with a Poisson objective gives the expected count; a Negative Binomial
turns that mean into a distribution, because corners are overdispersed.

Run:
    python -m app.modelling.corners
"""

import asyncio

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from app.db import SessionLocal, dispose_engine
from app.modelling.dataset import load

LINE = 9.5
CATEGORICAL = ["competition_id"]

MIN_TRAIN_ROWS = 1500          # warm-up before the first prediction
INNER_VALID_WEEKS = 26         # most recent weeks held back for early stopping
REPORT_SEASONS = (2024,)       # 2025 stays untouched until final evaluation
CALIBRATION_WINDOW = 100_000   # most recent earlier matches used to calibrate

PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}


def dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Negative Binomial dispersion r, by moment matching on residuals."""
    excess = float(np.mean((y - mu) ** 2 - mu))
    if excess <= 0:
        return 1e6  # no overdispersion found; effectively Poisson
    return float(np.mean(mu**2) / excess)


def prob_over(mu: np.ndarray, r: np.ndarray | float, line: float) -> np.ndarray:
    """P(count > line) under a Negative Binomial with mean mu."""
    from scipy.stats import nbinom

    p = r / (r + mu)
    return nbinom.sf(np.floor(line), r, p)


def brier(actual: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob - actual) ** 2))


def expected_calibration_error(actual, prob, bins: int = 10) -> float:
    """Weighted gap between predicted probability and observed frequency."""
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (prob >= lo) & (prob < hi)
        if band.sum():
            total += band.sum() * abs(prob[band].mean() - actual[band].mean())
    return float(total / len(prob))


def log_loss(actual: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(prob, 1e-9, 1 - 1e-9)
    return float(-np.mean(actual * np.log(prob) + (1 - actual) * np.log(1 - prob)))


def fit_model(X, y, weeks, names, week):
    """Train on every row before `week`, exactly as each walk-forward step does.

    Returns (model, dispersion r, boosting rounds, inner-validation mask,
    inner-validation predictions). Walk-forward evaluation and live prediction
    both call this, so the model that is scored is the model that forecasts.
    """
    history = weeks < week
    cutoff = week - INNER_VALID_WEEKS
    inner_train = history & (weeks < cutoff)
    inner_valid = history & (weeks >= cutoff)
    if inner_train.sum() < MIN_TRAIN_ROWS // 2 or inner_valid.sum() < 100:
        inner_train = inner_valid = history

    train_set = lgb.Dataset(
        X[inner_train], label=y[inner_train],
        feature_name=names, categorical_feature=CATEGORICAL,
    )
    valid_set = lgb.Dataset(
        X[inner_valid], label=y[inner_valid], feature_name=names,
        categorical_feature=CATEGORICAL, reference=train_set,
    )
    model = lgb.train(
        PARAMS, train_set, num_boost_round=2000, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )
    best = model.best_iteration or 100

    # Refit on the whole history at the chosen number of rounds, so the
    # most recent matches inform the model that makes the prediction.
    full = lgb.Dataset(
        X[history], label=y[history],
        feature_name=names, categorical_feature=CATEGORICAL,
    )
    final = lgb.train(PARAMS, full, num_boost_round=best)

    mu_valid = model.predict(X[inner_valid], num_iteration=best)
    return final, dispersion(y[inner_valid], mu_valid), best, inner_valid, mu_valid


def walk_forward(X, y, weeks, names):
    """Out-of-sample mean and dispersion for every match after the warm-up."""
    mu_out = np.full(len(y), np.nan)
    r_out = np.full(len(y), np.nan)
    fits = 0

    for week in np.unique(weeks):
        if (weeks < week).sum() < MIN_TRAIN_ROWS:
            continue
        final, r, *_ = fit_model(X, y, weeks, names, week)
        predict = weeks == week
        mu_out[predict] = final.predict(X[predict])
        r_out[predict] = r
        fits += 1

    return mu_out, r_out, fits


async def main() -> None:
    async with SessionLocal() as session:
        X, y, seasons, kickoffs, names = await load(session)
    await dispose_engine()

    weeks = kickoffs.astype("datetime64[W]").astype(int)
    print(f"{len(y)} matches, {len(np.unique(weeks))} matchweeks, {len(names)} features")

    mu, r, fits = walk_forward(X, y, weeks, names)
    print(f"walk-forward: {fits} refits, {int(np.isfinite(mu).sum())} predictions")

    report = np.isfinite(mu) & np.isin(seasons, REPORT_SEASONS)
    actual = (y[report] > LINE).astype(float)
    model_prob = prob_over(mu[report], r[report], LINE)

    warm = np.isfinite(mu) & ~np.isin(seasons, REPORT_SEASONS)
    base_rate = float((y[warm] > LINE).mean())
    base_prob = np.full_like(model_prob, base_rate)

    # Isotonic calibration (FR-ML-07), fitted only on out-of-sample predictions
    # from seasons BEFORE the reported one.
    # Only the most recent CALIBRATION_WINDOW earlier matches: how a model is
    # miscalibrated drifts, so a stale window calibrates to the wrong era.
    earlier = np.isfinite(mu) & (seasons < min(REPORT_SEASONS))
    if earlier.sum() > CALIBRATION_WINDOW:
        cut = np.sort(kickoffs[earlier])[-CALIBRATION_WINDOW]
        earlier &= kickoffs >= cut
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    calibrator.fit(
        prob_over(mu[earlier], r[earlier], LINE),
        (y[earlier] > LINE).astype(float),
    )
    calibrated = calibrator.predict(model_prob)
    print(f"calibration fitted on {int(earlier.sum())} earlier out-of-sample matches")

    picked = np.where(model_prob > 0.5, actual, 1 - actual)
    print(f"\nover {LINE}, seasons {REPORT_SEASONS}: {len(actual)} matches, "
          f"{actual.mean():.3f} went over")
    print(f"  baseline log loss : {log_loss(actual, base_prob):.5f} "
          f"(always predicts {base_rate:.3f})")
    print(f"  model    log loss : {log_loss(actual, model_prob):.5f}")
    print(f"  hit rate          : {picked.mean():.1%}   (break-even 52.4%)")
    print(f"  correlation       : {float(np.corrcoef(mu[report], y[report])[0, 1]):+.4f}")
    print(f"  P(over) spread    : sd {float(model_prob.std()):.4f}, "
          f"range {model_prob.min():.3f}-{model_prob.max():.3f}")

    print(f"\n  calibrated log loss: {log_loss(actual, calibrated):.5f}")
    print(f"  Brier  raw {brier(actual, model_prob):.5f} -> "
          f"calibrated {brier(actual, calibrated):.5f}")
    print(f"  ECE    raw {expected_calibration_error(actual, model_prob):.5f} -> "
          f"calibrated {expected_calibration_error(actual, calibrated):.5f}"
          f"   (B7.6 wants < 0.02)")

    print("\n  reliability (raw):  predicted -> observed")
    edges = np.quantile(model_prob, np.linspace(0, 1, 9))
    edges[0], edges[-1] = -np.inf, np.inf
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (model_prob >= lo) & (model_prob < hi)
        if band.sum():
            print(f"    {model_prob[band].mean():.3f} -> {actual[band].mean():.3f}"
                  f"   ({int(band.sum())} matches)")

    # Bootstrap confidence intervals (FR-ML-09, FR-PERF-02).
    rng = np.random.default_rng(7)
    per_match = -(
        actual * np.log(np.clip(model_prob, 1e-9, 1 - 1e-9))
        + (1 - actual) * np.log(np.clip(1 - model_prob, 1e-9, 1 - 1e-9))
    )
    base_per_match = -(
        actual * np.log(base_rate) + (1 - actual) * np.log(1 - base_rate)
    )
    advantage = base_per_match - per_match
    n = len(actual)
    draws = rng.integers(0, n, size=(4000, n))
    adv_boot = advantage[draws].mean(axis=1)
    hit_boot = picked[draws].mean(axis=1)
    print(f"\n  log-loss advantage : {advantage.mean():+.5f} "
          f"[{np.percentile(adv_boot, 2.5):+.5f}, {np.percentile(adv_boot, 97.5):+.5f}]")
    print(f"  hit rate           : {picked.mean():.1%} "
          f"[{np.percentile(hit_boot, 2.5):.1%}, {np.percentile(hit_boot, 97.5):.1%}]")

    confidence = np.abs(model_prob - 0.5)
    for lo, hi, label in ((0.00, 0.02, "toss-up   (<2pp)"),
                          (0.02, 0.05, "mild    (2-5pp)"),
                          (0.05, 1.00, "confident (>5pp)")):
        band = (confidence >= lo) & (confidence < hi)
        if band.sum():
            print(f"    {label:18}: {picked[band].mean():.1%}  ({int(band.sum())} matches)")


if __name__ == "__main__":
    asyncio.run(main())
