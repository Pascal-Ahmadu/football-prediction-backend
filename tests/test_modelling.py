"""Model maths, training rows and pick ranking."""

import asyncio
import math
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import poisson

from app.api.picks import _base_rate
from app.evaluation.track import _baseline_log_loss
from app.modelling import dataset
from app.modelling.corners import dispersion, expected_calibration_error, log_loss, prob_over
from app.modelling.predict import FORECASTS
from app.pipeline.weekly import feed_verdict, month_windows
from app.pricing.devig import devig, implied_probability, margin


class TestDistribution:
    def test_dispersion_recovers_overdispersion(self):
        rng = np.random.default_rng(0)
        r, mu = 10.0, 20.0
        y = rng.negative_binomial(r, r / (r + mu), size=200_000)
        assert dispersion(y, np.full(len(y), mu)) == pytest.approx(r, rel=0.1)

    def test_dispersion_of_underdispersed_data_is_effectively_poisson(self):
        y = np.full(1000, 20.0)
        assert dispersion(y, y) == 1e6

    def test_huge_dispersion_matches_poisson(self):
        mu = np.array([9.0, 24.0])
        assert prob_over(mu, 1e6, 9.5) == pytest.approx(poisson.sf(9, mu), abs=1e-4)

    def test_over_a_half_line_means_strictly_more(self):
        # P(over 9.5) = P(X >= 10) = P(X > 9)
        assert prob_over(np.array([10.0]), 1e6, 9.5)[0] == pytest.approx(poisson.sf(9, 10.0), abs=1e-4)


class TestScores:
    def test_log_loss_of_a_coin_flip(self):
        assert log_loss(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(math.log(2))

    def test_calibrated_forecasts_have_small_ece(self):
        rng = np.random.default_rng(1)
        prob = rng.uniform(0.2, 0.8, size=50_000)
        actual = (rng.uniform(size=prob.size) < prob).astype(float)
        assert expected_calibration_error(actual, prob) < 0.01

    def test_miscalibrated_forecasts_have_large_ece(self):
        rng = np.random.default_rng(1)
        prob = rng.uniform(0.2, 0.8, size=50_000)
        actual = (rng.uniform(size=prob.size) < prob - 0.15).astype(float)
        assert expected_calibration_error(actual, prob) > 0.1


class TestDevig:
    def test_implied_probability(self):
        assert implied_probability(2.0) == 0.5

    def test_rejects_impossible_price(self):
        with pytest.raises(ValueError):
            implied_probability(1.0)

    def test_fair_probabilities_sum_to_one(self):
        assert margin([1.9, 1.9]) == pytest.approx(2 / 1.9 - 1)
        assert devig([1.9, 1.9]) == pytest.approx([0.5, 0.5])
        assert sum(devig([2.5, 3.2, 3.0])) == pytest.approx(1.0)


class TestDatasetRows:
    def test_vector_matches_feature_names(self):
        for market in dataset.MARKETS.values():
            home = {n: 1.0 for n in market.team_features}
            away = {n: 2.0 for n in market.team_features}
            row = dataset._vector(market, home, away, None, competition_id=42)
            names = dataset.feature_names(market)
            assert len(row) == len(names)
            assert row[-1] == 42.0 and names[-1] == "competition_id"
            referee_slots = [row[i] for i, n in enumerate(names) if n.startswith("ref_")]
            assert all(math.isnan(v) for v in referee_slots), "unknown referee must be NaN, not 0"

    def test_missing_team_feature_is_nan(self):
        market = dataset.MARKETS["total_corners"]
        row = dataset._vector(market, {}, {}, None, competition_id=1)
        assert math.isnan(row[0])

    def test_every_forecast_has_a_market(self):
        assert set(FORECASTS) <= set(dataset.MARKETS)
        for spec in FORECASTS.values():
            assert spec.reference_line in spec.lines

    def test_load_builds_targets_and_skips_bad_rows(self):
        market = dataset.MARKETS["total_fouls"]
        snapshot_features = {n: 1.0 for n in market.team_features}
        snapshots = [
            SimpleNamespace(match_id=m, entity_type="team", entity_id=t, features=snapshot_features)
            for m in (1, 2, 3)
            for t in (10, 20)
        ] + [
            SimpleNamespace(
                match_id=1, entity_type="referee", entity_id=5,
                features={"ref_matches_seen": 3, "ref_fouls_pm": 25.0},
            )
        ]

        def match(match_id, h_fouls, a_fouls, referee_id=None):
            return SimpleNamespace(
                match_id=match_id, kickoff_utc=datetime(2026, 1, 1),
                season_start=SimpleNamespace(year=2025), competition_id=1,
                home_team_id=10, away_team_id=20, referee_id=referee_id,
                h_fouls=h_fouls, a_fouls=a_fouls,
                h_corners=None, a_corners=None, h_yellow=None, a_yellow=None, h_red=None, a_red=None,
            )

        matches = [
            match(1, 12, 11, referee_id=5),  # kept, referee known
            match(2, 1, 13),                 # partial statistics: skipped
            match(3, None, 13),              # missing: skipped
            match(4, 12, 12),                # no snapshots: skipped
        ]

        class Result(list):
            def all(self):
                return list(self)

        class FakeSession:
            async def execute(self, statement, params=None):
                return Result(snapshots if statement is dataset.SNAPSHOTS else matches)

        X, y, seasons, _kickoffs, names = asyncio.run(dataset.load(FakeSession(), "total_fouls"))
        assert y.tolist() == [23.0]
        assert seasons.tolist() == [2025]
        assert X[0, names.index("ref_fouls_pm")] == 25.0


class TestPicks:
    METRICS = {
        "base_rates": {"9.5": 0.49},
        "base_rates_by_competition": {"3": {"9.5": 0.36}},
    }

    def test_league_rate_is_preferred(self):
        assert _base_rate(self.METRICS, 3, "9.5", "league") == 0.36

    def test_falls_back_to_global_rate(self):
        assert _base_rate(self.METRICS, 99, "9.5", "league") == 0.49
        assert _base_rate(self.METRICS, 3, "9.5", "global") == 0.49


class TestMonthWindows:
    def test_splits_at_month_boundaries(self):
        assert month_windows(date(2026, 1, 20), date(2026, 3, 5)) == [
            (date(2026, 1, 20), date(2026, 1, 31)),
            (date(2026, 2, 1), date(2026, 2, 28)),
            (date(2026, 3, 1), date(2026, 3, 5)),
        ]

    def test_single_day(self):
        assert month_windows(date(2026, 9, 17), date(2026, 9, 17)) == [
            (date(2026, 9, 17), date(2026, 9, 17))
        ]


class TestLiveTracking:
    def test_baseline_log_loss_of_a_coin_flip(self):
        assert _baseline_log_loss(0.5) == pytest.approx(math.log(2))

    def test_baseline_log_loss_of_a_one_sided_market(self):
        # a market that always goes over is trivially predictable
        assert _baseline_log_loss(0.99) < 0.1

    def test_extremes_do_not_blow_up(self):
        assert _baseline_log_loss(0.0) == pytest.approx(0.0, abs=1e-6)
        assert _baseline_log_loss(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_profit_needs_a_price_better_than_the_model_says(self):
        """A 60% chance is only a bet above 1.67; below it, the price is against you."""
        assert 0.60 * 1.80 - 1 > 0      # value
        assert 0.60 * 1.60 - 1 < 0      # not value, however likely the outcome


class TestFeedHealth:
    """A lapsed subscription answers politely with nothing; that must not look fine."""

    def ok(self, league_id, events=12):
        return {"league_id": league_id, "events": events, "created": 2, "updated": 10,
                "rejected": 0, "error": None}

    def test_healthy_feed_passes(self):
        assert feed_verdict([self.ok("152"), self.ok("302")]) is None

    def test_a_quiet_league_is_not_a_broken_feed(self):
        assert feed_verdict([self.ok("152"), self.ok("302", events=0)]) is None

    def test_no_events_anywhere_is_broken(self):
        verdict = feed_verdict([self.ok("152", events=0), self.ok("302", events=0)])
        assert verdict and "no matches at all" in verdict

    def test_every_league_failing_is_broken(self):
        failed = [{"league_id": "152", "events": 0, "created": 0, "updated": 0,
                   "rejected": 0, "error": "timeout"}]
        assert feed_verdict(failed) and "every request failed" in feed_verdict(failed)

    def test_subscription_refusals_are_called_out(self):
        results = [
            self.ok("152"),
            {"league_id": "302", "events": 0, "created": 0, "updated": 0, "rejected": 0,
             "error": "get_events: No event found (please check your plan)!"},
        ]
        verdict = feed_verdict(results)
        assert verdict and "subscription" in verdict and "302" in verdict
