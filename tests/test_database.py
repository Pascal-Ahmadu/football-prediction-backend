"""Checks against the real database: the API, and the point-in-time rule.

Read-only. Skipped when Postgres is not running.
"""

import asyncio
import random

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import settings
from app.db import SessionLocal, dispose_engine
from app.features.rates import decay_factor
from app.main import app

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestApi:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_upcoming_fixtures(self, client):
        response = client.get("/api/v1/fixtures")
        assert response.status_code == 200

    def test_unknown_fixture_is_a_problem_document(self, client):
        response = client.get("/api/v1/fixtures/0/predictions")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/problem+json")
        assert response.json()["status"] == 404

    def test_fixture_forecast_has_every_market_and_consistent_distributions(self, client):
        """Markets are forecast in separate runs; the newest run must not hide the others."""
        match_id = client.get("/api/v1/fixtures", params={"limit": 1}).json()[0]["match_id"]
        predictions = client.get(f"/api/v1/fixtures/{match_id}/predictions").json()
        markets = {m["market"]: m for m in predictions["markets"]}
        assert {"total_corners", "total_cards", "total_fouls"} <= set(markets)

        distributions = client.get(f"/api/v1/fixtures/{match_id}/distributions").json()
        assert {d["market"] for d in distributions["markets"]} == set(markets)
        for dist in distributions["markets"]:
            total = sum(p["probability"] for p in dist["probabilities"]) + dist["tail"]["probability"]
            assert total == pytest.approx(1.0, abs=1e-3)
            for line in markets[dist["market"]]["lines"]:
                over = dist["tail"]["probability"] + sum(
                    p["probability"] for p in dist["probabilities"] if p["count"] > line["line"]
                )
                assert over == pytest.approx(line["prob_over"], abs=1e-3)

    def test_unknown_market_is_rejected(self, client):
        response = client.get("/api/v1/picks", params={"market": "total_goals"})
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/problem+json")

    @pytest.mark.parametrize(
        "market",
        ["total_corners", "total_cards", "total_fouls",
         "player_shots_on_target", "player_fouls_committed"],
    )
    def test_picks_respect_confidence(self, client, market):
        response = client.get(
            "/api/v1/picks", params={"market": market, "min_confidence": 0.1, "limit": 200}
        )
        assert response.status_code == 200
        for pick in response.json():
            assert pick["confidence"] >= 0.1 - 1e-9
            assert pick["selection"] in ("over", "under")

    @pytest.mark.parametrize("market", ["total_corners", "player_fouls_committed"])
    def test_picks_back_the_side_the_model_favours_over_the_base_rate(self, client, market):
        """The pick is the disagreement, not simply whichever side is likelier."""
        picks = client.get("/api/v1/picks", params={"market": market, "limit": 50}).json()
        assert picks
        for pick in picks:
            if pick["base_rate"] is None:
                continue
            probability_over = (
                pick["probability"] if pick["selection"] == "over" else 1 - pick["probability"]
            )
            if pick["selection"] == "over":
                assert probability_over >= pick["base_rate"]
            else:
                assert probability_over <= pick["base_rate"]


LEAK_CHECK = text("""
    with sides as (
        select m.match_id, m.kickoff_utc, s.team_id, s.corners, s.fouls
        from core.matches m
        join core.team_match_stats s on s.match_id = m.match_id
        where m.status = 'Finished'
          and (select count(*) from core.team_match_stats x where x.match_id = m.match_id) = 2
    )
    select corners, fouls from sides
    where team_id = :team_id
      and (kickoff_utc, match_id) < (:kickoff, :match_id)
    order by kickoff_utc, match_id
""")


def test_team_snapshots_use_only_earlier_matches():
    """Recompute sampled team snapshots from strictly earlier matches (risk R-02)."""
    lam = decay_factor(settings.feature_half_life)

    async def check() -> int:
        try:
            async with SessionLocal() as session:
                snapshots = (
                    await session.execute(
                        text("""
                            select f.match_id, f.entity_id, f.features, m.kickoff_utc
                            from features.feature_snapshots f
                            join core.matches m on m.match_id = f.match_id
                            where f.feature_set_version = :version
                              and f.entity_type = 'team' and m.status = 'Finished'
                        """),
                        {"version": settings.feature_set_version},
                    )
                ).all()
                assert snapshots, f"no snapshots for {settings.feature_set_version}"
                sample = random.Random(7).sample(snapshots, 25)

                for snap in sample:
                    earlier = (
                        await session.execute(
                            LEAK_CHECK,
                            {"team_id": snap.entity_id, "kickoff": snap.kickoff_utc, "match_id": snap.match_id},
                        )
                    ).all()
                    for column in ("corners", "fouls"):
                        expected = None
                        for row in earlier:
                            value = getattr(row, column)
                            if value is not None:
                                expected = float(value) if expected is None else (1 - lam) * value + lam * expected
                        stored = snap.features[f"{column}_for"]
                        if expected is None:
                            assert stored is None
                        else:
                            assert stored == pytest.approx(expected, abs=1e-9), (snap.match_id, column)
                return len(sample)
        finally:
            await dispose_engine()

    assert asyncio.run(check()) == 25


class TestPerformance:
    def test_endpoint_returns_settled_markets(self, client):
        response = client.get("/api/v1/performance", params={"days": 3650})
        assert response.status_code == 200
        for row in response.json():
            assert 0.0 <= row["hit_rate"] <= 1.0
            assert row["settled"] > 0
            assert row["hours_before"] > 0, "a forecast must predate its kickoff"

    def test_player_picks_list_each_player_once(self, client):
        picks = client.get(
            "/api/v1/picks", params={"market": "player_fouls_committed", "limit": 50}
        ).json()
        names = [p["entity_name"] for p in picks]
        assert all(names), "player picks must name the player"
        assert len(names) == len(set(names))


class TestAccumulator:
    def test_one_leg_per_match_and_consistent_arithmetic(self, client):
        data = client.get("/api/v1/accumulator", params={"legs": 4}).json()
        if not data["legs"]:
            # Legitimate state: no upcoming fixture has a price yet.
            assert data["expected_return"] == 0.0 and data["note"]
            return
        matches = [leg["match_id"] for leg in data["legs"]]
        assert len(matches) == len(set(matches)), "same match twice overstates the odds"
        odds = 1.0
        probability = 1.0
        for leg in data["legs"]:
            odds *= leg["price"]
            probability *= leg["probability"]
        assert data["combined_odds"] == pytest.approx(odds)
        assert data["probability"] == pytest.approx(probability)
        assert data["expected_return"] == pytest.approx(probability * odds)

    def test_unreachable_target_is_said_plainly(self, client):
        data = client.get(
            "/api/v1/accumulator", params={"target_odds": 100000, "legs": 20}
        ).json()
        assert data["combined_odds"] < 100000
        assert data["note"], "an unreachable target must be explained, not implied"
        assert "would need more" in data["note"] or "nothing to back" in data["note"]

    def test_demanding_value_never_returns_a_worse_leg(self, client):
        strict = client.get("/api/v1/accumulator", params={"legs": 5, "min_edge": 0.05}).json()
        for leg in strict["legs"]:
            assert leg["edge"] >= 0.05 - 1e-9


class TestNotifications:
    def test_raised_notifications_are_listed_and_can_be_read(self, client):
        async def raise_one() -> int:
            try:
                async with SessionLocal() as session:
                    from app.notifications import notify

                    item = await notify(
                        session, "warning", "pipeline", "Test notification",
                        detail="Raised by the test suite.", context={"test": True},
                    )
                    return item.notification_id
            finally:
                await dispose_engine()

        notification_id = asyncio.run(raise_one())

        listed = client.get("/api/v1/notifications", params={"limit": 50}).json()
        assert listed["unread"] >= 1
        mine = [n for n in listed["items"] if n["notification_id"] == notification_id]
        assert mine and mine[0]["level"] == "warning" and mine[0]["read_at"] is None

        after = client.post("/api/v1/notifications/read",
                            json={"notification_ids": [notification_id]}).json()
        read = [n for n in after["items"] if n["notification_id"] == notification_id]
        assert read and read[0]["read_at"] is not None

    def test_unread_filter_hides_read_ones(self, client):
        unread = client.get("/api/v1/notifications", params={"unread_only": True}).json()
        assert all(item["read_at"] is None for item in unread["items"])
