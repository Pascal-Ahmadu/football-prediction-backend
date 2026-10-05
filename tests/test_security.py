"""API keys and rate limiting."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.security import reset_rate_limits

KEY = "test-key-0123456789"


@pytest.fixture
def client():
    # raise_server_exceptions=False: with no database behind it the endpoint
    # raises, and the default client re-raises that instead of returning a
    # response. These tests are about the door, not the room, so a 500 is a
    # perfectly good "the request got past the guard".
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clean_settings():
    keys, limit = settings.api_keys, settings.rate_limit_per_minute
    reset_rate_limits()
    yield
    settings.api_keys, settings.rate_limit_per_minute = keys, limit
    reset_rate_limits()


class TestApiKeys:
    """Refusals are checked without a database: the key is verified before the
    endpoint runs, so a 401 needs nothing behind it. The cases that must reach
    the endpoint are marked `db`.
    """

    @pytest.mark.db
    def test_open_when_no_keys_are_configured(self, client):
        """A laptop with no keys set stays usable."""
        settings.api_keys = []
        assert client.get("/api/v1/picks", params={"limit": 1}).status_code == 200

    def test_closed_once_keys_are_configured(self, client):
        settings.api_keys = [KEY]
        response = client.get("/api/v1/picks", params={"limit": 1})
        assert response.status_code == 401
        assert response.headers["content-type"].startswith("application/problem+json")
        assert response.headers["WWW-Authenticate"] == "X-API-Key"

    def test_a_wrong_key_is_refused(self, client):
        settings.api_keys = [KEY]
        assert client.get(
            "/api/v1/picks", headers={"X-API-Key": "wrong"}, params={"limit": 1}
        ).status_code == 401

    @pytest.mark.db
    def test_the_right_key_is_accepted(self, client):
        settings.api_keys = [KEY]
        assert client.get(
            "/api/v1/picks", headers={"X-API-Key": KEY}, params={"limit": 1}
        ).status_code == 200

    @pytest.mark.db
    def test_several_keys_can_be_valid_at_once(self, client):
        settings.api_keys = ["first-key", KEY]
        for key in settings.api_keys:
            assert client.get(
                "/api/v1/picks", headers={"X-API-Key": key}, params={"limit": 1}
            ).status_code == 200

    def test_health_stays_open_for_monitoring(self, client):
        settings.api_keys = [KEY]
        # 200 when the database answers, 503 when it does not -- either way the
        # endpoint is reachable without a key, which is the point.
        assert client.get("/health").status_code in (200, 503)

    @pytest.mark.parametrize(
        "path",
        ["/api/v1/picks", "/api/v1/performance", "/api/v1/notifications",
         "/api/v1/accumulator", "/api/v1/fixtures"],
    )
    def test_every_data_endpoint_is_guarded(self, client, path):
        settings.api_keys = [KEY]
        assert client.get(path).status_code == 401, f"{path} is unprotected"

    def test_marking_notifications_read_is_guarded(self, client):
        settings.api_keys = [KEY]
        assert client.post("/api/v1/notifications/read", json={}).status_code == 401


class TestRateLimit:
    """Counting happens before the endpoint, so these need no database either."""

    def test_requests_are_refused_past_the_limit(self, client):
        settings.api_keys = []
        settings.rate_limit_per_minute = 3
        codes = [client.get("/api/v1/picks", params={"limit": 1}).status_code for _ in range(5)]
        assert 429 not in codes[:3], codes
        assert codes[3:] == [429, 429], codes

    def test_a_refusal_says_when_to_retry(self, client):
        settings.api_keys = []
        settings.rate_limit_per_minute = 1
        client.get("/api/v1/picks", params={"limit": 1})
        response = client.get("/api/v1/picks", params={"limit": 1})
        assert response.status_code == 429
        assert 0 < int(response.headers["Retry-After"]) <= 61
        assert response.headers["content-type"].startswith("application/problem+json")

    def test_zero_means_no_limit(self, client):
        settings.api_keys = []
        settings.rate_limit_per_minute = 0
        codes = [client.get("/api/v1/picks", params={"limit": 1}).status_code for _ in range(4)]
        assert 429 not in codes, codes

    def test_callers_are_counted_separately(self, client):
        settings.api_keys = ["key-one", "key-two"]
        settings.rate_limit_per_minute = 2
        for _ in range(2):
            assert client.get("/api/v1/picks", headers={"X-API-Key": "key-one"}).status_code != 429
        assert client.get("/api/v1/picks", headers={"X-API-Key": "key-one"}).status_code == 429
        assert client.get("/api/v1/picks", headers={"X-API-Key": "key-two"}).status_code != 429
