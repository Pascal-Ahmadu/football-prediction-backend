"""Runtime configuration.

Every secret and environment-specific value is read from the environment, never
hardcoded and never shipped to the frontend (NFR-SEC-02).
"""

from functools import  lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    feature_half_life: float = 6.0
    referee_half_life: float = 10.0
    feature_set_version: str = "rates-v5"
    rating_lambda: float = 5.0
    rating_min_rows: int = 300
    player_half_life: float = 10.0
    player_feature_set_version: str = "players-v2"


    environment: str = "local"
    debug: bool = False

    database_url: str = Field(
        default="postgresql+psycopg://postgres:devpass@localhost:55432/fmep",
        description="SQLAlchemy URL. Use the psycopg (v3) driver",
    )
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    # Set these on a deployment and the API is closed to everyone else. Left
    # empty (the default) the API is open, which is what a local machine wants.
    api_keys: list[str] = []
    rate_limit_per_minute: int = 120

    redis_url: str = "redis://localhost:6380/0"
    cors_origins: list[str] = ["http://localhost:3000"]
    api_football_key: str = ""
    api_football_base_url: str = "https://apiv3.apifootball.com/"
    league_ids: list[str] = [
        # top tiers
        "152",  # England Premier League
        "302",  # Spain La Liga
        "207",  # Italy Serie A
        "175",  # Germany Bundesliga
        "168",  # France Ligue 1
        "244",  # Netherlands Eredivisie
        "266",  # Portugal Primeira Liga
        "63",   # Belgium First Division A
        "322",  # Turkey Super Lig
        "279",  # Scotland Premiership
        "178",  # Greece Super League
        "56",   # Austria Bundesliga
        # second tiers
        "153",  # England Championship
        "301",  # Spain Segunda Division
        "206",  # Italy Serie B
        "171",  # Germany 2. Bundesliga
        "164",  # France Ligue 2
    ]
    api_football_timezone: str = "Europe/Berlin"

    # The Odds API (the-odds-api.com): market prices for the markets we predict.
    # The free plan allows 500 credits a month and one credit buys one market for
    # one fixture in one region, so fixtures are sampled, never swept.
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4/"
    # One credit buys one market in ONE region, so each extra region doubles the
    # cost of a fixture. "uk" alone covers the bookmakers quoting corners and cards.
    odds_api_regions: str = "uk"
    odds_sample_size: int = 20  # fixtures priced per run
    odds_sample_days: int = 4   # how far ahead to price

    odds_days_ahead: int = 7
    odds_near_kickoff_hours: float = 6.0
    odds_near_interval_seconds: int = 300
    odds_far_interval_seconds: int = 3600

    @property
    def async_database_url(self) -> str:
        """psycopg 3 speaks both sync and async over the same URL scheme."""
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process."""
    return Settings()


settings = get_settings()
