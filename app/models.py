"""SQLAlchemy models for the core schema (B4).
core holds canonical entities and match data -- the single source of truth
every other schema reads from (B1.1).
"""

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped,  mapped_column, relationship
from app.db import Base


class Competition(Base):
     __tablename__ = "competitions"
     __table_args__ = {"schema": "core"}

     competition_id: Mapped[int] = mapped_column(primary_key=True)
     name: Mapped[str] = mapped_column(String(120))
     country:Mapped[str] = mapped_column(String(80))
     tier: Mapped[int | None]

class Season(Base):
     __tablename__ = "seasons"
     __table_args__ = {"schema": "core"}

     season_id: Mapped[int] = mapped_column(primary_key=True)
     competition_id: Mapped[int] = mapped_column(
        ForeignKey("core.competitions.competition_id")
     )
     start_date: Mapped[date]
     end_date: Mapped[date]

class Team(Base):
     __tablename__ = "teams"
     __table_args__ = {"schema": "core"}

     team_id: Mapped[int] = mapped_column(primary_key=True)
     name: Mapped[str]= mapped_column(String(120))
     country: Mapped[str | None]= mapped_column(String(80))


class Venue(Base):
    __tablename__ = "venues"
    __table_args__={"schema": "core"}

    venue_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]= mapped_column(String(120))
    latitude: Mapped[float | None]
    longitude: Mapped[float | None]
    pitch_type: Mapped[str | None] = mapped_column(String(40))

class Referee(Base): 
    __tablename__= "referees"
    __table_args__={"schema": "core"}

    referee_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))

class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint(
            "season_id",
            "home_team_id",
            "away_team_id",
            "kickoff_utc",
            name="uq_matches_fixture",
        ),
        {"schema": "core"},   
    )

    match_id: Mapped[int] = mapped_column(primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("core.seasons.season_id"))
    kickoff_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("core.teams.team_id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("core.teams.team_id"))
    referee_id: Mapped[int | None] = mapped_column(ForeignKey("core.referees.referee_id"))
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("core.venues.venue_id"))
    status: Mapped[str] = mapped_column(String(20))
    home_goals: Mapped[int | None]
    away_goals: Mapped[int | None]
    home_team: Mapped["Team"] = relationship(
        foreign_keys=[home_team_id], lazy="selectin"
    )
    away_team: Mapped["Team"] = relationship(
        foreign_keys=[away_team_id], lazy="selectin"
    )


class EntityMap(Base):
    """Maps provider identifiers to canonical Platform ids (FR-DATA-07)."""

    __tablename__ = "entity_map"
    __table_args__ = {"schema": "core"}

    provider: Mapped[str] = mapped_column(String(40), primary_key=True)
    provider_entity_type: Mapped[str] = mapped_column(String(20), primary_key=True)
    provider_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    canonical_id: Mapped[int]


class RawPayloadRef(Base):
    """Lineage from a stored row back to the raw provider response (FR-DATA-06)."""

    __tablename__ = "raw_payload_refs"
    __table_args__ = {"schema": "core"}

    ref_id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(40))
    endpoint: Mapped[str] = mapped_column(String(200))
    content_hash: Mapped[str] = mapped_column(String(64))
    object_key: Mapped[str] = mapped_column(String(400))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RejectedPayload(Base):
    """Quarantine for payloads that failed schema validation (FR-DATA-05)."""

    __tablename__ = "rejected_payloads"
    __table_args__ = {"schema": "raw"}

    rejected_id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(40))
    endpoint: Mapped[str] = mapped_column(String(200))
    payload: Mapped[dict] = mapped_column(JSONB)
    validation_error: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
class TeamMatchStats(Base):
    """Per-team per-match aggregates (B4). One row per team, two per match."""

    __tablename__ = "team_match_stats"
    __table_args__ = {"schema": "core"}

    match_id: Mapped[int] = mapped_column(
        ForeignKey("core.matches.match_id"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(
        ForeignKey("core.teams.team_id"), primary_key=True
    )

    corners: Mapped[int | None]
    fouls: Mapped[int | None]
    yellow: Mapped[int | None]
    red: Mapped[int | None]
    shots: Mapped[int | None]
    shots_on_target: Mapped[int | None]
    offsides: Mapped[int | None]
    saves: Mapped[int | None]
    attacks: Mapped[int | None]
    dangerous_attacks: Mapped[int | None]
    passes: Mapped[int | None]
    passes_accurate: Mapped[int | None]
    possession: Mapped[float | None]
    crosses: Mapped[int | None]
    ppda: Mapped[float | None]


class Player(Base):
    __tablename__ = "players"
    __table_args__ = {"schema": "core"}

    player_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))


class PlayerMatchStats(Base):
    """Per-player per-match statistics (B4), for player props.

    Every listed player gets a row, unused substitutes included (minutes 0), so
    "in the squad but did not play" can be told apart from "not in the squad".
    None means the provider did not say. Shots follow the provider's definition,
    which EXCLUDES blocked shots (on target + off target only).
    """

    __tablename__ = "player_match_stats"
    __table_args__ = (
        Index("ix_player_match_stats_player", "player_id"),
        {"schema": "core"},
    )

    match_id: Mapped[int] = mapped_column(
        ForeignKey("core.matches.match_id"), primary_key=True
    )
    player_id: Mapped[int] = mapped_column(
        ForeignKey("core.players.player_id"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(ForeignKey("core.teams.team_id"))
    is_starter: Mapped[bool]
    position: Mapped[str | None] = mapped_column(String(30))
    minutes: Mapped[int | None]
    shots: Mapped[int | None]
    shots_on_target: Mapped[int | None]
    fouls_committed: Mapped[int | None]
    yellow: Mapped[int | None]
    red: Mapped[int | None]
    goals: Mapped[int | None]
    assists: Mapped[int | None]
    offsides: Mapped[int | None]
    tackles: Mapped[int | None]
    passes: Mapped[int | None]
    rating: Mapped[float | None]

class FeatureSnapshot(Base):
    """Point-in-time features for one entity before one match (FR-FEAT-08)."""

    __tablename__ = "feature_snapshots"
    __table_args__ = {"schema": "features"}

    match_id: Mapped[int] = mapped_column(
        ForeignKey("core.matches.match_id"), primary_key=True
    )
    entity_type: Mapped[str] = mapped_column(String(20), primary_key=True)
    entity_id: Mapped[int] = mapped_column(primary_key=True)
    feature_set_version: Mapped[str] = mapped_column(String(40), primary_key=True)
    computed_as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    features: Mapped[dict] = mapped_column(JSONB)


class Bookmaker(Base):
    """Bookmaker registry (B4)."""

    __tablename__ = "bookmakers"
    __table_args__ = {"schema": "core"}

    bookmaker_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)
    is_sharp: Mapped[bool] = mapped_column(default=False)


class OddsSnapshot(Base):
    """Append-only odds history (FR-DATA-02, NFR-DATA-01).

    A row is never updated. A changed price is a new row, so the price on offer
    at any moment can always be reconstructed.
    """

    __tablename__ = "odds_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "bookmaker_id", "match_id", "market", "entity_type", "entity_id",
            "line", "selection", "captured_at",
            name="uq_odds_snapshot",
            postgresql_nulls_not_distinct=True,
        ),
        Index("ix_odds_match_market_time", "match_id", "market", "captured_at"),
        {"schema": "core"},
    )

    snapshot_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    bookmaker_id: Mapped[int] = mapped_column(
        ForeignKey("core.bookmakers.bookmaker_id")
    )
    match_id: Mapped[int] = mapped_column(ForeignKey("core.matches.match_id"))
    market: Mapped[str] = mapped_column(String(30))
    entity_type: Mapped[str] = mapped_column(String(10), default="match")
    entity_id: Mapped[int | None]
    line: Mapped[float | None] = mapped_column(Numeric(5, 2, asdecimal=False))
    selection: Mapped[str] = mapped_column(String(10))
    decimal_price: Mapped[float] = mapped_column(Numeric(8, 3, asdecimal=False))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    provider_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class Notification(Base):
    """Something the operator should see, raised by the platform itself (B5).

    The scheduled runs happen unattended, so anything worth knowing -- a failed
    run, a pick that beats the market price, accuracy slipping -- is recorded here
    rather than only in a log file nobody opens.
    """

    __tablename__ = "notifications"
    __table_args__ = (
        Index("ix_notifications_unread", "read_at", "created_at"),
        {"schema": "core"},
    )

    notification_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    level: Mapped[str] = mapped_column(String(10))     # info, warning, error
    category: Mapped[str] = mapped_column(String(20))  # pipeline, value, accuracy
    title: Mapped[str] = mapped_column(String(200))
    detail: Mapped[str | None] = mapped_column(Text)
    context: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PredictionOutcome(Base):
    """What a forecast said before kickoff, and what actually happened (B7.5).

    Written after a match is played, from the last forecast made before its
    kickoff -- the one anybody acting on the model would have had. Every model's
    test season is now spent, so this is the only honest evidence left about how
    the live models are doing.
    """

    __tablename__ = "prediction_outcomes"
    __table_args__ = (
        UniqueConstraint(
            "match_id", "market", "entity_type", "entity_id", "line",
            name="uq_prediction_outcome",
            postgresql_nulls_not_distinct=True,
        ),
        Index("ix_prediction_outcomes_market_kickoff", "market", "kickoff_utc"),
        {"schema": "models"},
    )

    outcome_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("core.matches.match_id"))
    market: Mapped[str] = mapped_column(String(30))
    entity_type: Mapped[str] = mapped_column(String(10))
    entity_id: Mapped[int | None]
    line: Mapped[float] = mapped_column(Numeric(5, 2, asdecimal=False))
    model_version: Mapped[str] = mapped_column(
        ForeignKey("models.model_registry.model_version")
    )
    kickoff_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    hours_before_kickoff: Mapped[float]
    prob_over: Mapped[float]
    expected_value: Mapped[float | None]
    baseline_expected_value: Mapped[float | None]
    actual: Mapped[float]
    went_over: Mapped[bool]
    # The best price on each side shortly before kickoff, when one was collected.
    # Both sides are kept so profit can be worked out later under any staking rule.
    over_price: Mapped[float | None] = mapped_column(Numeric(8, 3, asdecimal=False))
    under_price: Mapped[float | None] = mapped_column(Numeric(8, 3, asdecimal=False))
    priced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    settled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelRegistry(Base):
    """One row per trained model (B4, FR-ML-06)."""

    __tablename__ = "model_registry"
    __table_args__ = {"schema": "models"}

    model_version: Mapped[str] = mapped_column(String(60), primary_key=True)
    market: Mapped[str] = mapped_column(String(30))
    algorithm: Mapped[str] = mapped_column(String(60))
    feature_set_version: Mapped[str] = mapped_column(String(40))
    train_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    train_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    params: Mapped[dict] = mapped_column(JSONB)
    validation_metrics: Mapped[dict] = mapped_column(JSONB)
    calibration: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Distribution(Base):
    """Full probability of every count for one forecast (B4, FR-PRED-01)."""

    __tablename__ = "distributions"
    __table_args__ = (
        UniqueConstraint(
            "match_id", "market", "entity_type", "entity_id", "model_version", "computed_at",
            name="uq_distribution",
            postgresql_nulls_not_distinct=True,
        ),
        {"schema": "models"},
    )

    distribution_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("core.matches.match_id"))
    market: Mapped[str] = mapped_column(String(30))
    entity_type: Mapped[str] = mapped_column(String(10))
    entity_id: Mapped[int | None]
    model_version: Mapped[str] = mapped_column(
        ForeignKey("models.model_registry.model_version")
    )
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expected_value: Mapped[float]
    # What a naive forecast would expect: for a player prop, his own rate per 90
    # times his usual minutes. Picks are ranked by how far the model departs from
    # it, the player-level equivalent of a league base rate. Null for match markets.
    baseline_expected_value: Mapped[float | None]
    pmf: Mapped[dict] = mapped_column(JSONB)


class Prediction(Base):
    """One priced line of one forecast (B4, FR-PRED-02)."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "match_id", "market", "entity_type", "entity_id", "line",
            "model_version", "computed_at",
            name="uq_prediction",
            postgresql_nulls_not_distinct=True,
        ),
        Index("ix_predictions_match_market_time", "match_id", "market", "computed_at"),
        {"schema": "models"},
    )

    prediction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("core.matches.match_id"))
    market: Mapped[str] = mapped_column(String(30))
    entity_type: Mapped[str] = mapped_column(String(10))
    entity_id: Mapped[int | None]
    line: Mapped[float] = mapped_column(Numeric(5, 2, asdecimal=False))
    model_version: Mapped[str] = mapped_column(
        ForeignKey("models.model_registry.model_version")
    )
    prob_over: Mapped[float]
    prob_under: Mapped[float]
    fair_odds_over: Mapped[float]
    fair_odds_under: Mapped[float]
    best_odds: Mapped[float | None]
    best_bookmaker_id: Mapped[int | None] = mapped_column(
        ForeignKey("core.bookmakers.bookmaker_id")
    )
    implied_prob: Mapped[float | None]
    edge: Mapped[float | None]
    flagged: Mapped[bool] = mapped_column(default=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
