"""
Thes define the shape of every JSON payload the service returns, and they generates the openAPI document the frontend's Typescript types
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class TeamSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    team_id: int
    name: str

class FixtureSummary(BaseModel):
        model_config = ConfigDict(from_attributes=True)

        match_id: int
        kickoff_utc: datetime
        status: str
        home_team: TeamSummary
        away_team: TeamSummary


class LinePrediction(BaseModel):
    """One priced line: the model's probability either side of it (FR-PRED-02)."""

    model_config = ConfigDict(from_attributes=True)

    line: float
    prob_over: float
    prob_under: float
    fair_odds_over: float
    fair_odds_under: float


class MarketPrediction(BaseModel):
    """Every line of one market from one forecast run (B5.2)."""

    market: str
    entity_type: str
    entity_id: int | None
    model_version: str
    computed_at: datetime
    expected_value: float | None
    lines: list[LinePrediction]


class FixturePredictions(BaseModel):
    """The latest forecast for one fixture, across all markets."""

    match_id: int
    kickoff_utc: datetime
    home_team: TeamSummary
    away_team: TeamSummary
    markets: list[MarketPrediction]


class CountProbability(BaseModel):
    count: int
    probability: float


class TailProbability(BaseModel):
    """Everything from `at_least` upwards, lumped together."""

    at_least: int
    probability: float


class MarketDistribution(BaseModel):
    """The chance of every count for one market from one forecast run (FR-PRED-01)."""

    market: str
    entity_type: str
    entity_id: int | None
    model_version: str
    computed_at: datetime
    expected_value: float
    probabilities: list[CountProbability]
    tail: TailProbability


class FixtureDistributions(BaseModel):
    match_id: int
    kickoff_utc: datetime
    home_team: TeamSummary
    away_team: TeamSummary
    markets: list[MarketDistribution]


class Pick(BaseModel):
    """One confident forecast (FR-WEB-05), with model confidence in place of edge."""

    match_id: int
    kickoff_utc: datetime
    competition: str
    country: str
    home_team: TeamSummary
    away_team: TeamSummary
    market: str
    entity_type: str = "match"
    entity_id: int | None = None
    entity_name: str | None = None  # the player, for player markets
    line: float
    selection: Literal["over", "under"]
    probability: float
    confidence: float
    base_rate: float | None
    edge_over_base: float | None
    expected_value: float | None
    model_version: str
    computed_at: datetime


class MarketPerformance(BaseModel):
    """Live accuracy of one market at one line, from settled forecasts (B7.5)."""

    market: str
    line: float
    settled: int
    hit_rate: float
    log_loss: float
    baseline_log_loss: float
    over_rate: float
    mean_prob: float
    mean_abs_error: float | None
    first_kickoff: datetime
    last_kickoff: datetime
    hours_before: float


class AccumulatorLeg(BaseModel):
    """One selection of an accumulator, at the price on offer."""

    match_id: int
    kickoff_utc: datetime
    fixture: str
    competition: str
    market: str
    line: float
    selection: Literal["over", "under"]
    probability: float
    price: float
    edge: float


class Accumulator(BaseModel):
    """A combination and its true chance (FR-WEB-05).

    expected_return is per unit staked: above 1.0 is value, below it is not,
    however likely the individual legs look.
    """

    legs: list[AccumulatorLeg]
    combined_odds: float
    probability: float
    expected_return: float
    note: str | None = None


class NotificationRead(BaseModel):
    """One thing the platform wants the operator to see."""

    model_config = ConfigDict(from_attributes=True)

    notification_id: int
    level: Literal["info", "warning", "error"]
    category: str
    title: str
    detail: str | None
    context: dict | None
    created_at: datetime
    read_at: datetime | None


class NotificationSummary(BaseModel):
    unread: int
    items: list[NotificationRead]
