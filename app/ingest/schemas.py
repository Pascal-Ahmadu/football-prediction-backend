"""Pydanctic schemas for apifootball payloads """


from datetime import  UTC, datetime, date, time
from typing import Any
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
from app.config import settings


class StatEntry(BaseModel):
    type: str
    home: str = ""
    away: str = ""


class CardEvent(BaseModel):
    """One booking. Exactly one of home_fault / away_fault names the player."""

    card: str = ""
    home_fault: str = ""
    away_fault: str = ""
    time: str = ""

class ProviderEvent(BaseModel):
    """ One match, exactly as apifootball returns it"""

    match_id: str
    league_id: str
    league_name: str
    country_name: str
    match_date: date
    match_time: str
    match_status: str
    match_hometeam_id: str
    match_hometeam_name: str
    match_awayteam_id: str
    match_awayteam_name: str
    match_hometeam_score: str =""
    match_awayteam_score: str =""
    match_referee: str=""
    match_stadium: str=""
    statistics: list[StatEntry] = Field(default_factory=list)
    cards: list[CardEvent] = Field(default_factory=list)
    # Only present when requested with withPlayerStats=1: {"home": [...], "away": [...]}.
    # The provider sends an empty list instead of an empty object when it has none.
    player_stats: dict[str, list[dict[str, Any]]] | list = Field(default_factory=dict)

    def players(self, side: str) -> list[dict[str, Any]]:
        """Player stat rows for 'home' or 'away'; empty when the provider has none."""
        if not isinstance(self.player_stats, dict):
            return []
        return self.player_stats.get(side, [])

    @property
    def kickoff_utc(self) -> datetime:
        """Kickoff as an aware UTC datetime.
        apifootball does not state a timezone for match_time.
        settings.api_football_timezone records our assumption --verify it again a know kickoff before trusting any backtest
        """
        naive = datetime.combine(self.match_date, time.fromisoformat(self.match_time))
        local = naive.replace(tzinfo=ZoneInfo(settings.api_football_timezone))
        return local.astimezone(UTC)

    @property
    def is_finished(self) -> bool:
        return self.match_status.strip().lower() == 'finished'

    def score(self, raw: str) -> int | None:
        return int(raw) if raw.strip().isdigit() else None
    

    @property
    def home_goals(self) -> int | None:
        return self.score(self.match_hometeam_score)

    @property
    def away_goals(self) -> int | None:
        return self.score(self.match_awayteam_score)

    def stat(self, name: str) -> tuple[int, int] | None:
        """First occurrence of a statistic, as (home, away) integers."""
        for entry in self.statistics:
            if entry.type.strip().lower() == name.strip().lower():
                if entry.home.strip().isdigit() and entry.away.strip().isdigit():
                    return int(entry.home), int(entry.away)
                return None
        return None
    
    def card_count(self, stat_name: str, label: str) -> tuple[int, int] | None:
        """Cards per side: the statistic if present, otherwise the booking events.

        apifootball omits a card statistic when neither side was booked, but
        occasionally also when bookings did happen. The booking events are an
        independent record, so they settle which it was. If both are missing
        while fouls were recorded, the match genuinely had no such cards.
        Returns None when no source can say.
        """
        pair = self.stat(stat_name)
        if pair is not None:
            return pair
        if self.cards:
            home = sum(1 for c in self.cards if c.card == label and c.home_fault.strip())
            away = sum(1 for c in self.cards if c.card == label and c.away_fault.strip())
            return home, away
        if self.stat("Fouls") is not None or self.stat("Yellow Cards") is not None:
            return 0, 0
        return None

    def percentage(self, name: str) -> tuple[float, float] | None:
        """A percentage statistic such as '45%' as (home, away) floats."""
        for entry in self.statistics:
            if entry.type.strip().lower() == name.strip().lower():
                try:
                    return (
                        float(entry.home.strip().rstrip("%")),
                        float(entry.away.strip().rstrip("%")),
                    )
                except ValueError:
                    return None
        return None

