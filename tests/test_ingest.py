"""Parsing apifootball payloads and naming referees."""

from datetime import UTC, date, datetime

import pytest

from app.ingest.fixtures import _count, _rating, competition_name, fold_accents, referee_base_name, referee_key
from app.ingest.football_data import ALIASES, DIVISIONS, current_season_code, parse_kickoff
from app.ingest.market_odds import MIN_AVERAGE, pair_score, similarity
from app.ingest.merge_referees import Components, _is_abbreviated
from app.ingest.schemas import ProviderEvent


def event(**overrides) -> ProviderEvent:
    payload = {
        "match_id": "1",
        "league_id": "152",
        "league_name": "Premier League",
        "country_name": "England",
        "match_date": "2026-09-19",
        "match_time": "15:30",
        "match_status": "Finished",
        "match_hometeam_id": "10",
        "match_hometeam_name": "Home",
        "match_awayteam_id": "20",
        "match_awayteam_name": "Away",
    }
    payload.update(overrides)
    return ProviderEvent.model_validate(payload)


def stats(*entries: tuple[str, str, str]) -> list[dict]:
    return [{"type": t, "home": h, "away": a} for t, h, a in entries]


def booking(card: str, side: str) -> dict:
    return {"card": card, f"{side}_fault": "Some Player"}


class TestKickoff:
    def test_summer_time_is_two_hours_ahead_of_utc(self):
        assert event().kickoff_utc == datetime(2026, 9, 19, 13, 30, tzinfo=UTC)

    def test_winter_time_is_one_hour_ahead_of_utc(self):
        assert event(match_date="2026-01-10").kickoff_utc == datetime(2026, 1, 10, 14, 30, tzinfo=UTC)


class TestScoresAndStats:
    def test_blank_score_is_unknown_not_zero(self):
        ev = event(match_hometeam_score="", match_awayteam_score="2")
        assert ev.home_goals is None
        assert ev.away_goals == 2

    def test_stat_is_case_insensitive(self):
        assert event(statistics=stats(("Corners", "5", "3"))).stat("corners") == (5, 3)

    def test_non_numeric_stat_is_unknown(self):
        assert event(statistics=stats(("Corners", "", "3"))).stat("Corners") is None

    def test_percentage(self):
        ev = event(statistics=stats(("Ball Possession", "61%", "39%")))
        assert ev.percentage("Ball Possession") == (61.0, 39.0)


class TestCardCount:
    """The card statistic is sometimes missing even when bookings happened."""

    def test_statistic_wins_when_present(self):
        ev = event(
            statistics=stats(("Yellow Cards", "2", "1")),
            cards=[booking("yellow card", "home")],
        )
        assert ev.card_count("Yellow Cards", "yellow card") == (2, 1)

    def test_falls_back_to_booking_events(self):
        ev = event(
            statistics=stats(("Fouls", "10", "12")),
            cards=[
                booking("yellow card", "home"),
                booking("yellow card", "away"),
                booking("yellow card", "away"),
                booking("red card", "home"),
            ],
        )
        assert ev.card_count("Yellow Cards", "yellow card") == (1, 2)
        assert ev.card_count("Red Cards", "red card") == (1, 0)

    def test_no_bookings_but_fouls_recorded_means_zero(self):
        ev = event(statistics=stats(("Fouls", "10", "12")))
        assert ev.card_count("Yellow Cards", "yellow card") == (0, 0)

    def test_nothing_recorded_is_unknown(self):
        assert event().card_count("Yellow Cards", "yellow card") is None


class TestRefereeNames:
    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("Michael Oliver", "michael oliver"),
            ("Michael Oliver, England", "michael oliver"),
            ("  Michael   Oliver ,England ", "michael oliver"),
            ("Carlos del Cerro", "carlos del cerro"),
            ("Carlos Del Cerro, Spain", "carlos del cerro"),
            ("Florian Badstübner", "florian badstubner"),
            ("F. Badstübner", "f. badstubner"),
        ],
    )
    def test_key_ignores_case_accents_and_country(self, name, key):
        assert referee_key(name) == key

    def test_abbreviated_form_is_not_joined_by_the_key(self):
        # An initial alone could be two people; merge_referees decides that.
        assert referee_key("M. Oliver") != referee_key("Michael Oliver")

    def test_base_name_and_accent_folding(self):
        assert referee_base_name("David Webb, England") == "David Webb"
        assert fold_accents("Martínez Munuera") == "Martinez Munuera"

    def test_abbreviation_detection(self):
        assert _is_abbreviated("M. Oliver")
        assert not _is_abbreviated("Michael Oliver")


class TestPlayerStats:
    def test_players_by_side(self):
        ev = event(player_stats={"home": [{"player_key": "1"}], "away": []})
        assert ev.players("home") == [{"player_key": "1"}]
        assert ev.players("away") == []

    def test_provider_sends_empty_list_when_there_are_none(self):
        assert event(player_stats=[]).players("home") == []
        assert event().players("away") == []

    @pytest.mark.parametrize(("raw", "value"), [("3", 3), (" 0 ", 0), ("", None), ("None", None), (None, None)])
    def test_counts(self, raw, value):
        assert _count(raw) == value

    @pytest.mark.parametrize(("raw", "value"), [("7.3", 7.3), ("", None), ("-", None)])
    def test_rating(self, raw, value):
        assert _rating(raw) == value


@pytest.mark.parametrize(
    ("league_name", "name"),
    [
        ("Championship - Promotion Play-offs - Final", "Championship"),
        ("Bundesliga - Regular Season", "Bundesliga"),
        ("Premier League", "Premier League"),
        ("2. Bundesliga", "2. Bundesliga"),
    ],
)
def test_competition_name_drops_the_stage(league_name, name):
    assert competition_name(league_name) == name


def test_union_find_joins_transitively():
    groups = Components([1, 2, 3, 4])
    groups.join(1, 2)
    groups.join(2, 3)
    assert groups.find(1) == groups.find(3)
    assert sorted(len(members) for members in groups.groups().values()) == [1, 3]


class TestOddsProviderNames:
    """Their team names differ from ours; fixtures are matched on name and kickoff."""

    @pytest.mark.parametrize(
        ("ours", "theirs"),
        [
            ("Brighton", "Brighton and Hove Albion"),
            ("Tottenham", "Tottenham Hotspur"),
            ("Bayern Munich", "Bayern Munich"),
            ("Köln", "FC Cologne"),
            ("Ath Bilbao", "Athletic Bilbao"),
        ],
    )
    def test_same_club_scores_well(self, ours, theirs):
        assert similarity(ours, theirs) >= 0.6, (ours, theirs)

    @pytest.mark.parametrize(
        ("ours", "theirs"),
        [
            ("Real Madrid", "Real Sociedad"),
            ("Nottingham", "Northampton"),
        ],
    )
    def test_different_clubs_score_poorly(self, ours, theirs):
        assert similarity(ours, theirs) < 0.6, (ours, theirs)

    def test_a_fixture_must_match_on_both_teams(self):
        """'Manchester Utd' alone looks like 'Manchester City'; the pair must not."""
        derby = {"home_team": "Manchester City", "away_team": "Liverpool"}
        assert pair_score("Manchester Utd", "Liverpool", derby) < MIN_AVERAGE

    def test_a_real_pairing_clears_the_bar(self):
        event = {"home_team": "Brighton and Hove Albion", "away_team": "Arsenal"}
        assert pair_score("Brighton", "Arsenal", event) >= MIN_AVERAGE

    def test_noise_words_are_ignored(self):
        assert similarity("AC Milan", "Milan") == 1.0


class TestFreeSource:
    """football-data.co.uk: the free fallback when the paid feed lapses."""

    @pytest.mark.parametrize(
        ("day", "code"),
        [(date(2026, 10, 4), "2627"), (date(2026, 3, 1), "2526"),
         (date(2026, 7, 1), "2627"), (date(2026, 6, 30), "2526")],
    )
    def test_season_code_follows_the_july_boundary(self, day, code):
        assert current_season_code(day) == code

    def test_kickoff_converts_uk_time_to_utc(self):
        summer = parse_kickoff({"Date": "20/09/2026", "Time": "15:00"})
        winter = parse_kickoff({"Date": "10/01/2027", "Time": "15:00"})
        assert summer == datetime(2026, 9, 20, 14, 0, tzinfo=UTC)
        assert winter == datetime(2027, 1, 10, 15, 0, tzinfo=UTC)

    def test_missing_time_falls_back_to_afternoon(self):
        assert parse_kickoff({"Date": "20/09/2026", "Time": ""}).hour in (14, 15)

    def test_every_division_maps_to_a_league_we_model(self):
        from app.config import settings

        assert set(DIVISIONS.values()) <= set(settings.league_ids)
        assert len(DIVISIONS) == 16, "16 of our 17 leagues; Austria is not published"

    def test_aliases_are_lower_case_keys(self):
        # resolve() looks them up casefolded, so a capitalised key would never hit
        assert all(key == key.casefold() for key in ALIASES)
