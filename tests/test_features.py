"""Team and referee features."""

from types import SimpleNamespace

import pytest

from app.features.rates import RATED, STATS, RefereeState, TeamState, _fold, decay_factor, fit_ratings


def side(team_id: int, home_team_id: int, competition_id: int = 1, **values) -> SimpleNamespace:
    stats = {name: None for name in STATS} | values
    return SimpleNamespace(
        team_id=team_id, home_team_id=home_team_id, competition_id=competition_id, **stats
    )


def test_decay_halves_after_half_life():
    assert decay_factor(6.0) ** 6 == pytest.approx(0.5)


class TestFold:
    def test_first_value_is_taken_as_is(self):
        ewma: dict = {}
        _fold(ewma, 1, "corners_for", 7, lam=0.9)
        assert ewma[(1, "corners_for")] == 7.0

    def test_later_values_are_weighted(self):
        ewma = {(1, "corners_for"): 10.0}
        _fold(ewma, 1, "corners_for", 4, lam=0.75)
        assert ewma[(1, "corners_for")] == pytest.approx(0.25 * 4 + 0.75 * 10)

    def test_missing_value_changes_nothing(self):
        ewma = {(1, "corners_for"): 10.0}
        _fold(ewma, 1, "corners_for", None, lam=0.75)
        assert ewma == {(1, "corners_for"): 10.0}


class TestTeamState:
    def test_features_before_a_match_do_not_contain_it(self):
        """Point-in-time rule (R-02): a snapshot is taken before the match is folded in."""
        state = TeamState()
        home = side(1, home_team_id=1, corners=6, fouls=11)
        away = side(2, home_team_id=1, corners=3, fouls=14)

        before = state.features(1, "home", competition_id=1)
        state.fold(home, away, lam=0.9)
        state.fold(away, home, lam=0.9)
        after = state.features(1, "home", competition_id=1)

        assert before["matches_seen"] == 0
        assert before["corners_for"] is None
        assert after["matches_seen"] == 1
        assert after["corners_for"] == 6
        assert after["corners_against"] == 3
        assert after["fouls_against_venue"] == 14
        assert state.features(1, "away", competition_id=1)["corners_for_venue"] is None

    def test_every_rated_stat_gets_attack_and_defence(self):
        features = TeamState().features(1, "home", competition_id=1)
        for name in RATED:
            assert f"{name}_attack" in features and f"{name}_defence" in features


def test_ratings_find_the_strong_attack():
    rows = []
    for _ in range(20):
        for team in range(1, 6):
            for opponent in range(1, 6):
                if team != opponent:
                    rows.append((team, opponent, 1, 8.0 if team == 1 else 4.0))
    ratings = fit_ratings(rows, lam=1.0)
    assert ratings[1][0] > 2.5
    assert all(abs(ratings[t][0]) < 1.5 for t in range(2, 6))


class TestRefereeState:
    def test_rates_and_home_share(self):
        referees = RefereeState()
        home = side(1, 1, yellow=3, red=1, fouls=12)
        away = side(2, 1, yellow=1, red=0, fouls=10)
        referees.fold(7, home, away, lam=0.9)

        features = referees.features(7)
        assert features["ref_matches_seen"] == 1
        assert features["ref_cards_pm"] == 5
        assert features["ref_fouls_pm"] == 22
        assert features["ref_cards_per_foul"] == pytest.approx(5 / 22)
        assert features["ref_home_card_share"] == pytest.approx(4 / 5)

    def test_card_free_match_leaves_home_share_undefined(self):
        referees = RefereeState()
        home = side(1, 1, yellow=0, red=0, fouls=12)
        away = side(2, 1, yellow=0, red=0, fouls=10)
        referees.fold(7, home, away, lam=0.9)
        assert referees.features(7)["ref_home_card_share"] is None

    def test_unknown_referee_has_no_history(self):
        features = RefereeState().features(99)
        assert features["ref_matches_seen"] == 0
        assert features["ref_fouls_pm"] is None
