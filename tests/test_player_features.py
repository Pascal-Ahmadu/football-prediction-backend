"""Player features: rates per 90 and involvement, from earlier matches only."""

from types import SimpleNamespace

import pytest

from app.features.players import RECENT_TEAM_MATCHES, PlayerState


def row(player_id, minutes, starter=True, position="Forwards", shots=0, sot=0, fouls=0):
    return SimpleNamespace(
        player_id=player_id, minutes=minutes, is_starter=starter, position=position,
        shots=shots, shots_on_target=sot, fouls_committed=fouls,
    )


def test_snapshot_before_a_match_does_not_contain_it():
    state = PlayerState(lam=0.9)
    before = state.features(7, team_id=1)
    state.fold_team_match(1, [row(7, 90, sot=2, fouls=1)])
    after = state.features(7, team_id=1)

    assert before["appearances_seen"] == 0
    assert before["shots_on_target_p90"] is None
    assert before["team_matches_since_played"] is None
    assert after["appearances_seen"] == 1
    assert after["shots_on_target_p90"] == pytest.approx(2.0)
    assert after["fouls_committed_p90"] == pytest.approx(1.0)
    assert after["position"] == 3


def test_rates_are_per_90_minutes_played():
    state = PlayerState(lam=0.5)
    state.fold_team_match(1, [row(7, 90, sot=1)])
    state.fold_team_match(1, [row(7, 30, sot=1)])
    # weighted counts 1 + 0.5*1 = 1.5 over weighted minutes 30 + 0.5*90 = 75
    assert state.features(7, 1)["shots_on_target_p90"] == pytest.approx(90 * 1.5 / 75)


def test_unused_substitute_counts_as_not_playing():
    state = PlayerState(lam=0.9)
    state.fold_team_match(1, [row(7, 90), row(8, 0, starter=False)])
    features = state.features(8, 1)
    assert features["appearances_seen"] == 0
    assert features["apps_recent"] == 0
    assert features["minutes_pa"] is None


def test_absence_is_seen_even_when_the_player_is_not_listed():
    state = PlayerState(lam=0.9)
    state.fold_team_match(1, [row(7, 90), row(8, 90)])
    for _ in range(3):  # player 8 injured: not in the squad at all
        state.fold_team_match(1, [row(7, 90)])
    features = state.features(8, 1)
    assert features["team_matches_since_played"] == 3
    assert features["apps_recent"] == 1
    assert features["starts_recent"] == 1
    assert features["minutes_recent"] == 90
    assert features["team_matches_recent"] == 4


def test_recent_window_forgets_old_matches():
    state = PlayerState(lam=0.9)
    state.fold_team_match(1, [row(8, 90)])
    for _ in range(RECENT_TEAM_MATCHES):
        state.fold_team_match(1, [row(7, 90)])
    assert state.features(8, 1)["apps_recent"] == 0
    assert 8 not in state.candidates(1)


def test_transferred_player_is_a_candidate_only_for_his_new_team():
    state = PlayerState(lam=0.9)
    state.fold_team_match(1, [row(9, 90)])
    state.fold_team_match(2, [row(9, 90)])
    assert 9 not in state.candidates(1)
    assert 9 in state.candidates(2)


def test_usual_minutes_is_a_weighted_average_not_anchored_on_the_debut():
    """A 4-minute debut must not make a regular starter look like a 4-minute player."""
    state = PlayerState(lam=0.933)  # half-life 10 appearances
    state.fold_team_match(1, [row(7, 4)])
    assert state.features(7, 1)["minutes_pa"] == pytest.approx(4)
    for _ in range(2):
        state.fold_team_match(1, [row(7, 90)])
    minutes_pa = state.features(7, 1)["minutes_pa"]
    assert 55 < minutes_pa < 65, minutes_pa  # the plain EWMA gave about 15


def test_rates_weight_the_latest_appearance_most():
    state = PlayerState(lam=0.5)
    state.fold_team_match(1, [row(7, 90, sot=0)])
    state.fold_team_match(1, [row(7, 90, sot=2)])
    # the recent 2 counts double the older 0: (2 + 0.5*0) / (90 + 0.5*90) * 90
    assert state.features(7, 1)["shots_on_target_p90"] == pytest.approx(90 * 2 / 135)
