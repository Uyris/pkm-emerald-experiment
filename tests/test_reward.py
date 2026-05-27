"""Testes da função de recompensa EmeraldReward."""

import pytest

from src.rewards.reward import EmeraldReward


def make_reward() -> EmeraldReward:
    return EmeraldReward(
        exploration_weight=0.01,
        new_map_weight=1.0,
        party_weight=5.0,
        level_weight=1.0,
        badge_weight=10.0,
        step_penalty=0.0,
    )


def test_none_info_is_zero():
    r = make_reward()
    r.reset()
    assert r.compute(None) == 0.0
    assert r.compute({}) == 0.0


def test_all_none_fields_is_zero():
    """Com tudo None (memória ainda não mapeada), a recompensa é 0 e não quebra."""
    r = make_reward()
    r.reset()
    info = {
        "party_count": None,
        "max_level": None,
        "badge_count": None,
        "map_id": None,
        "position": None,
    }
    assert r.compute(info) == 0.0


def test_exploration_rewards_new_positions_only():
    r = make_reward()
    r.reset()
    info = {"map_id": (1, 1), "position": (5, 5)}

    first = r.compute(info)
    second = r.compute(info)  # mesma posição → sem bônus de exploração

    assert first == pytest.approx(0.01 + 1.0)  # exploração + novo mapa
    assert second == 0.0


def test_new_map_reward_once_per_map():
    r = make_reward()
    r.reset()
    r.compute({"map_id": (1, 1), "position": (0, 0)})
    again = r.compute({"map_id": (1, 1), "position": (9, 9)})  # mesmo mapa, nova pos

    # novo mapa não pontua de novo; só a exploração da nova posição
    assert again == pytest.approx(0.01)


def test_level_increase_rewarded():
    r = make_reward()
    r.reset()
    # primeiro snapshot apenas registra a baseline
    assert r.compute({"max_level": 5}) == 0.0
    # aumento de 5 -> 7 = +2 * weight(1.0)
    assert r.compute({"max_level": 7}) == pytest.approx(2.0)
    # sem aumento -> 0
    assert r.compute({"max_level": 7}) == 0.0


def test_level_decrease_not_penalized():
    r = make_reward()
    r.reset()
    r.compute({"max_level": 10})
    assert r.compute({"max_level": 8}) == 0.0


def test_getting_starter_rewarded():
    """party_count 0 -> 1 (pegar o starter) dá party_weight."""
    r = make_reward()
    r.reset()
    assert r.compute({"party_count": 0}) == 0.0  # baseline
    assert r.compute({"party_count": 1}) == pytest.approx(5.0)
    assert r.compute({"party_count": 1}) == 0.0


def test_step_penalty_applied():
    r = EmeraldReward(exploration_weight=0.0, new_map_weight=0.0, step_penalty=0.1)
    r.reset()
    assert r.compute({"position": None, "map_id": None}) == pytest.approx(-0.1)
    assert r.compute(None) == pytest.approx(-0.1)


def test_badge_increase_rewarded():
    r = make_reward()
    r.reset()
    assert r.compute({"badge_count": 0}) == 0.0
    assert r.compute({"badge_count": 1}) == pytest.approx(10.0)
    assert r.compute({"badge_count": 1}) == 0.0


def test_reset_clears_state():
    r = make_reward()
    r.reset()
    info = {"map_id": (1, 1), "position": (5, 5)}
    r.compute(info)
    r.reset()
    # após reset, a mesma posição volta a pontuar como novidade
    assert r.compute(info) == pytest.approx(0.01 + 1.0)
