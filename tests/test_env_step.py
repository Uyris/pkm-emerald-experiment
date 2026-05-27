"""Testes do step do PokemonEmeraldEnv (usa o backend placeholder)."""

import numpy as np
import pytest

from src.emulator.gba_backend import MockGbaBackend
from src.envs.emerald_env import ACTIONS, PokemonEmeraldEnv


def make_test_env(max_steps: int = 8) -> PokemonEmeraldEnv:
    return PokemonEmeraldEnv(max_steps=max_steps, backend=MockGbaBackend(seed=0))


def test_step_returns_five_tuple():
    env = make_test_env()
    env.reset(seed=0)
    result = env.step(0)

    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_step_obs_matches_observation_space():
    env = make_test_env()
    env.reset(seed=0)
    obs, *_ = env.step(0)

    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize("action", range(len(ACTIONS)))
def test_all_actions_are_valid(action):
    env = make_test_env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

    assert info["action"] == ACTIONS[action]
    env.close()


def test_invalid_action_raises():
    env = make_test_env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(len(ACTIONS))  # fora do Discrete
    env.close()


def test_episode_truncates_at_max_steps():
    max_steps = 5
    env = make_test_env(max_steps=max_steps)
    env.reset(seed=0)

    truncated = False
    for _ in range(max_steps):
        _, _, terminated, truncated, _ = env.step(0)

    assert truncated is True
    env.close()


def _env_with_snapshots(snapshots, goals):
    """Cria um env cujo memory.snapshot() devolve `snapshots` em sequência."""
    env = PokemonEmeraldEnv(max_steps=100, goals=goals, backend=MockGbaBackend(seed=0))
    seq = iter(snapshots)
    last = {"value": snapshots[-1]}

    def fake_snapshot():
        try:
            last["value"] = next(seq)
        except StopIteration:
            pass
        return dict(last["value"])

    env.memory.snapshot = fake_snapshot
    return env


def test_goal_leave_start_map_terminates():
    # reset vê mapa (1,3); depois muda para (1,2) -> termina.
    env = _env_with_snapshots(
        [{"map_id": (1, 3)}, {"map_id": (1, 3)}, {"map_id": (1, 2)}],
        goals={"leave_start_map": True, "goal_reward": 10.0},
    )
    env.reset(seed=0)
    _, r1, term1, _, _ = env.step(0)   # ainda no mapa inicial
    assert term1 is False
    _, r2, term2, _, info = env.step(0)  # saiu do mapa inicial
    assert term2 is True
    assert info["goal_reached"] == "leave_start_map"
    assert r2 >= 10.0  # inclui o goal_reward
    env.close()


def test_goal_party_count_terminates():
    env = _env_with_snapshots(
        [{"party_count": 0}, {"party_count": 0}, {"party_count": 1}],
        goals={"party_count": 1},
    )
    env.reset(seed=0)
    _, _, term1, _, _ = env.step(0)
    assert term1 is False
    _, _, term2, _, info = env.step(0)
    assert term2 is True
    assert info["goal_reached"] == "party_count"
    env.close()


def test_no_goal_never_terminates():
    env = PokemonEmeraldEnv(max_steps=3, backend=MockGbaBackend(seed=0))
    env.reset(seed=0)
    terms = [env.step(0)[2] for _ in range(3)]
    assert not any(terms)  # terminated sempre False sem goals
    env.close()


def test_step_increments_step_counter():
    env = make_test_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert info["steps"] == 1
    _, _, _, _, info = env.step(0)
    assert info["steps"] == 2
    env.close()
