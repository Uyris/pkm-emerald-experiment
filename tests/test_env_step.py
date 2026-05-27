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


def test_step_increments_step_counter():
    env = make_test_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert info["steps"] == 1
    _, _, _, _, info = env.step(0)
    assert info["steps"] == 2
    env.close()
