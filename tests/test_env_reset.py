"""Testes do reset do PokemonEmeraldEnv (usa o backend placeholder)."""

import numpy as np

from src.emulator.gba_backend import MockGbaBackend
from src.envs.emerald_env import ACTIONS, PokemonEmeraldEnv


def make_test_env() -> PokemonEmeraldEnv:
    # Usa o backend mock (sem ROM/emulador) para testes determinísticos.
    return PokemonEmeraldEnv(
        obs_width=84,
        obs_height=84,
        max_steps=32,
        backend=MockGbaBackend(seed=0),
    )


def test_reset_returns_obs_and_info():
    env = make_test_env()
    obs, info = env.reset(seed=0)

    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    env.close()


def test_reset_obs_shape_and_dtype():
    env = make_test_env()
    obs, _ = env.reset(seed=0)

    assert obs.shape == (84, 84, 1)
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)
    env.close()


def test_action_space_matches_default_actions():
    env = make_test_env()
    # Default sem START -> 6 ações (UP, DOWN, LEFT, RIGHT, A, B).
    assert env.action_space.n == len(ACTIONS) == 6
    assert "START" not in env.actions
    env.close()


def test_custom_actions_can_include_start():
    from src.emulator.gba_backend import MockGbaBackend
    env = PokemonEmeraldEnv(
        actions=["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"],
        backend=MockGbaBackend(seed=0),
    )
    assert env.action_space.n == 7
    assert env.actions[6] == "START"
    env.close()


def test_reset_can_be_called_multiple_times():
    env = make_test_env()
    obs_a, _ = env.reset(seed=1)
    obs_b, _ = env.reset(seed=2)

    assert env.observation_space.contains(obs_a)
    assert env.observation_space.contains(obs_b)
    env.close()


def test_reset_info_contains_expected_keys():
    env = make_test_env()
    _, info = env.reset(seed=0)

    for key in ("party_count", "max_level", "badge_count", "map_id", "position"):
        assert key in info
    env.close()
