"""Carregamento de configuração e fábricas de ambiente.

Centraliza a leitura do YAML e a construção do ``PokemonEmeraldEnv`` para que
``train.py``, ``evaluate.py`` e ``play_random.py`` compartilhem a mesma lógica.

Fábricas:

* :func:`make_env`  → um ``PokemonEmeraldEnv`` único, observação ``(84, 84, 1)``.
  Usado por ``play_random`` e pelos testes.
* :func:`make_vec_env` → ``VecEnv`` do SB3 (``DummyVecEnv`` ou ``SubprocVecEnv``
  conforme ``train.n_envs``), com **frame stacking no eixo de canais** via
  ``VecFrameStack`` → ``(84, 84, N)`` (imagem válida para ``CnnPolicy``).

> Por que não empilhar com o wrapper do Gymnasium? ``FrameStackObservation``
> adiciona um eixo novo → ``(N, 84, 84, 1)`` (4D), rejeitado pela NatureCNN.
> ``VecFrameStack(channels_order="last")`` empilha no canal.
"""

from __future__ import annotations

from typing import Optional

import yaml

from src.envs.emerald_env import PokemonEmeraldEnv


def load_config(path: str) -> dict:
    """Lê um arquivo YAML de configuração e devolve um dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(
    config: dict,
    render_mode: Optional[str] = None,
    backend=None,
) -> PokemonEmeraldEnv:
    """Constrói um ``PokemonEmeraldEnv`` único a partir da config.

    Args:
        config: dict de configuração.
        render_mode: passado ao env (use ``"rgb_array"`` para visualização).
        backend: backend já instanciado (injeção para testes). Se ``None``, o
            env cria um ``GbaBackend`` real.
    """
    emulator = config.get("emulator", {})
    env_cfg = config.get("env", {})
    reward_cfg = config.get("reward", {})

    return PokemonEmeraldEnv(
        rom_path=emulator.get("rom_path"),
        init_state=emulator.get("init_state"),
        init_states=emulator.get("init_states"),
        init_state_weights=emulator.get("init_state_weights"),
        init_state_strategy=emulator.get("init_state_strategy", "random"),
        headless=emulator.get("headless", True),
        obs_width=env_cfg.get("obs_width", 84),
        obs_height=env_cfg.get("obs_height", 84),
        frame_skip=env_cfg.get("frame_skip", 4),
        max_steps=env_cfg.get("max_steps", 4096),
        actions=env_cfg.get("actions"),  # None -> DEFAULT_ACTIONS (sem START)
        goals=env_cfg.get("goals"),
        auto_advance_dialogue=env_cfg.get("auto_advance_dialogue", False),
        auto_advance_button=env_cfg.get("auto_advance_button", "A"),
        auto_advance_frames=env_cfg.get("auto_advance_frames", 4),
        reward_config=reward_cfg,
        render_mode=render_mode,
        backend=backend,
    )


def make_vec_env(config: dict, render_mode: Optional[str] = "rgb_array"):
    """Constrói um ``VecEnv`` do SB3 pronto para PPO + ``CnnPolicy``.

    * ``train.n_envs == 1`` → ``DummyVecEnv`` (in-process, ideal para depurar).
    * ``train.n_envs > 1``  → ``SubprocVecEnv`` (ambientes em paralelo).

    Aplica ``VecFrameStack`` no eixo de canais quando ``env.frame_stack > 1``.
    ``render_mode="rgb_array"`` mantém ``get_images()`` funcional (visualização).
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecFrameStack,
    )

    env_cfg = config.get("env", {})
    train_cfg = config.get("train", {})
    n_envs = int(train_cfg.get("n_envs", 1))
    n_stack = int(env_cfg.get("frame_stack", 1))

    def _thunk():
        # Monitor por-env funciona tanto em DummyVecEnv quanto em SubprocVecEnv.
        return Monitor(make_env(config, render_mode=render_mode))

    env_fns = [_thunk for _ in range(n_envs)]

    if n_envs > 1:
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        venv = DummyVecEnv(env_fns)

    if n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack, channels_order="last")
    return venv
