"""Ambiente Gymnasium para Pokémon Emerald.

:class:`PokemonEmeraldEnv` conecta o emulador (via :class:`GbaBackend`), a
leitura de memória (:class:`EmeraldMemory`), o pré-processamento de tela e a
função de recompensa (:class:`EmeraldReward`) em uma API Gymnasium padrão.

Observação:  grayscale ``(84, 84, 1)`` ``uint8`` (configurável).
Ações:       ``Discrete(n)`` sobre um conjunto configurável de botões.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.emulator.gba_backend import BUTTONS as VALID_BUTTONS
from src.emulator.gba_backend import GbaBackend
from src.memory.emerald_memory import EmeraldMemory
from src.rewards.reward import EmeraldReward
from src.utils.preprocessing import preprocess_frame

# Conjunto de ações PADRÃO: sem START/SELECT.
# START abre o menu (salvar, Pokédex, itens...) e atrapalha a meta inicial — a
# IA fica abrindo menu/salvando. Remover o botão é mais eficiente do que
# penalizar via recompensa (ela nem tem a opção de desperdiçar passos no menu).
# Reative incluindo "START" em `env.actions` no YAML quando precisar.
DEFAULT_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B")

# Compatibilidade retroativa (imports antigos).
ACTIONS = DEFAULT_ACTIONS


class PokemonEmeraldEnv(gym.Env):
    """Ambiente RL para Pokémon Emerald.

    Args:
        rom_path: caminho da ROM (passado ao backend). Pode ser ``None`` no
            modo placeholder.
        init_state: save state opcional aplicado no reset.
        obs_width / obs_height: tamanho da observação grayscale.
        frame_skip: nº de frames do emulador avançados por ação (action repeat).
        max_steps: passos máximos por episódio antes de ``truncated=True``.
        reward_config: dict de pesos para :class:`EmeraldReward`.
        headless: roda o emulador sem janela.
        actions: sequência de botões que compõem o action space (na ordem do
            índice ``Discrete``). Default :data:`DEFAULT_ACTIONS` (sem START).
        backend: instância de backend já criada (injeção para testes). Se
            ``None``, cria um :class:`GbaBackend`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: Optional[str] = None,
        init_state: Optional[str] = None,
        obs_width: int = 84,
        obs_height: int = 84,
        frame_skip: int = 4,
        max_steps: int = 4096,
        reward_config: Optional[dict] = None,
        headless: bool = True,
        actions: Optional[Sequence[str]] = None,
        render_mode: Optional[str] = None,
        backend: Optional[GbaBackend] = None,
    ) -> None:
        super().__init__()

        self.obs_width = obs_width
        self.obs_height = obs_height
        self.frame_skip = max(1, frame_skip)
        self.max_steps = max_steps
        self.render_mode = render_mode

        # --- Ações (configurável; valida contra os botões do backend) ---
        self.actions = tuple(actions) if actions else DEFAULT_ACTIONS
        invalid = [b for b in self.actions if b not in VALID_BUTTONS]
        if invalid:
            raise ValueError(
                f"Botões inválidos em actions: {invalid}. Válidos: {VALID_BUTTONS}"
            )

        # --- Camada de emulação (injetável para testes) ---
        self.backend = backend or GbaBackend(
            rom_path=rom_path, init_state=init_state, headless=headless
        )
        self.memory = EmeraldMemory(self.backend)
        self.reward_fn = EmeraldReward(**(reward_config or {}))

        # --- Espaços ---
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_height, self.obs_width, 1),
            dtype=np.uint8,
        )

        self._steps = 0
        self._last_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.backend.reset()
        self.reward_fn.reset()
        self._steps = 0

        obs = self._get_obs()
        info = self.memory.snapshot()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Ação inválida: {action!r}")

        button = self.actions[int(action)]

        # Pressiona o botão, avança frames (action repeat), solta tudo.
        self.backend.press_button(button)
        self.backend.step_frame(self.frame_skip)
        self.backend.release_all()

        self._steps += 1

        obs = self._get_obs()
        info = self.memory.snapshot()
        reward = self.reward_fn.compute(info)

        # Por enquanto não há condição de fim "vitória/derrota" mapeada.
        terminated = False
        truncated = self._steps >= self.max_steps

        info["steps"] = self._steps
        info["action"] = button
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Retorna o frame RGB atual (modo ``rgb_array``)."""
        if self.render_mode == "rgb_array":
            return self._last_frame if self._last_frame is not None else self.backend.get_screen()
        return None

    def close(self) -> None:
        self.backend.close()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        frame = self.backend.get_screen()
        self._last_frame = frame
        return preprocess_frame(frame, width=self.obs_width, height=self.obs_height)
