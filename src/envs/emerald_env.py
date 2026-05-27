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
        goals: dict com condições de término do episódio (``terminated=True``).
            Chaves suportadas:
              * ``leave_start_map`` (bool): termina ao sair do mapa inicial
                (ex.: sair do quarto).
              * ``party_count`` (int): termina quando o time tiver >= N Pokémon
                (ex.: 1 = pegou o starter).
              * ``map`` ([group, num]): termina ao chegar nesse mapa
                (ex.: Oldale Town, quando você souber o id).
              * ``goal_reward`` (float): bônus somado à recompensa ao atingir
                a meta. Default 0.0.
            ``None``/vazio → sem término por meta (só trunca por max_steps).
        backend: instância de backend já criada (injeção para testes). Se
            ``None``, cria um :class:`GbaBackend`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: Optional[str] = None,
        init_state: Optional[str] = None,
        init_states: Optional[Sequence[str]] = None,
        init_state_weights: Optional[Sequence[float]] = None,
        init_state_strategy: str = "random",
        obs_width: int = 84,
        obs_height: int = 84,
        frame_skip: int = 4,
        max_steps: int = 4096,
        reward_config: Optional[dict] = None,
        headless: bool = True,
        actions: Optional[Sequence[str]] = None,
        goals: Optional[dict] = None,
        render_mode: Optional[str] = None,
        backend: Optional[GbaBackend] = None,
    ) -> None:
        super().__init__()

        self.obs_width = obs_width
        self.obs_height = obs_height
        self.frame_skip = max(1, frame_skip)
        self.max_steps = max_steps
        self.goals = goals or {}
        self.goal_reward = float(self.goals.get("goal_reward", 0.0))
        self.render_mode = render_mode

        # --- Currículo de save states ---
        # Se `init_states` (lista) for dado, um estado é sorteado a cada reset
        # (do mais perto da meta ao mais longe → aprende do fácil ao difícil).
        # Carregamos os bytes uma vez; o backend recebe via set_state no reset.
        self._curriculum_paths = list(init_states) if init_states else []
        self._curriculum_states = [self._read_state(p) for p in self._curriculum_paths]
        # Pesos só fazem sentido com currículo ativo; sem estados, são ignorados
        # (evita erro de tamanho quando o currículo é desligado, ex.: no evaluate).
        if self._curriculum_states:
            self._curriculum_weights = list(init_state_weights) if init_state_weights else None
            if self._curriculum_weights and len(self._curriculum_weights) != len(self._curriculum_states):
                raise ValueError("init_state_weights deve ter o mesmo tamanho de init_states.")
        else:
            self._curriculum_weights = None
        self.init_state_strategy = init_state_strategy
        self._seq_idx = 0
        # No modo currículo, o env gerencia o estado: backend não auto-carrega.
        self._use_curriculum = bool(self._curriculum_states)
        backend_init_state = None if self._use_curriculum else init_state

        # --- Ações (configurável; valida contra os botões do backend) ---
        self.actions = tuple(actions) if actions else DEFAULT_ACTIONS
        invalid = [b for b in self.actions if b not in VALID_BUTTONS]
        if invalid:
            raise ValueError(
                f"Botões inválidos em actions: {invalid}. Válidos: {VALID_BUTTONS}"
            )

        # --- Camada de emulação (injetável para testes) ---
        self.backend = backend or GbaBackend(
            rom_path=rom_path, init_state=backend_init_state, headless=headless
        )
        self.memory = EmeraldMemory(self.backend)
        self.reward_fn = EmeraldReward(**(reward_config or {}))
        # Flags de evento a ler a cada passo (para os milestones do reward).
        self._milestone_flags = [
            m["flag"] for m in (reward_config or {}).get("flag_milestones", [])
            if "flag" in m
        ]

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
        self._start_map: Optional[tuple] = None

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

        # Currículo: sorteia qual save state usar neste episódio.
        chosen_path = None
        if self._use_curriculum:
            idx = self._pick_state_index()
            self.backend.set_state(self._curriculum_states[idx])
            chosen_path = self._curriculum_paths[idx]

        self.backend.reset()
        self.reward_fn.reset()
        self._steps = 0

        obs = self._get_obs()
        info = self.memory.snapshot()
        # Mapa inicial do episódio (referência para a meta "sair do mapa").
        self._start_map = info.get("map_id")
        if chosen_path is not None:
            info["init_state"] = chosen_path
        return obs, info

    def _pick_state_index(self) -> int:
        """Escolhe o índice do save state do currículo para este episódio."""
        n = len(self._curriculum_states)
        if self.init_state_strategy == "sequential":
            idx = self._seq_idx % n
            self._seq_idx += 1
            return idx
        # "random" (com pesos opcionais). Usa o RNG do Gymnasium (semeável).
        if self._curriculum_weights:
            probs = np.asarray(self._curriculum_weights, dtype=float)
            probs = probs / probs.sum()
            return int(self.np_random.choice(n, p=probs))
        return int(self.np_random.integers(n))

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
        if self._milestone_flags:
            info["flags"] = {fid: self.memory.get_flag(fid) for fid in self._milestone_flags}
        reward = self.reward_fn.compute(info)

        # Término por meta atingida (configurável). Bônus terminal opcional.
        terminated, reason = self._check_goal(info)
        if terminated:
            reward += self.goal_reward
            info["goal_reached"] = reason
        truncated = self._steps >= self.max_steps

        info["steps"] = self._steps
        info["action"] = button
        return obs, reward, terminated, truncated, info

    def _check_goal(self, info: dict) -> tuple[bool, Optional[str]]:
        """Avalia as condições de meta. Retorna ``(terminated, motivo)``.

        Valores ``None`` na memória nunca disparam término (tolerante a falhas).
        """
        map_id = info.get("map_id")
        party = info.get("party_count")

        if self.goals.get("leave_start_map") and map_id is not None \
                and self._start_map is not None and map_id != self._start_map:
            return True, "leave_start_map"

        target_party = self.goals.get("party_count")
        if target_party is not None and party is not None and party >= target_party:
            return True, "party_count"

        target_map = self.goals.get("map")
        if target_map is not None and map_id is not None \
                and tuple(target_map) == tuple(map_id):
            return True, "map"

        return False, None

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
    @staticmethod
    def _read_state(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _get_obs(self) -> np.ndarray:
        frame = self.backend.get_screen()
        self._last_frame = frame
        return preprocess_frame(frame, width=self.obs_width, height=self.obs_height)
