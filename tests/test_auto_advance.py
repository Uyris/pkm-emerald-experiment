"""Testes do auto-avanço de diálogo (gated por combate)."""

from src.emulator.gba_backend import MockGbaBackend
from src.envs.emerald_env import PokemonEmeraldEnv


class _StubBackend(MockGbaBackend):
    """Mock controlável: posição fixa/variável, em batalha ou não.

    Registra os botões pressionados para checar se o A do auto-avanço disparou.
    """

    def __init__(self, moving: bool, in_battle: bool):
        super().__init__(seed=0)
        self._moving = moving
        self._in_battle = in_battle
        self._t = 0
        self.presses = []

    def press_button(self, button: str) -> None:
        self.presses.append(button)
        super().press_button(button)

    def step_frame(self, frames: int = 1) -> None:
        self._t += 1
        super().step_frame(frames)


def _make(moving, in_battle):
    backend = _StubBackend(moving=moving, in_battle=in_battle)
    env = PokemonEmeraldEnv(
        max_steps=8, backend=backend, auto_advance_dialogue=True,
    )
    # Sobrescreve a leitura de memória para o cenário do teste.
    pos_state = {"n": 0}

    def fake_position():
        if moving:
            pos_state["n"] += 1
            return (pos_state["n"], 0)   # muda a cada leitura -> "andando"
        return (5, 5)                    # fixo -> "travado"

    env.memory.get_position = fake_position
    env.memory.get_map_id = lambda: (1, 3)
    env.memory.get_enemy_hp_fraction = lambda: (0.8 if in_battle else None)
    env.memory.snapshot = lambda: {
        "party_count": 1, "max_level": 5, "total_level": 5,
        "badge_count": 0, "map_id": (1, 3), "position": (5, 5),
    }
    return env, backend


def test_autopass_taps_a_when_blocked_out_of_battle():
    env, backend = _make(moving=False, in_battle=False)
    env.reset(seed=0)
    backend.presses.clear()
    _, _, _, _, info = env.step(0)  # ação 0 = UP
    assert "A" in backend.presses          # auto-avanço disparou
    assert info.get("auto_advanced") is True
    env.close()


def test_autopass_blocked_during_battle():
    """Travado MAS em combate (inimigo vivo) -> NÃO deve apertar A."""
    env, backend = _make(moving=False, in_battle=True)
    env.reset(seed=0)
    backend.presses.clear()
    _, _, _, _, info = env.step(0)
    assert "A" not in backend.presses      # bloqueado na batalha
    assert info.get("auto_advanced") is None
    env.close()


def test_autopass_not_triggered_when_moving():
    """Andando (posição muda) -> não está travado -> sem auto-avanço."""
    env, backend = _make(moving=True, in_battle=False)
    env.reset(seed=0)
    backend.presses.clear()
    _, _, _, _, info = env.step(0)
    assert info.get("auto_advanced") is None
    env.close()


def test_autopass_disabled_by_default():
    env = PokemonEmeraldEnv(max_steps=8, backend=MockGbaBackend(seed=0))
    assert env.auto_advance is False
    env.close()
