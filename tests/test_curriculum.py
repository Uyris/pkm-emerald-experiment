"""Testes do currículo de save states (amostragem de init_states no reset).

Usa MockGbaBackend (set_state é no-op), então não precisa de ROM/emulador —
testamos a lógica de seleção do env e a leitura dos arquivos de state.
"""

import pytest

from src.emulator.gba_backend import MockGbaBackend
from src.envs.emerald_env import PokemonEmeraldEnv


@pytest.fixture
def state_files(tmp_path):
    """Cria 3 arquivos de 'state' fake e devolve seus caminhos."""
    paths = []
    for i in range(3):
        p = tmp_path / f"curr_{i:03d}.state"
        p.write_bytes(b"FAKE_STATE_" + bytes([i]))
        paths.append(str(p))
    return paths


def _env(state_files, **kwargs):
    return PokemonEmeraldEnv(
        init_states=state_files,
        max_steps=8,
        backend=MockGbaBackend(seed=0),
        **kwargs,
    )


def test_curriculum_loads_state_bytes(state_files):
    env = _env(state_files)
    # Os bytes dos 3 arquivos foram lidos uma vez no __init__.
    assert len(env._curriculum_states) == 3
    assert env._use_curriculum is True
    env.close()


def test_reset_reports_chosen_state(state_files):
    env = _env(state_files)
    _, info = env.reset(seed=0)
    assert info["init_state"] in state_files
    env.close()


def test_sequential_strategy_cycles(state_files):
    env = _env(state_files, init_state_strategy="sequential")
    chosen = [env.reset(seed=0)[1]["init_state"] for _ in range(6)]
    # Deve percorrer 0,1,2,0,1,2 na ordem dos arquivos.
    assert chosen == state_files + state_files
    env.close()


def test_weights_length_must_match(state_files):
    with pytest.raises(ValueError):
        PokemonEmeraldEnv(
            init_states=state_files,
            init_state_weights=[0.5, 0.5],  # 2 != 3
            backend=MockGbaBackend(seed=0),
        )


def test_weighted_sampling_respects_zero_weight(state_files):
    # Peso 0 nos dois primeiros -> só o índice 2 deve ser sorteado.
    env = _env(state_files, init_state_weights=[0.0, 0.0, 1.0])
    chosen = {env.reset(seed=s)[1]["init_state"] for s in range(10)}
    assert chosen == {state_files[2]}
    env.close()


def test_no_curriculum_when_not_provided():
    env = PokemonEmeraldEnv(max_steps=8, backend=MockGbaBackend(seed=0))
    assert env._use_curriculum is False
    _, info = env.reset(seed=0)
    assert "init_state" not in info
    env.close()
