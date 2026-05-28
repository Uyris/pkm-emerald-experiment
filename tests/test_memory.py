"""Testes do EmeraldMemory usando o backend mock.

Com o mock (memória zerada), os getters que dependem do ponteiro de SaveBlock1
devem degradar para ``None`` (ponteiro inválido), e o snapshot deve conter
todas as chaves esperadas.
"""

from src.emulator.gba_backend import MockGbaBackend
from src.memory.emerald_memory import EmeraldMemory


def make_memory() -> EmeraldMemory:
    return EmeraldMemory(MockGbaBackend(seed=0))


def test_snapshot_has_all_keys():
    snap = make_memory().snapshot()
    for key in ("party_count", "max_level", "badge_count", "map_id", "position"):
        assert key in snap


def test_party_count_zero_on_mock():
    # gPlayerPartyCount lê 0 no mock (memória zerada) -> 0 é válido (<= 6).
    assert make_memory().get_party_count() == 0


def test_saveblock_fields_none_on_mock():
    # Ponteiro de SaveBlock1 = 0 no mock -> inválido -> None.
    mem = make_memory()
    assert mem.get_map_id() is None
    assert mem.get_position() is None
    assert mem.get_badge_count() is None


def test_max_level_none_when_party_empty():
    assert make_memory().get_max_level() is None


def test_get_flag_none_on_mock():
    # Ponteiro de SaveBlock1 inválido no mock -> flag indisponível -> None.
    assert make_memory().get_flag(0x867) is None


def test_read_flag_bytes_none_on_mock():
    assert make_memory().read_flag_bytes(16) is None


def test_event_flag_count_none_on_mock():
    assert make_memory().get_event_flag_count() is None
