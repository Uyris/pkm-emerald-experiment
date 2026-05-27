"""Descobre qual(is) flag(s) de evento mudaram entre dois save states.

Útil para achar a flag que uma cutscene/evento liga (ex.: a interação na casa
do vizinho, pré-requisito do starter). Faça dois save states com
``make_init_state.py`` — um ANTES e um DEPOIS do evento — e rode:

    python scripts/find_event_flag.py \
        --rom "roms/Pokemon - Emerald Version (USA, Europe).gba" \
        --before roms/before.state \
        --after  roms/after.state

A saída lista os ids de flag que passaram de 0 -> 1 (candidatos ao evento).
Depois ponha o id em `reward.flag_milestones` (configs/default.yaml).

Dica: quanto mais "cirúrgicos" forem os dois states (salvar imediatamente
antes e imediatamente depois da cutscene, sem andar/mexer em mais nada), menos
flags aparecem no diff — idealmente só a do evento.
"""

from __future__ import annotations

import argparse
import os
import sys

# Permite `python scripts/find_event_flag.py` da raiz do projeto: garante que a
# raiz (e não a pasta scripts/) esteja no sys.path para importar `src.*`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff de flags de evento entre 2 save states.")
    parser.add_argument("--rom", default="roms/Pokemon - Emerald Version (USA, Europe).gba")
    parser.add_argument("--before", required=True, help="Save state ANTES do evento.")
    parser.add_argument("--after", required=True, help="Save state DEPOIS do evento.")
    args = parser.parse_args()

    import mgba.log
    mgba.log.silence()

    from src.emulator.gba_backend import GbaBackend
    from src.memory.emerald_memory import EmeraldMemory

    backend = GbaBackend(rom_path=args.rom)
    mem = EmeraldMemory(backend)

    backend.load_state(args.before)
    before = mem.read_flag_bytes()
    backend.load_state(args.after)
    after = mem.read_flag_bytes()

    if before is None or after is None:
        print("[erro] Não consegui ler as flags (ponteiro de SaveBlock1 inválido). "
              "Os states estão dentro do jogo, com o personagem controlável?")
        return 1

    turned_on, turned_off = [], []
    for byte_idx in range(min(len(before), len(after))):
        a, b = before[byte_idx], after[byte_idx]
        if a == b:
            continue
        for bit in range(8):
            am, bm = (a >> bit) & 1, (b >> bit) & 1
            if am == bm:
                continue
            flag_id = byte_idx * 8 + bit
            (turned_on if bm else turned_off).append(flag_id)

    print(f"Flags que LIGARAM (0 -> 1): {[hex(f) for f in turned_on] or 'nenhuma'}")
    if turned_off:
        print(f"Flags que desligaram (1 -> 0): {[hex(f) for f in turned_off]}")

    if len(turned_on) == 1:
        f = turned_on[0]
        print(f"\nCandidato único: flag {hex(f)} ({f}). Use em flag_milestones:")
        print(f"  flag_milestones:\n    - {{flag: {hex(f)}, reward: 3.0}}")
    elif turned_on:
        print("\nVárias flags mudaram. Refaça os states mais 'cirúrgicos' (salvar "
              "imediatamente antes/depois do evento) para reduzir os candidatos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
