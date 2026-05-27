"""Joga a ROM interativamente para criar save states do treino.

A IA não consegue navegar a introdução / criação de novo save do Pokémon
Emerald (logos, gênero, nome, relógio). Use este script para jogar com o
teclado e salvar save states em pontos-chave do caminho — para um único
``init_state`` ou para um **currículo** (vários checkpoints da jornada até a
bolsa do Prof. Birch). Depois aponte ``emulator.init_state`` (único) ou
``emulator.init_states`` (lista) em configs/default.yaml.

Controles:
    Setas      → D-Pad
    Z          → A
    X          → B
    Enter      → START
    Backspace  → SELECT
    A / S      → L / R
    F5         → salvar state numerado (curr_000.state, curr_001.state, ...)
    F9         → CARREGAR um state (o de --load, ou o último salvo nesta sessão)
    ESC        → sair

Dica de currículo: jogue do quarto até a bolsa do Birch e dê F5 em vários
pontos (perto da bolsa, na Route 101, fora de casa, no quarto). Cada F5 gera
um arquivo numerado; liste-os em `init_states` (os mais perto da meta primeiro).

Uso:
    python scripts/make_init_state.py \
        --rom "roms/Pokemon - Emerald Version (USA, Europe).gba" \
        --out-prefix roms/curr

Isso salva roms/curr_000.state, roms/curr_001.state, ... a cada F5.
Para um único arquivo com nome fixo, use --out:
    python scripts/make_init_state.py --out roms/init_state.state

Continuar a partir de um checkpoint já feito (sem rejogar do começo):
    python scripts/make_init_state.py --load roms/curr_002.state
    # começa nesse state; F9 recarrega ele a qualquer momento. Novos F5
    # continuam a numeração e F9 passa a recarregar o último salvo.

Requer: pygame (`pip install pygame`).
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Cria save states jogando (único ou currículo).")
    parser.add_argument("--rom", default="roms/Pokemon - Emerald Version (USA, Europe).gba")
    parser.add_argument("--out", default=None,
                        help="Caminho FIXO (sobrescreve a cada F5). Ex.: roms/init_state.state")
    parser.add_argument("--out-prefix", default="roms/curr",
                        help="Prefixo p/ states NUMERADOS (curr_000.state, ...). Usado se --out não for dado.")
    parser.add_argument("--load", default=None,
                        help="State a carregar no início e ao apertar F9. Ex.: roms/curr_002.state")
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    try:
        import pygame
    except ImportError:
        print("[erro] pygame não instalado. Rode: pip install pygame")
        return 1

    import mgba.image
    import mgba.log
    from mgba.gba import GBA
    from pygba import PyGBA

    mgba.log.silence()

    gba = PyGBA.load(args.rom)
    w, h = gba.core.desired_video_dimensions()
    fb = mgba.image.Image(w, h)
    gba.core.set_video_buffer(fb)
    gba.core.reset()

    key_map = {
        pygame.K_UP: GBA.KEY_UP,
        pygame.K_DOWN: GBA.KEY_DOWN,
        pygame.K_LEFT: GBA.KEY_LEFT,
        pygame.K_RIGHT: GBA.KEY_RIGHT,
        pygame.K_z: GBA.KEY_A,
        pygame.K_x: GBA.KEY_B,
        pygame.K_RETURN: GBA.KEY_START,
        pygame.K_BACKSPACE: GBA.KEY_SELECT,
        pygame.K_a: GBA.KEY_L,
        pygame.K_s: GBA.KEY_R,
    }

    # Numeração inicial dos checkpoints (não sobrescreve os já existentes).
    save_idx = 0
    if args.out is None:
        out_dir = os.path.dirname(args.out_prefix) or "."
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(args.out_prefix)
        while os.path.exists(f"{args.out_prefix}_{save_idx:03d}.state"):
            save_idx += 1

    # Caminho a recarregar com F9 (o --load, ou o último salvo nesta sessão).
    load_path = args.load

    def save_state() -> None:
        nonlocal save_idx, load_path
        raw = bytes(gba.core.save_raw_state())
        if args.out is not None:
            path = args.out
        else:
            path = f"{args.out_prefix}_{save_idx:03d}.state"
            save_idx += 1
        with open(path, "wb") as f:
            f.write(raw)
        load_path = path  # F9 passa a recarregar o último salvo
        print(f"[ok] state salvo em {path} ({len(raw)} bytes)")

    def load_state() -> None:
        if not load_path:
            print("[aviso] nada para carregar (use --load ou salve um state com F5).")
            return
        if not os.path.exists(load_path):
            print(f"[erro] state não encontrado: {load_path}")
            return
        with open(load_path, "rb") as f:
            gba.core.load_raw_state(f.read())
        gba.core.run_frame()
        print(f"[ok] state carregado: {load_path}")

    # Carrega o state inicial, se pedido.
    if args.load:
        load_state()

    pygame.init()
    screen = pygame.display.set_mode((w * args.scale, h * args.scale))
    pygame.display.set_caption("make_init_state — F5 salva, F9 carrega, ESC sai")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F5:
                    save_state()
                elif event.key == pygame.K_F9:
                    load_state()

        pressed = pygame.key.get_pressed()
        keys = [bit for k, bit in key_map.items() if pressed[k]]
        gba.core.set_keys(*keys)
        gba.core.run_frame()

        img = fb.to_pil().convert("RGB")
        surf = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
        surf = pygame.transform.scale(surf, (w * args.scale, h * args.scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
