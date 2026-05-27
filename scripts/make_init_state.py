"""Joga a ROM interativamente para criar o save state inicial do treino.

A IA não consegue navegar a introdução / criação de novo save do Pokémon
Emerald (logos, gênero, nome, relógio). Use este script para jogar com o
teclado até o ponto desejado — idealmente **dentro do quarto inicial, com
controle do personagem** — e salvar um state. Depois aponte
``emulator.init_state`` (em configs/default.yaml) para esse arquivo.

Controles:
    Setas      → D-Pad
    Z          → A
    X          → B
    Enter      → START
    Backspace  → SELECT
    A / S      → L / R
    F5         → salvar state no caminho de saída
    ESC        → sair

Uso:
    python scripts/make_init_state.py \
        --rom "roms/Pokemon - Emerald Version (USA, Europe).gba" \
        --out roms/init_state.state

Requer: pygame (`pip install pygame`).
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Cria um save state inicial jogando.")
    parser.add_argument("--rom", default="roms/Pokemon - Emerald Version (USA, Europe).gba")
    parser.add_argument("--out", default="roms/init_state.state")
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

    pygame.init()
    screen = pygame.display.set_mode((w * args.scale, h * args.scale))
    pygame.display.set_caption("make_init_state — F5 salva, ESC sai")
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
                    raw = bytes(gba.core.save_raw_state())
                    with open(args.out, "wb") as f:
                        f.write(raw)
                    print(f"[ok] state salvo em {args.out} ({len(raw)} bytes)")

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
