"""Smoke test: roda o ambiente com ações aleatórias, com visualização do jogo.

Verifica que ambiente, backend (real ou mock), pré-processamento e recompensa
funcionam end-to-end — e mostra a tela do jogo numa janela.

Uso:
    python -m src.play_random --config configs/default.yaml --steps 400
    python -m src.play_random --config configs/default.yaml --no-render
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.utils.config import load_config, make_env

WINDOW = "pokemon-emerald-rl [random]"


def _show(env, scale: int = 3) -> None:
    frame = env.render()
    if frame is None:
        return
    frame = np.asarray(frame)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(
        bgr,
        (frame.shape[1] * scale, frame.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow(WINDOW, bgr)
    cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda ações aleatórias no ambiente.")
    parser.add_argument("--config", default="configs/default.yaml", help="Caminho do YAML.")
    parser.add_argument("--steps", type=int, default=400, help="Nº de passos por episódio.")
    parser.add_argument("--episodes", type=int, default=1, help="Nº de episódios.")
    parser.add_argument("--seed", type=int, default=0, help="Seed do action_space.")
    parser.add_argument("--no-render", action="store_true", help="Desativa a janela.")
    args = parser.parse_args()

    config = load_config(args.config)
    render = not args.no_render
    env = make_env(config, render_mode="rgb_array" if render else None)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            env.action_space.seed(args.seed + ep)
            total_reward = 0.0

            for step in range(args.steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if render:
                    _show(env)
                if terminated or truncated:
                    break

            print(
                f"[ep {ep}] passos={step + 1} "
                f"recompensa_total={total_reward:.3f} "
                f"obs_shape={obs.shape} map_id={info.get('map_id')} "
                f"party={info.get('party_count')}"
            )
    finally:
        if render:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        env.close()


if __name__ == "__main__":
    main()
