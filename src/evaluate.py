"""Avaliação de um modelo PPO salvo, com visualização ao vivo do jogo.

Uso:
    python -m src.evaluate --config configs/default.yaml --model models/ppo_emerald.zip
    python -m src.evaluate --config configs/default.yaml --no-render
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from stable_baselines3 import PPO

from src.utils.config import load_config, make_vec_env

WINDOW = "pokemon-emerald-rl [eval]"


def _show(env, scale: int = 3) -> None:
    """Desenha a tela do ambiente 0 numa janela OpenCV."""
    frames = env.env_method("render", indices=[0])
    if not frames or frames[0] is None:
        return
    frame = np.asarray(frames[0])
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(
        bgr,
        (frame.shape[1] * scale, frame.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow(WINDOW, bgr)
    cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um modelo PPO salvo.")
    parser.add_argument("--config", default="configs/default.yaml", help="Caminho do YAML.")
    parser.add_argument("--model", default="models/ppo_emerald.zip", help="Caminho do modelo.")
    parser.add_argument("--episodes", type=int, default=None, help="Sobrescreve eval.n_episodes.")
    parser.add_argument("--no-render", action="store_true", help="Desativa a janela de visualização.")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    n_episodes = args.episodes or eval_cfg.get("n_episodes", 5)
    deterministic = eval_cfg.get("deterministic", True)
    render = not args.no_render

    # Avaliação roda 1 ambiente (n_envs=1), mesma construção vetorizada do treino.
    config.setdefault("train", {})["n_envs"] = 1
    env = make_vec_env(config)

    model = PPO.load(args.model, env=env)

    rewards = []
    try:
        for ep in range(n_episodes):
            obs = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, dones, infos = env.step(action)
                total_reward += float(reward[0])
                done = bool(dones[0])
                if render:
                    _show(env, scale=eval_cfg.get("render_scale", 3))
            rewards.append(total_reward)
            print(f"[ep {ep}] recompensa_total={total_reward:.3f}")
    finally:
        if render:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        env.close()

    print(
        f"\nMédia em {n_episodes} episódios: "
        f"{np.mean(rewards):.3f} ± {np.std(rewards):.3f}"
    )


if __name__ == "__main__":
    main()
