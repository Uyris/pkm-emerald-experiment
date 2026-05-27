"""Treino com Stable-Baselines3 PPO + CnnPolicy.

Uso:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --no-render   # sem janela
    python -m src.train --config configs/default.yaml --wandb       # + W&B

Recursos:
    * Observação ``(84, 84, N)`` (frame stack) → ``CnnPolicy``.
    * ``train.n_envs`` controla paralelismo (DummyVecEnv / SubprocVecEnv).
    * Visualização ao vivo do jogo (janela OpenCV do ambiente 0).
    * Logging em TensorBoard (+ Weights & Biases opcional).
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from src.utils.callbacks import GameStatsCallback, RenderCallback
from src.utils.config import load_config, make_vec_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina PPO no PokemonEmeraldEnv.")
    parser.add_argument("--config", default="configs/default.yaml", help="Caminho do YAML.")
    parser.add_argument("--no-render", action="store_true", help="Desativa a janela de visualização.")
    parser.add_argument("--wandb", action="store_true", help="Ativa logging no Weights & Biases.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("train", {})
    render_enabled = train_cfg.get("render", True) and not args.no_render

    # VecEnv (paralelo conforme n_envs) + frame stacking no eixo de canais.
    env = make_vec_env(config)

    save_path = train_cfg.get("save_path", "models/ppo_emerald.zip")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tensorboard_log = train_cfg.get("tensorboard_log", "tensorboard/")

    model = PPO(
        policy=train_cfg.get("policy", "CnnPolicy"),
        env=env,
        n_steps=train_cfg.get("n_steps", 2048),
        batch_size=train_cfg.get("batch_size", 256),
        n_epochs=train_cfg.get("n_epochs", 4),
        gamma=train_cfg.get("gamma", 0.999),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        ent_coef=train_cfg.get("ent_coef", 0.01),
        learning_rate=train_cfg.get("learning_rate", 2.5e-4),
        clip_range=train_cfg.get("clip_range", 0.2),
        seed=train_cfg.get("seed", 42),
        tensorboard_log=tensorboard_log,
        verbose=1,
    )

    # ---- Callbacks ----
    callbacks = [
        CheckpointCallback(
            save_freq=max(train_cfg.get("checkpoint_freq", 50000) // max(1, train_cfg.get("n_envs", 1)), 1),
            save_path="checkpoints/",
            name_prefix="ppo_emerald",
        ),
        GameStatsCallback(log_freq=train_cfg.get("stats_log_freq", 256)),
    ]
    if render_enabled:
        callbacks.append(
            RenderCallback(
                render_freq=train_cfg.get("render_freq", 1),
                scale=train_cfg.get("render_scale", 3),
                show_all=train_cfg.get("render_all_envs", True),
            )
        )

    if args.wandb or train_cfg.get("wandb", False):
        _maybe_add_wandb(callbacks, config)

    model.learn(
        total_timesteps=train_cfg.get("total_timesteps", 1_000_000),
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    model.save(save_path)
    print(f"\nModelo salvo em: {save_path}")
    env.close()


def _maybe_add_wandb(callbacks: list, config: dict) -> None:
    """Adiciona o callback do W&B se o pacote estiver instalado."""
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
    except ImportError:
        print("[aviso] wandb não instalado; ignorando --wandb. "
              "Instale com `pip install wandb`.")
        return

    run = wandb.init(
        project=config.get("train", {}).get("wandb_project", "pokemon-emerald-rl"),
        config=config,
        sync_tensorboard=True,
        save_code=True,
    )
    callbacks.append(WandbCallback(verbose=1))
    print(f"[wandb] run iniciada: {run.url}")


if __name__ == "__main__":
    main()
