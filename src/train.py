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
import copy
import datetime
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure

from src.utils.callbacks import (
    EntCoefDecayCallback,
    GameStatsCallback,
    RenderCallback,
)
from src.utils.config import load_config, make_vec_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina PPO no PokemonEmeraldEnv.")
    parser.add_argument("--config", default="configs/default.yaml", help="Caminho do YAML.")
    parser.add_argument("--no-render", action="store_true", help="Desativa a janela de visualização.")
    parser.add_argument("--wandb", action="store_true", help="Ativa logging no Weights & Biases.")
    parser.add_argument("--resume", action="store_true",
                        help="Continua o treino a partir do modelo em train.save_path (transfere o aprendizado).")
    parser.add_argument("--resume-from", default=None,
                        help="Continua a partir deste modelo .zip específico.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("train", {})
    render_enabled = train_cfg.get("render", True) and not args.no_render

    # VecEnv (paralelo conforme n_envs) + frame stacking no eixo de canais.
    env = make_vec_env(config)

    save_path = train_cfg.get("save_path", "models/ppo_emerald.zip")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tensorboard_log = train_cfg.get("tensorboard_log", "tensorboard/")

    # --- Resumir treino: carrega pesos existentes (transfere o que já aprendeu) ---
    resume_path = args.resume_from or (save_path if args.resume else None)
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"--resume pedido mas o modelo não existe: {resume_path}. "
                "Treine uma primeira vez (sem --resume) ou cheque o caminho."
            )
        print(f"[resume] Carregando pesos de {resume_path} e continuando o treino.")
        model = PPO.load(resume_path, env=env, tensorboard_log=tensorboard_log, verbose=1)
    else:
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

    # --- Logger: salva TODOS os logs do run em arquivo (além do console) ---
    # Gera, por run, uma pasta com:
    #   log.txt       -> as mesmas tabelas que aparecem no terminal (completas)
    #   progress.csv  -> métricas por iteração (fácil de plotar/analisar)
    #   eventos do TensorBoard
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(train_cfg.get("log_dir", "logs"), f"PPO_{run_name}")
    model.set_logger(configure(run_dir, ["stdout", "log", "csv", "tensorboard"]))
    print(f"[logs] salvando o run em {run_dir} (log.txt, progress.csv, tensorboard)")

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

    total_timesteps = train_cfg.get("total_timesteps", 1_000_000)

    # Decaimento de ent_coef (explora cedo, exploita no fim) — se configurado.
    ent_coef_final = train_cfg.get("ent_coef_final")
    if ent_coef_final is not None:
        ent_coef_start = train_cfg.get("ent_coef", 0.01)
        callbacks.append(
            EntCoefDecayCallback(
                initial=ent_coef_start,
                final=ent_coef_final,
                total_timesteps=total_timesteps,
            )
        )
        print(f"[ent_coef] decaindo de {ent_coef_start} -> {ent_coef_final} ao longo de {total_timesteps} passos.")

    # --- Melhor modelo: avalia periodicamente e salva models/best_model.zip ---
    # Resolve o problema "o modelo final é pior que o pico": guarda o melhor por
    # recompensa de avaliação, independente de quando o treino degradar.
    eval_env = None
    if train_cfg.get("eval_best", True):
        n_envs = max(1, int(train_cfg.get("n_envs", 1)))
        eval_config = copy.deepcopy(config)
        eval_config.setdefault("train", {})["n_envs"] = 1  # avaliação com 1 env
        eval_env = make_vec_env(eval_config, render_mode=None)
        best_dir = os.path.dirname(save_path) or "."
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=best_dir,
                log_path=run_dir,
                eval_freq=max(int(train_cfg.get("eval_freq", 50000)) // n_envs, 1),
                n_eval_episodes=int(train_cfg.get("n_eval_episodes", 3)),
                deterministic=config.get("eval", {}).get("deterministic", True),
                render=False,
            )
        )
        print(f"[eval] melhor modelo -> {os.path.join(best_dir, 'best_model.zip')} "
              f"(a cada ~{train_cfg.get('eval_freq', 50000)} passos)")

    if args.wandb or train_cfg.get("wandb", False):
        _maybe_add_wandb(callbacks, config)

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    model.save(save_path)
    print(f"\nModelo salvo em: {save_path}")
    if train_cfg.get("eval_best", True):
        print(f"Melhor modelo (por avaliação): {os.path.join(os.path.dirname(save_path) or '.', 'best_model.zip')}")
    env.close()
    if eval_env is not None:
        eval_env.close()


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
