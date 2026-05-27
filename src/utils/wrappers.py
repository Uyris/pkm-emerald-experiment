"""Wrappers Gymnasium opcionais para o PokemonEmeraldEnv.

Mantemos o ambiente base simples. Ponto de extensão para wrappers de
observação/recompensa no nível do ambiente único.

> **Frame stacking:** NÃO faça aqui com ``FrameStackObservation`` — ele cria um
> eixo novo (``(N, H, W, C)``, 4D) e a NatureCNN do SB3 rejeita. O stacking é
> feito no eixo de canais via ``VecFrameStack`` em
> :func:`src.utils.config.make_vec_env`.
"""

from __future__ import annotations

import gymnasium as gym


def apply_wrappers(env: gym.Env) -> gym.Env:
    """Aplica a pilha padrão de wrappers de ambiente único.

    Atualmente um no-op — ponto de extensão para wrappers futuros (ex.: sticky
    actions, recorte de tela, normalização de recompensa). O frame stacking é
    tratado no nível vetorizado (veja o módulo de config).
    """
    return env
