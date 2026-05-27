"""Callbacks de treino: visualização ao vivo e métricas de jogo.

* :class:`RenderCallback` — abre uma janela OpenCV mostrando a tela real do GBA
  **durante o treino**. Por padrão monta um mosaico com **todos** os ambientes
  (``show_all=True``); pode mostrar só o ambiente 0 (``show_all=False``).
  Funciona com ``DummyVecEnv`` e ``SubprocVecEnv`` (os frames são serializados
  ao cruzar a fronteira de processo).
* :class:`GameStatsCallback` — registra métricas do jogo (mapas visitados,
  party_count, level, badges, posição) no TensorBoard.
"""

from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _tile_frames(frames: List[np.ndarray], scale: int) -> Optional[np.ndarray]:
    """Monta um mosaico (grade) BGR a partir de uma lista de frames RGB.

    A grade é o mais quadrada possível; células faltantes ficam pretas. Cada
    célula recebe um rótulo com o índice do ambiente.
    """
    frames = [f for f in frames if f is not None]
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_w, cell_h = w * scale, h * scale
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        if scale != 1:
            bgr = cv2.resize(bgr, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, cols)
        y0, x0 = r * cell_h, c * cell_w
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = bgr
        cv2.putText(
            canvas, f"env {i}", (x0 + 4, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return canvas


class RenderCallback(BaseCallback):
    """Mostra os ambientes numa janela OpenCV durante o treino.

    Args:
        render_freq: a cada quantos passos de treino atualizar a janela.
            Com muitos ambientes, valores >1 reduzem o overhead.
        scale: fator de ampliação por célula (GBA é 240x160).
        show_all: ``True`` (default) → mosaico com todos os envs; ``False`` →
            só o ambiente 0.
        window_name: título da janela.
    """

    def __init__(
        self,
        render_freq: int = 1,
        scale: int = 3,
        show_all: bool = True,
        window_name: str = "pokemon-emerald-rl",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.render_freq = max(1, render_freq)
        self.scale = scale
        self.show_all = show_all
        self.window_name = window_name
        self._window_ready = False
        self._disabled = False

    def _grab_frames(self) -> Optional[List[np.ndarray]]:
        """Pega os frames RGB (todos os envs, ou só o 0)."""
        try:
            if self.show_all:
                # get_images() chama render("rgb_array") em cada env.
                frames = self.training_env.get_images()
            else:
                frames = self.training_env.env_method("render", indices=[0])
        except Exception:
            return None
        if not frames:
            return None
        return [np.asarray(f) for f in frames if f is not None]

    def _on_step(self) -> bool:
        if self._disabled or self.n_calls % self.render_freq != 0:
            return True

        frames = self._grab_frames()
        if not frames:
            return True

        canvas = _tile_frames(frames, self.scale) if self.show_all else (
            cv2.resize(
                cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR),
                (frames[0].shape[1] * self.scale, frames[0].shape[0] * self.scale),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        if canvas is None:
            return True

        cv2.putText(
            canvas, f"steps={self.num_timesteps}", (6, canvas.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )

        try:
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)
            self._window_ready = True
        except cv2.error as e:
            # Sem suporte a GUI (build headless do OpenCV / sem display):
            # desativa a visualização e segue o treino normalmente.
            self._disabled = True
            print(f"[RenderCallback] visualização desativada (OpenCV sem GUI): {e}")
        return True

    def _on_training_end(self) -> None:
        if self._window_ready:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)


class GameStatsCallback(BaseCallback):
    """Registra métricas do jogo (do ``info`` do ambiente) no TensorBoard."""

    def __init__(self, log_freq: int = 256, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = max(1, log_freq)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        infos = self.locals.get("infos")
        if not infos:
            return True

        info = infos[0]
        party = info.get("party_count")
        level = info.get("max_level")
        badges = info.get("badge_count")
        map_id = info.get("map_id")

        if party is not None:
            self.logger.record("game/party_count", party)
        if level is not None:
            self.logger.record("game/max_level", level)
        if badges is not None:
            self.logger.record("game/badge_count", badges)
        if map_id is not None:
            # Identificador numérico estável para o mapa (group*1000 + num).
            self.logger.record("game/map_code", map_id[0] * 1000 + map_id[1])
        return True
