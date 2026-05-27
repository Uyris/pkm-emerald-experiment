"""Pré-processamento de observações (telas do emulador) com OpenCV.

Converte um frame RGB do GBA em uma observação compacta para a CNN:
grayscale, redimensionada para ``(height, width)`` e com canal explícito,
no formato ``(H, W, 1)`` ``uint8`` — compatível com a ``CnnPolicy`` do SB3.
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    width: int = 84,
    height: int = 84,
) -> np.ndarray:
    """Converte um frame RGB em observação grayscale ``(height, width, 1)``.

    Args:
        frame: imagem ``(H, W, 3)`` (RGB) ou ``(H, W)`` (já grayscale), ``uint8``.
        width: largura final.
        height: altura final.

    Returns:
        ``np.ndarray`` de shape ``(height, width, 1)`` e dtype ``uint8``.
    """
    if frame is None:
        raise ValueError("frame não pode ser None")

    frame = np.asarray(frame)

    # Para grayscale, se necessário.
    if frame.ndim == 3 and frame.shape[-1] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        gray = frame[..., 0]
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(f"Shape de frame não suportado: {frame.shape}")

    # Redimensiona (INTER_AREA é bom para reduzir tamanho).
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    # Garante uint8 e adiciona o canal.
    obs = resized.astype(np.uint8)
    return obs[..., np.newaxis]
