"""Camada de integração com o emulador de Game Boy Advance.

Esta é a *única* parte do projeto que conhece detalhes do emulador. Tudo acima
dela (ambiente Gymnasium, recompensas, treino) fala apenas com a interface
:class:`GbaBackend`.

Duas implementações:

* :class:`GbaBackend` — **real**, sobre PyGBA/mGBA. Requer a ROM e o binding
  Python ``mgba`` (veja ``scripts/install_mgba.py`` e o README).
* :class:`MockGbaBackend` — **placeholder** sem ROM/emulador. Gera telas de
  ruído e memória zerada; usado nos testes e para rodar o pipeline a seco.

Contrato (ambas respeitam):
    * :meth:`get_screen` → ``np.ndarray`` ``(H, W, 3)`` ``uint8`` (RGB).
    * :meth:`read_u8/read_u16/read_u32` → ``int`` (ou ``None`` se indisponível).
    * botões são strings de :data:`BUTTONS`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Resolução nativa da tela do GBA (largura x altura).
GBA_WIDTH = 240
GBA_HEIGHT = 160

# Botões lógicos expostos para as camadas superiores.
BUTTONS = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "L", "R")

# Mapeamento dos botões lógicos para os nomes de tecla do PyGBA (KEY_MAP).
_PYGBA_KEY = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "A": "A",
    "B": "B",
    "START": "start",
    "SELECT": "select",
    "L": "L",
    "R": "R",
}


def _validate_button(button: str) -> None:
    if button not in BUTTONS:
        raise ValueError(f"Botão inválido: {button!r}. Use um de {BUTTONS}.")


class GbaBackend:
    """Backend real sobre PyGBA/mGBA.

    Os imports de ``pygba``/``mgba`` são *lazy* (feitos no ``__init__``), para
    que importar este módulo não exija o emulador — só instanciar a classe
    exige. Isso mantém os testes (que usam :class:`MockGbaBackend`) leves.

    Args:
        rom_path: caminho da ROM ``.gba`` (obrigatório).
        init_state: caminho de um save state aplicado no :meth:`reset` (ex.:
            começar já dentro do quarto). Opcional.
        headless: ignorado aqui (o backend nunca abre janela; a visualização é
            feita por fora via :meth:`get_screen`).
        seed: aceito por compatibilidade de assinatura; não usado.
    """

    def __init__(
        self,
        rom_path: Optional[str] = None,
        init_state: Optional[str] = None,
        headless: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if not rom_path:
            raise ValueError(
                "GbaBackend requer rom_path. Para rodar sem ROM (testes), "
                "use MockGbaBackend."
            )

        # Imports locais + silêncio do logger do mGBA (evita poluir o console).
        import mgba.image
        import mgba.log
        from pygba import PyGBA
        from pygba.utils import KEY_MAP

        mgba.log.silence()

        self._KEY_MAP = KEY_MAP
        self.rom_path = rom_path
        self.init_state = init_state

        self._gba = PyGBA.load(rom_path)
        self._core = self._gba.core
        # Buffer de vídeo: precisa existir antes do reset para capturar a tela.
        self._framebuffer = mgba.image.Image(*self._core.desired_video_dimensions())
        self._core.set_video_buffer(self._framebuffer)
        self._core.reset()

        # Estado inicial opcional (carregado de arquivo).
        self._init_state_bytes: Optional[bytes] = None
        if init_state:
            with open(init_state, "rb") as f:
                self._init_state_bytes = f.read()
        # Estado pendente p/ o próximo reset (usado pelo currículo do env).
        self._pending_state: Optional[bytes] = None

        self._pressed: set[str] = set()

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def set_state(self, raw: Optional[bytes]) -> None:
        """Define o save state (bytes) a aplicar no PRÓXIMO :meth:`reset`.

        Usado pelo currículo: o env sorteia um estado e o injeta antes do reset.
        ``None`` volta ao comportamento padrão (usa o ``init_state`` do construtor).
        """
        self._pending_state = raw

    def reset(self) -> None:
        """Reinicia o emulador e aplica o estado pendente / ``init_state``."""
        self._pressed.clear()
        self._core.reset()
        state = self._pending_state if self._pending_state is not None else self._init_state_bytes
        if state is not None:
            self._core.load_raw_state(state)
            self._core.run_frame()  # renderiza o estado carregado

    def close(self) -> None:
        """Libera o core do emulador."""
        # mGBA libera recursos via GC; nada crítico a fazer aqui.
        self._pressed.clear()

    # ------------------------------------------------------------------ #
    # Avanço de tempo
    # ------------------------------------------------------------------ #
    def step_frame(self, frames: int = 1) -> None:
        """Avança ``frames`` frames mantendo os botões atualmente pressionados."""
        if frames < 1:
            raise ValueError("frames deve ser >= 1")
        keys = [self._KEY_MAP[_PYGBA_KEY[b]] for b in self._pressed]
        self._core.set_keys(*keys)
        for _ in range(frames):
            self._core.run_frame()

    # ------------------------------------------------------------------ #
    # Entrada (botões)
    # ------------------------------------------------------------------ #
    def press_button(self, button: str) -> None:
        _validate_button(button)
        self._pressed.add(button)

    def release_button(self, button: str) -> None:
        _validate_button(button)
        self._pressed.discard(button)

    def release_all(self) -> None:
        self._pressed.clear()
        self._core.set_keys()

    # ------------------------------------------------------------------ #
    # Saída (tela / memória)
    # ------------------------------------------------------------------ #
    def get_screen(self) -> np.ndarray:
        """Frame atual como ``np.ndarray`` ``(H, W, 3)`` ``uint8`` (RGB)."""
        return np.array(self._framebuffer.to_pil().convert("RGB"), dtype=np.uint8)

    def read_u8(self, address: int) -> int:
        return self._gba.read_u8(address)

    def read_u16(self, address: int) -> int:
        return self._gba.read_u16(address)

    def read_u32(self, address: int) -> int:
        return self._gba.read_u32(address)

    def read_memory(self, address: int, size: int = 1):
        """Lê ``size`` bytes a partir de ``address`` (little-endian → int)."""
        if size == 1:
            return self.read_u8(address)
        if size == 2:
            return self.read_u16(address)
        if size == 4:
            return self.read_u32(address)
        return self._gba.read_memory(address, size)

    # ------------------------------------------------------------------ #
    # Save states
    # ------------------------------------------------------------------ #
    def save_state(self, path: str) -> None:
        """Salva o estado atual do emulador em ``path``."""
        raw = bytes(self._core.save_raw_state())
        with open(path, "wb") as f:
            f.write(raw)

    def load_state(self, path: str) -> None:
        """Carrega um save state de ``path``."""
        with open(path, "rb") as f:
            self._core.load_raw_state(f.read())
        self._core.run_frame()

    # ------------------------------------------------------------------ #
    @property
    def pressed_buttons(self) -> frozenset[str]:
        return frozenset(self._pressed)


class MockGbaBackend:
    """Backend placeholder: sem ROM, sem emulador.

    Gera telas de ruído determinístico (a partir de ``seed``) e memória zerada.
    Usado nos testes e para validar o pipeline (env, wrappers, recompensa) sem
    depender do emulador. Respeita o mesmo contrato de :class:`GbaBackend`.
    """

    def __init__(
        self,
        rom_path: Optional[str] = None,
        init_state: Optional[str] = None,
        headless: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.rom_path = rom_path
        self.init_state = init_state
        self.headless = headless
        self._rng = np.random.default_rng(seed)
        self._pressed: set[str] = set()
        self._frame_count = 0

    # Ciclo de vida
    def set_state(self, raw: Optional[bytes]) -> None:
        """No-op no mock (mantém o contrato do currículo)."""
        return None

    def reset(self) -> None:
        self._pressed.clear()
        self._frame_count = 0

    def close(self) -> None:
        self._pressed.clear()

    # Tempo
    def step_frame(self, frames: int = 1) -> None:
        if frames < 1:
            raise ValueError("frames deve ser >= 1")
        self._frame_count += frames

    # Botões
    def press_button(self, button: str) -> None:
        _validate_button(button)
        self._pressed.add(button)

    def release_button(self, button: str) -> None:
        _validate_button(button)
        self._pressed.discard(button)

    def release_all(self) -> None:
        self._pressed.clear()

    # Saída
    def get_screen(self) -> np.ndarray:
        return self._rng.integers(
            0, 256, size=(GBA_HEIGHT, GBA_WIDTH, 3), dtype=np.uint8
        )

    def read_u8(self, address: int) -> int:
        return 0

    def read_u16(self, address: int) -> int:
        return 0

    def read_u32(self, address: int) -> int:
        return 0

    def read_memory(self, address: int, size: int = 1):
        if address < 0:
            raise ValueError("address deve ser >= 0")
        return 0 if size in (1, 2, 4) else bytes(size)

    # Save states (no-op no mock)
    def save_state(self, path: str) -> None:
        raise NotImplementedError("MockGbaBackend não persiste save states.")

    def load_state(self, path: str) -> None:
        raise NotImplementedError("MockGbaBackend não carrega save states.")

    @property
    def pressed_buttons(self) -> frozenset[str]:
        return frozenset(self._pressed)
