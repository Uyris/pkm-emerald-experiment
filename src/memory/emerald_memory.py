"""Leitura de estado de jogo a partir da memória do emulador.

Traduz endereços/bytes brutos (via :class:`GbaBackend`) em informações de alto
nível usadas pela recompensa: contagem do time, level, badges, mapa atual e
posição do jogador.

Endereços (Pokémon Emerald — US/Europe, build padrão)
-----------------------------------------------------
Baseados na decompilação `pret/pokeemerald` e em RAM maps públicos. Alguns
campos vivem em endereços fixos de EWRAM; outros ficam dentro do ``SaveBlock1``,
acessado por um **ponteiro** (``gSaveBlock1Ptr``) que muda ao longo da execução
— por isso lemos o ponteiro primeiro e somamos os offsets.

> ⚠️ Se a sua ROM for de outra revisão/região, confirme os endereços com o
> debugger do mGBA. Todos os getters degradam para ``None`` quando a leitura é
> implausível, então o treino não quebra.

Referências:
    * gSaveBlock1Ptr = 0x03005D8C
    * gPlayerParty       = 0x020244EC
    * gPlayerPartyCount  = 0x020244E9
"""

from __future__ import annotations

from typing import Optional, Tuple

# --- Endereços fixos em EWRAM/IWRAM (Emerald US) ---
GSAVEBLOCK1_PTR = 0x03005D8C       # u32: ponteiro para SaveBlock1
GPLAYER_PARTY = 0x020244EC         # base do array gPlayerParty
GPLAYER_PARTY_COUNT = 0x020244E9   # u8: nº de Pokémon no time
GENEMY_PARTY = 0x02024744          # gEnemyParty (= gPlayerParty + 6*100); verificado

# --- Offsets dentro de SaveBlock1 ---
SB1_POS_X = 0x0000        # s16: coordenada X do jogador (em tiles)
SB1_POS_Y = 0x0002        # s16: coordenada Y do jogador
SB1_MAP_GROUP = 0x0004    # s8: grupo do mapa atual (WarpData.mapGroup)
SB1_MAP_NUM = 0x0005      # s8: número do mapa atual (WarpData.mapNum)
SB1_FLAGS = 0x1270        # base do array de flags (badges, eventos, ...)

# --- Estrutura do Pokémon no time (campos fora da área criptografada) ---
PARTY_MON_SIZE = 100      # bytes por struct Pokemon
PARTY_MON_LEVEL_OFFSET = 84   # u8: level
PARTY_MON_HP_OFFSET = 86      # u16: HP atual
PARTY_MON_MAXHP_OFFSET = 88   # u16: HP máximo
PARTY_SIZE = 6

# --- Flags de badges (Emerald): FLAG_BADGE01_GET = 0x867 ---
FIRST_BADGE_FLAG = 0x867

# Quantos bytes do array de flags ler ao fazer diff (cobre as flags relevantes;
# 0x140 bytes = 2560 flags, mais que suficiente para os eventos iniciais).
FLAGS_NUM_BYTES = 0x140

# Tamanho do array de flags do Emerald (FLAGS_COUNT/8 ~= 0x12C). Usado para
# contar quantas flags de evento estão setadas (sinal de progresso de história,
# estilo PWhiddy). Os primeiros bytes contêm flags TEMP (que ligam/desligam na
# jogatina normal); pulamos eles para reduzir ruído na contagem.
EVENT_FLAGS_NUM_BYTES = 0x12C
EVENT_FLAGS_SKIP_BYTES = 0x08  # ~64 primeiras flags (TEMP/diárias) ignoradas

# Faixa válida de EWRAM, usada para sanidade do ponteiro de SaveBlock1.
_EWRAM_LO = 0x02000000
_EWRAM_HI = 0x02040000


class EmeraldMemory:
    """Lê o estado do jogo a partir de um backend que implementa o contrato.

    Todos os métodos são tolerantes a falhas: se a leitura falhar ou der um
    valor implausível (ex.: ponteiro de SaveBlock1 ainda não inicializado, como
    na tela de título), o método devolve ``None``.
    """

    def __init__(self, backend) -> None:
        self.backend = backend

    # ------------------------------------------------------------------ #
    # Leituras de baixo nível com proteção
    # ------------------------------------------------------------------ #
    def _u8(self, address: int) -> Optional[int]:
        try:
            return int(self.backend.read_u8(address))
        except Exception:
            return None

    def _u16(self, address: int) -> Optional[int]:
        try:
            return int(self.backend.read_u16(address))
        except Exception:
            return None

    def _saveblock1(self) -> Optional[int]:
        """Endereço-base de SaveBlock1 (ou ``None`` se o ponteiro não é válido)."""
        try:
            ptr = int(self.backend.read_u32(GSAVEBLOCK1_PTR))
        except Exception:
            return None
        if _EWRAM_LO <= ptr < _EWRAM_HI:
            return ptr
        return None

    @staticmethod
    def _to_signed16(value: int) -> int:
        return value - 0x10000 if value >= 0x8000 else value

    # ------------------------------------------------------------------ #
    # Estado de alto nível
    # ------------------------------------------------------------------ #
    def get_party_count(self) -> Optional[int]:
        """Quantidade de Pokémon no time (0 antes de pegar o starter)."""
        count = self._u8(GPLAYER_PARTY_COUNT)
        if count is None or count > PARTY_SIZE:
            return None
        return count

    def _party_levels(self) -> list:
        """Lista de levels válidos do time (vazia se indisponível)."""
        count = self.get_party_count()
        if not count:
            return []
        levels = []
        for i in range(count):
            addr = GPLAYER_PARTY + i * PARTY_MON_SIZE + PARTY_MON_LEVEL_OFFSET
            lvl = self._u8(addr)
            if lvl and 1 <= lvl <= 100:
                levels.append(lvl)
        return levels

    def get_max_level(self) -> Optional[int]:
        """Maior level do time, ou ``None`` se o time estiver vazio/indisponível."""
        levels = self._party_levels()
        return max(levels) if levels else None

    def get_total_level(self) -> Optional[int]:
        """Soma dos levels do time (estilo PWhiddy) — sinal denso para batalha.

        Ganhar batalha sobe o level → a soma aumenta → recompensa. ``None`` se o
        time estiver vazio/indisponível.
        """
        levels = self._party_levels()
        return sum(levels) if levels else None

    def get_enemy_hp_fraction(self) -> Optional[float]:
        """Fração de HP do time INIMIGO (0.0 a 1.0), ou ``None`` se fora de batalha.

        Lê ``gEnemyParty`` (HP/maxHP são campos planos). Recompensar a QUEDA
        dessa fração dá um sinal DENSO de dano: cada golpe que acerta reduz o HP
        → reward, derrotar o inimigo → fração chega a 0. Fora de batalha o
        ``maxHP`` total costuma ser 0 (ou dados antigos estáticos) → ``None``/sem
        variação, então não gera recompensa espúria.
        """
        cur_total = 0
        max_total = 0
        for i in range(PARTY_SIZE):
            base = GENEMY_PARTY + i * PARTY_MON_SIZE
            mh = self._u16(base + PARTY_MON_MAXHP_OFFSET)
            if mh is None:
                return None
            if mh == 0 or mh > 999:  # slot vazio / valor implausível
                continue
            hp = self._u16(base + PARTY_MON_HP_OFFSET)
            if hp is None:
                return None
            cur_total += min(hp, mh)
            max_total += mh
        if max_total == 0:
            return None
        return cur_total / max_total

    def get_badge_count(self) -> Optional[int]:
        """Número de badges conquistadas (0..8), ou ``None``."""
        base = self._saveblock1()
        if base is None:
            return None
        count = 0
        for i in range(8):
            flag = FIRST_BADGE_FLAG + i
            byte = self._u8(base + SB1_FLAGS + (flag >> 3))
            if byte is None:
                return None
            if byte & (1 << (flag & 7)):
                count += 1
        return count

    def get_flag(self, flag_id: int) -> Optional[bool]:
        """Lê uma flag de evento por id (ex.: cutscene), ou ``None``.

        As flags ficam num bitarray em ``SaveBlock1 + SB1_FLAGS``: o byte é
        ``flag_id >> 3`` e o bit é ``flag_id & 7``.
        """
        base = self._saveblock1()
        if base is None:
            return None
        byte = self._u8(base + SB1_FLAGS + (flag_id >> 3))
        if byte is None:
            return None
        return bool(byte & (1 << (flag_id & 7)))

    def read_flag_bytes(self, count: int = FLAGS_NUM_BYTES) -> Optional[bytes]:
        """Lê ``count`` bytes crus do array de flags (leitura em bloco)."""
        base = self._saveblock1()
        if base is None:
            return None
        try:
            data = self.backend.read_memory(base + SB1_FLAGS, count)
        except Exception:
            return None
        if data is None:
            return None
        return bytes(data)

    def get_event_flag_count(self) -> Optional[int]:
        """Número de flags de evento setadas — proxy de progresso de história.

        Conta os bits ligados no array de flags (pulando as TEMP iniciais). Não
        é "farmável" conversando com NPC: a maioria das interações não seta flag
        permanente; só eventos de progresso setam. ``None`` se indisponível.
        """
        data = self.read_flag_bytes(EVENT_FLAGS_NUM_BYTES)
        if data is None:
            return None
        return sum(bin(b).count("1") for b in data[EVENT_FLAGS_SKIP_BYTES:])

    def get_map_id(self) -> Optional[Tuple[int, int]]:
        """Identificador do mapa atual ``(group, num)``, ou ``None``."""
        base = self._saveblock1()
        if base is None:
            return None
        group = self._u8(base + SB1_MAP_GROUP)
        num = self._u8(base + SB1_MAP_NUM)
        if group is None or num is None:
            return None
        return (group, num)

    def get_position(self) -> Optional[Tuple[int, int]]:
        """Posição do jogador ``(x, y)`` em tiles, ou ``None``."""
        base = self._saveblock1()
        if base is None:
            return None
        x = self._u16(base + SB1_POS_X)
        y = self._u16(base + SB1_POS_Y)
        if x is None or y is None:
            return None
        return (self._to_signed16(x), self._to_signed16(y))

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        """Dicionário com todo o estado de jogo conhecido (valores podem ser ``None``)."""
        return {
            "party_count": self.get_party_count(),
            "max_level": self.get_max_level(),
            "total_level": self.get_total_level(),
            "badge_count": self.get_badge_count(),
            "map_id": self.get_map_id(),
            "position": self.get_position(),
        }
