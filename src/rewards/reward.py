"""Função de recompensa do PokemonEmeraldEnv.

:class:`EmeraldReward` combina componentes simples e extensíveis, pensados para
a **meta inicial** do projeto (sair do quarto → explorar Littleroot → pegar o
starter → chegar em Oldale Town):

* **Exploração**: bônus por visitar posições ``(map, x, y)`` inéditas no episódio.
* **Novos mapas**: bônus ao entrar em um ``map_id`` inédito (sair de casa,
  entrar na rota, chegar em Oldale...).
* **Starter / novo Pokémon**: bônus quando ``party_count`` aumenta (pegar o
  Pokémon inicial leva o time de 0 → 1).
* **Level**: recompensa proporcional ao aumento do maior level do time.
* **Badges**: bônus por cada nova badge.
* **Penalidade por passo**: pequeno custo por passo para incentivar progresso.

Princípio de robustez: **qualquer informação pode ser ``None``** (mapa de
memória indisponível, ex.: tela de título). Nesse caso o componente contribui
com ``0.0`` e o treino não quebra.
"""

from __future__ import annotations

from typing import Optional


class EmeraldReward:
    """Calcula a recompensa por passo a partir de snapshots de estado de jogo.

    Uso:
        reward_fn = EmeraldReward(exploration_weight=0.01, ...)
        reward_fn.reset()                 # início de cada episódio
        r = reward_fn.compute(info)       # a cada step, com o dict de info
    """

    def __init__(
        self,
        exploration_weight: float = 0.01,
        new_map_weight: float = 1.0,
        party_weight: float = 5.0,
        level_weight: float = 1.0,
        total_level_weight: float = 0.0,
        battle_damage_weight: float = 0.0,
        badge_weight: float = 10.0,
        event_weight: float = 0.0,
        step_penalty: float = 0.0,
        flag_milestones: Optional[list] = None,
    ) -> None:
        self.exploration_weight = exploration_weight
        self.new_map_weight = new_map_weight
        self.party_weight = party_weight
        self.level_weight = level_weight
        # Soma dos levels do time (estilo PWhiddy): cada level ganho (por vencer
        # batalha) pontua. Veja get_total_level.
        self.total_level_weight = total_level_weight
        # Dano causado: recompensa a QUEDA da fração de HP do inimigo. Sinal
        # DENSO de batalha (cada golpe que acerta pontua), que ensina a atacar
        # mesmo sem vencer a batalha inteira. Veja get_enemy_hp_fraction.
        self.battle_damage_weight = battle_damage_weight
        self.badge_weight = badge_weight
        # Progresso de história: recompensa por NOVA flag de evento setada
        # (estilo PWhiddy). Robusto contra "farm" de diálogo. Veja
        # EmeraldMemory.get_event_flag_count.
        self.event_weight = event_weight
        self.step_penalty = step_penalty
        # Marcos por flag de evento: lista de {"flag": int, "reward": float}.
        # Recompensa uma única vez por episódio, quando a flag liga (ex.: cutscene).
        self.flag_milestones = list(flag_milestones or [])

        self._visited_positions: set = set()
        self._visited_maps: set = set()
        self._prev_party_count: Optional[int] = None
        self._prev_max_level: Optional[int] = None
        self._prev_total_level: Optional[int] = None
        self._prev_badge_count: Optional[int] = None
        self._prev_enemy_hp_frac: Optional[float] = None
        self._max_event_count: Optional[int] = None
        self._fired_flags: set = set()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Zera o estado acumulado. Chame no início de cada episódio."""
        self._visited_positions = set()
        self._visited_maps = set()
        self._prev_party_count = None
        self._prev_max_level = None
        self._prev_total_level = None
        self._prev_badge_count = None
        self._prev_enemy_hp_frac = None
        self._max_event_count = None
        self._fired_flags = set()

    # ------------------------------------------------------------------ #
    def compute(self, info: Optional[dict]) -> float:
        """Calcula a recompensa do passo atual. ``info`` pode ser ``None``."""
        if not info:
            return -self.step_penalty

        reward = 0.0
        reward += self._exploration_reward(info)
        reward += self._new_map_reward(info)
        reward += self._party_reward(info)
        reward += self._level_reward(info)
        reward += self._total_level_reward(info)
        reward += self._battle_damage_reward(info)
        reward += self._badge_reward(info)
        reward += self._event_reward(info)
        reward += self._flag_milestone_reward(info)
        reward -= self.step_penalty
        return float(reward)

    def _event_reward(self, info: dict) -> float:
        """Recompensa por progresso de história (novas flags de evento setadas).

        Usa o MÁXIMO visto no episódio como base: só pontua quando a contagem
        ultrapassa esse máximo (robusto a flags TEMP que ligam/desligam). O 1º
        passo só calibra a base (flags já setadas no save state não pontuam).
        """
        if self.event_weight == 0:
            return 0.0
        count = info.get("event_flag_count")
        if count is None:
            return 0.0
        if self._max_event_count is None:
            self._max_event_count = count
            return 0.0
        if count > self._max_event_count:
            delta = count - self._max_event_count
            self._max_event_count = count
            return self.event_weight * delta
        return 0.0

    def _flag_milestone_reward(self, info: dict) -> float:
        """Recompensa (uma vez/episódio) quando uma flag de evento liga."""
        if not self.flag_milestones:
            return 0.0
        flags = info.get("flags") or {}
        total = 0.0
        for m in self.flag_milestones:
            fid = m.get("flag")
            if fid is None or fid in self._fired_flags:
                continue
            if flags.get(fid):  # True e não-None
                self._fired_flags.add(fid)
                total += float(m.get("reward", 1.0))
        return total

    # ------------------------------------------------------------------ #
    # Componentes
    # ------------------------------------------------------------------ #
    def _exploration_reward(self, info: dict) -> float:
        position = info.get("position")
        if position is None:
            return 0.0
        key = (info.get("map_id"), position)
        if key in self._visited_positions:
            return 0.0
        self._visited_positions.add(key)
        return self.exploration_weight

    def _new_map_reward(self, info: dict) -> float:
        map_id = info.get("map_id")
        if map_id is None or map_id in self._visited_maps:
            return 0.0
        self._visited_maps.add(map_id)
        return self.new_map_weight

    def _party_reward(self, info: dict) -> float:
        """Bônus por novos Pokémon no time (o starter é 0 → 1)."""
        count = info.get("party_count")
        if count is None:
            return 0.0
        if self._prev_party_count is None:
            self._prev_party_count = count
            return 0.0
        delta = count - self._prev_party_count
        self._prev_party_count = count
        return self.party_weight * delta if delta > 0 else 0.0

    def _level_reward(self, info: dict) -> float:
        level = info.get("max_level")
        if level is None:
            return 0.0
        if self._prev_max_level is None:
            self._prev_max_level = level
            return 0.0
        delta = level - self._prev_max_level
        self._prev_max_level = level
        return self.level_weight * delta if delta > 0 else 0.0

    def _total_level_reward(self, info: dict) -> float:
        """Recompensa por aumento da SOMA dos levels do time (sinal de batalha)."""
        if self.total_level_weight == 0:
            return 0.0
        level = info.get("total_level")
        if level is None:
            return 0.0
        if self._prev_total_level is None:
            self._prev_total_level = level
            return 0.0
        delta = level - self._prev_total_level
        self._prev_total_level = level
        return self.total_level_weight * delta if delta > 0 else 0.0

    def _battle_damage_reward(self, info: dict) -> float:
        """Recompensa a QUEDA da fração de HP do inimigo (dano causado).

        Derrotar um inimigo (fração 1.0 → 0.0) rende ~battle_damage_weight no
        total. Só pontua quedas: início de batalha/novo inimigo (fração sobe)
        não gera recompensa.
        """
        if self.battle_damage_weight == 0:
            return 0.0
        frac = info.get("enemy_hp_frac")
        if frac is None:
            self._prev_enemy_hp_frac = None  # saiu da batalha: zera a base
            return 0.0
        if self._prev_enemy_hp_frac is None:
            self._prev_enemy_hp_frac = frac
            return 0.0
        drop = self._prev_enemy_hp_frac - frac
        self._prev_enemy_hp_frac = frac
        return self.battle_damage_weight * drop if drop > 0 else 0.0

    def _badge_reward(self, info: dict) -> float:
        badges = info.get("badge_count")
        if badges is None:
            return 0.0
        if self._prev_badge_count is None:
            self._prev_badge_count = badges
            return 0.0
        delta = badges - self._prev_badge_count
        self._prev_badge_count = badges
        return self.badge_weight * delta if delta > 0 else 0.0
