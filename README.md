# pokemon-emerald-rl

Treinamento de uma IA com **Deep Reinforcement Learning** para jogar **Pokémon Emerald** (Game Boy Advance).

A base é compatível com [Gymnasium](https://gymnasium.farama.org/) e [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), usando **PPO** como primeiro algoritmo. A integração com o emulador é isolada em uma camada própria (`src/emulator/`), pensada para [PyGBA](https://github.com/davidwhitman/pygba) / [mGBA](https://mgba.io/).

> ⚠️ **Sem ROMs.** Este repositório **não** contém nem distribui ROMs. Você deve fornecer sua própria cópia legal do jogo. Veja [Configuração da ROM](#configuração-da-rom).

---

## 🎯 Meta inicial do projeto

O objetivo **não** é resolver o jogo inteiro de uma vez. A primeira meta concreta de treinamento é fazer o agente:

1. **Sair do quarto** inicial (casa do jogador).
2. **Explorar Littleroot Town**.
3. **Pegar o Pokémon inicial (starter)**.
4. **Chegar em Oldale Town**.

A partir daí, a estrutura modular permite estender recompensas e objetivos.

---

## 🗂️ Estrutura

```
pokemon-emerald-rl/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml          # Hiperparâmetros e configuração do ambiente
├── src/
│   ├── envs/
│   │   └── emerald_env.py     # PokemonEmeraldEnv (Gymnasium)
│   ├── emulator/
│   │   └── gba_backend.py     # GbaBackend — camada de integração com o emulador
│   ├── memory/
│   │   └── emerald_memory.py  # Leitura de endereços de memória (level, badges, mapa...)
│   ├── rewards/
│   │   └── reward.py          # EmeraldReward — função de recompensa
│   ├── utils/
│   │   ├── preprocessing.py   # Pré-processamento de tela (grayscale 84x84)
│   │   ├── wrappers.py        # Wrappers Gymnasium (ponto de extensão)
│   │   ├── config.py          # Loader YAML + fábricas make_env / make_vec_env
│   │   └── callbacks.py       # Visualização ao vivo + métricas no TensorBoard
│   ├── play_random.py         # Roda ações aleatórias (smoke test)
│   ├── train.py               # Treino com SB3 PPO + CnnPolicy
│   └── evaluate.py            # Carrega e avalia um modelo salvo
├── scripts/
│   ├── install_mgba.py        # Baixa o binding Python pré-compilado do mGBA
│   └── make_init_state.py     # Joga interativamente p/ criar o save state inicial
└── tests/
    ├── test_env_reset.py
    ├── test_env_step.py
    ├── test_reward.py
    └── test_memory.py
```

---

## 🚀 Instalação

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Binding Python do mGBA (obrigatório para o emulador real)

O `pygba` dirige o mGBA através do módulo Python `mgba`, que **não tem wheel no
PyPI para Windows**. Instale o binding pré-compilado (do projeto
[libmgba-py](https://github.com/hanzi/libmgba-py)) com o script incluso:

```bash
python scripts/install_mgba.py
# valide:
python -c "import mgba.core; print('mgba OK')"
```

> Os binários do libmgba-py 0.2.0 são compilados para **Python 3.10**. Use um
> ambiente 3.10. O script já instala no `site-packages` do venv ativo; o `cffi`
> (em `requirements.txt`) é exigido pelo binding.

Para a visualização ao vivo basta o `opencv-python` (já no `requirements.txt`).
O script `make_init_state.py` usa `pygame` (`pip install pygame`).

---

## 🎮 Configuração da ROM e do estado inicial

1. Forneça sua própria ROM legal de Pokémon Emerald (`.gba`) em `roms/`
   (o `.gitignore` já ignora `*.gba` e `roms/`).
2. Aponte o caminho em [configs/default.yaml](configs/default.yaml):

```yaml
emulator:
  rom_path: "roms/Pokemon - Emerald Version (USA, Europe).gba"
  init_state: "roms/init_state.state"   # recomendado (veja abaixo)
```

**Estado inicial (recomendado).** A IA não navega o intro / criação de save
(logos, gênero, nome, relógio). Crie um save state já **dentro do quarto
inicial** e aponte `init_state` para ele:

```bash
pip install pygame
python scripts/make_init_state.py \
  --rom "roms/Pokemon - Emerald Version (USA, Europe).gba" \
  --out roms/init_state.state
# Jogue com setas + Z/X/Enter; pressione F5 para salvar o state; ESC para sair.
```

Sem `init_state`, o treino começa do boot da ROM (a IA dificilmente passa do
intro) — útil só para testar o pipeline.

---

## 🎓 Currículo de save states (metas distantes)

Para metas longe do início (ex.: **pegar o starter** na bolsa do Prof. Birch),
exigir a jornada inteira de uma vez é quase impossível para o agente descobrir
(recompensa esparsa, horizonte longo). A solução é um **currículo**: começar de
estados próximos da meta (fáceis) e ir afastando.

**1. Crie os checkpoints** jogando do quarto até a bolsa, dando F5 em vários pontos:

```bash
python scripts/make_init_state.py \
  --rom "roms/Pokemon - Emerald Version (USA, Europe).gba" \
  --out-prefix roms/curr
# Cada F5 salva roms/curr_000.state, roms/curr_001.state, ...
# Sugestão: F5 em (a) frente à bolsa, (b) Route 101, (c) fora de casa, (d) no quarto.
```

**2. Liste-os** em `emulator.init_states` (em [configs/default.yaml](configs/default.yaml)),
os mais perto da meta primeiro, com pesos maiores:

```yaml
emulator:
  init_states:
    - "roms/curr_000.state"   # em frente à bolsa (fácil)
    - "roms/curr_001.state"   # Route 101
    - "roms/curr_002.state"   # fora de casa
    - "roms/init_state.state" # no quarto (difícil)
  init_state_weights: [0.5, 0.25, 0.15, 0.10]   # opcional (default = uniforme)
  init_state_strategy: "random"                 # "random" (pesos) ou "sequential"
```

A cada `reset`, um estado é sorteado. Conforme o agente domina os fáceis,
aumente os pesos dos mais distantes. Defina a meta correspondente em
`env.goals` (ex.: `party_count: 1`). Quando `init_states` está definido, ele
tem **prioridade** sobre o `init_state` único.

### Recompensar pré-requisitos (flags de evento)

Alguns objetivos têm **pré-requisitos obrigatórios sem recompensa própria** —
ex.: no Emerald, antes de pegar o starter é preciso entrar na casa do vizinho e
apertar A num item, disparando uma cutscene. A navegação entre mapas já é
premiada por `new_map_weight`, mas essa interação no meio da cadeia não tem
sinal. Solução: recompensar a **flag de evento** que a cutscene liga.

**1. Descubra a flag** com dois save states (antes/depois do evento):

```bash
# salve um state ANTES e um DEPOIS da cutscene (com make_init_state.py)
python scripts/find_event_flag.py \
  --before roms/before.state --after roms/after.state
# imprime as flags que ligaram (0 -> 1); idealmente uma só
```

**2. Configure o milestone** em `reward.flag_milestones` ([configs/default.yaml](configs/default.yaml)):

```yaml
reward:
  flag_milestones:
    - {flag: 0x123, reward: 3.0}   # use a flag encontrada
```

O agente ganha `reward` na primeira vez que a flag liga no episódio — um marco
intermediário que torna a cadeia longa (quarto → cutscene → Route 101 → starter)
aprendível.

---

## 🧪 Uso

### Smoke test com ações aleatórias (com janela do jogo)

```bash
python -m src.play_random --config configs/default.yaml
python -m src.play_random --config configs/default.yaml --no-render   # sem janela
```

### Treino (PPO) — com visualização ao vivo

```bash
python -m src.train --config configs/default.yaml
python -m src.train --config configs/default.yaml --no-render   # treino sem janela
python -m src.train --config configs/default.yaml --wandb       # + Weights & Biases
```

**Continuar um treino (transferir o aprendizado).** Sem `--resume`, o treino
começa do **zero** e **sobrescreve** o modelo salvo. Para aproveitar o que o
agente já aprendeu (ex.: já sabe sair de casa) ao mudar para uma meta mais
difícil (ex.: pegar o starter):

```bash
python -m src.train --config configs/default.yaml --resume            # de train.save_path
python -m src.train --config configs/default.yaml --resume-from models/ppo_emerald.zip
```

Os pesos da rede são carregados e o treino continua a partir deles (o PPO não
guarda replay buffer; o "aprendizado" são os pesos). `total_timesteps` passa a
ser o orçamento **desta** rodada. Requisito: o espaço de observação/ação deve
bater com o do modelo salvo — ou seja, **não mude `env.frame_stack` nem
`env.actions`** entre o treino original e o resume (mudar `goals`, `reward` e
`init_states` é OK e é justamente o ponto).

Uma janela OpenCV (`pokemon-emerald-rl`) mostra a tela real do GBA do ambiente 0
**durante o treino**. Funciona com 1 ou vários ambientes paralelos. Ajuste
`train.render`, `train.render_freq` e `train.render_scale` no YAML.

### Avaliação de um modelo salvo (com janela do jogo)

```bash
python -m src.evaluate --config configs/default.yaml --model models/ppo_emerald.zip
```

### TensorBoard

```bash
tensorboard --logdir tensorboard/
```
Métricas do jogo (`game/party_count`, `game/max_level`, `game/badge_count`,
`game/map_code`) e do PPO ficam registradas ali.

### Testes

```bash
pytest -q
```

Os testes usam o **`MockGbaBackend`**, então **não precisam de ROM nem de emulador**.

---

## ⚡ Paralelismo e frame stacking

- **`train.n_envs`** controla o paralelismo: `1` → `DummyVecEnv` (in-process,
  ideal para depurar); `>1` → `SubprocVecEnv` (vários emuladores em paralelo,
  acelera o treino).
- **`env.frame_stack`** empilha N frames no eixo de canais via `VecFrameStack`
  (observação `(84, 84, N)`), dando ao agente noção de movimento.
- **`env.frame_skip`** = action repeat (frames de emulação por ação do agente).

---

## 🧩 Ambiente (PokemonEmeraldEnv)

- **Observação:** tela do jogo em **grayscale 84×84×1** (`uint8`); empilhada para
  `84×84×N` no treino.
- **Ações (`Discrete(7)`):** `UP, DOWN, LEFT, RIGHT, A, B, START`.
- **Recompensa:** ver [src/rewards/reward.py](src/rewards/reward.py) — exploração,
  novos mapas, pegar Pokémon (starter), level e badges. Tolerante a `None`.

---

## 🛣️ Roadmap

- [x] Implementar `GbaBackend` real com PyGBA/mGBA.
- [x] Mapear endereços de memória reais em `emerald_memory.py` (Emerald US).
- [x] Ajustar a função de recompensa para a meta inicial (sair do quarto → Oldale Town).
- [x] Adicionar `VecEnv` paralelos para acelerar o treino.
- [x] Experimentar frame stacking e action repeat.
- [x] Logging com TensorBoard / Weights & Biases.
- [x] Visualização ao vivo do jogo durante o treino.

Próximos passos sugeridos: criar/curar `init_state`, calibrar pesos de
recompensa para a meta inicial e validar os endereços de memória da sua ROM com
o debugger do mGBA.

---

## ⚖️ Aviso legal

Este projeto é apenas para fins **educacionais e de pesquisa**. Pokémon e Pokémon Emerald são marcas registradas da Nintendo / Game Freak / The Pokémon Company. Não distribuímos ROMs nem material protegido por direitos autorais.
