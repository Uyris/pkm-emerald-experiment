# Mantido vazio de propósito.
#
# Importar `src.utils.config` aqui criaria um ciclo de import:
#   utils/__init__ -> config -> envs.emerald_env -> utils.preprocessing
#                                                 -> utils/__init__ (parcial)
# Importe os submódulos diretamente, ex.:
#   from src.utils.config import load_config, make_env
#   from src.utils.preprocessing import preprocess_frame
