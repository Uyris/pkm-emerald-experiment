"""Instala o binding Python pré-compilado do mGBA (libmgba-py).

O módulo ``mgba`` (exigido pelo PyGBA) não tem wheel no PyPI para Windows. Este
script baixa o pacote pré-compilado do projeto `hanzi/libmgba-py` para a sua
plataforma e o extrai dentro do ``site-packages`` do ambiente atual.

Uso:
    python scripts/install_mgba.py

Requisitos:
    * Python 3.10 (os binários do libmgba-py 0.2.0 são compilados por versão).
    * cffi instalado (`pip install cffi`).

Após rodar, valide com:
    python -c "import mgba.core; print('mgba OK')"
"""

from __future__ import annotations

import io
import platform
import site
import sys
import sysconfig
import urllib.request
import zipfile

RELEASE = "0.2.0-2"
BASE = f"https://github.com/hanzi/libmgba-py/releases/download/{RELEASE}"

ASSETS = {
    ("Windows", "AMD64"): "libmgba-py_0.2.0_win64.zip",
    ("Linux", "x86_64"): "libmgba-py_0.2.0_ubuntu-lunar.zip",
    ("Darwin", "x86_64"): "libmgba-py_0.2.0_macos-x86_64.zip",
    ("Darwin", "arm64"): "libmgba-py_0.2.0_macos-arm64.zip",
}


def _target_site_packages() -> str:
    # Prefere o site-packages do venv ativo.
    paths = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    paths = paths or [sysconfig.get_paths()["purelib"]]
    return paths[0]


def main() -> int:
    system = platform.system()
    machine = platform.machine()
    asset = ASSETS.get((system, machine))

    if asset is None:
        print(f"[erro] Plataforma não suportada automaticamente: {system}/{machine}.")
        print("Veja https://github.com/hanzi/libmgba-py/releases e instale manualmente.")
        return 1

    if sys.version_info[:2] != (3, 10):
        print(f"[aviso] Os binários do libmgba-py {RELEASE} são para Python 3.10; "
              f"você está em {sys.version_info.major}.{sys.version_info.minor}. "
              "O import pode falhar (ABI incompatível).")

    url = f"{BASE}/{asset}"
    dest = _target_site_packages()
    print(f"Baixando {url} ...")
    data = urllib.request.urlopen(url).read()
    print(f"  {len(data) / 1e6:.1f} MB. Extraindo em {dest} ...")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dest)

    print("Pronto. Validando import...")
    try:
        import mgba.core  # noqa: F401
        print("mgba OK ✅  (lembre de `pip install cffi` se ainda não tiver)")
        return 0
    except Exception as e:  # pragma: no cover
        print(f"[erro] Import falhou: {type(e).__name__}: {e}")
        print("Verifique se `cffi` está instalado e se a versão do Python é 3.10.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
