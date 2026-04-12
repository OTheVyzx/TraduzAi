from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_stack.runtime import run_debug_experiments


def main():
    parser = argparse.ArgumentParser(description="Roda experimentos de debug do pipeline visual.")
    parser.add_argument("image_path", help="Imagem de entrada para depurar")
    parser.add_argument("--models-dir", default="", help="Diretorio de modelos")
    parser.add_argument("--profile", default="quality", help="Perfil do pipeline")
    parser.add_argument(
        "--debug-root",
        default=str(Path(__file__).resolve().parents[1] / "debug_runs"),
        help="Diretorio raiz onde os artefatos serao salvos",
    )
    args = parser.parse_args()

    result = run_debug_experiments(
        image_path=args.image_path,
        models_dir=args.models_dir,
        profile=args.profile,
        debug_root=args.debug_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
