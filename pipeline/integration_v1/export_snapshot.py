"""Publish the immutable JSON snapshot of the Integration V1 contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import contract_snapshot


def write_contract_snapshot(destination: str | Path) -> dict[str, Any]:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(contract_snapshot(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    target.write_bytes(encoded)
    return {
        "path": str(target),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_contract_snapshot(args.destination), ensure_ascii=False))


if __name__ == "__main__":
    main()
