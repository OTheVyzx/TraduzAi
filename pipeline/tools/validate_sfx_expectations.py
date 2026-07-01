from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.run_sfx_detection_probe import _validate_expectations


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an existing SFX probe summary against expectations.")
    parser.add_argument("--summary", required=True, help="Path to run_sfx_detection_probe summary.json.")
    parser.add_argument("--expect", required=True, help="Path to SFX expectations JSON.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    expect_path = Path(args.expect)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = _validate_expectations(summary, expect_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
