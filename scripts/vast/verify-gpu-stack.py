from __future__ import annotations

import json
import os
import sys


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    require_gpu = _truthy(os.getenv("TRADUZAI_REQUIRE_GPU"))
    result: dict[str, object] = {"require_gpu": require_gpu}
    ok = True

    try:
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
        result["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch_cuda_available,
            "gpu": torch.cuda.get_device_name(0) if torch_cuda_available else None,
        }
        ok = ok and torch_cuda_available
    except Exception as exc:
        result["torch"] = {"error": str(exc)}
        ok = False

    try:
        import paddle

        paddle_cuda = bool(paddle.device.is_compiled_with_cuda())
        result["paddle"] = {
            "version": paddle.__version__,
            "cuda_compiled": paddle_cuda,
            "device": paddle.device.get_device() if paddle_cuda else "cpu",
        }
        ok = ok and paddle_cuda
    except Exception as exc:
        result["paddle"] = {"error": str(exc)}
        ok = False

    try:
        import paddleocr

        result["paddleocr"] = {"version": getattr(paddleocr, "__version__", "unknown")}
    except Exception as exc:
        result["paddleocr"] = {"error": str(exc)}
        ok = False

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if require_gpu and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
