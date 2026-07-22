"""Adaptador local FLUX Fill para o TraduzAI Studio.

Recebe jobs JSONL pelo stdin e devolve um resultado JSON por linha no stdout.
Imports de ML são lazy para manter status/testes leves. Por padrão, o processo
fica residente e o modelo precisa existir no cache/local path; downloads só são
permitidos por opt-in.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import sys
from typing import Any

CONTRACT_VERSION = "1.0"
DEFAULT_PROMPT = "complete the selected area seamlessly, matching the surrounding manga artwork"
_PIPELINES: dict[tuple[str, str, bool], Any] = {}


def data_url_to_bytes(value: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("A entrada FLUX deve ser um PNG local em data URL")
    try:
        return base64.b64decode(value[len(prefix) :], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"PNG base64 invalido: {error}") from error


def variant_seeds(seed: int, count: int) -> list[int]:
    return [int(seed) + index for index in range(count)]


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("O job FLUX precisa ser um objeto JSON")
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Versao de contrato FLUX incompatível")
    job_id = str(payload.get("job_id") or "").strip()
    model = str(payload.get("model") or "").strip()
    if not job_id:
        raise ValueError("O job FLUX precisa de um identificador")
    if not model:
        raise ValueError("Informe um modelo FLUX Fill local")
    width = int(payload.get("width") or 0)
    height = int(payload.get("height") or 0)
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise ValueError("Dimensoes FLUX devem ficar entre 1 e 4096 pixels")
    variant_count = int(payload.get("variant_count") or 0)
    if not 2 <= variant_count <= 4:
        raise ValueError("O FLUX precisa gerar entre 2 e 4 variantes")
    steps = int(payload.get("steps") or 0)
    if not 1 <= steps <= 100:
        raise ValueError("Steps FLUX devem ficar entre 1 e 100")
    prompt = str(payload.get("prompt") or "")
    negative_prompt = str(payload.get("negative_prompt") or "")
    if len(prompt) > 4000 or len(negative_prompt) > 4000:
        raise ValueError("Prompt FLUX excede o limite de 4000 caracteres")
    data_url_to_bytes(str(payload.get("source_png_data") or ""))
    data_url_to_bytes(str(payload.get("mask_png_data") or ""))
    return {
        **payload,
        "job_id": job_id,
        "model": model,
        "width": width,
        "height": height,
        "variant_count": variant_count,
        "steps": steps,
        "seed": int(payload.get("seed") or 0),
        "guidance_scale": float(payload.get("guidance_scale") or 18.0),
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
    }


def _allow_model_download() -> bool:
    return os.environ.get("TRADUZAI_STUDIO_FLUX_ALLOW_MODEL_DOWNLOAD", "0").strip() == "1"


def _load_pipeline(model: str) -> tuple[Any, Any, str]:
    try:
        import torch
        from diffusers import FluxFillPipeline
    except ImportError as error:
        raise RuntimeError(
            "Dependencias FLUX ausentes. Instale torch, diffusers, transformers, accelerate, pillow e safetensors."
        ) from error

    device = os.environ.get("TRADUZAI_STUDIO_FLUX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu").strip()
    dtype_name = os.environ.get("TRADUZAI_STUDIO_FLUX_DTYPE", "bfloat16" if device == "cuda" else "float32").strip()
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise RuntimeError(f"TRADUZAI_STUDIO_FLUX_DTYPE invalido: {dtype_name}")
    cpu_offload = device == "cuda" and os.environ.get("TRADUZAI_STUDIO_FLUX_CPU_OFFLOAD", "1").strip() != "0"
    key = (model, dtype_name, cpu_offload)
    pipe = _PIPELINES.get(key)
    if pipe is None:
        allow_download = _allow_model_download()
        if not allow_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        pipe = FluxFillPipeline.from_pretrained(
            model,
            torch_dtype=dtype,
            local_files_only=not allow_download,
        )
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        _PIPELINES[key] = pipe
    return pipe, torch, device


def _effective_prompt(prompt: str, negative_prompt: str) -> str:
    value = prompt or DEFAULT_PROMPT
    if negative_prompt:
        value = f"{value}. Avoid: {negative_prompt}"
    return value


def _png_data_url(image: Any) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def run_generation(payload: dict[str, Any]) -> dict[str, Any]:
    request = validate_request(payload)
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow nao esta instalado no ambiente do adaptador FLUX") from error

    source = Image.open(io.BytesIO(data_url_to_bytes(request["source_png_data"]))).convert("RGB")
    mask = Image.open(io.BytesIO(data_url_to_bytes(request["mask_png_data"]))).convert("L")
    size = (request["width"], request["height"])
    if source.size != size:
        source = source.resize(size, Image.Resampling.LANCZOS)
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.LANCZOS)

    pipe, torch, _device = _load_pipeline(request["model"])
    prompt = _effective_prompt(request["prompt"], request["negative_prompt"])
    variants = []
    for index, seed in enumerate(variant_seeds(request["seed"], request["variant_count"])):
        print(f"FLUX local: variante {index + 1}/{request['variant_count']} seed={seed}", file=sys.stderr, flush=True)
        generator = torch.Generator("cpu").manual_seed(seed)
        output = pipe(
            prompt=prompt,
            image=source,
            mask_image=mask,
            width=request["width"],
            height=request["height"],
            guidance_scale=request["guidance_scale"],
            num_inference_steps=request["steps"],
            max_sequence_length=512,
            generator=generator,
        ).images[0]
        if output.size != size:
            output = output.resize(size, Image.Resampling.LANCZOS)
        variants.append({
            "id": f"variant-{index + 1}",
            "seed": seed,
            "png_data": _png_data_url(output),
            "path": None,
        })
    return {
        "contract_version": CONTRACT_VERSION,
        "job_id": request["job_id"],
        "provider": "diffusers-local",
        "model": request["model"],
        "variants": variants,
    }


def serve(input_stream: Any, output_stream: Any, runner: Any = run_generation) -> int:
    """Processa jobs JSONL sem descarregar a pipeline entre gerações."""
    for raw_line in input_stream:
        if not raw_line.strip():
            continue
        payload: Any = None
        try:
            payload = json.loads(raw_line)
            result = runner(payload)
        except Exception as error:  # noqa: BLE001 - boundary reports errors per job
            result = {
                "contract_version": CONTRACT_VERSION,
                "job_id": str(payload.get("job_id") or "") if isinstance(payload, dict) else "",
                "error": str(error),
            }
        json.dump(result, output_stream, ensure_ascii=False, separators=(",", ":"))
        output_stream.write("\n")
        output_stream.flush()
    return 0


def main() -> int:
    try:
        return serve(sys.stdin, sys.stdout)
    except Exception as error:  # noqa: BLE001 - boundary must report adapter errors to Rust
        print(str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
