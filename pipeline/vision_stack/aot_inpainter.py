from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DebugCallback = Optional[Callable[[dict], None]]

RELU_NF_SCALE = 1.7139588594436646
WEIGHT_STANDARDIZATION_EPS = 1e-4
LAYER_NORM_EPS = 1e-9
HF_AOT_REPO_DIR = "models--mayocream--aot-inpainting"
MIN_FORWARD_SIDE = 128


class AotInpaintingUnavailable(RuntimeError):
    """Raised when the AOT preset is selected but the runtime cannot run it."""


@dataclass(frozen=True)
class AotModelPaths:
    config: Path
    weights: Path


@dataclass(frozen=True)
class AotConfig:
    model_type: str
    input_channels: int
    output_channels: int
    base_channels: int
    num_blocks: int
    dilation_rates: tuple[int, ...]
    pad_multiple: int
    default_max_side: int

    @classmethod
    def load(cls, path: Path) -> "AotConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = cls(
            model_type=str(payload.get("model_type") or ""),
            input_channels=int(payload.get("input_channels") or 0),
            output_channels=int(payload.get("output_channels") or 0),
            base_channels=int(payload.get("base_channels") or 0),
            num_blocks=int(payload.get("num_blocks") or 0),
            dilation_rates=tuple(int(v) for v in payload.get("dilation_rates") or ()),
            pad_multiple=int(payload.get("pad_multiple") or 0),
            default_max_side=int(payload.get("default_max_side") or 0),
        )
        config.validate(path)
        return config

    def validate(self, path: Path) -> None:
        if self.model_type != "manga-image-translator-aot":
            raise AotInpaintingUnavailable(
                f"Unsupported AOT config at {path}: model_type={self.model_type!r}"
            )
        if self.input_channels != 4 or self.output_channels != 3:
            raise AotInpaintingUnavailable(
                f"Unsupported AOT channel config at {path}: "
                f"input_channels={self.input_channels}, output_channels={self.output_channels}"
            )
        if self.base_channels <= 0 or self.num_blocks <= 0:
            raise AotInpaintingUnavailable(f"Invalid AOT config at {path}: empty model dimensions")
        if not self.dilation_rates:
            raise AotInpaintingUnavailable(f"Invalid AOT config at {path}: empty dilation_rates")
        if self.pad_multiple <= 0 or self.default_max_side <= 0:
            raise AotInpaintingUnavailable(f"Invalid AOT config at {path}: invalid size limits")


def aot_inpainting_enabled() -> bool:
    raw = os.getenv("TRADUZAI_AOT_INPAINT", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_path(name: str) -> Path | None:
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else None


def _existing_pair(config: Path, weights: Path) -> AotModelPaths | None:
    if config.exists() and weights.exists():
        return AotModelPaths(config=config, weights=weights)
    return None


def find_aot_model_paths(models_dir: str | Path | None = None) -> AotModelPaths | None:
    env_config = _env_path("TRADUZAI_AOT_CONFIG")
    env_weights = _env_path("TRADUZAI_AOT_WEIGHTS")
    if env_config is not None or env_weights is not None:
        if env_config is None or env_weights is None:
            raise AotInpaintingUnavailable(
                "Set both TRADUZAI_AOT_CONFIG and TRADUZAI_AOT_WEIGHTS to enable explicit AOT paths"
            )
        found = _existing_pair(env_config, env_weights)
        if found is None:
            raise AotInpaintingUnavailable(
                f"AOT model files not found: config={env_config}, weights={env_weights}"
            )
        return found

    bases: list[Path] = []
    if models_dir:
        bases.append(Path(models_dir))
    env_models = _env_path("TRADUZAI_MODELS_DIR") or _env_path("MANGATL_MODELS_DIR")
    if env_models is not None:
        bases.append(env_models)
    bases.append(Path.cwd() / "models")
    bases.append(Path.home() / ".traduzai" / "models")
    bases.append(Path.home() / ".mangatl" / "models")

    seen: set[Path] = set()
    for base in bases:
        base = base.expanduser()
        if base in seen:
            continue
        seen.add(base)
        direct_candidates = [
            base / "aot-inpainting",
            base / "huggingface" / HF_AOT_REPO_DIR,
            base / HF_AOT_REPO_DIR,
        ]
        for root in direct_candidates:
            found = _existing_pair(root / "config.json", root / "model.safetensors")
            if found is not None:
                return found

            snapshots = root / "snapshots"
            if snapshots.exists():
                for snapshot in sorted(snapshots.iterdir(), reverse=True):
                    if not snapshot.is_dir():
                        continue
                    found = _existing_pair(snapshot / "config.json", snapshot / "model.safetensors")
                    if found is not None:
                        return found

    return None


def resolve_aot_model_paths(models_dir: str | Path | None = None) -> AotModelPaths:
    found = find_aot_model_paths(models_dir)
    if found is None:
        raise AotInpaintingUnavailable(
            "AOT model files not found. Expected config.json and model.safetensors under "
            "models/aot-inpainting or models/huggingface/models--mayocream--aot-inpainting."
        )
    return found


def _standardize_conv_weight(weight: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    weight_f = weight.float()
    gain_f = gain.float()
    out_channels = weight_f.shape[0]
    flat = weight_f.flatten(1)
    fan_in = float(flat.shape[1])
    mean = flat.mean(dim=1, keepdim=True)
    var = flat.var(dim=1, keepdim=True, unbiased=False)
    variance = var * fan_in
    scale = torch.rsqrt(torch.clamp(variance, min=WEIGHT_STANDARDIZATION_EPS))
    scale = scale * gain_f.reshape(out_channels, 1)
    standardized = flat * scale - mean * scale
    return standardized.reshape_as(weight_f)


class GatedWsConvPadded(nn.Module):
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        prefix: str,
        *,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.pad = ((kernel_size - 1) * dilation) // 2
        self.register_buffer(
            "weight",
            _standardize_conv_weight(state[f"{prefix}.conv.weight"], state[f"{prefix}.conv.gain"]).to(
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer("bias", state[f"{prefix}.conv.bias"].to(device=device, dtype=dtype))
        self.register_buffer(
            "gate_weight",
            _standardize_conv_weight(
                state[f"{prefix}.conv_gate.weight"],
                state[f"{prefix}.conv_gate.gain"],
            ).to(device=device, dtype=dtype),
        )
        self.register_buffer("gate_bias", state[f"{prefix}.conv_gate.bias"].to(device=device, dtype=dtype))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        if self.pad:
            xs = F.pad(xs, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        signal = F.conv2d(xs, self.weight, self.bias, stride=self.stride, dilation=self.dilation)
        gate = torch.sigmoid(F.conv2d(xs, self.gate_weight, self.gate_bias, stride=self.stride, dilation=self.dilation))
        return signal * gate * 1.8


class GatedWsTransposeConvPadded(nn.Module):
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        prefix: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "weight",
            _standardize_conv_weight(state[f"{prefix}.conv.weight"], state[f"{prefix}.conv.gain"]).to(
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer("bias", state[f"{prefix}.conv.bias"].to(device=device, dtype=dtype))
        self.register_buffer(
            "gate_weight",
            _standardize_conv_weight(
                state[f"{prefix}.conv_gate.weight"],
                state[f"{prefix}.conv_gate.gain"],
            ).to(device=device, dtype=dtype),
        )
        self.register_buffer("gate_bias", state[f"{prefix}.conv_gate.bias"].to(device=device, dtype=dtype))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        signal = F.conv_transpose2d(xs, self.weight, self.bias, stride=2, padding=1)
        gate = torch.sigmoid(F.conv_transpose2d(xs, self.gate_weight, self.gate_bias, stride=2, padding=1))
        return signal * gate * 1.8


class PaddedConvRelu(nn.Module):
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        prefix: str,
        *,
        kernel_size: int,
        dilation: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.pad = ((kernel_size - 1) * dilation) // 2
        self.dilation = dilation
        self.register_buffer("weight", state[f"{prefix}.weight"].to(device=device, dtype=dtype))
        self.register_buffer("bias", state[f"{prefix}.bias"].to(device=device, dtype=dtype))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        if self.pad:
            xs = F.pad(xs, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        return F.conv2d(xs, self.weight, self.bias, dilation=self.dilation).relu()


class PaddedConv(nn.Module):
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        prefix: str,
        *,
        kernel_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.register_buffer("weight", state[f"{prefix}.weight"].to(device=device, dtype=dtype))
        self.register_buffer("bias", state[f"{prefix}.bias"].to(device=device, dtype=dtype))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        if self.pad:
            xs = F.pad(xs, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        return F.conv2d(xs, self.weight, self.bias)


class AotBlock(nn.Module):
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        prefix: str,
        *,
        dilation_rates: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                PaddedConvRelu(
                    state,
                    f"{prefix}.block{index:02}.1",
                    kernel_size=3,
                    dilation=rate,
                    dtype=dtype,
                    device=device,
                )
                for index, rate in enumerate(dilation_rates)
            ]
        )
        self.fuse = PaddedConv(state, f"{prefix}.fuse.1", kernel_size=3, dtype=dtype, device=device)
        self.gate = PaddedConv(state, f"{prefix}.gate.1", kernel_size=3, dtype=dtype, device=device)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(torch.cat([branch(xs) for branch in self.branches], dim=1))
        gate = torch.sigmoid(_aot_layer_norm(self.gate(xs)))
        return xs * (1.0 - gate) + fused * gate


class AotGenerator(nn.Module):
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        config: AotConfig,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        ch = config.base_channels
        body_channels = ch * 4
        self.head0 = GatedWsConvPadded(state, "head.0", kernel_size=3, dtype=dtype, device=device)
        self.head1 = GatedWsConvPadded(state, "head.2", kernel_size=4, stride=2, dtype=dtype, device=device)
        self.head2 = GatedWsConvPadded(state, "head.4", kernel_size=4, stride=2, dtype=dtype, device=device)
        self.body = nn.ModuleList(
            [
                AotBlock(
                    state,
                    f"body_conv.{index}",
                    dilation_rates=config.dilation_rates,
                    dtype=dtype,
                    device=device,
                )
                for index in range(config.num_blocks)
            ]
        )
        self.tail0 = GatedWsConvPadded(state, "tail.0", kernel_size=3, dtype=dtype, device=device)
        self.tail1 = GatedWsConvPadded(state, "tail.2", kernel_size=3, dtype=dtype, device=device)
        self.up0 = GatedWsTransposeConvPadded(state, "tail.4", dtype=dtype, device=device)
        self.up1 = GatedWsTransposeConvPadded(state, "tail.6", dtype=dtype, device=device)
        self.output = GatedWsConvPadded(state, "tail.8", kernel_size=3, dtype=dtype, device=device)
        del body_channels

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xs = torch.cat([mask, image], dim=1)
        xs = self.head0(xs).relu() * RELU_NF_SCALE
        xs = self.head1(xs).relu() * RELU_NF_SCALE
        xs = self.head2(xs)
        for block in self.body:
            xs = block(xs)
        xs = self.tail0(xs).relu() * RELU_NF_SCALE
        xs = self.tail1(xs).relu() * RELU_NF_SCALE
        xs = self.up0(xs).relu() * RELU_NF_SCALE
        xs = self.up1(xs).relu() * RELU_NF_SCALE
        return self.output(xs).clamp(-1.0, 1.0)


def _aot_layer_norm(xs: torch.Tensor) -> torch.Tensor:
    dtype = xs.dtype
    xs_f = xs.float()
    batch, channels, height, width = xs_f.shape
    flat = xs_f.flatten(2)
    mean = flat.mean(dim=2, keepdim=True)
    std = torch.sqrt(flat.var(dim=2, keepdim=True, unbiased=False) + LAYER_NORM_EPS)
    normalized = ((flat - mean) * 2.0 / std) - 1.0
    return (normalized * 5.0).reshape(batch, channels, height, width).to(dtype=dtype)


class AotInpainter:
    backend = "aot-inpainting"

    def __init__(
        self,
        *,
        models_dir: str | Path | None = None,
        device: torch.device | str = "cuda",
        half: bool = True,
        config_path: str | Path | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        if not aot_inpainting_enabled():
            raise AotInpaintingUnavailable(
                "AOT inpainting is disabled. Set TRADUZAI_AOT_INPAINT=1 to use the CJK AOT preset."
            )

        if config_path is not None or weights_path is not None:
            if config_path is None or weights_path is None:
                raise AotInpaintingUnavailable("AOT requires both config_path and weights_path")
            paths = AotModelPaths(Path(config_path), Path(weights_path))
            if not paths.config.exists() or not paths.weights.exists():
                raise AotInpaintingUnavailable(
                    f"AOT model files not found: config={paths.config}, weights={paths.weights}"
                )
        else:
            paths = resolve_aot_model_paths(models_dir)

        self.paths = paths
        self.config = AotConfig.load(paths.config)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.dtype = torch.float16 if half and self.device.type == "cuda" else torch.float32

        try:
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover - depends on local env
            raise AotInpaintingUnavailable("safetensors is required for AOT inpainting") from exc

        state = load_file(str(paths.weights), device="cpu")
        self.model = AotGenerator(state, self.config, dtype=self.dtype, device=self.device)
        self.model.eval()

    def inpaint(
        self,
        img_np: np.ndarray,
        mask: np.ndarray,
        batch_size: int = 4,
        debug: DebugCallback = None,
        force_no_tiling: bool = False,
    ) -> np.ndarray:
        del batch_size, force_no_tiling
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise ValueError(f"img_np invalid for AOT: {img_np.shape}")
        if mask.ndim != 2 or mask.shape[:2] != img_np.shape[:2]:
            raise ValueError(f"mask/image mismatch for AOT: mask={mask.shape} image={img_np.shape}")

        binary_mask = (mask > 127).astype(np.uint8) * 255
        if not np.any(binary_mask):
            return img_np.copy()

        if debug is not None:
            debug(
                {
                    "event": "aot_inpaint_call",
                    "backend": self.backend,
                    "image_shape": list(img_np.shape),
                    "mask_shape": list(binary_mask.shape),
                    "default_max_side": int(self.config.default_max_side),
                }
            )

        max_side = max(img_np.shape[:2])
        if max_side > self.config.default_max_side:
            result = self._resize_forward(img_np, binary_mask)
        else:
            result = self._pad_forward(img_np, binary_mask)

        result = _normalize_output(result, img_np.shape[:2])
        return _composite_masked(img_np, result, binary_mask)

    def _resize_forward(self, img_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
        height, width = img_np.shape[:2]
        scale = float(self.config.default_max_side) / float(max(height, width))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        small_img = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_AREA)
        small_mask = (small_mask > 127).astype(np.uint8) * 255
        small_out = self._pad_forward(small_img, small_mask)
        return cv2.resize(small_out, (width, height), interpolation=cv2.INTER_CUBIC)

    def _pad_forward(self, img_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
        height, width = img_np.shape[:2]
        pad_h = max(MIN_FORWARD_SIDE, _ceil_multiple(height, self.config.pad_multiple))
        pad_w = max(MIN_FORWARD_SIDE, _ceil_multiple(width, self.config.pad_multiple))
        if (pad_h, pad_w) != (height, width):
            img_np = _symmetric_pad_np(img_np, pad_h, pad_w)
            mask = _symmetric_pad_np(mask, pad_h, pad_w)
        result = self._forward_rgb(img_np, mask)
        return result[:height, :width].copy()

    def _forward_rgb(self, img_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image = torch.from_numpy(np.ascontiguousarray(img_np)).to(device=self.device)
        image = image.permute(2, 0, 1).unsqueeze(0).to(dtype=self.dtype) / 127.5 - 1.0

        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(device=self.device)
        mask_t = mask_t.unsqueeze(0).unsqueeze(0).to(dtype=self.dtype) / 255.0
        masked_image = image * (1.0 - mask_t).expand_as(image)

        with torch.inference_mode():
            output = self.model(masked_image, mask_t)

        output = output.squeeze(0).float().cpu().permute(1, 2, 0).numpy()
        output = np.clip((output + 1.0) * 127.5, 0.0, 255.0)
        return output.astype(np.uint8)


def _ceil_multiple(value: int, multiple: int) -> int:
    remainder = value % multiple
    return value if remainder == 0 else value + (multiple - remainder)


def _symmetric_pad_np(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    height, width = array.shape[:2]
    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    if pad_h == 0 and pad_w == 0:
        return array
    if array.ndim == 2:
        pad_width = ((0, pad_h), (0, pad_w))
    else:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(array, pad_width, mode="symmetric")


def _normalize_output(result_np: np.ndarray, expected_hw: tuple[int, int]) -> np.ndarray:
    expected_h, expected_w = expected_hw
    if result_np.shape[:2] == (expected_h, expected_w):
        return result_np.astype(np.uint8, copy=False)
    return cv2.resize(result_np, (expected_w, expected_h), interpolation=cv2.INTER_CUBIC).astype(np.uint8)


def _composite_masked(original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = original.copy()
    result[mask > 0] = inpainted[mask > 0]
    return result
