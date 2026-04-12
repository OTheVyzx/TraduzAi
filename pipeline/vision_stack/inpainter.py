"""
Inpainter — Remoção de texto via inpainting
Usa LaMA (Large Mask inpainting) — mesmo modelo do Koharu (AnimeMangaInpainting)
Suporte a batching e tiles para imagens grandes
"""

import logging
import os
from pathlib import Path
import shutil
import sysconfig
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

DebugCallback = Optional[Callable[[dict], None]]

def _default_models_dir() -> Path:
    env_dir = (os.getenv("TRADUZAI_MODELS_DIR") or os.getenv("MANGATL_MODELS_DIR") or "").strip()
    if env_dir:
        return Path(env_dir)

    default_dir = Path.home() / ".traduzai" / "models"
    legacy_dir = Path.home() / ".mangatl" / "models"
    if not default_dir.exists() and legacy_dir.exists():
        return legacy_dir
    return default_dir


MODELS_DIR = _default_models_dir()

MODEL_URLS = {
    "lama": "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/big-lama.pt",
    "lama-manga": "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/big-lama.pt",
}


class Inpainter:
    """
    Inpainting de alta qualidade usando LaMA.
    
    O mesmo modelo que o Koharu usa (AnimeMangaInpainting / dreMaz).
    Otimizações:
        - Processa em tiles se a imagem for grande (>1024px)
        - Batch de múltiplas regiões quando possível
        - fp16 na GPU para 2x throughput
    """

    def __init__(
        self,
        model: str = "lama",
        device: str = "cuda",
        half: bool = True,
        model_path: Optional[str] = None,
    ):
        self.device = self._resolve_device(device)
        self.half = half and self.device.type == "cuda"
        self._model = None
        self._load_model(model, model_path)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Inpainter: GPU com {vram_gb:.1f}GB VRAM")
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self, model_name: str, model_path: Optional[str] = None):
        """Carrega LaMA com checkpoint pré-treinado."""
        onnx_backend = self._try_load_onnx_backend(model_name, model_path)
        if onnx_backend is not None:
            self._model = onnx_backend
            provider_name = getattr(self, "_onnx_provider", "cuda")
            self._backend = "lama_onnx_tensorrt" if provider_name == "tensorrt" else "lama_onnx_cuda"
            logger.info("LaMA carregado via ONNX %s", provider_name.upper())
            return

        # Tenta usar simple-lama (wrapper pip simples)
        try:
            from simple_lama_inpainting import SimpleLama
            self._model = SimpleLama(device=str(self.device))
            self._backend = "simple_lama"
            logger.info(f"LaMA carregado via simple-lama ({self.device})")
            return
        except ImportError:
            pass

        # Fallback: carrega checkpoint diretamente
        ckpt_path = Path(model_path) if model_path else MODELS_DIR / f"{model_name}.pt"
        if not ckpt_path.exists():
            self._download_model(model_name, ckpt_path)

        try:
            self._model = self._load_lama_checkpoint(ckpt_path)
            self._backend = "lama_direct"
            logger.info(f"LaMA carregado diretamente ({self.device})")
        except Exception as e:
            logger.error(f"Falha ao carregar LaMA: {e}")
            logger.info("Usando OpenCV inpainting como fallback")
            self._model = None
            self._backend = "opencv"

    def _try_load_onnx_backend(self, model_name: str, model_path: Optional[str] = None):
        del model_path
        if model_name not in {"lama", "lama-manga"}:
            return None

        try:
            import onnxruntime as ort
            from inpainter.lama_onnx import get_lama_session, is_lama_manga_available

            if not is_lama_manga_available():
                return None

            try:
                ort.preload_dlls()
            except Exception:
                logger.debug("preload_dlls falhou; continuando com busca padrao de DLLs", exc_info=True)

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers and "TensorrtExecutionProvider" not in providers:
                return None

            tensorrt_flag = (
                os.getenv("TRADUZAI_ENABLE_TENSORRT")
                or os.getenv("MANGATL_ENABLE_TENSORRT")
                or ""
            ).strip().lower()
            allow_tensorrt = tensorrt_flag in {
                "1",
                "true",
                "yes",
                "on",
            }
            preferred_providers: list[str] = []
            if "CUDAExecutionProvider" in providers:
                preferred_providers.append("CUDAExecutionProvider")
            if (
                allow_tensorrt
                and "TensorrtExecutionProvider" in providers
                and self._tensorrt_runtime_available()
            ):
                preferred_providers = ["TensorrtExecutionProvider", *preferred_providers]
            elif "TensorrtExecutionProvider" in providers and not allow_tensorrt:
                logger.info("TensorRT disponivel, mas desativado por padrao; usando CUDAExecutionProvider")
            elif "TensorrtExecutionProvider" in providers and allow_tensorrt and not self._tensorrt_runtime_available():
                logger.warning("TensorRT solicitado, mas runtime indisponivel; usando CUDAExecutionProvider")
            preferred_providers.append("CPUExecutionProvider")

            try:
                session = get_lama_session(MODELS_DIR, providers=preferred_providers)
                active = [str(p) for p in session.get_providers()]
                if active and active[0] == "TensorrtExecutionProvider":
                    self._onnx_provider = "tensorrt"
                else:
                    self._onnx_provider = "cuda"
                return session
            except Exception as exc:
                logger.warning("LaMA ONNX TensorRT/CUDA falhou (%s); tentando CUDA puro", exc)
                if "CUDAExecutionProvider" not in providers:
                    return None
                session = get_lama_session(MODELS_DIR, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                self._onnx_provider = "cuda"
                return session
        except Exception as exc:
            logger.warning("LaMA ONNX CUDA nao carregou: %s", exc)
            return None

    @staticmethod
    def _tensorrt_runtime_available() -> bool:
        package_paths: list[Path] = []
        purelib = sysconfig.get_paths().get("purelib")
        if purelib:
            package_paths.append(Path(purelib) / "tensorrt_libs")
        try:
            import tensorrt_libs  # type: ignore

            package_paths.append(Path(tensorrt_libs.__file__).resolve().parent)
        except Exception:
            pass
        candidates = [
            "nvinfer_10.dll",
            "nvinfer.dll",
        ]
        for base in package_paths:
            for name in candidates:
                if (base / name).exists():
                    return True
        for name in candidates:
            if shutil.which(name):
                return True
        for path_entry in os.environ.get("PATH", "").split(os.pathsep):
            if not path_entry:
                continue
            path = Path(path_entry)
            for name in candidates:
                if (path / name).exists():
                    return True
        return False

    def _download_model(self, model_name: str, dest: Path):
        import urllib.request
        url = MODEL_URLS.get(model_name, MODEL_URLS["lama"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Baixando modelo LaMA ({model_name})...")
        
        def progress(count, block, total):
            if total > 0:
                print(f"\r  LaMA: {count*block*100/total:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest, progress)
        print()

    def _load_lama_checkpoint(self, ckpt_path: Path):
        """Carrega checkpoint LaMA manualmente."""
        state = torch.load(str(ckpt_path), map_location="cpu")
        
        # Suporte a diferentes formatos de checkpoint
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "generator" in state:
                state = state["generator"]

        # Importa a arquitetura LaMA
        from .lama_arch import LaMa
        model = LaMa()
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        
        if self.half:
            model = model.half()
        
        return model

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def inpaint(
        self,
        img_np: np.ndarray,
        mask: np.ndarray,
        batch_size: int = 4,
        debug: DebugCallback = None,
        force_no_tiling: bool = False,
    ) -> np.ndarray:
        """
        Remove o texto da imagem usando inpainting.
        
        Args:
            img_np: imagem RGB uint8 (H, W, 3)
            mask: máscara binária uint8 (H, W) — 255=apagar, 0=manter
            batch_size: não usado diretamente, reservado para tiles
            
        Returns:
            imagem RGB uint8 com texto removido
        """
        assert img_np.ndim == 3 and img_np.shape[2] == 3, f"img_np invalida: {img_np.shape}"
        assert mask.ndim == 2, f"mask invalida: {mask.shape}"
        assert mask.shape[:2] == img_np.shape[:2], (
            f"mask/image mismatch: mask={mask.shape[:2]} image={img_np.shape[:2]}"
        )

        if debug:
            debug(
                {
                    "event": "inpaint_call",
                    "backend": self._backend,
                    "image_shape": list(img_np.shape),
                    "mask_shape": list(mask.shape),
                    "batch_size": int(batch_size),
                    "force_no_tiling": bool(force_no_tiling),
                }
            )

        if self._backend == "opencv":
            return self._opencv_inpaint(img_np, mask)

        h, w = img_np.shape[:2]

        # Imagens grandes: processa por tiles sobrepostos
        if max(h, w) > 1024 and not force_no_tiling:
            return self._tiled_inpaint(img_np, mask, tile_size=512, overlap=64, debug=debug)

        return self._run_inpaint(img_np, mask, debug=debug)

    def _run_inpaint(self, img_np: np.ndarray, mask: np.ndarray, debug: DebugCallback = None) -> np.ndarray:
        """Inferência LaMA em uma única passagem."""
        
        assert mask.shape[:2] == img_np.shape[:2], (
            f"mask/image mismatch before run: mask={mask.shape[:2]} image={img_np.shape[:2]}"
        )
        if self._backend in {"lama_onnx_cuda", "lama_onnx_tensorrt"}:
            from inpainter.lama_onnx import inpaint_region_with_lama

            result_np = inpaint_region_with_lama(self._model, img_np, mask)
            if debug:
                debug(
                    {
                        "event": "single_run",
                        "backend": self._backend,
                        "input_shape": list(img_np.shape),
                        "mask_shape": list(mask.shape),
                        "pad": {"top": 0, "bottom": 0, "left": 0, "right": 0},
                        "output_shape": list(result_np.shape),
                    }
                )
            return result_np

        if self._backend == "simple_lama":
            pil_img = Image.fromarray(img_np)
            pil_mask = Image.fromarray(mask)
            result = self._model(pil_img, pil_mask)
            result_np = self._normalize_output_shape(np.array(result), img_np.shape[:2])
            if debug:
                debug(
                    {
                        "event": "single_run",
                        "backend": self._backend,
                        "input_shape": list(img_np.shape),
                        "mask_shape": list(mask.shape),
                        "pad": {"top": 0, "bottom": 0, "left": 0, "right": 0},
                        "output_shape": list(result_np.shape),
                    }
                )
            return result_np

        # Backend direto
        img_t = self._to_tensor(img_np)        # (1, 3, H, W)
        mask_t = self._mask_to_tensor(mask)    # (1, 1, H, W)

        # Padding para múltiplo de 8 (requisito LaMA)
        img_t, mask_t, pad = self._pad_to_multiple(img_t, mask_t, multiple=8)
        padded_h = int(img_t.shape[-2])
        padded_w = int(img_t.shape[-1])

        with torch.inference_mode():
            output = self._model(img_t, mask_t)

        # Remove padding e converte de volta
        output = self._unpad(output, pad)
        assert tuple(output.shape[-2:]) == tuple(img_np.shape[:2]), (
            f"unpad falhou: output={tuple(output.shape[-2:])} esperado={tuple(img_np.shape[:2])}"
        )
        result = self._to_numpy(output)
        assert result.shape[:2] == img_np.shape[:2], (
            f"numpy shape mismatch: result={result.shape[:2]} esperado={img_np.shape[:2]}"
        )
        if debug:
            orig_h, orig_w = img_np.shape[:2]
            debug(
                {
                    "event": "single_run",
                    "backend": self._backend,
                    "input_shape": list(img_np.shape),
                    "mask_shape": list(mask.shape),
                    "tensor_shape_after_pad": [int(v) for v in img_t.shape],
                    "pad": {
                        "top": 0,
                        "bottom": int(padded_h - orig_h),
                        "left": 0,
                        "right": int(padded_w - orig_w),
                    },
                    "tensor_shape_after_unpad": [int(v) for v in output.shape],
                    "output_shape": list(result.shape),
                }
            )
        return result

    @staticmethod
    def _normalize_output_shape(result_np: np.ndarray, expected_hw: tuple[int, int]) -> np.ndarray:
        expected_h, expected_w = expected_hw
        if result_np.ndim == 2:
            result_np = cv2.cvtColor(result_np, cv2.COLOR_GRAY2RGB)
        if result_np.shape[:2] == (expected_h, expected_w):
            return result_np

        out_h, out_w = result_np.shape[:2]
        if out_h >= expected_h and out_w >= expected_w and (out_h - expected_h) <= 8 and (out_w - expected_w) <= 8:
            return result_np[:expected_h, :expected_w].copy()

        interpolation = cv2.INTER_CUBIC if out_h < expected_h or out_w < expected_w else cv2.INTER_AREA
        return cv2.resize(result_np, (expected_w, expected_h), interpolation=interpolation)

    def _tiled_inpaint(
        self,
        img_np: np.ndarray,
        mask: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
        debug: DebugCallback = None,
    ) -> np.ndarray:
        """
        Inpainting por tiles para imagens grandes.
        Só processa tiles que contêm máscara (evita trabalho desnecessário).
        """
        h, w = img_np.shape[:2]
        result = img_np.copy().astype(np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)
        step = tile_size - overlap
        tile_logs = []

        ys = list(range(0, h - tile_size + 1, step)) + ([h - tile_size] if h > tile_size else [])
        xs = list(range(0, w - tile_size + 1, step)) + ([w - tile_size] if w > tile_size else [])

        for y in set(max(0, y) for y in ys):
            for x in set(max(0, x) for x in xs):
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                y1 = max(0, y2 - tile_size)
                x1 = max(0, x2 - tile_size)
                assert 0 <= x1 <= x2 <= w, f"tile x fora da imagem: {(x1, x2, w)}"
                assert 0 <= y1 <= y2 <= h, f"tile y fora da imagem: {(y1, y2, h)}"

                tile_mask = mask[y1:y2, x1:x2]
                if tile_mask.max() == 0:
                    # Tile sem texto — pula
                    continue

                tile_img = img_np[y1:y2, x1:x2]
                tile_h = y2 - y1
                tile_w = x2 - x1
                pad_h = (8 - tile_h % 8) % 8
                pad_w = (8 - tile_w % 8) % 8
                tile_result = self._run_inpaint(tile_img, tile_mask).astype(np.float32)
                assert tile_result.shape[:2] == tile_img.shape[:2], (
                    f"tile_result shape mismatch: {tile_result.shape[:2]} vs {tile_img.shape[:2]}"
                )

                # Blending suave nas bordas (feather)
                blend_w = self._feather_weight(max(tile_h, tile_w), min(overlap, tile_h // 2 or 1, tile_w // 2 or 1))
                blend_w = blend_w[:tile_h, :tile_w]
                blend_3d = blend_w[:, :, np.newaxis]
                target_roi = result[y1:y2, x1:x2]
                assert target_roi.shape[:2] == tile_result.shape[:2], (
                    f"patch/roi mismatch: patch={tile_result.shape[:2]} roi={target_roi.shape[:2]}"
                )
                result[y1:y2, x1:x2] = (
                    target_roi * weight[y1:y2, x1:x2]
                    + tile_result * blend_3d
                ) / (weight[y1:y2, x1:x2] + blend_3d + 1e-8)
                weight[y1:y2, x1:x2] += blend_3d
                tile_logs.append(
                    {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "width": int(tile_w),
                        "height": int(tile_h),
                        "resize_width": int(tile_w),
                        "resize_height": int(tile_h),
                        "padding": {"top": 0, "bottom": int(pad_h), "left": 0, "right": int(pad_w)},
                        "shape_before_inpaint": list(tile_img.shape),
                        "shape_after_inpaint": list(tile_result.shape),
                        "shape_before_paste": list(target_roi.shape),
                        "shape_after_paste": list(result[y1:y2, x1:x2].shape),
                        "paste_offsets": {"x": int(x1), "y": int(y1)},
                        "clamped": {
                            "left": bool(x1 != x),
                            "top": bool(y1 != y),
                            "right": bool(x2 != x + tile_size),
                            "bottom": bool(y2 != y + tile_size),
                        },
                    }
                )

        if debug:
            debug(
                {
                    "event": "tiled_inpaint",
                    "tile_size": int(tile_size),
                    "overlap": int(overlap),
                    "step": int(step),
                    "tiles": tile_logs,
                }
            )
        return result.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _feather_weight(size: int, overlap: int) -> np.ndarray:
        """Máscara de blending com borda suave."""
        w = np.ones((size, size), dtype=np.float32)
        fade = np.linspace(0, 1, overlap)
        w[:overlap, :] *= fade[:, np.newaxis]
        w[-overlap:, :] *= fade[::-1, np.newaxis]
        w[:, :overlap] *= fade[np.newaxis, :]
        w[:, -overlap:] *= fade[::-1][np.newaxis, :]
        return w

    def _to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img_np).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.half:
            t = t.half()
        return t

    def _mask_to_tensor(self, mask: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(mask).float() / 255.0
        t = t.unsqueeze(0).unsqueeze(0).to(self.device)
        if self.half:
            t = t.half()
        return t

    @staticmethod
    def _pad_to_multiple(img_t, mask_t, multiple=8):
        _, _, h, w = img_t.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        pad = (0, pad_w, 0, pad_h)
        img_t = F.pad(img_t, pad, mode="reflect")
        mask_t = F.pad(mask_t, pad, mode="reflect")
        assert tuple(img_t.shape[-2:]) == (h + pad_h, w + pad_w)
        assert tuple(mask_t.shape[-2:]) == (h + pad_h, w + pad_w)
        return img_t, mask_t, (h, w)

    @staticmethod
    def _unpad(t, pad):
        h, w = pad
        unpadded = t[:, :, :h, :w]
        assert tuple(unpadded.shape[-2:]) == (h, w)
        return unpadded

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze(0).permute(1, 2, 0)
        if self.half:
            arr = arr.float()
        arr = arr.cpu().numpy()
        return (arr * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _opencv_inpaint(img_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback: OpenCV Telea inpainting (sem ML, qualidade inferior)."""
        logger.warning("Usando OpenCV inpainting (fallback — qualidade inferior)")
        return cv2.inpaint(img_np, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def unload(self):
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
