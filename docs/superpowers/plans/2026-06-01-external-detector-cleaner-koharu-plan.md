# External Detector Cleaner Koharu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the current over-detection/mask failures and integrate Bubble-Detector-YOLOv4, NotAnotherBubbleCleaner, manga-cleaner, and the full Koharu renderer behind testable contracts without losing the existing `bubble-mask-cleaner-v2` work.

**Architecture:** Start from `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2`, because it already contains `component_bubble_cleaner`, safe ROI modes, and BubbleMask preservation. External projects are integrated as adapters with strict contracts first, then promoted to primary only after visual validation. UI/form text, speech balloons, CJK/SFX/art text, mask cleaning, and rendering must remain separate routes.

**Tech Stack:** Python 3.12, OpenCV, NumPy, PaddleOCR, PyTorch/ONNXRuntime where optional adapters require it, Tauri/Rust renderer bridge, pytest, cargo test, TS typecheck.

---

## Resumo Operacional

Este plano tem duas partes, nessa ordem:

1. Corrigir o problema atual de detect/mask: parar de aceitar UI, SFX, arte e scans brancos como balão de fala; exigir BubbleMask real antes de inpaint/render automático; manter texto de formulário/UI em rota própria.
2. Implementar o que ainda não foi implementado: Bubble-Detector-YOLOv4, NotAnotherBubbleCleaner original, manga-cleaner original e a troca completa para o renderer Koharu, todos por contratos testáveis e sem contaminar o pipeline principal antes da validação visual.

Regra principal: os motores externos entram primeiro como adaptadores opcionais no worktree `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2`. Eles só viram padrão depois de passar testes unitários e rodada visual. O checkout `N:\TraduzAI` está em outra branch e sujo, então não deve ser usado como base direta desta correção.

## Definição De Feito Por Item Não Implementado

- **Bubble-Detector-YOLOv4:** existe wrapper real em Python, flags de configuração, teste com modelo ausente, teste com detector fake, conversão para o contrato interno de balões e comparação visual contra o detector atual.
- **NotAnotherBubbleCleaner original:** existe adaptador que chama clone/modelo externo quando configurado, e existe implementação própria equivalente para o contrato de máscara quando o original não puder ser vendorizado. O código original não é copiado para o repo sem licença compatível.
- **manga-cleaner original:** existe adaptador opcional por clone/local path/subprocess para usar o runtime externo quando configurado; as ideias já compatíveis ficam no nosso ROI seguro. O source do manga-cleaner não é vendorizado porque o README declara restrição de direitos.
- **Koharu renderer completo:** o backend Rust/Koharu cobre o caminho inteiro de typesetting, não só bridge parcial; recebe BubbleMask/safe box/rotação/fonte/stroke, devolve contrato QA, e falha fechado se não conseguir renderizar.

## Correção Atual Que Deve Vir Antes Dos Motores Externos

- UI/form/layout não pode virar `speech_balloon`.
- Scan branco órfão precisa evidência de formato de balão, contorno e área interna coerente.
- CJK/SFX fora de BubbleMask real deve ser rejeitado para obra marcada como EN.
- `text_pixel_bbox` sem `line_polygons` não pode autorizar fast fill/render automático.
- Inpaint só pode alterar pixels da máscara real; pasteback por retângulo inteiro continua proibido.
- Renderer só recebe rota de balão quando existir BubbleMask/bubble ID real; caso contrário, rota vira UI/caption/review.

## Source Facts To Respect

- [ry-eon/Bubble-Detector-YOLOv4](https://github.com/ry-eon/Bubble-Detector-YOLOv4) is a PyTorch YOLOv4 speech bubble detector prototype. Its README says it is not final, has no GitHub release, and points to separate pretrained weights.
- [LiteralGenie/NotAnotherBubbleCleaner](https://github.com/LiteralGenie/NotAnotherBubbleCleaner) is archived and read-only. It uses Mask R-CNN plus thresholded connected components, centroid-in-mask checks, overlap filters, hole fill, and component shrink to avoid eating bubble outlines.
- [NeTRuNNeRGLiTCH/manga-cleaner](https://github.com/NeTRuNNeRGLiTCH/manga-cleaner) v3 is a Python 3.12/PySide6/OpenCV/NumPy studio app. Its useful runtime ideas are dynamic ROI, snap-to-8 reflection padding, and ONNX LaMa inpaint. Its README states restrictive rights, so do not vendor source unless permission/license is clarified.
- The current run failures are mostly caused by permissive white/UI scans, missing BubbleMask IDs, and mask generation from `text_pixel_bbox` without `line_polygons`.

## Branch And Worktree Policy

- Base execution on `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2`.
- Do not edit `N:\TraduzAI` directly until the branch is validated and deliberately merged or cherry-picked.
- Create branch `codex/external-detector-cleaner-koharu` from `codex/bubble-mask-cleaner-v2`.
- Keep third-party code out of the repo until a license gate passes. If license is absent or incompatible, integrate through an optional local path/subprocess adapter or reimplement the algorithmic contract in our own code.

## File Map

- Modify `pipeline/strip/detect_balloons.py`: disable UI/white fallback scans as speech-balloon sources; route them to typed candidates.
- Modify `pipeline/vision_stack/runtime.py`: enforce route contracts before OCR, attach external detector results, reject CJK/SFX outside real balloon for EN source, and keep UI separate.
- Modify `pipeline/vision_stack/ui_layout.py`: keep UI detection as UI only, never speech balloon.
- Create `pipeline/vision_stack/external_bubble_detector.py`: adapter for Bubble-Detector-YOLOv4 or compatible weights.
- Create `pipeline/inpainter/notanother_adapter.py`: adapter/owned implementation of NotAnotherBubbleCleaner component filtering.
- Modify `pipeline/inpainter/mask_builder.py`: require real BubbleMask plus numeric/valid ID for automatic component cleaner output.
- Modify `pipeline/inpainter/region_strategy.py`: finish manga-cleaner ROI contract and expose snap-to-8/pasteback helpers.
- Modify `pipeline/inpainter/__init__.py`: route AOT/LaMa through safe ROI only when selected and keep mask-only pasteback.
- Modify `pipeline/typesetter/backend_contract.py`, `pipeline/typesetter/rust_backend.py`, `pipeline/typesetter/renderer.py`: finish full Koharu backend switch contract.
- Modify `src-tauri/renderer-bridge/src/lib.rs`: finish missing renderer features needed for parity.
- Add/modify focused tests in `pipeline/tests/` and `src-tauri/renderer-bridge/tests/`.

---

## Phase 0: Baseline And Safety

### Task 0.1: Lock Baseline Worktree

**Files:** none

- [ ] Run:

```powershell
git -C N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2 status --short
git -C N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2 branch --show-current
git -C N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2 log --oneline -n 12
```

- [ ] Expected: branch is `codex/bubble-mask-cleaner-v2`, top commit includes `fix: preserve labeled bubble mask ids`.
- [ ] Create branch:

```powershell
git -C N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2 switch -c codex/external-detector-cleaner-koharu
```

- [ ] If branch exists, switch to it instead:

```powershell
git -C N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2 switch codex/external-detector-cleaner-koharu
```

### Task 0.2: Baseline Tests

**Files:** none

- [ ] Run the current focused tests before any behavior change:

```powershell
cd N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline
python -m pytest tests\test_mask_builder.py tests\test_inpaint_region_strategy.py tests\test_vision_stack_runtime.py tests\test_strip_detect.py tests\test_vision_stack_ui_layout.py -q
```

- [ ] Expected: either pass, or record exact failures before edits.

---

## Phase 1: Stop Current False Positives Before Adding New Engines

### Task 1.1: UI Scan Must Not Become Speech Balloon

**Files:**
- Modify `pipeline/strip/detect_balloons.py`
- Modify `pipeline/vision_stack/runtime.py`
- Test `pipeline/tests/test_strip_detect.py`
- Test `pipeline/tests/test_vision_stack_ui_layout.py`

- [ ] Add failing tests:
  - `test_ui_layout_band_scan_does_not_emit_speech_balloon_by_default`
  - `test_uied_candidate_keeps_ui_form_profile_not_white_balloon`

- [ ] Required behavior:
  - `TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN` defaults to off for speech-balloon detection.
  - UI components may still be detected by `ui_layout.py`, but they must emit `candidate_kind="ui_layout"` and `layout_profile="ui_form"`.
  - UI candidates must not be accepted by `_scan_orphan_white_balloon_blocks`.

- [ ] Run:

```powershell
python -m pytest tests\test_strip_detect.py::test_ui_layout_band_scan_does_not_emit_speech_balloon_by_default tests\test_vision_stack_ui_layout.py::test_uied_candidate_keeps_ui_form_profile_not_white_balloon -q
```

### Task 1.2: White Balloon Orphan Scan Requires Balloon Shape Evidence

**Files:**
- Modify `pipeline/vision_stack/runtime.py`
- Test `pipeline/tests/test_vision_stack_runtime.py`

- [ ] Add failing tests:
  - `test_white_orphan_scan_rejects_rectangular_form_panel`
  - `test_white_orphan_scan_accepts_real_speech_balloon_with_outline`

- [ ] Required behavior:
  - Rectangular form panels like page 3 are rejected as speech balloons.
  - Organic/oval/spiky bubbles with text remain eligible.
  - Rejected candidates get debug reason `white_orphan_rejected_not_speech_balloon_shape`.

- [ ] Run:

```powershell
python -m pytest tests\test_vision_stack_runtime.py -q -k "white_orphan_scan"
```

### Task 1.3: EN Source Must Reject CJK/SFX Outside Real Bubble

**Files:**
- Modify `pipeline/vision_stack/runtime.py`
- Test `pipeline/tests/test_vision_stack_runtime.py`
- Test `pipeline/tests/test_mask_builder.py`

- [ ] Add failing tests:
  - `test_en_source_rejects_cjk_sfx_candidate_without_real_bubble`
  - `test_en_source_keeps_cjk_inside_confirmed_dialogue_bubble_for_review`

- [ ] Required behavior:
  - If `idioma_origem=en` and OCR text is mostly CJK or gibberish-like from CJK strokes, reject unless a real BubbleMask/bubble ID exists.
  - Rejected CJK/SFX must not reach translation, inpaint, or render.
  - Debug reason: `cjk_sfx_outside_real_bubble_en_source`.

---

## Phase 2: Bubble-Detector-YOLOv4 Adapter

### Task 2.1: Add Adapter Contract

**Files:**
- Create `pipeline/vision_stack/external_bubble_detector.py`
- Test `pipeline/tests/test_external_bubble_detector.py`

- [ ] Add a dataclass contract:

```python
@dataclass(frozen=True)
class ExternalBubbleDetection:
    bbox: list[int]
    confidence: float
    source: str
    class_name: str = "speech_bubble"
    mask: np.ndarray | None = None
```

- [ ] Add `detect_bubbles_yolov4(image_rgb, model_dir, confidence_threshold=0.39) -> list[ExternalBubbleDetection]`.
- [ ] The first implementation may load lazily and return `[]` with reason `model_unavailable` when weights are absent.
- [ ] Add test that missing model is non-fatal and emits no detections.

### Task 2.2: Integrate As Candidate Source, Not Sole Truth

**Files:**
- Modify `pipeline/strip/detect_balloons.py`
- Modify `pipeline/vision_stack/runtime.py`
- Test `pipeline/tests/test_strip_detect.py`

- [ ] Add env flags:
  - `TRADUZAI_BUBBLE_DETECTOR_PRIMARY=yolov4|native`
  - `TRADUZAI_BUBBLE_DETECTOR_YOLOV4_DIR=<path>`
  - `TRADUZAI_BUBBLE_DETECTOR_MIN_CONF=0.39`

- [ ] Required behavior:
  - YOLOv4 detections must produce candidate source `bubble_yolov4`.
  - Native detector remains available as fallback/ensemble.
  - UI/form candidates cannot be upgraded to speech balloons just because they are rectangular.

- [ ] Run:

```powershell
python -m pytest tests\test_external_bubble_detector.py tests\test_strip_detect.py -q -k "yolov4 or bubble_detector"
```

### Task 2.3: Visual Gate For YOLOv4

**Files:**
- Modify `pipeline/debug_tools` only if needed
- No production behavior change

- [ ] Run One Second chapter 1 with YOLOv4 disabled and enabled into separate debug folders.
- [ ] Compare page 3, 22, 44, 47, 56.
- [ ] Promote YOLOv4 to primary only if:
  - page 3 UI is not treated as speech balloon;
  - real bubbles on 7/22/56 still detected;
  - SFX/painting text on 44/47 does not increase.

---

## Phase 3: NotAnotherBubbleCleaner Original Behavior

### Task 3.1: License And Source Gate

**Files:**
- Create `docs/third_party/notanotherbubblecleaner.md`

- [ ] Record:
  - repo is archived/read-only;
  - dependency stack is old Mask R-CNN;
  - no vendoring unless license is confirmed compatible.
- [ ] If license is absent or incompatible, implement the algorithmic behavior in our own `notanother_adapter.py` and allow optional external executable/model path only.

### Task 3.2: Component Filter Adapter

**Files:**
- Create `pipeline/inpainter/notanother_adapter.py`
- Modify `pipeline/inpainter/mask_builder.py`
- Test `pipeline/tests/test_notanother_adapter.py`

- [ ] Implement:
  - threshold dark text components;
  - connected component extraction;
  - minimum blob size;
  - centroid inside BubbleMask;
  - minimum overlap with BubbleMask;
  - hole fill;
  - shrink/erode accepted components to preserve outline;
  - return mask plus debug metrics.

- [ ] Required behavior:
  - Components outside BubbleMask are rejected.
  - Bubble outline is not masked.
  - Small noise specks are rejected.

### Task 3.3: Replace Current `component_bubble_cleaner` Internals

**Files:**
- Modify `pipeline/inpainter/mask_builder.py`
- Test `pipeline/tests/test_mask_builder.py`

- [ ] Keep public evidence kind `component_bubble_cleaner`.
- [ ] Internally call `notanother_adapter.build_notanother_text_mask(...)`.
- [ ] Require real BubbleMask and valid `bubble_id` for automatic inpaint/render.
- [ ] If no BubbleMask/bubble ID, return `None` and add reject reason `component_bubble_cleaner_missing_bubble_mask`.

---

## Phase 4: manga-cleaner Original Runtime Ideas

### Task 4.1: License And Source Gate

**Files:**
- Create `docs/third_party/manga-cleaner.md`

- [ ] Record:
  - project says v3 uses Python 3.12, OpenCV, NumPy, PySide6, ONNXRuntime, Dynamic ROI, snap-to-8 reflection padding, and ONNX LaMa;
  - README states copyright/all rights reserved;
  - do not vendor source unless license/permission is clarified.

### Task 4.2: Finish ROI Contract

**Files:**
- Modify `pipeline/inpainter/region_strategy.py`
- Test `pipeline/tests/test_inpaint_region_strategy.py`

- [ ] Ensure `manga_cleaner_roi_from_mask(mask, padding=16, multiple=8)` returns:
  - original source crop bbox;
  - reflected padded crop bbox;
  - pasteback bbox;
  - mask crop;
  - snap-to-8 dimensions.

- [ ] Add tests:
  - `test_manga_cleaner_roi_reflect_pads_to_multiple_of_8`
  - `test_manga_cleaner_roi_pasteback_uses_mask_only`

### Task 4.3: Route AOT/LaMa Through Safe ROI

**Files:**
- Modify `pipeline/vision_stack/runtime.py`
- Modify `pipeline/inpainter/__init__.py`
- Modify `pipeline/inpainter/lama_onnx.py`
- Test `pipeline/tests/test_vision_stack_runtime.py`

- [ ] Support:
  - `TRADUZAI_INPAINT_PRIMARY_ENGINE=aot_manga_roi`
  - `TRADUZAI_INPAINT_PRIMARY_ENGINE=lama_onnx`
  - `TRADUZAI_INPAINT_PRIMARY_ENGINE=manga_cleaner_roi_lama`

- [ ] Required behavior:
  - only pixels under mask are pasted back;
  - no full-rectangle pasteback;
  - ROI respects provided `inpaint_roi_bbox`;
  - output debug records `roi_source_bbox`, `roi_padded_bbox`, `pasteback_pixels`.

---

## Phase 5: Full Koharu Renderer Migration

### Task 5.1: Define Final Renderer Contract

**Files:**
- Modify `pipeline/typesetter/backend_contract.py`
- Modify `pipeline/tests/test_typesetting_backend_contract.py`
- Modify `src-tauri/renderer-bridge/tests/render_contract.rs`

- [ ] Contract fields:
  - `text`, `translated`, `bbox`, `safe_text_box`, `render_bbox`;
  - `rotation_deg`, `line_polygons`;
  - `bubble_mask_path`, `bubble_mask_value`, `bubble_id`;
  - `font_family`, `font_weight`, `font_size_px`;
  - `stroke_width`, `fill_rgb`, `stroke_rgb`;
  - QA return fields: `render_bbox`, `font_size_px`, `fit_status`, `backend`.

### Task 5.2: Make Rust Renderer Feature-Complete Enough For Primary

**Files:**
- Modify `src-tauri/renderer-bridge/src/lib.rs`
- Test `src-tauri/renderer-bridge/tests/*.rs`

- [ ] Implement or verify:
  - multiline wrapping;
  - centered and left layout;
  - bubble-safe bbox from BubbleMask ID;
  - rotation;
  - stroke/outline;
  - Comic Neue Bold default;
  - fallback font lookup;
  - no text outside selected BubbleMask.

- [ ] Run:

```powershell
cd N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\src-tauri\renderer-bridge
cargo test
```

### Task 5.3: Switch Python Typesetter To Koharu Primary Behind Env

**Files:**
- Modify `pipeline/typesetter/rust_backend.py`
- Modify `pipeline/typesetter/renderer.py`
- Test `pipeline/tests/test_typesetting_renderer.py`
- Test `pipeline/tests/test_renderer_backend_parity.py`

- [ ] Add `TRADUZAI_TYPESETTER=v1_python|v2_koharu`.
- [ ] Default remains `v1_python` until visual parity passes.
- [ ] `v2_koharu` must emit the same `render_plan_final.jsonl` contract.
- [ ] If Rust renderer fails, fail the page as blocked, not silent fallback success.

### Task 5.4: Promote Koharu Renderer To Default

**Files:**
- Modify `pipeline/typesetter/renderer.py`
- Modify app settings if needed

- [ ] Only after visual parity:
  - One Second ch1/ch2 pass user review;
  - Monster/other EN chapter pass smoke;
  - no regression in page 3 UI, page 7/22/56 residuals, page 44/47 SFX.

---

## Phase 6: Unified Routing Rules After External Engines

### Task 6.1: Route Matrix

**Files:**
- Create `pipeline/vision_stack/route_contract.py`
- Modify `pipeline/vision_stack/runtime.py`
- Test `pipeline/tests/test_vision_route_contract.py`

- [ ] Implement route categories:
  - `speech_balloon_text`: real BubbleMask + text evidence;
  - `ui_form_text`: UI component evidence, no speech balloon inpaint;
  - `non_balloon_caption`: text, no BubbleMask, rectangular/white page area;
  - `sfx_or_art_text`: CJK/SFX/art evidence, no automatic translation for EN source unless explicitly configured;
  - `reject_noise`: gibberish/low evidence.

- [ ] Each route explicitly sets:
  - `route_action`;
  - `render_policy`;
  - `inpaint_policy`;
  - `qa_flags`;
  - `blocks_export`.

### Task 6.2: Export Gate Must Enforce Route Contract

**Files:**
- Modify `pipeline/qa/export_gate.py`
- Test `pipeline/tests/test_export_gate.py`

- [ ] Required behavior:
  - UI/form text with UI route does not raise speech balloon mask errors.
  - Speech balloon text without BubbleMask blocks export before render.
  - CJK/SFX outside real bubble is not rendered and does not create false success.
  - `export_gate=BLOCK` still propagates to app.

---

## Phase 7: Validation

### Task 7.1: Unit Tests

Run:

```powershell
cd N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline
python -m pytest tests\test_external_bubble_detector.py tests\test_notanother_adapter.py tests\test_mask_builder.py tests\test_inpaint_region_strategy.py tests\test_vision_stack_runtime.py tests\test_strip_detect.py tests\test_vision_stack_ui_layout.py tests\test_export_gate.py -q
```

### Task 7.2: Rust Renderer Tests

Run:

```powershell
cd N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\src-tauri\renderer-bridge
cargo test
```

### Task 7.3: TypeScript/Rust App Smoke

Run:

```powershell
cd N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2
npm run typecheck
cd src-tauri
cargo test
```

### Task 7.4: Visual Runs

Run One Second chapter 1 and 2 with:

```powershell
$env:TRADUZAI_TEXT_MASK_ENGINE='component_bubble_cleaner'
$env:TRADUZAI_INPAINT_PRIMARY_ENGINE='aot_manga_roi'
$env:TRADUZAI_TYPESETTER='v2_koharu'
```

Then inspect:

- page 2/3 UI form;
- page 7 small balloons;
- page 22 `Sim, não funciona`, SFX, city signs;
- page 44/47 SFX/art;
- page 56 residual text.

Pass criteria:

- UI is not treated as speech balloon.
- CJK/SFX outside real bubble is not rendered as translated text.
- Real balloon original text is removed without eating border.
- Renderer text is legible and inside BubbleMask/safe box.
- Export gate blocks only real unresolved visual failures.

---

## Agent Split

- **Agent A: Detect/Route** owns Phase 1, Phase 2, and Phase 6.
- **Agent B: Cleaner/Inpaint** owns Phase 3 and Phase 4.
- **Agent C: Koharu Renderer** owns Phase 5.
- **Agent D: QA/Validation** owns Phase 7 and visual comparison reports.

Execution order must be A -> B -> C -> D. Do not start C as default until A and B stabilize the visual input contract.

## Review Gates

- Gate 1: after Phase 1, page 3/44/47 false positives must be reduced before integrating external engines.
- Gate 2: after Phase 3, `component_bubble_cleaner` cannot be `fast_fill_allowed` without real BubbleMask.
- Gate 3: after Phase 4, AOT/LaMa pasteback cannot alter pixels outside the mask.
- Gate 4: after Phase 5, Koharu renderer must match current Python output contract and fail closed.
- Gate 5: after Phase 7, user reviews visual outputs before broad reruns.
