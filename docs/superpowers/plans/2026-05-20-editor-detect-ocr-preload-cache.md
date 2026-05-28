# Editor Detect/OCR Preload Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make editor Detect/OCR feel almost instant by precomputing page vision results in the persistent fast worker and applying a valid cache on click.

**Architecture:** Keep the current Tauri -> Rust -> Python fast worker boundary. Add an editor-only vision cache keyed by image, source language, engine preset, page index, and layer geometry; preload cache entries from the editor while preserving the existing foreground action fallback. The automatic chapter pipeline must keep its current behavior.

**Tech Stack:** React 19, Zustand, Tauri v2 commands in Rust, Python 3.12, existing `pipeline/fast_page_server.py`, existing `pipeline/main.py` editor page actions, focused pytest plus `cargo check` and `npx tsc --noEmit`.

---

## Scope

This plan targets the manual editor only:

- Detect button should apply a precomputed full-page detect+OCR result when valid.
- OCR button should apply a precomputed OCR result for the current text layer geometry when valid.
- If cache is missing, stale, or still running, the existing foreground action must still work.
- Region/lasso actions keep the current synchronous behavior in the first implementation pass.
- The full automatic chapter pipeline is not migrated to this cache.

## File Structure

- Create `pipeline/editor_vision_cache.py`
  - Owns cache key generation, cache paths, cache JSON read/write, and stale checks.
  - Does not mutate `project.json`.

- Modify `pipeline/main.py`
  - Split editor detect/OCR page actions into compute/apply helpers.
  - Save cache after foreground compute.
  - Apply cache before compute when a valid cache exists.
  - Add preload runners callable by the fast worker.

- Modify `pipeline/fast_page_server.py`
  - Add `editor_preload_detect_ocr` and `editor_preload_ocr_layers`.
  - Use the existing single fast-worker request serialization for the first pass.
  - Let foreground actions reuse a finished cache or run the current fallback path when cache is absent.

- Modify `pipeline/tests/test_fast_page_server.py`
  - Cover preload request routing, cache-ready response, in-flight wait behavior, and foreground fallback.

- Create `pipeline/tests/test_editor_vision_cache.py`
  - Cover key stability, stale image invalidation, layer geometry invalidation, and corrupt cache handling.

- Modify `src-tauri/src/commands/pipeline.rs`
  - Add a fast-worker request helper for preload commands.
  - Add a Tauri command `preload_editor_vision_page`.
  - Do not emit blocking progress UI for preload.

- Modify `src-tauri/src/lib.rs`
  - Register `preload_editor_vision_page`.

- Modify `src/lib/tauri.ts`
  - Add `preloadEditorVisionPage`.

- Modify `src/lib/editorBackend.ts`
  - Add optional backend method `preloadEditorVisionPage`.

- Modify `src/lib/editorBackends/tauriEditorBackend.ts`
  - Expose the new Tauri binding.

- Modify `src/lib/stores/editorStore.ts`
  - Add preload scheduling state and action.
  - Schedule current page preload after `loadCurrentPage`.
  - Keep foreground click path unchanged except it benefits from Python cache.

---

### Task 1: Python Cache Module

**Files:**
- Create: `pipeline/editor_vision_cache.py`
- Test: `pipeline/tests/test_editor_vision_cache.py`

- [x] **Step 1: Write failing cache tests**

Create `pipeline/tests/test_editor_vision_cache.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from editor_vision_cache import (
    EditorVisionCacheKey,
    build_detect_ocr_cache_key,
    build_ocr_layers_cache_key,
    read_cache_entry,
    write_cache_entry,
)


def test_detect_cache_key_changes_when_image_mtime_changes(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"first")

    first = build_detect_ocr_cache_key(
        project_path=tmp_path / "project.json",
        page_index=0,
        image_path=image,
        idioma_origem="en",
        engine_preset_id="",
        schema_version=1,
    )

    image.write_bytes(b"second")
    second = build_detect_ocr_cache_key(
        project_path=tmp_path / "project.json",
        page_index=0,
        image_path=image,
        idioma_origem="en",
        engine_preset_id="",
        schema_version=1,
    )

    assert first.digest != second.digest


def test_ocr_layers_cache_key_changes_when_layer_bbox_changes(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    layers = [{"id": "a", "bbox": [1, 2, 3, 4]}, {"id": "b", "bbox": [10, 20, 30, 40]}]

    first = build_ocr_layers_cache_key(
        project_path=tmp_path / "project.json",
        page_index=1,
        image_path=image,
        layers=layers,
        idioma_origem="ja",
        engine_preset_id="manga",
        schema_version=1,
    )
    layers[0]["bbox"] = [1, 2, 33, 44]
    second = build_ocr_layers_cache_key(
        project_path=tmp_path / "project.json",
        page_index=1,
        image_path=image,
        layers=layers,
        idioma_origem="ja",
        engine_preset_id="manga",
        schema_version=1,
    )

    assert first.digest != second.digest


def test_cache_round_trip_and_corrupt_file_handling(tmp_path: Path) -> None:
    key = EditorVisionCacheKey(kind="detect_ocr", digest="abc123", cache_dir=tmp_path)
    payload = {"status": "ready", "page_index": 0, "texts": [{"text": "HELLO"}]}

    write_cache_entry(key, payload)

    assert read_cache_entry(key) == payload
    key.path.write_text("{bad json", encoding="utf-8")
    assert read_cache_entry(key) is None
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_editor_vision_cache.py -q
```

Expected: fails with `ModuleNotFoundError: No module named 'editor_vision_cache'`.

- [x] **Step 3: Implement `pipeline/editor_vision_cache.py`**

Add:

```python
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EditorVisionCacheKey:
    kind: str
    digest: str
    cache_dir: Path

    @property
    def path(self) -> Path:
        return self.cache_dir / f"{self.kind}-{self.digest}.json"


def _stat_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _normal_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _cache_dir(project_path: Path) -> Path:
    return project_path.parent / "layers" / "vision-cache"


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _layer_geometry(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for layer in layers:
        bbox = layer.get("bbox") or layer.get("layout_bbox") or layer.get("source_bbox")
        normalized.append({"id": str(layer.get("id") or ""), "bbox": bbox})
    return normalized


def build_detect_ocr_cache_key(
    *,
    project_path: Path,
    page_index: int,
    image_path: Path,
    idioma_origem: str,
    engine_preset_id: str,
    schema_version: int,
) -> EditorVisionCacheKey:
    payload = {
        "kind": "detect_ocr",
        "schema_version": schema_version,
        "project": str(project_path.resolve()),
        "page_index": int(page_index),
        "image": _stat_signature(image_path),
        "idioma_origem": _normal_text(idioma_origem),
        "engine_preset_id": _normal_text(engine_preset_id),
    }
    return EditorVisionCacheKey("detect_ocr", _digest(payload), _cache_dir(project_path))


def build_ocr_layers_cache_key(
    *,
    project_path: Path,
    page_index: int,
    image_path: Path,
    layers: list[dict[str, Any]],
    idioma_origem: str,
    engine_preset_id: str,
    schema_version: int,
) -> EditorVisionCacheKey:
    payload = {
        "kind": "ocr_layers",
        "schema_version": schema_version,
        "project": str(project_path.resolve()),
        "page_index": int(page_index),
        "image": _stat_signature(image_path),
        "layers": _layer_geometry(layers),
        "idioma_origem": _normal_text(idioma_origem),
        "engine_preset_id": _normal_text(engine_preset_id),
    }
    return EditorVisionCacheKey("ocr_layers", _digest(payload), _cache_dir(project_path))


def read_cache_entry(key: EditorVisionCacheKey) -> dict[str, Any] | None:
    try:
        data = json.loads(key.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("status") == "ready" else None


def write_cache_entry(key: EditorVisionCacheKey, payload: dict[str, Any]) -> None:
    key.cache_dir.mkdir(parents=True, exist_ok=True)
    temp = key.path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, key.path)
```

- [x] **Step 4: Run test to verify it passes**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_editor_vision_cache.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

Skipped for now: user requested execution in the current dirty checkout; no files have been staged or committed.

```powershell
git add pipeline/editor_vision_cache.py pipeline/tests/test_editor_vision_cache.py
git commit -m "feat: add editor vision cache keys"
```

---

### Task 2: Refactor Editor Detect/OCR Helpers Without Behavior Change

**Files:**
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_fast_page_server.py`

- [x] **Step 1: Add regression test around existing fast worker actions**

Extend `test_fast_page_session_routes_editor_detect_and_ocr_actions` in `pipeline/tests/test_fast_page_server.py` so it asserts the runner receives `idioma_origem` and `engine_preset_id` unchanged:

```python
assert calls == [
    ("detect", str(project_path), 2, None, {"idioma_origem": "ja", "idioma_destino": None, "engine_preset_id": "manga"}),
    ("ocr", str(project_path), 2, None, {"idioma_origem": "ja", "idioma_destino": None, "engine_preset_id": "manga"}),
]
```

Use request payloads that include:

```python
"idioma_origem": "ja",
"engine_preset_id": "manga",
```

- [x] **Step 2: Run test before refactor**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_fast_page_server.py::test_fast_page_session_routes_editor_detect_and_ocr_actions -q
```

Expected: pass before refactor.

- [x] **Step 3: Extract helpers in `pipeline/main.py`**

Refactor without changing command outputs:

```python
def _load_editor_project_page(project_path: Path, page_idx: int) -> tuple[dict, Path, dict, Path, str, Path]:
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)
    work_dir = project_path.parent
    _attach_work_dir_log_handler(work_dir)
    project["_work_dir"] = str(work_dir)
    page = project["paginas"][page_idx]
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    img_name = Path(original_rel).name
    orig_img = work_dir / original_rel
    if not orig_img.exists():
        orig_img = work_dir / "originals" / img_name
    if not orig_img.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            orig_img = candidate
    return project, work_dir, page, orig_img, img_name, Path(original_rel)
```

Then make `_run_detect_page` and `_run_ocr_page` call this helper. Keep the emitted progress and complete events identical.

- [x] **Step 4: Run focused test after refactor**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_fast_page_server.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Skipped for now: user requested execution in the current dirty checkout; no files have been staged or committed.

```powershell
git add pipeline/main.py pipeline/tests/test_fast_page_server.py
git commit -m "refactor: isolate editor page action loading"
```

---

### Task 3: Save And Apply Detect/OCR Cache In Python

**Files:**
- Modify: `pipeline/main.py`
- Modify: `pipeline/editor_vision_cache.py`
- Test: `pipeline/tests/test_editor_vision_cache.py`

- [x] **Step 1: Add cache payload tests**

Append to `pipeline/tests/test_editor_vision_cache.py`:

```python
from editor_vision_cache import build_detect_ocr_payload, build_ocr_layers_payload


def test_detect_payload_contains_page_patch() -> None:
    payload = build_detect_ocr_payload(
        page_index=0,
        text_layers=[{"id": "t1", "original": "HELLO"}],
        inpaint_blocks=[{"bbox": [1, 2, 3, 4]}],
    )

    assert payload["status"] == "ready"
    assert payload["kind"] == "detect_ocr"
    assert payload["page_index"] == 0
    assert payload["text_layers"][0]["original"] == "HELLO"
    assert payload["inpaint_blocks"][0]["bbox"] == [1, 2, 3, 4]


def test_ocr_payload_contains_layer_text_updates() -> None:
    payload = build_ocr_layers_payload(
        page_index=3,
        layer_updates=[{"id": "a", "original": "OK", "ocr_confidence": 0.91, "confianca_ocr": 0.91}],
    )

    assert payload["status"] == "ready"
    assert payload["kind"] == "ocr_layers"
    assert payload["page_index"] == 3
    assert payload["layer_updates"][0]["id"] == "a"
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_editor_vision_cache.py -q
```

Expected: fails because payload builders do not exist.

- [x] **Step 3: Add payload builders**

Add to `pipeline/editor_vision_cache.py`:

```python
def build_detect_ocr_payload(*, page_index: int, text_layers: list[dict[str, Any]], inpaint_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "ready",
        "kind": "detect_ocr",
        "schema_version": 1,
        "page_index": int(page_index),
        "text_layers": text_layers,
        "inpaint_blocks": inpaint_blocks,
    }


def build_ocr_layers_payload(*, page_index: int, layer_updates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "ready",
        "kind": "ocr_layers",
        "schema_version": 1,
        "page_index": int(page_index),
        "layer_updates": layer_updates,
    }
```

- [x] **Step 4: Wire cache into `main.py`**

In `_run_detect_page`, before computing full OCR, build the `detect_ocr` key for global actions only:

```python
from editor_vision_cache import build_detect_ocr_cache_key, read_cache_entry

is_regional = _region_bbox(region) is not None
cache_key = None
if not is_regional:
    cache_key = build_detect_ocr_cache_key(
        project_path=project_path,
        page_index=page_idx,
        image_path=orig_img,
        idioma_origem=project.get("idioma_origem", "en"),
        engine_preset_id=project.get("engine_preset_id", ""),
        schema_version=1,
    )
    cached = read_cache_entry(cache_key)
    if cached:
        page["text_layers"] = cached["text_layers"]
        page["inpaint_blocks"] = cached["inpaint_blocks"]
        _sync_page_legacy_aliases(page)
        _save_project_json(project_path, project)
        emit_progress("render", 80, 95, message="Aplicando deteccao em cache...")
        out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
        render_page_image(project, page_idx, str(out_img))
        emit("complete", output_path=str(out_img))
        return
```

After existing detect compute updates `page["text_layers"]` and `page["inpaint_blocks"]`, write the cache:

```python
from editor_vision_cache import build_detect_ocr_payload, write_cache_entry

if cache_key is not None:
    write_cache_entry(
        cache_key,
        build_detect_ocr_payload(
            page_index=page_idx,
            text_layers=page.get("text_layers") or [],
            inpaint_blocks=page.get("inpaint_blocks") or [],
        ),
    )
```

In `_run_ocr_page`, build `ocr_layers` cache for global actions only after `layers` is resolved:

```python
from editor_vision_cache import build_ocr_layers_cache_key, read_cache_entry

is_regional = _region_bbox(region) is not None
cache_key = None
if not is_regional:
    cache_key = build_ocr_layers_cache_key(
        project_path=project_path,
        page_index=page_idx,
        image_path=orig_img,
        layers=layers,
        idioma_origem=project.get("idioma_origem", "en"),
        engine_preset_id=project.get("engine_preset_id", ""),
        schema_version=1,
    )
    cached = read_cache_entry(cache_key)
    if cached:
        updates = {item["id"]: item for item in cached.get("layer_updates", []) if isinstance(item, dict) and item.get("id")}
        for layer in layers:
            update = updates.get(layer.get("id"))
            if update:
                layer["original"] = update.get("original", "")
                layer["ocr_confidence"] = update.get("ocr_confidence", 0.0)
                layer["confianca_ocr"] = update.get("confianca_ocr", layer["ocr_confidence"])
        page["text_layers"] = _page_text_layers_for_renderer(page, page_idx)
        _sync_page_legacy_aliases(page)
        _save_project_json(project_path, project)
        emit_progress("render", 80, 95, message="Aplicando OCR em cache...")
        out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
        render_page_image(project, page_idx, str(out_img))
        emit("complete", output_path=str(out_img))
        return
```

After OCR loop, write:

```python
from editor_vision_cache import build_ocr_layers_payload, write_cache_entry

if cache_key is not None:
    write_cache_entry(
        cache_key,
        build_ocr_layers_payload(
            page_index=page_idx,
            layer_updates=[
                {
                    "id": layer.get("id"),
                    "original": layer.get("original", ""),
                    "ocr_confidence": layer.get("ocr_confidence", layer.get("confianca_ocr", 0.0)),
                    "confianca_ocr": layer.get("confianca_ocr", layer.get("ocr_confidence", 0.0)),
                }
                for layer in layers
                if isinstance(layer, dict) and layer.get("id")
            ],
        ),
    )
```

- [x] **Step 5: Run focused Python tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_editor_vision_cache.py pipeline\tests\test_fast_page_server.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

Skipped for now: user requested execution in the current dirty checkout; no files have been staged or committed.

```powershell
git add pipeline/main.py pipeline/editor_vision_cache.py pipeline/tests/test_editor_vision_cache.py
git commit -m "feat: cache editor detect and ocr results"
```

---

### Task 4: Fast Worker Preload Requests

**Files:**
- Modify: `pipeline/fast_page_server.py`
- Modify: `pipeline/main.py`
- Modify: `pipeline/tests/test_fast_page_server.py`

- [x] **Step 1: Add fast worker preload tests**

Append a focused test to `pipeline/tests/test_fast_page_server.py`:

```python
def test_fast_page_session_routes_preload_request(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []
    project_path = tmp_path / "project.json"
    project_path.write_text('{"paginas":[]}', encoding="utf-8")

    def preload_detect(project_path_arg: Path, page_index: int, options: dict | None) -> dict:
        calls.append((str(project_path_arg), page_index))
        return {"cache": "ready", "kind": "detect_ocr"}

    session = FastPageSession(
        pipeline_runner=lambda config_path: None,
        preload_detect_runner=preload_detect,
    )

    events = session.handle({
        "type": "editor_preload_detect_ocr",
        "project_path": str(project_path),
        "page_index": 4,
        "idioma_origem": "en",
        "engine_preset_id": "",
    })

    assert events[0]["type"] in {"accepted", "ready"}
    assert events[0]["target"] == "detect_ocr"
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_fast_page_server.py::test_fast_page_session_routes_preload_request -q
```

Expected: fails because `preload_detect_runner` and request handling do not exist.

- [x] **Step 3: Add preload runner injection**

In `pipeline/fast_page_server.py`, extend types and constructor:

```python
PreloadDetectRunner = Callable[[Path, int, PageActionOptions | None], dict]
PreloadOcrRunner = Callable[[Path, int, PageActionOptions | None], dict]
```

Add constructor args:

```python
preload_detect_runner: PreloadDetectRunner | None = None,
preload_ocr_runner: PreloadOcrRunner | None = None,
```

Assign:

```python
self._preload_detect_runner = preload_detect_runner or _load_default_preload_detect_runner()
self._preload_ocr_runner = preload_ocr_runner or _load_default_preload_ocr_runner()
```

- [x] **Step 4: Add request handling**

In `FastPageSession.handle`:

```python
if request_type == "editor_preload_detect_ocr":
    return self._handle_editor_preload(request, self._preload_detect_runner, "detect_ocr")
if request_type == "editor_preload_ocr_layers":
    return self._handle_editor_preload(request, self._preload_ocr_runner, "ocr_layers")
```

Add method:

```python
def _handle_editor_preload(self, request: dict, runner, target: str) -> list[dict]:
    project_path = Path(_require_text(request, "project_path"))
    page_index = _require_int(request, "page_index")
    options = _page_action_options_from_request(request)
    result = runner(project_path, page_index, options)
    state = str(result.get("cache") or "ready")
    return [{"type": "ready" if state == "ready" else "accepted", "target": target, "cache": state, "session_id": self.session_id}]
```

Use synchronous preload in this task. Do not add thread concurrency until the cache path is proven.

- [x] **Step 5: Add default preload runners**

Implementation note: default preload runners were made cache-only after review. They write `layers/vision-cache` but do not save `project.json`, render output, or emit foreground events.

In `pipeline/main.py`, add:

```python
def _preload_detect_ocr_page(project_path: Path, page_idx: int, language_options: dict | None = None) -> dict:
    _run_detect_page(project_path, page_idx, None, language_options)
    return {"cache": "ready", "kind": "detect_ocr"}


def _preload_ocr_layers_page(project_path: Path, page_idx: int, language_options: dict | None = None) -> dict:
    _run_ocr_page(project_path, page_idx, None, language_options)
    return {"cache": "ready", "kind": "ocr_layers"}
```

In `pipeline/fast_page_server.py`, add loaders:

```python
def _load_default_preload_detect_runner() -> PreloadDetectRunner:
    from main import _preload_detect_ocr_page
    return _preload_detect_ocr_page


def _load_default_preload_ocr_runner() -> PreloadOcrRunner:
    from main import _preload_ocr_layers_page
    return _preload_ocr_layers_page
```

- [x] **Step 6: Run focused tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_fast_page_server.py -q
```

Expected: all fast page tests pass.

- [ ] **Step 7: Commit**

Skipped for now: user requested execution in the current dirty checkout; no files have been staged or committed.

```powershell
git add pipeline/fast_page_server.py pipeline/main.py pipeline/tests/test_fast_page_server.py
git commit -m "feat: add editor vision preload requests"
```

---

### Task 5: Rust Preload Command

**Files:**
- Modify: `src-tauri/src/commands/pipeline.rs`
- Modify: `src-tauri/src/lib.rs`

- [x] **Step 1: Add Rust unit coverage for preload request shape**

Inside the existing test module in `src-tauri/src/commands/pipeline.rs`, add a pure helper test by first adding this helper outside the test module:

```rust
fn editor_preload_request_json(
    request_type: &str,
    project_file: &Path,
    page_index: u32,
    idioma_origem: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "type": request_type,
        "project_path": project_file.to_string_lossy().to_string(),
        "page_index": page_index,
        "idioma_origem": clean_language(idioma_origem).unwrap_or_else(|| "en".to_string()),
    })
}
```

Then add the test:

```rust
#[test]
fn editor_preload_request_json_uses_project_file_and_language() {
    let project_file = std::path::Path::new("N:/work/project.json");
    let request = super::editor_preload_request_json(
        "editor_preload_detect_ocr",
        project_file,
        7,
        Some("ja"),
    );

    assert_eq!(request["type"], "editor_preload_detect_ocr");
    assert_eq!(request["page_index"], 7);
    assert_eq!(request["idioma_origem"], "ja");
    assert!(request["project_path"].as_str().unwrap().ends_with("project.json"));
}
```

- [x] **Step 2: Run Rust test to verify helper compiles**

Run:

```powershell
cd src-tauri
cargo test editor_preload_request_json_uses_project_file_and_language
```

Expected: pass.

- [x] **Step 3: Add Tauri command**

In `src-tauri/src/commands/pipeline.rs`, add:

```rust
#[tauri::command]
pub async fn preload_editor_vision_page(
    app: AppHandle,
    project_path: String,
    page_index: u32,
    #[allow(non_snake_case)] idioma_origem: Option<String>,
) -> Result<String, String> {
    let pf = resolve_project_json_path(&project_path)?;
    let detect_request = editor_preload_request_json(
        "editor_preload_detect_ocr",
        &pf,
        page_index,
        idioma_origem.as_deref(),
    );
    run_fast_page_worker_request(&app, detect_request).await?;
    Ok("queued".to_string())
}
```

This first command schedules only full-page detect+OCR. Add OCR-layer preload after frontend usage proves the first path is useful.

- [x] **Step 4: Register command**

In `src-tauri/src/lib.rs`, add to `tauri::generate_handler!`:

```rust
commands::pipeline::preload_editor_vision_page,
```

Place it next to `warmup_visual_stack`, `detect_page`, and `ocr_page`.

- [x] **Step 5: Run Rust checks**

Run:

```powershell
cd src-tauri
cargo fmt --check
cargo check
```

Expected: both pass.

- [ ] **Step 6: Commit**

Skipped for now: user requested execution in the current dirty checkout; no files have been staged or committed.

```powershell
git add src-tauri/src/commands/pipeline.rs src-tauri/src/lib.rs
git commit -m "feat: expose editor vision preload command"
```

---

### Task 6: Frontend Binding And Store Scheduling

**Files:**
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/editorBackend.ts`
- Modify: `src/lib/editorBackends/tauriEditorBackend.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

- [x] **Step 1: Add frontend API test**

In `src/lib/stores/__tests__/editorBitmapTools.test.ts`, extend the existing hoisted mock destructuring:

```ts
const {
  updateBrushRegion,
  updateMaskRegion,
  updateRecoveryRegion,
  updateReinpaintRegion,
  healInpaintRegion,
  patchEditorTextLayer,
  processBlock,
  runProcessRegion,
  runPageActionWithOptionalMask,
  writeMaskFromPng,
  loadEditorPage,
  preloadEditorVisionPage,
} = vi.hoisted(() => ({
```

Add the mock function inside the existing `vi.hoisted` returned object:

```ts
preloadEditorVisionPage: vi.fn(async () => "queued"),
```

Add the method to the mocked backend returned by `getEditorBackend`:

```ts
preloadEditorVisionPage,
```

Then add the test using the existing `makePage` and `makeProject` helpers:

```ts
it("requests editor vision preload after loading a page", async () => {
  const page = makePage();
  useAppStore.setState({ project: makeProject(page) });
  useEditorStore.setState({ currentPage: null, currentPageIndex: 0 });

  await useEditorStore.getState().loadCurrentPage();

  expect(preloadEditorVisionPage).toHaveBeenCalledWith({
    project_path: "D:/tmp/project.json",
    page_index: 0,
    idioma_origem: "en",
  });
});
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
npx vitest run src/lib/stores/__tests__/editorBitmapTools.test.ts -t "requests editor vision preload"
```

Expected: fails because the backend method and store scheduling do not exist.

- [x] **Step 3: Add Tauri binding**

In `src/lib/tauri.ts`:

```ts
export async function preloadEditorVisionPage(args: {
  project_path: string;
  page_index: number;
  idioma_origem?: string;
}): Promise<string> {
  return await invoke("preload_editor_vision_page", buildPlainPageCommandArgs(args));
}
```

- [x] **Step 4: Extend editor backend types**

In `src/lib/editorBackend.ts`, add:

```ts
preloadEditorVisionPage?(args: {
  project_path: string;
  page_index: number;
  idioma_origem?: string;
}): Promise<string>;
```

In `src/lib/editorBackends/tauriEditorBackend.ts`, import `preloadEditorVisionPage` from `../tauri` and add it to the existing `tauriEditorBackend` object:

```ts
export const tauriEditorBackend: EditorBackendApi = {
  saveProjectJson,
  loadEditorPage,
  patchEditorTextLayer,
  setEditorLayerVisibility,
  updateMaskRegion,
  updateBrushRegion,
  updateRecoveryRegion,
  updateReinpaintRegion,
  writeMaskFromPng,
  writeHealingMask,
  healInpaintRegion,
  renderPreviewPage,
  runPageActionWithOptionalMask,
  runProcessRegion,
  retypesetPage,
  detectPage,
  ocrPage,
  translatePage,
  reinpaintPage,
  processBlock,
  preloadEditorVisionPage,
};
```

- [x] **Step 5: Schedule preload in store**

Implementation note: page-load preload is target-aware and delayed by a short idle timer. Pages with existing text layers request `ocr_layers`; pages without layers request `detect_ocr`. The delayed scheduling reduces the chance that invisible preload competes with immediate foreground clicks.

In `src/lib/stores/editorStore.ts`, add a helper near other local helpers:

```ts
async function scheduleEditorVisionPreload(pageIndex: number) {
  const path = projectPath();
  if (!path) return;
  const { preloadEditorVisionPage } = await getTauriEditorApi();
  if (!preloadEditorVisionPage) return;
  await preloadEditorVisionPage({
    project_path: path,
    page_index: pageIndex,
    idioma_origem: projectLanguages().idioma_origem,
  });
}
```

At the end of successful `loadCurrentPage`, after `currentPage` is set, call without blocking UI:

```ts
void scheduleEditorVisionPreload(get().currentPageIndex).catch((error) => {
  console.warn("[EditorVisionPreload] falhou:", error);
});
```

Do not set `isRetypesetting` or `activePageAction` for preload.

- [x] **Step 6: Run frontend test**

Run:

```powershell
npx vitest run src/lib/stores/__tests__/editorBitmapTools.test.ts -t "requests editor vision preload"
```

Expected: pass.

- [x] **Step 7: Run TypeScript check**

Run:

```powershell
npx tsc --noEmit
```

Expected: pass.

- [ ] **Step 8: Commit**

Skipped for now: user requested execution in the current dirty checkout; no files have been staged or committed.

```powershell
git add src/lib/tauri.ts src/lib/editorBackend.ts src/lib/editorBackends/tauriEditorBackend.ts src/lib/stores/editorStore.ts src/lib/stores/__tests__/editorBitmapTools.test.ts
git commit -m "feat: preload editor vision results from page load"
```

---

### Task 7: Foreground Click Uses Cache And Keeps Fallback

**Files:**
- Modify: `pipeline/main.py`
- Modify: `pipeline/tests/test_fast_page_server.py`

- [x] **Step 1: Add test that cached detect avoids runner recompute**

Add a Python test around the cache module or a lightweight integration wrapper. The assertion must prove that a second detect request returns complete without calling the expensive fake runner twice:

```python
def test_editor_detect_cache_is_reused_for_second_request(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}
    project_path = tmp_path / "project.json"
    originals = tmp_path / "originals"
    translated = tmp_path / "translated"
    images = tmp_path / "images"
    originals.mkdir()
    translated.mkdir()
    images.mkdir()
    image_path = originals / "001.png"
    image_path.write_bytes(b"fake image bytes")
    project_path.write_text(
        json.dumps({
            "idioma_origem": "en",
            "idioma_destino": "pt-BR",
            "paginas": [{
                "numero": 1,
                "arquivo_original": "originals/001.png",
                "arquivo_traduzido": "translated/001.png",
                "image_layers": {
                    "base": {"key": "base", "path": "originals/001.png", "visible": True, "locked": True},
                    "rendered": {"key": "rendered", "path": "translated/001.png", "visible": True, "locked": True},
                },
                "text_layers": [],
                "textos": [],
                "inpaint_blocks": [],
            }],
        }),
        encoding="utf-8",
    )

    def expensive_detect(*args, **kwargs):
        calls["count"] += 1
        return {
            "texts": [{"text": "HELLO", "bbox": [1, 2, 30, 40], "confidence": 0.9, "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [1, 2, 30, 40], "confidence": 0.9}],
        }

    monkeypatch.setattr("ocr.detector.run_ocr", expensive_detect)
    monkeypatch.setattr("main.render_page_image", lambda *args, **kwargs: None)

    _run_detect_page(project_path, 0, None, {"idioma_origem": "en", "engine_preset_id": ""})
    _run_detect_page(project_path, 0, None, {"idioma_origem": "en", "engine_preset_id": ""})

    assert calls["count"] == 1
```

Add imports in the test file:

```python
import json
from main import _run_detect_page
```

- [x] **Step 2: Run test to verify current behavior**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests/test_fast_page_server.py -k "cache_is_reused" -q
```

Expected: fail until the cache check is wired fully into `_run_detect_page`.

- [x] **Step 3: Ensure `_run_detect_page` checks cache before expensive compute**

No production change was needed here; Task 3 had already wired the cache read before `run_ocr`. Task 7 added regression coverage for second-click reuse and regional bypass.

Move the cache read before:

```python
ocr_data = run_ocr(...)
```

Keep this exact rule:

```python
if _region_bbox(region) is not None:
    cache_key = None
```

This avoids applying full-page cache to lasso/mask actions.

- [x] **Step 4: Run focused tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_editor_vision_cache.py pipeline\tests\test_fast_page_server.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Skipped for now: user requested execution in the current dirty checkout; no files have been staged or committed.

```powershell
git add pipeline/main.py pipeline/tests/test_fast_page_server.py
git commit -m "fix: reuse editor vision cache on foreground actions"
```

---

### Task 8: Measured Validation

**Files:**
- Create: `scripts/bench_editor_vision_preload.py`
- No production code changes unless the benchmark exposes a bug.

- [x] **Step 1: Add benchmark script**

Create `scripts/bench_editor_vision_preload.py`:

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pipeline.main import _preload_detect_ocr_page, _run_detect_page


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_json")
    parser.add_argument("--page", type=int, default=0)
    args = parser.parse_args()

    project_path = Path(args.project_json)
    page = int(args.page)

    t0 = time.perf_counter()
    _preload_detect_ocr_page(project_path, page, {"idioma_origem": "en", "engine_preset_id": ""})
    preload_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    _run_detect_page(project_path, page, None, {"idioma_origem": "en", "engine_preset_id": ""})
    cached_click_sec = time.perf_counter() - t1

    print(json.dumps({
        "project_json": str(project_path),
        "page": page,
        "preload_sec": round(preload_sec, 3),
        "cached_click_sec": round(cached_click_sec, 3),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Run benchmark on one real manual project**

Measured on a copied one-page project under `.codex-tmp/editor_vision_bench_20260519_234830/project.json`: preload `13.025s`, cached click `0.224s`, speedup `58.13x`.

Run with a known local project:

```powershell
$project = Get-ChildItem data\works -Recurse -Filter project.json | Select-Object -First 1 -ExpandProperty FullName
pipeline\venv\Scripts\python.exe scripts\bench_editor_vision_preload.py $project --page 0
```

Expected:

```json
{
  "preload_sec": 1.0,
  "cached_click_sec": 0.3
}
```

The exact numbers can differ by hardware. The acceptance criterion is that `cached_click_sec` is at least 3x faster than `preload_sec` and subjectively below one second on an already warmed worker.

- [x] **Step 3: Run full focused verification bundle**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_editor_vision_cache.py pipeline\tests\test_fast_page_server.py -q
npx vitest run src/lib/stores/__tests__/editorBitmapTools.test.ts
npx tsc --noEmit
cd src-tauri
cargo fmt --check
cargo check
```

Expected: all pass.

- [ ] **Step 4: Commit**

Skipped for now: user requested execution in the current dirty checkout; no files have been staged or committed.

```powershell
git add scripts/bench_editor_vision_preload.py
git commit -m "test: add editor vision preload benchmark"
```

---

### Task 9: Optional Follow-Up, Split Detect From OCR

**Files:**
- Modify only after Tasks 1-8 are stable:
  - `pipeline/main.py`
  - `pipeline/fast_page_server.py`
  - `src-tauri/src/commands/pipeline.rs`
  - `src/pages/Editor.tsx`

- [ ] **Step 1: Add detector-only runner**

Add a new Python helper:

```python
def _run_detect_boxes_page(project_path: Path, page_idx: int, region: dict | None = None, language_options: dict | None = None):
    from vision_stack.runtime import _get_detector, _profile_to_detection_threshold
    from PIL import Image
    import numpy as np

    project, work_dir, page, orig_img, img_name, original_rel_path = _load_editor_project_page(project_path, page_idx)
    _apply_page_action_language_options(project, language_options)
    image_rgb = np.array(Image.open(orig_img).convert("RGB"))
    detector = _get_detector("quality")
    blocks = detector.detect(image_rgb, conf_threshold=_profile_to_detection_threshold("quality"))
    page["inpaint_blocks"] = [{"bbox": [int(v) for v in block.xyxy], "confidence": float(getattr(block, "confidence", 0.0))} for block in blocks]
    _sync_page_legacy_aliases(page)
    _save_project_json(project_path, project)
    emit("complete", output_path=str(work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")))
```

- [ ] **Step 2: Wire as a separate request only after explicit product approval**

Use request type:

```json
{"type": "editor_detect_boxes_page"}
```

Keep the current `editor_detect_page` as detect+OCR until the UI copy is changed deliberately.

- [ ] **Step 3: Validate with screenshots**

Run the editor, click Detectar, and confirm boxes appear before OCR text is complete only when the detector-only UX has been explicitly enabled.

---

## Self-Review

- Spec coverage: cache key, preload, foreground reuse, fallback, editor-only boundary, and validation are covered by Tasks 1-8.
- Placeholder scan: no task uses an undefined deferred implementation path; the optional detector-only follow-up is explicitly outside the first acceptance boundary.
- Type consistency: request names are `editor_preload_detect_ocr`, `editor_preload_ocr_layers`, `editor_detect_page`, and `editor_ocr_page`; frontend binding is `preloadEditorVisionPage`; Rust command is `preload_editor_vision_page`.
- Risk: Task 4 starts with synchronous preload to keep concurrency safe. A subsequent improvement can add a one-worker background queue, but only after the cache path is validated.
