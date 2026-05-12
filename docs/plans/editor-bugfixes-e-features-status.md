# Editor Bugfixes + Features Status

Updated: 2026-05-07

## Initial state

The worktree was already dirty before this execution. Notable pre-existing modified areas included:

- Frontend editor: `src/lib/stores/editorStore.ts`, `src/lib/tauri.ts`, `src/lib/editorHistory.ts`, `src/lib/e2e/tauriMock.ts`, `src/components/editor/stage/*`, `src/pages/Editor.tsx`
- Rust/Tauri: `src-tauri/src/commands/pipeline.rs`
- Pipeline: `pipeline/main.py`, `pipeline/project_writer.py`, `pipeline/typesetter/renderer.py`, OCR/translation/strip tests and runtime files
- Site/server/worker additions and runtime artifacts under `data/`, `debug/`, `pipeline/scratch/`, `test-results/`

Preserve unrelated existing changes. Do not reset or revert.

## Progress

- [x] Fase 1: project.json robusto
  - `run_page_action_with_optional_mask` now resolves project paths through `project_schema::resolve_project_file`.
  - `project_schema::load_project_value` now retries transient Windows `PermissionDenied`.
  - `save_project_value` retries temp write/remove/rename operations.
  - Editor read-modify-write commands now use `edit_project_value` with a per-project JSON lock.
- [x] Fase 2: persistencia de texto/cor e auto-save
  - `commitEditsPatchOnly` and `commitEdits` now clean only the saved snapshot.
  - Text/style edits made while IPC is pending remain in `pendingEdits`.
  - `runMaskedAction` now uses `flushAutoSave` and stops if the pre-pipeline save fails.
- [x] Fase 3: tipografia unica sem apagar cor/tamanho
  - Added `src/lib/editorTextStylePolicy.ts`.
  - Hydration and new local layers use ComicNeue-Bold.
  - Renderer canonicalizes font/decorative style while preserving color/size/alignment.
  - Font controls are fixed to ComicNeue-Bold.
- [x] Fase 4: lasso persistente e menu
  - Lasso now creates `activeLassoSelection` instead of immediately writing the mask.
  - Added Stage overlay and context menu.
  - Menu actions call `runMaskedActionFromLasso` with the selection bbox.
  - `Aplicar a máscara` keeps the old persistent mask behavior.
- [x] Fase 5: contrato regional real
  - Frontend and Tauri config accept optional `bbox`/`mask_path`.
  - `pipeline.rs` now propagates `--region-bbox` / `--external-mask` to the Python sidecar.
  - `pipeline/main.py` parses regional args for Detect/OCR/Translate/Inpaint.
  - Detect preserves layers outside the selected region and replaces only intersecting detections.
  - OCR and Translate operate only on text layers intersecting the region.
  - Inpaint filters blocks by region and merges only the regional output back over the existing inpaint layer.
- [x] Fase 6: recovery brush
  - Added `recovery` to TS image layer keys, Tauri schema default paths, Rust command registration, Tauri binding, E2E mock, toolbar, shortcut, and editor stroke routing.
  - Added editor preview composition for recovery, using original pixels where the recovery mask is active.
  - Added `pipeline/recovery.py` and bake integration for render preview, retypeset, and page action renders.
  - Mock runner now creates real transparent `mask`, `brush`, and `recovery` PNG layers.

## Checks run

- `cd D:\TraduzAi\src-tauri && cargo test project_schema` - pass, 4 tests.
- `cd D:\TraduzAi\src-tauri && cargo check` - pass.
- `cd D:\TraduzAi && npm run test -- src/lib/__tests__/editorTextStylePolicy.test.ts src/lib/stores/__tests__/editorStoreHistory.test.ts` - pass, 11 tests.
- `cd D:\TraduzAi && npm run check` - pass.
- `cd D:\TraduzAi && $env:PYTHONPATH='D:\TraduzAi\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_renderer.py -q` - pass, 25 tests.
- `cd D:\TraduzAi\src-tauri && cargo check` - pass after regional sidecar propagation.
- `cd D:\TraduzAi && npm run check` - pass after recovery preview composition.
- `cd D:\TraduzAi && $env:PYTHONPATH='D:\TraduzAi\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_page_action_region.py pipeline\tests\test_recovery_bake.py -q` - pass, 5 tests.
- `cd D:\TraduzAi && $env:PYTHONPATH='D:\TraduzAi\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_page_action_region.py pipeline\tests\test_recovery_bake.py pipeline\tests\test_main_emit.py pipeline\tests\test_main_strip_config.py -q` - pass, 63 tests.
- `cd D:\TraduzAi && npm run test` - pass, 16 files / 57 tests.
- `cd D:\TraduzAi\src-tauri && cargo test` - pass, 73 tests.
- `cd D:\TraduzAi && npx playwright test e2e/editor-rebuild.spec.ts --project=chromium` - pass, 12 tests.
- `cd D:\TraduzAi && $env:PYTHONPATH='D:\TraduzAi\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests -q -k "page_action or regional or manual"` - pass, 3 tests.

## Resume point

Implementation complete for the written plan. Remaining risk is manual UX acceptance on a real project:

1. Draw lasso and run Detect/OCR/Translate/Inpaint in a small area; confirm only the selected region changes.
2. Paint with `Restaurar original (R)` and confirm recovery appears in editor preview and exported/rendered output.
3. Repeat on a real project with non-empty masks because automated tests cover the contracts, not visual quality.
