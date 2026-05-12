# Commit and Push Local Changes Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Commit the current source/documentation changes safely and push all local branch work to `origin/feat/editor-brush-mask-typesetting`.

**Architecture:** Treat the current checkout as dirty and mixed: source changes, docs, app/site/server/worker code, and many generated chapter/debug artifacts are present together. Stage by explicit path groups, verify the staged diff, commit in coherent batches, then push the branch that is currently `13` commits ahead and `0` behind after `git fetch --prune origin`.

**Tech Stack:** Git on Windows PowerShell, TraduzAi desktop app, Python pipeline, Rust/Tauri, React/Vite, site workspace, server/worker package, pytest, Vitest/TypeScript, Cargo.

---

## Current Snapshot

Checked on 2026-05-11 in `N:\TraduzAI\TraduzAi`.

- Branch: `feat/editor-brush-mask-typesetting`
- Remote: `origin https://github.com/OTheVyzx/TraduzAi.git`
- Divergence after fetch: `0 behind / 13 ahead`
- Last local HEAD: `f60e80d fix(editor): 8 correcoes pos-feedback do usuario`
- Modified tracked files: 88 files, about `28012 insertions` and `2374 deletions`
- Untracked source-like files outside generated folders: 165 files
- Large generated/untracked groups:
  - `debug/`: 16921 files
  - `data/`: 10762 files
  - `pipeline/scratch/`: 5569 files
  - chapter/output folders such as `AAAAAAA/`, `BBBBBBBBBBBBB/`, `CCCCCCCCCCCC/`, `DDDDDDDDDDDDDDDDDDDD/`, `eeee.../`

Do not use `git add .`.

---

## Never Stage

These are runtime/generated/local artifacts unless the user explicitly asks to version a fixture:

- `debug/`
- `data/logs/`
- `data/works/`
- `data/runtime/`
- `data/models/`
- `pipeline/scratch/`
- `AAAAAAA/`
- `BBBBBBBBBBBBB/`
- `CCCCCCCCCCCC/`
- `DDDDDDDDDDDDDDDDDDDD/`
- `eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee/`
- `test-results/`
- `SITEEEEE.bat`
- generated translated chapter images/zips/layers

Also do not stage these tracked runtime changes by default:

- `data/credits.json`: only `week_start` changed.
- `backups/traduzai_backup_20260420_234656.zip`: deleted binary backup; leave it out unless cleanup of tracked backup files is explicitly desired.

---

## Phase 1: Preflight

Run:

```powershell
git fetch --prune origin
git status --short --branch --untracked-files=no
git rev-list --left-right --count origin/feat/editor-brush-mask-typesetting...HEAD
```

Expected:

```text
0    13
```

If the first number is not `0`, stop and inspect remote changes before pushing.

---

## Phase 2: Stage Repo Hygiene

Stage only hygiene/config files that should be shared:

```powershell
git add -- .gitignore .env.example requirements-server.txt requirements-worker.txt
git diff --cached --check
git diff --cached --stat
git commit -m "chore(repo): add runtime hygiene and service requirements"
```

Before committing, inspect `.env.example` and ensure it contains no secrets.

Do not include `data/credits.json` or deleted `backups/*.zip` in this commit.

---

## Phase 3: Stage Pipeline Quality Changes

Stage tracked pipeline changes plus new source/test/tool files, excluding `pipeline/scratch/`.

```powershell
git add -u -- pipeline
git add -- `
  pipeline/fast_page_server.py `
  pipeline/ocr/macro_ocr.py `
  pipeline/recovery.py `
  pipeline/runtime_profiles.py `
  pipeline/scripts/diff_inpaint.py `
  pipeline/strip/scheduler.py `
  pipeline/strip/smart_skip.py `
  pipeline/tools `
  pipeline/typesetter/style_policy.py `
  pipeline/tests
```

Verify no scratch files are staged:

```powershell
git diff --cached --name-only | Select-String "^(pipeline/scratch|debug|data|AAAAAAA|BBBBBBBBBBBBB|CCCCCCCCCCCC|DDDDDDDDDDDDDDDDDDDD)/"
```

Expected: no output.

Run focused tests:

```powershell
.\pipeline\venv\Scripts\python.exe -m pytest `
  pipeline\tests\test_ocr_normalizer.py `
  pipeline\tests\test_ocr_postprocess.py `
  pipeline\tests\test_strip_run.py `
  pipeline\tests\test_strip_process_bands.py `
  pipeline\tests\test_vision_stack_runtime.py `
  pipeline\tests\test_typesetting_renderer.py `
  pipeline\tests\test_runtime_profiles.py `
  -q
```

Then:

```powershell
git diff --cached --check
git commit -m "feat(pipeline): harden fast-page OCR inpaint and typesetting"
```

---

## Phase 4: Stage Worker and Server Changes

Stage the SaaS/local server and worker code:

```powershell
git add -- server worker
git diff --cached --name-only | Select-String "^(debug|data|pipeline/scratch|AAAAAAA|BBBBBBBBBBBBB|CCCCCCCCCCCC|DDDDDDDDDDDDDDDDDDDD)/"
```

Expected: no output.

Run:

```powershell
.\pipeline\venv\Scripts\python.exe -m pytest server\tests worker\tests -q
```

Then:

```powershell
git diff --cached --check
git commit -m "feat(worker): add local server worker and fast-page orchestration"
```

---

## Phase 5: Stage Desktop Editor and Tauri Changes

Stage tracked desktop/editor/Tauri files plus new editor helpers:

```powershell
git add -u -- src src-tauri e2e package.json package-lock.json
git add -- `
  src/components/editor/LassoContextMenu.tsx `
  src/components/editor/stage/EditorRotationHotspots.tsx `
  src/components/editor/stage/LassoSelectionOverlay.tsx `
  src/components/editor/stage/bitmapStrokePreview.ts `
  src/components/editor/stage/recoveryComposite.ts `
  src/components/editor/stage/textFit.ts `
  src/lib/editorBackend.ts `
  src/lib/editorBackends/tauriEditorBackend.ts `
  src/lib/editorTextStylePolicy.ts `
  src/lib/imageSource.ts `
  src/lib/konvaExportRenderer.ts `
  src/lib/lassoSelection.ts `
  src/lib/__tests__ `
  src/lib/stores/__tests__
```

Run:

```powershell
npm run check
npm run test -- --run
cargo check --manifest-path src-tauri\Cargo.toml
```

Then:

```powershell
git diff --cached --check
git commit -m "feat(editor): align preview export and bitmap text editing"
```

---

## Phase 6: Stage Vision Worker

Stage:

```powershell
git add -- vision-worker\Cargo.lock vision-worker\src\main.rs
```

Run:

```powershell
cargo test --manifest-path vision-worker\Cargo.toml
cargo build --manifest-path vision-worker\Cargo.toml
```

Then:

```powershell
git diff --cached --check
git commit -m "feat(vision): add persistent Koharu worker routing"
```

---

## Phase 7: Stage Site Workspace

Stage the site code and fonts/assets:

```powershell
git add -u -- site
git add -- `
  site/public/assets/traduzai-logo.svg `
  site/public/fonts `
  site/src/editor `
  site/src/projectApi.ts `
  site/src/projectConfig.ts `
  site/src/projectSetupApi.ts `
  site/src/vite-env.d.ts
```

Run:

```powershell
cd site
npm run build
cd ..
```

Then:

```powershell
git diff --cached --check
git commit -m "feat(site): add web project and editor flow"
```

---

## Phase 8: Stage Documentation and Handoff

Stage plans and context:

```powershell
git add -- context.md docs\plans
```

Review that docs do not include private manga pages or local-only secrets.

Then:

```powershell
git diff --cached --check
git commit -m "docs: update TraduzAi plans and handoff context"
```

---

## Phase 9: Final Gate Before Push

Run:

```powershell
git status --short --branch --untracked-files=no
git diff --check
git log --oneline origin/feat/editor-brush-mask-typesetting..HEAD
```

Expected:

- no staged changes;
- remaining modified files should only be intentionally excluded runtime artifacts such as `data/credits.json` and `backups/*.zip`, or nothing;
- local commits are listed above the remote.

If generated files still show as untracked, do not delete them in this plan. Either leave them local or add safe ignore rules in a separate reviewed cleanup commit.

---

## Phase 10: Push

Push the current branch:

```powershell
git push origin feat/editor-brush-mask-typesetting
```

Verify:

```powershell
git fetch --prune origin
git rev-list --left-right --count origin/feat/editor-brush-mask-typesetting...HEAD
git status --short --branch --untracked-files=no
```

Expected:

```text
0    0
```

The working tree may still show ignored/untracked generated folders locally. That is acceptable if no code/documentation changes remain unstaged.
