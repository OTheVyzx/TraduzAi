# System Fonts via dafont Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add installed system fonts to the TraduzAI editor font picker using `fcoury/dafont` as the local font scanner/resolver, with preview and render support in both the main app and TraduzAI Studio.

**Architecture:** Add Tauri commands that expose system font metadata from the OS using the `dafont` Rust crate, then extend the shared editor font catalog with a `system` source. The frontend should list/search system fonts, lazily register the chosen font in `document.fonts`, and persist a stable `SystemFont__...` value that Python rendering can resolve through a project/system font manifest.

**Tech Stack:** React 19 + TypeScript, Tauri v2 Rust, `fcoury/dafont` Rust crate from GitHub, Python typesetter font resolution.

---

## Key Decisions

- Use `fcoury/dafont` for local/system font discovery only. It is not a DaFont website downloader.
- Keep Google Fonts and bundled fonts unchanged.
- Add a third picker group: `Sistema`.
- Store selected system fonts with stable values like `SystemFont__Arial__Regular.ttf`, not raw absolute paths inside `estilo.fonte`.
- Save a project-side/system-font manifest mapping stable value to actual font path so the Python renderer can resolve the font later.
- Do not copy every system font into the project. Register/copy only selected fonts if needed.
- Treat system fonts as local user assets. Export portability requires either copying selected font files into project cache or graceful fallback if unavailable.

## Task 1: Add Rust Font Metadata Types and Commands in Main Tauri

**Files:**
- Modify: `src-tauri/Cargo.toml`
- Modify: `src-tauri/src/commands/project.rs`
- Modify: `src-tauri/src/lib.rs`

**Step 1: Add failing Rust command tests**

Add tests near existing project command tests in `src-tauri/src/commands/project.rs`:

```rust
#[test]
fn system_font_cache_filename_is_stable_and_safe() {
    assert_eq!(
        system_font_cache_filename("Arial", "Regular", "ttf").unwrap(),
        "SystemFont__Arial__Regular.ttf"
    );
    assert!(system_font_cache_filename("..", "Regular", "ttf").is_err());
}

#[test]
fn normalizes_system_font_query() {
    assert_eq!(normalize_system_font_query("  Times-New  "), "times new");
}
```

Run:

```bash
cd src-tauri
cargo test system_font --lib
```

Expected: fail because helpers do not exist.

**Step 2: Add `dafont` dependency**

In `src-tauri/Cargo.toml`, add:

```toml
dafont = { git = "https://github.com/fcoury/dafont", rev = "<pin-current-master-commit>" }
```

Pin a commit after checking the repo commit hash. Do not depend on floating `master`.

**Step 3: Add serializable structs**

In `src-tauri/src/commands/project.rs`:

```rust
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SystemFontInfo {
    pub family: String,
    pub full_name: String,
    pub filename: String,
    pub path: String,
    pub weight: String,
    pub style: String,
    pub monospace: bool,
}
```

**Step 4: Implement helpers**

Add:

```rust
fn normalize_system_font_query(value: &str) -> String { /* ascii/space normalization */ }
fn system_font_cache_slug(value: &str) -> Result<String, String> { /* safe slug */ }
fn system_font_cache_filename(family: &str, style: &str, extension: &str) -> Result<String, String> { /* SystemFont__... */ }
```

**Step 5: Implement commands**

Add:

```rust
#[tauri::command]
pub async fn list_system_fonts(query: Option<String>) -> Result<Vec<SystemFontInfo>, String> {
    let cache = dafont::FcFontCache::build();
    let normalized_query = normalize_system_font_query(query.as_deref().unwrap_or(""));
    let mut fonts = Vec::new();
    for (pattern, path) in cache.list() {
        // family from pattern.family, full_name from pattern.name.
        // filter by query if len >= 2.
        // infer extension from path.path.
        // weight/style from pattern.bold / pattern.italic.
        // monospace from pattern.monospace.
        // build stable filename.
    }
    fonts.sort_by(|a, b| a.family.cmp(&b.family).then(a.full_name.cmp(&b.full_name)));
    fonts.dedup_by(|a, b| a.filename == b.filename);
    Ok(fonts)
}

#[tauri::command]
pub async fn resolve_system_font(filename: String) -> Result<Option<SystemFontInfo>, String> {
    let fonts = list_system_fonts(None).await?;
    Ok(fonts.into_iter().find(|font| font.filename == filename))
}
```

**Step 6: Register commands**

In `src-tauri/src/lib.rs`, add to `generate_handler!`:

```rust
commands::project::list_system_fonts,
commands::project::resolve_system_font,
```

**Step 7: Verify**

Run:

```bash
cd src-tauri
cargo test system_font --lib
cargo check
```

Expected: pass.

## Task 2: Add the Same Commands to Studio Tauri

**Files:**
- Modify: `studio/src-tauri/Cargo.toml`
- Modify: `studio/src-tauri/src/main.rs`

**Step 1: Add dependency**

```toml
dafont = { git = "https://github.com/fcoury/dafont", rev = "<same-pinned-commit>" }
```

**Step 2: Mirror minimal structs/helpers/commands**

Add the same `SystemFontInfo`, helper functions, `list_system_fonts`, and `resolve_system_font` to `studio/src-tauri/src/main.rs`.

**Step 3: Register commands**

In the Studio `generate_handler!`, add:

```rust
list_system_fonts,
resolve_system_font,
```

**Step 4: Verify**

Run:

```bash
cd studio/src-tauri
cargo check
```

Expected: pass.

## Task 3: Extend Tauri TypeScript Bindings

**Files:**
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/e2e/tauriMock.ts`

**Step 1: Add types**

In `src/lib/tauri.ts`:

```ts
export interface SystemFontInfo {
  family: string;
  full_name: string;
  filename: string;
  path: string;
  weight: "400" | "700" | string;
  style: "normal" | "italic" | string;
  monospace: boolean;
}
```

**Step 2: Add invoke wrappers**

```ts
export async function listSystemFonts(query?: string): Promise<SystemFontInfo[]> {
  if (isE2E()) return tauriMock.listSystemFonts(query);
  return invoke<SystemFontInfo[]>("list_system_fonts", { query: query ?? null });
}

export async function resolveSystemFont(filename: string): Promise<SystemFontInfo | null> {
  if (isE2E()) return tauriMock.resolveSystemFont(filename);
  return invoke<SystemFontInfo | null>("resolve_system_font", { filename });
}
```

**Step 3: Mock E2E**

In `src/lib/e2e/tauriMock.ts`, add deterministic fonts:

```ts
async listSystemFonts(query?: string) {
  const fonts = [{ family: "Arial", full_name: "Arial Regular", filename: "SystemFont__Arial__Regular.ttf", path: "C:/Windows/Fonts/arial.ttf", weight: "400", style: "normal", monospace: false }];
  return query ? fonts.filter((font) => font.family.toLowerCase().includes(query.toLowerCase())) : fonts;
}
```

**Step 4: Verify**

Run:

```bash
npm run check
```

Expected: pass.

## Task 4: Extend Font Catalog with System Source

**Files:**
- Modify: `src/lib/fontCatalog.ts`
- Modify: `src/lib/__tests__/fontCatalog.test.ts`

**Step 1: Add failing tests**

In `src/lib/__tests__/fontCatalog.test.ts`:

```ts
it("converts system font metadata into editor options", () => {
  expect(systemFontInfoToOption({
    family: "Arial",
    full_name: "Arial Regular",
    filename: "SystemFont__Arial__Regular.ttf",
    path: "C:/Windows/Fonts/arial.ttf",
    weight: "400",
    style: "normal",
    monospace: false,
  })).toMatchObject({
    label: "Arial",
    value: "SystemFont__Arial__Regular.ttf",
    cssFamily: "Arial",
    source: "system",
    groupLabel: "Sistema",
  });
});
```

Run:

```bash
npm test -- src/lib/__tests__/fontCatalog.test.ts
```

Expected: fail.

**Step 2: Extend source union**

Change:

```ts
export type EditorFontCatalogSource = "bundle" | "google";
```

to:

```ts
export type EditorFontCatalogSource = "bundle" | "google" | "system";
```

Add:

```ts
system: "Sistema",
```

to `GROUP_LABELS`.

**Step 3: Add converter**

```ts
export function systemFontInfoToOption(font: SystemFontInfo): EditorFontOption {
  return {
    label: font.family,
    value: font.filename,
    cssFamily: font.family,
    source: "system",
    groupLabel: GROUP_LABELS.system,
    variants: [font.weight],
    variant: font.weight,
    localPath: font.path,
    style: font.style,
  };
}
```

Extend `EditorFontOption` with:

```ts
localPath?: string;
style?: string;
```

**Step 4: Extend selected font fallback**

Any unknown `SystemFont__...` value should display as source `system`, not `google`.

**Step 5: Verify**

Run:

```bash
npm test -- src/lib/__tests__/fontCatalog.test.ts
```

Expected: pass.

## Task 5: Add System Search to Font Picker

**Files:**
- Modify: `src/components/editor/EditorFontPicker.tsx`
- Test: add or extend component test if existing; otherwise validate with a focused unit test around helpers.

**Step 1: Add search states**

Split remote state:

```ts
const [googleOptions, setGoogleOptions] = useState<EditorFontOption[]>([]);
const [systemOptions, setSystemOptions] = useState<EditorFontOption[]>([]);
```

**Step 2: Query system fonts**

When `query.length >= 2`, call both:

```ts
listSystemFonts(query).then((fonts) => fonts.map(systemFontInfoToOption))
searchGoogleFonts(query).then((results) => results.map(googleFontSearchResultToOption))
```

A Google failure should not hide system results. A system failure should not hide Google results.

**Step 3: Group results**

Replace `optionGroupsFromOptions` with grouped result builder:

```ts
function optionGroupsFromSearchResults(system: EditorFontOption[], google: EditorFontOption[]): EditorFontGroup[] {
  return [
    system.length ? { label: "Sistema", source: "system", options: system } : null,
    google.length ? { label: "Google Fonts", source: "google", options: google } : null,
  ].filter(Boolean) as EditorFontGroup[];
}
```

**Step 4: Update menu badges**

Show badge text:

- `Sistema` for system fonts.
- `Google` for Google Fonts.

**Step 5: Verify manually**

Run:

```bash
npm --prefix studio run dev -- --host 127.0.0.1 --port 1430
```

Open the font picker and search `arial`.

Expected:

- System group appears.
- Google group may appear if network works.
- No global “Falha ao buscar” if one provider succeeds.

## Task 6: Load and Preview Selected System Fonts

**Files:**
- Modify: `src/lib/fontCatalog.ts`
- Modify: `src/lib/fonts.ts`

**Step 1: Add failing unit test**

In `src/lib/__tests__/fontCatalog.test.ts`, test that `ensureEditorFontOptionReady` calls the system resolver path through a helper. If direct DOM `FontFace` is awkward, test a pure helper:

```ts
it("detects system font option values", () => {
  expect(isSystemFontValue("SystemFont__Arial__Regular.ttf")).toBe(true);
});
```

**Step 2: Add system readiness branch**

In `ensureEditorFontOptionReady`:

```ts
if (option.source === "system") {
  const { resolveSystemFont } = await import("./tauri");
  const font = await resolveSystemFont(option.value);
  if (!font) throw new Error(`Fonte do sistema nao encontrada: ${option.label}`);
  const { readFile } = await import("@tauri-apps/plugin-fs");
  const bytes = await readFile(font.path);
  await registerImportedFont(option.cssFamily, bytesToArrayBuffer(bytes), font.weight === "700" ? "700" : "400", font.style === "italic" ? "italic" : "normal");
  return { ...option, localPath: font.path };
}
```

**Step 3: Keep browser fallback**

If `readFile(font.path)` fails but CSS can load the family from the system:

```ts
await ensureEditorFontLoaded(option.cssFamily, 32, option.style ?? "normal");
```

Only use this fallback for preview; rendering still needs a path.

**Step 4: Verify**

Run:

```bash
npm test -- src/lib/__tests__/fontCatalog.test.ts
npm run check
```

Expected: pass.

## Task 7: Persist System Font Manifest for Python Rendering

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src/lib/stores/appStore.ts` if schema typing needs a new optional field.
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Decide manifest shape**

Add optional project field:

```json
"font_assets": {
  "system": {
    "SystemFont__Arial__Regular.ttf": {
      "family": "Arial",
      "path": "C:/Windows/Fonts/arial.ttf",
      "weight": "400",
      "style": "normal"
    }
  }
}
```

**Step 2: Update save path when applying a system font**

When `handleFontChange` applies a system font option, update project metadata with the resolved path. Keep text layer style:

```ts
estilo.fonte = "SystemFont__Arial__Regular.ttf"
```

Do not store absolute path directly in each text layer.

**Step 3: Add renderer resolution test**

In `pipeline/tests/test_typesetting_renderer.py`:

```py
def test_find_font_resolves_project_system_font_manifest(tmp_path):
    font_path = tmp_path / "Arial.ttf"
    font_path.write_bytes((Path(__file__).parents[2] / "fonts" / "ComicNeue-Bold.ttf").read_bytes())
    manifest = {"system": {"SystemFont__Arial__Regular.ttf": {"path": str(font_path)}}}
    assert find_font("SystemFont__Arial__Regular.ttf", font_assets=manifest) == str(font_path)
```

Adjust exact function signature based on current `renderer.find_font`.

**Step 4: Implement renderer lookup**

In `pipeline/typesetter/renderer.py`, before scanning `FONT_DIRS`, check project font manifest for `font_name`.

If the path does not exist, fall back to current behavior and log/debug the miss.

**Step 5: Verify**

Run:

```bash
cd pipeline
python -m pytest tests/test_typesetting_renderer.py -k system_font
```

Expected: pass.

## Task 8: Studio Compatibility

**Files:**
- Modify only if needed: `studio/src/backend/editorBackendCompat.ts`
- Modify only if needed: `studio/src/project/studioProject.ts`
- Test: `studio/src/backend/__tests__/editorBackendCompat.test.ts`

**Step 1: Ensure Studio preserves `font_assets`**

The adapter must not strip unknown project fields. Confirm `toAppProject` and save paths preserve:

```ts
font_assets
```

**Step 2: Add test**

In `studio/src/backend/__tests__/editorBackendCompat.test.ts`, verify save/load preserves `font_assets.system`.

**Step 3: Verify**

Run:

```bash
npm --prefix studio test -- src/backend/__tests__/editorBackendCompat.test.ts
npm --prefix studio run build
```

Expected: pass.

## Task 9: UX Polish and Safety

**Files:**
- Modify: `src/components/editor/EditorFontPicker.tsx`

**Step 1: Add source labels**

Menu should show:

- `Sistema`
- `Google`
- `Embutida`

**Step 2: Add non-blocking provider errors**

If system scan fails:

```txt
Fontes do sistema indisponiveis
```

If Google fails but system works, do not show full menu as error.

**Step 3: Add loading row**

When both providers are loading:

```txt
Buscando fontes...
```

**Step 4: Verify visually**

Use Studio:

```bash
N:\TraduzAI\STUUUUUUUUUUDIO.bat
```

Search:

- `arial`
- `comic`
- `noto`

Expected:

- Existing bundled fonts still show.
- Installed system fonts show under `Sistema`.
- Google results still show under `Google Fonts`.

## Task 10: Full Validation

Run:

```bash
npm test -- src/lib/__tests__/fontCatalog.test.ts
npm run check
cd src-tauri && cargo check
cd ..\studio\src-tauri && cargo check
cd .. && npm --prefix studio run build
cd pipeline && python -m pytest tests/test_typesetting_renderer.py -k "font or system_font"
```

Expected:

- All pass.
- Existing Google Fonts behavior remains intact.
- Studio font picker can select a system font and preview it.
- Rendering can resolve the selected system font through the manifest.

## Risks and Notes

- `fcoury/dafont` is small and old; pin the exact commit and be ready to vendor or replace if the crate fails under current Rust.
- System fonts have licensing constraints. Because they are installed locally by the user, the app should not assume export redistribution rights.
- Absolute system font paths make projects less portable. The manifest should be treated as local resolution metadata, not a portable asset contract.
- For portable exports later, add an explicit “copiar fonte para o projeto” option.
