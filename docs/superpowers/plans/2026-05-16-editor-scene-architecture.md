# Editor Scene Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the TraduzAi editor around a cleaner scene and operation architecture while preserving every existing editor tool and interaction.

**Architecture:** Keep Konva/react-konva as the primary interaction surface. Add a Koharu-inspired read model (`EditorScene`) and small operation/helper modules so panels, Stage, and store code share normalized layer data and typed mutation helpers instead of duplicating page-shape logic. Preserve current tool modes (`select`, `block`, `brush`, `repairBrush`, `reinpaintBrush`, `eraser`, `mask`), existing store methods, undo/redo, lasso, snap guides, rotation, floating text editor, and bitmap persistence.

**Tech Stack:** React 19, TypeScript, Vite, Tauri v2, Zustand, Konva/react-konva, Vitest, Playwright.

---

## Non-Negotiable Scope Guards

- Do not replace Konva with DOM/canvas-only rendering.
- Do not remove any current tool mode from `EditorToolMode`.
- Do not bypass `editorStore.ts` for persistence, history, autosave, or pipeline actions.
- Do not create a parallel editor path or legacy mode.
- UI text added in this plan must be PT-BR and dark-only.
- `project.json` compatibility stays on current `image_layers`, `text_layers`, and `textos` aliases.

## File Structure

- Create `src/lib/editorScene.ts`
  - Pure read model for the active page.
  - Merges `currentPage`, `pendingEdits`, `selectedLayerId`, and image-layer metadata.
  - Exposes sorted text layers, role-based image layers, selected text, visible text count, and search helpers.
- Create `src/lib/__tests__/editorScene.test.ts`
  - Unit tests for pending edit merge, text ordering, selected text lookup, image layer fallback, and search.
- Create `src/lib/editorOps.ts`
  - Pure typed helpers for command construction and layer update metadata.
  - Keeps `editorStore.ts` as executor, but moves repeated command-building rules into one module.
- Create `src/lib/__tests__/editorOps.test.ts`
  - Unit tests for command IDs, visibility commands, lock commands, reorder commands, and no-op guards.
- Create `src/lib/editorStroke.ts`
  - Pure helpers extracted from `useEditorStageController.ts`: point conversion, stroke decimation, dirty bbox calculation, lasso clipping, and bitmap target selection.
- Create `src/lib/__tests__/editorStroke.test.ts`
  - Unit tests for coordinate conversion, dirty bbox padding, clipped bbox, and target selection.
- Create `src/components/editor/stage/useEditorBitmapDrawing.ts`
  - Hook owning bitmap paint/recovery/healing queues after pure helpers are in place.
  - Keeps behavior identical to current `brush`, `repairBrush`, `reinpaintBrush`, and `eraser` flows.
- Modify `src/components/editor/stage/useEditorStageController.ts`
  - Consume `editorScene.ts` for `layers`.
  - Delegate bitmap stroke details to `useEditorBitmapDrawing`.
  - Keep Stage pointer handlers and lasso flow intact.
- Modify `src/components/editor/stage/EditorStage.tsx`
  - Consume the normalized `layers` already returned by controller.
  - No visual rewrite.
- Modify `src/components/editor/LayersPanel.tsx`
  - Use `EditorScene`.
  - Keep text list, add image-layer section using existing `image_layers` controls.
- Modify `src/components/editor/LayerItem.tsx`
  - Keep current text-layer actions.
  - Accept normalized text item shape from `EditorScene`.
- Modify `src/components/editor/PropertyEditor.tsx`
  - Use `EditorScene.selectedTextLayer` instead of repeating selected-layer lookup and pending merge logic.
- Modify `src/pages/Editor.tsx`
  - Only if needed to host the enhanced panel state. Keep `ToolSidebar`, `TypesettingBar`, and `EditorStage`.
- Test with existing focused editor suites and one Playwright smoke path.

---

### Task 1: Add `EditorScene` Read Model

**Files:**
- Create: `src/lib/editorScene.ts`
- Create: `src/lib/__tests__/editorScene.test.ts`

- [ ] **Step 1: Write failing tests for normalized text and image layers**

Create `src/lib/__tests__/editorScene.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { buildEditorScene, searchTextLayers } from "../editorScene";
import type { PageData, TextEntry } from "../stores/appStore";

const style = {
  fonte: "Newrotic.ttf",
  tamanho: 32,
  cor: "#000000",
  cor_gradiente: [],
  contorno: "#ffffff",
  contorno_px: 0,
  glow: false,
  glow_cor: "#ffffff",
  glow_px: 0,
  sombra: false,
  sombra_cor: "#000000",
  sombra_offset: [0, 0] as [number, number],
  bold: false,
  italico: false,
  rotacao: 0,
  alinhamento: "center" as const,
};

function text(id: string, order: number, translated: string): TextEntry {
  return {
    id,
    bbox: [10 * order, 10, 100, 60],
    tipo: "fala",
    original: `original ${id}`,
    traduzido: translated,
    translated,
    confianca_ocr: 0.9,
    estilo: style,
    order,
    visible: true,
    locked: false,
  };
}

function page(): PageData {
  return {
    numero: 1,
    arquivo_original: "page-001.png",
    arquivo_traduzido: "rendered.png",
    image_layers: {
      base: { key: "base", path: "base.png", visible: true, locked: true, order: 0 },
      mask: { key: "mask", path: "mask.png", visible: false, locked: false, opacity: 0.5, order: 1, technical: true },
      inpaint: { key: "inpaint", path: "inpaint.png", visible: true, locked: false, order: 2 },
      rendered: { key: "rendered", path: "rendered.png", visible: false, locked: false, order: 5 },
    },
    text_layers: [text("b", 2, "Segundo"), text("a", 1, "Primeiro")],
    textos: [],
  };
}

describe("buildEditorScene", () => {
  it("sorts text layers and merges pending edits without mutating the page", () => {
    const source = page();
    const scene = buildEditorScene({
      page: source,
      pendingEdits: { a: { traduzido: "Alterado", visible: false } },
      selectedLayerId: "a",
    });

    expect(scene.textLayers.map((layer) => layer.id)).toEqual(["a", "b"]);
    expect(scene.textLayers[0].displayText).toBe("Alterado");
    expect(scene.textLayers[0].visible).toBe(false);
    expect(scene.selectedTextLayer?.id).toBe("a");
    expect(source.text_layers[1].traduzido).toBe("Primeiro");
  });

  it("returns stable image layer roles with defaults", () => {
    const scene = buildEditorScene({ page: page(), pendingEdits: {}, selectedLayerId: null });

    expect(scene.imageLayers.map((layer) => layer.key)).toEqual([
      "base",
      "mask",
      "inpaint",
      "brush",
      "recovery",
      "rendered",
    ]);
    expect(scene.imageLayers.find((layer) => layer.key === "brush")?.hasContent).toBe(false);
    expect(scene.imageLayers.find((layer) => layer.key === "base")?.locked).toBe(true);
  });
});

describe("searchTextLayers", () => {
  it("matches translated text, original text, and type", () => {
    const scene = buildEditorScene({ page: page(), pendingEdits: {}, selectedLayerId: null });

    expect(searchTextLayers(scene.textLayers, "primeiro").map((layer) => layer.id)).toEqual(["a"]);
    expect(searchTextLayers(scene.textLayers, "original b").map((layer) => layer.id)).toEqual(["b"]);
    expect(searchTextLayers(scene.textLayers, "fala").map((layer) => layer.id)).toEqual(["a", "b"]);
  });
});
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
npx vitest run src/lib/__tests__/editorScene.test.ts
```

Expected: FAIL because `src/lib/editorScene.ts` does not exist.

- [ ] **Step 3: Implement the pure read model**

Create `src/lib/editorScene.ts`:

```ts
import type { ImageLayer, ImageLayerKey, PageData, TextEntry } from "./stores/appStore";

const IMAGE_LAYER_ORDER: ImageLayerKey[] = ["base", "mask", "inpaint", "brush", "recovery", "rendered"];

export type EditorSceneTextLayer = TextEntry & {
  displayText: string;
  displayOriginal: string;
  effectiveBbox: TextEntry["bbox"];
  visible: boolean;
  locked: boolean;
  order: number;
};

export type EditorSceneImageLayer = ImageLayer & {
  hasContent: boolean;
  visible: boolean;
  locked: boolean;
  opacity: number;
  order: number;
};

export type EditorScene = {
  page: PageData | null;
  textLayers: EditorSceneTextLayer[];
  imageLayers: EditorSceneImageLayer[];
  selectedTextLayer: EditorSceneTextLayer | null;
  selectedLayerId: string | null;
  textCount: number;
  visibleTextCount: number;
};

export type BuildEditorSceneInput = {
  page: PageData | null | undefined;
  pendingEdits: Record<string, Partial<TextEntry>>;
  selectedLayerId: string | null | undefined;
};

function sortTextLayers(layers: TextEntry[]): TextEntry[] {
  return [...layers].sort((left, right) => {
    const leftOrder = left.order ?? Number.MAX_SAFE_INTEGER;
    const rightOrder = right.order ?? Number.MAX_SAFE_INTEGER;
    return leftOrder - rightOrder || left.id.localeCompare(right.id);
  });
}

function mergeLayer(layer: TextEntry, patch: Partial<TextEntry> | undefined): EditorSceneTextLayer {
  const merged: TextEntry = {
    ...layer,
    ...patch,
    estilo: patch?.estilo ? { ...layer.estilo, ...patch.estilo } : layer.estilo,
  };
  return {
    ...merged,
    displayText: merged.traduzido ?? merged.translated ?? "",
    displayOriginal: merged.original ?? "",
    effectiveBbox: merged.layout_bbox ?? merged.bbox,
    visible: merged.visible !== false,
    locked: merged.locked === true,
    order: merged.order ?? 0,
  };
}

function normalizeImageLayer(key: ImageLayerKey, page: PageData): EditorSceneImageLayer {
  const existing = page.image_layers?.[key];
  const fallbackPath =
    key === "base"
      ? page.arquivo_original
      : key === "rendered"
        ? page.image_layers?.rendered?.path ?? page.arquivo_traduzido ?? null
        : null;
  const path = existing?.path ?? fallbackPath ?? null;
  return {
    key,
    path,
    visible: existing?.visible ?? (key === "base" || key === "rendered"),
    locked: existing?.locked ?? (key === "base" || key === "rendered"),
    opacity: existing?.opacity ?? 1,
    order: existing?.order ?? IMAGE_LAYER_ORDER.indexOf(key),
    technical: existing?.technical ?? key === "mask",
    hasContent: Boolean(path),
  };
}

export function buildEditorScene({ page, pendingEdits, selectedLayerId }: BuildEditorSceneInput): EditorScene {
  if (!page) {
    return {
      page: null,
      textLayers: [],
      imageLayers: [],
      selectedTextLayer: null,
      selectedLayerId: selectedLayerId ?? null,
      textCount: 0,
      visibleTextCount: 0,
    };
  }
  const textLayers = sortTextLayers(page.text_layers ?? []).map((layer) => mergeLayer(layer, pendingEdits[layer.id]));
  const imageLayers = IMAGE_LAYER_ORDER.map((key) => normalizeImageLayer(key, page)).sort(
    (left, right) => left.order - right.order,
  );
  const selected = selectedLayerId ? textLayers.find((layer) => layer.id === selectedLayerId) ?? null : null;
  return {
    page,
    textLayers,
    imageLayers,
    selectedTextLayer: selected,
    selectedLayerId: selectedLayerId ?? null,
    textCount: textLayers.length,
    visibleTextCount: textLayers.filter((layer) => layer.visible).length,
  };
}

export function searchTextLayers(layers: EditorSceneTextLayer[], query: string): EditorSceneTextLayer[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return layers;
  return layers.filter((entry) => {
    const haystack = `${entry.displayText} ${entry.displayOriginal} ${entry.tipo}`.toLowerCase();
    return haystack.includes(normalized);
  });
}
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
npx vitest run src/lib/__tests__/editorScene.test.ts
```

Expected: PASS.

- [ ] **Step 5: Run typecheck**

Run:

```bash
npm run check
```

Expected: PASS.

- [ ] **Step 6: Commit checkpoint**

```bash
git add src/lib/editorScene.ts src/lib/__tests__/editorScene.test.ts
git commit -m "refactor: add editor scene read model"
```

---

### Task 2: Refactor Panels To Use `EditorScene`

**Files:**
- Modify: `src/components/editor/LayersPanel.tsx`
- Modify: `src/components/editor/LayerItem.tsx`
- Modify: `src/components/editor/PropertyEditor.tsx`
- Test: `src/lib/__tests__/editorScene.test.ts`

- [ ] **Step 1: Add a selected-layer regression test**

Extend `src/lib/__tests__/editorScene.test.ts`:

```ts
it("returns null selectedTextLayer when selection points to a missing layer", () => {
  const scene = buildEditorScene({
    page: page(),
    pendingEdits: {},
    selectedLayerId: "missing",
  });

  expect(scene.selectedTextLayer).toBeNull();
  expect(scene.selectedLayerId).toBe("missing");
});
```

- [ ] **Step 2: Run focused test**

Run:

```bash
npx vitest run src/lib/__tests__/editorScene.test.ts
```

Expected: PASS after Task 1, then keep passing after panel refactor.

- [ ] **Step 3: Update `LayersPanel.tsx` to build the scene once**

Replace the direct text layer derivation at the top of `LayersPanel` with:

```ts
import { buildEditorScene, searchTextLayers } from "../../lib/editorScene";

// inside LayersPanel()
const scene = useMemo(
  () =>
    buildEditorScene({
      page: currentPage,
      pendingEdits,
      selectedLayerId,
    }),
  [currentPage, pendingEdits, selectedLayerId],
);

const filteredTextLayers = useMemo(
  () => searchTextLayers(scene.textLayers, query),
  [query, scene.textLayers],
);
```

Keep existing delete/save buttons and existing `LayerItem` rendering. Add a compact image-layer section above the text list:

```tsx
<div className="border-b border-border px-3 py-2">
  <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-muted">
    Camadas da pagina
  </div>
  <div className="space-y-1">
    {scene.imageLayers.map((layer) => (
      <button
        key={layer.key}
        type="button"
        disabled={!layer.hasContent}
        className="flex w-full items-center justify-between rounded-md px-2 py-1 text-[11px] text-text-secondary transition-smooth hover:bg-white/[0.04] disabled:opacity-35"
        title={layer.hasContent ? `Camada ${layer.key}` : `Camada ${layer.key} sem conteudo`}
      >
        <span className="font-mono">{layer.key}</span>
        <span>{layer.visible ? "visivel" : "oculta"}</span>
      </button>
    ))}
  </div>
</div>
```

- [ ] **Step 4: Update `LayerItem.tsx` typing**

Change the `entry` prop type from `TextEntry` to `EditorSceneTextLayer`:

```ts
import type { EditorSceneTextLayer } from "../../lib/editorScene";

type LayerItemProps = {
  entry: EditorSceneTextLayer;
  index: number;
};
```

Use `entry.displayText`, `entry.displayOriginal`, `entry.effectiveBbox`, `entry.visible`, and `entry.locked` where the component currently recomputes those values.

- [ ] **Step 5: Update `PropertyEditor.tsx` selected layer lookup**

Build the scene near the existing store reads:

```ts
import { buildEditorScene } from "../../lib/editorScene";

const scene = buildEditorScene({ page: currentPage, pendingEdits, selectedLayerId });
const entry = scene.selectedTextLayer;
```

Remove the local duplicate selected-layer lookup:

```ts
const selectedLayer = currentPage?.text_layers.find((t) => t.id === selectedLayerId);
const entry = selectedLayer;
```

Keep all existing style controls and calls to `updatePendingEdit` / `updatePendingEstilo`.

- [ ] **Step 6: Validate panels**

Run:

```bash
npm run check
npx vitest run src/lib/__tests__/editorScene.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit checkpoint**

```bash
git add src/components/editor/LayersPanel.tsx src/components/editor/LayerItem.tsx src/components/editor/PropertyEditor.tsx src/lib/__tests__/editorScene.test.ts
git commit -m "refactor: drive editor panels from scene model"
```

---

### Task 3: Add Typed Editor Operation Helpers

**Files:**
- Create: `src/lib/editorOps.ts`
- Create: `src/lib/__tests__/editorOps.test.ts`
- Modify: `src/lib/stores/editorStore.ts`

- [ ] **Step 1: Write failing tests for command helpers**

Create `src/lib/__tests__/editorOps.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  buildToggleLockCommand,
  buildToggleVisibilityCommand,
  buildReorderLayersCommand,
  nextEditorCommandId,
} from "../editorOps";

describe("editorOps", () => {
  it("creates stable command ids with a readable prefix", () => {
    expect(nextEditorCommandId("visibility")).toMatch(/^visibility-/);
  });

  it("builds visibility commands with before and after values", () => {
    const command = buildToggleVisibilityCommand({
      pageKey: "project:0:path",
      layerId: "layer-a",
      before: true,
      after: false,
    });

    expect(command.type).toBe("toggle-visibility");
    expect(command.before).toBe(true);
    expect(command.after).toBe(false);
  });

  it("builds lock commands with before and after values", () => {
    const command = buildToggleLockCommand({
      pageKey: "project:0:path",
      layerId: "layer-a",
      before: false,
      after: true,
    });

    expect(command.type).toBe("toggle-lock");
    expect(command.before).toBe(false);
    expect(command.after).toBe(true);
  });

  it("builds reorder commands with defensive copies", () => {
    const before = ["a", "b"];
    const after = ["b", "a"];
    const command = buildReorderLayersCommand({ pageKey: "p", before, after });

    before.reverse();
    after.reverse();

    expect(command.before).toEqual(["a", "b"]);
    expect(command.after).toEqual(["b", "a"]);
  });
});
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
npx vitest run src/lib/__tests__/editorOps.test.ts
```

Expected: FAIL because `src/lib/editorOps.ts` does not exist.

- [ ] **Step 3: Implement helpers**

Create `src/lib/editorOps.ts`:

```ts
import type { EditorCommand } from "./editorHistory";

export function nextEditorCommandId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

function meta(prefix: string, pageKey: string) {
  return {
    commandId: nextEditorCommandId(prefix),
    pageKey,
    createdAt: Date.now(),
  };
}

export function buildToggleVisibilityCommand(args: {
  pageKey: string;
  layerId: string;
  before: boolean;
  after: boolean;
}): EditorCommand {
  return {
    ...meta("visibility", args.pageKey),
    type: "toggle-visibility",
    layerId: args.layerId,
    before: args.before,
    after: args.after,
  };
}

export function buildToggleLockCommand(args: {
  pageKey: string;
  layerId: string;
  before: boolean;
  after: boolean;
}): EditorCommand {
  return {
    ...meta("lock", args.pageKey),
    type: "toggle-lock",
    layerId: args.layerId,
    before: args.before,
    after: args.after,
  };
}

export function buildReorderLayersCommand(args: {
  pageKey: string;
  before: string[];
  after: string[];
}): EditorCommand {
  return {
    ...meta("reorder", args.pageKey),
    type: "reorder-layers",
    before: [...args.before],
    after: [...args.after],
  };
}
```

- [ ] **Step 4: Refactor store command construction**

In `src/lib/stores/editorStore.ts`, import:

```ts
import {
  buildReorderLayersCommand,
  buildToggleLockCommand,
  buildToggleVisibilityCommand,
} from "../editorOps";
```

Use the helpers in the existing methods:

- `toggleTextLayerVisibility`
- `toggleTextLayerLock`
- `reorderWorkingLayers` call sites that record a command

Keep method names and signatures unchanged.

- [ ] **Step 5: Validate history behavior**

Run:

```bash
npx vitest run src/lib/__tests__/editorHistory.test.ts src/lib/stores/__tests__/editorStoreHistory.test.ts src/lib/__tests__/editorOps.test.ts
npm run check
```

Expected: PASS.

- [ ] **Step 6: Commit checkpoint**

```bash
git add src/lib/editorOps.ts src/lib/__tests__/editorOps.test.ts src/lib/stores/editorStore.ts
git commit -m "refactor: centralize editor operation commands"
```

---

### Task 4: Extract Pure Stroke Helpers Before Moving Hook Logic

**Files:**
- Create: `src/lib/editorStroke.ts`
- Create: `src/lib/__tests__/editorStroke.test.ts`
- Modify: `src/components/editor/stage/useEditorStageController.ts`

- [ ] **Step 1: Write failing tests for stroke geometry**

Create `src/lib/__tests__/editorStroke.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  bitmapTargetForEditorTool,
  pointFromStageClientRect,
  shouldAppendStrokePoint,
  strokeDirtyBbox,
} from "../editorStroke";

describe("editorStroke", () => {
  it("converts client coordinates to image coordinates", () => {
    expect(
      pointFromStageClientRect({
        clientX: 150,
        clientY: 90,
        rect: { left: 50, top: 40, width: 200, height: 100 },
        imageWidth: 1000,
        imageHeight: 500,
      }),
    ).toEqual({ x: 500, y: 250 });
  });

  it("expands stroke bbox by brush radius and clamps to page", () => {
    expect(
      strokeDirtyBbox({
        stroke: [
          [10, 20],
          [30, 40],
        ],
        brushSize: 20,
        width: 100,
        height: 100,
      }),
    ).toEqual([0, 8, 42, 52]);
  });

  it("filters duplicate points", () => {
    expect(shouldAppendStrokePoint([10, 10], { x: 10, y: 10 })).toBe(false);
    expect(shouldAppendStrokePoint([10, 10], { x: 11, y: 10 })).toBe(true);
  });

  it("maps tools to mutable bitmap targets", () => {
    expect(bitmapTargetForEditorTool("brush", null)).toBe("brush");
    expect(bitmapTargetForEditorTool("repairBrush", null)).toBe("recovery");
    expect(bitmapTargetForEditorTool("reinpaintBrush", null)).toBe("reinpaint");
    expect(bitmapTargetForEditorTool("eraser", "mask")).toBe("mask");
  });
});
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
npx vitest run src/lib/__tests__/editorStroke.test.ts
```

Expected: FAIL because `src/lib/editorStroke.ts` does not exist.

- [ ] **Step 3: Implement pure stroke helpers**

Create `src/lib/editorStroke.ts`:

```ts
import type { EditorToolMode } from "./stores/editorStore";

export type ImagePoint = { x: number; y: number };
export type Bbox = [number, number, number, number];
export type BitmapTarget = "brush" | "mask" | "recovery" | "reinpaint";

export function pointFromStageClientRect(args: {
  clientX: number;
  clientY: number;
  rect: Pick<DOMRect, "left" | "top" | "width" | "height">;
  imageWidth: number;
  imageHeight: number;
}): ImagePoint | null {
  if (args.rect.width <= 0 || args.rect.height <= 0 || args.imageWidth <= 0 || args.imageHeight <= 0) return null;
  const x = ((args.clientX - args.rect.left) / args.rect.width) * args.imageWidth;
  const y = ((args.clientY - args.rect.top) / args.rect.height) * args.imageHeight;
  return {
    x: Math.max(0, Math.min(args.imageWidth, Math.round(x))),
    y: Math.max(0, Math.min(args.imageHeight, Math.round(y))),
  };
}

export function shouldAppendStrokePoint(last: [number, number] | undefined, point: ImagePoint): boolean {
  return !last || last[0] !== point.x || last[1] !== point.y;
}

export function strokeDirtyBbox(args: {
  stroke: [number, number][];
  brushSize: number;
  width: number;
  height: number;
}): Bbox | null {
  if (args.stroke.length === 0 || args.width <= 0 || args.height <= 0) return null;
  const pad = Math.max(1, Math.ceil(args.brushSize / 2) + 2);
  const xs = args.stroke.map(([x]) => x);
  const ys = args.stroke.map(([, y]) => y);
  return [
    Math.max(0, Math.floor(Math.min(...xs) - pad)),
    Math.max(0, Math.floor(Math.min(...ys) - pad)),
    Math.min(args.width, Math.ceil(Math.max(...xs) + pad)),
    Math.min(args.height, Math.ceil(Math.max(...ys) + pad)),
  ];
}

export function bitmapTargetForEditorTool(
  toolMode: EditorToolMode,
  eraserTarget: "brush" | "mask" | "recovery" | null,
): BitmapTarget | null {
  if (toolMode === "brush") return "brush";
  if (toolMode === "repairBrush") return "recovery";
  if (toolMode === "reinpaintBrush") return "reinpaint";
  if (toolMode === "mask") return "mask";
  if (toolMode === "eraser") return eraserTarget ?? "brush";
  return null;
}
```

- [ ] **Step 4: Use helpers in `useEditorStageController.ts`**

Replace local point conversion and dirty-bbox construction with imports:

```ts
import {
  bitmapTargetForEditorTool,
  pointFromStageClientRect,
  shouldAppendStrokePoint,
  strokeDirtyBbox,
} from "../../../lib/editorStroke";
```

Keep the existing `pointFromStageEvent` wrapper, but delegate the math:

```ts
const pointFromStageEvent = (event: Konva.KonvaEventObject<MouseEvent>) => {
  const stage = event.target.getStage();
  const rect = stage?.container().getBoundingClientRect();
  if (!rect) return null;
  return pointFromStageClientRect({
    clientX: event.evt.clientX,
    clientY: event.evt.clientY,
    rect,
    imageWidth: baseImage.size.width,
    imageHeight: baseImage.size.height,
  });
};
```

Use `shouldAppendStrokePoint` in paint stroke accumulation:

```ts
setPaintStroke((points) => {
  const last = points[points.length - 1];
  if (!shouldAppendStrokePoint(last, point)) return points;
  return [...points, [point.x, point.y]];
});
```

- [ ] **Step 5: Validate no behavior drift**

Run:

```bash
npx vitest run src/lib/__tests__/editorStroke.test.ts src/lib/stores/__tests__/editorBitmapTools.test.ts src/lib/editorStage/__tests__/bitmapStrokePreview.test.ts src/lib/__tests__/lassoSelection.test.ts
npm run check
```

Expected: PASS.

- [ ] **Step 6: Commit checkpoint**

```bash
git add src/lib/editorStroke.ts src/lib/__tests__/editorStroke.test.ts src/components/editor/stage/useEditorStageController.ts
git commit -m "refactor: extract editor stroke geometry helpers"
```

---

### Task 5: Move Bitmap Drawing Queues Into A Focused Hook

**Files:**
- Create: `src/components/editor/stage/useEditorBitmapDrawing.ts`
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Test: `src/lib/__tests__/editorStroke.test.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`
- Test: `src/lib/editorStage/__tests__/bitmapStrokePreview.test.ts`

- [ ] **Step 1: Add a focused target-selection test**

Extend `src/lib/__tests__/editorStroke.test.ts`:

```ts
it("keeps eraser target explicit when the user selected a target", () => {
  expect(bitmapTargetForEditorTool("eraser", "mask")).toBe("mask");
  expect(bitmapTargetForEditorTool("eraser", "recovery")).toBe("recovery");
});
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
npx vitest run src/lib/__tests__/editorStroke.test.ts
```

Expected: PASS.

- [ ] **Step 3: Create hook shell with dependency injection**

Create `src/components/editor/stage/useEditorBitmapDrawing.ts` with this shape:

```ts
import { useCallback, useRef } from "react";
import type { Bbox } from "../../../lib/editorHistory";

type BitmapDrawingDeps = {
  pageKey: string;
  width: number;
  height: number;
  applyBitmapStroke: (payload: {
    width: number;
    height: number;
    strokes: [number, number][][];
    clear?: boolean;
    layerKey?: "brush" | "mask" | "recovery" | "reinpaint";
    erase?: boolean;
    brushSize?: number;
    color?: string;
    opacity?: number;
    hardness?: number;
    optimisticPath?: string;
    pngData?: string;
    clipMaskPng?: string;
    dirty_bbox?: Bbox;
  }) => Promise<void>;
  healPaintedRegion: (payload: { bbox: Bbox; maskPngData?: string }) => Promise<void>;
};

export function useEditorBitmapDrawing(deps: BitmapDrawingDeps) {
  const bitmapPersistQueueRef = useRef<Partial<Record<"brush" | "mask", Promise<void>>>>({});
  const recoveryPersistQueueRef = useRef<Promise<void>>(Promise.resolve());

  const enqueueBitmapPersist = useCallback((layerKey: "brush" | "mask", run: () => Promise<void>) => {
    bitmapPersistQueueRef.current[layerKey] = (bitmapPersistQueueRef.current[layerKey] ?? Promise.resolve())
      .catch(() => undefined)
      .then(run);
    return bitmapPersistQueueRef.current[layerKey];
  }, []);

  const enqueueRecoveryPersist = useCallback((run: () => Promise<void>) => {
    recoveryPersistQueueRef.current = recoveryPersistQueueRef.current.catch(() => undefined).then(run);
    return recoveryPersistQueueRef.current;
  }, []);

  return {
    enqueueBitmapPersist,
    enqueueRecoveryPersist,
    applyBitmapStroke: deps.applyBitmapStroke,
    healPaintedRegion: deps.healPaintedRegion,
  };
}
```

- [ ] **Step 4: Move queue ownership from controller to hook**

In `useEditorStageController.ts`:

- Remove local `bitmapPersistQueueRef`.
- Remove local `recoveryPersistQueueRef`.
- Instantiate `useEditorBitmapDrawing`.
- Replace queue assignments with `enqueueBitmapPersist(...)` and `enqueueRecoveryPersist(...)`.

Keep local preview construction in the controller during this task. Do not move `createBitmapStrokePreviewOnCanvas`, recovery preview, or healing mask logic yet.

- [ ] **Step 5: Validate bitmap flows**

Run:

```bash
npx vitest run src/lib/stores/__tests__/editorBitmapTools.test.ts src/lib/editorStage/__tests__/bitmapStrokePreview.test.ts src/lib/__tests__/healingBrushMask.test.ts
npm run check
```

Expected: PASS.

- [ ] **Step 6: Commit checkpoint**

```bash
git add src/components/editor/stage/useEditorBitmapDrawing.ts src/components/editor/stage/useEditorStageController.ts src/lib/__tests__/editorStroke.test.ts
git commit -m "refactor: isolate editor bitmap persistence queues"
```

---

### Task 6: Add Koharu-Style Text Block Workflow Without Removing Tools

**Files:**
- Modify: `src/components/editor/LayersPanel.tsx`
- Modify: `src/components/editor/LayerItem.tsx`
- Modify: `src/lib/editorScene.ts`
- Test: `src/lib/__tests__/editorScene.test.ts`

- [ ] **Step 1: Add derived status fields to scene tests**

Extend `src/lib/__tests__/editorScene.test.ts`:

```ts
it("derives text block status for panel badges", () => {
  const scene = buildEditorScene({ page: page(), pendingEdits: {}, selectedLayerId: null });

  expect(scene.textLayers[0].hasOriginal).toBe(true);
  expect(scene.textLayers[0].hasTranslation).toBe(true);
  expect(scene.textLayers[0].confidencePercent).toBe(90);
});
```

- [ ] **Step 2: Update `EditorSceneTextLayer`**

Add these fields in `src/lib/editorScene.ts`:

```ts
hasOriginal: boolean;
hasTranslation: boolean;
confidencePercent: number;
```

Populate them in `mergeLayer`:

```ts
const displayText = merged.traduzido ?? merged.translated ?? "";
const displayOriginal = merged.original ?? "";
return {
  ...merged,
  displayText,
  displayOriginal,
  hasOriginal: displayOriginal.trim().length > 0,
  hasTranslation: displayText.trim().length > 0,
  confidencePercent: Math.round(((merged.confianca_ocr ?? merged.ocr_confidence ?? 0) || 0) * 100),
  effectiveBbox: merged.layout_bbox ?? merged.bbox,
  visible: merged.visible !== false,
  locked: merged.locked === true,
  order: merged.order ?? 0,
};
```

- [ ] **Step 3: Add per-block actions to `LayerItem.tsx`**

Use existing store methods only:

```ts
const selectLayer = useEditorStore((s) => s.selectLayer);
const reProcessBlock = useEditorStore((s) => s.reProcessBlock);
```

Add action buttons for the selected/current item:

```tsx
<div className="mt-2 flex items-center gap-1">
  <button
    type="button"
    onClick={(event) => {
      event.stopPropagation();
      selectLayer(entry.id);
      void reProcessBlock("ocr");
    }}
    className="rounded-md border border-border px-2 py-1 text-[10px] text-text-secondary hover:bg-white/[0.04]"
  >
    OCR
  </button>
  <button
    type="button"
    onClick={(event) => {
      event.stopPropagation();
      selectLayer(entry.id);
      void reProcessBlock("translate");
    }}
    className="rounded-md border border-border px-2 py-1 text-[10px] text-text-secondary hover:bg-white/[0.04]"
  >
    Traduzir
  </button>
  <button
    type="button"
    onClick={(event) => {
      event.stopPropagation();
      selectLayer(entry.id);
      void reProcessBlock("inpaint");
    }}
    className="rounded-md border border-border px-2 py-1 text-[10px] text-text-secondary hover:bg-white/[0.04]"
  >
    Limpar
  </button>
</div>
```

Do not change tool modes. These buttons call existing manual block actions.

- [ ] **Step 4: Validate text block panel behavior**

Run:

```bash
npx vitest run src/lib/__tests__/editorScene.test.ts
npm run check
```

Expected: PASS.

- [ ] **Step 5: Run focused Playwright smoke**

Run:

```bash
npx playwright test e2e/editor-rebuild.spec.ts --grep "@smoke|editor Konva" --timeout=90000
```

Expected: PASS if the local dev server path is healthy. If `page.goto` times out while Vitest and TypeScript pass, record it as an environment/server issue and do not claim E2E passed.

- [ ] **Step 6: Commit checkpoint**

```bash
git add src/components/editor/LayersPanel.tsx src/components/editor/LayerItem.tsx src/lib/editorScene.ts src/lib/__tests__/editorScene.test.ts
git commit -m "feat: add text block workflow to editor panel"
```

---

### Task 7: Final Integration And Regression Gate

**Files:**
- Verify all modified files from Tasks 1-6.
- No new files unless a test failure identifies a concrete missing fixture.

- [ ] **Step 1: Run full focused frontend gate**

Run:

```bash
npm run check
npx vitest run src/lib/__tests__/editorScene.test.ts src/lib/__tests__/editorOps.test.ts src/lib/__tests__/editorStroke.test.ts
npx vitest run src/lib/__tests__/editorHistory.test.ts src/lib/stores/__tests__/editorStoreHistory.test.ts src/lib/stores/__tests__/editorBitmapTools.test.ts
npx vitest run src/lib/editorStage/__tests__/editorTransformConstraints.test.ts src/lib/editorStage/__tests__/bitmapStrokePreview.test.ts src/lib/editorStage/__tests__/editorSnapGuides.test.ts
```

Expected: PASS.

- [ ] **Step 2: Run editor smoke**

Run:

```bash
npx playwright test e2e/editor-rebuild.spec.ts --grep "@smoke|@manual-flow|editor Konva" --timeout=90000
```

Expected: PASS if local browser/dev-server setup is healthy.

- [ ] **Step 3: Inspect diff for accidental tool removal**

Run:

```bash
rg -n "export type EditorToolMode" src/lib/stores/editorStore.ts
rg -n "select|block|brush|repairBrush|reinpaintBrush|eraser|mask" src/components/editor/toolbar/ToolSidebar.tsx src/lib/stores/editorStore.ts
```

Expected:

- `EditorToolMode` still includes `select`, `block`, `brush`, `repairBrush`, `reinpaintBrush`, `eraser`, `mask`.
- `ToolSidebar.tsx` still exposes all existing tools.

- [ ] **Step 4: Run whitespace/diff sanity**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Commit final integration**

```bash
git add src docs/superpowers/plans/2026-05-16-editor-scene-architecture.md
git commit -m "refactor: improve editor scene architecture"
```

---

## Rollback Plan

- If panel behavior regresses, revert Task 6 first. It is additive UI behavior and should not block the architectural read model.
- If Stage brush behavior regresses, revert Task 5 and keep Task 4 pure helpers. The helpers are independently tested and lower risk.
- If history behavior regresses, revert Task 3. `editorStore.ts` method signatures must remain unchanged, so callers should recover immediately.
- If `EditorScene` has a bad assumption, fix Task 1 tests first and keep consumers behind `buildEditorScene` instead of reintroducing duplicated merging logic.

## Self-Review

- Spec coverage: The plan preserves current tools, keeps Konva, adds `EditorScene`, centralizes operation helpers, isolates bitmap drawing queues, and adds Koharu-style block workflow.
- Placeholder scan: No `TBD`, `TODO`, incomplete test commands, or undefined task dependencies remain.
- Type consistency: `EditorSceneTextLayer`, `EditorSceneImageLayer`, `EditorCommand`, `EditorToolMode`, `Bbox`, and existing store method names match the current codebase.
- Scope check: This is one cohesive editor architecture plan. It does not include pipeline rewrites, backend schema changes, or site migration.
