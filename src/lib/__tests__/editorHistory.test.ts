import { describe, expect, it, vi } from "vitest";
import type { TextEntry, TextLayerStyle } from "../stores/appStore";
import {
  bitmapCache,
  commandMatchesWorkingState,
  createHistoryStack,
  disposeAll,
  executeCommand,
  historyBaseMatches,
  recordCommand,
  redo,
  undo,
  type Bbox,
  type EditorCommand,
  type WorkingStateDraft,
} from "../editorHistory";

const pageKey = "project:0:page.png";

function makeLayer(id: string, overrides: Partial<TextEntry> = {}): TextEntry {
  const estilo: TextLayerStyle = {
    fonte: "ComicNeue-Bold.ttf",
    tamanho: 28,
    cor: "#ffffff",
    cor_gradiente: [],
    contorno: "#000000",
    contorno_px: 2,
    glow: false,
    glow_cor: "",
    glow_px: 0,
    sombra: false,
    sombra_cor: "",
    sombra_offset: [0, 0],
    bold: false,
    italico: false,
    rotacao: 0,
    alinhamento: "center",
  };

  return {
    id,
    kind: "text",
    bbox: [0, 0, 100, 100],
    layout_bbox: [0, 0, 100, 100],
    source_bbox: [0, 0, 100, 100],
    render_bbox: null,
    tipo: "fala",
    original: "Hello",
    traduzido: "Ola",
    translated: "Ola",
    confianca_ocr: 1,
    ocr_confidence: 1,
    estilo,
    style: estilo,
    visible: true,
    locked: false,
    order: 0,
    ...overrides,
  };
}

class Draft implements WorkingStateDraft {
  layers = new Map<string, TextEntry>();

  constructor(layers: TextEntry[] = [makeLayer("a")]) {
    for (const layer of layers) this.layers.set(layer.id, layer);
  }

  setWorkingTraduzido(_pageKey: string, layerId: string, value: string): void {
    const layer = this.layers.get(layerId);
    if (!layer) return;
    this.layers.set(layerId, { ...layer, traduzido: value, translated: value });
  }

  setWorkingEstiloPatch(
    _pageKey: string,
    layerId: string,
    patch: Partial<TextLayerStyle>,
    touchedKeys: (keyof TextLayerStyle)[],
  ): void {
    const layer = this.layers.get(layerId);
    if (!layer) return;
    const estilo = { ...layer.estilo };
    for (const key of touchedKeys) {
      estilo[key] = patch[key] as never;
    }
    this.layers.set(layerId, { ...layer, estilo, style: estilo });
  }

  setWorkingBbox(_pageKey: string, layerId: string, bbox: Bbox): void {
    const layer = this.layers.get(layerId);
    if (!layer) return;
    this.layers.set(layerId, { ...layer, bbox, layout_bbox: bbox });
  }

  insertWorkingLayer(_pageKey: string, layer: TextEntry, insertIndex: number): void {
    const next = [...this.layers.values()];
    next.splice(insertIndex, 0, layer);
    this.layers = new Map(next.map((item, index) => [item.id, { ...item, order: index }]));
  }

  deleteWorkingLayer(_pageKey: string, layerId: string): void {
    this.layers.delete(layerId);
  }

  reorderWorkingLayers(_pageKey: string, orderedIds: string[]): void {
    const next: TextEntry[] = [];
    for (const [index, id] of orderedIds.entries()) {
      const layer = this.layers.get(id);
      if (layer) next.push({ ...layer, order: index });
    }
    this.layers = new Map(next.map((layer) => [layer.id, layer]));
  }

  applyWorkingBitmapRegion(): void {
    return;
  }

  setWorkingVisibility(_pageKey: string, layerId: string, visible: boolean): void {
    const layer = this.layers.get(layerId);
    if (!layer) return;
    this.layers.set(layerId, { ...layer, visible });
  }

  setWorkingLocked(_pageKey: string, layerId: string, locked: boolean): void {
    const layer = this.layers.get(layerId);
    if (!layer) return;
    this.layers.set(layerId, { ...layer, locked });
  }

  hasLayer(_pageKey: string, layerId: string): boolean {
    return this.layers.has(layerId);
  }

  getLayer(_pageKey: string, layerId: string): TextEntry | null {
    return this.layers.get(layerId) ?? null;
  }

  getOrderedLayerIds(): string[] {
    return [...this.layers.values()].sort((a, b) => (a.order ?? 0) - (b.order ?? 0)).map((layer) => layer.id);
  }

  currentPageKey(): string {
    return pageKey;
  }

  sanitizeSelection(): void {
    return;
  }
}

function meta(id: string) {
  return { commandId: id, pageKey, createdAt: Number(id.replace(/\D/g, "")) || 1 };
}

describe("editorHistory", () => {
  it("pushes, undoes and redoes an edit command", () => {
    const draft = new Draft();
    const stack = createHistoryStack(pageKey, "base-a");

    const result = executeCommand(
      { ...meta("cmd1"), type: "edit-traduzido", layerId: "a", before: "Ola", after: "Oi", sessionId: "s1" },
      draft,
      stack,
    );

    expect(result).toEqual({ ok: true });
    expect(draft.getLayer(pageKey, "a")?.traduzido).toBe("Oi");
    expect(stack.index).toBe(1);

    expect(undo(stack, draft)).toEqual({ ok: true });
    expect(draft.getLayer(pageKey, "a")?.traduzido).toBe("Ola");
    expect(stack.index).toBe(0);

    expect(redo(stack, draft)).toEqual({ ok: true });
    expect(draft.getLayer(pageKey, "a")?.traduzido).toBe("Oi");
    expect(stack.index).toBe(1);
  });

  it("records no-op commands as rejected without changing stack", () => {
    const draft = new Draft();
    const stack = createHistoryStack(pageKey, "base-a");

    const result = recordCommand(
      { ...meta("cmd1"), type: "edit-bbox", layerId: "a", before: [0, 0, 100, 100], after: [0, 0, 100, 100] },
      draft,
      stack,
    );

    expect(result).toEqual({ ok: false, reason: "no-op" });
    expect(stack.commands).toHaveLength(0);
  });

  it("discards redo commands and disposes their bitmap cache after a new command", () => {
    disposeAll();
    const draft = new Draft();
    const stack = createHistoryStack(pageKey, "base-a");
    bitmapCache.set("bitmap1", {
      pageKey,
      commandId: "bitmap1",
      before: new Uint8Array([1]),
      after: new Uint8Array([2]),
      byteLength: 2,
    });

    expect(executeCommand({ ...meta("bitmap1"), type: "bitmap-stroke", bbox: [0, 0, 1, 1] }, draft, stack)).toEqual({
      ok: true,
    });
    expect(undo(stack, draft)).toEqual({ ok: true });

    expect(
      executeCommand(
        { ...meta("cmd2"), type: "edit-traduzido", layerId: "a", before: "Ola", after: "Oi", sessionId: "s1" },
        draft,
        stack,
      ),
    ).toEqual({ ok: true });

    expect(bitmapCache.has("bitmap1")).toBe(false);
    expect(stack.commands.map((cmd) => cmd.commandId)).toEqual(["cmd2"]);
  });

  it("prunes oldest commands when a stack exceeds 200 commands", () => {
    const draft = new Draft();
    const stack = createHistoryStack(pageKey, "base-a");

    for (let index = 0; index < 205; index += 1) {
      const before = index === 0 ? "Ola" : `texto ${index - 1}`;
      const after = `texto ${index}`;
      expect(
        executeCommand(
          { ...meta(`cmd${index}`), type: "edit-traduzido", layerId: "a", before, after, sessionId: `s${index}` },
          draft,
          stack,
        ),
      ).toEqual({ ok: true });
    }

    expect(stack.commands).toHaveLength(200);
    expect(stack.index).toBe(200);
    expect(stack.commands[0].commandId).toBe("cmd5");
  });

  it("applies batch commands in order and reverts them in reverse order", () => {
    const draft = new Draft();
    const stack = createHistoryStack(pageKey, "base-a");
    const batch: EditorCommand = {
      ...meta("batch1"),
      type: "batch",
      label: "Editar texto e caixa",
      commands: [
        { ...meta("cmd1"), type: "edit-traduzido", layerId: "a", before: "Ola", after: "Oi", sessionId: "s1" },
        { ...meta("cmd2"), type: "edit-bbox", layerId: "a", before: [0, 0, 100, 100], after: [10, 10, 120, 120] },
      ],
    };

    expect(executeCommand(batch, draft, stack)).toEqual({ ok: true });
    expect(draft.getLayer(pageKey, "a")?.traduzido).toBe("Oi");
    expect(draft.getLayer(pageKey, "a")?.bbox).toEqual([10, 10, 120, 120]);

    expect(undo(stack, draft)).toEqual({ ok: true });
    expect(draft.getLayer(pageKey, "a")?.traduzido).toBe("Ola");
    expect(draft.getLayer(pageKey, "a")?.bbox).toEqual([0, 0, 100, 100]);
  });

  it("rejects nested batches defensively at runtime", () => {
    const draft = new Draft();
    const stack = createHistoryStack(pageKey, "base-a");
    const nested = {
      ...meta("batch1"),
      type: "batch",
      label: "Nested",
      commands: [{ ...meta("batch2"), type: "batch", label: "Inner", commands: [] }],
    } as unknown as EditorCommand;

    expect(executeCommand(nested, draft, stack)).toEqual({ ok: false, reason: "nested batch rejeitado" });
  });

  it("rejects recordCommand when command does not match the working state and warns in dev", () => {
    const draft = new Draft();
    const stack = createHistoryStack(pageKey, "base-a");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const result = recordCommand(
      { ...meta("cmd1"), type: "edit-traduzido", layerId: "a", before: "Ola", after: "Divergente", sessionId: "s1" },
      draft,
      stack,
    );

    expect(result).toEqual({ ok: false, reason: "texto traduzido diverge do working state" });
    expect(commandMatchesWorkingState(stack.commands[0] ?? ({ ...meta("none"), type: "batch", label: "", commands: [] } as EditorCommand), draft).ok).toBe(true);
    warn.mockRestore();
  });

  it("detects base fingerprint changes explicitly", () => {
    const stack = createHistoryStack(pageKey, "base-a");

    expect(historyBaseMatches(stack, "base-a")).toBe(true);
    expect(historyBaseMatches(stack, "base-b")).toBe(false);
  });

  it("applies and reverts persistent visibility and lock commands", () => {
    const draft = new Draft();
    const stack = createHistoryStack(pageKey, "base-a");

    expect(
      executeCommand(
        { ...meta("cmd1"), type: "toggle-visibility", layerId: "a", before: true, after: false },
        draft,
        stack,
      ),
    ).toEqual({ ok: true });
    expect(draft.getLayer(pageKey, "a")?.visible).toBe(false);

    expect(
      executeCommand(
        { ...meta("cmd2"), type: "toggle-lock", layerId: "a", before: false, after: true },
        draft,
        stack,
      ),
    ).toEqual({ ok: true });
    expect(draft.getLayer(pageKey, "a")?.locked).toBe(true);

    expect(undo(stack, draft)).toEqual({ ok: true });
    expect(draft.getLayer(pageKey, "a")?.locked).toBe(false);

    expect(undo(stack, draft)).toEqual({ ok: true });
    expect(draft.getLayer(pageKey, "a")?.visible).toBe(true);
  });
});
