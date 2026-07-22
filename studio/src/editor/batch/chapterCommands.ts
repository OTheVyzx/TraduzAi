import type {
  StudioProject,
  StudioTextLayer,
  StudioTextStyle,
} from "../../project/studioProject";

export interface ChapterLayerRef {
  pageIndex: number;
  layerId: string;
}

export interface ChapterStyleClipboard {
  source: ChapterLayerRef;
  style: StudioTextStyle;
}

export interface ChapterCommand {
  id: string;
  label: string;
  before: StudioProject;
  after: StudioProject;
  affectedLayers: ChapterLayerRef[];
  createdAt: number;
}

interface ChapterFieldSnapshot {
  exists: boolean;
  value?: unknown;
}

export interface ChapterLayerPatch extends ChapterLayerRef {
  fields: Record<string, { before: ChapterFieldSnapshot; after: ChapterFieldSnapshot }>;
}

export interface ChapterHistoryEntry {
  id: string;
  label: string;
  patches: ChapterLayerPatch[];
  createdAt: number;
}

export interface ChapterReplaceOptions {
  query: string;
  replacement: string;
  caseSensitive?: boolean;
  wholeWord?: boolean;
}

export interface ChapterReplacementPreview extends ChapterLayerRef {
  id: string;
  before: string;
  after: string;
  occurrences: number;
}

export interface ChapterReviewItem extends ChapterLayerRef {
  id: string;
  pageNumber: number;
  original: string;
  translated: string;
  reasons: string[];
}

function clone<T>(value: T): T {
  if (value === undefined) return value;
  return JSON.parse(JSON.stringify(value)) as T;
}

function valuesEqual(left: unknown, right: unknown) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function fieldSnapshot(layer: StudioTextLayer, key: string): ChapterFieldSnapshot {
  if (!Object.prototype.hasOwnProperty.call(layer, key)) return { exists: false };
  return { exists: true, value: clone((layer as unknown as Record<string, unknown>)[key]) };
}

function randomId() {
  return typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function layerAt(project: StudioProject, ref: ChapterLayerRef): StudioTextLayer {
  const page = project.paginas[ref.pageIndex];
  if (!page) throw new Error(`Pagina ${ref.pageIndex + 1} nao encontrada`);
  const layer = page.text_layers.find((item) => item.id === ref.layerId);
  if (!layer) throw new Error(`Camada de texto nao encontrada: ${ref.layerId}`);
  return layer;
}

function createCommand(
  project: StudioProject,
  label: string,
  affectedLayers: ChapterLayerRef[],
  transform: (draft: StudioProject) => void,
): ChapterCommand {
  const before = clone(project);
  const after = clone(project);
  transform(after);
  return {
    id: randomId(),
    label,
    before,
    after,
    affectedLayers: clone(affectedLayers),
    createdAt: Date.now(),
  };
}

function syncPageTextAliases(project: StudioProject, pageIndexes: Iterable<number>) {
  for (const pageIndex of new Set(pageIndexes)) {
    const page = project.paginas[pageIndex];
    if (page) page.textos = page.text_layers;
  }
}

function preferredStyle(layer: StudioTextLayer): StudioTextStyle {
  const style = layer.style ?? {};
  const legacy = layer.estilo ?? {};
  return Object.keys(style).length > 0 ? style : legacy;
}

export function copyStyleFromLayer(
  project: StudioProject,
  source: ChapterLayerRef,
): ChapterStyleClipboard {
  return {
    source: clone(source),
    style: clone(preferredStyle(layerAt(project, source))),
  };
}

export function createApplyStyleCommand(
  project: StudioProject,
  clipboard: ChapterStyleClipboard,
  targets: ChapterLayerRef[],
): ChapterCommand {
  const uniqueTargets = targets.filter(
    (target, index, all) => all.findIndex(
      (candidate) => candidate.pageIndex === target.pageIndex && candidate.layerId === target.layerId,
    ) === index,
  ).filter((target) => layerAt(project, target).locked !== true);
  return createCommand(project, "Aplicar estilo", uniqueTargets, (draft) => {
    for (const target of uniqueTargets) {
      const layer = layerAt(draft, target);
      layer.style = clone(clipboard.style);
      layer.estilo = clone(clipboard.style);
    }
    syncPageTextAliases(draft, uniqueTargets.map((target) => target.pageIndex));
  });
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function replacementPattern(options: ChapterReplaceOptions) {
  if (!options.query) throw new Error("Digite o texto que deseja localizar");
  const source = escapeRegExp(options.query);
  const bounded = options.wholeWord
    ? `(?<![\\p{L}\\p{N}_])${source}(?![\\p{L}\\p{N}_])`
    : source;
  return new RegExp(bounded, options.caseSensitive ? "gu" : "giu");
}

export function previewChapterReplacements(
  project: StudioProject,
  options: ChapterReplaceOptions,
): ChapterReplacementPreview[] {
  const pattern = replacementPattern(options);
  const previews: ChapterReplacementPreview[] = [];
  project.paginas.forEach((page, pageIndex) => {
    page.text_layers.forEach((layer) => {
      if (layer.locked === true) return;
      const before = layer.translated ?? layer.traduzido ?? "";
      const matches = [...before.matchAll(pattern)];
      if (matches.length === 0) return;
      previews.push({
        id: `replace:${pageIndex}:${layer.id}`,
        pageIndex,
        layerId: layer.id,
        before,
        after: before.replace(pattern, () => options.replacement),
        occurrences: matches.length,
      });
    });
  });
  return previews;
}

export function createReplaceTextCommand(
  project: StudioProject,
  replacements: ChapterReplacementPreview[],
): ChapterCommand {
  const valid = replacements.filter(
    (item) => item.before !== item.after && layerAt(project, item).locked !== true,
  );
  return createCommand(project, "Substituir texto no capitulo", valid, (draft) => {
    for (const replacement of valid) {
      const layer = layerAt(draft, replacement);
      const current = layer.translated ?? layer.traduzido ?? "";
      if (current !== replacement.before) {
        throw new Error(`O texto da camada ${replacement.layerId} mudou depois da pre-visualizacao`);
      }
      layer.translated = replacement.after;
      layer.traduzido = replacement.after;
    }
    syncPageTextAliases(draft, valid.map((item) => item.pageIndex));
  });
}

function reviewWasResolved(layer: StudioTextLayer, reasons: string[]) {
  const review = layer.studio_review;
  if (typeof review !== "object" || review === null) return false;
  const record = review as Record<string, unknown>;
  if (record.status !== "resolved") return false;
  const translated = layer.translated ?? layer.traduzido ?? "";
  const recordedReasons = Array.isArray(record.reasons) ? record.reasons.map(String) : [];
  return record.translated === translated
    && JSON.stringify(recordedReasons) === JSON.stringify(reasons);
}

function reviewReasons(layer: StudioTextLayer) {
  const reasons: string[] = [];
  if (layer.review_required === true || layer.route_action === "review_required") {
    reasons.push("Revisao solicitada");
  }
  if (Array.isArray(layer.qa_flags)) {
    for (const flag of layer.qa_flags) {
      const reason = String(flag).trim();
      if (reason && !reasons.includes(reason)) reasons.push(reason);
    }
  }
  const original = String(layer.original ?? "").trim();
  const translated = String(layer.translated ?? layer.traduzido ?? "").trim();
  if (original && !translated) reasons.push("Traducao vazia");
  return reasons;
}

export function buildChapterReviewQueue(project: StudioProject): ChapterReviewItem[] {
  const queue: ChapterReviewItem[] = [];
  project.paginas.forEach((page, pageIndex) => {
    page.text_layers.forEach((layer) => {
      const reasons = reviewReasons(layer);
      if (reasons.length === 0) return;
      if (reviewWasResolved(layer, reasons)) return;
      queue.push({
        id: `text:${pageIndex}:${layer.id}`,
        pageIndex,
        pageNumber: page.numero ?? pageIndex + 1,
        layerId: layer.id,
        original: layer.original ?? "",
        translated: layer.translated ?? layer.traduzido ?? "",
        reasons,
      });
    });
  });
  return queue;
}

export function createResolveReviewCommand(
  project: StudioProject,
  items: ChapterReviewItem[],
  resolvedAt = new Date().toISOString(),
): ChapterCommand {
  return createCommand(project, "Resolver itens de revisao", items, (draft) => {
    for (const item of items) {
      const layer = layerAt(draft, item);
      layer.studio_review = {
        status: "resolved",
        resolved_at: resolvedAt,
        reasons: clone(item.reasons),
        translated: layer.translated ?? layer.traduzido ?? "",
      };
    }
    syncPageTextAliases(draft, items.map((item) => item.pageIndex));
  });
}

export function restoreChapterCommand(
  command: ChapterCommand,
  direction: "undo" | "redo",
): StudioProject {
  return clone(direction === "undo" ? command.before : command.after);
}

export function createChapterHistoryEntry(command: ChapterCommand): ChapterHistoryEntry {
  const patches = command.affectedLayers.flatMap((ref): ChapterLayerPatch[] => {
    const beforeLayer = layerAt(command.before, ref);
    const afterLayer = layerAt(command.after, ref);
    const keys = new Set([
      ...Object.keys(beforeLayer),
      ...Object.keys(afterLayer),
    ]);
    const fields: ChapterLayerPatch["fields"] = {};
    for (const key of keys) {
      const before = fieldSnapshot(beforeLayer, key);
      const after = fieldSnapshot(afterLayer, key);
      if (before.exists === after.exists && valuesEqual(before.value, after.value)) continue;
      fields[key] = { before, after };
    }
    return Object.keys(fields).length > 0 ? [{ ...clone(ref), fields }] : [];
  });
  return {
    id: command.id,
    label: command.label,
    patches,
    createdAt: command.createdAt,
  };
}

export function applyChapterHistoryEntry(
  project: StudioProject,
  entry: ChapterHistoryEntry,
  direction: "undo" | "redo",
): StudioProject {
  const draft = clone(project);
  const expectedSide = direction === "undo" ? "after" : "before";
  const targetSide = direction === "undo" ? "before" : "after";
  for (const patch of entry.patches) {
    const layer = layerAt(draft, patch);
    const record = layer as unknown as Record<string, unknown>;
    for (const [key, states] of Object.entries(patch.fields)) {
      const current = fieldSnapshot(layer, key);
      const expected = states[expectedSide];
      if (current.exists !== expected.exists || !valuesEqual(current.value, expected.value)) {
        throw new Error(`A camada ${patch.layerId} mudou no campo ${key}; o historico de lote foi preservado sem sobrescrever a edicao`);
      }
      const target = states[targetSide];
      if (target.exists) record[key] = clone(target.value);
      else delete record[key];
    }
  }
  syncPageTextAliases(draft, entry.patches.map((patch) => patch.pageIndex));
  return draft;
}
