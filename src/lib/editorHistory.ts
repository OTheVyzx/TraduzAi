import type { Project, TextEntry, TextLayerStyle } from "./stores/appStore";

export type Bbox = TextEntry["bbox"];

export type BaseCmdMeta = {
  commandId: string;
  pageKey: string;
  createdAt: number;
};

export type SelectionState = {
  ids: string[];
  primary: string | null;
};

export type NonBatchCommand = BaseCmdMeta &
  (
    | {
        type: "edit-traduzido";
        layerId: string;
        before: string;
        after: string;
        sessionId: string;
      }
    | {
        type: "edit-estilo";
        layerId: string;
        before: Partial<TextLayerStyle>;
        after: Partial<TextLayerStyle>;
        touchedKeys: (keyof TextLayerStyle)[];
      }
    | { type: "edit-bbox"; layerId: string; before: Bbox; after: Bbox }
    | {
        type: "create-layer";
        layerId: string;
        layer: TextEntry;
        sourceLayerId?: string;
        insertIndex: number;
        selectionBefore?: SelectionState;
        selectionAfter?: SelectionState;
      }
    | {
        type: "delete-layer";
        layerId: string;
        layer: TextEntry;
        index: number;
        selectionBefore?: SelectionState;
        selectionAfter?: SelectionState;
      }
    | { type: "toggle-visibility"; layerId: string; before: boolean; after: boolean }
    | { type: "toggle-lock"; layerId: string; before: boolean; after: boolean }
    | { type: "reorder-layers"; before: string[]; after: string[] }
    | { type: "bitmap-stroke"; bbox: Bbox }
  );

export type BatchCommand = BaseCmdMeta & {
  type: "batch";
  label: string;
  commands: NonBatchCommand[];
};

export type EditorCommand = NonBatchCommand | BatchCommand;

export type HistoryStack = {
  pageKey: string;
  baseFingerprint: string;
  commands: EditorCommand[];
  index: number;
  memoryBytes: number;
};

export type ValidationResult = { ok: true } | { ok: false; reason: string };

export interface WorkingStateDraft {
  setWorkingTraduzido(pageKey: string, layerId: string, value: string): void;
  setWorkingEstiloPatch(
    pageKey: string,
    layerId: string,
    patch: Partial<TextLayerStyle>,
    touchedKeys: (keyof TextLayerStyle)[],
  ): void;
  setWorkingBbox(pageKey: string, layerId: string, bbox: Bbox): void;
  insertWorkingLayer(pageKey: string, layer: TextEntry, insertIndex: number): void;
  deleteWorkingLayer(pageKey: string, layerId: string): void;
  reorderWorkingLayers(pageKey: string, orderedIds: string[]): void;
  applyWorkingBitmapRegion(pageKey: string, bbox: Bbox, bytes: Uint8Array): void;
  setWorkingVisibility(pageKey: string, layerId: string, visible: boolean): void;
  setWorkingLocked(pageKey: string, layerId: string, locked: boolean): void;
  hasLayer(pageKey: string, layerId: string): boolean;
  getLayer(pageKey: string, layerId: string): TextEntry | null;
  getOrderedLayerIds(pageKey: string): string[];
  currentPageKey(): string;
  sanitizeSelection(): void;
}

export type BitmapCacheEntry = {
  pageKey: string;
  commandId: string;
  before: Uint8Array;
  after: Uint8Array;
  byteLength: number;
};

export const bitmapCache = new Map<string, BitmapCacheEntry>();

const TRIVIAL_COMMAND_BYTES = 256;
export const MAX_COMMANDS_PER_PAGE = 200;
export const MAX_BYTES_PER_PAGE = 50 * 1024 * 1024;
export const MAX_BYTES_GLOBAL = 150 * 1024 * 1024;

function isDevBuild() {
  return ((import.meta as ImportMeta & { env?: { DEV?: boolean } }).env?.DEV ?? false) === true;
}

function hashString(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

export function getPageKey(project: Pick<Project, "id" | "paginas">, currentPageIndex: number) {
  const page = project.paginas[currentPageIndex];
  const originalPath = page?.arquivo_original ?? "";
  const projectPart = project.id || hashString(originalPath);
  return `${projectPart}:${currentPageIndex}:${originalPath}`;
}

export function createHistoryStack(pageKey: string, baseFingerprint: string): HistoryStack {
  return {
    pageKey,
    baseFingerprint,
    commands: [],
    index: 0,
    memoryBytes: 0,
  };
}

export function historyBaseMatches(stack: HistoryStack, baseFingerprint: string): boolean {
  return stack.baseFingerprint === baseFingerprint;
}

export function updateHistoryBaseFingerprint(stack: HistoryStack, baseFingerprint: string): void {
  stack.baseFingerprint = baseFingerprint;
}

export function estimateCommandBytes(cmd: EditorCommand): number {
  if (cmd.type === "batch") {
    return cmd.commands.reduce((total, child) => total + estimateCommandBytes(child), TRIVIAL_COMMAND_BYTES);
  }
  if (cmd.type !== "bitmap-stroke") return TRIVIAL_COMMAND_BYTES;
  return bitmapCache.get(cmd.commandId)?.byteLength ?? TRIVIAL_COMMAND_BYTES;
}

export function validateCommand(
  cmd: EditorCommand,
  direction: "apply" | "revert",
  draft: WorkingStateDraft,
): ValidationResult {
  if (cmd.type === "batch") {
    for (const child of cmd.commands) {
      if ((child as { type: string }).type === "batch") {
        return { ok: false, reason: "nested batch rejeitado" };
      }
      const result = validateCommand(child, direction, draft);
      if (!result.ok) return result;
    }
    return { ok: true };
  }

  if (cmd.pageKey !== draft.currentPageKey()) {
    return { ok: false, reason: "comando pertence a outra pagina" };
  }

  if (cmd.type === "create-layer") {
    const exists = draft.hasLayer(cmd.pageKey, cmd.layerId);
    if (direction === "apply" && exists) return { ok: false, reason: "camada ja existe" };
    if (direction === "revert" && !exists) return { ok: false, reason: "camada nao encontrada" };
    return { ok: true };
  }

  if (cmd.type === "delete-layer") {
    const exists = draft.hasLayer(cmd.pageKey, cmd.layerId);
    if (direction === "apply" && !exists) return { ok: false, reason: "camada nao encontrada" };
    if (direction === "revert" && exists) return { ok: false, reason: "camada ja existe" };
    return { ok: true };
  }

  if (cmd.type === "reorder-layers") {
    return cmd.before.length > 0 || cmd.after.length > 0
      ? { ok: true }
      : { ok: false, reason: "reorder sem camadas" };
  }

  if (cmd.type === "bitmap-stroke") {
    return bitmapCache.has(cmd.commandId)
      ? { ok: true }
      : { ok: false, reason: "bitmap-stroke sem cache valido" };
  }

  if ("layerId" in cmd && !draft.hasLayer(cmd.pageKey, cmd.layerId)) {
    return { ok: false, reason: "camada nao encontrada" };
  }

  return { ok: true };
}

export function commandMatchesWorkingState(cmd: EditorCommand, draft: WorkingStateDraft): ValidationResult {
  if (cmd.type === "batch") {
    for (const child of cmd.commands) {
      const result = commandMatchesWorkingState(child, draft);
      if (!result.ok) return result;
    }
    return { ok: true };
  }

  const layer = "layerId" in cmd ? draft.getLayer(cmd.pageKey, cmd.layerId) : null;

  switch (cmd.type) {
    case "edit-traduzido":
      return layer?.traduzido === cmd.after || layer?.translated === cmd.after
        ? { ok: true }
        : { ok: false, reason: "texto traduzido diverge do working state" };
    case "edit-estilo":
      for (const key of cmd.touchedKeys) {
        if (layer?.estilo?.[key] !== cmd.after[key]) {
          return { ok: false, reason: `estilo.${String(key)} diverge do working state` };
        }
      }
      return { ok: true };
    case "edit-bbox":
      return bboxEquals(layer?.bbox, cmd.after) || bboxEquals(layer?.layout_bbox, cmd.after)
        ? { ok: true }
        : { ok: false, reason: "bbox diverge do working state" };
    case "create-layer":
      return draft.hasLayer(cmd.pageKey, cmd.layerId)
        ? { ok: true }
        : { ok: false, reason: "camada criada nao existe no working state" };
    case "delete-layer":
      return !draft.hasLayer(cmd.pageKey, cmd.layerId)
        ? { ok: true }
        : { ok: false, reason: "camada deletada ainda existe no working state" };
    case "toggle-visibility":
      return layer?.visible === cmd.after
        ? { ok: true }
        : { ok: false, reason: "visibilidade diverge do working state" };
    case "toggle-lock":
      return layer?.locked === cmd.after
        ? { ok: true }
        : { ok: false, reason: "lock diverge do working state" };
    case "reorder-layers":
      return arrayEquals(draft.getOrderedLayerIds(cmd.pageKey), cmd.after)
        ? { ok: true }
        : { ok: false, reason: "ordem das camadas diverge do working state" };
    case "bitmap-stroke": {
      const entry = bitmapCache.get(cmd.commandId);
      return entry && entry.pageKey === cmd.pageKey && entry.byteLength > 0
        ? { ok: true }
        : { ok: false, reason: "bitmap-stroke sem cache valido" };
    }
  }
}

export function applyCommand(cmd: EditorCommand, draft: WorkingStateDraft): void {
  if (cmd.type === "batch") {
    for (const child of cmd.commands) applyCommand(child, draft);
    draft.sanitizeSelection();
    return;
  }

  switch (cmd.type) {
    case "edit-traduzido":
      draft.setWorkingTraduzido(cmd.pageKey, cmd.layerId, cmd.after);
      break;
    case "edit-estilo":
      draft.setWorkingEstiloPatch(cmd.pageKey, cmd.layerId, cmd.after, cmd.touchedKeys);
      break;
    case "edit-bbox":
      draft.setWorkingBbox(cmd.pageKey, cmd.layerId, cmd.after);
      break;
    case "create-layer":
      draft.insertWorkingLayer(cmd.pageKey, cmd.layer, cmd.insertIndex);
      break;
    case "delete-layer":
      draft.deleteWorkingLayer(cmd.pageKey, cmd.layerId);
      break;
    case "reorder-layers":
      draft.reorderWorkingLayers(cmd.pageKey, cmd.after);
      break;
    case "bitmap-stroke": {
      const entry = bitmapCache.get(cmd.commandId);
      if (entry) draft.applyWorkingBitmapRegion(cmd.pageKey, cmd.bbox, entry.after);
      break;
    }
    case "toggle-visibility":
      draft.setWorkingVisibility(cmd.pageKey, cmd.layerId, cmd.after);
      break;
    case "toggle-lock":
      draft.setWorkingLocked(cmd.pageKey, cmd.layerId, cmd.after);
      break;
  }
  draft.sanitizeSelection();
}

export function revertCommand(cmd: EditorCommand, draft: WorkingStateDraft): void {
  if (cmd.type === "batch") {
    for (const child of [...cmd.commands].reverse()) revertCommand(child, draft);
    draft.sanitizeSelection();
    return;
  }

  switch (cmd.type) {
    case "edit-traduzido":
      draft.setWorkingTraduzido(cmd.pageKey, cmd.layerId, cmd.before);
      break;
    case "edit-estilo":
      draft.setWorkingEstiloPatch(cmd.pageKey, cmd.layerId, cmd.before, cmd.touchedKeys);
      break;
    case "edit-bbox":
      draft.setWorkingBbox(cmd.pageKey, cmd.layerId, cmd.before);
      break;
    case "create-layer":
      draft.deleteWorkingLayer(cmd.pageKey, cmd.layerId);
      break;
    case "delete-layer":
      draft.insertWorkingLayer(cmd.pageKey, cmd.layer, cmd.index);
      break;
    case "reorder-layers":
      draft.reorderWorkingLayers(cmd.pageKey, cmd.before);
      break;
    case "bitmap-stroke": {
      const entry = bitmapCache.get(cmd.commandId);
      if (entry) draft.applyWorkingBitmapRegion(cmd.pageKey, cmd.bbox, entry.before);
      break;
    }
    case "toggle-visibility":
      draft.setWorkingVisibility(cmd.pageKey, cmd.layerId, cmd.before);
      break;
    case "toggle-lock":
      draft.setWorkingLocked(cmd.pageKey, cmd.layerId, cmd.before);
      break;
  }
  draft.sanitizeSelection();
}

export function executeCommand(
  cmd: EditorCommand,
  draft: WorkingStateDraft,
  stack: HistoryStack,
): ValidationResult {
  const validation = validateCommand(cmd, "apply", draft);
  if (!validation.ok) return validation;
  applyCommand(cmd, draft);
  return pushCommand(cmd, stack);
}

export function recordCommand(
  cmd: EditorCommand,
  draft: WorkingStateDraft,
  stack: HistoryStack,
): ValidationResult {
  if (isNoOpCommand(cmd)) return { ok: false, reason: "no-op" };

  const match = commandMatchesWorkingState(cmd, draft);
  if (!match.ok) {
    if (isDevBuild()) console.warn(match.reason);
    return match;
  }

  return pushCommand(cmd, stack);
}

export function undo(stack: HistoryStack, draft: WorkingStateDraft): ValidationResult {
  if (stack.index <= 0) return { ok: false, reason: "nada para desfazer" };
  const cmd = stack.commands[stack.index - 1];
  const validation = validateCommand(cmd, "revert", draft);
  if (!validation.ok) return validation;
  revertCommand(cmd, draft);
  stack.index -= 1;
  return { ok: true };
}

export function redo(stack: HistoryStack, draft: WorkingStateDraft): ValidationResult {
  if (stack.index >= stack.commands.length) return { ok: false, reason: "nada para refazer" };
  const cmd = stack.commands[stack.index];
  const validation = validateCommand(cmd, "apply", draft);
  if (!validation.ok) return validation;
  applyCommand(cmd, draft);
  stack.index += 1;
  return { ok: true };
}

export function disposeCommand(cmd: EditorCommand): void {
  if (cmd.type === "batch") {
    for (const child of cmd.commands) disposeCommand(child);
    return;
  }
  bitmapCache.delete(cmd.commandId);
}

export function disposeCommandById(commandId: string): void {
  bitmapCache.delete(commandId);
}

export function disposeAllForPage(pageKey: string): void {
  for (const [commandId, entry] of bitmapCache) {
    if (entry.pageKey === pageKey) bitmapCache.delete(commandId);
  }
}

export function disposeAll(): void {
  bitmapCache.clear();
}

export function pruneHistoryStacksByGlobalCap(stacks: HistoryStack[]): void {
  let totalBytes = stacks.reduce((total, stack) => total + stack.memoryBytes, 0);
  while (totalBytes > MAX_BYTES_GLOBAL) {
    const candidate = stacks
      .filter((stack) => stack.commands.length > 0)
      .sort((a, b) => {
        const aCreatedAt = a.commands[0]?.createdAt ?? Number.POSITIVE_INFINITY;
        const bCreatedAt = b.commands[0]?.createdAt ?? Number.POSITIVE_INFINITY;
        return aCreatedAt - bCreatedAt;
      })[0];
    if (!candidate) return;
    const before = candidate.memoryBytes;
    pruneOldest(candidate, 1);
    if (candidate.memoryBytes >= before) return;
    totalBytes -= before - candidate.memoryBytes;
  }
}

function pushCommand(cmd: EditorCommand, stack: HistoryStack): ValidationResult {
  const removed = stack.commands.slice(stack.index);
  for (const oldCommand of removed) disposeCommand(oldCommand);
  stack.commands = [...stack.commands.slice(0, stack.index), cmd];
  stack.index = stack.commands.length;
  stack.memoryBytes = stack.commands.reduce((total, item) => total + estimateCommandBytes(item), 0);
  enforcePageCaps(stack);
  return { ok: true };
}

function enforcePageCaps(stack: HistoryStack): void {
  while (stack.commands.length > MAX_COMMANDS_PER_PAGE) {
    pruneOldest(stack, 1);
  }
  while (stack.memoryBytes > MAX_BYTES_PER_PAGE && stack.commands.length > 0) {
    pruneOldest(stack, 1);
  }
}

function pruneOldest(stack: HistoryStack, count: number): void {
  const removed = stack.commands.splice(0, count);
  for (const cmd of removed) disposeCommand(cmd);
  stack.index = Math.max(0, stack.index - removed.length);
  stack.memoryBytes = stack.commands.reduce((total, item) => total + estimateCommandBytes(item), 0);
}

function isNoOpCommand(cmd: EditorCommand): boolean {
  if (cmd.type === "batch") return cmd.commands.every(isNoOpCommand);
  switch (cmd.type) {
    case "edit-traduzido":
      return cmd.before === cmd.after;
    case "edit-estilo":
      return cmd.touchedKeys.every((key) => cmd.before[key] === cmd.after[key]);
    case "edit-bbox":
      return bboxEquals(cmd.before, cmd.after);
    case "toggle-visibility":
    case "toggle-lock":
      return cmd.before === cmd.after;
    case "reorder-layers":
      return arrayEquals(cmd.before, cmd.after);
    case "create-layer":
    case "delete-layer":
    case "bitmap-stroke":
      return false;
  }
}

function bboxEquals(a: Bbox | undefined, b: Bbox | undefined) {
  if (!a || !b) return false;
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) < 0.01);
}

function arrayEquals(a: string[], b: string[]) {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}
