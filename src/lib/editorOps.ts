import type { EditorCommand } from "./editorHistory";

export function nextEditorCommandId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

export function buildToggleVisibilityCommand(args: {
  pageKey: string;
  layerId: string;
  before: boolean;
  after: boolean;
}): EditorCommand {
  return {
    commandId: nextEditorCommandId("toggle-visibility"),
    pageKey: args.pageKey,
    createdAt: Date.now(),
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
    commandId: nextEditorCommandId("toggle-lock"),
    pageKey: args.pageKey,
    createdAt: Date.now(),
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
    commandId: nextEditorCommandId("reorder-layers"),
    pageKey: args.pageKey,
    createdAt: Date.now(),
    type: "reorder-layers",
    before: [...args.before],
    after: [...args.after],
  };
}
