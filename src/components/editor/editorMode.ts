export type EditorMode = "traduzai" | "studio";

export type EditorToolKey =
  | "select"
  | "block"
  | "brush"
  | "repairBrush"
  | "reinpaintBrush"
  | "eraser"
  | "mask"
  | "process";

export type LayerProcessingAction = "ocr" | "translate" | "inpaint";

export interface EditorCapabilities {
  showPipelineActions: boolean;
  showSourceLanguage: boolean;
  showBlockProcessingActions: boolean;
  useProfessionalLayersPresentation: boolean;
}

const ALL_EDITOR_TOOLS: EditorToolKey[] = [
  "select",
  "block",
  "brush",
  "repairBrush",
  "reinpaintBrush",
  "eraser",
  "mask",
  "process",
];

const STUDIO_EDITOR_TOOLS: EditorToolKey[] = ["select", "block", "brush", "eraser", "mask"];

const TRADUZAI_CAPABILITIES: EditorCapabilities = {
  showPipelineActions: true,
  showSourceLanguage: true,
  showBlockProcessingActions: true,
  useProfessionalLayersPresentation: false,
};

const STUDIO_CAPABILITIES: EditorCapabilities = {
  showPipelineActions: false,
  showSourceLanguage: false,
  showBlockProcessingActions: false,
  useProfessionalLayersPresentation: true,
};

export function resolveEditorCapabilities(mode: EditorMode): EditorCapabilities {
  return mode === "studio" ? STUDIO_CAPABILITIES : TRADUZAI_CAPABILITIES;
}

export function editorToolsForMode(mode: EditorMode): EditorToolKey[] {
  return mode === "studio" ? [...STUDIO_EDITOR_TOOLS] : [...ALL_EDITOR_TOOLS];
}

export function isEditorToolVisible(mode: EditorMode, tool: EditorToolKey): boolean {
  return editorToolsForMode(mode).includes(tool);
}

export function layerProcessingActionsForMode(mode: EditorMode): LayerProcessingAction[] {
  return resolveEditorCapabilities(mode).showBlockProcessingActions
    ? ["ocr", "translate", "inpaint"]
    : [];
}
