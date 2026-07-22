export type EditorMode = "traduzai" | "studio" | "studio-translation";
export type EditorViewKey = "original" | "inpainted" | "translated";

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
  showTypesettingControls: boolean;
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
const STUDIO_TRANSLATION_TOOLS: EditorToolKey[] = ["select", "block"];

const TRADUZAI_CAPABILITIES: EditorCapabilities = {
  showPipelineActions: true,
  showSourceLanguage: true,
  showBlockProcessingActions: true,
  useProfessionalLayersPresentation: false,
  showTypesettingControls: true,
};

const STUDIO_CAPABILITIES: EditorCapabilities = {
  showPipelineActions: false,
  showSourceLanguage: false,
  showBlockProcessingActions: false,
  useProfessionalLayersPresentation: true,
  showTypesettingControls: true,
};

const STUDIO_TRANSLATION_CAPABILITIES: EditorCapabilities = {
  ...STUDIO_CAPABILITIES,
  showTypesettingControls: false,
};

export function resolveEditorCapabilities(mode: EditorMode): EditorCapabilities {
  if (mode === "studio-translation") return STUDIO_TRANSLATION_CAPABILITIES;
  return mode === "studio" ? STUDIO_CAPABILITIES : TRADUZAI_CAPABILITIES;
}

export function editorToolsForMode(mode: EditorMode): EditorToolKey[] {
  if (mode === "studio-translation") return [...STUDIO_TRANSLATION_TOOLS];
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

export function editorViewLabelForMode(mode: EditorMode, view: EditorViewKey): string {
  if (view === "original") return "Original";
  if (view === "inpainted") return "Limpa";
  return mode === "studio-translation" ? "Traduzida" : "Camadas";
}

export function isEditorViewAvailable(
  mode: EditorMode,
  view: EditorViewKey,
  hasInpaintLayer: boolean,
): boolean {
  return mode !== "studio-translation" || view !== "inpainted" || hasInpaintLayer;
}
