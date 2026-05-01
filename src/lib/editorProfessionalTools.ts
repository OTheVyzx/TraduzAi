export type EditorProfessionalToolKey =
  | "select"
  | "move_text"
  | "edit_text"
  | "mask_brush"
  | "eraser"
  | "region_reprocess"
  | "compare";

export type EditorProfessionalTool = {
  key: EditorProfessionalToolKey;
  label: string;
  hotkey: string;
  area: "texto" | "mascara" | "pagina";
};

export const EDITOR_PROFESSIONAL_TOOLS: EditorProfessionalTool[] = [
  { key: "select", label: "Selecionar", hotkey: "V", area: "texto" },
  { key: "move_text", label: "Mover texto", hotkey: "arrastar", area: "texto" },
  { key: "edit_text", label: "Editar texto", hotkey: "campo", area: "texto" },
  { key: "mask_brush", label: "Brush de mascara", hotkey: "M", area: "mascara" },
  { key: "eraser", label: "Borracha", hotkey: "E", area: "mascara" },
  { key: "region_reprocess", label: "Reprocessar regiao", hotkey: "painel", area: "pagina" },
  { key: "compare", label: "Comparar original/final", hotkey: "1/3", area: "pagina" },
];

export const EDITOR_PROFESSIONAL_SHORTCUTS = [
  "Ctrl+Z desfaz",
  "Ctrl+Shift+Z refaz",
  "Ctrl+S salva",
  "1/2/3 troca visualizacao",
  "Alt+setas troca pagina",
];

export function editorProfessionalReadiness(input: {
  hasLayersPanel: boolean;
  hasTextProperties: boolean;
  hasMaskTools: boolean;
  hasBeforeAfter: boolean;
  hasUndoRedo: boolean;
}) {
  const checks = [
    input.hasLayersPanel,
    input.hasTextProperties,
    input.hasMaskTools,
    input.hasBeforeAfter,
    input.hasUndoRedo,
  ];
  const passed = checks.filter(Boolean).length;
  return {
    passed,
    total: checks.length,
    ready: passed === checks.length,
  };
}
