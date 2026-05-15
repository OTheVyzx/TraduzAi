/**
 * ToolSidebar — Fase 4 do refactor.
 *
 * Sidebar vertical (~48px) posicionada entre PageThumbnails e o canvas.
 * Substitui o segmented horizontal de TOOL_MODES na toolbar principal.
 *
 * Ferramentas:
 *  V  Selecionar
 *  H  Mover/Pan     (placeholder — Space+drag no canvas)
 *  T  Novo bloco    (block)
 *  B  Brush
 *  E  Borracha
 *  L  Máscara Lasso (freehand/poligonal — Fase 8)
 */

import {
  MousePointer2,
  Hand,
  PenTool,
  Brush,
  RotateCcw,
  Sparkles,
  Eraser,
  Scissors,
} from "lucide-react";
import { useEditorStore, type EditorToolMode } from "../../../lib/stores/editorStore";

const TOOLS: {
  key: EditorToolMode;
  label: string;
  icon: typeof MousePointer2;
  hotkey: string;
  title: string;
}[] = [
  { key: "select", label: "V", icon: MousePointer2, hotkey: "V", title: "Selecionar (V)" },
  { key: "block", label: "T", icon: PenTool, hotkey: "T", title: "Novo bloco de texto (T)" },
  { key: "brush", label: "B", icon: Brush, hotkey: "B", title: "Brush (B)" },
  { key: "repairBrush", label: "R", icon: RotateCcw, hotkey: "R", title: "Pincel de recuperação (R)" },
  { key: "reinpaintBrush", label: "I", icon: Sparkles, hotkey: "I", title: "Pincel corretor: corrigir área pintada (I)" },
  { key: "eraser", label: "E", icon: Eraser, hotkey: "E", title: "Borracha (E)" },
  { key: "mask", label: "L", icon: Scissors, hotkey: "L", title: "Máscara Lasso (L)" },
];

// Ferramenta "Hand/Pan" — mapeada para select até termos o modo pan dedicado (Fase 7)
const PAN_TOOL = { key: "select" as EditorToolMode, label: "H", icon: Hand, hotkey: "H", title: "Mover/Pan (H) — atalho Space no canvas" };

const ALL_TOOLS = [TOOLS[0], PAN_TOOL, ...TOOLS.slice(1)];

export function ToolSidebar() {
  const toolMode = useEditorStore((s) => s.toolMode);
  const setToolMode = useEditorStore((s) => s.setToolMode);

  return (
    <div className="flex flex-col items-center gap-1 border-r border-border bg-bg-secondary/60 px-1 py-2 w-[44px] shrink-0">
      {ALL_TOOLS.map(({ key, icon: Icon, hotkey, title }) => {
        // Para o PAN_TOOL, nunca mostra ativo (pois key='select' e select já cobre)
        const isActive = toolMode === key && !(hotkey === "H");
        return (
          <button
            key={hotkey}
            onClick={() => setToolMode(key)}
            title={title}
            className={`relative flex h-8 w-8 items-center justify-center rounded-lg transition-smooth ${
              isActive
                ? "bg-accent-cyan/15 text-accent-cyan"
                : "text-text-muted hover:bg-white/[0.04] hover:text-text-primary"
            }`}
          >
            <Icon size={15} />
            {/* Hotkey badge — tiny letter in bottom-right corner */}
            <span className="absolute bottom-0.5 right-0.5 text-[8px] font-bold leading-none text-text-muted/50">
              {hotkey}
            </span>
          </button>
        );
      })}

    </div>
  );
}
