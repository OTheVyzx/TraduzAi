import { Minus, Plus, Maximize2, LocateFixed } from "lucide-react";
import { useEditorStore } from "../../../lib/stores/editorStore";

/**
 * Controles de zoom compactos para a toolbar superior do Editor.
 *
 * Antes ficavam absolute no canto inferior direito do canvas
 * (EditorStage). Movidos pra toolbar (Fase 1 do refactor) para liberar a
 * área visual do canvas e ficar próximos das ações de pipeline.
 */
export function ZoomControls() {
  const { zoom, zoomIn, zoomOut, resetViewport, setPan } = useEditorStore();

  return (
    <div className="flex items-center gap-0.5 rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
      <button
        onClick={zoomOut}
        className="rounded-md p-1 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
        title="Diminuir zoom (-)"
      >
        <Minus size={12} />
      </button>
      <span className="min-w-[34px] text-center font-mono text-[10px] text-text-secondary">
        {Math.round(zoom * 100)}%
      </span>
      <button
        onClick={zoomIn}
        className="rounded-md p-1 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
        title="Aumentar zoom (+)"
      >
        <Plus size={12} />
      </button>
      <div className="mx-0.5 h-3 w-px bg-border" />
      <button
        onClick={resetViewport}
        className="rounded-md p-1 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
        title="Ajustar à tela (0)"
      >
        <Maximize2 size={11} />
      </button>
      <button
        onClick={() => setPan({ x: 0, y: 0 })}
        className="rounded-md p-1 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
        title="Centralizar"
      >
        <LocateFixed size={11} />
      </button>
    </div>
  );
}
