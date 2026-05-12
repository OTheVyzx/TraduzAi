/**
 * RenderStatusBadge — Fase 6 do refactor.
 *
 * Indicador visual do Auto Fidelity Render (renderização FT2Font em background).
 *
 * Estados:
 *  - "rendering"  → 🔄 Renderizando fiel…
 *  - "updated"    → ✓ Fiel atualizado
 *  - "stale"      → ⚠ Render desatualizado (clique força render imediato)
 *  - "error"      → ✗ Erro no render fiel
 *  - "idle"       → null (não aparece)
 */

import { useEffect, useState } from "react";
import { Loader2, Check, AlertCircle, RefreshCw } from "lucide-react";
import { useEditorStore } from "../../../lib/stores/editorStore";

export function RenderStatusBadge() {
  const status = useEditorStore((s) => s.renderStatus);
  const renderError = useEditorStore((s) => s.renderError);
  const forceFidelityRender = useEditorStore((s) => s.forceFidelityRender);

  // Oculta "Fiel atualizado" após 4 segundos
  const [showUpdated, setShowUpdated] = useState(false);
  useEffect(() => {
    if (status === "updated") {
      setShowUpdated(true);
      const id = window.setTimeout(() => setShowUpdated(false), 4000);
      return () => window.clearTimeout(id);
    } else {
      setShowUpdated(false);
    }
  }, [status]);

  if (status === "idle") return null;

  if (status === "rendering") {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-brand/20 bg-brand/8 px-2 py-1 text-[10px] text-brand">
        <Loader2 size={10} className="animate-spin" />
        Renderizando fiel…
      </div>
    );
  }

  if (status === "updated" && showUpdated) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-status-success/25 bg-status-success/10 px-2 py-1 text-[10px] text-status-success">
        <Check size={10} />
        Fiel atualizado
      </div>
    );
  }

  if (status === "stale") {
    return (
      <button
        onClick={() => void forceFidelityRender()}
        className="flex items-center gap-1.5 rounded-lg border border-status-warning/25 bg-status-warning/10 px-2 py-1 text-[10px] text-status-warning hover:bg-status-warning/15 transition-smooth"
        title="Render desatualizado — clique para forçar"
      >
        <RefreshCw size={10} />
        Render desatualizado
      </button>
    );
  }

  if (status === "error") {
    return (
      <button
        onClick={() => void forceFidelityRender()}
        className="flex items-center gap-1.5 rounded-lg border border-status-error/30 bg-status-error/10 px-2 py-1 text-[10px] text-status-error hover:bg-status-error/15 transition-smooth"
        title={renderError ?? "Erro no render fiel — clique para tentar novamente"}
      >
        <AlertCircle size={10} />
        Erro no render fiel
      </button>
    );
  }

  return null;
}
