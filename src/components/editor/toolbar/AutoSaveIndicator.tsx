import { useEffect, useState } from "react";
import { Loader2, Check, AlertCircle, Clock } from "lucide-react";
import { useEditorStore } from "../../../lib/stores/editorStore";

/**
 * Indicador visual do auto-save (Fase 3 do refactor).
 *
 * Substitui o botão "Salvar" antigo. Mostra:
 *  - "Salvando…"        durante runAutoSave / flushAutoSave
 *  - "Salvo agora"       no segundo após sucesso
 *  - "Salvo há 12s"      contagem regressiva live
 *  - "Alterações pendentes" quando dirty=true mas auto-save ainda não rodou
 *  - "Erro ao salvar"    com retry inline
 */
export function AutoSaveIndicator() {
  const status = useEditorStore((s) => s.autoSaveStatus);
  const lastSavedAt = useEditorStore((s) => s.lastSavedAt);
  const lastSaveError = useEditorStore((s) => s.lastSaveError);
  const dirty = useEditorStore((s) => s.dirty);
  const runAutoSave = useEditorStore((s) => s.runAutoSave);

  // Atualiza o "há Xs" a cada segundo enquanto status === 'saved'.
  const [, forceTick] = useState(0);
  useEffect(() => {
    if (status !== "saved") return;
    const id = window.setInterval(() => forceTick((n) => n + 1), 1000);
    return () => window.clearInterval(id);
  }, [status]);

  if (status === "idle" && !dirty) {
    return null; // não polui a UI quando não há nada a comunicar
  }

  if (status === "saving") {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-border bg-bg-tertiary/30 px-2 py-1 text-[10px] text-text-muted">
        <Loader2 size={11} className="animate-spin" />
        Salvando…
      </div>
    );
  }

  if (status === "error") {
    return (
      <div
        className="flex items-center gap-1.5 rounded-lg border border-status-error/30 bg-status-error/10 px-2 py-1 text-[10px] text-status-error"
        title={lastSaveError ?? "Erro ao salvar"}
      >
        <AlertCircle size={11} />
        Erro ao salvar
        <button
          onClick={() => void runAutoSave()}
          className="ml-1 rounded px-1.5 py-0.5 text-[10px] hover:bg-status-error/20"
        >
          Retry
        </button>
      </div>
    );
  }

  if (status === "saved" && lastSavedAt !== null) {
    const secondsAgo = Math.max(0, Math.floor((Date.now() - lastSavedAt) / 1000));
    const label = secondsAgo < 2 ? "Salvo agora" : `Salvo há ${secondsAgo}s`;
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-status-success/25 bg-status-success/10 px-2 py-1 text-[10px] text-status-success">
        <Check size={11} />
        {label}
      </div>
    );
  }

  // pending / dirty mas ainda não disparou
  return (
    <div
      className="flex items-center gap-1.5 rounded-lg border border-status-warning/25 bg-status-warning/10 px-2 py-1 text-[10px] text-status-warning"
      title="Alterações ainda não salvas — auto-save em até 3s"
    >
      <Clock size={11} />
      Alterações pendentes
    </div>
  );
}
