/**
 * UndoRedoControls — Fase 11 do refactor.
 *
 * Dois botões pequenos na topbar com Ctrl+Z / Ctrl+Y.
 * A tooltip mostra a descrição da ação no topo da stack.
 */

import { Redo2, Undo2 } from "lucide-react";
import { useEditorStore } from "../../../lib/stores/editorStore";

function getTopAction(stack: ReturnType<typeof useEditorStore.getState>["historyByPageKey"][string] | undefined, direction: "undo" | "redo") {
  if (!stack) return null;
  const { commands, index } = stack;
  if (direction === "undo") {
    // O comando que será desfeito é o último executado (index - 1)
    const cmd = commands[index - 1];
    return cmd ?? null;
  } else {
    // O comando que será refeito é o próximo na fila (index)
    const cmd = commands[index];
    return cmd ?? null;
  }
}

function labelForAction(cmd: ReturnType<typeof getTopAction>): string {
  if (!cmd) return "";
  switch (cmd.type) {
    case "edit-traduzido": return "Editar texto";
    case "edit-estilo": return "Editar estilo";
    case "edit-bbox": return "Mover balão";
    case "create-layer": return "Criar camada";
    case "delete-layer": return "Excluir camada";
    case "reorder-layers": return "Reordenar camadas";
    case "bitmap-stroke":
      if (cmd.layerKey === "inpaint") return "Recuperacao";
      if (cmd.layerKey === "mask") return "Mascara";
      return "Pincel";
    case "toggle-visibility": return "Alternar visibilidade";
    case "toggle-lock": return "Alternar bloqueio";
    case "batch": return cmd.label;
    default: return "Ação";
  }
}

export function UndoRedoControls() {
  const undoEditor = useEditorStore((s) => s.undoEditor);
  const redoEditor = useEditorStore((s) => s.redoEditor);
  const currentPageKey = useEditorStore((s) => s.currentPageKey());
  const historyByPageKey = useEditorStore((s) => s.historyByPageKey);

  const stack = historyByPageKey[currentPageKey];
  const canUndo = stack ? stack.index > 0 : false;
  const canRedo = stack ? stack.index < stack.commands.length : false;

  const undoAction = getTopAction(stack, "undo");
  const redoAction = getTopAction(stack, "redo");

  return (
    <div className="flex items-center gap-0.5">
      <button
        onClick={() => undoEditor()}
        disabled={!canUndo}
        title={canUndo ? `Desfazer: ${labelForAction(undoAction)} (Ctrl+Z)` : "Nada para desfazer"}
        className="rounded-md p-1.5 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-25"
      >
        <Undo2 size={13} />
      </button>
      <button
        onClick={() => redoEditor()}
        disabled={!canRedo}
        title={canRedo ? `Refazer: ${labelForAction(redoAction)} (Ctrl+Y)` : "Nada para refazer"}
        className="rounded-md p-1.5 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-25"
      >
        <Redo2 size={13} />
      </button>
    </div>
  );
}
