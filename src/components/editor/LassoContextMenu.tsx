import { Search, ScanText, Languages, Sparkles, ClipboardCheck, X } from "lucide-react";
import { useEditorStore } from "../../lib/stores/editorStore";

type PageAction = "detect" | "detect_boxes" | "ocr" | "translate" | "inpaint";

const ACTIONS: { action: PageAction; label: string; icon: typeof Search }[] = [
  { action: "detect", label: "Detectar area", icon: Search },
  { action: "detect_boxes", label: "Caixas area", icon: Search },
  { action: "ocr", label: "OCR area", icon: ScanText },
  { action: "translate", label: "Traduzir area", icon: Languages },
  { action: "inpaint", label: "Inpaint area", icon: Sparkles },
];

export function LassoContextMenu({
  x,
  y,
  position = "fixed",
  onClose,
}: {
  x: number;
  y: number;
  position?: "fixed" | "absolute";
  onClose: () => void;
}) {
  const activePageAction = useEditorStore((s) => s.activePageAction);
  const runMaskedActionFromLasso = useEditorStore((s) => s.runMaskedActionFromLasso);
  const applyLassoSelectionToMask = useEditorStore((s) => s.applyLassoSelectionToMask);
  const setActiveLassoSelection = useEditorStore((s) => s.setActiveLassoSelection);
  const disabled = activePageAction !== null;

  return (
    <div
      data-testid="lasso-context-menu"
      className={`${position} z-[60] min-w-[180px] rounded-lg border border-border bg-bg-secondary p-1.5 shadow-[0_16px_40px_rgba(0,0,0,0.45)]`}
      style={{ left: x, top: y }}
      onMouseDownCapture={(event) => event.stopPropagation()}
      onMouseDown={(event) => event.stopPropagation()}
      onContextMenu={(event) => event.preventDefault()}
    >
      {ACTIONS.map(({ action, label, icon: Icon }) => (
        <button
          key={action}
          disabled={disabled}
          onClick={() => {
            onClose();
            void runMaskedActionFromLasso(action);
          }}
          title={label}
          className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-[11px] text-text-primary hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Icon size={13} />
          {label}
        </button>
      ))}
      <div className="my-1 h-px bg-border" />
      <button
        disabled={disabled}
        onClick={() => {
          onClose();
          void applyLassoSelectionToMask();
        }}
        className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-[11px] text-text-primary hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-50"
      >
        <ClipboardCheck size={13} />
        Aplicar a máscara
      </button>
      <button
        onClick={() => {
          setActiveLassoSelection(null);
          onClose();
        }}
        className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-[11px] text-text-muted hover:bg-white/[0.06] hover:text-text-primary"
      >
        <X size={13} />
        Cancelar seleção
      </button>
    </div>
  );
}
