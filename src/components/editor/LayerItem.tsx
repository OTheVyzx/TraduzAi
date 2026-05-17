import { useEffect, useRef, useState, type MouseEvent } from "react";
import {
  Eye,
  EyeOff,
  Lock,
  LockOpen,
  MessageSquare,
  BookOpen,
  Zap,
  Cloud,
  ScanText,
  Languages,
  Eraser,
} from "lucide-react";
import { useEditorStore } from "../../lib/stores/editorStore";
import type { NormalizedTextLayer } from "../../lib/editorScene";

const TIPO_ICONS = {
  fala: MessageSquare,
  narracao: BookOpen,
  sfx: Zap,
  pensamento: Cloud,
} as const;

interface LayerItemProps {
  entry: NormalizedTextLayer;
  index: number;
  hasEdits?: boolean;
}

export function LayerItem({ entry, index, hasEdits = false }: LayerItemProps) {
  const rowRef = useRef<HTMLDivElement>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<"ocr" | "translate" | "inpaint" | null>(null);
  const { selectedLayerId, hoveredLayerId } = useEditorStore();
  const selectLayer = useEditorStore((s) => s.selectLayer);
  const hoverLayer = useEditorStore((s) => s.hoverLayer);
  const toggleVisibility = useEditorStore((s) => s.toggleTextLayerVisibility);
  const toggleLock = useEditorStore((s) => s.toggleTextLayerLock);
  const reProcessBlock = useEditorStore((s) => s.reProcessBlock);

  const isSelected = selectedLayerId === entry.id;
  const isHovered = hoveredLayerId === entry.id;
  const isHidden = !entry.visible;
  const isLocked = entry.locked;

  useEffect(() => {
    if (!isSelected || !rowRef.current) return;
    rowRef.current.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [isSelected]);

  useEffect(() => {
    setActionError(null);
    setPendingAction(null);
  }, [entry.id]);

  const displayText = entry.displayText || entry.displayOriginal;

  const TipoIcon = TIPO_ICONS[entry.tipo] ?? MessageSquare;

  const confidenceColor =
    entry.confidencePercent >= 80
      ? "bg-status-success"
      : entry.confidencePercent >= 50
        ? "bg-status-warning"
        : "bg-status-error";

  const handleBlockAction = async (
    event: MouseEvent<HTMLButtonElement>,
    mode: "ocr" | "translate" | "inpaint",
  ) => {
    event.stopPropagation();
    selectLayer(entry.id);
    setActionError(null);
    setPendingAction(mode);
    try {
      await reProcessBlock(mode);
    } catch (error) {
      console.error("Falha ao reprocessar bloco de texto", error);
      setActionError("Nao foi possivel executar a acao neste texto.");
    } finally {
      setPendingAction(null);
    }
  };

  return (
    <div
      ref={rowRef}
      className={`border-l-2 px-3 py-2.5 transition-smooth ${
        isSelected
          ? "border-brand bg-brand/5"
          : isHovered
            ? "border-brand/30 bg-bg-hover"
            : "border-transparent"
      }`}
      onClick={() => selectLayer(entry.id)}
      onMouseEnter={() => hoverLayer(entry.id)}
      onMouseLeave={() => hoverLayer(null)}
    >
      <div className="flex cursor-pointer items-start gap-2">
        <button
          className={`mt-0.5 flex-shrink-0 p-0.5 transition-smooth ${
            isHidden ? "text-text-muted" : "text-text-secondary hover:text-text-primary"
          }`}
          onClick={(event) => {
            event.stopPropagation();
            void toggleVisibility(entry.id);
          }}
          title={isHidden ? "Mostrar camada de texto" : "Ocultar camada de texto"}
        >
          {isHidden ? <EyeOff size={14} /> : <Eye size={14} />}
        </button>
        <button
          className={`mt-0.5 flex-shrink-0 p-0.5 transition-smooth ${
            isLocked ? "text-status-warning" : "text-text-muted hover:text-text-primary"
          }`}
          onClick={(event) => {
            event.stopPropagation();
            toggleLock(entry.id);
          }}
          title={isLocked ? "Desbloquear camada de texto" : "Bloquear camada de texto"}
        >
          {isLocked ? <Lock size={14} /> : <LockOpen size={14} />}
        </button>

        <div className="flex min-w-0 flex-1 gap-2">
          <div className="mt-0.5 flex shrink-0 items-center gap-1">
            <span className="rounded bg-bg-tertiary px-1.5 py-0.5 text-[10px] font-mono text-text-muted">
              {index}
            </span>
            <TipoIcon size={14} className="text-text-muted" />
          </div>

          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="rounded-full border border-border px-1.5 py-0.5 text-[10px] uppercase tracking-[0.14em] text-text-muted">
                {entry.tipo}
              </span>
              {hasEdits && (
                <span className="rounded-full bg-brand/12 px-1.5 py-0.5 text-[10px] text-brand">
                  editado
                </span>
              )}
            </div>
            <p
              className={`mt-1 line-clamp-2 text-sm ${
                isHidden ? "text-text-muted line-through" : "text-text-primary"
              }`}
            >
              {displayText.trim() || "(vazio)"}
            </p>
            <div className="mt-1.5 flex items-center justify-between text-[10px] text-text-muted">
              <span className="truncate">{entry.effectiveBbox.join(", ")}</span>
              <span className="flex items-center gap-1">
                <span className={`h-1.5 w-1.5 rounded-full ${confidenceColor}`} />
                {entry.confidencePercent}%
              </span>
            </div>
            <div className="mt-2 flex items-center gap-1">
              <button
                disabled={pendingAction !== null}
                className="inline-flex items-center gap-1 rounded border border-border bg-bg-tertiary/45 px-1.5 py-1 text-[10px] text-text-secondary transition-smooth hover:border-brand/30 hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-50"
                onClick={(event) => handleBlockAction(event, "ocr")}
                title="Refazer OCR deste texto"
              >
                <ScanText size={11} />
                OCR
              </button>
              <button
                disabled={pendingAction !== null}
                className="inline-flex items-center gap-1 rounded border border-border bg-bg-tertiary/45 px-1.5 py-1 text-[10px] text-text-secondary transition-smooth hover:border-brand/30 hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-50"
                onClick={(event) => handleBlockAction(event, "translate")}
                title="Traduzir este texto"
              >
                <Languages size={11} />
                Traduzir
              </button>
              <button
                disabled={pendingAction !== null}
                className="inline-flex items-center gap-1 rounded border border-border bg-bg-tertiary/45 px-1.5 py-1 text-[10px] text-text-secondary transition-smooth hover:border-brand/30 hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-50"
                onClick={(event) => handleBlockAction(event, "inpaint")}
                title="Limpar fundo deste texto"
              >
                <Eraser size={11} />
                Limpar
              </button>
            </div>
            {actionError && (
              <p className="mt-1 text-[10px] leading-tight text-status-error">{actionError}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
