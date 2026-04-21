import { useEffect, useRef } from "react";
import {
  Eye,
  EyeOff,
  MessageSquare,
  BookOpen,
  Zap,
  Cloud,
} from "lucide-react";
import { useEditorStore } from "../../lib/stores/editorStore";
import type { TextEntry } from "../../lib/stores/appStore";

const TIPO_ICONS = {
  fala: MessageSquare,
  narracao: BookOpen,
  sfx: Zap,
  pensamento: Cloud,
} as const;

interface LayerItemProps {
  entry: TextEntry;
  index: number;
}

export function LayerItem({ entry, index }: LayerItemProps) {
  const rowRef = useRef<HTMLDivElement>(null);
  const { selectedLayerId, hoveredLayerId, pendingEdits } = useEditorStore();
  const selectLayer = useEditorStore((s) => s.selectLayer);
  const hoverLayer = useEditorStore((s) => s.hoverLayer);
  const toggleVisibility = useEditorStore((s) => s.toggleTextLayerVisibility);

  const isSelected = selectedLayerId === entry.id;
  const isHovered = hoveredLayerId === entry.id;
  const isHidden = entry.visible === false;

  useEffect(() => {
    if (!isSelected || !rowRef.current) return;
    rowRef.current.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [isSelected]);

  const edit = pendingEdits[entry.id];
  const displayText = edit?.traduzido ?? edit?.translated ?? entry.traduzido ?? entry.translated ?? entry.original;
  const hasEdits = !!edit;

  const TipoIcon = TIPO_ICONS[entry.tipo] ?? MessageSquare;

  const confidence = entry.confianca_ocr ?? entry.ocr_confidence ?? 0;
  const confidenceColor =
    confidence >= 0.8
      ? "bg-status-success"
      : confidence >= 0.5
        ? "bg-status-warning"
        : "bg-status-error";

  return (
    <div
      ref={rowRef}
      className={`border-l-2 px-3 py-2.5 transition-smooth ${
        isSelected
          ? "border-accent-purple bg-accent-purple/5"
          : isHovered
            ? "border-accent-purple/30 bg-bg-hover"
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
        >
          {isHidden ? <EyeOff size={14} /> : <Eye size={14} />}
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
              <span className="rounded-full border border-white/10 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.14em] text-text-muted">
                {entry.tipo}
              </span>
              {hasEdits && (
                <span className="rounded-full bg-accent-purple/12 px-1.5 py-0.5 text-[10px] text-accent-purple">
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
              <span className="truncate">{(edit?.bbox ?? entry.bbox).join(", ")}</span>
              <span className="flex items-center gap-1">
                <span className={`h-1.5 w-1.5 rounded-full ${confidenceColor}`} />
                {Math.round(confidence * 100)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
