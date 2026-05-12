/**
 * FloatingTextEditor — Fase 5 do refactor.
 *
 * Painel flutuante próximo ao balão selecionado. Mostra Original (readonly)
 * e Tradução (editável) sem precisar abrir o painel lateral PROPRIEDADES.
 *
 * Posicionamento:
 *  - Calcula a posição da borda direita superior do bbox no espaço de tela
 *  - Preferencialmente renderiza à direita do balão; se não couber, à esquerda;
 *    se ainda não couber, abaixo.
 *  - Clampado na viewport com margem de 12px.
 *
 * Props:
 *  - stageScale: escala atual do canvas Konva
 *  - panOffset: deslocamento atual do pan
 *  - imageWidth / imageHeight: dimensões naturais da imagem
 *  - containerSize: dimensões do container do stage
 */

import { useEffect, useRef, useCallback } from "react";
import { RotateCcw, X } from "lucide-react";
import { useEditorStore } from "../../../lib/stores/editorStore";
import { useTextEditSession } from "../../../lib/useTextEditSession";

interface FloatingTextEditorProps {
  stageScale: number;
  panOffset: { x: number; y: number };
  imageWidth: number;
  imageHeight: number;
  containerSize: { width: number; height: number };
}

const PANEL_WIDTH = 260;
const PANEL_MARGIN = 12;

/**
 * Converte ponto de imagem → posição dentro do container.
 * O Stage é centralizado no container via flex; o panOffset é relativo ao centro.
 */
function imageToContainer(
  ix: number,
  iy: number,
  stageScale: number,
  panOffset: { x: number; y: number },
  imageWidth: number,
  imageHeight: number,
  containerSize: { width: number; height: number },
) {
  const cx = containerSize.width / 2;
  const cy = containerSize.height / 2;
  const x = cx + panOffset.x + (ix - imageWidth / 2) * stageScale;
  const y = cy + panOffset.y + (iy - imageHeight / 2) * stageScale;
  return { x, y };
}

export function FloatingTextEditor({
  stageScale,
  panOffset,
  imageWidth,
  imageHeight,
  containerSize,
}: FloatingTextEditorProps) {
  const selectedLayerId = useEditorStore((s) => s.selectedLayerId);
  const currentPage = useEditorStore((s) => s.currentPage);
  const updateEdit = useEditorStore((s) => s.updatePendingEdit);
  const selectLayer = useEditorStore((s) => s.selectLayer);

  const panelRef = useRef<HTMLDivElement>(null);

  const entry = currentPage?.text_layers.find((t) => t.id === selectedLayerId);
  const original = entry?.original ?? "";

  // Fase 11: usa session de edição para gravar undo/redo automaticamente
  const textSession = useTextEditSession(selectedLayerId ?? "");
  const traduzido = textSession.value;

  // Fechar ao clicar fora do painel
  useEffect(() => {
    if (!selectedLayerId) return;
    function onMouseDown(e: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        // Não fechar — clique no canvas fica com EditorStage. O ESC fecha.
        // Mantemos o painel aberto para o usuário continuar editando após clicar
        // fora sem intenção de fechar (ex: clicar num botão da toolbar).
      }
    }
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [selectedLayerId]);

  // ESC fecha (deselect layer)
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && selectedLayerId) {
        // Só fecha se o foco está dentro do painel
        if (panelRef.current?.contains(document.activeElement)) {
          selectLayer(null);
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedLayerId, selectLayer]);

  const handleRestore = useCallback(() => {
    if (!selectedLayerId || !entry) return;
    // Registra undo: mudança de volta para o original
    textSession.onFocus();
    updateEdit(selectedLayerId, {
      traduzido: entry.original,
      translated: entry.original,
    });
    textSession.onBlur();
  }, [selectedLayerId, entry, updateEdit, textSession]);

  if (!entry || !selectedLayerId || containerSize.width === 0 || imageWidth === 0) {
    return null;
  }

  const [x1, y1, , y2] = entry.layout_bbox ?? entry.bbox;
  const x2 = (entry.layout_bbox ?? entry.bbox)[2];
  const bboxH = (y2 - y1) * stageScale;

  // Ponto de ancoragem: topo direito do bbox
  const anchorRight = imageToContainer(x2, y1, stageScale, panOffset, imageWidth, imageHeight, containerSize);
  // Ponto de ancoragem: topo esquerdo do bbox
  const anchorLeft = imageToContainer(x1, y1, stageScale, panOffset, imageWidth, imageHeight, containerSize);

  // Tenta posicionar à direita
  let left: number;
  let top = anchorRight.y;

  if (anchorRight.x + PANEL_MARGIN + PANEL_WIDTH <= containerSize.width - PANEL_MARGIN) {
    // À direita do balão
    left = anchorRight.x + PANEL_MARGIN;
  } else if (anchorLeft.x - PANEL_MARGIN - PANEL_WIDTH >= PANEL_MARGIN) {
    // À esquerda do balão
    left = anchorLeft.x - PANEL_MARGIN - PANEL_WIDTH;
  } else {
    // Abaixo do balão
    left = Math.max(PANEL_MARGIN, Math.min(anchorRight.x - PANEL_WIDTH / 2, containerSize.width - PANEL_WIDTH - PANEL_MARGIN));
    top = anchorRight.y + bboxH + PANEL_MARGIN;
  }

  // Clamp vertical
  top = Math.max(PANEL_MARGIN, Math.min(top, containerSize.height - PANEL_MARGIN - 200));

  const confidencePercent = Math.round(
    ((entry.confianca_ocr ?? entry.ocr_confidence ?? 0) || 0) * 100,
  );
  const confidenceColor =
    confidencePercent >= 80
      ? "text-status-success"
      : confidencePercent >= 50
        ? "text-status-warning"
        : "text-status-error";

  return (
    <div
      className="pointer-events-none absolute inset-0 z-40"
      aria-hidden={!selectedLayerId}
    >
      <div
        ref={panelRef}
        className="pointer-events-auto absolute rounded-xl border border-border bg-bg-secondary/95 shadow-[0_8px_32px_rgba(0,0,0,0.45)] backdrop-blur-md"
        style={{ left, top, width: PANEL_WIDTH }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-border">
          <div className="flex items-center gap-2">
            <span className="rounded-md border border-border bg-bg-tertiary/50 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
              {entry.tipo}
            </span>
            <span className={`font-mono text-[10px] font-medium ${confidenceColor}`}>
              {confidencePercent}%
            </span>
          </div>
          <button
            onClick={() => selectLayer(null)}
            className="rounded-md p-0.5 text-text-muted hover:bg-white/[0.06] hover:text-text-primary transition-smooth"
            title="Fechar (Esc)"
          >
            <X size={12} />
          </button>
        </div>

        <div className="p-3 space-y-2.5">
          {/* Original (readonly) */}
          <div>
            <label className="block text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted/70 mb-1">
              Original
            </label>
            <div className="rounded-lg border border-border bg-bg-tertiary/30 px-2.5 py-2 text-[11px] text-text-muted font-mono leading-relaxed max-h-20 overflow-y-auto">
              {original || <span className="italic opacity-40">— vazio —</span>}
            </div>
          </div>

          {/* Tradução (editable) */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted/70">
                Tradução
              </label>
              <button
                onClick={handleRestore}
                className="flex items-center gap-0.5 rounded-md px-1.5 py-0.5 text-[9px] text-text-muted hover:bg-white/[0.06] hover:text-text-primary transition-smooth"
                title="Restaurar texto original"
              >
                <RotateCcw size={9} />
                Restaurar
              </button>
            </div>
            <textarea
              value={traduzido}
              autoFocus
              rows={4}
              placeholder="Tradução..."
              onFocus={textSession.onFocus}
              onBlur={textSession.onBlur}
              onCompositionStart={textSession.onCompositionStart}
              onCompositionEnd={textSession.onCompositionEnd}
              onChange={(e) => textSession.onChange(e.target.value)}
              onKeyDown={(e) => {
                // Ctrl+Enter confirma e fecha
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  textSession.onBlur();
                  selectLayer(null);
                }
                // Impede ESC de se propagar para o handler global quando foco aqui
                if (e.key === "Escape") {
                  e.stopPropagation();
                  textSession.onBlur();
                  selectLayer(null);
                }
              }}
              className="w-full resize-none rounded-lg border border-border bg-bg-tertiary/60 px-2.5 py-2 text-[11px] text-text-primary leading-relaxed focus:border-brand/40 focus:outline-none transition-smooth"
            />
          </div>
        </div>

        {/* Footer hint */}
        <div className="border-t border-border px-3 py-1.5">
          <p className="text-[9px] text-text-muted/50">
            Ctrl+Enter fecha · ESC cancela
          </p>
        </div>
      </div>
    </div>
  );
}
