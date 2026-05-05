/**
 * LayersPanel — Fase 10: BITMAP como Layers MVP
 *
 * Seções:
 *   1. BITMAP — drag reorder, opacity slider, eye toggle, lock toggle, thumbnail 32×32
 *   2. TEXTO — lista filtrada de TextEntry (igual ao anterior)
 */

import { useEffect, useMemo, useState } from "react";
import {
  Check,
  Eye,
  EyeOff,
  GripVertical,
  Image as ImageIcon,
  Layers,
  Lock,
  LockOpen,
  Search,
  Trash2,
  Wand2,
} from "lucide-react";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
  arrayMove,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { useEditorStore } from "../../lib/stores/editorStore";
import type { ImageLayerKey } from "../../lib/stores/appStore";
import { LayerItem } from "./LayerItem";

// Ordem padrão: paint/brush em cima de rendered em cima de inpaint em cima de base; mask no fundo
const DEFAULT_IMAGE_LAYER_ORDER: ImageLayerKey[] = ["brush", "rendered", "inpaint", "base", "mask"];

const IMAGE_LAYER_LABELS: Record<ImageLayerKey, string> = {
  base: "Original",
  mask: "Máscara",
  inpaint: "Inpaint",
  brush: "Pintura",
  rendered: "Render",
};

// Camadas técnicas — não exportadas no resultado final (ex: máscara)
const TECHNICAL_LAYERS: Set<ImageLayerKey> = new Set(["mask"]);

// Referência estável para fallback de image_layers vazio (evita loop por ref nova)
const EMPTY_IMAGE_LAYERS: Partial<Record<ImageLayerKey, never>> = Object.freeze({});

// ─── SortableImageLayerRow ───────────────────────────────────────────────────

interface SortableRowProps {
  layerKey: ImageLayerKey;
  visible: boolean;
  locked: boolean;
  opacity: number;
  selected: boolean;
  thumbnail: string | undefined;
  onSelect: () => void;
  onToggleVisibility: () => void;
  onToggleLock: () => void;
  onOpacityChange: (v: number) => void;
}

function SortableImageLayerRow({
  layerKey,
  visible,
  locked,
  opacity,
  selected,
  thumbnail,
  onSelect,
  onToggleVisibility,
  onToggleLock,
  onOpacityChange,
}: SortableRowProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: layerKey,
    disabled: TECHNICAL_LAYERS.has(layerKey),
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const isTechnical = TECHNICAL_LAYERS.has(layerKey);

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`flex items-center gap-1.5 rounded-lg border px-2 py-1.5 transition-smooth ${
        selected
          ? "border-accent-cyan/25 bg-accent-cyan/8"
          : "border-transparent hover:bg-white/[0.03]"
      }`}
    >
      {/* Drag handle — desabilitado para camadas técnicas */}
      <button
        {...(!isTechnical ? { ...attributes, ...listeners } : {})}
        className={`shrink-0 rounded p-0.5 transition-smooth ${
          isTechnical
            ? "cursor-default text-transparent"
            : "cursor-grab text-text-muted/30 hover:text-text-muted active:cursor-grabbing"
        }`}
        tabIndex={-1}
        aria-hidden
      >
        <GripVertical size={11} />
      </button>

      {/* Thumbnail */}
      <div
        className="shrink-0 w-8 h-8 rounded overflow-hidden border border-border/60 bg-white/[0.04] cursor-pointer"
        onClick={onSelect}
      >
        {thumbnail ? (
          <img src={thumbnail} alt={IMAGE_LAYER_LABELS[layerKey]} className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <ImageIcon size={10} className="text-text-muted/40" />
          </div>
        )}
      </div>

      {/* Label + opacity */}
      <div className="min-w-0 flex-1 cursor-pointer" onClick={onSelect}>
        <div className="flex items-center gap-1">
          <p className="text-[11px] font-medium text-text-primary">{IMAGE_LAYER_LABELS[layerKey]}</p>
          {isTechnical && (
            <span className="text-[8px] text-text-muted/40 font-mono uppercase tracking-wider">tec</span>
          )}
        </div>
        {/* Opacity slider */}
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={Math.round(opacity * 100)}
          onClick={(e) => e.stopPropagation()}
          onChange={(e) => onOpacityChange(Number(e.target.value) / 100)}
          className="w-full h-1 mt-0.5 appearance-none bg-border/40 rounded cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:h-2.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent-cyan"
          title={`Opacidade: ${Math.round(opacity * 100)}%`}
        />
      </div>

      {/* Controls */}
      <div className="flex shrink-0 items-center gap-0.5">
        <button
          type="button"
          className="rounded-md p-1 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
          onClick={(e) => { e.stopPropagation(); onToggleLock(); }}
          title={locked ? "Desbloquear camada" : "Bloquear camada"}
        >
          {locked ? <Lock size={11} /> : <LockOpen size={11} />}
        </button>
        <button
          type="button"
          className="rounded-md p-1 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
          onClick={(e) => { e.stopPropagation(); onToggleVisibility(); }}
          title={visible ? "Ocultar" : "Mostrar"}
        >
          {visible ? <Eye size={11} /> : <EyeOff size={11} />}
        </button>
      </div>
    </div>
  );
}

// ─── LayersPanel ─────────────────────────────────────────────────────────────

export function LayersPanel() {
  // Selectors individuais — evita re-render do panel inteiro quando qualquer
  // outra parte do store muda (ex: zoom, brushColor, etc.)
  const currentPage = useEditorStore((s) => s.currentPage);
  const selectedLayerId = useEditorStore((s) => s.selectedLayerId);
  const selectedImageLayerKey = useEditorStore((s) => s.selectedImageLayerKey);
  const pendingEdits = useEditorStore((s) => s.pendingEdits);
  const layerThumbnails = useEditorStore((s) => s.layerThumbnails);
  const toggleImageLayerVisibility = useEditorStore((s) => s.toggleImageLayerVisibility);
  const selectImageLayer = useEditorStore((s) => s.selectImageLayer);
  const deleteSelectedLayer = useEditorStore((s) => s.deleteSelectedLayer);
  const commitEdits = useEditorStore((s) => s.commitEdits);
  const setImageLayerOpacity = useEditorStore((s) => s.setImageLayerOpacity);
  const setImageLayerLocked = useEditorStore((s) => s.setImageLayerLocked);
  const reorderImageLayers = useEditorStore((s) => s.reorderImageLayers);
  const generateLayerThumbnail = useEditorStore((s) => s.generateLayerThumbnail);
  const bitmapLayerVersions = useEditorStore((s) => s.bitmapLayerVersions);

  const [query, setQuery] = useState("");

  const textLayers = currentPage?.text_layers ?? [];
  // Referência ESTÁVEL: usa o objeto real do store (mesma ref enquanto não muda)
  // ou o EMPTY_IMAGE_LAYERS pré-congelado. Sem `?? {}` que cria objeto novo a cada render.
  const imageLayers = (currentPage?.image_layers ?? EMPTY_IMAGE_LAYERS) as Partial<
    Record<ImageLayerKey, { path: string | null; visible: boolean; locked: boolean; opacity?: number; order?: number }>
  >;
  const hasPendingEdits = Object.keys(pendingEdits).length > 0;

  // Compute display order for bitmap layers
  const orderedBitmapKeys = useMemo<ImageLayerKey[]>(() => {
    const keys = DEFAULT_IMAGE_LAYER_ORDER.filter((k) => {
      // Sempre mostrar base/mask; mostrar outros somente se tiverem path
      if (k === "base" || k === "mask") return true;
      return !!imageLayers[k]?.path;
    });
    // Respeitar order numérico se disponível
    return [...keys].sort((a, b) => {
      const oa = imageLayers[a]?.order ?? DEFAULT_IMAGE_LAYER_ORDER.indexOf(a);
      const ob = imageLayers[b]?.order ?? DEFAULT_IMAGE_LAYER_ORDER.indexOf(b);
      return oa - ob;
    });
  }, [imageLayers]);

  // Gerar thumbnails APENAS para camadas que têm path mas não têm thumbnail.
  // Não inclui `layerThumbnails` nas deps — o guard previne loop e a inclusão
  // dispararia uma rerun a cada thumbnail gerado.
  useEffect(() => {
    for (const key of orderedBitmapKeys) {
      const layer = imageLayers[key];
      const hasThumb = !!useEditorStore.getState().layerThumbnails[key];
      if (layer?.path && !hasThumb) {
        void generateLayerThumbnail(key);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orderedBitmapKeys, bitmapLayerVersions]);

  const filteredTextLayers = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return textLayers;
    return textLayers.filter((entry) => {
      const haystack = `${entry.traduzido ?? entry.translated ?? ""} ${entry.original} ${entry.tipo}`.toLowerCase();
      return haystack.includes(normalized);
    });
  }, [query, textLayers]);

  // DnD sensors
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;

    const activeKey = active.id as ImageLayerKey;
    const overKey = over.id as ImageLayerKey;

    // Não reordenar camadas técnicas
    if (TECHNICAL_LAYERS.has(activeKey) || TECHNICAL_LAYERS.has(overKey)) return;

    const oldIndex = orderedBitmapKeys.indexOf(activeKey);
    const newIndex = orderedBitmapKeys.indexOf(overKey);
    if (oldIndex === -1 || newIndex === -1) return;

    const reordered = arrayMove(orderedBitmapKeys, oldIndex, newIndex);
    void reorderImageLayers(reordered);
  };

  return (
    <div className="flex h-full w-[340px] flex-col border-l border-border bg-bg-primary">
      {/* Header */}
      <div className="border-b border-border px-4 py-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers size={13} className="text-brand" />
            <span className="text-[13px] font-semibold tracking-tight">Camadas</span>
            <span className="rounded-full bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-text-muted">
              {filteredTextLayers.length}/{textLayers.length}
            </span>
          </div>

          <div className="flex items-center gap-0.5">
            <button
              onClick={() => void commitEdits()}
              disabled={!hasPendingEdits}
              className="rounded-md p-1.5 text-status-success transition-smooth hover:bg-status-success/8 disabled:opacity-25"
              title="Salvar alterações"
            >
              <Check size={13} />
            </button>
            <button
              onClick={() => void deleteSelectedLayer()}
              disabled={!selectedLayerId}
              className="rounded-md p-1.5 text-status-error transition-smooth hover:bg-status-error/8 disabled:opacity-25"
              title="Excluir camada"
            >
              <Trash2 size={13} />
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="relative mt-2">
          <Search size={12} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Buscar camada..."
            className="w-full rounded-lg border border-border bg-bg-tertiary/50 py-1.5 pl-7 pr-3 text-[11px] text-text-primary outline-none transition-smooth placeholder:text-text-muted focus:border-brand/30 focus:bg-bg-tertiary"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* ── Bitmap layers (drag & drop) ── */}
        <div className="px-3 py-2.5">
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted">
            <ImageIcon size={10} />
            Bitmap
          </div>

          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext items={orderedBitmapKeys} strategy={verticalListSortingStrategy}>
              <div className="space-y-0.5">
                {orderedBitmapKeys.map((key) => {
                  const layer = imageLayers[key];
                  const visible = layer?.visible ?? (key === "base");
                  const locked = layer?.locked ?? false;
                  const opacity = layer?.opacity ?? 1;
                  const selected = selectedImageLayerKey === key;
                  return (
                    <SortableImageLayerRow
                      key={key}
                      layerKey={key}
                      visible={visible}
                      locked={locked}
                      opacity={opacity}
                      selected={selected}
                      thumbnail={layerThumbnails[key]}
                      onSelect={() => selectImageLayer(key)}
                      onToggleVisibility={() => void toggleImageLayerVisibility(key)}
                      onToggleLock={() => void setImageLayerLocked(key, !locked)}
                      onOpacityChange={(v) => void setImageLayerOpacity(key, v)}
                    />
                  );
                })}
              </div>
            </SortableContext>
          </DndContext>
        </div>

        {/* ── Text layers ── */}
        <div className="border-t border-border px-3 py-2.5">
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted">
            <Wand2 size={10} />
            Texto
          </div>
          {filteredTextLayers.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border bg-bg-tertiary/30 px-4 py-5 text-center text-[11px] text-text-muted">
              {textLayers.length === 0 ? "Nenhuma camada de texto" : "Sem resultados"}
            </div>
          ) : (
            <div className="space-y-0.5">
              {filteredTextLayers.map((entry, index) => (
                <LayerItem key={entry.id} entry={entry} index={index + 1} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
