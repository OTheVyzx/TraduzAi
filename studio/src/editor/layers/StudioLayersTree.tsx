import { useEffect, useMemo, useState, type DragEvent, type MouseEvent } from "react";
import {
  AlertTriangle,
  Eye,
  EyeOff,
  FileImage,
  Folder,
  FolderPlus,
  GripVertical,
  Layers3,
  Loader2,
  Lock,
  LockOpen,
  PaintBucket,
  Redo2,
  ScanLine,
  SlidersHorizontal,
  Sparkles,
  Type,
  Undo2,
  type LucideIcon,
} from "lucide-react";
import type { StudioSceneNode, StudioSceneNodeKind } from "../../project/studioProject";
import { orderedSceneChildren, useStudioSceneStore } from "../../store/studioSceneStore";

const BLEND_MODES = [
  { value: "normal", label: "Normal" },
  { value: "multiply", label: "Multiplicar" },
  { value: "screen", label: "Tela" },
  { value: "overlay", label: "Sobrepor" },
  { value: "darken", label: "Escurecer" },
  { value: "lighten", label: "Clarear" },
  { value: "color-dodge", label: "Subexposição de cor" },
  { value: "color-burn", label: "Superexposição de cor" },
] as const;

const NODE_PRESENTATION: Record<StudioSceneNodeKind, { label: string; icon: LucideIcon; color: string }> = {
  raster: { label: "Raster", icon: FileImage, color: "text-accent-cyan" },
  text: { label: "Texto", icon: Type, color: "text-brand" },
  group: { label: "Grupo", icon: Folder, color: "text-status-warning" },
  mask: { label: "Máscara", icon: ScanLine, color: "text-text-secondary" },
  generated: { label: "Gerada", icon: Sparkles, color: "text-accent-pink" },
  adjustment: { label: "Ajuste", icon: SlidersHorizontal, color: "text-status-success" },
  fill: { label: "Preenchimento", icon: PaintBucket, color: "text-accent-purple-light" },
};

export interface StudioLayersTreeProps {
  onSelectTextLayer?: (layerId: string | null) => void;
}

export function StudioLayersTree({ onSelectTextLayer }: StudioLayersTreeProps = {}) {
  const reactiveState = useStudioSceneStore();
  const state = reactiveState.scene ? reactiveState : useStudioSceneStore.getState();
  const {
    scene,
    selectedNodeIds,
    primaryNodeId,
    history,
    historyIndex,
    isSaving,
    error,
    groupSelected,
    undo,
    redo,
    clearError,
  } = state;
  const primaryNode = scene?.nodes.find((node) => node.id === primaryNodeId) ?? null;
  const rootNodes = useMemo(
    () => (scene ? [...orderedSceneChildren(scene, null)].reverse() : []),
    [scene],
  );

  if (!scene) {
    return (
      <aside className="flex h-full w-[340px] items-center justify-center border-l border-border bg-bg-primary text-[11px] text-text-muted">
        Carregando camadas...
      </aside>
    );
  }

  return (
    <aside data-testid="studio-layers-tree" className="flex h-full w-[340px] flex-col border-l border-border bg-bg-primary">
      <div className="border-b border-border bg-bg-secondary/45 px-3 py-2.5">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg border border-brand/25 bg-brand/10 text-brand">
            <Layers3 size={14} />
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="text-[12px] font-semibold text-text-primary">Camadas</h2>
            <p className="font-mono text-[9px] uppercase tracking-[0.12em] text-text-muted">
              {scene.nodes.length} nós · {selectedNodeIds.length} selecionada(s)
            </p>
          </div>
          {isSaving && <Loader2 size={13} className="animate-spin text-brand" aria-label="Salvando camadas" />}
        </div>

        <div className="mt-2 flex items-center gap-1">
          <button
            type="button"
            disabled={isSaving || historyIndex <= 0}
            onClick={() => void undo().catch(() => undefined)}
            className="rounded-md border border-border bg-bg-tertiary/40 p-1.5 text-text-muted transition-smooth hover:border-brand/30 hover:text-text-primary disabled:opacity-25"
            title={historyIndex > 0 ? `Desfazer: ${history[historyIndex - 1]?.label}` : "Nada para desfazer"}
          >
            <Undo2 size={12} />
          </button>
          <button
            type="button"
            disabled={isSaving || historyIndex >= history.length}
            onClick={() => void redo().catch(() => undefined)}
            className="rounded-md border border-border bg-bg-tertiary/40 p-1.5 text-text-muted transition-smooth hover:border-brand/30 hover:text-text-primary disabled:opacity-25"
            title={historyIndex < history.length ? `Refazer: ${history[historyIndex]?.label}` : "Nada para refazer"}
          >
            <Redo2 size={12} />
          </button>
          <div className="mx-0.5 h-4 w-px bg-border" />
          <button
            type="button"
            disabled={isSaving || selectedNodeIds.length === 0}
            onClick={() => void groupSelected().catch(() => undefined)}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-brand/25 bg-brand/8 px-2 py-1.5 text-[10px] font-medium text-brand transition-smooth hover:bg-brand/14 disabled:opacity-25"
            title="Criar grupo com a seleção"
          >
            <FolderPlus size={12} />
            Novo grupo
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-2 border-b border-status-error/25 bg-status-error/8 px-3 py-2 text-[10px] text-status-error">
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          <span className="min-w-0 flex-1 leading-relaxed">{error}</span>
          <button type="button" onClick={clearError} className="text-text-muted hover:text-text-primary" title="Fechar erro">×</button>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto py-1.5">
        {rootNodes.map((node) => (
          <StudioLayerNodeRow
            key={node.id}
            node={node}
            depth={0}
            onSelectTextLayer={onSelectTextLayer}
          />
        ))}
        {rootNodes.length === 0 && (
          <div className="mx-3 mt-3 rounded-lg border border-dashed border-border px-3 py-6 text-center text-[11px] text-text-muted">
            Nenhuma camada nesta página
          </div>
        )}
      </div>

      {primaryNode && <StudioLayerInspector node={primaryNode} />}
    </aside>
  );
}

function StudioLayerNodeRow({
  node,
  depth,
  onSelectTextLayer,
}: {
  node: StudioSceneNode;
  depth: number;
  onSelectTextLayer?: (layerId: string | null) => void;
}) {
  const reactiveState = useStudioSceneStore();
  const state = reactiveState.scene ? reactiveState : useStudioSceneStore.getState();
  const { scene, selectedNodeIds, isSaving, selectNode, patchNode, moveNodeBefore } = state;
  const presentation = NODE_PRESENTATION[node.kind];
  const Icon = presentation.icon;
  const selected = selectedNodeIds.includes(node.id);
  const children = scene ? [...orderedSceneChildren(scene, node.id)].reverse() : [];

  const select = (event: MouseEvent<HTMLDivElement>) => {
    selectNode(node.id, event.ctrlKey || event.metaKey || event.shiftKey);
    onSelectTextLayer?.(node.text_layer_id ?? null);
  };

  const drop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const draggedId = event.dataTransfer.getData("application/x-traduzai-scene-node");
    if (!draggedId || draggedId === node.id) return;
    void moveNodeBefore(node.id, draggedId).catch(() => undefined);
  };

  return (
    <>
      <div
        draggable={!isSaving}
        onDragStart={(event) => {
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("application/x-traduzai-scene-node", node.id);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDrop={drop}
        onClick={select}
        style={{ paddingLeft: `${8 + depth * 14}px` }}
        className={`group flex min-h-10 cursor-default items-center gap-1 border-l-2 pr-2 transition-smooth ${
          selected
            ? "border-brand bg-brand/10"
            : "border-transparent hover:border-brand/25 hover:bg-white/[0.035]"
        } ${node.visible ? "" : "opacity-55"}`}
        title={`${node.name} · ${presentation.label}`}
      >
        <GripVertical size={11} className="shrink-0 cursor-grab text-text-muted/60 group-hover:text-text-muted" />
        <button
          type="button"
          disabled={isSaving}
          onClick={(event) => {
            event.stopPropagation();
            void patchNode(node.id, { visible: !node.visible }).catch(() => undefined);
          }}
          className="rounded p-1 text-text-muted transition-smooth hover:bg-white/[0.06] hover:text-text-primary disabled:opacity-30"
          title={node.visible ? "Ocultar camada" : "Mostrar camada"}
        >
          {node.visible ? <Eye size={12} /> : <EyeOff size={12} />}
        </button>
        <Icon size={14} className={`shrink-0 ${presentation.color}`} />
        <div className="min-w-0 flex-1 py-1">
          <p className={`truncate text-[11px] font-medium ${node.visible ? "text-text-primary" : "text-text-muted line-through"}`}>
            {node.name || "Sem nome"}
          </p>
          <p className="truncate font-mono text-[8px] uppercase tracking-[0.12em] text-text-muted">
            {presentation.label}{children.length > 0 ? ` · ${children.length}` : ""}
          </p>
        </div>
        <button
          type="button"
          disabled={isSaving}
          onClick={(event) => {
            event.stopPropagation();
            void patchNode(node.id, { locked: !node.locked }).catch(() => undefined);
          }}
          className={`rounded p-1 transition-smooth hover:bg-white/[0.06] disabled:opacity-30 ${
            node.locked ? "text-status-warning" : "text-text-muted hover:text-text-primary"
          }`}
          title={node.locked ? "Desbloquear camada" : "Bloquear camada"}
        >
          {node.locked ? <Lock size={12} /> : <LockOpen size={12} />}
        </button>
      </div>
      {children.map((child) => (
        <StudioLayerNodeRow
          key={child.id}
          node={child}
          depth={depth + 1}
          onSelectTextLayer={onSelectTextLayer}
        />
      ))}
    </>
  );
}

function StudioLayerInspector({ node }: { node: StudioSceneNode }) {
  const reactiveState = useStudioSceneStore();
  const state = reactiveState.scene ? reactiveState : useStudioSceneStore.getState();
  const { patchNode, isSaving } = state;
  const [opacity, setOpacity] = useState(() => Math.round(node.opacity * 100));

  useEffect(() => {
    setOpacity(Math.round(node.opacity * 100));
  }, [node.id, node.opacity]);

  const commitOpacity = () => {
    const next = opacity / 100;
    if (Math.abs(next - node.opacity) < 0.001) return;
    void patchNode(node.id, { opacity: next }).catch(() => undefined);
  };

  return (
    <div className="border-t border-border bg-bg-secondary/35 px-3 py-2.5">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-[11px] font-semibold text-text-primary">{node.name}</p>
          <p className="font-mono text-[8px] uppercase tracking-[0.14em] text-text-muted">Propriedades da camada</p>
        </div>
        <span className="rounded border border-border bg-bg-tertiary/45 px-1.5 py-0.5 font-mono text-[9px] text-text-muted">
          {Math.round(node.opacity * 100)}%
        </span>
      </div>

      <label className="block text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
        Opacidade
        <input
          type="range"
          min={0}
          max={100}
          value={opacity}
          disabled={isSaving}
          onChange={(event) => setOpacity(Number(event.target.value))}
          onPointerUp={commitOpacity}
          onKeyUp={commitOpacity}
          onBlur={commitOpacity}
          className="mt-1 block w-full accent-[rgb(var(--color-brand))]"
        />
      </label>

      <label className="mt-2 block text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
        Modo de mesclagem
        <select
          value={node.blend_mode}
          disabled={isSaving}
          onChange={(event) => void patchNode(node.id, { blend_mode: event.target.value }).catch(() => undefined)}
          className="mt-1 w-full rounded-md border border-border bg-bg-tertiary/55 px-2 py-1.5 text-[10px] font-medium normal-case tracking-normal text-text-primary outline-none transition-smooth focus:border-brand/40"
        >
          {BLEND_MODES.map((mode) => (
            <option key={mode.value} value={mode.value}>{mode.label}</option>
          ))}
        </select>
      </label>
    </div>
  );
}
