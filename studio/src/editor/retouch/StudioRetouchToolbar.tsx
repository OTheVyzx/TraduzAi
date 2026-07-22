import { useState } from "react";
import { loadImageSource } from "../../../../src/lib/imageSource";
import { useEditorStore } from "../../../../src/lib/stores/editorStore";
import { getStudioEditorBackend } from "../../backend/editorBackend";
import type { StudioPage, StudioSceneNode } from "../../project/studioProject";
import { useStudioSceneStore } from "../../store/studioSceneStore";
import { resolveStudioAssetPath } from "../compositor/studioSceneCompositor";
import { studioSelectionFromLasso } from "../selection/selectionModel";
import {
  applyRetouchCommandToScene,
  assertRetouchExecutionContext,
  createRetouchCommand,
  renderRetouchCommandBitmap,
  type RetouchTool,
} from "./retouchCommands";

const TOOL_LABELS: Record<RetouchTool, string> = {
  clone: "Clone",
  healing: "Correção",
  patch: "Remendo",
};

function nodeSourcePath(page: StudioPage, node: StudioSceneNode) {
  if (node.kind === "generated") {
    const path = node.metadata.image_path;
    return typeof path === "string" && path.trim() ? path : null;
  }
  if (node.kind !== "raster" || !node.image_layer_key) return null;
  return page.image_layers[node.image_layer_key]?.path
    ?? (node.image_layer_key === "base" ? page.arquivo_original : null)
    ?? null;
}

async function loadRetouchImage(path: string) {
  const loaded = await loadImageSource(path, "image/png");
  return new Promise<{ image: HTMLImageElement; width: number; height: number }>((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      resolve({ image, width: image.naturalWidth, height: image.naturalHeight });
      if (loaded.revoke) window.setTimeout(loaded.revoke, 1000);
    };
    image.onerror = () => {
      loaded.revoke?.();
      reject(new Error("Não foi possível carregar a camada selecionada"));
    };
    image.src = loaded.src;
  });
}

export function StudioRetouchToolbar({
  projectPath,
  page,
}: {
  projectPath: string;
  page: StudioPage | null;
}) {
  const activeSelection = useEditorStore((state) => state.activeLassoSelection);
  const setActiveSelection = useEditorStore((state) => state.setActiveLassoSelection);
  const scene = useStudioSceneStore((state) => state.scene);
  const primaryNodeId = useStudioSceneStore((state) => state.primaryNodeId);
  const isSaving = useStudioSceneStore((state) => state.isSaving);
  const [sourceOffsetX, setSourceOffsetX] = useState(24);
  const [sourceOffsetY, setSourceOffsetY] = useState(0);
  const [busyTool, setBusyTool] = useState<RetouchTool | null>(null);
  const [error, setError] = useState<string | null>(null);
  const target = scene?.nodes.find((node) => node.id === primaryNodeId) ?? null;
  const targetIsSupported = target?.kind === "raster" || target?.kind === "generated";
  const canRetouch = Boolean(page && scene && activeSelection && target && targetIsSupported && !target.locked);

  const executeRetouch = async (tool: RetouchTool) => {
    if (!page || !scene || !activeSelection || !target || !targetIsSupported) return;
    setBusyTool(tool);
    setError(null);
    try {
      const initialSceneState = useStudioSceneStore.getState();
      const contextToken = {
        pageKey: initialSceneState.pageKey,
        pageIndex: activeSelection.pageIndex,
        scene,
      };
      const assertCurrentContext = () => {
        const currentSceneState = useStudioSceneStore.getState();
        assertRetouchExecutionContext(contextToken, {
          pageKey: currentSceneState.pageKey,
          pageIndex: useEditorStore.getState().currentPageIndex,
          scene: currentSceneState.scene,
        });
      };
      assertCurrentContext();
      const sourcePath = nodeSourcePath(page, target);
      if (!sourcePath) throw new Error("A camada selecionada não possui pixels para o retoque");
      const loadedSource = await loadRetouchImage(resolveStudioAssetPath(projectPath, sourcePath));
      const commandId = `retouch-${crypto.randomUUID()}`;
      const selection = studioSelectionFromLasso(activeSelection, target.id);
      const baseCommand = createRetouchCommand({
        id: commandId,
        tool,
        targetNodeId: target.id,
        selection,
        sampling: tool === "healing"
          ? { mode: "automatic" }
          : {
              mode: "sampled",
              sourceNodeId: target.id,
              sourceOffset: [sourceOffsetX, sourceOffsetY],
              aligned: true,
            },
      });
      const pngData = await renderRetouchCommandBitmap(baseCommand, {
        width: loadedSource.width,
        height: loadedSource.height,
        createCanvas: (width, height) => {
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          return canvas;
        },
        loadNodeImage: async () => ({ image: loadedSource.image }),
      });
      assertCurrentContext();
      const resultPath = await getStudioEditorBackend().saveGeneratedAsset({
        project_path: projectPath,
        page_index: activeSelection.pageIndex,
        asset_id: commandId,
        png_data: pngData,
      });
      assertCurrentContext();
      const readyCommand = createRetouchCommand({
        ...baseCommand,
        resultPath,
      });
      assertCurrentContext();
      const changed = await useStudioSceneStore.getState().executeSceneCommand(
        `${TOOL_LABELS[tool]} em nova camada`,
        (currentScene) => applyRetouchCommandToScene(currentScene, readyCommand),
      );
      if (changed) setActiveSelection(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusyTool(null);
    }
  };

  return (
    <div className="flex items-center gap-1.5" title={error ?? "Retoque não destrutivo em uma nova camada"}>
      <div className="flex items-center gap-1 rounded-lg border border-border bg-bg-tertiary/45 px-1.5 py-1">
        <span className="text-[9px] font-semibold uppercase tracking-[0.08em] text-text-muted">Amostra</span>
        <input
          aria-label="Deslocamento horizontal da amostra"
          type="number"
          value={sourceOffsetX}
          onChange={(event) => setSourceOffsetX(Number(event.target.value) || 0)}
          className="w-12 rounded border border-border bg-bg-primary px-1 py-0.5 text-center font-mono text-[9px] text-text-primary outline-none focus:border-brand/40"
          title="Deslocamento X da origem"
        />
        <input
          aria-label="Deslocamento vertical da amostra"
          type="number"
          value={sourceOffsetY}
          onChange={(event) => setSourceOffsetY(Number(event.target.value) || 0)}
          className="w-12 rounded border border-border bg-bg-primary px-1 py-0.5 text-center font-mono text-[9px] text-text-primary outline-none focus:border-brand/40"
          title="Deslocamento Y da origem"
        />
      </div>
      {(["clone", "healing", "patch"] as const).map((tool) => (
        <button
          key={tool}
          type="button"
          disabled={!canRetouch || isSaving || busyTool !== null}
          onClick={() => void executeRetouch(tool)}
          className="rounded-lg border border-brand/25 bg-brand/8 px-2 py-1.5 text-[10px] font-medium text-brand transition-smooth hover:bg-brand/14 disabled:opacity-25"
          title={canRetouch ? `${TOOL_LABELS[tool]} na seleção e criar nova camada` : "Selecione uma área e uma camada raster"}
        >
          {busyTool === tool ? "Aplicando…" : TOOL_LABELS[tool]}
        </button>
      ))}
      {error && <span className="max-w-40 truncate text-[9px] text-status-error">{error}</span>}
    </div>
  );
}
