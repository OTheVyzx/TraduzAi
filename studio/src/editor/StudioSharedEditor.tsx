import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MemoryRouter } from "react-router-dom";
import { FileDown } from "lucide-react";
import { Editor } from "../../../src/pages/Editor";
import type { EditorSceneVisualNode } from "../../../src/components/editor/stage/editorSceneVisual";
import { useAppStore, useEditorStore, type Project, type TextLayerStyle } from "../../../src/editor-shared";
import { createLegacyEditorBackendAdapter } from "../backend/editorBackendCompat";
import { getStudioEditorBackend } from "../backend/editorBackend";
import { downloadStudioPagePsd } from "../export/psd";
import type { StudioPage, StudioProject, StudioScene } from "../project/studioProject";
import { projectStudioSceneToPage, useStudioSceneStore } from "../store/studioSceneStore";
import { configureEditorBackend, type EditorBackendApi } from "../shims/currentEditorBackend";
import { StudioLayersTree } from "./layers/StudioLayersTree";
import { persistStudioScene } from "./layers/studioScenePersistence";
import { attachStudioSelectionMask, studioSelectionFromLasso } from "./selection/selectionModel";
import {
  composeStudioSceneLayerBitmaps,
  compositeStudioSceneLayerBitmaps,
  resolveStudioAssetPath,
  resolveStudioSceneVisualOrder,
} from "./compositor/studioSceneCompositor";
import { StudioRetouchToolbar } from "./retouch/StudioRetouchToolbar";
import { GenerativeFillPanel } from "./generative/GenerativeFillPanel";
import { ChapterToolsPanel } from "./batch/ChapterToolsPanel";
import { runStudioAutosaveCycle } from "../autosave/recovery";
import { useStudioProjectStore } from "../store/projectStore";
import { reconcileStudioEditorPage } from "./sharedEditorProjectSync";

const DEFAULT_TEXT_STYLE: TextLayerStyle = {
  fonte: "Comic Neue",
  tamanho: 34,
  cor: "#111827",
  cor_gradiente: [],
  contorno: "#ffffff",
  contorno_px: 2,
  glow: false,
  glow_cor: "#ffffff",
  glow_px: 0,
  sombra: false,
  sombra_cor: "#000000",
  sombra_offset: [0, 0],
  bold: true,
  italico: false,
  rotacao: 0,
  alinhamento: "center",
};

function canonicalizeTextStyle(style: unknown): TextLayerStyle {
  const record = typeof style === "object" && style !== null ? (style as Record<string, unknown>) : {};
  return {
    ...DEFAULT_TEXT_STYLE,
    ...record,
    fonte: String(record.fonte ?? record.fontFamily ?? DEFAULT_TEXT_STYLE.fonte),
    tamanho: Number(record.tamanho ?? record.fontSize ?? DEFAULT_TEXT_STYLE.tamanho),
    cor: String(record.cor ?? record.color ?? DEFAULT_TEXT_STYLE.cor),
    contorno: String(record.contorno ?? record.strokeColor ?? DEFAULT_TEXT_STYLE.contorno),
    contorno_px: Number(record.contorno_px ?? record.strokeWidth ?? DEFAULT_TEXT_STYLE.contorno_px),
    rotacao: Number(record.rotacao ?? record.rotation ?? DEFAULT_TEXT_STYLE.rotacao),
    alinhamento:
      record.alinhamento === "left" || record.alinhamento === "right" || record.alinhamento === "center"
        ? record.alinhamento
        : DEFAULT_TEXT_STYLE.alinhamento,
  };
}

function repairMaskLeakedIntoInpaint(page: StudioProject["paginas"][number]) {
  const maskPath = page.image_layers.mask?.path;
  const inpaintPath = page.image_layers.inpaint?.path;
  const fallbackPath = page.image_layers.base?.path ?? page.arquivo_original ?? page.arquivo_traduzido ?? null;
  if (!maskPath || !inpaintPath || inpaintPath !== maskPath || !fallbackPath) return page;
  return {
    ...page,
    image_layers: {
      ...page.image_layers,
      inpaint: {
        ...(page.image_layers.inpaint ?? {}),
        key: "inpaint" as const,
        path: fallbackPath,
        visible: true,
      },
    },
  };
}

function toAppProject(project: StudioProject, projectPath: string): Project {
  const paginas = project.paginas.map((inputPage) => {
    const page = repairMaskLeakedIntoInpaint(inputPage);
    const textLayers = page.text_layers.map((layer, index) => {
      const estilo = canonicalizeTextStyle(layer.estilo ?? layer.style);
      return {
        ...layer,
        bbox: layer.bbox,
        layout_bbox: layer.layout_bbox ?? layer.bbox,
        original: layer.original ?? "",
        traduzido: layer.traduzido ?? layer.translated ?? "",
        translated: layer.translated ?? layer.traduzido ?? "",
        tipo: layer.tipo ?? "fala",
        confianca_ocr: layer.confianca_ocr ?? layer.ocr_confidence ?? 1,
        ocr_confidence: layer.ocr_confidence ?? layer.confianca_ocr ?? 1,
        visible: layer.visible !== false,
        locked: layer.locked === true,
        order: layer.order ?? index,
        estilo,
        style: estilo,
      };
    });
    return {
      ...page,
      arquivo_original: page.arquivo_original ?? page.image_layers.base?.path ?? "",
      arquivo_traduzido: page.arquivo_traduzido ?? page.image_layers.rendered?.path ?? "",
      text_layers: textLayers,
      textos: textLayers,
    };
  }) as Project["paginas"];
  return {
    ...(project as unknown as Record<string, unknown>),
    id: project.id ?? "traduzai-studio-project",
    obra: project.obra ?? "TraduzAI Studio",
    capitulo: Number(project.capitulo ?? 1),
    idioma_origem: project.idioma_origem ?? "en",
    idioma_destino: project.idioma_destino ?? "pt-BR",
    qualidade: "normal",
    contexto: {
      sinopse: "",
      genero: [],
      personagens: [],
      glossario: {},
      aliases: [],
      termos: [],
      relacoes: [],
      faccoes: [],
      resumo_por_arco: [],
      memoria_lexical: {},
      fontes_usadas: [],
    },
    paginas,
    status: "done",
    source_path: projectPath,
    output_path: projectPath,
    totalPages: project.paginas.length,
    mode: "manual",
  };
}

export function StudioSharedEditor({
  project,
  projectPath,
}: {
  project: StudioProject;
  projectPath: string;
}) {
  const appProject = useMemo(() => toAppProject(project, projectPath), [project, projectPath]);
  const resetEditor = useEditorStore((state) => state.resetEditor);
  const currentPageIndex = useEditorStore((state) => state.currentPageIndex);
  const commitEdits = useEditorStore((state) => state.commitEdits);
  const editorPage = useEditorStore((state) => state.currentPage);
  const selectedLayerId = useEditorStore((state) => state.selectedLayerId);
  const scene = useStudioSceneStore((state) => state.scene);
  const primaryNodeId = useStudioSceneStore((state) => state.primaryNodeId);
  const [isExportingPsd, setIsExportingPsd] = useState(false);
  const initializedProjectPath = useRef<string | null>(null);
  const recoverySnapshot = useStudioProjectStore((state) => state.recoverySnapshot);
  const restoreRecovery = useStudioProjectStore((state) => state.restoreRecovery);
  const dismissRecovery = useStudioProjectStore((state) => state.dismissRecovery);
  const isProjectSaving = useStudioProjectStore((state) => state.isProjectSaving);
  const [bitmapComposite, setBitmapComposite] = useState<{
    pageKey: string;
    source: string;
    visualNodes: EditorSceneVisualNode[];
  } | null>(null);

  useEffect(() => {
    const backend = createLegacyEditorBackendAdapter(getStudioEditorBackend()) as unknown as EditorBackendApi;
    configureEditorBackend(backend);
    useAppStore.getState().setProject(appProject);
    if (initializedProjectPath.current !== projectPath) {
      initializedProjectPath.current = projectPath;
      resetEditor();
      return;
    }
    const editorState = useEditorStore.getState();
    useEditorStore.setState(reconcileStudioEditorPage(
      appProject,
      editorState.currentPageIndex,
      editorState.selectedLayerId,
      editorState.currentPage,
      editorState.dirty,
    ));
  }, [appProject, projectPath, resetEditor]);

  useEffect(() => {
    if (recoverySnapshot) return;
    let disposed = false;
    let running = false;
    const capture = async (force: boolean) => {
      const editorState = useEditorStore.getState();
      if (disposed || running || (!force && !editorState.dirty)) return;
      running = true;
      try {
        await runStudioAutosaveCycle({
          backend: getStudioEditorBackend(),
          projectPath,
          dirty: editorState.dirty,
          runAutoSave: () => useEditorStore.getState().runAutoSave(),
        });
      } catch (error) {
        console.error("Falha no autosave/recuperacao do Studio:", error);
      } finally {
        running = false;
      }
    };
    void capture(true);
    const interval = window.setInterval(() => void capture(false), 3000);
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!useEditorStore.getState().dirty) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => {
      disposed = true;
      window.clearInterval(interval);
      window.removeEventListener("beforeunload", beforeUnload);
    };
  }, [projectPath, recoverySnapshot]);

  const currentPage = project.paginas[currentPageIndex] ?? project.paginas[0] ?? null;
  const primaryNode = scene?.nodes.find((node) => node.id === primaryNodeId) ?? null;
  const selectionTargetNode = primaryNode
    && (primaryNode.kind === "raster" || primaryNode.kind === "generated")
    && !primaryNode.locked
    ? primaryNode
    : null;
  const selectionTargetLabel = primaryNode
    ? `${primaryNode.name}${primaryNode.locked ? " (bloqueada)" : ""}`
    : null;

  const persistCurrentScene = useCallback(async (scene: StudioScene) => {
    const backend = getStudioEditorBackend();
    await persistStudioScene({
      backend,
      projectPath,
      pageIndex: currentPageIndex,
      scene,
    });
    const editorState = useEditorStore.getState();
    if (editorState.currentPageIndex !== currentPageIndex || !editorState.currentPage) return;
    const synchronizedPage = projectStudioSceneToPage(
      editorState.currentPage as unknown as StudioPage,
      scene,
    );
    useEditorStore.setState({
      currentPage: synchronizedPage as unknown as typeof editorState.currentPage,
    });
  }, [currentPageIndex, projectPath]);

  useEffect(() => {
    const pageKey = `${projectPath}::${currentPageIndex}`;
    const fallbackScene = project.paginas[currentPageIndex]?.studio_scene ?? null;
    let disposed = false;
    useStudioSceneStore.setState({
      pageKey,
      scene: null,
      selectedNodeIds: [],
      primaryNodeId: null,
      history: [],
      historyIndex: 0,
      isSaving: false,
      error: null,
      persist: persistCurrentScene,
    });
    getStudioEditorBackend()
      .loadProject({ project_path: projectPath })
      .then((latestProject) => {
        if (disposed) return;
        const scene = latestProject.paginas[currentPageIndex]?.studio_scene ?? fallbackScene;
        if (!scene) throw new Error(`Página ${currentPageIndex + 1} sem studio_scene`);
        useStudioSceneStore.getState().hydrate(pageKey, scene, persistCurrentScene);
      })
      .catch((error) => {
        if (disposed) return;
        if (fallbackScene) {
          useStudioSceneStore.getState().hydrate(pageKey, fallbackScene, persistCurrentScene);
        }
        useStudioSceneStore.setState({ error: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      disposed = true;
    };
  }, [currentPageIndex, persistCurrentScene, project.paginas, projectPath]);

  useEffect(() => {
    if (!scene || !editorPage) return;
    const pageKey = `${projectPath}::${currentPageIndex}`;
    let disposed = false;
    void composeStudioSceneLayerBitmaps({
      page: editorPage as unknown as StudioPage,
      scene,
      resolveSourcePath: (path) => resolveStudioAssetPath(projectPath, path),
    })
      .then((rendered) => {
        if (disposed) return;
        const bitmapByNodeId = new Map(rendered.layers.map((layer) => [layer.nodeId, layer]));
        const visualNodes = resolveStudioSceneVisualOrder(editorPage as unknown as StudioPage, scene).flatMap(
          (item): EditorSceneVisualNode[] => {
            if (item.kind === "text") {
              return [{ id: item.nodeId, kind: "text", textLayerId: item.textLayerId }];
            }
            const layer = bitmapByNodeId.get(item.nodeId);
            if (!layer) return [];
            return [{
              id: item.nodeId,
              kind: "bitmap",
              source: layer.canvas.toDataURL("image/png"),
              opacity: layer.opacity,
              blendMode: layer.blendMode,
            }];
          },
        );
        const composite = compositeStudioSceneLayerBitmaps(rendered);
        setBitmapComposite({
          pageKey,
          source: composite.toDataURL("image/png"),
          visualNodes,
        });
      })
      .catch((error) => {
        if (!disposed) console.error("Falha ao compor camadas do Studio:", error);
      });
    return () => {
      disposed = true;
    };
  }, [currentPageIndex, editorPage, projectPath, scene]);

  const selectSceneTextLayer = useCallback((layerId: string | null) => {
    useEditorStore.getState().selectLayer(layerId);
  }, []);

  const prepareChapterProject = useCallback(async () => {
    const editorState = useEditorStore.getState();
    if (editorState.dirty) await editorState.flushAutoSave();
    return getStudioEditorBackend().loadProject({ project_path: projectPath });
  }, [projectPath]);

  const navigateToChapterLayer = useCallback(async (pageIndex: number, layerId: string) => {
    const editorState = useEditorStore.getState();
    if (editorState.dirty) await editorState.flushAutoSave();
    await editorState.setCurrentPage(pageIndex);
    useEditorStore.getState().selectLayer(layerId);
  }, []);

  const changeStudioPage = useCallback(async (pageIndex: number) => {
    const editorState = useEditorStore.getState();
    if (editorState.dirty) await editorState.flushAutoSave();
    await useEditorStore.getState().setCurrentPage(pageIndex);
  }, []);

  const attachCurrentSelectionMask = useCallback(async () => {
    const editorState = useEditorStore.getState();
    const sceneState = useStudioSceneStore.getState();
    const targetNodeId = sceneState.primaryNodeId;
    if (!editorState.activeLassoSelection) throw new Error("Crie uma seleção antes de adicionar a máscara");
    if (!targetNodeId) throw new Error("Selecione uma camada-alvo na árvore de camadas");
    const selection = studioSelectionFromLasso(editorState.activeLassoSelection, targetNodeId);
    const changed = await sceneState.executeSceneCommand(
      "Adicionar máscara de camada",
      (currentScene) => attachStudioSelectionMask(currentScene, selection),
    );
    if (changed) editorState.setActiveLassoSelection(null);
  }, []);

  const exportCurrentPagePsd = async () => {
    setIsExportingPsd(true);
    try {
      await commitEdits();
      const latestProject = await getStudioEditorBackend().loadProject({ project_path: projectPath });
      await downloadStudioPagePsd({
        ...latestProject,
        source_path: latestProject.source_path ?? projectPath,
      }, currentPageIndex);
    } catch (error) {
      console.error("Falha ao exportar PSD no Studio:", error);
    } finally {
      setIsExportingPsd(false);
    }
  };

  return (
    <MemoryRouter>
      {recoverySnapshot && (
        <div className="fixed inset-0 z-[120] flex items-center justify-center bg-black/75 p-6" role="dialog" aria-modal="true" aria-labelledby="studio-recovery-title">
          <div className="w-full max-w-md rounded-2xl border border-status-warning/40 bg-bg-secondary p-5 shadow-2xl">
            <p id="studio-recovery-title" className="text-sm font-semibold text-text-primary">Sessão de recuperação encontrada</p>
            <p className="mt-2 text-xs leading-5 text-text-muted">
              Escolha recuperar a sessão anterior ou ignorá-la antes de continuar editando este projeto.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                disabled={isProjectSaving}
                onClick={() => void dismissRecovery()}
                className="rounded-md border border-border px-3 py-2 text-xs text-text-secondary disabled:opacity-40"
              >
                Ignorar
              </button>
              <button
                type="button"
                disabled={isProjectSaving}
                onClick={() => void restoreRecovery()}
                className="rounded-md bg-status-warning px-3 py-2 text-xs font-semibold text-black disabled:opacity-40"
              >
                Recuperar sessão
              </button>
            </div>
          </div>
        </div>
      )}
      <Editor
        mode="studio"
        onBack={() => undefined}
        emptyBackLabel="Voltar ao Studio"
        layersPanel={<StudioLayersTree onSelectTextLayer={selectSceneTextLayer} />}
        selectionTargetNodeId={selectionTargetNode?.id ?? null}
        selectionTargetLabel={selectionTargetLabel}
        onAttachSelectionMask={attachCurrentSelectionMask}
        bitmapCompositeSource={
          bitmapComposite?.pageKey === `${projectPath}::${currentPageIndex}` ? bitmapComposite.source : null
        }
        sceneVisualNodes={
          bitmapComposite?.pageKey === `${projectPath}::${currentPageIndex}` ? bitmapComposite.visualNodes : null
        }
        onRequestPageChange={changeStudioPage}
        headerActions={
          <>
            <ChapterToolsPanel
              project={project}
              currentPageIndex={currentPageIndex}
              selectedLayerId={selectedLayerId}
              onPrepareProject={prepareChapterProject}
              onNavigateToLayer={navigateToChapterLayer}
            />
            <GenerativeFillPanel
              projectPath={projectPath}
              page={editorPage as unknown as StudioPage | null}
            />
            <StudioRetouchToolbar
              projectPath={projectPath}
              page={editorPage as unknown as StudioPage | null}
            />
            <button
              type="button"
              onClick={() => void exportCurrentPagePsd()}
              disabled={isExportingPsd || !currentPage}
              className="flex items-center gap-1 rounded-lg border border-status-success/30 bg-status-success/10 px-2.5 py-1 text-[11px] font-medium text-status-success transition-smooth hover:bg-status-success/15 disabled:opacity-30"
              title="Salvar pagina atual em PSD"
            >
              <FileDown size={12} />
              Salvar em PSD
            </button>
          </>
        }
      />
    </MemoryRouter>
  );
}
