import { useEffect, useMemo, useState } from "react";
import { Check, LoaderCircle, Sparkles, Trash2, X } from "lucide-react";
import { useEditorStore } from "../../../../src/lib/stores/editorStore";
import { getStudioEditorBackend } from "../../backend/editorBackend";
import type { StudioPage, StudioSceneNode } from "../../project/studioProject";
import { useStudioSceneStore } from "../../store/studioSceneStore";
import {
  DEFAULT_FLUX_MODEL,
  FLUX_ADAPTER_CONTRACT_VERSION,
  acceptFluxVariant,
  activateFluxVariant,
  applyFluxGenerationToScene,
  assertFluxExecutionContext,
  createFluxGeneration,
  findFluxGenerationId,
  rejectFluxGeneration,
  type FluxGenerateConfig,
  type FluxGeneration,
} from "../../ai/fluxContract";
import { useFluxStore } from "../../ai/fluxStore";
import { resolveStudioAssetPath } from "../compositor/studioSceneCompositor";
import { studioSelectionFromLasso } from "../selection/selectionModel";
import {
  materializeFluxVariantOverlay,
  prepareFluxCrop,
} from "./fluxImagePreparation";

export const GENERATIVE_FILL_COPY = {
  title: "Preenchimento FLUX",
  prompt: "Prompt opcional",
  defaultVariants: "2 variantes",
  localAdapter: "Configure o adaptador local para usar FLUX",
} as const;

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

function safeVariantAssetId(jobId: string, variantId: string) {
  const safeVariant = variantId.replace(/[^A-Za-z0-9_-]+/g, "-").slice(0, 48) || "variant";
  return `${jobId}-${safeVariant}`.slice(0, 128);
}

export function GenerativeFillPanel({
  projectPath,
  page,
  initiallyOpen = false,
}: {
  projectPath: string;
  page: StudioPage | null;
  initiallyOpen?: boolean;
}) {
  const [open, setOpen] = useState(initiallyOpen);
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("texto, letras, assinatura, artefatos");
  const [variantCount, setVariantCount] = useState(2);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 2_000_000_000));
  const [localError, setLocalError] = useState<string | null>(null);
  const activeSelection = useEditorStore((state) => state.activeLassoSelection);
  const currentPageIndex = useEditorStore((state) => state.currentPageIndex);
  const setActiveSelection = useEditorStore((state) => state.setActiveLassoSelection);
  const scene = useStudioSceneStore((state) => state.scene);
  const pageKey = useStudioSceneStore((state) => state.pageKey);
  const primaryNodeId = useStudioSceneStore((state) => state.primaryNodeId);
  const isSavingScene = useStudioSceneStore((state) => state.isSaving);
  const providerStatus = useFluxStore((state) => state.providerStatus);
  const isCheckingProvider = useFluxStore((state) => state.isCheckingProvider);
  const activeJob = useFluxStore((state) => state.activeJob);
  const fluxError = useFluxStore((state) => state.error);
  const checkProvider = useFluxStore((state) => state.checkProvider);
  const generate = useFluxStore((state) => state.generate);
  const cancelActiveJob = useFluxStore((state) => state.cancelActiveJob);
  const clearCompletedJob = useFluxStore((state) => state.clearCompletedJob);
  const target = scene?.nodes.find((node) => node.id === primaryNodeId) ?? null;
  const targetIsSupported = target?.kind === "raster" || target?.kind === "generated";
  const sceneGenerationId = useMemo(
    () => findFluxGenerationId(scene, primaryNodeId),
    [primaryNodeId, scene],
  );
  const activeJobGenerationId = activeJob?.request.job_id ?? null;
  const generationId = sceneGenerationId ?? (
    activeJob?.status === "generating" || activeJob?.status === "cancelling"
      ? activeJobGenerationId
      : null
  );
  const generationGroup = generationId
    ? scene?.nodes.find((node) => node.id === `group:${generationId}`) ?? null
    : null;
  const sceneGeneration = generationGroup?.metadata.flux_generation as FluxGeneration | undefined;
  const displayVariants = activeJob?.request.job_id === generationId
    ? activeJob.result?.variants ?? sceneGeneration?.variants ?? []
    : sceneGeneration?.variants ?? [];
  const activeVariantId = useMemo(() => {
    if (!scene || !generationId) return null;
    const active = scene.nodes.find((node) => (
      node.kind === "generated" &&
      node.visible &&
      node.metadata.generation_id === generationId
    ));
    return typeof active?.metadata.variant_id === "string" ? active.metadata.variant_id : null;
  }, [generationId, scene]);
  const isCancelling = activeJob?.status === "cancelling";
  const isGenerating = activeJob?.status === "generating" || isCancelling;
  const canGenerate = Boolean(
    page &&
    scene &&
    activeSelection &&
    target &&
    targetIsSupported &&
    !target.locked &&
    (providerStatus?.status === "ready" || providerStatus?.status === "configured") &&
    !isSavingScene &&
    !isGenerating,
  );

  useEffect(() => {
    if (!open || providerStatus || isCheckingProvider) return;
    void checkProvider(() => getStudioEditorBackend().fluxProviderStatus());
  }, [checkProvider, isCheckingProvider, open, providerStatus]);

  useEffect(() => () => {
    const fluxState = useFluxStore.getState();
    if (fluxState.activeJob?.status === "generating") {
      void fluxState.cancelActiveJob((jobId) => (
        getStudioEditorBackend().cancelFluxFill(jobId)
      ));
    } else if (fluxState.activeJob?.status !== "cancelling") {
      fluxState.clearCompletedJob();
    }
  }, [currentPageIndex, pageKey]);

  const refreshStatus = () => {
    useFluxStore.setState({ providerStatus: null, error: null });
    void checkProvider(() => getStudioEditorBackend().fluxProviderStatus());
  };

  const assertCurrentContext = (expected: { pageKey: string | null; pageIndex: number; scene: NonNullable<typeof scene> }) => {
    const currentSceneState = useStudioSceneStore.getState();
    assertFluxExecutionContext(expected, {
      pageKey: currentSceneState.pageKey,
      pageIndex: useEditorStore.getState().currentPageIndex,
      scene: currentSceneState.scene,
    });
  };

  const runGeneration = async () => {
    if (!page || !scene || !activeSelection || !target || !targetIsSupported || target.locked) return;
    setLocalError(null);
    const context = { pageKey, pageIndex: currentPageIndex, scene };
    const savedAssetIds: string[] = [];
    let sceneCommitted = false;
    try {
      assertCurrentContext(context);
      const sourcePath = nodeSourcePath(page, target);
      if (!sourcePath) throw new Error("A camada selecionada não possui pixels para o FLUX");
      const selection = studioSelectionFromLasso(activeSelection, target.id);
      const prepared = await prepareFluxCrop(resolveStudioAssetPath(projectPath, sourcePath), selection);
      assertCurrentContext(context);
      const jobId = `flux-${crypto.randomUUID()}`;
      const request: FluxGenerateConfig = {
        contract_version: FLUX_ADAPTER_CONTRACT_VERSION,
        job_id: jobId,
        prompt,
        negative_prompt: negativePrompt,
        model: providerStatus?.model?.trim() || DEFAULT_FLUX_MODEL,
        source_png_data: prepared.sourcePngData,
        mask_png_data: prepared.maskPngData,
        width: prepared.width,
        height: prepared.height,
        variant_count: variantCount,
        seed: Math.trunc(seed),
        steps: 20,
        guidance_scale: 18,
      };
      const backend = getStudioEditorBackend();
      const result = await generate(request, (config) => backend.generateFluxFill(config));
      if (!result) return;
      assertCurrentContext(context);

      const variants = [];
      for (const variant of result.variants) {
        assertCurrentContext(context);
        const variantSource = variant.png_data?.trim()
          || (variant.path ? resolveStudioAssetPath(projectPath, variant.path) : "");
        if (!variantSource) throw new Error(`A variante ${variant.id} não possui imagem local`);
        const pngData = await materializeFluxVariantOverlay(prepared, variantSource);
        assertCurrentContext(context);
        const assetId = safeVariantAssetId(jobId, variant.id);
        const resultPath = await backend.saveGeneratedAsset({
          project_path: projectPath,
          page_index: currentPageIndex,
          asset_id: assetId,
          png_data: pngData,
        });
        savedAssetIds.push(assetId);
        assertCurrentContext(context);
        variants.push({ id: variant.id, seed: variant.seed, resultPath });
      }

      const fluxGeneration = createFluxGeneration({
        id: jobId,
        targetNodeId: target.id,
        selection,
        prompt,
        negativePrompt,
        provider: result.provider,
        model: result.model,
        cropBbox: prepared.bbox,
        seed,
        variants,
        createdAt: Date.now(),
      });
      assertCurrentContext(context);
      const changed = await useStudioSceneStore.getState().executeSceneCommand(
        "Adicionar variantes FLUX",
        (currentScene) => applyFluxGenerationToScene(currentScene, fluxGeneration),
      );
      if (!changed) throw new Error("Não foi possível adicionar as variantes FLUX à cena");
      sceneCommitted = true;
      useStudioSceneStore.getState().selectNode(`group:${jobId}`);
      setActiveSelection(null);
    } catch (error) {
      let message = error instanceof Error ? error.message : String(error);
      if (!sceneCommitted && savedAssetIds.length > 0) {
        try {
          await getStudioEditorBackend().deleteGeneratedAssets({
            project_path: projectPath,
            page_index: context.pageIndex,
            asset_ids: savedAssetIds,
          });
        } catch (cleanupError) {
          message += ` · Falha ao limpar variantes parciais: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}`;
        }
      }
      setLocalError(message);
    }
  };

  const previewVariant = async (variantId: string) => {
    if (!generationId) return;
    setLocalError(null);
    try {
      await useStudioSceneStore.getState().executeSceneCommand(
        "Visualizar variante FLUX",
        (currentScene) => activateFluxVariant(currentScene, generationId, variantId),
      );
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    }
  };

  const keepVariant = async () => {
    if (!generationId || !activeVariantId) return;
    setLocalError(null);
    try {
      const changed = await useStudioSceneStore.getState().executeSceneCommand(
        "Aceitar variante FLUX",
        (currentScene) => acceptFluxVariant(currentScene, generationId, activeVariantId),
      );
      if (changed) {
        useStudioSceneStore.getState().selectNode(`generated:${generationId}:${activeVariantId}`);
        clearCompletedJob();
      }
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    }
  };

  const discardGeneration = async () => {
    if (!generationId || !generationGroup) return;
    setLocalError(null);
    try {
      const sourceNodeId = generationGroup.metadata.source_node_id;
      const changed = await useStudioSceneStore.getState().executeSceneCommand(
        "Descartar variantes FLUX",
        (currentScene) => rejectFluxGeneration(currentScene, generationId),
      );
      if (changed) {
        if (typeof sourceNodeId === "string") useStudioSceneStore.getState().selectNode(sourceNodeId);
        clearCompletedJob();
      }
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    }
  };

  const statusLabel = isCheckingProvider
    ? "Verificando adaptador local…"
    : providerStatus?.status === "ready"
      ? `Adaptador local pronto · ${providerStatus.model ?? "FLUX"}`
    : providerStatus?.status === "configured"
      ? `Adaptador configurado · validação ao gerar · ${providerStatus.model ?? "FLUX"}`
    : providerStatus?.message ?? GENERATIVE_FILL_COPY.localAdapter;
  const error = localError ?? fluxError;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex items-center gap-1 rounded-lg border border-fuchsia-400/30 bg-fuchsia-400/10 px-2.5 py-1 text-[11px] font-medium text-fuchsia-200 transition-smooth hover:bg-fuchsia-400/15"
        title="Preenchimento generativo local por seleção"
      >
        <Sparkles size={12} />
        {GENERATIVE_FILL_COPY.title}
      </button>

      {open && (
        <section className="absolute right-0 top-[calc(100%+8px)] z-[80] w-[380px] rounded-xl border border-border bg-bg-secondary p-3 shadow-2xl">
          <div className="mb-2 flex items-start justify-between gap-3">
            <div>
              <h3 className="text-xs font-semibold text-text-primary">{GENERATIVE_FILL_COPY.title}</h3>
              <p className="mt-0.5 text-[9px] text-text-muted">Seleção local → variantes em novas camadas</p>
            </div>
            <button type="button" onClick={() => setOpen(false)} className="rounded p-1 text-text-muted hover:bg-bg-tertiary" aria-label="Fechar FLUX">
              <X size={13} />
            </button>
          </div>

          <label className="block text-[9px] font-semibold uppercase tracking-[0.08em] text-text-muted">
            {GENERATIVE_FILL_COPY.prompt}
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={2}
              maxLength={4000}
              placeholder="Ex.: reconstruir o tecido mantendo o traço do mangá"
              className="mt-1 w-full resize-none rounded-lg border border-border bg-bg-primary px-2 py-1.5 text-[11px] font-normal normal-case tracking-normal text-text-primary outline-none focus:border-fuchsia-400/45"
            />
          </label>
          <label className="mt-2 block text-[9px] font-semibold uppercase tracking-[0.08em] text-text-muted">
            Evitar
            <input
              value={negativePrompt}
              onChange={(event) => setNegativePrompt(event.target.value)}
              maxLength={4000}
              className="mt-1 w-full rounded-lg border border-border bg-bg-primary px-2 py-1.5 text-[10px] font-normal normal-case tracking-normal text-text-primary outline-none focus:border-fuchsia-400/45"
            />
          </label>

          <div className="mt-2 grid grid-cols-2 gap-2">
            <label className="text-[9px] font-semibold uppercase tracking-[0.08em] text-text-muted">
              Variantes
              <select
                value={variantCount}
                onChange={(event) => setVariantCount(Number(event.target.value))}
                className="mt-1 w-full rounded-lg border border-border bg-bg-primary px-2 py-1.5 text-[10px] font-normal normal-case tracking-normal text-text-primary"
              >
                <option value={2}>{GENERATIVE_FILL_COPY.defaultVariants}</option>
                <option value={3}>3 variantes</option>
                <option value={4}>4 variantes</option>
              </select>
            </label>
            <label className="text-[9px] font-semibold uppercase tracking-[0.08em] text-text-muted">
              Seed
              <input
                type="number"
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value) || 0)}
                className="mt-1 w-full rounded-lg border border-border bg-bg-primary px-2 py-1.5 font-mono text-[10px] font-normal normal-case tracking-normal text-text-primary"
              />
            </label>
          </div>

          <div className="mt-2 flex items-center justify-between gap-2 rounded-lg border border-border/80 bg-bg-tertiary/40 px-2 py-1.5">
            <span className={`text-[9px] ${providerStatus?.status === "ready" ? "text-status-success" : "text-text-muted"}`}>
              {statusLabel}
            </span>
            <button type="button" onClick={refreshStatus} disabled={isCheckingProvider || isGenerating} className="text-[9px] text-brand hover:underline disabled:opacity-30">
              Reverificar
            </button>
          </div>

          {displayVariants.length > 0 && generationGroup && (
            <div className="mt-2 rounded-lg border border-fuchsia-400/20 bg-fuchsia-400/5 p-2">
              <p className="mb-1.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-fuchsia-200">Variantes geradas</p>
              <div className="grid grid-cols-2 gap-1">
                {displayVariants.map((variant, index) => (
                  <button
                    key={variant.id}
                    type="button"
                    onClick={() => void previewVariant(variant.id)}
                    className={`rounded-md border px-2 py-1 text-left text-[9px] ${activeVariantId === variant.id ? "border-fuchsia-300 bg-fuchsia-300/15 text-fuchsia-100" : "border-border bg-bg-primary text-text-secondary"}`}
                  >
                    Variante {index + 1}<span className="ml-1 font-mono text-[8px] opacity-60">#{variant.seed}</span>
                  </button>
                ))}
              </div>
              <div className="mt-2 flex gap-1">
                <button type="button" onClick={() => void keepVariant()} disabled={!activeVariantId || isSavingScene} className="flex flex-1 items-center justify-center gap-1 rounded-md bg-status-success/15 px-2 py-1 text-[9px] text-status-success disabled:opacity-30">
                  <Check size={10} /> Manter variante
                </button>
                <button type="button" onClick={() => void discardGeneration()} disabled={isSavingScene} className="flex items-center justify-center gap-1 rounded-md bg-status-error/10 px-2 py-1 text-[9px] text-status-error disabled:opacity-30">
                  <Trash2 size={10} /> Descartar
                </button>
              </div>
            </div>
          )}

          {error && <p className="mt-2 rounded-md bg-status-error/10 px-2 py-1 text-[9px] text-status-error">{error}</p>}
          {!activeSelection && !generationGroup && <p className="mt-2 text-[9px] text-text-muted">Crie uma seleção Lasso sobre uma camada raster desbloqueada.</p>}

          <div className="mt-3 flex gap-1.5">
            {isGenerating ? (
              <button
                type="button"
                onClick={() => void cancelActiveJob((jobId) => getStudioEditorBackend().cancelFluxFill(jobId))}
                disabled={isCancelling}
                className="flex flex-1 items-center justify-center gap-1 rounded-lg border border-status-warning/30 bg-status-warning/10 px-2 py-1.5 text-[10px] text-status-warning"
              >
                <LoaderCircle size={11} className="animate-spin" /> {isCancelling ? "Cancelando e liberando GPU…" : "Cancelar geração"}
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void runGeneration()}
                disabled={!canGenerate}
                className="flex flex-1 items-center justify-center gap-1 rounded-lg bg-fuchsia-500 px-2 py-1.5 text-[10px] font-semibold text-white disabled:opacity-25"
              >
                <Sparkles size={11} /> Gerar em novas camadas
              </button>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
