import type { EditorPagePayload, ProjectJson } from "../tauri";
import {
  useAppStore,
  type ImageLayerKey,
  type PageData,
  type ProcessRegionOverlay,
  type Project,
  type TextEntry,
} from "../stores/appStore";
import { getE2EFixtureProject, setE2EFixtureProject } from "./fixtureProject";

function clone<T>(value: T): T {
  return structuredClone(value);
}

function updatePage(pageIndex: number, updater: (page: PageData) => PageData) {
  const project = getE2EFixtureProject();
  const paginas = [...project.paginas];
  paginas[pageIndex] = updater(clone(paginas[pageIndex]));
  setE2EFixtureProject({ ...project, paginas });
}

let pipelineCompleteCallback: ((result: { success: boolean; output_path: string }) => void) | null = null;
const pipelineOutputs = new Map<string, Project>();

export const tauriMock = {
  async openFiles(): Promise<string | null> {
    return "e2e/source.zip";
  },

  async validateImport(): Promise<{
    valid: boolean;
    pages: number;
    has_project_json: boolean;
    error?: string;
  }> {
    return { valid: true, pages: getE2EFixtureProject().paginas.length, has_project_json: true };
  },

  async loadProjectJson(path?: string): Promise<ProjectJson> {
    const project = path ? pipelineOutputs.get(path) : null;
    return clone(project ?? getE2EFixtureProject()) as ProjectJson;
  },

  async saveProjectJson(config: { project_json: Project }): Promise<void> {
    setE2EFixtureProject(clone(config.project_json));
  },

  async loadEditorPage(config: { page_index: number }): Promise<EditorPagePayload> {
    const project = getE2EFixtureProject();
    const appProject = useAppStore.getState().project;
    const page = project.paginas[config.page_index] ?? appProject?.paginas[config.page_index];
    if (!page) throw new Error(`Pagina mock nao encontrada: ${config.page_index}`);
    return {
      project_file: "e2e/project-basic.json",
      project_dir: "e2e",
      page_index: config.page_index,
      total_pages: appProject?.paginas.length ?? project.paginas.length,
      page: clone(page),
    };
  },

  async patchEditorTextLayer(config: {
    page_index: number;
    layer_id: string;
    patch: Record<string, unknown>;
  }): Promise<TextEntry> {
    let patched: TextEntry | null = null;
    updatePage(config.page_index, (page) => {
      const text_layers = page.text_layers.map((layer) => {
        if (layer.id !== config.layer_id) return layer;
        const stylePatch = config.patch.style && typeof config.patch.style === "object"
          ? (config.patch.style as Partial<TextEntry["estilo"]>)
          : null;
        patched = {
          ...layer,
          traduzido: (config.patch.translated as string | undefined) ?? layer.traduzido,
          translated: (config.patch.translated as string | undefined) ?? layer.translated,
          bbox: (config.patch.bbox as TextEntry["bbox"] | undefined) ?? layer.bbox,
          layout_bbox: (config.patch.layout_bbox as TextEntry["bbox"] | undefined) ?? layer.layout_bbox,
          balloon_bbox: (config.patch.balloon_bbox as TextEntry["bbox"] | undefined) ?? layer.balloon_bbox,
          visible: (config.patch.visible as boolean | undefined) ?? layer.visible,
          locked: (config.patch.locked as boolean | undefined) ?? layer.locked,
          estilo: stylePatch ? { ...layer.estilo, ...stylePatch } : layer.estilo,
        };
        patched.style = patched.estilo;
        return patched;
      });
      return { ...page, text_layers, textos: text_layers };
    });
    if (!patched) throw new Error("Camada mock nao encontrada");
    return clone(patched);
  },

  async setEditorLayerVisibility(config: {
    page_index: number;
    layer_kind: "image" | "text";
    layer_key?: string | null;
    layer_id?: string | null;
    visible: boolean;
  }): Promise<void> {
    updatePage(config.page_index, (page) => {
      if (config.layer_kind === "text" && config.layer_id) {
        const text_layers = page.text_layers.map((layer) =>
          layer.id === config.layer_id ? { ...layer, visible: config.visible } : layer,
        );
        return { ...page, text_layers, textos: text_layers };
      }
      if (config.layer_kind === "image" && config.layer_key) {
        const key = config.layer_key as ImageLayerKey;
        const layer = page.image_layers?.[key];
        return {
          ...page,
          image_layers: {
            ...page.image_layers,
            [key]: {
              key,
              path: layer?.path ?? null,
              visible: config.visible,
              locked: layer?.locked ?? false,
            },
          },
        };
      }
      return page;
    });
  },

  async renderPreviewPage(): Promise<string> {
    return "/e2e-rendered-fresh.png";
  },

  async updateMaskRegion(): Promise<string> {
    return "/e2e-mask.png";
  },

  async updateBrushRegion(): Promise<string> {
    return "/e2e-brush.png";
  },

  async updateRecoveryRegion(): Promise<string> {
    return getE2EFixtureProject().paginas[0].image_layers?.inpaint?.path ?? "/e2e-recovery.png";
  },

  async updateReinpaintRegion(): Promise<string> {
    return getE2EFixtureProject().paginas[0].image_layers?.inpaint?.path ?? "/e2e-reinpaint.png";
  },

  async writeHealingMask(config?: { page_index?: number }): Promise<string> {
    const page = Number(config?.page_index ?? 0) + 1;
    return `/e2e/editor_cache/healing_masks/page-${String(page).padStart(4, "0")}/mask.png`;
  },

  async writeMaskFromPng(config?: { page_index?: number }): Promise<string> {
    const page = Number(config?.page_index ?? 0) + 1;
    return `/e2e/editor_cache/masks/page-${String(page).padStart(4, "0")}/lasso.png`;
  },

  async healInpaintRegion(config: {
    page_index: number;
    bbox: [number, number, number, number];
  }): Promise<{
    page_index: number;
    inpaint_path: string;
    before_inpaint_path: string | null;
    bbox: [number, number, number, number];
  }> {
    const before = getE2EFixtureProject().paginas[config.page_index].image_layers?.inpaint?.path ?? null;
    const path = `/e2e-healed-page-${config.page_index + 1}.png`;
    updatePage(config.page_index, (page) => ({
      ...page,
      image_layers: {
        ...page.image_layers,
        inpaint: {
          key: "inpaint",
          path,
          visible: true,
          locked: page.image_layers?.inpaint?.locked ?? false,
        },
      },
    }));
    return {
      page_index: config.page_index,
      inpaint_path: path,
      before_inpaint_path: before,
      bbox: config.bbox,
    };
  },

  async runPageActionWithOptionalMask(config: {
    page_index: number;
    action: "detect" | "ocr" | "translate" | "inpaint";
    bbox?: [number, number, number, number] | null;
    mask_path?: string | null;
    engine_preset_id?: string;
  }): Promise<{
    action: "detect" | "ocr" | "translate" | "inpaint";
    mode: "global" | "regional";
    bbox?: [number, number, number, number] | null;
    changed_assets: Array<"brush" | "mask" | "inpaint" | "rendered" | "preview" | "project_json">;
    changed_layers: string[];
    message: string;
  }> {
    if (config.action === "inpaint") {
      updatePage(config.page_index, (page) => ({
        ...page,
        image_layers: {
          ...page.image_layers,
          inpaint: {
            key: "inpaint",
            path: `/e2e-inpaint-page-${config.page_index + 1}.png`,
            visible: true,
            locked: page.image_layers?.inpaint?.locked ?? false,
          },
        },
      }));
    }
    return {
      action: config.action,
      mode: config.bbox ? "regional" : "global",
      bbox: config.bbox ?? null,
      changed_assets:
        config.action === "inpaint"
          ? ["project_json", "inpaint", "rendered"]
          : ["project_json", "rendered"],
      changed_layers: [],
      message: "ok",
    };
  },

  async runProcessRegion(config: {
    page_index: number;
    bbox: [number, number, number, number];
    mask_path?: string | null;
  }): Promise<{
    page_index: number;
    overlay: ProcessRegionOverlay;
    changed_assets: Array<"brush" | "mask" | "inpaint" | "rendered" | "preview" | "project_json">;
    changed_layers: string[];
    message: string;
  }> {
    const page = getE2EFixtureProject().paginas[config.page_index];
    const overlay: ProcessRegionOverlay = {
      id: `e2e-process-${Date.now()}`,
      page_index: config.page_index,
      bbox: config.bbox,
      crop_path: `/e2e-process-page-${config.page_index + 1}.png`,
      text_layer_ids: page.text_layers.map((layer) => layer.id).slice(0, 1),
      visible: true,
      locked: false,
      order: page.process_overlays?.length ?? 0,
    };
    updatePage(config.page_index, (current) => ({
      ...current,
      process_overlays: [...(current.process_overlays ?? []), overlay],
      image_layers: {
        ...current.image_layers,
        mask: current.image_layers?.mask
          ? { ...current.image_layers.mask, path: config.mask_path ?? current.image_layers.mask.path }
          : { key: "mask", path: config.mask_path ?? null, visible: true, locked: false },
      },
    }));
    return {
      page_index: config.page_index,
      overlay,
      changed_assets: ["project_json", "inpaint", "rendered"],
      changed_layers: overlay.text_layer_ids,
      message: "ok",
    };
  },

  async startPipeline(config?: { source_path?: string; obra?: string; capitulo?: number }): Promise<{ job_id: string }> {
    const capitulo = Number(config?.capitulo ?? getE2EFixtureProject().capitulo);
    const outputPath = `e2e/project-cap-${capitulo}.json`;
    const project = clone(getE2EFixtureProject());
    const outputProject: Project = {
      ...project,
      id: `${project.id}-cap-${capitulo}`,
      obra: config?.obra?.trim() || project.obra,
      capitulo,
      source_path: config?.source_path ?? project.source_path,
      output_path: outputPath,
      status: "done",
    };
    pipelineOutputs.set(outputPath, outputProject);
    window.setTimeout(() => {
      pipelineCompleteCallback?.({
        success: true,
        output_path: outputPath,
      });
    }, 25);
    return { job_id: "e2e-job" };
  },

  async onPipelineProgress(_callback?: unknown): Promise<() => void> {
    return () => {};
  },

  async onPipelineComplete(callback?: unknown): Promise<() => void> {
    if (typeof callback === "function") {
      pipelineCompleteCallback = callback as (result: { success: boolean; output_path: string }) => void;
    }
    return () => {
      pipelineCompleteCallback = null;
    };
  },
};
