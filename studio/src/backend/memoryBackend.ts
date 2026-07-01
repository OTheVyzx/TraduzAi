import { importStudioProject, toTraduzAiV2Compat } from "../project/adapters";
import type { ImageLayerKey, StudioProject, StudioTextLayer } from "../project/studioProject";
import type { BitmapLayerKey, BitmapRegionConfig, EditorPagePayload, StudioEditorBackend } from "./editorBackend";

function cloneProject(project: StudioProject): StudioProject {
  return JSON.parse(JSON.stringify(project)) as StudioProject;
}

function normalizeStoredProject(project: StudioProject): StudioProject {
  return importStudioProject(toTraduzAiV2Compat(project)).project;
}

function pageAt(project: StudioProject, pageIndex: number) {
  const page = project.paginas[pageIndex];
  if (!page) throw new Error(`Pagina ${pageIndex + 1} nao encontrada`);
  return page;
}

function syncTextAliases(page: ReturnType<typeof pageAt>) {
  page.textos = page.text_layers;
}

function createTextLayer(index: number, bbox: [number, number, number, number]): StudioTextLayer {
  return {
    id: `studio-text-${crypto.randomUUID()}`,
    kind: "text",
    original: "",
    translated: "",
    traduzido: "",
    bbox,
    layout_bbox: bbox,
    style: {},
    estilo: {},
    visible: true,
    locked: false,
    order: index,
  };
}

export class MemoryStudioEditorBackend implements StudioEditorBackend {
  private projects = new Map<string, StudioProject>();

  constructor(initialProjects: Record<string, StudioProject> = {}) {
    for (const [path, project] of Object.entries(initialProjects)) {
      this.projects.set(path, normalizeStoredProject(project));
    }
  }

  putProject(projectPath: string, project: StudioProject) {
    this.projects.set(projectPath, normalizeStoredProject(project));
  }

  async loadProject(config: { project_path: string }): Promise<StudioProject> {
    const project = this.projects.get(config.project_path);
    if (!project) throw new Error(`Projeto nao encontrado: ${config.project_path}`);
    return cloneProject(project);
  }

  async saveProjectJson(config: { project_path: string; project_json: StudioProject }): Promise<void> {
    this.projects.set(config.project_path, normalizeStoredProject(config.project_json));
  }

  async loadEditorPage(config: { project_path: string; page_index: number }): Promise<EditorPagePayload> {
    const project = await this.loadProject({ project_path: config.project_path });
    const page = pageAt(project, config.page_index);
    return {
      project_file: `${config.project_path}/project.json`,
      project_dir: config.project_path,
      page_index: config.page_index,
      total_pages: project.paginas.length,
      page,
      project,
    };
  }

  async createEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layout_bbox: [number, number, number, number];
  }): Promise<StudioTextLayer> {
    const project = await this.loadProject({ project_path: config.project_path });
    const page = pageAt(project, config.page_index);
    const layer = createTextLayer(page.text_layers.length, config.layout_bbox);
    page.text_layers.push(layer);
    syncTextAliases(page);
    await this.saveProjectJson({ project_path: config.project_path, project_json: project });
    return layer;
  }

  async patchEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layer_id: string;
    patch: Record<string, unknown>;
  }): Promise<StudioTextLayer> {
    const project = await this.loadProject({ project_path: config.project_path });
    const page = pageAt(project, config.page_index);
    const index = page.text_layers.findIndex((layer) => layer.id === config.layer_id);
    if (index < 0) throw new Error(`Camada nao encontrada: ${config.layer_id}`);
    const next = {
      ...page.text_layers[index],
      ...config.patch,
    } as StudioTextLayer;
    if (typeof next.translated === "string" || typeof next.traduzido === "string") {
      const translated = next.translated || next.traduzido || "";
      next.translated = translated;
      next.traduzido = translated;
    }
    if (next.style || next.estilo) {
      const style = Object.keys(next.style ?? {}).length > 0 ? next.style : next.estilo;
      next.style = style ?? {};
      next.estilo = style ?? {};
    }
    page.text_layers[index] = next;
    syncTextAliases(page);
    await this.saveProjectJson({ project_path: config.project_path, project_json: project });
    return next;
  }

  async deleteEditorTextLayer(config: { project_path: string; page_index: number; layer_id: string }): Promise<void> {
    const project = await this.loadProject({ project_path: config.project_path });
    const page = pageAt(project, config.page_index);
    const next = page.text_layers.filter((layer) => layer.id !== config.layer_id);
    if (next.length === page.text_layers.length) throw new Error(`Camada nao encontrada: ${config.layer_id}`);
    page.text_layers = next.map((layer, index) => ({ ...layer, order: index }));
    syncTextAliases(page);
    await this.saveProjectJson({ project_path: config.project_path, project_json: project });
  }

  async setEditorLayerVisibility(config: {
    project_path: string;
    page_index: number;
    layer_kind: "image" | "text";
    layer_key?: ImageLayerKey | null;
    layer_id?: string | null;
    visible: boolean;
  }): Promise<void> {
    const project = await this.loadProject({ project_path: config.project_path });
    const page = pageAt(project, config.page_index);
    if (config.layer_kind === "image") {
      const key = config.layer_key;
      if (!key) throw new Error("layer_key e obrigatorio para camada de imagem");
      const layer = page.image_layers[key] ?? { key, path: null, locked: false, visible: true };
      page.image_layers[key] = { ...layer, visible: config.visible };
    } else {
      const layer = page.text_layers.find((item) => item.id === config.layer_id);
      if (!layer) throw new Error(`Camada nao encontrada: ${config.layer_id}`);
      layer.visible = config.visible;
      syncTextAliases(page);
    }
    await this.saveProjectJson({ project_path: config.project_path, project_json: project });
  }

  async updateBitmapLayer(config: BitmapRegionConfig): Promise<string> {
    const project = await this.loadProject({ project_path: config.project_path });
    const page = pageAt(project, config.page_index);
    const key: BitmapLayerKey = config.layer_key;
    const path = config.png_data;
    page.image_layers[key] = {
      key,
      path,
      visible: true,
      locked: page.image_layers[key]?.locked === true,
    };
    if (key === "rendered") page.arquivo_traduzido = path;
    await this.saveProjectJson({ project_path: config.project_path, project_json: project });
    return path;
  }
}
