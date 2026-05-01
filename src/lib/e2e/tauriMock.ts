import type { EditorPagePayload, ProjectJson } from "../tauri";
import type { ImageLayerKey, PageData, Project, TextEntry } from "../stores/appStore";
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

export const tauriMock = {
  async loadProjectJson(): Promise<ProjectJson> {
    return clone(getE2EFixtureProject()) as ProjectJson;
  },

  async saveProjectJson(config: { project_json: Project }): Promise<void> {
    setE2EFixtureProject(clone(config.project_json));
  },

  async loadEditorPage(config: { page_index: number }): Promise<EditorPagePayload> {
    const project = getE2EFixtureProject();
    return {
      project_file: "e2e/project-basic.json",
      project_dir: "e2e",
      page_index: config.page_index,
      total_pages: project.paginas.length,
      page: clone(project.paginas[config.page_index]),
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
};
