import type { EditorBackendApi, PageActionName, PageActionResult } from "../../../src/lib/editorBackend";
import type { ProcessRegionOverlay, TextEntry } from "../../../src/lib/stores/appStore";
import { projectApi } from "../projectApi";
import { editorApi } from "./editorApi";
import {
  assetPathToUrl,
  denormalizeWebProject,
  normalizeWebPage,
  normalizeWebProject,
  projectIdFromWebPath,
} from "./webProjectAdapter";

function projectId(config: { project_path: string }) {
  return projectIdFromWebPath(config.project_path);
}

function changedAssets(value: unknown): PageActionResult["changed_assets"] {
  if (!Array.isArray(value)) return ["project_json"];
  return value.filter((item): item is PageActionResult["changed_assets"][number] =>
    ["brush", "mask", "inpaint", "rendered", "preview", "project_json"].includes(String(item)),
  );
}

export const httpEditorBackend: EditorBackendApi = {
  async saveProjectJson(config) {
    const id = projectId(config);
    await projectApi.saveProject(id, denormalizeWebProject(id, config.project_json));
  },

  async loadEditorPage(config) {
    const id = projectId(config);
    const payload = await editorApi.loadEditorPage(id, config.page_index);
    return {
      project: normalizeWebProject(id, payload.project),
      page: normalizeWebPage(id, payload.page, payload.page_index),
      page_index: payload.page_index,
      total_pages: Array.isArray(payload.project?.paginas) ? payload.project.paginas.length : undefined,
      project_dir: config.project_path,
      project_file: config.project_path,
    };
  },

  async patchEditorTextLayer(config) {
    const id = projectId(config);
    const response = await editorApi.patchTextLayer(id, config.page_index, config.layer_id, config.patch);
    return normalizeWebPage(id, { text_layers: [response.layer] }, config.page_index).text_layers[0] as TextEntry;
  },

  async setEditorLayerVisibility(config) {
    const id = projectId(config);
    await editorApi.setLayerVisibility(id, {
      layer: config.layer_key ?? config.layer_id ?? "",
      visible: config.visible,
      page_index: config.page_index,
      layer_kind: config.layer_kind,
      layer_key: config.layer_key,
      layer_id: config.layer_id,
    });
  },

  async updateMaskRegion(config) {
    const id = projectId(config);
    const response = await editorApi.updateBitmapLayer(id, config.page_index, "mask", {
      width: config.width,
      height: config.height,
      brush_size: config.brush_size,
      clear: config.clear,
      erase: config.erase,
      strokes: config.strokes,
      png_data: config.png_data,
      dirty_bbox: config.dirty_bbox,
      op: config.erase ? "subtract" : "add",
    });
    return assetPathToUrl(id, response.asset_path) ?? response.url;
  },

  async updateBrushRegion(config) {
    const id = projectId(config);
    const response = await editorApi.updateBitmapLayer(id, config.page_index, "brush", {
      width: config.width,
      height: config.height,
      brush_size: config.brush_size,
      clear: config.clear,
      erase: config.erase,
      strokes: config.strokes,
      png_data: config.png_data,
      color: config.color,
      opacity: config.opacity,
      hardness: config.hardness,
      dirty_bbox: config.dirty_bbox,
    });
    return assetPathToUrl(id, response.asset_path) ?? response.url;
  },

  async updateRecoveryRegion(config) {
    const id = projectId(config);
    const response = await editorApi.updateBitmapLayer(id, config.page_index, "recovery", {
      width: config.width,
      height: config.height,
      brush_size: config.brush_size,
      clear: config.clear,
      erase: config.erase,
      strokes: config.strokes,
      png_data: config.png_data,
      color: config.color,
      opacity: config.opacity,
      hardness: config.hardness,
      dirty_bbox: config.dirty_bbox,
    });
    return assetPathToUrl(id, response.asset_path) ?? response.url;
  },

  async updateReinpaintRegion(config) {
    const id = projectId(config);
    const response = await editorApi.updateBitmapLayer(id, config.page_index, "recovery", {
      width: config.width,
      height: config.height,
      brush_size: config.brush_size,
      clear: config.clear,
      erase: config.erase,
      strokes: config.strokes,
      png_data: config.png_data,
      color: config.color,
      opacity: config.opacity,
      hardness: config.hardness,
      dirty_bbox: config.dirty_bbox,
    });
    return assetPathToUrl(id, response.asset_path) ?? response.url;
  },

  async writeMaskFromPng(config) {
    const id = projectId(config);
    const response = await editorApi.writeMaskFromPng(id, config.page_index, {
      png_data: config.png_data,
      op: config.op,
    });
    return assetPathToUrl(id, response.asset_path) ?? response.url;
  },

  async writeHealingMask(config) {
    const id = projectId(config);
    const response = await editorApi.updateBitmapLayer(id, config.page_index, "recovery", {
      png_data: config.png_data,
      dirty_bbox: config.bbox,
      op: "replace",
    });
    return assetPathToUrl(id, response.asset_path) ?? response.url;
  },

  async healInpaintRegion(config) {
    const id = projectId(config);
    const response = await editorApi.runPageAction(id, config.page_index, {
      action: "inpaint",
      region: {
        bbox: config.bbox,
        mask_path: config.mask_path,
      },
    });
    const inpaintPath =
      response.page?.image_layers?.inpaint?.path ??
      response.page?.arquivo_traduzido ??
      response.page?.rendered_path ??
      "";
    return {
      page_index: config.page_index,
      inpaint_path: assetPathToUrl(id, inpaintPath) ?? inpaintPath,
      before_inpaint_path: null,
      bbox: config.bbox,
    };
  },

  async renderPreviewPage(args) {
    const id = projectId(args);
    const response = await projectApi.renderPreview(id, args.page_index);
    return {
      output_path: assetPathToUrl(id, response.asset_path) ?? response.preview_url,
      renderer_backend: "web",
    };
  },

  async runPageActionWithOptionalMask(config) {
    const id = projectId(config);
    const response = await editorApi.runPageAction(id, config.page_index, {
      action: config.action,
      region: {
        bbox: config.bbox ?? undefined,
        mask_path: config.mask_path ?? undefined,
      },
    });
    return {
      action: config.action,
      mode: config.bbox || config.mask_path ? "regional" : "global",
      bbox: config.bbox ?? null,
      changed_assets: changedAssets(response.changed_assets),
      changed_layers: [],
      message: "ok",
    };
  },

  async runProcessRegion(config) {
    const id = projectId(config);
    const response = await editorApi.runPageAction(id, config.page_index, {
      action: "inpaint",
      region: {
        bbox: config.bbox,
        mask_path: config.mask_path ?? undefined,
      },
    });
    const textLayers = Array.isArray(response.page?.text_layers) ? response.page.text_layers : [];
    const overlay: ProcessRegionOverlay = {
      id: `web-process-${Date.now()}`,
      page_index: config.page_index,
      bbox: config.bbox,
      crop_path:
        response.page?.image_layers?.inpaint?.path ??
        response.page?.arquivo_traduzido ??
        config.mask_path ??
        "",
      text_layer_ids: textLayers.map((layer: { id?: unknown }) => layer.id).filter((value: unknown): value is string => typeof value === "string"),
      visible: true,
      locked: false,
      order: Array.isArray(response.page?.process_overlays) ? response.page.process_overlays.length : 0,
    };
    return {
      page_index: config.page_index,
      overlay,
      changed_assets: changedAssets(response.changed_assets),
      changed_layers: overlay.text_layer_ids,
      message: "ok",
    };
  },

  async retypesetPage(args) {
    const id = projectId(args);
    await editorApi.runPageAction(id, args.page_index, { action: "retypeset" });
    return "ok";
  },

  async detectPage(args) {
    return runSimpleAction(args, "detect");
  },

  async ocrPage(args) {
    return runSimpleAction(args, "ocr");
  },

  async translatePage(args) {
    return runSimpleAction(args, "translate");
  },

  async reinpaintPage(args) {
    return runSimpleAction(args, "inpaint");
  },

  async processBlock(config) {
    const id = projectId(config);
    await editorApi.runPageAction(id, config.page_index, {
      action: "process-block",
      block_id: config.block_id,
      mode: config.mode,
    });
    return "ok";
  },
};

async function runSimpleAction(
  args: { project_path: string; page_index: number },
  action: PageActionName,
) {
  const id = projectId(args);
  await editorApi.runPageAction(id, args.page_index, { action });
  return "ok";
}
