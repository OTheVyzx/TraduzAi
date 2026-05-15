import Konva from "konva";
import type { PageData, TextEntry } from "./stores/appStore";
import { loadImageSource } from "./imageSource";
import { ensureEditorFontLoaded } from "./fonts";
import { EDITOR_TEXT_LINE_HEIGHT, fitEditorTextFontSize } from "../components/editor/stage/textFit";
import {
  fontFamilyFromStyle,
  fontStyleFromStyle,
  styleForLayer,
  textForLayer,
} from "../components/editor/stage/textLayerStyleUtils";
import { addStyledKonvaTextNodes } from "../components/editor/stage/konvaTextStyleRenderer";

type RenderPageOptions = {
  page: PageData;
  projectImageBasePath?: string | null;
  mimeType?: string;
  quality?: number;
};

function normalizePath(path: string) {
  return path.replace(/\\/g, "/");
}

function isAbsolutePath(path: string) {
  return /^[A-Za-z]:[\\/]/.test(path) || path.startsWith("/") || /^(data|blob|asset|file|https?):/i.test(path);
}

function projectBaseDir(baseDir?: string | null) {
  if (!baseDir) return "";
  return normalizePath(baseDir).replace(/\/project\.json$/i, "");
}

function joinProjectPath(baseDir: string | null | undefined, maybeRelative?: string | null) {
  if (!maybeRelative) return null;
  if (isAbsolutePath(maybeRelative)) return normalizePath(maybeRelative);
  const base = projectBaseDir(baseDir);
  return base ? `${base}/${maybeRelative}`.replace(/\\/g, "/") : maybeRelative;
}

function editingBaseImagePath(page: PageData, projectImageBasePath?: string | null) {
  return joinProjectPath(
    projectImageBasePath,
    page.image_layers?.inpaint?.path ?? page.image_layers?.base?.path ?? page.arquivo_original ?? null,
  );
}

function normalizeRotationDegrees(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  let normalized = numeric % 360;
  if (normalized > 180) normalized -= 360;
  if (normalized <= -180) normalized += 360;
  if (Math.abs(normalized) < 0.01) return 0;
  return Math.round(normalized * 100) / 100;
}

function bboxRect(entry: TextEntry) {
  const bbox = entry.layout_bbox ?? entry.bbox;
  const [x1, y1, x2, y2] = bbox;
  return {
    x: Number(x1) || 0,
    y: Number(y1) || 0,
    width: Math.max(1, (Number(x2) || 0) - (Number(x1) || 0)),
    height: Math.max(1, (Number(y2) || 0) - (Number(y1) || 0)),
  };
}

function loadHtmlImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("imagem indisponivel"));
    image.src = src;
  });
}

async function waitForStyledFont(fontFamily: string, fontSize: number, fontStyle: string) {
  await ensureEditorFontLoaded(fontFamily, fontSize, fontStyle);
}

function addTextLayer(layer: Konva.Layer, entry: TextEntry) {
  if (entry.visible === false) return;
  const rect = bboxRect(entry);
  const style = styleForLayer(entry);
  const text = textForLayer(entry);
  const fontFamily = fontFamilyFromStyle(style);
  const fontStyle = fontStyleFromStyle(style);
  const textBoxWidth = Math.max(1, rect.width - 16);
  const textBoxHeight = Math.max(1, rect.height - 12);
  const fontSize = fitEditorTextFontSize({
    text,
    fontFamily,
    fontStyle,
    maxFontSize: Math.max(8, style.tamanho),
    maxWidth: textBoxWidth,
    maxHeight: textBoxHeight,
  });

  const group = new Konva.Group({
    x: rect.x + rect.width / 2,
    y: rect.y + rect.height / 2,
    offsetX: rect.width / 2,
    offsetY: rect.height / 2,
    width: rect.width,
    height: rect.height,
    rotation: normalizeRotationDegrees(style.rotacao),
    listening: false,
  });
  addStyledKonvaTextNodes(group, {
    x: 8,
    y: 6,
    width: textBoxWidth,
    height: textBoxHeight,
    text,
    align: style.alinhamento,
    fontSize,
    fontFamily,
    fontStyle,
    lineHeight: EDITOR_TEXT_LINE_HEIGHT,
    style,
    listening: false,
  });
  layer.add(group);
}

export function shouldUseKonvaPreviewRenderer() {
  const raw = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env
    ?.VITE_TRADUZAI_KONVA_RENDER_PREVIEW;
  return raw !== "0";
}

export async function renderPageWithKonvaToDataUrl({
  page,
  projectImageBasePath,
  mimeType = "image/jpeg",
  quality = 0.92,
}: RenderPageOptions) {
  if (typeof document === "undefined") {
    throw new Error("Konva preview requer ambiente de navegador");
  }
  const basePath = editingBaseImagePath(page, projectImageBasePath);
  if (!basePath) throw new Error("pagina sem imagem base para preview Konva");

  const loaded = await loadImageSource(basePath, mimeType);
  const container = document.createElement("div");
  container.style.position = "fixed";
  container.style.left = "-100000px";
  container.style.top = "0";
  document.body.appendChild(container);

  try {
    const image = await loadHtmlImage(loaded.src);
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    const stage = new Konva.Stage({ container, width, height });
    const baseLayer = new Konva.Layer();
    baseLayer.add(new Konva.Image({ image, x: 0, y: 0, width, height, listening: false }));
    stage.add(baseLayer);

    const textLayer = new Konva.Layer();
    const textEntries = [...(page.text_layers ?? page.textos ?? [])].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
    await Promise.all(
      textEntries.map((entry) => {
        const style = styleForLayer(entry);
        return waitForStyledFont(fontFamilyFromStyle(style), style.tamanho, fontStyleFromStyle(style));
      }),
    );
    textEntries.forEach((entry) => addTextLayer(textLayer, entry));
    stage.add(textLayer);
    stage.draw();
    const dataUrl = stage.toDataURL({ mimeType, quality, pixelRatio: 1 });
    stage.destroy();
    return dataUrl;
  } finally {
    loaded.revoke?.();
    container.remove();
  }
}
