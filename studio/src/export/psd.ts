import type { StudioPage, StudioProject } from "../project/studioProject";
import { writePsdUint8Array } from "ag-psd";
import type { Layer, LayerTextData, Psd, UnitsBounds } from "ag-psd";
import { serializeEngineData } from "ag-psd/dist-es/engineData";
import { encodeEngineData as encodeAgEngineData } from "ag-psd/dist-es/text";
import { loadImageSource } from "../../../src/lib/imageSource";

const MAX_PSD_SLICE_HEIGHT = 2000;

export interface PsdRasterLayer {
  name: string;
  pixels: Uint8Array;
  hidden?: boolean;
  textSpec?: PsdTextSpec;
  left?: number;
  top?: number;
  right?: number;
  bottom?: number;
}

export interface PsdTextSpec {
  text: string;
  x: number;
  y: number;
  width: number;
  height: number;
  fontName: string;
  fontSize: number;
  color: [number, number, number, number];
  vertical: boolean;
  justification: "left" | "center" | "right";
}

interface PsdPagePayload {
  width: number;
  height: number;
  layers: PsdRasterLayer[];
}

interface PsdExportPart {
  filename: string;
  bytes: Uint8Array;
}

export async function exportStudioPagePsd(project: StudioProject, pageIndex: number) {
  const parts = await exportStudioPagePsdParts(project, pageIndex);
  return parts[0]?.bytes ?? writePsdRasterLayers(1, 1, []);
}

export async function exportStudioPagePsdParts(project: StudioProject, pageIndex: number): Promise<PsdExportPart[]> {
  const page = project.paginas[pageIndex];
  if (!page) throw new Error("Pagina nao encontrada");

  const payload = await buildPsdPagePayload(page);
  const sliceCount = Math.ceil(payload.height / MAX_PSD_SLICE_HEIGHT);
  const baseName = project.obra ?? "traduzai-studio";
  return Array.from({ length: sliceCount }, (_, index) => {
    const top = index * MAX_PSD_SLICE_HEIGHT;
    const height = Math.min(MAX_PSD_SLICE_HEIGHT, payload.height - top);
    const suffix = sliceCount > 1 ? `-parte-${index + 1}` : "";
    return {
      filename: safeFilename(`${baseName}-pg-${pageIndex + 1}${suffix}.psd`),
      bytes: writePsdRasterLayers(payload.width, height, sliceLayers(payload.layers, payload.width, payload.height, top, height)),
    };
  });
}

async function buildPsdPagePayload(page: StudioPage): Promise<PsdPagePayload> {
  const baseSource = firstPageImagePath(page, "base") ?? firstPageImagePath(page, "rendered");
  const base = baseSource ? await loadImagePixels(baseSource) : null;
  const width = base?.width ?? inferPageWidth(page);
  const height = base?.height ?? inferPageHeight(page);
  const layers: PsdRasterLayer[] = [
    {
      name: "Original",
      pixels: base?.pixels ?? solidPixels(width, height, [255, 255, 255, 255]),
    },
  ];

  const inpaintLayer = page.image_layers.inpaint;
  const inpaintSource = inpaintLayer?.visible === false ? null : firstPageImagePath(page, "inpaint");
  const inpaint = inpaintSource ? await loadImagePixels(inpaintSource, width, height) : null;
  if (inpaint) {
    layers.push({ name: "Limpeza (Inpaint)", pixels: inpaint.pixels });
  }

  const maskSource = page.image_layers.mask?.visible === false ? null : page.image_layers.mask?.path;
  const mask = maskSource ? await loadImagePixels(maskSource, width, height).catch(() => null) : null;
  if (mask) layers.push({ name: "Mascara de Deteccao", pixels: mask.pixels, hidden: true });

  const brushSource = page.image_layers.brush?.visible === false ? null : page.image_layers.brush?.path;
  const brush = brushSource ? await loadImagePixels(brushSource, width, height).catch(() => null) : null;
  if (brush) layers.push({ name: "Pincel de Edicao", pixels: brush.pixels, hidden: true });

  for (const [index, textLayer] of page.text_layers.entries()) {
    const text = textLayer.translated ?? textLayer.traduzido ?? textLayer.original ?? "";
    if (text.trim()) {
      const textSpec = textSpecFromLayer(text, textLayer);
      layers.push({
        name: `Texto ${index + 1}`,
        pixels: transparentPixels(textSpec.width, textSpec.height),
        left: textSpec.x,
        top: textSpec.y,
        right: textSpec.x + textSpec.width,
        bottom: textSpec.y + textSpec.height,
        textSpec,
      });
    }
  }

  return { width, height, layers };
}

export function downloadStudioPagePsd(project: StudioProject, pageIndex: number): Promise<string | null> {
  return exportStudioPagePsdParts(project, pageIndex).then(async (parts) => {
    if (isTauriRuntime()) {
      const outputs: string[] = [];
      for (const part of parts) {
        outputs.push(await writePsdToProjectExports(project, part.filename, part.bytes));
      }
      return outputs.join(", ");
    }
    for (const part of parts) {
      const bytes = new Uint8Array(part.bytes).buffer;
      const url = URL.createObjectURL(new Blob([bytes], { type: "image/vnd.adobe.photoshop" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = part.filename;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 500);
    }
    return null;
  });
}

function isTauriRuntime() {
  return typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);
}

function normalizePath(path: string) {
  return path.replace(/\\/g, "/");
}

function projectDirectory(project: StudioProject) {
  const sourcePath = typeof project.source_path === "string" ? normalizePath(project.source_path) : "";
  if (!sourcePath) return null;
  return sourcePath.toLowerCase().endsWith(".json")
    ? sourcePath.slice(0, sourcePath.lastIndexOf("/"))
    : sourcePath;
}

async function writePsdToProjectExports(project: StudioProject, filename: string, bytes: Uint8Array): Promise<string> {
  const dir = projectDirectory(project);
  if (!dir) throw new Error("Caminho do projeto indisponivel para exportar PSD");
  const { invoke } = await import("@tauri-apps/api/core");
  const { writeFile } = await import("@tauri-apps/plugin-fs");
  const outputPath = await invoke<string>("studio_prepare_psd_export", {
    config: {
      project_path: project.source_path ?? dir,
      file_name: filename,
    },
  });
  await writeFile(outputPath, bytes);
  return outputPath.replace(/\\/g, "/");
}

export function writePsdRasterLayers(width: number, height: number, layers: PsdRasterLayer[]) {
  const psd: Psd = {
    width,
    height,
    imageData: { data: mergedComposite(width, height, layers), width, height },
    children: [...layers].reverse().map((layer) => agPsdLayerFromRaster(width, height, layer)),
  };
  return writePsdUint8Array(psd, {
    invalidateTextLayers: true,
    noBackground: true,
    trimImageData: false,
  });
}

function agPsdLayerFromRaster(width: number, height: number, layer: PsdRasterLayer): Layer {
  const bounds = boundsForLayer(width, height, layer);
  const psdLayer: Layer = {
    name: layer.name,
    top: bounds.top,
    left: bounds.left,
    bottom: bounds.bottom,
    right: bounds.right,
    hidden: layer.hidden,
    blendMode: "normal",
    opacity: 255,
    imageData: {
      data: layer.pixels,
      width: Math.max(0, bounds.right - bounds.left),
      height: Math.max(0, bounds.bottom - bounds.top),
    },
  };
  if (layer.textSpec) psdLayer.text = layerTextData(layer.textSpec);
  return psdLayer;
}

function firstPageImagePath(page: StudioPage, kind: "base" | "rendered" | "inpaint") {
  if (kind === "base") return page.arquivo_original ?? page.image_layers.base?.path ?? null;
  if (kind === "inpaint") {
    const value = page.image_layers.inpaint?.path ?? page.arquivo_final ?? page.inpaint_path ?? "";
    return String(value).trim() || null;
  }
  return page.arquivo_traduzido ?? page.image_layers.rendered?.path ?? page.image_layers.inpaint?.path ?? null;
}

async function loadImagePixels(source: string, targetWidth?: number, targetHeight?: number) {
  const image = await loadImage(source);
  const width = Math.max(1, Math.round(targetWidth ?? image.naturalWidth));
  const height = Math.max(1, Math.round(targetHeight ?? image.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D indisponivel");
  ctx.drawImage(image, 0, 0, width, height);
  return { width, height, pixels: new Uint8Array(ctx.getImageData(0, 0, width, height).data) };
}

function loadImage(source: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    let revoke: (() => void) | undefined;
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Falha ao carregar imagem para PSD: ${source}`));
    loadImageSource(source, "image/png")
      .then((loaded) => {
        revoke = loaded.revoke;
        image.src = loaded.src;
      })
      .catch((error) => reject(error));
    image.addEventListener("load", () => {
      if (revoke) window.setTimeout(revoke, 1000);
    }, { once: true });
  });
}

function inferPageWidth(page: StudioPage) {
  return Math.max(1, ...page.text_layers.map((layer) => Math.ceil(layer.bbox[2])));
}

function inferPageHeight(page: StudioPage) {
  return Math.max(1, ...page.text_layers.map((layer) => Math.ceil(layer.bbox[3])));
}

function solidPixels(width: number, height: number, rgba: [number, number, number, number]) {
  const pixels = new Uint8Array(width * height * 4);
  for (let index = 0; index < pixels.length; index += 4) {
    pixels[index] = rgba[0];
    pixels[index + 1] = rgba[1];
    pixels[index + 2] = rgba[2];
    pixels[index + 3] = rgba[3];
  }
  return pixels;
}

function transparentPixels(width: number, height: number) {
  return new Uint8Array(width * height * 4);
}

function sliceLayers(layers: PsdRasterLayer[], width: number, pageHeight: number, top: number, height: number) {
  const bottom = top + height;
  return layers.flatMap((layer) => {
    const bounds = boundsForLayer(width, pageHeight, layer);
    if (bounds.bottom <= top || bounds.top >= bottom) return [];
    const localTop = Math.max(bounds.top, top);
    const localBottom = Math.min(bounds.bottom, bottom);
    const sourceWidth = bounds.right - bounds.left;
    const sourceTop = localTop - bounds.top;
    const sourceHeight = localBottom - localTop;
    const pixels = cropPixels(layer.pixels, sourceWidth, sourceTop, sourceHeight);
    const textSpec = layer.textSpec
      ? {
          ...layer.textSpec,
          y: Math.max(0, layer.textSpec.y - top),
          height: Math.max(1, sourceHeight),
        }
      : undefined;
    return [{
      ...layer,
      pixels,
      top: localTop - top,
      bottom: localBottom - top,
      textSpec,
    }];
  });
}

function cropPixels(pixels: Uint8Array, width: number, top: number, height: number) {
  if (top === 0 && height * width * 4 === pixels.length) return pixels;
  const start = top * width * 4;
  const end = start + height * width * 4;
  return pixels.slice(start, end);
}

function textSpecFromLayer(text: string, layer: StudioPage["text_layers"][number]): PsdTextSpec {
  const [x, y, width, height] = textBoundsFromLayer(layer);
  const style = textStyleFromLayer(layer);
  return {
    text,
    x: Math.max(0, Math.round(x)),
    y: Math.max(0, Math.round(y)),
    width: Math.max(1, Math.round(width)),
    height: Math.max(1, Math.round(height)),
    fontName: style.fontName,
    fontSize: style.fontSize,
    color: parseColor(style.color),
    vertical: style.vertical,
    justification: style.justification,
  };
}

function textBoundsFromLayer(layer: StudioPage["text_layers"][number]): [number, number, number, number] {
  const raw = layer.render_bbox ?? layer.bbox ?? layer.layout_bbox;
  const [x, y, third, fourth] = raw.map((value) => Number(value)) as [number, number, number, number];
  if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(third) && Number.isFinite(fourth)) {
    if (third > x && fourth > y) {
      const rightWidth = third - x;
      const bottomHeight = fourth - y;
      if (rightWidth <= 1200 && bottomHeight <= 1200 && bottomHeight <= fourth) {
        return [x, y, rightWidth, bottomHeight];
      }
    }
    return [x, y, third, fourth];
  }
  return [0, 0, 1, 1];
}

function textStyleFromLayer(layer: StudioPage["text_layers"][number]) {
  const style = layer.style ?? {};
  const legacy = layer.estilo ?? {};
  const fontName =
    stringValue(style.fontFamily) ??
    stringValue(style.fonte) ??
    stringValue(legacy.fontFamily) ??
    stringValue(legacy.fonte) ??
    "ArialMT";
  const fontSize =
    finiteNumber(style.fontSize) ??
    finiteNumber(style.tamanho) ??
    finiteNumber(legacy.fontSize) ??
    finiteNumber(legacy.tamanho) ??
    24;
  const color =
    stringValue(style.color) ??
    stringValue(style.cor) ??
    stringValue(legacy.color) ??
    stringValue(legacy.cor) ??
    "#000000";
  const align = stringValue(style.align) ?? stringValue(style.alinhamento) ?? stringValue(legacy.align) ?? stringValue(legacy.alinhamento);
  const vertical = style.vertical === true || legacy.vertical === true;
  return {
    fontName: normalizeFontName(fontName),
    fontSize,
    color,
    vertical,
    justification: align === "left" || align === "right" ? align : "center" as "left" | "center" | "right",
  };
}

function stringValue(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function finiteNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeFontName(value: string) {
  return value.replace(/\.(ttf|otf)$/i, "").trim() || "ArialMT";
}

function parseColor(value: unknown): [number, number, number, number] {
  if (typeof value !== "string") return [0, 0, 0, 255];
  const hex = value.trim().replace(/^#/, "");
  if (/^[0-9a-f]{6}$/i.test(hex)) {
    return [Number.parseInt(hex.slice(0, 2), 16), Number.parseInt(hex.slice(2, 4), 16), Number.parseInt(hex.slice(4, 6), 16), 255];
  }
  return [0, 0, 0, 255];
}

function writeHeader(out: ByteWriter, width: number, height: number) {
  out.ascii("8BPS");
  out.u16(1);
  out.zeroes(6);
  out.u16(4);
  out.u32(height);
  out.u32(width);
  out.u16(8);
  out.u16(3);
}

function layerAndMaskInfo(width: number, height: number, layers: PsdRasterLayer[]) {
  const layerInfo = new ByteWriter();
  layerInfo.i16(layers.length > 0 ? -layers.length : 0);
  const recordLayers = [...layers].reverse();
  const layerBounds = recordLayers.map((layer) => boundsForLayer(width, height, layer));
  const channelPayloads = recordLayers.map((layer, index) => {
    const bounds = layerBounds[index];
    const expectedBytes = Math.max(0, bounds.right - bounds.left) * Math.max(0, bounds.bottom - bounds.top) * 4;
    if (layer.pixels.length !== expectedBytes) {
      throw new Error(`Camada PSD invalida (${layer.name}): ${layer.pixels.length} bytes para area de ${expectedBytes} bytes`);
    }
    return layerChannels(layer.pixels);
  });

  recordLayers.forEach((layer, index) => {
    const bounds = layerBounds[index];
    layerInfo.i32(bounds.top);
    layerInfo.i32(bounds.left);
    layerInfo.i32(bounds.bottom);
    layerInfo.i32(bounds.right);
    layerInfo.u16(4);
    [0, 1, 2, -1].forEach((channelId, channelIndex) => {
      layerInfo.i16(channelId);
      layerInfo.u32(2 + channelPayloads[index][channelIndex].length);
    });
    layerInfo.ascii("8BIM");
    layerInfo.ascii("norm");
    layerInfo.u8(255);
    layerInfo.u8(0);
    layerInfo.u8(layer.hidden ? 0x0a : 0x08);
    layerInfo.u8(0);
    const extra = layerExtraData(layer.name, layer.textSpec);
    layerInfo.u32(extra.length);
    layerInfo.bytes(extra);
  });

  for (const channels of channelPayloads) {
    for (const channel of channels) {
      layerInfo.u16(0);
      layerInfo.bytes(channel);
    }
  }
  layerInfo.padTo(4);

  const payload = layerInfo.toUint8Array();
  const full = new ByteWriter();
  full.u32(payload.length);
  full.bytes(payload);
  full.u32(0);
  return full.toUint8Array();
}

function boundsForLayer(width: number, height: number, layer: PsdRasterLayer) {
  const left = Math.max(0, Math.min(width, Math.round(layer.left ?? 0)));
  const top = Math.max(0, Math.min(height, Math.round(layer.top ?? 0)));
  const right = Math.max(left, Math.min(width, Math.round(layer.right ?? width)));
  const bottom = Math.max(top, Math.min(height, Math.round(layer.bottom ?? height)));
  return { left, top, right, bottom };
}

function layerExtraData(name: string, textSpec?: PsdTextSpec) {
  const extra = new ByteWriter();
  extra.u32(0);
  extra.u32(0);
  extra.pascalString(name, 4);
  if (textSpec) {
    const luni = new ByteWriter();
    luni.unicodeString(name);
    writeAdditionalInfoBlock(extra, "luni", luni.toUint8Array(), 4);
    writeAdditionalInfoBlock(extra, "TySh", tyshBody(textSpec), 2);
  }
  return extra.toUint8Array();
}

function writeAdditionalInfoBlock(out: ByteWriter, key: string, body: Uint8Array, alignment: number) {
  const padding = (alignment - (body.length % alignment)) % alignment;
  out.ascii("8BIM");
  out.ascii(key);
  out.u32(body.length + padding);
  out.bytes(body);
  out.zeroes(padding);
}

function tyshBody(spec: PsdTextSpec) {
  const left = 0;
  const top = 0;
  const right = spec.width;
  const bottom = spec.height;
  const body = new ByteWriter();
  body.i16(1);
  body.f64(1);
  body.f64(0);
  body.f64(0);
  body.f64(1);
  body.f64(0);
  body.f64(0);
  body.i16(50);
  writeVersionedDescriptor(body, textDescriptor(spec, left, top, right, bottom));
  body.i16(1);
  writeVersionedDescriptor(body, warpDescriptor(spec, left, top, right, bottom));
  body.f32(left);
  body.f32(top);
  body.f32(right);
  body.f32(bottom);
  return body.toUint8Array();
}

type DescriptorValue =
  | { type: "text"; value: string }
  | { type: "enum"; typeId: string; value: string }
  | { type: "integer"; value: number }
  | { type: "double"; value: number }
  | { type: "unitPixels"; value: number }
  | { type: "raw"; value: Uint8Array }
  | { type: "object"; value: DescriptorObject };

interface DescriptorObject {
  name: string;
  classId: string;
  items: Array<[string, DescriptorValue]>;
}

function textDescriptor(spec: PsdTextSpec, left: number, top: number, right: number, bottom: number): DescriptorObject {
  const bounds = boundsDescriptor("bounds", left, top, right, bottom);
  const boundingBox = boundsDescriptor("boundingBox", left, top, right, bottom);
  return {
    name: "",
    classId: "TxLr",
    items: [
      ["Txt ", { type: "text", value: spec.text }],
      ["textGridding", { type: "enum", typeId: "textGridding", value: "None" }],
      ["Ornt", { type: "enum", typeId: "Ornt", value: spec.vertical ? "Vrtc" : "Hrzn" }],
      ["AntA", { type: "enum", typeId: "Annt", value: "antiAliasSharp" }],
      ["bounds", { type: "object", value: bounds }],
      ["boundingBox", { type: "object", value: boundingBox }],
      ["TextIndex", { type: "integer", value: 0 }],
      ["EngineData", { type: "raw", value: encodeEngineData(spec) }],
    ],
  };
}

function warpDescriptor(spec: PsdTextSpec, left: number, top: number, right: number, bottom: number): DescriptorObject {
  return {
    name: "",
    classId: "warp",
    items: [
      ["warpStyle", { type: "enum", typeId: "warpStyle", value: "warpNone" }],
      ["warpValue", { type: "double", value: 0 }],
      ["warpPerspective", { type: "double", value: 0 }],
      ["warpPerspectiveOther", { type: "double", value: 0 }],
      ["warpRotate", { type: "enum", typeId: "Ornt", value: spec.vertical ? "Vrtc" : "Hrzn" }],
      ["bounds", { type: "object", value: boundsDescriptor("bounds", left, top, right, bottom) }],
    ],
  };
}

function boundsDescriptor(classId: string, left: number, top: number, right: number, bottom: number): DescriptorObject {
  return {
    name: "",
    classId,
    items: [
      ["Left", { type: "unitPixels", value: left }],
      ["Top ", { type: "unitPixels", value: top }],
      ["Rght", { type: "unitPixels", value: right }],
      ["Btom", { type: "unitPixels", value: bottom }],
    ],
  };
}

function writeVersionedDescriptor(out: ByteWriter, descriptor: DescriptorObject) {
  out.u32(16);
  writeDescriptorObject(out, descriptor);
}

function writeDescriptorObject(out: ByteWriter, descriptor: DescriptorObject) {
  out.unicodeStringWithPadding(descriptor.name);
  out.asciiOrClassId(descriptor.classId);
  out.u32(descriptor.items.length);
  for (const [key, value] of descriptor.items) {
    out.asciiOrClassId(key);
    writeDescriptorValue(out, value);
  }
}

function writeDescriptorValue(out: ByteWriter, value: DescriptorValue) {
  if (value.type === "text") {
    out.ascii("TEXT");
    out.unicodeStringWithPadding(value.value);
  } else if (value.type === "enum") {
    out.ascii("enum");
    out.asciiOrClassId(value.typeId);
    out.asciiOrClassId(value.value);
  } else if (value.type === "integer") {
    out.ascii("long");
    out.i32(value.value);
  } else if (value.type === "double") {
    out.ascii("doub");
    out.f64(value.value);
  } else if (value.type === "unitPixels") {
    out.ascii("UntF");
    out.ascii("#Pxl");
    out.f64(value.value);
  } else if (value.type === "raw") {
    out.ascii("tdta");
    out.u32(value.value.length);
    out.bytes(value.value);
  } else {
    out.ascii("Objc");
    writeDescriptorObject(out, value.value);
  }
}

function encodeEngineData(spec: PsdTextSpec) {
  return serializeEngineData(encodeAgEngineData(layerTextData(spec)));
}

function layerTextData(spec: PsdTextSpec): LayerTextData {
  const text = spec.text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const textLength = utf16Length(`${text.replace(/\n/g, "\r")}\r`);
  const bounds = textUnitsBounds(0, 0, spec.width, spec.height);
  const font = { name: spec.fontName.trim() || "ArialMT", script: 0, type: 0, synthetic: 0 };
  const style = {
    font,
    fontSize: spec.fontSize,
    fillColor: { r: spec.color[0], g: spec.color[1], b: spec.color[2], a: spec.color[3] },
    autoKerning: true,
    kerning: 0,
  };
  return {
    text,
    transform: [1, 0, 0, 1, spec.x, spec.y],
    antiAlias: "sharp",
    gridding: "none",
    orientation: spec.vertical ? "vertical" : "horizontal",
    index: 0,
    left: 0,
    top: 0,
    right: spec.width,
    bottom: spec.height,
    bounds,
    boundingBox: bounds,
    shapeType: "box",
    boxBounds: [0, 0, spec.width, spec.height],
    style,
    styleRuns: [{ length: textLength, style }],
    paragraphStyle: { justification: spec.justification },
    paragraphStyleRuns: [{ length: textLength, style: { justification: spec.justification } }],
  };
}

function textUnitsBounds(left: number, top: number, right: number, bottom: number): UnitsBounds {
  return {
    left: { units: "Pixels", value: left },
    top: { units: "Pixels", value: top },
    right: { units: "Pixels", value: right },
    bottom: { units: "Pixels", value: bottom },
  };
}

function encodeLegacyEngineData(spec: PsdTextSpec) {
  const text = normalizeEngineText(spec.text);
  const fontName = spec.fontName.trim() || "ArialMT";
  const fontIndex = fontName === "AdobeInvisFont" ? 0 : 1;
  const totalLength = utf16Length(text);
  const justification = spec.justification === "left" ? 0 : spec.justification === "right" ? 1 : 2;
  const color = `[ ${formatFloat(spec.color[3] / 255)} ${formatFloat(spec.color[0] / 255)} ${formatFloat(spec.color[1] / 255)} ${formatFloat(spec.color[2] / 255)} ]`;
  const paragraphProperties = paragraphPropertiesBlock(justification);
  const baseStyleSheet = baseStyleSheetBlock(fontIndex, spec.fontSize, color);
  const styleRunSheet = styleRunSheetBlock(fontIndex, spec.fontSize, color);
  const resources = resourceDict(fontName, paragraphProperties, baseStyleSheet);
  const engineData = `\n\n<<
\t/EngineDict <<
\t\t/Editor <<
\t\t\t/Text ${engineString(text)}
\t\t>>
\t\t/ParagraphRun <<
\t\t\t/DefaultRunData <<
\t\t\t\t/ParagraphSheet <<
\t\t\t\t\t/DefaultStyleSheet 0
\t\t\t\t\t/Properties <<
\t\t\t\t\t>>
\t\t\t\t>>
\t\t\t\t/Adjustments <<
\t\t\t\t\t/Axis [ 1 0 1 ]
\t\t\t\t\t/XY [ 0 0 ]
\t\t\t\t>>
\t\t\t>>
\t\t\t/RunArray [
\t\t\t\t<<
\t\t\t\t\t/ParagraphSheet <<
\t\t\t\t\t\t/DefaultStyleSheet 0
\t\t\t\t\t\t/Properties ${paragraphProperties}
\t\t\t\t\t>>
\t\t\t\t\t/Adjustments <<
\t\t\t\t\t\t/Axis [ 1 0 1 ]
\t\t\t\t\t\t/XY [ 0 0 ]
\t\t\t\t\t>>
\t\t\t\t>>
\t\t\t]
\t\t\t/RunLengthArray [ ${totalLength} ]
\t\t\t/IsJoinable 1
\t\t>>
\t\t/StyleRun <<
\t\t\t/DefaultRunData <<
\t\t\t\t/StyleSheet <<
\t\t\t\t\t/StyleSheetData ${styleRunSheet}
\t\t\t\t>>
\t\t\t>>
\t\t\t/RunArray [
\t\t\t\t<<
\t\t\t\t\t/StyleSheet <<
\t\t\t\t\t\t/StyleSheetData ${styleRunSheet}
\t\t\t\t\t>>
\t\t\t\t>>
\t\t\t]
\t\t\t/RunLengthArray [ ${totalLength} ]
\t\t\t/IsJoinable 2
\t\t>>
\t\t/GridInfo <<
\t\t\t/GridIsOn false
\t\t\t/ShowGrid false
\t\t\t/GridSize 18
\t\t\t/GridLeading 22
\t\t\t/GridColor <<
\t\t\t\t/Type 1
\t\t\t\t/Values [ 1 0 0 1 ]
\t\t\t>>
\t\t\t/GridLeadingFillColor <<
\t\t\t\t/Type 1
\t\t\t\t/Values [ 1 0 0 1 ]
\t\t\t>>
\t\t\t/AlignLineHeightToGridFlags false
\t\t>>
\t\t/AntiAlias 4
\t\t/UseFractionalGlyphWidths true
\t\t/Rendered <<
\t\t\t/Version 1
\t\t\t/Shapes <<
\t\t\t\t/WritingDirection ${spec.vertical ? 2 : 0}
\t\t\t\t/Children [
\t\t\t\t\t<<
\t\t\t\t\t\t/ShapeType 1
\t\t\t\t\t\t/Procession ${spec.vertical ? 1 : 0}
\t\t\t\t\t\t/Lines <<
\t\t\t\t\t\t\t/WritingDirection ${spec.vertical ? 2 : 0}
\t\t\t\t\t\t\t/Children [
\t\t\t\t\t\t\t]
\t\t\t\t\t\t>>
\t\t\t\t\t\t/Cookie <<
\t\t\t\t\t\t\t/Photoshop <<
\t\t\t\t\t\t\t\t/ShapeType 1
\t\t\t\t\t\t\t\t/BoxBounds [ 0 0 ${formatFloat(spec.width)} ${formatFloat(spec.height)} ]
\t\t\t\t\t\t\t\t/Base <<
\t\t\t\t\t\t\t\t\t/ShapeType 1
\t\t\t\t\t\t\t\t\t/TransformPoint0 [ 1 0 ]
\t\t\t\t\t\t\t\t\t/TransformPoint1 [ 0 1 ]
\t\t\t\t\t\t\t\t\t/TransformPoint2 [ 0 0 ]
\t\t\t\t\t\t\t\t>>
\t\t\t\t\t\t\t>>
\t\t\t\t\t\t>>
\t\t\t\t\t>>
\t\t\t\t]
\t\t\t>>
\t\t>>
\t>>
\t/ResourceDict ${resources}
\t/DocumentResources ${resources}
>>`;
  return latin1Bytes(engineData);
}

function paragraphPropertiesBlock(justification: number) {
  return `<<
\t\t\t/Justification ${justification}
\t\t\t/FirstLineIndent 0
\t\t\t/StartIndent 0
\t\t\t/EndIndent 0
\t\t\t/SpaceBefore 0
\t\t\t/SpaceAfter 0
\t\t\t/AutoHyphenate true
\t\t\t/HyphenatedWordSize 6
\t\t\t/PreHyphen 2
\t\t\t/PostHyphen 2
\t\t\t/ConsecutiveHyphens 8
\t\t\t/Zone 36
\t\t\t/WordSpacing [ .8 1 1.33 ]
\t\t\t/LetterSpacing [ 0 0 0 ]
\t\t\t/GlyphSpacing [ 1 1 1 ]
\t\t\t/AutoLeading 1.2
\t\t\t/LeadingType 0
\t\t\t/Hanging false
\t\t\t/Burasagari false
\t\t\t/KinsokuOrder 0
\t\t\t/EveryLineComposer false
\t\t>>`;
}

function baseStyleSheetBlock(fontIndex: number, fontSize: number, color: string) {
  return `<<
\t\t\t/Font ${fontIndex}
\t\t\t/FontSize ${formatFloat(fontSize)}
\t\t\t/FauxBold false
\t\t\t/FauxItalic false
\t\t\t/AutoLeading true
\t\t\t/Leading 0
\t\t\t/HorizontalScale 1
\t\t\t/VerticalScale 1
\t\t\t/Tracking 0
\t\t\t/AutoKerning true
\t\t\t/Kerning 0
\t\t\t/BaselineShift 0
\t\t\t/FontCaps 0
\t\t\t/FontBaseline 0
\t\t\t/Underline false
\t\t\t/Strikethrough false
\t\t\t/Ligatures true
\t\t\t/DLigatures false
\t\t\t/BaselineDirection 2
\t\t\t/Tsume 0
\t\t\t/StyleRunAlignment 2
\t\t\t/Language 0
\t\t\t/NoBreak false
\t\t\t/FillColor <<
\t\t\t\t/Type 1
\t\t\t\t/Values ${color}
\t\t\t>>
\t\t\t/StrokeColor <<
\t\t\t\t/Type 1
\t\t\t\t/Values [ 1 0 0 0 ]
\t\t\t>>
\t\t\t/FillFlag true
\t\t\t/StrokeFlag false
\t\t\t/FillFirst true
\t\t\t/YUnderline 1
\t\t\t/OutlineWidth 1
\t\t\t/CharacterDirection 0
\t\t\t/HindiNumbers false
\t\t\t/Kashida 1
\t\t\t/DiacriticPos 2
\t\t>>`;
}

function styleRunSheetBlock(fontIndex: number, fontSize: number, color: string) {
  return `<<
\t\t\t/Font ${fontIndex}
\t\t\t/FontSize ${formatFloat(fontSize)}
\t\t\t/FauxBold false
\t\t\t/FauxItalic false
\t\t\t/AutoKerning true
\t\t\t/Kerning 0
\t\t\t/FillColor <<
\t\t\t\t/Type 1
\t\t\t\t/Values ${color}
\t\t\t>>
\t\t>>`;
}

function resourceDict(fontName: string, paragraphProperties: string, baseStyleSheet: string) {
  const extraFont = fontName === "AdobeInvisFont" ? "" : `
\t\t\t<<
\t\t\t\t/Name ${engineString(fontName)}
\t\t\t\t/Script 0
\t\t\t\t/FontType 0
\t\t\t\t/Synthetic 0
\t\t\t>>`;
  return `<<
\t\t/KinsokuSet [
\t\t]
\t\t/MojiKumiSet [
\t\t]
\t\t/TheNormalStyleSheet 0
\t\t/TheNormalParagraphSheet 0
\t\t/ParagraphSheetSet [
\t\t\t<<
\t\t\t\t/Name ${engineString("Normal RGB")}
\t\t\t\t/DefaultStyleSheet 0
\t\t\t\t/Properties ${paragraphProperties}
\t\t\t>>
\t\t]
\t\t/StyleSheetSet [
\t\t\t<<
\t\t\t\t/Name ${engineString("Normal RGB")}
\t\t\t\t/StyleSheetData ${baseStyleSheet}
\t\t\t>>
\t\t]
\t\t/FontSet [
\t\t\t<<
\t\t\t\t/Name ${engineString("AdobeInvisFont")}
\t\t\t\t/Script 0
\t\t\t\t/FontType 0
\t\t\t\t/Synthetic 0
\t\t\t>>${extraFont}
\t\t]
\t\t/SuperscriptSize .583
\t\t/SuperscriptPosition .333
\t\t/SubscriptSize .583
\t\t/SubscriptPosition .333
\t\t/SmallCapSize .7
\t>>`;
}

function normalizeEngineText(text: string) {
  return `${text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/\n/g, "\r")}\r`;
}

function utf16Length(text: string) {
  let count = 0;
  for (const char of text) count += char.codePointAt(0)! > 0xffff ? 2 : 1;
  return count;
}

function engineString(text: string) {
  const bytes: number[] = [0xfe, 0xff];
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    bytes.push((code >>> 8) & 0xff, code & 0xff);
  }
  return `(${bytes.map((byte) => {
    if (byte === 0x28 || byte === 0x29 || byte === 0x5c) return `\\${String.fromCharCode(byte)}`;
    return String.fromCharCode(byte);
  }).join("")})`;
}

function formatFloat(value: number) {
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
}

function latin1Bytes(value: string) {
  const bytes = new Uint8Array(value.length);
  for (let index = 0; index < value.length; index += 1) bytes[index] = value.charCodeAt(index) & 0xff;
  return bytes;
}

function layerChannels(pixels: Uint8Array) {
  const channels = [new Uint8Array(pixels.length / 4), new Uint8Array(pixels.length / 4), new Uint8Array(pixels.length / 4), new Uint8Array(pixels.length / 4)];
  for (let source = 0, pixel = 0; source < pixels.length; source += 4, pixel += 1) {
    channels[0][pixel] = pixels[source];
    channels[1][pixel] = pixels[source + 1];
    channels[2][pixel] = pixels[source + 2];
    channels[3][pixel] = pixels[source + 3];
  }
  return channels;
}

function mergedComposite(width: number, height: number, layers: PsdRasterLayer[]) {
  const output = solidPixels(width, height, [255, 255, 255, 255]);
  for (const layer of layers) {
    if (layer.hidden) continue;
    alphaCompositeLayer(output, width, height, layer);
  }
  return output;
}

function alphaCompositeLayer(base: Uint8Array, width: number, height: number, layer: PsdRasterLayer) {
  const bounds = boundsForLayer(width, height, layer);
  const layerWidth = bounds.right - bounds.left;
  const layerHeight = bounds.bottom - bounds.top;
  for (let y = 0; y < layerHeight; y += 1) {
    for (let x = 0; x < layerWidth; x += 1) {
      const topIndex = (y * layerWidth + x) * 4;
      const baseIndex = ((bounds.top + y) * width + bounds.left + x) * 4;
      alphaCompositePixel(base, baseIndex, layer.pixels, topIndex);
    }
  }
}

function alphaCompositePixel(base: Uint8Array, baseIndex: number, top: Uint8Array, topIndex: number) {
  const alpha = top[topIndex + 3] / 255;
  if (alpha <= 0) return;
  base[baseIndex] = Math.round(top[topIndex] * alpha + base[baseIndex] * (1 - alpha));
  base[baseIndex + 1] = Math.round(top[topIndex + 1] * alpha + base[baseIndex + 1] * (1 - alpha));
  base[baseIndex + 2] = Math.round(top[topIndex + 2] * alpha + base[baseIndex + 2] * (1 - alpha));
  base[baseIndex + 3] = 255;
}

function writeImageData(out: ByteWriter, pixels: Uint8Array) {
  out.u16(0);
  for (const channel of layerChannels(pixels)) {
    out.bytes(channel);
  }
}

function safeFilename(name: string) {
  const cleaned = name.replace(/[<>:"/\\|?*\u0000-\u001f]+/g, "-").replace(/\s+/g, " ").trim();
  return cleaned.endsWith(".psd") ? cleaned : `${cleaned || "traduzai-studio"}.psd`;
}

class ByteWriter {
  private values: number[] = [];

  toUint8Array() {
    return new Uint8Array(this.values);
  }

  get length() {
    return this.values.length;
  }

  bytes(bytes: Uint8Array) {
    for (let index = 0; index < bytes.length; index += 1) {
      this.values.push(bytes[index]);
    }
  }

  ascii(value: string) {
    for (let index = 0; index < value.length; index += 1) this.values.push(value.charCodeAt(index));
  }

  zeroes(count: number) {
    for (let index = 0; index < count; index += 1) this.values.push(0);
  }

  u8(value: number) {
    this.values.push(value & 0xff);
  }

  u16(value: number) {
    this.values.push((value >>> 8) & 0xff, value & 0xff);
  }

  i16(value: number) {
    this.u16(value < 0 ? 0x10000 + value : value);
  }

  u32(value: number) {
    this.values.push((value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff);
  }

  i32(value: number) {
    this.u32(value < 0 ? 0x100000000 + value : value);
  }

  f32(value: number) {
    const bytes = new ArrayBuffer(4);
    new DataView(bytes).setFloat32(0, value, false);
    this.bytes(new Uint8Array(bytes));
  }

  f64(value: number) {
    const bytes = new ArrayBuffer(8);
    new DataView(bytes).setFloat64(0, value, false);
    this.bytes(new Uint8Array(bytes));
  }

  asciiOrClassId(value: string) {
    const treatAsClassId = value.length === 4 && !["warp", "time", "hold", "list"].includes(value);
    if (treatAsClassId) {
      this.i32(0);
      this.ascii(value);
    } else {
      this.i32(value.length);
      this.ascii(value);
    }
  }

  unicodeString(value: string) {
    const units = utf16Units(value);
    this.u32(units.length);
    for (const unit of units) this.u16(unit);
  }

  unicodeStringWithPadding(value: string) {
    const units = utf16Units(value);
    this.u32(units.length + 1);
    for (const unit of units) this.u16(unit);
    this.u16(0);
  }

  pascalString(value: string, padTo: number) {
    const bytes = new TextEncoder().encode(value.replace(/[^\x00-\x7f]/g, "?")).slice(0, 255);
    this.u8(bytes.length);
    this.bytes(bytes);
    this.padTo(padTo);
  }

  padTo(multiple: number) {
    while (this.length % multiple !== 0) this.u8(0);
  }
}

function utf16Units(value: string) {
  const units: number[] = [];
  for (let index = 0; index < value.length; index += 1) units.push(value.charCodeAt(index));
  return units;
}
