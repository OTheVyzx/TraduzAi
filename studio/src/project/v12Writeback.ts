import type { StudioPage, StudioProject, StudioTextLayer } from "./studioProject";

type JsonRecord = Record<string, unknown>;
function record(value: unknown): JsonRecord { return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {}; }
function regionFromLayer(layer: StudioTextLayer, previous: JsonRecord | undefined): JsonRecord {
  const translation = record(previous?.translation);
  return { ...previous, id: layer.id, raw_ocr: layer.original, translation: { ...translation, text: layer.translated }, bbox: layer.bbox, style: layer.estilo, visible: layer.visible, order: layer.order, ...(previous ? {} : { studio_created: true }) };
}
function writePage(page: StudioPage, previous: JsonRecord | undefined): JsonRecord {
  const regions = Array.isArray(previous?.regions) ? previous.regions.map(record) : [];
  const layersById = new Map(page.text_layers.map((layer) => [layer.id, layer]));
  const nextRegions = regions.map((region) => {
    const layer = layersById.get(String(region.id ?? ""));
    return layer ? regionFromLayer(layer, region) : { ...region, studio_deleted: true };
  });
  for (const layer of page.text_layers) if (!regions.some((region) => region.id === layer.id)) nextRegions.push(regionFromLayer(layer, undefined));
  return { ...previous, number: page.numero, source_path: page.arquivo_original ?? previous?.source_path, rendered_path: page.arquivo_traduzido ?? previous?.rendered_path, regions: nextRegions };
}
export function writeV12Project(project: StudioProject, canonicalPages: JsonRecord[]): JsonRecord {
  const rawPages = Array.isArray(project.pages) ? project.pages.map(record) : [];
  const pages = canonicalPages.map((page) => writePage(page as StudioPage, rawPages.find((candidate) => Number(candidate.number ?? candidate.numero) === page.numero)));
  const legacy = record(project.legacy);
  return { ...project, schema_version: "12.0", studio_schema_version: "1.0", paginas: canonicalPages, pages, legacy: { ...legacy, paginas: canonicalPages } };
}
