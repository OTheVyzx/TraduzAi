import {
  isTranslationStatus,
  type StudioPage,
  type StudioProject,
  type StudioTextLayer,
  type TranslationStatus,
} from "../project/studioProject";

export type TranslationQueueFilter = "all" | TranslationStatus;

export interface TranslationQueueQuery {
  query?: string;
  textClass?: string;
}

export interface TranslationQueueItem {
  pageIndex: number;
  pageNumber: number;
  blockIndex: number;
  layerId: string;
  original: string;
  translated: string;
  status: TranslationStatus;
  textClass: string;
  qaFlags: string[];
  notes?: string;
}

export interface TranslationProgress {
  total: number;
  completed: number;
  pending: number;
  translated: number;
  review: number;
  approved: number;
  percentage: number;
}

export function resolveTranslationStatus(layer: StudioTextLayer): TranslationStatus {
  if (isTranslationStatus(layer.translation_status)) return layer.translation_status;
  return layer.translated.trim().length > 0 ? "translated" : "pending";
}

export function buildTranslationQueue(
  project: StudioProject,
  filter: TranslationQueueFilter = "all",
  options: TranslationQueueQuery = {},
): TranslationQueueItem[] {
  const query = normalizeSearch(options.query ?? "");
  const requestedClass = normalizeSearch(options.textClass ?? "");
  return project.paginas.flatMap((page, pageIndex) => (
    page.text_layers.flatMap((layer, blockIndex) => {
      const status = resolveTranslationStatus(layer);
      if (filter !== "all" && status !== filter) return [];
      const textClass = String(layer.content_class ?? layer.tipo ?? "sem classe").trim() || "sem classe";
      const notes = typeof layer.translation_notes === "string" ? layer.translation_notes : undefined;
      const qaFlags = Array.isArray(layer.qa_flags)
        ? layer.qa_flags.filter((flag): flag is string => typeof flag === "string" && flag.trim().length > 0)
        : [];
      if (requestedClass && normalizeSearch(textClass) !== requestedClass) return [];
      if (query && !normalizeSearch([layer.original, layer.translated, notes, textClass, ...qaFlags].filter(Boolean).join(" ")).includes(query)) {
        return [];
      }
      return [{
        pageIndex,
        pageNumber: page.numero,
        blockIndex,
        layerId: layer.id,
        original: layer.original,
        translated: layer.translated,
        status,
        textClass,
        qaFlags,
        ...(notes ? { notes } : {}),
      }];
    })
  ));
}

function normalizeSearch(value: string) {
  return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("pt-BR").trim();
}

export function calculatePageTranslationProgress(page: StudioPage): TranslationProgress {
  return calculateLayerProgress(page.text_layers);
}

export function calculateTranslationProgress(project: StudioProject): TranslationProgress {
  return calculateLayerProgress(project.paginas.flatMap((page) => page.text_layers));
}

function calculateLayerProgress(layers: StudioTextLayer[]): TranslationProgress {
  const counts: Record<TranslationStatus, number> = {
    pending: 0,
    translated: 0,
    review: 0,
    approved: 0,
  };
  for (const layer of layers) counts[resolveTranslationStatus(layer)] += 1;

  const total = layers.length;
  const completed = total - counts.pending;
  return {
    total,
    completed,
    ...counts,
    percentage: total === 0 ? 0 : Math.round((completed / total) * 100),
  };
}
