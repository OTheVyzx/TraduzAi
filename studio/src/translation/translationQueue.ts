import {
  isTranslationStatus,
  type StudioPage,
  type StudioProject,
  type StudioTextLayer,
  type TranslationStatus,
} from "../project/studioProject";

export type TranslationQueueFilter = "all" | TranslationStatus;

export interface TranslationQueueItem {
  pageIndex: number;
  pageNumber: number;
  blockIndex: number;
  layerId: string;
  original: string;
  translated: string;
  status: TranslationStatus;
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
): TranslationQueueItem[] {
  return project.paginas.flatMap((page, pageIndex) => (
    page.text_layers.flatMap((layer, blockIndex) => {
      const status = resolveTranslationStatus(layer);
      if (filter !== "all" && status !== filter) return [];
      return [{
        pageIndex,
        pageNumber: page.numero,
        blockIndex,
        layerId: layer.id,
        original: layer.original,
        translated: layer.translated,
        status,
        ...(typeof layer.translation_notes === "string" ? { notes: layer.translation_notes } : {}),
      }];
    })
  ));
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
