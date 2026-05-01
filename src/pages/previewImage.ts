import type { PageData } from "../lib/stores/appStore";

function normalizePreviewPath(path?: string | null) {
  if (!path) return null;
  return path.replace(/\\/g, "/");
}

function dedupePaths(paths: Array<string | null>) {
  const seen = new Set<string>();
  const output: string[] = [];
  for (const candidate of paths) {
    const normalized = normalizePreviewPath(candidate);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    output.push(normalized);
  }
  return output;
}

export function getPreviewImageCandidates(page: PageData, showOriginal: boolean) {
  const basePath = page.image_layers?.base?.path ?? page.arquivo_original ?? null;
  const renderedPath = page.image_layers?.rendered?.path ?? page.arquivo_traduzido ?? null;
  const inpaintPath = page.image_layers?.inpaint?.path ?? null;

  if (showOriginal) {
    return dedupePaths([basePath]);
  }

  return dedupePaths([renderedPath, inpaintPath, basePath]);
}

export function getPreviewToggleLabel(showOriginal: boolean) {
  return showOriginal ? "Ver traduzido" : "Ver original";
}
