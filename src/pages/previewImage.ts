import type { PageData } from "../lib/stores/appStore";

function isAbsoluteOrDirectPath(path: string) {
  return (
    /^[A-Za-z]:\//.test(path) ||
    path.startsWith("/") ||
    /^(data|blob|asset|file):/i.test(path) ||
    /^https?:\/\//i.test(path)
  );
}

function normalizePreviewPath(path?: string | null) {
  if (!path) return null;
  return path.replace(/\\/g, "/");
}

function projectBasePath(projectPath?: string | null) {
  return normalizePreviewPath(projectPath)?.replace(/\/project\.json$/i, "") ?? null;
}

function resolvePreviewPath(path: string | null, projectPath?: string | null) {
  const normalized = normalizePreviewPath(path);
  if (!normalized) return null;
  if (isAbsoluteOrDirectPath(normalized)) return normalized;
  const base = projectBasePath(projectPath);
  if (!base) return normalized;
  return `${base}/${normalized}`.replace(/\\/g, "/");
}

function dedupePaths(paths: Array<string | null>, projectPath?: string | null) {
  const seen = new Set<string>();
  const output: string[] = [];
  for (const candidate of paths) {
    const normalized = resolvePreviewPath(candidate, projectPath);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    output.push(normalized);
  }
  return output;
}

export function getPreviewImageCandidates(
  page: PageData,
  showOriginal: boolean,
  projectPath?: string | null,
  faithfulPreviewPath?: string | null,
) {
  const basePath = page.image_layers?.base?.path ?? page.arquivo_original ?? null;
  const renderedPath = page.image_layers?.rendered?.path ?? page.arquivo_traduzido ?? null;
  const inpaintPath = page.image_layers?.inpaint?.path ?? null;

  if (showOriginal) {
    return dedupePaths([basePath], projectPath);
  }

  return dedupePaths([faithfulPreviewPath ?? null, renderedPath, inpaintPath, basePath], projectPath);
}

export function getPreviewToggleLabel(showOriginal: boolean) {
  return showOriginal ? "Ver traduzido" : "Ver original";
}
