import type { ImageLayerKey, PageData } from "../../../lib/stores/appStore";
import type { EditorViewMode, RenderPreviewCacheEntry } from "../../../lib/stores/editorStore";

export function editingBaseImagePath(page: PageData | null | undefined) {
  if (!page) return null;
  return (
    page.image_layers?.inpaint?.path ??
    page.image_layers?.base?.path ??
    page.arquivo_original ??
    null
  );
}

export function originalImagePath(page: PageData | null | undefined) {
  if (!page) return null;
  return page.image_layers?.base?.path ?? page.arquivo_original ?? null;
}

export function displayImagePathForMode(
  page: PageData | null | undefined,
  viewMode: EditorViewMode,
  renderPreviewState: RenderPreviewCacheEntry,
  selectedImageLayerKey: ImageLayerKey | null = null,
) {
  if (!page) return null;
  if (
    selectedImageLayerKey === "base" ||
    selectedImageLayerKey === "inpaint" ||
    selectedImageLayerKey === "rendered"
  ) {
    return page.image_layers?.[selectedImageLayerKey]?.path ?? null;
  }
  if (viewMode === "original") return originalImagePath(page);
  if (viewMode === "inpainted") return editingBaseImagePath(page);

  if (renderPreviewState.status === "fresh" && renderPreviewState.previewPath) {
    return renderPreviewState.previewPath;
  }

  return editingBaseImagePath(page);
}

export function isFaithfulPreviewMode(
  viewMode: EditorViewMode,
  renderPreviewState: RenderPreviewCacheEntry,
) {
  return viewMode === "translated" && renderPreviewState.status === "fresh" && !!renderPreviewState.previewPath;
}

export function isBitmapInspectionLayer(layerKey: ImageLayerKey | null) {
  return layerKey === "base" || layerKey === "inpaint" || layerKey === "rendered";
}
