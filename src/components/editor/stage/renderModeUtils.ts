import type { PageData } from "../../../lib/stores/appStore";
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
  _renderPreviewState: RenderPreviewCacheEntry,
) {
  if (!page) return null;
  if (viewMode === "original") return originalImagePath(page);
  if (viewMode === "inpainted") return editingBaseImagePath(page);

  // O editor é editável/WYSIWYG: o preview renderizado é mantido no cache para
  // exportação e para a tela de Preview, mas não deve substituir o canvas de
  // edição. Se o arquivo de preview falhar ao carregar, a página inteira some.
  return editingBaseImagePath(page);
}

export function isFaithfulPreviewMode(
  _viewMode: EditorViewMode,
  _renderPreviewState: RenderPreviewCacheEntry,
) {
  return false;
}
