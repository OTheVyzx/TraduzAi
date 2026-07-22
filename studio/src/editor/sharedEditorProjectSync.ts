import type { PageData, Project } from "../../../src/lib/stores/appStore";

export interface ReconciledStudioEditorPage {
  currentPageIndex: number;
  currentPage: PageData | null;
  selectedLayerId: string | null;
}

export function reconcileStudioEditorPage(
  project: Project,
  currentPageIndex: number,
  selectedLayerId: string | null,
  workingPage: PageData | null = null,
  dirty = false,
): ReconciledStudioEditorPage {
  if (dirty && workingPage) {
    return { currentPageIndex, currentPage: workingPage, selectedLayerId };
  }
  const pageIndex = Math.max(0, Math.min(currentPageIndex, project.paginas.length - 1));
  const page = project.paginas[pageIndex] ?? null;
  return {
    currentPageIndex: pageIndex,
    currentPage: page,
    selectedLayerId: page?.text_layers.some((layer) => layer.id === selectedLayerId)
      ? selectedLayerId
      : null,
  };
}
