import {
  chapterProgress,
  type ChapterWorkflowStatus,
  type LibraryChapter,
  type LibraryWork,
  type StudioLibrary,
} from "./libraryModel";

export interface HomeWorkMetrics {
  totalWorks: number;
  translatedWorks: number;
  editingWorks: number;
  reviewWorks: number;
}

export interface SelectedWorkChapterMetrics {
  totalChapters: number;
  translatedChapters: number;
  editingChapters: number;
  reviewChapters: number;
  completedPages: number;
  totalPages: number;
}

function hasWorkflowStatus(work: LibraryWork, status: ChapterWorkflowStatus): boolean {
  return work.chapters.some((chapter) => chapter.workflowStatus === status);
}

function isTranslatedWork(work: LibraryWork): boolean {
  return work.chapters.length > 0 && work.chapters.every((chapter) => chapter.workflowStatus === "completed");
}

function latestChapterEditTime(work: LibraryWork): number {
  return Math.max(0, ...work.chapters.map((chapter) => {
    const timestamp = chapter.lastOpenedAt ? Date.parse(chapter.lastOpenedAt) : Number.NaN;
    return Number.isFinite(timestamp) ? timestamp : 0;
  }));
}

export function selectHomeWorkMetrics(library: StudioLibrary): HomeWorkMetrics {
  return {
    totalWorks: library.works.length,
    translatedWorks: library.works.filter(isTranslatedWork).length,
    editingWorks: library.works.filter((work) => hasWorkflowStatus(work, "editing")).length,
    reviewWorks: library.works.filter((work) => hasWorkflowStatus(work, "review")).length,
  };
}

export function selectRecentWorks(library: StudioLibrary): LibraryWork[] {
  return [...library.works].sort((left, right) => {
    const difference = latestChapterEditTime(right) - latestChapterEditTime(left);
    return difference || left.title.localeCompare(right.title, "pt-BR", { sensitivity: "base" });
  });
}

export function selectSelectedWorkChapterMetrics(library: StudioLibrary): SelectedWorkChapterMetrics | null {
  const work = library.works.find((candidate) => candidate.id === library.selectedWorkId);
  if (!work) return null;

  const chapterHasStatus = (status: ChapterWorkflowStatus) => work.chapters.filter((chapter) => chapter.workflowStatus === status).length;
  const totalPages = work.chapters.reduce((total, chapter) => total + Math.max(0, chapter.pageCount ?? 0), 0);
  const completedPages = work.chapters.reduce((total, chapter) => {
    const pageCount = Math.max(0, chapter.pageCount ?? 0);
    return total + Math.round((chapterProgress(chapter) / 100) * pageCount);
  }, 0);

  return {
    totalChapters: work.chapters.length,
    translatedChapters: chapterHasStatus("completed"),
    editingChapters: chapterHasStatus("editing"),
    reviewChapters: chapterHasStatus("review"),
    completedPages,
    totalPages,
  };
}

export function selectChapterLastEditedAt(chapter: LibraryChapter): Date | null {
  if (!chapter.lastOpenedAt) return null;
  const timestamp = Date.parse(chapter.lastOpenedAt);
  return Number.isFinite(timestamp) ? new Date(timestamp) : null;
}
