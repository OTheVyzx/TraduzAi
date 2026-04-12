import { create } from "zustand";
import type { LabReferencePreview, LabSnapshot } from "../tauri";

interface LabState {
  snapshot: LabSnapshot | null;
  referencePreview: LabReferencePreview | null;
  previewLoading: boolean;
  previewError: string | null;
  selectedChapter: number | null;
  selectedPage: number;
  highlightedProposalId: string | null;

  setSnapshot: (snapshot: LabSnapshot) => void;
  setReferencePreview: (preview: LabReferencePreview | null) => void;
  setPreviewLoading: (loading: boolean) => void;
  setPreviewError: (error: string | null) => void;
  setSelectedChapter: (chapter: number | null) => void;
  setSelectedPage: (page: number) => void;
  setHighlightedProposalId: (proposalId: string | null) => void;
  reset: () => void;
}

function resolveSelectedChapter(snapshot: LabSnapshot, currentChapter: number | null): number | null {
  const pool = snapshot.chapter_pairs.length > 0
    ? snapshot.chapter_pairs
    : snapshot.available_chapter_pairs;

  if (pool.length === 0) {
    return null;
  }

  if (currentChapter !== null) {
    const exists = pool.some((pair) => pair.chapter_number === currentChapter);
    if (exists) {
      return currentChapter;
    }
  }

  return pool[0]?.chapter_number ?? null;
}

export const useLabStore = create<LabState>((set) => ({
  snapshot: null,
  referencePreview: null,
  previewLoading: false,
  previewError: null,
  selectedChapter: null,
  selectedPage: 0,
  highlightedProposalId: null,

  setSnapshot: (snapshot) =>
    set((state) => ({
      snapshot,
      selectedChapter: resolveSelectedChapter(snapshot, state.selectedChapter),
      selectedPage:
        state.selectedChapter !== null &&
        state.selectedChapter !== resolveSelectedChapter(snapshot, state.selectedChapter)
          ? 0
          : state.selectedPage,
    })),
  setReferencePreview: (referencePreview) => set({ referencePreview }),
  setPreviewLoading: (previewLoading) => set({ previewLoading }),
  setPreviewError: (previewError) => set({ previewError }),
  setSelectedChapter: (selectedChapter) =>
    set({
      selectedChapter,
      selectedPage: 0,
      referencePreview: null,
      previewError: null,
    }),
  setSelectedPage: (selectedPage) =>
    set({
      selectedPage: Math.max(0, selectedPage),
      referencePreview: null,
      previewError: null,
    }),
  setHighlightedProposalId: (highlightedProposalId) => set({ highlightedProposalId }),
  reset: () =>
    set({
      snapshot: null,
      referencePreview: null,
      previewLoading: false,
      previewError: null,
      selectedChapter: null,
      selectedPage: 0,
      highlightedProposalId: null,
    }),
}));
