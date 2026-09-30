import { describe, expect, it } from "vitest";
import {
  selectHomeWorkMetrics,
  selectRecentWorks,
  selectSelectedWorkChapterMetrics,
} from "../librarySelectors";
import { createEmptyLibrary, type StudioLibrary } from "../libraryModel";

function libraryWithWorks(): StudioLibrary {
  return {
    ...createEmptyLibrary(),
    selectedWorkId: "editing",
    works: [
      {
        id: "finished",
        title: "Finalizada",
        aliases: [],
        publicationStatus: "completed",
        external: {},
        chapters: [{
          id: "finished-1", label: "1", projectPath: "memory://finished/1", pageCount: 10, completedPages: 10,
          workflowStatus: "completed", lastOpenedAt: "2026-09-01T09:00:00Z",
        }],
      },
      {
        id: "editing",
        title: "Em edi\u00e7\u00e3o",
        aliases: [],
        publicationStatus: "releasing",
        external: {},
        chapters: [
          { id: "editing-1", label: "1", projectPath: "memory://editing/1", pageCount: 10, completedPages: 10, workflowStatus: "completed" },
          { id: "editing-2", label: "2", projectPath: "memory://editing/2", pageCount: 8, completedPages: 3, workflowStatus: "editing", lastOpenedAt: "2026-09-02T10:00:00Z" },
        ],
      },
      {
        id: "review",
        title: "Em revis\u00e3o",
        aliases: [],
        publicationStatus: "releasing",
        external: {},
        chapters: [{
          id: "review-1", label: "1", projectPath: "memory://review/1", pageCount: 12, completedPages: 9,
          workflowStatus: "review", lastOpenedAt: "2026-09-01T20:00:00Z",
        }],
      },
    ],
  };
}

describe("librarySelectors", () => {
  it("calculates Home metrics by work rather than by chapter", () => {
    expect(selectHomeWorkMetrics(libraryWithWorks())).toEqual({
      totalWorks: 3,
      translatedWorks: 1,
      editingWorks: 1,
      reviewWorks: 1,
    });
  });

  it("orders recently edited works by their newest real chapter timestamp", () => {
    expect(selectRecentWorks(libraryWithWorks()).map((work) => work.id)).toEqual(["editing", "review", "finished"]);
  });

  it("keeps selected-work metrics at chapter level for Biblioteca", () => {
    expect(selectSelectedWorkChapterMetrics(libraryWithWorks())).toEqual({
      totalChapters: 2,
      translatedChapters: 1,
      editingChapters: 1,
      reviewChapters: 0,
      completedPages: 13,
      totalPages: 18,
    });
  });
});
