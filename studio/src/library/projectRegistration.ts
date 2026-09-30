import type { LibraryChapter, LibraryWork } from "./libraryModel";

function comparablePath(path: string): string {
  return path.trim().replace(/\\/g, "/").replace(/\/+$/, "").toLocaleLowerCase("en-US");
}

export function findChapterByProjectPath(
  works: readonly LibraryWork[],
  projectPath: string,
): { work: LibraryWork; chapter: LibraryChapter } | undefined {
  const normalizedProjectPath = comparablePath(projectPath);
  for (const work of works) {
    const chapter = work.chapters.find(
      (candidate) => comparablePath(candidate.projectPath) === normalizedProjectPath,
    );
    if (chapter) return { work, chapter };
  }
  return undefined;
}

export function findChapterForProjectRegistration(
  work: LibraryWork | undefined,
  projectPath: string,
  chapterLabel: string,
): LibraryChapter | undefined {
  if (!work) return undefined;
  const normalizedProjectPath = comparablePath(projectPath);
  return work.chapters.find(
    (chapter) => comparablePath(chapter.projectPath) === normalizedProjectPath,
  ) ?? work.chapters.find(
    (chapter) => chapter.label.localeCompare(chapterLabel, "pt-BR", { sensitivity: "base" }) === 0,
  );
}

export function findWorkForProjectRegistration(
  works: readonly LibraryWork[],
  projectTitle: string,
  projectPath: string,
): LibraryWork | undefined {
  return findChapterByProjectPath(works, projectPath)?.work ?? works.find(
    (work) => work.title.localeCompare(projectTitle, "pt-BR", { sensitivity: "base" }) === 0,
  );
}
