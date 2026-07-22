import type { LibraryWork } from "./libraryModel";

function comparablePath(path: string): string {
  return path.trim().replace(/\\/g, "/").replace(/\/+$/, "").toLocaleLowerCase("en-US");
}

export function findWorkForProjectRegistration(
  works: readonly LibraryWork[],
  projectTitle: string,
  projectPath: string,
): LibraryWork | undefined {
  const normalizedProjectPath = comparablePath(projectPath);
  return works.find((work) => work.chapters.some(
    (chapter) => comparablePath(chapter.projectPath) === normalizedProjectPath,
  )) ?? works.find(
    (work) => work.title.localeCompare(projectTitle, "pt-BR", { sensitivity: "base" }) === 0,
  );
}
