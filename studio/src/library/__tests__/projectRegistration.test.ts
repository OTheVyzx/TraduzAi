import { describe, expect, it } from "vitest";
import type { LibraryWork } from "../libraryModel";
import { findWorkForProjectRegistration } from "../projectRegistration";

function work(id: string, title: string, projectPath?: string): LibraryWork {
  return {
    id,
    title,
    aliases: [],
    publicationStatus: "unknown",
    external: {},
    chapters: projectPath ? [{ id: `${id}-chapter`, label: "1", projectPath }] : [],
  };
}

describe("registro automático de projetos na biblioteca", () => {
  it("mantém a obra que já contém o caminho mesmo quando o título interno diverge", () => {
    const attached = work("attached", "Obra escolhida pelo usuário", "N:/obra/001/project.json");
    const titleMatch = work("title-match", "Título interno");

    expect(findWorkForProjectRegistration(
      [attached, titleMatch],
      "Título interno",
      "n:\\obra\\001\\project.json",
    )?.id).toBe("attached");
  });
});
