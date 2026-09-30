import { describe, expect, it } from "vitest";
import type { LibraryWork } from "../libraryModel";
import {
  findChapterByProjectPath,
  findChapterForProjectRegistration,
  findWorkForProjectRegistration,
} from "../projectRegistration";

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

  it("localiza o capítulo existente para uma promoção sem trocar sua identidade", () => {
    const attached = work("attached", "Obra", "N:/obra/001/project.json");

    expect(findChapterByProjectPath(
      [attached],
      "n:\\obra\\001\\project.json",
    )).toMatchObject({
      work: { id: "attached" },
      chapter: { id: "attached-chapter" },
    });
  });

  it("reutiliza o capítulo de mesmo rótulo quando o backend promove o project.json", () => {
    const attached = work("attached", "Obra", "N:/runtime/seed/project.json");

    expect(findChapterForProjectRegistration(
      attached,
      "N:/runtime/promoted/project.json",
      "1",
    )?.id).toBe("attached-chapter");
  });
});
