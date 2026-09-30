import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ReaderMangaDetails } from "../ReaderMangaDetails";

const manga = {
  sourceId: "20",
  url: "/necromancer",
  title: "Necromancer, the Ultimate Scourge!",
  thumbnailUrl: "https://covers.invalid/necromancer.webp",
  author: "AtlasScan",
  artist: "Studio Atlas",
  status: 1,
  description: "Uma calamidade ambulante.",
  genre: "Ação, Fantasia, Magia",
};

describe("ReaderMangaDetails", () => {
  it("renders remote metadata, explicit library action and chapter information", () => {
    const html = renderToStaticMarkup(createElement(ReaderMangaDetails, {
      manga,
      sourceName: "MangaFlix (PT-BR)",
      chapters: [{ url: "/chapter/302", name: "Capítulo 302", scanlator: "AtlasScan", dateUpload: Date.UTC(2026, 4, 5) }],
      inLibrary: false,
      busy: false,
      chaptersLoading: false,
      chaptersError: null,
      onBack: () => undefined,
      onToggleLibrary: () => undefined,
      onRetryChapters: () => undefined,
      renderChapterAction: () => createElement("button", { type: "button" }, "Baixar capítulo"),
    }));

    expect(html).toContain("Necromancer, the Ultimate Scourge!");
    expect(html).toContain("AtlasScan");
    expect(html).toContain("Em andamento");
    expect(html).toContain("Uma calamidade ambulante.");
    expect(html).toContain("Fantasia");
    expect(html).toContain("1 capítulo");
    expect(html).toContain("Adicionar à biblioteca");
    expect(html).toContain("Capítulo 302");
    expect(html).toContain("Baixar capítulo");
    expect(html).toContain("05/05/2026");
    expect(html).toContain('src="https://covers.invalid/necromancer.webp"');
  });

  it("keeps details visible when chapters fail and exposes retry", () => {
    const html = renderToStaticMarkup(createElement(ReaderMangaDetails, {
      manga,
      sourceName: "MangaFlix (PT-BR)",
      chapters: [],
      inLibrary: true,
      busy: false,
      chaptersLoading: false,
      chaptersError: "Fonte indisponível",
      onBack: () => undefined,
      onToggleLibrary: () => undefined,
      onRetryChapters: () => undefined,
    }));

    expect(html).toContain("Necromancer, the Ultimate Scourge!");
    expect(html).toContain("Fonte indisponível");
    expect(html).toContain("Tentar novamente");
    expect(html).toContain("Remover da biblioteca");
  });

  it("loads long chapter lists in bounded pages", () => {
    const chapters = Array.from({ length: 125 }, (_, index) => ({
      url: `/chapter/${index + 1}`,
      name: `Capítulo ${index + 1}`,
    }));
    const html = renderToStaticMarkup(createElement(ReaderMangaDetails, {
      manga,
      sourceName: "MangaFlix (PT-BR)",
      chapters,
      inLibrary: true,
      busy: false,
      chaptersLoading: false,
      chaptersError: null,
      onBack: () => undefined,
      onToggleLibrary: () => undefined,
      onRetryChapters: () => undefined,
    }));

    expect(html.match(/reader-manga-chapter-open/g)).toHaveLength(80);
    expect(html).toContain("Carregar mais 45 capítulos");
    expect(html).toContain("Capítulo 80");
    expect(html).not.toContain(">Capítulo 81<");
  });
});
