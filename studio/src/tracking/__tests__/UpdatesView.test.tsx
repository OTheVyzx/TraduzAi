import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { LibraryWork } from "../../library/libraryModel";
import { UpdatesView } from "../UpdatesView";

const work: LibraryWork = {
  id: "work-1",
  title: "Obra acompanhada",
  aliases: [],
  publicationStatus: "hiatus",
  external: {
    mangaDexId: "manga-uuid",
    manualStatusOverride: "hiatus",
    tracking: {
      fetchedAt: "2026-07-22T10:00:00Z",
      expiresAt: "2026-07-22T10:30:00Z",
      lastError: "Sem conexao",
      snapshots: [{
        provider: "mangadex",
        providerId: "manga-uuid",
        title: "Obra acompanhada",
        status: "releasing",
        remoteChapterCount: 3,
        latestChapter: "10.5",
        coverUrl: null,
        siteUrl: "https://mangadex.org/title/manga-uuid",
        fetchedAt: "2026-07-22T10:00:00Z",
      }],
    },
  },
  chapters: [
    { id: "chapter-10", label: "10", projectPath: "C:/obra/10/project.json" },
  ],
};

describe("UpdatesView", () => {
  it("keeps expired offline metadata visible, shows manual conflicts and never offers downloads", () => {
    const html = renderToStaticMarkup(
      <UpdatesView
        open
        works={[work]}
        trackingLanguage="en"
        now={new Date("2026-07-22T12:00:00Z")}
        onClose={() => undefined}
        onOpenWork={() => undefined}
        onPersistWork={() => undefined}
      />,
    );

    expect(html).toContain("Desatualizado");
    expect(html).toContain("Última atualização");
    expect(html).toContain('dateTime="2026-07-22T10:00:00Z"');
    expect(html).toContain("10.5");
    expect(html).toContain("Status manual");
    expect(html).toContain("Conflito");
    expect(html).toContain("Abrir obra");
    expect(html).not.toContain("Baixar");
  });
});
