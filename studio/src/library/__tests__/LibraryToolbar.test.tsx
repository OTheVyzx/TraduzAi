import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LibraryToolbar } from "../LibraryToolbar";

describe("LibraryToolbar", () => {
  it("exposes Updates as an icon-only action in the upper toolbar", () => {
    const html = renderToStaticMarkup(
      <LibraryToolbar
        title="Solo Max-Level Newbie"
        chapterCount={1}
        query=""
        view="grid"
        thumbnailSize={176}
        onQueryChange={() => undefined}
        onSetView={() => undefined}
        onSetThumbnailSize={() => undefined}
        onOpenUpdates={() => undefined}
      />,
    );

    expect(html).toContain('class="studio-library-updates-trigger"');
    expect(html).toContain('aria-label="Atualizações"');
    expect(html).toContain('title="Atualizações"');
    expect(html).not.toContain(">Atualizações</button>");
  });
});
