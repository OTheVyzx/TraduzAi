import { describe, expect, it } from "vitest";
import { importStudioProject, toTraduzAiV2Compat } from "../adapters";
describe("v12 Studio writeback", () => {
  it("prefers Studio paginas and preserves unknown pipeline metadata across a round trip", () => {
    const fixture = { schema_version: "12.0", pages: [{ number: 1, regions: [{ id: "r1", raw_ocr: "old", translation: { text: "old" }, bbox: [1, 2, 3, 4], pipeline_only_metadata: { qa: "keep" } }] }], legacy: { paginas: [{ numero: 1, textos: [{ id: "r1", texto: "stale", traduzido: "stale" }] }] } };
    const once = importStudioProject(fixture).project;
    once.paginas[0].text_layers[0].original = "OCR manual";
    once.paginas[0].text_layers[0].translated = "Tradução manual";
    once.paginas[0].text_layers[0].traduzido = "Tradução manual";
    const raw = toTraduzAiV2Compat(once) as { pages: Array<{ regions: Array<Record<string, unknown>> }> };
    const twice = importStudioProject(raw).project;
    expect(twice.paginas[0].text_layers[0]).toMatchObject({ original: "OCR manual", translated: "Tradução manual" });
    expect(raw.pages[0].regions[0].pipeline_only_metadata).toEqual({ qa: "keep" });
  });
});
