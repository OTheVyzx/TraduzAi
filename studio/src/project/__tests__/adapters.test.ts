import { describe, expect, it } from "vitest";
import { finalImagePathForPage, importStudioProject, toTraduzAiV2Compat } from "../adapters";

describe("studio project adapters", () => {
  it("imports legacy v1 pages and regenerates text aliases", () => {
    const result = importStudioProject({
      versao: "1.0",
      app: "TraduzAi",
      paginas: [
        {
          numero: 1,
          arquivo_original: "original/page-001.png",
          arquivo_traduzido: "translated/page-001.png",
          textos: [
            {
              id: "t1",
              bbox: [10, 20, 50, 80],
              texto: "HELLO",
              traduzido: "OLA",
              estilo: { fontSize: 24 },
              confidence: 0.9,
            },
          ],
        },
      ],
    });

    expect(result.kind).toBe("traduzai_v1");
    expect(result.project.studio_schema_version).toBe("1.0");
    expect(result.project.paginas[0].text_layers[0]).toMatchObject({
      id: "t1",
      original: "HELLO",
      translated: "OLA",
      traduzido: "OLA",
      ocr_confidence: 0.9,
      confianca_ocr: 0.9,
    });
    expect(result.project.paginas[0].image_layers.base?.path).toBe("original/page-001.png");
  });

  it("imports v12 projects through legacy.paginas when available", () => {
    const result = importStudioProject({
      schema_version: "12.0",
      legacy: {
        paginas: [
          {
            numero: 2,
            arquivo_original: "original/002.png",
            textos: [{ id: "legacy", bbox: [1, 2, 3, 4], texto: "A", traduzido: "B" }],
          },
        ],
      },
      pages: [],
    });

    expect(result.kind).toBe("v12_analysis_project");
    expect(result.project.paginas[0].numero).toBe(2);
    expect(result.project.paginas[0].text_layers[0].translated).toBe("B");
  });

  it("projects v12 pages regions when legacy pages are missing", () => {
    const result = importStudioProject({
      schema_version: 12,
      pages: [
        {
          source_path: "original/001.png",
          rendered_path: "rendered/001.png",
          regions: [
            {
              id: "r1",
              region_id: "region-1",
              reading_order: 3,
              bbox: [5, 6, 40, 60],
              raw_ocr: "YES",
              translation: { text: "SIM" },
              layout: { fontSize: 18 },
            },
          ],
        },
      ],
    });

    expect(result.project.paginas[0].text_layers[0]).toMatchObject({
      id: "r1",
      original: "YES",
      translated: "SIM",
      region_id: "region-1",
      reading_order: 3,
    });
    expect(result.project.paginas[0].text_layers[0].v12_region).toMatchObject({ region_id: "region-1" });
    expect(result.warnings.length).toBeGreaterThan(0);
  });

  it("exports v2 compatibility aliases", () => {
    const result = importStudioProject({
      versao: "2.0",
      paginas: [
        {
          numero: 1,
          image_layers: { base: { path: "base.png" }, rendered: { path: "rendered.png" } },
          text_layers: [{ id: "a", bbox: [0, 0, 1, 1], original: "A", translated: "B", style: {} }],
        },
      ],
    });

    const compat = toTraduzAiV2Compat(result.project) as { paginas: Array<Record<string, unknown>> };
    expect(compat.paginas[0].arquivo_original).toBe("base.png");
    expect(compat.paginas[0].arquivo_traduzido).toBe("rendered.png");
    expect(compat.paginas[0].textos).toEqual(compat.paginas[0].text_layers);
  });

  it("keeps final image fallback order compatible with site exports", () => {
    const result = importStudioProject({
      versao: "2.0",
      paginas: [
        {
          numero: 1,
          image_layers: {
            base: { path: "base.png" },
            inpaint: { path: "inpaint.png" },
            rendered: { path: "rendered.png" },
          },
        },
      ],
    });

    expect(finalImagePathForPage(result.project.paginas[0])).toBe("rendered.png");
  });

  it("round-trips desktop project fields without dropping editor/export metadata", () => {
    const desktopProject = {
      app: "traduzai",
      versao: "2.0",
      id: "desktop-project",
      obra: "Obra Desktop",
      capitulo: 12,
      idioma_origem: "en",
      idioma_destino: "pt-BR",
      engine_preset_id: "manhwa_manhua",
      _work_dir: "N:/TraduzAI/out/obra/cap-12",
      qa: { summary: { total: 1 } },
      estatisticas: { pages: 1 },
      paginas: [
        {
          numero: 1,
          arquivo_original: "originals/001.png",
          arquivo_traduzido: "translated/001.png",
          arquivo_final: "translated/001.png",
          rendered_path: "rendered/001.png",
          translated_path: "translated/001.png",
          image_layers: {
            base: { key: "base", path: "originals/001.png", visible: true, locked: true, checksum: "base-sha" },
            inpaint: { key: "inpaint", path: "images/001.png", visible: true, locked: false, opacity: 0.8 },
            rendered: { key: "rendered", path: "rendered/001.png", visible: true, locked: false, generated_at: "now" },
            mask: { key: "mask", path: "layers/masks/001.png", visible: false, locked: false, technical: true },
          },
          inpaint_blocks: [{ bbox: [10, 20, 40, 60], confidence: 0.7 }],
          process_overlays: [{ id: "overlay-1", bbox: [1, 2, 3, 4] }],
          text_layers: [
            {
              id: "region-1",
              bbox: [100, 120, 300, 220],
              layout_bbox: [100, 120, 300, 220],
              source_bbox: [95, 115, 305, 225],
              render_bbox: [102, 122, 298, 218],
              original: "HELLO",
              traduzido: "OLA",
              translated: "OLA",
              tipo: "fala",
              estilo: { fonte: "Comic Neue", tamanho: 32, cor: "#111111" },
              style: { fonte: "Comic Neue", tamanho: 32, cor: "#111111" },
              qa_flags: ["ocr_gibberish"],
              qa_actions: [{ flag_id: "ocr_gibberish", status: "ignored" }],
              glossary_hits: [{ term: "knight" }],
              normalization: { changed: true },
            },
          ],
          textos: [{ id: "stale-textos", bbox: [0, 0, 1, 1], traduzido: "STALE" }],
        },
      ],
    };

    const imported = importStudioProject(desktopProject);
    const compat = toTraduzAiV2Compat(imported.project) as {
      qa: unknown;
      estatisticas: unknown;
      paginas: Array<{
        arquivo_original: string | null;
        arquivo_traduzido: string | null;
        arquivo_final?: string;
        original_path?: string;
        rendered_path?: string;
        translated_path?: string;
        inpaint_path?: string | null;
        image_layers: Record<string, Record<string, unknown>>;
        text_layers: Array<Record<string, unknown>>;
        textos: Array<Record<string, unknown>>;
        inpaint_blocks?: unknown[];
        process_overlays?: unknown[];
      }>;
    };

    expect(compat.qa).toEqual(desktopProject.qa);
    expect(compat.estatisticas).toEqual(desktopProject.estatisticas);
    expect(compat.paginas[0].arquivo_original).toBe("originals/001.png");
    expect(compat.paginas[0].arquivo_traduzido).toBe("rendered/001.png");
    expect(compat.paginas[0].arquivo_final).toBe("images/001.png");
    expect(compat.paginas[0].inpaint_path).toBe("images/001.png");
    expect(compat.paginas[0].original_path).toBe("originals/001.png");
    expect(compat.paginas[0].rendered_path).toBe("rendered/001.png");
    expect(compat.paginas[0].translated_path).toBe("rendered/001.png");
    expect(compat.paginas[0].image_layers.base.checksum).toBe("base-sha");
    expect(compat.paginas[0].image_layers.rendered.generated_at).toBe("now");
    expect(compat.paginas[0].text_layers[0].qa_actions).toEqual([{ flag_id: "ocr_gibberish", status: "ignored" }]);
    expect(compat.paginas[0].text_layers[0].glossary_hits).toEqual([{ term: "knight" }]);
    expect(compat.paginas[0].text_layers[0].texto).toBe("HELLO");
    expect(compat.paginas[0].textos).toEqual(compat.paginas[0].text_layers);
    expect(compat.paginas[0].textos[0].id).toBe("region-1");
    expect(compat.paginas[0].inpaint_blocks).toEqual(desktopProject.paginas[0].inpaint_blocks);
    expect(compat.paginas[0].process_overlays).toEqual(desktopProject.paginas[0].process_overlays);
  });

  it("round-trips site-style pages that only expose rendered_path aliases", () => {
    const result = importStudioProject({
      app: "traduzai-site",
      versao: "2.0",
      source_path: "site://project",
      paginas: [
        {
          numero: 7,
          original_path: "uploads/original-007.webp",
          rendered_path: "exports/final-007.webp",
          text_layers: [
            {
              id: "site-layer",
              bbox: [20, 30, 120, 160],
              text: "RAW",
              translated: "TRADUZIDO",
              style: { fontFamily: "Comic Neue", fontSize: 28 },
              web_only_metadata: { review_state: "approved" },
            },
          ],
        },
      ],
    });

    const page = result.project.paginas[0];
    const compat = toTraduzAiV2Compat(result.project) as { paginas: Array<Record<string, unknown>> };

    expect(page.arquivo_original).toBe("uploads/original-007.webp");
    expect(page.arquivo_traduzido).toBe("exports/final-007.webp");
    expect(page.image_layers.base?.path).toBe("uploads/original-007.webp");
    expect(page.image_layers.rendered?.path).toBe("exports/final-007.webp");
    expect(finalImagePathForPage(page)).toBe("exports/final-007.webp");
    expect((compat.paginas[0].text_layers as Array<Record<string, unknown>>)[0].web_only_metadata).toEqual({
      review_state: "approved",
    });
  });

  it("uses editor/site bbox priority when only derived geometry is reliable", () => {
    const result = importStudioProject({
      versao: "2.0",
      paginas: [
        {
          numero: 1,
          text_layers: [
            {
              id: "render-box",
              bbox: [0, 0, 1, 1],
              source_bbox: [5, 5, 10, 10],
              balloon_bbox: [10, 20, 200, 240],
              layout_bbox: [20, 30, 210, 250],
              render_bbox: [30, 40, 220, 260],
              original: "A",
              translated: "B",
            },
            {
              id: "balloon-only",
              balloon_bbox: [50, 60, 260, 300],
              original: "C",
              translated: "D",
            },
          ],
        },
      ],
    });

    expect(result.project.paginas[0].text_layers[0].bbox).toEqual([30, 40, 220, 260]);
    expect(result.project.paginas[0].text_layers[1].bbox).toEqual([50, 60, 260, 300]);
  });
});
