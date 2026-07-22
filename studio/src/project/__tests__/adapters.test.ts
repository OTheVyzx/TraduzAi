import { describe, expect, it } from "vitest";
import studioProjectSchema from "../../../schemas/studio_project.schema.json";
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

  it("derives manual translation status and preserves status, notes, and aliases on round trip", () => {
    const imported = importStudioProject({
      versao: "2.0",
      paginas: [{
        numero: 1,
        textos: [
          { id: "pending", bbox: [0, 0, 10, 10], texto: "WAIT", traduzido: "" },
          {
            id: "translated",
            bbox: [0, 10, 10, 20],
            texto: "DONE",
            translated: "",
            traduzido: "FEITO",
          },
          {
            id: "review",
            bbox: [0, 20, 10, 30],
            texto: "CHECK",
            traduzido: "REVISAR",
            translation_status: "review",
            translation_notes: "Confirmar o nome próprio",
          },
        ],
      }],
    });

    expect(imported.project.paginas[0].text_layers).toMatchObject([
      { id: "pending", translation_status: "pending" },
      { id: "translated", translation_status: "translated" },
      {
        id: "review",
        translation_status: "review",
        translation_notes: "Confirmar o nome próprio",
      },
    ]);

    const compat = toTraduzAiV2Compat(imported.project);
    const reopened = importStudioProject(compat).project.paginas[0];
    expect(reopened.text_layers[2]).toMatchObject({
      translated: "REVISAR",
      traduzido: "REVISAR",
      translation_status: "review",
      translation_notes: "Confirmar o nome próprio",
    });
    expect(reopened.textos).toEqual(reopened.text_layers);
  });

  it("declares additive manual translation fields in the Studio JSON schema", () => {
    const textLayerProperties = studioProjectSchema.$defs.textLayer.properties;

    expect(textLayerProperties.translation_status).toEqual({
      enum: ["pending", "translated", "review", "approved"],
    });
    expect(textLayerProperties.translation_notes).toEqual({ type: "string" });
    expect(studioProjectSchema.$defs.textLayer.required).not.toContain("translation_status");
    expect(studioProjectSchema.$defs.textLayer.required).not.toContain("translation_notes");
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

  it("round-trips the additive professional text style without flattening its passes", () => {
    const imported = importStudioProject({
      app: "traduzai",
      versao: "2.0",
      paginas: [
        {
          numero: 1,
          text_layers: [
            {
              id: "styled-text",
              bbox: [10, 20, 200, 120],
              original: "STYLE",
              translated: "ESTILO",
              style: {
                fonte: "Legacy.ttf",
                studio_style: {
                  version: "1.0",
                  typography: { fontFamily: "Anime Ace", fontSize: 38, fontWeight: 700 },
                  fills: [{ type: "solid", color: "#fefefe", opacity: 1 }],
                  strokes: [
                    { color: "#ffffff", width: 3, position: "outside" },
                    { color: "#000000", width: 8, position: "outside" },
                  ],
                  effects: {
                    dropShadows: [{ color: "#000000", offsetX: 4, offsetY: 6, blur: 3 }],
                  },
                },
              },
            },
          ],
        },
      ],
    });

    const reopened = importStudioProject(toTraduzAiV2Compat(imported.project));
    const style = reopened.project.paginas[0].text_layers[0].style.studio_style;

    expect(style?.typography).toMatchObject({ fontFamily: "Anime Ace", fontSize: 38, fontWeight: 700 });
    expect(style?.strokes).toHaveLength(2);
    expect(style?.effects?.dropShadows).toHaveLength(1);
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

  it("derives an additive scene for legacy image and text layers", () => {
    const result = importStudioProject({
      versao: "2.0",
      paginas: [
        {
          numero: 3,
          image_layers: {
            base: { path: "original/003.png", visible: true, locked: true, opacity: 0.75 },
            mask: { path: "layers/masks/003.png", visible: false, technical: true },
          },
          text_layers: [
            {
              id: "dialogue-1",
              bbox: [10, 20, 200, 120],
              original: "HELLO",
              translated: "OLA",
              visible: true,
              locked: false,
              order: 4,
            },
          ],
        },
      ],
    });

    const scene = result.project.paginas[0].studio_scene;

    expect(scene.version).toBe("1.0");
    expect(scene.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "image:base",
          kind: "raster",
          name: "Original",
          image_layer_key: "base",
          visible: true,
          locked: true,
          opacity: 0.75,
          blend_mode: "normal",
          parent_id: null,
        }),
        expect.objectContaining({
          id: "image:mask",
          kind: "mask",
          name: "Máscara",
          image_layer_key: "mask",
          visible: false,
        }),
        expect.objectContaining({
          id: "text:dialogue-1",
          kind: "text",
          name: "OLA",
          text_layer_id: "dialogue-1",
        }),
      ]),
    );
    expect(scene.roots).toEqual(expect.arrayContaining(["image:base", "image:mask", "text:dialogue-1"]));
    expect(scene.nodes.some((node) => node.id === "image:rendered")).toBe(false);
  });

  it("round-trips every Studio scene node kind and custom metadata", () => {
    const nodeKinds = ["raster", "text", "group", "mask", "generated", "adjustment", "fill"] as const;
    const nodes = nodeKinds.map((kind, order) => ({
      id: `custom:${kind}`,
      kind,
      name: `Custom ${kind}`,
      visible: kind !== "mask",
      locked: kind === "raster",
      opacity: 1 - order * 0.1,
      blend_mode: kind === "adjustment" ? "multiply" : "normal",
      parent_id: kind === "generated" ? "custom:group" : null,
      order,
      mask_ids: kind === "generated" ? ["custom:mask"] : [],
      metadata: kind === "generated" ? { prompt: "reconstruir o fundo", seed: 42 } : { preserved: true },
    }));
    const imported = importStudioProject({
      app: "traduzai",
      versao: "2.0",
      studio_schema_version: "1.0",
      paginas: [
        {
          numero: 1,
          image_layers: {},
          text_layers: [],
          studio_scene: {
            version: "1.0",
            roots: nodes.filter((node) => node.parent_id === null).map((node) => node.id),
            nodes,
            metadata: { workspace: "lettering" },
          },
        },
      ],
    });

    const compat = toTraduzAiV2Compat(imported.project);
    const reopened = importStudioProject(compat);
    const scene = reopened.project.paginas[0].studio_scene;

    expect(scene.nodes.map((node) => node.kind)).toEqual(nodeKinds);
    expect(scene.nodes.find((node) => node.kind === "generated")?.metadata).toEqual({
      prompt: "reconstruir o fundo",
      seed: 42,
    });
    expect(scene.nodes.find((node) => node.kind === "generated")?.parent_id).toBe("custom:group");
    expect(scene.metadata).toEqual({ workspace: "lettering" });
  });

  it("reconciles projected scene state without dropping free scene nodes", () => {
    const imported = importStudioProject({
      app: "traduzai",
      versao: "2.0",
      studio_schema_version: "1.0",
      paginas: [
        {
          numero: 1,
          image_layers: {
            base: { path: "base.png", visible: false, locked: true, opacity: 0.4 },
          },
          text_layers: [
            {
              id: "t1",
              bbox: [0, 0, 100, 100],
              original: "A",
              translated: "B",
              visible: false,
              locked: true,
              order: 2,
            },
          ],
          studio_scene: {
            version: "1.0",
            roots: ["free:group", "image:base", "text:t1"],
            nodes: [
              {
                id: "free:group",
                kind: "group",
                name: "Retoques",
                visible: true,
                locked: false,
                opacity: 1,
                blend_mode: "normal",
                parent_id: null,
                order: 0,
                mask_ids: [],
                metadata: { custom: true },
              },
              {
                id: "image:base",
                kind: "raster",
                name: "Original",
                image_layer_key: "base",
                visible: true,
                locked: false,
                opacity: 1,
                blend_mode: "normal",
                parent_id: null,
                order: 1,
                mask_ids: [],
                metadata: {},
              },
              {
                id: "text:t1",
                kind: "text",
                name: "B",
                text_layer_id: "t1",
                visible: true,
                locked: false,
                opacity: 1,
                blend_mode: "normal",
                parent_id: null,
                order: 2,
                mask_ids: [],
                metadata: {},
              },
            ],
          },
        },
      ],
    });

    const importedScene = imported.project.paginas[0].studio_scene;
    expect(importedScene.nodes.find((node) => node.id === "image:base")).toMatchObject({
      visible: false,
      locked: true,
      opacity: 0.4,
    });
    expect(importedScene.nodes.find((node) => node.id === "text:t1")).toMatchObject({
      visible: false,
      locked: true,
    });

    imported.project.paginas[0].image_layers.base!.opacity = 0.25;
    imported.project.paginas[0].text_layers[0].visible = true;
    const compat = toTraduzAiV2Compat(imported.project) as {
      paginas: Array<{ studio_scene: { nodes: Array<Record<string, unknown>> } }>;
    };

    expect(compat.paginas[0].studio_scene.nodes.find((node) => node.id === "image:base")).toMatchObject({
      opacity: 0.25,
    });
    expect(compat.paginas[0].studio_scene.nodes.find((node) => node.id === "text:t1")).toMatchObject({
      visible: true,
    });
    expect(compat.paginas[0].studio_scene.nodes.find((node) => node.id === "free:group")?.metadata).toEqual({
      custom: true,
    });
  });

  it("declares the additive scene contract in the Studio JSON schema", () => {
    const pageSchema = studioProjectSchema.properties.paginas.items;

    expect(pageSchema.required).toContain("studio_scene");
    expect(pageSchema.properties.studio_scene).toEqual({ $ref: "#/$defs/studioScene" });
    expect(studioProjectSchema.$defs.sceneNode.properties.kind.enum).toEqual([
      "raster",
      "text",
      "group",
      "mask",
      "generated",
      "adjustment",
      "fill",
    ]);
  });

  it("keeps scene-owned properties intrinsic when compatibility state is flattened", () => {
    const result = importStudioProject({
      app: "traduzai",
      versao: "2.0",
      studio_schema_version: "1.0",
      paginas: [
        {
          numero: 1,
          image_layers: {},
          text_layers: [
            {
              id: "t1",
              bbox: [0, 0, 100, 100],
              original: "A",
              translated: "B",
              visible: false,
              locked: true,
              opacity: 0.25,
            },
          ],
          studio_scene: {
            version: "1.0",
            roots: ["text:t1"],
            nodes: [
              {
                id: "text:t1",
                kind: "text",
                name: "B",
                visible: true,
                locked: false,
                opacity: 1,
                blend_mode: "normal",
                parent_id: null,
                order: 0,
                mask_ids: [],
                text_layer_id: "t1",
                metadata: { projected_from: "text_layers", scene_owned: true },
              },
            ],
          },
        },
      ],
    });

    expect(result.project.paginas[0].studio_scene.nodes[0]).toMatchObject({
      visible: true,
      locked: false,
      opacity: 1,
    });
    expect(result.project.paginas[0].text_layers[0]).toMatchObject({
      visible: false,
      locked: true,
      opacity: 0.25,
    });
  });
});
