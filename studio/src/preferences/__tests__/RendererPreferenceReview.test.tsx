import { createElement } from "react";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  RendererPreferenceReview,
  createPreferenceResponse,
  parsePreferenceComparison,
  type PreferenceCandidate,
  type PreferenceComparison,
} from "../RendererPreferenceReview";

const hash = (character: string) => character.repeat(64);

function rasterSafety(overrides: Record<string, unknown> = {}) {
  return {
    schema: "traduzai.raster-safety.v1",
    status: "pass",
    bbox: [10, 20, 30, 40],
    ink_pixel_count: 120,
    alpha_sha256: hash("9"),
    authorized_body_sha256: hash("a"),
    protected_art_sha256: hash("b"),
    outside_authorized_body_px: 0,
    protected_art_overlap_px: 0,
    evidence_sha256: hash("c"),
    ...overrides,
  };
}

function candidate(overrides: Partial<PreferenceCandidate> = {}): PreferenceCandidate {
  return {
    schema: "traduzai.renderer-preference.v1",
    owner_id: "owner-001",
    target_text: "Mesmo texto",
    target_sha256: hash("a"),
    source_sha256: hash("b"),
    style_sha256: hash("c"),
    layout_plan_sha256: hash("d"),
    recipe_sha256: hash("e"),
    output_sha256: hash("f"),
    preview_ref: { relative_path: "review/candidate.png", sha256: hash("1") },
    context_ref: { relative_path: "review/context.png", sha256: hash("2") },
    metrics: { font_size: 31, raster_safety: rasterSafety() },
    hard_safety_passed: true,
    preference_profile: "uncalibrated",
    candidate_id: "renderer-candidate:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ...overrides,
  };
}

function comparison(overrides: Partial<PreferenceComparison> = {}): PreferenceComparison {
  const first = candidate();
  const second = candidate({
    candidate_id: "renderer-candidate:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    layout_plan_sha256: hash("3"),
    recipe_sha256: hash("4"),
    output_sha256: hash("5"),
    preview_ref: { relative_path: "review/candidate-b.png", sha256: hash("6") },
  });
  return {
    schema: "traduzai.renderer-preference.v1",
    candidates: [first, second],
    positions: { A: second.candidate_id, B: first.candidate_id },
    randomization_sha256: hash("7"),
    preference_profile: "uncalibrated",
    comparison_sha256: hash("8"),
    ...overrides,
  };
}

describe("RendererPreferenceReview", () => {
  it("consumes the Renderer-owned synthetic fixture without turning it into human training data", () => {
    const fixtureRoot = fileURLToPath(new URL("../__fixtures__/renderer-preference-v1", import.meta.url));
    const indexPath = resolve(fixtureRoot, "index.json");
    const comparisonPath = resolve(fixtureRoot, "comparison.json");
    const digest = (path: string) => createHash("sha256").update(readFileSync(path, "utf8").replace(/\r\n/g, "\n")).digest("hex");
    const index = JSON.parse(readFileSync(indexPath, "utf8")) as Record<string, any>;
    const rawComparison = JSON.parse(readFileSync(comparisonPath, "utf8"));
    const owner = index.owners["fixture:renderer-preference:owner-001"];

    expect(digest(indexPath)).toBe("40327aab0c065cf2dbe496053a248fdb245e6275f9d66232efd906be2a175a7a");
    expect(digest(comparisonPath)).toBe(owner.comparison_ref.sha256);
    expect(owner.state).toBe("pending");
    expect(index).toMatchObject({
      fixture_kind: "synthetic_contract_test",
      human_preference: false,
      training_eligible: false,
      actor: { kind: "system", id: "test/synthetic" },
    });

    const parsed = parsePreferenceComparison(rawComparison);
    expect(parsed.comparison_sha256).toBe(owner.comparison_sha256);
    for (const candidate of parsed.candidates) {
      for (const artifact of [candidate.preview_ref, candidate.context_ref]) {
        expect(digest(resolve(dirname(comparisonPath), artifact.relative_path))).toBe(artifact.sha256);
      }
    }
    expect(createPreferenceResponse({
      comparison: parsed,
      choice: "equivalent",
      actor: index.actor,
      recordedAt: "2026-09-27T15:12:00.000Z",
      trainingOptIn: true,
    })).toMatchObject({
      actor_kind: "system",
      actor_id: "test/synthetic",
      selected_candidate_id: null,
      training_eligible: false,
    });
  });

  it("accepts only a hard-safe same-owner same-target comparison with auditable positions", () => {
    expect(parsePreferenceComparison(comparison()).positions).toEqual({
      A: "renderer-candidate:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      B: "renderer-candidate:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    });
    expect(() => parsePreferenceComparison(comparison({
      candidates: [candidate(), candidate({
        candidate_id: "renderer-candidate:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        hard_safety_passed: false,
      })],
    }))).toThrow("segurança");
    expect(() => parsePreferenceComparison(comparison({ positions: { A: "missing", B: "also-missing" } })))
      .toThrow("posições");
    expect(() => parsePreferenceComparison(comparison({
      candidates: [candidate({ metrics: { font_size: 31 } }), candidate({
        candidate_id: "renderer-candidate:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        layout_plan_sha256: hash("3"),
        recipe_sha256: hash("4"),
        output_sha256: hash("5"),
        preview_ref: { relative_path: "review/candidate-b.png", sha256: hash("6") },
      })],
    }))).toThrow("raster");
    expect(() => parsePreferenceComparison(comparison({
      candidates: [candidate({ metrics: { raster_safety: rasterSafety({
        status: "review_required",
        protected_art_overlap_px: 1,
      }) } }), candidate({
        candidate_id: "renderer-candidate:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        layout_plan_sha256: hash("3"),
        recipe_sha256: hash("4"),
        output_sha256: hash("5"),
        preview_ref: { relative_path: "review/candidate-b.png", sha256: hash("6") },
      })],
    }))).toThrow("arte protegida");
  });

  it("records the displayed positions and stable selected candidate without inventing a human actor", () => {
    expect(createPreferenceResponse({
      comparison: comparison(),
      choice: "A",
      actor: { kind: "system", id: "test/synthetic" },
      recordedAt: "2026-09-27T14:30:00.000Z",
    })).toEqual({
      schema: "traduzai.renderer-preference-response.v1",
      comparison_sha256: hash("8"),
      choice: "A",
      selected_candidate_id: "renderer-candidate:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      displayed_candidate_ids: {
        A: "renderer-candidate:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        B: "renderer-candidate:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      },
      candidate_recipe_sha256s: [hash("4"), hash("e")],
      candidate_output_sha256s: [hash("5"), hash("f")],
      target_sha256: hash("a"),
      randomization_sha256: hash("7"),
      actor_kind: "system",
      actor_id: "test/synthetic",
      recorded_at: "2026-09-27T14:30:00.000Z",
      training_eligible: false,
    });
  });

  it("renders equal-size A/B previews, source context and all five honest choices", () => {
    const html = renderToStaticMarkup(createElement(RendererPreferenceReview, {
      comparison: comparison(),
      resolveArtifact: (path: string) => `asset://${path}`,
      actor: { kind: "human" as const, id: "local-user" },
      onSubmit: async () => undefined,
    }));

    expect(html).toContain("Comparar composição");
    expect(html).toContain("Contexto original");
    expect(html).toContain("Opção A");
    expect(html).toContain("Opção B");
    expect(html).toContain("Equivalentes");
    expect(html).toContain("Nenhuma");
    expect(html).toContain("Não tenho certeza");
    expect(html.match(/studio-preference-preview-frame/g)).toHaveLength(2);
  });
});
