import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import type { StudioIpcClient } from "../../backend/studioIpcClient";
import type { StudioProject, StudioTextLayer } from "../../project/studioProject";
import {
  loadRendererPreference,
  preferenceOwnerId,
  StudioRendererPreference,
  submitRendererPreference,
} from "../StudioRendererPreference";

const fixture = JSON.parse(readFileSync(
  fileURLToPath(new URL("../__fixtures__/renderer-preference-v1/comparison.json", import.meta.url)),
  "utf8",
));

const project: StudioProject = {
  app: "traduzai",
  versao: "2.0",
  studio_schema_version: "1.0",
  id: "project-001",
  capitulo: "57",
  project_revision: 12,
  paginas: [],
};

describe("StudioRendererPreference", () => {
  it("finds the canonical owner without inventing one from the layer id", () => {
    expect(preferenceOwnerId({ owner_id: "owner-007" } as unknown as StudioTextLayer)).toBe("owner-007");
    expect(preferenceOwnerId({ logical_owner_id: "owner-008" } as unknown as StudioTextLayer)).toBe("owner-008");
    expect(preferenceOwnerId({ id: "text-1" } as StudioTextLayer)).toBeNull();
  });

  it("loads and validates only a pending comparison returned for the exact owner", async () => {
    const readRendererPreferenceComparison = vi.fn().mockResolvedValue({
      owner_id: "fixture:renderer-preference:owner-001",
      project_revision: 12,
      state: "pending",
      comparison: fixture,
      artifact_base: "N:/p/preference",
    });
    const result = await loadRendererPreference({
      client: { readRendererPreferenceComparison } as unknown as StudioIpcClient,
      project,
      projectPath: "N:/p/project.json",
      ownerId: "fixture:renderer-preference:owner-001",
    });

    expect(result?.comparison.comparison_sha256).toBe(fixture.comparison_sha256);
    expect(result?.artifactBase).toBe("N:/p/preference");
    expect(readRendererPreferenceComparison).toHaveBeenCalledWith({
      projectPath: "N:/p/project.json",
      ownerId: "fixture:renderer-preference:owner-001",
      expectedRevision: 12,
    });
  });

  it("persists the synthetic fixture as system test evidence with all seven hashes", async () => {
    const submitReviewDecision = vi.fn().mockResolvedValue({
      decision_id: "decision-001",
      project_revision: 13,
      owner_status: "approved",
    });
    await submitRendererPreference({
      client: { submitReviewDecision } as unknown as StudioIpcClient,
      project,
      projectPath: "N:/p/project.json",
      comparison: fixture,
      choice: "A",
      recordedAt: "2026-09-27T15:30:00.000Z",
    });

    const call = submitReviewDecision.mock.calls[0][0];
    expect(call.decision).toMatchObject({
      actor_kind: "system",
      actor_id: "test/synthetic",
      preference_response: {
        choice: "A",
        training_eligible: false,
      },
    });
    expect(call.decision.evidence_sha256s).toHaveLength(7);
  });

  it("shows the compact A/B entry point only for a layer with a canonical owner", () => {
    const withOwner = renderToStaticMarkup(createElement(StudioRendererPreference, {
      project,
      projectPath: "N:/p/project.json",
      layer: { owner_id: "owner-001" } as unknown as StudioTextLayer,
    }));
    const withoutOwner = renderToStaticMarkup(createElement(StudioRendererPreference, {
      project,
      projectPath: "N:/p/project.json",
      layer: { id: "text-1" } as StudioTextLayer,
    }));

    expect(withOwner).toContain("Comparar A/B");
    expect(withOwner).toContain("Candidatas do Renderer");
    expect(withoutOwner).toBe("");
  });
});
