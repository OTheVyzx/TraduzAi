import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { ExportDecision } from "../../backend/studioIpcClient";
import {
  StudioExportDecisionNotice,
  canApproveFinalReview,
  exportBlockReasons,
  requestStudioFinalExport,
} from "../StudioFinalExport";

const decision = (overrides: Partial<ExportDecision> = {}): ExportDecision => ({
  allowed: false,
  status: "BLOCK",
  gate_allowed: false,
  verified: false,
  completion_status: "incomplete",
  output_review_state: "pending",
  blocker_count: 1,
  critical_issue_count: 0,
  critical_flag_count: 0,
  review_issue_count: 2,
  review_flag_count: 0,
  issue_count: 0,
  ...overrides,
});

describe("Studio final export", () => {
  it("does not treat PASS alone as permission to export", () => {
    expect(exportBlockReasons(decision({ status: "PASS" }))).toEqual(expect.arrayContaining([
      "O backend não autorizou a exportação final",
      "O projeto ainda não foi verificado",
      "Há 1 blocker não resolvido",
    ]));
  });

  it("renders backend reasons and directs the user back to review", () => {
    const html = renderToStaticMarkup(
      <StudioExportDecisionNotice decision={decision()} />,
    );

    expect(html).toContain("Exportação final bloqueada");
    expect(html).toContain("2 achados de revisão");
    expect(html).toContain("Revise as pendências");
  });

  it("shows approval only when the backend decision is explicitly allowed", () => {
    const allowed = decision({
      allowed: true,
      status: "PASS",
      gate_allowed: true,
      verified: true,
      completion_status: "approved",
      output_review_state: "approved",
      blocker_count: 0,
      review_issue_count: 0,
    });

    expect(exportBlockReasons(allowed)).toEqual([]);
    expect(renderToStaticMarkup(<StudioExportDecisionNotice decision={allowed} />))
      .toContain("Exportação final autorizada");
  });

  it("offers human final review only after the real QA gate is clean", () => {
    const awaitingHuman = decision({
      status: "PASS",
      gate_allowed: true,
      blocker_count: 0,
      review_issue_count: 0,
    });
    expect(canApproveFinalReview(awaitingHuman)).toBe(true);
    expect(canApproveFinalReview(decision({
      status: "PASS",
      gate_allowed: true,
      blocker_count: 0,
      review_issue_count: 0,
      issue_count: 1,
    }))).toBe(false);
  });

  it("never asks for a destination or invokes final export while blocked", async () => {
    const chooseDestination = vi.fn();
    const exportFinal = vi.fn();
    const blocked = decision();
    const result = await requestStudioFinalExport({
      client: {
        decideExport: vi.fn().mockResolvedValue({ export_decision: blocked }),
        exportFinal,
      } as never,
      projectPath: "N:/p/project.json",
      expectedRevision: 4,
      chooseDestination,
      idempotencyKey: "export-r4",
    });

    expect(result).toEqual({ kind: "blocked", decision: blocked });
    expect(chooseDestination).not.toHaveBeenCalled();
    expect(exportFinal).not.toHaveBeenCalled();
  });

  it("rechecks the gate before invoking final export", async () => {
    const allowed = decision({
      allowed: true,
      status: "PASS",
      gate_allowed: true,
      verified: true,
      completion_status: "approved",
      output_review_state: "approved",
      blocker_count: 0,
      review_issue_count: 0,
    });
    const exportFinal = vi.fn().mockResolvedValue({
      project_revision: 5,
      export_manifest: {},
      publication_receipt: { receipt_id: "receipt-001" },
    });
    const result = await requestStudioFinalExport({
      client: {
        decideExport: vi.fn().mockResolvedValue({ export_decision: allowed }),
        exportFinal,
      } as never,
      projectPath: "N:/p/project.json",
      expectedRevision: 4,
      chooseDestination: vi.fn().mockResolvedValue("N:/exports/final.cbz"),
      idempotencyKey: "export-r4",
    });

    expect(result.kind).toBe("exported");
    expect(exportFinal).toHaveBeenCalledWith({
      projectPath: "N:/p/project.json",
      destination: "N:/exports/final.cbz",
      expectedRevision: 4,
      idempotencyKey: "export-r4",
    });
  });
});
