import { describe, expect, it } from "vitest";
import {
  buildRecentProjectReviewBadge,
  deriveCompletionProjectStatus,
  isBlockedProjectReviewState,
  type PipelineCompleteEvent,
} from "../pipelineCompletion";

describe("pipeline completion review state", () => {
  it("treats export_gate BLOCK as done_blocked instead of approved", () => {
    const event: PipelineCompleteEvent = {
      success: true,
      job_id: "job-1",
      output_path: "N:/TraduzAI/data/works/run",
      completion_status: "blocked",
      export_gate: {
        status: "BLOCK",
        critical_issue_count: 1,
        critical_flag_count: 1,
        review_issue_count: 2,
        needs_review: true,
      },
      blocking_flags: ["visual_text_leak"],
      review_flags: ["ocr_suspect"],
    };

    const status = deriveCompletionProjectStatus(event, {
      output_review_state: "blocked_preview",
    });

    expect(status).toBe("done_blocked");
    expect(isBlockedProjectReviewState({ status, output_review_state: "blocked_preview" })).toBe(true);
    expect(buildRecentProjectReviewBadge({ status, critical_issue_count: 1 })).toEqual({
      label: "Bloqueado (1 issue critica)",
      tone: "error",
    });
  });
});
