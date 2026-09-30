import { describe, expect, it } from "vitest";

import { createStudioIpcClient, type StudioIpcTransport } from "../studioIpcClient";
import type { ReviewDecision } from "../studioIpcV1";

function recordingTransport(result: unknown = {}) {
  const calls: Array<{ command: string; args: Record<string, unknown> }> = [];
  const transport: StudioIpcTransport = {
    async invoke<T>(command: string, args: Record<string, unknown>) {
      calls.push({ command, args });
      return result as T;
    },
  };
  return { calls, transport };
}

describe("Studio IPC v1 client", () => {
  it("starts Consumer Fast with the canonical mutation envelope", async () => {
    const { calls, transport } = recordingTransport({
      job_id: "job-001",
      project_revision: 8,
    });
    const client = createStudioIpcClient(transport);

    await client.startConsumerFast({
      projectPath: "N:/projects/chapter/project.json",
      chapterId: "chapter-057",
      expectedRevision: 7,
      idempotencyKey: "start-chapter-057-r7",
    });

    expect(calls).toEqual([{
      command: "start_consumer_fast",
      args: {
        projectPath: "N:/projects/chapter/project.json",
        chapterId: "chapter-057",
        expectedRevision: 7,
        idempotencyKey: "start-chapter-057-r7",
      },
    }]);
  });

  it("sends job controls with job, revision and idempotency identity", async () => {
    const { calls, transport } = recordingTransport({
      job_id: "job-001",
      project_revision: 9,
      status: "cancelled",
    });
    const client = createStudioIpcClient(transport);

    await client.cancelConsumerFast({
      jobId: "job-001",
      expectedRevision: 8,
      idempotencyKey: "cancel-job-001-r8",
    });

    expect(calls[0]).toEqual({
      command: "cancel_consumer_fast",
      args: {
        jobId: "job-001",
        expectedRevision: 8,
        idempotencyKey: "cancel-job-001-r8",
      },
    });
  });

  it("retries only the terminal units explicitly selected", async () => {
    const { calls, transport } = recordingTransport();
    const client = createStudioIpcClient(transport);

    await client.retryConsumerFast({
      jobId: "job-001",
      terminalUnitIds: ["owner-3", "owner-9"],
      expectedRevision: 9,
      idempotencyKey: "retry-job-001-r9",
    });

    expect(calls[0]).toEqual({
      command: "retry_consumer_fast",
      args: {
        jobId: "job-001",
        terminalUnitIds: ["owner-3", "owner-9"],
        expectedRevision: 9,
        idempotencyKey: "retry-job-001-r9",
      },
    });
  });

  it("maps pause and resume to distinct canonical commands", async () => {
    const { calls, transport } = recordingTransport();
    const client = createStudioIpcClient(transport);
    const identity = {
      jobId: "job-001",
      expectedRevision: 9,
      idempotencyKey: "job-001-r9-action",
    };

    await client.pauseConsumerFast(identity);
    await client.resumeConsumerFast(identity);

    expect(calls.map(({ command }) => command)).toEqual([
      "pause_consumer_fast",
      "resume_consumer_fast",
    ]);
  });

  it("persists canonical events and requests owner retypeset without invoking OCR", async () => {
    const { calls, transport } = recordingTransport();
    const client = createStudioIpcClient(transport);
    const projectEvent = {
      job_id: "job-001",
      project_id: "project-001",
      expected_revision: 9,
      project_revision: 9,
      sequence: 4,
      stage: "review",
      status: "awaiting_review" as const,
      reason_code: "owner_requires_review",
      payload: { owner_id: "owner-3" },
    };

    await client.persistProjectEvent({
      event: projectEvent,
      expectedRevision: 9,
      idempotencyKey: "event-job-001-4",
    });
    await client.retypesetOwner({
      projectPath: "N:/p/project.json",
      ownerId: "owner-3",
      layoutRequest: { owner_id: "owner-3", text: "Texto corrigido" },
      expectedRevision: 10,
      idempotencyKey: "retypeset-owner-3-r10",
    });

    expect(calls[0]).toEqual({
      command: "persist_project_event",
      args: {
        event: projectEvent,
        expectedRevision: 9,
        idempotencyKey: "event-job-001-4",
      },
    });
    expect(calls[1]).toEqual({
      command: "retypeset_owner",
      args: {
        projectPath: "N:/p/project.json",
        ownerId: "owner-3",
        layoutRequest: { owner_id: "owner-3", text: "Texto corrigido" },
        expectedRevision: 10,
        idempotencyKey: "retypeset-owner-3-r10",
      },
    });
  });

  it("keeps export decision, final export and diagnostic export separate", async () => {
    const { calls, transport } = recordingTransport();
    const client = createStudioIpcClient(transport);

    await client.decideExport({ projectPath: "N:/p/project.json", expectedRevision: 10 });
    await client.approveFinalReview({
      projectPath: "N:/p/project.json",
      expectedRevision: 10,
      idempotencyKey: "approve-final-r10",
      actorKind: "human",
      actorId: "local-user",
    });
    await client.exportFinal({
      projectPath: "N:/p/project.json",
      destination: "N:/exports/final.cbz",
      expectedRevision: 10,
      idempotencyKey: "export-final-r10",
    });
    await client.exportDiagnostic({
      projectPath: "N:/p/project.json",
      destination: "N:/exports/diagnostic.zip",
      expectedRevision: 10,
    });

    expect(calls.map(({ command }) => command)).toEqual([
      "decide_export",
      "approve_final_review",
      "export_final",
      "export_diagnostic",
    ]);
    expect(calls[1].args).toMatchObject({
      idempotencyKey: "approve-final-r10",
      actorKind: "human",
      actorId: "local-user",
    });
    expect(calls[2].args).toMatchObject({ idempotencyKey: "export-final-r10" });
    expect(calls[3].args).not.toHaveProperty("idempotencyKey");
  });

  it("submits the Studio-owned ReviewDecision through the canonical server-validated command", async () => {
    const { calls, transport } = recordingTransport({
      decision_id: "review-001",
      project_revision: 13,
      owner_status: "approved",
    });
    const client = createStudioIpcClient(transport);
    const decision: ReviewDecision = {
      project_id: "project-001",
      owner_id: "owner-007",
      expected_revision: 12,
      actor_kind: "human",
      actor_id: "local-user",
      decision: "accept_candidate",
      reason_code: "renderer_preference_A",
      evidence_sha256s: ["a".repeat(64)],
      idempotency_key: "preference-owner-007-r12",
    };

    await client.submitReviewDecision({
      projectPath: "N:/p/project.json",
      decision,
      expectedRevision: 12,
      idempotencyKey: "preference-owner-007-r12",
    });

    expect(calls[0]).toEqual({
      command: "submit_review_decision",
      args: {
        projectPath: "N:/p/project.json",
        decision,
        expectedRevision: 12,
        idempotencyKey: "preference-owner-007-r12",
      },
    });
  });

  it("reads a pending Renderer comparison by owner and exact project revision", async () => {
    const { calls, transport } = recordingTransport({
      owner_id: "owner-007",
      project_revision: 12,
      state: "pending",
      comparison: { schema: "traduzai.renderer-preference.v1" },
      artifact_base: "N:/p/review/owner-007",
    });
    const client = createStudioIpcClient(transport);

    await client.readRendererPreferenceComparison({
      projectPath: "N:/p/project.json",
      ownerId: "owner-007",
      expectedRevision: 12,
    });

    expect(calls).toEqual([{
      command: "read_renderer_preference_comparison",
      args: {
        projectPath: "N:/p/project.json",
        ownerId: "owner-007",
        expectedRevision: 12,
      },
    }]);
  });

  it("rejects a ReviewDecision whose nested revision or idempotency key diverges from the command", async () => {
    const { calls, transport } = recordingTransport();
    const client = createStudioIpcClient(transport);
    const decision: ReviewDecision = {
      project_id: "project-001",
      owner_id: "owner-007",
      expected_revision: 11,
      actor_kind: "system",
      actor_id: "test/synthetic",
      decision: "unsure",
      reason_code: "synthetic_preference_test",
      evidence_sha256s: ["b".repeat(64)],
      idempotency_key: "nested-key",
    };

    await expect(client.submitReviewDecision({
      projectPath: "N:/p/project.json",
      decision,
      expectedRevision: 12,
      idempotencyKey: "command-key",
    })).rejects.toThrow("identidade");
    expect(calls).toEqual([]);
  });
});
