import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { StudioProject } from "../../project/studioProject";
import { createStudioJobState } from "../../backend/studioIpcV1";
import {
  StudioJobStatusCard,
  canonicalStudioProjectIdentity,
  newestStudioJobState,
  persistedStudioJobState,
  promotedProjectPath,
} from "../StudioJobControls";

const project = (overrides: Partial<StudioProject> = {}): StudioProject => ({
  app: "traduzai",
  versao: "2.0",
  studio_schema_version: "1.0",
  id: "project-001",
  capitulo: "57",
  paginas: [],
  project_revision: 7,
  ...overrides,
});

describe("Studio job controls", () => {
  it("requires canonical project, chapter and revision identity", () => {
    expect(canonicalStudioProjectIdentity(project(), "N:/p/project.json")).toEqual({
      projectId: "project-001",
      chapterId: "57",
      projectPath: "N:/p/project.json",
      expectedRevision: 7,
    });
    expect(canonicalStudioProjectIdentity(project({ id: undefined }), "N:/p/project.json"))
      .toEqual({ error: "Projeto sem ID canônico" });
    expect(canonicalStudioProjectIdentity(project({ project_revision: undefined }), "N:/p/project.json"))
      .toEqual({ error: "Projeto sem revisão canônica" });
  });

  it("shows real stage and controls without inventing a percentage", () => {
    const state = {
      ...createStudioJobState({ jobId: "job-001", projectId: "project-001", expectedRevision: 7 }),
      stage: "analysis",
      status: "running" as const,
      reasonCode: "provider_started",
    };
    const html = renderToStaticMarkup(
      <StudioJobStatusCard state={state} busy={false} error={null} onAction={() => undefined} />,
    );

    expect(html).toContain("Análise");
    expect(html).toContain("Em execução");
    expect(html).toContain("Pausar");
    expect(html).toContain("Cancelar");
    expect(html).not.toContain("%");
  });

  it("keeps cancelled distinct from completed and terminal", () => {
    const state = {
      ...createStudioJobState({ jobId: "job-001", projectId: "project-001", expectedRevision: 7 }),
      status: "cancelled" as const,
      reasonCode: "cancelled_by_user",
    };
    const html = renderToStaticMarkup(
      <StudioJobStatusCard state={state} busy={false} error={null} onAction={() => undefined} />,
    );

    expect(html).toContain("Cancelado");
    expect(html).toContain("Tentar novamente");
    expect(html).not.toContain("Retomar");
    expect(html).not.toContain("Concluído");
  });

  it("allows an explicit reprocess after review or a blocked result", () => {
    for (const status of ["awaiting_review", "blocked"] as const) {
      const state = {
        ...createStudioJobState({ jobId: "job-001", projectId: "project-001", expectedRevision: 7 }),
        status,
        stage: "review",
        reasonCode: "review_required",
      };
      const html = renderToStaticMarkup(
        <StudioJobStatusCard state={state} busy={false} error={null} onAction={() => undefined} />,
      );

      expect(html).toContain("Tentar novamente");
    }
  });

  it("reopens a persisted paused job without pretending it completed", () => {
    const persisted = persistedStudioJobState(project({
      job_id: "job-001",
      job_status: "paused",
      job_stage: "translation",
      job_sequence: 14,
      project_revision: 9,
      job_expected_revision: 7,
    }), {
      projectId: "project-001",
      chapterId: "57",
      projectPath: "N:/p/project.json",
      expectedRevision: 9,
    });

    expect(persisted).toMatchObject({
      jobId: "job-001",
      status: "paused",
      stage: "translation",
      lastSequence: 14,
      expectedRevision: 7,
      projectRevision: 9,
    });
  });

  it("treats a persisted job without status as resumable unknown state", () => {
    expect(persistedStudioJobState(project({ job_id: "job-001" }), {
      projectId: "project-001",
      chapterId: "57",
      projectPath: "N:/p/project.json",
      expectedRevision: 7,
    })?.status).toBe("paused");
  });

  it("reconciles a newer persisted worker transition over a stale action result", () => {
    const stale = {
      ...createStudioJobState({ jobId: "job-001", projectId: "project-001", expectedRevision: 7 }),
      projectRevision: 8,
      lastSequence: 2,
      status: "queued" as const,
      reasonCode: "retry_requested",
    };
    const running = {
      ...stale,
      projectRevision: 9,
      lastSequence: 3,
      status: "running" as const,
      reasonCode: "worker_started",
    };

    expect(newestStudioJobState(stale, running)).toBe(running);
    expect(newestStudioJobState(running, stale)).toBe(running);
  });

  it("reopens the promoted physical project from the completion event", () => {
    expect(promotedProjectPath({
      reason_code: "physical_pipeline_completed",
      payload: {
        detail: {
          output_project_path: "C:/runtime/pipeline-staging-r40/project.json",
        },
      },
    })).toBe("C:/runtime/pipeline-staging-r40/project.json");
    expect(promotedProjectPath({ reason_code: "worker_started", payload: {} })).toBeNull();
  });

  it("reopens the newest canonical integration job persisted by the backend", () => {
    const persisted = persistedStudioJobState(project({
      project_revision: 12,
      integration_v1: {
        jobs: {
          "job-old": {
            job_id: "job-old",
            project_id: "project-001",
            expected_revision: 7,
            project_revision: 8,
            sequence: 2,
            stage: "analysis",
            status: "cancelled",
            reason_code: "job_cancelled",
          },
          "job-current": {
            job_id: "job-current",
            project_id: "project-001",
            expected_revision: 11,
            project_revision: 12,
            sequence: 9,
            stage: "translation",
            status: "paused",
            reason_code: "pause_requested",
            units: {
              "owner-ok": { status: "complete", terminal: true },
              "owner-pending": { status: "queued", terminal: false },
            },
          },
        },
      },
    }), {
      projectId: "project-001",
      chapterId: "57",
      projectPath: "N:/p/project.json",
      expectedRevision: 12,
    });

    expect(persisted).toMatchObject({
      jobId: "job-current",
      status: "paused",
      stage: "translation",
      reasonCode: "pause_requested",
      lastSequence: 9,
      expectedRevision: 11,
      projectRevision: 12,
      payload: { terminal_unit_ids: ["owner-ok"] },
    });
  });
});
