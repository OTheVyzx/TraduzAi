import { createElement, useState, type ReactNode } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { Scale } from "lucide-react";
import type { StudioIpcClient } from "../backend/studioIpcClient";
import { createTauriStudioIpcClient } from "../backend/studioIpcTauri";
import { buildReviewDecision, type ReviewDecision } from "../backend/studioIpcV1";
import { canonicalStudioProjectIdentity } from "../jobs/StudioJobControls";
import type { StudioProject, StudioTextLayer } from "../project/studioProject";
import {
  createPreferenceResponse,
  parsePreferenceComparison,
  type PreferenceActor,
  type PreferenceChoice,
  type PreferenceComparison,
  RendererPreferenceReview,
} from "./RendererPreferenceReview";

export interface LoadedRendererPreference {
  comparison: PreferenceComparison;
  artifactBase: string;
}

export function preferenceOwnerId(layer: StudioTextLayer | null): string | null {
  if (!layer) return null;
  for (const value of [layer.owner_id, layer.logical_owner_id]) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

function requireProjectIdentity(project: StudioProject, projectPath: string) {
  const identity = canonicalStudioProjectIdentity(project, projectPath);
  if ("error" in identity) throw new Error(identity.error);
  return identity;
}

export async function loadRendererPreference(input: {
  client: StudioIpcClient;
  project: StudioProject;
  projectPath: string;
  ownerId: string;
}): Promise<LoadedRendererPreference | null> {
  const identity = requireProjectIdentity(input.project, input.projectPath);
  const result = await input.client.readRendererPreferenceComparison({
    projectPath: identity.projectPath,
    ownerId: input.ownerId,
    expectedRevision: identity.expectedRevision,
  });
  if (result.owner_id !== input.ownerId || result.project_revision !== identity.expectedRevision) {
    throw new Error("A comparação retornada não corresponde ao owner e à revisão solicitados");
  }
  if (result.state !== "pending" || result.comparison === null || !result.artifact_base) return null;
  return {
    comparison: parsePreferenceComparison(result.comparison),
    artifactBase: result.artifact_base,
  };
}

export function rendererPreferenceActor(comparison: PreferenceComparison): PreferenceActor {
  const synthetic = comparison.candidates.every(
    (candidate) => candidate.metrics.fixture_kind === "synthetic_contract_test",
  );
  return synthetic
    ? { kind: "system", id: "test/synthetic" }
    : { kind: "human", id: "local-user" };
}

function decisionForChoice(choice: PreferenceChoice): ReviewDecision["decision"] {
  if (choice === "A" || choice === "B") return "accept_candidate";
  if (choice === "equivalent") return "equivalent";
  if (choice === "neither") return "reject_candidates";
  return "defer";
}

export async function submitRendererPreference(input: {
  client: StudioIpcClient;
  project: StudioProject;
  projectPath: string;
  comparison: PreferenceComparison | unknown;
  choice: PreferenceChoice;
  recordedAt?: string;
  trainingOptIn?: boolean;
}) {
  const identity = requireProjectIdentity(input.project, input.projectPath);
  const comparison = parsePreferenceComparison(input.comparison);
  const ownerId = comparison.candidates[0].owner_id;
  const actor = rendererPreferenceActor(comparison);
  const recordedAt = input.recordedAt ?? new Date().toISOString();
  const response = createPreferenceResponse({
    comparison,
    choice: input.choice,
    actor,
    recordedAt,
    trainingOptIn: actor.kind === "human" && input.trainingOptIn === true,
  });
  const evidenceSha256s = [
    response.comparison_sha256,
    response.target_sha256,
    response.randomization_sha256,
    ...response.candidate_recipe_sha256s,
    ...response.candidate_output_sha256s,
  ];
  const nonce = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
  const idempotencyKey = `renderer-preference:${ownerId}:r${identity.expectedRevision}:${nonce}`;
  const decision = buildReviewDecision({
    projectId: identity.projectId,
    ownerId,
    expectedRevision: identity.expectedRevision,
    actorKind: actor.kind,
    actorId: actor.id,
    decision: decisionForChoice(input.choice),
    reasonCode: `renderer_preference_${input.choice}`,
    evidenceSha256s,
    idempotencyKey,
    preferenceResponse: response,
  });
  return input.client.submitReviewDecision({
    projectPath: identity.projectPath,
    decision,
    expectedRevision: identity.expectedRevision,
    idempotencyKey,
  });
}

export function StudioRendererPreference({
  project,
  projectPath,
  layer,
  client,
  onPersisted,
}: {
  project: StudioProject;
  projectPath: string;
  layer: StudioTextLayer | null;
  client?: StudioIpcClient;
  onPersisted?: () => void | Promise<void>;
}) {
  const ownerId = preferenceOwnerId(layer);
  const [loaded, setLoaded] = useState<LoadedRendererPreference | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!ownerId) return null;

  const ipc = () => client ?? createTauriStudioIpcClient();
  const open = async () => {
    setLoading(true);
    setError(null);
    try {
      const pending = await loadRendererPreference({ client: ipc(), project, projectPath, ownerId });
      setLoaded(pending);
      if (!pending) setError("Não há comparação pendente para esta unidade.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  };

  const children: ReactNode[] = [
    createElement("button", {
      key: "trigger",
      type: "button",
      disabled: loading,
      onClick: () => void open(),
      className: "flex w-full items-center justify-center gap-2 rounded-lg border border-accent-purple/30 bg-accent-purple/10 px-3 py-2 text-[11px] font-semibold text-accent-purple transition hover:bg-accent-purple/15 disabled:opacity-40",
      title: "Candidatas do Renderer para a mesma unidade e o mesmo texto",
    }, createElement(Scale, { size: 13 }), loading ? "Carregando…" : "Comparar A/B"),
  ];
  if (error) {
    children.push(createElement("p", {
      key: "error",
      role: "status",
      className: "mt-2 text-[10px] leading-4 text-text-muted",
    }, error));
  }
  if (loaded) {
    const actor = rendererPreferenceActor(loaded.comparison);
    children.push(createElement("div", {
      key: "dialog",
      className: "studio-floating-layer studio-editor-recovery-backdrop",
      role: "dialog",
      "aria-modal": "true",
      "aria-label": "Comparação A/B do Renderer",
    }, createElement("section", {
      className: "relative max-h-[88vh] w-[min(980px,92vw)] overflow-auto rounded-2xl border border-border bg-bg-secondary p-4 shadow-2xl",
    },
    createElement("button", {
      type: "button",
      onClick: () => setLoaded(null),
      className: "absolute right-3 top-3 z-10 rounded-md border border-border bg-bg-primary px-2 py-1 text-[10px] text-text-secondary",
    }, "Fechar"),
    createElement(RendererPreferenceReview, {
      comparison: loaded.comparison,
      actor,
      resolveArtifact: (relativePath: string) => convertFileSrc(`${loaded.artifactBase}/${relativePath}`),
      onSubmit: async (response) => {
        await submitRendererPreference({
          client: ipc(),
          project,
          projectPath,
          comparison: loaded.comparison,
          choice: response.choice,
          trainingOptIn: response.training_eligible,
          recordedAt: response.recorded_at,
        });
        setLoaded(null);
        await onPersisted?.();
      },
    }))));
  }
  return createElement("section", { className: "px-3 pb-3", "aria-label": "Preferência de composição" }, ...children);
}
