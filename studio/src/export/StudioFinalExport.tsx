import { useMemo, useState } from "react";
import { FileCheck2, ShieldAlert } from "lucide-react";

import type { ExportDecision, StudioIpcClient } from "../backend/studioIpcClient";
import { createTauriStudioIpcClient } from "../backend/studioIpcTauri";
import { explainStudioIpcError } from "../backend/studioIpcV1";
import type { StudioProject } from "../project/studioProject";
import { canonicalStudioProjectIdentity } from "../jobs/StudioJobControls";

function plural(count: number, singular: string, pluralForm: string) {
  return `Há ${count} ${count === 1 ? singular : pluralForm}`;
}

export function exportBlockReasons(decision: ExportDecision): string[] {
  const reasons: string[] = [];
  if (!decision.allowed) reasons.push("O backend não autorizou a exportação final");
  if (decision.status !== "PASS" || !decision.gate_allowed) reasons.push("O gate de exportação não está aprovado");
  if (!decision.verified) reasons.push("O projeto ainda não foi verificado");
  if (decision.completion_status !== "approved") reasons.push("O processamento ainda não foi aprovado");
  if (decision.output_review_state !== "approved") reasons.push("A revisão visual final ainda não foi aprovada");
  if (decision.blocker_count > 0) reasons.push(plural(decision.blocker_count, "blocker não resolvido", "blockers não resolvidos"));
  if (decision.critical_issue_count > 0) reasons.push(plural(decision.critical_issue_count, "erro crítico", "erros críticos"));
  if (decision.critical_flag_count > 0) reasons.push(plural(decision.critical_flag_count, "flag crítica", "flags críticas"));
  if (decision.review_issue_count > 0) reasons.push(plural(decision.review_issue_count, "achado de revisão", "achados de revisão"));
  if (decision.review_flag_count > 0) reasons.push(plural(decision.review_flag_count, "flag de revisão", "flags de revisão"));
  if (decision.issue_count > 0
    && decision.critical_issue_count === 0
    && decision.review_issue_count === 0) {
    reasons.push(plural(decision.issue_count, "pendência do gate", "pendências do gate"));
  }
  return reasons;
}

export function canApproveFinalReview(decision: ExportDecision) {
  return !decision.allowed
    && decision.status === "PASS"
    && decision.gate_allowed
    && decision.blocker_count === 0
    && decision.issue_count === 0
    && decision.critical_issue_count === 0
    && decision.critical_flag_count === 0
    && decision.review_issue_count === 0
    && decision.review_flag_count === 0;
}

export function StudioExportDecisionNotice({ decision }: { decision: ExportDecision }) {
  const reasons = exportBlockReasons(decision);
  if (reasons.length === 0) {
    return (
      <div className="rounded-lg border border-status-success/30 bg-status-success/10 p-2 text-[10px] text-status-success">
        Exportação final autorizada pelo backend.
      </div>
    );
  }
  return (
    <div role="alert" className="rounded-lg border border-status-error/30 bg-status-error/10 p-2 text-[10px] text-status-error">
      <p className="font-semibold">Exportação final bloqueada</p>
      <ul className="mt-1 list-disc space-y-0.5 pl-4">
        {reasons.map((reason) => <li key={reason}>{reason}</li>)}
      </ul>
      <p className="mt-1 text-text-muted">Revise as pendências indicadas pelo backend antes de tentar novamente.</p>
    </div>
  );
}

async function chooseFinalExportDestination() {
  const { save } = await import("@tauri-apps/plugin-dialog");
  return save({
    title: "Exportar resultado final",
    defaultPath: "traduzai-final.cbz",
    filters: [{ name: "Comic Book ZIP", extensions: ["cbz", "zip"] }],
  });
}

export async function requestStudioFinalExport({
  client,
  projectPath,
  expectedRevision,
  chooseDestination,
  idempotencyKey,
}: {
  client: StudioIpcClient;
  projectPath: string;
  expectedRevision: number;
  chooseDestination: () => Promise<string | null>;
  idempotencyKey: string;
}) {
  const response = await client.decideExport({ projectPath, expectedRevision });
  const decision = response.export_decision;
  if (exportBlockReasons(decision).length > 0) {
    return { kind: "blocked" as const, decision };
  }
  const destination = await chooseDestination();
  if (!destination) return { kind: "cancelled" as const, decision };
  const result = await client.exportFinal({
    projectPath,
    destination,
    expectedRevision,
    idempotencyKey,
  });
  return { kind: "exported" as const, decision, destination, result };
}

export function StudioFinalExport({
  project,
  projectPath,
  client = createTauriStudioIpcClient(),
  chooseDestination = chooseFinalExportDestination,
  onPersisted,
}: {
  project: StudioProject;
  projectPath: string;
  client?: StudioIpcClient;
  chooseDestination?: () => Promise<string | null>;
  onPersisted?: () => void | Promise<void>;
}) {
  const identity = useMemo(
    () => canonicalStudioProjectIdentity(project, projectPath),
    [project, projectPath],
  );
  const [decision, setDecision] = useState<ExportDecision | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [receipt, setReceipt] = useState<string | null>(null);
  const [confirmingApproval, setConfirmingApproval] = useState(false);

  const requestExport = async () => {
    if ("error" in identity) {
      setError(identity.error);
      return;
    }
    setBusy(true);
    setError(null);
    setReceipt(null);
    try {
      const outcome = await requestStudioFinalExport({
        client,
        projectPath: identity.projectPath,
        expectedRevision: identity.expectedRevision,
        chooseDestination,
        idempotencyKey: `export-final-r${identity.expectedRevision}-${crypto.randomUUID()}`,
      });
      setDecision(outcome.decision);
      if (outcome.kind === "exported") {
        setReceipt(String(outcome.result.publication_receipt.receipt_id ?? outcome.destination));
      }
    } catch (exportError) {
      setError(explainStudioIpcError(exportError));
    } finally {
      setBusy(false);
    }
  };

  const approveFinalReview = async () => {
    if ("error" in identity || !decision || !canApproveFinalReview(decision)) return;
    setBusy(true);
    setError(null);
    try {
      const result = await client.approveFinalReview({
        projectPath: identity.projectPath,
        expectedRevision: identity.expectedRevision,
        idempotencyKey: `approve-final-r${identity.expectedRevision}-${crypto.randomUUID()}`,
        actorKind: "human",
        actorId: "local-user",
      });
      setDecision(result.export_decision);
      setConfirmingApproval(false);
      await onPersisted?.();
    } catch (approvalError) {
      setError(explainStudioIpcError(approvalError));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="relative">
      <button
        type="button"
        disabled={busy || "error" in identity}
        onClick={() => void requestExport()}
        className="flex items-center gap-1 rounded-lg border border-status-success/30 bg-status-success/10 px-2.5 py-1 text-[10px] font-medium text-status-success transition-smooth hover:bg-status-success/15 disabled:opacity-40"
        title={"error" in identity ? identity.error : "Verificar o gate e exportar o capítulo final"}
      >
        {decision && exportBlockReasons(decision).length > 0 ? <ShieldAlert size={11} /> : <FileCheck2 size={11} />}
        {busy ? "Verificando…" : "Exportar final"}
      </button>
      {(decision || error || receipt) && (
        <div className="absolute right-0 top-8 z-[90] w-80 rounded-xl border border-border bg-bg-secondary p-2 shadow-2xl">
          {decision && <StudioExportDecisionNotice decision={decision} />}
          {decision && canApproveFinalReview(decision) && (
            <div className="mt-2 rounded-lg border border-accent/30 bg-accent/10 p-2 text-[10px] text-text-primary">
              <p>O QA terminou sem pendências. A exportação ainda exige sua revisão visual explícita.</p>
              {!confirmingApproval ? (
                <button
                  type="button"
                  className="mt-2 rounded-md border border-accent/40 px-2 py-1 font-medium text-accent"
                  onClick={() => setConfirmingApproval(true)}
                >
                  Aprovar revisão final
                </button>
              ) : (
                <div className="mt-2 space-y-2">
                  <p>Confirme somente depois de revisar o resultado final exibido.</p>
                  <div className="flex justify-end gap-2">
                    <button type="button" onClick={() => setConfirmingApproval(false)} className="rounded-md px-2 py-1 text-text-muted">
                      Cancelar
                    </button>
                    <button type="button" disabled={busy} onClick={() => void approveFinalReview()} className="rounded-md bg-accent px-2 py-1 font-medium text-white disabled:opacity-40">
                      Confirmar aprovação humana
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
          {error && <p role="alert" className="mt-1 text-[10px] text-status-error">{error}</p>}
          {receipt && <p className="mt-1 text-[10px] text-status-success">Exportação concluída: {receipt}</p>}
        </div>
      )}
    </div>
  );
}
