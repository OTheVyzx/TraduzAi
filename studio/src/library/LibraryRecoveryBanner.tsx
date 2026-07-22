import { AlertTriangle, DatabaseBackup, Save } from "lucide-react";

export function LibraryRecoveryBanner({
  recoveredFromBackup,
  hasUnsavedChanges,
  error,
  saving,
  onSaveRecoveredCopy,
}: {
  recoveredFromBackup: boolean;
  hasUnsavedChanges: boolean;
  error: string | null;
  saving: boolean;
  onSaveRecoveredCopy: () => void | Promise<void>;
}) {
  const saveFailed = hasUnsavedChanges && Boolean(error);
  if (!recoveredFromBackup && !saveFailed) return null;

  return (
    <div
      className={`studio-library-recovery${saveFailed ? " studio-library-recovery-error" : ""}`}
      role={saveFailed ? "alert" : "status"}
      aria-live={saveFailed ? "assertive" : "polite"}
    >
      {saveFailed ? <AlertTriangle size={16} aria-hidden="true" /> : <DatabaseBackup size={16} aria-hidden="true" />}
      <span>
        <strong>{saveFailed ? "Falha ao salvar o catálogo." : "Catálogo recuperado do backup."}</strong>{" "}
        {saveFailed
          ? <>As alterações continuam nesta sessão. {error}</>
          : "Revise os dados e grave uma nova cópia principal para concluir a recuperação."}
      </span>
      <button type="button" disabled={saving} onClick={() => void onSaveRecoveredCopy()}>
        <Save size={13} aria-hidden="true" />
        {saving ? "Salvando…" : saveFailed ? "Tentar salvar" : "Salvar cópia recuperada"}
      </button>
    </div>
  );
}
