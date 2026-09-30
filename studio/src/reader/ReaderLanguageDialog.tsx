import { useEffect, useRef, useState } from "react";
import { toggleAllowedLanguage, type AllowedLanguages } from "./readerPreferences";

export function ReaderLanguageDialog({ open, languages, allowed, onCancel, onApply }: {
  open: boolean;
  languages: Array<{ code: string; label: string; count: number }>;
  allowed: AllowedLanguages;
  onCancel(): void;
  onApply(allowed: AllowedLanguages): void;
}) {
  const [draft, setDraft] = useState<AllowedLanguages>(allowed);
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (open) setDraft(allowed); }, [allowed, open]);
  useEffect(() => {
    if (!open) return;
    const dialog = dialogRef.current;
    dialog?.querySelector<HTMLElement>("input, button")?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
      if (event.key !== "Tab" || !dialog) return;
      const focusable = [...dialog.querySelectorAll<HTMLElement>("input:not(:disabled), button:not(:disabled)")];
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onCancel, open]);
  if (!open) return null;
  const codes = languages.map((language) => language.code);
  return <div className="reader-language-backdrop studio-floating-layer" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onCancel(); }}>
    <div className="reader-language-dialog" role="dialog" aria-modal="true" aria-labelledby="reader-language-title" ref={dialogRef}>
      <h2 id="reader-language-title">Idiomas permitidos</h2>
      <div className="reader-language-options">
        <label><span>Todos</span><input type="checkbox" role="switch" checked={draft === null} aria-checked={draft === null} onChange={(event) => setDraft(toggleAllowedLanguage(draft, "*", event.currentTarget.checked, codes))} /></label>
        {languages.map((language) => {
          const checked = draft === null || draft.includes(language.code);
          return <label key={language.code}><span>{language.label}<small>{language.count} fonte(s)</small></span><input type="checkbox" role="switch" checked={checked} aria-checked={checked} onChange={(event) => setDraft(toggleAllowedLanguage(draft, language.code, event.currentTarget.checked, codes))} /></label>;
        })}
      </div>
      <footer><button type="button" onClick={onCancel}>Cancelar</button><button type="button" onClick={() => onApply(draft)}>Aplicar</button></footer>
    </div>
  </div>;
}
