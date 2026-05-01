import { useDeferredValue, useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown, Globe, Search } from "lucide-react";
import type { SupportedLanguage } from "../../lib/tauri";
import { normalizeLanguageCodeForSelection } from "../../lib/languages";

interface LanguageSelectFieldProps {
  label: string;
  value: string;
  languages: SupportedLanguage[];
  fallbackCode: string;
  onChange: (code: string) => void;
  sourceMode?: boolean;
  helperText?: string;
  searchPlaceholder?: string;
  emptyText?: string;
}

export function LanguageSelectField({
  label,
  value,
  languages,
  fallbackCode,
  onChange,
  sourceMode = false,
  helperText,
  searchPlaceholder = "Buscar idioma ou codigo...",
  emptyText = "Nenhum idioma encontrado.",
}: LanguageSelectFieldProps) {
  const fieldRef = useRef<HTMLDivElement | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const listboxId = useId();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);

  const selectedCode = normalizeLanguageCodeForSelection(value, languages, fallbackCode);
  const selectedLanguage =
    languages.find((language) => language.code === selectedCode) ?? languages[0] ?? null;
  const normalizedQuery = deferredQuery.trim().toLowerCase();

  const filteredLanguages = languages.filter((language) => {
    if (!normalizedQuery) return true;

    const labelText = language.label.toLowerCase();
    const codeText = language.code.toLowerCase();
    const sourceText =
      sourceMode && language.ocr_strategy === "best_effort" ? "ocr experimental" : "";

    return [labelText, codeText, sourceText].some((text) => text.includes(normalizedQuery));
  });

  useEffect(() => {
    if (!open) {
      setQuery("");
      return;
    }

    const timer = window.setTimeout(() => {
      searchInputRef.current?.focus();
      searchInputRef.current?.select();
    }, 10);

    function handlePointerDown(event: MouseEvent) {
      if (!fieldRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    window.addEventListener("mousedown", handlePointerDown);
    window.addEventListener("keydown", handleEscape);

    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("mousedown", handlePointerDown);
      window.removeEventListener("keydown", handleEscape);
    };
  }, [open]);

  function handleSelect(code: string) {
    onChange(code);
    setOpen(false);
  }

  return (
    <div ref={fieldRef} className="relative">
      <label className="text-sm text-text-secondary mb-2 block">{label}</label>

      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-controls={listboxId}
        className={`w-full rounded-xl border px-4 py-3 text-left transition-smooth ${
          open
            ? "border-accent-purple/40 bg-bg-tertiary shadow-[0_0_0_1px_rgba(168,85,247,0.16)]"
            : "border-white/10 bg-bg-secondary hover:border-white/20"
        }`}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 text-sm text-text-primary">
              <Globe size={14} className="text-accent-purple flex-shrink-0" />
              <span className="truncate">
                {selectedLanguage ? selectedLanguage.label : "Selecione um idioma"}
              </span>
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-text-secondary/80">
              <span className="rounded-md bg-white/5 px-2 py-0.5 font-mono uppercase tracking-wide">
                {selectedLanguage?.code ?? fallbackCode}
              </span>
              {sourceMode && selectedLanguage?.ocr_strategy === "best_effort" && (
                <span className="rounded-md bg-status-warning/10 px-2 py-0.5 text-status-warning">
                  OCR experimental
                </span>
              )}
            </div>
          </div>
          <ChevronDown
            size={16}
            className={`mt-0.5 flex-shrink-0 text-text-secondary transition-transform ${
              open ? "rotate-180" : ""
            }`}
          />
        </div>
      </button>

      {helperText && (
        <p className="mt-2 text-xs text-text-secondary/80">{helperText}</p>
      )}

      {open && (
        <div className="absolute left-0 right-0 top-full z-30 mt-2 overflow-hidden rounded-2xl border border-white/10 bg-bg-primary shadow-2xl shadow-black/30">
          <div className="border-b border-white/5 p-3">
            <div className="relative">
              <Search
                size={14}
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary/50"
              />
              <input
                ref={searchInputRef}
                type="text"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={searchPlaceholder}
                className="w-full rounded-lg border border-white/10 bg-bg-secondary py-2 pl-9 pr-3 text-sm text-text-primary placeholder:text-text-secondary/45 focus:border-accent-purple/40 focus:outline-none"
              />
            </div>
          </div>

          <div
            id={listboxId}
            role="listbox"
            className="max-h-72 overflow-y-auto p-2"
          >
            {filteredLanguages.length === 0 ? (
              <div className="px-3 py-6 text-center text-sm text-text-secondary">
                {emptyText}
              </div>
            ) : (
              filteredLanguages.map((language) => {
                const isSelected = language.code === selectedCode;
                return (
                  <button
                    key={language.code}
                    type="button"
                    role="option"
                    aria-selected={isSelected}
                    onClick={() => handleSelect(language.code)}
                    className={`flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2.5 text-left transition-smooth ${
                      isSelected
                        ? "bg-accent-purple/12 text-text-primary"
                        : "text-text-secondary hover:bg-white/5 hover:text-text-primary"
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="truncate text-sm font-medium">
                          {language.label}
                        </span>
                        {sourceMode && language.ocr_strategy === "best_effort" && (
                          <span className="rounded-md bg-status-warning/10 px-2 py-0.5 text-[10px] text-status-warning">
                            OCR experimental
                          </span>
                        )}
                      </div>
                      <div className="mt-1 text-[11px] text-text-secondary/75">
                        <span className="font-mono uppercase tracking-wide">
                          {language.code}
                        </span>
                        {sourceMode && language.ocr_strategy === "dedicated" && (
                          <span className="ml-2">OCR pronto</span>
                        )}
                      </div>
                    </div>
                    <Check
                      size={15}
                      className={isSelected ? "text-accent-purple" : "text-transparent"}
                    />
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
