import { Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  findEditorFontOption,
  googleFontSearchResultToOption,
  listEditorFontGroups,
  type EditorFontGroup,
  type EditorFontOption,
} from "../../lib/fontCatalog";
import { searchGoogleFonts } from "../../lib/tauri";

type SearchStatus = "idle" | "loading" | "error";

function joinClassNames(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

function containsFontValue(groups: EditorFontGroup[], value: string): boolean {
  return groups.some((group) => group.options.some((option) => option.value === value));
}

function labelFromStoredFontValue(value?: string): string {
  if (!value) return "";
  const localOption = findEditorFontOption(value);
  if (localOption) return localOption.label;
  const googleMatch = /^GoogleFont__(.+)__[^.]+\.(?:ttf|otf)$/i.exec(value);
  if (googleMatch) return googleMatch[1].replace(/_/g, " ");
  return value.replace(/\.(ttf|otf)$/i, "");
}

function includeSelectedFont(groups: EditorFontGroup[], value?: string, selectedOption?: EditorFontOption | null): EditorFontGroup[] {
  if (!value || containsFontValue(groups, value)) return groups;
  const option =
    selectedOption?.value === value
      ? selectedOption
      : findEditorFontOption(value) ?? {
          label: labelFromStoredFontValue(value),
          value,
          cssFamily: labelFromStoredFontValue(value),
          source: "google" as const,
          groupLabel: "Selecionada",
          variants: ["regular"],
          variant: "regular",
        };
  return [
    {
      label: "Selecionada",
      source: option.source,
      options: [option],
    },
    ...groups,
  ];
}

function optionGroupsFromOptions(options: EditorFontOption[]): EditorFontGroup[] {
  if (options.length === 0) return [];
  return [
    {
      label: "Google Fonts",
      source: "google",
      options,
    },
  ];
}

function FontMenu({
  groups,
  status,
  query,
  left,
  top,
  width,
  onPick,
}: {
  groups: EditorFontGroup[];
  status: SearchStatus;
  query: string;
  left: number;
  top: number;
  width: number;
  onPick: (option: EditorFontOption) => void;
}) {
  const trimmedQuery = query.trim();
  return createPortal(
    <div
      style={{ position: "fixed", left, top, width, zIndex: 9999 }}
      className="max-h-[260px] overflow-y-auto rounded-md border border-border bg-bg-secondary shadow-[0_8px_32px_rgba(0,0,0,0.45)]"
    >
      {status === "loading" && <div className="px-3 py-2 text-[11px] text-text-muted">Buscando...</div>}
      {status === "error" && (
        <div className="px-3 py-2 text-[11px] text-status-error">Falha ao buscar no Google Fonts</div>
      )}
      {status !== "loading" && status !== "error" && trimmedQuery.length > 0 && trimmedQuery.length < 2 && (
        <div className="px-3 py-2 text-[11px] text-text-muted">Digite pelo menos 2 letras</div>
      )}
      {status !== "loading" && status !== "error" && trimmedQuery.length >= 2 && groups.length === 0 && (
        <div className="px-3 py-2 text-[11px] text-text-muted">Nenhuma fonte encontrada</div>
      )}
      {groups.map((group) => (
        <div key={`${group.source}:${group.label}`} className="py-1">
          <div className="px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-muted">
            {group.label}
          </div>
          {group.options.map((font) => (
            <button
              key={font.value}
              type="button"
              onMouseDown={(event) => {
                event.preventDefault();
                onPick(font);
              }}
              className="flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-[11px] text-text-primary hover:bg-white/[0.06]"
            >
              <span className="truncate">{font.label}</span>
              {font.source === "google" && (
                <span className="shrink-0 text-[9px] uppercase tracking-[0.08em] text-brand">Google</span>
              )}
            </button>
          ))}
        </div>
      ))}
    </div>,
    document.body,
  );
}

export function EditorFontPicker({
  value,
  loadingFont,
  onChange,
  variant = "panel",
  selectTestId,
}: {
  value?: string;
  loadingFont?: string | null;
  onChange: (value: string, option?: EditorFontOption) => void | Promise<void>;
  variant?: "toolbar" | "panel";
  selectTestId?: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const remoteCacheRef = useRef(new Map<string, EditorFontOption[]>());
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState(() => labelFromStoredFontValue(value));
  const [remoteOptions, setRemoteOptions] = useState<EditorFontOption[]>([]);
  const [remoteStatus, setRemoteStatus] = useState<SearchStatus>("idle");
  const [selectedRemoteOption, setSelectedRemoteOption] = useState<EditorFontOption | null>(null);
  const [menuRect, setMenuRect] = useState({ left: 0, top: 0, width: 220 });
  const isToolbar = variant === "toolbar";
  const disabled = loadingFont !== null && loadingFont !== undefined;
  const query = inputValue.trim();

  useEffect(() => {
    if (!open) setInputValue(labelFromStoredFontValue(value));
  }, [open, value]);

  useEffect(() => {
    if (!open || query.length < 2) {
      setRemoteOptions([]);
      setRemoteStatus("idle");
      return;
    }

    const normalizedQuery = query.toLowerCase();
    const cached = remoteCacheRef.current.get(normalizedQuery);
    if (cached) {
      setRemoteOptions(cached);
      setRemoteStatus("idle");
      return;
    }

    let cancelled = false;
    setRemoteStatus("loading");
    const timeout = window.setTimeout(() => {
      searchGoogleFonts(query)
        .then((results) => {
          if (cancelled) return;
          const options = results.map(googleFontSearchResultToOption);
          remoteCacheRef.current.set(normalizedQuery, options);
          setRemoteOptions(options);
          setRemoteStatus("idle");
        })
        .catch((error) => {
          if (cancelled) return;
          console.warn("[fonts] falha ao buscar Google Fonts:", error);
          setRemoteOptions([]);
          setRemoteStatus("error");
        });
    }, 350);

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [open, query]);

  useEffect(() => {
    if (!open || !inputRef.current) return;
    const updatePosition = () => {
      const rect = inputRef.current?.getBoundingClientRect();
      if (!rect) return;
      setMenuRect({
        left: rect.left,
        top: rect.bottom + 4,
        width: Math.max(rect.width, isToolbar ? 260 : 220),
      });
    };
    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [isToolbar, open]);

  const groups = useMemo(() => {
    const baseGroups = query.length >= 2 ? optionGroupsFromOptions(remoteOptions) : listEditorFontGroups();
    return includeSelectedFont(baseGroups, value, selectedRemoteOption);
  }, [query.length, remoteOptions, selectedRemoteOption, value]);

  function pickFont(option: EditorFontOption) {
    setSelectedRemoteOption(option.source === "google" ? option : null);
    setInputValue(option.label);
    setOpen(false);
    void onChange(option.value, option);
  }

  return (
    <div className={joinClassNames("relative shrink-0", isToolbar ? "w-[260px]" : "w-full")}>
      <Search
        size={12}
        className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-text-muted"
      />
      <input
        ref={inputRef}
        type="search"
        value={inputValue}
        title={loadingFont ? "Baixando fonte..." : "Fonte"}
        data-testid={selectTestId}
        disabled={disabled}
        placeholder="Digite uma fonte"
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 120)}
        onChange={(event) => {
          setInputValue(event.target.value);
          setOpen(true);
        }}
        className={joinClassNames(
          "w-full rounded-md border border-border bg-bg-tertiary/60 pl-7 pr-2 text-[11px] text-text-primary transition-smooth placeholder:text-text-muted/70 focus:border-brand/40 focus:outline-none",
          isToolbar ? "h-7" : "h-8",
        )}
      />
      {open && typeof document !== "undefined" && (
        <FontMenu
          groups={groups}
          status={remoteStatus}
          query={query}
          left={menuRect.left}
          top={menuRect.top}
          width={menuRect.width}
          onPick={pickFont}
        />
      )}
    </div>
  );
}
