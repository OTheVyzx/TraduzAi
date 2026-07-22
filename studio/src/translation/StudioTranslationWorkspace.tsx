import { useEffect } from "react";
import { Languages } from "lucide-react";
import type { StudioProject, StudioTextLayer, TranslationStatus } from "../project/studioProject";
import { buildTranslationQueue, resolveTranslationStatus } from "./translationQueue";
import { GlossaryPanel } from "./GlossaryPanel";
import { TranslationInspector } from "./TranslationInspector";
import type { TranslationTarget } from "./TranslationQueuePanel";

export interface TranslationDraft {
  translated: string;
  type: string;
  notes: string;
  status: TranslationStatus;
}

export type TranslationShortcut = "confirm-next" | "next-block" | "previous-block";

export function createTranslationPatch(draft: TranslationDraft): Partial<StudioTextLayer> {
  return {
    translated: draft.translated,
    traduzido: draft.translated,
    tipo: draft.type,
    translation_notes: draft.notes,
    translation_status: draft.status,
  };
}

export function findNextPendingTranslationTarget(
  project: StudioProject,
  currentPageIndex: number,
  currentLayerId: string | null,
): TranslationTarget | null {
  const all = buildTranslationQueue(project);
  if (all.length === 0) return null;
  const currentIndex = all.findIndex((item) => item.pageIndex === currentPageIndex && item.layerId === currentLayerId);
  if (currentIndex < 0) {
    const first = all.find((item) => item.status === "pending");
    return first ? { pageIndex: first.pageIndex, layerId: first.layerId } : null;
  }
  for (let offset = 1; offset < all.length; offset += 1) {
    const candidate = all[(currentIndex + offset) % all.length];
    if (candidate.status === "pending") return { pageIndex: candidate.pageIndex, layerId: candidate.layerId };
  }
  return null;
}

export function findAdjacentTranslationTarget(
  project: StudioProject,
  currentPageIndex: number,
  currentLayerId: string | null,
  direction: "next" | "previous",
): TranslationTarget | null {
  const queue = buildTranslationQueue(project);
  if (queue.length === 0) return null;
  const currentIndex = queue.findIndex((item) => item.pageIndex === currentPageIndex && item.layerId === currentLayerId);
  if (currentIndex < 0) {
    const fallback = direction === "next" ? queue[0] : queue[queue.length - 1];
    return { pageIndex: fallback.pageIndex, layerId: fallback.layerId };
  }
  const delta = direction === "next" ? 1 : -1;
  const candidate = queue[(currentIndex + delta + queue.length) % queue.length];
  return { pageIndex: candidate.pageIndex, layerId: candidate.layerId };
}

export function translationTargetRequiresPageChange(
  currentPageIndex: number,
  target: TranslationTarget,
): boolean {
  return currentPageIndex !== target.pageIndex;
}

export function translationUsesStudioComposite(
  viewMode: "original" | "inpainted" | "translated",
): boolean {
  return viewMode === "translated";
}

export function translationShortcutFor(input: {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  editableTarget?: boolean;
}): TranslationShortcut | null {
  if ((input.ctrlKey || input.metaKey) && input.key === "Enter") return "confirm-next";
  if (input.altKey && input.key === "ArrowDown") return "next-block";
  if (input.altKey && input.key === "ArrowUp") return "previous-block";
  if (input.editableTarget) return null;
  return null;
}

function isEditableTarget(target: EventTarget | null) {
  return target instanceof HTMLElement && (
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.tagName === "SELECT" ||
    target.isContentEditable ||
    Boolean(target.closest("[contenteditable='true']"))
  );
}

function projectGlossary(project: StudioProject): Record<string, string> {
  const value = project.work_context?.glossary;
  if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
  );
}

export function StudioTranslationWorkspace({
  project,
  layer,
  onChange,
  onConfirmNext,
  onNavigateBlock,
  onUpdateGlossary,
  isSaving = false,
}: {
  project: StudioProject;
  layer: StudioTextLayer | null;
  onChange: (patch: Partial<StudioTextLayer>) => void;
  onConfirmNext: () => void | Promise<void>;
  onNavigateBlock: (direction: "next" | "previous") => void | Promise<void>;
  onUpdateGlossary: (glossary: Record<string, string>) => void | Promise<void>;
  isSaving?: boolean;
}) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const shortcut = translationShortcutFor({
        key: event.key,
        ctrlKey: event.ctrlKey,
        metaKey: event.metaKey,
        altKey: event.altKey,
        editableTarget: isEditableTarget(event.target),
      });
      if (!shortcut) return;
      event.preventDefault();
      if (shortcut === "confirm-next") void onConfirmNext();
      else void onNavigateBlock(shortcut === "next-block" ? "next" : "previous");
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onConfirmNext, onNavigateBlock]);

  const normalizedLayer = layer ? {
    ...layer,
    translation_status: resolveTranslationStatus(layer),
  } : null;

  return (
    <aside
      className="flex h-full w-[318px] shrink-0 flex-col border-l border-border bg-bg-secondary/70"
      aria-label="Tradução manual"
      data-editor-preserve-text-selection="true"
    >
      <header className="flex items-center gap-2 border-b border-border px-3 py-2.5">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-accent-cyan/10 text-accent-cyan">
          <Languages size={13} />
        </span>
        <div>
          <h2 className="text-[11px] font-semibold text-text-primary">Tradução manual</h2>
          <p className="text-[9px] text-text-muted">Texto, revisão e contexto local</p>
        </div>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <TranslationInspector
          layer={normalizedLayer}
          onChange={onChange}
          onConfirmNext={onConfirmNext}
          isSaving={isSaving}
        />
        <GlossaryPanel glossary={projectGlossary(project)} onChange={onUpdateGlossary} />
      </div>
    </aside>
  );
}
