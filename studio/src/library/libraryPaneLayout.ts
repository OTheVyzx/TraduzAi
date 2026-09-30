export type LibraryPaneSide = "left" | "right";

export type LibraryPaneLayout = {
  left: number;
  right: number;
};

export const LIBRARY_PANE_STORAGE_KEY = "traduzai:library-pane-layout:v1";
export const DEFAULT_LIBRARY_PANES: LibraryPaneLayout = { left: 292, right: 300 };

const MIN_LEFT = 220;
const MAX_LEFT = 480;
const MIN_RIGHT = 240;
const MAX_RIGHT = 520;
const MIN_CENTER = 400;
const DIVIDER_SPACE = 20;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
}

export function clampLibraryPaneLayout(layout: LibraryPaneLayout, availableWidth: number): LibraryPaneLayout {
  const right = clamp(layout.right, MIN_RIGHT, MAX_RIGHT);
  const maximumLeft = Math.min(MAX_LEFT, availableWidth - right - MIN_CENTER - DIVIDER_SPACE);
  const left = clamp(layout.left, MIN_LEFT, maximumLeft);
  const maximumRight = Math.min(MAX_RIGHT, availableWidth - left - MIN_CENTER - DIVIDER_SPACE);
  return { left, right: clamp(right, MIN_RIGHT, maximumRight) };
}

export function resizeLibraryPane(
  layout: LibraryPaneLayout,
  side: LibraryPaneSide,
  pointerDelta: number,
  availableWidth: number,
): LibraryPaneLayout {
  const proposed = side === "left"
    ? { ...layout, left: layout.left + pointerDelta }
    : { ...layout, right: layout.right - pointerDelta };
  return clampLibraryPaneLayout(proposed, availableWidth);
}

export function parseLibraryPaneLayout(raw: string | null, availableWidth: number): LibraryPaneLayout {
  if (!raw) return clampLibraryPaneLayout(DEFAULT_LIBRARY_PANES, availableWidth);
  try {
    const parsed = JSON.parse(raw) as Partial<LibraryPaneLayout>;
    if (!Number.isFinite(parsed.left) || !Number.isFinite(parsed.right)) return clampLibraryPaneLayout(DEFAULT_LIBRARY_PANES, availableWidth);
    return clampLibraryPaneLayout({ left: Number(parsed.left), right: Number(parsed.right) }, availableWidth);
  } catch {
    return clampLibraryPaneLayout(DEFAULT_LIBRARY_PANES, availableWidth);
  }
}
