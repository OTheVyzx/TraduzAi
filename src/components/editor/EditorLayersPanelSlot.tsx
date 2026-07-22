import type { ReactNode } from "react";
import type { EditorMode } from "./editorMode";
import { LayersPanel } from "./LayersPanel";

export function EditorLayersPanelSlot({
  mode,
  panel,
}: {
  mode: EditorMode;
  panel?: ReactNode;
}) {
  return panel ?? <LayersPanel mode={mode} />;
}
