// Shared boundary for TraduzAI Studio editor reuse.
// This barrel intentionally exports UI/state modules that are coupled to the
// current editor runtime. Do not use it as a lightweight utility import path.
export { LayersPanel } from "../components/editor/LayersPanel";
export { PageThumbnails } from "../components/editor/PageThumbnails";
export { EditorStage } from "../components/editor/stage/EditorStage";
export { ToolSidebar } from "../components/editor/toolbar/ToolSidebar";
export { UndoRedoControls } from "../components/editor/toolbar/UndoRedoControls";
export { ZoomControls } from "../components/editor/toolbar/ZoomControls";
export { LayeredBitmapCanvas, bitmapStrokePasses } from "./bitmap/layeredBitmapCanvas";
export { useAppStore } from "../lib/stores/appStore";
export { useEditorStore } from "../lib/stores/editorStore";

export type { Project, TextLayerStyle } from "../lib/stores/appStore";
export type {
  Canvas2DLike,
  CanvasLike,
  DrawBitmapStrokeOptions,
  LayeredBitmapKey,
  LayeredBitmapLayer,
} from "./bitmap/layeredBitmapCanvas";
