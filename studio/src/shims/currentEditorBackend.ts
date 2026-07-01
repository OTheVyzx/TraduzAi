import type { EditorBackendApi } from "../../../src/lib/editorBackend";

let configuredBackend: EditorBackendApi | null = null;

export function configureEditorBackend(backend: EditorBackendApi | null) {
  configuredBackend = backend;
}

export async function getEditorBackend(): Promise<EditorBackendApi> {
  if (configuredBackend) return configuredBackend;
  throw new Error("Studio editor backend is not configured");
}

export type {
  BitmapRegionConfig,
  EditorBackendApi,
  EditorPagePayload,
  PageActionChangedAsset,
  PageActionName,
  PageActionResult,
  ProcessRegionResult,
  RegionalInpaintResult,
} from "../../../src/lib/editorBackend";
