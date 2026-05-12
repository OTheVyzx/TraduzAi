const API_URL = import.meta.env.VITE_API_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: "Falha na API" }));
    throw new Error(detail.detail ?? "Falha na API");
  }
  return response.json();
}

export type PageActionRegion = {
  bbox?: [number, number, number, number];
  mask_path?: string;
};

export type PageActionPayload = {
  action: string;
  region?: PageActionRegion;
  block_id?: string;
  mode?: string;
};

export type BitmapLayerUpdatePayload = {
  png_data?: string;
  op?: "replace" | "add" | "subtract";
  dirty_bbox?: number[];
  color?: string;
  opacity?: number;
  hardness?: number;
  width?: number;
  height?: number;
  brush_size?: number;
  clear?: boolean;
  erase?: boolean;
  strokes?: [number, number][][];
};

export const editorApi = {
  loadEditorPage: (projectId: string, pageIndex: number) => request<{ project: any; page: any; page_index: number }>(`/api/projects/${projectId}/editor/pages/${pageIndex}`),
  patchTextLayer: (projectId: string, pageIndex: number, layerId: string, patch: unknown) => request<{ layer: any }>(`/api/projects/${projectId}/editor/pages/${pageIndex}/text-layers/${layerId}`, {
    method: "PATCH",
    body: JSON.stringify({ patch }),
  }),
  createTextLayer: (projectId: string, pageIndex: number, layer: unknown) => request<{ layer: any }>(`/api/projects/${projectId}/editor/pages/${pageIndex}/text-layers`, {
    method: "POST",
    body: JSON.stringify({ layer }),
  }),
  deleteTextLayer: (projectId: string, pageIndex: number, layerId: string) => request<{ ok: true }>(`/api/projects/${projectId}/editor/pages/${pageIndex}/text-layers/${layerId}`, { method: "DELETE" }),
  setLayerVisibility: (
    projectId: string,
    payload: { layer: string; visible: boolean; page_index?: number; layer_kind?: "image" | "text"; layer_key?: string | null; layer_id?: string | null },
  ) => request<{ ok: true }>(`/api/projects/${projectId}/editor/visibility`, {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  updateBitmapLayer: (projectId: string, pageIndex: number, layer: "mask" | "brush" | "recovery", payload: BitmapLayerUpdatePayload) => request<{ asset_path: string; url: string }>(`/api/projects/${projectId}/editor/pages/${pageIndex}/${layer}`, {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  writeMaskFromPng: (projectId: string, pageIndex: number, payload: BitmapLayerUpdatePayload) => request<{ asset_path: string; url: string }>(`/api/projects/${projectId}/editor/pages/${pageIndex}/mask/png`, {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  runPageAction: (
    projectId: string,
    pageIndex: number,
    actionOrPayload: string | PageActionPayload,
    region?: PageActionRegion,
  ) => {
    const payload = typeof actionOrPayload === "string" ? { action: actionOrPayload, region } : actionOrPayload;
    return request<{ ok: true; changed_assets: string[]; page: any }>(`/api/projects/${projectId}/editor/pages/${pageIndex}/actions`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
};
