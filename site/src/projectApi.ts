const API_URL = import.meta.env.VITE_API_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    credentials: "include",
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: "Falha na API" }));
    throw new Error(detail.detail ?? "Falha na API");
  }
  return response.json();
}

export type ProjectLayerMap = Record<string, { asset_path: string; url: string }>;

export const assetUrl = (path: string) => `${API_URL}${path}`;

export const projectApi = {
  materialize: (projectId: string) => request<{ project_id: string; page_count: number }>(`/api/jobs/${projectId}/materialize-project`, { method: "POST" }),
  getProject: (projectId: string) => request<{ project: any; state: any }>(`/api/projects/${projectId}`),
  saveProject: (projectId: string, project: any) => request<{ ok: true }>(`/api/projects/${projectId}`, {
    method: "PUT",
    body: JSON.stringify(project),
  }),
  getPage: (projectId: string, pageIndex: number) => request<{ page: any; layers: ProjectLayerMap; state: any }>(`/api/projects/${projectId}/pages/${pageIndex}`),
  renderPreview: (projectId: string, pageIndex: number) => request<{ preview_url: string; asset_path: string }>(`/api/projects/${projectId}/pages/${pageIndex}/render-preview`, { method: "POST" }),
  exportProject: (projectId: string, format: "zip-full" | "cbz" | "jpg-zip") => request<{ artifact: { id: string; filename: string; download_url: string } }>(`/api/projects/${projectId}/exports/${format}`, { method: "POST" }),
  exportPsdPage: (projectId: string, pageIndex: number) => request<{ artifact: { id: string; filename: string; download_url: string } }>(`/api/projects/${projectId}/exports/psd-page`, {
    method: "POST",
    body: JSON.stringify({ page_index: pageIndex }),
  }),
  saveSettings: (projectId: string, payload: any) => request<{ ok: true }>(`/api/projects/${projectId}/settings`, {
    method: "PUT",
    body: JSON.stringify(payload),
  }),
};
