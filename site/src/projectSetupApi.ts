import type { GlossaryCandidate } from "./projectConfig";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8787";

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

export type WorkSearchResult = {
  work_id: string;
  title: string;
  source: string;
  risk_level: "high" | "medium" | "low";
  synopsis?: string;
  source_url?: string;
  cover_url?: string;
  score?: number;
  genres?: string[];
  characters?: string[];
};

export type PresetOption = {
  id: string;
  label: string;
  quality: "rapida" | "normal" | "alta";
  description: string;
};

export const setupApi = {
  languages: () => request<{ languages: { id: string; label: string }[] }>("/api/setup/languages"),
  presets: () => request<{ presets: PresetOption[] }>("/api/setup/presets"),
  searchWork: (query: string) => request<{ results: WorkSearchResult[] }>("/api/setup/work-search", {
    method: "POST",
    body: JSON.stringify({ query }),
  }),
  workContext: (work: WorkSearchResult) => request<{ context: any }>("/api/setup/work-context", {
    method: "POST",
    body: JSON.stringify(work),
  }),
  acceptGlossary: (workId: string, entry: GlossaryCandidate) => request<{ entry: GlossaryCandidate }>(`/api/setup/glossary/${workId}/entries`, {
    method: "POST",
    body: JSON.stringify(entry),
  }),
};
