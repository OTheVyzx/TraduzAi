export const PROJECT_SCHEMA_VERSION_V12 = "12.0" as const;

export type ProjectRunModeV12 = "mock" | "real" | "debug";
export type ProjectExportStatusV12 = "not_exported" | "clean" | "with_warnings" | "blocked";
export type ProjectRegionTypeV12 =
  | "speech_balloon"
  | "caption"
  | "sfx"
  | "background_text"
  | "unknown";
export type ProjectRenderStatusV12 = "pending" | "approved" | "warning" | "blocked" | "ignored";

export interface ProjectRegionV12 {
  region_id: string;
  page: number;
  bbox: [number, number, number, number];
  polygon: unknown[];
  group_id: string | null;
  reading_order: number;
  region_type: ProjectRegionTypeV12;
  raw_ocr: string;
  normalized_ocr: string;
  ocr_confidence: number;
  normalization: {
    changed: boolean;
    corrections: unknown[];
    is_gibberish: boolean;
  };
  entities: unknown[];
  term_protection: {
    protected_text: string;
    placeholders: unknown[];
  };
  translation: {
    text: string;
    engine: string;
    confidence: number;
    used_glossary: unknown[];
    warnings: string[];
  };
  layout: {
    font: string;
    font_size: number;
    fit_score: number;
    overflow: boolean;
  };
  mask: {
    path: string | null;
    type: string | null;
    bbox: [number, number, number, number] | null;
    valid: boolean;
  };
  render_status: ProjectRenderStatusV12;
  qa_flags: string[];
}

export interface ProjectPageV12 {
  page: number;
  source_path?: string | null;
  rendered_path?: string | null;
  width?: number | null;
  height?: number | null;
  regions: ProjectRegionV12[];
}

export interface ProjectSchemaV12 {
  schema_version: typeof PROJECT_SCHEMA_VERSION_V12;
  app: "traduzai";
  run: {
    run_id: string;
    created_at?: string | null;
    started_at?: string | null;
    finished_at?: string | null;
    duration_ms?: number;
    mode: ProjectRunModeV12;
    pipeline_version: typeof PROJECT_SCHEMA_VERSION_V12;
  };
  source: {
    input_path: string;
    page_count: number;
    hash: string;
  };
  work_context: {
    selected: boolean;
    work_id: string | null;
    title: string | null;
    context_loaded: boolean;
    glossary_loaded: boolean;
    glossary_entries_count: number;
    risk_level: "unknown" | "low" | "medium" | "high";
    user_ignored_warning: boolean;
  };
  pages: ProjectPageV12[];
  glossary_hits: unknown[];
  entity_flags: unknown[];
  qa: {
    summary: {
      total_pages: number;
      pages_with_flags: number;
      critical: number;
      high: number;
      medium: number;
      low: number;
    };
    flags: Array<Record<string, unknown>>;
  };
  export_report: {
    status: ProjectExportStatusV12;
    files: unknown[];
  };
  legacy: {
    paginas: unknown[];
  };
}

export function isProjectSchemaV12(value: unknown): value is ProjectSchemaV12 {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { schema_version?: unknown }).schema_version === PROJECT_SCHEMA_VERSION_V12 &&
    (value as { app?: unknown }).app === "traduzai" &&
    Array.isArray((value as { pages?: unknown }).pages)
  );
}
