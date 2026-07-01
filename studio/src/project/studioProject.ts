export const STUDIO_SCHEMA_VERSION = "1.0" as const;
export const COMPAT_PROJECT_VERSION = "2.0" as const;

export type ImageLayerKey = "base" | "mask" | "inpaint" | "brush" | "recovery" | "rendered";

export interface StudioImageLayer {
  key: ImageLayerKey;
  path: string | null;
  visible: boolean;
  locked: boolean;
  opacity?: number;
  order?: number;
  technical?: boolean;
}

export interface StudioTextStyle {
  fontFamily?: string;
  fontSize?: number;
  color?: string;
  strokeColor?: string;
  strokeWidth?: number;
  align?: "left" | "center" | "right";
  vertical?: boolean;
  [key: string]: unknown;
}

export interface StudioTextLayer {
  id: string;
  kind: "text";
  original: string;
  translated: string;
  traduzido: string;
  bbox: [number, number, number, number];
  source_bbox?: [number, number, number, number];
  layout_bbox?: [number, number, number, number];
  render_bbox?: [number, number, number, number];
  tipo?: string;
  style: StudioTextStyle;
  estilo: StudioTextStyle;
  visible: boolean;
  locked: boolean;
  order: number;
  ocr_confidence?: number | null;
  confianca_ocr?: number | null;
  qa_flags?: unknown[];
  [key: string]: unknown;
}

export interface StudioPage {
  numero: number;
  arquivo_original?: string | null;
  arquivo_traduzido?: string | null;
  image_layers: Partial<Record<ImageLayerKey, StudioImageLayer>>;
  text_layers: StudioTextLayer[];
  textos: StudioTextLayer[];
  inpaint_blocks?: unknown[];
  process_overlays?: unknown[];
  editor_cache?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface StudioProject {
  app: "traduzai";
  versao: typeof COMPAT_PROJECT_VERSION;
  studio_schema_version: typeof STUDIO_SCHEMA_VERSION;
  id?: string;
  job_id?: string;
  obra?: string;
  capitulo?: string | number;
  idioma_origem?: string;
  idioma_destino?: string;
  source_path?: string | null;
  output_path?: string | null;
  _work_dir?: string | null;
  engine_preset_id?: string;
  paginas: StudioPage[];
  qa?: unknown;
  estatisticas?: Record<string, unknown>;
  work_context?: Record<string, unknown>;
  [key: string]: unknown;
}

export type ProjectImportKind = "studio_project" | "traduzai_v1" | "traduzai_v2" | "v12_analysis_project";

export interface ProjectImportResult {
  kind: ProjectImportKind;
  project: StudioProject;
  warnings: string[];
}
