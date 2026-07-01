export type WebProjectMode = "auto" | "manual" | "batch";
export type WebProjectQuality = "rapida" | "normal" | "alta";

export type GlossaryCandidate = {
  kind: string;
  source: string;
  target: string;
  confidence: number;
  status?: "pending" | "reviewed" | "rejected";
};

export interface WebProjectConfig {
  mode: WebProjectMode;
  obra: string;
  capitulo: string;
  idioma_origem: string;
  idioma_destino: string;
  preset_id: string;
  preset?: unknown;
  qualidade: WebProjectQuality;
  export_mode: "clean" | "with_warnings" | "debug";
  contexto: {
    sinopse: string;
    genero: string[];
    personagens: string[];
    termos: string[];
    faccoes: string[];
    aliases: Record<string, string[]>;
    glossario: Record<string, string>;
    memoria_lexical: Record<string, string>;
    internet_context?: {
      internet_context_loaded?: boolean;
      rejected_glossary_candidates?: string[];
      glossary_candidates?: GlossaryCandidate[];
    };
  };
  work_context?: {
    selected: boolean;
    work_id: string;
    title: string;
    cover_url?: string;
    context_loaded: boolean;
    internet_context_loaded: boolean;
    glossary_loaded: boolean;
    glossary_entries_count: number;
    risk_level: "high" | "medium" | "low";
    user_ignored_warning?: boolean;
  };
}

export const emptyProjectConfig = (mode: WebProjectMode): WebProjectConfig => ({
  mode,
  obra: "Projeto sem nome",
  capitulo: "1",
  idioma_origem: "en",
  idioma_destino: "pt-BR",
  preset_id: "scan-clean",
  qualidade: "normal",
  export_mode: "clean",
  contexto: {
    sinopse: "",
    genero: [],
    personagens: [],
    termos: [],
    faccoes: [],
    aliases: {},
    glossario: {},
    memoria_lexical: {},
    internet_context: {
      internet_context_loaded: false,
      rejected_glossary_candidates: [],
      glossary_candidates: [],
    },
  },
});
