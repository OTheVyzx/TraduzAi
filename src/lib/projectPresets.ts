import type { ProjectQuality } from "./stores/appStore";

export interface ProjectPresetSettings {
  ocr_sensitivity: "low" | "normal" | "high";
  ocr_cleanup: "light" | "normal" | "strong";
  translation_style: "natural_br" | "literal";
  typesetting_font_scale: number;
  balloon_margin: number;
  sfx_mode: "preserve" | "partial_translate";
  inpaint_mode: "conservative" | "normal" | "aggressive";
  qa_mode: "review" | "strict" | "debug";
}

export interface ProjectPreset {
  id: string;
  name: string;
  description: string;
  quality: ProjectQuality;
  settings: ProjectPresetSettings;
  custom?: boolean;
}

export const PROJECT_PRESETS: ProjectPreset[] = [
  {
    id: "manhwa_webtoon_color",
    name: "Manhwa/Webtoon colorido",
    description: "Ideal para capitulos coloridos verticais. Traducao natural, fonte maior, revisao de SFX manual.",
    quality: "alta",
    settings: {
      ocr_sensitivity: "high",
      ocr_cleanup: "normal",
      translation_style: "natural_br",
      typesetting_font_scale: 1.08,
      balloon_margin: 0.12,
      sfx_mode: "preserve",
      inpaint_mode: "normal",
      qa_mode: "review",
    },
  },
  {
    id: "manga_bw",
    name: "Manga preto e branco",
    description: "Melhor para paginas P&B, com inpaint conservador e texto mais compacto.",
    quality: "alta",
    settings: {
      ocr_sensitivity: "normal",
      ocr_cleanup: "strong",
      translation_style: "natural_br",
      typesetting_font_scale: 0.98,
      balloon_margin: 0.09,
      sfx_mode: "preserve",
      inpaint_mode: "conservative",
      qa_mode: "review",
    },
  },
  {
    id: "manhua_color",
    name: "Manhua colorido",
    description: "Para obras coloridas com termos de cultivo, nomes e faccoes recorrentes.",
    quality: "alta",
    settings: {
      ocr_sensitivity: "high",
      ocr_cleanup: "normal",
      translation_style: "literal",
      typesetting_font_scale: 1.04,
      balloon_margin: 0.1,
      sfx_mode: "partial_translate",
      inpaint_mode: "normal",
      qa_mode: "strict",
    },
  },
  {
    id: "small_balloons",
    name: "Baloes pequenos",
    description: "Reduz tamanho de fonte, aumenta rigor de QA e evita texto apertado.",
    quality: "alta",
    settings: {
      ocr_sensitivity: "high",
      ocr_cleanup: "strong",
      translation_style: "natural_br",
      typesetting_font_scale: 0.9,
      balloon_margin: 0.16,
      sfx_mode: "preserve",
      inpaint_mode: "conservative",
      qa_mode: "strict",
    },
  },
  {
    id: "scanlation_clean",
    name: "Scanlation clean",
    description: "Foco em limpeza visual, revisao estrita e export com menos tolerancia a falhas.",
    quality: "alta",
    settings: {
      ocr_sensitivity: "normal",
      ocr_cleanup: "strong",
      translation_style: "natural_br",
      typesetting_font_scale: 1,
      balloon_margin: 0.12,
      sfx_mode: "partial_translate",
      inpaint_mode: "aggressive",
      qa_mode: "strict",
    },
  },
  {
    id: "natural_br",
    name: "Traducao natural BR",
    description: "Prioriza fluidez em portugues brasileiro sem perder nomes protegidos.",
    quality: "normal",
    settings: {
      ocr_sensitivity: "normal",
      ocr_cleanup: "normal",
      translation_style: "natural_br",
      typesetting_font_scale: 1,
      balloon_margin: 0.11,
      sfx_mode: "preserve",
      inpaint_mode: "normal",
      qa_mode: "review",
    },
  },
  {
    id: "literal",
    name: "Traducao mais literal",
    description: "Mantem estrutura mais fiel ao original e aumenta peso do glossario.",
    quality: "normal",
    settings: {
      ocr_sensitivity: "normal",
      ocr_cleanup: "normal",
      translation_style: "literal",
      typesetting_font_scale: 0.98,
      balloon_margin: 0.1,
      sfx_mode: "preserve",
      inpaint_mode: "normal",
      qa_mode: "review",
    },
  },
  {
    id: "sfx_preserve",
    name: "SFX preservar",
    description: "Mantem efeitos sonoros no original e sinaliza revisao manual quando necessario.",
    quality: "normal",
    settings: {
      ocr_sensitivity: "normal",
      ocr_cleanup: "light",
      translation_style: "natural_br",
      typesetting_font_scale: 1,
      balloon_margin: 0.1,
      sfx_mode: "preserve",
      inpaint_mode: "conservative",
      qa_mode: "review",
    },
  },
  {
    id: "sfx_partial",
    name: "SFX traduzir parcial",
    description: "Traduz SFX simples e preserva efeitos complexos para revisao.",
    quality: "normal",
    settings: {
      ocr_sensitivity: "normal",
      ocr_cleanup: "normal",
      translation_style: "natural_br",
      typesetting_font_scale: 1,
      balloon_margin: 0.1,
      sfx_mode: "partial_translate",
      inpaint_mode: "normal",
      qa_mode: "review",
    },
  },
];

export function getProjectPreset(id?: string | null) {
  return PROJECT_PRESETS.find((preset) => preset.id === id) ?? PROJECT_PRESETS[0];
}

export function createCustomPreset(base: ProjectPreset, name: string): ProjectPreset {
  return {
    ...base,
    id: `custom_${Date.now()}`,
    name: name.trim() || "Preset customizado",
    description: "Preset customizado criado a partir da configuracao atual.",
    custom: true,
  };
}
