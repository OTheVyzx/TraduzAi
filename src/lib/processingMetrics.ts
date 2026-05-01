import type { PipelineLogEntry, PipelineProgress, SystemProfile } from "./stores/appStore";

export const PERCEIVED_PROCESSING_STEPS = [
  "Extraindo imagens",
  "Detectando regioes",
  "Lendo texto",
  "Normalizando OCR",
  "Buscando contexto",
  "Aplicando glossario",
  "Traduzindo",
  "Gerando mascaras",
  "Inpaintando",
  "Renderizando",
  "Rodando QA",
  "Exportando",
];

export function pagesPerMinute(pipeline: PipelineProgress | null, elapsedSeconds: number) {
  if (!pipeline || elapsedSeconds <= 0 || pipeline.current_page <= 0) return 0;
  return (pipeline.current_page / elapsedSeconds) * 60;
}

export function countFlagLogs(logs: PipelineLogEntry[]) {
  return logs.filter((entry) => /flag|qa|critical|warning/i.test(entry.message)).length;
}

export function hardwareUsageLabel(profile: SystemProfile | null) {
  if (!profile) return "Hardware em deteccao";
  return profile.gpu_available ? `GPU ativa: ${profile.gpu_name}` : "CPU ativa";
}
