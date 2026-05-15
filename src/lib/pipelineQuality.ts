export type PipelineQuality = "normal" | "ultra";
export type LegacyPipelineQuality = "rapida" | "alta" | "max" | "maximum";
export type PipelineQualityInput = PipelineQuality | LegacyPipelineQuality | string | null | undefined;

export function normalizePipelineQuality(value: unknown): PipelineQuality {
  const raw = typeof value === "string" ? value.trim().toLocaleLowerCase("pt-BR") : "";
  if (raw === "ultra" || raw === "alta" || raw === "max" || raw === "maximum") {
    return "ultra";
  }
  return "normal";
}

export function formatPipelineQualityLabel(value: unknown): string {
  return normalizePipelineQuality(value) === "ultra" ? "Ultra" : "Normal";
}
