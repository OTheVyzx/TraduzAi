import type { Project } from "./stores/appStore";

export interface OnboardingStep {
  id: string;
  label: string;
  status: "done" | "warning" | "todo";
}

export const ONBOARDING_FLOW = [
  "Importe um capitulo",
  "Selecione a obra",
  "Busque contexto online",
  "Revise o glossario",
  "Traduza",
  "Corrija alertas",
  "Exporte",
];

export function buildOnboardingChecklist(project: Project | null, translated = false, reviewed = false, exported = false): OnboardingStep[] {
  const hasWork = Boolean(project?.obra?.trim() || project?.work_context?.selected);
  const contextLoaded = Boolean(project?.work_context?.internet_context_loaded || project?.work_context?.context_loaded);
  const glossaryLoaded = Boolean(project?.work_context?.glossary_loaded || Object.keys(project?.contexto.glossario ?? {}).length > 0);

  return [
    { id: "import", label: "Capitulo importado", status: project ? "done" : "todo" },
    { id: "work", label: "Obra selecionada", status: hasWork ? "done" : "warning" },
    { id: "context", label: "Contexto online carregado", status: contextLoaded ? "done" : "warning" },
    { id: "glossary", label: "Glossario revisado", status: glossaryLoaded ? "done" : "warning" },
    { id: "translation", label: "Traducao iniciada", status: translated ? "done" : "todo" },
    { id: "review", label: "Revisao final", status: reviewed ? "done" : "todo" },
    { id: "export", label: "Export", status: exported ? "done" : "todo" },
  ];
}
