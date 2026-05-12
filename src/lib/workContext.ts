// Esquema de contexto da obra — usado para guiar a tradução e o typesetting
// Persiste no project.json sob a chave "work_context"

export type WorkCharacter = {
  id: string;
  name: string;
  aliases?: string[];
  gender?: string;
  role?: string;
  relationshipNotes?: string;
  speechStyle?: string;
  doNotTranslateName?: boolean;
  preferredPortugueseName?: string;
};

export type WorkGlossaryEntry = {
  id: string;
  source: string;
  target: string;
  category?: "name" | "place" | "skill" | "title" | "item" | "organization" | "other";
  notes?: string;
  /** Se verdadeiro, a entrada sempre aparece no prompt de tradução e tem prioridade máxima */
  locked?: boolean;
};

export type WorkStyleGuide = {
  formality?: "auto" | "informal" | "neutral" | "formal";
  honorifics?: "keep" | "adapt" | "remove";
  soundEffects?: "keep_original" | "translate" | "both";
};

export type WorkContext = {
  title?: string;
  synopsis?: string;
  genre?: string[];
  tone?: string;
  translationRules?: string[];
  characters?: WorkCharacter[];
  glossary?: WorkGlossaryEntry[];
  styleGuide?: WorkStyleGuide;
  chapterSummary?: string;
  version?: number;
  updatedAt?: string;
};

/** Retorna um WorkContext vazio e válido (usado para projetos sem contexto salvo) */
export const defaultWorkContext = (): WorkContext => ({
  genre: [],
  characters: [],
  glossary: [],
  translationRules: [],
  version: 1,
});
