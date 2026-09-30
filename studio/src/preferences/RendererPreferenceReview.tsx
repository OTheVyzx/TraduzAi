import { useMemo, useState } from "react";
import { Check, Equal, HelpCircle, ThumbsDown, ZoomIn } from "lucide-react";
import type { RendererPreferenceResponsePayload } from "../backend/studioIpcV1";

export const RENDERER_PREFERENCE_SCHEMA = "traduzai.renderer-preference.v1" as const;
export const RENDERER_PREFERENCE_RESPONSE_SCHEMA = "traduzai.renderer-preference-response.v1" as const;

export type PreferenceChoice = "A" | "B" | "equivalent" | "neither" | "unsure";
export type PreferenceActor = { kind: "human" | "model" | "system"; id: string };

export interface PreferenceArtifactRef {
  relative_path: string;
  sha256: string;
}

export interface RasterSafetyEvidence {
  schema: "traduzai.raster-safety.v1";
  status: "pass" | "review_required";
  bbox: [number, number, number, number];
  ink_pixel_count: number;
  alpha_sha256: string;
  authorized_body_sha256: string;
  protected_art_sha256: string;
  outside_authorized_body_px: number;
  protected_art_overlap_px: number;
  evidence_sha256: string;
}

export interface PreferenceCandidate {
  schema: typeof RENDERER_PREFERENCE_SCHEMA;
  owner_id: string;
  target_text: string;
  target_sha256: string;
  source_sha256: string;
  style_sha256: string;
  layout_plan_sha256: string;
  recipe_sha256: string;
  output_sha256: string;
  preview_ref: PreferenceArtifactRef;
  context_ref: PreferenceArtifactRef;
  metrics: Record<string, unknown>;
  hard_safety_passed: boolean;
  preference_profile: "uncalibrated";
  candidate_id: string;
}

export interface PreferenceComparison {
  schema: typeof RENDERER_PREFERENCE_SCHEMA;
  candidates: [PreferenceCandidate, PreferenceCandidate];
  positions: { A: string; B: string };
  randomization_sha256: string;
  preference_profile: "uncalibrated";
  comparison_sha256: string;
}

export interface PreferenceResponse extends RendererPreferenceResponsePayload {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireSha256(value: unknown, field: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new Error(`${field} exige SHA-256 minúsculo`);
  }
  return value;
}

function parseArtifactRef(value: unknown, field: string): PreferenceArtifactRef {
  if (!isRecord(value) || typeof value.relative_path !== "string" || !value.relative_path
    || value.relative_path.includes("\\") || value.relative_path.startsWith("/")
    || value.relative_path.includes(":") || value.relative_path.split("/").some((part) => !part || part === "." || part === "..")) {
    throw new Error(`${field} exige caminho relativo portátil`);
  }
  return { relative_path: value.relative_path, sha256: requireSha256(value.sha256, `${field}.sha256`) };
}

function parseRasterSafety(value: unknown): RasterSafetyEvidence {
  if (!isRecord(value) || value.schema !== "traduzai.raster-safety.v1") {
    throw new Error("Candidata sem evidência raster de segurança");
  }
  const bbox = value.bbox;
  if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every(Number.isInteger)
    || Number(bbox[2]) <= Number(bbox[0]) || Number(bbox[3]) <= Number(bbox[1])) {
    throw new Error("Evidência raster exige uma caixa válida");
  }
  const integerFields = ["ink_pixel_count", "outside_authorized_body_px", "protected_art_overlap_px"] as const;
  for (const field of integerFields) {
    if (!Number.isInteger(value[field]) || Number(value[field]) < 0) {
      throw new Error(`Evidência raster inválida em ${field}`);
    }
  }
  const outside = Number(value.outside_authorized_body_px);
  const protectedOverlap = Number(value.protected_art_overlap_px);
  const expectedStatus = outside === 0 && protectedOverlap === 0 ? "pass" : "review_required";
  if (value.status !== expectedStatus) throw new Error("Status raster diverge das colisões medidas");
  if (outside > 0) throw new Error("Composição ultrapassa a área de escrita autorizada");
  if (protectedOverlap > 0) throw new Error("Composição intersecta arte protegida");
  return {
    schema: "traduzai.raster-safety.v1",
    status: "pass",
    bbox: [Number(bbox[0]), Number(bbox[1]), Number(bbox[2]), Number(bbox[3])],
    ink_pixel_count: Number(value.ink_pixel_count),
    alpha_sha256: requireSha256(value.alpha_sha256, "raster_safety.alpha_sha256"),
    authorized_body_sha256: requireSha256(value.authorized_body_sha256, "raster_safety.authorized_body_sha256"),
    protected_art_sha256: requireSha256(value.protected_art_sha256, "raster_safety.protected_art_sha256"),
    outside_authorized_body_px: outside,
    protected_art_overlap_px: protectedOverlap,
    evidence_sha256: requireSha256(value.evidence_sha256, "raster_safety.evidence_sha256"),
  };
}

function parseCandidate(value: unknown): PreferenceCandidate {
  if (!isRecord(value) || value.schema !== RENDERER_PREFERENCE_SCHEMA
    || value.preference_profile !== "uncalibrated") {
    throw new Error("Candidata usa contrato de preferência incompatível");
  }
  const textFields = ["owner_id", "target_text", "candidate_id"] as const;
  for (const field of textFields) {
    if (typeof value[field] !== "string" || !value[field]) throw new Error(`Candidata sem ${field}`);
  }
  if (value.hard_safety_passed !== true) throw new Error("Candidatas sem segurança aprovada não podem ser comparadas");
  if (!isRecord(value.metrics)) throw new Error("Candidata sem métricas do Renderer");
  const rasterSafety = parseRasterSafety(value.metrics.raster_safety);
  return {
    schema: RENDERER_PREFERENCE_SCHEMA,
    owner_id: value.owner_id as string,
    target_text: value.target_text as string,
    target_sha256: requireSha256(value.target_sha256, "target_sha256"),
    source_sha256: requireSha256(value.source_sha256, "source_sha256"),
    style_sha256: requireSha256(value.style_sha256, "style_sha256"),
    layout_plan_sha256: requireSha256(value.layout_plan_sha256, "layout_plan_sha256"),
    recipe_sha256: requireSha256(value.recipe_sha256, "recipe_sha256"),
    output_sha256: requireSha256(value.output_sha256, "output_sha256"),
    preview_ref: parseArtifactRef(value.preview_ref, "preview_ref"),
    context_ref: parseArtifactRef(value.context_ref, "context_ref"),
    metrics: { ...value.metrics, raster_safety: rasterSafety },
    hard_safety_passed: true,
    preference_profile: "uncalibrated",
    candidate_id: value.candidate_id as string,
  };
}

export function parsePreferenceComparison(value: unknown): PreferenceComparison {
  if (!isRecord(value) || value.schema !== RENDERER_PREFERENCE_SCHEMA
    || value.preference_profile !== "uncalibrated"
    || !Array.isArray(value.candidates) || value.candidates.length !== 2) {
    throw new Error("Comparação de preferência incompleta ou incompatível");
  }
  const candidates = [parseCandidate(value.candidates[0]), parseCandidate(value.candidates[1])] as [PreferenceCandidate, PreferenceCandidate];
  const [first, second] = candidates;
  if (first.candidate_id === second.candidate_id) throw new Error("A comparação exige candidatas distintas");
  if (first.owner_id !== second.owner_id || first.target_sha256 !== second.target_sha256
    || first.source_sha256 !== second.source_sha256 || first.style_sha256 !== second.style_sha256) {
    throw new Error("A comparação exige mesmo owner, texto, origem e estilo");
  }
  const positions = value.positions;
  if (!isRecord(positions) || typeof positions.A !== "string" || typeof positions.B !== "string"
    || new Set([positions.A, positions.B]).size !== 2
    || ![first.candidate_id, second.candidate_id].every((id) => id === positions.A || id === positions.B)) {
    throw new Error("As posições A/B não correspondem às candidatas do Renderer");
  }
  return {
    schema: RENDERER_PREFERENCE_SCHEMA,
    candidates,
    positions: { A: positions.A, B: positions.B },
    randomization_sha256: requireSha256(value.randomization_sha256, "randomization_sha256"),
    preference_profile: "uncalibrated",
    comparison_sha256: requireSha256(value.comparison_sha256, "comparison_sha256"),
  };
}

function candidateAt(comparison: PreferenceComparison, position: "A" | "B") {
  const candidate = comparison.candidates.find((item) => item.candidate_id === comparison.positions[position]);
  if (!candidate) throw new Error(`Posição ${position} sem candidata`);
  return candidate;
}

export function createPreferenceResponse(input: {
  comparison: PreferenceComparison;
  choice: PreferenceChoice;
  actor: PreferenceActor;
  recordedAt: string;
  trainingOptIn?: boolean;
}): PreferenceResponse {
  const comparison = parsePreferenceComparison(input.comparison);
  if (!input.actor.id.trim()) throw new Error("A preferência exige identificação de proveniência");
  if (input.actor.kind === "system" && input.actor.id !== "test/synthetic") {
    throw new Error("Automação de preferência deve usar o ator test/synthetic");
  }
  const displayedA = candidateAt(comparison, "A");
  const displayedB = candidateAt(comparison, "B");
  return {
    schema: RENDERER_PREFERENCE_RESPONSE_SCHEMA,
    comparison_sha256: comparison.comparison_sha256,
    choice: input.choice,
    selected_candidate_id: input.choice === "A" || input.choice === "B"
      ? comparison.positions[input.choice]
      : null,
    displayed_candidate_ids: { ...comparison.positions },
    candidate_recipe_sha256s: [displayedA.recipe_sha256, displayedB.recipe_sha256],
    candidate_output_sha256s: [displayedA.output_sha256, displayedB.output_sha256],
    target_sha256: displayedA.target_sha256,
    randomization_sha256: comparison.randomization_sha256,
    actor_kind: input.actor.kind,
    actor_id: input.actor.id,
    recorded_at: input.recordedAt,
    training_eligible: input.actor.kind === "human" && input.trainingOptIn === true,
  };
}

const CHOICES: Array<{ value: PreferenceChoice; label: string; icon: typeof Check }> = [
  { value: "A", label: "Escolher A", icon: Check },
  { value: "B", label: "Escolher B", icon: Check },
  { value: "equivalent", label: "Equivalentes", icon: Equal },
  { value: "neither", label: "Nenhuma", icon: ThumbsDown },
  { value: "unsure", label: "Não tenho certeza", icon: HelpCircle },
];

export function RendererPreferenceReview({
  comparison: rawComparison,
  actor,
  resolveArtifact,
  onSubmit,
}: {
  comparison: PreferenceComparison;
  actor: PreferenceActor;
  resolveArtifact: (relativePath: string) => string;
  onSubmit: (response: PreferenceResponse) => Promise<void>;
}) {
  const comparison = useMemo(() => parsePreferenceComparison(rawComparison), [rawComparison]);
  const [zoom, setZoom] = useState(100);
  const [trainingOptIn, setTrainingOptIn] = useState(false);
  const [saving, setSaving] = useState<PreferenceChoice | null>(null);
  const [error, setError] = useState<string | null>(null);
  const context = candidateAt(comparison, "A").context_ref;

  const submit = async (choice: PreferenceChoice) => {
    setSaving(choice);
    setError(null);
    try {
      await onSubmit(createPreferenceResponse({
        comparison,
        choice,
        actor,
        recordedAt: new Date().toISOString(),
        trainingOptIn,
      }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSaving(null);
    }
  };

  return <section className="studio-preference-review" aria-label="Comparação de preferência do Renderer">
    <header>
      <div><p>Preferência de composição</p><h2>Comparar composição</h2></div>
      <label><ZoomIn size={15} /> Zoom igual
        <input type="range" min="80" max="160" value={zoom} onChange={(event) => setZoom(Number(event.currentTarget.value))} />
        <span>{zoom}%</span>
      </label>
    </header>
    <figure className="studio-preference-context">
      <figcaption>Contexto original</figcaption>
      <img src={resolveArtifact(context.relative_path)} alt="Contexto original da fala" />
    </figure>
    <p className="studio-preference-target">Texto comparado: <strong>{candidateAt(comparison, "A").target_text}</strong></p>
    <div className="studio-preference-grid">
      {(["A", "B"] as const).map((position) => {
        const item = candidateAt(comparison, position);
        return <figure key={position}>
          <figcaption>Opção {position}</figcaption>
          <div className="studio-preference-preview-frame">
            <img style={{ transform: `scale(${zoom / 100})` }} src={resolveArtifact(item.preview_ref.relative_path)} alt={`Composição ${position}`} />
          </div>
        </figure>;
      })}
    </div>
    <div className="studio-preference-actions" aria-label="Escolha de preferência">
      {CHOICES.map(({ value, label, icon: Icon }) => <button key={value} type="button" disabled={saving !== null} onClick={() => void submit(value)}>
        <Icon size={14} /> {saving === value ? "Salvando…" : label}
      </button>)}
    </div>
    <label className="studio-preference-opt-in">
      <input type="checkbox" checked={trainingOptIn} disabled={actor.kind !== "human"} onChange={(event) => setTrainingOptIn(event.currentTarget.checked)} />
      Permitir que esta escolha humana seja exportada para aprendizado futuro
    </label>
    {actor.kind !== "human" && <p className="studio-preference-provenance">Teste automatizado: registrado como {actor.id} e excluído de treinamento.</p>}
    {error && <p role="alert" className="studio-preference-error">{error}</p>}
  </section>;
}
