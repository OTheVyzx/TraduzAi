/**
 * RenderStatusBadge — Fase 6 do refactor.
 *
 * Indicador visual do Auto Fidelity Render (renderização FT2Font em background).
 *
 * Estados:
 *  - "rendering"  → 🔄 Renderizando fiel…
 *  - "updated"    → ✓ Fiel atualizado
 *  - "stale"      → ⚠ Render desatualizado (clique força render imediato)
 *  - "error"      → ✗ Erro no render fiel
 *  - "idle"       → null (não aparece)
 */

import { useEffect, useMemo, useState } from "react";
import { Loader2, Check, AlertCircle, RefreshCw, Zap } from "lucide-react";
import { getRenderPreviewStateForPage, useEditorStore } from "../../../lib/stores/editorStore";
import type { TextEntry } from "../../../lib/stores/appStore";

const SFX_REVIEW_FLAGS = new Set([
  "sfx_render_missing",
  "sfx_render_outside_source_region",
  "sfx_inpaint_damaged_art_risk",
  "sfx_translation_unknown",
  "sfx_style_low_confidence",
]);

export function RenderStatusBadge() {
  const status = useEditorStore((s) => s.renderStatus);
  const renderError = useEditorStore((s) => s.renderError);
  const forceFidelityRender = useEditorStore((s) => s.forceFidelityRender);
  const pageKey = useEditorStore((s) => s.currentPageKey());
  const currentPage = useEditorStore((s) => s.currentPage);
  const renderPreviewCacheByPageKey = useEditorStore((s) => s.renderPreviewCacheByPageKey);
  const previewState = useMemo(
    () => getRenderPreviewStateForPage(pageKey, currentPage, renderPreviewCacheByPageKey),
    [currentPage, pageKey, renderPreviewCacheByPageKey],
  );
  const previewStatus = previewState.status;
  const backend = previewState.rendererBackend;
  const sfxSummary = useMemo(() => summarizeSfxReview(currentPage?.text_layers ?? []), [currentPage]);

  // Oculta "Fiel atualizado" após 4 segundos
  const [showUpdated, setShowUpdated] = useState(false);
  useEffect(() => {
    if (status === "updated") {
      setShowUpdated(true);
      const id = window.setTimeout(() => setShowUpdated(false), 4000);
      return () => window.clearTimeout(id);
    } else {
      setShowUpdated(false);
    }
  }, [status]);

  if (status === "idle" && sfxSummary.reviewCount > 0) {
    return (
      <div
        className="flex items-center gap-1.5 rounded-lg border border-status-warning/25 bg-status-warning/10 px-2 py-1 text-[10px] text-status-warning"
        title={sfxSummary.title}
      >
        <Zap size={10} />
        SFX: revisar {sfxSummary.reviewCount}
      </div>
    );
  }

  if (status === "idle" && previewStatus === "fresh" && backend) {
    const label = backend === "koharu_rust" ? "Rust" : "Python";
    return (
      <div
        className="flex items-center gap-1.5 rounded-lg border border-border bg-bg-tertiary/40 px-2 py-1 text-[10px] text-text-muted"
        title={`Renderer: ${backend}`}
      >
        <Check size={10} />
        Renderer {label}
      </div>
    );
  }

  if (status === "idle") return null;

  if (status === "rendering") {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-brand/20 bg-brand/8 px-2 py-1 text-[10px] text-brand">
        <Loader2 size={10} className="animate-spin" />
        Renderizando fiel…
      </div>
    );
  }

  if (status === "updated" && showUpdated) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-status-success/25 bg-status-success/10 px-2 py-1 text-[10px] text-status-success">
        <Check size={10} />
        Fiel atualizado
      </div>
    );
  }

  if (status === "stale") {
    return (
      <button
        onClick={() => void forceFidelityRender()}
        className="flex items-center gap-1.5 rounded-lg border border-status-warning/25 bg-status-warning/10 px-2 py-1 text-[10px] text-status-warning hover:bg-status-warning/15 transition-smooth"
        title="Render desatualizado — clique para forçar"
      >
        <RefreshCw size={10} />
        Render desatualizado
      </button>
    );
  }

  if (status === "error") {
    return (
      <button
        onClick={() => void forceFidelityRender()}
        className="flex items-center gap-1.5 rounded-lg border border-status-error/30 bg-status-error/10 px-2 py-1 text-[10px] text-status-error hover:bg-status-error/15 transition-smooth"
        title={renderError ?? "Erro no render fiel — clique para tentar novamente"}
      >
        <AlertCircle size={10} />
        Erro no render fiel
      </button>
    );
  }

  return null;
}

function summarizeSfxReview(layers: TextEntry[]) {
  const flagged = layers.filter((layer) => {
    if (!isSfxLayer(layer)) return false;
    const sfx = layer.sfx ?? {};
    const flags = new Set([...(layer.qa_flags ?? []), ...(sfx.qa_flags ?? [])]);
    return (
      sfx.review_required === true
      || sfx.inpaint_allowed === false
      || layer.route_action === "review_required"
      || [...flags].some((flag) => SFX_REVIEW_FLAGS.has(flag))
    );
  });
  const flagList = flagged.flatMap((layer) => [...(layer.qa_flags ?? []), ...(layer.sfx?.qa_flags ?? [])]);
  const uniqueFlags = [...new Set(flagList)].slice(0, 6);
  const title = uniqueFlags.length > 0
    ? `SFX precisa de revisão: ${uniqueFlags.join(", ")}`
    : "SFX precisa de revisão manual";
  return { reviewCount: flagged.length, title };
}

function isSfxLayer(layer: TextEntry) {
  return (
    layer.tipo === "sfx"
    || layer.content_class === "sfx"
    || layer.route_action === "translate_sfx_inpaint_render"
    || layer.route_action === "review_required"
    || Boolean(layer.sfx)
  );
}
