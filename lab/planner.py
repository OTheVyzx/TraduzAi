"""Improvement Planner — converte findings dos critics em Proposals executaveis.

Fluxo:
    findings: list[Finding] + aggregate_benchmark: dict → list[Proposal]

Agrupa findings por (issue_type, suggested_file), prioriza e gera Proposal
estruturado com motivation, target_file, change_kind e metric_gap esperado.

- `change_kind="regex_add"` ou `"threshold_tune"` → patch pode ser gerado
  localmente (sem LLM) a partir de `local_patch_hint`.
- `change_kind="logic_fix"` → marcado com `needs_coder=True`, escalável para
  um Coder (Ollama ou Claude Code) na camada seguinte.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable

from lab.critics.base import Finding


SEVERITY_WEIGHTS = {"info": 1.0, "warning": 2.5, "error": 4.5}

# Mapa issue_type → change_kind + métrica ligada
ISSUE_PROFILES: dict[str, dict] = {
    # OCR
    "watermark_leaked": {
        "change_kind": "regex_add",
        "linked_metric": "visual_cleanup",
    },
    "low_confidence": {
        "change_kind": "threshold_tune",
        "linked_metric": "readability",
    },
    "ocr_artifact_repeated_digits": {
        "change_kind": "regex_add",
        "linked_metric": "readability",
    },
    "duplicated_ocr_box": {
        "change_kind": "logic_fix",
        "linked_metric": "layout_occupancy",
    },
    # Translation
    "untranslated": {
        "change_kind": "logic_fix",
        "linked_metric": "textual_similarity",
    },
    "encoding_artifact": {
        "change_kind": "logic_fix",
        "linked_metric": "textual_similarity",
    },
    "length_ratio_outlier": {
        "change_kind": "threshold_tune",
        "linked_metric": "textual_similarity",
    },
    "term_inconsistency": {
        "change_kind": "logic_fix",
        "linked_metric": "term_consistency",
    },
    # Typeset
    "occupancy_too_low": {
        "change_kind": "threshold_tune",
        "linked_metric": "layout_occupancy",
    },
    "occupancy_too_high": {
        "change_kind": "threshold_tune",
        "linked_metric": "layout_occupancy",
    },
    "font_too_small_for_balloon": {
        "change_kind": "threshold_tune",
        "linked_metric": "readability",
    },
    "text_overflow_bbox": {
        "change_kind": "logic_fix",
        "linked_metric": "layout_occupancy",
    },
    "connected_lobe_imbalance": {
        "change_kind": "logic_fix",
        "linked_metric": "readability",
    },
    # Inpaint
    "residual_text_in_balloon": {
        "change_kind": "logic_fix",
        "linked_metric": "visual_cleanup",
    },
    # Fallback
    "critic_crashed": {
        "change_kind": "logic_fix",
        "linked_metric": "readability",
    },
}

DEFAULT_BENCHMARK_TARGET = 80.0


@dataclass
class Proposal:
    proposal_id: str
    title: str
    motivation: str
    target_file: str
    target_anchor: str
    change_kind: str  # "regex_add" | "threshold_tune" | "logic_fix"
    needs_coder: bool = False
    local_patch_hint: dict = field(default_factory=dict)
    expected_metric_gain: dict = field(default_factory=dict)
    findings_sample: list[dict] = field(default_factory=list)
    touched_domains: list[str] = field(default_factory=list)
    priority_score: float = 0.0
    issue_type: str = ""
    severity: str = "warning"

    def to_dict(self) -> dict:
        return asdict(self)


def _infer_touched_domains(target_file: str) -> list[str]:
    if not target_file:
        return ["pipeline/**"]
    if target_file.startswith("pipeline/"):
        return ["pipeline/**"]
    if target_file.startswith("src-tauri/"):
        return ["src-tauri/**"]
    if target_file.startswith("src/"):
        return ["src/**"]
    if target_file.startswith("lab/"):
        return ["pipeline/**"]  # lab vive junto do pipeline p/ routing de reviewer
    return ["pipeline/**"]


def _severity_weight(findings: list[Finding]) -> float:
    if not findings:
        return 0.0
    return max(SEVERITY_WEIGHTS.get(f.severity, 1.0) for f in findings)


def _metric_gap(aggregate_benchmark: dict, metric_name: str) -> float:
    metrics = aggregate_benchmark.get("metrics", {}) or {}
    value = float(metrics.get(metric_name, DEFAULT_BENCHMARK_TARGET) or 0.0)
    return max(0.0, DEFAULT_BENCHMARK_TARGET - value)


def _deepest_severity(findings: list[Finding]) -> str:
    for level in ("error", "warning", "info"):
        if any(f.severity == level for f in findings):
            return level
    return "warning"


def _humanize_issue_type(issue_type: str) -> str:
    mapping = {
        "watermark_leaked": "Endurecer filtro de watermark",
        "low_confidence": "Elevar threshold de confiança OCR",
        "ocr_artifact_repeated_digits": "Adicionar regex contra artefatos do PaddleOCR",
        "duplicated_ocr_box": "Corrigir merge de bboxes duplicadas",
        "untranslated": "Garantir fallback Google/Ollama em traduções idênticas",
        "encoding_artifact": "Corrigir decodificação UTF-8 no translator",
        "length_ratio_outlier": "Ajustar tolerância de razão EN→PT-BR",
        "term_inconsistency": "Introduzir glossário persistente de termos",
        "occupancy_too_low": "Elevar font_search_floor para preencher balões",
        "occupancy_too_high": "Reduzir font máximo ou aumentar margem no balão",
        "font_too_small_for_balloon": "Proteger fonte mínima em balões grandes",
        "text_overflow_bbox": "Reforçar wrap / split no renderer",
        "connected_lobe_imbalance": "Rebalancear split semântico entre lobos",
        "residual_text_in_balloon": "Expandir máscara de inpaint para cobrir resíduo",
        "critic_crashed": "Investigar crash de critic no Lab",
    }
    return mapping.get(issue_type, issue_type.replace("_", " ").capitalize())


def _build_local_patch_hint(issue_type: str, findings: list[Finding]) -> dict:
    """Para change_kind = regex_add ou threshold_tune, gera hint machine-readable."""
    hint: dict = {"kind": ISSUE_PROFILES.get(issue_type, {}).get("change_kind", "logic_fix")}

    if issue_type == "watermark_leaked":
        tokens: set[str] = set()
        for finding in findings:
            token = str(finding.evidence.get("matched_token", "")).strip().lower()
            if token:
                tokens.add(token)
        hint["new_watermark_tokens"] = sorted(tokens)
        return hint

    if issue_type == "ocr_artifact_repeated_digits":
        hint["regex_pattern"] = r"1{5,}|0{5,}|-{5,}"
        return hint

    if issue_type == "low_confidence":
        confidences = [
            float(f.evidence.get("confidence", 0.0)) for f in findings
            if "confidence" in f.evidence
        ]
        if confidences:
            hint["median_observed"] = round(sorted(confidences)[len(confidences) // 2], 3)
        hint["suggestion"] = "raise_threshold_to_0.65"
        return hint

    if issue_type in {"occupancy_too_low", "occupancy_too_high"}:
        occupancies = [
            float(f.evidence.get("occupancy", 0.0)) for f in findings
            if "occupancy" in f.evidence
        ]
        if occupancies:
            hint["mean_observed"] = round(sum(occupancies) / len(occupancies), 3)
        hint["direction"] = "increase_floor" if issue_type == "occupancy_too_low" else "decrease_cap"
        return hint

    if issue_type == "font_too_small_for_balloon":
        hint["min_font_size"] = 14
        return hint

    if issue_type == "length_ratio_outlier":
        ratios = [float(f.evidence.get("ratio", 0.0)) for f in findings if "ratio" in f.evidence]
        if ratios:
            hint["min_ratio"] = min(ratios)
            hint["max_ratio"] = max(ratios)
        return hint

    return hint


def build_proposals(
    findings: Iterable[Finding],
    aggregate_benchmark: dict,
    run_id: str,
) -> list[Proposal]:
    """Agrupa findings e gera propostas priorizadas."""
    findings_list = list(findings)
    if not findings_list:
        return []

    grouped: dict[tuple[str, str], list[Finding]] = defaultdict(list)
    for finding in findings_list:
        key = (finding.issue_type, finding.suggested_file or "pipeline/**")
        grouped[key].append(finding)

    proposals: list[Proposal] = []
    for idx, ((issue_type, target_file), bucket) in enumerate(grouped.items()):
        profile = ISSUE_PROFILES.get(issue_type, {"change_kind": "logic_fix", "linked_metric": "readability"})
        change_kind: str = profile["change_kind"]
        linked_metric: str = profile["linked_metric"]

        severity_weight = _severity_weight(bucket)
        metric_gap = _metric_gap(aggregate_benchmark, linked_metric)
        priority = severity_weight * len(bucket) * (1.0 + metric_gap / 20.0)

        anchor = ""
        for finding in bucket:
            if finding.suggested_anchor:
                anchor = finding.suggested_anchor
                break

        sample = [f.to_dict() for f in bucket[:5]]
        needs_coder = change_kind == "logic_fix"
        motivation_lines = [
            f"{len(bucket)} ocorrência(s) de `{issue_type}` detectada(s) em {_count_pages(bucket)} página(s).",
            f"Métrica ligada: {linked_metric} (gap atual vs {DEFAULT_BENCHMARK_TARGET:.0f}: {metric_gap:.1f} pts).",
            f"Severidade dominante: {_deepest_severity(bucket)}. Alvo: `{target_file}`.",
        ]
        motivation = " ".join(motivation_lines)

        proposal = Proposal(
            proposal_id=f"proposal-{run_id[:8]}-{idx:02d}",
            title=_humanize_issue_type(issue_type),
            motivation=motivation,
            target_file=target_file,
            target_anchor=anchor,
            change_kind=change_kind,
            needs_coder=needs_coder,
            local_patch_hint=_build_local_patch_hint(issue_type, bucket),
            expected_metric_gain={linked_metric: round(min(metric_gap, 6.0), 1)},
            findings_sample=sample,
            touched_domains=_infer_touched_domains(target_file),
            priority_score=round(priority, 2),
            issue_type=issue_type,
            severity=_deepest_severity(bucket),
        )
        proposals.append(proposal)

    proposals.sort(key=lambda p: p.priority_score, reverse=True)
    return proposals


def _count_pages(findings: Iterable[Finding]) -> int:
    return len({f.page_index for f in findings if f.page_index >= 0})


def proposal_to_lab_payload(proposal: Proposal, run_id: str, git_available: bool) -> dict:
    """Converte Proposal para o shape que o Rust/LabSnapshot espera.

    Mantém compatibilidade com `LabProposal` existente e adiciona campos opcionais
    (motivation, target_file, change_kind, expected_metric_gain) via chaves extras
    — o Rust aceita via `#[serde(default)]` se atualizarmos a struct depois.
    """
    return {
        "proposal_id": proposal.proposal_id,
        "batch_id": f"batch-{run_id[:8]}",
        "title": proposal.title,
        "summary": proposal.motivation,
        "author": "improvement_planner",
        "risk": "alto" if proposal.severity == "error" else "medio",
        "touched_domains": proposal.touched_domains,
        "required_reviewers": [],  # preenchido pelo caller via required_reviewers_for
        # Transforma Finding.to_dict() → shape LabReviewFinding (title/body/severity/file_path).
        "review_findings": [
            {
                "title": str(f.get("issue_type", "finding")).replace("_", " ").capitalize(),
                "body": str(f.get("suggested_fix", f.get("body", ""))),
                "severity": str(f.get("severity", "warning")),
                "file_path": str(f.get("suggested_file", f.get("file_path", ""))),
            }
            for f in proposal.findings_sample[:3]
        ],
        "integration_verdict": "",
        "benchmark_batch_id": f"batch-{run_id[:8]}",
        "proposal_status": "reviewing",
        "pr_status": "awaiting_review",
        "git_available": git_available,
        # Campos novos (backwards-compatible)
        "motivation": proposal.motivation,
        "target_file": proposal.target_file,
        "target_anchor": proposal.target_anchor,
        "change_kind": proposal.change_kind,
        "needs_coder": proposal.needs_coder,
        "expected_metric_gain": proposal.expected_metric_gain,
        "priority_score": proposal.priority_score,
        "issue_type": proposal.issue_type,
        "local_patch_hint": proposal.local_patch_hint,
    }


__all__ = [
    "ISSUE_PROFILES",
    "Proposal",
    "build_proposals",
    "proposal_to_lab_payload",
]
