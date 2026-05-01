"""Protocol base para coders do Lab + gerador de patch local para hints conhecidos.

`build_local_patch_from_hint` lida com 80% dos casos sem chamar LLM:
- `regex_add` → injeta regex na variavel de registry apropriada
- `threshold_tune` → abre arquivo alvo, procura a constante e sugere novo valor

Para `logic_fix` (change_kind restante), a proposta segue para o coder LLM.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class PatchProposal:
    """Resultado de um coder. NUNCA aplicado automaticamente."""

    proposal_id: str
    patch_unified_diff: str
    files_affected: list[str] = field(default_factory=list)
    rationale: str = ""
    author: str = "unknown_coder"  # "ollama_coder" | "claude_code_coder" | "local_patch_hint"
    confidence: float = 0.0  # [0, 1]
    model_used: str = ""
    generated_at_iso: str = ""
    dry_run: bool = True
    error: str = ""  # preenchido se a geracao falhou

    def to_dict(self) -> dict:
        return asdict(self)


class Coder(Protocol):
    """Contrato minimo de um coder.

    Implementacoes concretas:
    - OllamaCoder → invoca modelo local via API HTTP do Ollama
    - ClaudeCodeCoder → invoca `claude -p` CLI
    """

    coder_id: str

    def propose_patch(self, proposal: dict, repo_root: Path) -> PatchProposal:
        """Recebe proposal (shape do `proposal_to_lab_payload`) e devolve patch."""
        ...


# ---------------------------------------------------------------------------
# Local patch builder — resolve regex_add/threshold_tune sem LLM
# ---------------------------------------------------------------------------


def build_local_patch_from_hint(
    proposal: dict, repo_root: Path
) -> PatchProposal | None:
    """Gera PatchProposal deterministico a partir do `local_patch_hint`.

    Retorna None se o caso nao puder ser resolvido sem LLM — nesse caso
    o caller deve fallback para um coder com LLM.
    """
    hint = proposal.get("local_patch_hint") or {}
    kind = str(hint.get("kind", "")).strip()
    target_file_rel = str(proposal.get("target_file", "")).strip()
    proposal_id = str(proposal.get("proposal_id", "unknown"))

    if not target_file_rel:
        return None

    target_path = (repo_root / target_file_rel).resolve()
    if not target_path.exists():
        return None

    try:
        original_source = target_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if kind == "regex_add" and proposal.get("issue_type") == "watermark_leaked":
        return _patch_watermark_tokens(
            proposal_id=proposal_id,
            target_file_rel=target_file_rel,
            target_path=target_path,
            original_source=original_source,
            new_tokens=list(hint.get("new_watermark_tokens", [])),
        )

    if kind == "regex_add" and proposal.get("issue_type") == "ocr_artifact_repeated_digits":
        return _patch_repeated_digits(
            proposal_id=proposal_id,
            target_file_rel=target_file_rel,
            target_path=target_path,
            original_source=original_source,
            regex_pattern=str(hint.get("regex_pattern", r"1{5,}|0{5,}|-{5,}")),
        )

    if kind == "threshold_tune" and proposal.get("issue_type") == "low_confidence":
        return _patch_low_confidence(
            proposal_id=proposal_id,
            target_file_rel=target_file_rel,
            target_path=target_path,
            original_source=original_source,
        )

    if kind == "threshold_tune" and proposal.get("issue_type") == "font_too_small_for_balloon":
        min_font = int(hint.get("min_font_size", 14) or 14)
        return _patch_min_font_size(
            proposal_id=proposal_id,
            target_file_rel=target_file_rel,
            target_path=target_path,
            original_source=original_source,
            min_font=min_font,
        )

    # Demais threshold_tune / logic_fix → fica para LLM
    return None


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _make_unified_diff(target_file_rel: str, original: str, patched: str) -> str:
    """Gera um unified diff estilo `git diff` (tolerante a Windows)."""
    import difflib

    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=f"a/{target_file_rel}",
        tofile=f"b/{target_file_rel}",
        lineterm="",
    )
    return "".join(diff)


def _patch_watermark_tokens(
    *,
    proposal_id: str,
    target_file_rel: str,
    target_path: Path,
    original_source: str,
    new_tokens: list[str],
) -> PatchProposal | None:
    """Para `watermark_leaked`: injeta novos tokens no regex existente.

    Procura padrao `WATERMARK_RE = re.compile(r"(?i)\\b(...)\\b")` e adiciona
    os novos tokens ao grupo alternativo.
    """
    new_tokens = [t.strip().lower() for t in new_tokens if t and t.strip()]
    if not new_tokens:
        return None

    pattern = re.compile(
        r"WATERMARK_RE\s*=\s*re\.compile\(\s*r\"(?P<pref>\(\?i\))\\b\((?P<inner>[^)]*)\)\\b\"\s*\)",
        re.MULTILINE,
    )
    match = pattern.search(original_source)
    if not match:
        return None

    existing_raw = match.group("inner")
    existing = [tok.strip() for tok in existing_raw.split("|") if tok.strip()]
    deduped = sorted({*existing, *new_tokens})
    if set(deduped) == set(existing):
        return PatchProposal(
            proposal_id=proposal_id,
            patch_unified_diff="",
            files_affected=[target_file_rel],
            rationale="Todos os tokens de watermark ja estavam no regex — no-op.",
            author="local_patch_hint",
            confidence=1.0,
            dry_run=True,
        )

    new_regex = 'WATERMARK_RE = re.compile(r"(?i)\\b(' + "|".join(deduped) + ')\\b")'
    patched_source = original_source[: match.start()] + new_regex + original_source[match.end() :]
    diff = _make_unified_diff(target_file_rel, original_source, patched_source)

    return PatchProposal(
        proposal_id=proposal_id,
        patch_unified_diff=diff,
        files_affected=[target_file_rel],
        rationale=(
            "Injetei os tokens novos no WATERMARK_RE: "
            + ", ".join(sorted(set(new_tokens) - set(existing)))
        ),
        author="local_patch_hint",
        confidence=0.9,
        dry_run=True,
    )


def _patch_repeated_digits(
    *,
    proposal_id: str,
    target_file_rel: str,
    target_path: Path,
    original_source: str,
    regex_pattern: str,
) -> PatchProposal | None:
    """Para `ocr_artifact_repeated_digits`: garante regex de filtragem no arquivo."""
    if regex_pattern in original_source:
        return PatchProposal(
            proposal_id=proposal_id,
            patch_unified_diff="",
            files_affected=[target_file_rel],
            rationale="Regex de artefato repetido ja presente — no-op.",
            author="local_patch_hint",
            confidence=1.0,
        )

    insert_snippet = (
        "\n# Auto-injetado pelo Lab: filtra artefatos do PaddleOCR\n"
        'OCR_ARTIFACT_REPEATED_DIGITS_RE = re.compile(r"'
        + regex_pattern.replace("\\", "\\\\")
        + '")\n'
    )
    patched_source = original_source.rstrip() + insert_snippet
    diff = _make_unified_diff(target_file_rel, original_source, patched_source)
    return PatchProposal(
        proposal_id=proposal_id,
        patch_unified_diff=diff,
        files_affected=[target_file_rel],
        rationale=(
            f"Adicionei OCR_ARTIFACT_REPEATED_DIGITS_RE com padrao `{regex_pattern}`. "
            "Integracao manual: aplicar o regex no pipeline de pos-filtro do OCR."
        ),
        author="local_patch_hint",
        confidence=0.6,  # usuario precisa conectar o regex no fluxo
        dry_run=True,
    )


def _patch_low_confidence(
    *,
    proposal_id: str,
    target_file_rel: str,
    target_path: Path,
    original_source: str,
) -> PatchProposal | None:
    """Para `low_confidence`: aumenta LOW_CONFIDENCE_THRESHOLD se existir."""
    pattern = re.compile(
        r"(LOW_CONFIDENCE_THRESHOLD\s*=\s*)(0\.\d+)",
        re.MULTILINE,
    )
    match = pattern.search(original_source)
    if not match:
        return None

    current = float(match.group(2))
    new_value = min(0.75, max(current + 0.05, 0.65))
    if abs(new_value - current) < 1e-3:
        return PatchProposal(
            proposal_id=proposal_id,
            patch_unified_diff="",
            files_affected=[target_file_rel],
            rationale=f"LOW_CONFIDENCE_THRESHOLD ja esta em {current}. No-op.",
            author="local_patch_hint",
            confidence=1.0,
        )

    patched_source = (
        original_source[: match.start(2)]
        + f"{new_value:.2f}"
        + original_source[match.end(2) :]
    )
    diff = _make_unified_diff(target_file_rel, original_source, patched_source)
    return PatchProposal(
        proposal_id=proposal_id,
        patch_unified_diff=diff,
        files_affected=[target_file_rel],
        rationale=(
            f"Elevei LOW_CONFIDENCE_THRESHOLD de {current} para {new_value:.2f} "
            "para reduzir ruido do OCR em bordas."
        ),
        author="local_patch_hint",
        confidence=0.85,
        dry_run=True,
    )


def _patch_min_font_size(
    *,
    proposal_id: str,
    target_file_rel: str,
    target_path: Path,
    original_source: str,
    min_font: int,
) -> PatchProposal | None:
    """Para `font_too_small_for_balloon`: garante floor minimo no renderer."""
    pattern = re.compile(
        r"(MIN_FONT_SIZE\s*=\s*)(\d+(?:\.\d+)?)",
        re.MULTILINE,
    )
    match = pattern.search(original_source)
    if not match:
        return None

    current = float(match.group(2))
    new_value = max(current, float(min_font))
    if abs(new_value - current) < 1e-3:
        return PatchProposal(
            proposal_id=proposal_id,
            patch_unified_diff="",
            files_affected=[target_file_rel],
            rationale=f"MIN_FONT_SIZE ja esta em {current}. No-op.",
            author="local_patch_hint",
            confidence=1.0,
        )

    patched_source = (
        original_source[: match.start(2)]
        + f"{int(new_value) if new_value.is_integer() else new_value}"
        + original_source[match.end(2) :]
    )
    diff = _make_unified_diff(target_file_rel, original_source, patched_source)
    return PatchProposal(
        proposal_id=proposal_id,
        patch_unified_diff=diff,
        files_affected=[target_file_rel],
        rationale=(
            f"Elevei MIN_FONT_SIZE de {current} para {new_value} "
            "para evitar fonte ilegivel em baloes grandes."
        ),
        author="local_patch_hint",
        confidence=0.8,
        dry_run=True,
    )


__all__ = [
    "Coder",
    "PatchProposal",
    "build_local_patch_from_hint",
]
