"""Tipos compartilhados pelos critics do Lab.

Finding eh o payload estruturado que cada critic emite. O Planner (camada
seguinte) consome lista de Findings, agrupa por tipo/arquivo e gera Proposal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


SEVERITIES = ("info", "warning", "error")


@dataclass
class Finding:
    """Observacao estruturada emitida por um critic."""

    critic_id: str
    chapter_number: int
    page_index: int
    issue_type: str
    severity: str
    evidence: dict = field(default_factory=dict)
    suggested_fix: str = ""
    suggested_file: str = ""
    suggested_anchor: str = ""
    bbox: list[int] | None = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            self.severity = "warning"

    def to_dict(self) -> dict:
        return asdict(self)


class Critic(Protocol):
    """Contrato minimo de um critic."""

    critic_id: str

    def analyze(self, chapter_artifact: dict) -> list[Finding]:
        """Recebe o artefato do capitulo e retorna findings.

        chapter_artifact esperado (ver lab/runner.build_artifact_record):
            {
                "chapter_number": int,
                "source_path": str,         # .cbz EN
                "reference_path": str,      # .cbz PT-BR
                "output_dir": str,          # pasta com project.json + translated/
                "project_json": str,        # caminho direto para project.json
                "benchmark": {              # BenchmarkResult.to_dict()
                    "score_before", "score_after", "green", "summary",
                    "metrics": {...}
                },
            }
        """
        ...


def load_project_json_from_artifact(chapter_artifact: dict) -> dict:
    """Helper: abre project.json do artefato ou retorna {} se falhar."""
    import json
    from pathlib import Path

    project_path = chapter_artifact.get("project_json")
    if not project_path:
        return {}
    path = Path(project_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


__all__ = [
    "Critic",
    "Finding",
    "SEVERITIES",
    "load_project_json_from_artifact",
]
