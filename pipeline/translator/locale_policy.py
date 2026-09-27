"""Deterministic, non-rewriting target-locale validation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re


_SOURCE_NUMBER_RE = re.compile(
    r"(?<![\w.,])(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s*(?P<magnitude>thousand|million|billion|trillion))?",
    re.IGNORECASE,
)
_TARGET_NUMBER_RE = re.compile(
    r"(?<![\w.,])(?P<number>\d+(?:\.\d{3})*(?:,\d+)?)"
    r"(?:\s*(?P<magnitude>trilh(?:ão|ões)|bilh(?:ão|ões)|milh(?:ão|ões)|mil))?",
    re.IGNORECASE,
)
_SOURCE_MAGNITUDES = {
    None: Decimal(1),
    "thousand": Decimal(1_000),
    "million": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
}
_TARGET_MAGNITUDES = {
    None: Decimal(1),
    "mil": Decimal(1_000),
    "milhão": Decimal(1_000_000),
    "milhões": Decimal(1_000_000),
    "bilhão": Decimal(1_000_000_000),
    "bilhões": Decimal(1_000_000_000),
    "trilhão": Decimal(1_000_000_000_000),
    "trilhões": Decimal(1_000_000_000_000),
}
_PT_PT_LONG_SCALE_RE = re.compile(r"\bmil\s+milh(?:ão|ões)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LocaleValidationIssue:
    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True)
class LocaleValidationResult:
    target_locale: str
    target_text: str
    status: str
    issues: tuple[LocaleValidationIssue, ...]
    numeric_equivalent: bool | None

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"

    @property
    def numbers_equivalent_for_binding(self) -> bool:
        """Stricter numeric verdict used before an owner translation is bound."""

        return self.numeric_equivalent is not False and not any(
            issue.code in {
                "numeric_magnitude_mismatch",
                "numeric_equivalence_ambiguous",
            }
            for issue in self.issues
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_locale": self.target_locale,
            "target_text": self.target_text,
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
            "numeric_equivalent": self.numeric_equivalent,
        }


def canonical_target_locale(value: str) -> str:
    normalized = str(value or "").strip().replace("_", "-")
    if normalized.casefold() in {"pt", "pt-br"}:
        return "pt-BR"
    return normalized


def _numeric_values(
    text: str,
    *,
    target: bool,
) -> tuple[Decimal, ...]:
    pattern = _TARGET_NUMBER_RE if target else _SOURCE_NUMBER_RE
    magnitudes = _TARGET_MAGNITUDES if target else _SOURCE_MAGNITUDES
    values: list[Decimal] = []
    for match in pattern.finditer(text):
        raw_number = match.group("number")
        normalized_number = (
            raw_number.replace(".", "").replace(",", ".")
            if target
            else raw_number.replace(",", "")
        )
        try:
            number = Decimal(normalized_number)
        except InvalidOperation:
            continue
        magnitude = match.group("magnitude")
        magnitude_key = magnitude.casefold() if magnitude else None
        values.append(number * magnitudes[magnitude_key])
    return tuple(values)


def validate_target_locale(
    *,
    source_text: str,
    target_text: str,
    target_locale: str,
) -> LocaleValidationResult:
    """Validate locale and numeric meaning without changing the target text."""

    locale = canonical_target_locale(target_locale)
    source = str(source_text or "")
    target = str(target_text or "")
    if locale != "pt-BR":
        return LocaleValidationResult(locale, target, "ok", (), None)

    issues: list[LocaleValidationIssue] = []
    if _PT_PT_LONG_SCALE_RE.search(target):
        issues.append(
            LocaleValidationIssue(
                code="pt_pt_long_scale_lexeme",
                severity="critical",
                message="A tradução usa escala longa de PT-PT em um alvo PT-BR.",
            )
        )

    source_values = _numeric_values(source, target=False)
    target_values = _numeric_values(target, target=True)
    numeric_equivalent: bool | None = None
    if source_values or target_values:
        if len(source_values) == len(target_values):
            numeric_equivalent = source_values == target_values
            if not numeric_equivalent:
                issues.append(
                    LocaleValidationIssue(
                        code="numeric_magnitude_mismatch",
                        severity="critical",
                        message="A tradução altera deterministicamente o valor numérico.",
                    )
                )
        else:
            issues.append(
                LocaleValidationIssue(
                    code="numeric_equivalence_ambiguous",
                    severity="review",
                    message="Não foi possível parear todos os valores numéricos.",
                )
            )

    status = "ok"
    if any(issue.severity == "critical" for issue in issues):
        status = "blocked"
    elif issues:
        status = "review"
    return LocaleValidationResult(
        target_locale=locale,
        target_text=target,
        status=status,
        issues=tuple(issues),
        numeric_equivalent=numeric_equivalent,
    )


__all__ = [
    "LocaleValidationIssue",
    "LocaleValidationResult",
    "canonical_target_locale",
    "validate_target_locale",
]
