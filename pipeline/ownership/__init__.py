"""Page-global text ownership contracts for the automatic pipeline."""

from .model import (
    ComponentDisposition,
    OwnerGraph,
    OwnerGraphValidationError,
    OwnerProjection,
    OwnerViolation,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)

__all__ = [
    "ComponentDisposition",
    "OwnerGraph",
    "OwnerGraphValidationError",
    "OwnerProjection",
    "OwnerViolation",
    "SourceTextComponent",
    "TextObservation",
    "TextOwner",
]
