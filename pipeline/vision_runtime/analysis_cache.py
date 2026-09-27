"""Dependency-scoped cache identity and publication for visual analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping
from uuid import uuid4


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CAPABILITY_PROVIDERS = {
    "detection": ("detection",),
    "ocr": ("detection", "ocr"),
    "structure": ("detection", "ocr", "structure"),
    "masks": ("detection", "masks"),
}
_DETERMINISTIC_FAILURE_REASONS = {
    "capability_unavailable",
    "illegible_or_unrecognized",
    "no_observations",
    "unsupported_language",
}


def _canonical_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    digest = str(value or "")
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return digest


@dataclass(frozen=True)
class AnalysisCacheIdentity:
    """Only dependencies that may change visual analysis belong in this identity."""

    payload: dict[str, Any]
    sha256: str

    @classmethod
    def build(cls, request: Mapping[str, Any]) -> "AnalysisCacheIdentity":
        providers = request.get("providers")
        if not isinstance(providers, Mapping):
            raise ValueError("providers are required")
        missing = set(_CAPABILITY_PROVIDERS) - set(providers)
        if missing:
            raise ValueError("provider identities are incomplete")
        source_sha256 = _require_sha256(request.get("source_sha256"), "source_sha256")
        neighbors = [
            _require_sha256(value, "authenticated_neighbor_sha256")
            for value in request.get("authenticated_neighbor_sha256s", ())
        ]
        for provider_name, provider in providers.items():
            if not isinstance(provider, Mapping):
                raise ValueError(f"provider {provider_name} identity is invalid")
            if any(not str(provider.get(field) or "") for field in ("family", "name", "model", "version")):
                raise ValueError(f"provider {provider_name} identity is incomplete")
        region = request.get("region")
        if not isinstance(region, Mapping) or len(tuple(region.get("bbox") or ())) != 4:
            raise ValueError("region bbox is required")
        payload = {
            "source_sha256": source_sha256,
            "authenticated_neighbor_sha256s": neighbors,
            "region": _canonical_copy(region),
            "coordinate_space": str(request.get("coordinate_space") or ""),
            "transform_sha256": _require_sha256(
                request.get("transform_sha256"), "transform_sha256"
            ),
            "source_language": str(request.get("source_language") or ""),
            "analysis_config_sha256": _require_sha256(
                request.get("analysis_config_sha256"), "analysis_config_sha256"
            ),
            "providers": _canonical_copy(providers),
        }
        return cls(payload=payload, sha256=_sha256(payload))

    def capability_sha256(self, capability: str) -> str:
        provider_names = _CAPABILITY_PROVIDERS.get(capability)
        if provider_names is None:
            raise ValueError(f"unsupported analysis capability: {capability}")
        scoped = {
            key: _canonical_copy(value)
            for key, value in self.payload.items()
            if key != "providers"
        }
        if capability in {"detection", "masks"}:
            scoped.pop("source_language", None)
        scoped["capability"] = capability
        scoped["providers"] = {
            name: _canonical_copy(self.payload["providers"][name])
            for name in provider_names
        }
        return _sha256(scoped)

    def revision_sha256(self, selected_observation_id: str) -> str:
        selected = str(selected_observation_id or "").strip()
        if not selected:
            raise ValueError("selected_observation_id is required")
        return _sha256({
            "analysis_identity_sha256": self.sha256,
            "selected_observation_id": selected,
        })


@dataclass(frozen=True)
class AnalysisCacheResult:
    payload: dict[str, Any]
    cache_hit: bool
    identity_sha256: str


class AnalysisArtifactCache:
    """Publish complete records atomically and single-flight concurrent consumers."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}

    def _path(self, identity: AnalysisCacheIdentity) -> Path:
        return self.root / identity.sha256[:2] / f"{identity.sha256}.json"

    def lookup(self, identity: AnalysisCacheIdentity) -> AnalysisCacheResult | None:
        path = self._path(identity)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        payload = envelope.get("payload")
        if (
            envelope.get("identity_sha256") != identity.sha256
            or not isinstance(payload, Mapping)
            or payload.get("status") != "complete"
            or envelope.get("payload_sha256") != _sha256(payload)
        ):
            return None
        return AnalysisCacheResult(
            payload=_canonical_copy(payload),
            cache_hit=True,
            identity_sha256=identity.sha256,
        )

    def _publish(self, identity: AnalysisCacheIdentity, payload: Mapping[str, Any]) -> None:
        if payload.get("status") != "complete":
            raise ValueError("only complete visual analysis may be published")
        path = self._path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        copied = _canonical_copy(payload)
        envelope = {
            "identity": identity.payload,
            "identity_sha256": identity.sha256,
            "payload": copied,
            "payload_sha256": _sha256(copied),
        }
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_bytes(_canonical_bytes(envelope))
        os.replace(temporary, path)

    def get_or_compute(
        self,
        identity: AnalysisCacheIdentity,
        producer: Callable[[], Mapping[str, Any]],
    ) -> AnalysisCacheResult:
        cached = self.lookup(identity)
        if cached is not None:
            return cached
        with self._lock:
            event = self._inflight.get(identity.sha256)
            owner = event is None
            if owner:
                event = threading.Event()
                self._inflight[identity.sha256] = event
        assert event is not None
        if not owner:
            event.wait()
            cached = self.lookup(identity)
            if cached is None:
                raise RuntimeError("visual analysis producer did not publish a complete record")
            return cached
        try:
            payload = _canonical_copy(producer())
            self._publish(identity, payload)
            return AnalysisCacheResult(
                payload=payload,
                cache_hit=False,
                identity_sha256=identity.sha256,
            )
        finally:
            with self._lock:
                self._inflight.pop(identity.sha256, None)
                event.set()


class DeterministicFailureCache:
    """Persist stable capability failures without masking transient outages."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, identity: AnalysisCacheIdentity, capability: str) -> Path:
        digest = identity.capability_sha256(capability)
        return self.root / str(capability) / digest[:2] / f"{digest}.json"

    def lookup(
        self, identity: AnalysisCacheIdentity, *, capability: str
    ) -> dict[str, Any] | None:
        path = self._path(identity, capability)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        receipt = envelope.get("receipt")
        if not isinstance(receipt, Mapping):
            return None
        expected_capability_sha256 = identity.capability_sha256(capability)
        if (
            receipt.get("status") != "deterministic_failure"
            or receipt.get("capability") != capability
            or receipt.get("capability_sha256") != expected_capability_sha256
            or receipt.get("reason_code") not in _DETERMINISTIC_FAILURE_REASONS
            or envelope.get("receipt_sha256") != _sha256(receipt)
        ):
            return None
        return _canonical_copy(receipt)

    def record(
        self,
        identity: AnalysisCacheIdentity,
        *,
        capability: str,
        reason_code: str,
        evidence_sha256: str,
    ) -> dict[str, Any]:
        reason = str(reason_code or "")
        if reason not in _DETERMINISTIC_FAILURE_REASONS:
            raise ValueError(f"failure reason is not deterministic: {reason}")
        evidence = _require_sha256(evidence_sha256, "evidence_sha256")
        capability_sha256 = identity.capability_sha256(capability)
        existing = self.lookup(identity, capability=capability)
        if existing is not None:
            return existing
        receipt = {
            "status": "deterministic_failure",
            "cache_hit": True,
            "analysis_identity_sha256": identity.sha256,
            "capability": str(capability),
            "capability_sha256": capability_sha256,
            "reason_code": reason,
            "evidence_sha256": evidence,
        }
        envelope = {
            "receipt": receipt,
            "receipt_sha256": _sha256(receipt),
        }
        path = self._path(identity, capability)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_bytes(_canonical_bytes(envelope))
        os.replace(temporary, path)
        return _canonical_copy(receipt)
