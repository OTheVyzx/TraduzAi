"""Crash-safe chapter publication locks and transactions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Literal

from .hash_contract import canonical_json_bytes, canonical_json_sha256


class PublicationBusyError(RuntimeError):
    """Raised when another incompatible publication lock is already held."""


class PublicationRecoveryError(RuntimeError):
    """Raised when a publication journal cannot be restored safely."""


_REGISTRY_GUARD = threading.Lock()
_HELD_LOCKS: dict[str, list[str]] = {}


def _replace_with_transient_retry(
    source: str | Path,
    target: str | Path,
    *,
    attempts: int = 10,
) -> None:
    """Preserve atomic replace semantics across short Windows scanner locks."""

    if attempts < 1:
        raise ValueError("publication replace attempts must be positive")
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.25, 0.02 * (1.6**attempt)))


@dataclass
class PublicationLock:
    resolved_run_root: Path
    mode: Literal["shared_reader", "exclusive_writer"]
    _handle: Any = field(default=None, init=False, repr=False)
    _overlapped: Any = field(default=None, init=False, repr=False)
    _registry_key: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"shared_reader", "exclusive_writer"}:
            raise ValueError(f"unsupported publication lock mode: {self.mode}")
        self.resolved_run_root = Path(self.resolved_run_root).resolve()

    def __enter__(self) -> "PublicationLock":
        key = os.path.normcase(str(self.resolved_run_root))
        with _REGISTRY_GUARD:
            held = _HELD_LOCKS.get(key, [])
            incompatible = self.mode == "exclusive_writer" or "exclusive_writer" in held
            if held and incompatible:
                raise PublicationBusyError(f"publication root is already locked: {self.resolved_run_root}")
            _HELD_LOCKS.setdefault(key, []).append(self.mode)
        self._registry_key = key
        try:
            lock_path = self.resolved_run_root.parent / f".{self.resolved_run_root.name}.publication.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = lock_path.open("a+b")
            self._acquire_os_lock()
            return self
        except Exception:
            self._release_registry()
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            raise

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        try:
            self._release_os_lock()
        finally:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._release_registry()

    def _release_registry(self) -> None:
        if not self._registry_key:
            return
        with _REGISTRY_GUARD:
            held = _HELD_LOCKS.get(self._registry_key, [])
            if self.mode in held:
                held.remove(self.mode)
            if not held:
                _HELD_LOCKS.pop(self._registry_key, None)
        self._registry_key = ""

    def _acquire_os_lock(self) -> None:
        if os.name == "nt":
            self._windows_lock(acquire=True)
            return
        import fcntl

        operation = fcntl.LOCK_SH if self.mode == "shared_reader" else fcntl.LOCK_EX
        try:
            fcntl.flock(self._handle.fileno(), operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PublicationBusyError(
                f"publication root is locked by another process: {self.resolved_run_root}"
            ) from exc

    def _release_os_lock(self) -> None:
        if self._handle is None:
            return
        if os.name == "nt":
            self._windows_lock(acquire=False)
            return
        import fcntl

        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)

    def _windows_lock(self, *, acquire: bool) -> None:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class OVERLAPPED(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_void_p),
                ("InternalHigh", ctypes.c_void_p),
                ("Offset", wintypes.DWORD),
                ("OffsetHigh", wintypes.DWORD),
                ("hEvent", wintypes.HANDLE),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        os_handle = msvcrt.get_osfhandle(self._handle.fileno())
        if acquire:
            overlapped = OVERLAPPED()
            flags = 0x00000001
            if self.mode == "exclusive_writer":
                flags |= 0x00000002
            ok = kernel32.LockFileEx(
                wintypes.HANDLE(os_handle),
                wintypes.DWORD(flags),
                wintypes.DWORD(0),
                wintypes.DWORD(1),
                wintypes.DWORD(0),
                ctypes.byref(overlapped),
            )
            if not ok:
                error = ctypes.get_last_error()
                raise PublicationBusyError(
                    f"publication root is locked by another process ({error}): {self.resolved_run_root}"
                )
            self._overlapped = overlapped
            return
        overlapped = self._overlapped or OVERLAPPED()
        ok = kernel32.UnlockFileEx(
            wintypes.HANDLE(os_handle),
            wintypes.DWORD(0),
            wintypes.DWORD(1),
            wintypes.DWORD(0),
            ctypes.byref(overlapped),
        )
        self._overlapped = None
        if not ok:
            error = ctypes.get_last_error()
            raise PublicationRecoveryError(f"failed to release publication lock: {error}")


@dataclass
class PublicationTransaction:
    run_root: Path
    transaction_id: str
    staging_root: Path
    backup_root: Path
    journal_path: Path
    fault_phase: str | None = None
    _bundle_identity: tuple[str, str, str] | None = field(default=None, init=False, repr=False)

    @classmethod
    def for_bundle(cls, run_root: str | Path, bundle: Any) -> "PublicationTransaction":
        root = Path(run_root).resolve()
        bundle_sha = str(getattr(bundle, "bundle_sha256", "") or "")
        if len(bundle_sha) != 64:
            raise ValueError("publication bundle hash is invalid")
        transaction_id = f"publication-{bundle_sha[:24]}"
        transaction_base = root.parent / f".{root.name}.publication-transactions"
        return cls(
            run_root=root,
            transaction_id=transaction_id,
            staging_root=transaction_base / transaction_id / "staging",
            backup_root=transaction_base / transaction_id / "backup",
            journal_path=transaction_base / transaction_id / "publication.wal.json",
        )

    def stage(self, bundle: Any) -> None:
        source = Path(getattr(bundle, "runtime_staging_root", "")).resolve(strict=True)
        if not source.is_dir():
            raise ValueError("publication bundle staging root is missing")
        files = _safe_tree_files(source)
        if "publication_receipt.json" not in files:
            raise ValueError("publication bundle lacks its reader commit marker")
        if self.staging_root.exists() or self.journal_path.exists() or self.backup_root.exists():
            raise PublicationRecoveryError("publication transaction already has pending state")
        self.staging_root.mkdir(parents=True, exist_ok=False)
        for relative in files:
            source_path = source / Path(*relative.split("/"))
            target = self.staging_root / Path(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
        self._bundle_identity = (
            str(getattr(bundle, "run_id", "") or ""),
            str(getattr(bundle, "execution_id", "") or ""),
            str(getattr(bundle, "bundle_sha256", "") or ""),
        )

    def commit(self, bundle: Any) -> None:
        identity = (
            str(getattr(bundle, "run_id", "") or ""),
            str(getattr(bundle, "execution_id", "") or ""),
            str(getattr(bundle, "bundle_sha256", "") or ""),
        )
        if identity != self._bundle_identity or not self.staging_root.is_dir():
            raise ValueError("publication bundle differs from staged transaction")
        with PublicationLock(self.run_root, "exclusive_writer"):
            self.run_root.mkdir(parents=True, exist_ok=True)
            staged_files = set(_safe_tree_files(self.staging_root))
            current_files = set(_safe_tree_files(self.run_root))
            operations = []
            for relative in sorted(staged_files | current_files):
                if relative == "publication_receipt.json":
                    continue
                operations.append(_operation(relative, "replace" if relative in staged_files else "remove"))
            operations.append(_operation("publication_receipt.json", "replace"))
            journal = {
                "schema_version": 1,
                "transaction_id": self.transaction_id,
                "run_id": identity[0],
                "execution_id": identity[1],
                "bundle_sha256": identity[2],
                "run_root": str(self.run_root),
                "operations": operations,
                "status": "committing",
            }
            self.backup_root.mkdir(parents=True, exist_ok=False)
            self._write_journal(journal)
            try:
                for operation in operations:
                    relative = operation["relative_path"]
                    target = _contained_target(self.run_root, relative)
                    staged = _contained_target(self.staging_root, relative)
                    backup = _contained_target(self.backup_root, relative)
                    if target.exists():
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        operation["backup_intent"] = True
                        self._write_journal(journal)
                        self._fault_checkpoint(f"before_backup:{relative}")
                        _replace_with_transient_retry(target, backup)
                        operation["backup_done"] = True
                        self._write_journal(journal)
                        self._fault_checkpoint(f"after_backup:{relative}")
                    if operation["action"] == "replace":
                        if not staged.is_file():
                            raise PublicationRecoveryError(f"staged publication file is missing: {relative}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        operation["promote_intent"] = True
                        self._write_journal(journal)
                        self._fault_checkpoint(f"before_promote:{relative}")
                        _replace_with_transient_retry(staged, target)
                        operation["promote_done"] = True
                        self._write_journal(journal)
                        self._fault_checkpoint(f"after_promote:{relative}")
                    else:
                        operation["remove_done"] = True
                        self._write_journal(journal)
                        self._fault_checkpoint(f"after_remove:{relative}")
                journal["status"] = "committed"
                self._write_journal(journal)
                if not (self.run_root / "publication_receipt.json").is_file():
                    raise PublicationRecoveryError("publication receipt was not promoted")
            except Exception:
                self._rollback_operations(journal)
                raise
            self._discard_transaction_files()

    def rollback(self) -> None:
        if not self.journal_path.is_file():
            return
        journal = _read_verified_journal(self.journal_path, self.run_root)
        self._rollback_operations(journal)

    def _rollback_operations(self, journal: dict[str, Any]) -> None:
        failures = []
        for operation in reversed(journal.get("operations") or []):
            relative = operation["relative_path"]
            target = _contained_target(self.run_root, relative)
            backup = _contained_target(self.backup_root, relative)
            try:
                if operation.get("promote_done") and target.is_file():
                    target.unlink()
                if operation.get("backup_done") and backup.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _replace_with_transient_retry(backup, target)
            except OSError as exc:
                failures.append(f"{relative}: {exc}")
        if failures:
            raise PublicationRecoveryError("publication rollback failed: " + "; ".join(failures))
        self._discard_transaction_files()

    def _write_journal(self, journal: dict[str, Any]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(journal)
        payload["wal_sha256"] = canonical_json_sha256(payload)
        temporary = self.journal_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_transient_retry(temporary, self.journal_path)

    def _discard_transaction_files(self) -> None:
        if self.journal_path.exists():
            self.journal_path.unlink()
        for root in (self.staging_root, self.backup_root):
            if root.exists():
                shutil.rmtree(root)

    def _fault_checkpoint(self, phase: str) -> None:
        if self.fault_phase == phase:
            raise RuntimeError(f"fault injection: {phase}")


def recover_chapter_publication(run_root: str | Path) -> None:
    root = Path(run_root).resolve()
    transactions = root.parent / f".{root.name}.publication-transactions"
    if not transactions.is_dir():
        return
    with PublicationLock(root, "exclusive_writer"):
        for journal_path in sorted(transactions.glob("*/publication.wal.json")):
            transaction_root = journal_path.parent
            journal = _read_verified_journal(journal_path, root)
            transaction = PublicationTransaction(
                run_root=root,
                transaction_id=str(journal["transaction_id"]),
                staging_root=transaction_root / "staging",
                backup_root=transaction_root / "backup",
                journal_path=journal_path,
            )
            transaction._rollback_operations(journal)


@dataclass(frozen=True)
class ReopenedVerifiedPublication:
    verified_inputs: Any
    asset_manifest: Any
    export_manifest: Any
    receipt: Any


def reopen_verified_publication(run_root: str | Path) -> ReopenedVerifiedPublication:
    """Open a publication only after its complete control/hash chain verifies."""

    from .chapter_contract import (
        ChapterAssetManifest,
        ExportManifest,
        PublicationReceipt,
        VerifiedProjectInputs,
    )

    root = Path(run_root).resolve(strict=True)
    with PublicationLock(root, "shared_reader"):
        receipt_path = root / "publication_receipt.json"
        asset_path = root / "chapter_asset_manifest.json"
        export_path = root / "export_manifest.json"
        if not receipt_path.is_file() or not asset_path.is_file() or not export_path.is_file():
            raise PublicationRecoveryError("publication control chain is incomplete")
        verified_inputs = VerifiedProjectInputs.read_verified(root)
        asset_manifest = ChapterAssetManifest.build(
            root,
            run_id=verified_inputs.run_id,
            execution_id=verified_inputs.execution_id,
            replay_of_execution_id=verified_inputs.replay_of_execution_id,
            artifact_store_id=verified_inputs.artifact_store_id,
            generation_id=verified_inputs.generation_id,
        )
        if asset_manifest.canonical_json_bytes != asset_path.read_bytes():
            raise PublicationRecoveryError("chapter asset manifest differs from published tree")
        export_manifest = ExportManifest.build(verified_inputs, asset_manifest)
        if export_manifest.canonical_json_bytes != export_path.read_bytes():
            raise PublicationRecoveryError("export manifest differs from verified page authority")
        receipt = PublicationReceipt.build(verified_inputs, asset_manifest, export_manifest)
        if receipt.canonical_json_bytes != receipt_path.read_bytes():
            raise PublicationRecoveryError("publication receipt hash chain mismatch")
        translated_entries = {entry.relative_path: entry for entry in asset_manifest.entries}
        for source, page, export_page in zip(
            verified_inputs.source_manifest.pages,
            verified_inputs.pages,
            export_manifest.pages,
            strict=True,
        ):
            if (
                source.page_id != page.page_id
                or export_page.page_id != page.page_id
                or export_page.page_source_sha256 != source.page_source_sha256
                or export_page.page_result_sha256 != page.page_result_sha256
                or export_page.page_execution_evidence_sha256 != page.page_execution_evidence.sha256
                or export_page.final_pixel_sha256 != page.final_artifact.pixel_sha256
                or export_page.translated_path != page.final_artifact.relative_path
            ):
                raise PublicationRecoveryError("export page is not bound to verified page authority")
            asset = translated_entries.get(export_page.translated_path)
            if asset is None or asset.file_sha256 != page.final_artifact.file_sha256:
                raise PublicationRecoveryError("export page final asset is absent from asset manifest")
        return ReopenedVerifiedPublication(
            verified_inputs=verified_inputs,
            asset_manifest=asset_manifest,
            export_manifest=export_manifest,
            receipt=receipt,
        )


def _operation(relative_path: str, action: str) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "action": action,
        "backup_intent": False,
        "backup_done": False,
        "promote_intent": False,
        "promote_done": False,
        "remove_done": False,
    }


def _safe_tree_files(root: Path) -> tuple[str, ...]:
    resolved = root.resolve(strict=True)
    paths = []
    folded = set()
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise PublicationRecoveryError(f"publication tree contains symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix()
        key = relative.casefold()
        if key in folded:
            raise PublicationRecoveryError(f"publication tree contains casefold collision: {relative}")
        folded.add(key)
        paths.append(relative)
    return tuple(sorted(paths))


def _contained_target(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise PublicationRecoveryError(f"unsafe publication target: {relative!r}")
    resolved_root = root.resolve()
    target = resolved_root / Path(*relative.split("/"))
    try:
        target.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise PublicationRecoveryError(f"publication target escapes root: {relative}") from exc
    return target


def _read_verified_journal(path: Path, expected_root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationRecoveryError("publication WAL is unreadable") from exc
    supplied_hash = payload.pop("wal_sha256", None)
    if canonical_json_sha256(payload) != supplied_hash:
        raise PublicationRecoveryError("publication WAL hash mismatch")
    if Path(payload.get("run_root", "")).resolve() != expected_root.resolve():
        raise PublicationRecoveryError("publication WAL targets another root")
    for operation in payload.get("operations") or []:
        _contained_target(expected_root, str(operation.get("relative_path") or ""))
    return payload


__all__ = [
    "PublicationBusyError",
    "PublicationLock",
    "PublicationRecoveryError",
    "PublicationTransaction",
    "ReopenedVerifiedPublication",
    "recover_chapter_publication",
    "reopen_verified_publication",
]
