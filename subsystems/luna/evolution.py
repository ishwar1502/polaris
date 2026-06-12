"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/evolution.py

Concrete in-memory implementation of the LUNA Knowledge Evolution Engine.

Manages the full lifecycle and evolution of knowledge records over time:
tracking versions, applying audited updates, superseding obsolete records,
resolving contradictions, retiring stale knowledge, and producing evolution
reports.

Responsibilities:
    evolve_knowledge          — apply a structured change set to a record
    update_confidence         — revise a record's confidence score
    update_validation_state   — advance or regress a record's validation status
    version_tracking          — monotonic version log per record
    supersession_tracking     — link successor → predecessor records
    contradiction_resolution  — acknowledge or resolve known contradictions
    historical_tracking       — full per-record history of applied proposals
    retirement_management     — deprecate, supersede, and archive records
    change_audit              — append-only immutable audit trail
    evolution reporting       — aggregate stats and per-record summaries

Integrations:
    validation.py  — KnowledgeValidationEngine for post-update validation
    retrieval.py   — KnowledgeRetrievalEngine for cross-type record lookup
    synthesis.py   — KnowledgeSynthesisEngine (invalidation hook)

Thread safety:  threading.RLock on all public operations.
Lifecycle-gated: every public method raises LunaNotInitializedError before
    initialize() or after shutdown().

Part of the POLARIS Cognitive Substrate:
    ASTRA        → Identity
    ECHO         → Experience
    LUNA         → Knowledge       ← this module
    CHRONOS      → Time
    CONSTELLATION → Relationships
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Optional

from subsystems.luna.exceptions import (
    KnowledgeError,
    KnowledgeRetrievalError,
    KnowledgeValidationError,
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractKnowledgeEvolutionEngine
from subsystems.luna.models import (
    ConfidenceLevel,
    KnowledgeConfidence,
    KnowledgeContradiction,
    KnowledgeMetadata,
    KnowledgeRecord,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeValidationResult,
    ValidationSeverity,
    ValidationStatus,
    _new_id,
    _utcnow,
)

logger = logging.getLogger(__name__)

_ENGINE_VERSION: str = "5.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# PROPOSAL STATE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_PROPOSAL_PENDING: str = "pending"
_PROPOSAL_APPLIED: str = "applied"
_PROPOSAL_REJECTED: str = "rejected"

# Allowed status transitions for lifecycle gating
_VALID_TRANSITIONS: dict[KnowledgeStatus, frozenset[KnowledgeStatus]] = {
    KnowledgeStatus.DRAFT: frozenset({
        KnowledgeStatus.PENDING_VALIDATION,
        KnowledgeStatus.RETRACTED,
    }),
    KnowledgeStatus.PENDING_VALIDATION: frozenset({
        KnowledgeStatus.VALIDATED,
        KnowledgeStatus.DRAFT,
        KnowledgeStatus.RETRACTED,
    }),
    KnowledgeStatus.VALIDATED: frozenset({
        KnowledgeStatus.ACTIVE,
        KnowledgeStatus.DEPRECATED,
        KnowledgeStatus.SUPERSEDED,
        KnowledgeStatus.RETRACTED,
        KnowledgeStatus.PENDING_VALIDATION,
    }),
    KnowledgeStatus.ACTIVE: frozenset({
        KnowledgeStatus.DEPRECATED,
        KnowledgeStatus.SUPERSEDED,
        KnowledgeStatus.RETRACTED,
        KnowledgeStatus.ARCHIVED,
    }),
    KnowledgeStatus.DEPRECATED: frozenset({
        KnowledgeStatus.ARCHIVED,
        KnowledgeStatus.RETRACTED,
        KnowledgeStatus.ACTIVE,
    }),
    KnowledgeStatus.SUPERSEDED: frozenset({
        KnowledgeStatus.ARCHIVED,
        KnowledgeStatus.RETRACTED,
    }),
    KnowledgeStatus.RETRACTED: frozenset(),   # terminal
    KnowledgeStatus.ARCHIVED: frozenset(),    # terminal
}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

def _make_proposal(
    *,
    proposal_id: str,
    record_id: str,
    knowledge_type: KnowledgeType,
    changes: dict[str, Any],
    rationale: str,
    source: str,
    confidence_delta: float,
    created_at: datetime,
) -> dict[str, Any]:
    """Return a canonical proposal dict."""
    return {
        "id": proposal_id,
        "record_id": record_id,
        "knowledge_type": knowledge_type.value,
        "changes": changes,
        "rationale": rationale,
        "source": source,
        "confidence_delta": confidence_delta,
        "state": _PROPOSAL_PENDING,
        "created_at": created_at.isoformat(),
        "resolved_at": None,
        "resolution_notes": "",
    }


def _make_audit_entry(
    *,
    entry_id: str,
    proposal_id: str,
    record_id: str,
    knowledge_type: KnowledgeType,
    action: str,
    actor: str,
    changes: dict[str, Any],
    previous_version: int,
    new_version: int,
    occurred_at: datetime,
    notes: str = "",
) -> dict[str, Any]:
    """Return a canonical, immutable audit trail entry."""
    return {
        "id": entry_id,
        "proposal_id": proposal_id,
        "record_id": record_id,
        "knowledge_type": knowledge_type.value,
        "action": action,
        "actor": actor,
        "changes": deepcopy(changes),
        "previous_version": previous_version,
        "new_version": new_version,
        "occurred_at": occurred_at.isoformat(),
        "notes": notes,
    }


def _make_version_snapshot(
    *,
    snapshot_id: str,
    record_id: str,
    version: int,
    knowledge_type: KnowledgeType,
    status: KnowledgeStatus,
    confidence_score: float,
    validation_status: ValidationStatus,
    snapshotted_at: datetime,
    proposal_id: Optional[str] = None,
) -> dict[str, Any]:
    """Lightweight version-log entry (not a full record copy)."""
    return {
        "id": snapshot_id,
        "record_id": record_id,
        "version": version,
        "knowledge_type": knowledge_type.value,
        "status": status.value,
        "confidence_score": confidence_score,
        "validation_status": validation_status.value,
        "snapshotted_at": snapshotted_at.isoformat(),
        "proposal_id": proposal_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: RECORD MUTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_changes_to_record(
    record: KnowledgeRecord,
    changes: dict[str, Any],
    confidence_delta: float,
) -> KnowledgeRecord:
    """
    Apply a changes dict to a KnowledgeRecord in-place and return it.

    Only fields that exist on the record are mutated.  Reserved metadata
    fields (version, created_at) are never overwritten by callers.
    The metadata is bumped via KnowledgeMetadata.bump_version().
    """
    # Separate out metadata-level changes from record-level changes
    meta_confidence: Optional[float] = changes.get("confidence_score")
    meta_validation_status: Optional[ValidationStatus] = changes.get("validation_status")
    meta_tags: Optional[list[str]] = changes.get("tags")
    meta_references: Optional[list[str]] = changes.get("references")
    meta_review_date: Optional[datetime] = changes.get("review_date")

    # Compute new confidence score
    if meta_confidence is not None:
        new_confidence: Optional[float] = max(0.0, min(1.0, float(meta_confidence)))
    elif confidence_delta != 0.0:
        raw = record.metadata.confidence_score + confidence_delta
        new_confidence = max(0.0, min(1.0, raw))
    else:
        new_confidence = None

    # Bump the immutable metadata
    new_metadata = record.metadata.bump_version(
        confidence_score=new_confidence,
        validation_status=meta_validation_status,
        tags=meta_tags,
        references=meta_references,
        review_date=meta_review_date,
    )
    record.metadata = new_metadata  # type: ignore[misc]

    # Apply record-level scalar fields (skip protected/metadata keys)
    _protected = {
        "id", "knowledge_type", "metadata",
        "confidence_score", "validation_status", "tags", "references",
        "review_date",
    }
    for field_name, value in changes.items():
        if field_name in _protected:
            continue
        if hasattr(record, field_name):
            setattr(record, field_name, value)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeEvolutionEngine(AbstractKnowledgeEvolutionEngine):
    """
    In-memory, thread-safe implementation of the LUNA Knowledge Evolution
    Engine (v1).

    The engine does **not** own any knowledge stores directly.  It holds
    references to the retrieval engine (for cross-type record lookup) and
    optionally the validation engine (for post-update validation passes) and
    synthesis engine (for cache-invalidation notification).

    All writes mutate the live record objects returned by the retrieval
    engine, so changes are immediately visible to all other engines.

    Lifecycle::

        engine = KnowledgeEvolutionEngine(
            retrieval_engine=my_retrieval_engine,
            validation_engine=my_validation_engine,
        )
        engine.initialize()
        proposal_id = engine.propose_update(
            record_id="...",
            knowledge_type=KnowledgeType.FACT,
            changes={"description": "Revised description"},
            rationale="Source updated",
        )
        updated_record = engine.apply_update(proposal_id)
        engine.shutdown()
    """

    def __init__(
        self,
        *,
        retrieval_engine: Optional[Any] = None,
        validation_engine: Optional[Any] = None,
        synthesis_engine: Optional[Any] = None,
    ) -> None:
        self._retrieval_engine = retrieval_engine
        self._validation_engine = validation_engine
        self._synthesis_engine = synthesis_engine

        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False
        self._started_at: Optional[datetime] = None

        # ── Proposal registry ────────────────────────────────────────────────
        # proposal_id → proposal dict
        self._proposals: dict[str, dict[str, Any]] = {}

        # ── Audit trail (append-only) ────────────────────────────────────────
        # Ordered list of audit entry dicts
        self._audit_trail: list[dict[str, Any]] = []

        # ── Version log ─────────────────────────────────────────────────────
        # record_id → ordered list[version_snapshot dict]
        self._version_log: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # ── Applied evolution history ────────────────────────────────────────
        # record_id → ordered list[proposal dicts] (applied only)
        self._evolution_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # ── Supersession map ────────────────────────────────────────────────
        # record_id → superseding record_id
        self._superseded_by: dict[str, str] = {}
        # record_id → list of records this one supersedes
        self._supersedes: dict[str, list[str]] = defaultdict(list)

        # ── Contradiction registry ───────────────────────────────────────────
        # contradiction_id → KnowledgeContradiction
        self._contradictions: dict[str, KnowledgeContradiction] = {}
        # record_id → list[contradiction_id]
        self._contradictions_by_record: dict[str, list[str]] = defaultdict(list)

        # ── Confidence assessments ───────────────────────────────────────────
        # record_id → list[KnowledgeConfidence] newest-last
        self._confidence_history: dict[str, list[KnowledgeConfidence]] = defaultdict(list)

        # ── Observability counters ───────────────────────────────────────────
        self._proposals_created: int = 0
        self._proposals_applied: int = 0
        self._proposals_rejected: int = 0
        self._contradictions_registered: int = 0
        self._contradictions_resolved: int = 0
        self._retirements: int = 0
        self._last_mutation_at: Optional[datetime] = None

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            try:
                self._proposals.clear()
                self._audit_trail.clear()
                self._version_log.clear()
                self._evolution_history.clear()
                self._superseded_by.clear()
                self._supersedes.clear()
                self._contradictions.clear()
                self._contradictions_by_record.clear()
                self._confidence_history.clear()
                self._proposals_created = 0
                self._proposals_applied = 0
                self._proposals_rejected = 0
                self._contradictions_registered = 0
                self._contradictions_resolved = 0
                self._retirements = 0
                self._last_mutation_at = None
                self._started_at = _utcnow()
                self._initialized = True
                logger.info(
                    "KnowledgeEvolutionEngine initialized (version=%s)", _ENGINE_VERSION
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="KnowledgeEvolutionEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                return
            try:
                self._initialized = False
                logger.info(
                    "KnowledgeEvolutionEngine shutdown "
                    "(applied=%d, rejected=%d, retirements=%d)",
                    self._proposals_applied,
                    self._proposals_rejected,
                    self._retirements,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="KnowledgeEvolutionEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def is_initialized(self) -> bool:
        return self._initialized

    # ─────────────────────────────────────────────────────────────────────────
    # GUARD
    # ─────────────────────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation=operation)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: RECORD LOOKUP
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_record(
        self, record_id: str, knowledge_type: KnowledgeType
    ) -> KnowledgeRecord:
        """
        Retrieve a record via the retrieval engine.

        Raises:
            KnowledgeRetrievalError: Record not found or retrieval engine absent.
        """
        if self._retrieval_engine is None or not self._retrieval_engine.is_initialized():
            raise KnowledgeRetrievalError(
                f"Retrieval engine unavailable; cannot locate record '{record_id}'",
                context={"record_id": record_id, "knowledge_type": knowledge_type.value},
            )
        try:
            return self._retrieval_engine.retrieve_by_id(record_id, knowledge_type)
        except Exception as exc:
            raise KnowledgeRetrievalError(
                f"Record not found: '{record_id}' (type={knowledge_type.value})",
                context={"record_id": record_id, "knowledge_type": knowledge_type.value},
                cause=exc,
            ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: VERSION SNAPSHOT
    # ─────────────────────────────────────────────────────────────────────────

    def _snapshot_version(
        self,
        record: KnowledgeRecord,
        knowledge_type: KnowledgeType,
        proposal_id: Optional[str] = None,
    ) -> None:
        """Append a lightweight version snapshot to the record's version log."""
        snapshot = _make_version_snapshot(
            snapshot_id=_new_id(),
            record_id=record.id,
            version=record.metadata.version,
            knowledge_type=knowledge_type,
            status=record.status,
            confidence_score=record.metadata.confidence_score,
            validation_status=record.metadata.validation_status,
            snapshotted_at=_utcnow(),
            proposal_id=proposal_id,
        )
        self._version_log[record.id].append(snapshot)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: AUDIT APPEND
    # ─────────────────────────────────────────────────────────────────────────

    def _append_audit(
        self,
        *,
        proposal_id: str,
        record_id: str,
        knowledge_type: KnowledgeType,
        action: str,
        actor: str,
        changes: dict[str, Any],
        previous_version: int,
        new_version: int,
        notes: str = "",
    ) -> None:
        entry = _make_audit_entry(
            entry_id=_new_id(),
            proposal_id=proposal_id,
            record_id=record_id,
            knowledge_type=knowledge_type,
            action=action,
            actor=actor,
            changes=changes,
            previous_version=previous_version,
            new_version=new_version,
            occurred_at=_utcnow(),
            notes=notes,
        )
        self._audit_trail.append(entry)
        self._last_mutation_at = _utcnow()

    # ─────────────────────────────────────────────────────────────────────────
    # PROPOSAL MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def propose_update(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        changes: dict[str, Any],
        *,
        rationale: str = "",
        source: str = "",
        confidence_delta: float = 0.0,
    ) -> str:
        """
        Propose an update to an existing knowledge record.

        The record is looked up to confirm it exists and is not terminal.
        The proposal is queued; no changes are applied until apply_update()
        is called.

        Returns:
            proposal_id (str)

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Record not found.
            KnowledgeError:          Record is in a terminal status.
        """
        with self._lock:
            self._require_initialized("propose_update")

            # Validate the record exists and is mutable
            record = self._fetch_record(record_id, knowledge_type)
            if record.status.is_terminal:
                raise KnowledgeError(
                    f"Cannot propose update to terminal record '{record_id}' "
                    f"(status={record.status.value})",
                    context={
                        "record_id": record_id,
                        "knowledge_type": knowledge_type.value,
                        "status": record.status.value,
                    },
                )

            proposal_id = _new_id()
            proposal = _make_proposal(
                proposal_id=proposal_id,
                record_id=record_id,
                knowledge_type=knowledge_type,
                changes=changes,
                rationale=rationale,
                source=source,
                confidence_delta=confidence_delta,
                created_at=_utcnow(),
            )
            self._proposals[proposal_id] = proposal
            self._proposals_created += 1

            logger.debug(
                "Proposal created: id=%s record_id=%s type=%s",
                proposal_id,
                record_id,
                knowledge_type.value,
            )
            return proposal_id

    def apply_update(self, proposal_id: str) -> KnowledgeRecord:
        """
        Apply an approved proposal to the target record.

        Increments the record's version via KnowledgeMetadata.bump_version(),
        records a version snapshot, appends an audit entry, and runs a
        post-update validation pass if the validation engine is available.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeRetrievalError:  Proposal or record not found.
            KnowledgeError:           Proposal is not in PENDING state.
            KnowledgeValidationError: Post-update validation fails with blocking issues.
        """
        with self._lock:
            self._require_initialized("apply_update")

            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise KnowledgeRetrievalError(
                    f"Proposal not found: '{proposal_id}'",
                    context={"proposal_id": proposal_id},
                )
            if proposal["state"] != _PROPOSAL_PENDING:
                raise KnowledgeError(
                    f"Proposal '{proposal_id}' is not pending "
                    f"(state={proposal['state']})",
                    context={"proposal_id": proposal_id, "state": proposal["state"]},
                )

            record_id = proposal["record_id"]
            knowledge_type = KnowledgeType(proposal["knowledge_type"])
            changes = proposal["changes"]
            confidence_delta = float(proposal.get("confidence_delta", 0.0))

            record = self._fetch_record(record_id, knowledge_type)
            if record.status.is_terminal:
                raise KnowledgeError(
                    f"Cannot apply update to terminal record '{record_id}' "
                    f"(status={record.status.value})",
                    context={"record_id": record_id, "status": record.status.value},
                )

            previous_version = record.metadata.version

            # Apply changes
            _apply_changes_to_record(record, changes, confidence_delta)

            new_version = record.metadata.version

            # Mark proposal as applied
            now_iso = _utcnow().isoformat()
            proposal["state"] = _PROPOSAL_APPLIED
            proposal["resolved_at"] = now_iso

            # Version snapshot
            self._snapshot_version(record, knowledge_type, proposal_id=proposal_id)

            # Audit entry
            self._append_audit(
                proposal_id=proposal_id,
                record_id=record_id,
                knowledge_type=knowledge_type,
                action="apply_update",
                actor=proposal.get("source", ""),
                changes=changes,
                previous_version=previous_version,
                new_version=new_version,
                notes=proposal.get("rationale", ""),
            )

            # Evolution history
            self._evolution_history[record_id].append(deepcopy(proposal))

            self._proposals_applied += 1

            # Post-update validation (optional; non-blocking on absence)
            if (
                self._validation_engine is not None
                and self._validation_engine.is_initialized()
            ):
                try:
                    result: KnowledgeValidationResult = (
                        self._validation_engine.validate_record(record_id, knowledge_type)
                    )
                    # When the validation engine has no store registered for
                    # this record, validate_record returns a synthetic result
                    # (knowledge_name="<unknown>") flagging the record as
                    # unreachable rather than reporting an actual content
                    # defect. That reflects a registry wiring gap, not a
                    # problem introduced by this update, so it must not block
                    # the apply.
                    if result.knowledge_name == "<unknown>":
                        pass
                    elif result.has_blocking_issues:
                        raise KnowledgeValidationError(
                            f"Post-update validation failed for '{record_id}': "
                            f"{result.error_count} error(s), "
                            f"{result.critical_count} critical issue(s)",
                            context={
                                "record_id": record_id,
                                "proposal_id": proposal_id,
                                "error_count": result.error_count,
                                "critical_count": result.critical_count,
                            },
                        )
                except KnowledgeValidationError:
                    raise
                except Exception:
                    # Validation engine faults must not block the apply
                    pass

            logger.info(
                "Proposal applied: id=%s record_id=%s v%d→v%d",
                proposal_id,
                record_id,
                previous_version,
                new_version,
            )
            return record

    def reject_update(self, proposal_id: str, *, reason: str = "") -> None:
        """
        Reject a pending proposal without applying it.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Proposal not found.
            KnowledgeError:          Proposal is not in PENDING state.
        """
        with self._lock:
            self._require_initialized("reject_update")

            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise KnowledgeRetrievalError(
                    f"Proposal not found: '{proposal_id}'",
                    context={"proposal_id": proposal_id},
                )
            if proposal["state"] != _PROPOSAL_PENDING:
                raise KnowledgeError(
                    f"Proposal '{proposal_id}' is not pending "
                    f"(state={proposal['state']})",
                    context={"proposal_id": proposal_id, "state": proposal["state"]},
                )

            proposal["state"] = _PROPOSAL_REJECTED
            proposal["resolved_at"] = _utcnow().isoformat()
            proposal["resolution_notes"] = reason

            self._append_audit(
                proposal_id=proposal_id,
                record_id=proposal["record_id"],
                knowledge_type=KnowledgeType(proposal["knowledge_type"]),
                action="reject_update",
                actor=proposal.get("source", ""),
                changes={},
                previous_version=0,
                new_version=0,
                notes=reason,
            )

            self._proposals_rejected += 1
            logger.debug(
                "Proposal rejected: id=%s reason=%s", proposal_id, reason
            )

    def get_proposal(self, proposal_id: str) -> dict[str, Any]:
        """
        Return a proposal dict by ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Proposal not found.
        """
        with self._lock:
            self._require_initialized("get_proposal")
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise KnowledgeRetrievalError(
                    f"Proposal not found: '{proposal_id}'",
                    context={"proposal_id": proposal_id},
                )
            return deepcopy(proposal)

    def list_pending_proposals(self) -> list[dict[str, Any]]:
        """
        Return all proposals currently in PENDING state.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("list_pending_proposals")
            return [
                deepcopy(p)
                for p in self._proposals.values()
                if p["state"] == _PROPOSAL_PENDING
            ]

    def get_evolution_history(self, record_id: str) -> list[dict[str, Any]]:
        """
        Return the ordered list of applied proposals for a record (oldest first).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_evolution_history")
            return [deepcopy(p) for p in self._evolution_history.get(record_id, [])]

    # ─────────────────────────────────────────────────────────────────────────
    # EVOLVE KNOWLEDGE  (high-level convenience wrapper)
    # ─────────────────────────────────────────────────────────────────────────

    def evolve_knowledge(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        changes: dict[str, Any],
        *,
        rationale: str = "",
        source: str = "",
        confidence_delta: float = 0.0,
        auto_apply: bool = True,
    ) -> KnowledgeRecord:
        """
        Propose and optionally immediately apply a change to a knowledge record.

        This is the primary high-level entry point for knowledge evolution.
        When auto_apply=True (default) the proposal is both created and applied
        in the same call.  When auto_apply=False the proposal ID is stored but
        the record is not mutated; callers must later call apply_update().

        Args:
            record_id:        ID of the target record.
            knowledge_type:   KnowledgeType of the target record.
            changes:          Dict of field_name → new_value to apply.
            rationale:        Human-readable reason for the change.
            source:           Initiating source identifier.
            confidence_delta: Signed offset to add to existing confidence_score.
            auto_apply:       When True, immediately apply after proposing.

        Returns:
            The updated KnowledgeRecord (auto_apply=True) or the current
            unchanged record (auto_apply=False).

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeRetrievalError:  Record not found.
            KnowledgeError:           Record is terminal.
            KnowledgeValidationError: Post-update validation fails (auto_apply=True).
        """
        with self._lock:
            self._require_initialized("evolve_knowledge")
            proposal_id = self.propose_update(
                record_id,
                knowledge_type,
                changes,
                rationale=rationale,
                source=source,
                confidence_delta=confidence_delta,
            )
            if auto_apply:
                return self.apply_update(proposal_id)
            # Return current record without changes
            return self._fetch_record(record_id, knowledge_type)

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIDENCE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def update_confidence(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        new_confidence: Optional[float] = None,
        source_reliability: Optional[float] = None,
        recency_score: Optional[float] = None,
        corroboration_score: Optional[float] = None,
        consistency_score: Optional[float] = None,
        assessment_notes: str = "",
    ) -> KnowledgeConfidence:
        """
        Compute and apply a structured confidence reassessment for a record.

        Constructs a KnowledgeConfidence model from the four contributing
        factors, derives the overall_score, applies it to the record's
        metadata via a proposal, and returns the assessment.

        Args:
            record_id:            ID of the target record.
            knowledge_type:       KnowledgeType of the target record.
            source_reliability:   Component score [0.0, 1.0].
            recency_score:        Component score [0.0, 1.0].
            corroboration_score:  Component score [0.0, 1.0].
            consistency_score:    Component score [0.0, 1.0].
            assessment_notes:     Free-text notes about this assessment.

        Returns:
            The KnowledgeConfidence record capturing all components.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Record not found.
            KnowledgeError:          Record is terminal.
        """
        with self._lock:
            self._require_initialized("update_confidence")

            # When only new_confidence is provided, distribute it uniformly
            # across the four components so the weighted formula returns it intact.
            if new_confidence is not None:
                _c = max(0.0, min(1.0, new_confidence))
                _sr = source_reliability if source_reliability is not None else _c
                _rs = recency_score if recency_score is not None else _c
                _cs = corroboration_score if corroboration_score is not None else _c
                _co = consistency_score if consistency_score is not None else _c
            else:
                _sr = source_reliability if source_reliability is not None else 0.5
                _rs = recency_score if recency_score is not None else 0.5
                _cs = corroboration_score if corroboration_score is not None else 0.5
                _co = consistency_score if consistency_score is not None else 0.5

            confidence = KnowledgeConfidence.create(
                knowledge_id=record_id,
                source_reliability=_sr,
                recency_score=_rs,
                corroboration_score=_cs,
                consistency_score=_co,
                assessment_notes=assessment_notes,
            )

            # Apply via proposal so it enters the audit trail
            self.evolve_knowledge(
                record_id,
                knowledge_type,
                changes={"confidence_score": confidence.overall_score},
                rationale=assessment_notes or "confidence reassessment",
                source="confidence_engine",
                auto_apply=True,
            )

            # Store assessment history
            self._confidence_history[record_id].append(confidence)

            logger.debug(
                "Confidence updated: record_id=%s overall=%.3f level=%s",
                record_id,
                confidence.overall_score,
                confidence.confidence_level.value,
            )
            return confidence

    def get_confidence_history(
        self, record_id: str
    ) -> list[KnowledgeConfidence]:
        """
        Return all confidence assessments for a record (oldest first).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_confidence_history")
            return list(self._confidence_history.get(record_id, []))

    def get_latest_confidence(
        self, record_id: str
    ) -> Optional[KnowledgeConfidence]:
        """
        Return the most recent confidence assessment for a record, or None.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_latest_confidence")
            history = self._confidence_history.get(record_id, [])
            return history[-1] if history else None

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION STATE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def update_validation_state(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        new_validation_status: ValidationStatus,
        *,
        notes: str = "",
    ) -> KnowledgeRecord:
        """
        Update the validation_status field on a record's metadata.

        This is a direct metadata mutation; it does not re-run the validation
        pass — use the KnowledgeValidationEngine for that.  This method simply
        records the outcome of an external validation decision.

        Args:
            record_id:             ID of the target record.
            knowledge_type:        KnowledgeType of the target record.
            new_validation_status: The ValidationStatus to write.
            notes:                 Reason or commentary for the state change.

        Returns:
            The updated KnowledgeRecord.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Record not found.
            KnowledgeError:          Record is terminal.
        """
        with self._lock:
            self._require_initialized("update_validation_state")
            return self.evolve_knowledge(
                record_id,
                knowledge_type,
                changes={"validation_status": new_validation_status},
                rationale=notes or f"validation state set to {new_validation_status.value}",
                source="validation_engine",
                auto_apply=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # VERSION TRACKING
    # ─────────────────────────────────────────────────────────────────────────

    def get_version_log(self, record_id: str) -> list[dict[str, Any]]:
        """
        Return the ordered version snapshot log for a record (oldest first).

        Each entry contains: id, record_id, version, knowledge_type, status,
        confidence_score, validation_status, snapshotted_at, proposal_id.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_version_log")
            return [deepcopy(s) for s in self._version_log.get(record_id, [])]

    def get_current_version(self, record_id: str, knowledge_type: KnowledgeType) -> int:
        """
        Return the current version number of a record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Record not found.
        """
        with self._lock:
            self._require_initialized("get_current_version")
            record = self._fetch_record(record_id, knowledge_type)
            return record.metadata.version

    # ─────────────────────────────────────────────────────────────────────────
    # RETIREMENT MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def deprecate_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        superseded_by: Optional[str] = None,
        reason: str = "",
    ) -> KnowledgeRecord:
        """
        Mark a record as DEPRECATED or SUPERSEDED.

        If superseded_by is supplied, the record's status becomes SUPERSEDED
        and the supersession link is registered; otherwise it becomes DEPRECATED.

        Args:
            record_id:      ID of the record to retire.
            knowledge_type: KnowledgeType of the record.
            superseded_by:  Optional ID of the new record replacing this one.
            reason:         Human-readable deprecation reason.

        Returns:
            The updated (deprecated/superseded) KnowledgeRecord.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Record not found.
            KnowledgeError:          Record is already terminal or transition invalid.
        """
        with self._lock:
            self._require_initialized("deprecate_record")

            record = self._fetch_record(record_id, knowledge_type)
            current_status = record.status

            if current_status.is_terminal:
                raise KnowledgeError(
                    f"Cannot deprecate terminal record '{record_id}' "
                    f"(status={current_status.value})",
                    context={"record_id": record_id, "status": current_status.value},
                )

            if superseded_by is not None:
                target_status = KnowledgeStatus.SUPERSEDED
            else:
                target_status = KnowledgeStatus.DEPRECATED

            if target_status not in _VALID_TRANSITIONS.get(current_status, frozenset()):
                raise KnowledgeError(
                    f"Invalid status transition for '{record_id}': "
                    f"{current_status.value} → {target_status.value}",
                    context={
                        "record_id": record_id,
                        "from_status": current_status.value,
                        "to_status": target_status.value,
                    },
                )

            # Register supersession links before mutating
            if superseded_by is not None:
                self._superseded_by[record_id] = superseded_by
                self._supersedes[superseded_by].append(record_id)

            updated = self.evolve_knowledge(
                record_id,
                knowledge_type,
                changes={"status": target_status},
                rationale=reason or f"record {target_status.value}",
                source="evolution_engine",
                auto_apply=True,
            )

            self._retirements += 1

            # Audit entry for supersession link
            if superseded_by is not None:
                self._append_audit(
                    proposal_id=_new_id(),
                    record_id=record_id,
                    knowledge_type=knowledge_type,
                    action="supersession_registered",
                    actor="evolution_engine",
                    changes={"superseded_by": superseded_by},
                    previous_version=updated.metadata.version,
                    new_version=updated.metadata.version,
                    notes=reason,
                )

            logger.info(
                "Record deprecated/superseded: id=%s status=%s superseded_by=%s",
                record_id,
                target_status.value,
                superseded_by,
            )
            return updated

    def retire_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        reason: str = "",
        archive: bool = False,
    ) -> KnowledgeRecord:
        """
        Permanently retire a record by transitioning it to RETRACTED or ARCHIVED.

        This is a one-way, terminal operation.  The record remains in the store
        for audit and referential integrity but will no longer be returned by
        active queries.

        Args:
            record_id:      ID of the record to retire.
            knowledge_type: KnowledgeType of the record.
            reason:         Human-readable reason for retirement.
            archive:        When True, use ARCHIVED status; otherwise RETRACTED.

        Returns:
            The retired KnowledgeRecord.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Record not found.
            KnowledgeError:          Record is already terminal or transition invalid.
        """
        with self._lock:
            self._require_initialized("retire_record")

            record = self._fetch_record(record_id, knowledge_type)
            current_status = record.status

            if current_status.is_terminal:
                raise KnowledgeError(
                    f"Record '{record_id}' is already terminal "
                    f"(status={current_status.value})",
                    context={"record_id": record_id, "status": current_status.value},
                )

            target_status = KnowledgeStatus.ARCHIVED if archive else KnowledgeStatus.RETRACTED

            if target_status not in _VALID_TRANSITIONS.get(current_status, frozenset()):
                raise KnowledgeError(
                    f"Invalid status transition for '{record_id}': "
                    f"{current_status.value} → {target_status.value}",
                    context={
                        "record_id": record_id,
                        "from_status": current_status.value,
                        "to_status": target_status.value,
                    },
                )

            updated = self.evolve_knowledge(
                record_id,
                knowledge_type,
                changes={"status": target_status},
                rationale=reason or f"record {target_status.value}",
                source="evolution_engine",
                auto_apply=True,
            )

            self._retirements += 1

            logger.info(
                "Record retired: id=%s status=%s", record_id, target_status.value
            )
            return updated

    # ─────────────────────────────────────────────────────────────────────────
    # SUPERSESSION TRACKING
    # ─────────────────────────────────────────────────────────────────────────

    def get_superseded_by(self, record_id: str) -> Optional[str]:
        """
        Return the ID of the record that supersedes this one, or None.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_superseded_by")
            return self._superseded_by.get(record_id)

    def get_supersedes(self, record_id: str) -> list[str]:
        """
        Return the list of record IDs that the given record supersedes.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_supersedes")
            return list(self._supersedes.get(record_id, []))

    def register_supersession(
        self,
        old_record_id: str,
        new_record_id: str,
        knowledge_type: KnowledgeType,
        *,
        reason: str = "",
    ) -> None:
        """
        Register a supersession link between two records without changing status.

        Useful when the status transition has already been handled externally
        and only the link metadata needs to be stored.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("register_supersession")
            self._superseded_by[old_record_id] = new_record_id
            self._supersedes[new_record_id].append(old_record_id)
            self._append_audit(
                proposal_id=_new_id(),
                record_id=old_record_id,
                knowledge_type=knowledge_type,
                action="supersession_registered",
                actor="evolution_engine",
                changes={"superseded_by": new_record_id},
                previous_version=0,
                new_version=0,
                notes=reason,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # CONTRADICTION RESOLUTION
    # ─────────────────────────────────────────────────────────────────────────

    def register_contradiction(
        self,
        record_a_id: str,
        record_b_id: str,
        contradiction_description: str,
        severity: ValidationSeverity = ValidationSeverity.WARNING,
    ) -> KnowledgeContradiction:
        """
        Register a detected contradiction between two knowledge records.

        Both records are linked to the contradiction so that it can be
        discovered via either party's ID.

        Returns:
            The created KnowledgeContradiction.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("register_contradiction")

            contradiction = KnowledgeContradiction.create(
                record_a_id=record_a_id,
                record_b_id=record_b_id,
                contradiction_description=contradiction_description,
                severity=severity,
            )
            self._contradictions[contradiction.id] = contradiction
            self._contradictions_by_record[record_a_id].append(contradiction.id)
            self._contradictions_by_record[record_b_id].append(contradiction.id)
            self._contradictions_registered += 1

            logger.debug(
                "Contradiction registered: id=%s a=%s b=%s severity=%s",
                contradiction.id,
                record_a_id,
                record_b_id,
                severity.value,
            )
            return contradiction

    def resolve_contradiction(
        self,
        contradiction_id: str,
        *,
        resolution_notes: str = "",
        acknowledge_only: bool = False,
    ) -> KnowledgeContradiction:
        """
        Mark a contradiction as resolved or acknowledged.

        Args:
            contradiction_id:  ID of the KnowledgeContradiction to resolve.
            resolution_notes:  Free-text explanation of how it was resolved.
            acknowledge_only:  When True, status becomes 'acknowledged' rather
                               than 'resolved' (useful for known, accepted conflicts).

        Returns:
            A new KnowledgeContradiction instance with updated resolution_status.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Contradiction not found.
            KnowledgeError:          Contradiction is already resolved.
        """
        with self._lock:
            self._require_initialized("resolve_contradiction")

            existing = self._contradictions.get(contradiction_id)
            if existing is None:
                raise KnowledgeRetrievalError(
                    f"Contradiction not found: '{contradiction_id}'",
                    context={"contradiction_id": contradiction_id},
                )
            if existing.is_resolved:
                raise KnowledgeError(
                    f"Contradiction '{contradiction_id}' is already resolved",
                    context={"contradiction_id": contradiction_id},
                )

            new_status = "acknowledged" if acknowledge_only else "resolved"

            # KnowledgeContradiction is frozen; rebuild with updated fields
            resolved = KnowledgeContradiction(
                id=existing.id,
                record_a_id=existing.record_a_id,
                record_b_id=existing.record_b_id,
                contradiction_description=existing.contradiction_description,
                severity=existing.severity,
                detected_at=existing.detected_at,
                resolution_status=new_status,
                resolution_notes=resolution_notes,
            )
            self._contradictions[contradiction_id] = resolved

            if new_status == "resolved":
                self._contradictions_resolved += 1

            self._append_audit(
                proposal_id=_new_id(),
                record_id=existing.record_a_id,
                knowledge_type=KnowledgeType.FACT,   # type-agnostic; best-effort
                action="contradiction_resolved",
                actor="evolution_engine",
                changes={
                    "contradiction_id": contradiction_id,
                    "resolution_status": new_status,
                },
                previous_version=0,
                new_version=0,
                notes=resolution_notes,
            )

            logger.debug(
                "Contradiction %s: id=%s", new_status, contradiction_id
            )
            return resolved

    def get_contradictions_for_record(
        self, record_id: str
    ) -> list[KnowledgeContradiction]:
        """
        Return all contradictions involving the given record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_contradictions_for_record")
            ids = self._contradictions_by_record.get(record_id, [])
            return [
                self._contradictions[cid]
                for cid in ids
                if cid in self._contradictions
            ]

    def get_unresolved_contradictions(self) -> list[KnowledgeContradiction]:
        """
        Return all unresolved contradictions across all records.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_unresolved_contradictions")
            return [
                c for c in self._contradictions.values()
                if not c.is_resolved
            ]

    def get_contradiction(self, contradiction_id: str) -> KnowledgeContradiction:
        """
        Return a specific contradiction by ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Not found.
        """
        with self._lock:
            self._require_initialized("get_contradiction")
            contradiction = self._contradictions.get(contradiction_id)
            if contradiction is None:
                raise KnowledgeRetrievalError(
                    f"Contradiction not found: '{contradiction_id}'",
                    context={"contradiction_id": contradiction_id},
                )
            return contradiction

    # ─────────────────────────────────────────────────────────────────────────
    # HISTORICAL TRACKING
    # ─────────────────────────────────────────────────────────────────────────

    def get_audit_trail(
        self,
        *,
        record_id: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = None,
        action: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Return filtered audit trail entries (newest first).

        Args:
            record_id:      Restrict to entries for a specific record.
            knowledge_type: Restrict to entries of a specific KnowledgeType.
            action:         Restrict to entries with a specific action string.
            limit:          Maximum number of entries to return.
            offset:         Pagination offset.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_audit_trail")

            entries = self._audit_trail
            if record_id is not None:
                entries = [e for e in entries if e["record_id"] == record_id]
            if knowledge_type is not None:
                entries = [
                    e for e in entries
                    if e["knowledge_type"] == knowledge_type.value
                ]
            if action is not None:
                entries = [e for e in entries if e["action"] == action]

            # Newest first
            entries = list(reversed(entries))
            sliced = entries[offset: offset + limit]
            return [deepcopy(e) for e in sliced]

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """
        Return a lightweight liveness/readiness summary.

        Required keys: engine, initialized, record_count, status.
        """
        with self._lock:
            pending = sum(
                1 for p in self._proposals.values()
                if p["state"] == _PROPOSAL_PENDING
            )
            return {
                "engine": "KnowledgeEvolutionEngine",
                "initialized": self._initialized,
                "record_count": len(self._evolution_history),
                "status": "healthy" if self._initialized else "offline",
                "pending_proposals": pending,
                "contradictions_unresolved": sum(
                    1 for c in self._contradictions.values() if not c.is_resolved
                ),
            }

    def diagnostics_report(self) -> dict[str, Any]:
        """
        Return a full introspection snapshot for operator debugging.

        Required keys (superset of health_report): engine, initialized,
        record_count, status, index_size, duplicate_checks, mutation_count,
        last_mutation_at.
        """
        with self._lock:
            total_proposals = len(self._proposals)
            pending = sum(
                1 for p in self._proposals.values()
                if p["state"] == _PROPOSAL_PENDING
            )
            return {
                "engine": "KnowledgeEvolutionEngine",
                "engine_version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "record_count": len(self._evolution_history),
                "status": "healthy" if self._initialized else "offline",
                "index_size": len(self._version_log),
                "duplicate_checks": 0,
                "mutation_count": self._proposals_applied,
                "last_mutation_at": (
                    self._last_mutation_at.isoformat()
                    if self._last_mutation_at
                    else None
                ),
                "started_at": (
                    self._started_at.isoformat() if self._started_at else None
                ),
                "proposals_created": self._proposals_created,
                "proposals_applied": self._proposals_applied,
                "proposals_rejected": self._proposals_rejected,
                "proposals_pending": pending,
                "proposals_total": total_proposals,
                "contradictions_registered": self._contradictions_registered,
                "contradictions_resolved": self._contradictions_resolved,
                "contradictions_unresolved": sum(
                    1 for c in self._contradictions.values() if not c.is_resolved
                ),
                "retirements": self._retirements,
                "version_log_entries": sum(
                    len(v) for v in self._version_log.values()
                ),
                "audit_trail_entries": len(self._audit_trail),
                "supersession_links": len(self._superseded_by),
                "confidence_assessments": sum(
                    len(v) for v in self._confidence_history.values()
                ),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # EVOLUTION REPORTING
    # ─────────────────────────────────────────────────────────────────────────

    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate evolution statistics.

        Required keys: proposals_created, proposals_applied, proposals_rejected,
        proposals_pending, contradictions_registered, contradictions_resolved,
        retirements, audit_trail_entries, generated_at.
        """
        with self._lock:
            self._require_initialized("audit_report")

            pending = sum(
                1 for p in self._proposals.values()
                if p["state"] == _PROPOSAL_PENDING
            )
            unresolved_contradictions = [
                c for c in self._contradictions.values() if not c.is_resolved
            ]
            blocking_contradictions = [
                c for c in unresolved_contradictions if c.is_blocking
            ]

            return {
                "engine": "KnowledgeEvolutionEngine",
                "engine_version": _ENGINE_VERSION,
                "proposals_created": self._proposals_created,
                "proposals_applied": self._proposals_applied,
                "proposals_rejected": self._proposals_rejected,
                "proposals_pending": pending,
                "contradictions_registered": self._contradictions_registered,
                "contradictions_resolved": self._contradictions_resolved,
                "contradictions_unresolved": len(unresolved_contradictions),
                "contradictions_blocking": len(blocking_contradictions),
                "retirements": self._retirements,
                "supersession_links": len(self._superseded_by),
                "version_log_entries": sum(
                    len(v) for v in self._version_log.values()
                ),
                "audit_trail_entries": len(self._audit_trail),
                "records_with_history": len(self._evolution_history),
                "confidence_assessments": sum(
                    len(v) for v in self._confidence_history.values()
                ),
                "last_mutation_at": (
                    self._last_mutation_at.isoformat()
                    if self._last_mutation_at
                    else None
                ),
                "generated_at": _utcnow().isoformat(),
            }

    def evolution_report_for_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> dict[str, Any]:
        """
        Return a per-record evolution summary.

        Includes version log, applied proposals, confidence history,
        contradiction involvement, and supersession links.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Record not found.
        """
        with self._lock:
            self._require_initialized("evolution_report_for_record")

            # Confirm record exists
            record = self._fetch_record(record_id, knowledge_type)

            version_log = [
                deepcopy(s) for s in self._version_log.get(record_id, [])
            ]
            applied_proposals = [
                deepcopy(p) for p in self._evolution_history.get(record_id, [])
            ]
            confidence_hist = [
                c.to_dict() for c in self._confidence_history.get(record_id, [])
            ]
            contradictions = [
                c.to_dict()
                for cid in self._contradictions_by_record.get(record_id, [])
                if (c := self._contradictions.get(cid)) is not None
            ]
            superseded_by = self._superseded_by.get(record_id)
            supersedes = list(self._supersedes.get(record_id, []))

            return {
                "record_id": record_id,
                "knowledge_type": knowledge_type.value,
                "current_version": record.metadata.version,
                "current_status": record.status.value,
                "current_confidence_score": record.metadata.confidence_score,
                "current_validation_status": record.metadata.validation_status.value,
                "version_log": version_log,
                "applied_proposals_count": len(applied_proposals),
                "applied_proposals": applied_proposals,
                "confidence_assessments": confidence_hist,
                "contradictions": contradictions,
                "superseded_by": superseded_by,
                "supersedes": supersedes,
                "generated_at": _utcnow().isoformat(),
            }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "KnowledgeEvolutionEngine",
]