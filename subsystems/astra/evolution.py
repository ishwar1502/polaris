# subsystems/astra/evolution.py
"""
ASTRA v5 Identity Evolution Engine.

The Identity Evolution Engine is the only authorised pathway for updating
the identity model based on accumulated evidence.  It enforces the critical
ASTRA law:

    Short-term events ≠ Identity Change.

Evolution requires:
1. Minimum evidence count has been met.
2. Evidence confidence exceeds the threshold.
3. The evolution passes change validation.

All evolution events are logged as immutable :class:`~subsystems.astra.models.EvolutionRecord`
entries in a permanent history log.  Records are never modified or deleted.

All operations are thread-safe via an internal :class:`threading.RLock`.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from subsystems.astra.exceptions import (
    EvolutionError,
    IdentityNotFoundError,
    InsufficientEvidenceError,
)
from subsystems.astra.models import (
    EvolutionRecord,
    EvolutionTrigger,
    IdentityProfile,
)

_logger = logging.getLogger(__name__)

# Default thresholds
_DEFAULT_MIN_EVIDENCE: int = 3
_DEFAULT_MIN_CONFIDENCE: float = 0.65


class IdentityEvolutionEngine:
    """Safe, evidence-gated update pathway for the identity model.

    Parameters
    ----------
    min_evidence_count:
        Minimum number of evidence items required before an evolution can
        be approved.  Defaults to 3.
    min_confidence:
        Minimum confidence score (0.0-1.0) required.  Defaults to 0.65.
    """

    def __init__(
        self,
        min_evidence_count: int = _DEFAULT_MIN_EVIDENCE,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._min_evidence_count = min_evidence_count
        self._min_confidence = min_confidence
        self._evolution_log: list[EvolutionRecord] = []
        self._pending_evidence: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Evidence accumulation
    # ------------------------------------------------------------------

    def accumulate_evidence(
        self,
        *,
        source: str,
        signal_type: str,
        weight: float,
        description: str,
        related_fields: list[str] | None = None,
    ) -> None:
        """Add an evidence item to the pending pool.

        Evidence items accumulate until a threshold is met, at which point
        :meth:`evolve_identity` may be called to commit the evolution.

        Parameters
        ----------
        source:
            Subsystem providing the evidence.
        signal_type:
            Classification of the signal.
        weight:
            0.0-1.0 significance weight.
        description:
            Description of the observation.
        related_fields:
            Optional list of identity field names this evidence relates to.
        """
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Evidence weight must be 0.0-1.0, got {weight}.")

        item = {
            "source": source,
            "signal_type": signal_type,
            "weight": weight,
            "description": description,
            "related_fields": list(related_fields or []),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._pending_evidence.append(item)

        _logger.debug(
            "Evolution evidence accumulated: source=%s type=%s weight=%.2f",
            source,
            signal_type,
            weight,
        )

    def get_pending_evidence_count(self) -> int:
        """Return the current number of pending evidence items."""
        with self._lock:
            return len(self._pending_evidence)

    def clear_pending_evidence(self) -> int:
        """Clear pending evidence without committing an evolution.

        Returns the count of items cleared.
        """
        with self._lock:
            count = len(self._pending_evidence)
            self._pending_evidence.clear()
            return count

    def compute_confidence(self) -> float:
        """Compute a confidence score from the current pending evidence.

        The score is the weighted average of all pending evidence weights.

        Returns
        -------
        float
            Confidence score (0.0-1.0).  Returns 0.0 if no evidence exists.
        """
        with self._lock:
            if not self._pending_evidence:
                return 0.0
            total_weight = sum(e["weight"] for e in self._pending_evidence)
            return min(1.0, total_weight / len(self._pending_evidence))

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve_identity(
        self,
        identity: IdentityProfile,
        updates: dict[str, Any],
        *,
        trigger: EvolutionTrigger = EvolutionTrigger.EVIDENCE_THRESHOLD,
        notes: str = "",
        force: bool = False,
    ) -> tuple[IdentityProfile, EvolutionRecord]:
        """Attempt to evolve the identity model.

        This method validates that sufficient evidence exists, applies the
        *updates* to the *identity* profile, and logs an immutable
        :class:`~subsystems.astra.models.EvolutionRecord`.

        Parameters
        ----------
        identity:
            The current :class:`~subsystems.astra.models.IdentityProfile`.
        updates:
            Dictionary of field-name → new-value pairs to apply.
        trigger:
            What triggered this evolution attempt.
        notes:
            Human-readable rationale.
        force:
            If True, bypass evidence threshold checks (for manual updates
            via :meth:`~subsystems.astra.astra.AstraSubsystem.update_identity`).
            Should only be True for explicit user-initiated identity updates.

        Returns
        -------
        tuple[IdentityProfile, EvolutionRecord]
            The updated profile and the immutable evolution log entry.

        Raises
        ------
        IdentityNotFoundError
            If *identity* is None.
        InsufficientEvidenceError
            If evidence thresholds are not met (and force=False).
        EvolutionError
            If the updates contain invalid data.
        """
        if identity is None:
            raise IdentityNotFoundError()

        with self._lock:
            evidence_count = len(self._pending_evidence)
            confidence = self.compute_confidence()

        if not force:
            if evidence_count < self._min_evidence_count:
                raise InsufficientEvidenceError(
                    required=self._min_evidence_count,
                    available=evidence_count,
                    confidence=confidence,
                )
            if confidence < self._min_confidence:
                raise InsufficientEvidenceError(
                    required=self._min_evidence_count,
                    available=evidence_count,
                    confidence=confidence,
                )

        # Capture pre-evolution snapshot of changed fields
        changed_fields = [f for f in updates if hasattr(identity, f)]
        if not changed_fields:
            raise EvolutionError(
                "No valid identity fields specified in updates."
            )

        previous_snapshot = {
            field: getattr(identity, field) for field in changed_fields
        }

        # Apply updates to the profile
        allowed_fields = {
            "name",
            "background",
            "education",
            "career_direction",
            "interests",
            "core_identity_tags",
            "metadata",
        }
        invalid_fields = set(updates) - allowed_fields
        if invalid_fields:
            raise EvolutionError(
                f"Cannot evolve unknown identity fields: {sorted(invalid_fields)}."
            )

        self._apply_identity_updates(identity, updates)

        new_snapshot = {
            field: getattr(identity, field) for field in changed_fields
        }

        # Create the immutable evolution record
        record = EvolutionRecord.create(
            trigger=trigger,
            changed_fields=changed_fields,
            previous_snapshot={k: _serialise(v) for k, v in previous_snapshot.items()},
            new_snapshot={k: _serialise(v) for k, v in new_snapshot.items()},
            confidence=confidence if not force else 1.0,
            evidence_count=evidence_count,
            notes=notes,
        )

        with self._lock:
            self._evolution_log.append(record)
            # Clear pending evidence after successful evolution
            self._pending_evidence.clear()

        _logger.info(
            "Identity evolved: fields=%s trigger=%s confidence=%.2f record=%s",
            changed_fields,
            trigger.name,
            record.confidence,
            record.record_id,
        )
        return identity, record

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_evolution_history(self) -> list[EvolutionRecord]:
        """Return the full, immutable evolution history.

        Returns
        -------
        list[EvolutionRecord]
            All evolution records in chronological order.
        """
        with self._lock:
            return list(self._evolution_log)

    def get_evolution_count(self) -> int:
        """Return the total number of evolution events recorded."""
        with self._lock:
            return len(self._evolution_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_identity_updates(
        identity: IdentityProfile, updates: dict[str, Any]
    ) -> None:
        """Apply field updates directly to the identity profile."""
        if "name" in updates:
            val = str(updates["name"]).strip()
            if not val:
                raise EvolutionError("Identity name cannot be empty.")
            identity.name = val

        if "background" in updates:
            identity.background = str(updates["background"])

        if "education" in updates:
            identity.education = str(updates["education"])

        if "career_direction" in updates:
            identity.career_direction = str(updates["career_direction"])

        if "interests" in updates:
            val = updates["interests"]
            if not isinstance(val, list):
                raise EvolutionError("'interests' must be a list.")
            identity.interests = [str(i) for i in val]

        if "core_identity_tags" in updates:
            val = updates["core_identity_tags"]
            if not isinstance(val, list):
                raise EvolutionError("'core_identity_tags' must be a list.")
            identity.core_identity_tags = [str(t) for t in val]

        if "metadata" in updates:
            val = updates["metadata"]
            if not isinstance(val, dict):
                raise EvolutionError("'metadata' must be a dictionary.")
            identity.metadata.update(val)

        identity.bump_version()


def _serialise(value: Any) -> Any:
    """Convert a value to a JSON-compatible form for snapshots."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value