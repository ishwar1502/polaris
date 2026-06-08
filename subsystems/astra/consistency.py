# subsystems/astra/consistency.py
"""
ASTRA v5 Consistency Engine.

The Consistency Engine prevents identity drift by analysing whether observed
signals represent genuine, lasting changes to the user's identity — or merely
temporary fluctuations (frustration, context shifts, mood states).

Core Law: Short-term events ≠ Identity Change.

The engine accumulates evidence and computes a stability score.  Drift is
flagged when multiple independent signals suggest a meaningful change, but
the engine itself does NOT update the identity.  That is the sole
responsibility of the :class:`~subsystems.astra.evolution.IdentityEvolutionEngine`.

All operations are thread-safe via an internal :class:`threading.RLock`.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from subsystems.astra.models import (
    ConsistencyReport,
    DigitalTwinState,
    EvidenceItem,
    Goal,
    GoalState,
    IdentityProfile,
    PreferenceProfile,
)

_logger = logging.getLogger(__name__)

# Thresholds
_STABILITY_THRESHOLD: float = 0.6
_HIGH_EVIDENCE_WEIGHT: float = 0.7
_MAX_EVIDENCE_BUFFER: int = 200


class ConsistencyEngine:
    """Analyses ASTRA state dimensions for identity drift and stability.

    The engine maintains a rolling evidence buffer and computes per-dimension
    and overall consistency scores on demand.

    Parameters
    ----------
    stability_threshold:
        Minimum score below which drift is flagged.  Defaults to 0.6.
    """

    def __init__(self, stability_threshold: float = _STABILITY_THRESHOLD) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._evidence_buffer: list[EvidenceItem] = []
        self._stability_threshold = stability_threshold
        self._last_report: ConsistencyReport | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_evidence(
        self,
        *,
        source: str,
        signal_type: str,
        weight: float,
        description: str,
    ) -> None:
        """Append an evidence item to the rolling buffer.

        Parameters
        ----------
        source:
            Subsystem or component providing the evidence.
        signal_type:
            Classification (e.g. ``"behavior_observation"``, ``"event_pattern"``).
        weight:
            0.0-1.0 significance weight.
        description:
            Human-readable description of the observation.
        """
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Evidence weight must be 0.0-1.0, got {weight}.")

        item = EvidenceItem(
            source=source,
            signal_type=signal_type,
            weight=weight,
            description=description,
        )
        with self._lock:
            self._evidence_buffer.append(item)
            # Trim buffer to max size (FIFO)
            if len(self._evidence_buffer) > _MAX_EVIDENCE_BUFFER:
                self._evidence_buffer = self._evidence_buffer[-_MAX_EVIDENCE_BUFFER:]

    def run_consistency_check(
        self,
        *,
        identity: IdentityProfile | None,
        goals: list[Goal],
        preferences: PreferenceProfile,
        external_signals: list[dict[str, Any]] | None = None,
    ) -> ConsistencyReport:
        """Perform a full consistency analysis.

        This method evaluates the current ASTRA state dimensions against the
        accumulated evidence buffer to detect potential identity drift.

        Parameters
        ----------
        identity:
            Current identity profile (may be None if not yet created).
        goals:
            Current goal list.
        preferences:
            Current preference profile.
        external_signals:
            Optional additional signals from external sources (e.g. AURORA
            reporting emotional volatility).  Each dict must have ``source``,
            ``signal_type``, ``weight``, and ``description`` keys.

        Returns
        -------
        ConsistencyReport
            Full analysis report.
        """
        with self._lock:
            evidence_snapshot = list(self._evidence_buffer)

        # Incorporate any external signals (without persisting them)
        if external_signals:
            for sig in external_signals:
                evidence_snapshot.append(
                    EvidenceItem(
                        source=sig.get("source", "external"),
                        signal_type=sig.get("signal_type", "unknown"),
                        weight=float(sig.get("weight", 0.5)),
                        description=sig.get("description", ""),
                    )
                )

        # Per-dimension analysis
        dimension_scores: dict[str, float] = {}
        drift_fields: list[str] = []

        dimension_scores["identity"] = self._analyse_identity_stability(
            identity, evidence_snapshot, drift_fields
        )
        dimension_scores["goals"] = self._analyse_goal_stability(
            goals, evidence_snapshot, drift_fields
        )
        dimension_scores["preferences"] = self._analyse_preference_stability(
            preferences, evidence_snapshot, drift_fields
        )

        # Compute weighted overall score
        weights = {"identity": 0.5, "goals": 0.3, "preferences": 0.2}
        overall_score = sum(
            dimension_scores[dim] * weights.get(dim, 0.0)
            for dim in dimension_scores
        )

        drift_detected = (
            overall_score < self._stability_threshold or len(drift_fields) > 0
        )

        if drift_detected:
            recommendation = (
                f"Identity drift detected in: {', '.join(drift_fields) if drift_fields else 'overall'}. "
                "Consider running Identity Evolution Engine if evidence supports genuine change."
            )
        else:
            recommendation = (
                f"Identity is stable (score={overall_score:.2f}). No action required."
            )

        report = ConsistencyReport(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            drift_detected=drift_detected,
            drift_fields=drift_fields,
            evidence_items=evidence_snapshot,
            recommendation=recommendation,
        )

        with self._lock:
            self._last_report = report

        _logger.debug(
            "Consistency check completed: score=%.2f drift=%s fields=%s",
            overall_score,
            drift_detected,
            drift_fields,
        )
        return report

    def get_last_report(self) -> ConsistencyReport | None:
        """Return the most recent consistency report, or None.

        Returns
        -------
        ConsistencyReport | None
        """
        with self._lock:
            return self._last_report

    def get_evidence_count(self) -> int:
        """Return the current number of items in the evidence buffer."""
        with self._lock:
            return len(self._evidence_buffer)

    def clear_evidence(self) -> int:
        """Clear the evidence buffer and return the number of items removed."""
        with self._lock:
            count = len(self._evidence_buffer)
            self._evidence_buffer.clear()
            return count

    # ------------------------------------------------------------------
    # Private analysis helpers
    # ------------------------------------------------------------------

    def _analyse_identity_stability(
        self,
        identity: IdentityProfile | None,
        evidence: list[EvidenceItem],
        drift_fields: list[str],
    ) -> float:
        """Compute a stability score for the identity dimension.

        Strategy: count high-weight evidence items that signal identity
        instability.  A high concentration of such signals reduces the score.

        Returns float: 0.0-1.0 stability score.
        """
        if identity is None:
            # No identity yet — not instability, just absence
            return 1.0

        identity_signals = [
            e for e in evidence
            if "identity" in e.signal_type or "self_model" in e.signal_type
        ]

        if not identity_signals:
            return 1.0

        destabilising = [e for e in identity_signals if e.weight >= _HIGH_EVIDENCE_WEIGHT]
        ratio = len(destabilising) / max(len(identity_signals), 1)

        if ratio > 0.5:
            drift_fields.append("identity")
            return max(0.0, 1.0 - ratio)

        return max(0.6, 1.0 - (ratio * 0.5))

    def _analyse_goal_stability(
        self,
        goals: list[Goal],
        evidence: list[EvidenceItem],
        drift_fields: list[str],
    ) -> float:
        """Compute a stability score for the goals dimension.

        Strategy: check for rapid goal abandonment pattern or significant
        goal-direction signals in evidence.

        Returns float: 0.0-1.0 stability score.
        """
        if not goals:
            return 1.0

        # Check for goal volatility: many recent abandonments
        abandoned = [g for g in goals if g.state == GoalState.ABANDONED]
        total = len(goals)
        abandonment_ratio = len(abandoned) / total if total > 0 else 0.0

        goal_signals = [
            e for e in evidence
            if "goal" in e.signal_type or "direction" in e.signal_type
        ]
        high_weight_goal_signals = [e for e in goal_signals if e.weight >= _HIGH_EVIDENCE_WEIGHT]

        signal_pressure = len(high_weight_goal_signals) / max(len(goal_signals) + 1, 1)

        score = 1.0 - (abandonment_ratio * 0.4) - (signal_pressure * 0.3)
        score = max(0.0, min(1.0, score))

        if score < self._stability_threshold:
            drift_fields.append("goals")

        return score

    def _analyse_preference_stability(
        self,
        preferences: PreferenceProfile,
        evidence: list[EvidenceItem],
        drift_fields: list[str],
    ) -> float:
        """Compute a stability score for the preferences dimension.

        Strategy: scan evidence for high-frequency preference contradiction
        signals.

        Returns float: 0.0-1.0 stability score.
        """
        pref_signals = [
            e for e in evidence
            if "preference" in e.signal_type or "tendency" in e.signal_type
        ]

        if not pref_signals:
            return 1.0

        contradiction_signals = [
            e for e in pref_signals
            if "contradiction" in e.description.lower()
            or "conflict" in e.description.lower()
        ]

        if not contradiction_signals:
            return 1.0

        ratio = len(contradiction_signals) / max(len(pref_signals), 1)
        score = max(0.0, 1.0 - ratio)

        if score < self._stability_threshold:
            drift_fields.append("preferences")

        return score