# subsystems/astra/digital_twin.py
"""
ASTRA v5 Digital Twin Engine.

The Digital Twin Engine is the heart of ASTRA.  It aggregates all ASTRA-owned
dimensions into a single :class:`~subsystems.astra.models.DigitalTwinState`
that represents the user as a living cognitive model.

All operations are thread-safe via an internal :class:`threading.RLock`.
"""

from __future__ import annotations

import copy
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from subsystems.astra.exceptions import DigitalTwinError, IdentityNotFoundError
from subsystems.astra.models import (
    CapabilitySnapshot,
    ConsistencyReport,
    DigitalTwinState,
    EvolutionRecord,
    FutureSelfReference,
    Goal,
    GoalState,
    IdentityProfile,
    PreferenceProfile,
)

_logger = logging.getLogger(__name__)

# Default future-self horizon definitions (years)
_DEFAULT_HORIZONS: tuple[float, ...] = (1.0, 5.0, 10.0)


class DigitalTwinEngine:
    """Aggregates all ASTRA-owned dimensions into the DigitalTwinState.

    The engine is responsible for:

    * **Generating** a fresh :class:`~subsystems.astra.models.DigitalTwinState`
      on demand by pulling from all provided data sources.
    * **Refreshing** the twin when any source dimension changes.
    * **Snapshotting** the current twin for point-in-time audit purposes.
    * **Exporting** the twin as a serialisable dictionary for downstream
      subsystems (ORION, VEGA, ODYSSEY, APOLLO, JANUS, PROMETHEUS, ZENITH).

    All write operations are guarded by an internal :class:`threading.RLock`
    so that concurrent calls from multiple engine threads remain safe.

    Parameters
    ----------
    subsystem_id:
        The owning subsystem's identifier (used in log messages).
    """

    def __init__(self, subsystem_id: str = "polaris.astra") -> None:
        self._lock: threading.RLock = threading.RLock()
        self._subsystem_id = subsystem_id
        self._current_twin: DigitalTwinState | None = None
        self._twin_version: int = 0
        self._snapshot_history: list[DigitalTwinState] = []
        self._max_snapshots: int = 50

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        identity: IdentityProfile,
        goals: list[Goal],
        preferences: PreferenceProfile,
        capabilities: CapabilitySnapshot,
        evolution_records: list[EvolutionRecord] | None = None,
        consistency_report: ConsistencyReport | None = None,
        future_self_refs: list[FutureSelfReference] | None = None,
        growth_indicators: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DigitalTwinState:
        """Generate a new :class:`~subsystems.astra.models.DigitalTwinState`.

        This is the primary entry-point for twin creation.  Every call
        produces a fresh, versioned twin from the supplied dimensions.

        Parameters
        ----------
        identity:
            Current :class:`~subsystems.astra.models.IdentityProfile`.
        goals:
            All goals (all states).  The twin will filter and order them.
        preferences:
            Current :class:`~subsystems.astra.models.PreferenceProfile`.
        capabilities:
            Latest :class:`~subsystems.astra.models.CapabilitySnapshot`.
        evolution_records:
            Optional list of recent evolution records for growth metrics.
        consistency_report:
            Optional most-recent :class:`~subsystems.astra.models.ConsistencyReport`.
        future_self_refs:
            Optional list of :class:`~subsystems.astra.models.FutureSelfReference`
            projections.
        growth_indicators:
            Optional pre-computed growth indicator map.
        metadata:
            Optional annotations to embed in the twin.

        Returns
        -------
        DigitalTwinState
            Newly generated and cached digital twin.

        Raises
        ------
        DigitalTwinError
            If required dimensions are missing or invalid.
        """
        if not isinstance(identity, IdentityProfile):
            raise DigitalTwinError(
                "generate() requires a valid IdentityProfile."
            )
        if not isinstance(preferences, PreferenceProfile):
            raise DigitalTwinError(
                "generate() requires a valid PreferenceProfile."
            )
        if not isinstance(capabilities, CapabilitySnapshot):
            raise DigitalTwinError(
                "generate() requires a valid CapabilitySnapshot."
            )

        with self._lock:
            self._twin_version += 1

            # Derive consistency score from the latest report (default 1.0).
            consistency_score: float = 1.0
            if consistency_report is not None:
                consistency_score = max(
                    0.0, min(1.0, consistency_report.overall_score)
                )

            # Build growth indicators if not provided.
            computed_growth = self._compute_growth_indicators(
                goals=goals,
                evolution_records=evolution_records or [],
                base_indicators=growth_indicators or {},
            )

            # Sort goals: non-terminal first, by priority descending.
            sorted_goals = sorted(
                [g for g in goals if not g.is_terminal()],
                key=lambda g: g.priority.value,
                reverse=True,
            ) + [g for g in goals if g.is_terminal()]

            twin = DigitalTwinState(
                identity=identity,
                goals=sorted_goals,
                preferences=preferences,
                capabilities=capabilities,
                growth_indicators=computed_growth,
                future_self_refs=list(future_self_refs or []),
                consistency_score=consistency_score,
                generated_at=datetime.now(timezone.utc),
                twin_version=self._twin_version,
                metadata=dict(metadata or {}),
            )

            self._current_twin = twin
            _logger.debug(
                "Digital Twin v%d generated for subsystem %r "
                "(goals=%d, consistency=%.2f).",
                self._twin_version,
                self._subsystem_id,
                len(sorted_goals),
                consistency_score,
            )
            return twin

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(
        self,
        *,
        identity: IdentityProfile | None = None,
        goals: list[Goal] | None = None,
        preferences: PreferenceProfile | None = None,
        capabilities: CapabilitySnapshot | None = None,
        evolution_records: list[EvolutionRecord] | None = None,
        consistency_report: ConsistencyReport | None = None,
        future_self_refs: list[FutureSelfReference] | None = None,
        growth_indicators: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DigitalTwinState:
        """Refresh the current twin by merging updated dimensions.

        Only the dimensions supplied as non-``None`` arguments are updated.
        All other dimensions are carried over from the most recently generated
        twin.  A new :class:`~subsystems.astra.models.DigitalTwinState` is
        always produced — twins are immutable once generated.

        Parameters
        ----------
        identity:
            Updated identity profile, or ``None`` to keep existing.
        goals:
            Updated goal list, or ``None`` to keep existing.
        preferences:
            Updated preference profile, or ``None`` to keep existing.
        capabilities:
            Updated capability snapshot, or ``None`` to keep existing.
        evolution_records:
            Updated evolution records, or ``None`` to keep existing.
        consistency_report:
            Updated consistency report, or ``None`` to keep existing.
        future_self_refs:
            Updated future-self references, or ``None`` to keep existing.
        growth_indicators:
            Updated growth indicators, or ``None`` to recompute.
        metadata:
            Additional metadata to merge, or ``None`` for no change.

        Returns
        -------
        DigitalTwinState
            Newly generated twin with merged dimensions.

        Raises
        ------
        DigitalTwinError
            If no twin has been generated yet (call :meth:`generate` first).
        """
        with self._lock:
            if self._current_twin is None:
                raise DigitalTwinError(
                    "Cannot refresh: no digital twin has been generated yet. "
                    "Call generate() first."
                )
            current = self._current_twin

            merged_metadata = dict(current.metadata)
            if metadata:
                merged_metadata.update(metadata)

            return self.generate(
                identity=identity if identity is not None else current.identity,
                goals=goals if goals is not None else list(current.goals),
                preferences=(
                    preferences
                    if preferences is not None
                    else current.preferences
                ),
                capabilities=(
                    capabilities
                    if capabilities is not None
                    else current.capabilities
                ),
                evolution_records=evolution_records,
                consistency_report=consistency_report,
                future_self_refs=(
                    future_self_refs
                    if future_self_refs is not None
                    else list(current.future_self_refs)
                ),
                growth_indicators=(
                    growth_indicators
                    if growth_indicators is not None
                    else dict(current.growth_indicators)
                ),
                metadata=merged_metadata,
            )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> DigitalTwinState:
        """Capture a point-in-time snapshot of the current twin.

        The snapshot is appended to an internal history buffer (capped at
        :attr:`_max_snapshots`) and a deep copy is returned to the caller.

        Returns
        -------
        DigitalTwinState
            Deep copy of the current twin.

        Raises
        ------
        DigitalTwinError
            If no twin has been generated yet.
        """
        with self._lock:
            if self._current_twin is None:
                raise DigitalTwinError(
                    "Cannot snapshot: no digital twin has been generated yet."
                )
            snap = copy.deepcopy(self._current_twin)
            self._snapshot_history.append(snap)
            # Trim history to max
            if len(self._snapshot_history) > self._max_snapshots:
                self._snapshot_history = self._snapshot_history[
                    -self._max_snapshots :
                ]
            _logger.debug(
                "Twin snapshot taken (v%d, history=%d).",
                snap.twin_version,
                len(self._snapshot_history),
            )
            return snap

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """Export the current twin as a serialisable dictionary.

        Consuming subsystems (ORION, VEGA, ODYSSEY, APOLLO, JANUS,
        PROMETHEUS, ZENITH) call this method to obtain a portable
        representation of the current self-model.

        Returns
        -------
        dict[str, Any]
            Serialisable representation of the current twin.

        Raises
        ------
        DigitalTwinError
            If no twin has been generated yet.
        """
        with self._lock:
            if self._current_twin is None:
                raise DigitalTwinError(
                    "Cannot export: no digital twin has been generated yet."
                )
            twin = self._current_twin
            identity = twin.identity
            prefs = twin.preferences

            return {
                "twin_version": twin.twin_version,
                "generated_at": twin.generated_at.isoformat(),
                "consistency_score": twin.consistency_score,
                "is_consistent": twin.is_consistent,
                "identity": {
                    "name": identity.name,
                    "background": identity.background,
                    "education": identity.education,
                    "career_direction": identity.career_direction,
                    "interests": list(identity.interests),
                    "core_identity_tags": list(identity.core_identity_tags),
                    "version": identity.version,
                    "updated_at": identity.updated_at.isoformat(),
                },
                "goals": [
                    {
                        "goal_id": g.goal_id,
                        "title": g.title,
                        "goal_type": g.goal_type.name,
                        "state": g.state.name,
                        "priority": g.priority.name,
                        "description": g.description,
                        "motivation": g.motivation,
                        "progress_pct": g.progress_pct,
                        "tags": list(g.tags),
                        "parent_goal_id": g.parent_goal_id,
                        "created_at": g.created_at.isoformat(),
                        "updated_at": g.updated_at.isoformat(),
                        "target_date": (
                            g.target_date.isoformat() if g.target_date else None
                        ),
                        "completed_at": (
                            g.completed_at.isoformat() if g.completed_at else None
                        ),
                    }
                    for g in twin.goals
                ],
                "active_goal_count": len(twin.active_goals),
                "preferences": {
                    "communication": {
                        "preferred_detail_level": prefs.communication.preferred_detail_level,
                        "prefers_examples": prefs.communication.prefers_examples,
                        "prefers_analogies": prefs.communication.prefers_analogies,
                        "preferred_tone": prefs.communication.preferred_tone,
                        "structured_output_preferred": prefs.communication.structured_output_preferred,
                    },
                    "learning": {
                        "preferred_approach": prefs.learning.preferred_approach,
                        "prefers_big_picture_first": prefs.learning.prefers_big_picture_first,
                        "learns_by_building": prefs.learning.learns_by_building,
                        "preferred_learning_pace": prefs.learning.preferred_learning_pace,
                        "depth_over_breadth": prefs.learning.depth_over_breadth,
                    },
                    "development": {
                        "preferred_languages": list(prefs.development.preferred_languages),
                        "architecture_first": prefs.development.architecture_first,
                        "prefers_modularity": prefs.development.prefers_modularity,
                        "testing_discipline": prefs.development.testing_discipline,
                        "documentation_style": prefs.development.documentation_style,
                    },
                    "workflow": {
                        "deep_work_sessions": prefs.workflow.deep_work_sessions,
                        "preferred_session_length_hours": prefs.workflow.preferred_session_length_hours,
                        "async_communication_preferred": prefs.workflow.async_communication_preferred,
                        "batch_decisions": prefs.workflow.batch_decisions,
                        "peak_hours": list(prefs.workflow.peak_hours),
                    },
                    "version": prefs.version,
                },
                "capabilities": {
                    "strengths": [
                        {
                            "name": e.name,
                            "domain": e.domain,
                            "confidence": e.confidence,
                            "evidence_count": e.evidence_count,
                            "notes": e.notes,
                        }
                        for e in twin.capabilities.strengths
                    ],
                    "growth_areas": [
                        {
                            "name": e.name,
                            "domain": e.domain,
                            "confidence": e.confidence,
                            "evidence_count": e.evidence_count,
                            "notes": e.notes,
                        }
                        for e in twin.capabilities.growth_areas
                    ],
                    "snapshot_at": twin.capabilities.snapshot_at.isoformat(),
                },
                "growth_indicators": dict(twin.growth_indicators),
                "future_self_refs": [
                    {
                        "horizon_label": ref.horizon_label,
                        "horizon_years": ref.horizon_years,
                        "headline_identity": ref.headline_identity,
                        "projected_goals": list(ref.projected_goals),
                        "confidence": ref.confidence,
                        "last_updated": ref.last_updated.isoformat(),
                    }
                    for ref in twin.future_self_refs
                ],
                "metadata": dict(twin.metadata),
            }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_current_twin(self) -> DigitalTwinState | None:
        """Return the most recently generated twin, or ``None``."""
        with self._lock:
            return self._current_twin

    def get_twin_version(self) -> int:
        """Return the current twin generation counter."""
        with self._lock:
            return self._twin_version

    def get_snapshot_history(self) -> list[DigitalTwinState]:
        """Return a copy of the snapshot history list."""
        with self._lock:
            return list(self._snapshot_history)

    def clear_snapshot_history(self) -> None:
        """Clear all retained snapshots (does not affect current twin)."""
        with self._lock:
            self._snapshot_history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_growth_indicators(
        self,
        *,
        goals: list[Goal],
        evolution_records: list[EvolutionRecord],
        base_indicators: dict[str, Any],
    ) -> dict[str, Any]:
        """Derive growth indicators from available ASTRA dimensions.

        Parameters
        ----------
        goals:
            Full goal list.
        evolution_records:
            Evolution history records.
        base_indicators:
            Caller-supplied base indicators to merge into.

        Returns
        -------
        dict[str, Any]
            Merged and enriched growth indicators map.
        """
        indicators: dict[str, Any] = dict(base_indicators)

        # Aggregate goal counts by state
        state_counts: dict[str, int] = {
            state.name: 0 for state in GoalState
        }
        for goal in goals:
            state_counts[goal.state.name] += 1

        indicators.setdefault("goals_total", len(goals))
        indicators.setdefault("goals_active", state_counts.get("ACTIVE", 0))
        indicators.setdefault(
            "goals_completed", state_counts.get("COMPLETED", 0)
        )
        indicators.setdefault(
            "goals_abandoned", state_counts.get("ABANDONED", 0)
        )

        # Evolution depth
        indicators.setdefault(
            "identity_evolution_count", len(evolution_records)
        )

        # Average goal progress (active goals only)
        active_goals = [g for g in goals if g.state == GoalState.ACTIVE]
        if active_goals:
            avg_progress = sum(g.progress_pct for g in active_goals) / len(
                active_goals
            )
            indicators.setdefault("avg_active_goal_progress", round(avg_progress, 1))
        else:
            indicators.setdefault("avg_active_goal_progress", 0.0)

        return indicators