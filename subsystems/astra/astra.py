# subsystems/astra/astra.py
"""
ASTRA v5 Subsystem.

Digital Twin Core of POLARIS.

ASTRA is the cognitive model of the user — it answers:
  * Who is this person?
  * What matters to them?
  * How do they think?
  * How are they changing?
  * Where are they going?

This module implements :class:`AstraSubsystem`, which wires together every
ASTRA engine under the :class:`~core.contracts.subsystem.SubsystemContract`
lifecycle and exposes a unified public API to the rest of POLARIS.

All public API methods are thread-safe.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from core.contracts.health import HealthReport, HealthStatus
from core.contracts.subsystem import SubsystemContract, SubsystemMetadata
from core.types.identifiers import make_version

from subsystems.astra.capability import CapabilityManager
from subsystems.astra.consistency import ConsistencyEngine
from subsystems.astra.digital_twin import DigitalTwinEngine
from subsystems.astra.events import (
    consistency_check_completed,
    digital_twin_updated,
    goal_created,
    goal_removed,
    goal_updated,
    identity_evolved,
    identity_updated,
    preference_changed,
)
from subsystems.astra.evolution import IdentityEvolutionEngine
from subsystems.astra.exceptions import (
    AstraNotInitializedError,
    GoalNotFoundError,
    IdentityNotFoundError,
)
from subsystems.astra.goals import GoalEngine
from subsystems.astra.identity import IdentityEngine
from subsystems.astra.models import (
    CapabilityEntry,
    CapabilitySnapshot,
    ConsistencyReport,
    DigitalTwinState,
    EvolutionRecord,
    EvolutionTrigger,
    FutureSelfReference,
    Goal,
    GoalPriority,
    GoalState,
    GoalType,
    IdentityProfile,
    PreferenceProfile,
)
from subsystems.astra.preferences import PreferenceEngine

_logger = logging.getLogger(__name__)

# Subsystem identifier
_ASTRA_ID: str = "polaris.astra"
_ASTRA_VERSION: str = "5.0.0"
_ASTRA_NAME: str = "ASTRA"
_ASTRA_DESCRIPTION: str = (
    "Digital Twin Core of POLARIS. Maintains the user's cognitive model: "
    "identity, goals, preferences, capabilities, and growth trajectory."
)


def _make_astra_metadata() -> SubsystemMetadata:
    """Build the static metadata descriptor for ASTRA."""
    return SubsystemMetadata(
        id=_ASTRA_ID,
        name=_ASTRA_NAME,
        version=make_version(_ASTRA_VERSION),
        description=_ASTRA_DESCRIPTION,
        dependencies=frozenset(),
        capabilities=tuple(),
    )


class AstraSubsystem(SubsystemContract):
    """ASTRA v5 Digital Twin Core subsystem.

    Integrates
    ----------
    * :class:`~subsystems.astra.identity.IdentityEngine`
    * :class:`~subsystems.astra.goals.GoalEngine`
    * :class:`~subsystems.astra.preferences.PreferenceEngine`
    * :class:`~subsystems.astra.capability.CapabilityManager`
    * :class:`~subsystems.astra.digital_twin.DigitalTwinEngine`
    * :class:`~subsystems.astra.consistency.ConsistencyEngine`
    * :class:`~subsystems.astra.evolution.IdentityEvolutionEngine`

    Public API
    ----------
    Identity:
        :meth:`get_identity`, :meth:`update_identity`
    Goals:
        :meth:`get_goals`, :meth:`create_goal`, :meth:`update_goal`,
        :meth:`remove_goal`
    Preferences:
        :meth:`get_preferences`, :meth:`update_preferences`
    Capabilities:
        :meth:`register_capability`, :meth:`unregister_capability`
    Digital Twin:
        :meth:`generate_digital_twin`, :meth:`get_digital_twin`
    Consistency:
        :meth:`run_consistency_check`
    Evolution:
        :meth:`evolve_identity`

    Parameters
    ----------
    event_bus:
        Optional event bus for publishing ASTRA events.  If ``None``, events
        are silently dropped.
    memory_gateway:
        Optional memory gateway for state persistence.  If ``None``, all
        state is in-memory only.
    """

    def __init__(
        self,
        *,
        event_bus: Any | None = None,
        memory_gateway: Any | None = None,
    ) -> None:
        super().__init__(_make_astra_metadata())

        self._event_bus = event_bus
        self._memory_gateway = memory_gateway

        # Internal state lock (guards _paused flag only; engines have own locks)
        self._state_lock: threading.RLock = threading.RLock()
        self._paused: bool = False

        # Engines — created at __init__, fully wired in _do_initialize
        self._identity_engine: IdentityEngine = IdentityEngine()
        self._goal_engine: GoalEngine = GoalEngine()
        self._preference_engine: PreferenceEngine = PreferenceEngine()
        self._capability_manager: CapabilityManager = CapabilityManager()
        self._consistency_engine: ConsistencyEngine = ConsistencyEngine()
        self._evolution_engine: IdentityEvolutionEngine = IdentityEvolutionEngine()
        self._twin_engine: DigitalTwinEngine = DigitalTwinEngine(
            subsystem_id=_ASTRA_ID
        )

        # Metrics
        self._metrics: dict[str, int] = {
            "identity_updates": 0,
            "goals_created": 0,
            "goals_removed": 0,
            "preference_updates": 0,
            "twin_generations": 0,
            "consistency_checks": 0,
            "evolution_events": 0,
        }

    # ------------------------------------------------------------------
    # SubsystemContract lifecycle hooks
    # ------------------------------------------------------------------

    def _do_initialize(self) -> None:
        """Wire engines and load any persisted state."""
        _logger.info("ASTRA: initializing engines.")
        # Engines are constructed in __init__; nothing else required for
        # in-memory operation.  Persistence hydration would happen here.

    def _do_start(self) -> None:
        """Mark ASTRA as active."""
        with self._state_lock:
            self._paused = False
        _logger.info("ASTRA: subsystem running.")

    def _do_pause(self) -> None:
        """Suspend new API calls."""
        with self._state_lock:
            self._paused = True
        _logger.info("ASTRA: subsystem paused.")

    def _do_resume(self) -> None:
        """Resume API calls."""
        with self._state_lock:
            self._paused = False
        _logger.info("ASTRA: subsystem resumed.")

    def _do_stop(self) -> None:
        """Release resources."""
        with self._state_lock:
            self._paused = False
        _logger.info("ASTRA: subsystem stopped.")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> HealthReport:
        """Return the current health status of ASTRA."""
        try:
            has_identity = self._identity_engine.has_identity()
            goal_count = self._goal_engine.get_goal_count()
            twin = self._twin_engine.get_current_twin()

            status = HealthStatus.HEALTHY
            message = "ASTRA is operational."
            meta: dict[str, Any] = {
                "has_identity": has_identity,
                "goal_count": goal_count,
                "twin_version": twin.twin_version if twin else 0,
                "metrics": dict(self._metrics),
            }

            if not has_identity:
                status = HealthStatus.DEGRADED
                message = "ASTRA is running but no identity profile has been created."

            return HealthReport(
                status=status,
                message=message,
                metadata=meta,
            )
        except Exception as exc:  # pragma: no cover
            return HealthReport(
                status=HealthStatus.UNHEALTHY,
                message=f"ASTRA health check failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Guard helper
    # ------------------------------------------------------------------

    def _require_running(self, operation: str) -> None:
        """Raise :class:`AstraNotInitializedError` unless subsystem is running."""
        if not self.is_running:
            raise AstraNotInitializedError(operation)

    # ------------------------------------------------------------------
    # Identity API
    # ------------------------------------------------------------------

    def get_identity(self) -> IdentityProfile:
        """Return the current identity profile.

        Returns
        -------
        IdentityProfile
            Current identity profile.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        IdentityNotFoundError
            If no identity profile has been created.
        """
        self._require_running("get_identity")
        return self._identity_engine.get_identity()

    def update_identity(self, updates: dict[str, Any]) -> IdentityProfile:
        """Apply validated updates to the identity profile.

        If no profile exists, one is created from *updates* (``name`` is
        required in that case).  The profile is versioned and an
        :func:`~subsystems.astra.events.identity_updated` event is published.

        Parameters
        ----------
        updates:
            Mapping of field-name → new-value pairs.

        Returns
        -------
        IdentityProfile
            The updated profile.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        IdentityValidationError
            If any update fails validation.
        """
        self._require_running("update_identity")
        changed_fields = self._identity_engine.get_changed_fields(updates)
        profile = self._identity_engine.update_identity(updates)

        with self._state_lock:
            self._metrics["identity_updates"] += 1

        if changed_fields:
            self._publish(
                identity_updated(
                    changed_fields=changed_fields,
                    identity_version=profile.version,
                )
            )
        return profile

    # ------------------------------------------------------------------
    # Goal API
    # ------------------------------------------------------------------

    def get_goals(
        self,
        *,
        state: GoalState | None = None,
        goal_type: GoalType | None = None,
        priority: GoalPriority | None = None,
    ) -> list[Goal]:
        """Return goals, optionally filtered.

        Parameters
        ----------
        state:
            Filter by :class:`~subsystems.astra.models.GoalState`.
        goal_type:
            Filter by :class:`~subsystems.astra.models.GoalType`.
        priority:
            Filter by :class:`~subsystems.astra.models.GoalPriority`.

        Returns
        -------
        list[Goal]
            Matching goals.
        """
        self._require_running("get_goals")
        return self._goal_engine.get_goals(
            state=state, goal_type=goal_type, priority=priority
        )

    def create_goal(self, goal_data: dict[str, Any]) -> Goal:
        """Register a new goal.

        Parameters
        ----------
        goal_data:
            Dictionary containing at minimum ``title`` and ``goal_type``.
            See :class:`~subsystems.astra.models.Goal` for all supported fields.

        Returns
        -------
        Goal
            The newly created goal.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        GoalValidationError
            If required fields are missing or invalid.
        """
        self._require_running("create_goal")
        goal = self._goal_engine.create_goal(goal_data)

        with self._state_lock:
            self._metrics["goals_created"] += 1

        self._publish(
            goal_created(
                goal_id=goal.goal_id,
                title=goal.title,
                goal_type=goal.goal_type.name,
                priority=goal.priority.name,
            )
        )
        return goal

    def update_goal(self, goal_id: str, updates: dict[str, Any]) -> Goal:
        """Update an existing goal.

        Parameters
        ----------
        goal_id:
            UUID of the goal to update.
        updates:
            Mapping of field-name → new-value pairs.

        Returns
        -------
        Goal
            The updated goal.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        GoalNotFoundError
            If the goal_id does not exist.
        GoalStateError
            If an invalid state transition is attempted.
        GoalValidationError
            If the updates contain invalid data.
        """
        self._require_running("update_goal")
        changed_fields = self._goal_engine.get_changed_fields(goal_id, updates)
        goal = self._goal_engine.update_goal(goal_id, updates)

        new_state = goal.state.name if "state" in updates else None
        self._publish(
            goal_updated(
                goal_id=goal_id,
                changed_fields=changed_fields,
                new_state=new_state,
            )
        )
        return goal

    def remove_goal(self, goal_id: str) -> Goal:
        """Permanently remove a goal.

        Parameters
        ----------
        goal_id:
            UUID of the goal to remove.

        Returns
        -------
        Goal
            The removed goal (for audit purposes).

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        GoalNotFoundError
            If the goal_id does not exist.
        """
        self._require_running("remove_goal")
        goal = self._goal_engine.remove_goal(goal_id)

        with self._state_lock:
            self._metrics["goals_removed"] += 1

        self._publish(goal_removed(goal_id=goal_id, title=goal.title))
        return goal

    # ------------------------------------------------------------------
    # Preference API
    # ------------------------------------------------------------------

    def get_preferences(self) -> PreferenceProfile:
        """Return the current preference profile.

        Returns
        -------
        PreferenceProfile
            Current preference profile.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        """
        self._require_running("get_preferences")
        return self._preference_engine.get_preferences()

    def update_preferences(self, updates: dict[str, Any]) -> PreferenceProfile:
        """Apply updates to the preference profile.

        Parameters
        ----------
        updates:
            Nested dict mapping dimension names to field updates.
            Example::

                {
                    "communication": {"preferred_tone": "direct"},
                    "learning": {"depth_over_breadth": True},
                }

        Returns
        -------
        PreferenceProfile
            The updated preference profile.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        PreferenceValidationError
            If any update fails validation.
        """
        self._require_running("update_preferences")
        profile = self._preference_engine.update_preferences(updates)
        changed_dimensions = list(updates.keys())

        with self._state_lock:
            self._metrics["preference_updates"] += 1

        self._publish(
            preference_changed(
                changed_dimensions=changed_dimensions,
                preference_version=profile.version,
            )
        )
        return profile

    # ------------------------------------------------------------------
    # Capability API
    # ------------------------------------------------------------------

    def register_capability(
        self,
        *,
        name: str,
        domain: str,
        confidence: float,
        evidence_count: int = 0,
        notes: str = "",
        category: str = "strength",
    ) -> CapabilityEntry:
        """Register a new user capability.

        Parameters
        ----------
        name:
            Capability label.
        domain:
            Domain category.
        confidence:
            0.0-1.0 assessment confidence.
        evidence_count:
            Number of supporting evidence instances.
        notes:
            Human-readable rationale.
        category:
            ``"strength"`` or ``"growth_area"``.

        Returns
        -------
        CapabilityEntry
            The registered capability entry.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        CapabilityAlreadyExistsError
            If the capability is already registered.
        CapabilityValidationError
            If the data fails validation.
        """
        self._require_running("register_capability")
        entry = CapabilityEntry(
            name=name,
            domain=domain,
            confidence=confidence,
            evidence_count=evidence_count,
            notes=notes,
        )
        self._capability_manager.register_capability(
            entry=entry, category=category
        )
        return entry

    def unregister_capability(self, *, name: str, domain: str) -> CapabilityEntry:
        """Remove a registered capability.

        Parameters
        ----------
        name:
            Capability label.
        domain:
            Domain category.

        Returns
        -------
        CapabilityEntry
            The removed entry.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        CapabilityNotFoundError
            If the capability does not exist.
        """
        self._require_running("unregister_capability")
        return self._capability_manager.unregister_capability(
            name=name, domain=domain
        )

    # ------------------------------------------------------------------
    # Digital Twin API
    # ------------------------------------------------------------------

    def generate_digital_twin(
        self,
        *,
        future_self_refs: list[FutureSelfReference] | None = None,
        growth_indicators: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DigitalTwinState:
        """Generate a fresh digital twin from the current ASTRA state.

        Pulls the latest identity, goals, preferences, capabilities, and
        consistency data, then delegates to the
        :class:`~subsystems.astra.digital_twin.DigitalTwinEngine`.

        Parameters
        ----------
        future_self_refs:
            Optional future-self reference models.
        growth_indicators:
            Optional additional growth indicators to merge.
        metadata:
            Optional twin-level annotations.

        Returns
        -------
        DigitalTwinState
            Freshly generated digital twin.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        DigitalTwinError
            If a required ASTRA dimension is not yet available.
        """
        self._require_running("generate_digital_twin")

        # Gather all ASTRA-owned dimensions
        try:
            identity = self._identity_engine.get_identity()
        except IdentityNotFoundError:
            identity = None  # type: ignore[assignment]

        goals = self._goal_engine.get_goals()
        preferences = self._preference_engine.get_preferences()
        capabilities_snap = self._capability_manager.snapshot()
        last_report = self._consistency_engine.get_last_report()
        evolution_records = self._evolution_engine.get_evolution_history()

        if identity is None:
            # Create a minimal placeholder identity so the twin can still be built
            identity = IdentityProfile(name="Unknown")

        twin = self._twin_engine.generate(
            identity=identity,
            goals=goals,
            preferences=preferences,
            capabilities=capabilities_snap,
            evolution_records=evolution_records,
            consistency_report=last_report,
            future_self_refs=future_self_refs or [],
            growth_indicators=growth_indicators,
            metadata=metadata,
        )

        with self._state_lock:
            self._metrics["twin_generations"] += 1

        self._publish(
            digital_twin_updated(
                twin_version=twin.twin_version,
                consistency_score=twin.consistency_score,
                active_goal_count=len(twin.active_goals),
            )
        )
        return twin

    def get_digital_twin(self) -> DigitalTwinState | None:
        """Return the most recently generated digital twin, or ``None``.

        Returns
        -------
        DigitalTwinState | None
            The current twin, or ``None`` if not yet generated.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        """
        self._require_running("get_digital_twin")
        return self._twin_engine.get_current_twin()

    # ------------------------------------------------------------------
    # Consistency API
    # ------------------------------------------------------------------

    def run_consistency_check(
        self,
        *,
        external_signals: list[dict[str, Any]] | None = None,
    ) -> ConsistencyReport:
        """Run a consistency analysis against the current ASTRA state.

        Parameters
        ----------
        external_signals:
            Optional signals from external subsystems (e.g. AURORA's emotional
            volatility signals).  Each dict must have ``source``, ``signal_type``,
            ``weight``, and ``description`` keys.

        Returns
        -------
        ConsistencyReport
            Full analysis report including per-dimension scores and drift
            detection.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        """
        self._require_running("run_consistency_check")

        try:
            identity = self._identity_engine.get_identity()
        except IdentityNotFoundError:
            identity = None

        goals = self._goal_engine.get_goals()
        preferences = self._preference_engine.get_preferences()

        report = self._consistency_engine.run_consistency_check(
            identity=identity,
            goals=goals,
            preferences=preferences,
            external_signals=external_signals,
        )

        with self._state_lock:
            self._metrics["consistency_checks"] += 1

        self._publish(
            consistency_check_completed(
                overall_score=report.overall_score,
                drift_detected=report.drift_detected,
                drift_fields=report.drift_fields,
            )
        )
        return report

    # ------------------------------------------------------------------
    # Evolution API
    # ------------------------------------------------------------------

    def evolve_identity(
        self,
        updates: dict[str, Any],
        *,
        trigger: EvolutionTrigger = EvolutionTrigger.EVIDENCE_THRESHOLD,
        notes: str = "",
        force: bool = False,
    ) -> tuple[IdentityProfile, EvolutionRecord]:
        """Attempt an evidence-gated identity evolution.

        Parameters
        ----------
        updates:
            Field-name → new-value mapping to apply to the identity profile.
        trigger:
            :class:`~subsystems.astra.models.EvolutionTrigger` classification.
        notes:
            Human-readable rationale.
        force:
            If ``True``, bypass evidence threshold checks.  Only use for
            explicit, user-initiated updates.

        Returns
        -------
        tuple[IdentityProfile, EvolutionRecord]
            The updated identity profile and the immutable evolution log entry.

        Raises
        ------
        AstraNotInitializedError
            If the subsystem is not running.
        IdentityNotFoundError
            If no identity profile exists.
        InsufficientEvidenceError
            If evidence thresholds are not met and ``force=False``.
        EvolutionError
            If the updates contain invalid data.
        """
        self._require_running("evolve_identity")
        identity = self._identity_engine.get_identity()

        updated_profile, record = self._evolution_engine.evolve_identity(
            identity,
            updates,
            trigger=trigger,
            notes=notes,
            force=force,
        )

        # Sync the evolved profile back into the identity engine
        # Use force=True here since the evolution engine already validated
        self._identity_engine.update_identity(
            {
                field: getattr(updated_profile, field)
                for field in record.changed_fields
            }
        )

        with self._state_lock:
            self._metrics["evolution_events"] += 1

        self._publish(
            identity_evolved(
                record_id=record.record_id,
                trigger=record.trigger.name,
                changed_fields=list(record.changed_fields),
                confidence=record.confidence,
                evidence_count=record.evidence_count,
            )
        )
        return updated_profile, record

    # ------------------------------------------------------------------
    # Accessors for engines (for testing / advanced use)
    # ------------------------------------------------------------------

    @property
    def identity_engine(self) -> IdentityEngine:
        """Direct access to the Identity Engine."""
        return self._identity_engine

    @property
    def goal_engine(self) -> GoalEngine:
        """Direct access to the Goal Engine."""
        return self._goal_engine

    @property
    def preference_engine(self) -> PreferenceEngine:
        """Direct access to the Preference Engine."""
        return self._preference_engine

    @property
    def capability_manager(self) -> CapabilityManager:
        """Direct access to the Capability Manager."""
        return self._capability_manager

    @property
    def consistency_engine(self) -> ConsistencyEngine:
        """Direct access to the Consistency Engine."""
        return self._consistency_engine

    @property
    def evolution_engine(self) -> IdentityEvolutionEngine:
        """Direct access to the Identity Evolution Engine."""
        return self._evolution_engine

    @property
    def twin_engine(self) -> DigitalTwinEngine:
        """Direct access to the Digital Twin Engine."""
        return self._twin_engine

    def get_metrics(self) -> dict[str, int]:
        """Return a snapshot of internal operation metrics."""
        with self._state_lock:
            return dict(self._metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish(self, event: Any) -> None:
        """Publish an event to the event bus, swallowing errors."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(event)
        except Exception:  # pragma: no cover
            _logger.warning(
                "ASTRA: failed to publish event %r.",
                getattr(event, "event_type", event),
                exc_info=True,
            )