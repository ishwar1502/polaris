"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/timeline_projection.py

Implementation of the Timeline Projection Engine.

Responsibility: Project future states and milestones across time horizons.

Constitutional rule:
    CHRONOS owns time.
    JANUS predicts future states across time.
    This engine never manages, stores, or modifies time data.

JANUS Law 6:
    All forecasts require uncertainty. No certainty claims. Ever.

JANUS Law 7:
    JANUS cannot create plans. ZENITH owns planning.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .models import (
    ConfidenceProfile,
    EvidenceProfile,
    ForecastHorizon,
    ProbabilityLevel,
    ProjectionMilestone,
    ProjectionStatus,
    TimelineProjection,
    UncertaintyProfile,
)
from .interfaces import ITimelineProjectionEngine
from .schemas import (
    TimelineProjectionCreateRequest,
    TimelineProjectionCreateResponse,
    TimelineProjectionGetRequest,
    TimelineProjectionGetResponse,
    TimelineProjectionUpdateRequest,
    TimelineProjectionUpdateResponse,
    TimelineProjectionCompletionQuery,
    TimelineProjectionCompletionResult,
)
from .exceptions import (
    JanusNotInitializedError,
    JanusAlreadyInitializedError,
    JanusShutdownError,
    JanusMissingRequiredFieldError,
    JanusInvalidProbabilityError,
    JanusTimelineProjectionNotFoundError,
    JanusTimelineProjectionConstructionError,
    JanusProjectionMilestoneError,
    JanusProjectionStatusTransitionError,
    JanusChronosOwnershipViolationError,
    JanusMissingUncertaintyError,
)


# ---------------------------------------------------------------------------
# Engine Identity
# ---------------------------------------------------------------------------

_ENGINE_NAME: str = "TimelineProjectionEngine"
_ENGINE_VERSION: str = "5.1.0"


# ---------------------------------------------------------------------------
# Legal Projection Status Transitions
# ---------------------------------------------------------------------------

_LEGAL_TRANSITIONS: dict[ProjectionStatus, frozenset[ProjectionStatus]] = {
    ProjectionStatus.DRAFT: frozenset(
        {ProjectionStatus.ACTIVE, ProjectionStatus.REVISED, ProjectionStatus.EXPIRED, ProjectionStatus.SUPERSEDED}
    ),
    ProjectionStatus.ACTIVE: frozenset(
        {ProjectionStatus.REVISED, ProjectionStatus.EXPIRED, ProjectionStatus.SUPERSEDED}
    ),
    ProjectionStatus.REVISED: frozenset(
        {ProjectionStatus.ACTIVE, ProjectionStatus.EXPIRED, ProjectionStatus.SUPERSEDED}
    ),
    ProjectionStatus.EXPIRED: frozenset(),
    ProjectionStatus.SUPERSEDED: frozenset(),
}


# ---------------------------------------------------------------------------
# Diagnostics / Health / Statistics Value Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineProjectionHealthReport:
    """Health snapshot for the Timeline Projection Engine."""

    engine_name: str
    engine_version: str
    is_initialized: bool
    is_shutdown: bool
    total_projections: int
    active_projections: int
    draft_projections: int
    revised_projections: int
    expired_projections: int
    superseded_projections: int
    healthy: bool
    generated_at: datetime


@dataclass(frozen=True)
class TimelineProjectionDiagnosticsReport:
    """Diagnostics snapshot covering integrity of stored projections."""

    engine_name: str
    engine_version: str
    total_projections: int
    projections_with_critical_milestones: int
    projections_missing_evidence: int
    projections_with_zero_uncertainty: int
    average_milestone_count: float
    average_completion_probability: float
    horizon_distribution: dict[str, int]
    status_distribution: dict[str, int]
    generated_at: datetime


@dataclass(frozen=True)
class TimelineProjectionStatistics:
    """Aggregate statistics across all registered TimelineProjections."""

    total_projections: int
    by_status: dict[str, int]
    by_horizon: dict[str, int]
    total_milestones: int
    total_critical_milestones: int
    average_milestone_probability: Optional[float]
    average_completion_probability: Optional[float]
    average_confidence: Optional[float]
    average_uncertainty_volatility: Optional[float]
    generated_at: datetime

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class TimelineProjectionComparisonResult:
    """Structured comparison between two TimelineProjections."""

    projection_id_a: str
    projection_id_b: str
    horizon_a: str
    horizon_b: str
    milestone_count_a: int
    milestone_count_b: int
    completion_probability_a: float
    completion_probability_b: float
    completion_probability_delta: float
    confidence_a: float
    confidence_b: float
    confidence_delta: float
    shared_milestone_labels: tuple[str, ...]
    unique_to_a: tuple[str, ...]
    unique_to_b: tuple[str, ...]
    compared_at: datetime


@dataclass(frozen=True)
class HorizonAnalysisResult:
    """Analysis of all projections falling within a given ForecastHorizon."""

    horizon: str
    projection_count: int
    total_milestones: int
    critical_milestone_count: int
    average_milestone_probability: Optional[float]
    average_completion_probability: Optional[float]
    most_uncertain_projection_id: Optional[str]
    least_uncertain_projection_id: Optional[str]
    analyzed_at: datetime


@dataclass(frozen=True)
class MilestoneProjectionResult:
    """Result of projecting milestones for a given TimelineProjection."""

    projection_id: str
    milestones: tuple[ProjectionMilestone, ...]
    critical_milestones: tuple[ProjectionMilestone, ...]
    earliest_milestone_label: Optional[str]
    latest_milestone_label: Optional[str]
    overall_probability_level: ProbabilityLevel
    generated_at: datetime


@dataclass(frozen=True)
class TemporalDependencyAnalysisResult:
    """Result of analyzing inter-milestone dependency structure."""

    projection_id: str
    total_milestones: int
    milestones_with_dependencies: int
    unresolved_dependency_ids: tuple[str, ...]
    dependency_chains: tuple[tuple[str, ...], ...]
    has_circular_dependency: bool
    max_chain_depth: int
    analyzed_at: datetime


# ---------------------------------------------------------------------------
# Timeline Projection Engine
# ---------------------------------------------------------------------------


class TimelineProjectionEngine(ITimelineProjectionEngine):
    """
    Production implementation of the Timeline Projection Engine.

    Owns:
        - TimelineProjection creation, retrieval, update, and lifecycle.
        - Milestone projection and temporal dependency analysis.
        - Horizon-based analysis and projection comparison.
        - Completion-probability computation from critical milestones.

    Never owns:
        - Time itself (CHRONOS owns time).
        - Plans (ZENITH owns planning).
        - Decisions (VEGA owns decisions).

    Thread-safety:
        All mutable state is guarded by a single re-entrant lock. Stored
        TimelineProjection and ProjectionMilestone instances are treated as
        immutable (frozen dataclasses / dataclasses replaced wholesale on
        update, never mutated in place).
    """

    _RESTRICTED_OPERATIONS: frozenset[str] = frozenset(
        {
            "set_current_time",
            "advance_clock",
            "schedule_event",
            "modify_calendar",
            "set_system_time",
            "manage_timeline_clock",
        }
    )

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._initialized: bool = False
        self._shutdown: bool = False
        self._projections: dict[str, TimelineProjection] = {}

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._shutdown:
                raise JanusShutdownError(_ENGINE_NAME)
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._projections = {}
            self._initialized = True

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                raise JanusNotInitializedError(_ENGINE_NAME)
            if self._shutdown:
                return
            self._shutdown = True

    @property
    def is_initialized(self) -> bool:
        with self._lock:
            return self._initialized and not self._shutdown

    @property
    def engine_name(self) -> str:
        return _ENGINE_NAME

    @property
    def engine_version(self) -> str:
        return _ENGINE_VERSION

    def _ensure_operational(self) -> None:
        if self._shutdown:
            raise JanusShutdownError(_ENGINE_NAME)
        if not self._initialized:
            raise JanusNotInitializedError(_ENGINE_NAME)

    # -----------------------------------------------------------------
    # Constitutional Boundary Guard
    # -----------------------------------------------------------------

    def _guard_chronos_boundary(self, operation: str) -> None:
        """
        Raise JanusChronosOwnershipViolationError if the requested
        operation would imply management of time itself.

        CHRONOS owns time. JANUS predicts future states across time.
        """
        if operation.lower() in self._RESTRICTED_OPERATIONS:
            raise JanusChronosOwnershipViolationError(operation)

    # -----------------------------------------------------------------
    # Validation Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_milestones(
        milestones: tuple[ProjectionMilestone, ...]
    ) -> None:
        if not milestones:
            raise JanusMissingRequiredFieldError(
                "milestones", engine=_ENGINE_NAME
            )

        known_ids = {m.milestone_id for m in milestones}

        for milestone in milestones:
            if not 0.0 <= milestone.probability <= 1.0:
                raise JanusInvalidProbabilityError(
                    milestone.probability,
                    field=f"milestones[{milestone.milestone_id}].probability",
                    engine=_ENGINE_NAME,
                )
            if not milestone.label.strip():
                raise JanusProjectionMilestoneError(
                    milestone.milestone_id,
                    "label must be a non-empty string.",
                )
            for dep_id in milestone.dependencies:
                if dep_id == milestone.milestone_id:
                    raise JanusProjectionMilestoneError(
                        milestone.milestone_id,
                        "milestone cannot depend on itself.",
                    )

        # External dependency references are permitted (they may resolve
        # against milestones in other projections), but self-referential
        # cycles within this projection are rejected.
        for milestone in milestones:
            internal_deps = [d for d in milestone.dependencies if d in known_ids]
            for dep_id in internal_deps:
                if milestone.milestone_id in _transitive_dependency_closure(
                    dep_id, {m.milestone_id: m for m in milestones}
                ):
                    raise JanusProjectionMilestoneError(
                        milestone.milestone_id,
                        f"circular dependency detected with milestone '{dep_id}'.",
                    )

    @staticmethod
    def _validate_uncertainty(
        projection_id: str, uncertainty: Optional[UncertaintyProfile]
    ) -> None:
        if uncertainty is None:
            raise JanusMissingUncertaintyError(projection_id, "TimelineProjection")

    @staticmethod
    def _validate_evidence(evidence: Optional[EvidenceProfile]) -> None:
        if evidence is None:
            raise JanusMissingRequiredFieldError("evidence", engine=_ENGINE_NAME)

    # -----------------------------------------------------------------
    # Core Interface: Create
    # -----------------------------------------------------------------

    def create_projection(
        self, request: TimelineProjectionCreateRequest
    ) -> TimelineProjectionCreateResponse:
        with self._lock:
            self._ensure_operational()
            self._guard_chronos_boundary("create_projection")

            if not request.title.strip():
                raise JanusMissingRequiredFieldError("title", engine=_ENGINE_NAME)
            if not request.description.strip():
                raise JanusMissingRequiredFieldError(
                    "description", engine=_ENGINE_NAME
                )
            if not request.context.strip():
                raise JanusMissingRequiredFieldError("context", engine=_ENGINE_NAME)

            try:
                self._validate_milestones(tuple(request.milestones))
                self._validate_evidence(request.evidence)
            except (
                JanusMissingRequiredFieldError,
                JanusInvalidProbabilityError,
                JanusProjectionMilestoneError,
            ) as exc:
                raise JanusTimelineProjectionConstructionError(
                    str(exc), context={"title": request.title}
                ) from exc

            self._validate_uncertainty("<pending>", request.uncertainty)

            try:
                projection = TimelineProjection.create(
                    title=request.title,
                    description=request.description,
                    context=request.context,
                    milestones=list(request.milestones),
                    horizon=request.horizon,
                    uncertainty=request.uncertainty,
                    confidence=request.confidence,
                    evidence=request.evidence,
                )
            except ValueError as exc:
                raise JanusTimelineProjectionConstructionError(
                    str(exc), context={"title": request.title}
                ) from exc

            self._projections[projection.projection_id] = projection

            return TimelineProjectionCreateResponse(
                projection=projection,
                created_at=projection.created_at,
                engine_version=_ENGINE_VERSION,
            )

    # -----------------------------------------------------------------
    # Core Interface: Retrieve
    # -----------------------------------------------------------------

    def get_projection(
        self, request: TimelineProjectionGetRequest
    ) -> TimelineProjectionGetResponse:
        with self._lock:
            self._ensure_operational()

            projection = self._projections.get(request.projection_id)
            if projection is None:
                raise JanusTimelineProjectionNotFoundError(request.projection_id)

            return TimelineProjectionGetResponse(
                projection=projection,
                retrieved_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Core Interface: Update
    # -----------------------------------------------------------------

    def update_projection(
        self, request: TimelineProjectionUpdateRequest
    ) -> TimelineProjectionUpdateResponse:
        with self._lock:
            self._ensure_operational()
            self._guard_chronos_boundary("update_projection")

            projection = self._projections.get(request.projection_id)
            if projection is None:
                raise JanusTimelineProjectionNotFoundError(request.projection_id)

            try:
                self._validate_milestones(tuple(request.updated_milestones))
                self._validate_evidence(request.updated_evidence)
            except (
                JanusMissingRequiredFieldError,
                JanusInvalidProbabilityError,
                JanusProjectionMilestoneError,
            ) as exc:
                raise JanusTimelineProjectionConstructionError(
                    str(exc), context={"projection_id": request.projection_id}
                ) from exc

            self._validate_uncertainty(
                request.projection_id, request.updated_uncertainty
            )

            previous_status = projection.status
            target_status = ProjectionStatus.REVISED

            if target_status not in _LEGAL_TRANSITIONS.get(previous_status, frozenset()):
                if previous_status != target_status:
                    raise JanusProjectionStatusTransitionError(
                        request.projection_id,
                        previous_status.name,
                        target_status.name,
                    )

            updated = TimelineProjection(
                projection_id=projection.projection_id,
                title=projection.title,
                description=projection.description,
                context=projection.context,
                milestones=list(request.updated_milestones),
                horizon=projection.horizon,
                status=target_status,
                uncertainty=request.updated_uncertainty,
                confidence=projection.confidence,
                evidence=request.updated_evidence,
                created_at=projection.created_at,
                updated_at=_utcnow(),
            )

            self._projections[updated.projection_id] = updated

            return TimelineProjectionUpdateResponse(
                projection_id=updated.projection_id,
                previous_status=previous_status,
                new_status=updated.status,
                updated_at=updated.updated_at,
            )

    # -----------------------------------------------------------------
    # Core Interface: Completion Probability
    # -----------------------------------------------------------------

    def query_completion_probability(
        self, request: TimelineProjectionCompletionQuery
    ) -> TimelineProjectionCompletionResult:
        with self._lock:
            self._ensure_operational()

            projection = self._projections.get(request.projection_id)
            if projection is None:
                raise JanusTimelineProjectionNotFoundError(request.projection_id)

            completion_probability = projection.completion_probability
            probability_level = ProbabilityLevel.from_float(completion_probability)
            critical_count = len(projection.critical_milestones)

            return TimelineProjectionCompletionResult(
                projection_id=projection.projection_id,
                completion_probability=completion_probability,
                probability_level=probability_level,
                critical_milestone_count=critical_count,
                computed_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Core Interface: List by Status
    # -----------------------------------------------------------------

    def list_projections_by_status(
        self, status: ProjectionStatus
    ) -> tuple[TimelineProjection, ...]:
        with self._lock:
            self._ensure_operational()

            return tuple(
                p for p in self._projections.values() if p.status == status
            )

    # -----------------------------------------------------------------
    # Core Interface: Expire
    # -----------------------------------------------------------------

    def expire_projection(
        self, projection_id: str, reason: str
    ) -> TimelineProjection:
        with self._lock:
            self._ensure_operational()
            self._guard_chronos_boundary("expire_projection")

            if not reason.strip():
                raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)

            projection = self._projections.get(projection_id)
            if projection is None:
                raise JanusTimelineProjectionNotFoundError(projection_id)

            target_status = ProjectionStatus.EXPIRED
            if target_status not in _LEGAL_TRANSITIONS.get(
                projection.status, frozenset()
            ):
                raise JanusProjectionStatusTransitionError(
                    projection_id, projection.status.name, target_status.name
                )

            updated = _with_status(projection, target_status, _utcnow())
            self._projections[projection_id] = updated
            return updated

    # -----------------------------------------------------------------
    # Core Interface: Supersede
    # -----------------------------------------------------------------

    def supersede_projection(
        self, projection_id: str, replacement_projection_id: str, reason: str
    ) -> TimelineProjection:
        with self._lock:
            self._ensure_operational()
            self._guard_chronos_boundary("supersede_projection")

            if not reason.strip():
                raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)

            projection = self._projections.get(projection_id)
            if projection is None:
                raise JanusTimelineProjectionNotFoundError(projection_id)

            replacement = self._projections.get(replacement_projection_id)
            if replacement is None:
                raise JanusTimelineProjectionNotFoundError(replacement_projection_id)

            if replacement_projection_id == projection_id:
                raise JanusTimelineProjectionConstructionError(
                    "A TimelineProjection cannot supersede itself.",
                    context={
                        "projection_id": projection_id,
                        "replacement_projection_id": replacement_projection_id,
                    },
                )

            target_status = ProjectionStatus.SUPERSEDED
            if target_status not in _LEGAL_TRANSITIONS.get(
                projection.status, frozenset()
            ):
                raise JanusProjectionStatusTransitionError(
                    projection_id, projection.status.name, target_status.name
                )

            updated = _with_status(projection, target_status, _utcnow())
            self._projections[projection_id] = updated
            return updated

    # -----------------------------------------------------------------
    # Projection Lifecycle Management — Activation
    # -----------------------------------------------------------------

    def activate_projection(self, projection_id: str) -> TimelineProjection:
        """
        Transition a TimelineProjection from DRAFT (or REVISED) to ACTIVE.

        Part of projection lifecycle management. Not part of the abstract
        interface's minimum surface but provided as a production-quality
        lifecycle operation consistent with the legal transition table.
        """
        with self._lock:
            self._ensure_operational()
            self._guard_chronos_boundary("activate_projection")

            projection = self._projections.get(projection_id)
            if projection is None:
                raise JanusTimelineProjectionNotFoundError(projection_id)

            target_status = ProjectionStatus.ACTIVE
            if target_status not in _LEGAL_TRANSITIONS.get(
                projection.status, frozenset()
            ):
                raise JanusProjectionStatusTransitionError(
                    projection_id, projection.status.name, target_status.name
                )

            updated = _with_status(projection, target_status, _utcnow())
            self._projections[projection_id] = updated
            return updated

    # -----------------------------------------------------------------
    # Projection Comparison
    # -----------------------------------------------------------------

    def compare_projections(
        self, projection_id_a: str, projection_id_b: str
    ) -> TimelineProjectionComparisonResult:
        """
        Produce a structured comparison between two TimelineProjections,
        covering milestone overlap, completion probability, and confidence.
        """
        with self._lock:
            self._ensure_operational()

            projection_a = self._projections.get(projection_id_a)
            if projection_a is None:
                raise JanusTimelineProjectionNotFoundError(projection_id_a)

            projection_b = self._projections.get(projection_id_b)
            if projection_b is None:
                raise JanusTimelineProjectionNotFoundError(projection_id_b)

            labels_a = {m.label for m in projection_a.milestones}
            labels_b = {m.label for m in projection_b.milestones}

            shared = tuple(sorted(labels_a & labels_b))
            unique_a = tuple(sorted(labels_a - labels_b))
            unique_b = tuple(sorted(labels_b - labels_a))

            completion_a = projection_a.completion_probability
            completion_b = projection_b.completion_probability

            return TimelineProjectionComparisonResult(
                projection_id_a=projection_a.projection_id,
                projection_id_b=projection_b.projection_id,
                horizon_a=projection_a.horizon.value,
                horizon_b=projection_b.horizon.value,
                milestone_count_a=len(projection_a.milestones),
                milestone_count_b=len(projection_b.milestones),
                completion_probability_a=completion_a,
                completion_probability_b=completion_b,
                completion_probability_delta=completion_b - completion_a,
                confidence_a=projection_a.confidence.overall,
                confidence_b=projection_b.confidence.overall,
                confidence_delta=(
                    projection_b.confidence.overall - projection_a.confidence.overall
                ),
                shared_milestone_labels=shared,
                unique_to_a=unique_a,
                unique_to_b=unique_b,
                compared_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Horizon Analysis
    # -----------------------------------------------------------------

    def analyze_horizon(self, horizon: ForecastHorizon) -> HorizonAnalysisResult:
        """
        Analyze all TimelineProjections registered at the given
        ForecastHorizon, aggregating milestone and uncertainty statistics.
        """
        with self._lock:
            self._ensure_operational()

            matching = [
                p for p in self._projections.values() if p.horizon == horizon
            ]

            total_milestones = sum(len(p.milestones) for p in matching)
            critical_count = sum(len(p.critical_milestones) for p in matching)

            all_probabilities = [
                m.probability for p in matching for m in p.milestones
            ]
            avg_milestone_probability = (
                statistics.fmean(all_probabilities) if all_probabilities else None
            )

            completion_probs = [p.completion_probability for p in matching]
            avg_completion = (
                statistics.fmean(completion_probs) if completion_probs else None
            )

            most_uncertain_id: Optional[str] = None
            least_uncertain_id: Optional[str] = None
            if matching:
                most_uncertain = max(
                    matching, key=lambda p: p.uncertainty.volatility_score
                )
                least_uncertain = min(
                    matching, key=lambda p: p.uncertainty.volatility_score
                )
                most_uncertain_id = most_uncertain.projection_id
                least_uncertain_id = least_uncertain.projection_id

            return HorizonAnalysisResult(
                horizon=horizon.value,
                projection_count=len(matching),
                total_milestones=total_milestones,
                critical_milestone_count=critical_count,
                average_milestone_probability=avg_milestone_probability,
                average_completion_probability=avg_completion,
                most_uncertain_projection_id=most_uncertain_id,
                least_uncertain_projection_id=least_uncertain_id,
                analyzed_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Milestone Projection
    # -----------------------------------------------------------------

    def project_milestones(self, projection_id: str) -> MilestoneProjectionResult:
        """
        Return the full milestone projection for a TimelineProjection,
        including critical milestones and earliest/latest temporal bounds.

        Note: milestone `projected_at` timestamps are produced by upstream
        engines (e.g., the Future Modeling / Forecasting Engines) and
        consumed here as-is. This engine never sets, advances, or modifies
        clock time — CHRONOS owns time.
        """
        with self._lock:
            self._ensure_operational()

            projection = self._projections.get(projection_id)
            if projection is None:
                raise JanusTimelineProjectionNotFoundError(projection_id)

            milestones = tuple(projection.milestones)
            critical = tuple(projection.critical_milestones)

            earliest_label: Optional[str] = None
            latest_label: Optional[str] = None
            if milestones:
                earliest = min(milestones, key=lambda m: m.projected_at)
                latest = max(milestones, key=lambda m: m.projected_at)
                earliest_label = earliest.label
                latest_label = latest.label

            overall_probability = (
                statistics.fmean(m.probability for m in milestones)
                if milestones
                else 0.0
            )

            return MilestoneProjectionResult(
                projection_id=projection.projection_id,
                milestones=milestones,
                critical_milestones=critical,
                earliest_milestone_label=earliest_label,
                latest_milestone_label=latest_label,
                overall_probability_level=ProbabilityLevel.from_float(
                    overall_probability
                ),
                generated_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Temporal Dependency Analysis
    # -----------------------------------------------------------------

    def analyze_temporal_dependencies(
        self, projection_id: str
    ) -> TemporalDependencyAnalysisResult:
        """
        Analyze the dependency structure among the milestones of a
        TimelineProjection: dependency chains, unresolved external
        dependency references, and circular-dependency detection.
        """
        with self._lock:
            self._ensure_operational()

            projection = self._projections.get(projection_id)
            if projection is None:
                raise JanusTimelineProjectionNotFoundError(projection_id)

            milestones_by_id = {m.milestone_id: m for m in projection.milestones}
            with_deps = [m for m in projection.milestones if m.dependencies]

            unresolved: list[str] = []
            for milestone in projection.milestones:
                for dep_id in milestone.dependencies:
                    if dep_id not in milestones_by_id:
                        unresolved.append(dep_id)

            has_cycle = False
            for milestone in projection.milestones:
                closure = _transitive_dependency_closure(
                    milestone.milestone_id, milestones_by_id
                )
                if milestone.milestone_id in closure:
                    has_cycle = True
                    break

            chains: list[tuple[str, ...]] = []
            max_depth = 0
            for milestone in projection.milestones:
                chain = _build_dependency_chain(milestone, milestones_by_id)
                chains.append(chain)
                max_depth = max(max_depth, len(chain))

            return TemporalDependencyAnalysisResult(
                projection_id=projection.projection_id,
                total_milestones=len(projection.milestones),
                milestones_with_dependencies=len(with_deps),
                unresolved_dependency_ids=tuple(sorted(set(unresolved))),
                dependency_chains=tuple(chains),
                has_circular_dependency=has_cycle,
                max_chain_depth=max_depth,
                analyzed_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def get_statistics(self) -> TimelineProjectionStatistics:
        """Return aggregate statistics across all registered projections."""
        with self._lock:
            self._ensure_operational()

            projections = list(self._projections.values())

            by_status: dict[str, int] = {}
            for status in ProjectionStatus:
                by_status[status.name] = sum(
                    1 for p in projections if p.status == status
                )

            by_horizon: dict[str, int] = {}
            for horizon in ForecastHorizon:
                by_horizon[horizon.value] = sum(
                    1 for p in projections if p.horizon == horizon
                )

            total_milestones = sum(len(p.milestones) for p in projections)
            total_critical = sum(len(p.critical_milestones) for p in projections)

            all_milestone_probs = [
                m.probability for p in projections for m in p.milestones
            ]
            avg_milestone_probability = (
                statistics.fmean(all_milestone_probs)
                if all_milestone_probs
                else None
            )

            completion_probs = [p.completion_probability for p in projections]
            avg_completion = (
                statistics.fmean(completion_probs) if completion_probs else None
            )

            confidences = [p.confidence.overall for p in projections]
            avg_confidence = statistics.fmean(confidences) if confidences else None

            volatilities = [p.uncertainty.volatility_score for p in projections]
            avg_volatility = (
                statistics.fmean(volatilities) if volatilities else None
            )

            return TimelineProjectionStatistics(
                total_projections=len(projections),
                by_status=by_status,
                by_horizon=by_horizon,
                total_milestones=total_milestones,
                total_critical_milestones=total_critical,
                average_milestone_probability=avg_milestone_probability,
                average_completion_probability=avg_completion,
                average_confidence=avg_confidence,
                average_uncertainty_volatility=avg_volatility,
                generated_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Health Report
    # -----------------------------------------------------------------

    def check_health(self) -> TimelineProjectionHealthReport:
        """Return a health snapshot of the engine and its registry."""
        with self._lock:
            projections = list(self._projections.values())

            active = sum(
                1 for p in projections if p.status == ProjectionStatus.ACTIVE
            )
            draft = sum(
                1 for p in projections if p.status == ProjectionStatus.DRAFT
            )
            revised = sum(
                1 for p in projections if p.status == ProjectionStatus.REVISED
            )
            expired = sum(
                1 for p in projections if p.status == ProjectionStatus.EXPIRED
            )
            superseded = sum(
                1 for p in projections if p.status == ProjectionStatus.SUPERSEDED
            )

            healthy = self._initialized and not self._shutdown

            return TimelineProjectionHealthReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                is_initialized=self._initialized,
                is_shutdown=self._shutdown,
                total_projections=len(projections),
                active_projections=active,
                draft_projections=draft,
                revised_projections=revised,
                expired_projections=expired,
                superseded_projections=superseded,
                healthy=healthy,
                generated_at=_utcnow(),
            )

    def get_health(self) -> dict:
        """Return health as a dict compatible with h['is_healthy']."""
        report = self.check_health()
        return {
            "is_healthy": report.healthy,
            "engine_name": report.engine_name,
            "engine_version": report.engine_version,
            "is_initialized": report.is_initialized,
            "is_shutdown": report.is_shutdown,
            "total_projections": report.total_projections,
            "active_projections": report.active_projections,
            "draft_projections": report.draft_projections,
            "revised_projections": report.revised_projections,
            "expired_projections": report.expired_projections,
            "superseded_projections": report.superseded_projections,
            "healthy": report.healthy,
            "generated_at": report.generated_at,
        }

    # -----------------------------------------------------------------
    # Diagnostics Report
    # -----------------------------------------------------------------

    def run_diagnostics(self) -> TimelineProjectionDiagnosticsReport:
        """
        Run a diagnostics pass over all registered projections, flagging
        integrity concerns (missing evidence, zero uncertainty) without
        mutating any state.
        """
        with self._lock:
            self._ensure_operational()

            projections = list(self._projections.values())

            with_critical = sum(
                1 for p in projections if p.critical_milestones
            )

            missing_evidence = sum(
                1
                for p in projections
                if not p.evidence.sources and not p.evidence.patterns_observed
            )

            zero_uncertainty = sum(
                1
                for p in projections
                if p.uncertainty.volatility_score == 0.0
                and p.uncertainty.unknown_risk_exposure == 0.0
            )

            milestone_counts = [len(p.milestones) for p in projections]
            avg_milestone_count = (
                statistics.fmean(milestone_counts) if milestone_counts else 0.0
            )

            completion_probs = [p.completion_probability for p in projections]
            avg_completion = (
                statistics.fmean(completion_probs) if completion_probs else 0.0
            )

            horizon_distribution: dict[str, int] = {}
            for p in projections:
                horizon_distribution[p.horizon.value] = (
                    horizon_distribution.get(p.horizon.value, 0) + 1
                )

            status_distribution: dict[str, int] = {}
            for p in projections:
                status_distribution[p.status.name] = (
                    status_distribution.get(p.status.name, 0) + 1
                )

            return TimelineProjectionDiagnosticsReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                total_projections=len(projections),
                projections_with_critical_milestones=with_critical,
                projections_missing_evidence=missing_evidence,
                projections_with_zero_uncertainty=zero_uncertainty,
                average_milestone_count=avg_milestone_count,
                average_completion_probability=avg_completion,
                horizon_distribution=horizon_distribution,
                status_distribution=status_distribution,
                generated_at=_utcnow(),
            )


# ---------------------------------------------------------------------------
# Module-level Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _with_status(
    projection: TimelineProjection, status: ProjectionStatus, updated_at: datetime
) -> TimelineProjection:
    """Return a copy of the given TimelineProjection with a new status/timestamp."""
    return TimelineProjection(
        projection_id=projection.projection_id,
        title=projection.title,
        description=projection.description,
        context=projection.context,
        milestones=projection.milestones,
        horizon=projection.horizon,
        status=status,
        uncertainty=projection.uncertainty,
        confidence=projection.confidence,
        evidence=projection.evidence,
        created_at=projection.created_at,
        updated_at=updated_at,
    )


def _transitive_dependency_closure(
    milestone_id: str,
    milestones_by_id: dict[str, ProjectionMilestone],
    _visited: Optional[set[str]] = None,
) -> set[str]:
    """
    Return the set of all milestone_ids transitively reachable via
    `dependencies` starting from `milestone_id` (exclusive of itself,
    unless a cycle routes back to it).
    """
    if _visited is None:
        _visited = set()

    milestone = milestones_by_id.get(milestone_id)
    if milestone is None:
        return set()

    closure: set[str] = set()
    for dep_id in milestone.dependencies:
        if dep_id in _visited:
            closure.add(dep_id)
            continue
        if dep_id not in milestones_by_id:
            continue
        closure.add(dep_id)
        new_visited = _visited | {milestone_id}
        closure |= _transitive_dependency_closure(
            dep_id, milestones_by_id, new_visited
        )

    return closure


def _build_dependency_chain(
    milestone: ProjectionMilestone,
    milestones_by_id: dict[str, ProjectionMilestone],
) -> tuple[str, ...]:
    """
    Build an ordered dependency chain (labels) starting from `milestone`,
    following the first dependency at each step. Stops on missing
    dependency, cycle detection, or chain exhaustion.
    """
    chain: list[str] = [milestone.label]
    seen_ids: set[str] = {milestone.milestone_id}
    current = milestone

    while current.dependencies:
        next_id = current.dependencies[0]
        if next_id in seen_ids or next_id not in milestones_by_id:
            break
        current = milestones_by_id[next_id]
        chain.append(current.label)
        seen_ids.add(current.milestone_id)

    return tuple(chain)