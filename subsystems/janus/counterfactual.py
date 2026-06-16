"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/counterfactual.py

Counterfactual Engine — ICounterfactualEngine implementation.

Responsibility:
    Construct alternate-reality scenarios that answer 'What if this decision
    had been different?' Used for learning and strategic evaluation.

    Counterfactual analysis may simulate alternative histories and alternative
    decision paths. It may never modify or mutate any historical record.
    Historical records are owned by ECHO. Historical integrity is inviolable.

Constitutional rule:
    JANUS owns counterfactual analysis.
    ECHO owns historical records.
    VEGA owns decisions.
    ZENITH owns planning.
    ASTRA owns identity modeling.

    This engine explores simulated alternatives only.
    It never writes to, modifies, or claims ownership of historical data.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .exceptions import (
    JanusAlreadyInitializedError,
    JanusConstitutionalViolationError,
    JanusCounterfactualConditionError,
    JanusCounterfactualConstructionError,
    JanusCounterfactualDivergenceError,
    JanusCounterfactualError,
    JanusCounterfactualNotFoundError,
    JanusNotInitializedError,
    JanusShutdownError,
    JanusValidationError,
)
from .interfaces import ICounterfactualEngine
from .models import (
    ConfidenceProfile,
    CounterfactualScenario,
    EvidenceProfile,
    FutureModel,
    UncertaintyProfile,
)

_LOG = logging.getLogger(__name__)

_ENGINE_NAME: str = "CounterfactualEngine"
_ENGINE_VERSION: str = "5.1.0"

# ---------------------------------------------------------------------------
# Constitutional Constants
# ---------------------------------------------------------------------------

# Operations that would imply history mutation — strictly forbidden
_HISTORY_MUTATION_TERMS: frozenset[str] = frozenset(
    {
        "modify_history",
        "update_history",
        "delete_history",
        "write_history",
        "mutate_history",
        "overwrite_history",
        "patch_history",
        "rewrite_history",
        "edit_history",
        "amend_history",
    }
)

# Operations that imply decision selection — owned by VEGA
_SELECTION_TERMS: frozenset[str] = frozenset(
    {"select", "choose", "decide", "approve", "commit"}
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
# schemas.py was not supplied; these dataclasses fulfil the interface contract
# that interfaces.py imports from .schemas. They are defined here as the
# authoritative local implementation types.


@dataclass(frozen=True)
class CounterfactualCreateRequest:
    """
    Request to construct a CounterfactualScenario.

    Fields:
        title                    — Human-readable title for the counterfactual.
        description              — Narrative description.
        original_event           — The actual historical event being re-examined.
        counterfactual_condition — The alternate condition to simulate.
        divergence_point         — The historical moment at which paths diverge.
        resulting_future_model   — The FutureModel produced under the alternate condition.
        delta_risk               — Change in risk relative to original outcome (−1 to +1).
        delta_opportunity        — Change in opportunity relative to original (−1 to +1).
        uncertainty              — UncertaintyProfile for this counterfactual.
        confidence               — ConfidenceProfile for this counterfactual.
        evidence                 — EvidenceProfile supporting this counterfactual.
        learning_insights        — Initial learning insights derived from this alternate path.
        requested_by             — Identity of the requesting subsystem or user.
    """

    title: str
    description: str
    original_event: str
    counterfactual_condition: str
    divergence_point: datetime
    resulting_future_model: FutureModel
    delta_risk: float
    delta_opportunity: float
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    learning_insights: tuple[str, ...]
    requested_by: str


@dataclass(frozen=True)
class CounterfactualCreateResponse:
    """Response from a successful counterfactual creation."""

    counterfactual: CounterfactualScenario
    created_at: datetime
    engine_name: str
    engine_version: str


@dataclass(frozen=True)
class CounterfactualGetRequest:
    """Request to retrieve a CounterfactualScenario by its ID."""

    counterfactual_id: str


@dataclass(frozen=True)
class CounterfactualGetResponse:
    """Response containing the retrieved CounterfactualScenario."""

    counterfactual: CounterfactualScenario
    retrieved_at: datetime
    engine_name: str


@dataclass(frozen=True)
class CounterfactualCompareRequest:
    """
    Request to compare a CounterfactualScenario against a reference.

    Fields:
        counterfactual_id       — The counterfactual to evaluate.
        reference_event_label   — Label identifying the original/baseline event
                                  against which deltas are measured.
        include_learning_synthesis — Whether to synthesise a combined insight summary.
    """

    counterfactual_id: str
    reference_event_label: str
    include_learning_synthesis: bool = True


@dataclass(frozen=True)
class CounterfactualComparisonMetrics:
    """Quantitative metrics from a counterfactual comparison."""

    delta_risk: float
    delta_opportunity: float
    risk_direction: str          # 'improved', 'worsened', 'neutral'
    opportunity_direction: str   # 'improved', 'worsened', 'neutral'
    confidence_overall: float
    evidence_strength: float
    net_delta: float             # (delta_opportunity − delta_risk) composite
    probability_level: str       # ProbabilityLevel.value of the confidence


@dataclass(frozen=True)
class CounterfactualCompareResponse:
    """Response from a counterfactual comparison operation."""

    counterfactual_id: str
    reference_event_label: str
    metrics: CounterfactualComparisonMetrics
    learning_insights: tuple[str, ...]
    synthesised_insight: Optional[str]
    compared_at: datetime
    engine_name: str


# ---------------------------------------------------------------------------
# Statistics & Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualStatistics:
    """Point-in-time statistics snapshot for the Counterfactual Engine."""

    total_counterfactuals: int
    counterfactuals_by_event: dict[str, int]
    average_delta_risk: float
    average_delta_opportunity: float
    average_confidence: float
    average_evidence_strength: float
    total_learning_insights: int
    average_insights_per_counterfactual: float
    generated_at: datetime


@dataclass(frozen=True)
class CounterfactualHealthReport:
    """Health report for the Counterfactual Engine lifecycle."""

    engine_name: str
    engine_version: str
    is_initialized: bool
    is_shut_down: bool
    total_counterfactuals: int
    events_tracked: int
    generated_at: datetime


@dataclass(frozen=True)
class CounterfactualDiagnosticsReport:
    """Detailed diagnostics snapshot for engineering inspection."""

    engine_name: str
    engine_version: str
    is_initialized: bool
    total_counterfactuals: int
    counterfactual_ids: list[str]
    events_tracked: int
    counterfactual_ids_by_event: dict[str, list[str]]
    total_learning_insights: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# Counterfactual Engine Implementation
# ---------------------------------------------------------------------------


class CounterfactualEngine(ICounterfactualEngine):
    """
    Production implementation of ICounterfactualEngine.

    Thread-safe. All public methods acquire the internal RLock.

    Historical Integrity Invariant:
        This engine constructs simulated alternatives. It never mutates,
        writes, or claims ownership of historical records. Any operation
        that implies history mutation raises JanusConstitutionalViolationError.

    Constitutional Invariants:
        - ECHO owns historical records.
        - VEGA owns decisions.
        - JANUS owns counterfactual analysis.
        - Divergence points must be in the past relative to creation time.
        - Counterfactual conditions must be non-self-contradictory (validated
          syntactically; semantic validation is caller responsibility).
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False
        self._shut_down: bool = False

        # Primary store: counterfactual_id → CounterfactualScenario
        self._store: dict[str, CounterfactualScenario] = {}

        # Event index: original_event → list[counterfactual_id]
        self._event_index: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._store.clear()
            self._event_index.clear()
            self._initialized = True
            self._shut_down = False
            _LOG.info(
                "[%s] initialized (version=%s)",
                _ENGINE_NAME,
                _ENGINE_VERSION,
            )

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                raise JanusNotInitializedError(_ENGINE_NAME)
            if self._shut_down:
                return
            count = len(self._store)
            self._store.clear()
            self._event_index.clear()
            self._shut_down = True
            _LOG.info(
                "[%s] shut down (released %d counterfactual records)",
                _ENGINE_NAME,
                count,
            )

    @property
    def is_initialized(self) -> bool:
        return self._initialized and not self._shut_down

    @property
    def engine_name(self) -> str:
        return _ENGINE_NAME

    @property
    def engine_version(self) -> str:
        return _ENGINE_VERSION

    # ------------------------------------------------------------------
    # ICounterfactualEngine — Core Operations
    # ------------------------------------------------------------------

    def create_counterfactual(
        self, request: CounterfactualCreateRequest
    ) -> CounterfactualCreateResponse:
        """
        Construct a CounterfactualScenario from the request.

        Validates all inputs, enforces historical integrity (divergence point
        must not be in the future), enforces condition validity, and registers
        the counterfactual in the engine store.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: required fields are missing or blank.
            JanusCounterfactualDivergenceError: divergence_point is in the future.
            JanusCounterfactualConditionError: counterfactual_condition is invalid.
            JanusCounterfactualConstructionError: CounterfactualScenario creation fails.
        """
        self._assert_operational()
        self._validate_create_request(request)

        with self._lock:
            self._assert_operational()
            try:
                counterfactual = CounterfactualScenario.create(
                    title=request.title,
                    description=request.description,
                    original_event=request.original_event,
                    counterfactual_condition=request.counterfactual_condition,
                    divergence_point=request.divergence_point,
                    resulting_future_model=request.resulting_future_model,
                    delta_risk=request.delta_risk,
                    delta_opportunity=request.delta_opportunity,
                    uncertainty=request.uncertainty,
                    confidence=request.confidence,
                    evidence=request.evidence,
                    learning_insights=list(request.learning_insights),
                )
            except (ValueError, TypeError) as exc:
                raise JanusCounterfactualConstructionError(
                    f"CounterfactualScenario.create failed: {exc}",
                    context={
                        "original_event": request.original_event,
                        "counterfactual_condition": request.counterfactual_condition,
                    },
                ) from exc

            self._store[counterfactual.counterfactual_id] = counterfactual
            self._event_index.setdefault(request.original_event, []).append(
                counterfactual.counterfactual_id
            )

            _LOG.debug(
                "[%s] created counterfactual '%s' for event '%s' "
                "(condition='%s', divergence=%s)",
                _ENGINE_NAME,
                counterfactual.counterfactual_id,
                request.original_event,
                request.counterfactual_condition,
                request.divergence_point.isoformat(),
            )

            now = datetime.utcnow()
            return CounterfactualCreateResponse(
                counterfactual=counterfactual,
                created_at=now,
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
            )

    def get_counterfactual(
        self, request: CounterfactualGetRequest
    ) -> CounterfactualGetResponse:
        """
        Retrieve a CounterfactualScenario by its counterfactual_id.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: counterfactual_id is blank.
            JanusCounterfactualNotFoundError: ID does not exist.
        """
        self._assert_operational()
        if not request.counterfactual_id or not request.counterfactual_id.strip():
            raise JanusValidationError(
                "counterfactual_id must be a non-empty string.",
                field="counterfactual_id",
                value=request.counterfactual_id,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            counterfactual = self._store.get(request.counterfactual_id)
            if counterfactual is None:
                raise JanusCounterfactualNotFoundError(request.counterfactual_id)
            return CounterfactualGetResponse(
                counterfactual=counterfactual,
                retrieved_at=datetime.utcnow(),
                engine_name=_ENGINE_NAME,
            )

    def compare_counterfactual(
        self, request: CounterfactualCompareRequest
    ) -> CounterfactualCompareResponse:
        """
        Compare a CounterfactualScenario against its reference event, producing
        delta metrics and consolidated learning insights.

        Constitutional note: comparison evaluates; it never selects or approves.
        VEGA selects. JANUS evaluates.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: required fields are missing.
            JanusCounterfactualNotFoundError: counterfactual_id does not exist.
        """
        self._assert_operational()
        if not request.counterfactual_id or not request.counterfactual_id.strip():
            raise JanusValidationError(
                "counterfactual_id must be a non-empty string.",
                field="counterfactual_id",
                value=request.counterfactual_id,
                engine=_ENGINE_NAME,
            )
        if (
            not request.reference_event_label
            or not request.reference_event_label.strip()
        ):
            raise JanusValidationError(
                "reference_event_label must be a non-empty string.",
                field="reference_event_label",
                value=request.reference_event_label,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            counterfactual = self._store.get(request.counterfactual_id)
            if counterfactual is None:
                raise JanusCounterfactualNotFoundError(request.counterfactual_id)

            metrics = self._build_comparison_metrics(counterfactual)
            insights = tuple(counterfactual.learning_insights)
            synthesised: Optional[str] = None

            if request.include_learning_synthesis and insights:
                synthesised = self._synthesise_insights(
                    insights=insights,
                    reference_event=request.reference_event_label,
                    counterfactual_condition=counterfactual.counterfactual_condition,
                    metrics=metrics,
                )

            _LOG.debug(
                "[%s] compared counterfactual '%s' against reference '%s'",
                _ENGINE_NAME,
                request.counterfactual_id,
                request.reference_event_label,
            )

            return CounterfactualCompareResponse(
                counterfactual_id=request.counterfactual_id,
                reference_event_label=request.reference_event_label,
                metrics=metrics,
                learning_insights=insights,
                synthesised_insight=synthesised,
                compared_at=datetime.utcnow(),
                engine_name=_ENGINE_NAME,
            )

    def list_counterfactuals_for_event(
        self, original_event: str
    ) -> tuple[CounterfactualScenario, ...]:
        """
        Return all CounterfactualScenarios derived from the given original event.

        Returns an empty tuple if no counterfactuals exist for the event.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: original_event is blank.
        """
        self._assert_operational()
        if not original_event or not original_event.strip():
            raise JanusValidationError(
                "original_event must be a non-empty string.",
                field="original_event",
                value=original_event,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            ids = self._event_index.get(original_event, [])
            result: list[CounterfactualScenario] = []
            for cid in ids:
                cf = self._store.get(cid)
                if cf is not None:
                    result.append(cf)
            return tuple(result)

    def extract_learning_insights(
        self, counterfactual_id: str
    ) -> tuple[str, ...]:
        """
        Return the learning insights recorded on a CounterfactualScenario.

        These insights feed NOVA via ECHO's reflection pipeline.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: counterfactual_id is blank.
            JanusCounterfactualNotFoundError: ID does not exist.
        """
        self._assert_operational()
        if not counterfactual_id or not counterfactual_id.strip():
            raise JanusValidationError(
                "counterfactual_id must be a non-empty string.",
                field="counterfactual_id",
                value=counterfactual_id,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            cf = self._store.get(counterfactual_id)
            if cf is None:
                raise JanusCounterfactualNotFoundError(counterfactual_id)
            return tuple(cf.learning_insights)

    # ------------------------------------------------------------------
    # Extended Operations (beyond interface minimum)
    # ------------------------------------------------------------------

    def list_all_counterfactuals(self) -> tuple[CounterfactualScenario, ...]:
        """
        Return all registered CounterfactualScenarios.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            return tuple(self._store.values())

    def count_counterfactuals_for_event(self, original_event: str) -> int:
        """
        Return the number of counterfactuals registered for a given event.

        Returns 0 if the event is not tracked.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: original_event is blank.
        """
        self._assert_operational()
        if not original_event or not original_event.strip():
            raise JanusValidationError(
                "original_event must be a non-empty string.",
                field="original_event",
                value=original_event,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            return len(self._event_index.get(original_event, []))

    def counterfactual_exists(self, counterfactual_id: str) -> bool:
        """
        Return True if a CounterfactualScenario with the given ID exists.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            return counterfactual_id in self._store

    def list_tracked_events(self) -> tuple[str, ...]:
        """
        Return the unique original event labels currently tracked.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            return tuple(self._event_index.keys())

    def generate_alternative_decision_paths(
        self,
        original_event: str,
        alternative_conditions: tuple[str, ...],
        divergence_point: datetime,
        resulting_future_models: tuple[FutureModel, ...],
        delta_risks: tuple[float, ...],
        delta_opportunities: tuple[float, ...],
        uncertainties: tuple[UncertaintyProfile, ...],
        confidences: tuple[ConfidenceProfile, ...],
        evidences: tuple[EvidenceProfile, ...],
        learning_insights_per_path: tuple[tuple[str, ...], ...],
        requested_by: str,
    ) -> tuple[CounterfactualScenario, ...]:
        """
        Bulk-generate CounterfactualScenarios for multiple alternative
        decision conditions from a single original event.

        All input tuples must be of equal length. Each index corresponds to
        one alternative decision path.

        Historical integrity is enforced: divergence_point must be in the past
        and no historical record is modified.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: mismatched tuple lengths or missing fields.
            JanusCounterfactualDivergenceError: divergence_point in the future.
            JanusCounterfactualConstructionError: any individual creation fails.
        """
        self._assert_operational()
        n = len(alternative_conditions)
        if n == 0:
            raise JanusValidationError(
                "alternative_conditions must contain at least one condition.",
                field="alternative_conditions",
                value=alternative_conditions,
                engine=_ENGINE_NAME,
            )
        _lengths = {
            "resulting_future_models": len(resulting_future_models),
            "delta_risks": len(delta_risks),
            "delta_opportunities": len(delta_opportunities),
            "uncertainties": len(uncertainties),
            "confidences": len(confidences),
            "evidences": len(evidences),
            "learning_insights_per_path": len(learning_insights_per_path),
        }
        mismatched = {k: v for k, v in _lengths.items() if v != n}
        if mismatched:
            raise JanusValidationError(
                f"All input tuples must have the same length as "
                f"alternative_conditions ({n}). "
                f"Mismatched lengths: {mismatched}.",
                field="input_tuples",
                value=mismatched,
                engine=_ENGINE_NAME,
            )

        # Validate divergence point once (shared across all paths)
        self._validate_divergence_point(divergence_point)

        if not original_event or not original_event.strip():
            raise JanusValidationError(
                "original_event must be a non-empty string.",
                field="original_event",
                value=original_event,
                engine=_ENGINE_NAME,
            )

        created: list[CounterfactualScenario] = []
        for i in range(n):
            condition = alternative_conditions[i]
            self._validate_condition(condition)
            request = CounterfactualCreateRequest(
                title=f"Alternative path {i + 1}: {condition[:80]}",
                description=(
                    f"Simulated alternative for event '{original_event}' "
                    f"under condition: {condition}"
                ),
                original_event=original_event,
                counterfactual_condition=condition,
                divergence_point=divergence_point,
                resulting_future_model=resulting_future_models[i],
                delta_risk=delta_risks[i],
                delta_opportunity=delta_opportunities[i],
                uncertainty=uncertainties[i],
                confidence=confidences[i],
                evidence=evidences[i],
                learning_insights=learning_insights_per_path[i],
                requested_by=requested_by,
            )
            response = self.create_counterfactual(request)
            created.append(response.counterfactual)

        return tuple(created)

    def evaluate_counterfactual(
        self,
        counterfactual_id: str,
    ) -> dict[str, Any]:
        """
        Evaluate a CounterfactualScenario against its baseline, returning
        a structured evaluation summary.

        Constitutional note: evaluation informs; it never selects or approves.
        VEGA selects. JANUS evaluates.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusCounterfactualNotFoundError: ID does not exist.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            cf = self._store.get(counterfactual_id)
            if cf is None:
                raise JanusCounterfactualNotFoundError(counterfactual_id)

            metrics = self._build_comparison_metrics(cf)
            net_delta = cf.delta_opportunity - cf.delta_risk
            overall_verdict: str
            if net_delta > 0.10:
                overall_verdict = "net_positive"
            elif net_delta < -0.10:
                overall_verdict = "net_negative"
            else:
                overall_verdict = "net_neutral"

            return {
                "counterfactual_id": counterfactual_id,
                "title": cf.title,
                "original_event": cf.original_event,
                "counterfactual_condition": cf.counterfactual_condition,
                "divergence_point": cf.divergence_point.isoformat(),
                "delta_risk": cf.delta_risk,
                "delta_opportunity": cf.delta_opportunity,
                "net_delta": net_delta,
                "overall_verdict": overall_verdict,
                "risk_direction": metrics.risk_direction,
                "opportunity_direction": metrics.opportunity_direction,
                "confidence_overall": cf.confidence.overall,
                "evidence_strength": cf.evidence.evidence_strength,
                "learning_insights_count": len(cf.learning_insights),
                "evaluated_at": datetime.utcnow().isoformat(),
            }

    def append_learning_insight(
        self,
        counterfactual_id: str,
        insight: str,
    ) -> CounterfactualScenario:
        """
        Append a new learning insight to an existing CounterfactualScenario.

        Returns the updated CounterfactualScenario.

        Note: CounterfactualScenario is a mutable dataclass (not frozen),
        so learning_insights list can be appended in-place. The scenario's
        created_at timestamp is preserved; no historical field is mutated.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: insight is blank or counterfactual_id blank.
            JanusCounterfactualNotFoundError: ID does not exist.
        """
        self._assert_operational()
        if not counterfactual_id or not counterfactual_id.strip():
            raise JanusValidationError(
                "counterfactual_id must be a non-empty string.",
                field="counterfactual_id",
                value=counterfactual_id,
                engine=_ENGINE_NAME,
            )
        if not insight or not insight.strip():
            raise JanusValidationError(
                "insight must be a non-empty string.",
                field="insight",
                value=insight,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            cf = self._store.get(counterfactual_id)
            if cf is None:
                raise JanusCounterfactualNotFoundError(counterfactual_id)
            cf.learning_insights.append(insight.strip())
            _LOG.debug(
                "[%s] appended learning insight to counterfactual '%s'",
                _ENGINE_NAME,
                counterfactual_id,
            )
            return cf

    def compare_counterfactuals_for_event(
        self,
        original_event: str,
    ) -> list[dict[str, Any]]:
        """
        Compare all CounterfactualScenarios registered under the same
        original event, returning a ranked list of evaluation summaries.

        Ranked by net_delta (delta_opportunity − delta_risk) descending.

        Constitutional note: ranking evaluates alternatives. VEGA selects.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: original_event is blank.
        """
        self._assert_operational()
        counterfactuals = self.list_counterfactuals_for_event(original_event)
        if not counterfactuals:
            return []

        evaluations: list[dict[str, Any]] = []
        for cf in counterfactuals:
            evaluation = self.evaluate_counterfactual(cf.counterfactual_id)
            evaluations.append(evaluation)

        evaluations.sort(key=lambda e: e["net_delta"], reverse=True)
        for rank, ev in enumerate(evaluations, start=1):
            ev["rank"] = rank

        return evaluations

    # ------------------------------------------------------------------
    # Statistics & Observability
    # ------------------------------------------------------------------

    def get_statistics(self) -> CounterfactualStatistics:
        """
        Return a point-in-time statistics snapshot.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            all_cf = list(self._store.values())
            total = len(all_cf)

            counterfactuals_by_event: dict[str, int] = {
                event: len(ids) for event, ids in self._event_index.items()
            }
            avg_delta_risk = (
                sum(cf.delta_risk for cf in all_cf) / total if total else 0.0
            )
            avg_delta_opportunity = (
                sum(cf.delta_opportunity for cf in all_cf) / total if total else 0.0
            )
            avg_confidence = (
                sum(cf.confidence.overall for cf in all_cf) / total if total else 0.0
            )
            avg_evidence = (
                sum(cf.evidence.evidence_strength for cf in all_cf) / total
                if total
                else 0.0
            )
            total_insights = sum(len(cf.learning_insights) for cf in all_cf)
            avg_insights = total_insights / total if total else 0.0

            return CounterfactualStatistics(
                total_counterfactuals=total,
                counterfactuals_by_event=counterfactuals_by_event,
                average_delta_risk=avg_delta_risk,
                average_delta_opportunity=avg_delta_opportunity,
                average_confidence=avg_confidence,
                average_evidence_strength=avg_evidence,
                total_learning_insights=total_insights,
                average_insights_per_counterfactual=avg_insights,
                generated_at=datetime.utcnow(),
            )

    def get_health_report(self) -> CounterfactualHealthReport:
        """
        Return a lifecycle health report.

        Safe to call even before initialization or after shutdown.
        """
        with self._lock:
            return CounterfactualHealthReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                is_initialized=self._initialized,
                is_shut_down=self._shut_down,
                total_counterfactuals=len(self._store),
                events_tracked=len(self._event_index),
                generated_at=datetime.utcnow(),
            )

    def get_diagnostics_report(self) -> CounterfactualDiagnosticsReport:
        """
        Return a detailed diagnostics snapshot for engineering inspection.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            all_cf = list(self._store.values())
            total_insights = sum(len(cf.learning_insights) for cf in all_cf)
            ids_by_event = {
                event: list(ids)
                for event, ids in self._event_index.items()
            }
            return CounterfactualDiagnosticsReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                is_initialized=self._initialized,
                total_counterfactuals=len(self._store),
                counterfactual_ids=list(self._store.keys()),
                events_tracked=len(self._event_index),
                counterfactual_ids_by_event=ids_by_event,
                total_learning_insights=total_insights,
                generated_at=datetime.utcnow(),
            )

    # ------------------------------------------------------------------
    # Constitutional Guards
    # ------------------------------------------------------------------

    def assert_no_history_mutation(self, operation: str) -> None:
        """
        Constitutional guard: raise JanusConstitutionalViolationError if
        `operation` implies mutation of historical records.

        Historical records are owned exclusively by ECHO.
        JANUS may read and simulate; it may never write or mutate.

        Raises:
            JanusConstitutionalViolationError: operation implies history mutation.
        """
        lower_op = operation.lower().replace(" ", "_")
        for term in _HISTORY_MUTATION_TERMS:
            if term in lower_op:
                raise JanusConstitutionalViolationError(
                    engine=_ENGINE_NAME,
                    operation=operation,
                    rightful_owner="ECHO",
                )

    def assert_no_selection(self, operation: str) -> None:
        """
        Constitutional guard: raise JanusConstitutionalViolationError if
        `operation` implies branch/scenario selection.

        Selection is owned by VEGA. JANUS evaluates; VEGA selects.

        Raises:
            JanusConstitutionalViolationError: operation implies selection.
        """
        lower_op = operation.lower()
        for term in _SELECTION_TERMS:
            if term in lower_op:
                raise JanusConstitutionalViolationError(
                    engine=_ENGINE_NAME,
                    operation=operation,
                    rightful_owner="VEGA",
                )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _assert_operational(self) -> None:
        """Raise the appropriate lifecycle exception if the engine is not usable."""
        if not self._initialized:
            raise JanusNotInitializedError(_ENGINE_NAME)
        if self._shut_down:
            raise JanusShutdownError(_ENGINE_NAME)

    def _validate_create_request(
        self, request: CounterfactualCreateRequest
    ) -> None:
        """Validate all fields of a CounterfactualCreateRequest."""
        _required_str_fields: list[tuple[str, str]] = [
            ("title", request.title),
            ("description", request.description),
            ("original_event", request.original_event),
            ("counterfactual_condition", request.counterfactual_condition),
            ("requested_by", request.requested_by),
        ]
        for field_name, value in _required_str_fields:
            if not value or not value.strip():
                raise JanusValidationError(
                    f"'{field_name}' must be a non-empty string.",
                    field=field_name,
                    value=value,
                    engine=_ENGINE_NAME,
                )

        if request.divergence_point is None:
            raise JanusValidationError(
                "divergence_point must not be None.",
                field="divergence_point",
                value=None,
                engine=_ENGINE_NAME,
            )

        self._validate_divergence_point(request.divergence_point)
        self._validate_condition(request.counterfactual_condition)

        for field_name, value in (
            ("delta_risk", request.delta_risk),
            ("delta_opportunity", request.delta_opportunity),
        ):
            if not -1.0 <= value <= 1.0:
                raise JanusValidationError(
                    f"'{field_name}' must be in [-1.0, 1.0], got {value}.",
                    field=field_name,
                    value=value,
                    engine=_ENGINE_NAME,
                )

        if request.resulting_future_model is None:
            raise JanusValidationError(
                "resulting_future_model must not be None.",
                field="resulting_future_model",
                value=None,
                engine=_ENGINE_NAME,
            )
        if request.uncertainty is None:
            raise JanusValidationError(
                "uncertainty must not be None.",
                field="uncertainty",
                value=None,
                engine=_ENGINE_NAME,
            )
        if request.confidence is None:
            raise JanusValidationError(
                "confidence must not be None.",
                field="confidence",
                value=None,
                engine=_ENGINE_NAME,
            )
        if request.evidence is None:
            raise JanusValidationError(
                "evidence must not be None.",
                field="evidence",
                value=None,
                engine=_ENGINE_NAME,
            )

    def _validate_divergence_point(self, divergence_point: datetime) -> None:
        """
        Enforce that the divergence point is not in the future.

        Historical integrity invariant: a counterfactual diverges from a
        historical moment. A future divergence point implies predetermination,
        which contradicts the counterfactual model.

        Raises:
            JanusCounterfactualDivergenceError: divergence_point is in the future.
        """
        now = datetime.utcnow()
        if divergence_point > now:
            raise JanusCounterfactualDivergenceError(
                divergence_point=divergence_point.isoformat(),
                reason=(
                    "Divergence point is in the future. "
                    "Counterfactual analysis requires a historical divergence point. "
                    "Historical integrity invariant: JANUS simulates alternative "
                    "histories from past events only."
                ),
            )

    def _validate_condition(self, condition: str) -> None:
        """
        Validate that the counterfactual condition is syntactically non-trivial.

        Raises:
            JanusCounterfactualConditionError: condition is blank or self-referential.
        """
        if not condition or not condition.strip():
            raise JanusCounterfactualConditionError(
                condition=condition,
                reason="Counterfactual condition must be a non-empty string.",
            )
        stripped = condition.strip().lower()
        # A condition identical to common null-value placeholders is invalid
        _NULL_PLACEHOLDERS: frozenset[str] = frozenset(
            {"none", "null", "n/a", "na", "nothing", "same", "unchanged", "no change"}
        )
        if stripped in _NULL_PLACEHOLDERS:
            raise JanusCounterfactualConditionError(
                condition=condition,
                reason=(
                    f"Counterfactual condition '{condition}' is a null placeholder "
                    "and does not describe a meaningful alternative. "
                    "A counterfactual must specify a concrete alternate condition."
                ),
            )

    def _build_comparison_metrics(
        self,
        cf: CounterfactualScenario,
    ) -> CounterfactualComparisonMetrics:
        """Build CounterfactualComparisonMetrics from a CounterfactualScenario."""
        _NEUTRAL_BAND = 0.05

        risk_direction: str
        if cf.delta_risk < -_NEUTRAL_BAND:
            risk_direction = "improved"   # less risk in the counterfactual
        elif cf.delta_risk > _NEUTRAL_BAND:
            risk_direction = "worsened"
        else:
            risk_direction = "neutral"

        opportunity_direction: str
        if cf.delta_opportunity > _NEUTRAL_BAND:
            opportunity_direction = "improved"
        elif cf.delta_opportunity < -_NEUTRAL_BAND:
            opportunity_direction = "worsened"
        else:
            opportunity_direction = "neutral"

        net_delta = cf.delta_opportunity - cf.delta_risk

        from .models import ProbabilityLevel

        probability_level = ProbabilityLevel.from_float(cf.confidence.overall).value

        return CounterfactualComparisonMetrics(
            delta_risk=cf.delta_risk,
            delta_opportunity=cf.delta_opportunity,
            risk_direction=risk_direction,
            opportunity_direction=opportunity_direction,
            confidence_overall=cf.confidence.overall,
            evidence_strength=cf.evidence.evidence_strength,
            net_delta=net_delta,
            probability_level=probability_level,
        )

    def _synthesise_insights(
        self,
        insights: tuple[str, ...],
        reference_event: str,
        counterfactual_condition: str,
        metrics: CounterfactualComparisonMetrics,
    ) -> str:
        """
        Produce a synthesised narrative summary of learning insights.

        This is a deterministic text synthesis. No generative model is
        called; the synthesis is assembled from the structured fields.
        """
        insight_count = len(insights)
        risk_note = (
            f"Risk was {metrics.risk_direction} "
            f"(Δ={metrics.delta_risk:+.3f})"
        )
        opp_note = (
            f"opportunity was {metrics.opportunity_direction} "
            f"(Δ={metrics.delta_opportunity:+.3f})"
        )
        net_note = (
            f"net delta: {metrics.net_delta:+.3f} "
            f"({metrics.probability_level} confidence)"
        )
        return (
            f"Counterfactual analysis of '{reference_event}' under condition "
            f"'{counterfactual_condition}': {risk_note}, {opp_note}, {net_note}. "
            f"{insight_count} learning insight(s) recorded."
        )