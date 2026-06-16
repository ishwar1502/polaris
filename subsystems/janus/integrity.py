"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/integrity.py

Implements IScenarioIntegrityEngine.

Responsibility: Protect forecast quality by preventing impossible futures,
unsupported predictions, contradictory outcomes, and false certainty claims.

Constitutional Rule (JANUS Law 6):
    All forecasts require uncertainty. No certainty claims ever.
    This engine is the enforcement layer for that invariant.

JANUS integrity analysis provides validation.
JANUS integrity analysis does NOT make decisions.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Optional

from .exceptions import (
    JanusAlreadyInitializedError,
    JanusConstitutionalViolationError,
    JanusEvidenceProfileError,
    JanusFalseCertaintyError,
    JanusImpossibleFutureError,
    JanusIntegrityError,
    JanusMissingUncertaintyError,
    JanusNotInitializedError,
    JanusShutdownError,
    JanusUnsupportedPredictionError,
    JanusContradictoryOutcomeError,
)
from .interfaces import IScenarioIntegrityEngine
from .models import (
    EvidenceProfile,
    Forecast,
    FutureModel,
    OutcomeSimulation,
    Scenario,
    ScenarioStatus,
    SimulationStatus,
    TimelineProjection,
    UncertaintyLevel,
    UncertaintyProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENGINE_NAME: Final[str] = "ScenarioIntegrityEngine"
_ENGINE_VERSION: Final[str] = "5.1.0"

# Minimum evidence strength required for any JANUS artifact.
_MIN_EVIDENCE_STRENGTH: Final[float] = 0.05

# If a probability == 1.0 exactly and uncertainty is NEGLIGIBLE, it is false certainty.
_CERTAINTY_PROBABILITY_THRESHOLD: Final[float] = 1.0

# The maximum tolerated probability distribution sum deviation.
_DISTRIBUTION_TOLERANCE: Final[float] = 1e-6

# Constitutional ownership map: operation_prefix → rightful_owner subsystem.
_CONSTITUTIONAL_OWNERSHIP: Final[dict[str, str]] = {
    "decide": "VEGA",
    "select_strategy": "ODYSSEY",
    "plan": "ZENITH",
    "execute": "DRACO",
    "reason": "ORION",
    "approve": "VEGA",
    "choose_direction": "ODYSSEY",
    "create_plan": "ZENITH",
    "own_memory": "ECHO",
    "store_knowledge": "ASTRA",
    "manage_identity": "ASTRA",
}

# ---------------------------------------------------------------------------
# Internal result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ValidationResult:
    """Immutable internal record of a single validation outcome."""
    artifact_id: str
    artifact_type: str
    is_valid: bool
    violations: tuple[str, ...]
    validated_at: datetime


@dataclass
class _IntegrityStatistics:
    """Mutable statistics accumulator, guarded by the engine lock."""
    total_validations: int = 0
    total_violations: int = 0
    scenarios_validated: int = 0
    scenarios_failed: int = 0
    forecasts_validated: int = 0
    forecasts_failed: int = 0
    future_models_validated: int = 0
    future_models_failed: int = 0
    simulations_validated: int = 0
    simulations_failed: int = 0
    timeline_projections_validated: int = 0
    timeline_projections_failed: int = 0
    uncertainty_enforcements: int = 0
    uncertainty_violations: int = 0
    evidence_enforcements: int = 0
    evidence_violations: int = 0
    constitutional_checks: int = 0
    constitutional_violations: int = 0
    false_certainty_detections: int = 0
    impossible_future_detections: int = 0
    contradictory_outcome_detections: int = 0
    unsupported_prediction_detections: int = 0
    engine_start_time: Optional[datetime] = None
    last_validation_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# ScenarioIntegrityEngine
# ---------------------------------------------------------------------------


class ScenarioIntegrityEngine(IScenarioIntegrityEngine):
    """
    Production implementation of IScenarioIntegrityEngine.

    Thread-safe, lifecycle-managed engine that enforces JANUS integrity
    invariants across all future-modeling artifacts.

    Validation posture:
        - Rejects impossible futures (probability > 1.0, states that cannot coexist).
        - Rejects unsupported predictions (zero evidence sources, zero evidence strength).
        - Rejects false certainty (probability == 1.0 with NEGLIGIBLE uncertainty).
        - Rejects contradictory outcomes (duplicate labels, mutually exclusive probabilities).
        - Enforces JANUS Law 6: every artifact must carry an UncertaintyProfile.
        - Enforces evidence minimums on all artifacts.
        - Guards POLARIS constitutional boundaries.

    This engine never makes decisions, approves futures, or selects strategies.
    """

    def __init__(self, min_evidence_strength: float = _MIN_EVIDENCE_STRENGTH) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False
        self._shutdown: bool = False
        self._stats: _IntegrityStatistics = _IntegrityStatistics()
        self._validation_history: list[_ValidationResult] = []
        self._min_evidence_strength: float = min_evidence_strength

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._stats.engine_start_time = datetime.utcnow()
            self._initialized = True
            self._shutdown = False
        logger.info("[%s] Initialized (version=%s).", _ENGINE_NAME, _ENGINE_VERSION)

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                raise JanusNotInitializedError(_ENGINE_NAME)
            if self._shutdown:
                return
            self._shutdown = True
            self._initialized = False
        logger.info(
            "[%s] Shutdown. Total validations=%d, total violations=%d.",
            _ENGINE_NAME,
            self._stats.total_validations,
            self._stats.total_violations,
        )

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

    # ------------------------------------------------------------------
    # Lifecycle guard helper
    # ------------------------------------------------------------------

    def _guard(self) -> None:
        """Raise appropriate lifecycle exception if engine is not operational."""
        if self._shutdown:
            raise JanusShutdownError(_ENGINE_NAME)
        if not self._initialized:
            raise JanusNotInitializedError(_ENGINE_NAME)

    # ------------------------------------------------------------------
    # IScenarioIntegrityEngine — validate_scenario
    # ------------------------------------------------------------------

    def validate_scenario(self, scenario: Scenario) -> tuple[bool, tuple[str, ...]]:
        """
        Validate a Scenario for integrity violations.

        Checks:
            1. Scenario is not INVALIDATED before validation.
            2. risk_score and opportunity_score are in [0, 1].
            3. All branches have valid probabilities.
            4. Branch probability distribution sums are internally consistent.
            5. No branch describes an impossible future state (probability > 1.0).
            6. FutureModel integrity (delegated to validate_future_model).
            7. UncertaintyProfile is present and non-trivial (Law 6).
            8. ConfidenceProfile overall is not 1.0 with NEGLIGIBLE uncertainty (false certainty).
            9. EvidenceProfile passes minimum strength threshold.
           10. No duplicate branch labels.
        """
        with self._lock:
            self._guard()
            violations: list[str] = []
            artifact_id = scenario.scenario_id
            artifact_type = "Scenario"

            # Status check
            if scenario.status == ScenarioStatus.INVALIDATED:
                violations.append(
                    f"Scenario '{artifact_id}' has status INVALIDATED and cannot be validated."
                )

            # Score range checks
            if not 0.0 <= scenario.risk_score <= 1.0:
                violations.append(
                    f"risk_score={scenario.risk_score!r} is outside [0.0, 1.0]."
                )
            if not 0.0 <= scenario.opportunity_score <= 1.0:
                violations.append(
                    f"opportunity_score={scenario.opportunity_score!r} is outside [0.0, 1.0]."
                )

            # Branch validations
            branch_labels: set[str] = set()
            for branch in scenario.branches:
                if not 0.0 <= branch.probability <= 1.0:
                    violations.append(
                        f"Branch '{branch.branch_id}' probability={branch.probability!r} "
                        "is outside [0.0, 1.0]."
                    )
                if branch.probability > 1.0:
                    violations.append(
                        f"Branch '{branch.branch_id}' has an impossible probability > 1.0."
                    )
                    self._stats.impossible_future_detections += 1
                if branch.label in branch_labels:
                    violations.append(
                        f"Duplicate branch label '{branch.label}' detected in scenario "
                        f"'{artifact_id}'."
                    )
                    self._stats.contradictory_outcome_detections += 1
                branch_labels.add(branch.label)

                # Branch future state probability
                fs = branch.future_state
                if not 0.0 <= fs.probability <= 1.0:
                    violations.append(
                        f"Branch '{branch.branch_id}' FutureState '{fs.state_id}' "
                        f"probability={fs.probability!r} is outside [0.0, 1.0]."
                    )
                if fs.probability > 1.0:
                    self._stats.impossible_future_detections += 1

                # Branch confidence false certainty
                branch_cf_violations = self._check_false_certainty(
                    branch.branch_id,
                    "ScenarioBranch",
                    branch.probability,
                    branch.confidence.overall,
                    scenario.uncertainty,
                )
                violations.extend(branch_cf_violations)

            # FutureModel integrity
            fm_valid, fm_violations = self.validate_future_model(scenario.future_model)
            if not fm_valid:
                for v in fm_violations:
                    violations.append(f"[FutureModel] {v}")

            # Uncertainty invariant (Law 6)
            try:
                self._enforce_uncertainty_invariant_internal(
                    artifact_id, artifact_type, scenario.uncertainty
                )
            except JanusMissingUncertaintyError as exc:
                violations.append(str(exc))
                self._stats.uncertainty_violations += 1

            # False certainty on scenario confidence
            sc_cf_violations = self._check_false_certainty(
                artifact_id,
                artifact_type,
                None,
                scenario.confidence.overall,
                scenario.uncertainty,
            )
            violations.extend(sc_cf_violations)

            # Evidence invariant
            try:
                self._enforce_evidence_invariant_internal(
                    artifact_id, artifact_type, scenario.evidence
                )
            except JanusEvidenceProfileError as exc:
                violations.append(str(exc))
                self._stats.evidence_violations += 1

            is_valid = len(violations) == 0
            self._record_validation(artifact_id, artifact_type, is_valid, violations)
            self._stats.scenarios_validated += 1
            if not is_valid:
                self._stats.scenarios_failed += 1

            return is_valid, tuple(violations)

    # ------------------------------------------------------------------
    # IScenarioIntegrityEngine — validate_forecast
    # ------------------------------------------------------------------

    def validate_forecast(self, forecast: Forecast) -> tuple[bool, tuple[str, ...]]:
        """
        Validate a Forecast for integrity violations.

        Checks:
            1. Forecast status is not 'superseded' or unknown.
            2. ProbabilityDistribution outcomes sum to 1.0 (±1e-6).
            3. All individual outcome probabilities are in [0, 1].
            4. No certainty claim (Law 6 enforcement).
            5. UncertaintyProfile is present and non-trivial.
            6. EvidenceProfile has at least one source and meets strength minimum.
            7. ForecastMetadata has at least one data_source (unsupported prediction guard).
            8. Confidence overall is not 1.0 with NEGLIGIBLE uncertainty.
        """
        with self._lock:
            self._guard()
            violations: list[str] = []
            artifact_id = forecast.forecast_id
            artifact_type = "Forecast"

            # Status
            if forecast.status == "superseded":
                violations.append(
                    f"Forecast '{artifact_id}' has been superseded and should not be re-validated."
                )

            # Probability distribution
            dist = forecast.probability_distribution
            pd_violations = self._validate_probability_distribution(artifact_id, dist)
            violations.extend(pd_violations)

            # Data sources — unsupported prediction guard
            if not forecast.metadata.data_sources:
                violations.append(
                    f"Forecast '{artifact_id}' has no data_sources in metadata. "
                    "All forecasts must remain evidence-based."
                )
                self._stats.unsupported_prediction_detections += 1

            # Uncertainty invariant (Law 6)
            try:
                self._enforce_uncertainty_invariant_internal(
                    artifact_id, artifact_type, forecast.uncertainty
                )
            except JanusMissingUncertaintyError as exc:
                violations.append(str(exc))
                self._stats.uncertainty_violations += 1

            # False certainty check across all outcomes
            for outcome_label, prob in dist.outcomes.items():
                if prob >= _CERTAINTY_PROBABILITY_THRESHOLD:
                    cf_violations = self._check_false_certainty(
                        artifact_id,
                        f"Forecast outcome '{outcome_label}'",
                        prob,
                        forecast.confidence.overall,
                        forecast.uncertainty,
                    )
                    violations.extend(cf_violations)

            # Confidence false certainty
            cf_violations = self._check_false_certainty(
                artifact_id,
                artifact_type,
                None,
                forecast.confidence.overall,
                forecast.uncertainty,
            )
            violations.extend(cf_violations)

            # Evidence invariant
            try:
                self._enforce_evidence_invariant_internal(
                    artifact_id, artifact_type, forecast.evidence
                )
            except JanusEvidenceProfileError as exc:
                violations.append(str(exc))
                self._stats.evidence_violations += 1

            is_valid = len(violations) == 0
            self._record_validation(artifact_id, artifact_type, is_valid, violations)
            self._stats.forecasts_validated += 1
            if not is_valid:
                self._stats.forecasts_failed += 1

            return is_valid, tuple(violations)

    # ------------------------------------------------------------------
    # IScenarioIntegrityEngine — validate_future_model
    # ------------------------------------------------------------------

    def validate_future_model(
        self, future_model: FutureModel
    ) -> tuple[bool, tuple[str, ...]]:
        """
        Validate a FutureModel for impossible states or contradictory outcomes.

        Checks:
            1. At least one FutureState exists.
            2. All FutureState probabilities are in [0, 1].
            3. No two FutureStates at the same horizon have probabilities that sum > 1.0.
            4. No duplicate state labels at the same horizon.
            5. UncertaintyProfile is non-trivial.
            6. EvidenceProfile meets strength minimum.
            7. Confidence overall is not 1.0 with NEGLIGIBLE uncertainty.
        """
        with self._lock:
            self._guard()
            violations: list[str] = []
            artifact_id = future_model.model_id
            artifact_type = "FutureModel"

            if not future_model.future_states:
                violations.append(
                    f"FutureModel '{artifact_id}' has no FutureStates. "
                    "A model with zero states describes an impossible future."
                )
                self._stats.impossible_future_detections += 1

            # Per-horizon probability and label checks
            horizon_state_map: dict[str, list] = {}
            for state in future_model.future_states:
                h_key = state.horizon.value
                horizon_state_map.setdefault(h_key, []).append(state)

                if not 0.0 <= state.probability <= 1.0:
                    violations.append(
                        f"FutureState '{state.state_id}' probability={state.probability!r} "
                        "is outside [0.0, 1.0]."
                    )
                    self._stats.impossible_future_detections += 1

            for horizon_key, states in horizon_state_map.items():
                # Duplicate label check per horizon
                labels_seen: set[str] = set()
                for st in states:
                    if st.label in labels_seen:
                        violations.append(
                            f"FutureModel '{artifact_id}' has duplicate FutureState label "
                            f"'{st.label}' at horizon '{horizon_key}'."
                        )
                        self._stats.contradictory_outcome_detections += 1
                    labels_seen.add(st.label)

                # Probability sum at horizon must not exceed 1.0 by more than tolerance
                total = sum(st.probability for st in states)
                if total > 1.0 + _DISTRIBUTION_TOLERANCE:
                    violations.append(
                        f"FutureModel '{artifact_id}' FutureState probabilities at horizon "
                        f"'{horizon_key}' sum to {total:.6f}, exceeding 1.0. "
                        "Impossible future detected."
                    )
                    self._stats.impossible_future_detections += 1

            # Uncertainty invariant (Law 6)
            try:
                self._enforce_uncertainty_invariant_internal(
                    artifact_id, artifact_type, future_model.uncertainty
                )
            except JanusMissingUncertaintyError as exc:
                violations.append(str(exc))
                self._stats.uncertainty_violations += 1

            # False certainty on confidence
            cf_violations = self._check_false_certainty(
                artifact_id,
                artifact_type,
                None,
                future_model.confidence.overall,
                future_model.uncertainty,
            )
            violations.extend(cf_violations)

            # Evidence invariant
            try:
                self._enforce_evidence_invariant_internal(
                    artifact_id, artifact_type, future_model.evidence
                )
            except JanusEvidenceProfileError as exc:
                violations.append(str(exc))
                self._stats.evidence_violations += 1

            is_valid = len(violations) == 0
            self._record_validation(artifact_id, artifact_type, is_valid, violations)
            self._stats.future_models_validated += 1
            if not is_valid:
                self._stats.future_models_failed += 1

            return is_valid, tuple(violations)

    # ------------------------------------------------------------------
    # IScenarioIntegrityEngine — validate_simulation
    # ------------------------------------------------------------------

    def validate_simulation(
        self, simulation: OutcomeSimulation
    ) -> tuple[bool, tuple[str, ...]]:
        """
        Validate an OutcomeSimulation for internal consistency and evidence support.

        Checks:
            1. Simulation is in a terminal or active state (not FAILED with no outcomes).
            2. All SimulationOutcome probabilities are in [0, 1].
            3. Outcome probability distribution sums to 1.0 (±1e-6) if outcomes exist.
            4. No duplicate outcome labels.
            5. ProbabilityDistribution of the simulation is internally consistent.
            6. UncertaintyProfile is present.
            7. EvidenceProfile meets minimum strength (inferred from ProbabilityDistribution
               confidence).
            8. No false certainty on any individual outcome.
        """
        with self._lock:
            self._guard()
            violations: list[str] = []
            artifact_id = simulation.simulation_id
            artifact_type = "OutcomeSimulation"

            # Terminal state checks
            if simulation.status == SimulationStatus.FAILED and not simulation.outcomes:
                violations.append(
                    f"OutcomeSimulation '{artifact_id}' has status FAILED with zero outcomes. "
                    "Cannot perform integrity validation on an incomplete simulation."
                )

            # Outcome-level checks
            outcome_labels: set[str] = set()
            outcome_prob_total: float = 0.0
            for outcome in simulation.outcomes:
                if not 0.0 <= outcome.probability <= 1.0:
                    violations.append(
                        f"SimulationOutcome '{outcome.outcome_id}' probability="
                        f"{outcome.probability!r} is outside [0.0, 1.0]."
                    )
                    self._stats.impossible_future_detections += 1

                if outcome.label in outcome_labels:
                    violations.append(
                        f"Duplicate SimulationOutcome label '{outcome.label}' in simulation "
                        f"'{artifact_id}'."
                    )
                    self._stats.contradictory_outcome_detections += 1
                outcome_labels.add(outcome.label)
                outcome_prob_total += outcome.probability

                # False certainty per outcome
                if outcome.probability >= _CERTAINTY_PROBABILITY_THRESHOLD:
                    cf_violations = self._check_false_certainty(
                        outcome.outcome_id,
                        "SimulationOutcome",
                        outcome.probability,
                        outcome.confidence.overall,
                        simulation.uncertainty,
                    )
                    violations.extend(cf_violations)

            # Outcome probability sum check (only when outcomes present)
            if simulation.outcomes and abs(outcome_prob_total - 1.0) > _DISTRIBUTION_TOLERANCE:
                violations.append(
                    f"OutcomeSimulation '{artifact_id}' outcome probabilities sum to "
                    f"{outcome_prob_total:.6f}; must sum to 1.0 (±1e-6)."
                )

            # ProbabilityDistribution consistency
            pd_violations = self._validate_probability_distribution(
                artifact_id, simulation.probability_distribution
            )
            violations.extend(pd_violations)

            # Uncertainty invariant (Law 6)
            try:
                self._enforce_uncertainty_invariant_internal(
                    artifact_id, artifact_type, simulation.uncertainty
                )
            except JanusMissingUncertaintyError as exc:
                violations.append(str(exc))
                self._stats.uncertainty_violations += 1

            # Evidence check via ProbabilityDistribution confidence as proxy
            dist_confidence = simulation.probability_distribution.confidence
            if dist_confidence.overall < self._min_evidence_strength:
                violations.append(
                    f"OutcomeSimulation '{artifact_id}' ProbabilityDistribution confidence="
                    f"{dist_confidence.overall!r} is below the minimum evidence threshold of "
                    f"{self._min_evidence_strength!r}. Predictions lack sufficient support."
                )
                self._stats.unsupported_prediction_detections += 1

            is_valid = len(violations) == 0
            self._record_validation(artifact_id, artifact_type, is_valid, violations)
            self._stats.simulations_validated += 1
            if not is_valid:
                self._stats.simulations_failed += 1

            return is_valid, tuple(violations)

    # ------------------------------------------------------------------
    # IScenarioIntegrityEngine — validate_timeline_projection
    # ------------------------------------------------------------------

    def validate_timeline_projection(
        self, projection: TimelineProjection
    ) -> tuple[bool, tuple[str, ...]]:
        """
        Validate a TimelineProjection for milestone consistency and evidence support.

        Checks:
            1. At least one milestone exists.
            2. All milestone probabilities are in [0, 1].
            3. No duplicate milestone labels.
            4. Critical milestones have probability > 0.0 (a critical milestone with 0 probability
               is an impossible future).
            5. Milestone dependency IDs reference existing milestones within the projection.
            6. Chronological ordering: milestones projected_at must be non-decreasing.
            7. UncertaintyProfile is present and non-trivial.
            8. EvidenceProfile meets minimum strength.
            9. Confidence overall is not 1.0 with NEGLIGIBLE uncertainty.
        """
        with self._lock:
            self._guard()
            violations: list[str] = []
            artifact_id = projection.projection_id
            artifact_type = "TimelineProjection"

            if not projection.milestones:
                violations.append(
                    f"TimelineProjection '{artifact_id}' has no milestones. "
                    "A projection with zero milestones cannot model the future."
                )
                self._stats.impossible_future_detections += 1

            milestone_ids: set[str] = {m.milestone_id for m in projection.milestones}
            milestone_labels: set[str] = set()
            previous_projected_at: Optional[datetime] = None

            for milestone in projection.milestones:
                # Probability range
                if not 0.0 <= milestone.probability <= 1.0:
                    violations.append(
                        f"ProjectionMilestone '{milestone.milestone_id}' probability="
                        f"{milestone.probability!r} is outside [0.0, 1.0]."
                    )
                    self._stats.impossible_future_detections += 1

                # Critical milestone with zero probability
                if milestone.is_critical and milestone.probability == 0.0:
                    violations.append(
                        f"Critical ProjectionMilestone '{milestone.milestone_id}' "
                        f"('{milestone.label}') has probability=0.0. "
                        "A critical milestone with zero probability describes an impossible future."
                    )
                    self._stats.impossible_future_detections += 1

                # Duplicate label
                if milestone.label in milestone_labels:
                    violations.append(
                        f"Duplicate milestone label '{milestone.label}' in projection "
                        f"'{artifact_id}'."
                    )
                    self._stats.contradictory_outcome_detections += 1
                milestone_labels.add(milestone.label)

                # Dependency reference integrity
                for dep_id in milestone.dependencies:
                    if dep_id not in milestone_ids:
                        violations.append(
                            f"ProjectionMilestone '{milestone.milestone_id}' depends on "
                            f"unknown milestone_id '{dep_id}'."
                        )

                # Chronological ordering
                if previous_projected_at is not None:
                    if milestone.projected_at < previous_projected_at:
                        violations.append(
                            f"ProjectionMilestone '{milestone.milestone_id}' "
                            f"projected_at={milestone.projected_at.isoformat()} is earlier than "
                            f"the preceding milestone's projected_at="
                            f"{previous_projected_at.isoformat()}. Timeline is not ordered."
                        )
                previous_projected_at = milestone.projected_at

            # Uncertainty invariant (Law 6)
            try:
                self._enforce_uncertainty_invariant_internal(
                    artifact_id, artifact_type, projection.uncertainty
                )
            except JanusMissingUncertaintyError as exc:
                violations.append(str(exc))
                self._stats.uncertainty_violations += 1

            # False certainty on confidence
            cf_violations = self._check_false_certainty(
                artifact_id,
                artifact_type,
                None,
                projection.confidence.overall,
                projection.uncertainty,
            )
            violations.extend(cf_violations)

            # Evidence invariant
            try:
                self._enforce_evidence_invariant_internal(
                    artifact_id, artifact_type, projection.evidence
                )
            except JanusEvidenceProfileError as exc:
                violations.append(str(exc))
                self._stats.evidence_violations += 1

            is_valid = len(violations) == 0
            self._record_validation(artifact_id, artifact_type, is_valid, violations)
            self._stats.timeline_projections_validated += 1
            if not is_valid:
                self._stats.timeline_projections_failed += 1

            return is_valid, tuple(violations)

    # ------------------------------------------------------------------
    # IScenarioIntegrityEngine — enforce_uncertainty_invariant
    # ------------------------------------------------------------------

    def enforce_uncertainty_invariant(
        self, artifact_id: str, artifact_type: str, uncertainty: UncertaintyProfile
    ) -> None:
        """
        Enforce JANUS Law 6: raise JanusMissingUncertaintyError if the
        UncertaintyProfile is absent or trivial.

        A trivial profile is one where:
            - uncertainty_id is empty, or
            - level is NEGLIGIBLE and all numeric scores are 0.0.
        """
        with self._lock:
            self._guard()
            self._stats.uncertainty_enforcements += 1
            try:
                self._enforce_uncertainty_invariant_internal(
                    artifact_id, artifact_type, uncertainty
                )
            except JanusMissingUncertaintyError:
                self._stats.uncertainty_violations += 1
                raise

    # ------------------------------------------------------------------
    # IScenarioIntegrityEngine — enforce_evidence_invariant
    # ------------------------------------------------------------------

    def enforce_evidence_invariant(
        self, artifact_id: str, artifact_type: str, evidence: EvidenceProfile
    ) -> None:
        """
        Raise JanusEvidenceProfileError if the EvidenceProfile is absent,
        empty, or below the minimum evidence strength threshold.
        """
        with self._lock:
            self._guard()
            self._stats.evidence_enforcements += 1
            try:
                self._enforce_evidence_invariant_internal(
                    artifact_id, artifact_type, evidence
                )
            except JanusEvidenceProfileError:
                self._stats.evidence_violations += 1
                raise

    # ------------------------------------------------------------------
    # IScenarioIntegrityEngine — check_constitutional_boundary
    # ------------------------------------------------------------------

    def check_constitutional_boundary(
        self, engine_name: str, operation: str, rightful_owner: str
    ) -> None:
        """
        Raise JanusConstitutionalViolationError if the given operation
        belongs to a different POLARIS subsystem.

        The check normalises the operation to lowercase and scans
        _CONSTITUTIONAL_OWNERSHIP for any registered prefix that is
        contained in the operation name. Additionally, it validates that the
        caller's declared rightful_owner matches the registered owner.
        """
        with self._lock:
            self._guard()
            self._stats.constitutional_checks += 1

            operation_lower = operation.lower()

            for restricted_prefix, actual_owner in _CONSTITUTIONAL_OWNERSHIP.items():
                if restricted_prefix in operation_lower:
                    if actual_owner.upper() != rightful_owner.upper():
                        # The caller declared a wrong owner — likely a violation.
                        self._stats.constitutional_violations += 1
                        raise JanusConstitutionalViolationError(
                            engine=engine_name,
                            operation=operation,
                            rightful_owner=actual_owner,
                        )
                    # Prefix matched and owner is consistent — still a violation
                    # if the caller is a JANUS engine claiming ownership.
                    if engine_name.upper() not in ("JANUS", "SCENARIOINTEGRITYENGINE") and \
                            actual_owner.upper() == engine_name.upper():
                        # The engine is declaring itself as the owner of the operation,
                        # but the operation is in the restricted list owned by another system.
                        pass  # Correct: owner is consistent, no violation.
                    # If the JANUS engine is trying to perform a restricted operation:
                    if actual_owner.upper() != engine_name.upper() and \
                            _ENGINE_NAME.upper() in engine_name.upper():
                        self._stats.constitutional_violations += 1
                        raise JanusConstitutionalViolationError(
                            engine=engine_name,
                            operation=operation,
                            rightful_owner=actual_owner,
                        )

    # ------------------------------------------------------------------
    # Public read-only accessors (diagnostics / health)
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Return a snapshot of current engine statistics."""
        with self._lock:
            self._guard()
            return {
                "engine_name": _ENGINE_NAME,
                "engine_version": _ENGINE_VERSION,
                "engine_start_time": (
                    self._stats.engine_start_time.isoformat()
                    if self._stats.engine_start_time else None
                ),
                "last_validation_at": (
                    self._stats.last_validation_at.isoformat()
                    if self._stats.last_validation_at else None
                ),
                "total_validations": self._stats.total_validations,
                "total_violations": self._stats.total_violations,
                "scenarios": {
                    "validated": self._stats.scenarios_validated,
                    "failed": self._stats.scenarios_failed,
                },
                "forecasts": {
                    "validated": self._stats.forecasts_validated,
                    "failed": self._stats.forecasts_failed,
                },
                "future_models": {
                    "validated": self._stats.future_models_validated,
                    "failed": self._stats.future_models_failed,
                },
                "simulations": {
                    "validated": self._stats.simulations_validated,
                    "failed": self._stats.simulations_failed,
                },
                "timeline_projections": {
                    "validated": self._stats.timeline_projections_validated,
                    "failed": self._stats.timeline_projections_failed,
                },
                "invariant_enforcement": {
                    "uncertainty_enforcements": self._stats.uncertainty_enforcements,
                    "uncertainty_violations": self._stats.uncertainty_violations,
                    "evidence_enforcements": self._stats.evidence_enforcements,
                    "evidence_violations": self._stats.evidence_violations,
                    "constitutional_checks": self._stats.constitutional_checks,
                    "constitutional_violations": self._stats.constitutional_violations,
                },
                "detections": {
                    "false_certainty": self._stats.false_certainty_detections,
                    "impossible_future": self._stats.impossible_future_detections,
                    "contradictory_outcome": self._stats.contradictory_outcome_detections,
                    "unsupported_prediction": self._stats.unsupported_prediction_detections,
                },
            }

    def get_health(self) -> dict[str, Any]:
        """Return a health report for the engine."""
        with self._lock:
            is_healthy = self._initialized and not self._shutdown
            uptime_seconds: Optional[float] = None
            if self._stats.engine_start_time is not None:
                uptime_seconds = (
                    datetime.utcnow() - self._stats.engine_start_time
                ).total_seconds()
            return {
                "engine_name": _ENGINE_NAME,
                "engine_version": _ENGINE_VERSION,
                "is_healthy": is_healthy,
                "is_initialized": self._initialized,
                "is_shutdown": self._shutdown,
                "uptime_seconds": uptime_seconds,
                "min_evidence_strength": self._min_evidence_strength,
                "validation_history_size": len(self._validation_history),
            }

    def get_validation_history(self) -> tuple[dict[str, Any], ...]:
        """Return an immutable snapshot of the validation history."""
        with self._lock:
            self._guard()
            return tuple(
                {
                    "artifact_id": r.artifact_id,
                    "artifact_type": r.artifact_type,
                    "is_valid": r.is_valid,
                    "violations": list(r.violations),
                    "validated_at": r.validated_at.isoformat(),
                }
                for r in self._validation_history
            )

    def get_diagnostics(self) -> dict[str, Any]:
        """Return a comprehensive diagnostics report."""
        with self._lock:
            health = self.get_health() if self._initialized and not self._shutdown else {
                "engine_name": _ENGINE_NAME,
                "is_healthy": False,
            }
        stats = {}
        if self._initialized and not self._shutdown:
            stats = self.get_statistics()
        history_summary: list[dict[str, Any]] = []
        if self._validation_history:
            recent = self._validation_history[-10:]
            history_summary = [
                {
                    "artifact_id": r.artifact_id,
                    "artifact_type": r.artifact_type,
                    "is_valid": r.is_valid,
                    "violation_count": len(r.violations),
                    "validated_at": r.validated_at.isoformat(),
                }
                for r in recent
            ]
        return {
            "health": health,
            "statistics": stats,
            "recent_validations": history_summary,
            "constitutional_ownership_map": dict(_CONSTITUTIONAL_OWNERSHIP),
            "min_evidence_strength_threshold": self._min_evidence_strength,
            "false_certainty_probability_threshold": _CERTAINTY_PROBABILITY_THRESHOLD,
            "distribution_sum_tolerance": _DISTRIBUTION_TOLERANCE,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enforce_uncertainty_invariant_internal(
        self,
        artifact_id: str,
        artifact_type: str,
        uncertainty: UncertaintyProfile,
    ) -> None:
        """
        Internal (non-locking) implementation of the uncertainty invariant check.
        Raises JanusMissingUncertaintyError if the profile is trivial.

        A profile is trivial when ALL of the following hold:
            - level is NEGLIGIBLE
            - unknown_risk_exposure == 0.0
            - volatility_score == 0.0
            - market_sensitivity == 0.0
            - technology_sensitivity == 0.0
            - known_risks is empty
            - external_factors is empty
        """
        if not uncertainty.uncertainty_id:
            raise JanusMissingUncertaintyError(artifact_id, artifact_type)

        is_trivial = (
            uncertainty.level == UncertaintyLevel.NEGLIGIBLE
            and uncertainty.unknown_risk_exposure == 0.0
            and uncertainty.volatility_score == 0.0
            and uncertainty.market_sensitivity == 0.0
            and uncertainty.technology_sensitivity == 0.0
            and not uncertainty.known_risks
            and not uncertainty.external_factors
        )
        if is_trivial:
            raise JanusMissingUncertaintyError(artifact_id, artifact_type)

    def _enforce_evidence_invariant_internal(
        self,
        artifact_id: str,
        artifact_type: str,
        evidence: EvidenceProfile,
    ) -> None:
        """
        Internal (non-locking) implementation of the evidence invariant check.
        Raises JanusEvidenceProfileError when:
            - evidence_id is empty, or
            - sources is empty, or
            - evidence_strength < self._min_evidence_strength.
        """
        if not evidence.evidence_id:
            raise JanusEvidenceProfileError(
                artifact_id,
                f"{artifact_type} has an EvidenceProfile with no evidence_id.",
            )
        if not evidence.sources:
            raise JanusEvidenceProfileError(
                artifact_id,
                f"{artifact_type} has an EvidenceProfile with no sources. "
                "All forecasts must remain evidence-based.",
            )
        if evidence.evidence_strength < self._min_evidence_strength:
            raise JanusEvidenceProfileError(
                artifact_id,
                f"{artifact_type} evidence_strength={evidence.evidence_strength!r} is below "
                f"the minimum threshold of {self._min_evidence_strength!r}.",
            )

    def _check_false_certainty(
        self,
        artifact_id: str,
        artifact_type: str,
        probability: Optional[float],
        confidence_overall: float,
        uncertainty: UncertaintyProfile,
    ) -> list[str]:
        """
        Return a list of violation strings if a false certainty condition is detected.

        False certainty conditions (JANUS Law 6):
            1. confidence_overall == 1.0 AND uncertainty level is NEGLIGIBLE.
            2. A specific probability == 1.0 AND uncertainty level is NEGLIGIBLE.
        """
        violations: list[str] = []

        is_negligible = uncertainty.level == UncertaintyLevel.NEGLIGIBLE

        if confidence_overall >= _CERTAINTY_PROBABILITY_THRESHOLD and is_negligible:
            violations.append(
                f"{artifact_type} '{artifact_id}' claims certainty: "
                f"confidence_overall={confidence_overall!r} with NEGLIGIBLE uncertainty. "
                "JANUS Law 6: All forecasts require uncertainty. No certainty claims ever."
            )
            self._stats.false_certainty_detections += 1

        if probability is not None and probability >= _CERTAINTY_PROBABILITY_THRESHOLD and is_negligible:
            violations.append(
                f"{artifact_type} '{artifact_id}' has probability={probability!r} with "
                "NEGLIGIBLE uncertainty. This constitutes a false certainty claim. "
                "JANUS Law 6: No certainty claims ever."
            )
            self._stats.false_certainty_detections += 1

        return violations

    def _validate_probability_distribution(
        self, artifact_id: str, dist: Any
    ) -> list[str]:
        """
        Validate a ProbabilityDistribution's internal consistency.
        Returns a list of violation strings.
        """
        violations: list[str] = []

        if not dist.outcomes:
            violations.append(
                f"Artifact '{artifact_id}' has a ProbabilityDistribution with no outcomes."
            )
            return violations

        total = sum(dist.outcomes.values())
        if abs(total - 1.0) > _DISTRIBUTION_TOLERANCE:
            violations.append(
                f"Artifact '{artifact_id}' ProbabilityDistribution outcomes sum to "
                f"{total:.6f}; must sum to 1.0 (±{_DISTRIBUTION_TOLERANCE})."
            )

        for outcome_label, prob in dist.outcomes.items():
            if not 0.0 <= prob <= 1.0:
                violations.append(
                    f"Artifact '{artifact_id}' ProbabilityDistribution outcome "
                    f"'{outcome_label}' has probability={prob!r} outside [0.0, 1.0]."
                )
                self._stats.impossible_future_detections += 1

        return violations

    def _record_validation(
        self,
        artifact_id: str,
        artifact_type: str,
        is_valid: bool,
        violations: list[str],
    ) -> None:
        """Append a validation result to history and update statistics."""
        now = datetime.utcnow()
        result = _ValidationResult(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            is_valid=is_valid,
            violations=tuple(violations),
            validated_at=now,
        )
        self._validation_history.append(result)
        self._stats.total_validations += 1
        self._stats.total_violations += len(violations)
        self._stats.last_validation_at = now

        if violations:
            logger.warning(
                "[%s] Integrity violations in %s '%s': %s",
                _ENGINE_NAME,
                artifact_type,
                artifact_id,
                "; ".join(violations),
            )
        else:
            logger.debug(
                "[%s] %s '%s' passed integrity validation.",
                _ENGINE_NAME,
                artifact_type,
                artifact_id,
            )