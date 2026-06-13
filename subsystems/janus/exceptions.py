"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: exceptions.py

Complete exception hierarchy for JANUS.
All JANUS operations raise from JanusError as the common root.
Subsystem-specific branches cover each engine domain.

Constitutional boundary: JANUS owns scenario modeling, future modeling,
forecasting, branch analysis, counterfactual analysis, uncertainty modeling,
risk/opportunity analysis, outcome simulation, strategic forecasting,
timeline projection, and probability estimation.
JANUS never owns decisions, plans, identity, or knowledge storage.
"""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class JanusError(Exception):
    """
    Root exception for all JANUS failures.
    Every JANUS engine raises a subclass of JanusError.
    """

    def __init__(
        self,
        message: str,
        *,
        engine: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.engine: Optional[str] = engine
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.engine:
            return f"[{self.engine}] {base}"
        return base


# ---------------------------------------------------------------------------
# Lifecycle Errors
# ---------------------------------------------------------------------------


class JanusLifecycleError(JanusError):
    """Raised when a JANUS operation is called in an invalid lifecycle state."""


class JanusNotInitializedError(JanusLifecycleError):
    """Raised when an engine method is invoked before the engine is initialized."""

    def __init__(self, engine: str) -> None:
        super().__init__(
            f"Engine '{engine}' has not been initialized. Call initialize() first.",
            engine=engine,
        )


class JanusAlreadyInitializedError(JanusLifecycleError):
    """Raised when initialize() is called on an already-initialized engine."""

    def __init__(self, engine: str) -> None:
        super().__init__(
            f"Engine '{engine}' is already initialized.",
            engine=engine,
        )


class JanusShutdownError(JanusLifecycleError):
    """Raised when an operation is attempted on a shut-down engine."""

    def __init__(self, engine: str) -> None:
        super().__init__(
            f"Engine '{engine}' has been shut down and cannot accept new operations.",
            engine=engine,
        )


# ---------------------------------------------------------------------------
# Validation Errors
# ---------------------------------------------------------------------------


class JanusValidationError(JanusError):
    """Raised when input data fails validation before processing."""

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Any = None,
        engine: Optional[str] = None,
    ) -> None:
        super().__init__(message, engine=engine, context={"field": field, "value": value})
        self.field = field
        self.value = value


class JanusMissingRequiredFieldError(JanusValidationError):
    """Raised when a required input field is absent or None."""

    def __init__(self, field: str, engine: Optional[str] = None) -> None:
        super().__init__(
            f"Required field '{field}' is missing.",
            field=field,
            engine=engine,
        )


class JanusInvalidProbabilityError(JanusValidationError):
    """Raised when a probability value is outside [0.0, 1.0]."""

    def __init__(self, value: float, field: str = "probability", engine: Optional[str] = None) -> None:
        super().__init__(
            f"Invalid probability value {value!r} for field '{field}'. Must be in [0.0, 1.0].",
            field=field,
            value=value,
            engine=engine,
        )


class JanusInvalidScoreError(JanusValidationError):
    """Raised when a normalized score (impact, value, feasibility, etc.) is out of range."""

    def __init__(self, value: float, field: str, engine: Optional[str] = None) -> None:
        super().__init__(
            f"Score value {value!r} for field '{field}' is out of range. Must be in [0.0, 1.0].",
            field=field,
            value=value,
            engine=engine,
        )


class JanusInvalidDeltaError(JanusValidationError):
    """Raised when a delta value (delta_risk, delta_opportunity) is outside [-1.0, 1.0]."""

    def __init__(self, value: float, field: str, engine: Optional[str] = None) -> None:
        super().__init__(
            f"Delta value {value!r} for field '{field}' is out of range. Must be in [-1.0, 1.0].",
            field=field,
            value=value,
            engine=engine,
        )


class JanusProbabilityDistributionError(JanusValidationError):
    """Raised when a ProbabilityDistribution's outcomes do not sum to 1.0."""

    def __init__(self, total: float, engine: Optional[str] = None) -> None:
        super().__init__(
            f"ProbabilityDistribution outcomes sum to {total:.6f}; must sum to 1.0 (±1e-6).",
            field="outcomes",
            value=total,
            engine=engine,
        )


# ---------------------------------------------------------------------------
# Scenario Engine Errors
# ---------------------------------------------------------------------------


class JanusScenarioError(JanusError):
    """Base class for all Scenario Engine failures."""


class JanusScenarioNotFoundError(JanusScenarioError):
    """Raised when a requested scenario_id does not exist."""

    def __init__(self, scenario_id: str) -> None:
        super().__init__(
            f"Scenario '{scenario_id}' not found.",
            engine="ScenarioEngine",
            context={"scenario_id": scenario_id},
        )
        self.scenario_id = scenario_id


class JanusScenarioGenerationError(JanusScenarioError):
    """Raised when scenario generation fails for a given context or decision."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Scenario generation failed: {reason}",
            engine="ScenarioEngine",
            context=context,
        )


class JanusScenarioStatusTransitionError(JanusScenarioError):
    """Raised when an illegal ScenarioStatus transition is attempted."""

    def __init__(self, scenario_id: str, current: str, target: str) -> None:
        super().__init__(
            f"Cannot transition scenario '{scenario_id}' from '{current}' to '{target}'.",
            engine="ScenarioEngine",
            context={"scenario_id": scenario_id, "current": current, "target": target},
        )


class JanusScenarioBranchError(JanusScenarioError):
    """Raised when a ScenarioBranch is malformed or inconsistent."""

    def __init__(self, branch_id: str, reason: str) -> None:
        super().__init__(
            f"Branch '{branch_id}' is invalid: {reason}",
            engine="ScenarioEngine",
            context={"branch_id": branch_id},
        )


class JanusScenarioLimitExceededError(JanusScenarioError):
    """Raised when scenario generation would exceed the configured combinatorial limit."""

    def __init__(self, current: int, limit: int) -> None:
        super().__init__(
            f"Scenario count {current} would exceed the configured limit of {limit}.",
            engine="ScenarioEngine",
            context={"current": current, "limit": limit},
        )


# ---------------------------------------------------------------------------
# Forecast Engine Errors
# ---------------------------------------------------------------------------


class JanusForecastError(JanusError):
    """Base class for all Forecasting Engine failures."""


class JanusForecastNotFoundError(JanusForecastError):
    """Raised when a requested forecast_id does not exist."""

    def __init__(self, forecast_id: str) -> None:
        super().__init__(
            f"Forecast '{forecast_id}' not found.",
            engine="ForecastingEngine",
            context={"forecast_id": forecast_id},
        )
        self.forecast_id = forecast_id


class JanusForecastGenerationError(JanusForecastError):
    """Raised when forecast generation fails due to insufficient data or model error."""

    def __init__(self, reason: str, horizon: Optional[str] = None) -> None:
        super().__init__(
            f"Forecast generation failed: {reason}",
            engine="ForecastingEngine",
            context={"horizon": horizon},
        )


class JanusForecastHorizonError(JanusForecastError):
    """Raised when an unsupported or invalid ForecastHorizon is specified."""

    def __init__(self, horizon: Any) -> None:
        super().__init__(
            f"Invalid or unsupported forecast horizon: {horizon!r}",
            engine="ForecastingEngine",
            context={"horizon": horizon},
        )


class JanusForecastSupersededError(JanusForecastError):
    """Raised when an operation is attempted on a superseded forecast."""

    def __init__(self, forecast_id: str, superseded_by: str) -> None:
        super().__init__(
            f"Forecast '{forecast_id}' has been superseded by '{superseded_by}'.",
            engine="ForecastingEngine",
            context={"forecast_id": forecast_id, "superseded_by": superseded_by},
        )


class JanusForecastAssessmentError(JanusForecastError):
    """Raised when forecast assessment or accuracy tracking fails."""

    def __init__(self, forecast_id: str, reason: str) -> None:
        super().__init__(
            f"Forecast assessment failed for '{forecast_id}': {reason}",
            engine="ForecastingEngine",
            context={"forecast_id": forecast_id},
        )


# ---------------------------------------------------------------------------
# Future Model Engine Errors
# ---------------------------------------------------------------------------


class JanusFutureModelError(JanusError):
    """Base class for all Future Modeling Engine failures."""


class JanusFutureModelNotFoundError(JanusFutureModelError):
    """Raised when a requested model_id does not exist."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"FutureModel '{model_id}' not found.",
            engine="FutureModelingEngine",
            context={"model_id": model_id},
        )
        self.model_id = model_id


class JanusFutureModelConstructionError(JanusFutureModelError):
    """Raised when FutureModel construction fails due to invalid or incomplete state data."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"FutureModel construction failed: {reason}",
            engine="FutureModelingEngine",
            context=context,
        )


class JanusFutureStateError(JanusFutureModelError):
    """Raised when a FutureState is invalid or conflicts with the enclosing model."""

    def __init__(self, state_id: str, reason: str) -> None:
        super().__init__(
            f"FutureState '{state_id}' is invalid: {reason}",
            engine="FutureModelingEngine",
            context={"state_id": state_id},
        )


class JanusFutureModelHorizonConflictError(JanusFutureModelError):
    """Raised when FutureStates within a model carry conflicting or duplicate horizons."""

    def __init__(self, model_id: str, horizon: str) -> None:
        super().__init__(
            f"FutureModel '{model_id}' has conflicting states at horizon '{horizon}'.",
            engine="FutureModelingEngine",
            context={"model_id": model_id, "horizon": horizon},
        )


# ---------------------------------------------------------------------------
# Simulation Engine Errors
# ---------------------------------------------------------------------------


class JanusSimulationError(JanusError):
    """Base class for all Outcome Simulation Engine failures."""


class JanusSimulationNotFoundError(JanusSimulationError):
    """Raised when a requested simulation_id does not exist."""

    def __init__(self, simulation_id: str) -> None:
        super().__init__(
            f"OutcomeSimulation '{simulation_id}' not found.",
            engine="OutcomeSimulationEngine",
            context={"simulation_id": simulation_id},
        )
        self.simulation_id = simulation_id


class JanusSimulationExecutionError(JanusSimulationError):
    """Raised when a simulation fails during execution."""

    def __init__(self, simulation_id: str, reason: str) -> None:
        super().__init__(
            f"Simulation '{simulation_id}' failed during execution: {reason}",
            engine="OutcomeSimulationEngine",
            context={"simulation_id": simulation_id},
        )


class JanusSimulationAbortedError(JanusSimulationError):
    """Raised when a simulation is aborted before completion."""

    def __init__(self, simulation_id: str, reason: str) -> None:
        super().__init__(
            f"Simulation '{simulation_id}' was aborted: {reason}",
            engine="OutcomeSimulationEngine",
            context={"simulation_id": simulation_id},
        )


class JanusSimulationStatusError(JanusSimulationError):
    """Raised when an illegal SimulationStatus transition is attempted."""

    def __init__(self, simulation_id: str, current: str, target: str) -> None:
        super().__init__(
            f"Cannot transition simulation '{simulation_id}' from '{current}' to '{target}'.",
            engine="OutcomeSimulationEngine",
            context={"simulation_id": simulation_id, "current": current, "target": target},
        )


class JanusSimulationOutcomeError(JanusSimulationError):
    """Raised when a SimulationOutcome is malformed or its probabilities are inconsistent."""

    def __init__(self, outcome_id: str, reason: str) -> None:
        super().__init__(
            f"SimulationOutcome '{outcome_id}' is invalid: {reason}",
            engine="OutcomeSimulationEngine",
            context={"outcome_id": outcome_id},
        )


# ---------------------------------------------------------------------------
# Probability Engine Errors
# ---------------------------------------------------------------------------


class JanusProbabilityError(JanusError):
    """Base class for all Probability Engine failures."""


class JanusProbabilityComputationError(JanusProbabilityError):
    """Raised when probability computation fails due to model or data error."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Probability computation failed: {reason}",
            engine="ProbabilityEngine",
            context=context,
        )


class JanusProbabilityNormalizationError(JanusProbabilityError):
    """Raised when a probability distribution cannot be normalized to sum to 1.0."""

    def __init__(self, total: float) -> None:
        super().__init__(
            f"Probability distribution cannot be normalized; current total: {total:.6f}.",
            engine="ProbabilityEngine",
            context={"total": total},
        )


class JanusProbabilityDistributionNotFoundError(JanusProbabilityError):
    """Raised when a requested distribution_id does not exist."""

    def __init__(self, distribution_id: str) -> None:
        super().__init__(
            f"ProbabilityDistribution '{distribution_id}' not found.",
            engine="ProbabilityEngine",
            context={"distribution_id": distribution_id},
        )


class JanusConfidenceProfileError(JanusProbabilityError):
    """Raised when a ConfidenceProfile is invalid or inconsistent."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"ConfidenceProfile is invalid: {reason}",
            engine="ProbabilityEngine",
            context=context,
        )


# ---------------------------------------------------------------------------
# Counterfactual Engine Errors
# ---------------------------------------------------------------------------


class JanusCounterfactualError(JanusError):
    """Base class for all Counterfactual Engine failures."""


class JanusCounterfactualNotFoundError(JanusCounterfactualError):
    """Raised when a requested counterfactual_id does not exist."""

    def __init__(self, counterfactual_id: str) -> None:
        super().__init__(
            f"CounterfactualScenario '{counterfactual_id}' not found.",
            engine="CounterfactualEngine",
            context={"counterfactual_id": counterfactual_id},
        )
        self.counterfactual_id = counterfactual_id


class JanusCounterfactualConstructionError(JanusCounterfactualError):
    """Raised when a counterfactual scenario cannot be constructed from the given inputs."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Counterfactual construction failed: {reason}",
            engine="CounterfactualEngine",
            context=context,
        )


class JanusCounterfactualDivergenceError(JanusCounterfactualError):
    """Raised when the divergence point is temporally invalid (e.g., in the future)."""

    def __init__(self, divergence_point: str, reason: str) -> None:
        super().__init__(
            f"Invalid divergence point '{divergence_point}': {reason}",
            engine="CounterfactualEngine",
            context={"divergence_point": divergence_point},
        )


class JanusCounterfactualConditionError(JanusCounterfactualError):
    """Raised when the counterfactual condition is logically impossible or self-contradictory."""

    def __init__(self, condition: str, reason: str) -> None:
        super().__init__(
            f"Counterfactual condition '{condition}' is invalid: {reason}",
            engine="CounterfactualEngine",
            context={"condition": condition},
        )


# ---------------------------------------------------------------------------
# Risk Analysis Engine Errors
# ---------------------------------------------------------------------------


class JanusRiskError(JanusError):
    """Base class for all Future Risk Engine failures."""


class JanusRiskAssessmentNotFoundError(JanusRiskError):
    """Raised when a requested risk_id does not exist."""

    def __init__(self, risk_id: str) -> None:
        super().__init__(
            f"RiskAssessment '{risk_id}' not found.",
            engine="FutureRiskEngine",
            context={"risk_id": risk_id},
        )
        self.risk_id = risk_id


class JanusRiskAnalysisError(JanusRiskError):
    """Raised when risk analysis fails due to insufficient data or computation error."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Risk analysis failed: {reason}",
            engine="FutureRiskEngine",
            context=context,
        )


class JanusRiskFactorError(JanusRiskError):
    """Raised when a RiskFactor is malformed or its scores are invalid."""

    def __init__(self, factor_id: str, reason: str) -> None:
        super().__init__(
            f"RiskFactor '{factor_id}' is invalid: {reason}",
            engine="FutureRiskEngine",
            context={"factor_id": factor_id},
        )


class JanusRiskLevelError(JanusRiskError):
    """Raised when a risk level assignment is inconsistent with the underlying factor scores."""

    def __init__(self, assigned: str, computed: str) -> None:
        super().__init__(
            f"Assigned RiskLevel '{assigned}' is inconsistent with computed level '{computed}'.",
            engine="FutureRiskEngine",
            context={"assigned": assigned, "computed": computed},
        )


# ---------------------------------------------------------------------------
# Opportunity Analysis Engine Errors
# ---------------------------------------------------------------------------


class JanusOpportunityError(JanusError):
    """Base class for all Future Opportunity Engine failures."""


class JanusOpportunityAssessmentNotFoundError(JanusOpportunityError):
    """Raised when a requested opportunity_id does not exist."""

    def __init__(self, opportunity_id: str) -> None:
        super().__init__(
            f"OpportunityAssessment '{opportunity_id}' not found.",
            engine="FutureOpportunityEngine",
            context={"opportunity_id": opportunity_id},
        )
        self.opportunity_id = opportunity_id


class JanusOpportunityAnalysisError(JanusOpportunityError):
    """Raised when opportunity analysis fails due to insufficient data or computation error."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Opportunity analysis failed: {reason}",
            engine="FutureOpportunityEngine",
            context=context,
        )


class JanusOpportunityFactorError(JanusOpportunityError):
    """Raised when an OpportunityFactor is malformed or its scores are invalid."""

    def __init__(self, factor_id: str, reason: str) -> None:
        super().__init__(
            f"OpportunityFactor '{factor_id}' is invalid: {reason}",
            engine="FutureOpportunityEngine",
            context={"factor_id": factor_id},
        )


class JanusOpportunityLevelError(JanusOpportunityError):
    """Raised when an opportunity level assignment is inconsistent with underlying factor scores."""

    def __init__(self, assigned: str, computed: str) -> None:
        super().__init__(
            f"Assigned OpportunityLevel '{assigned}' is inconsistent with computed level '{computed}'.",
            engine="FutureOpportunityEngine",
            context={"assigned": assigned, "computed": computed},
        )


# ---------------------------------------------------------------------------
# Timeline Projection Engine Errors
# ---------------------------------------------------------------------------


class JanusTimelineProjectionError(JanusError):
    """Base class for all Timeline Projection Engine failures."""


class JanusTimelineProjectionNotFoundError(JanusTimelineProjectionError):
    """Raised when a requested projection_id does not exist."""

    def __init__(self, projection_id: str) -> None:
        super().__init__(
            f"TimelineProjection '{projection_id}' not found.",
            engine="TimelineProjectionEngine",
            context={"projection_id": projection_id},
        )
        self.projection_id = projection_id


class JanusTimelineProjectionConstructionError(JanusTimelineProjectionError):
    """Raised when timeline projection construction fails."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"TimelineProjection construction failed: {reason}",
            engine="TimelineProjectionEngine",
            context=context,
        )


class JanusProjectionMilestoneError(JanusTimelineProjectionError):
    """Raised when a ProjectionMilestone is malformed or its dependencies are broken."""

    def __init__(self, milestone_id: str, reason: str) -> None:
        super().__init__(
            f"ProjectionMilestone '{milestone_id}' is invalid: {reason}",
            engine="TimelineProjectionEngine",
            context={"milestone_id": milestone_id},
        )


class JanusProjectionStatusTransitionError(JanusTimelineProjectionError):
    """Raised when an illegal ProjectionStatus transition is attempted."""

    def __init__(self, projection_id: str, current: str, target: str) -> None:
        super().__init__(
            f"Cannot transition projection '{projection_id}' from '{current}' to '{target}'.",
            engine="TimelineProjectionEngine",
            context={"projection_id": projection_id, "current": current, "target": target},
        )


class JanusChronosOwnershipViolationError(JanusTimelineProjectionError):
    """
    Raised when a Timeline Projection operation attempts to manage time data,
    which is owned exclusively by CHRONOS.
    Constitutional rule: CHRONOS owns time; JANUS predicts future states across time.
    """

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"Operation '{operation}' violates constitutional boundary: "
            "CHRONOS owns time. JANUS may only project future states across time.",
            engine="TimelineProjectionEngine",
            context={"operation": operation},
        )


# ---------------------------------------------------------------------------
# Uncertainty Engine Errors
# ---------------------------------------------------------------------------


class JanusUncertaintyError(JanusError):
    """Base class for all Uncertainty Engine failures."""


class JanusUncertaintyProfileNotFoundError(JanusUncertaintyError):
    """Raised when a requested uncertainty_id does not exist."""

    def __init__(self, uncertainty_id: str) -> None:
        super().__init__(
            f"UncertaintyProfile '{uncertainty_id}' not found.",
            engine="UncertaintyEngine",
            context={"uncertainty_id": uncertainty_id},
        )


class JanusUncertaintyModelingError(JanusUncertaintyError):
    """Raised when uncertainty modeling fails due to missing or contradictory inputs."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Uncertainty modeling failed: {reason}",
            engine="UncertaintyEngine",
            context=context,
        )


class JanusMissingUncertaintyError(JanusUncertaintyError):
    """
    Raised when a forecast or scenario is submitted without a required UncertaintyProfile.
    Constitutional rule: All forecasts must include uncertainty. No certainty claims ever.
    """

    def __init__(self, artifact_id: str, artifact_type: str) -> None:
        super().__init__(
            f"{artifact_type} '{artifact_id}' is missing a required UncertaintyProfile. "
            "All JANUS forecasts must carry uncertainty.",
            engine="UncertaintyEngine",
            context={"artifact_id": artifact_id, "artifact_type": artifact_type},
        )


# ---------------------------------------------------------------------------
# Branch Analysis Engine Errors
# ---------------------------------------------------------------------------


class JanusBranchAnalysisError(JanusError):
    """Base class for all Branch Analysis Engine failures."""


class JanusBranchNotFoundError(JanusBranchAnalysisError):
    """Raised when a requested branch_id does not exist."""

    def __init__(self, branch_id: str) -> None:
        super().__init__(
            f"ScenarioBranch '{branch_id}' not found.",
            engine="BranchAnalysisEngine",
            context={"branch_id": branch_id},
        )
        self.branch_id = branch_id


class JanusBranchConstructionError(JanusBranchAnalysisError):
    """Raised when branch construction fails due to inconsistent or incomplete data."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Branch construction failed: {reason}",
            engine="BranchAnalysisEngine",
            context=context,
        )


class JanusBranchProbabilityConflictError(JanusBranchAnalysisError):
    """Raised when branch probabilities within a scenario are inconsistent."""

    def __init__(self, scenario_id: str, total: float) -> None:
        super().__init__(
            f"Branch probabilities in scenario '{scenario_id}' are inconsistent: total={total:.6f}.",
            engine="BranchAnalysisEngine",
            context={"scenario_id": scenario_id, "total": total},
        )


# ---------------------------------------------------------------------------
# Scenario Evaluation Engine Errors
# ---------------------------------------------------------------------------


class JanusScenarioEvaluationError(JanusError):
    """Base class for all Scenario Evaluation Engine failures."""


class JanusScenarioComparisonError(JanusScenarioEvaluationError):
    """Raised when scenario comparison fails due to incompatible or incomplete scenarios."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Scenario comparison failed: {reason}",
            engine="ScenarioEvaluationEngine",
            context=context,
        )


class JanusScenarioComparisonNotFoundError(JanusScenarioEvaluationError):
    """Raised when a requested comparison_id does not exist."""

    def __init__(self, comparison_id: str) -> None:
        super().__init__(
            f"ScenarioComparison '{comparison_id}' not found.",
            engine="ScenarioEvaluationEngine",
            context={"comparison_id": comparison_id},
        )


class JanusScenarioRankingError(JanusScenarioEvaluationError):
    """Raised when scenario ranking cannot be determined due to missing scores."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Scenario ranking failed: {reason}",
            engine="ScenarioEvaluationEngine",
        )


class JanusApprovalBoundaryViolationError(JanusScenarioEvaluationError):
    """
    Raised when evaluation logic attempts to approve or select a scenario.
    Constitutional rule: JANUS evaluates futures; it never approves them. Approval belongs to VEGA.
    """

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"Operation '{operation}' violates constitutional boundary: "
            "JANUS evaluates futures; it never approves them. Approval belongs to VEGA.",
            engine="ScenarioEvaluationEngine",
            context={"operation": operation},
        )


# ---------------------------------------------------------------------------
# Strategic Forecast Engine Errors
# ---------------------------------------------------------------------------


class JanusStrategicForecastError(JanusError):
    """Base class for all Strategic Forecast Engine failures."""


class JanusStrategicForecastNotFoundError(JanusStrategicForecastError):
    """Raised when a requested strategic_forecast_id does not exist."""

    def __init__(self, strategic_forecast_id: str) -> None:
        super().__init__(
            f"StrategicForecast '{strategic_forecast_id}' not found.",
            engine="StrategicForecastEngine",
            context={"strategic_forecast_id": strategic_forecast_id},
        )


class JanusStrategicForecastConstructionError(JanusStrategicForecastError):
    """Raised when strategic forecast construction fails."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"StrategicForecast construction failed: {reason}",
            engine="StrategicForecastEngine",
            context=context,
        )


class JanusStrategySelectionViolationError(JanusStrategicForecastError):
    """
    Raised when strategic forecasting logic attempts to select a strategy.
    Constitutional rule: JANUS forecasts strategic outcomes; ODYSSEY chooses strategic direction.
    JANUS never selects strategy.
    """

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"Operation '{operation}' violates constitutional boundary: "
            "JANUS forecasts strategic outcomes; ODYSSEY chooses strategic direction. "
            "JANUS never selects strategy.",
            engine="StrategicForecastEngine",
            context={"operation": operation},
        )


# ---------------------------------------------------------------------------
# Scenario Integrity Engine Errors
# ---------------------------------------------------------------------------


class JanusIntegrityError(JanusError):
    """Base class for all Scenario Integrity Engine failures."""


class JanusImpossibleFutureError(JanusIntegrityError):
    """Raised when a scenario or future model describes a logically impossible future."""

    def __init__(self, artifact_id: str, artifact_type: str, reason: str) -> None:
        super().__init__(
            f"{artifact_type} '{artifact_id}' describes an impossible future: {reason}",
            engine="ScenarioIntegrityEngine",
            context={"artifact_id": artifact_id, "artifact_type": artifact_type},
        )


class JanusUnsupportedPredictionError(JanusIntegrityError):
    """Raised when a forecast or prediction lacks evidential support."""

    def __init__(self, artifact_id: str, artifact_type: str) -> None:
        super().__init__(
            f"{artifact_type} '{artifact_id}' contains unsupported predictions. "
            "All forecasts must remain evidence-based.",
            engine="ScenarioIntegrityEngine",
            context={"artifact_id": artifact_id, "artifact_type": artifact_type},
        )


class JanusContradictoryOutcomeError(JanusIntegrityError):
    """Raised when two or more outcomes within a forecast or scenario are mutually contradictory."""

    def __init__(self, artifact_id: str, outcome_a: str, outcome_b: str) -> None:
        super().__init__(
            f"Contradictory outcomes detected in '{artifact_id}': "
            f"'{outcome_a}' conflicts with '{outcome_b}'.",
            engine="ScenarioIntegrityEngine",
            context={"artifact_id": artifact_id, "outcome_a": outcome_a, "outcome_b": outcome_b},
        )


class JanusFalseCertaintyError(JanusIntegrityError):
    """
    Raised when a forecast or scenario claims certainty (probability == 1.0 with no uncertainty).
    Constitutional rule: All forecasts require uncertainty. No certainty claims ever.
    """

    def __init__(self, artifact_id: str, artifact_type: str) -> None:
        super().__init__(
            f"{artifact_type} '{artifact_id}' makes a false certainty claim. "
            "JANUS Law 6: All forecasts require uncertainty. No certainty claims ever.",
            engine="ScenarioIntegrityEngine",
            context={"artifact_id": artifact_id, "artifact_type": artifact_type},
        )


class JanusEvidenceProfileError(JanusIntegrityError):
    """Raised when an EvidenceProfile is missing, empty, or fails strength validation."""

    def __init__(self, artifact_id: str, reason: str) -> None:
        super().__init__(
            f"EvidenceProfile for '{artifact_id}' is invalid: {reason}",
            engine="ScenarioIntegrityEngine",
            context={"artifact_id": artifact_id},
        )


class JanusConstitutionalViolationError(JanusIntegrityError):
    """
    Raised when any JANUS engine attempts an operation that violates the
    POLARIS constitutional separation of responsibilities.
    """

    def __init__(self, engine: str, operation: str, rightful_owner: str) -> None:
        super().__init__(
            f"Engine '{engine}' attempted operation '{operation}', "
            f"which belongs to '{rightful_owner}'. Constitutional boundary violated.",
            engine=engine,
            context={"operation": operation, "rightful_owner": rightful_owner},
        )


# ---------------------------------------------------------------------------
# Future Memory Interface Errors
# ---------------------------------------------------------------------------


class JanusFutureMemoryError(JanusError):
    """Base class for all Future Memory Interface failures."""


class JanusFutureMemoryReadError(JanusFutureMemoryError):
    """Raised when JANUS cannot read historical outcome data from upstream subsystems."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(
            f"Failed to read historical data from '{source}': {reason}",
            engine="FutureMemoryInterface",
            context={"source": source},
        )
        self.source = source


class JanusFutureMemoryOwnershipViolationError(JanusFutureMemoryError):
    """
    Raised when the Future Memory Interface attempts to write or own memory data.
    Constitutional rule: JANUS does not own memory; it consumes historical outcomes.
    """

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"Operation '{operation}' violates constitutional boundary: "
            "JANUS does not own memory. It consumes historical outcomes to improve forecasting.",
            engine="FutureMemoryInterface",
            context={"operation": operation},
        )


class JanusFutureMemoryStaleDataError(JanusFutureMemoryError):
    """Raised when historical data consumed from upstream is detected as stale or outdated."""

    def __init__(self, source: str, age_description: str) -> None:
        super().__init__(
            f"Historical data from '{source}' is stale: {age_description}",
            engine="FutureMemoryInterface",
            context={"source": source, "age_description": age_description},
        )


# ---------------------------------------------------------------------------
# Future Orchestrator Errors
# ---------------------------------------------------------------------------


class JanusOrchestratorError(JanusError):
    """Base class for all Future Orchestrator failures."""


class JanusOrchestratorCoordinationError(JanusOrchestratorError):
    """Raised when the orchestrator cannot coordinate engine activities due to a subsystem failure."""

    def __init__(self, reason: str, failed_engine: Optional[str] = None) -> None:
        super().__init__(
            f"Orchestrator coordination failed: {reason}",
            engine="FutureOrchestrator",
            context={"failed_engine": failed_engine},
        )


class JanusOrchestratorCycleDetectedError(JanusOrchestratorError):
    """Raised when the orchestrator detects a dependency cycle among engine activations."""

    def __init__(self, cycle: list[str]) -> None:
        cycle_str = " → ".join(cycle)
        super().__init__(
            f"Dependency cycle detected in engine activation sequence: {cycle_str}",
            engine="FutureOrchestrator",
            context={"cycle": cycle},
        )


class JanusOrchestratorTimeoutError(JanusOrchestratorError):
    """Raised when an orchestrated operation exceeds the configured timeout."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Orchestrated operation '{operation}' timed out after {timeout_seconds}s.",
            engine="FutureOrchestrator",
            context={"operation": operation, "timeout_seconds": timeout_seconds},
        )


class JanusAssessmentNotFoundError(JanusOrchestratorError):
    """Raised when a requested assessment_id does not exist."""

    def __init__(self, assessment_id: str) -> None:
        super().__init__(
            f"FutureAssessment '{assessment_id}' not found.",
            engine="FutureOrchestrator",
            context={"assessment_id": assessment_id},
        )
        self.assessment_id = assessment_id


class JanusAssessmentStatusTransitionError(JanusOrchestratorError):
    """Raised when an illegal FutureAssessmentStatus transition is attempted."""

    def __init__(self, assessment_id: str, current: str, target: str) -> None:
        super().__init__(
            f"Cannot transition assessment '{assessment_id}' from '{current}' to '{target}'.",
            engine="FutureOrchestrator",
            context={"assessment_id": assessment_id, "current": current, "target": target},
        )