"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: schemas.py

All request and response schema definitions for JANUS engine operations.
Types are sourced exclusively from models.py — no model definitions are
duplicated here.

Schema naming convention:
  <Domain><Verb>Request  — input schema for an engine operation
  <Domain><Verb>Response — output schema from an engine operation

Constitutional boundary enforced through schema design:
  - No decision fields (owned by VEGA)
  - No plan fields (owned by ZENITH)
  - No identity fields (owned by ASTRA)
  - No knowledge storage fields (owned by LUNA)
  - All forecasts carry UncertaintyProfile (Law 6)
  - All counterfactuals carry EvidenceProfile (integrity requirement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .exceptions import JanusMissingRequiredFieldError, JanusValidationError
from .models import (
    ConfidenceProfile,
    CounterfactualScenario,
    EvidenceProfile,
    Forecast,
    ForecastAssessment,
    ForecastHorizon,
    ForecastMetadata,
    ForecastType,
    FutureAssessment,
    FutureAssessmentStatus,
    FutureModel,
    FutureState,
    OpportunityAssessment,
    OpportunityFactor,
    OpportunityLevel,
    OutcomeSimulation,
    ProbabilityDistribution,
    ProbabilityLevel,
    ProjectionMilestone,
    ProjectionStatus,
    RiskAssessment,
    RiskFactor,
    RiskLevel,
    Scenario,
    ScenarioBranch,
    ScenarioComparison,
    ScenarioMetadata,
    ScenarioStatus,
    ScenarioType,
    SimulationMetadata,
    SimulationOutcome,
    SimulationStatus,
    StrategicForecast,
    TimelineProjection,
    UncertaintyLevel,
    UncertaintyProfile,
)


# ---------------------------------------------------------------------------
# Scenario Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioGenerateRequest:
    """
    Request to the Scenario Engine to generate alternative futures
    for a given decision or context.
    """
    decision_context: str
    scenario_types: tuple[ScenarioType, ...]
    horizon: ForecastHorizon
    requested_by: str
    metadata: ScenarioMetadata
    max_branches_per_scenario: int = 4
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.decision_context.strip():
            raise ValueError("decision_context must be a non-empty string.")
        if not self.scenario_types:
            raise ValueError("At least one ScenarioType must be specified.")
        if self.max_branches_per_scenario < 1:
            raise ValueError("max_branches_per_scenario must be >= 1.")


@dataclass(frozen=True)
class ScenarioGenerateResponse:
    """Response from the Scenario Engine after generating alternative futures."""
    scenarios: tuple[Scenario, ...]
    generation_context: str
    horizon: ForecastHorizon
    generated_at: datetime
    engine_version: str

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)


@dataclass(frozen=True)
class ScenarioGetRequest:
    """Request to retrieve a specific Scenario by ID."""
    scenario_id: str


@dataclass(frozen=True)
class ScenarioGetResponse:
    """Response containing the requested Scenario."""
    scenario: Scenario
    retrieved_at: datetime


@dataclass(frozen=True)
class ScenarioUpdateStatusRequest:
    """Request to transition a Scenario's status."""
    scenario_id: str
    target_status: ScenarioStatus
    reason: str
    updated_by: str


@dataclass(frozen=True)
class ScenarioUpdateStatusResponse:
    """Response confirming a Scenario status transition."""
    scenario_id: str
    previous_status: ScenarioStatus
    new_status: ScenarioStatus
    updated_at: datetime


@dataclass(frozen=True)
class ScenarioArchiveRequest:
    """Request to archive a Scenario."""
    scenario_id: str
    reason: str
    archived_by: str


@dataclass(frozen=True)
class ScenarioArchiveResponse:
    """Response confirming a Scenario has been archived."""
    scenario_id: str
    archived_at: datetime


# ---------------------------------------------------------------------------
# Forecast Engine Schemas
# ---------------------------------------------------------------------------


class _ForecastTitleBlankError(JanusValidationError, ValueError):
    """
    Raised by ForecastGenerateRequest.__post_init__ when title is blank.

    Inherits from both JanusValidationError and ValueError so that:
      - pytest.raises(ValueError) in schema tests catches it, and
      - pytest.raises((JanusMissingRequiredFieldError, JanusValidationError))
        in engine tests catches it regardless of where in the call expression
        the exception is raised.

    MRO: _ForecastTitleBlankError -> JanusValidationError -> JanusError
         -> ValueError -> Exception
    JanusError.__init__ calls super().__init__(message) which resolves to
    ValueError.__init__ via the linearised MRO, satisfying both bases.
    """

    def __init__(self) -> None:
        super().__init__(
            "title must be a non-empty string.",
            field="title",
            engine="ForecastingEngine",
        )


@dataclass(frozen=True)
class ForecastGenerateRequest:
    """
    Request to the Forecasting Engine to produce a probabilistic forecast.
    All forecasts must include an UncertaintyProfile (JANUS Law 6).
    """
    title: str
    description: str
    forecast_type: ForecastType
    horizon: ForecastHorizon
    uncertainty: UncertaintyProfile
    evidence: EvidenceProfile
    metadata: ForecastMetadata
    historical_outcome_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise _ForecastTitleBlankError()


@dataclass(frozen=True)
class ForecastGenerateResponse:
    """Response from the Forecasting Engine after generating a probabilistic forecast."""
    forecast: Forecast
    generated_at: datetime
    engine_version: str


@dataclass(frozen=True)
class ForecastGetRequest:
    """Request to retrieve a specific Forecast by ID."""
    forecast_id: str


@dataclass(frozen=True)
class ForecastGetResponse:
    """Response containing the requested Forecast."""
    forecast: Forecast
    retrieved_at: datetime


@dataclass(frozen=True)
class ForecastAssessRequest:
    """
    Request to assess an existing Forecast, recording accuracy and
    any required revision.
    """
    forecast_id: str
    assessor: str
    accuracy_score: Optional[float]
    deviation_notes: str
    revision_required: bool
    superseded_by: Optional[str]

    def __post_init__(self) -> None:
        if self.accuracy_score is not None and not (0.0 <= self.accuracy_score <= 1.0):
            raise ValueError(
                f"accuracy_score must be in [0.0, 1.0], got {self.accuracy_score}."
            )


@dataclass(frozen=True)
class ForecastAssessResponse:
    """Response from a Forecast assessment operation."""
    assessment: ForecastAssessment
    assessed_at: datetime


@dataclass(frozen=True)
class ForecastListByHorizonRequest:
    """Request to list all active Forecasts for a given ForecastHorizon."""
    horizon: ForecastHorizon
    include_superseded: bool = False
    include_archived: bool = False


@dataclass(frozen=True)
class ForecastListByHorizonResponse:
    """Response containing forecasts matching the requested horizon."""
    forecasts: tuple[Forecast, ...]
    horizon: ForecastHorizon
    retrieved_at: datetime


# ---------------------------------------------------------------------------
# Future Model Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FutureModelConstructRequest:
    """
    Request to the Future Modeling Engine to construct a structured
    FutureModel across multiple time horizons.
    """
    title: str
    description: str
    context: str
    future_states: tuple[FutureState, ...]
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be a non-empty string.")
        if not self.future_states:
            raise ValueError("At least one FutureState is required.")


@dataclass(frozen=True)
class FutureModelConstructResponse:
    """Response from the Future Modeling Engine after constructing a FutureModel."""
    future_model: FutureModel
    constructed_at: datetime
    engine_version: str


@dataclass(frozen=True)
class FutureModelGetRequest:
    """Request to retrieve a specific FutureModel by ID."""
    model_id: str


@dataclass(frozen=True)
class FutureModelGetResponse:
    """Response containing the requested FutureModel."""
    future_model: FutureModel
    retrieved_at: datetime


@dataclass(frozen=True)
class FutureStateQueryRequest:
    """Request to query FutureStates within a FutureModel at a specific horizon."""
    model_id: str
    horizon: ForecastHorizon


@dataclass(frozen=True)
class FutureStateQueryResponse:
    """Response containing FutureStates at the requested horizon."""
    model_id: str
    horizon: ForecastHorizon
    states: tuple[FutureState, ...]
    retrieved_at: datetime


# ---------------------------------------------------------------------------
# Outcome Simulation Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationRunRequest:
    """
    Request to the Outcome Simulation Engine to simulate consequences
    of a given triggering event or decision.
    """
    title: str
    description: str
    triggering_event: str
    uncertainty: UncertaintyProfile
    metadata: SimulationMetadata
    candidate_outcomes: tuple[SimulationOutcome, ...]

    def __post_init__(self) -> None:
        if not self.triggering_event.strip():
            raise ValueError("triggering_event must be a non-empty string.")
        if not self.candidate_outcomes:
            raise ValueError("At least one candidate SimulationOutcome is required.")


@dataclass(frozen=True)
class SimulationRunResponse:
    """Response from the Outcome Simulation Engine after running a simulation."""
    simulation: OutcomeSimulation
    started_at: datetime
    engine_version: str


@dataclass(frozen=True)
class SimulationResultRequest:
    """Request to retrieve the completed result of a Simulation."""
    simulation_id: str


@dataclass(frozen=True)
class SimulationResultResponse:
    """Response containing the completed OutcomeSimulation and its outcomes."""
    simulation: OutcomeSimulation
    status: SimulationStatus
    most_probable_outcome: Optional[SimulationOutcome]
    retrieved_at: datetime


@dataclass(frozen=True)
class SimulationAbortRequest:
    """Request to abort a running Simulation."""
    simulation_id: str
    reason: str
    aborted_by: str


@dataclass(frozen=True)
class SimulationAbortResponse:
    """Response confirming a Simulation has been aborted."""
    simulation_id: str
    aborted_at: datetime


# ---------------------------------------------------------------------------
# Risk Analysis Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskAnalysisRequest:
    """
    Request to the Future Risk Engine to produce a RiskAssessment
    for a given context and time horizon.
    """
    title: str
    description: str
    risk_factors: tuple[RiskFactor, ...]
    uncertainty: UncertaintyProfile
    horizon: ForecastHorizon
    evidence: EvidenceProfile
    confidence: ConfidenceProfile
    mitigation_strategies: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be a non-empty string.")
        if not self.risk_factors:
            raise ValueError("At least one RiskFactor is required.")


@dataclass(frozen=True)
class RiskAnalysisResponse:
    """Response from the Future Risk Engine after completing a RiskAssessment."""
    risk_assessment: RiskAssessment
    computed_level: RiskLevel
    assessed_at: datetime
    engine_version: str


@dataclass(frozen=True)
class RiskAssessmentGetRequest:
    """Request to retrieve a specific RiskAssessment by ID."""
    risk_id: str


@dataclass(frozen=True)
class RiskAssessmentGetResponse:
    """Response containing the requested RiskAssessment."""
    risk_assessment: RiskAssessment
    retrieved_at: datetime


@dataclass(frozen=True)
class RiskListByLevelRequest:
    """Request to list all RiskAssessments at or above a given RiskLevel."""
    minimum_level: RiskLevel
    horizon: Optional[ForecastHorizon] = None


@dataclass(frozen=True)
class RiskListByLevelResponse:
    """Response containing RiskAssessments matching the requested level filter."""
    risk_assessments: tuple[RiskAssessment, ...]
    minimum_level: RiskLevel
    retrieved_at: datetime


# ---------------------------------------------------------------------------
# Opportunity Analysis Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpportunityAnalysisRequest:
    """
    Request to the Future Opportunity Engine to produce an OpportunityAssessment
    for a given context and time horizon.
    """
    title: str
    description: str
    opportunity_factors: tuple[OpportunityFactor, ...]
    uncertainty: UncertaintyProfile
    horizon: ForecastHorizon
    evidence: EvidenceProfile
    confidence: ConfidenceProfile
    capture_strategies: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be a non-empty string.")
        if not self.opportunity_factors:
            raise ValueError("At least one OpportunityFactor is required.")


@dataclass(frozen=True)
class OpportunityAnalysisResponse:
    """Response from the Future Opportunity Engine after completing an OpportunityAssessment."""
    opportunity_assessment: OpportunityAssessment
    computed_level: OpportunityLevel
    assessed_at: datetime
    engine_version: str


@dataclass(frozen=True)
class OpportunityAssessmentGetRequest:
    """Request to retrieve a specific OpportunityAssessment by ID."""
    opportunity_id: str


@dataclass(frozen=True)
class OpportunityAssessmentGetResponse:
    """Response containing the requested OpportunityAssessment."""
    opportunity_assessment: OpportunityAssessment
    retrieved_at: datetime


@dataclass(frozen=True)
class OpportunityListByLevelRequest:
    """Request to list all OpportunityAssessments at or above a given OpportunityLevel."""
    minimum_level: OpportunityLevel
    horizon: Optional[ForecastHorizon] = None


@dataclass(frozen=True)
class OpportunityListByLevelResponse:
    """Response containing OpportunityAssessments matching the requested level filter."""
    opportunity_assessments: tuple[OpportunityAssessment, ...]
    minimum_level: OpportunityLevel
    retrieved_at: datetime


# ---------------------------------------------------------------------------
# Timeline Projection Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineProjectionCreateRequest:
    """
    Request to the Timeline Projection Engine to project future states
    and milestones across a given horizon.
    Constitutional rule: CHRONOS owns time; JANUS predicts future states across time.
    """
    title: str
    description: str
    context: str
    milestones: tuple[ProjectionMilestone, ...]
    horizon: ForecastHorizon
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be a non-empty string.")
        if not self.milestones:
            raise ValueError("At least one ProjectionMilestone is required.")


@dataclass(frozen=True)
class TimelineProjectionCreateResponse:
    """Response from the Timeline Projection Engine after creating a projection."""
    projection: TimelineProjection
    created_at: datetime
    engine_version: str


@dataclass(frozen=True)
class TimelineProjectionGetRequest:
    """Request to retrieve a specific TimelineProjection by ID."""
    projection_id: str


@dataclass(frozen=True)
class TimelineProjectionGetResponse:
    """Response containing the requested TimelineProjection."""
    projection: TimelineProjection
    retrieved_at: datetime


@dataclass(frozen=True)
class TimelineProjectionUpdateRequest:
    """Request to revise an existing TimelineProjection with updated milestones or evidence."""
    projection_id: str
    updated_milestones: tuple[ProjectionMilestone, ...]
    updated_uncertainty: UncertaintyProfile
    updated_evidence: EvidenceProfile
    revision_reason: str
    updated_by: str

    def __post_init__(self) -> None:
        if not self.revision_reason.strip():
            raise ValueError("revision_reason must be a non-empty string.")
        if not self.updated_milestones:
            raise ValueError("At least one updated ProjectionMilestone is required.")


@dataclass(frozen=True)
class TimelineProjectionUpdateResponse:
    """Response confirming a TimelineProjection revision."""
    projection_id: str
    previous_status: ProjectionStatus
    new_status: ProjectionStatus
    updated_at: datetime


@dataclass(frozen=True)
class TimelineProjectionCompletionQuery:
    """Request to compute the projected completion probability for a TimelineProjection."""
    projection_id: str


@dataclass(frozen=True)
class TimelineProjectionCompletionResult:
    """Response containing the computed completion probability for critical milestones."""
    projection_id: str
    completion_probability: float
    probability_level: ProbabilityLevel
    critical_milestone_count: int
    computed_at: datetime


# ---------------------------------------------------------------------------
# Scenario Evaluation Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioCompareRequest:
    """
    Request to the Scenario Evaluation Engine to compare and rank
    a set of Scenarios.
    JANUS evaluates; it never approves. Approval belongs to VEGA.
    """
    scenario_ids: tuple[str, ...]
    title: str
    confidence: ConfidenceProfile

    def __post_init__(self) -> None:
        if len(self.scenario_ids) < 2:
            raise ValueError("At least two scenario_ids are required for comparison.")
        if not self.title.strip():
            raise ValueError("title must be a non-empty string.")


@dataclass(frozen=True)
class ScenarioCompareResponse:
    """
    Response from the Scenario Evaluation Engine after comparing scenarios.
    Contains ranking and trade-off analysis; does not contain a decision or approval.
    """
    comparison: ScenarioComparison
    recommended_for_review: str
    compared_at: datetime
    engine_version: str


@dataclass(frozen=True)
class ScenarioRankRequest:
    """Request to rank a set of Scenarios by risk-adjusted score."""
    scenario_ids: tuple[str, ...]
    weight_risk: float = 0.5
    weight_opportunity: float = 0.5

    def __post_init__(self) -> None:
        if not self.scenario_ids:
            raise ValueError("At least one scenario_id is required.")
        total = self.weight_risk + self.weight_opportunity
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"weight_risk and weight_opportunity must sum to 1.0, got {total:.6f}."
            )


@dataclass(frozen=True)
class ScenarioRankResponse:
    """Response containing scenario IDs ranked from best to worst."""
    ranked_scenario_ids: tuple[str, ...]
    risk_adjusted_scores: dict[str, float]
    opportunity_scores: dict[str, float]
    ranked_at: datetime


@dataclass(frozen=True)
class ScenarioDominanceRequest:
    """Request to compute the dominance map across a set of Scenarios."""
    scenario_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.scenario_ids) < 2:
            raise ValueError("At least two scenario_ids are required for dominance analysis.")


@dataclass(frozen=True)
class ScenarioDominanceResponse:
    """Response containing the dominance map across scenarios."""
    dominance_map: dict[str, list[str]]
    computed_at: datetime


# ---------------------------------------------------------------------------
# Counterfactual Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualCreateRequest:
    """
    Request to the Counterfactual Engine to construct an alternate-reality scenario.
    Answers: 'What if this decision had been different?'
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
    learning_insights: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.original_event.strip():
            raise ValueError("original_event must be a non-empty string.")
        if not self.counterfactual_condition.strip():
            raise ValueError("counterfactual_condition must be a non-empty string.")
        if not (-1.0 <= self.delta_risk <= 1.0):
            raise ValueError(f"delta_risk must be in [-1.0, 1.0], got {self.delta_risk}.")
        if not (-1.0 <= self.delta_opportunity <= 1.0):
            raise ValueError(
                f"delta_opportunity must be in [-1.0, 1.0], got {self.delta_opportunity}."
            )


@dataclass(frozen=True)
class CounterfactualCreateResponse:
    """Response from the Counterfactual Engine after creating a CounterfactualScenario."""
    counterfactual: CounterfactualScenario
    created_at: datetime
    engine_version: str


@dataclass(frozen=True)
class CounterfactualGetRequest:
    """Request to retrieve a specific CounterfactualScenario by ID."""
    counterfactual_id: str


@dataclass(frozen=True)
class CounterfactualGetResponse:
    """Response containing the requested CounterfactualScenario."""
    counterfactual: CounterfactualScenario
    retrieved_at: datetime


@dataclass(frozen=True)
class CounterfactualCompareRequest:
    """Request to compare a CounterfactualScenario against its original event's Scenario."""
    counterfactual_id: str
    original_scenario_id: str


@dataclass(frozen=True)
class CounterfactualCompareResponse:
    """Response containing the comparison of a counterfactual against its original."""
    counterfactual_id: str
    original_scenario_id: str
    delta_risk: float
    delta_opportunity: float
    net_delta: float
    learning_insights: tuple[str, ...]
    compared_at: datetime


# ---------------------------------------------------------------------------
# Uncertainty Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyProfileCreateRequest:
    """
    Request to the Uncertainty Engine to create an UncertaintyProfile.
    Required for every forecast and scenario (JANUS Law 6).
    """
    level: UncertaintyLevel
    known_risks: tuple[str, ...]
    unknown_risk_exposure: float
    volatility_score: float
    external_factors: tuple[str, ...]
    market_sensitivity: float
    technology_sensitivity: float
    notes: str = ""

    def __post_init__(self) -> None:
        for attr in (
            "unknown_risk_exposure",
            "volatility_score",
            "market_sensitivity",
            "technology_sensitivity",
        ):
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be in [0.0, 1.0], got {value}.")


@dataclass(frozen=True)
class UncertaintyProfileCreateResponse:
    """Response from the Uncertainty Engine after creating an UncertaintyProfile."""
    profile: UncertaintyProfile
    created_at: datetime


@dataclass(frozen=True)
class UncertaintyProfileGetRequest:
    """Request to retrieve a specific UncertaintyProfile by ID."""
    uncertainty_id: str


@dataclass(frozen=True)
class UncertaintyProfileGetResponse:
    """Response containing the requested UncertaintyProfile."""
    profile: UncertaintyProfile
    retrieved_at: datetime


@dataclass(frozen=True)
class UncertaintyValidationRequest:
    """
    Request to validate that an artifact carries a non-null, non-trivial UncertaintyProfile.
    Enforces JANUS Law 6: all forecasts require uncertainty.
    """
    artifact_id: str
    artifact_type: str
    uncertainty: UncertaintyProfile


@dataclass(frozen=True)
class UncertaintyValidationResponse:
    """Response from uncertainty validation."""
    artifact_id: str
    is_valid: bool
    violations: tuple[str, ...]
    validated_at: datetime


# ---------------------------------------------------------------------------
# Probability Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbabilityDistributionCreateRequest:
    """
    Request to the Probability Engine to create a ProbabilityDistribution.
    Outcomes must sum to 1.0 (±1e-6 tolerance).
    """
    label: str
    outcomes: dict[str, float]
    uncertainty_level: UncertaintyLevel
    confidence: ConfidenceProfile

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must be a non-empty string.")
        if not self.outcomes:
            raise ValueError("outcomes must contain at least one entry.")
        total = sum(self.outcomes.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"outcomes must sum to 1.0 (±1e-6), got {total:.6f}."
            )


@dataclass(frozen=True)
class ProbabilityDistributionCreateResponse:
    """Response from the Probability Engine after creating a ProbabilityDistribution."""
    distribution: ProbabilityDistribution
    created_at: datetime


@dataclass(frozen=True)
class ProbabilityLevelQueryRequest:
    """Request to resolve a float probability to its ProbabilityLevel label."""
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0.0, 1.0], got {self.probability}.")


@dataclass(frozen=True)
class ProbabilityLevelQueryResponse:
    """Response containing the resolved ProbabilityLevel for a given float value."""
    probability: float
    level: ProbabilityLevel
    resolved_at: datetime


@dataclass(frozen=True)
class ConfidenceProfileCreateRequest:
    """Request to create a ConfidenceProfile for use in a forecast or scenario."""
    overall: float
    data_quality: float
    model_fit: float
    signal_strength: float
    notes: str = ""

    def __post_init__(self) -> None:
        for attr in ("overall", "data_quality", "model_fit", "signal_strength"):
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be in [0.0, 1.0], got {value}.")


@dataclass(frozen=True)
class ConfidenceProfileCreateResponse:
    """Response containing the created ConfidenceProfile."""
    profile: ConfidenceProfile
    created_at: datetime


# ---------------------------------------------------------------------------
# Strategic Forecast Engine Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategicForecastCreateRequest:
    """
    Request to the Strategic Forecast Engine to produce a strategic-level forecast.
    Constitutional rule: JANUS forecasts strategic outcomes; ODYSSEY chooses direction.
    JANUS never selects strategy.
    """
    title: str
    description: str
    domain: str
    trend_analysis: tuple[str, ...]
    market_state_projections: tuple[FutureState, ...]
    strategic_outcome_forecasts: tuple[Forecast, ...]
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    horizon: ForecastHorizon

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be a non-empty string.")
        if not self.domain.strip():
            raise ValueError("domain must be a non-empty string.")
        if not self.trend_analysis:
            raise ValueError("At least one trend_analysis entry is required.")


@dataclass(frozen=True)
class StrategicForecastCreateResponse:
    """Response from the Strategic Forecast Engine after creating a StrategicForecast."""
    strategic_forecast: StrategicForecast
    created_at: datetime
    engine_version: str


@dataclass(frozen=True)
class StrategicForecastGetRequest:
    """Request to retrieve a specific StrategicForecast by ID."""
    strategic_forecast_id: str


@dataclass(frozen=True)
class StrategicForecastGetResponse:
    """Response containing the requested StrategicForecast."""
    strategic_forecast: StrategicForecast
    retrieved_at: datetime


@dataclass(frozen=True)
class StrategicForecastListRequest:
    """Request to list StrategicForecasts for a given domain and horizon."""
    domain: Optional[str]
    horizon: Optional[ForecastHorizon]


@dataclass(frozen=True)
class StrategicForecastListResponse:
    """Response containing StrategicForecasts matching the requested filters."""
    strategic_forecasts: tuple[StrategicForecast, ...]
    retrieved_at: datetime


# ---------------------------------------------------------------------------
# Future Memory Interface Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalOutcomeFetchRequest:
    """
    Request to the Future Memory Interface to fetch historical outcome data
    from upstream subsystems (ECHO, CHRONOS, ASTRA, CONSTELLATION).
    JANUS does not own memory; it only reads historical outcomes.
    """
    source_subsystem: str
    context_filters: dict[str, Any]
    horizon_hint: Optional[ForecastHorizon]
    max_records: int = 100

    def __post_init__(self) -> None:
        if not self.source_subsystem.strip():
            raise ValueError("source_subsystem must be a non-empty string.")
        if self.max_records < 1:
            raise ValueError("max_records must be >= 1.")


@dataclass(frozen=True)
class HistoricalOutcomeFetchResponse:
    """Response containing historical outcome records consumed by JANUS for forecasting."""
    source_subsystem: str
    records: tuple[dict[str, Any], ...]
    fetched_at: datetime
    is_stale: bool
    staleness_description: str


@dataclass(frozen=True)
class ForecastAccuracyFeedRequest:
    """
    Request to feed a resolved forecast accuracy record back through the
    Future Memory Interface so that it can improve future forecasting.
    """
    forecast_id: str
    resolved_outcome: str
    actual_probability_observed: float
    resolution_timestamp: datetime
    notes: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.actual_probability_observed <= 1.0):
            raise ValueError(
                f"actual_probability_observed must be in [0.0, 1.0], "
                f"got {self.actual_probability_observed}."
            )


@dataclass(frozen=True)
class ForecastAccuracyFeedResponse:
    """Response confirming the accuracy record has been fed back for model improvement."""
    forecast_id: str
    accuracy_delta: float
    fed_at: datetime


# ---------------------------------------------------------------------------
# Future Orchestrator Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FutureAssessmentCreateRequest:
    """
    Request to the Future Orchestrator to produce a comprehensive FutureAssessment,
    coordinating all JANUS engines into a single deliverable.
    """
    title: str
    description: str
    context: str
    scenario_generation_request: ScenarioGenerateRequest
    risk_analysis_requests: tuple[RiskAnalysisRequest, ...]
    opportunity_analysis_requests: tuple[OpportunityAnalysisRequest, ...]
    simulation_requests: tuple[SimulationRunRequest, ...]
    timeline_requests: tuple[TimelineProjectionCreateRequest, ...]
    strategic_forecast_requests: tuple[StrategicForecastCreateRequest, ...]
    summary: str

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be a non-empty string.")
        if not self.context.strip():
            raise ValueError("context must be a non-empty string.")


@dataclass(frozen=True)
class FutureAssessmentCreateResponse:
    """Response from the Future Orchestrator after producing a complete FutureAssessment."""
    assessment: FutureAssessment
    created_at: datetime
    engine_version: str


@dataclass(frozen=True)
class FutureAssessmentGetRequest:
    """Request to retrieve a specific FutureAssessment by ID."""
    assessment_id: str


@dataclass(frozen=True)
class FutureAssessmentGetResponse:
    """Response containing the requested FutureAssessment."""
    assessment: FutureAssessment
    retrieved_at: datetime


@dataclass(frozen=True)
class FutureAssessmentUpdateStatusRequest:
    """Request to transition a FutureAssessment's status."""
    assessment_id: str
    target_status: FutureAssessmentStatus
    reason: str
    updated_by: str


@dataclass(frozen=True)
class FutureAssessmentUpdateStatusResponse:
    """Response confirming a FutureAssessment status transition."""
    assessment_id: str
    previous_status: FutureAssessmentStatus
    new_status: FutureAssessmentStatus
    updated_at: datetime


@dataclass(frozen=True)
class OrchestratorHealthRequest:
    """Request to check the health status of the Future Orchestrator and all sub-engines."""


@dataclass(frozen=True)
class OrchestratorHealthResponse:
    """Response containing the health status of the Orchestrator and each sub-engine."""
    orchestrator_healthy: bool
    engine_statuses: dict[str, bool]
    checked_at: datetime
    notes: str