"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: interfaces.py

Abstract interface contracts for every JANUS engine.
Each interface enforces:
  - Lifecycle gating (initialize / shutdown)
  - Domain ownership boundaries
  - Constitutional separation (JANUS Law 1–8)
  - Production-quality typing throughout

Engine inventory (16 total):
  IScenarioEngine            — Scenario Engine
  IForecastingEngine         — Forecasting Engine
  IFutureModelingEngine      — Future Modeling Engine
  IBranchAnalysisEngine      — Branch Analysis Engine
  ICounterfactualEngine      — Counterfactual Engine
  IUncertaintyEngine         — Uncertainty Engine
  IFutureRiskEngine          — Future Risk Engine
  IFutureOpportunityEngine   — Future Opportunity Engine
  IOutcomeSimulationEngine   — Outcome Simulation Engine
  IStrategicForecastEngine   — Strategic Forecast Engine
  ITimelineProjectionEngine  — Timeline Projection Engine
  IProbabilityEngine         — Probability Engine
  IScenarioEvaluationEngine  — Scenario Evaluation Engine
  IScenarioIntegrityEngine   — Scenario Integrity Engine
  IFutureMemoryInterface     — Future Memory Interface
  IFutureOrchestrator        — Future Orchestrator
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import (
    ConfidenceProfile,
    CounterfactualScenario,
    EvidenceProfile,
    Forecast,
    ForecastAssessment,
    ForecastHorizon,
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
from .schemas import (
    # Scenario
    ScenarioGenerateRequest,
    ScenarioGenerateResponse,
    ScenarioGetRequest,
    ScenarioGetResponse,
    ScenarioUpdateStatusRequest,
    ScenarioUpdateStatusResponse,
    ScenarioArchiveRequest,
    ScenarioArchiveResponse,
    # Forecast
    ForecastGenerateRequest,
    ForecastGenerateResponse,
    ForecastGetRequest,
    ForecastGetResponse,
    ForecastAssessRequest,
    ForecastAssessResponse,
    ForecastListByHorizonRequest,
    ForecastListByHorizonResponse,
    # Future Model
    FutureModelConstructRequest,
    FutureModelConstructResponse,
    FutureModelGetRequest,
    FutureModelGetResponse,
    FutureStateQueryRequest,
    FutureStateQueryResponse,
    # Simulation
    SimulationRunRequest,
    SimulationRunResponse,
    SimulationResultRequest,
    SimulationResultResponse,
    SimulationAbortRequest,
    SimulationAbortResponse,
    # Risk
    RiskAnalysisRequest,
    RiskAnalysisResponse,
    RiskAssessmentGetRequest,
    RiskAssessmentGetResponse,
    RiskListByLevelRequest,
    RiskListByLevelResponse,
    # Opportunity
    OpportunityAnalysisRequest,
    OpportunityAnalysisResponse,
    OpportunityAssessmentGetRequest,
    OpportunityAssessmentGetResponse,
    OpportunityListByLevelRequest,
    OpportunityListByLevelResponse,
    # Timeline Projection
    TimelineProjectionCreateRequest,
    TimelineProjectionCreateResponse,
    TimelineProjectionGetRequest,
    TimelineProjectionGetResponse,
    TimelineProjectionUpdateRequest,
    TimelineProjectionUpdateResponse,
    TimelineProjectionCompletionQuery,
    TimelineProjectionCompletionResult,
    # Scenario Evaluation
    ScenarioCompareRequest,
    ScenarioCompareResponse,
    ScenarioRankRequest,
    ScenarioRankResponse,
    ScenarioDominanceRequest,
    ScenarioDominanceResponse,
    # Counterfactual
    CounterfactualCreateRequest,
    CounterfactualCreateResponse,
    CounterfactualGetRequest,
    CounterfactualGetResponse,
    CounterfactualCompareRequest,
    CounterfactualCompareResponse,
    # Uncertainty
    UncertaintyProfileCreateRequest,
    UncertaintyProfileCreateResponse,
    UncertaintyProfileGetRequest,
    UncertaintyProfileGetResponse,
    UncertaintyValidationRequest,
    UncertaintyValidationResponse,
    # Probability
    ProbabilityDistributionCreateRequest,
    ProbabilityDistributionCreateResponse,
    ProbabilityLevelQueryRequest,
    ProbabilityLevelQueryResponse,
    ConfidenceProfileCreateRequest,
    ConfidenceProfileCreateResponse,
    # Strategic Forecast
    StrategicForecastCreateRequest,
    StrategicForecastCreateResponse,
    StrategicForecastGetRequest,
    StrategicForecastGetResponse,
    StrategicForecastListRequest,
    StrategicForecastListResponse,
    # Future Memory Interface
    HistoricalOutcomeFetchRequest,
    HistoricalOutcomeFetchResponse,
    ForecastAccuracyFeedRequest,
    ForecastAccuracyFeedResponse,
    # Orchestrator
    FutureAssessmentCreateRequest,
    FutureAssessmentCreateResponse,
    FutureAssessmentGetRequest,
    FutureAssessmentGetResponse,
    FutureAssessmentUpdateStatusRequest,
    FutureAssessmentUpdateStatusResponse,
    OrchestratorHealthRequest,
    OrchestratorHealthResponse,
)


# ---------------------------------------------------------------------------
# Lifecycle Mixin
# ---------------------------------------------------------------------------


class JanusEngineLifecycle(ABC):
    """
    Lifecycle contract shared by all JANUS engines.

    Required sequence:
        initialize() → [operational methods] → shutdown()

    Implementations must:
      - Raise JanusAlreadyInitializedError if initialize() is called twice.
      - Raise JanusNotInitializedError if any operational method is called
        before initialize() completes.
      - Raise JanusShutdownError if any operational method is called after
        shutdown() completes.
      - Be idempotent on a second shutdown() call (no-op or raise, consistently).
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the engine. Acquire all required resources.
        Must be called before any operational method.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shut down the engine. Release all resources.
        No operational methods may be called after shutdown.
        """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Return True if the engine has been successfully initialized."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return the canonical name of this engine (e.g., 'ScenarioEngine')."""

    @property
    @abstractmethod
    def engine_version(self) -> str:
        """Return the version string of this engine implementation."""


# ---------------------------------------------------------------------------
# IScenarioEngine
# ---------------------------------------------------------------------------


class IScenarioEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Scenario Engine.

    Responsibility: Generate alternative future scenarios for a given
    decision context. Each scenario represents a possible future branch.

    Constitutional rule:
        JANUS generates and evaluates scenarios.
        VEGA selects among them.
        This engine never selects or approves a scenario.
    """

    @abstractmethod
    def generate_scenarios(
        self, request: ScenarioGenerateRequest
    ) -> ScenarioGenerateResponse:
        """
        Generate one or more alternative-future Scenarios for the given
        decision context and requested ScenarioTypes.
        """

    @abstractmethod
    def get_scenario(self, request: ScenarioGetRequest) -> ScenarioGetResponse:
        """Retrieve a Scenario by its scenario_id."""

    @abstractmethod
    def update_scenario_status(
        self, request: ScenarioUpdateStatusRequest
    ) -> ScenarioUpdateStatusResponse:
        """
        Transition a Scenario's ScenarioStatus.
        Validates the transition is legal before applying.
        """

    @abstractmethod
    def archive_scenario(
        self, request: ScenarioArchiveRequest
    ) -> ScenarioArchiveResponse:
        """Archive a Scenario, marking it as no longer active."""

    @abstractmethod
    def list_scenarios_by_type(
        self, scenario_type: ScenarioType
    ) -> tuple[Scenario, ...]:
        """Return all Scenarios of the given ScenarioType."""

    @abstractmethod
    def list_scenarios_by_status(
        self, status: ScenarioStatus
    ) -> tuple[Scenario, ...]:
        """Return all Scenarios with the given ScenarioStatus."""


# ---------------------------------------------------------------------------
# IForecastingEngine
# ---------------------------------------------------------------------------


class IForecastingEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Forecasting Engine.

    Responsibility: Produce probabilistic forecasts from historical data,
    current state, patterns, and trends.

    Constitutional rule (Law 6):
        All forecasts must include an UncertaintyProfile.
        No certainty claims ever.
    """

    @abstractmethod
    def generate_forecast(
        self, request: ForecastGenerateRequest
    ) -> ForecastGenerateResponse:
        """
        Generate a probabilistic Forecast for the given type and horizon.
        The request must carry an UncertaintyProfile; generation will be
        rejected if uncertainty is absent.
        """

    @abstractmethod
    def get_forecast(self, request: ForecastGetRequest) -> ForecastGetResponse:
        """Retrieve a Forecast by its forecast_id."""

    @abstractmethod
    def assess_forecast(
        self, request: ForecastAssessRequest
    ) -> ForecastAssessResponse:
        """
        Assess an existing Forecast for accuracy after its horizon has resolved.
        Records deviation notes and flags revision if required.
        """

    @abstractmethod
    def list_forecasts_by_horizon(
        self, request: ForecastListByHorizonRequest
    ) -> ForecastListByHorizonResponse:
        """Return all Forecasts matching the given ForecastHorizon."""

    @abstractmethod
    def supersede_forecast(
        self, forecast_id: str, replacement_forecast_id: str, reason: str
    ) -> Forecast:
        """
        Mark an existing Forecast as superseded by a newer forecast.
        Returns the updated (superseded) Forecast.
        """

    @abstractmethod
    def list_forecasts_by_type(
        self, forecast_type: ForecastType
    ) -> tuple[Forecast, ...]:
        """Return all active Forecasts of the given ForecastType."""


# ---------------------------------------------------------------------------
# IFutureModelingEngine
# ---------------------------------------------------------------------------


class IFutureModelingEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Future Modeling Engine.

    Responsibility: Construct structured FutureModel objects that span
    multiple time horizons, incorporating FutureStates with associated
    risks, opportunities, and uncertainty.
    """

    @abstractmethod
    def construct_future_model(
        self, request: FutureModelConstructRequest
    ) -> FutureModelConstructResponse:
        """
        Construct a FutureModel from the given future states, uncertainty,
        confidence, and evidence profiles.
        """

    @abstractmethod
    def get_future_model(
        self, request: FutureModelGetRequest
    ) -> FutureModelGetResponse:
        """Retrieve a FutureModel by its model_id."""

    @abstractmethod
    def query_states_at_horizon(
        self, request: FutureStateQueryRequest
    ) -> FutureStateQueryResponse:
        """Return all FutureStates within a FutureModel at the given horizon."""

    @abstractmethod
    def update_future_model(
        self,
        model_id: str,
        updated_states: tuple[FutureState, ...],
        updated_uncertainty: UncertaintyProfile,
        updated_evidence: EvidenceProfile,
        reason: str,
    ) -> FutureModel:
        """
        Update a FutureModel with revised states, uncertainty, and evidence.
        Returns the updated FutureModel.
        """

    @abstractmethod
    def list_future_models(self) -> tuple[FutureModel, ...]:
        """Return all currently active FutureModels."""


# ---------------------------------------------------------------------------
# IBranchAnalysisEngine
# ---------------------------------------------------------------------------


class IBranchAnalysisEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Branch Analysis Engine.

    Responsibility: Explore alternative decision branches, constructing
    ScenarioBranch objects that map triggering choices to FutureStates.

    Constitutional rule:
        Branch analysis supports strategic exploration.
        It never selects a branch; VEGA selects.
    """

    @abstractmethod
    def construct_branch(
        self,
        label: str,
        description: str,
        triggering_choice: str,
        future_state: FutureState,
        probability: float,
        risk_assessment: RiskAssessment,
        opportunity_assessment: OpportunityAssessment,
        confidence: ConfidenceProfile,
    ) -> ScenarioBranch:
        """
        Construct a ScenarioBranch for a given triggering choice and its
        resulting FutureState.
        """

    @abstractmethod
    def get_branch(self, branch_id: str) -> ScenarioBranch:
        """Retrieve a ScenarioBranch by its branch_id."""

    @abstractmethod
    def analyze_branches(
        self, scenario_id: str
    ) -> tuple[ScenarioBranch, ...]:
        """
        Return all ScenarioBranches for the given scenario, ordered by
        probability descending.
        """

    @abstractmethod
    def validate_branch_probabilities(self, scenario_id: str) -> bool:
        """
        Validate that the branch probabilities within a scenario are
        internally consistent (i.e., do not imply mutual exclusivity violations).
        Returns True if consistent.
        """

    @abstractmethod
    def dominant_branch(self, scenario_id: str) -> ScenarioBranch:
        """
        Return the ScenarioBranch with the highest probability for the given scenario.
        """

    @abstractmethod
    def compare_branches(
        self, branch_ids: tuple[str, ...]
    ) -> dict[str, dict[str, Any]]:
        """
        Produce a structured comparison of the given ScenarioBranches.
        Returns a dict keyed by branch_id containing comparative metrics.
        """


# ---------------------------------------------------------------------------
# ICounterfactualEngine
# ---------------------------------------------------------------------------


class ICounterfactualEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Counterfactual Engine.

    Responsibility: Construct alternate-reality scenarios that answer
    'What if this decision had been different?' Used for learning and
    strategic evaluation.
    """

    @abstractmethod
    def create_counterfactual(
        self, request: CounterfactualCreateRequest
    ) -> CounterfactualCreateResponse:
        """
        Construct a CounterfactualScenario from a given original event
        and alternate condition, anchored at the specified divergence point.
        """

    @abstractmethod
    def get_counterfactual(
        self, request: CounterfactualGetRequest
    ) -> CounterfactualGetResponse:
        """Retrieve a CounterfactualScenario by its counterfactual_id."""

    @abstractmethod
    def compare_counterfactual(
        self, request: CounterfactualCompareRequest
    ) -> CounterfactualCompareResponse:
        """
        Compare a CounterfactualScenario against its original Scenario,
        producing delta metrics and consolidated learning insights.
        """

    @abstractmethod
    def list_counterfactuals_for_event(
        self, original_event: str
    ) -> tuple[CounterfactualScenario, ...]:
        """
        Return all CounterfactualScenarios derived from the given original event.
        """

    @abstractmethod
    def extract_learning_insights(
        self, counterfactual_id: str
    ) -> tuple[str, ...]:
        """
        Return the learning insights recorded on a CounterfactualScenario.
        These feed NOVA via ECHO's reflection pipeline.
        """


# ---------------------------------------------------------------------------
# IUncertaintyEngine
# ---------------------------------------------------------------------------


class IUncertaintyEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Uncertainty Engine.

    Responsibility: Model and quantify uncertainty across all JANUS forecast
    and scenario artifacts.

    Constitutional rule (Law 6):
        Every forecast must include uncertainty.
        No certainty claims ever.
        This engine enforces that invariant across all JANUS artifacts.
    """

    @abstractmethod
    def create_uncertainty_profile(
        self, request: UncertaintyProfileCreateRequest
    ) -> UncertaintyProfileCreateResponse:
        """Create an UncertaintyProfile for use in a forecast or scenario."""

    @abstractmethod
    def get_uncertainty_profile(
        self, request: UncertaintyProfileGetRequest
    ) -> UncertaintyProfileGetResponse:
        """Retrieve an UncertaintyProfile by its uncertainty_id."""

    @abstractmethod
    def validate_uncertainty(
        self, request: UncertaintyValidationRequest
    ) -> UncertaintyValidationResponse:
        """
        Validate that a JANUS artifact carries a non-null, non-trivial
        UncertaintyProfile. Enforces JANUS Law 6.
        """

    @abstractmethod
    def aggregate_uncertainty(
        self, profiles: tuple[UncertaintyProfile, ...]
    ) -> UncertaintyProfile:
        """
        Aggregate multiple UncertaintyProfiles into a single consolidated profile.
        Used by the Orchestrator when assembling FutureAssessments.
        """

    @abstractmethod
    def classify_uncertainty_level(
        self, profile: UncertaintyProfile
    ) -> UncertaintyLevel:
        """
        Derive the overall UncertaintyLevel classification from a full
        UncertaintyProfile's component scores.
        """


# ---------------------------------------------------------------------------
# IFutureRiskEngine
# ---------------------------------------------------------------------------


class IFutureRiskEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Future Risk Engine.

    Responsibility: Identify and assess future threats across skill obsolescence,
    technology disruption, market changes, project failure, and resource exhaustion.
    Focuses exclusively on future risk, not present-state risk.
    """

    @abstractmethod
    def analyze_risk(self, request: RiskAnalysisRequest) -> RiskAnalysisResponse:
        """
        Produce a RiskAssessment for the given risk factors, horizon, and context.
        Computes composite risk scores and derives the RiskLevel.
        """

    @abstractmethod
    def get_risk_assessment(
        self, request: RiskAssessmentGetRequest
    ) -> RiskAssessmentGetResponse:
        """Retrieve a RiskAssessment by its risk_id."""

    @abstractmethod
    def list_risks_by_level(
        self, request: RiskListByLevelRequest
    ) -> RiskListByLevelResponse:
        """
        Return all RiskAssessments at or above the specified RiskLevel,
        optionally filtered by ForecastHorizon.
        """

    @abstractmethod
    def compute_composite_risk_score(
        self, risk_factors: tuple[RiskFactor, ...]
    ) -> float:
        """
        Compute the composite risk score (0.0–1.0) from a set of RiskFactors.
        Returns the mean composite score across all factors.
        """

    @abstractmethod
    def derive_risk_level(self, composite_score: float) -> RiskLevel:
        """
        Derive the RiskLevel classification from a composite risk score.
        """

    @abstractmethod
    def list_risk_factors_by_category(
        self, category: str
    ) -> tuple[RiskFactor, ...]:
        """Return all RiskFactors belonging to the given category."""


# ---------------------------------------------------------------------------
# IFutureOpportunityEngine
# ---------------------------------------------------------------------------


class IFutureOpportunityEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Future Opportunity Engine.

    Responsibility: Identify and assess future opportunities across emerging
    technologies, industry growth, research breakthroughs, and startup openings.
    Feeds ODYSSEY and VEGA.
    """

    @abstractmethod
    def analyze_opportunity(
        self, request: OpportunityAnalysisRequest
    ) -> OpportunityAnalysisResponse:
        """
        Produce an OpportunityAssessment for the given opportunity factors,
        horizon, and context. Computes composite scores and derives the
        OpportunityLevel.
        """

    @abstractmethod
    def get_opportunity_assessment(
        self, request: OpportunityAssessmentGetRequest
    ) -> OpportunityAssessmentGetResponse:
        """Retrieve an OpportunityAssessment by its opportunity_id."""

    @abstractmethod
    def list_opportunities_by_level(
        self, request: OpportunityListByLevelRequest
    ) -> OpportunityListByLevelResponse:
        """
        Return all OpportunityAssessments at or above the specified
        OpportunityLevel, optionally filtered by ForecastHorizon.
        """

    @abstractmethod
    def compute_composite_opportunity_score(
        self, opportunity_factors: tuple[OpportunityFactor, ...]
    ) -> float:
        """
        Compute the composite opportunity score (0.0–1.0) from a set of
        OpportunityFactors. Returns the mean composite score across all factors.
        """

    @abstractmethod
    def derive_opportunity_level(self, composite_score: float) -> OpportunityLevel:
        """Derive the OpportunityLevel classification from a composite score."""

    @abstractmethod
    def list_opportunity_factors_by_category(
        self, category: str
    ) -> tuple[OpportunityFactor, ...]:
        """Return all OpportunityFactors belonging to the given category."""


# ---------------------------------------------------------------------------
# IOutcomeSimulationEngine
# ---------------------------------------------------------------------------


class IOutcomeSimulationEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Outcome Simulation Engine.

    Responsibility: Simulate consequence chains from a given decision or event,
    producing short-, medium-, and long-term effect chains.

    Constitutional rule (Law 3):
        ORION reasons. JANUS simulates. Never merged.
    """

    @abstractmethod
    def run_simulation(
        self, request: SimulationRunRequest
    ) -> SimulationRunResponse:
        """
        Begin a simulation for the given triggering event with the supplied
        candidate outcomes and uncertainty model.
        Returns immediately with PENDING status; poll via get_simulation_result.
        """

    @abstractmethod
    def get_simulation_result(
        self, request: SimulationResultRequest
    ) -> SimulationResultResponse:
        """
        Retrieve the current state and result of a simulation.
        Returns the OutcomeSimulation with its current SimulationStatus.
        """

    @abstractmethod
    def abort_simulation(
        self, request: SimulationAbortRequest
    ) -> SimulationAbortResponse:
        """Abort a running simulation before it completes."""

    @abstractmethod
    def most_probable_outcome(
        self, simulation_id: str
    ) -> SimulationOutcome:
        """
        Return the SimulationOutcome with the highest probability for the
        given completed simulation.
        """

    @abstractmethod
    def list_simulations_by_status(
        self, status: SimulationStatus
    ) -> tuple[OutcomeSimulation, ...]:
        """Return all OutcomeSimulations with the given SimulationStatus."""

    @abstractmethod
    def validate_outcome_probabilities(
        self, simulation_id: str
    ) -> bool:
        """
        Validate that the SimulationOutcome probabilities for a simulation
        are internally consistent. Returns True if consistent.
        """


# ---------------------------------------------------------------------------
# IStrategicForecastEngine
# ---------------------------------------------------------------------------


class IStrategicForecastEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Strategic Forecast Engine.

    Responsibility: Forecast strategic-level outcomes from trend analysis and
    market state projections.

    Constitutional rule:
        JANUS forecasts strategic outcomes.
        ODYSSEY chooses strategic direction.
        This engine never selects strategy.
    """

    @abstractmethod
    def create_strategic_forecast(
        self, request: StrategicForecastCreateRequest
    ) -> StrategicForecastCreateResponse:
        """
        Produce a StrategicForecast for the given domain, integrating trend
        analysis, market state projections, and strategic outcome forecasts.
        """

    @abstractmethod
    def get_strategic_forecast(
        self, request: StrategicForecastGetRequest
    ) -> StrategicForecastGetResponse:
        """Retrieve a StrategicForecast by its strategic_forecast_id."""

    @abstractmethod
    def list_strategic_forecasts(
        self, request: StrategicForecastListRequest
    ) -> StrategicForecastListResponse:
        """
        Return all StrategicForecasts matching the given domain and/or horizon
        filters.
        """

    @abstractmethod
    def update_strategic_forecast(
        self,
        strategic_forecast_id: str,
        updated_trend_analysis: tuple[str, ...],
        updated_market_projections: tuple[FutureState, ...],
        updated_outcome_forecasts: tuple[Forecast, ...],
        updated_uncertainty: UncertaintyProfile,
        updated_evidence: EvidenceProfile,
        reason: str,
    ) -> StrategicForecast:
        """
        Revise a StrategicForecast with updated trend data, projections, and
        uncertainty. Returns the updated StrategicForecast.
        """

    @abstractmethod
    def list_strategic_forecasts_by_domain(
        self, domain: str
    ) -> tuple[StrategicForecast, ...]:
        """Return all StrategicForecasts for a given strategic domain."""


# ---------------------------------------------------------------------------
# ITimelineProjectionEngine
# ---------------------------------------------------------------------------


class ITimelineProjectionEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Timeline Projection Engine.

    Responsibility: Project future states and milestones across time horizons.

    Constitutional rule:
        CHRONOS owns time.
        JANUS predicts future states across time.
        This engine never manages, stores, or modifies time data.
    """

    @abstractmethod
    def create_projection(
        self, request: TimelineProjectionCreateRequest
    ) -> TimelineProjectionCreateResponse:
        """
        Create a TimelineProjection from the given milestones, horizon,
        uncertainty, and evidence.
        """

    @abstractmethod
    def get_projection(
        self, request: TimelineProjectionGetRequest
    ) -> TimelineProjectionGetResponse:
        """Retrieve a TimelineProjection by its projection_id."""

    @abstractmethod
    def update_projection(
        self, request: TimelineProjectionUpdateRequest
    ) -> TimelineProjectionUpdateResponse:
        """
        Revise a TimelineProjection with updated milestones and evidence.
        Transitions status to REVISED.
        """

    @abstractmethod
    def query_completion_probability(
        self, request: TimelineProjectionCompletionQuery
    ) -> TimelineProjectionCompletionResult:
        """
        Compute the completion probability of a TimelineProjection based
        on its critical milestones.
        """

    @abstractmethod
    def list_projections_by_status(
        self, status: ProjectionStatus
    ) -> tuple[TimelineProjection, ...]:
        """Return all TimelineProjections with the given ProjectionStatus."""

    @abstractmethod
    def expire_projection(
        self, projection_id: str, reason: str
    ) -> TimelineProjection:
        """
        Mark a TimelineProjection as expired when its horizon has passed
        without resolution. Returns the updated projection.
        """

    @abstractmethod
    def supersede_projection(
        self, projection_id: str, replacement_projection_id: str, reason: str
    ) -> TimelineProjection:
        """
        Mark a TimelineProjection as superseded by a newer projection.
        Returns the updated (superseded) projection.
        """


# ---------------------------------------------------------------------------
# IProbabilityEngine
# ---------------------------------------------------------------------------


class IProbabilityEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Probability Engine.

    Responsibility: Create and validate ProbabilityDistributions, resolve
    float probabilities to ProbabilityLevel labels, and create ConfidenceProfiles.

    Constitutional rule (Law 6):
        Required for all forecasts.
        No certainty claims — probability == 1.0 without uncertainty
        constitutes a false certainty violation.
    """

    @abstractmethod
    def create_distribution(
        self, request: ProbabilityDistributionCreateRequest
    ) -> ProbabilityDistributionCreateResponse:
        """
        Create a ProbabilityDistribution over named outcomes.
        Validates that outcomes sum to 1.0 (±1e-6).
        """

    @abstractmethod
    def get_distribution(
        self, distribution_id: str
    ) -> ProbabilityDistribution:
        """Retrieve a ProbabilityDistribution by its distribution_id."""

    @abstractmethod
    def resolve_probability_level(
        self, request: ProbabilityLevelQueryRequest
    ) -> ProbabilityLevelQueryResponse:
        """
        Resolve a float probability value to its ProbabilityLevel label
        using the standard JANUS thresholds.
        """

    @abstractmethod
    def create_confidence_profile(
        self, request: ConfidenceProfileCreateRequest
    ) -> ConfidenceProfileCreateResponse:
        """
        Create a ConfidenceProfile from its component scores (overall,
        data_quality, model_fit, signal_strength).
        """

    @abstractmethod
    def validate_no_false_certainty(
        self,
        probability: float,
        uncertainty: UncertaintyProfile,
        artifact_id: str,
        artifact_type: str,
    ) -> bool:
        """
        Validate that a probability == 1.0 is not accompanied by a trivial
        (zero) UncertaintyProfile. Returns True if the claim is valid.
        Raises JanusFalseCertaintyError on violation.
        """

    @abstractmethod
    def normalize_distribution(
        self, outcomes: dict[str, float]
    ) -> dict[str, float]:
        """
        Normalize an outcome probability dict to sum exactly to 1.0.
        Raises JanusProbabilityNormalizationError if normalization is impossible.
        """


# ---------------------------------------------------------------------------
# IScenarioEvaluationEngine
# ---------------------------------------------------------------------------


class IScenarioEvaluationEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Scenario Evaluation Engine.

    Responsibility: Evaluate, rank, compare, and analyze trade-offs across
    a set of Scenarios.

    Constitutional rule (Law 8):
        JANUS evaluates futures.
        It never approves futures.
        Approval belongs to VEGA.
        This engine must never produce an 'approved' or 'selected' field.
    """

    @abstractmethod
    def compare_scenarios(
        self, request: ScenarioCompareRequest
    ) -> ScenarioCompareResponse:
        """
        Compare and rank the given set of Scenarios.
        Produces a ScenarioComparison with ranking, trade-off analysis,
        and a recommended_for_review field (recommendation, not decision).
        """

    @abstractmethod
    def rank_scenarios(
        self, request: ScenarioRankRequest
    ) -> ScenarioRankResponse:
        """
        Rank the given Scenarios by risk-adjusted composite score.
        Applies the configured weight_risk / weight_opportunity weighting.
        """

    @abstractmethod
    def compute_dominance(
        self, request: ScenarioDominanceRequest
    ) -> ScenarioDominanceResponse:
        """
        Compute the dominance map across a set of Scenarios.
        A Scenario A dominates B if A has both lower risk and higher opportunity.
        """

    @abstractmethod
    def compute_risk_adjusted_score(
        self,
        scenario: Scenario,
        weight_risk: float,
        weight_opportunity: float,
    ) -> float:
        """
        Compute the risk-adjusted composite score for a single Scenario.
        score = (weight_opportunity × opportunity_score) - (weight_risk × risk_score)
        """

    @abstractmethod
    def get_scenario_comparison(
        self, comparison_id: str
    ) -> ScenarioComparison:
        """Retrieve a ScenarioComparison by its comparison_id."""

    @abstractmethod
    def analyze_trade_offs(
        self, scenario_ids: tuple[str, ...]
    ) -> dict[str, str]:
        """
        Produce a trade-off summary dict (scenario_id → trade-off description)
        for the given Scenarios.
        """


# ---------------------------------------------------------------------------
# IScenarioIntegrityEngine
# ---------------------------------------------------------------------------


class IScenarioIntegrityEngine(JanusEngineLifecycle, ABC):
    """
    Interface for the Scenario Integrity Engine.

    Responsibility: Protect forecast quality by preventing impossible futures,
    unsupported predictions, contradictory outcomes, and false certainty claims.

    Constitutional rule (Law 6):
        All forecasts require uncertainty.
        No certainty claims ever.
        This engine is the enforcement layer for that invariant.
    """

    @abstractmethod
    def validate_scenario(self, scenario: Scenario) -> tuple[bool, tuple[str, ...]]:
        """
        Validate a Scenario for integrity violations.
        Returns (is_valid, violations) where violations is a tuple of
        human-readable violation descriptions.
        """

    @abstractmethod
    def validate_forecast(self, forecast: Forecast) -> tuple[bool, tuple[str, ...]]:
        """
        Validate a Forecast for integrity violations (missing uncertainty,
        false certainty, unsupported predictions).
        Returns (is_valid, violations).
        """

    @abstractmethod
    def validate_future_model(
        self, future_model: FutureModel
    ) -> tuple[bool, tuple[str, ...]]:
        """
        Validate a FutureModel for impossible states or contradictory outcomes.
        Returns (is_valid, violations).
        """

    @abstractmethod
    def validate_simulation(
        self, simulation: OutcomeSimulation
    ) -> tuple[bool, tuple[str, ...]]:
        """
        Validate an OutcomeSimulation for internal consistency and evidence
        support. Returns (is_valid, violations).
        """

    @abstractmethod
    def validate_timeline_projection(
        self, projection: TimelineProjection
    ) -> tuple[bool, tuple[str, ...]]:
        """
        Validate a TimelineProjection for milestone consistency and
        evidence support. Returns (is_valid, violations).
        """

    @abstractmethod
    def enforce_uncertainty_invariant(
        self, artifact_id: str, artifact_type: str, uncertainty: UncertaintyProfile
    ) -> None:
        """
        Enforce JANUS Law 6: raise JanusMissingUncertaintyError if the
        UncertaintyProfile is absent or trivial.
        """

    @abstractmethod
    def enforce_evidence_invariant(
        self, artifact_id: str, artifact_type: str, evidence: EvidenceProfile
    ) -> None:
        """
        Raise JanusEvidenceProfileError if the EvidenceProfile is absent,
        empty, or below the minimum evidence strength threshold.
        """

    @abstractmethod
    def check_constitutional_boundary(
        self, engine_name: str, operation: str, rightful_owner: str
    ) -> None:
        """
        Raise JanusConstitutionalViolationError if the given operation
        belongs to a different POLARIS subsystem.
        """


# ---------------------------------------------------------------------------
# IFutureMemoryInterface
# ---------------------------------------------------------------------------


class IFutureMemoryInterface(JanusEngineLifecycle, ABC):
    """
    Interface for the Future Memory Interface.

    Responsibility: Consume historical outcome data from upstream subsystems
    (ECHO, CHRONOS, ASTRA, CONSTELLATION) to improve forecasting accuracy.
    Feed resolved forecast accuracy records back for model improvement.

    Constitutional rule:
        JANUS does not own memory.
        It consumes historical outcomes to improve forecasting accuracy.
        This interface must never write to or claim ownership of memory data.
    """

    @abstractmethod
    def fetch_historical_outcomes(
        self, request: HistoricalOutcomeFetchRequest
    ) -> HistoricalOutcomeFetchResponse:
        """
        Fetch historical outcome records from the specified upstream subsystem.
        Reads from ECHO, CHRONOS, ASTRA, or CONSTELLATION.
        Never writes to those subsystems.
        """

    @abstractmethod
    def feed_forecast_accuracy(
        self, request: ForecastAccuracyFeedRequest
    ) -> ForecastAccuracyFeedResponse:
        """
        Feed a resolved forecast accuracy record back into the forecasting
        model to improve future predictions.
        """

    @abstractmethod
    def is_data_stale(
        self, source_subsystem: str, context_filters: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Check whether the available data from the specified upstream subsystem
        is stale for the given context.
        Returns (is_stale, staleness_description).
        """

    @abstractmethod
    def list_available_sources(self) -> tuple[str, ...]:
        """
        Return the canonical names of all upstream subsystems this interface
        is configured to read from (e.g., 'ECHO', 'CHRONOS', 'ASTRA',
        'CONSTELLATION').
        """

    @abstractmethod
    def validate_source_subsystem(self, source_subsystem: str) -> bool:
        """
        Validate that the given subsystem name is a permitted upstream source.
        Returns True if permitted; raises JanusFutureMemoryOwnershipViolationError
        if the operation would imply memory ownership.
        """


# ---------------------------------------------------------------------------
# IFutureOrchestrator
# ---------------------------------------------------------------------------


class IFutureOrchestrator(JanusEngineLifecycle, ABC):
    """
    Interface for the Future Orchestrator.

    Responsibility: Coordinate all JANUS engine activities into a single
    coherent FutureAssessment. Acts as JANUS's executive controller.

    The Orchestrator does not re-implement engine logic; it sequences
    engine invocations, aggregates results, and manages confidence and
    uncertainty across the full assessment pipeline.
    """

    @abstractmethod
    def create_future_assessment(
        self, request: FutureAssessmentCreateRequest
    ) -> FutureAssessmentCreateResponse:
        """
        Orchestrate all JANUS engines to produce a comprehensive
        FutureAssessment.

        Sequence:
            1. Validate all sub-requests via IScenarioIntegrityEngine.
            2. Generate scenarios via IScenarioEngine.
            3. Run risk analyses via IFutureRiskEngine.
            4. Run opportunity analyses via IFutureOpportunityEngine.
            5. Run outcome simulations via IOutcomeSimulationEngine.
            6. Create timeline projections via ITimelineProjectionEngine.
            7. Create strategic forecasts via IStrategicForecastEngine.
            8. Generate forecasts via IForecastingEngine.
            9. Evaluate and rank scenarios via IScenarioEvaluationEngine.
            10. Aggregate uncertainty via IUncertaintyEngine.
            11. Assemble and return FutureAssessment.
        """

    @abstractmethod
    def get_future_assessment(
        self, request: FutureAssessmentGetRequest
    ) -> FutureAssessmentGetResponse:
        """Retrieve a FutureAssessment by its assessment_id."""

    @abstractmethod
    def update_assessment_status(
        self, request: FutureAssessmentUpdateStatusRequest
    ) -> FutureAssessmentUpdateStatusResponse:
        """
        Transition a FutureAssessment's FutureAssessmentStatus.
        Validates the transition is legal before applying.
        """

    @abstractmethod
    def check_health(
        self, request: OrchestratorHealthRequest
    ) -> OrchestratorHealthResponse:
        """
        Check the health status of the Orchestrator and all sub-engines.
        Returns a per-engine health map.
        """

    @abstractmethod
    def list_assessments_by_status(
        self, status: FutureAssessmentStatus
    ) -> tuple[FutureAssessment, ...]:
        """Return all FutureAssessments with the given FutureAssessmentStatus."""

    @abstractmethod
    def invalidate_assessment(
        self, assessment_id: str, reason: str, invalidated_by: str
    ) -> FutureAssessment:
        """
        Mark a FutureAssessment as INVALIDATED.
        Returns the updated assessment.
        """

    @abstractmethod
    def regenerate_assessment(
        self,
        assessment_id: str,
        reason: str,
        requested_by: str,
    ) -> FutureAssessmentCreateResponse:
        """
        Trigger a full regeneration of a FutureAssessment using the
        original request parameters. Supersedes the existing assessment.
        """