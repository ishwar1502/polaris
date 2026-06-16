"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: orchestrator.py

FutureOrchestrator — executive coordinator for all JANUS engine activities.

Responsibilities:
  - Forecast workflow orchestration
  - Scenario workflow orchestration
  - Branch-analysis orchestration
  - Counterfactual workflow orchestration
  - Risk-analysis orchestration
  - Opportunity-analysis orchestration
  - Timeline workflow orchestration
  - Simulation workflow orchestration
  - Evaluation workflow orchestration
  - End-to-end forecasting and scenario pipelines
  - Forecast and scenario lifecycle coordination
  - Failure recovery
  - Workflow diagnostics and statistics

Constitutional boundaries (JANUS Critical Laws 1-8):
  - The orchestrator coordinates engines; it never owns domain logic.
  - The orchestrator delegates to engines; it never makes decisions.
  - VEGA makes decisions. ZENITH creates plans.
  - All orchestrated workflows respect bounded-exploration limits.
  - Unlimited exploration is forbidden.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Optional

from .exceptions import (
    JanusAlreadyInitializedError,
    JanusAssessmentNotFoundError,
    JanusAssessmentStatusTransitionError,
    JanusError,
    JanusNotInitializedError,
    JanusOrchestratorCoordinationError,
    JanusShutdownError,
)
from .interfaces import (
    IBranchAnalysisEngine,
    ICounterfactualEngine,
    IForecastingEngine,
    IFutureMemoryInterface,
    IFutureModelingEngine,
    IFutureOpportunityEngine,
    IFutureOrchestrator,
    IFutureRiskEngine,
    IOutcomeSimulationEngine,
    IProbabilityEngine,
    IScenarioEngine,
    IScenarioEvaluationEngine,
    IScenarioIntegrityEngine,
    IStrategicForecastEngine,
    ITimelineProjectionEngine,
    IUncertaintyEngine,
)
from .models import (
    ConfidenceProfile,
    Forecast,
    ForecastMetadata,
    ForecastType,
    FutureAssessment,
    FutureAssessmentStatus,
    OpportunityAssessment,
    OutcomeSimulation,
    RiskAssessment,
    Scenario,
    ScenarioComparison,
    SimulationStatus,
    StrategicForecast,
    TimelineProjection,
    UncertaintyLevel,
    UncertaintyProfile,
)
from .schemas import (
    ForecastGenerateRequest,
    FutureAssessmentCreateRequest,
    FutureAssessmentCreateResponse,
    FutureAssessmentGetRequest,
    FutureAssessmentGetResponse,
    FutureAssessmentUpdateStatusRequest,
    FutureAssessmentUpdateStatusResponse,
    OrchestratorHealthRequest,
    OrchestratorHealthResponse,
    ScenarioCompareRequest,
    ScenarioRankRequest,
    SimulationResultRequest,
)

logger = logging.getLogger(__name__)

_ENGINE_NAME: str = "FutureOrchestrator"
_ENGINE_VERSION: str = "5.1.0"

# ---------------------------------------------------------------------------
# Bounded-exploration defaults (overridable via config)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_SCENARIOS: int = 20
_DEFAULT_MAX_SIMULATIONS: int = 10
_DEFAULT_MIN_CONFIDENCE: float = 0.10
_DEFAULT_MIN_SCENARIO_PROBABILITY: float = 0.05
_DEFAULT_SIMULATION_POLL_MAX_ATTEMPTS: int = 100
_DEFAULT_SIMULATION_POLL_SLEEP_S: float = 0.05

# ---------------------------------------------------------------------------
# Legal FutureAssessmentStatus transitions
# ---------------------------------------------------------------------------

_LEGAL_TRANSITIONS: dict[FutureAssessmentStatus, frozenset[FutureAssessmentStatus]] = {
    FutureAssessmentStatus.PENDING: frozenset({
        FutureAssessmentStatus.IN_PROGRESS,
        FutureAssessmentStatus.INVALIDATED,
    }),
    FutureAssessmentStatus.IN_PROGRESS: frozenset({
        FutureAssessmentStatus.COMPLETE,
        FutureAssessmentStatus.INVALIDATED,
    }),
    FutureAssessmentStatus.COMPLETE: frozenset({
        FutureAssessmentStatus.INVALIDATED,
    }),
    FutureAssessmentStatus.INVALIDATED: frozenset(),
}


# ---------------------------------------------------------------------------
# FutureOrchestrator
# ---------------------------------------------------------------------------


class FutureOrchestrator(IFutureOrchestrator):
    """
    Executive coordinator for all JANUS engine activities.

    FutureOrchestrator sequences engine invocations, aggregates results,
    manages confidence and uncertainty across the full assessment pipeline,
    and produces FutureAssessment deliverables for ODYSSEY, VEGA, ZENITH,
    PROMETHEUS, and DRACO.

    Constitutional constraints:
      - Coordinates engines; never owns domain logic.
      - Delegates to engines; never makes decisions or plans.
      - VEGA selects among evaluated scenarios; this class only evaluates.
      - ZENITH produces plans; this class only sequences engine calls.
      - Bounded-exploration limits are always enforced.

    Thread safety:
      - All public methods are guarded by an internal RLock.
      - Assessment store and counter mutations are atomic within the lock.
    """

    def __init__(
        self,
        *,
        scenario_engine: IScenarioEngine,
        forecasting_engine: IForecastingEngine,
        future_modeling_engine: IFutureModelingEngine,
        branch_analysis_engine: IBranchAnalysisEngine,
        counterfactual_engine: ICounterfactualEngine,
        uncertainty_engine: IUncertaintyEngine,
        future_risk_engine: IFutureRiskEngine,
        future_opportunity_engine: IFutureOpportunityEngine,
        outcome_simulation_engine: IOutcomeSimulationEngine,
        strategic_forecast_engine: IStrategicForecastEngine,
        timeline_projection_engine: ITimelineProjectionEngine,
        probability_engine: IProbabilityEngine,
        scenario_evaluation_engine: IScenarioEvaluationEngine,
        scenario_integrity_engine: IScenarioIntegrityEngine,
        future_memory_interface: IFutureMemoryInterface,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._scenario_engine = scenario_engine
        self._forecasting_engine = forecasting_engine
        self._future_modeling_engine = future_modeling_engine
        self._branch_analysis_engine = branch_analysis_engine
        self._counterfactual_engine = counterfactual_engine
        self._uncertainty_engine = uncertainty_engine
        self._future_risk_engine = future_risk_engine
        self._future_opportunity_engine = future_opportunity_engine
        self._outcome_simulation_engine = outcome_simulation_engine
        self._strategic_forecast_engine = strategic_forecast_engine
        self._timeline_projection_engine = timeline_projection_engine
        self._probability_engine = probability_engine
        self._scenario_evaluation_engine = scenario_evaluation_engine
        self._scenario_integrity_engine = scenario_integrity_engine
        self._future_memory_interface = future_memory_interface
        self._config: dict[str, Any] = dict(config or {})

        self._lock = threading.RLock()
        self._initialized: bool = False
        self._shutdown: bool = False

        # Assessment store: assessment_id → FutureAssessment
        self._assessments: dict[str, FutureAssessment] = {}
        # Original request store for regeneration: assessment_id → request
        self._assessment_requests: dict[str, FutureAssessmentCreateRequest] = {}

        # Bounded-exploration configuration
        self._max_scenarios: int = int(
            self._config.get("max_scenarios", _DEFAULT_MAX_SCENARIOS)
        )
        self._max_simulations: int = int(
            self._config.get("max_simulations", _DEFAULT_MAX_SIMULATIONS)
        )
        self._min_confidence: float = float(
            self._config.get("min_confidence", _DEFAULT_MIN_CONFIDENCE)
        )
        self._min_scenario_probability: float = float(
            self._config.get("min_scenario_probability", _DEFAULT_MIN_SCENARIO_PROBABILITY)
        )
        self._simulation_poll_max_attempts: int = int(
            self._config.get(
                "simulation_poll_max_attempts", _DEFAULT_SIMULATION_POLL_MAX_ATTEMPTS
            )
        )
        self._simulation_poll_sleep_s: float = float(
            self._config.get("simulation_poll_sleep_s", _DEFAULT_SIMULATION_POLL_SLEEP_S)
        )

        # Workflow diagnostic counters
        self._assessments_created: int = 0
        self._assessments_invalidated: int = 0
        self._assessments_regenerated: int = 0
        self._workflow_step_failures: int = 0

    # ------------------------------------------------------------------
    # Lifecycle — JanusEngineLifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._shutdown:
                raise JanusShutdownError(_ENGINE_NAME)
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._initialized = True
            logger.info("%s v%s initialized.", _ENGINE_NAME, _ENGINE_VERSION)

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._initialized = False
            logger.info(
                "%s shut down. Assessments retained in store: %d.",
                _ENGINE_NAME,
                len(self._assessments),
            )

    @property
    def is_initialized(self) -> bool:
        with self._lock:
            return self._initialized

    @property
    def engine_name(self) -> str:
        return _ENGINE_NAME

    @property
    def engine_version(self) -> str:
        return _ENGINE_VERSION

    # ------------------------------------------------------------------
    # Lifecycle guard
    # ------------------------------------------------------------------

    def _require_running(self) -> None:
        """Raise appropriate lifecycle error if the orchestrator is not operational."""
        if self._shutdown:
            raise JanusShutdownError(_ENGINE_NAME)
        if not self._initialized:
            raise JanusNotInitializedError(_ENGINE_NAME)

    # ------------------------------------------------------------------
    # IFutureOrchestrator — create_future_assessment
    # ------------------------------------------------------------------

    def create_future_assessment(
        self, request: FutureAssessmentCreateRequest
    ) -> FutureAssessmentCreateResponse:
        """
        Orchestrate all JANUS engines to produce a comprehensive FutureAssessment.

        Sequence:
          1.  Validate all sub-requests via IScenarioIntegrityEngine.
          2.  Generate scenarios via IScenarioEngine (bounded).
          3.  Run risk analyses via IFutureRiskEngine.
          4.  Run opportunity analyses via IFutureOpportunityEngine.
          5.  Run outcome simulations via IOutcomeSimulationEngine (bounded).
          6.  Create timeline projections via ITimelineProjectionEngine.
          7.  Create strategic forecasts via IStrategicForecastEngine.
          8.  Generate forecasts via IForecastingEngine (one per scenario).
          9.  Evaluate and rank scenarios via IScenarioEvaluationEngine.
          10. Aggregate uncertainty via IUncertaintyEngine.
          11. Assemble, persist, and return the FutureAssessment.
        """
        with self._lock:
            self._require_running()
            return self._execute_assessment_pipeline(request)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _execute_assessment_pipeline(
        self, request: FutureAssessmentCreateRequest
    ) -> FutureAssessmentCreateResponse:
        """Full 11-step orchestration pipeline for a FutureAssessment."""
        logger.info(
            "[%s] Starting assessment pipeline: %r (context=%r).",
            _ENGINE_NAME,
            request.title,
            request.context[:80],
        )

        # Step 1 — Integrity validation
        self._validate_sub_requests(request)

        # Step 2 — Scenario generation (bounded)
        scenarios = self._generate_scenarios(request)

        # Step 3 — Risk analyses
        risk_assessments = self._run_risk_analyses(request)

        # Step 4 — Opportunity analyses
        opportunity_assessments = self._run_opportunity_analyses(request)

        # Step 5 — Outcome simulations (bounded)
        simulations = self._run_simulations(request)

        # Step 6 — Timeline projections
        timeline_projections = self._create_timeline_projections(request)

        # Step 7 — Strategic forecasts
        strategic_forecasts = self._create_strategic_forecasts(request)

        # Step 8 — Forecast generation
        forecasts = self._generate_forecasts(
            request,
            scenarios,
            strategic_forecasts,
        )

        # Step 9 — Scenario evaluation
        scenario_comparison = self._evaluate_scenarios(scenarios)

        # Step 10 — Uncertainty aggregation
        overall_uncertainty, overall_confidence = self._aggregate_uncertainty(
            scenarios,
            risk_assessments,
            opportunity_assessments,
            simulations,
            forecasts,
            timeline_projections,
        )

        # Step 11 — Assembly and persistence
        assessment = FutureAssessment.create(
            title=request.title,
            description=request.description,
            context=request.context,
            scenarios=list(scenarios),
            forecasts=list(forecasts),
            simulations=list(simulations),
            risk_assessments=list(risk_assessments),
            opportunity_assessments=list(opportunity_assessments),
            timeline_projections=list(timeline_projections),
            counterfactuals=[],
            strategic_forecasts=list(strategic_forecasts),
            overall_uncertainty=overall_uncertainty,
            overall_confidence=overall_confidence,
            summary=request.summary,
            scenario_comparison=scenario_comparison,
        )
        assessment.status = FutureAssessmentStatus.COMPLETE
        assessment.updated_at = datetime.utcnow()

        self._assessments[assessment.assessment_id] = assessment
        self._assessment_requests[assessment.assessment_id] = request
        self._assessments_created += 1

        logger.info(
            "[%s] Assessment pipeline complete: %s "
            "(%d scenarios, %d risks, %d opportunities, %d simulations, "
            "%d projections, %d strategic forecasts, %d forecasts).",
            _ENGINE_NAME,
            assessment.assessment_id,
            len(assessment.scenarios),
            len(assessment.risk_assessments),
            len(assessment.opportunity_assessments),
            len(assessment.simulations),
            len(assessment.timeline_projections),
            len(assessment.strategic_forecasts),
            len(assessment.forecasts),
        )

        return FutureAssessmentCreateResponse(
            assessment=assessment,
            created_at=datetime.utcnow(),
            engine_version=_ENGINE_VERSION,
        )

    # ------------------------------------------------------------------
    # Step 1 — Integrity validation
    # ------------------------------------------------------------------

    def _validate_sub_requests(self, request: FutureAssessmentCreateRequest) -> None:
        """
        Validate all sub-requests via IScenarioIntegrityEngine.
        Enforces uncertainty invariants (JANUS Law 6) and evidence invariants
        on every request object in the assessment payload.
        """
        self._scenario_integrity_engine.check_constitutional_boundary(
            engine_name=_ENGINE_NAME,
            operation="create_future_assessment",
            rightful_owner="FutureOrchestrator",
        )

        for risk_req in request.risk_analysis_requests:
            self._scenario_integrity_engine.enforce_uncertainty_invariant(
                artifact_id=risk_req.title,
                artifact_type="RiskAnalysisRequest",
                uncertainty=risk_req.uncertainty,
            )
            self._scenario_integrity_engine.enforce_evidence_invariant(
                artifact_id=risk_req.title,
                artifact_type="RiskAnalysisRequest",
                evidence=risk_req.evidence,
            )

        for opp_req in request.opportunity_analysis_requests:
            self._scenario_integrity_engine.enforce_uncertainty_invariant(
                artifact_id=opp_req.title,
                artifact_type="OpportunityAnalysisRequest",
                uncertainty=opp_req.uncertainty,
            )
            self._scenario_integrity_engine.enforce_evidence_invariant(
                artifact_id=opp_req.title,
                artifact_type="OpportunityAnalysisRequest",
                evidence=opp_req.evidence,
            )

        for sim_req in request.simulation_requests:
            self._scenario_integrity_engine.enforce_uncertainty_invariant(
                artifact_id=sim_req.title,
                artifact_type="SimulationRunRequest",
                uncertainty=sim_req.uncertainty,
            )

        for tl_req in request.timeline_requests:
            self._scenario_integrity_engine.enforce_uncertainty_invariant(
                artifact_id=tl_req.title,
                artifact_type="TimelineProjectionCreateRequest",
                uncertainty=tl_req.uncertainty,
            )
            self._scenario_integrity_engine.enforce_evidence_invariant(
                artifact_id=tl_req.title,
                artifact_type="TimelineProjectionCreateRequest",
                evidence=tl_req.evidence,
            )

        for sf_req in request.strategic_forecast_requests:
            self._scenario_integrity_engine.enforce_uncertainty_invariant(
                artifact_id=sf_req.title,
                artifact_type="StrategicForecastCreateRequest",
                uncertainty=sf_req.uncertainty,
            )
            self._scenario_integrity_engine.enforce_evidence_invariant(
                artifact_id=sf_req.title,
                artifact_type="StrategicForecastCreateRequest",
                evidence=sf_req.evidence,
            )

        logger.debug("[%s] Sub-request integrity validation passed.", _ENGINE_NAME)

    # ------------------------------------------------------------------
    # Step 2 — Scenario generation
    # ------------------------------------------------------------------

    def _generate_scenarios(
        self, request: FutureAssessmentCreateRequest
    ) -> tuple[Scenario, ...]:
        """
        Generate scenarios via IScenarioEngine.

        Bounded exploration:
          - Enforces self._max_scenarios as an absolute ceiling.
          - Prunes scenarios whose confidence.overall falls below
            self._min_confidence.
          - Respects request.scenario_generation_request.max_branches_per_scenario.
        """
        gen_req = request.scenario_generation_request
        logger.debug(
            "[%s] Generating scenarios for: %r (types=%s, horizon=%s).",
            _ENGINE_NAME,
            gen_req.decision_context[:80],
            [t.name for t in gen_req.scenario_types],
            gen_req.horizon.name,
        )

        response = self._scenario_engine.generate_scenarios(gen_req)
        scenarios: tuple[Scenario, ...] = response.scenarios

        # Bounded exploration: absolute scenario ceiling
        if len(scenarios) > self._max_scenarios:
            logger.warning(
                "[%s] Scenario count %d exceeds configured ceiling %d; truncating.",
                _ENGINE_NAME,
                len(scenarios),
                self._max_scenarios,
            )
            scenarios = scenarios[: self._max_scenarios]

        # Bounded exploration: confidence pruning
        pruned = tuple(
            s for s in scenarios
            if s.confidence.overall >= self._min_confidence
        )
        if len(pruned) < len(scenarios):
            discarded = len(scenarios) - len(pruned)
            logger.debug(
                "[%s] Pruned %d scenario(s) below confidence threshold %.2f.",
                _ENGINE_NAME,
                discarded,
                self._min_confidence,
            )
            # Retain at least one scenario even if all fall below threshold.
            scenarios = pruned if pruned else scenarios[:1]

        logger.debug(
            "[%s] %d scenario(s) retained after bounded pruning.",
            _ENGINE_NAME,
            len(scenarios),
        )
        return scenarios

    # ------------------------------------------------------------------
    # Step 3 — Risk analyses
    # ------------------------------------------------------------------

    def _run_risk_analyses(
        self, request: FutureAssessmentCreateRequest
    ) -> tuple[RiskAssessment, ...]:
        """
        Run all risk analyses via IFutureRiskEngine.
        Individual failures are logged and skipped; the pipeline continues.
        """
        risk_assessments: list[RiskAssessment] = []
        total = len(request.risk_analysis_requests)
        for i, risk_req in enumerate(request.risk_analysis_requests):
            try:
                response = self._future_risk_engine.analyze_risk(risk_req)
                risk_assessments.append(response.risk_assessment)
                logger.debug(
                    "[%s] Risk analysis %d/%d complete: %s → level=%s.",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    risk_req.title,
                    response.computed_level.name,
                )
            except JanusError as exc:
                logger.warning(
                    "[%s] Risk analysis %d/%d (%r) failed — %s: %s; skipping.",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    risk_req.title,
                    type(exc).__name__,
                    exc,
                )
                self._workflow_step_failures += 1
        return tuple(risk_assessments)

    # ------------------------------------------------------------------
    # Step 4 — Opportunity analyses
    # ------------------------------------------------------------------

    def _run_opportunity_analyses(
        self, request: FutureAssessmentCreateRequest
    ) -> tuple[OpportunityAssessment, ...]:
        """
        Run all opportunity analyses via IFutureOpportunityEngine.
        Individual failures are logged and skipped; the pipeline continues.
        """
        opportunity_assessments: list[OpportunityAssessment] = []
        total = len(request.opportunity_analysis_requests)
        for i, opp_req in enumerate(request.opportunity_analysis_requests):
            try:
                response = self._future_opportunity_engine.analyze_opportunity(opp_req)
                opportunity_assessments.append(response.opportunity_assessment)
                logger.debug(
                    "[%s] Opportunity analysis %d/%d complete: %s → level=%s.",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    opp_req.title,
                    response.computed_level.name,
                )
            except JanusError as exc:
                logger.warning(
                    "[%s] Opportunity analysis %d/%d (%r) failed — %s: %s; skipping.",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    opp_req.title,
                    type(exc).__name__,
                    exc,
                )
                self._workflow_step_failures += 1
        return tuple(opportunity_assessments)

    # ------------------------------------------------------------------
    # Step 5 — Outcome simulations
    # ------------------------------------------------------------------

    def _run_simulations(
        self, request: FutureAssessmentCreateRequest
    ) -> tuple[OutcomeSimulation, ...]:
        """
        Run outcome simulations via IOutcomeSimulationEngine.

        Bounded exploration:
          - Enforces self._max_simulations as an absolute ceiling.
          - Polls each simulation until terminal status or poll limit.
        Individual failures are logged and skipped.
        """
        sim_requests = request.simulation_requests

        # Bounded exploration: absolute simulation ceiling
        if len(sim_requests) > self._max_simulations:
            logger.warning(
                "[%s] Simulation count %d exceeds configured ceiling %d; truncating.",
                _ENGINE_NAME,
                len(sim_requests),
                self._max_simulations,
            )
            sim_requests = sim_requests[: self._max_simulations]

        simulations: list[OutcomeSimulation] = []
        total = len(sim_requests)
        for i, sim_req in enumerate(sim_requests):
            try:
                run_response = self._outcome_simulation_engine.run_simulation(sim_req)
                simulation_id = run_response.simulation.simulation_id
                simulation = self._poll_simulation_to_terminal(simulation_id)
                simulations.append(simulation)
                logger.debug(
                    "[%s] Simulation %d/%d complete: %s → status=%s.",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    simulation_id,
                    simulation.status.name,
                )
            except JanusError as exc:
                logger.warning(
                    "[%s] Simulation %d/%d (%r) failed — %s: %s; skipping.",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    sim_req.title,
                    type(exc).__name__,
                    exc,
                )
                self._workflow_step_failures += 1
        return tuple(simulations)

    def _poll_simulation_to_terminal(self, simulation_id: str) -> OutcomeSimulation:
        """
        Poll a simulation until it reaches a terminal status
        (COMPLETED, FAILED, ABORTED) or the attempt ceiling is exhausted.
        """
        result_req = SimulationResultRequest(simulation_id=simulation_id)
        result = self._outcome_simulation_engine.get_simulation_result(result_req)

        for attempt in range(self._simulation_poll_max_attempts):
            if result.status in (
                SimulationStatus.COMPLETED,
                SimulationStatus.FAILED,
                SimulationStatus.ABORTED,
            ):
                return result.simulation
            time.sleep(self._simulation_poll_sleep_s)
            result = self._outcome_simulation_engine.get_simulation_result(result_req)

        logger.warning(
            "[%s] Simulation %s did not reach terminal status after %d poll attempts; "
            "returning last known state (%s).",
            _ENGINE_NAME,
            simulation_id,
            self._simulation_poll_max_attempts,
            result.status.name,
        )
        return result.simulation

    # ------------------------------------------------------------------
    # Step 6 — Timeline projections
    # ------------------------------------------------------------------

    def _create_timeline_projections(
        self, request: FutureAssessmentCreateRequest
    ) -> tuple[TimelineProjection, ...]:
        """
        Create timeline projections via ITimelineProjectionEngine.
        Individual failures are logged and skipped; the pipeline continues.
        """
        projections: list[TimelineProjection] = []
        total = len(request.timeline_requests)
        for i, tl_req in enumerate(request.timeline_requests):
            try:
                response = self._timeline_projection_engine.create_projection(tl_req)
                projections.append(response.projection)
                logger.debug(
                    "[%s] Timeline projection %d/%d created: %s.",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    response.projection.projection_id,
                )
            except JanusError as exc:
                logger.warning(
                    "[%s] Timeline projection %d/%d (%r) failed — %s: %s; skipping.",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    tl_req.title,
                    type(exc).__name__,
                    exc,
                )
                self._workflow_step_failures += 1
        return tuple(projections)

    # ------------------------------------------------------------------
    # Step 7 — Strategic forecasts
    # ------------------------------------------------------------------

    def _create_strategic_forecasts(
        self, request: FutureAssessmentCreateRequest
    ) -> tuple[StrategicForecast, ...]:
        """
        Create strategic forecasts via IStrategicForecastEngine.
        Individual failures are logged and skipped; the pipeline continues.
        """
        strategic_forecasts: list[StrategicForecast] = []
        total = len(request.strategic_forecast_requests)
        for i, sf_req in enumerate(request.strategic_forecast_requests):
            try:
                response = self._strategic_forecast_engine.create_strategic_forecast(sf_req)
                strategic_forecasts.append(response.strategic_forecast)
                logger.debug(
                    "[%s] Strategic forecast %d/%d created: %s (domain=%s).",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    response.strategic_forecast.strategic_forecast_id,
                    response.strategic_forecast.domain,
                )
            except JanusError as exc:
                logger.warning(
                    "[%s] Strategic forecast %d/%d (%r) failed — %s: %s; skipping.",
                    _ENGINE_NAME,
                    i + 1,
                    total,
                    sf_req.title,
                    type(exc).__name__,
                    exc,
                )
                self._workflow_step_failures += 1
        return tuple(strategic_forecasts)

    # ------------------------------------------------------------------
    # Step 8 — Forecast generation
    # ------------------------------------------------------------------

    def _generate_forecasts(
        self,
        request: FutureAssessmentCreateRequest,
        scenarios: tuple[Scenario, ...],
        strategic_forecasts: tuple[StrategicForecast, ...],
    ) -> tuple[Forecast, ...]:
        """
        Generate probabilistic forecasts via IForecastingEngine.

        Sources:
          (a) Collect forecasts embedded in StrategicForecast deliverables and
              validate each via IScenarioIntegrityEngine before inclusion.
          (b) Synthesise one ForecastGenerateRequest per scenario and delegate
              to IForecastingEngine, using the scenario's uncertainty, evidence,
              and the scenario-generation horizon.
        """
        forecasts: list[Forecast] = []

        # (a) Collect validated forecasts from strategic forecast deliverables.
        for sf in strategic_forecasts:
            for embedded_fc in sf.strategic_outcome_forecasts:
                is_valid, violations = self._scenario_integrity_engine.validate_forecast(
                    embedded_fc
                )
                if is_valid:
                    forecasts.append(embedded_fc)
                else:
                    logger.debug(
                        "[%s] Embedded forecast %s in strategic forecast %s failed "
                        "integrity validation: %s; excluding.",
                        _ENGINE_NAME,
                        embedded_fc.forecast_id,
                        sf.strategic_forecast_id,
                        "; ".join(violations),
                    )

        # (b) Synthesise one forecast per scenario via IForecastingEngine.
        horizon = request.scenario_generation_request.horizon
        for scenario in scenarios:
            try:
                metadata = ForecastMetadata.create(
                    model_version=_ENGINE_VERSION,
                    data_sources=list(scenario.evidence.sources),
                    horizon=horizon,
                    generated_by=_ENGINE_NAME,
                )
                forecast_req = ForecastGenerateRequest(
                    title=f"Scenario Forecast: {scenario.title}",
                    description=(
                        f"Probabilistic forecast synthesised from scenario "
                        f"'{scenario.title}' for assessment context: "
                        f"{request.context[:120]}"
                    ),
                    forecast_type=ForecastType.PROBABILISTIC,
                    horizon=horizon,
                    uncertainty=scenario.uncertainty,
                    evidence=scenario.evidence,
                    metadata=metadata,
                )
                response = self._forecasting_engine.generate_forecast(forecast_req)
                forecasts.append(response.forecast)
                logger.debug(
                    "[%s] Forecast generated for scenario %s: %s.",
                    _ENGINE_NAME,
                    scenario.scenario_id,
                    response.forecast.forecast_id,
                )
            except JanusError as exc:
                logger.warning(
                    "[%s] Forecast generation for scenario %s failed — %s: %s; skipping.",
                    _ENGINE_NAME,
                    scenario.scenario_id,
                    type(exc).__name__,
                    exc,
                )
                self._workflow_step_failures += 1

        return tuple(forecasts)

    # ------------------------------------------------------------------
    # Step 9 — Scenario evaluation
    # ------------------------------------------------------------------

    def _evaluate_scenarios(
        self, scenarios: tuple[Scenario, ...]
    ) -> Optional[ScenarioComparison]:
        """
        Evaluate and rank scenarios via IScenarioEvaluationEngine.

        Produces a ScenarioComparison with ranking and trade-off analysis.
        Returns None when fewer than two scenarios are available.

        Constitutional boundary: JANUS evaluates; VEGA selects.
        The returned comparison carries a recommended_for_review field,
        not a decision.
        """
        if len(scenarios) < 2:
            logger.debug(
                "[%s] Only %d scenario(s) available; skipping evaluation.",
                _ENGINE_NAME,
                len(scenarios),
            )
            return None

        scenario_ids = tuple(s.scenario_id for s in scenarios)

        try:
            # Rank scenarios first (bounded exploration: equal weights)
            rank_req = ScenarioRankRequest(
                scenario_ids=scenario_ids,
                weight_risk=0.5,
                weight_opportunity=0.5,
            )
            self._scenario_evaluation_engine.rank_scenarios(rank_req)

            # Produce the full comparison artifact
            compare_confidence = ConfidenceProfile.create(
                overall=0.70,
                data_quality=0.70,
                model_fit=0.70,
                signal_strength=0.70,
                notes="Orchestrator-assembled scenario comparison confidence.",
            )
            compare_req = ScenarioCompareRequest(
                scenario_ids=scenario_ids,
                title=(
                    f"JANUS Scenario Comparison — "
                    f"{len(scenarios)} scenario(s) evaluated"
                ),
                confidence=compare_confidence,
            )
            compare_response = self._scenario_evaluation_engine.compare_scenarios(compare_req)
            logger.debug(
                "[%s] Scenario evaluation complete: %d scenario(s) ranked; "
                "recommended_for_review=%s.",
                _ENGINE_NAME,
                len(scenarios),
                compare_response.recommended_for_review,
            )
            return compare_response.comparison

        except JanusError as exc:
            logger.warning(
                "[%s] Scenario evaluation failed — %s: %s; proceeding without comparison.",
                _ENGINE_NAME,
                type(exc).__name__,
                exc,
            )
            self._workflow_step_failures += 1
            return None

    # ------------------------------------------------------------------
    # Step 10 — Uncertainty aggregation
    # ------------------------------------------------------------------

    def _aggregate_uncertainty(
        self,
        scenarios: tuple[Scenario, ...],
        risk_assessments: tuple[RiskAssessment, ...],
        opportunity_assessments: tuple[OpportunityAssessment, ...],
        simulations: tuple[OutcomeSimulation, ...],
        forecasts: tuple[Forecast, ...],
        timeline_projections: tuple[TimelineProjection, ...],
    ) -> tuple[UncertaintyProfile, ConfidenceProfile]:
        """
        Aggregate uncertainty across all JANUS artifacts via IUncertaintyEngine.
        Returns (overall_uncertainty, overall_confidence).
        """
        profiles: list[UncertaintyProfile] = []

        for s in scenarios:
            profiles.append(s.uncertainty)
        for r in risk_assessments:
            profiles.append(r.uncertainty)
        for o in opportunity_assessments:
            profiles.append(o.uncertainty)
        for sim in simulations:
            profiles.append(sim.uncertainty)
        for fc in forecasts:
            profiles.append(fc.uncertainty)
        for tp in timeline_projections:
            profiles.append(tp.uncertainty)

        if not profiles:
            overall_uncertainty = UncertaintyProfile.create(
                level=UncertaintyLevel.EXTREME,
                known_risks=[],
                unknown_risk_exposure=1.0,
                volatility_score=1.0,
                external_factors=[],
                market_sensitivity=1.0,
                technology_sensitivity=1.0,
                notes="No artifacts were available to aggregate uncertainty from.",
            )
        else:
            overall_uncertainty = self._uncertainty_engine.aggregate_uncertainty(
                tuple(profiles)
            )

        # Derive overall confidence from scenario confidence profiles (mean of component scores).
        if scenarios:
            n = len(scenarios)
            overall_confidence = ConfidenceProfile.create(
                overall=sum(s.confidence.overall for s in scenarios) / n,
                data_quality=sum(s.confidence.data_quality for s in scenarios) / n,
                model_fit=sum(s.confidence.model_fit for s in scenarios) / n,
                signal_strength=sum(s.confidence.signal_strength for s in scenarios) / n,
                notes=(
                    f"Mean of {n} scenario confidence profile(s) "
                    f"assembled by {_ENGINE_NAME}."
                ),
            )
        else:
            overall_confidence = ConfidenceProfile.create(
                overall=0.10,
                data_quality=0.10,
                model_fit=0.10,
                signal_strength=0.10,
                notes="No scenarios available; confidence defaults to minimum.",
            )

        return overall_uncertainty, overall_confidence

    # ------------------------------------------------------------------
    # IFutureOrchestrator — retrieval and lifecycle management
    # ------------------------------------------------------------------

    def get_future_assessment(
        self, request: FutureAssessmentGetRequest
    ) -> FutureAssessmentGetResponse:
        with self._lock:
            self._require_running()
            assessment = self._get_assessment_or_raise(request.assessment_id)
            return FutureAssessmentGetResponse(
                assessment=assessment,
                retrieved_at=datetime.utcnow(),
            )

    def update_assessment_status(
        self, request: FutureAssessmentUpdateStatusRequest
    ) -> FutureAssessmentUpdateStatusResponse:
        with self._lock:
            self._require_running()
            assessment = self._get_assessment_or_raise(request.assessment_id)
            self._assert_legal_transition(
                request.assessment_id,
                assessment.status,
                request.target_status,
            )
            previous_status = assessment.status
            assessment.status = request.target_status
            assessment.updated_at = datetime.utcnow()
            logger.debug(
                "[%s] Assessment %s status: %s → %s (by=%s).",
                _ENGINE_NAME,
                request.assessment_id,
                previous_status.name,
                request.target_status.name,
                request.updated_by,
            )
            return FutureAssessmentUpdateStatusResponse(
                assessment_id=request.assessment_id,
                previous_status=previous_status,
                new_status=request.target_status,
                updated_at=datetime.utcnow(),
            )

    def check_health(self, request: OrchestratorHealthRequest) -> OrchestratorHealthResponse:
        with self._lock:
            orchestrator_healthy = self._initialized and not self._shutdown

        sub_engines: dict[str, object] = {
            "scenario_engine": self._scenario_engine,
            "forecasting_engine": self._forecasting_engine,
            "future_modeling_engine": self._future_modeling_engine,
            "branch_analysis_engine": self._branch_analysis_engine,
            "counterfactual_engine": self._counterfactual_engine,
            "uncertainty_engine": self._uncertainty_engine,
            "future_risk_engine": self._future_risk_engine,
            "future_opportunity_engine": self._future_opportunity_engine,
            "outcome_simulation_engine": self._outcome_simulation_engine,
            "strategic_forecast_engine": self._strategic_forecast_engine,
            "timeline_projection_engine": self._timeline_projection_engine,
            "probability_engine": self._probability_engine,
            "scenario_evaluation_engine": self._scenario_evaluation_engine,
            "scenario_integrity_engine": self._scenario_integrity_engine,
            "future_memory_interface": self._future_memory_interface,
        }

        engine_statuses: dict[str, bool] = {}
        for name, engine in sub_engines.items():
            try:
                engine_statuses[name] = bool(engine.is_initialized)
            except Exception:  # noqa: BLE001
                engine_statuses[name] = False

        unhealthy_engines = [k for k, v in engine_statuses.items() if not v]
        if unhealthy_engines:
            orchestrator_healthy = False
            notes = f"Unhealthy engines: {', '.join(unhealthy_engines)}."
        else:
            notes = "All engines operational."

        return OrchestratorHealthResponse(
            orchestrator_healthy=orchestrator_healthy,
            engine_statuses=engine_statuses,
            checked_at=datetime.utcnow(),
            notes=notes,
        )

    def list_assessments_by_status(
        self, status: FutureAssessmentStatus
    ) -> tuple[FutureAssessment, ...]:
        with self._lock:
            self._require_running()
            return tuple(
                a for a in self._assessments.values() if a.status == status
            )

    def invalidate_assessment(
        self, assessment_id: str, reason: str, invalidated_by: str
    ) -> FutureAssessment:
        with self._lock:
            self._require_running()
            assessment = self._get_assessment_or_raise(assessment_id)
            self._assert_legal_transition(
                assessment_id,
                assessment.status,
                FutureAssessmentStatus.INVALIDATED,
            )
            assessment.status = FutureAssessmentStatus.INVALIDATED
            assessment.updated_at = datetime.utcnow()
            self._assessments_invalidated += 1
            logger.info(
                "[%s] Assessment %s invalidated by %s: %s",
                _ENGINE_NAME,
                assessment_id,
                invalidated_by,
                reason,
            )
            return assessment

    def regenerate_assessment(
        self,
        assessment_id: str,
        reason: str,
        requested_by: str,
    ) -> FutureAssessmentCreateResponse:
        """
        Trigger a full regeneration of a FutureAssessment using its original
        request parameters. The existing assessment is invalidated before the
        new pipeline executes.
        """
        with self._lock:
            self._require_running()

            original_request = self._assessment_requests.get(assessment_id)
            if original_request is None:
                raise JanusAssessmentNotFoundError(assessment_id)

            # Invalidate the existing assessment unconditionally before regeneration.
            existing = self._assessments.get(assessment_id)
            if existing is not None and existing.status != FutureAssessmentStatus.INVALIDATED:
                existing.status = FutureAssessmentStatus.INVALIDATED
                existing.updated_at = datetime.utcnow()

            logger.info(
                "[%s] Regenerating assessment %s (requested_by=%s): %s",
                _ENGINE_NAME,
                assessment_id,
                requested_by,
                reason,
            )
            self._assessments_regenerated += 1

            return self._execute_assessment_pipeline(original_request)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_assessment_or_raise(self, assessment_id: str) -> FutureAssessment:
        assessment = self._assessments.get(assessment_id)
        if assessment is None:
            raise JanusAssessmentNotFoundError(assessment_id)
        return assessment

    def _assert_legal_transition(
        self,
        assessment_id: str,
        current: FutureAssessmentStatus,
        target: FutureAssessmentStatus,
    ) -> None:
        allowed = _LEGAL_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise JanusAssessmentStatusTransitionError(
                assessment_id=assessment_id,
                current=current.name,
                target=target.name,
            )

    # ------------------------------------------------------------------
    # Workflow diagnostics and statistics
    # ------------------------------------------------------------------

    def workflow_diagnostics(self) -> dict[str, Any]:
        """
        Return a structured diagnostic snapshot of the orchestrator's workflow
        state, including per-engine availability, assessment counts by status,
        and bounded-exploration configuration.

        Used by JanusSubsystem.get_diagnostics().
        """
        with self._lock:
            initialized = self._initialized
            shutdown = self._shutdown
            total_assessments = len(self._assessments)
            assessments_by_status: dict[str, int] = {
                status.name: sum(
                    1 for a in self._assessments.values() if a.status == status
                )
                for status in FutureAssessmentStatus
            }
            assessments_created = self._assessments_created
            assessments_invalidated = self._assessments_invalidated
            assessments_regenerated = self._assessments_regenerated
            workflow_step_failures = self._workflow_step_failures

        engine_availability: dict[str, bool] = {}
        for attr_name, engine in (
            ("scenario_engine", self._scenario_engine),
            ("forecasting_engine", self._forecasting_engine),
            ("future_modeling_engine", self._future_modeling_engine),
            ("branch_analysis_engine", self._branch_analysis_engine),
            ("counterfactual_engine", self._counterfactual_engine),
            ("uncertainty_engine", self._uncertainty_engine),
            ("future_risk_engine", self._future_risk_engine),
            ("future_opportunity_engine", self._future_opportunity_engine),
            ("outcome_simulation_engine", self._outcome_simulation_engine),
            ("strategic_forecast_engine", self._strategic_forecast_engine),
            ("timeline_projection_engine", self._timeline_projection_engine),
            ("probability_engine", self._probability_engine),
            ("scenario_evaluation_engine", self._scenario_evaluation_engine),
            ("scenario_integrity_engine", self._scenario_integrity_engine),
            ("future_memory_interface", self._future_memory_interface),
        ):
            try:
                engine_availability[attr_name] = bool(engine.is_initialized)
            except Exception:  # noqa: BLE001
                engine_availability[attr_name] = False

        return {
            "engine_name": _ENGINE_NAME,
            "engine_version": _ENGINE_VERSION,
            "initialized": initialized,
            "shutdown": shutdown,
            "total_assessments": total_assessments,
            "assessments_by_status": assessments_by_status,
            "assessments_created": assessments_created,
            "assessments_invalidated": assessments_invalidated,
            "assessments_regenerated": assessments_regenerated,
            "workflow_step_failures": workflow_step_failures,
            "bounded_exploration_limits": {
                "max_scenarios": self._max_scenarios,
                "max_simulations": self._max_simulations,
                "min_confidence": self._min_confidence,
                "min_scenario_probability": self._min_scenario_probability,
                "simulation_poll_max_attempts": self._simulation_poll_max_attempts,
            },
            "engine_availability": engine_availability,
        }

    def workflow_statistics(self) -> dict[str, Any]:
        """
        Return aggregated workflow statistics for JanusSubsystem.get_statistics().
        """
        with self._lock:
            return {
                "engine_name": _ENGINE_NAME,
                "engine_version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "assessments_created": self._assessments_created,
                "assessments_invalidated": self._assessments_invalidated,
                "assessments_regenerated": self._assessments_regenerated,
                "workflow_step_failures": self._workflow_step_failures,
                "total_stored_assessments": len(self._assessments),
                "assessments_by_status": {
                    status.name: sum(
                        1 for a in self._assessments.values() if a.status == status
                    )
                    for status in FutureAssessmentStatus
                },
            }

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            initialized = self._initialized
            shutdown = self._shutdown
            total = len(self._assessments)
        return (
            f"FutureOrchestrator("
            f"version={_ENGINE_VERSION!r}, "
            f"initialized={initialized}, "
            f"shutdown={shutdown}, "
            f"assessments={total})"
        )