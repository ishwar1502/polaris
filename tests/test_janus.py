"""
JANUS v5.1 — Comprehensive pytest suite.

Coverage:
  - models (enums, value objects, core models)
  - schemas (request/response dataclasses)
  - exceptions (hierarchy and attributes)
  - interfaces (lifecycle contract)
  - branch_analysis.BranchAnalysisEngine
  - counterfactual.CounterfactualEngine
  - forecasting.ForecastingEngine
  - future_modeling.FutureModelingEngine
  - integrity.ScenarioIntegrityEngine
  - risk_analysis.RiskAnalysisEngine
  - opportunity_analysis.OpportunityAnalysisEngine
  - simulation.OutcomeSimulationEngine
  - timeline_projection.TimelineProjectionEngine
  - strategic_forecasting.StrategicForecastEngine
  - scenario.ScenarioEngine
  - uncertainty.UncertaintyEngine
  - orchestrator.FutureOrchestrator (lifecycle + status management)
"""

from __future__ import annotations

import sys
import os
import threading
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Path setup: the janus package lives at parent/janus inside pkg2
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from subsystems.janus.models import (
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

from subsystems.janus.exceptions import (
    JanusError,
    JanusLifecycleError,
    JanusNotInitializedError,
    JanusAlreadyInitializedError,
    JanusShutdownError,
    JanusValidationError,
    JanusMissingRequiredFieldError,
    JanusInvalidProbabilityError,
    JanusInvalidScoreError,
    JanusInvalidDeltaError,
    JanusProbabilityDistributionError,
    JanusScenarioError,
    JanusScenarioNotFoundError,
    JanusScenarioGenerationError,
    JanusScenarioStatusTransitionError,
    JanusForecastError,
    JanusForecastNotFoundError,
    JanusForecastHorizonError,
    JanusForecastSupersededError,
    JanusFutureModelError,
    JanusFutureModelNotFoundError,
    JanusFutureModelConstructionError,
    JanusSimulationError,
    JanusSimulationNotFoundError,
    JanusBranchAnalysisError,
    JanusBranchNotFoundError,
    JanusBranchConstructionError,
    JanusBranchProbabilityConflictError,
    JanusCounterfactualError,
    JanusCounterfactualNotFoundError,
    JanusCounterfactualDivergenceError,
    JanusCounterfactualConditionError,
    JanusRiskError,
    JanusRiskAssessmentNotFoundError,
    JanusOpportunityError,
    JanusOpportunityAssessmentNotFoundError,
    JanusTimelineProjectionError,
    JanusTimelineProjectionNotFoundError,
    JanusUncertaintyError,
    JanusUncertaintyProfileNotFoundError,
    JanusMissingUncertaintyError,
    JanusIntegrityError,
    JanusImpossibleFutureError,
    JanusFalseCertaintyError,
    JanusEvidenceProfileError,
    JanusConstitutionalViolationError,
    JanusStrategicForecastError,
    JanusStrategicForecastNotFoundError,
    JanusOrchestratorError,
    JanusAssessmentNotFoundError,
    JanusAssessmentStatusTransitionError,
    JanusOrchestratorCoordinationError,
)

from subsystems.janus.schemas import (
    ForecastGenerateRequest,
    ForecastGetRequest,
    ForecastAssessRequest,
    ForecastListByHorizonRequest,
    FutureModelConstructRequest,
    FutureModelGetRequest,
    FutureStateQueryRequest,
    ScenarioGenerateRequest,
    ScenarioGetRequest,
    ScenarioUpdateStatusRequest,
    ScenarioArchiveRequest,
    ScenarioCompareRequest,
    ScenarioRankRequest,
    ScenarioDominanceRequest,
    RiskAnalysisRequest,
    RiskAssessmentGetRequest,
    RiskListByLevelRequest,
    OpportunityAnalysisRequest,
    OpportunityAssessmentGetRequest,
    OpportunityListByLevelRequest,
    TimelineProjectionCreateRequest,
    TimelineProjectionGetRequest,
    TimelineProjectionUpdateRequest,
    TimelineProjectionCompletionQuery,
    UncertaintyProfileCreateRequest,
    UncertaintyProfileGetRequest,
    UncertaintyValidationRequest,
    ProbabilityDistributionCreateRequest,
    ProbabilityLevelQueryRequest,
    ConfidenceProfileCreateRequest,
    StrategicForecastCreateRequest,
    StrategicForecastGetRequest,
    StrategicForecastListRequest,
    SimulationRunRequest,
    SimulationResultRequest,
    SimulationAbortRequest,
    CounterfactualCreateRequest as CFCreateReq,
    CounterfactualGetRequest as CFGetReq,
    CounterfactualCompareRequest as CFCompareReq,
    FutureAssessmentGetRequest,
    FutureAssessmentUpdateStatusRequest,
    OrchestratorHealthRequest,
)

from subsystems.janus.branch_analysis import (
    BranchAnalysisEngine,
    BranchAnalysisConfig,
    BranchStatistics,
    BranchHealthReport,
    BranchDiagnosticsReport,
    BranchLineage,
)
from subsystems.janus.counterfactual import (
    CounterfactualEngine,
    CounterfactualCreateRequest,
    CounterfactualGetRequest,
    CounterfactualCompareRequest,
    CounterfactualStatistics,
    CounterfactualHealthReport,
    CounterfactualDiagnosticsReport,
)
from subsystems.janus.forecasting import ForecastingEngine
from subsystems.janus.future_modeling import FutureModelingEngine
from subsystems.janus.integrity import ScenarioIntegrityEngine
from subsystems.janus.risk_analysis import RiskAnalysisEngine, RiskAnalysisEngineConfig
from subsystems.janus.opportunity_analysis import (
    OpportunityAnalysisEngine,
    OpportunityAnalysisEngineConfig,
)
from subsystems.janus.simulation import OutcomeSimulationEngine, SimulationBoundsConfig
from subsystems.janus.timeline_projection import TimelineProjectionEngine
from subsystems.janus.strategic_forecasting import StrategicForecastEngine
from subsystems.janus.scenario import ScenarioEngine
from subsystems.janus.uncertainty import UncertaintyEngine
from subsystems.janus.orchestrator import FutureOrchestrator

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def make_confidence(overall: float = 0.70) -> ConfidenceProfile:
    return ConfidenceProfile.create(
        overall=overall,
        data_quality=overall,
        model_fit=overall,
        signal_strength=overall,
        notes="test",
    )


def make_evidence(strength: float = 0.60) -> EvidenceProfile:
    return EvidenceProfile.create(
        sources=["source-A"],
        patterns_observed=["pattern-1"],
        contradicting_evidence=[],
        evidence_strength=strength,
    )


def make_uncertainty(level: UncertaintyLevel = UncertaintyLevel.MODERATE) -> UncertaintyProfile:
    return UncertaintyProfile.create(
        level=level,
        known_risks=["risk-A"],
        unknown_risk_exposure=0.30,
        volatility_score=0.25,
        external_factors=["factor-A"],
        market_sensitivity=0.20,
        technology_sensitivity=0.15,
    )


def make_prob_dist(outcomes: dict | None = None) -> ProbabilityDistribution:
    if outcomes is None:
        outcomes = {"outcome_a": 0.60, "outcome_b": 0.25, "outcome_c": 0.15}
    conf = make_confidence(0.65)
    return ProbabilityDistribution.create(
        label="test-dist",
        outcomes=outcomes,
        uncertainty_level=UncertaintyLevel.MODERATE,
        confidence=conf,
    )


def make_risk_factor(impact: float = 0.5, likelihood: float = 0.5) -> RiskFactor:
    return RiskFactor.create(
        name="test-risk",
        description="A test risk factor",
        category="technical",
        impact_score=impact,
        likelihood_score=likelihood,
        time_horizon=ForecastHorizon.ONE_YEAR,
        mitigations=["mitigation-A"],
    )


def make_opportunity_factor(value: float = 0.6, feasibility: float = 0.7) -> OpportunityFactor:
    return OpportunityFactor.create(
        name="test-opportunity",
        description="A test opportunity factor",
        category="market",
        value_score=value,
        feasibility_score=feasibility,
        time_horizon=ForecastHorizon.ONE_YEAR,
        enablers=["enabler-A"],
    )


def make_risk_assessment() -> RiskAssessment:
    return RiskAssessment.create(
        title="Test Risk",
        description="test",
        level=RiskLevel.MODERATE,
        risk_factors=[make_risk_factor()],
        probability_distribution=make_prob_dist(),
        uncertainty=make_uncertainty(),
        horizon=ForecastHorizon.ONE_YEAR,
        confidence=make_confidence(),
        evidence=make_evidence(),
    )


def make_opportunity_assessment() -> OpportunityAssessment:
    return OpportunityAssessment.create(
        title="Test Opportunity",
        description="test",
        level=OpportunityLevel.MODERATE,
        opportunity_factors=[make_opportunity_factor()],
        probability_distribution=make_prob_dist(),
        uncertainty=make_uncertainty(),
        horizon=ForecastHorizon.ONE_YEAR,
        confidence=make_confidence(),
        evidence=make_evidence(),
    )


def make_future_state(
    label: str = "state-A",
    horizon: ForecastHorizon = ForecastHorizon.ONE_YEAR,
    probability: float = 0.60,
) -> FutureState:
    return FutureState.create(
        label=label,
        description="test state",
        horizon=horizon,
        attributes={"key": "value"},
        probability=probability,
        uncertainty=make_uncertainty(),
    )


def make_future_model() -> FutureModel:
    state = make_future_state()
    return FutureModel.create(
        title="Test Model",
        description="desc",
        context="context",
        future_states=[state],
        uncertainty=make_uncertainty(),
        confidence=make_confidence(),
        evidence=make_evidence(),
    )


def make_scenario_branch(probability: float = 0.70) -> ScenarioBranch:
    return ScenarioBranch.create(
        label="branch-A",
        description="test branch",
        triggering_choice="choice-A",
        future_state=make_future_state(),
        probability=probability,
        risk_assessment=make_risk_assessment(),
        opportunity_assessment=make_opportunity_assessment(),
        confidence=make_confidence(),
    )


def make_scenario() -> Scenario:
    return Scenario.create(
        title="Test Scenario",
        description="desc",
        scenario_type=ScenarioType.BASELINE,
        branches=[make_scenario_branch()],
        future_model=make_future_model(),
        uncertainty=make_uncertainty(),
        confidence=make_confidence(),
        evidence=make_evidence(),
        metadata=ScenarioMetadata.create(
            created_by="test",
            source_subsystem="JANUS",
        ),
        risk_score=0.30,
        opportunity_score=0.60,
    )


def make_forecast() -> Forecast:
    return Forecast.create(
        title="Test Forecast",
        description="desc",
        forecast_type=ForecastType.PROBABILISTIC,
        horizon=ForecastHorizon.ONE_YEAR,
        probability_distribution=make_prob_dist(),
        uncertainty=make_uncertainty(),
        confidence=make_confidence(),
        evidence=make_evidence(),
        metadata=ForecastMetadata.create(
            model_version="5.1.0",
            data_sources=["source-A"],
            horizon=ForecastHorizon.ONE_YEAR,
            generated_by="test",
        ),
    )


def make_projection_milestone(
    label: str = "M1",
    probability: float = 0.80,
    is_critical: bool = False,
    offset_days: int = 30,
) -> ProjectionMilestone:
    return ProjectionMilestone.create(
        label=label,
        description="test milestone",
        projected_at=datetime.utcnow() + timedelta(days=offset_days),
        probability=probability,
        is_critical=is_critical,
    )


def make_simulation_outcome(label: str = "primary", probability: float = 0.70) -> SimulationOutcome:
    return SimulationOutcome.create(
        label=label,
        description="test outcome",
        probability=probability,
        short_term_effects=["effect-A"],
        medium_term_effects=["effect-B"],
        long_term_effects=["effect-C"],
        risk_implications=["risk-implication"],
        opportunity_implications=["opp-implication"],
        confidence=make_confidence(),
    )


def make_simulation_metadata() -> SimulationMetadata:
    return SimulationMetadata.create(iterations=100, engine_version="5.1.0")


# ---------------------------------------------------------------------------
# ============================================================
# EXCEPTION HIERARCHY
# ============================================================
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_janus_error_is_exception(self):
        err = JanusError("base error")
        assert isinstance(err, Exception)

    def test_janus_error_engine_context(self):
        err = JanusError("msg", engine="TestEngine", context={"k": "v"})
        assert err.engine == "TestEngine"
        assert err.context == {"k": "v"}

    def test_janus_error_str_includes_engine(self):
        err = JanusError("hello", engine="MyEngine")
        assert "MyEngine" in str(err)

    def test_not_initialized_error(self):
        err = JanusNotInitializedError("TestEngine")
        assert isinstance(err, JanusLifecycleError)
        assert isinstance(err, JanusError)
        assert "TestEngine" in str(err)

    def test_already_initialized_error(self):
        err = JanusAlreadyInitializedError("TestEngine")
        assert isinstance(err, JanusLifecycleError)
        assert "TestEngine" in str(err)

    def test_shutdown_error(self):
        err = JanusShutdownError("TestEngine")
        assert isinstance(err, JanusLifecycleError)
        assert "TestEngine" in str(err)

    def test_validation_error_fields(self):
        err = JanusValidationError("bad field", field="x", value=42, engine="Eng")
        assert err.field == "x"
        assert err.value == 42

    def test_missing_required_field_error(self):
        err = JanusMissingRequiredFieldError("title", engine="Eng")
        assert isinstance(err, JanusValidationError)
        assert "title" in str(err)

    def test_invalid_probability_error(self):
        err = JanusInvalidProbabilityError(1.5)
        assert isinstance(err, JanusValidationError)
        assert "1.5" in str(err)

    def test_invalid_score_error(self):
        err = JanusInvalidScoreError(1.2, field="impact_score")
        assert isinstance(err, JanusValidationError)

    def test_invalid_delta_error(self):
        err = JanusInvalidDeltaError(-1.5, field="delta_risk")
        assert isinstance(err, JanusValidationError)

    def test_probability_distribution_error(self):
        err = JanusProbabilityDistributionError(1.5)
        assert isinstance(err, JanusValidationError)
        assert "1.5" in str(err)

    def test_scenario_not_found_error(self):
        err = JanusScenarioNotFoundError("sc-123")
        assert isinstance(err, JanusScenarioError)
        assert err.scenario_id == "sc-123"

    def test_scenario_generation_error(self):
        err = JanusScenarioGenerationError("no data", context={"k": "v"})
        assert isinstance(err, JanusScenarioError)

    def test_scenario_status_transition_error(self):
        err = JanusScenarioStatusTransitionError("sc-1", "PENDING", "ARCHIVED")
        assert isinstance(err, JanusScenarioError)

    def test_forecast_not_found_error(self):
        err = JanusForecastNotFoundError("fc-abc")
        assert err.forecast_id == "fc-abc"
        assert isinstance(err, JanusForecastError)

    def test_forecast_horizon_error(self):
        err = JanusForecastHorizonError("bad_horizon")
        assert isinstance(err, JanusForecastError)

    def test_forecast_superseded_error(self):
        err = JanusForecastSupersededError("fc-1", "fc-2")
        assert isinstance(err, JanusForecastError)

    def test_future_model_not_found_error(self):
        err = JanusFutureModelNotFoundError("m-1")
        assert err.model_id == "m-1"
        assert isinstance(err, JanusFutureModelError)

    def test_future_model_construction_error(self):
        err = JanusFutureModelConstructionError("bad state")
        assert isinstance(err, JanusFutureModelError)

    def test_simulation_not_found_error(self):
        err = JanusSimulationNotFoundError("sim-1")
        assert isinstance(err, JanusSimulationError)

    def test_branch_not_found_error(self):
        err = JanusBranchNotFoundError("br-1")
        assert err.branch_id == "br-1"
        assert isinstance(err, JanusBranchAnalysisError)

    def test_branch_construction_error(self):
        err = JanusBranchConstructionError("bad branch")
        assert isinstance(err, JanusBranchAnalysisError)

    def test_branch_probability_conflict_error(self):
        err = JanusBranchProbabilityConflictError("sc-1", 1.8)
        assert isinstance(err, JanusBranchAnalysisError)

    def test_counterfactual_not_found_error(self):
        err = JanusCounterfactualNotFoundError("cf-1")
        assert err.counterfactual_id == "cf-1"
        assert isinstance(err, JanusCounterfactualError)

    def test_counterfactual_divergence_error(self):
        err = JanusCounterfactualDivergenceError("2030-01-01", "future")
        assert isinstance(err, JanusCounterfactualError)

    def test_counterfactual_condition_error(self):
        err = JanusCounterfactualConditionError("none", "null placeholder")
        assert isinstance(err, JanusCounterfactualError)

    def test_risk_assessment_not_found_error(self):
        err = JanusRiskAssessmentNotFoundError("r-1")
        assert isinstance(err, JanusRiskError)

    def test_opportunity_assessment_not_found_error(self):
        err = JanusOpportunityAssessmentNotFoundError("o-1")
        assert isinstance(err, JanusOpportunityError)

    def test_timeline_projection_not_found_error(self):
        err = JanusTimelineProjectionNotFoundError("tp-1")
        assert isinstance(err, JanusTimelineProjectionError)

    def test_uncertainty_profile_not_found_error(self):
        err = JanusUncertaintyProfileNotFoundError("u-1")
        assert isinstance(err, JanusUncertaintyError)

    def test_missing_uncertainty_error(self):
        err = JanusMissingUncertaintyError("artifact-1", "Forecast")
        assert isinstance(err, JanusUncertaintyError)

    def test_impossible_future_error(self):
        err = JanusImpossibleFutureError("art-1", "Scenario", "prob > 1")
        assert isinstance(err, JanusIntegrityError)

    def test_false_certainty_error(self):
        err = JanusFalseCertaintyError("art-1", "Forecast")
        assert isinstance(err, JanusIntegrityError)

    def test_evidence_profile_error(self):
        err = JanusEvidenceProfileError("art-1", "no sources")
        assert isinstance(err, JanusIntegrityError)

    def test_constitutional_violation_error(self):
        err = JanusConstitutionalViolationError(
            engine="BranchAnalysisEngine",
            operation="select_branch",
            rightful_owner="VEGA",
        )
        assert isinstance(err, JanusIntegrityError)
        assert "VEGA" in str(err)

    def test_strategic_forecast_not_found(self):
        err = JanusStrategicForecastNotFoundError("sf-1")
        assert isinstance(err, JanusStrategicForecastError)

    def test_assessment_not_found_error(self):
        err = JanusAssessmentNotFoundError("a-1")
        assert err.assessment_id == "a-1"
        assert isinstance(err, JanusOrchestratorError)

    def test_assessment_status_transition_error(self):
        err = JanusAssessmentStatusTransitionError("a-1", "COMPLETE", "PENDING")
        assert isinstance(err, JanusOrchestratorError)

    def test_orchestrator_coordination_error(self):
        err = JanusOrchestratorCoordinationError("engines failed", failed_engine="engine-1")
        assert isinstance(err, JanusOrchestratorError)


# ---------------------------------------------------------------------------
# ============================================================
# MODELS
# ============================================================
# ---------------------------------------------------------------------------


class TestEnums:
    def test_scenario_status_members(self):
        assert ScenarioStatus.PENDING in ScenarioStatus
        assert ScenarioStatus.ACTIVE in ScenarioStatus
        assert ScenarioStatus.ARCHIVED in ScenarioStatus
        assert ScenarioStatus.INVALIDATED in ScenarioStatus

    def test_scenario_type_members(self):
        assert ScenarioType.BASELINE in ScenarioType
        assert ScenarioType.OPTIMISTIC in ScenarioType
        assert ScenarioType.COUNTERFACTUAL in ScenarioType

    def test_forecast_type_members(self):
        assert ForecastType.PROBABILISTIC in ForecastType
        assert ForecastType.STRATEGIC in ForecastType
        assert ForecastType.RISK in ForecastType

    def test_forecast_horizon_values(self):
        assert ForecastHorizon.ONE_YEAR.value == "1_year"
        assert ForecastHorizon.ONE_MONTH.value == "1_month"
        assert ForecastHorizon.TEN_YEARS.value == "10_years"
        assert "Year" in ForecastHorizon.ONE_YEAR.label

    def test_probability_level_from_float(self):
        assert ProbabilityLevel.from_float(0.90) == ProbabilityLevel.HIGHLY_LIKELY
        assert ProbabilityLevel.from_float(0.70) == ProbabilityLevel.LIKELY
        assert ProbabilityLevel.from_float(0.50) == ProbabilityLevel.POSSIBLE
        assert ProbabilityLevel.from_float(0.30) == ProbabilityLevel.UNLIKELY
        assert ProbabilityLevel.from_float(0.10) == ProbabilityLevel.HIGHLY_UNCERTAIN

    def test_probability_level_from_float_boundaries(self):
        assert ProbabilityLevel.from_float(0.0) == ProbabilityLevel.HIGHLY_UNCERTAIN
        assert ProbabilityLevel.from_float(1.0) == ProbabilityLevel.HIGHLY_LIKELY
        assert ProbabilityLevel.from_float(0.80) == ProbabilityLevel.LIKELY
        assert ProbabilityLevel.from_float(0.81) == ProbabilityLevel.HIGHLY_LIKELY

    def test_probability_level_invalid_raises(self):
        with pytest.raises(ValueError):
            ProbabilityLevel.from_float(1.5)
        with pytest.raises(ValueError):
            ProbabilityLevel.from_float(-0.1)

    def test_uncertainty_level_members(self):
        assert UncertaintyLevel.NEGLIGIBLE in UncertaintyLevel
        assert UncertaintyLevel.EXTREME in UncertaintyLevel

    def test_risk_level_members(self):
        assert RiskLevel.NEGLIGIBLE in RiskLevel
        assert RiskLevel.CRITICAL in RiskLevel

    def test_opportunity_level_members(self):
        assert OpportunityLevel.MARGINAL in OpportunityLevel
        assert OpportunityLevel.TRANSFORMATIVE in OpportunityLevel

    def test_simulation_status_members(self):
        assert SimulationStatus.PENDING in SimulationStatus
        assert SimulationStatus.COMPLETED in SimulationStatus
        assert SimulationStatus.ABORTED in SimulationStatus

    def test_projection_status_members(self):
        assert ProjectionStatus.DRAFT in ProjectionStatus
        assert ProjectionStatus.SUPERSEDED in ProjectionStatus

    def test_future_assessment_status_members(self):
        assert FutureAssessmentStatus.PENDING in FutureAssessmentStatus
        assert FutureAssessmentStatus.COMPLETE in FutureAssessmentStatus
        assert FutureAssessmentStatus.INVALIDATED in FutureAssessmentStatus


class TestConfidenceProfile:
    def test_create_valid(self):
        cp = make_confidence(0.75)
        assert cp.overall == 0.75
        assert isinstance(cp.level, ProbabilityLevel)

    def test_level_property(self):
        cp = make_confidence(0.85)
        assert cp.level == ProbabilityLevel.HIGHLY_LIKELY

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ConfidenceProfile.create(overall=1.1, data_quality=0.5, model_fit=0.5, signal_strength=0.5)

    def test_zero_values_valid(self):
        cp = ConfidenceProfile.create(overall=0.0, data_quality=0.0, model_fit=0.0, signal_strength=0.0)
        assert cp.overall == 0.0


class TestEvidenceProfile:
    def test_create_valid(self):
        ev = make_evidence(0.75)
        assert ev.evidence_strength == 0.75
        assert len(ev.sources) > 0
        assert ev.evidence_id != ""

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError):
            EvidenceProfile.create(
                sources=["s"], patterns_observed=[], contradicting_evidence=[], evidence_strength=1.5
            )


class TestUncertaintyProfile:
    def test_create_valid(self):
        up = make_uncertainty()
        assert up.level == UncertaintyLevel.MODERATE
        assert up.uncertainty_id != ""
        assert up.unknown_risk_exposure == 0.30

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError):
            UncertaintyProfile.create(
                level=UncertaintyLevel.HIGH,
                known_risks=[],
                unknown_risk_exposure=1.5,
                volatility_score=0.5,
                external_factors=[],
                market_sensitivity=0.5,
                technology_sensitivity=0.5,
            )


class TestProbabilityDistribution:
    def test_create_valid(self):
        dist = make_prob_dist()
        assert abs(sum(dist.outcomes.values()) - 1.0) < 1e-6
        assert dist.distribution_id != ""

    def test_mode_property(self):
        dist = make_prob_dist({"a": 0.70, "b": 0.20, "c": 0.10})
        assert dist.mode == "a"

    def test_entropy_property(self):
        dist = make_prob_dist()
        assert dist.entropy > 0.0

    def test_invalid_sum_raises(self):
        with pytest.raises(ValueError):
            ProbabilityDistribution.create(
                label="bad",
                outcomes={"a": 0.5, "b": 0.6},
                uncertainty_level=UncertaintyLevel.MODERATE,
                confidence=make_confidence(),
            )

    def test_invalid_probability_in_outcomes_raises(self):
        with pytest.raises(ValueError):
            ProbabilityDistribution.create(
                label="bad",
                outcomes={"a": -0.1, "b": 1.1},
                uncertainty_level=UncertaintyLevel.MODERATE,
                confidence=make_confidence(),
            )


class TestRiskFactor:
    def test_create_valid(self):
        rf = make_risk_factor(0.6, 0.7)
        assert rf.composite_score == pytest.approx(0.6 * 0.7)
        assert rf.factor_id != ""

    def test_invalid_impact_raises(self):
        with pytest.raises(ValueError):
            RiskFactor.create(
                name="r", description="d", category="c",
                impact_score=1.5, likelihood_score=0.5,
                time_horizon=ForecastHorizon.ONE_YEAR,
            )

    def test_invalid_likelihood_raises(self):
        with pytest.raises(ValueError):
            RiskFactor.create(
                name="r", description="d", category="c",
                impact_score=0.5, likelihood_score=-0.1,
                time_horizon=ForecastHorizon.ONE_YEAR,
            )


class TestOpportunityFactor:
    def test_create_valid(self):
        of = make_opportunity_factor()
        assert of.composite_score == pytest.approx(0.6 * 0.7)
        assert of.factor_id != ""

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            OpportunityFactor.create(
                name="o", description="d", category="c",
                value_score=1.2, feasibility_score=0.5,
                time_horizon=ForecastHorizon.ONE_YEAR,
            )


class TestProjectionMilestone:
    def test_create_valid(self):
        m = make_projection_milestone()
        assert m.milestone_id != ""
        assert m.probability == 0.80

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            ProjectionMilestone.create(
                label="m", description="d",
                projected_at=datetime.utcnow(),
                probability=1.5,
            )


class TestFutureState:
    def test_create_valid(self):
        fs = make_future_state()
        assert fs.state_id != ""
        assert fs.probability_level == ProbabilityLevel.LIKELY

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            FutureState.create(
                label="l", description="d",
                horizon=ForecastHorizon.ONE_YEAR,
                attributes={}, probability=1.5,
                uncertainty=make_uncertainty(),
            )


class TestFutureModel:
    def test_create_valid(self):
        m = make_future_model()
        assert m.model_id != ""
        assert len(m.future_states) == 1

    def test_horizons_property(self):
        m = make_future_model()
        assert ForecastHorizon.ONE_YEAR in m.horizons

    def test_state_at(self):
        m = make_future_model()
        states = m.state_at(ForecastHorizon.ONE_YEAR)
        assert len(states) == 1
        states_other = m.state_at(ForecastHorizon.TEN_YEARS)
        assert len(states_other) == 0


class TestScenarioBranch:
    def test_create_valid(self):
        b = make_scenario_branch()
        assert b.branch_id != ""
        assert b.probability == 0.70

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            ScenarioBranch.create(
                label="l", description="d", triggering_choice="c",
                future_state=make_future_state(), probability=1.5,
                risk_assessment=make_risk_assessment(),
                opportunity_assessment=make_opportunity_assessment(),
                confidence=make_confidence(),
            )


class TestScenario:
    def test_create_valid(self):
        s = make_scenario()
        assert s.scenario_id != ""
        assert s.status == ScenarioStatus.PENDING

    def test_dominant_branch(self):
        s = make_scenario()
        db = s.dominant_branch
        assert db is not None
        assert db.probability == 0.70

    def test_dominant_branch_empty(self):
        s = make_scenario()
        s.branches = []
        assert s.dominant_branch is None

    def test_invalid_risk_score_raises(self):
        with pytest.raises(ValueError):
            Scenario.create(
                title="t", description="d", scenario_type=ScenarioType.BASELINE,
                branches=[], future_model=make_future_model(),
                uncertainty=make_uncertainty(), confidence=make_confidence(),
                evidence=make_evidence(),
                metadata=ScenarioMetadata.create(created_by="t", source_subsystem="J"),
                risk_score=1.5, opportunity_score=0.5,
            )


class TestForecast:
    def test_create_valid(self):
        f = make_forecast()
        assert f.forecast_id != ""
        assert f.status == "active"

    def test_create_default_status(self):
        f = make_forecast()
        assert f.status == "active"


class TestCounterfactualScenario:
    def test_create_valid(self):
        fm = make_future_model()
        cf = CounterfactualScenario.create(
            title="CF Test",
            description="desc",
            original_event="event-A",
            counterfactual_condition="if X then Y",
            divergence_point=datetime.utcnow() - timedelta(days=30),
            resulting_future_model=fm,
            delta_risk=-0.10,
            delta_opportunity=0.20,
            uncertainty=make_uncertainty(),
            confidence=make_confidence(),
            evidence=make_evidence(),
            learning_insights=["insight-1"],
        )
        assert cf.counterfactual_id != ""
        assert cf.delta_risk == -0.10

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            CounterfactualScenario.create(
                title="CF", description="d", original_event="e",
                counterfactual_condition="c",
                divergence_point=datetime.utcnow() - timedelta(days=1),
                resulting_future_model=make_future_model(),
                delta_risk=2.0, delta_opportunity=0.0,
                uncertainty=make_uncertainty(), confidence=make_confidence(),
                evidence=make_evidence(),
            )


class TestTimelineProjection:
    def test_create_valid(self):
        m = make_projection_milestone(is_critical=True)
        tp = TimelineProjection.create(
            title="TP Test", description="d", context="ctx",
            milestones=[m], horizon=ForecastHorizon.ONE_YEAR,
            uncertainty=make_uncertainty(), confidence=make_confidence(),
            evidence=make_evidence(),
        )
        assert tp.projection_id != ""
        assert tp.status == ProjectionStatus.DRAFT
        assert len(tp.critical_milestones) == 1

    def test_completion_probability(self):
        m1 = make_projection_milestone(label="M1", probability=0.80, is_critical=True)
        m2 = make_projection_milestone(label="M2", probability=0.90, is_critical=True, offset_days=60)
        tp = TimelineProjection.create(
            title="TP", description="d", context="c",
            milestones=[m1, m2], horizon=ForecastHorizon.ONE_YEAR,
            uncertainty=make_uncertainty(), confidence=make_confidence(),
            evidence=make_evidence(),
        )
        assert tp.completion_probability == 0.80

    def test_completion_probability_no_critical(self):
        m = make_projection_milestone(is_critical=False)
        tp = TimelineProjection.create(
            title="TP", description="d", context="c",
            milestones=[m], horizon=ForecastHorizon.ONE_YEAR,
            uncertainty=make_uncertainty(), confidence=make_confidence(),
            evidence=make_evidence(),
        )
        assert tp.completion_probability == 1.0


class TestFutureAssessment:
    def test_create_valid(self):
        fa = FutureAssessment.create(
            title="FA Test", description="desc", context="ctx",
            scenarios=[make_scenario()], forecasts=[make_forecast()],
            simulations=[], risk_assessments=[make_risk_assessment()],
            opportunity_assessments=[make_opportunity_assessment()],
            timeline_projections=[], counterfactuals=[],
            strategic_forecasts=[],
            overall_uncertainty=make_uncertainty(),
            overall_confidence=make_confidence(), summary="summary",
        )
        assert fa.assessment_id != ""
        assert fa.status == FutureAssessmentStatus.PENDING
        assert fa.scenario_count == 1

    def test_has_critical_risks(self):
        ra = make_risk_assessment()
        ra.level = RiskLevel.CRITICAL
        fa = FutureAssessment.create(
            title="FA", description="d", context="c",
            scenarios=[], forecasts=[], simulations=[],
            risk_assessments=[ra], opportunity_assessments=[],
            timeline_projections=[], counterfactuals=[], strategic_forecasts=[],
            overall_uncertainty=make_uncertainty(), overall_confidence=make_confidence(),
            summary="",
        )
        assert fa.has_critical_risks is True

    def test_has_transformative_opportunities(self):
        oa = make_opportunity_assessment()
        oa.level = OpportunityLevel.TRANSFORMATIVE
        fa = FutureAssessment.create(
            title="FA", description="d", context="c",
            scenarios=[], forecasts=[], simulations=[],
            risk_assessments=[], opportunity_assessments=[oa],
            timeline_projections=[], counterfactuals=[], strategic_forecasts=[],
            overall_uncertainty=make_uncertainty(), overall_confidence=make_confidence(),
            summary="",
        )
        assert fa.has_transformative_opportunities is True


# ---------------------------------------------------------------------------
# ============================================================
# SCHEMAS
# ============================================================
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_scenario_generate_request_valid(self):
        meta = ScenarioMetadata.create(created_by="test", source_subsystem="JANUS")
        req = ScenarioGenerateRequest(
            decision_context="launch product",
            scenario_types=(ScenarioType.BASELINE, ScenarioType.OPTIMISTIC),
            horizon=ForecastHorizon.ONE_YEAR,
            requested_by="VEGA",
            metadata=meta,
        )
        assert req.decision_context == "launch product"

    def test_scenario_generate_request_empty_context_raises(self):
        meta = ScenarioMetadata.create(created_by="t", source_subsystem="J")
        with pytest.raises(ValueError):
            ScenarioGenerateRequest(
                decision_context="   ",
                scenario_types=(ScenarioType.BASELINE,),
                horizon=ForecastHorizon.ONE_YEAR,
                requested_by="VEGA",
                metadata=meta,
            )

    def test_scenario_generate_request_no_types_raises(self):
        meta = ScenarioMetadata.create(created_by="t", source_subsystem="J")
        with pytest.raises(ValueError):
            ScenarioGenerateRequest(
                decision_context="ctx",
                scenario_types=(),
                horizon=ForecastHorizon.ONE_YEAR,
                requested_by="VEGA",
                metadata=meta,
            )

    def test_forecast_generate_request_valid(self):
        req = ForecastGenerateRequest(
            title="Test Forecast",
            description="desc",
            forecast_type=ForecastType.PROBABILISTIC,
            horizon=ForecastHorizon.ONE_YEAR,
            uncertainty=make_uncertainty(),
            evidence=make_evidence(),
            metadata=ForecastMetadata.create(
                model_version="1.0", data_sources=["s"],
                horizon=ForecastHorizon.ONE_YEAR, generated_by="test",
            ),
        )
        assert req.title == "Test Forecast"

    def test_forecast_generate_request_blank_title_raises(self):
        with pytest.raises(ValueError):
            ForecastGenerateRequest(
                title="",
                description="desc",
                forecast_type=ForecastType.PROBABILISTIC,
                horizon=ForecastHorizon.ONE_YEAR,
                uncertainty=make_uncertainty(),
                evidence=make_evidence(),
                metadata=ForecastMetadata.create(
                    model_version="1.0", data_sources=["s"],
                    horizon=ForecastHorizon.ONE_YEAR, generated_by="test",
                ),
            )

    def test_risk_analysis_request_valid(self):
        req = RiskAnalysisRequest(
            title="Risk Test",
            description="desc",
            risk_factors=(make_risk_factor(),),
            uncertainty=make_uncertainty(),
            horizon=ForecastHorizon.ONE_YEAR,
            evidence=make_evidence(),
            confidence=make_confidence(),
        )
        assert req.title == "Risk Test"

    def test_risk_analysis_request_no_factors_raises(self):
        with pytest.raises(ValueError):
            RiskAnalysisRequest(
                title="Risk",
                description="desc",
                risk_factors=(),
                uncertainty=make_uncertainty(),
                horizon=ForecastHorizon.ONE_YEAR,
                evidence=make_evidence(),
                confidence=make_confidence(),
            )

    def test_opportunity_analysis_request_valid(self):
        req = OpportunityAnalysisRequest(
            title="Opp Test",
            description="desc",
            opportunity_factors=(make_opportunity_factor(),),
            uncertainty=make_uncertainty(),
            horizon=ForecastHorizon.ONE_YEAR,
            evidence=make_evidence(),
            confidence=make_confidence(),
        )
        assert req.title == "Opp Test"

    def test_opportunity_analysis_request_no_factors_raises(self):
        with pytest.raises(ValueError):
            OpportunityAnalysisRequest(
                title="Opp",
                description="desc",
                opportunity_factors=(),
                uncertainty=make_uncertainty(),
                horizon=ForecastHorizon.ONE_YEAR,
                evidence=make_evidence(),
                confidence=make_confidence(),
            )

    def test_timeline_projection_create_request_valid(self):
        req = TimelineProjectionCreateRequest(
            title="TL Test", description="d", context="ctx",
            milestones=(make_projection_milestone(),),
            horizon=ForecastHorizon.ONE_YEAR,
            uncertainty=make_uncertainty(),
            confidence=make_confidence(),
            evidence=make_evidence(),
        )
        assert req.title == "TL Test"

    def test_timeline_projection_create_request_no_milestones_raises(self):
        with pytest.raises(ValueError):
            TimelineProjectionCreateRequest(
                title="TL", description="d", context="c",
                milestones=(),
                horizon=ForecastHorizon.ONE_YEAR,
                uncertainty=make_uncertainty(),
                confidence=make_confidence(),
                evidence=make_evidence(),
            )

    def test_probability_distribution_create_request_valid(self):
        req = ProbabilityDistributionCreateRequest(
            label="test-dist",
            outcomes={"a": 0.60, "b": 0.40},
            uncertainty_level=UncertaintyLevel.MODERATE,
            confidence=make_confidence(),
        )
        assert req.label == "test-dist"

    def test_probability_distribution_create_request_bad_sum_raises(self):
        with pytest.raises(ValueError):
            ProbabilityDistributionCreateRequest(
                label="bad",
                outcomes={"a": 0.60, "b": 0.60},
                uncertainty_level=UncertaintyLevel.MODERATE,
                confidence=make_confidence(),
            )

    def test_uncertainty_profile_create_request_valid(self):
        req = UncertaintyProfileCreateRequest(
            level=UncertaintyLevel.HIGH,
            known_risks=("risk-A",),
            unknown_risk_exposure=0.40,
            volatility_score=0.35,
            external_factors=("factor-A",),
            market_sensitivity=0.25,
            technology_sensitivity=0.20,
        )
        assert req.level == UncertaintyLevel.HIGH

    def test_uncertainty_profile_create_request_invalid_score_raises(self):
        with pytest.raises(ValueError):
            UncertaintyProfileCreateRequest(
                level=UncertaintyLevel.HIGH,
                known_risks=(),
                unknown_risk_exposure=1.5,
                volatility_score=0.5,
                external_factors=(),
                market_sensitivity=0.5,
                technology_sensitivity=0.5,
            )

    def test_scenario_rank_request_weights_not_one_raises(self):
        with pytest.raises(ValueError):
            ScenarioRankRequest(
                scenario_ids=("id-1", "id-2"),
                weight_risk=0.6,
                weight_opportunity=0.6,
            )

    def test_scenario_compare_request_one_id_raises(self):
        with pytest.raises(ValueError):
            ScenarioCompareRequest(
                scenario_ids=("only-one",),
                title="compare",
                confidence=make_confidence(),
            )

    def test_forecast_assess_request_invalid_score_raises(self):
        with pytest.raises(ValueError):
            ForecastAssessRequest(
                forecast_id="fc-1",
                assessor="test",
                accuracy_score=1.5,
                deviation_notes="",
                revision_required=False,
                superseded_by=None,
            )

    def test_orchestrator_health_request_constructable(self):
        req = OrchestratorHealthRequest()
        assert req is not None


# ---------------------------------------------------------------------------
# ============================================================
# BRANCH ANALYSIS ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestBranchAnalysisEngineLifecycle:
    def test_initial_state(self):
        e = BranchAnalysisEngine()
        assert not e.is_initialized
        assert e.engine_name == "BranchAnalysisEngine"
        assert "5.1" in e.engine_version

    def test_initialize(self):
        e = BranchAnalysisEngine()
        e.initialize()
        assert e.is_initialized

    def test_double_initialize_raises(self):
        e = BranchAnalysisEngine()
        e.initialize()
        with pytest.raises(JanusAlreadyInitializedError):
            e.initialize()

    def test_shutdown(self):
        e = BranchAnalysisEngine()
        e.initialize()
        e.shutdown()
        assert not e.is_initialized

    def test_shutdown_without_init_raises(self):
        e = BranchAnalysisEngine()
        with pytest.raises(JanusNotInitializedError):
            e.shutdown()

    def test_shutdown_idempotent(self):
        e = BranchAnalysisEngine()
        e.initialize()
        e.shutdown()
        e.shutdown()  # second call should be no-op

    def test_operation_before_init_raises(self):
        e = BranchAnalysisEngine()
        with pytest.raises(JanusNotInitializedError):
            e.get_branch("br-1")

    def test_operation_after_shutdown_raises(self):
        e = BranchAnalysisEngine()
        e.initialize()
        e.shutdown()
        with pytest.raises(JanusShutdownError):
            e.get_branch("br-1")

    def test_health_report_before_init(self):
        e = BranchAnalysisEngine()
        h = e.get_health_report()
        assert not h.is_initialized
        assert h.engine_name == "BranchAnalysisEngine"

    def test_health_report_after_init(self):
        e = BranchAnalysisEngine()
        e.initialize()
        h = e.get_health_report()
        assert h.is_initialized
        assert not h.is_shut_down
        e.shutdown()


class TestBranchAnalysisEngineConfig:
    def test_default_config(self):
        config = BranchAnalysisConfig()
        assert config.max_branches_per_scenario == 32
        assert config.max_branch_depth == 8
        assert config.min_probability_threshold == 0.05

    def test_custom_config(self):
        config = BranchAnalysisConfig(max_branches_per_scenario=5, max_branch_depth=3)
        assert config.max_branches_per_scenario == 5
        assert config.max_branch_depth == 3

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            BranchAnalysisConfig(max_branches_per_scenario=0)
        with pytest.raises(ValueError):
            BranchAnalysisConfig(max_branch_depth=0)
        with pytest.raises(ValueError):
            BranchAnalysisConfig(min_probability_threshold=-0.1)
        with pytest.raises(ValueError):
            BranchAnalysisConfig(min_probability_threshold=1.1)


class TestBranchAnalysisEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = BranchAnalysisEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def _make_branch(self, e, label="B1", scenario_id="sc-1", probability=0.60, parent=None):
        return e.construct_branch(
            label=label,
            description="test branch",
            triggering_choice="choice-A",
            future_state=make_future_state(),
            probability=probability,
            risk_assessment=make_risk_assessment(),
            opportunity_assessment=make_opportunity_assessment(),
            confidence=make_confidence(0.70),
            scenario_id=scenario_id,
            parent_branch_id=parent,
        )

    def test_construct_branch_happy_path(self, engine):
        b = self._make_branch(engine)
        assert b.branch_id != ""
        assert b.label == "B1"
        assert b.probability == 0.60

    def test_get_branch_happy_path(self, engine):
        b = self._make_branch(engine)
        retrieved = engine.get_branch(b.branch_id)
        assert retrieved.branch_id == b.branch_id

    def test_get_branch_not_found_raises(self, engine):
        with pytest.raises(JanusBranchNotFoundError):
            engine.get_branch("nonexistent-id")

    def test_get_branch_blank_id_raises(self, engine):
        with pytest.raises(JanusValidationError):
            engine.get_branch("  ")

    def test_analyze_branches_returns_ordered_by_probability(self, engine):
        b1 = self._make_branch(engine, label="B1", probability=0.30, scenario_id="sc-x")
        b2 = self._make_branch(engine, label="B2", probability=0.60, scenario_id="sc-x")
        branches = engine.analyze_branches("sc-x")
        assert branches[0].probability >= branches[1].probability

    def test_analyze_branches_empty_scenario(self, engine):
        branches = engine.analyze_branches("nonexistent-scenario")
        assert branches == ()

    def test_analyze_branches_blank_scenario_raises(self, engine):
        with pytest.raises(JanusValidationError):
            engine.analyze_branches("")

    def test_dominant_branch(self, engine):
        self._make_branch(engine, label="B-low", probability=0.20, scenario_id="sc-dom")
        self._make_branch(engine, label="B-high", probability=0.70, scenario_id="sc-dom")
        dom = engine.dominant_branch("sc-dom")
        assert dom.label == "B-high"

    def test_dominant_branch_no_branches_raises(self, engine):
        with pytest.raises(JanusBranchNotFoundError):
            engine.dominant_branch("empty-scenario")

    def test_construct_branch_below_probability_threshold_raises(self, engine):
        config = BranchAnalysisConfig(min_probability_threshold=0.10)
        e = BranchAnalysisEngine(config=config)
        e.initialize()
        with pytest.raises(JanusBranchConstructionError):
            e.construct_branch(
                label="B", description="d", triggering_choice="c",
                future_state=make_future_state(), probability=0.04,
                risk_assessment=make_risk_assessment(),
                opportunity_assessment=make_opportunity_assessment(),
                confidence=make_confidence(0.70),
                scenario_id="sc-1",
            )
        e.shutdown()

    def test_construct_branch_below_confidence_threshold_raises(self, engine):
        config = BranchAnalysisConfig(min_confidence_threshold=0.50)
        e = BranchAnalysisEngine(config=config)
        e.initialize()
        with pytest.raises(JanusBranchConstructionError):
            e.construct_branch(
                label="B", description="d", triggering_choice="c",
                future_state=make_future_state(), probability=0.60,
                risk_assessment=make_risk_assessment(),
                opportunity_assessment=make_opportunity_assessment(),
                confidence=make_confidence(0.30),
                scenario_id="sc-1",
            )
        e.shutdown()

    def test_construct_branch_validates_blank_label(self, engine):
        with pytest.raises(JanusValidationError):
            engine.construct_branch(
                label="", description="d", triggering_choice="c",
                future_state=make_future_state(), probability=0.60,
                risk_assessment=make_risk_assessment(),
                opportunity_assessment=make_opportunity_assessment(),
                confidence=make_confidence(),
                scenario_id="sc-1",
            )

    def test_construct_branch_validates_blank_description(self, engine):
        with pytest.raises(JanusValidationError):
            engine.construct_branch(
                label="B", description="", triggering_choice="c",
                future_state=make_future_state(), probability=0.60,
                risk_assessment=make_risk_assessment(),
                opportunity_assessment=make_opportunity_assessment(),
                confidence=make_confidence(),
                scenario_id="sc-1",
            )

    def test_construct_branch_validates_out_of_range_probability(self, engine):
        with pytest.raises(JanusValidationError):
            engine.construct_branch(
                label="B", description="d", triggering_choice="c",
                future_state=make_future_state(), probability=1.5,
                risk_assessment=make_risk_assessment(),
                opportunity_assessment=make_opportunity_assessment(),
                confidence=make_confidence(),
                scenario_id="sc-1",
            )

    def test_branch_lineage_root(self, engine):
        b = self._make_branch(engine, scenario_id="sc-lin")
        lineage = engine.get_branch_lineage(b.branch_id)
        assert lineage.depth == 0
        assert lineage.ancestors == ()

    def test_branch_lineage_child(self, engine):
        parent = self._make_branch(engine, label="Parent", scenario_id="sc-child")
        child = self._make_branch(engine, label="Child", scenario_id="sc-child",
                                   probability=0.30, parent=parent.branch_id)
        lineage = engine.get_branch_lineage(child.branch_id)
        assert lineage.depth == 1
        assert parent.branch_id in lineage.ancestors

    def test_branch_ancestry(self, engine):
        root = self._make_branch(engine, label="Root", scenario_id="sc-anc")
        child = self._make_branch(engine, label="Child", scenario_id="sc-anc",
                                   probability=0.30, parent=root.branch_id)
        ancestry = engine.get_branch_ancestry(child.branch_id)
        assert len(ancestry) == 1
        assert ancestry[0].branch_id == root.branch_id

    def test_get_branch_children(self, engine):
        parent = self._make_branch(engine, label="Parent", scenario_id="sc-ch")
        child1 = self._make_branch(engine, label="C1", scenario_id="sc-ch",
                                    probability=0.30, parent=parent.branch_id)
        child2 = self._make_branch(engine, label="C2", scenario_id="sc-ch",
                                    probability=0.20, parent=parent.branch_id)
        children = engine.get_branch_children(parent.branch_id)
        assert len(children) == 2

    def test_get_branch_depth(self, engine):
        b = self._make_branch(engine, scenario_id="sc-dep")
        assert engine.get_branch_depth(b.branch_id) == 0

    def test_rank_branches_by_probability(self, engine):
        self._make_branch(engine, label="B1", probability=0.20, scenario_id="sc-rank")
        self._make_branch(engine, label="B2", probability=0.60, scenario_id="sc-rank")
        ranked = engine.rank_branches("sc-rank", by="probability")
        assert ranked[0].probability >= ranked[-1].probability

    def test_rank_branches_invalid_criterion_raises(self, engine):
        with pytest.raises(JanusValidationError):
            engine.rank_branches("sc-1", by="invalid_criterion")

    def test_validate_branch_probabilities_valid(self, engine):
        self._make_branch(engine, label="B1", probability=0.30, scenario_id="sc-vp")
        self._make_branch(engine, label="B2", probability=0.40, scenario_id="sc-vp")
        assert engine.validate_branch_probabilities("sc-vp") is True

    def test_validate_branch_probabilities_invalid_raises(self, engine):
        e2 = BranchAnalysisEngine(BranchAnalysisConfig(min_probability_threshold=0.05))
        e2.initialize()
        e2.construct_branch(
            label="B1", description="d", triggering_choice="c",
            future_state=make_future_state(), probability=0.70,
            risk_assessment=make_risk_assessment(),
            opportunity_assessment=make_opportunity_assessment(),
            confidence=make_confidence(), scenario_id="sc-invalid",
        )
        e2.construct_branch(
            label="B2", description="d", triggering_choice="c",
            future_state=make_future_state(), probability=0.70,
            risk_assessment=make_risk_assessment(),
            opportunity_assessment=make_opportunity_assessment(),
            confidence=make_confidence(), scenario_id="sc-invalid",
        )
        with pytest.raises(JanusBranchProbabilityConflictError):
            e2.validate_branch_probabilities("sc-invalid")
        e2.shutdown()

    def test_compare_branches(self, engine):
        b1 = self._make_branch(engine, label="B1", probability=0.60, scenario_id="sc-cmp")
        b2 = self._make_branch(engine, label="B2", probability=0.30, scenario_id="sc-cmp")
        result = engine.compare_branches((b1.branch_id, b2.branch_id))
        assert b1.branch_id in result
        assert result[b1.branch_id]["rank_by_probability"] == 1

    def test_compare_branches_empty_raises(self, engine):
        with pytest.raises(JanusValidationError):
            engine.compare_branches(())

    def test_compare_branches_not_found_raises(self, engine):
        with pytest.raises(JanusBranchNotFoundError):
            engine.compare_branches(("nonexistent",))

    def test_prune_branch(self, engine):
        b = self._make_branch(engine, scenario_id="sc-prune")
        engine.prune_branch(b.branch_id, "test pruning")
        assert engine.is_branch_pruned(b.branch_id)

    def test_prune_branch_not_found_raises(self, engine):
        with pytest.raises(JanusBranchNotFoundError):
            engine.prune_branch("nonexistent", "reason")

    def test_prune_branches_by_probability(self, engine):
        self._make_branch(engine, label="High", probability=0.60, scenario_id="sc-p2")
        self._make_branch(engine, label="Low", probability=0.12, scenario_id="sc-p2")
        pruned_count = engine.prune_branches_by_probability("sc-p2", threshold=0.15)
        assert pruned_count >= 1

    def test_prune_branches_by_confidence(self, engine):
        engine.construct_branch(
            label="LowConf", description="d", triggering_choice="c",
            future_state=make_future_state(), probability=0.60,
            risk_assessment=make_risk_assessment(),
            opportunity_assessment=make_opportunity_assessment(),
            confidence=make_confidence(0.20), scenario_id="sc-pc",
        )
        pruned = engine.prune_branches_by_confidence("sc-pc", threshold=0.30)
        assert pruned >= 1

    def test_prune_all_branches_for_scenario(self, engine):
        self._make_branch(engine, label="B1", scenario_id="sc-pa")
        self._make_branch(engine, label="B2", probability=0.30, scenario_id="sc-pa")
        pruned = engine.prune_all_branches_for_scenario("sc-pa", "scenario archived")
        assert pruned == 2
        assert engine.count_active_branches("sc-pa") == 0

    def test_branch_exists(self, engine):
        b = self._make_branch(engine, scenario_id="sc-ex")
        assert engine.branch_exists(b.branch_id)
        assert not engine.branch_exists("nonexistent")

    def test_get_scenario_for_branch(self, engine):
        b = self._make_branch(engine, scenario_id="sc-gsf")
        assert engine.get_scenario_for_branch(b.branch_id) == "sc-gsf"

    def test_list_scenario_ids(self, engine):
        self._make_branch(engine, scenario_id="sc-list-A")
        self._make_branch(engine, scenario_id="sc-list-B")
        ids = engine.list_scenario_ids()
        assert "sc-list-A" in ids
        assert "sc-list-B" in ids

    def test_count_active_branches(self, engine):
        self._make_branch(engine, label="B1", scenario_id="sc-cnt")
        self._make_branch(engine, label="B2", probability=0.30, scenario_id="sc-cnt")
        assert engine.count_active_branches("sc-cnt") == 2

    def test_assert_no_selection_raises_on_select(self, engine):
        with pytest.raises(JanusConstitutionalViolationError):
            engine.assert_no_selection("select_branch")

    def test_assert_no_selection_passes_on_analyze(self, engine):
        engine.assert_no_selection("analyze_branch")  # should not raise

    def test_get_statistics(self, engine):
        self._make_branch(engine, scenario_id="sc-stats")
        stats = engine.get_statistics()
        assert stats.total_branches >= 1
        assert stats.active_branches >= 1
        assert isinstance(stats.generated_at, datetime)

    def test_get_diagnostics_report(self, engine):
        self._make_branch(engine, scenario_id="sc-diag")
        diag = engine.get_diagnostics_report()
        assert diag.engine_name == "BranchAnalysisEngine"
        assert diag.total_records >= 1

    def test_per_scenario_limit_raises(self):
        config = BranchAnalysisConfig(max_branches_per_scenario=2)
        e = BranchAnalysisEngine(config=config)
        e.initialize()
        for i in range(2):
            e.construct_branch(
                label=f"B{i}", description="d", triggering_choice="c",
                future_state=make_future_state(), probability=0.30,
                risk_assessment=make_risk_assessment(),
                opportunity_assessment=make_opportunity_assessment(),
                confidence=make_confidence(), scenario_id="sc-lim",
            )
        with pytest.raises(JanusBranchAnalysisError):
            e.construct_branch(
                label="B3", description="d", triggering_choice="c",
                future_state=make_future_state(), probability=0.30,
                risk_assessment=make_risk_assessment(),
                opportunity_assessment=make_opportunity_assessment(),
                confidence=make_confidence(), scenario_id="sc-lim",
            )
        e.shutdown()

    def test_max_depth_limit_raises(self):
        config = BranchAnalysisConfig(max_branch_depth=1)
        e = BranchAnalysisEngine(config=config)
        e.initialize()
        root = e.construct_branch(
            label="Root", description="d", triggering_choice="c",
            future_state=make_future_state(), probability=0.60,
            risk_assessment=make_risk_assessment(),
            opportunity_assessment=make_opportunity_assessment(),
            confidence=make_confidence(), scenario_id="sc-dep",
        )
        child = e.construct_branch(
            label="Child", description="d", triggering_choice="c",
            future_state=make_future_state(), probability=0.30,
            risk_assessment=make_risk_assessment(),
            opportunity_assessment=make_opportunity_assessment(),
            confidence=make_confidence(), scenario_id="sc-dep",
            parent_branch_id=root.branch_id,
        )
        with pytest.raises(JanusBranchAnalysisError):
            e.construct_branch(
                label="GrandChild", description="d", triggering_choice="c",
                future_state=make_future_state(), probability=0.15,
                risk_assessment=make_risk_assessment(),
                opportunity_assessment=make_opportunity_assessment(),
                confidence=make_confidence(), scenario_id="sc-dep",
                parent_branch_id=child.branch_id,
            )
        e.shutdown()

    def test_thread_safety(self, engine):
        errors = []

        def add_branch(i):
            try:
                engine.construct_branch(
                    label=f"B{i}", description="d", triggering_choice="c",
                    future_state=make_future_state(), probability=0.20,
                    risk_assessment=make_risk_assessment(),
                    opportunity_assessment=make_opportunity_assessment(),
                    confidence=make_confidence(), scenario_id="sc-thread",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_branch, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# ============================================================
# COUNTERFACTUAL ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestCounterfactualEngineLifecycle:
    def test_initial_state(self):
        e = CounterfactualEngine()
        assert not e.is_initialized
        assert e.engine_name == "CounterfactualEngine"

    def test_initialize_and_shutdown(self):
        e = CounterfactualEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized

    def test_double_init_raises(self):
        e = CounterfactualEngine()
        e.initialize()
        with pytest.raises(JanusAlreadyInitializedError):
            e.initialize()
        e.shutdown()

    def test_operation_before_init_raises(self):
        e = CounterfactualEngine()
        with pytest.raises(JanusNotInitializedError):
            e.list_all_counterfactuals()

    def test_operation_after_shutdown_raises(self):
        e = CounterfactualEngine()
        e.initialize()
        e.shutdown()
        with pytest.raises(JanusShutdownError):
            e.list_all_counterfactuals()

    def test_health_report_before_init(self):
        e = CounterfactualEngine()
        h = e.get_health_report()
        assert not h.is_initialized
        assert h.engine_name == "CounterfactualEngine"


def _make_cf_request(
    event="event-A",
    condition="if X had been Y",
    delta_risk=-0.10,
    delta_opportunity=0.20,
) -> CounterfactualCreateRequest:
    return CounterfactualCreateRequest(
        title=f"CF: {event}",
        description="counterfactual desc",
        original_event=event,
        counterfactual_condition=condition,
        divergence_point=datetime.utcnow() - timedelta(days=30),
        resulting_future_model=make_future_model(),
        delta_risk=delta_risk,
        delta_opportunity=delta_opportunity,
        uncertainty=make_uncertainty(),
        confidence=make_confidence(),
        evidence=make_evidence(),
        learning_insights=("insight-1", "insight-2"),
        requested_by="JANUS-TEST",
    )


class TestCounterfactualEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = CounterfactualEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def test_create_counterfactual_happy_path(self, engine):
        req = _make_cf_request()
        resp = engine.create_counterfactual(req)
        assert resp.counterfactual.counterfactual_id != ""
        assert resp.engine_name == "CounterfactualEngine"

    def test_get_counterfactual_happy_path(self, engine):
        req = _make_cf_request()
        resp = engine.create_counterfactual(req)
        cf_id = resp.counterfactual.counterfactual_id
        get_resp = engine.get_counterfactual(CounterfactualGetRequest(counterfactual_id=cf_id))
        assert get_resp.counterfactual.counterfactual_id == cf_id

    def test_get_counterfactual_not_found_raises(self, engine):
        with pytest.raises(JanusCounterfactualNotFoundError):
            engine.get_counterfactual(CounterfactualGetRequest(counterfactual_id="nonexistent"))

    def test_get_counterfactual_blank_id_raises(self, engine):
        with pytest.raises(JanusValidationError):
            engine.get_counterfactual(CounterfactualGetRequest(counterfactual_id=""))

    def test_create_with_future_divergence_point_raises(self, engine):
        req = CounterfactualCreateRequest(
            title="Future CF",
            description="desc",
            original_event="event-A",
            counterfactual_condition="if X",
            divergence_point=datetime.utcnow() + timedelta(days=10),
            resulting_future_model=make_future_model(),
            delta_risk=0.0,
            delta_opportunity=0.0,
            uncertainty=make_uncertainty(),
            confidence=make_confidence(),
            evidence=make_evidence(),
            learning_insights=(),
            requested_by="test",
        )
        with pytest.raises(JanusCounterfactualDivergenceError):
            engine.create_counterfactual(req)

    def test_create_with_null_condition_raises(self, engine):
        with pytest.raises(JanusCounterfactualConditionError):
            engine.create_counterfactual(_make_cf_request(condition="none"))

    def test_create_with_blank_event_raises(self, engine):
        with pytest.raises(JanusValidationError):
            engine.create_counterfactual(_make_cf_request(event="  "))

    def test_compare_counterfactual(self, engine):
        req = _make_cf_request()
        resp = engine.create_counterfactual(req)
        cf_id = resp.counterfactual.counterfactual_id
        compare_req = CounterfactualCompareRequest(
            counterfactual_id=cf_id,
            reference_event_label="event-A-baseline",
            include_learning_synthesis=True,
        )
        cmp_resp = engine.compare_counterfactual(compare_req)
        assert cmp_resp.counterfactual_id == cf_id
        assert cmp_resp.metrics.delta_risk == pytest.approx(-0.10)

    def test_list_counterfactuals_for_event(self, engine):
        engine.create_counterfactual(_make_cf_request(event="shared-event"))
        engine.create_counterfactual(_make_cf_request(event="shared-event", condition="if Y"))
        results = engine.list_counterfactuals_for_event("shared-event")
        assert len(results) == 2

    def test_list_counterfactuals_blank_event_raises(self, engine):
        with pytest.raises(JanusValidationError):
            engine.list_counterfactuals_for_event("")

    def test_extract_learning_insights(self, engine):
        req = _make_cf_request()
        resp = engine.create_counterfactual(req)
        insights = engine.extract_learning_insights(resp.counterfactual.counterfactual_id)
        assert "insight-1" in insights

    def test_append_learning_insight(self, engine):
        req = _make_cf_request()
        resp = engine.create_counterfactual(req)
        cf_id = resp.counterfactual.counterfactual_id
        updated = engine.append_learning_insight(cf_id, "new insight")
        assert "new insight" in updated.learning_insights

    def test_append_blank_insight_raises(self, engine):
        req = _make_cf_request()
        resp = engine.create_counterfactual(req)
        with pytest.raises(JanusValidationError):
            engine.append_learning_insight(resp.counterfactual.counterfactual_id, "")

    def test_counterfactual_exists(self, engine):
        req = _make_cf_request()
        resp = engine.create_counterfactual(req)
        assert engine.counterfactual_exists(resp.counterfactual.counterfactual_id)
        assert not engine.counterfactual_exists("nonexistent")

    def test_list_tracked_events(self, engine):
        engine.create_counterfactual(_make_cf_request(event="evt-X"))
        events = engine.list_tracked_events()
        assert "evt-X" in events

    def test_evaluate_counterfactual_net_positive(self, engine):
        req = _make_cf_request(delta_risk=-0.30, delta_opportunity=0.50)
        resp = engine.create_counterfactual(req)
        eval_result = engine.evaluate_counterfactual(resp.counterfactual.counterfactual_id)
        assert eval_result["overall_verdict"] == "net_positive"

    def test_evaluate_counterfactual_net_negative(self, engine):
        req = _make_cf_request(delta_risk=0.50, delta_opportunity=-0.30)
        resp = engine.create_counterfactual(req)
        eval_result = engine.evaluate_counterfactual(resp.counterfactual.counterfactual_id)
        assert eval_result["overall_verdict"] == "net_negative"

    def test_compare_counterfactuals_for_event(self, engine):
        engine.create_counterfactual(_make_cf_request(event="cmp-evt", condition="if X"))
        engine.create_counterfactual(_make_cf_request(event="cmp-evt", condition="if Y", delta_risk=-0.20, delta_opportunity=0.30))
        evals = engine.compare_counterfactuals_for_event("cmp-evt")
        assert len(evals) == 2
        assert evals[0]["rank"] == 1

    def test_count_counterfactuals_for_event(self, engine):
        engine.create_counterfactual(_make_cf_request(event="cnt-evt"))
        assert engine.count_counterfactuals_for_event("cnt-evt") == 1

    def test_generate_alternative_decision_paths(self, engine):
        conditions = ("if A", "if B")
        futures = (make_future_model(), make_future_model())
        div_point = datetime.utcnow() - timedelta(days=10)
        cfs = engine.generate_alternative_decision_paths(
            original_event="bulk-event",
            alternative_conditions=conditions,
            divergence_point=div_point,
            resulting_future_models=futures,
            delta_risks=(-0.10, 0.20),
            delta_opportunities=(0.30, -0.10),
            uncertainties=(make_uncertainty(), make_uncertainty()),
            confidences=(make_confidence(), make_confidence()),
            evidences=(make_evidence(), make_evidence()),
            learning_insights_per_path=(("ins-1",), ("ins-2",)),
            requested_by="test",
        )
        assert len(cfs) == 2

    def test_get_statistics(self, engine):
        engine.create_counterfactual(_make_cf_request())
        stats = engine.get_statistics()
        assert stats.total_counterfactuals == 1
        assert isinstance(stats.generated_at, datetime)

    def test_get_diagnostics_report(self, engine):
        engine.create_counterfactual(_make_cf_request())
        diag = engine.get_diagnostics_report()
        assert diag.engine_name == "CounterfactualEngine"
        assert diag.total_counterfactuals == 1

    def test_assert_no_history_mutation_raises(self, engine):
        with pytest.raises(JanusConstitutionalViolationError):
            engine.assert_no_history_mutation("modify_history")

    def test_assert_no_history_mutation_passes_for_read(self, engine):
        engine.assert_no_history_mutation("read_history")  # should not raise

    def test_assert_no_selection_raises(self, engine):
        with pytest.raises(JanusConstitutionalViolationError):
            engine.assert_no_selection("select_counterfactual")


# ---------------------------------------------------------------------------
# ============================================================
# FORECASTING ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestForecastingEngineLifecycle:
    def test_initial_state(self):
        e = ForecastingEngine()
        assert not e.is_initialized
        assert e.engine_name == "ForecastingEngine"

    def test_initialize_and_shutdown(self):
        e = ForecastingEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized

    def test_double_init_raises(self):
        e = ForecastingEngine()
        e.initialize()
        with pytest.raises(JanusAlreadyInitializedError):
            e.initialize()
        e.shutdown()

    def test_not_initialized_raises(self):
        e = ForecastingEngine()
        req = ForecastGetRequest(forecast_id="fc-1")
        with pytest.raises(JanusNotInitializedError):
            e.get_forecast(req)

    def test_health_report(self):
        e = ForecastingEngine()
        h = e.health()
        assert h["engine"] == "ForecastingEngine"
        assert not h["healthy"]
        e.initialize()
        h2 = e.health()
        assert h2["healthy"]
        e.shutdown()


def _make_forecast_req() -> ForecastGenerateRequest:
    return ForecastGenerateRequest(
        title="Test Forecast",
        description="forecast description",
        forecast_type=ForecastType.PROBABILISTIC,
        horizon=ForecastHorizon.ONE_YEAR,
        uncertainty=make_uncertainty(),
        evidence=make_evidence(0.70),
        metadata=ForecastMetadata.create(
            model_version="5.1.0",
            data_sources=["source-A"],
            horizon=ForecastHorizon.ONE_YEAR,
            generated_by="test",
        ),
    )


class TestForecastingEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = ForecastingEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def test_generate_forecast_happy_path(self, engine):
        req = _make_forecast_req()
        resp = engine.generate_forecast(req)
        assert resp.forecast.forecast_id != ""
        assert resp.forecast.status == "active"

    def test_get_forecast_happy_path(self, engine):
        req = _make_forecast_req()
        resp = engine.generate_forecast(req)
        get_resp = engine.get_forecast(ForecastGetRequest(forecast_id=resp.forecast.forecast_id))
        assert get_resp.forecast.forecast_id == resp.forecast.forecast_id

    def test_get_forecast_not_found_raises(self, engine):
        with pytest.raises(JanusForecastNotFoundError):
            engine.get_forecast(ForecastGetRequest(forecast_id="nonexistent"))

    def test_generate_blank_title_raises(self, engine):
        req = ForecastGenerateRequest(
            title="Valid", description="d", forecast_type=ForecastType.PROBABILISTIC,
            horizon=ForecastHorizon.ONE_YEAR, uncertainty=make_uncertainty(),
            evidence=make_evidence(), metadata=ForecastMetadata.create(
                model_version="1.0", data_sources=["s"],
                horizon=ForecastHorizon.ONE_YEAR, generated_by="test",
            ),
        )
        # Inject blank title after construction via patching the attribute
        # (frozen dataclass - just test missing field via engine validate)
        with pytest.raises((JanusMissingRequiredFieldError, JanusValidationError)):
            engine.generate_forecast(
                ForecastGenerateRequest(
                    title="  ",
                    description="d",
                    forecast_type=ForecastType.PROBABILISTIC,
                    horizon=ForecastHorizon.ONE_YEAR,
                    uncertainty=make_uncertainty(),
                    evidence=make_evidence(),
                    metadata=ForecastMetadata.create(
                        model_version="1.0", data_sources=["s"],
                        horizon=ForecastHorizon.ONE_YEAR, generated_by="test",
                    ),
                )
            )

    def test_generate_missing_uncertainty_raises(self, engine):
        with pytest.raises((JanusMissingUncertaintyError, JanusMissingRequiredFieldError)):
            engine.generate_forecast(
                ForecastGenerateRequest(
                    title="T", description="d",
                    forecast_type=ForecastType.PROBABILISTIC,
                    horizon=ForecastHorizon.ONE_YEAR,
                    uncertainty=None,
                    evidence=make_evidence(),
                    metadata=ForecastMetadata.create(
                        model_version="1.0", data_sources=["s"],
                        horizon=ForecastHorizon.ONE_YEAR, generated_by="test",
                    ),
                )
            )

    def test_assess_forecast(self, engine):
        req = _make_forecast_req()
        resp = engine.generate_forecast(req)
        assess_req = ForecastAssessRequest(
            forecast_id=resp.forecast.forecast_id,
            assessor="test-assessor",
            accuracy_score=0.85,
            deviation_notes="slight deviation",
            revision_required=False,
            superseded_by=None,
        )
        assess_resp = engine.assess_forecast(assess_req)
        assert assess_resp.assessment.forecast.forecast_id == resp.forecast.forecast_id

    def test_supersede_forecast(self, engine):
        fc1 = engine.generate_forecast(_make_forecast_req())
        fc2 = engine.generate_forecast(_make_forecast_req())
        superseded = engine.supersede_forecast(
            fc1.forecast.forecast_id,
            fc2.forecast.forecast_id,
            "newer forecast available",
        )
        assert superseded.status == "superseded"

    def test_supersede_nonexistent_raises(self, engine):
        fc = engine.generate_forecast(_make_forecast_req())
        with pytest.raises(JanusForecastNotFoundError):
            engine.supersede_forecast("nonexistent", fc.forecast.forecast_id, "reason")

    def test_list_forecasts_by_horizon(self, engine):
        engine.generate_forecast(_make_forecast_req())
        req = ForecastListByHorizonRequest(horizon=ForecastHorizon.ONE_YEAR)
        resp = engine.list_forecasts_by_horizon(req)
        assert len(resp.forecasts) >= 1
        assert resp.horizon == ForecastHorizon.ONE_YEAR

    def test_list_forecasts_by_type(self, engine):
        engine.generate_forecast(_make_forecast_req())
        results = engine.list_forecasts_by_type(ForecastType.PROBABILISTIC)
        assert len(results) >= 1

    def test_archive_forecast(self, engine):
        fc = engine.generate_forecast(_make_forecast_req())
        archived = engine.archive_forecast(
            fc.forecast.forecast_id, reason="no longer relevant", archived_by="test"
        )
        assert archived.status == "archived"

    def test_forecast_exists(self, engine):
        fc = engine.generate_forecast(_make_forecast_req())
        assert engine.forecast_exists(fc.forecast.forecast_id)
        assert not engine.forecast_exists("nonexistent")

    def test_average_accuracy_no_data(self, engine):
        fc = engine.generate_forecast(_make_forecast_req())
        avg = engine.average_accuracy(fc.forecast.forecast_id)
        assert avg is None

    def test_average_accuracy_with_scores(self, engine):
        req = _make_forecast_req()
        fc = engine.generate_forecast(req)
        engine.assess_forecast(ForecastAssessRequest(
            forecast_id=fc.forecast.forecast_id, assessor="a",
            accuracy_score=0.80, deviation_notes="", revision_required=False, superseded_by=None,
        ))
        engine.assess_forecast(ForecastAssessRequest(
            forecast_id=fc.forecast.forecast_id, assessor="b",
            accuracy_score=0.90, deviation_notes="", revision_required=False, superseded_by=None,
        ))
        avg = engine.average_accuracy(fc.forecast.forecast_id)
        assert avg == pytest.approx(0.85)

    def test_statistics(self, engine):
        engine.generate_forecast(_make_forecast_req())
        stats = engine.statistics()
        assert stats["engine"] == "ForecastingEngine"
        assert stats["total_registered"] >= 1

    def test_register_forecast(self, engine):
        fc = make_forecast()
        engine.register_forecast(fc)
        assert engine.forecast_exists(fc.forecast_id)

    def test_list_forecasts(self, engine):
        engine.generate_forecast(_make_forecast_req())
        forecasts = engine.list_forecasts()
        assert len(forecasts) >= 1


# ---------------------------------------------------------------------------
# ============================================================
# FUTURE MODELING ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestFutureModelingEngineLifecycle:
    def test_initial_state(self):
        e = FutureModelingEngine()
        assert not e.is_initialized
        assert e.engine_name == "FutureModelingEngine"

    def test_initialize_and_shutdown(self):
        e = FutureModelingEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized

    def test_double_init_raises(self):
        e = FutureModelingEngine()
        e.initialize()
        with pytest.raises(JanusAlreadyInitializedError):
            e.initialize()
        e.shutdown()

    def test_health(self):
        e = FutureModelingEngine()
        h = e.health()
        assert not h["healthy"]
        e.initialize()
        h2 = e.health()
        assert h2["healthy"]
        e.shutdown()


class TestFutureModelingEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = FutureModelingEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def _make_construct_req(self, n_states: int = 1) -> FutureModelConstructRequest:
        states = tuple(
            make_future_state(label=f"state-{i}", probability=0.30)
            for i in range(n_states)
        )
        return FutureModelConstructRequest(
            title="Test Model", description="desc", context="ctx",
            future_states=states,
            uncertainty=make_uncertainty(),
            confidence=make_confidence(),
            evidence=make_evidence(),
        )

    def test_construct_future_model_happy_path(self, engine):
        req = self._make_construct_req()
        resp = engine.construct_future_model(req)
        assert resp.future_model.model_id != ""

    def test_get_future_model_happy_path(self, engine):
        resp = engine.construct_future_model(self._make_construct_req())
        mid = resp.future_model.model_id
        get_resp = engine.get_future_model(FutureModelGetRequest(model_id=mid))
        assert get_resp.future_model.model_id == mid

    def test_get_future_model_not_found_raises(self, engine):
        with pytest.raises(JanusFutureModelNotFoundError):
            engine.get_future_model(FutureModelGetRequest(model_id="nonexistent"))

    def test_construct_no_states_raises(self, engine):
        with pytest.raises((JanusValidationError, ValueError)):
            engine.construct_future_model(
                FutureModelConstructRequest(
                    title="T", description="d", context="c",
                    future_states=(),
                    uncertainty=make_uncertainty(),
                    confidence=make_confidence(),
                    evidence=make_evidence(),
                )
            )

    def test_query_states_at_horizon(self, engine):
        state = make_future_state(horizon=ForecastHorizon.ONE_YEAR)
        resp = engine.construct_future_model(
            FutureModelConstructRequest(
                title="T", description="d", context="c",
                future_states=(state,),
                uncertainty=make_uncertainty(),
                confidence=make_confidence(),
                evidence=make_evidence(),
            )
        )
        query = FutureStateQueryRequest(
            model_id=resp.future_model.model_id,
            horizon=ForecastHorizon.ONE_YEAR,
        )
        q_resp = engine.query_states_at_horizon(query)
        assert len(q_resp.states) == 1

    def test_list_future_models(self, engine):
        engine.construct_future_model(self._make_construct_req())
        models = engine.list_future_models()
        assert len(models) >= 1

    def test_model_exists(self, engine):
        resp = engine.construct_future_model(self._make_construct_req())
        assert engine.model_exists(resp.future_model.model_id)
        assert not engine.model_exists("nonexistent")

    def test_create_future_state(self, engine):
        state = engine.create_future_state(
            label="standalone-state",
            description="desc",
            horizon=ForecastHorizon.ONE_YEAR,
            attributes={"key": "value"},
            probability=0.60,
            uncertainty=make_uncertainty(),
        )
        assert state.state_id != ""

    def test_update_future_model(self, engine):
        resp = engine.construct_future_model(self._make_construct_req())
        model_id = resp.future_model.model_id
        new_state = make_future_state(label="updated-state", probability=0.50)
        updated = engine.update_future_model(
            model_id=model_id,
            updated_states=(new_state,),
            updated_uncertainty=make_uncertainty(UncertaintyLevel.HIGH),
            updated_evidence=make_evidence(0.75),
            reason="test update",
        )
        assert len(updated.future_states) == 1

    def test_horizon_coverage(self, engine):
        resp = engine.construct_future_model(self._make_construct_req())
        coverage = engine.horizon_coverage(resp.future_model.model_id)
        assert ForecastHorizon.ONE_YEAR.value in coverage

    def test_weighted_probability_summary(self, engine):
        resp = engine.construct_future_model(self._make_construct_req())
        summary = engine.weighted_probability_summary(resp.future_model.model_id)
        assert ForecastHorizon.ONE_YEAR.value in summary

    def test_statistics(self, engine):
        engine.construct_future_model(self._make_construct_req())
        stats = engine.statistics()
        assert stats["total_models"] >= 1

    def test_register_future_model_and_exists(self, engine):
        fm = make_future_model()
        engine.register_future_model(fm)
        assert engine.model_exists(fm.model_id)


# ---------------------------------------------------------------------------
# ============================================================
# INTEGRITY ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestScenarioIntegrityEngineLifecycle:
    def test_initial_state(self):
        e = ScenarioIntegrityEngine()
        assert not e.is_initialized
        assert e.engine_name == "ScenarioIntegrityEngine"

    def test_initialize_and_shutdown(self):
        e = ScenarioIntegrityEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized

    def test_double_init_raises(self):
        e = ScenarioIntegrityEngine()
        e.initialize()
        with pytest.raises(JanusAlreadyInitializedError):
            e.initialize()
        e.shutdown()

    def test_health_before_init(self):
        e = ScenarioIntegrityEngine()
        h = e.get_health()
        assert not h["is_healthy"]


class TestScenarioIntegrityEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = ScenarioIntegrityEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def test_validate_scenario_happy_path(self, engine):
        s = make_scenario()
        is_valid, violations = engine.validate_scenario(s)
        assert is_valid
        assert violations == ()

    def test_validate_forecast_happy_path(self, engine):
        f = make_forecast()
        is_valid, violations = engine.validate_forecast(f)
        assert is_valid
        assert violations == ()

    def test_validate_future_model_happy_path(self, engine):
        fm = make_future_model()
        is_valid, violations = engine.validate_future_model(fm)
        assert is_valid
        assert violations == ()

    def test_validate_future_model_no_states_fails(self, engine):
        fm = make_future_model()
        fm.future_states = []
        is_valid, violations = engine.validate_future_model(fm)
        assert not is_valid
        assert len(violations) > 0

    def test_validate_timeline_projection_happy_path(self, engine):
        m = make_projection_milestone(probability=0.80, is_critical=True)
        tp = TimelineProjection.create(
            title="TP", description="d", context="c",
            milestones=[m], horizon=ForecastHorizon.ONE_YEAR,
            uncertainty=make_uncertainty(), confidence=make_confidence(),
            evidence=make_evidence(),
        )
        is_valid, violations = engine.validate_timeline_projection(tp)
        assert is_valid

    def test_validate_timeline_projection_no_milestones_fails(self, engine):
        tp = TimelineProjection.create(
            title="TP", description="d", context="c",
            milestones=[], horizon=ForecastHorizon.ONE_YEAR,
            uncertainty=make_uncertainty(), confidence=make_confidence(),
            evidence=make_evidence(),
        )
        is_valid, violations = engine.validate_timeline_projection(tp)
        assert not is_valid

    def test_enforce_uncertainty_invariant_passes(self, engine):
        engine.enforce_uncertainty_invariant("art-1", "Forecast", make_uncertainty())

    def test_enforce_uncertainty_invariant_trivial_raises(self, engine):
        trivial = UncertaintyProfile.create(
            level=UncertaintyLevel.NEGLIGIBLE,
            known_risks=[], unknown_risk_exposure=0.0, volatility_score=0.0,
            external_factors=[], market_sensitivity=0.0, technology_sensitivity=0.0,
        )
        with pytest.raises(JanusMissingUncertaintyError):
            engine.enforce_uncertainty_invariant("art-1", "Forecast", trivial)

    def test_enforce_evidence_invariant_passes(self, engine):
        engine.enforce_evidence_invariant("art-1", "Scenario", make_evidence())

    def test_enforce_evidence_invariant_no_sources_raises(self, engine):
        ev_no_source = EvidenceProfile(
            evidence_id="ev-1",
            sources=(),
            patterns_observed=(),
            contradicting_evidence=(),
            evidence_strength=0.50,
            collected_at=datetime.utcnow(),
        )
        with pytest.raises(JanusEvidenceProfileError):
            engine.enforce_evidence_invariant("art-1", "Forecast", ev_no_source)

    def test_validate_scenario_invalidated_fails(self, engine):
        s = make_scenario()
        s.status = ScenarioStatus.INVALIDATED
        is_valid, violations = engine.validate_scenario(s)
        assert not is_valid

    def test_get_statistics(self, engine):
        engine.validate_forecast(make_forecast())
        stats = engine.get_statistics()
        assert stats["total_validations"] >= 1
        assert "forecasts" in stats

    def test_get_validation_history(self, engine):
        engine.validate_forecast(make_forecast())
        history = engine.get_validation_history()
        assert len(history) >= 1
        assert "artifact_id" in history[0]

    def test_get_diagnostics(self, engine):
        diag = engine.get_diagnostics()
        assert "health" in diag
        assert "statistics" in diag

    def test_check_constitutional_boundary_passes_for_janus(self, engine):
        # JANUS is allowed to perform create_future_assessment - no violation
        engine.check_constitutional_boundary(
            engine_name="FutureOrchestrator",
            operation="create_future_assessment",
            rightful_owner="FutureOrchestrator",
        )

    def test_check_constitutional_boundary_raises_for_violation(self, engine):
        with pytest.raises(JanusConstitutionalViolationError):
            engine.check_constitutional_boundary(
                engine_name="ScenarioIntegrityEngine",
                operation="decide_strategy",
                rightful_owner="VEGA",
            )


# ---------------------------------------------------------------------------
# ============================================================
# RISK ANALYSIS ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestRiskAnalysisEngineLifecycle:
    def test_initial_state(self):
        e = RiskAnalysisEngine()
        assert not e.is_initialized
        assert e.engine_name == "RiskAnalysisEngine"

    def test_initialize_and_shutdown(self):
        e = RiskAnalysisEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized

    def test_double_init_raises(self):
        e = RiskAnalysisEngine()
        e.initialize()
        with pytest.raises(JanusAlreadyInitializedError):
            e.initialize()
        e.shutdown()

    def test_operation_before_init_raises(self):
        e = RiskAnalysisEngine()
        with pytest.raises(JanusNotInitializedError):
            e.compute_composite_risk_score((make_risk_factor(),))


class TestRiskAnalysisEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = RiskAnalysisEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def _make_risk_req(self) -> RiskAnalysisRequest:
        return RiskAnalysisRequest(
            title="Risk Analysis",
            description="desc",
            risk_factors=(make_risk_factor(0.6, 0.7),),
            uncertainty=make_uncertainty(),
            horizon=ForecastHorizon.ONE_YEAR,
            evidence=make_evidence(),
            confidence=make_confidence(),
        )

    def test_analyze_risk_happy_path(self, engine):
        resp = engine.analyze_risk(self._make_risk_req())
        assert resp.risk_assessment.risk_id != ""
        assert isinstance(resp.computed_level, RiskLevel)

    def test_get_risk_assessment(self, engine):
        resp = engine.analyze_risk(self._make_risk_req())
        rid = resp.risk_assessment.risk_id
        get_resp = engine.get_risk_assessment(RiskAssessmentGetRequest(risk_id=rid))
        assert get_resp.risk_assessment.risk_id == rid

    def test_get_risk_assessment_not_found_raises(self, engine):
        with pytest.raises(JanusRiskAssessmentNotFoundError):
            engine.get_risk_assessment(RiskAssessmentGetRequest(risk_id="nonexistent"))

    def test_list_risks_by_level(self, engine):
        engine.analyze_risk(self._make_risk_req())
        resp = engine.list_risks_by_level(RiskListByLevelRequest(minimum_level=RiskLevel.LOW))
        assert len(resp.risk_assessments) >= 1

    def test_compute_composite_risk_score(self, engine):
        factors = (make_risk_factor(0.6, 0.8),)
        score = engine.compute_composite_risk_score(factors)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.6 * 0.8)

    def test_derive_risk_level_critical(self, engine):
        level = engine.derive_risk_level(0.95)
        assert level == RiskLevel.CRITICAL

    def test_derive_risk_level_low(self, engine):
        level = engine.derive_risk_level(0.10)
        assert level in (RiskLevel.NEGLIGIBLE, RiskLevel.LOW)

    def test_list_risk_factors_by_category(self, engine):
        engine.analyze_risk(self._make_risk_req())
        factors = engine.list_risk_factors_by_category("technical")
        assert len(factors) >= 1

    def test_health_report(self, engine):
        h = engine.get_health()
        assert h["is_healthy"]


# ---------------------------------------------------------------------------
# ============================================================
# OPPORTUNITY ANALYSIS ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestOpportunityAnalysisEngineLifecycle:
    def test_initial_state(self):
        e = OpportunityAnalysisEngine()
        assert not e.is_initialized

    def test_initialize_and_shutdown(self):
        e = OpportunityAnalysisEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized


class TestOpportunityAnalysisEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = OpportunityAnalysisEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def _make_opp_req(self) -> OpportunityAnalysisRequest:
        return OpportunityAnalysisRequest(
            title="Opportunity Analysis",
            description="desc",
            opportunity_factors=(make_opportunity_factor(),),
            uncertainty=make_uncertainty(),
            horizon=ForecastHorizon.ONE_YEAR,
            evidence=make_evidence(),
            confidence=make_confidence(),
        )

    def test_analyze_opportunity_happy_path(self, engine):
        resp = engine.analyze_opportunity(self._make_opp_req())
        assert resp.opportunity_assessment.opportunity_id != ""
        assert isinstance(resp.computed_level, OpportunityLevel)

    def test_get_opportunity_assessment(self, engine):
        resp = engine.analyze_opportunity(self._make_opp_req())
        oid = resp.opportunity_assessment.opportunity_id
        get_resp = engine.get_opportunity_assessment(
            OpportunityAssessmentGetRequest(opportunity_id=oid)
        )
        assert get_resp.opportunity_assessment.opportunity_id == oid

    def test_get_opportunity_assessment_not_found_raises(self, engine):
        with pytest.raises(JanusOpportunityAssessmentNotFoundError):
            engine.get_opportunity_assessment(
                OpportunityAssessmentGetRequest(opportunity_id="nonexistent")
            )

    def test_list_opportunities_by_level(self, engine):
        engine.analyze_opportunity(self._make_opp_req())
        resp = engine.list_opportunities_by_level(
            OpportunityListByLevelRequest(minimum_level=OpportunityLevel.LOW)
        )
        assert len(resp.opportunity_assessments) >= 1

    def test_compute_composite_opportunity_score(self, engine):
        factors = (make_opportunity_factor(0.7, 0.8),)
        score = engine.compute_composite_opportunity_score(factors)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.7 * 0.8)

    def test_derive_opportunity_level_high(self, engine):
        level = engine.derive_opportunity_level(0.85)
        assert level in (OpportunityLevel.HIGH, OpportunityLevel.TRANSFORMATIVE)

    def test_list_opportunity_factors_by_category(self, engine):
        engine.analyze_opportunity(self._make_opp_req())
        factors = engine.list_opportunity_factors_by_category("market")
        assert len(factors) >= 1


# ---------------------------------------------------------------------------
# ============================================================
# SIMULATION ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestSimulationEngineLifecycle:
    def test_initial_state(self):
        e = OutcomeSimulationEngine()
        assert not e.is_initialized
        assert e.engine_name == "OutcomeSimulationEngine"

    def test_initialize_and_shutdown(self):
        e = OutcomeSimulationEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized

    def test_double_init_raises(self):
        e = OutcomeSimulationEngine()
        e.initialize()
        with pytest.raises(JanusAlreadyInitializedError):
            e.initialize()
        e.shutdown()


class TestSimulationEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = OutcomeSimulationEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def _make_sim_req(self, title: str = "Sim Test") -> SimulationRunRequest:
        outcomes = (
            make_simulation_outcome("primary", 0.70),
            make_simulation_outcome("alternate", 0.20),
            make_simulation_outcome("pessimistic", 0.10),
        )
        return SimulationRunRequest(
            title=title,
            description="sim desc",
            triggering_event="market-shock",
            uncertainty=make_uncertainty(),
            metadata=make_simulation_metadata(),
            candidate_outcomes=outcomes,
        )

    def test_run_simulation_happy_path(self, engine):
        resp = engine.run_simulation(self._make_sim_req())
        assert resp.simulation.simulation_id != ""

    def test_get_simulation_result(self, engine):
        run_resp = engine.run_simulation(self._make_sim_req())
        sim_id = run_resp.simulation.simulation_id
        result_resp = engine.get_simulation_result(SimulationResultRequest(simulation_id=sim_id))
        assert result_resp.simulation.simulation_id == sim_id

    def test_get_simulation_result_not_found_raises(self, engine):
        with pytest.raises(JanusSimulationNotFoundError):
            engine.get_simulation_result(SimulationResultRequest(simulation_id="nonexistent"))

    def test_abort_simulation(self, engine):
        run_resp = engine.run_simulation(self._make_sim_req())
        sim_id = run_resp.simulation.simulation_id
        abort_resp = engine.abort_simulation(
            SimulationAbortRequest(simulation_id=sim_id, reason="cancelled", aborted_by="test")
        )
        assert abort_resp.simulation_id == sim_id

    def test_most_probable_outcome(self, engine):
        run_resp = engine.run_simulation(self._make_sim_req())
        sim_id = run_resp.simulation.simulation_id
        outcome = engine.most_probable_outcome(sim_id)
        assert outcome.label == "primary"

    def test_list_simulations_by_status(self, engine):
        engine.run_simulation(self._make_sim_req())
        sims = engine.list_simulations_by_status(SimulationStatus.PENDING)
        # May be COMPLETED immediately depending on implementation
        assert isinstance(sims, tuple)

    def test_validate_outcome_probabilities(self, engine):
        run_resp = engine.run_simulation(self._make_sim_req())
        result = engine.validate_outcome_probabilities(run_resp.simulation.simulation_id)
        assert isinstance(result, bool)

    def test_run_simulation_blank_event_raises(self, engine):
        with pytest.raises((ValueError, JanusValidationError)):
            engine.run_simulation(
                SimulationRunRequest(
                    title="T", description="d", triggering_event="  ",
                    uncertainty=make_uncertainty(), metadata=make_simulation_metadata(),
                    candidate_outcomes=(make_simulation_outcome(),),
                )
            )

    def test_run_simulation_no_outcomes_raises(self, engine):
        with pytest.raises((ValueError, JanusValidationError)):
            engine.run_simulation(
                SimulationRunRequest(
                    title="T", description="d", triggering_event="event",
                    uncertainty=make_uncertainty(), metadata=make_simulation_metadata(),
                    candidate_outcomes=(),
                )
            )


# ---------------------------------------------------------------------------
# ============================================================
# TIMELINE PROJECTION ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestTimelineProjectionEngineLifecycle:
    def test_initial_state(self):
        e = TimelineProjectionEngine()
        assert not e.is_initialized

    def test_initialize_and_shutdown(self):
        e = TimelineProjectionEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized

    def test_not_initialized_raises(self):
        e = TimelineProjectionEngine()
        with pytest.raises(JanusNotInitializedError):
            e.list_projections_by_status(ProjectionStatus.DRAFT)


class TestTimelineProjectionEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = TimelineProjectionEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def _make_tl_req(self) -> TimelineProjectionCreateRequest:
        return TimelineProjectionCreateRequest(
            title="Timeline Test",
            description="desc",
            context="context",
            milestones=(
                make_projection_milestone("M1", probability=0.80, is_critical=True, offset_days=30),
                make_projection_milestone("M2", probability=0.90, is_critical=False, offset_days=60),
            ),
            horizon=ForecastHorizon.ONE_YEAR,
            uncertainty=make_uncertainty(),
            confidence=make_confidence(),
            evidence=make_evidence(),
        )

    def test_create_projection_happy_path(self, engine):
        resp = engine.create_projection(self._make_tl_req())
        assert resp.projection.projection_id != ""
        assert resp.projection.status == ProjectionStatus.DRAFT

    def test_get_projection_happy_path(self, engine):
        resp = engine.create_projection(self._make_tl_req())
        pid = resp.projection.projection_id
        get_resp = engine.get_projection(TimelineProjectionGetRequest(projection_id=pid))
        assert get_resp.projection.projection_id == pid

    def test_get_projection_not_found_raises(self, engine):
        with pytest.raises(JanusTimelineProjectionNotFoundError):
            engine.get_projection(TimelineProjectionGetRequest(projection_id="nonexistent"))

    def test_update_projection(self, engine):
        resp = engine.create_projection(self._make_tl_req())
        pid = resp.projection.projection_id
        new_milestone = make_projection_milestone("M-updated", offset_days=45)
        update_req = TimelineProjectionUpdateRequest(
            projection_id=pid,
            updated_milestones=(new_milestone,),
            updated_uncertainty=make_uncertainty(UncertaintyLevel.HIGH),
            updated_evidence=make_evidence(0.75),
            revision_reason="new data available",
            updated_by="test",
        )
        update_resp = engine.update_projection(update_req)
        assert update_resp.new_status == ProjectionStatus.REVISED

    def test_query_completion_probability(self, engine):
        resp = engine.create_projection(self._make_tl_req())
        pid = resp.projection.projection_id
        result = engine.query_completion_probability(
            TimelineProjectionCompletionQuery(projection_id=pid)
        )
        assert 0.0 <= result.completion_probability <= 1.0
        assert isinstance(result.probability_level, ProbabilityLevel)

    def test_list_projections_by_status(self, engine):
        engine.create_projection(self._make_tl_req())
        projections = engine.list_projections_by_status(ProjectionStatus.DRAFT)
        assert len(projections) >= 1

    def test_expire_projection(self, engine):
        resp = engine.create_projection(self._make_tl_req())
        pid = resp.projection.projection_id
        expired = engine.expire_projection(pid, "horizon passed")
        assert expired.status == ProjectionStatus.EXPIRED

    def test_supersede_projection(self, engine):
        resp1 = engine.create_projection(self._make_tl_req())
        resp2 = engine.create_projection(self._make_tl_req())
        superseded = engine.supersede_projection(
            resp1.projection.projection_id,
            resp2.projection.projection_id,
            "replacement available",
        )
        assert superseded.status == ProjectionStatus.SUPERSEDED

    def test_statistics(self, engine):
        engine.create_projection(self._make_tl_req())
        stats = engine.get_statistics()
        assert stats["total_projections"] >= 1

    def test_health_report(self, engine):
        h = engine.get_health()
        assert h["is_healthy"]


# ---------------------------------------------------------------------------
# ============================================================
# STRATEGIC FORECAST ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestStrategicForecastEngineLifecycle:
    def test_initial_state(self):
        e = StrategicForecastEngine()
        assert not e.is_initialized

    def test_initialize_and_shutdown(self):
        e = StrategicForecastEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized


class TestStrategicForecastEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = StrategicForecastEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def _make_sf_req(self, domain: str = "technology") -> StrategicForecastCreateRequest:
        return StrategicForecastCreateRequest(
            title="Strategic Forecast Test",
            description="desc",
            domain=domain,
            trend_analysis=("trend-1", "trend-2"),
            market_state_projections=(make_future_state(),),
            strategic_outcome_forecasts=(make_forecast(),),
            uncertainty=make_uncertainty(),
            confidence=make_confidence(),
            evidence=make_evidence(),
            horizon=ForecastHorizon.FIVE_YEARS,
        )

    def test_create_strategic_forecast_happy_path(self, engine):
        resp = engine.create_strategic_forecast(self._make_sf_req())
        assert resp.strategic_forecast.strategic_forecast_id != ""
        assert resp.strategic_forecast.domain == "technology"

    def test_get_strategic_forecast_happy_path(self, engine):
        resp = engine.create_strategic_forecast(self._make_sf_req())
        sfid = resp.strategic_forecast.strategic_forecast_id
        get_resp = engine.get_strategic_forecast(
            StrategicForecastGetRequest(strategic_forecast_id=sfid)
        )
        assert get_resp.strategic_forecast.strategic_forecast_id == sfid

    def test_get_strategic_forecast_not_found_raises(self, engine):
        with pytest.raises(JanusStrategicForecastNotFoundError):
            engine.get_strategic_forecast(
                StrategicForecastGetRequest(strategic_forecast_id="nonexistent")
            )

    def test_list_strategic_forecasts(self, engine):
        engine.create_strategic_forecast(self._make_sf_req())
        resp = engine.list_strategic_forecasts(
            StrategicForecastListRequest(domain=None, horizon=None)
        )
        assert len(resp.strategic_forecasts) >= 1

    def test_list_strategic_forecasts_by_domain(self, engine):
        engine.create_strategic_forecast(self._make_sf_req(domain="finance"))
        results = engine.list_strategic_forecasts_by_domain("finance")
        assert len(results) >= 1

    def test_update_strategic_forecast(self, engine):
        resp = engine.create_strategic_forecast(self._make_sf_req())
        sfid = resp.strategic_forecast.strategic_forecast_id
        updated = engine.update_strategic_forecast(
            strategic_forecast_id=sfid,
            updated_trend_analysis=("new-trend",),
            updated_market_projections=(make_future_state(),),
            updated_outcome_forecasts=(make_forecast(),),
            updated_uncertainty=make_uncertainty(UncertaintyLevel.HIGH),
            updated_evidence=make_evidence(0.80),
            reason="updated with new market data",
        )
        assert updated.strategic_forecast_id == sfid

    def test_statistics(self, engine):
        engine.create_strategic_forecast(self._make_sf_req())
        stats = engine.get_statistics()
        assert stats["total_forecasts"] >= 1

    def test_health(self, engine):
        h = engine.get_health()
        assert h["is_healthy"]


# ---------------------------------------------------------------------------
# ============================================================
# SCENARIO ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestScenarioEngineLifecycle:
    def test_initial_state(self):
        e = ScenarioEngine()
        assert not e.is_initialized

    def test_initialize_and_shutdown(self):
        e = ScenarioEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized

    def test_not_initialized_raises(self):
        e = ScenarioEngine()
        with pytest.raises(JanusNotInitializedError):
            e.list_scenarios_by_status(ScenarioStatus.ACTIVE)


class TestScenarioEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = ScenarioEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def _make_gen_req(self) -> ScenarioGenerateRequest:
        meta = ScenarioMetadata.create(created_by="test", source_subsystem="JANUS")
        return ScenarioGenerateRequest(
            decision_context="Should we expand into new markets?",
            scenario_types=(ScenarioType.BASELINE, ScenarioType.OPTIMISTIC),
            horizon=ForecastHorizon.ONE_YEAR,
            requested_by="VEGA",
            metadata=meta,
        )

    def test_generate_scenarios_happy_path(self, engine):
        resp = engine.generate_scenarios(self._make_gen_req())
        assert resp.scenario_count >= 1
        for s in resp.scenarios:
            assert s.scenario_id != ""

    def test_get_scenario_happy_path(self, engine):
        resp = engine.generate_scenarios(self._make_gen_req())
        sid = resp.scenarios[0].scenario_id
        get_resp = engine.get_scenario(ScenarioGetRequest(scenario_id=sid))
        assert get_resp.scenario.scenario_id == sid

    def test_get_scenario_not_found_raises(self, engine):
        with pytest.raises(JanusScenarioNotFoundError):
            engine.get_scenario(ScenarioGetRequest(scenario_id="nonexistent"))

    def test_update_scenario_status(self, engine):
        resp = engine.generate_scenarios(self._make_gen_req())
        sid = resp.scenarios[0].scenario_id
        update_resp = engine.update_scenario_status(
            ScenarioUpdateStatusRequest(
                scenario_id=sid,
                target_status=ScenarioStatus.ACTIVE,
                reason="activating",
                updated_by="test",
            )
        )
        assert update_resp.new_status == ScenarioStatus.ACTIVE

    def test_archive_scenario(self, engine):
        resp = engine.generate_scenarios(self._make_gen_req())
        sid = resp.scenarios[0].scenario_id
        archive_resp = engine.archive_scenario(
            ScenarioArchiveRequest(scenario_id=sid, reason="no longer needed", archived_by="test")
        )
        assert archive_resp.scenario_id == sid

    def test_list_scenarios_by_type(self, engine):
        engine.generate_scenarios(self._make_gen_req())
        scenarios = engine.list_scenarios_by_type(ScenarioType.BASELINE)
        assert len(scenarios) >= 1

    def test_list_scenarios_by_status(self, engine):
        engine.generate_scenarios(self._make_gen_req())
        scenarios = engine.list_scenarios_by_status(ScenarioStatus.PENDING)
        assert len(scenarios) >= 1

    def test_statistics(self, engine):
        engine.generate_scenarios(self._make_gen_req())
        stats = engine.get_statistics()
        assert stats["total_scenarios"] >= 1


# ---------------------------------------------------------------------------
# ============================================================
# UNCERTAINTY ENGINE
# ============================================================
# ---------------------------------------------------------------------------


class TestUncertaintyEngineLifecycle:
    def test_initial_state(self):
        e = UncertaintyEngine()
        assert not e.is_initialized

    def test_initialize_and_shutdown(self):
        e = UncertaintyEngine()
        e.initialize()
        assert e.is_initialized
        e.shutdown()
        assert not e.is_initialized


class TestUncertaintyEngineOperations:
    @pytest.fixture(autouse=True)
    def engine(self):
        e = UncertaintyEngine()
        e.initialize()
        yield e
        if e.is_initialized:
            e.shutdown()

    def _make_create_req(self, level=UncertaintyLevel.MODERATE) -> UncertaintyProfileCreateRequest:
        return UncertaintyProfileCreateRequest(
            level=level,
            known_risks=("risk-A",),
            unknown_risk_exposure=0.30,
            volatility_score=0.25,
            external_factors=("factor-A",),
            market_sensitivity=0.20,
            technology_sensitivity=0.15,
        )

    def test_create_uncertainty_profile(self, engine):
        resp = engine.create_uncertainty_profile(self._make_create_req())
        assert resp.profile.uncertainty_id != ""

    def test_get_uncertainty_profile(self, engine):
        resp = engine.create_uncertainty_profile(self._make_create_req())
        uid = resp.profile.uncertainty_id
        get_resp = engine.get_uncertainty_profile(
            UncertaintyProfileGetRequest(uncertainty_id=uid)
        )
        assert get_resp.profile.uncertainty_id == uid

    def test_get_uncertainty_profile_not_found_raises(self, engine):
        with pytest.raises(JanusUncertaintyProfileNotFoundError):
            engine.get_uncertainty_profile(
                UncertaintyProfileGetRequest(uncertainty_id="nonexistent")
            )

    def test_validate_uncertainty(self, engine):
        up = make_uncertainty()
        resp = engine.validate_uncertainty(
            UncertaintyValidationRequest(
                artifact_id="art-1",
                artifact_type="Forecast",
                uncertainty=up,
            )
        )
        assert resp.is_valid

    def test_validate_uncertainty_trivial_fails(self, engine):
        trivial = UncertaintyProfile.create(
            level=UncertaintyLevel.NEGLIGIBLE,
            known_risks=[], unknown_risk_exposure=0.0,
            volatility_score=0.0, external_factors=[],
            market_sensitivity=0.0, technology_sensitivity=0.0,
        )
        resp = engine.validate_uncertainty(
            UncertaintyValidationRequest(
                artifact_id="art-1",
                artifact_type="Forecast",
                uncertainty=trivial,
            )
        )
        assert not resp.is_valid

    def test_aggregate_uncertainty(self, engine):
        profiles = (make_uncertainty(), make_uncertainty(UncertaintyLevel.HIGH))
        aggregated = engine.aggregate_uncertainty(profiles)
        assert aggregated.uncertainty_id != ""
        assert isinstance(aggregated.level, UncertaintyLevel)

    def test_classify_uncertainty_level(self, engine):
        up = make_uncertainty(UncertaintyLevel.EXTREME)
        level = engine.classify_uncertainty_level(up)
        assert isinstance(level, UncertaintyLevel)

    def test_statistics(self, engine):
        engine.create_uncertainty_profile(self._make_create_req())
        stats = engine.get_statistics()
        assert stats["total_profiles"] >= 1


# ---------------------------------------------------------------------------
# ============================================================
# ORCHESTRATOR
# ============================================================
# ---------------------------------------------------------------------------


def _make_mock_engine(name="MockEngine", version="5.1.0"):
    """Create a mock that satisfies the JanusEngineLifecycle interface."""
    e = MagicMock()
    e.engine_name = name
    e.engine_version = version
    e.is_initialized = True
    return e


def _make_full_orchestrator() -> FutureOrchestrator:
    """Build an orchestrator with all-mock dependencies."""
    return FutureOrchestrator(
        scenario_engine=_make_mock_engine("ScenarioEngine"),
        forecasting_engine=_make_mock_engine("ForecastingEngine"),
        future_modeling_engine=_make_mock_engine("FutureModelingEngine"),
        branch_analysis_engine=_make_mock_engine("BranchAnalysisEngine"),
        counterfactual_engine=_make_mock_engine("CounterfactualEngine"),
        uncertainty_engine=_make_mock_engine("UncertaintyEngine"),
        future_risk_engine=_make_mock_engine("FutureRiskEngine"),
        future_opportunity_engine=_make_mock_engine("FutureOpportunityEngine"),
        outcome_simulation_engine=_make_mock_engine("OutcomeSimulationEngine"),
        strategic_forecast_engine=_make_mock_engine("StrategicForecastEngine"),
        timeline_projection_engine=_make_mock_engine("TimelineProjectionEngine"),
        probability_engine=_make_mock_engine("ProbabilityEngine"),
        scenario_evaluation_engine=_make_mock_engine("ScenarioEvaluationEngine"),
        scenario_integrity_engine=_make_mock_engine("ScenarioIntegrityEngine"),
        future_memory_interface=_make_mock_engine("FutureMemoryInterface"),
    )


class TestFutureOrchestratorLifecycle:
    def test_initial_state(self):
        orch = _make_full_orchestrator()
        assert not orch.is_initialized
        assert orch.engine_name == "FutureOrchestrator"
        assert "5.1" in orch.engine_version

    def test_initialize_and_shutdown(self):
        orch = _make_full_orchestrator()
        orch.initialize()
        assert orch.is_initialized
        orch.shutdown()
        assert not orch.is_initialized

    def test_double_init_raises(self):
        orch = _make_full_orchestrator()
        orch.initialize()
        with pytest.raises(JanusAlreadyInitializedError):
            orch.initialize()
        orch.shutdown()

    def test_shutdown_idempotent(self):
        orch = _make_full_orchestrator()
        orch.initialize()
        orch.shutdown()
        orch.shutdown()  # should be no-op

    def test_operation_before_init_raises(self):
        orch = _make_full_orchestrator()
        with pytest.raises(JanusNotInitializedError):
            orch.list_assessments_by_status(FutureAssessmentStatus.PENDING)

    def test_operation_after_shutdown_raises(self):
        orch = _make_full_orchestrator()
        orch.initialize()
        orch.shutdown()
        with pytest.raises(JanusShutdownError):
            orch.list_assessments_by_status(FutureAssessmentStatus.PENDING)

    def test_check_health_before_init(self):
        orch = _make_full_orchestrator()
        resp = orch.check_health(OrchestratorHealthRequest())
        assert not resp.orchestrator_healthy

    def test_check_health_after_init(self):
        orch = _make_full_orchestrator()
        orch.initialize()
        resp = orch.check_health(OrchestratorHealthRequest())
        assert resp.orchestrator_healthy
        orch.shutdown()

    def test_repr(self):
        orch = _make_full_orchestrator()
        r = repr(orch)
        assert "FutureOrchestrator" in r

    def test_workflow_diagnostics_before_init(self):
        orch = _make_full_orchestrator()
        diag = orch.workflow_diagnostics()
        assert not diag["initialized"]

    def test_workflow_diagnostics_after_init(self):
        orch = _make_full_orchestrator()
        orch.initialize()
        diag = orch.workflow_diagnostics()
        assert diag["initialized"]
        assert "bounded_exploration_limits" in diag
        orch.shutdown()

    def test_workflow_statistics(self):
        orch = _make_full_orchestrator()
        orch.initialize()
        stats = orch.workflow_statistics()
        assert stats["assessments_created"] == 0
        orch.shutdown()


class TestFutureOrchestratorStatusManagement:
    @pytest.fixture(autouse=True)
    def orch(self):
        o = _make_full_orchestrator()
        o.initialize()
        yield o
        if o.is_initialized:
            o.shutdown()

    def _create_fake_assessment(self, orch) -> FutureAssessment:
        fa = FutureAssessment.create(
            title="Test Assessment", description="desc", context="ctx",
            scenarios=[], forecasts=[], simulations=[],
            risk_assessments=[], opportunity_assessments=[],
            timeline_projections=[], counterfactuals=[], strategic_forecasts=[],
            overall_uncertainty=make_uncertainty(), overall_confidence=make_confidence(),
            summary="summary",
        )
        orch._assessments[fa.assessment_id] = fa
        orch._assessment_requests[fa.assessment_id] = None
        return fa

    def test_get_future_assessment_not_found_raises(self, orch):
        with pytest.raises(JanusAssessmentNotFoundError):
            orch.get_future_assessment(FutureAssessmentGetRequest(assessment_id="nonexistent"))

    def test_list_assessments_by_status_empty(self, orch):
        results = orch.list_assessments_by_status(FutureAssessmentStatus.PENDING)
        assert results == ()

    def test_list_assessments_by_status_with_assessment(self, orch):
        fa = self._create_fake_assessment(orch)
        results = orch.list_assessments_by_status(FutureAssessmentStatus.PENDING)
        assert any(a.assessment_id == fa.assessment_id for a in results)

    def test_update_assessment_status_valid_transition(self, orch):
        fa = self._create_fake_assessment(orch)
        resp = orch.update_assessment_status(
            FutureAssessmentUpdateStatusRequest(
                assessment_id=fa.assessment_id,
                target_status=FutureAssessmentStatus.IN_PROGRESS,
                reason="starting",
                updated_by="test",
            )
        )
        assert resp.previous_status == FutureAssessmentStatus.PENDING
        assert resp.new_status == FutureAssessmentStatus.IN_PROGRESS

    def test_update_assessment_status_invalid_transition_raises(self, orch):
        fa = self._create_fake_assessment(orch)
        with pytest.raises(JanusAssessmentStatusTransitionError):
            orch.update_assessment_status(
                FutureAssessmentUpdateStatusRequest(
                    assessment_id=fa.assessment_id,
                    target_status=FutureAssessmentStatus.COMPLETE,  # skips IN_PROGRESS
                    reason="jump",
                    updated_by="test",
                )
            )

    def test_invalidate_assessment(self, orch):
        fa = self._create_fake_assessment(orch)
        invalidated = orch.invalidate_assessment(fa.assessment_id, "test reason", "test-user")
        assert invalidated.status == FutureAssessmentStatus.INVALIDATED

    def test_invalidate_assessment_not_found_raises(self, orch):
        with pytest.raises(JanusAssessmentNotFoundError):
            orch.invalidate_assessment("nonexistent", "reason", "user")

    def test_invalidate_already_invalidated_raises(self, orch):
        fa = self._create_fake_assessment(orch)
        orch.invalidate_assessment(fa.assessment_id, "first", "user")
        with pytest.raises(JanusAssessmentStatusTransitionError):
            orch.invalidate_assessment(fa.assessment_id, "second", "user")

    def test_check_health_notes_unhealthy_engine(self, orch):
        orch._scenario_engine.is_initialized = False
        resp = orch.check_health(OrchestratorHealthRequest())
        assert not resp.orchestrator_healthy
        assert "scenario_engine" in resp.notes


# ---------------------------------------------------------------------------
# ============================================================
# END-TO-END INTEGRATION: engines used together
# ============================================================
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests exercising multiple engines in concert."""

    def test_branch_analysis_and_integrity(self):
        integrity = ScenarioIntegrityEngine()
        integrity.initialize()
        branch_engine = BranchAnalysisEngine()
        branch_engine.initialize()

        branch = branch_engine.construct_branch(
            label="Integrated Branch",
            description="integration test",
            triggering_choice="go-to-market",
            future_state=make_future_state(),
            probability=0.65,
            risk_assessment=make_risk_assessment(),
            opportunity_assessment=make_opportunity_assessment(),
            confidence=make_confidence(),
            scenario_id="sc-integration",
        )
        assert branch.branch_id != ""

        scenario = make_scenario()
        is_valid, _ = integrity.validate_scenario(scenario)
        assert is_valid

        branch_engine.shutdown()
        integrity.shutdown()

    def test_counterfactual_and_forecasting(self):
        cf_engine = CounterfactualEngine()
        cf_engine.initialize()
        fc_engine = ForecastingEngine()
        fc_engine.initialize()

        req = _make_cf_request()
        cf_resp = cf_engine.create_counterfactual(req)
        assert cf_resp.counterfactual.counterfactual_id != ""

        fc_req = _make_forecast_req()
        fc_resp = fc_engine.generate_forecast(fc_req)
        assert fc_resp.forecast.forecast_id != ""

        cf_engine.shutdown()
        fc_engine.shutdown()

    def test_risk_and_opportunity_together(self):
        risk_engine = RiskAnalysisEngine()
        risk_engine.initialize()
        opp_engine = OpportunityAnalysisEngine()
        opp_engine.initialize()

        risk_resp = risk_engine.analyze_risk(
            RiskAnalysisRequest(
                title="Risk A", description="d",
                risk_factors=(make_risk_factor(),),
                uncertainty=make_uncertainty(),
                horizon=ForecastHorizon.ONE_YEAR,
                evidence=make_evidence(),
                confidence=make_confidence(),
            )
        )
        opp_resp = opp_engine.analyze_opportunity(
            OpportunityAnalysisRequest(
                title="Opp A", description="d",
                opportunity_factors=(make_opportunity_factor(),),
                uncertainty=make_uncertainty(),
                horizon=ForecastHorizon.ONE_YEAR,
                evidence=make_evidence(),
                confidence=make_confidence(),
            )
        )
        ra = risk_resp.risk_assessment
        oa = opp_resp.opportunity_assessment
        assert ra.composite_risk_score >= 0.0
        assert oa.composite_opportunity_score >= 0.0

        risk_engine.shutdown()
        opp_engine.shutdown()

    def test_future_modeling_and_integrity(self):
        fm_engine = FutureModelingEngine()
        fm_engine.initialize()
        integrity = ScenarioIntegrityEngine()
        integrity.initialize()

        req = FutureModelConstructRequest(
            title="Integration FM", description="d", context="c",
            future_states=(make_future_state(),),
            uncertainty=make_uncertainty(),
            confidence=make_confidence(),
            evidence=make_evidence(),
        )
        resp = fm_engine.construct_future_model(req)
        is_valid, violations = integrity.validate_future_model(resp.future_model)
        assert is_valid, f"Violations: {violations}"

        fm_engine.shutdown()
        integrity.shutdown()

    def test_scenario_engine_generates_multiple_types(self):
        e = ScenarioEngine()
        e.initialize()
        meta = ScenarioMetadata.create(created_by="test", source_subsystem="JANUS")
        req = ScenarioGenerateRequest(
            decision_context="Expand or contract?",
            scenario_types=(ScenarioType.BASELINE, ScenarioType.OPTIMISTIC, ScenarioType.PESSIMISTIC),
            horizon=ForecastHorizon.ONE_YEAR,
            requested_by="VEGA",
            metadata=meta,
        )
        resp = e.generate_scenarios(req)
        assert resp.scenario_count >= 1
        e.shutdown()

    def test_uncertainty_aggregation_across_engines(self):
        unc_engine = UncertaintyEngine()
        unc_engine.initialize()

        profiles = tuple(
            make_uncertainty(level)
            for level in [UncertaintyLevel.LOW, UncertaintyLevel.HIGH, UncertaintyLevel.EXTREME]
        )
        aggregated = unc_engine.aggregate_uncertainty(profiles)
        assert aggregated is not None
        assert aggregated.level in UncertaintyLevel
        unc_engine.shutdown()


# ---------------------------------------------------------------------------
# ============================================================
# SCENARIO METADATA
# ============================================================
# ---------------------------------------------------------------------------


class TestScenarioMetadata:
    def test_create_valid(self):
        meta = ScenarioMetadata.create(
            created_by="JANUS-test",
            source_subsystem="JANUS",
            tags=["tag-A", "tag-B"],
            version=2,
        )
        assert meta.created_by == "JANUS-test"
        assert "tag-A" in meta.tags
        assert meta.version == 2

    def test_default_version(self):
        meta = ScenarioMetadata.create(created_by="t", source_subsystem="J")
        assert meta.version == 1

    def test_tags_immutable_tuple(self):
        meta = ScenarioMetadata.create(created_by="t", source_subsystem="J", tags=["x"])
        assert isinstance(meta.tags, tuple)


class TestForecastMetadata:
    def test_create_valid(self):
        meta = ForecastMetadata.create(
            model_version="5.1.0",
            data_sources=["source-A", "source-B"],
            horizon=ForecastHorizon.ONE_YEAR,
            generated_by="ForecastingEngine",
        )
        assert meta.model_version == "5.1.0"
        assert len(meta.data_sources) == 2

    def test_data_sources_immutable_tuple(self):
        meta = ForecastMetadata.create(
            model_version="1.0", data_sources=["s"],
            horizon=ForecastHorizon.ONE_MONTH, generated_by="test",
        )
        assert isinstance(meta.data_sources, tuple)


class TestSimulationMetadata:
    def test_create_valid(self):
        meta = SimulationMetadata.create(iterations=1000, engine_version="5.1.0", seed=42)
        assert meta.iterations == 1000
        assert meta.seed == 42

    def test_no_seed(self):
        meta = SimulationMetadata.create(iterations=100, engine_version="5.1.0")
        assert meta.seed is None


# ---------------------------------------------------------------------------
# ============================================================
# SCHEMA RESPONSE TYPES
# ============================================================
# ---------------------------------------------------------------------------


class TestResponseSchemas:
    def test_scenario_generate_response(self):
        from subsystems.janus.schemas import ScenarioGenerateResponse
        resp = ScenarioGenerateResponse(
            scenarios=(make_scenario(),),
            generation_context="ctx",
            horizon=ForecastHorizon.ONE_YEAR,
            generated_at=datetime.utcnow(),
            engine_version="5.1.0",
        )
        assert resp.scenario_count == 1

    def test_forecast_generate_response(self):
        from subsystems.janus.schemas import ForecastGenerateResponse
        resp = ForecastGenerateResponse(
            forecast=make_forecast(),
            generated_at=datetime.utcnow(),
            engine_version="5.1.0",
        )
        assert resp.forecast.forecast_id != ""

    def test_counterfactual_compare_response(self):
        from subsystems.janus.schemas import CounterfactualCompareResponse
        resp = CounterfactualCompareResponse(
            counterfactual_id="cf-1",
            original_scenario_id="sc-1",
            delta_risk=-0.1,
            delta_opportunity=0.2,
            net_delta=0.3,
            learning_insights=("insight-1",),
            compared_at=datetime.utcnow(),
        )
        assert resp.net_delta == 0.3

    def test_uncertainty_validation_response(self):
        from subsystems.janus.schemas import UncertaintyValidationResponse
        resp = UncertaintyValidationResponse(
            artifact_id="art-1",
            is_valid=True,
            violations=(),
            validated_at=datetime.utcnow(),
        )
        assert resp.is_valid

    def test_risk_list_by_level_response(self):
        from subsystems.janus.schemas import RiskListByLevelResponse
        resp = RiskListByLevelResponse(
            risk_assessments=(make_risk_assessment(),),
            minimum_level=RiskLevel.MODERATE,
            retrieved_at=datetime.utcnow(),
        )
        assert len(resp.risk_assessments) == 1

    def test_orchestrator_health_response(self):
        from subsystems.janus.schemas import OrchestratorHealthResponse
        resp = OrchestratorHealthResponse(
            orchestrator_healthy=True,
            engine_statuses={"scenario_engine": True},
            checked_at=datetime.utcnow(),
            notes="all ok",
        )
        assert resp.orchestrator_healthy


# ---------------------------------------------------------------------------
# ============================================================
# ADDITIONAL EDGE CASES
# ============================================================
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_probability_level_boundary_values(self):
        assert ProbabilityLevel.from_float(0.8001) == ProbabilityLevel.HIGHLY_LIKELY
        assert ProbabilityLevel.from_float(0.8000) == ProbabilityLevel.LIKELY
        assert ProbabilityLevel.from_float(0.6001) == ProbabilityLevel.LIKELY
        assert ProbabilityLevel.from_float(0.6000) == ProbabilityLevel.POSSIBLE
        assert ProbabilityLevel.from_float(0.4001) == ProbabilityLevel.POSSIBLE
        assert ProbabilityLevel.from_float(0.4000) == ProbabilityLevel.UNLIKELY
        assert ProbabilityLevel.from_float(0.2001) == ProbabilityLevel.UNLIKELY
        assert ProbabilityLevel.from_float(0.2000) == ProbabilityLevel.HIGHLY_UNCERTAIN

    def test_probability_distribution_edge_sum(self):
        # Sum exactly 1.0 by construction
        dist = ProbabilityDistribution.create(
            label="edge",
            outcomes={"a": 1.0},
            uncertainty_level=UncertaintyLevel.LOW,
            confidence=make_confidence(),
        )
        assert dist.mode == "a"

    def test_risk_factor_zero_scores(self):
        rf = RiskFactor.create(
            name="r", description="d", category="c",
            impact_score=0.0, likelihood_score=0.0,
            time_horizon=ForecastHorizon.ONE_MONTH,
        )
        assert rf.composite_score == 0.0

    def test_future_assessment_empty_collections(self):
        fa = FutureAssessment.create(
            title="Empty FA", description="d", context="c",
            scenarios=[], forecasts=[], simulations=[],
            risk_assessments=[], opportunity_assessments=[],
            timeline_projections=[], counterfactuals=[], strategic_forecasts=[],
            overall_uncertainty=make_uncertainty(), overall_confidence=make_confidence(),
            summary="",
        )
        assert fa.scenario_count == 0
        assert not fa.has_critical_risks
        assert not fa.has_transformative_opportunities

    def test_scenario_comparison_unknown_ranked_id_raises(self):
        s = make_scenario()
        with pytest.raises(ValueError):
            ScenarioComparison.create(
                title="cmp",
                scenarios=[s],
                ranked_scenario_ids=["nonexistent-id"],
                trade_off_analysis={},
                dominance_map={},
                risk_adjusted_scores={},
                opportunity_scores={},
                recommended_for_review="nonexistent-id",
                rationale="test",
                confidence=make_confidence(),
            )

    def test_branch_analysis_config_immutable(self):
        config = BranchAnalysisConfig()
        with pytest.raises(Exception):
            config.max_branch_depth = 99

    def test_counterfactual_learning_insights_are_mutable_list(self):
        fm = make_future_model()
        cf = CounterfactualScenario.create(
            title="CF", description="d", original_event="e",
            counterfactual_condition="if X",
            divergence_point=datetime.utcnow() - timedelta(days=1),
            resulting_future_model=fm,
            delta_risk=0.0, delta_opportunity=0.0,
            uncertainty=make_uncertainty(), confidence=make_confidence(),
            evidence=make_evidence(),
            learning_insights=["initial"],
        )
        cf.learning_insights.append("added")
        assert "added" in cf.learning_insights

    def test_branch_analysis_engine_name_and_version(self):
        e = BranchAnalysisEngine()
        assert e.engine_name == "BranchAnalysisEngine"
        assert e.engine_version.startswith("5.")

    def test_counterfactual_engine_name_and_version(self):
        e = CounterfactualEngine()
        assert e.engine_name == "CounterfactualEngine"
        assert e.engine_version.startswith("5.")

    def test_orchestrator_name_and_version(self):
        orch = _make_full_orchestrator()
        assert orch.engine_name == "FutureOrchestrator"
        assert "5.1" in orch.engine_version

    def test_scenario_branch_probability_level(self):
        b = make_scenario_branch(probability=0.85)
        # Branch does not expose probability_level but FutureState does
        assert b.probability == 0.85

    def test_timeline_projection_no_critical_milestones_completion_is_one(self):
        m = make_projection_milestone(is_critical=False)
        tp = TimelineProjection.create(
            title="TP", description="d", context="c",
            milestones=[m], horizon=ForecastHorizon.ONE_YEAR,
            uncertainty=make_uncertainty(), confidence=make_confidence(),
            evidence=make_evidence(),
        )
        assert tp.completion_probability == 1.0

    def test_janus_error_no_engine_str(self):
        err = JanusError("plain error")
        assert "plain error" in str(err)
        assert "[" not in str(err)

    def test_evidence_profile_empty_patterns(self):
        ev = EvidenceProfile.create(
            sources=["source-A"],
            patterns_observed=[],
            contradicting_evidence=[],
            evidence_strength=0.50,
        )
        assert ev.patterns_observed == ()