"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: models.py

Defines all core data models, enums, and value objects for JANUS.
JANUS owns: Scenario Modeling, Future Modeling, Forecasting, Branch Analysis,
Counterfactual Analysis, Uncertainty Modeling, Future Risk/Opportunity Analysis,
Outcome Simulation, Strategic Forecasting, Timeline Projection, Probability Estimation,
Scenario Evaluation.

JANUS never owns: Reasoning, Decision Making, Planning, Knowledge Storage, Identity.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Final, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScenarioStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    EVALUATING = auto()
    EVALUATED = auto()
    ARCHIVED = auto()
    INVALIDATED = auto()


class ScenarioType(Enum):
    BASELINE = auto()
    OPTIMISTIC = auto()
    PESSIMISTIC = auto()
    ALTERNATIVE = auto()
    COUNTERFACTUAL = auto()
    EXPLORATORY = auto()
    STRESS_TEST = auto()
    DOMINANT = auto()


class ForecastType(Enum):
    PROBABILISTIC = auto()
    STRATEGIC = auto()
    TREND = auto()
    RISK = auto()
    OPPORTUNITY = auto()
    OUTCOME = auto()
    TIMELINE = auto()


class ForecastHorizon(Enum):
    ONE_MONTH = "1_month"
    THREE_MONTHS = "3_months"
    SIX_MONTHS = "6_months"
    ONE_YEAR = "1_year"
    FIVE_YEARS = "5_years"
    TEN_YEARS = "10_years"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class ProbabilityLevel(Enum):
    HIGHLY_LIKELY = "highly_likely"       # > 80%
    LIKELY = "likely"                      # 60–80%
    POSSIBLE = "possible"                  # 40–60%
    UNLIKELY = "unlikely"                  # 20–40%
    HIGHLY_UNCERTAIN = "highly_uncertain"  # < 20%

    @classmethod
    def from_float(cls, probability: float) -> "ProbabilityLevel":
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        if probability > 0.80:
            return cls.HIGHLY_LIKELY
        if probability > 0.60:
            return cls.LIKELY
        if probability > 0.40:
            return cls.POSSIBLE
        if probability > 0.20:
            return cls.UNLIKELY
        return cls.HIGHLY_UNCERTAIN


class UncertaintyLevel(Enum):
    NEGLIGIBLE = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    EXTREME = auto()


class RiskLevel(Enum):
    NEGLIGIBLE = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()


class OpportunityLevel(Enum):
    MARGINAL = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    TRANSFORMATIVE = auto()


class SimulationStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()


class ProjectionStatus(Enum):
    DRAFT = auto()
    ACTIVE = auto()
    REVISED = auto()
    EXPIRED = auto()
    SUPERSEDED = auto()


class FutureAssessmentStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()
    INVALIDATED = auto()


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioMetadata:
    """Immutable metadata envelope for a Scenario."""
    created_at: datetime
    created_by: str
    source_subsystem: str
    tags: tuple[str, ...]
    version: int = 1

    @classmethod
    def create(
        cls,
        created_by: str,
        source_subsystem: str,
        tags: Optional[list[str]] = None,
        version: int = 1,
    ) -> "ScenarioMetadata":
        return cls(
            created_at=datetime.utcnow(),
            created_by=created_by,
            source_subsystem=source_subsystem,
            tags=tuple(tags or []),
            version=version,
        )


@dataclass(frozen=True)
class ForecastMetadata:
    """Immutable metadata envelope for a Forecast."""
    created_at: datetime
    model_version: str
    data_sources: tuple[str, ...]
    horizon: ForecastHorizon
    generated_by: str

    @classmethod
    def create(
        cls,
        model_version: str,
        data_sources: list[str],
        horizon: ForecastHorizon,
        generated_by: str,
    ) -> "ForecastMetadata":
        return cls(
            created_at=datetime.utcnow(),
            model_version=model_version,
            data_sources=tuple(data_sources),
            horizon=horizon,
            generated_by=generated_by,
        )


@dataclass(frozen=True)
class SimulationMetadata:
    """Immutable metadata envelope for a Simulation."""
    created_at: datetime
    iterations: int
    seed: Optional[int]
    engine_version: str

    @classmethod
    def create(
        cls,
        iterations: int,
        engine_version: str,
        seed: Optional[int] = None,
    ) -> "SimulationMetadata":
        return cls(
            created_at=datetime.utcnow(),
            iterations=iterations,
            seed=seed,
            engine_version=engine_version,
        )


@dataclass(frozen=True)
class RiskFactor:
    """Immutable representation of a single future risk factor."""
    factor_id: str
    name: str
    description: str
    category: str
    impact_score: float           # 0.0 – 1.0
    likelihood_score: float       # 0.0 – 1.0
    time_horizon: ForecastHorizon
    mitigations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.impact_score <= 1.0:
            raise ValueError(f"impact_score must be in [0, 1], got {self.impact_score}")
        if not 0.0 <= self.likelihood_score <= 1.0:
            raise ValueError(f"likelihood_score must be in [0, 1], got {self.likelihood_score}")

    @property
    def composite_score(self) -> float:
        return self.impact_score * self.likelihood_score

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        category: str,
        impact_score: float,
        likelihood_score: float,
        time_horizon: ForecastHorizon,
        mitigations: Optional[list[str]] = None,
    ) -> "RiskFactor":
        return cls(
            factor_id=str(uuid.uuid4()),
            name=name,
            description=description,
            category=category,
            impact_score=impact_score,
            likelihood_score=likelihood_score,
            time_horizon=time_horizon,
            mitigations=tuple(mitigations or []),
        )


@dataclass(frozen=True)
class OpportunityFactor:
    """Immutable representation of a single future opportunity factor."""
    factor_id: str
    name: str
    description: str
    category: str
    value_score: float            # 0.0 – 1.0
    feasibility_score: float      # 0.0 – 1.0
    time_horizon: ForecastHorizon
    enablers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.value_score <= 1.0:
            raise ValueError(f"value_score must be in [0, 1], got {self.value_score}")
        if not 0.0 <= self.feasibility_score <= 1.0:
            raise ValueError(f"feasibility_score must be in [0, 1], got {self.feasibility_score}")

    @property
    def composite_score(self) -> float:
        return self.value_score * self.feasibility_score

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        category: str,
        value_score: float,
        feasibility_score: float,
        time_horizon: ForecastHorizon,
        enablers: Optional[list[str]] = None,
    ) -> "OpportunityFactor":
        return cls(
            factor_id=str(uuid.uuid4()),
            name=name,
            description=description,
            category=category,
            value_score=value_score,
            feasibility_score=feasibility_score,
            time_horizon=time_horizon,
            enablers=tuple(enablers or []),
        )


@dataclass(frozen=True)
class ProjectionMilestone:
    """Immutable milestone within a timeline projection."""
    milestone_id: str
    label: str
    description: str
    projected_at: datetime
    probability: float            # 0.0 – 1.0
    dependencies: tuple[str, ...]
    is_critical: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")

    @classmethod
    def create(
        cls,
        label: str,
        description: str,
        projected_at: datetime,
        probability: float,
        is_critical: bool = False,
        dependencies: Optional[list[str]] = None,
    ) -> "ProjectionMilestone":
        return cls(
            milestone_id=str(uuid.uuid4()),
            label=label,
            description=description,
            projected_at=projected_at,
            probability=probability,
            dependencies=tuple(dependencies or []),
            is_critical=is_critical,
        )


@dataclass(frozen=True)
class ConfidenceProfile:
    """Immutable confidence envelope for any forecast or scenario."""
    overall: float                # 0.0 – 1.0
    data_quality: float           # 0.0 – 1.0
    model_fit: float              # 0.0 – 1.0
    signal_strength: float        # 0.0 – 1.0
    notes: str

    def __post_init__(self) -> None:
        for attr in ("overall", "data_quality", "model_fit", "signal_strength"):
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be in [0, 1], got {value}")

    @property
    def level(self) -> ProbabilityLevel:
        return ProbabilityLevel.from_float(self.overall)

    @classmethod
    def create(
        cls,
        overall: float,
        data_quality: float,
        model_fit: float,
        signal_strength: float,
        notes: str = "",
    ) -> "ConfidenceProfile":
        return cls(
            overall=overall,
            data_quality=data_quality,
            model_fit=model_fit,
            signal_strength=signal_strength,
            notes=notes,
        )


@dataclass(frozen=True)
class EvidenceProfile:
    """Immutable evidence record supporting a forecast or scenario."""
    evidence_id: str
    sources: tuple[str, ...]
    patterns_observed: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    evidence_strength: float      # 0.0 – 1.0
    collected_at: datetime

    def __post_init__(self) -> None:
        if not 0.0 <= self.evidence_strength <= 1.0:
            raise ValueError(
                f"evidence_strength must be in [0, 1], got {self.evidence_strength}"
            )

    @classmethod
    def create(
        cls,
        sources: list[str],
        patterns_observed: list[str],
        contradicting_evidence: list[str],
        evidence_strength: float,
    ) -> "EvidenceProfile":
        return cls(
            evidence_id=str(uuid.uuid4()),
            sources=tuple(sources),
            patterns_observed=tuple(patterns_observed),
            contradicting_evidence=tuple(contradicting_evidence),
            evidence_strength=evidence_strength,
            collected_at=datetime.utcnow(),
        )


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


@dataclass
class ProbabilityDistribution:
    """
    Discrete probability distribution over named outcomes.
    All weights must sum to 1.0 (±1e-6 tolerance).
    """
    distribution_id: str
    label: str
    outcomes: dict[str, float]         # outcome_label → probability
    uncertainty_level: UncertaintyLevel
    confidence: ConfidenceProfile
    created_at: datetime

    _TOLERANCE: Final[float] = field(default=1e-6, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        total = sum(self.outcomes.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ProbabilityDistribution outcomes must sum to 1.0, got {total:.6f}"
            )
        for key, prob in self.outcomes.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"Probability for '{key}' must be in [0, 1], got {prob}")

    @property
    def mode(self) -> str:
        return max(self.outcomes, key=lambda k: self.outcomes[k])

    @property
    def entropy(self) -> float:
        import math
        return -sum(
            p * math.log2(p) for p in self.outcomes.values() if p > 0.0
        )

    @classmethod
    def create(
        cls,
        label: str,
        outcomes: dict[str, float],
        uncertainty_level: UncertaintyLevel,
        confidence: ConfidenceProfile,
    ) -> "ProbabilityDistribution":
        return cls(
            distribution_id=str(uuid.uuid4()),
            label=label,
            outcomes=outcomes,
            uncertainty_level=uncertainty_level,
            confidence=confidence,
            created_at=datetime.utcnow(),
        )


@dataclass
class UncertaintyProfile:
    """Models uncertainty across known, unknown, and external dimensions."""
    uncertainty_id: str
    level: UncertaintyLevel
    known_risks: list[str]
    unknown_risk_exposure: float       # 0.0 – 1.0
    volatility_score: float            # 0.0 – 1.0
    external_factors: list[str]
    market_sensitivity: float          # 0.0 – 1.0
    technology_sensitivity: float      # 0.0 – 1.0
    notes: str
    created_at: datetime

    def __post_init__(self) -> None:
        for attr in (
            "unknown_risk_exposure",
            "volatility_score",
            "market_sensitivity",
            "technology_sensitivity",
        ):
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be in [0, 1], got {value}")

    @classmethod
    def create(
        cls,
        level: UncertaintyLevel,
        known_risks: list[str],
        unknown_risk_exposure: float,
        volatility_score: float,
        external_factors: list[str],
        market_sensitivity: float,
        technology_sensitivity: float,
        notes: str = "",
    ) -> "UncertaintyProfile":
        return cls(
            uncertainty_id=str(uuid.uuid4()),
            level=level,
            known_risks=known_risks,
            unknown_risk_exposure=unknown_risk_exposure,
            volatility_score=volatility_score,
            external_factors=external_factors,
            market_sensitivity=market_sensitivity,
            technology_sensitivity=technology_sensitivity,
            notes=notes,
            created_at=datetime.utcnow(),
        )


@dataclass
class RiskAssessment:
    """Future risk assessment produced by the Future Risk Engine."""
    risk_id: str
    title: str
    description: str
    level: RiskLevel
    risk_factors: list[RiskFactor]
    probability_distribution: ProbabilityDistribution
    uncertainty: UncertaintyProfile
    horizon: ForecastHorizon
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    mitigation_strategies: list[str]
    created_at: datetime
    updated_at: datetime

    @property
    def composite_risk_score(self) -> float:
        if not self.risk_factors:
            return 0.0
        return sum(rf.composite_score for rf in self.risk_factors) / len(self.risk_factors)

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        level: RiskLevel,
        risk_factors: list[RiskFactor],
        probability_distribution: ProbabilityDistribution,
        uncertainty: UncertaintyProfile,
        horizon: ForecastHorizon,
        confidence: ConfidenceProfile,
        evidence: EvidenceProfile,
        mitigation_strategies: Optional[list[str]] = None,
    ) -> "RiskAssessment":
        now = datetime.utcnow()
        return cls(
            risk_id=str(uuid.uuid4()),
            title=title,
            description=description,
            level=level,
            risk_factors=risk_factors,
            probability_distribution=probability_distribution,
            uncertainty=uncertainty,
            horizon=horizon,
            confidence=confidence,
            evidence=evidence,
            mitigation_strategies=mitigation_strategies or [],
            created_at=now,
            updated_at=now,
        )


@dataclass
class OpportunityAssessment:
    """Future opportunity assessment produced by the Future Opportunity Engine."""
    opportunity_id: str
    title: str
    description: str
    level: OpportunityLevel
    opportunity_factors: list[OpportunityFactor]
    probability_distribution: ProbabilityDistribution
    uncertainty: UncertaintyProfile
    horizon: ForecastHorizon
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    capture_strategies: list[str]
    created_at: datetime
    updated_at: datetime

    @property
    def composite_opportunity_score(self) -> float:
        if not self.opportunity_factors:
            return 0.0
        return sum(f.composite_score for f in self.opportunity_factors) / len(
            self.opportunity_factors
        )

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        level: OpportunityLevel,
        opportunity_factors: list[OpportunityFactor],
        probability_distribution: ProbabilityDistribution,
        uncertainty: UncertaintyProfile,
        horizon: ForecastHorizon,
        confidence: ConfidenceProfile,
        evidence: EvidenceProfile,
        capture_strategies: Optional[list[str]] = None,
    ) -> "OpportunityAssessment":
        now = datetime.utcnow()
        return cls(
            opportunity_id=str(uuid.uuid4()),
            title=title,
            description=description,
            level=level,
            opportunity_factors=opportunity_factors,
            probability_distribution=probability_distribution,
            uncertainty=uncertainty,
            horizon=horizon,
            confidence=confidence,
            evidence=evidence,
            capture_strategies=capture_strategies or [],
            created_at=now,
            updated_at=now,
        )


@dataclass
class FutureState:
    """A single modeled future state at a particular time horizon."""
    state_id: str
    label: str
    description: str
    horizon: ForecastHorizon
    attributes: dict[str, Any]           # domain-specific state attributes
    probability: float                   # 0.0 – 1.0
    uncertainty: UncertaintyProfile
    risks: list[RiskAssessment]
    opportunities: list[OpportunityAssessment]
    created_at: datetime

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")

    @property
    def probability_level(self) -> ProbabilityLevel:
        return ProbabilityLevel.from_float(self.probability)

    @classmethod
    def create(
        cls,
        label: str,
        description: str,
        horizon: ForecastHorizon,
        attributes: dict[str, Any],
        probability: float,
        uncertainty: UncertaintyProfile,
        risks: Optional[list[RiskAssessment]] = None,
        opportunities: Optional[list[OpportunityAssessment]] = None,
    ) -> "FutureState":
        return cls(
            state_id=str(uuid.uuid4()),
            label=label,
            description=description,
            horizon=horizon,
            attributes=attributes,
            probability=probability,
            uncertainty=uncertainty,
            risks=risks or [],
            opportunities=opportunities or [],
            created_at=datetime.utcnow(),
        )


@dataclass
class FutureModel:
    """
    Structured future model spanning multiple time horizons.
    Produced by the Future Modeling Engine.
    """
    model_id: str
    title: str
    description: str
    context: str
    future_states: list[FutureState]
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    created_at: datetime
    updated_at: datetime

    @property
    def horizons(self) -> list[ForecastHorizon]:
        return list({fs.horizon for fs in self.future_states})

    def state_at(self, horizon: ForecastHorizon) -> list[FutureState]:
        return [fs for fs in self.future_states if fs.horizon == horizon]

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        context: str,
        future_states: list[FutureState],
        uncertainty: UncertaintyProfile,
        confidence: ConfidenceProfile,
        evidence: EvidenceProfile,
    ) -> "FutureModel":
        now = datetime.utcnow()
        return cls(
            model_id=str(uuid.uuid4()),
            title=title,
            description=description,
            context=context,
            future_states=future_states,
            uncertainty=uncertainty,
            confidence=confidence,
            evidence=evidence,
            created_at=now,
            updated_at=now,
        )


@dataclass
class ScenarioBranch:
    """A single decision branch within a Scenario."""
    branch_id: str
    label: str
    description: str
    triggering_choice: str
    future_state: FutureState
    probability: float             # 0.0 – 1.0
    risk_assessment: RiskAssessment
    opportunity_assessment: OpportunityAssessment
    confidence: ConfidenceProfile
    created_at: datetime

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")

    @classmethod
    def create(
        cls,
        label: str,
        description: str,
        triggering_choice: str,
        future_state: FutureState,
        probability: float,
        risk_assessment: RiskAssessment,
        opportunity_assessment: OpportunityAssessment,
        confidence: ConfidenceProfile,
    ) -> "ScenarioBranch":
        return cls(
            branch_id=str(uuid.uuid4()),
            label=label,
            description=description,
            triggering_choice=triggering_choice,
            future_state=future_state,
            probability=probability,
            risk_assessment=risk_assessment,
            opportunity_assessment=opportunity_assessment,
            confidence=confidence,
            created_at=datetime.utcnow(),
        )


@dataclass
class Scenario:
    """
    Core scenario entity — a possible future generated by the Scenario Engine.
    JANUS generates and evaluates scenarios; VEGA selects among them.
    """
    scenario_id: str
    title: str
    description: str
    scenario_type: ScenarioType
    status: ScenarioStatus
    branches: list[ScenarioBranch]
    future_model: FutureModel
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    metadata: ScenarioMetadata
    risk_score: float              # 0.0 – 1.0
    opportunity_score: float       # 0.0 – 1.0
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        for attr in ("risk_score", "opportunity_score"):
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be in [0, 1], got {value}")

    @property
    def dominant_branch(self) -> Optional[ScenarioBranch]:
        if not self.branches:
            return None
        return max(self.branches, key=lambda b: b.probability)

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        scenario_type: ScenarioType,
        branches: list[ScenarioBranch],
        future_model: FutureModel,
        uncertainty: UncertaintyProfile,
        confidence: ConfidenceProfile,
        evidence: EvidenceProfile,
        metadata: ScenarioMetadata,
        risk_score: float = 0.0,
        opportunity_score: float = 0.0,
    ) -> "Scenario":
        now = datetime.utcnow()
        return cls(
            scenario_id=str(uuid.uuid4()),
            title=title,
            description=description,
            scenario_type=scenario_type,
            status=ScenarioStatus.PENDING,
            branches=branches,
            future_model=future_model,
            uncertainty=uncertainty,
            confidence=confidence,
            evidence=evidence,
            metadata=metadata,
            risk_score=risk_score,
            opportunity_score=opportunity_score,
            created_at=now,
            updated_at=now,
        )


@dataclass
class ScenarioComparison:
    """
    Structured comparison of two or more scenarios.
    Produced by the Scenario Evaluation Engine.
    """
    comparison_id: str
    title: str
    scenarios: list[Scenario]
    ranked_scenario_ids: list[str]        # scenario_ids ordered best → worst
    trade_off_analysis: dict[str, str]    # scenario_id → trade-off summary
    dominance_map: dict[str, list[str]]   # scenario_id → list of dominated scenario_ids
    risk_adjusted_scores: dict[str, float]
    opportunity_scores: dict[str, float]
    recommended_for_review: str           # scenario_id — not a decision, a recommendation
    rationale: str
    confidence: ConfidenceProfile
    created_at: datetime

    def __post_init__(self) -> None:
        ids = {s.scenario_id for s in self.scenarios}
        unknown = set(self.ranked_scenario_ids) - ids
        if unknown:
            raise ValueError(f"ranked_scenario_ids contains unknown scenario IDs: {unknown}")

    @classmethod
    def create(
        cls,
        title: str,
        scenarios: list[Scenario],
        ranked_scenario_ids: list[str],
        trade_off_analysis: dict[str, str],
        dominance_map: dict[str, list[str]],
        risk_adjusted_scores: dict[str, float],
        opportunity_scores: dict[str, float],
        recommended_for_review: str,
        rationale: str,
        confidence: ConfidenceProfile,
    ) -> "ScenarioComparison":
        return cls(
            comparison_id=str(uuid.uuid4()),
            title=title,
            scenarios=scenarios,
            ranked_scenario_ids=ranked_scenario_ids,
            trade_off_analysis=trade_off_analysis,
            dominance_map=dominance_map,
            risk_adjusted_scores=risk_adjusted_scores,
            opportunity_scores=opportunity_scores,
            recommended_for_review=recommended_for_review,
            rationale=rationale,
            confidence=confidence,
            created_at=datetime.utcnow(),
        )


@dataclass
class Forecast:
    """
    Probabilistic forecast produced by the Forecasting Engine.
    Forecasts are never certain; all forecasts carry uncertainty.
    """
    forecast_id: str
    title: str
    description: str
    forecast_type: ForecastType
    horizon: ForecastHorizon
    probability_distribution: ProbabilityDistribution
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    metadata: ForecastMetadata
    status: str                          # active | superseded | archived
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        forecast_type: ForecastType,
        horizon: ForecastHorizon,
        probability_distribution: ProbabilityDistribution,
        uncertainty: UncertaintyProfile,
        confidence: ConfidenceProfile,
        evidence: EvidenceProfile,
        metadata: ForecastMetadata,
    ) -> "Forecast":
        now = datetime.utcnow()
        return cls(
            forecast_id=str(uuid.uuid4()),
            title=title,
            description=description,
            forecast_type=forecast_type,
            horizon=horizon,
            probability_distribution=probability_distribution,
            uncertainty=uncertainty,
            confidence=confidence,
            evidence=evidence,
            metadata=metadata,
            status="active",
            created_at=now,
            updated_at=now,
        )


@dataclass
class ForecastAssessment:
    """
    Structured evaluation of a Forecast, including accuracy tracking
    and revision history.
    """
    assessment_id: str
    forecast: Forecast
    assessed_at: datetime
    accuracy_score: Optional[float]          # None if outcome not yet observable
    deviation_notes: str
    revision_required: bool
    superseded_by: Optional[str]             # forecast_id of the replacement
    assessor: str

    @classmethod
    def create(
        cls,
        forecast: Forecast,
        assessor: str,
        accuracy_score: Optional[float] = None,
        deviation_notes: str = "",
        revision_required: bool = False,
        superseded_by: Optional[str] = None,
    ) -> "ForecastAssessment":
        return cls(
            assessment_id=str(uuid.uuid4()),
            forecast=forecast,
            assessed_at=datetime.utcnow(),
            accuracy_score=accuracy_score,
            deviation_notes=deviation_notes,
            revision_required=revision_required,
            superseded_by=superseded_by,
            assessor=assessor,
        )


@dataclass
class SimulationOutcome:
    """A single outcome record from an Outcome Simulation run."""
    outcome_id: str
    label: str
    description: str
    probability: float                  # 0.0 – 1.0
    short_term_effects: list[str]
    medium_term_effects: list[str]
    long_term_effects: list[str]
    risk_implications: list[str]
    opportunity_implications: list[str]
    confidence: ConfidenceProfile

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")

    @classmethod
    def create(
        cls,
        label: str,
        description: str,
        probability: float,
        short_term_effects: list[str],
        medium_term_effects: list[str],
        long_term_effects: list[str],
        risk_implications: list[str],
        opportunity_implications: list[str],
        confidence: ConfidenceProfile,
    ) -> "SimulationOutcome":
        return cls(
            outcome_id=str(uuid.uuid4()),
            label=label,
            description=description,
            probability=probability,
            short_term_effects=short_term_effects,
            medium_term_effects=medium_term_effects,
            long_term_effects=long_term_effects,
            risk_implications=risk_implications,
            opportunity_implications=opportunity_implications,
            confidence=confidence,
        )


@dataclass
class OutcomeSimulation:
    """
    Full outcome simulation produced by the Outcome Simulation Engine.
    Models consequence chains from a given decision or event.
    """
    simulation_id: str
    title: str
    description: str
    triggering_event: str
    outcomes: list[SimulationOutcome]
    probability_distribution: ProbabilityDistribution
    uncertainty: UncertaintyProfile
    status: SimulationStatus
    metadata: SimulationMetadata
    created_at: datetime
    completed_at: Optional[datetime]

    @property
    def most_probable_outcome(self) -> Optional[SimulationOutcome]:
        if not self.outcomes:
            return None
        return max(self.outcomes, key=lambda o: o.probability)

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        triggering_event: str,
        outcomes: list[SimulationOutcome],
        probability_distribution: ProbabilityDistribution,
        uncertainty: UncertaintyProfile,
        metadata: SimulationMetadata,
    ) -> "OutcomeSimulation":
        return cls(
            simulation_id=str(uuid.uuid4()),
            title=title,
            description=description,
            triggering_event=triggering_event,
            outcomes=outcomes,
            probability_distribution=probability_distribution,
            uncertainty=uncertainty,
            status=SimulationStatus.PENDING,
            metadata=metadata,
            created_at=datetime.utcnow(),
            completed_at=None,
        )


@dataclass
class CounterfactualScenario:
    """
    Alternate-reality scenario produced by the Counterfactual Engine.
    Answers: 'What if this decision had been different?'
    """
    counterfactual_id: str
    title: str
    description: str
    original_event: str
    counterfactual_condition: str
    divergence_point: datetime
    resulting_future_model: FutureModel
    delta_risk: float                   # relative change vs original (-1 to +1)
    delta_opportunity: float            # relative change vs original (-1 to +1)
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    learning_insights: list[str]
    created_at: datetime

    def __post_init__(self) -> None:
        for attr in ("delta_risk", "delta_opportunity"):
            value = getattr(self, attr)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be in [-1, 1], got {value}")

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        original_event: str,
        counterfactual_condition: str,
        divergence_point: datetime,
        resulting_future_model: FutureModel,
        delta_risk: float,
        delta_opportunity: float,
        uncertainty: UncertaintyProfile,
        confidence: ConfidenceProfile,
        evidence: EvidenceProfile,
        learning_insights: Optional[list[str]] = None,
    ) -> "CounterfactualScenario":
        return cls(
            counterfactual_id=str(uuid.uuid4()),
            title=title,
            description=description,
            original_event=original_event,
            counterfactual_condition=counterfactual_condition,
            divergence_point=divergence_point,
            resulting_future_model=resulting_future_model,
            delta_risk=delta_risk,
            delta_opportunity=delta_opportunity,
            uncertainty=uncertainty,
            confidence=confidence,
            evidence=evidence,
            learning_insights=learning_insights or [],
            created_at=datetime.utcnow(),
        )


@dataclass
class TimelineProjection:
    """
    Future timeline projection produced by the Timeline Projection Engine.
    Projects future states and milestones across time.
    Constitutional rule: CHRONOS owns time; JANUS predicts future states across time.
    """
    projection_id: str
    title: str
    description: str
    context: str
    milestones: list[ProjectionMilestone]
    horizon: ForecastHorizon
    status: ProjectionStatus
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    created_at: datetime
    updated_at: datetime

    @property
    def critical_milestones(self) -> list[ProjectionMilestone]:
        return [m for m in self.milestones if m.is_critical]

    @property
    def completion_probability(self) -> float:
        critical = self.critical_milestones
        if not critical:
            return 1.0
        return min(m.probability for m in critical)

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        context: str,
        milestones: list[ProjectionMilestone],
        horizon: ForecastHorizon,
        uncertainty: UncertaintyProfile,
        confidence: ConfidenceProfile,
        evidence: EvidenceProfile,
    ) -> "TimelineProjection":
        now = datetime.utcnow()
        return cls(
            projection_id=str(uuid.uuid4()),
            title=title,
            description=description,
            context=context,
            milestones=milestones,
            horizon=horizon,
            status=ProjectionStatus.DRAFT,
            uncertainty=uncertainty,
            confidence=confidence,
            evidence=evidence,
            created_at=now,
            updated_at=now,
        )


@dataclass
class StrategicForecast:
    """
    Strategic-level forecast produced by the Strategic Forecast Engine.
    Constitutional rule: JANUS forecasts strategic outcomes; ODYSSEY chooses strategic direction.
    JANUS never selects strategy.
    """
    strategic_forecast_id: str
    title: str
    description: str
    domain: str
    trend_analysis: list[str]
    market_state_projections: list[FutureState]
    strategic_outcome_forecasts: list[Forecast]
    uncertainty: UncertaintyProfile
    confidence: ConfidenceProfile
    evidence: EvidenceProfile
    horizon: ForecastHorizon
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        domain: str,
        trend_analysis: list[str],
        market_state_projections: list[FutureState],
        strategic_outcome_forecasts: list[Forecast],
        uncertainty: UncertaintyProfile,
        confidence: ConfidenceProfile,
        evidence: EvidenceProfile,
        horizon: ForecastHorizon,
    ) -> "StrategicForecast":
        now = datetime.utcnow()
        return cls(
            strategic_forecast_id=str(uuid.uuid4()),
            title=title,
            description=description,
            domain=domain,
            trend_analysis=trend_analysis,
            market_state_projections=market_state_projections,
            strategic_outcome_forecasts=strategic_outcome_forecasts,
            uncertainty=uncertainty,
            confidence=confidence,
            evidence=evidence,
            horizon=horizon,
            created_at=now,
            updated_at=now,
        )


@dataclass
class FutureAssessment:
    """
    Comprehensive future assessment — master output of JANUS.
    Aggregates scenarios, forecasts, simulations, risks, and opportunities
    into a single deliverable package for ODYSSEY, VEGA, ZENITH, PROMETHEUS, DRACO.
    JANUS evaluates futures; it never approves them. Approval belongs to VEGA.
    """
    assessment_id: str
    title: str
    description: str
    context: str
    status: FutureAssessmentStatus
    scenarios: list[Scenario]
    scenario_comparison: Optional[ScenarioComparison]
    forecasts: list[Forecast]
    simulations: list[OutcomeSimulation]
    risk_assessments: list[RiskAssessment]
    opportunity_assessments: list[OpportunityAssessment]
    timeline_projections: list[TimelineProjection]
    counterfactuals: list[CounterfactualScenario]
    strategic_forecasts: list[StrategicForecast]
    overall_uncertainty: UncertaintyProfile
    overall_confidence: ConfidenceProfile
    summary: str
    created_at: datetime
    updated_at: datetime

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    @property
    def has_critical_risks(self) -> bool:
        return any(r.level == RiskLevel.CRITICAL for r in self.risk_assessments)

    @property
    def has_transformative_opportunities(self) -> bool:
        return any(
            o.level == OpportunityLevel.TRANSFORMATIVE for o in self.opportunity_assessments
        )

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        context: str,
        scenarios: list[Scenario],
        forecasts: list[Forecast],
        simulations: list[OutcomeSimulation],
        risk_assessments: list[RiskAssessment],
        opportunity_assessments: list[OpportunityAssessment],
        timeline_projections: list[TimelineProjection],
        counterfactuals: list[CounterfactualScenario],
        strategic_forecasts: list[StrategicForecast],
        overall_uncertainty: UncertaintyProfile,
        overall_confidence: ConfidenceProfile,
        summary: str,
        scenario_comparison: Optional[ScenarioComparison] = None,
    ) -> "FutureAssessment":
        now = datetime.utcnow()
        return cls(
            assessment_id=str(uuid.uuid4()),
            title=title,
            description=description,
            context=context,
            status=FutureAssessmentStatus.PENDING,
            scenarios=scenarios,
            scenario_comparison=scenario_comparison,
            forecasts=forecasts,
            simulations=simulations,
            risk_assessments=risk_assessments,
            opportunity_assessments=opportunity_assessments,
            timeline_projections=timeline_projections,
            counterfactuals=counterfactuals,
            strategic_forecasts=strategic_forecasts,
            overall_uncertainty=overall_uncertainty,
            overall_confidence=overall_confidence,
            summary=summary,
            created_at=now,
            updated_at=now,
        )