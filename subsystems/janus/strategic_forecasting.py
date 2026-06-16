"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/strategic_forecasting.py

Implementation of the Strategic Forecast Engine.

Responsibility: Forecast strategic-level outcomes from trend analysis and
market state projections.

Constitutional rule:
    JANUS forecasts strategic outcomes.
    ODYSSEY chooses strategic direction.
    This engine never selects strategy.

JANUS Law 6:
    All forecasts require uncertainty. No certainty claims. Ever.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .models import (
    Forecast,
    ForecastHorizon,
    FutureState,
    ProbabilityLevel,
    StrategicForecast,
    UncertaintyProfile,
)
from .interfaces import IStrategicForecastEngine
from .schemas import (
    StrategicForecastCreateRequest,
    StrategicForecastCreateResponse,
    StrategicForecastGetRequest,
    StrategicForecastGetResponse,
    StrategicForecastListRequest,
    StrategicForecastListResponse,
)
from .exceptions import (
    JanusNotInitializedError,
    JanusAlreadyInitializedError,
    JanusShutdownError,
    JanusMissingRequiredFieldError,
    JanusInvalidProbabilityError,
    JanusStrategicForecastNotFoundError,
    JanusStrategicForecastConstructionError,
    JanusStrategySelectionViolationError,
    JanusMissingUncertaintyError,
)


# ---------------------------------------------------------------------------
# Engine Identity
# ---------------------------------------------------------------------------

_ENGINE_NAME: str = "StrategicForecastEngine"
_ENGINE_VERSION: str = "5.1.0"

# Horizons considered "long-horizon" for the purposes of long-horizon
# forecasting analysis.
_LONG_HORIZONS: frozenset[ForecastHorizon] = frozenset(
    {ForecastHorizon.FIVE_YEARS, ForecastHorizon.TEN_YEARS}
)

# Operations that would constitute selecting/choosing a strategic direction.
# JANUS forecasts strategic outcomes; ODYSSEY chooses strategic direction.
_RESTRICTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "select_strategy",
        "choose_direction",
        "set_strategic_direction",
        "approve_strategy",
        "pick_strategy",
        "decide_strategy",
    }
)


# ---------------------------------------------------------------------------
# Diagnostics / Health / Statistics Value Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategicForecastHealthReport:
    """Health snapshot for the Strategic Forecast Engine."""

    engine_name: str
    engine_version: str
    is_initialized: bool
    is_shutdown: bool
    total_forecasts: int
    domains_tracked: int
    horizons_tracked: int
    healthy: bool
    generated_at: datetime


@dataclass(frozen=True)
class StrategicForecastDiagnosticsReport:
    """Diagnostics snapshot covering integrity of stored StrategicForecasts."""

    engine_name: str
    engine_version: str
    total_forecasts: int
    forecasts_missing_trend_analysis: int
    forecasts_missing_market_projections: int
    forecasts_missing_outcome_forecasts: int
    forecasts_with_zero_uncertainty: int
    average_trend_count: float
    average_market_projection_count: float
    average_outcome_forecast_count: float
    domain_distribution: dict[str, int]
    horizon_distribution: dict[str, int]
    generated_at: datetime


@dataclass(frozen=True)
class StrategicForecastStatistics:
    """Aggregate statistics across all registered StrategicForecasts."""

    total_forecasts: int
    by_domain: dict[str, int]
    by_horizon: dict[str, int]
    total_trend_entries: int
    total_market_state_projections: int
    total_outcome_forecasts: int
    average_confidence: Optional[float]
    average_uncertainty_volatility: Optional[float]
    average_market_projection_probability: Optional[float]
    generated_at: datetime

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class StrategicForecastComparisonResult:
    """Structured comparison between two StrategicForecasts."""

    strategic_forecast_id_a: str
    strategic_forecast_id_b: str
    domain_a: str
    domain_b: str
    horizon_a: str
    horizon_b: str
    confidence_a: float
    confidence_b: float
    confidence_delta: float
    shared_trend_themes: tuple[str, ...]
    unique_trends_a: tuple[str, ...]
    unique_trends_b: tuple[str, ...]
    outcome_forecast_count_a: int
    outcome_forecast_count_b: int
    compared_at: datetime


@dataclass(frozen=True)
class StrategicForecastRanking:
    """A single ranked entry within a StrategicForecast ranking result."""

    strategic_forecast_id: str
    title: str
    domain: str
    horizon: str
    composite_score: float
    confidence_component: float
    opportunity_component: float
    risk_component: float


@dataclass(frozen=True)
class StrategicForecastRankingResult:
    """Result of ranking a set of StrategicForecasts."""

    ranked: tuple[StrategicForecastRanking, ...]
    weight_opportunity: float
    weight_risk: float
    weight_confidence: float
    ranked_at: datetime


@dataclass(frozen=True)
class StrategicForecastEvaluation:
    """
    Structured evaluation of a single StrategicForecast.

    This is an evaluation, not a selection or approval. JANUS forecasts
    strategic outcomes; ODYSSEY chooses strategic direction.
    """

    strategic_forecast_id: str
    title: str
    domain: str
    horizon: str
    trend_count: int
    market_projection_count: int
    outcome_forecast_count: int
    average_market_probability: Optional[float]
    average_market_probability_level: Optional[str]
    confidence_level: str
    uncertainty_volatility: float
    is_long_horizon: bool
    notable_trends: tuple[str, ...]
    evaluated_at: datetime


@dataclass(frozen=True)
class TrendAnalysisResult:
    """Result of analyzing trends across a set of StrategicForecasts."""

    domain: Optional[str]
    horizon: Optional[str]
    forecast_count: int
    total_trend_entries: int
    distinct_trend_themes: tuple[str, ...]
    recurring_trend_themes: tuple[str, ...]
    analyzed_at: datetime


@dataclass(frozen=True)
class OpportunityForecastResult:
    """
    Forecast of strategic opportunity exposure derived from market state
    projections within a StrategicForecast.
    """

    strategic_forecast_id: str
    domain: str
    horizon: str
    opportunity_signal_count: int
    average_opportunity_score: Optional[float]
    transformative_signal_count: int
    forecasted_at: datetime


@dataclass(frozen=True)
class RiskForecastResult:
    """
    Forecast of strategic risk exposure derived from risk assessments
    embedded within market state projections of a StrategicForecast.
    """

    strategic_forecast_id: str
    domain: str
    horizon: str
    risk_signal_count: int
    average_risk_score: Optional[float]
    critical_signal_count: int
    forecasted_at: datetime


# ---------------------------------------------------------------------------
# Strategic Forecast Engine
# ---------------------------------------------------------------------------


class StrategicForecastEngine(IStrategicForecastEngine):
    """
    Production implementation of the Strategic Forecast Engine.

    Owns:
        - StrategicForecast creation, retrieval, listing, and revision.
        - Long-horizon and multi-scenario strategic forecasting analysis.
        - Strategic trend analysis, opportunity/risk forecasting.
        - StrategicForecast comparison, ranking, and evaluation.

    Never owns:
        - Strategy selection or strategic direction (ODYSSEY).
        - Decisions (VEGA).

    Thread-safety:
        All mutable state is guarded by a single re-entrant lock. Stored
        StrategicForecast instances are treated as immutable; revisions
        produce new StrategicForecast instances replacing the prior
        registry entry.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._initialized: bool = False
        self._shutdown: bool = False
        self._forecasts: dict[str, StrategicForecast] = {}

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._shutdown:
                raise JanusShutdownError(_ENGINE_NAME)
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._forecasts = {}
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

    def _guard_strategy_selection_boundary(self, operation: str) -> None:
        """
        Raise JanusStrategySelectionViolationError if the requested
        operation would imply selecting or approving a strategic direction.

        JANUS forecasts strategic outcomes; ODYSSEY chooses strategic
        direction. JANUS never selects strategy.
        """
        if operation.lower() in _RESTRICTED_OPERATIONS:
            raise JanusStrategySelectionViolationError(operation)

    # -----------------------------------------------------------------
    # Validation Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_create_request(request: StrategicForecastCreateRequest) -> None:
        if not request.title.strip():
            raise JanusMissingRequiredFieldError("title", engine=_ENGINE_NAME)
        if not request.description.strip():
            raise JanusMissingRequiredFieldError("description", engine=_ENGINE_NAME)
        if not request.domain.strip():
            raise JanusMissingRequiredFieldError("domain", engine=_ENGINE_NAME)
        if not request.trend_analysis:
            raise JanusMissingRequiredFieldError(
                "trend_analysis", engine=_ENGINE_NAME
            )

        for state in request.market_state_projections:
            if not 0.0 <= state.probability <= 1.0:
                raise JanusInvalidProbabilityError(
                    state.probability,
                    field=f"market_state_projections[{state.state_id}].probability",
                    engine=_ENGINE_NAME,
                )

    @staticmethod
    def _validate_uncertainty(
        forecast_id: str, uncertainty: Optional[UncertaintyProfile]
    ) -> None:
        if uncertainty is None:
            raise JanusMissingUncertaintyError(forecast_id, "StrategicForecast")

    # -----------------------------------------------------------------
    # Core Interface: Create
    # -----------------------------------------------------------------

    def create_strategic_forecast(
        self, request: StrategicForecastCreateRequest
    ) -> StrategicForecastCreateResponse:
        with self._lock:
            self._ensure_operational()

            try:
                self._validate_create_request(request)
            except JanusMissingRequiredFieldError as exc:
                raise JanusStrategicForecastConstructionError(
                    str(exc), context={"title": request.title}
                ) from exc
            except JanusInvalidProbabilityError as exc:
                raise JanusStrategicForecastConstructionError(
                    str(exc), context={"title": request.title}
                ) from exc

            self._validate_uncertainty("<pending>", request.uncertainty)

            try:
                strategic_forecast = StrategicForecast.create(
                    title=request.title,
                    description=request.description,
                    domain=request.domain,
                    trend_analysis=list(request.trend_analysis),
                    market_state_projections=list(request.market_state_projections),
                    strategic_outcome_forecasts=list(
                        request.strategic_outcome_forecasts
                    ),
                    uncertainty=request.uncertainty,
                    confidence=request.confidence,
                    evidence=request.evidence,
                    horizon=request.horizon,
                )
            except ValueError as exc:
                raise JanusStrategicForecastConstructionError(
                    str(exc), context={"title": request.title}
                ) from exc

            self._forecasts[strategic_forecast.strategic_forecast_id] = (
                strategic_forecast
            )

            return StrategicForecastCreateResponse(
                strategic_forecast=strategic_forecast,
                created_at=strategic_forecast.created_at,
                engine_version=_ENGINE_VERSION,
            )

    # -----------------------------------------------------------------
    # Core Interface: Get
    # -----------------------------------------------------------------

    def get_strategic_forecast(
        self, request: StrategicForecastGetRequest
    ) -> StrategicForecastGetResponse:
        with self._lock:
            self._ensure_operational()

            forecast = self._forecasts.get(request.strategic_forecast_id)
            if forecast is None:
                raise JanusStrategicForecastNotFoundError(
                    request.strategic_forecast_id
                )

            return StrategicForecastGetResponse(
                strategic_forecast=forecast,
                retrieved_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Core Interface: List
    # -----------------------------------------------------------------

    def list_strategic_forecasts(
        self, request: StrategicForecastListRequest
    ) -> StrategicForecastListResponse:
        with self._lock:
            self._ensure_operational()

            results = list(self._forecasts.values())

            if request.domain is not None:
                results = [f for f in results if f.domain == request.domain]

            if request.horizon is not None:
                results = [f for f in results if f.horizon == request.horizon]

            return StrategicForecastListResponse(
                strategic_forecasts=tuple(results),
                retrieved_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Core Interface: Update / Revise
    # -----------------------------------------------------------------

    def update_strategic_forecast(
        self,
        strategic_forecast_id: str,
        updated_trend_analysis: tuple[str, ...],
        updated_market_projections: tuple[FutureState, ...],
        updated_outcome_forecasts: tuple[Forecast, ...],
        updated_uncertainty: UncertaintyProfile,
        updated_evidence,
        reason: str,
    ) -> StrategicForecast:
        with self._lock:
            self._ensure_operational()

            forecast = self._forecasts.get(strategic_forecast_id)
            if forecast is None:
                raise JanusStrategicForecastNotFoundError(strategic_forecast_id)

            if not reason.strip():
                raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)

            if not updated_trend_analysis:
                raise JanusStrategicForecastConstructionError(
                    "updated_trend_analysis must contain at least one entry.",
                    context={"strategic_forecast_id": strategic_forecast_id},
                )

            for state in updated_market_projections:
                if not 0.0 <= state.probability <= 1.0:
                    raise JanusInvalidProbabilityError(
                        state.probability,
                        field=f"updated_market_projections[{state.state_id}].probability",
                        engine=_ENGINE_NAME,
                    )

            self._validate_uncertainty(strategic_forecast_id, updated_uncertainty)

            if updated_evidence is None:
                raise JanusMissingRequiredFieldError(
                    "updated_evidence", engine=_ENGINE_NAME
                )

            updated = StrategicForecast(
                strategic_forecast_id=forecast.strategic_forecast_id,
                title=forecast.title,
                description=forecast.description,
                domain=forecast.domain,
                trend_analysis=list(updated_trend_analysis),
                market_state_projections=list(updated_market_projections),
                strategic_outcome_forecasts=list(updated_outcome_forecasts),
                uncertainty=updated_uncertainty,
                confidence=forecast.confidence,
                evidence=updated_evidence,
                horizon=forecast.horizon,
                created_at=forecast.created_at,
                updated_at=_utcnow(),
            )

            self._forecasts[strategic_forecast_id] = updated
            return updated

    # -----------------------------------------------------------------
    # Core Interface: List by Domain
    # -----------------------------------------------------------------

    def list_strategic_forecasts_by_domain(
        self, domain: str
    ) -> tuple[StrategicForecast, ...]:
        with self._lock:
            self._ensure_operational()

            if not domain.strip():
                raise JanusMissingRequiredFieldError("domain", engine=_ENGINE_NAME)

            return tuple(
                f for f in self._forecasts.values() if f.domain == domain
            )

    # -----------------------------------------------------------------
    # Long-Horizon Forecasting
    # -----------------------------------------------------------------

    def list_long_horizon_forecasts(
        self, domain: Optional[str] = None
    ) -> tuple[StrategicForecast, ...]:
        """
        Return all StrategicForecasts at FIVE_YEARS or TEN_YEARS horizons,
        optionally filtered by domain. Supports strategic foresight over
        extended time horizons.
        """
        with self._lock:
            self._ensure_operational()

            results = [
                f
                for f in self._forecasts.values()
                if f.horizon in _LONG_HORIZONS
            ]

            if domain is not None:
                results = [f for f in results if f.domain == domain]

            return tuple(results)

    # -----------------------------------------------------------------
    # Multi-Scenario Forecasting
    # -----------------------------------------------------------------

    def aggregate_multi_scenario_forecast(
        self, strategic_forecast_ids: tuple[str, ...]
    ) -> dict[str, object]:
        """
        Aggregate multiple StrategicForecasts (e.g., representing distinct
        scenarios for the same domain/horizon) into a single consolidated
        view: combined trend themes, pooled market projections, and pooled
        outcome forecasts.

        Does not select or approve any single scenario — JANUS forecasts
        strategic outcomes; ODYSSEY chooses strategic direction.
        """
        with self._lock:
            self._ensure_operational()

            if not strategic_forecast_ids:
                raise JanusMissingRequiredFieldError(
                    "strategic_forecast_ids", engine=_ENGINE_NAME
                )

            forecasts: list[StrategicForecast] = []
            for forecast_id in strategic_forecast_ids:
                forecast = self._forecasts.get(forecast_id)
                if forecast is None:
                    raise JanusStrategicForecastNotFoundError(forecast_id)
                forecasts.append(forecast)

            combined_trends: set[str] = set()
            total_market_projections = 0
            total_outcome_forecasts = 0
            domains: set[str] = set()
            horizons: set[str] = set()

            for forecast in forecasts:
                combined_trends.update(forecast.trend_analysis)
                total_market_projections += len(forecast.market_state_projections)
                total_outcome_forecasts += len(forecast.strategic_outcome_forecasts)
                domains.add(forecast.domain)
                horizons.add(forecast.horizon.value)

            confidences = [f.confidence.overall for f in forecasts]
            avg_confidence = statistics.fmean(confidences) if confidences else 0.0

            return {
                "scenario_count": len(forecasts),
                "strategic_forecast_ids": tuple(strategic_forecast_ids),
                "domains": tuple(sorted(domains)),
                "horizons": tuple(sorted(horizons)),
                "combined_trend_themes": tuple(sorted(combined_trends)),
                "total_market_state_projections": total_market_projections,
                "total_outcome_forecasts": total_outcome_forecasts,
                "average_confidence": avg_confidence,
                "aggregated_at": _utcnow(),
            }

    # -----------------------------------------------------------------
    # Forecast Comparison
    # -----------------------------------------------------------------

    def compare_strategic_forecasts(
        self, strategic_forecast_id_a: str, strategic_forecast_id_b: str
    ) -> StrategicForecastComparisonResult:
        """Produce a structured comparison between two StrategicForecasts."""
        with self._lock:
            self._ensure_operational()

            forecast_a = self._forecasts.get(strategic_forecast_id_a)
            if forecast_a is None:
                raise JanusStrategicForecastNotFoundError(strategic_forecast_id_a)

            forecast_b = self._forecasts.get(strategic_forecast_id_b)
            if forecast_b is None:
                raise JanusStrategicForecastNotFoundError(strategic_forecast_id_b)

            trends_a = set(forecast_a.trend_analysis)
            trends_b = set(forecast_b.trend_analysis)

            shared = tuple(sorted(trends_a & trends_b))
            unique_a = tuple(sorted(trends_a - trends_b))
            unique_b = tuple(sorted(trends_b - trends_a))

            return StrategicForecastComparisonResult(
                strategic_forecast_id_a=forecast_a.strategic_forecast_id,
                strategic_forecast_id_b=forecast_b.strategic_forecast_id,
                domain_a=forecast_a.domain,
                domain_b=forecast_b.domain,
                horizon_a=forecast_a.horizon.value,
                horizon_b=forecast_b.horizon.value,
                confidence_a=forecast_a.confidence.overall,
                confidence_b=forecast_b.confidence.overall,
                confidence_delta=(
                    forecast_b.confidence.overall - forecast_a.confidence.overall
                ),
                shared_trend_themes=shared,
                unique_trends_a=unique_a,
                unique_trends_b=unique_b,
                outcome_forecast_count_a=len(forecast_a.strategic_outcome_forecasts),
                outcome_forecast_count_b=len(forecast_b.strategic_outcome_forecasts),
                compared_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Strategic Trend Analysis
    # -----------------------------------------------------------------

    def analyze_strategic_trends(
        self,
        domain: Optional[str] = None,
        horizon: Optional[ForecastHorizon] = None,
    ) -> TrendAnalysisResult:
        """
        Analyze trend themes across all StrategicForecasts, optionally
        filtered by domain and/or horizon. Identifies recurring trend
        themes (appearing in more than one StrategicForecast).
        """
        with self._lock:
            self._ensure_operational()

            forecasts = list(self._forecasts.values())

            if domain is not None:
                forecasts = [f for f in forecasts if f.domain == domain]

            if horizon is not None:
                forecasts = [f for f in forecasts if f.horizon == horizon]

            total_entries = sum(len(f.trend_analysis) for f in forecasts)

            occurrence_count: dict[str, int] = {}
            for forecast in forecasts:
                for trend in set(forecast.trend_analysis):
                    occurrence_count[trend] = occurrence_count.get(trend, 0) + 1

            distinct = tuple(sorted(occurrence_count.keys()))
            recurring = tuple(
                sorted(
                    trend for trend, count in occurrence_count.items() if count > 1
                )
            )

            return TrendAnalysisResult(
                domain=domain,
                horizon=horizon.value if horizon is not None else None,
                forecast_count=len(forecasts),
                total_trend_entries=total_entries,
                distinct_trend_themes=distinct,
                recurring_trend_themes=recurring,
                analyzed_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Opportunity Forecasting
    # -----------------------------------------------------------------

    def forecast_opportunities(
        self, strategic_forecast_id: str
    ) -> OpportunityForecastResult:
        """
        Forecast strategic opportunity exposure for a StrategicForecast by
        aggregating OpportunityAssessments embedded within its market state
        projections.

        Feeds ODYSSEY and VEGA. Identifies opportunity signals; does not
        select or capture opportunities.
        """
        with self._lock:
            self._ensure_operational()

            forecast = self._forecasts.get(strategic_forecast_id)
            if forecast is None:
                raise JanusStrategicForecastNotFoundError(strategic_forecast_id)

            opportunity_scores: list[float] = []
            transformative_count = 0

            for state in forecast.market_state_projections:
                for opportunity in state.opportunities:
                    opportunity_scores.append(opportunity.composite_opportunity_score)
                    if opportunity.level.name == "TRANSFORMATIVE":
                        transformative_count += 1

            avg_score = (
                statistics.fmean(opportunity_scores) if opportunity_scores else None
            )

            return OpportunityForecastResult(
                strategic_forecast_id=forecast.strategic_forecast_id,
                domain=forecast.domain,
                horizon=forecast.horizon.value,
                opportunity_signal_count=len(opportunity_scores),
                average_opportunity_score=avg_score,
                transformative_signal_count=transformative_count,
                forecasted_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Risk Forecasting
    # -----------------------------------------------------------------

    def forecast_risks(self, strategic_forecast_id: str) -> RiskForecastResult:
        """
        Forecast strategic risk exposure for a StrategicForecast by
        aggregating RiskAssessments embedded within its market state
        projections.

        Focuses on future risk signals; does not select mitigations.
        """
        with self._lock:
            self._ensure_operational()

            forecast = self._forecasts.get(strategic_forecast_id)
            if forecast is None:
                raise JanusStrategicForecastNotFoundError(strategic_forecast_id)

            risk_scores: list[float] = []
            critical_count = 0

            for state in forecast.market_state_projections:
                for risk in state.risks:
                    risk_scores.append(risk.composite_risk_score)
                    if risk.level.name == "CRITICAL":
                        critical_count += 1

            avg_score = statistics.fmean(risk_scores) if risk_scores else None

            return RiskForecastResult(
                strategic_forecast_id=forecast.strategic_forecast_id,
                domain=forecast.domain,
                horizon=forecast.horizon.value,
                risk_signal_count=len(risk_scores),
                average_risk_score=avg_score,
                critical_signal_count=critical_count,
                forecasted_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Forecast Ranking
    # -----------------------------------------------------------------

    def rank_strategic_forecasts(
        self,
        strategic_forecast_ids: tuple[str, ...],
        weight_opportunity: float = 0.4,
        weight_risk: float = 0.4,
        weight_confidence: float = 0.2,
    ) -> StrategicForecastRankingResult:
        """
        Rank a set of StrategicForecasts by a composite score combining
        opportunity exposure, risk exposure (inverted), and forecast
        confidence.

        This produces a ranking for review; it is not a selection.
        JANUS evaluates and forecasts; ODYSSEY chooses direction and
        VEGA decides.
        """
        with self._lock:
            self._ensure_operational()
            self._guard_strategy_selection_boundary("rank_strategic_forecasts")

            if not strategic_forecast_ids:
                raise JanusMissingRequiredFieldError(
                    "strategic_forecast_ids", engine=_ENGINE_NAME
                )

            total_weight = weight_opportunity + weight_risk + weight_confidence
            if total_weight <= 0.0:
                raise JanusStrategicForecastConstructionError(
                    "Ranking weights must sum to a positive value.",
                    context={
                        "weight_opportunity": weight_opportunity,
                        "weight_risk": weight_risk,
                        "weight_confidence": weight_confidence,
                    },
                )

            rankings: list[StrategicForecastRanking] = []

            for forecast_id in strategic_forecast_ids:
                forecast = self._forecasts.get(forecast_id)
                if forecast is None:
                    raise JanusStrategicForecastNotFoundError(forecast_id)

                opportunity_result = self.forecast_opportunities(forecast_id)
                risk_result = self.forecast_risks(forecast_id)

                opportunity_component = (
                    opportunity_result.average_opportunity_score or 0.0
                )
                risk_component = risk_result.average_risk_score or 0.0
                confidence_component = forecast.confidence.overall

                composite = (
                    (weight_opportunity * opportunity_component)
                    - (weight_risk * risk_component)
                    + (weight_confidence * confidence_component)
                ) / total_weight

                rankings.append(
                    StrategicForecastRanking(
                        strategic_forecast_id=forecast.strategic_forecast_id,
                        title=forecast.title,
                        domain=forecast.domain,
                        horizon=forecast.horizon.value,
                        composite_score=composite,
                        confidence_component=confidence_component,
                        opportunity_component=opportunity_component,
                        risk_component=risk_component,
                    )
                )

            rankings.sort(key=lambda r: r.composite_score, reverse=True)

            return StrategicForecastRankingResult(
                ranked=tuple(rankings),
                weight_opportunity=weight_opportunity,
                weight_risk=weight_risk,
                weight_confidence=weight_confidence,
                ranked_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Forecast Evaluation
    # -----------------------------------------------------------------

    def evaluate_strategic_forecast(
        self, strategic_forecast_id: str
    ) -> StrategicForecastEvaluation:
        """
        Produce a structured evaluation of a single StrategicForecast.

        This is an evaluation for review, not an approval or selection.
        """
        with self._lock:
            self._ensure_operational()

            forecast = self._forecasts.get(strategic_forecast_id)
            if forecast is None:
                raise JanusStrategicForecastNotFoundError(strategic_forecast_id)

            market_probabilities = [
                state.probability for state in forecast.market_state_projections
            ]
            avg_market_probability = (
                statistics.fmean(market_probabilities)
                if market_probabilities
                else None
            )
            avg_market_probability_level = (
                ProbabilityLevel.from_float(avg_market_probability).value
                if avg_market_probability is not None
                else None
            )

            notable_trends = tuple(forecast.trend_analysis[:5])

            return StrategicForecastEvaluation(
                strategic_forecast_id=forecast.strategic_forecast_id,
                title=forecast.title,
                domain=forecast.domain,
                horizon=forecast.horizon.value,
                trend_count=len(forecast.trend_analysis),
                market_projection_count=len(forecast.market_state_projections),
                outcome_forecast_count=len(forecast.strategic_outcome_forecasts),
                average_market_probability=avg_market_probability,
                average_market_probability_level=avg_market_probability_level,
                confidence_level=forecast.confidence.level.value,
                uncertainty_volatility=forecast.uncertainty.volatility_score,
                is_long_horizon=forecast.horizon in _LONG_HORIZONS,
                notable_trends=notable_trends,
                evaluated_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def get_statistics(self) -> StrategicForecastStatistics:
        """Return aggregate statistics across all registered forecasts."""
        with self._lock:
            self._ensure_operational()

            forecasts = list(self._forecasts.values())

            by_domain: dict[str, int] = {}
            for forecast in forecasts:
                by_domain[forecast.domain] = by_domain.get(forecast.domain, 0) + 1

            by_horizon: dict[str, int] = {}
            for horizon in ForecastHorizon:
                by_horizon[horizon.value] = sum(
                    1 for f in forecasts if f.horizon == horizon
                )

            total_trend_entries = sum(len(f.trend_analysis) for f in forecasts)
            total_market_projections = sum(
                len(f.market_state_projections) for f in forecasts
            )
            total_outcome_forecasts = sum(
                len(f.strategic_outcome_forecasts) for f in forecasts
            )

            confidences = [f.confidence.overall for f in forecasts]
            avg_confidence = statistics.fmean(confidences) if confidences else None

            volatilities = [f.uncertainty.volatility_score for f in forecasts]
            avg_volatility = (
                statistics.fmean(volatilities) if volatilities else None
            )

            all_market_probabilities = [
                state.probability
                for f in forecasts
                for state in f.market_state_projections
            ]
            avg_market_probability = (
                statistics.fmean(all_market_probabilities)
                if all_market_probabilities
                else None
            )

            return StrategicForecastStatistics(
                total_forecasts=len(forecasts),
                by_domain=by_domain,
                by_horizon=by_horizon,
                total_trend_entries=total_trend_entries,
                total_market_state_projections=total_market_projections,
                total_outcome_forecasts=total_outcome_forecasts,
                average_confidence=avg_confidence,
                average_uncertainty_volatility=avg_volatility,
                average_market_projection_probability=avg_market_probability,
                generated_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Health Report
    # -----------------------------------------------------------------

    def check_health(self) -> StrategicForecastHealthReport:
        """Return a health snapshot of the engine and its registry."""
        with self._lock:
            forecasts = list(self._forecasts.values())

            domains = {f.domain for f in forecasts}
            horizons = {f.horizon for f in forecasts}

            healthy = self._initialized and not self._shutdown

            return StrategicForecastHealthReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                is_initialized=self._initialized,
                is_shutdown=self._shutdown,
                total_forecasts=len(forecasts),
                domains_tracked=len(domains),
                horizons_tracked=len(horizons),
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
            "total_forecasts": report.total_forecasts,
            "domains_tracked": report.domains_tracked,
            "horizons_tracked": report.horizons_tracked,
            "healthy": report.healthy,
            "generated_at": report.generated_at,
        }

    # -----------------------------------------------------------------
    # Diagnostics Report
    # -----------------------------------------------------------------

    def run_diagnostics(self) -> StrategicForecastDiagnosticsReport:
        """
        Run a diagnostics pass over all registered StrategicForecasts,
        flagging integrity concerns without mutating any state.
        """
        with self._lock:
            self._ensure_operational()

            forecasts = list(self._forecasts.values())

            missing_trends = sum(
                1 for f in forecasts if not f.trend_analysis
            )
            missing_market_projections = sum(
                1 for f in forecasts if not f.market_state_projections
            )
            missing_outcome_forecasts = sum(
                1 for f in forecasts if not f.strategic_outcome_forecasts
            )
            zero_uncertainty = sum(
                1
                for f in forecasts
                if f.uncertainty.volatility_score == 0.0
                and f.uncertainty.unknown_risk_exposure == 0.0
            )

            trend_counts = [len(f.trend_analysis) for f in forecasts]
            avg_trend_count = (
                statistics.fmean(trend_counts) if trend_counts else 0.0
            )

            market_counts = [len(f.market_state_projections) for f in forecasts]
            avg_market_count = (
                statistics.fmean(market_counts) if market_counts else 0.0
            )

            outcome_counts = [
                len(f.strategic_outcome_forecasts) for f in forecasts
            ]
            avg_outcome_count = (
                statistics.fmean(outcome_counts) if outcome_counts else 0.0
            )

            domain_distribution: dict[str, int] = {}
            for f in forecasts:
                domain_distribution[f.domain] = (
                    domain_distribution.get(f.domain, 0) + 1
                )

            horizon_distribution: dict[str, int] = {}
            for f in forecasts:
                horizon_distribution[f.horizon.value] = (
                    horizon_distribution.get(f.horizon.value, 0) + 1
                )

            return StrategicForecastDiagnosticsReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                total_forecasts=len(forecasts),
                forecasts_missing_trend_analysis=missing_trends,
                forecasts_missing_market_projections=missing_market_projections,
                forecasts_missing_outcome_forecasts=missing_outcome_forecasts,
                forecasts_with_zero_uncertainty=zero_uncertainty,
                average_trend_count=avg_trend_count,
                average_market_projection_count=avg_market_count,
                average_outcome_forecast_count=avg_outcome_count,
                domain_distribution=domain_distribution,
                horizon_distribution=horizon_distribution,
                generated_at=_utcnow(),
            )


# ---------------------------------------------------------------------------
# Module-level Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)