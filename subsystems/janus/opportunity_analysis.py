"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/opportunity_analysis.py

Future Opportunity Engine — IFutureOpportunityEngine implementation.

Responsibility:
    Identify and assess future opportunities across emerging technologies,
    industry growth, research breakthroughs, and startup openings.
    Feeds ODYSSEY and VEGA with structured opportunity intelligence.

Constitutional rule:
    JANUS identifies and evaluates future opportunities.
    It never decides whether to pursue them.
    VEGA decides which opportunities to pursue.
    ZENITH plans how to capture them.

JANUS owns future opportunity analysis exclusively.
VEGA owns opportunity-pursuit decisions.
ZENITH owns capture planning.
"""

from __future__ import annotations

import logging
import statistics
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .exceptions import (
    JanusAlreadyInitializedError,
    JanusInvalidScoreError,
    JanusNotInitializedError,
    JanusOpportunityAnalysisError,
    JanusOpportunityAssessmentNotFoundError,
    JanusOpportunityFactorError,
    JanusOpportunityLevelError,
    JanusShutdownError,
    JanusValidationError,
)
from .interfaces import IFutureOpportunityEngine
from .models import (
    ConfidenceProfile,
    EvidenceProfile,
    ForecastHorizon,
    OpportunityAssessment,
    OpportunityFactor,
    OpportunityLevel,
    ProbabilityDistribution,
    UncertaintyProfile,
)
from .schemas import (
    OpportunityAnalysisRequest,
    OpportunityAnalysisResponse,
    OpportunityAssessmentGetRequest,
    OpportunityAssessmentGetResponse,
    OpportunityListByLevelRequest,
    OpportunityListByLevelResponse,
)

_LOG = logging.getLogger(__name__)

_ENGINE_NAME: str = "FutureOpportunityEngine"
_ENGINE_VERSION: str = "5.1.0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpportunityAnalysisEngineConfig:
    """
    Tunable thresholds for the Future Opportunity Engine.

    Thresholds must be strictly increasing and lie within (0.0, 1.0).
    A composite opportunity score at or below `marginal_threshold` maps to
    OpportunityLevel.MARGINAL; above `high_threshold` maps to
    OpportunityLevel.TRANSFORMATIVE.
    """

    marginal_threshold: float = 0.10
    low_threshold: float = 0.30
    moderate_threshold: float = 0.55
    high_threshold: float = 0.80

    trend_stability_tolerance: float = 0.05

    def __post_init__(self) -> None:
        thresholds = (
            self.marginal_threshold,
            self.low_threshold,
            self.moderate_threshold,
            self.high_threshold,
        )
        for value in thresholds:
            if not 0.0 < value < 1.0:
                raise JanusValidationError(
                    f"Opportunity level threshold {value!r} must be in (0.0, 1.0).",
                    field="threshold",
                    value=value,
                    engine=_ENGINE_NAME,
                )
        if not (thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]):
            raise JanusValidationError(
                "Opportunity level thresholds must be strictly increasing.",
                field="thresholds",
                value=thresholds,
                engine=_ENGINE_NAME,
            )
        if not 0.0 <= self.trend_stability_tolerance <= 1.0:
            raise JanusValidationError(
                f"trend_stability_tolerance {self.trend_stability_tolerance!r} "
                "must be in [0.0, 1.0].",
                field="trend_stability_tolerance",
                value=self.trend_stability_tolerance,
                engine=_ENGINE_NAME,
            )


# ---------------------------------------------------------------------------
# History Record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpportunityHistoryEntry:
    """Immutable point-in-time record of an OpportunityAssessment's composite score."""

    recorded_at: datetime
    composite_score: float
    level: OpportunityLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _require_unit_score(value: float, field_name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise JanusInvalidScoreError(value, field_name, engine=_ENGINE_NAME)
    return value


# ---------------------------------------------------------------------------
# Future Opportunity Engine
# ---------------------------------------------------------------------------


class OpportunityAnalysisEngine(IFutureOpportunityEngine):
    """
    Production implementation of the Future Opportunity Engine.

    Thread-safe: all mutable state is guarded by an internal `threading.RLock`.
    Immutable model handling: stored `OpportunityAssessment` instances are never
    mutated in place. Updates are performed by constructing a new instance via
    `OpportunityAssessment.create(...)` and replacing the stored reference; the
    superseded instance's composite score is preserved in the opportunity's
    history for trend analysis.

    Constitutional rule: JANUS identifies opportunities. VEGA decides whether
    to pursue them. These responsibilities are never merged.
    """

    def __init__(
        self, config: Optional[OpportunityAnalysisEngineConfig] = None
    ) -> None:
        self._config: OpportunityAnalysisEngineConfig = (
            config or OpportunityAnalysisEngineConfig()
        )
        self._lock = threading.RLock()
        self._initialized: bool = False
        self._shutdown: bool = False
        self._initialized_at: Optional[datetime] = None
        self._shutdown_at: Optional[datetime] = None

        self._assessments: dict[str, OpportunityAssessment] = {}
        self._history: dict[str, list[OpportunityHistoryEntry]] = {}

        self._assessments_created: int = 0
        self._lookups_performed: int = 0
        self._listings_performed: int = 0

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._initialized and not self._shutdown:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._assessments = {}
            self._history = {}
            self._assessments_created = 0
            self._lookups_performed = 0
            self._listings_performed = 0
            self._initialized = True
            self._shutdown = False
            self._initialized_at = datetime.utcnow()
            self._shutdown_at = None
            _LOG.info("%s initialized (version=%s)", _ENGINE_NAME, _ENGINE_VERSION)

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                raise JanusNotInitializedError(_ENGINE_NAME)
            if self._shutdown:
                return
            self._shutdown = True
            self._shutdown_at = datetime.utcnow()
            _LOG.info("%s shut down", _ENGINE_NAME)

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
        if not self._initialized:
            raise JanusNotInitializedError(_ENGINE_NAME)
        if self._shutdown:
            raise JanusShutdownError(_ENGINE_NAME)

    # -----------------------------------------------------------------
    # Opportunity Scoring
    # -----------------------------------------------------------------

    def compute_composite_opportunity_score(
        self, opportunity_factors: tuple[OpportunityFactor, ...]
    ) -> float:
        with self._lock:
            self._ensure_operational()
            if not opportunity_factors:
                return 0.0
            return statistics.mean(f.composite_score for f in opportunity_factors)

    def derive_opportunity_level(self, composite_score: float) -> OpportunityLevel:
        with self._lock:
            self._ensure_operational()
            composite_score = _require_unit_score(composite_score, "composite_score")

            cfg = self._config
            if composite_score <= cfg.marginal_threshold:
                return OpportunityLevel.MARGINAL
            if composite_score <= cfg.low_threshold:
                return OpportunityLevel.LOW
            if composite_score <= cfg.moderate_threshold:
                return OpportunityLevel.MODERATE
            if composite_score <= cfg.high_threshold:
                return OpportunityLevel.HIGH
            return OpportunityLevel.TRANSFORMATIVE

    # -----------------------------------------------------------------
    # Opportunity Identification & Modeling
    # -----------------------------------------------------------------

    def build_opportunity_factor(
        self,
        name: str,
        description: str,
        category: str,
        value_score: float,
        feasibility_score: float,
        time_horizon: ForecastHorizon,
        enablers: Optional[tuple[str, ...]] = None,
    ) -> OpportunityFactor:
        """
        Construct a single OpportunityFactor from raw analytical inputs.

        Raises JanusOpportunityFactorError if the supplied scores are invalid.
        """
        with self._lock:
            self._ensure_operational()

            value_score = _require_unit_score(value_score, "value_score")
            feasibility_score = _require_unit_score(
                feasibility_score, "feasibility_score"
            )

            try:
                return OpportunityFactor.create(
                    name=name,
                    description=description,
                    category=category,
                    value_score=value_score,
                    feasibility_score=feasibility_score,
                    time_horizon=time_horizon,
                    enablers=list(enablers or ()),
                )
            except ValueError as exc:
                raise JanusOpportunityFactorError(name, str(exc)) from exc

    def identify_opportunity_factors(
        self,
        candidates: tuple[
            tuple[str, str, str, float, float, ForecastHorizon, tuple[str, ...]], ...
        ],
    ) -> tuple[OpportunityFactor, ...]:
        """
        Identify a batch of OpportunityFactors from raw candidate tuples of
        (name, description, category, value_score, feasibility_score,
        time_horizon, enablers).

        Raises JanusOpportunityAnalysisError if no candidates are supplied.
        """
        with self._lock:
            self._ensure_operational()

            if not candidates:
                raise JanusOpportunityAnalysisError(
                    "Cannot identify opportunity factors from an empty candidate set.",
                    context={"candidate_count": 0},
                )

            factors: list[OpportunityFactor] = []
            for (
                name,
                description,
                category,
                value_score,
                feasibility_score,
                time_horizon,
                enablers,
            ) in candidates:
                factors.append(
                    self.build_opportunity_factor(
                        name=name,
                        description=description,
                        category=category,
                        value_score=value_score,
                        feasibility_score=feasibility_score,
                        time_horizon=time_horizon,
                        enablers=enablers,
                    )
                )
            return tuple(factors)

    def list_opportunity_factors_by_category(
        self, category: str
    ) -> tuple[OpportunityFactor, ...]:
        with self._lock:
            self._ensure_operational()

            seen: dict[str, OpportunityFactor] = {}
            for assessment in self._assessments.values():
                for factor in assessment.opportunity_factors:
                    if (
                        factor.category == category
                        and factor.factor_id not in seen
                    ):
                        seen[factor.factor_id] = factor
            return tuple(seen.values())

    # -----------------------------------------------------------------
    # Default Probability Distribution
    # -----------------------------------------------------------------

    def _default_probability_distribution(
        self,
        composite_score: float,
        uncertainty: UncertaintyProfile,
        confidence: ConfidenceProfile,
    ) -> ProbabilityDistribution:
        """
        Build a default two-outcome ProbabilityDistribution
        ("materializes" / "does_not_materialize") from a composite
        opportunity score when the caller does not supply one explicitly.
        """
        materializes = _clamp(composite_score)
        does_not_materialize = 1.0 - materializes

        try:
            return ProbabilityDistribution.create(
                label="opportunity_materialization",
                outcomes={
                    "materializes": materializes,
                    "does_not_materialize": does_not_materialize,
                },
                uncertainty_level=uncertainty.level,
                confidence=confidence,
            )
        except ValueError as exc:
            raise JanusOpportunityAnalysisError(
                f"Failed to construct default probability distribution: {exc}",
                context={"composite_score": composite_score},
            ) from exc

    # -----------------------------------------------------------------
    # Opportunity Analysis
    # -----------------------------------------------------------------

    def analyze_opportunity(
        self, request: OpportunityAnalysisRequest
    ) -> OpportunityAnalysisResponse:
        with self._lock:
            self._ensure_operational()

            if not request.opportunity_factors:
                raise JanusOpportunityFactorError(
                    "<unspecified>",
                    "OpportunityAnalysisRequest.opportunity_factors must not be empty.",
                )

            composite_score = self.compute_composite_opportunity_score(
                request.opportunity_factors
            )
            level = self.derive_opportunity_level(composite_score)

            probability_distribution = self._default_probability_distribution(
                composite_score=composite_score,
                uncertainty=request.uncertainty,
                confidence=request.confidence,
            )

            try:
                assessment = OpportunityAssessment.create(
                    title=request.title,
                    description=request.description,
                    level=level,
                    opportunity_factors=list(request.opportunity_factors),
                    probability_distribution=probability_distribution,
                    uncertainty=request.uncertainty,
                    horizon=request.horizon,
                    confidence=request.confidence,
                    evidence=request.evidence,
                    capture_strategies=list(request.capture_strategies),
                )
            except ValueError as exc:
                raise JanusOpportunityAnalysisError(
                    str(exc), context={"title": request.title}
                ) from exc

            self._assessments[assessment.opportunity_id] = assessment
            self._history[assessment.opportunity_id] = [
                OpportunityHistoryEntry(
                    recorded_at=assessment.created_at,
                    composite_score=composite_score,
                    level=level,
                )
            ]
            self._assessments_created += 1

            return OpportunityAnalysisResponse(
                opportunity_assessment=assessment,
                computed_level=level,
                assessed_at=assessment.created_at,
                engine_version=_ENGINE_VERSION,
            )

    def get_opportunity_assessment(
        self, request: OpportunityAssessmentGetRequest
    ) -> OpportunityAssessmentGetResponse:
        with self._lock:
            self._ensure_operational()
            self._lookups_performed += 1

            assessment = self._assessments.get(request.opportunity_id)
            if assessment is None:
                raise JanusOpportunityAssessmentNotFoundError(request.opportunity_id)

            return OpportunityAssessmentGetResponse(
                opportunity_assessment=assessment, retrieved_at=datetime.utcnow()
            )

    def list_opportunities_by_level(
        self, request: OpportunityListByLevelRequest
    ) -> OpportunityListByLevelResponse:
        with self._lock:
            self._ensure_operational()
            self._listings_performed += 1

            matches = [
                assessment
                for assessment in self._assessments.values()
                if assessment.level.value >= request.minimum_level.value
                and (
                    request.horizon is None
                    or assessment.horizon == request.horizon
                )
            ]
            matches.sort(
                key=lambda a: a.composite_opportunity_score, reverse=True
            )

            return OpportunityListByLevelResponse(
                opportunity_assessments=tuple(matches),
                minimum_level=request.minimum_level,
                retrieved_at=datetime.utcnow(),
            )

    # -----------------------------------------------------------------
    # Opportunity Ranking & Aggregation
    # -----------------------------------------------------------------

    def rank_opportunity_assessments(
        self, opportunity_ids: Optional[tuple[str, ...]] = None
    ) -> tuple[OpportunityAssessment, ...]:
        """
        Return OpportunityAssessments ordered from most to least valuable by
        composite opportunity score. If `opportunity_ids` is provided, only
        those assessments are ranked; otherwise all stored assessments are ranked.

        Raises JanusOpportunityAssessmentNotFoundError if any requested
        opportunity_id is not present.
        """
        with self._lock:
            self._ensure_operational()

            if opportunity_ids is None:
                assessments = list(self._assessments.values())
            else:
                assessments = []
                for opportunity_id in opportunity_ids:
                    assessment = self._assessments.get(opportunity_id)
                    if assessment is None:
                        raise JanusOpportunityAssessmentNotFoundError(opportunity_id)
                    assessments.append(assessment)

            assessments.sort(
                key=lambda a: a.composite_opportunity_score, reverse=True
            )
            return tuple(assessments)

    def aggregate_opportunity_assessments(
        self, opportunity_ids: tuple[str, ...]
    ) -> dict[str, object]:
        """
        Aggregate a set of OpportunityAssessments into summary statistics:
        mean/min/max composite score, a count of assessments at each
        OpportunityLevel, and the categories appearing among their opportunity
        factors, ordered by frequency.

        Raises JanusOpportunityAnalysisError if `opportunity_ids` is empty,
        and JanusOpportunityAssessmentNotFoundError if any id is unknown.
        """
        with self._lock:
            self._ensure_operational()

            if not opportunity_ids:
                raise JanusOpportunityAnalysisError(
                    "Cannot aggregate an empty set of opportunity assessments.",
                    context={"opportunity_id_count": 0},
                )

            assessments: list[OpportunityAssessment] = []
            for opportunity_id in opportunity_ids:
                assessment = self._assessments.get(opportunity_id)
                if assessment is None:
                    raise JanusOpportunityAssessmentNotFoundError(opportunity_id)
                assessments.append(assessment)

            composite_scores = [a.composite_opportunity_score for a in assessments]

            level_distribution: dict[str, int] = {
                level.name: 0 for level in OpportunityLevel
            }
            category_counts: dict[str, int] = {}
            for assessment in assessments:
                level_distribution[assessment.level.name] += 1
                for factor in assessment.opportunity_factors:
                    category_counts[factor.category] = (
                        category_counts.get(factor.category, 0) + 1
                    )

            dominant_categories = tuple(
                sorted(
                    category_counts,
                    key=lambda c: category_counts[c],
                    reverse=True,
                )
            )

            return {
                "opportunity_count": len(assessments),
                "mean_composite_score": statistics.mean(composite_scores),
                "min_composite_score": min(composite_scores),
                "max_composite_score": max(composite_scores),
                "level_distribution": level_distribution,
                "dominant_categories": dominant_categories,
            }

    # -----------------------------------------------------------------
    # Opportunity Trend Analysis
    # -----------------------------------------------------------------

    def record_opportunity_observation(
        self,
        opportunity_id: str,
        composite_score: float,
        level: OpportunityLevel,
    ) -> None:
        """
        Record a new historical observation of an OpportunityAssessment's
        composite score, for use by `analyze_opportunity_trend`.

        Raises JanusOpportunityAssessmentNotFoundError if `opportunity_id`
        is unknown.
        """
        with self._lock:
            self._ensure_operational()

            if opportunity_id not in self._assessments:
                raise JanusOpportunityAssessmentNotFoundError(opportunity_id)

            _require_unit_score(composite_score, "composite_score")

            self._history.setdefault(opportunity_id, []).append(
                OpportunityHistoryEntry(
                    recorded_at=datetime.utcnow(),
                    composite_score=composite_score,
                    level=level,
                )
            )

    def analyze_opportunity_trend(self, opportunity_id: str) -> dict[str, object]:
        """
        Analyze the historical trend of an OpportunityAssessment's composite
        score.

        Returns a dict containing the observation history, the delta between
        the earliest and latest recorded composite scores, and a trend
        direction of "increasing", "decreasing", or "stable"
        (within `trend_stability_tolerance`), or "insufficient_data" if fewer
        than two observations have been recorded.

        Raises JanusOpportunityAssessmentNotFoundError if `opportunity_id`
        is unknown.
        """
        with self._lock:
            self._ensure_operational()

            if opportunity_id not in self._assessments:
                raise JanusOpportunityAssessmentNotFoundError(opportunity_id)

            history = self._history.get(opportunity_id, [])

            if len(history) < 2:
                return {
                    "opportunity_id": opportunity_id,
                    "observation_count": len(history),
                    "trend": "insufficient_data",
                    "delta": 0.0,
                    "history": tuple(
                        {
                            "recorded_at": entry.recorded_at.isoformat(),
                            "composite_score": entry.composite_score,
                            "level": entry.level.name,
                        }
                        for entry in history
                    ),
                }

            first = history[0]
            last = history[-1]
            delta = last.composite_score - first.composite_score

            if abs(delta) <= self._config.trend_stability_tolerance:
                trend = "stable"
            elif delta > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            return {
                "opportunity_id": opportunity_id,
                "observation_count": len(history),
                "trend": trend,
                "delta": delta,
                "history": tuple(
                    {
                        "recorded_at": entry.recorded_at.isoformat(),
                        "composite_score": entry.composite_score,
                        "level": entry.level.name,
                    }
                    for entry in history
                ),
            }

    # -----------------------------------------------------------------
    # Opportunity Statistics & Evaluation Utilities
    # -----------------------------------------------------------------

    def compute_opportunity_statistics(
        self, opportunity_ids: Optional[tuple[str, ...]] = None
    ) -> dict[str, object]:
        """
        Compute summary statistics (mean, population standard deviation,
        min, max) of composite opportunity scores across the given
        assessments, or across all stored assessments if `opportunity_ids`
        is None.

        Raises JanusOpportunityAnalysisError if there are no assessments
        to summarize, and JanusOpportunityAssessmentNotFoundError if any
        requested id is unknown.
        """
        with self._lock:
            self._ensure_operational()

            if opportunity_ids is None:
                assessments = list(self._assessments.values())
            else:
                assessments = []
                for opportunity_id in opportunity_ids:
                    assessment = self._assessments.get(opportunity_id)
                    if assessment is None:
                        raise JanusOpportunityAssessmentNotFoundError(opportunity_id)
                    assessments.append(assessment)

            if not assessments:
                raise JanusOpportunityAnalysisError(
                    "Cannot compute statistics over zero opportunity assessments.",
                    context={"opportunity_id_count": 0},
                )

            scores = [a.composite_opportunity_score for a in assessments]

            return {
                "count": len(scores),
                "mean": statistics.mean(scores),
                "population_stdev": (
                    statistics.pstdev(scores) if len(scores) > 1 else 0.0
                ),
                "min": min(scores),
                "max": max(scores),
            }

    def evaluate_opportunity_assessment(
        self, opportunity_id: str
    ) -> dict[str, object]:
        """
        Produce an evaluation summary for a single OpportunityAssessment:
        its level, composite score, dominant opportunity factor (highest
        composite_score), uncertainty level, and confidence level.

        This is an evaluation utility only — it makes no decision about
        whether to pursue the opportunity. VEGA decides.

        Raises JanusOpportunityAssessmentNotFoundError if `opportunity_id`
        is unknown.
        """
        with self._lock:
            self._ensure_operational()

            assessment = self._assessments.get(opportunity_id)
            if assessment is None:
                raise JanusOpportunityAssessmentNotFoundError(opportunity_id)

            dominant_factor_summary: Optional[dict[str, object]] = None
            if assessment.opportunity_factors:
                dominant_factor = max(
                    assessment.opportunity_factors,
                    key=lambda f: f.composite_score,
                )
                dominant_factor_summary = {
                    "factor_id": dominant_factor.factor_id,
                    "name": dominant_factor.name,
                    "category": dominant_factor.category,
                    "composite_score": dominant_factor.composite_score,
                }

            return {
                "opportunity_id": assessment.opportunity_id,
                "title": assessment.title,
                "level": assessment.level.name,
                "composite_opportunity_score": assessment.composite_opportunity_score,
                "horizon": assessment.horizon.value,
                "uncertainty_level": assessment.uncertainty.level.name,
                "confidence_level": assessment.confidence.level.name,
                "probability_mode": assessment.probability_distribution.mode,
                "dominant_opportunity_factor": dominant_factor_summary,
                "capture_strategy_count": len(assessment.capture_strategies),
            }

    def score_opportunity_feasibility(
        self, opportunity_id: str
    ) -> dict[str, object]:
        """
        Produce a focused feasibility breakdown for an OpportunityAssessment,
        reporting each factor's value_score, feasibility_score, composite_score,
        and enablers — structured for ODYSSEY and VEGA consumption.

        Raises JanusOpportunityAssessmentNotFoundError if `opportunity_id`
        is unknown.
        """
        with self._lock:
            self._ensure_operational()

            assessment = self._assessments.get(opportunity_id)
            if assessment is None:
                raise JanusOpportunityAssessmentNotFoundError(opportunity_id)

            factor_breakdown = tuple(
                {
                    "factor_id": f.factor_id,
                    "name": f.name,
                    "category": f.category,
                    "value_score": f.value_score,
                    "feasibility_score": f.feasibility_score,
                    "composite_score": f.composite_score,
                    "enablers": f.enablers,
                }
                for f in assessment.opportunity_factors
            )

            mean_value = (
                statistics.mean(f.value_score for f in assessment.opportunity_factors)
                if assessment.opportunity_factors
                else 0.0
            )
            mean_feasibility = (
                statistics.mean(
                    f.feasibility_score for f in assessment.opportunity_factors
                )
                if assessment.opportunity_factors
                else 0.0
            )

            return {
                "opportunity_id": opportunity_id,
                "title": assessment.title,
                "horizon": assessment.horizon.value,
                "level": assessment.level.name,
                "mean_value_score": mean_value,
                "mean_feasibility_score": mean_feasibility,
                "composite_opportunity_score": assessment.composite_opportunity_score,
                "factor_breakdown": factor_breakdown,
            }

    def validate_opportunity_level_assignment(
        self,
        assessment: OpportunityAssessment,
    ) -> None:
        """
        Validate that an OpportunityAssessment's declared level is consistent
        with the level derived from its composite score.

        Raises JanusOpportunityLevelError if the levels differ.
        """
        with self._lock:
            self._ensure_operational()

            composite_score = assessment.composite_opportunity_score
            computed_level = self.derive_opportunity_level(composite_score)

            if assessment.level != computed_level:
                raise JanusOpportunityLevelError(
                    assigned=assessment.level.name,
                    computed=computed_level.name,
                )

    # -----------------------------------------------------------------
    # Diagnostics & Health Reporting
    # -----------------------------------------------------------------

    def diagnostics(self) -> dict[str, object]:
        """Return internal diagnostic counters and configuration state."""
        with self._lock:
            return {
                "engine_name": _ENGINE_NAME,
                "engine_version": _ENGINE_VERSION,
                "is_initialized": self._initialized,
                "is_shutdown": self._shutdown,
                "initialized_at": (
                    self._initialized_at.isoformat()
                    if self._initialized_at
                    else None
                ),
                "shutdown_at": (
                    self._shutdown_at.isoformat() if self._shutdown_at else None
                ),
                "stored_assessment_count": len(self._assessments),
                "assessments_created": self._assessments_created,
                "lookups_performed": self._lookups_performed,
                "listings_performed": self._listings_performed,
                "config": {
                    "marginal_threshold": self._config.marginal_threshold,
                    "low_threshold": self._config.low_threshold,
                    "moderate_threshold": self._config.moderate_threshold,
                    "high_threshold": self._config.high_threshold,
                    "trend_stability_tolerance": (
                        self._config.trend_stability_tolerance
                    ),
                },
            }

    def health_check(self) -> dict[str, object]:
        """
        Return a health report for this engine.

        Status values:
          - "healthy":         initialized and operational.
          - "not_initialized": initialize() has not yet been called.
          - "shutdown":        the engine has been shut down.
        """
        with self._lock:
            if self._shutdown:
                status = "shutdown"
            elif not self._initialized:
                status = "not_initialized"
            else:
                status = "healthy"

            level_counts: dict[str, int] = {
                level.name: 0 for level in OpportunityLevel
            }
            for assessment in self._assessments.values():
                level_counts[assessment.level.name] += 1

            return {
                "engine_name": _ENGINE_NAME,
                "engine_version": _ENGINE_VERSION,
                "status": status,
                "checked_at": datetime.utcnow().isoformat(),
                "stored_assessment_count": len(self._assessments),
                "level_distribution": level_counts,
            }