"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/risk_analysis.py

Future Risk Engine — IFutureRiskEngine implementation.

Responsibility:
    Identify and assess future threats across skill obsolescence, technology
    disruption, market changes, project failure, and resource exhaustion.
    Focuses exclusively on future risk, not present-state risk.

Constitutional rule:
    JANUS predicts future risks. It never decides what to do about them.
    This engine produces RiskAssessments, rankings, aggregates, trends, and
    statistics — analysis only. VEGA decides; ZENITH plans; this engine does
    neither.

JANUS owns future risk analysis exclusively.
VEGA owns risk-response decisions.
ZENITH owns mitigation planning.
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
    JanusRiskAnalysisError,
    JanusRiskAssessmentNotFoundError,
    JanusRiskFactorError,
    JanusShutdownError,
    JanusValidationError,
)
from .interfaces import IFutureRiskEngine
from .models import (
    ConfidenceProfile,
    EvidenceProfile,
    ForecastHorizon,
    ProbabilityDistribution,
    RiskAssessment,
    RiskFactor,
    RiskLevel,
    UncertaintyProfile,
)
from .schemas import (
    RiskAnalysisRequest,
    RiskAnalysisResponse,
    RiskAssessmentGetRequest,
    RiskAssessmentGetResponse,
    RiskListByLevelRequest,
    RiskListByLevelResponse,
)

_LOG = logging.getLogger(__name__)

_ENGINE_NAME: str = "RiskAnalysisEngine"
_ENGINE_VERSION: str = "5.1.0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskAnalysisEngineConfig:
    """
    Tunable thresholds for the Future Risk Engine.

    Thresholds must be strictly increasing and lie within (0.0, 1.0).
    A composite risk score at or below `negligible_threshold` maps to
    RiskLevel.NEGLIGIBLE; above `high_threshold` maps to RiskLevel.CRITICAL.
    """

    negligible_threshold: float = 0.10
    low_threshold: float = 0.30
    moderate_threshold: float = 0.55
    high_threshold: float = 0.80

    trend_stability_tolerance: float = 0.05

    def __post_init__(self) -> None:
        thresholds = (
            self.negligible_threshold,
            self.low_threshold,
            self.moderate_threshold,
            self.high_threshold,
        )
        for value in thresholds:
            if not 0.0 < value < 1.0:
                raise JanusValidationError(
                    f"Risk level threshold {value!r} must be in (0.0, 1.0).",
                    field="threshold",
                    value=value,
                    engine=_ENGINE_NAME,
                )
        if not (thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]):
            raise JanusValidationError(
                "Risk level thresholds must be strictly increasing.",
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
class RiskHistoryEntry:
    """Immutable point-in-time record of a RiskAssessment's composite score."""

    recorded_at: datetime
    composite_score: float
    level: RiskLevel


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
# Future Risk Engine
# ---------------------------------------------------------------------------


class RiskAnalysisEngine(IFutureRiskEngine):
    """
    Production implementation of the Future Risk Engine.

    Thread-safe: all mutable state is guarded by an internal `threading.RLock`.
    Immutable model handling: stored `RiskAssessment` instances are never
    mutated in place. Updates are performed by constructing a new instance
    via `RiskAssessment.create(...)` and replacing the stored reference; the
    superseded instance's composite score is preserved in the risk's history
    for trend analysis.
    """

    def __init__(self, config: Optional[RiskAnalysisEngineConfig] = None) -> None:
        self._config: RiskAnalysisEngineConfig = config or RiskAnalysisEngineConfig()
        self._lock = threading.RLock()
        self._initialized: bool = False
        self._shutdown: bool = False
        self._initialized_at: Optional[datetime] = None
        self._shutdown_at: Optional[datetime] = None

        self._assessments: dict[str, RiskAssessment] = {}
        self._history: dict[str, list[RiskHistoryEntry]] = {}

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
                # Idempotent: a second shutdown() call is a no-op.
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
    # Risk Scoring
    # -----------------------------------------------------------------

    def compute_composite_risk_score(
        self, risk_factors: tuple[RiskFactor, ...]
    ) -> float:
        with self._lock:
            self._ensure_operational()
            if not risk_factors:
                return 0.0
            return statistics.mean(rf.composite_score for rf in risk_factors)

    def derive_risk_level(self, composite_score: float) -> RiskLevel:
        with self._lock:
            self._ensure_operational()
            composite_score = _require_unit_score(composite_score, "composite_score")

            cfg = self._config
            if composite_score <= cfg.negligible_threshold:
                return RiskLevel.NEGLIGIBLE
            if composite_score <= cfg.low_threshold:
                return RiskLevel.LOW
            if composite_score <= cfg.moderate_threshold:
                return RiskLevel.MODERATE
            if composite_score <= cfg.high_threshold:
                return RiskLevel.HIGH
            return RiskLevel.CRITICAL

    # -----------------------------------------------------------------
    # Risk Identification & Modeling
    # -----------------------------------------------------------------

    def build_risk_factor(
        self,
        name: str,
        description: str,
        category: str,
        impact_score: float,
        likelihood_score: float,
        time_horizon: ForecastHorizon,
        mitigations: Optional[tuple[str, ...]] = None,
    ) -> RiskFactor:
        """
        Construct a single RiskFactor from raw analytical inputs.

        Raises JanusRiskFactorError if the supplied scores are invalid.
        """
        with self._lock:
            self._ensure_operational()

            impact_score = _require_unit_score(impact_score, "impact_score")
            likelihood_score = _require_unit_score(likelihood_score, "likelihood_score")

            try:
                return RiskFactor.create(
                    name=name,
                    description=description,
                    category=category,
                    impact_score=impact_score,
                    likelihood_score=likelihood_score,
                    time_horizon=time_horizon,
                    mitigations=list(mitigations or ()),
                )
            except ValueError as exc:
                raise JanusRiskFactorError(name, str(exc)) from exc

    def identify_risk_factors(
        self,
        candidates: tuple[
            tuple[str, str, str, float, float, ForecastHorizon, tuple[str, ...]], ...
        ],
    ) -> tuple[RiskFactor, ...]:
        """
        Identify a batch of RiskFactors from raw candidate tuples of
        (name, description, category, impact_score, likelihood_score,
        time_horizon, mitigations).

        Raises JanusRiskAnalysisError if no candidates are supplied.
        """
        with self._lock:
            self._ensure_operational()

            if not candidates:
                raise JanusRiskAnalysisError(
                    "Cannot identify risk factors from an empty candidate set.",
                    context={"candidate_count": 0},
                )

            factors: list[RiskFactor] = []
            for (
                name,
                description,
                category,
                impact_score,
                likelihood_score,
                time_horizon,
                mitigations,
            ) in candidates:
                factors.append(
                    self.build_risk_factor(
                        name=name,
                        description=description,
                        category=category,
                        impact_score=impact_score,
                        likelihood_score=likelihood_score,
                        time_horizon=time_horizon,
                        mitigations=mitigations,
                    )
                )
            return tuple(factors)

    def list_risk_factors_by_category(
        self, category: str
    ) -> tuple[RiskFactor, ...]:
        with self._lock:
            self._ensure_operational()

            seen: dict[str, RiskFactor] = {}
            for assessment in self._assessments.values():
                for factor in assessment.risk_factors:
                    if factor.category == category and factor.factor_id not in seen:
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
        ("materializes" / "does_not_materialize") from a composite risk
        score when the caller does not supply one explicitly.
        """
        materializes = _clamp(composite_score)
        does_not_materialize = 1.0 - materializes

        try:
            return ProbabilityDistribution.create(
                label="risk_materialization",
                outcomes={
                    "materializes": materializes,
                    "does_not_materialize": does_not_materialize,
                },
                uncertainty_level=uncertainty.level,
                confidence=confidence,
            )
        except ValueError as exc:
            raise JanusRiskAnalysisError(
                f"Failed to construct default probability distribution: {exc}",
                context={"composite_score": composite_score},
            ) from exc

    # -----------------------------------------------------------------
    # Risk Analysis
    # -----------------------------------------------------------------

    def analyze_risk(self, request: RiskAnalysisRequest) -> RiskAnalysisResponse:
        with self._lock:
            self._ensure_operational()

            if not request.risk_factors:
                raise JanusRiskFactorError(
                    "<unspecified>",
                    "RiskAnalysisRequest.risk_factors must not be empty.",
                )

            composite_score = self.compute_composite_risk_score(request.risk_factors)
            level = self.derive_risk_level(composite_score)

            probability_distribution = self._default_probability_distribution(
                composite_score=composite_score,
                uncertainty=request.uncertainty,
                confidence=request.confidence,
            )

            try:
                assessment = RiskAssessment.create(
                    title=request.title,
                    description=request.description,
                    level=level,
                    risk_factors=list(request.risk_factors),
                    probability_distribution=probability_distribution,
                    uncertainty=request.uncertainty,
                    horizon=request.horizon,
                    confidence=request.confidence,
                    evidence=request.evidence,
                    mitigation_strategies=list(request.mitigation_strategies),
                )
            except ValueError as exc:
                raise JanusRiskAnalysisError(
                    str(exc), context={"title": request.title}
                ) from exc

            self._assessments[assessment.risk_id] = assessment
            self._history[assessment.risk_id] = [
                RiskHistoryEntry(
                    recorded_at=assessment.created_at,
                    composite_score=composite_score,
                    level=level,
                )
            ]
            self._assessments_created += 1

            return RiskAnalysisResponse(
                risk_assessment=assessment,
                computed_level=level,
                assessed_at=assessment.created_at,
                engine_version=_ENGINE_VERSION,
            )

    def get_risk_assessment(
        self, request: RiskAssessmentGetRequest
    ) -> RiskAssessmentGetResponse:
        with self._lock:
            self._ensure_operational()
            self._lookups_performed += 1

            assessment = self._assessments.get(request.risk_id)
            if assessment is None:
                raise JanusRiskAssessmentNotFoundError(request.risk_id)

            return RiskAssessmentGetResponse(
                risk_assessment=assessment, retrieved_at=datetime.utcnow()
            )

    def list_risks_by_level(
        self, request: RiskListByLevelRequest
    ) -> RiskListByLevelResponse:
        with self._lock:
            self._ensure_operational()
            self._listings_performed += 1

            matches = [
                assessment
                for assessment in self._assessments.values()
                if assessment.level.value >= request.minimum_level.value
                and (request.horizon is None or assessment.horizon == request.horizon)
            ]
            matches.sort(key=lambda a: a.composite_risk_score, reverse=True)

            return RiskListByLevelResponse(
                risk_assessments=tuple(matches),
                minimum_level=request.minimum_level,
                retrieved_at=datetime.utcnow(),
            )

    # -----------------------------------------------------------------
    # Risk Ranking & Aggregation
    # -----------------------------------------------------------------

    def rank_risk_assessments(
        self, risk_ids: Optional[tuple[str, ...]] = None
    ) -> tuple[RiskAssessment, ...]:
        """
        Return RiskAssessments ordered from most to least severe by
        composite risk score. If `risk_ids` is provided, only those
        assessments are ranked; otherwise all stored assessments are ranked.

        Raises JanusRiskAssessmentNotFoundError if any requested risk_id is
        not present.
        """
        with self._lock:
            self._ensure_operational()

            if risk_ids is None:
                assessments = list(self._assessments.values())
            else:
                assessments = []
                for risk_id in risk_ids:
                    assessment = self._assessments.get(risk_id)
                    if assessment is None:
                        raise JanusRiskAssessmentNotFoundError(risk_id)
                    assessments.append(assessment)

            assessments.sort(key=lambda a: a.composite_risk_score, reverse=True)
            return tuple(assessments)

    def aggregate_risk_assessments(
        self, risk_ids: tuple[str, ...]
    ) -> dict[str, object]:
        """
        Aggregate a set of RiskAssessments into summary statistics:
        mean/min/max composite score, a count of assessments at each
        RiskLevel, and the categories appearing among their risk factors,
        ordered by frequency.

        Raises JanusRiskAnalysisError if `risk_ids` is empty, and
        JanusRiskAssessmentNotFoundError if any id is unknown.
        """
        with self._lock:
            self._ensure_operational()

            if not risk_ids:
                raise JanusRiskAnalysisError(
                    "Cannot aggregate an empty set of risk assessments.",
                    context={"risk_id_count": 0},
                )

            assessments: list[RiskAssessment] = []
            for risk_id in risk_ids:
                assessment = self._assessments.get(risk_id)
                if assessment is None:
                    raise JanusRiskAssessmentNotFoundError(risk_id)
                assessments.append(assessment)

            composite_scores = [a.composite_risk_score for a in assessments]

            level_distribution: dict[str, int] = {level.name: 0 for level in RiskLevel}
            category_counts: dict[str, int] = {}
            for assessment in assessments:
                level_distribution[assessment.level.name] += 1
                for factor in assessment.risk_factors:
                    category_counts[factor.category] = (
                        category_counts.get(factor.category, 0) + 1
                    )

            dominant_categories = tuple(
                sorted(category_counts, key=lambda c: category_counts[c], reverse=True)
            )

            return {
                "risk_count": len(assessments),
                "mean_composite_score": statistics.mean(composite_scores),
                "min_composite_score": min(composite_scores),
                "max_composite_score": max(composite_scores),
                "level_distribution": level_distribution,
                "dominant_categories": dominant_categories,
            }

    # -----------------------------------------------------------------
    # Risk Trend Analysis
    # -----------------------------------------------------------------

    def record_risk_observation(
        self, risk_id: str, composite_score: float, level: RiskLevel
    ) -> None:
        """
        Record a new historical observation of a RiskAssessment's composite
        score, for use by `analyze_risk_trend`.

        Raises JanusRiskAssessmentNotFoundError if `risk_id` is unknown.
        """
        with self._lock:
            self._ensure_operational()

            if risk_id not in self._assessments:
                raise JanusRiskAssessmentNotFoundError(risk_id)

            composite_score = _require_unit_score(composite_score, "composite_score")

            self._history.setdefault(risk_id, []).append(
                RiskHistoryEntry(
                    recorded_at=datetime.utcnow(),
                    composite_score=composite_score,
                    level=level,
                )
            )

    def analyze_risk_trend(self, risk_id: str) -> dict[str, object]:
        """
        Analyze the historical trend of a RiskAssessment's composite score.

        Returns a dict containing the observation history, the delta
        between the earliest and latest recorded composite scores, and a
        trend direction of "increasing", "decreasing", or "stable"
        (within `trend_stability_tolerance`), or "insufficient_data" if
        fewer than two observations have been recorded.

        Raises JanusRiskAssessmentNotFoundError if `risk_id` is unknown.
        """
        with self._lock:
            self._ensure_operational()

            if risk_id not in self._assessments:
                raise JanusRiskAssessmentNotFoundError(risk_id)

            history = self._history.get(risk_id, [])

            if len(history) < 2:
                return {
                    "risk_id": risk_id,
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
                "risk_id": risk_id,
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
    # Risk Statistics & Evaluation Utilities
    # -----------------------------------------------------------------

    def compute_risk_statistics(
        self, risk_ids: Optional[tuple[str, ...]] = None
    ) -> dict[str, object]:
        """
        Compute summary statistics (mean, population standard deviation,
        min, max) of composite risk scores across the given assessments,
        or across all stored assessments if `risk_ids` is None.

        Raises JanusRiskAnalysisError if there are no assessments to
        summarize, and JanusRiskAssessmentNotFoundError if any requested
        id is unknown.
        """
        with self._lock:
            self._ensure_operational()

            if risk_ids is None:
                assessments = list(self._assessments.values())
            else:
                assessments = []
                for risk_id in risk_ids:
                    assessment = self._assessments.get(risk_id)
                    if assessment is None:
                        raise JanusRiskAssessmentNotFoundError(risk_id)
                    assessments.append(assessment)

            if not assessments:
                raise JanusRiskAnalysisError(
                    "Cannot compute statistics over zero risk assessments.",
                    context={"risk_id_count": 0},
                )

            scores = [a.composite_risk_score for a in assessments]

            return {
                "count": len(scores),
                "mean": statistics.mean(scores),
                "population_stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores),
            }

    def evaluate_risk_assessment(self, risk_id: str) -> dict[str, object]:
        """
        Produce an evaluation summary for a single RiskAssessment:
        its level, composite score, dominant risk factor (highest
        composite_score), uncertainty level, and confidence level.

        This is an evaluation utility only — it makes no decision about
        how to respond to the risk. VEGA decides.

        Raises JanusRiskAssessmentNotFoundError if `risk_id` is unknown.
        """
        with self._lock:
            self._ensure_operational()

            assessment = self._assessments.get(risk_id)
            if assessment is None:
                raise JanusRiskAssessmentNotFoundError(risk_id)

            if assessment.risk_factors:
                dominant_factor = max(
                    assessment.risk_factors, key=lambda rf: rf.composite_score
                )
                dominant_factor_summary: Optional[dict[str, object]] = {
                    "factor_id": dominant_factor.factor_id,
                    "name": dominant_factor.name,
                    "category": dominant_factor.category,
                    "composite_score": dominant_factor.composite_score,
                }
            else:
                dominant_factor_summary = None

            return {
                "risk_id": assessment.risk_id,
                "title": assessment.title,
                "level": assessment.level.name,
                "composite_risk_score": assessment.composite_risk_score,
                "horizon": assessment.horizon.value,
                "uncertainty_level": assessment.uncertainty.level.name,
                "confidence_level": assessment.confidence.level.name,
                "probability_mode": assessment.probability_distribution.mode,
                "dominant_risk_factor": dominant_factor_summary,
                "mitigation_strategy_count": len(assessment.mitigation_strategies),
            }

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
                    self._initialized_at.isoformat() if self._initialized_at else None
                ),
                "shutdown_at": (
                    self._shutdown_at.isoformat() if self._shutdown_at else None
                ),
                "stored_assessment_count": len(self._assessments),
                "assessments_created": self._assessments_created,
                "lookups_performed": self._lookups_performed,
                "listings_performed": self._listings_performed,
                "config": {
                    "negligible_threshold": self._config.negligible_threshold,
                    "low_threshold": self._config.low_threshold,
                    "moderate_threshold": self._config.moderate_threshold,
                    "high_threshold": self._config.high_threshold,
                    "trend_stability_tolerance": self._config.trend_stability_tolerance,
                },
            }

    def health_check(self) -> dict[str, object]:
        """
        Return a health report for this engine.

        Status values:
          - "healthy": initialized and operational.
          - "not_initialized": initialize() has not yet been called.
          - "shutdown": the engine has been shut down.
        """
        with self._lock:
            if self._shutdown:
                status = "shutdown"
            elif not self._initialized:
                status = "not_initialized"
            else:
                status = "healthy"

            level_counts: dict[str, int] = {level.name: 0 for level in RiskLevel}
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

    def get_health(self) -> dict[str, object]:
        """
        Return a health report including an `is_healthy` flag, for callers
        that prefer a boolean health indicator alongside the detailed
        `health_check` report.
        """
        report = self.health_check()
        report["is_healthy"] = report["status"] == "healthy"
        return report