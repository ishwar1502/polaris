"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/uncertainty.py

Uncertainty Engine — IUncertaintyEngine implementation.

Responsibility:
    Model and quantify uncertainty across all JANUS forecast and scenario
    artifacts. Provides confidence estimation, uncertainty aggregation,
    uncertainty propagation across forecast horizons, volatility analysis,
    unknown-risk modeling, confidence scoring, and statistical utilities.

Constitutional rule (Law 6):
    Every forecast must include uncertainty.
    No certainty claims ever.
    This engine is the enforcement point for that invariant; the
    Scenario Integrity Engine calls into it to verify compliance.

JANUS owns uncertainty modeling exclusively.
JANUS never owns decisions, plans, identity, or knowledge storage.
This engine produces analysis only — it never approves, selects, or
decides anything. VEGA decides.
"""

from __future__ import annotations

import logging
import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .exceptions import (
    JanusAlreadyInitializedError,
    JanusInvalidScoreError,
    JanusMissingUncertaintyError,
    JanusNotInitializedError,
    JanusShutdownError,
    JanusUncertaintyModelingError,
    JanusUncertaintyProfileNotFoundError,
    JanusValidationError,
)
from .interfaces import IUncertaintyEngine
from .models import (
    ConfidenceProfile,
    ForecastHorizon,
    ProbabilityDistribution,
    ProbabilityLevel,
    UncertaintyLevel,
    UncertaintyProfile,
)

_LOG = logging.getLogger(__name__)

_ENGINE_NAME: str = "UncertaintyEngine"
_ENGINE_VERSION: str = "5.1.0"


# ---------------------------------------------------------------------------
# Schema Contracts
#
# These mirror the request/response contracts declared in `.schemas` for
# IUncertaintyEngine. They are defined here as immutable dataclasses so this
# module is self-contained and exception-driven validation can occur at the
# boundary before any model object is constructed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyProfileCreateRequest:
    """Request to create a new UncertaintyProfile."""

    known_risks: tuple[str, ...]
    unknown_risk_exposure: float
    volatility_score: float
    external_factors: tuple[str, ...]
    market_sensitivity: float
    technology_sensitivity: float
    notes: str = ""


@dataclass(frozen=True)
class UncertaintyProfileCreateResponse:
    """Response containing the newly created UncertaintyProfile."""

    uncertainty_profile: UncertaintyProfile

    @property
    def profile(self) -> UncertaintyProfile:
        return self.uncertainty_profile


@dataclass(frozen=True)
class UncertaintyProfileGetRequest:
    """Request to retrieve an UncertaintyProfile by id."""

    uncertainty_id: str


@dataclass(frozen=True)
class UncertaintyProfileGetResponse:
    """Response containing the retrieved UncertaintyProfile."""

    uncertainty_profile: UncertaintyProfile

    @property
    def profile(self) -> UncertaintyProfile:
        return self.uncertainty_profile


@dataclass(frozen=True)
class UncertaintyValidationRequest:
    """
    Request to validate that an artifact carries a non-null, non-trivial
    UncertaintyProfile, per JANUS Law 6.
    """

    artifact_id: str
    artifact_type: str
    uncertainty: Optional[UncertaintyProfile]


@dataclass(frozen=True)
class UncertaintyValidationResponse:
    """Response describing the result of an uncertainty validation."""

    is_valid: bool
    violations: tuple[str, ...]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyEngineConfig:
    """
    Tunable thresholds and weights for the Uncertainty Engine.

    All weight values must sum to 1.0 (±1e-6). All thresholds must be
    strictly increasing and lie within (0.0, 1.0).
    """

    negligible_threshold: float = 0.10
    low_threshold: float = 0.25
    moderate_threshold: float = 0.55
    high_threshold: float = 0.80

    unknown_risk_weight: float = 0.30
    volatility_weight: float = 0.30
    market_sensitivity_weight: float = 0.20
    technology_sensitivity_weight: float = 0.20

    known_risk_increment: float = 0.02
    known_risk_increment_cap: float = 0.20
    external_factor_increment: float = 0.02
    external_factor_increment_cap: float = 0.20

    horizon_propagation_multipliers: dict[ForecastHorizon, float] = field(
        default_factory=lambda: {
            ForecastHorizon.ONE_MONTH: 1.00,
            ForecastHorizon.THREE_MONTHS: 1.10,
            ForecastHorizon.SIX_MONTHS: 1.25,
            ForecastHorizon.ONE_YEAR: 1.45,
            ForecastHorizon.FIVE_YEARS: 1.90,
            ForecastHorizon.TEN_YEARS: 2.40,
        }
    )

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
                    f"Uncertainty level threshold {value!r} must be in (0.0, 1.0).",
                    field="threshold",
                    value=value,
                    engine=_ENGINE_NAME,
                )
        if not (
            thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]
        ):
            raise JanusValidationError(
                "Uncertainty level thresholds must be strictly increasing.",
                field="thresholds",
                value=thresholds,
                engine=_ENGINE_NAME,
            )

        weights = (
            self.unknown_risk_weight,
            self.volatility_weight,
            self.market_sensitivity_weight,
            self.technology_sensitivity_weight,
        )
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 1e-6:
            raise JanusValidationError(
                f"Uncertainty component weights must sum to 1.0, got {total_weight:.6f}.",
                field="weights",
                value=total_weight,
                engine=_ENGINE_NAME,
            )


# ---------------------------------------------------------------------------
# Statistical Utilities
# ---------------------------------------------------------------------------


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a value into [lower, upper]."""
    return max(lower, min(upper, value))


def _require_unit_score(value: float, field_name: str) -> float:
    """Validate that a score lies in [0.0, 1.0]; raise JanusInvalidScoreError otherwise."""
    if not 0.0 <= value <= 1.0:
        raise JanusInvalidScoreError(value, field_name, engine=_ENGINE_NAME)
    return value


# ---------------------------------------------------------------------------
# Uncertainty Engine
# ---------------------------------------------------------------------------


class UncertaintyEngine(IUncertaintyEngine):
    """
    Production implementation of the Uncertainty Engine.

    Thread-safe: all mutable state is guarded by an internal `threading.RLock`.
    Immutable model handling: stored `UncertaintyProfile` instances are never
    mutated in place. Updates are performed by constructing a new instance
    via `UncertaintyProfile.create(...)` and replacing the stored reference.
    """

    def __init__(self, config: Optional[UncertaintyEngineConfig] = None) -> None:
        self._config: UncertaintyEngineConfig = config or UncertaintyEngineConfig()
        self._lock = threading.RLock()
        self._initialized: bool = False
        self._shutdown: bool = False
        self._initialized_at: Optional[datetime] = None
        self._shutdown_at: Optional[datetime] = None

        self._profiles: dict[str, UncertaintyProfile] = {}

        self._profiles_created: int = 0
        self._validations_performed: int = 0
        self._aggregations_performed: int = 0
        self._propagations_performed: int = 0

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._initialized and not self._shutdown:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._profiles = {}
            self._profiles_created = 0
            self._validations_performed = 0
            self._aggregations_performed = 0
            self._propagations_performed = 0
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
    # Level Classification
    # -----------------------------------------------------------------

    def _compute_level(
        self,
        unknown_risk_exposure: float,
        volatility_score: float,
        market_sensitivity: float,
        technology_sensitivity: float,
        known_risk_count: int,
        external_factor_count: int,
    ) -> UncertaintyLevel:
        """
        Derive an UncertaintyLevel from raw component scores.

        The composite score is a weighted blend of the four normalized
        component scores, with small additive contributions from the
        number of identified known risks and external factors (more
        identified factors increase composite uncertainty, reflecting a
        more complex risk landscape — even though each individual factor
        is "known").
        """
        cfg = self._config

        composite = (
            unknown_risk_exposure * cfg.unknown_risk_weight
            + volatility_score * cfg.volatility_weight
            + market_sensitivity * cfg.market_sensitivity_weight
            + technology_sensitivity * cfg.technology_sensitivity_weight
        )

        known_risk_addend = min(
            known_risk_count * cfg.known_risk_increment,
            cfg.known_risk_increment_cap,
        )
        external_factor_addend = min(
            external_factor_count * cfg.external_factor_increment,
            cfg.external_factor_increment_cap,
        )

        composite = _clamp(composite + known_risk_addend + external_factor_addend)

        if composite <= cfg.negligible_threshold:
            return UncertaintyLevel.NEGLIGIBLE
        if composite <= cfg.low_threshold:
            return UncertaintyLevel.LOW
        if composite <= cfg.moderate_threshold:
            return UncertaintyLevel.MODERATE
        if composite <= cfg.high_threshold:
            return UncertaintyLevel.HIGH
        return UncertaintyLevel.EXTREME

    def classify_uncertainty_level(self, profile: UncertaintyProfile) -> UncertaintyLevel:
        with self._lock:
            self._ensure_operational()
            return self._compute_level(
                unknown_risk_exposure=profile.unknown_risk_exposure,
                volatility_score=profile.volatility_score,
                market_sensitivity=profile.market_sensitivity,
                technology_sensitivity=profile.technology_sensitivity,
                known_risk_count=len(profile.known_risks),
                external_factor_count=len(profile.external_factors),
            )

    # -----------------------------------------------------------------
    # Profile Creation / Retrieval
    # -----------------------------------------------------------------

    def create_uncertainty_profile(
        self, request: UncertaintyProfileCreateRequest
    ) -> UncertaintyProfileCreateResponse:
        with self._lock:
            self._ensure_operational()

            unknown_risk_exposure = _require_unit_score(
                request.unknown_risk_exposure, "unknown_risk_exposure"
            )
            volatility_score = _require_unit_score(
                request.volatility_score, "volatility_score"
            )
            market_sensitivity = _require_unit_score(
                request.market_sensitivity, "market_sensitivity"
            )
            technology_sensitivity = _require_unit_score(
                request.technology_sensitivity, "technology_sensitivity"
            )

            known_risks = list(request.known_risks)
            external_factors = list(request.external_factors)

            level = self._compute_level(
                unknown_risk_exposure=unknown_risk_exposure,
                volatility_score=volatility_score,
                market_sensitivity=market_sensitivity,
                technology_sensitivity=technology_sensitivity,
                known_risk_count=len(known_risks),
                external_factor_count=len(external_factors),
            )

            try:
                profile = UncertaintyProfile.create(
                    level=level,
                    known_risks=known_risks,
                    unknown_risk_exposure=unknown_risk_exposure,
                    volatility_score=volatility_score,
                    external_factors=external_factors,
                    market_sensitivity=market_sensitivity,
                    technology_sensitivity=technology_sensitivity,
                    notes=request.notes,
                )
            except ValueError as exc:
                raise JanusUncertaintyModelingError(
                    str(exc), context={"request": "create_uncertainty_profile"}
                ) from exc

            self._profiles[profile.uncertainty_id] = profile
            self._profiles_created += 1

            return UncertaintyProfileCreateResponse(uncertainty_profile=profile)

    def get_uncertainty_profile(
        self, request: UncertaintyProfileGetRequest
    ) -> UncertaintyProfileGetResponse:
        with self._lock:
            self._ensure_operational()
            profile = self._profiles.get(request.uncertainty_id)
            if profile is None:
                raise JanusUncertaintyProfileNotFoundError(request.uncertainty_id)
            return UncertaintyProfileGetResponse(uncertainty_profile=profile)

    # -----------------------------------------------------------------
    # Validation (Law 6 enforcement)
    # -----------------------------------------------------------------

    def validate_uncertainty(
        self, request: UncertaintyValidationRequest
    ) -> UncertaintyValidationResponse:
        with self._lock:
            self._ensure_operational()
            self._validations_performed += 1

            if request.uncertainty is None:
                raise JanusMissingUncertaintyError(
                    request.artifact_id, request.artifact_type
                )

            profile = request.uncertainty
            violations: list[str] = []

            is_trivial = (
                profile.level == UncertaintyLevel.NEGLIGIBLE
                and profile.unknown_risk_exposure == 0.0
                and profile.volatility_score == 0.0
                and profile.market_sensitivity == 0.0
                and profile.technology_sensitivity == 0.0
                and not profile.known_risks
                and not profile.external_factors
            )
            if is_trivial:
                violations.append(
                    "UncertaintyProfile is trivial (all components zero, "
                    "no known risks or external factors). JANUS Law 6 "
                    "requires forecasts to carry meaningful uncertainty, "
                    "not a placeholder absence of uncertainty."
                )

            computed_level = self.classify_uncertainty_level(profile)
            if computed_level != profile.level:
                violations.append(
                    f"UncertaintyProfile level '{profile.level.name}' is "
                    f"inconsistent with the level computed from its "
                    f"component scores ('{computed_level.name}')."
                )

            return UncertaintyValidationResponse(
                is_valid=not violations,
                violations=tuple(violations),
            )

    # -----------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------

    def aggregate_uncertainty(
        self, profiles: tuple[UncertaintyProfile, ...]
    ) -> UncertaintyProfile:
        with self._lock:
            self._ensure_operational()

            if not profiles:
                raise JanusUncertaintyModelingError(
                    "Cannot aggregate an empty set of UncertaintyProfiles.",
                    context={"profile_count": 0},
                )

            unknown_risk_exposure = _clamp(
                self._mean(p.unknown_risk_exposure for p in profiles)
            )
            volatility_score = _clamp(
                self._mean(p.volatility_score for p in profiles)
            )
            market_sensitivity = _clamp(
                self._mean(p.market_sensitivity for p in profiles)
            )
            technology_sensitivity = _clamp(
                self._mean(p.technology_sensitivity for p in profiles)
            )

            known_risks: list[str] = []
            external_factors: list[str] = []
            for p in profiles:
                for risk in p.known_risks:
                    if risk not in known_risks:
                        known_risks.append(risk)
                for factor in p.external_factors:
                    if factor not in external_factors:
                        external_factors.append(factor)

            level = self._compute_level(
                unknown_risk_exposure=unknown_risk_exposure,
                volatility_score=volatility_score,
                market_sensitivity=market_sensitivity,
                technology_sensitivity=technology_sensitivity,
                known_risk_count=len(known_risks),
                external_factor_count=len(external_factors),
            )

            notes = (
                f"Aggregated from {len(profiles)} UncertaintyProfile(s): "
                + ", ".join(p.uncertainty_id for p in profiles)
            )

            try:
                aggregated = UncertaintyProfile.create(
                    level=level,
                    known_risks=known_risks,
                    unknown_risk_exposure=unknown_risk_exposure,
                    volatility_score=volatility_score,
                    external_factors=external_factors,
                    market_sensitivity=market_sensitivity,
                    technology_sensitivity=technology_sensitivity,
                    notes=notes,
                )
            except ValueError as exc:
                raise JanusUncertaintyModelingError(
                    str(exc), context={"request": "aggregate_uncertainty"}
                ) from exc

            self._profiles[aggregated.uncertainty_id] = aggregated
            self._aggregations_performed += 1

            return aggregated

    # -----------------------------------------------------------------
    # Propagation
    # -----------------------------------------------------------------

    def propagate_uncertainty(
        self,
        profile: UncertaintyProfile,
        horizon: ForecastHorizon,
        additional_unknown_risk_exposure: float = 0.0,
        additional_known_risks: Optional[tuple[str, ...]] = None,
    ) -> UncertaintyProfile:
        """
        Propagate an UncertaintyProfile forward across a forecast horizon.

        Uncertainty compounds with time: a profile describing the present
        carries proportionally more unknown-risk exposure and volatility
        when projected to a more distant horizon. The compounding model is:

            new_value = 1 - (1 - old_value) ** multiplier

        where `multiplier >= 1.0` for horizons further than the baseline
        (one month) and `multiplier == 1.0` leaves the profile unchanged.

        Sensitivity scores (market, technology) are preserved unless an
        explicit additional exposure is supplied, since sensitivity to
        external factors is a structural property rather than a function
        of elapsed time.
        """
        with self._lock:
            self._ensure_operational()

            additional_unknown_risk_exposure = _require_unit_score(
                additional_unknown_risk_exposure, "additional_unknown_risk_exposure"
            )

            multiplier = self._config.horizon_propagation_multipliers.get(horizon)
            if multiplier is None:
                raise JanusUncertaintyModelingError(
                    f"No propagation multiplier configured for horizon "
                    f"'{horizon.value}'.",
                    context={"horizon": horizon.value},
                )

            propagated_volatility = _clamp(
                1.0 - (1.0 - profile.volatility_score) ** multiplier
            )

            base_unknown = _clamp(
                1.0 - (1.0 - profile.unknown_risk_exposure) ** multiplier
            )
            propagated_unknown_risk = _clamp(
                base_unknown + additional_unknown_risk_exposure
            )

            known_risks = list(profile.known_risks)
            for risk in additional_known_risks or ():
                if risk not in known_risks:
                    known_risks.append(risk)

            level = self._compute_level(
                unknown_risk_exposure=propagated_unknown_risk,
                volatility_score=propagated_volatility,
                market_sensitivity=profile.market_sensitivity,
                technology_sensitivity=profile.technology_sensitivity,
                known_risk_count=len(known_risks),
                external_factor_count=len(profile.external_factors),
            )

            notes = (
                f"Propagated from UncertaintyProfile '{profile.uncertainty_id}' "
                f"to horizon '{horizon.label}' (multiplier={multiplier:.3f})."
            )

            try:
                propagated = UncertaintyProfile.create(
                    level=level,
                    known_risks=known_risks,
                    unknown_risk_exposure=propagated_unknown_risk,
                    volatility_score=propagated_volatility,
                    external_factors=list(profile.external_factors),
                    market_sensitivity=profile.market_sensitivity,
                    technology_sensitivity=profile.technology_sensitivity,
                    notes=notes,
                )
            except ValueError as exc:
                raise JanusUncertaintyModelingError(
                    str(exc), context={"request": "propagate_uncertainty"}
                ) from exc

            self._profiles[propagated.uncertainty_id] = propagated
            self._propagations_performed += 1

            return propagated

    # -----------------------------------------------------------------
    # Volatility Analysis
    # -----------------------------------------------------------------

    def analyze_volatility(self, historical_values: tuple[float, ...]) -> float:
        """
        Compute a normalized volatility score (0.0-1.0) from a series of
        historical numeric observations using the coefficient of variation
        (population standard deviation divided by the mean of absolute
        values), clamped to [0.0, 1.0].

        Fewer than two observations yields a volatility score of 0.0
        (no variation can be observed).
        """
        with self._lock:
            self._ensure_operational()

            if len(historical_values) < 2:
                return 0.0

            mean_value = statistics.mean(historical_values)
            stdev_value = statistics.pstdev(historical_values)

            if mean_value == 0.0:
                denominator = statistics.mean(abs(v) for v in historical_values)
                if denominator == 0.0:
                    return 0.0
                coefficient = stdev_value / denominator
            else:
                coefficient = stdev_value / abs(mean_value)

            return _clamp(coefficient)

    # -----------------------------------------------------------------
    # Unknown-Risk Modeling
    # -----------------------------------------------------------------

    def model_unknown_risk_exposure(
        self,
        known_risk_count: int,
        evidence_strength: float,
        domain_volatility: float,
    ) -> float:
        """
        Estimate unknown-risk exposure (0.0-1.0) for a domain.

        The model reflects three intuitions:
          - Weaker evidence implies a larger space of un-modeled risks.
          - Higher domain volatility implies a larger space of un-modeled
            risks.
          - Each additional *known* risk slightly reduces unknown-risk
            exposure, since cataloguing risks narrows the unknown space —
            but this reduction is capped, since cataloguing known risks
            does not guarantee completeness.
        """
        with self._lock:
            self._ensure_operational()

            if known_risk_count < 0:
                raise JanusInvalidScoreError(
                    float(known_risk_count), "known_risk_count", engine=_ENGINE_NAME
                )
            evidence_strength = _require_unit_score(evidence_strength, "evidence_strength")
            domain_volatility = _require_unit_score(domain_volatility, "domain_volatility")

            evidence_gap = 1.0 - evidence_strength
            cfg = self._config

            known_risk_reduction = min(
                known_risk_count * cfg.known_risk_increment,
                cfg.known_risk_increment_cap,
            )

            exposure = _clamp(
                (0.6 * evidence_gap) + (0.4 * domain_volatility) - known_risk_reduction
            )

            return exposure

    # -----------------------------------------------------------------
    # Confidence Estimation & Scoring
    # -----------------------------------------------------------------

    def estimate_confidence(
        self,
        data_quality: float,
        model_fit: float,
        signal_strength: float,
        notes: str = "",
    ) -> ConfidenceProfile:
        """
        Estimate an overall ConfidenceProfile from its three components.

        Overall confidence is a weighted blend, reflecting that data
        quality is the dominant determinant of confidence, followed by
        model fit, followed by signal strength.
        """
        with self._lock:
            self._ensure_operational()

            data_quality = _require_unit_score(data_quality, "data_quality")
            model_fit = _require_unit_score(model_fit, "model_fit")
            signal_strength = _require_unit_score(signal_strength, "signal_strength")

            overall = _clamp(
                (0.45 * data_quality) + (0.35 * model_fit) + (0.20 * signal_strength)
            )

            try:
                return ConfidenceProfile.create(
                    overall=overall,
                    data_quality=data_quality,
                    model_fit=model_fit,
                    signal_strength=signal_strength,
                    notes=notes,
                )
            except ValueError as exc:
                raise JanusUncertaintyModelingError(
                    str(exc), context={"request": "estimate_confidence"}
                ) from exc

    def score_confidence(self, profile: ConfidenceProfile) -> ProbabilityLevel:
        """Classify a ConfidenceProfile's overall score into a ProbabilityLevel."""
        with self._lock:
            self._ensure_operational()
            return ProbabilityLevel.from_float(profile.overall)

    # -----------------------------------------------------------------
    # Statistical Utilities
    # -----------------------------------------------------------------

    @staticmethod
    def _mean(values: "list[float] | tuple[float, ...] | object") -> float:
        values_list = list(values)
        if not values_list:
            return 0.0
        return statistics.mean(values_list)

    @staticmethod
    def compute_mean(values: tuple[float, ...]) -> float:
        if not values:
            raise JanusUncertaintyModelingError(
                "Cannot compute mean of an empty value set.",
                context={"operation": "compute_mean"},
            )
        return statistics.mean(values)

    @staticmethod
    def compute_population_stdev(values: tuple[float, ...]) -> float:
        if len(values) < 1:
            raise JanusUncertaintyModelingError(
                "Cannot compute standard deviation of an empty value set.",
                context={"operation": "compute_population_stdev"},
            )
        if len(values) == 1:
            return 0.0
        return statistics.pstdev(values)

    @staticmethod
    def compute_variance(values: tuple[float, ...]) -> float:
        if len(values) < 1:
            raise JanusUncertaintyModelingError(
                "Cannot compute variance of an empty value set.",
                context={"operation": "compute_variance"},
            )
        if len(values) == 1:
            return 0.0
        return statistics.pvariance(values)

    @staticmethod
    def normalize(values: tuple[float, ...]) -> tuple[float, ...]:
        """Normalize a tuple of non-negative values so they sum to 1.0."""
        if not values:
            raise JanusUncertaintyModelingError(
                "Cannot normalize an empty value set.",
                context={"operation": "normalize"},
            )
        for value in values:
            if value < 0.0:
                raise JanusInvalidScoreError(value, "normalize_input", engine=_ENGINE_NAME)
        total = sum(values)
        if total == 0.0:
            raise JanusUncertaintyModelingError(
                "Cannot normalize a value set that sums to zero.",
                context={"operation": "normalize"},
            )
        return tuple(v / total for v in values)

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
                "stored_profile_count": len(self._profiles),
                "profiles_created": self._profiles_created,
                "validations_performed": self._validations_performed,
                "aggregations_performed": self._aggregations_performed,
                "propagations_performed": self._propagations_performed,
                "config": {
                    "negligible_threshold": self._config.negligible_threshold,
                    "low_threshold": self._config.low_threshold,
                    "moderate_threshold": self._config.moderate_threshold,
                    "high_threshold": self._config.high_threshold,
                    "unknown_risk_weight": self._config.unknown_risk_weight,
                    "volatility_weight": self._config.volatility_weight,
                    "market_sensitivity_weight": self._config.market_sensitivity_weight,
                    "technology_sensitivity_weight": self._config.technology_sensitivity_weight,
                },
            }

    def get_statistics(self) -> dict[str, object]:
        """Return summary statistics for stored uncertainty profiles."""
        with self._lock:
            return {
                "total_profiles": len(self._profiles),
                "profiles_created": self._profiles_created,
                "validations_performed": self._validations_performed,
                "aggregations_performed": self._aggregations_performed,
                "propagations_performed": self._propagations_performed,
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

            return {
                "engine_name": _ENGINE_NAME,
                "engine_version": _ENGINE_VERSION,
                "status": status,
                "checked_at": datetime.utcnow().isoformat(),
                "stored_profile_count": len(self._profiles),
            }