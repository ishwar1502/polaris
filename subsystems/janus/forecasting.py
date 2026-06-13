"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/forecasting.py

Implements ForecastingEngine — the Forecasting Engine for JANUS.

Responsibilities:
  - Forecast creation from uncertainty, evidence, and historical context
  - Forecast management (retrieval, listing, supersession)
  - Forecast assessment and accuracy tracking
  - Forecast confidence management
  - Forecast statistics and health reporting

Constitutional boundaries enforced:
  - All forecasts carry an UncertaintyProfile (Law 6 — mandatory).
  - No certainty claims ever (Law 6).
  - JANUS forecasts outcomes; ODYSSEY chooses direction (Law 2).
  - JANUS never selects, decides, or plans.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Optional

from ..models import (
    ConfidenceProfile,
    EvidenceProfile,
    Forecast,
    ForecastAssessment,
    ForecastHorizon,
    ForecastMetadata,
    ForecastType,
    ProbabilityDistribution,
    ProbabilityLevel,
    UncertaintyLevel,
    UncertaintyProfile,
)
from ..exceptions import (
    JanusAlreadyInitializedError,
    JanusForecastAssessmentError,
    JanusForecastGenerationError,
    JanusForecastHorizonError,
    JanusForecastNotFoundError,
    JanusForecastSupersededError,
    JanusMissingRequiredFieldError,
    JanusMissingUncertaintyError,
    JanusNotInitializedError,
    JanusShutdownError,
    JanusValidationError,
)
from ..interfaces import IForecastingEngine
from ..schemas import (
    ForecastAssessRequest,
    ForecastAssessResponse,
    ForecastGenerateRequest,
    ForecastGenerateResponse,
    ForecastGetRequest,
    ForecastGetResponse,
    ForecastListByHorizonRequest,
    ForecastListByHorizonResponse,
)

_ENGINE_NAME    = "ForecastingEngine"
_ENGINE_VERSION = "5.1.0"

# Minimum evidence-strength threshold required to emit a forecast.
_MIN_EVIDENCE_STRENGTH: float = 0.05

# Statuses stored directly on the Forecast.status string field.
_STATUS_ACTIVE      = "active"
_STATUS_SUPERSEDED  = "superseded"
_STATUS_ARCHIVED    = "archived"


class ForecastingEngine(IForecastingEngine):
    """
    Thread-safe implementation of IForecastingEngine.

    Internal storage:
      _forecasts     — forecast_id → Forecast
      _assessments   — assessment_id → ForecastAssessment
      _accuracy_feed — forecast_id → list[float]  (resolved accuracy history)

    Lifecycle:
        engine = ForecastingEngine()
        engine.initialize()
        # … operational calls …
        engine.shutdown()
    """

    def __init__(self) -> None:
        self._forecasts:     dict[str, Forecast]           = {}
        self._assessments:   dict[str, ForecastAssessment] = {}
        self._accuracy_feed: dict[str, list[float]]        = {}

        self._lock:          threading.RLock = threading.RLock()
        self._initialized:   bool = False
        self._shutdown_flag: bool = False

        # Diagnostics
        self._total_created:    int = 0
        self._total_assessed:   int = 0
        self._total_superseded: int = 0
        self._created_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._forecasts.clear()
            self._assessments.clear()
            self._accuracy_feed.clear()
            self._total_created    = 0
            self._total_assessed   = 0
            self._total_superseded = 0
            self._created_at       = datetime.utcnow()
            self._initialized      = True

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_flag = True
            self._initialized   = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized and not self._shutdown_flag

    @property
    def engine_name(self) -> str:
        return _ENGINE_NAME

    @property
    def engine_version(self) -> str:
        return _ENGINE_VERSION

    # ------------------------------------------------------------------
    # Guard helpers
    # ------------------------------------------------------------------

    def _require_ready(self) -> None:
        if self._shutdown_flag:
            raise JanusShutdownError(_ENGINE_NAME)
        if not self._initialized:
            raise JanusNotInitializedError(_ENGINE_NAME)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_generate_request(self, request: ForecastGenerateRequest) -> None:
        if not request.title.strip():
            raise JanusMissingRequiredFieldError("title", engine=_ENGINE_NAME)
        if not request.description.strip():
            raise JanusMissingRequiredFieldError("description", engine=_ENGINE_NAME)
        if request.forecast_type is None:
            raise JanusMissingRequiredFieldError("forecast_type", engine=_ENGINE_NAME)
        if request.horizon is None:
            raise JanusForecastHorizonError(None)
        if not isinstance(request.horizon, ForecastHorizon):
            raise JanusForecastHorizonError(request.horizon)
        if request.uncertainty is None:
            raise JanusMissingUncertaintyError("<pending>", "Forecast")
        if request.evidence is None:
            raise JanusMissingRequiredFieldError("evidence", engine=_ENGINE_NAME)
        if request.evidence.evidence_strength < _MIN_EVIDENCE_STRENGTH:
            raise JanusValidationError(
                f"evidence_strength {request.evidence.evidence_strength!r} is below "
                f"the minimum threshold of {_MIN_EVIDENCE_STRENGTH}. "
                "All JANUS forecasts must be evidence-based.",
                field="evidence_strength",
                engine=_ENGINE_NAME,
            )
        if request.metadata is None:
            raise JanusMissingRequiredFieldError("metadata", engine=_ENGINE_NAME)

    def _validate_uncertainty_not_trivial(
        self,
        uncertainty: UncertaintyProfile,
        artifact_id: str,
    ) -> None:
        """
        Law 6: All forecasts require uncertainty. No certainty claims ever.

        A trivial uncertainty profile is one where every numeric field is 0.0
        and the level is NEGLIGIBLE, which would constitute a false certainty claim.
        """
        is_trivial = (
            uncertainty.level == UncertaintyLevel.NEGLIGIBLE
            and uncertainty.unknown_risk_exposure == 0.0
            and uncertainty.volatility_score == 0.0
            and uncertainty.market_sensitivity == 0.0
            and uncertainty.technology_sensitivity == 0.0
            and not uncertainty.known_risks
            and not uncertainty.external_factors
        )
        if is_trivial:
            raise JanusMissingUncertaintyError(artifact_id, "Forecast")

    # ------------------------------------------------------------------
    # Probability distribution construction
    # ------------------------------------------------------------------

    def _build_default_distribution(
        self,
        forecast_type: ForecastType,
        uncertainty: UncertaintyProfile,
        confidence: ConfidenceProfile,
    ) -> ProbabilityDistribution:
        """
        Build a default ProbabilityDistribution calibrated to the given
        ForecastType and UncertaintyProfile.

        The distribution is deliberately non-certain: probabilities reflect
        the uncertainty level and confidence so that Law 6 is respected.
        """
        level = uncertainty.level
        base_prob   = confidence.overall  # dominant outcome probability

        # Uncertainty level dampens the dominant outcome probability.
        dampeners = {
            UncertaintyLevel.NEGLIGIBLE: 0.00,
            UncertaintyLevel.LOW:        0.10,
            UncertaintyLevel.MODERATE:   0.20,
            UncertaintyLevel.HIGH:       0.30,
            UncertaintyLevel.EXTREME:    0.40,
        }
        dampener = dampeners.get(level, 0.20)
        dominant = max(0.10, min(0.90, base_prob - dampener))

        # Distribute remainder across three outcome buckets.
        remainder = 1.0 - dominant
        alt_a = round(remainder * 0.50, 6)
        alt_b = round(remainder * 0.30, 6)
        alt_c = round(1.0 - dominant - alt_a - alt_b, 6)  # absorb rounding

        outcomes: dict[str, float] = {
            f"{forecast_type.name}_outcome_primary":   dominant,
            f"{forecast_type.name}_outcome_alternate":  alt_a,
            f"{forecast_type.name}_outcome_pessimistic": alt_b,
            f"{forecast_type.name}_outcome_tail":       alt_c,
        }

        return ProbabilityDistribution.create(
            label=f"{forecast_type.name} distribution",
            outcomes=outcomes,
            uncertainty_level=level,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # IForecastingEngine — public interface
    # ------------------------------------------------------------------

    def generate_forecast(
        self, request: ForecastGenerateRequest
    ) -> ForecastGenerateResponse:
        """
        Generate a probabilistic Forecast.

        Constitutional enforcement:
          - Uncertainty is mandatory (Law 6).
          - No certainty claims (Law 6).
          - JANUS does not select outcomes; it forecasts probabilities.
        """
        self._require_ready()
        self._validate_generate_request(request)

        # Build a working confidence profile from metadata if none exists.
        confidence = ConfidenceProfile.create(
            overall=max(0.10, min(0.90, request.evidence.evidence_strength)),
            data_quality=max(0.10, request.evidence.evidence_strength),
            model_fit=0.6,
            signal_strength=max(0.10, request.evidence.evidence_strength * 0.8),
            notes=f"Auto-derived for {request.forecast_type.name} forecast.",
        )

        probability_distribution = self._build_default_distribution(
            forecast_type=request.forecast_type,
            uncertainty=request.uncertainty,
            confidence=confidence,
        )

        forecast = Forecast.create(
            title=request.title,
            description=request.description,
            forecast_type=request.forecast_type,
            horizon=request.horizon,
            probability_distribution=probability_distribution,
            uncertainty=request.uncertainty,
            confidence=confidence,
            evidence=request.evidence,
            metadata=request.metadata,
        )

        # Validate no false certainty post-construction.
        self._validate_uncertainty_not_trivial(request.uncertainty, forecast.forecast_id)

        with self._lock:
            self._forecasts[forecast.forecast_id] = forecast
            self._accuracy_feed[forecast.forecast_id] = []
            self._total_created += 1

        return ForecastGenerateResponse(
            forecast=forecast,
            generated_at=datetime.utcnow(),
            engine_version=_ENGINE_VERSION,
        )

    def get_forecast(self, request: ForecastGetRequest) -> ForecastGetResponse:
        self._require_ready()
        if not request.forecast_id.strip():
            raise JanusMissingRequiredFieldError("forecast_id", engine=_ENGINE_NAME)

        with self._lock:
            forecast = self._forecasts.get(request.forecast_id)
            if forecast is None:
                raise JanusForecastNotFoundError(request.forecast_id)
            return ForecastGetResponse(forecast=forecast, retrieved_at=datetime.utcnow())

    def assess_forecast(
        self, request: ForecastAssessRequest
    ) -> ForecastAssessResponse:
        """
        Assess an existing Forecast for accuracy.

        Records the assessment with deviation notes and optional accuracy score.
        If revision_required is True, the caller should subsequently call
        generate_forecast() with revised inputs and supersede_forecast().
        """
        self._require_ready()

        if not request.forecast_id.strip():
            raise JanusMissingRequiredFieldError("forecast_id", engine=_ENGINE_NAME)
        if not request.assessor.strip():
            raise JanusMissingRequiredFieldError("assessor", engine=_ENGINE_NAME)
        if request.accuracy_score is not None and not (0.0 <= request.accuracy_score <= 1.0):
            raise JanusValidationError(
                f"accuracy_score {request.accuracy_score!r} must be in [0.0, 1.0].",
                field="accuracy_score",
                engine=_ENGINE_NAME,
            )

        with self._lock:
            forecast = self._forecasts.get(request.forecast_id)
            if forecast is None:
                raise JanusForecastNotFoundError(request.forecast_id)

            if forecast.status == _STATUS_SUPERSEDED:
                raise JanusForecastSupersededError(
                    request.forecast_id,
                    superseded_by=request.superseded_by or "<unknown>",
                )

            try:
                assessment = ForecastAssessment.create(
                    forecast=forecast,
                    assessor=request.assessor,
                    accuracy_score=request.accuracy_score,
                    deviation_notes=request.deviation_notes,
                    revision_required=request.revision_required,
                    superseded_by=request.superseded_by,
                )
            except Exception as exc:
                raise JanusForecastAssessmentError(request.forecast_id, str(exc)) from exc

            self._assessments[assessment.assessment_id] = assessment
            self._total_assessed += 1

            # Track accuracy feed.
            if request.accuracy_score is not None:
                self._accuracy_feed.setdefault(request.forecast_id, []).append(
                    request.accuracy_score
                )

            # If superseded_by is provided, mark the forecast status accordingly.
            if request.superseded_by:
                forecast.status     = _STATUS_SUPERSEDED
                forecast.updated_at = datetime.utcnow()
                self._total_superseded += 1

        return ForecastAssessResponse(
            assessment=assessment,
            assessed_at=assessment.assessed_at,
        )

    def list_forecasts_by_horizon(
        self, request: ForecastListByHorizonRequest
    ) -> ForecastListByHorizonResponse:
        self._require_ready()
        if request.horizon is None:
            raise JanusForecastHorizonError(None)

        with self._lock:
            results: list[Forecast] = []
            for forecast in self._forecasts.values():
                if forecast.horizon != request.horizon:
                    continue
                if not request.include_superseded and forecast.status == _STATUS_SUPERSEDED:
                    continue
                if not request.include_archived and forecast.status == _STATUS_ARCHIVED:
                    continue
                results.append(forecast)

        return ForecastListByHorizonResponse(
            forecasts=tuple(results),
            horizon=request.horizon,
            retrieved_at=datetime.utcnow(),
        )

    def supersede_forecast(
        self, forecast_id: str, replacement_forecast_id: str, reason: str
    ) -> Forecast:
        """
        Mark an existing Forecast as superseded by a newer forecast.
        The replacement forecast must already exist in the registry.
        """
        self._require_ready()
        if not forecast_id.strip():
            raise JanusMissingRequiredFieldError("forecast_id", engine=_ENGINE_NAME)
        if not replacement_forecast_id.strip():
            raise JanusMissingRequiredFieldError("replacement_forecast_id", engine=_ENGINE_NAME)
        if not reason.strip():
            raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)

        with self._lock:
            forecast = self._forecasts.get(forecast_id)
            if forecast is None:
                raise JanusForecastNotFoundError(forecast_id)

            replacement = self._forecasts.get(replacement_forecast_id)
            if replacement is None:
                raise JanusForecastNotFoundError(replacement_forecast_id)

            if forecast.status == _STATUS_SUPERSEDED:
                raise JanusForecastSupersededError(
                    forecast_id, superseded_by=replacement_forecast_id
                )

            forecast.status     = _STATUS_SUPERSEDED
            forecast.updated_at = datetime.utcnow()
            self._total_superseded += 1

        return forecast

    def list_forecasts_by_type(
        self, forecast_type: ForecastType
    ) -> tuple[Forecast, ...]:
        self._require_ready()
        with self._lock:
            return tuple(
                f for f in self._forecasts.values()
                if f.forecast_type == forecast_type and f.status == _STATUS_ACTIVE
            )

    # ------------------------------------------------------------------
    # Extended public API (beyond interface minimum)
    # ------------------------------------------------------------------

    def register_forecast(self, forecast: Forecast) -> None:
        """
        Register an externally-constructed Forecast directly.

        Used when a Forecast is assembled by an upstream engine
        (e.g. StrategicForecastEngine) and needs to enter the registry.
        """
        self._require_ready()
        if forecast is None:
            raise JanusMissingRequiredFieldError("forecast", engine=_ENGINE_NAME)
        if forecast.uncertainty is None:
            raise JanusMissingUncertaintyError(forecast.forecast_id, "Forecast")
        self._validate_uncertainty_not_trivial(forecast.uncertainty, forecast.forecast_id)

        with self._lock:
            if forecast.forecast_id in self._forecasts:
                raise JanusForecastGenerationError(
                    f"Forecast '{forecast.forecast_id}' is already registered.",
                    horizon=forecast.horizon.value,
                )
            self._forecasts[forecast.forecast_id] = forecast
            self._accuracy_feed[forecast.forecast_id] = []
            self._total_created += 1

    def update_forecast(self, forecast: Forecast, reason: str) -> Forecast:
        """
        Replace a stored Forecast with an updated version.

        The forecast_id must match an existing entry.  The updated forecast
        must still carry a valid UncertaintyProfile.
        """
        self._require_ready()
        if not reason.strip():
            raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)
        if forecast.uncertainty is None:
            raise JanusMissingUncertaintyError(forecast.forecast_id, "Forecast")

        with self._lock:
            if forecast.forecast_id not in self._forecasts:
                raise JanusForecastNotFoundError(forecast.forecast_id)
            forecast.updated_at = datetime.utcnow()
            self._forecasts[forecast.forecast_id] = forecast

        return forecast

    def archive_forecast(self, forecast_id: str, reason: str, archived_by: str) -> Forecast:
        """
        Mark a Forecast as archived.  Only active forecasts may be archived.
        """
        self._require_ready()
        if not forecast_id.strip():
            raise JanusMissingRequiredFieldError("forecast_id", engine=_ENGINE_NAME)
        if not reason.strip():
            raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)
        if not archived_by.strip():
            raise JanusMissingRequiredFieldError("archived_by", engine=_ENGINE_NAME)

        with self._lock:
            forecast = self._forecasts.get(forecast_id)
            if forecast is None:
                raise JanusForecastNotFoundError(forecast_id)
            if forecast.status == _STATUS_SUPERSEDED:
                raise JanusForecastSupersededError(forecast_id, superseded_by="<unknown>")

            forecast.status     = _STATUS_ARCHIVED
            forecast.updated_at = datetime.utcnow()

        return forecast

    def get_assessment(self, assessment_id: str) -> ForecastAssessment:
        """Retrieve a ForecastAssessment by its assessment_id."""
        self._require_ready()
        with self._lock:
            assessment = self._assessments.get(assessment_id)
            if assessment is None:
                raise JanusForecastAssessmentError(
                    assessment_id, "Assessment not found."
                )
        return assessment

    def list_assessments_for_forecast(
        self, forecast_id: str
    ) -> tuple[ForecastAssessment, ...]:
        """Return all ForecastAssessments for a given forecast_id."""
        self._require_ready()
        with self._lock:
            return tuple(
                a for a in self._assessments.values()
                if a.forecast.forecast_id == forecast_id
            )

    def average_accuracy(self, forecast_id: str) -> Optional[float]:
        """
        Return the mean accuracy score for a forecast across all resolved
        assessments, or None if no accuracy data exists.
        """
        self._require_ready()
        with self._lock:
            scores = self._accuracy_feed.get(forecast_id, [])
        if not scores:
            return None
        return sum(scores) / len(scores)

    def list_forecasts(
        self,
        *,
        forecast_type: Optional[ForecastType] = None,
        horizon: Optional[ForecastHorizon] = None,
        include_superseded: bool = False,
        include_archived: bool = False,
    ) -> tuple[Forecast, ...]:
        """
        Return all Forecasts, optionally filtered by type, horizon, and status.
        """
        self._require_ready()
        with self._lock:
            results: list[Forecast] = list(self._forecasts.values())

        if forecast_type is not None:
            results = [f for f in results if f.forecast_type == forecast_type]
        if horizon is not None:
            results = [f for f in results if f.horizon == horizon]
        if not include_superseded:
            results = [f for f in results if f.status != _STATUS_SUPERSEDED]
        if not include_archived:
            results = [f for f in results if f.status != _STATUS_ARCHIVED]

        return tuple(results)

    def forecast_exists(self, forecast_id: str) -> bool:
        """Return True if the forecast_id is registered."""
        self._require_ready()
        with self._lock:
            return forecast_id in self._forecasts

    # ------------------------------------------------------------------
    # Statistics and health
    # ------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        """Return a snapshot of engine statistics for monitoring."""
        self._require_ready()
        with self._lock:
            total      = len(self._forecasts)
            by_type:   dict[str, int] = {}
            by_status: dict[str, int] = {}
            by_horizon: dict[str, int] = {}
            for f in self._forecasts.values():
                by_type[f.forecast_type.name]    = by_type.get(f.forecast_type.name, 0) + 1
                by_status[f.status]              = by_status.get(f.status, 0) + 1
                by_horizon[f.horizon.value]      = by_horizon.get(f.horizon.value, 0) + 1

        return {
            "engine":               _ENGINE_NAME,
            "version":              _ENGINE_VERSION,
            "initialized":          self._initialized,
            "total_registered":     total,
            "total_created":        self._total_created,
            "total_assessed":       self._total_assessed,
            "total_superseded":     self._total_superseded,
            "assessments_stored":   len(self._assessments),
            "forecasts_by_type":    by_type,
            "forecasts_by_status":  by_status,
            "forecasts_by_horizon": by_horizon,
            "engine_created_at":    self._created_at.isoformat() if self._created_at else None,
        }

    def health(self) -> dict[str, Any]:
        """Return engine health report."""
        is_healthy = self._initialized and not self._shutdown_flag
        return {
            "engine":           _ENGINE_NAME,
            "version":          _ENGINE_VERSION,
            "healthy":          is_healthy,
            "initialized":      self._initialized,
            "shutdown":         self._shutdown_flag,
            "forecast_count":   len(self._forecasts) if is_healthy else 0,
            "assessment_count": len(self._assessments) if is_healthy else 0,
        }
