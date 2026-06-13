"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/future_modeling.py

Implements FutureModelingEngine — the Future Modeling Engine for JANUS.

Responsibilities:
  - Future model creation spanning multiple time horizons
  - Future state management (create, retrieve, update)
  - Future model retrieval and listing
  - Future state projection and horizon queries
  - Future model analysis (horizon coverage, risk/opportunity summaries)
  - Model statistics and health reporting

Constitutional boundaries enforced:
  - JANUS constructs FutureModels; VEGA selects futures; ZENITH plans.
  - This engine never selects or approves a future model.
  - All models carry UncertaintyProfile (Law 6).
  - CHRONOS owns time; JANUS predicts future states across time.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Optional

from ..models import (
    ConfidenceProfile,
    EvidenceProfile,
    ForecastHorizon,
    FutureModel,
    FutureState,
    OpportunityAssessment,
    ProbabilityLevel,
    RiskAssessment,
    RiskLevel,
    OpportunityLevel,
    UncertaintyLevel,
    UncertaintyProfile,
)
from ..exceptions import (
    JanusAlreadyInitializedError,
    JanusFutureModelConstructionError,
    JanusFutureModelHorizonConflictError,
    JanusFutureModelNotFoundError,
    JanusFutureStateError,
    JanusMissingRequiredFieldError,
    JanusMissingUncertaintyError,
    JanusNotInitializedError,
    JanusShutdownError,
    JanusValidationError,
)
from ..interfaces import IFutureModelingEngine
from ..schemas import (
    FutureModelConstructRequest,
    FutureModelConstructResponse,
    FutureModelGetRequest,
    FutureModelGetResponse,
    FutureStateQueryRequest,
    FutureStateQueryResponse,
)

_ENGINE_NAME    = "FutureModelingEngine"
_ENGINE_VERSION = "5.1.0"

# Maximum number of FutureStates with the same horizon within a single model.
# Multiple states per horizon are valid (e.g. optimistic vs pessimistic at 1 year).
_MAX_STATES_PER_HORIZON = 20


class FutureModelingEngine(IFutureModelingEngine):
    """
    Thread-safe implementation of IFutureModelingEngine.

    Internal storage:
      _models       — model_id  → FutureModel
      _state_index  — state_id  → (model_id, FutureState)  for O(1) state lookup

    Lifecycle:
        engine = FutureModelingEngine()
        engine.initialize()
        # … operational calls …
        engine.shutdown()
    """

    def __init__(self) -> None:
        self._models:      dict[str, FutureModel]                   = {}
        self._state_index: dict[str, tuple[str, FutureState]]       = {}

        self._lock:          threading.RLock = threading.RLock()
        self._initialized:   bool = False
        self._shutdown_flag: bool = False

        # Diagnostics
        self._total_models_created: int = 0
        self._total_states_created: int = 0
        self._total_models_updated: int = 0
        self._created_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._models.clear()
            self._state_index.clear()
            self._total_models_created = 0
            self._total_states_created = 0
            self._total_models_updated = 0
            self._created_at           = datetime.utcnow()
            self._initialized          = True

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

    def _validate_construct_request(self, request: FutureModelConstructRequest) -> None:
        if not request.title.strip():
            raise JanusMissingRequiredFieldError("title", engine=_ENGINE_NAME)
        if not request.context.strip():
            raise JanusMissingRequiredFieldError("context", engine=_ENGINE_NAME)
        if not request.future_states:
            raise JanusValidationError(
                "At least one FutureState is required to construct a FutureModel.",
                field="future_states",
                engine=_ENGINE_NAME,
            )
        if request.uncertainty is None:
            raise JanusMissingUncertaintyError("<pending>", "FutureModel")
        if request.confidence is None:
            raise JanusMissingRequiredFieldError("confidence", engine=_ENGINE_NAME)
        if request.evidence is None:
            raise JanusMissingRequiredFieldError("evidence", engine=_ENGINE_NAME)

    def _validate_future_state(self, state: FutureState) -> None:
        if not state.state_id.strip():
            raise JanusFutureStateError("", "state_id is empty.")
        if not state.label.strip():
            raise JanusFutureStateError(state.state_id, "label is empty.")
        if not (0.0 <= state.probability <= 1.0):
            raise JanusFutureStateError(
                state.state_id,
                f"probability {state.probability!r} must be in [0.0, 1.0].",
            )
        if state.uncertainty is None:
            raise JanusFutureStateError(state.state_id, "uncertainty is None.")
        if state.horizon is None:
            raise JanusFutureStateError(state.state_id, "horizon is None.")

    def _validate_states_collection(
        self,
        states: tuple[FutureState, ...] | list[FutureState],
        model_id: str,
    ) -> None:
        """Validate all states and check for excessive per-horizon concentration."""
        horizon_counts: dict[ForecastHorizon, int] = {}
        for state in states:
            self._validate_future_state(state)
            horizon_counts[state.horizon] = horizon_counts.get(state.horizon, 0) + 1

        for horizon, count in horizon_counts.items():
            if count > _MAX_STATES_PER_HORIZON:
                raise JanusFutureModelHorizonConflictError(model_id, horizon.value)

    def _validate_model_integrity(self, model: FutureModel) -> None:
        if not model.model_id.strip():
            raise JanusFutureModelConstructionError("model_id is empty.")
        if not model.title.strip():
            raise JanusMissingRequiredFieldError("title", engine=_ENGINE_NAME)
        if model.uncertainty is None:
            raise JanusMissingUncertaintyError(model.model_id, "FutureModel")
        if model.confidence is None:
            raise JanusMissingRequiredFieldError("confidence", engine=_ENGINE_NAME)
        if model.evidence is None:
            raise JanusMissingRequiredFieldError("evidence", engine=_ENGINE_NAME)
        self._validate_states_collection(model.future_states, model.model_id)

    # ------------------------------------------------------------------
    # State index management
    # ------------------------------------------------------------------

    def _index_states(self, model_id: str, states: list[FutureState]) -> None:
        """Add states to the cross-model state index."""
        for state in states:
            self._state_index[state.state_id] = (model_id, state)

    def _deindex_states(self, states: list[FutureState]) -> None:
        """Remove states from the cross-model state index."""
        for state in states:
            self._state_index.pop(state.state_id, None)

    # ------------------------------------------------------------------
    # IFutureModelingEngine — public interface
    # ------------------------------------------------------------------

    def construct_future_model(
        self, request: FutureModelConstructRequest
    ) -> FutureModelConstructResponse:
        """
        Construct a FutureModel from the given future states and profiles.

        Constitutional rule:
          - JANUS constructs models; VEGA selects preferred futures.
          - This method never indicates which future is preferred.
        """
        self._require_ready()
        self._validate_construct_request(request)

        states_list = list(request.future_states)
        # Pre-validate states before model construction.
        placeholder_id = "<pending>"
        self._validate_states_collection(states_list, placeholder_id)

        try:
            model = FutureModel.create(
                title=request.title,
                description=request.description,
                context=request.context,
                future_states=states_list,
                uncertainty=request.uncertainty,
                confidence=request.confidence,
                evidence=request.evidence,
            )
        except (ValueError, TypeError) as exc:
            raise JanusFutureModelConstructionError(str(exc)) from exc

        self._validate_model_integrity(model)

        with self._lock:
            self._models[model.model_id] = model
            self._index_states(model.model_id, model.future_states)
            self._total_models_created += 1
            self._total_states_created += len(model.future_states)

        return FutureModelConstructResponse(
            future_model=model,
            constructed_at=datetime.utcnow(),
            engine_version=_ENGINE_VERSION,
        )

    def get_future_model(
        self, request: FutureModelGetRequest
    ) -> FutureModelGetResponse:
        self._require_ready()
        if not request.model_id.strip():
            raise JanusMissingRequiredFieldError("model_id", engine=_ENGINE_NAME)

        with self._lock:
            model = self._models.get(request.model_id)
            if model is None:
                raise JanusFutureModelNotFoundError(request.model_id)
            return FutureModelGetResponse(
                future_model=model, retrieved_at=datetime.utcnow()
            )

    def query_states_at_horizon(
        self, request: FutureStateQueryRequest
    ) -> FutureStateQueryResponse:
        """Return all FutureStates within a FutureModel at the given horizon."""
        self._require_ready()
        if not request.model_id.strip():
            raise JanusMissingRequiredFieldError("model_id", engine=_ENGINE_NAME)
        if request.horizon is None:
            raise JanusValidationError(
                "horizon is required.", field="horizon", engine=_ENGINE_NAME
            )

        with self._lock:
            model = self._models.get(request.model_id)
            if model is None:
                raise JanusFutureModelNotFoundError(request.model_id)
            states = tuple(model.state_at(request.horizon))

        return FutureStateQueryResponse(
            model_id=request.model_id,
            horizon=request.horizon,
            states=states,
            retrieved_at=datetime.utcnow(),
        )

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

        The existing model is replaced atomically.  The state index is
        updated to reflect added and removed states.
        """
        self._require_ready()

        if not model_id.strip():
            raise JanusMissingRequiredFieldError("model_id", engine=_ENGINE_NAME)
        if not reason.strip():
            raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)
        if updated_uncertainty is None:
            raise JanusMissingUncertaintyError(model_id, "FutureModel")
        if updated_evidence is None:
            raise JanusMissingRequiredFieldError("updated_evidence", engine=_ENGINE_NAME)

        states_list = list(updated_states)
        self._validate_states_collection(states_list, model_id)
        for state in states_list:
            self._validate_future_state(state)

        with self._lock:
            existing = self._models.get(model_id)
            if existing is None:
                raise JanusFutureModelNotFoundError(model_id)

            # Deindex old states; index new states.
            self._deindex_states(existing.future_states)

            existing.future_states = states_list
            existing.uncertainty   = updated_uncertainty
            existing.evidence      = updated_evidence
            existing.updated_at    = datetime.utcnow()

            self._index_states(model_id, states_list)
            self._total_models_updated += 1

        return existing

    def list_future_models(self) -> tuple[FutureModel, ...]:
        """Return all currently registered FutureModels."""
        self._require_ready()
        with self._lock:
            return tuple(self._models.values())

    # ------------------------------------------------------------------
    # Extended public API (beyond interface minimum)
    # ------------------------------------------------------------------

    def create_future_state(
        self,
        label: str,
        description: str,
        horizon: ForecastHorizon,
        attributes: dict[str, Any],
        probability: float,
        uncertainty: UncertaintyProfile,
        risks: Optional[list[RiskAssessment]] = None,
        opportunities: Optional[list[OpportunityAssessment]] = None,
    ) -> FutureState:
        """
        Construct and validate a standalone FutureState.

        The returned state is not yet attached to any FutureModel; the
        caller registers it by passing it to construct_future_model() or
        update_future_model().
        """
        self._require_ready()

        if not label.strip():
            raise JanusMissingRequiredFieldError("label", engine=_ENGINE_NAME)
        if horizon is None:
            raise JanusMissingRequiredFieldError("horizon", engine=_ENGINE_NAME)
        if not (0.0 <= probability <= 1.0):
            raise JanusValidationError(
                f"probability {probability!r} must be in [0.0, 1.0].",
                field="probability",
                engine=_ENGINE_NAME,
            )
        if uncertainty is None:
            raise JanusMissingUncertaintyError("<pending>", "FutureState")

        try:
            state = FutureState.create(
                label=label,
                description=description,
                horizon=horizon,
                attributes=attributes,
                probability=probability,
                uncertainty=uncertainty,
                risks=risks or [],
                opportunities=opportunities or [],
            )
        except (ValueError, TypeError) as exc:
            raise JanusFutureStateError("<pending>", str(exc)) from exc

        self._validate_future_state(state)

        with self._lock:
            self._total_states_created += 1

        return state

    def get_future_state(self, state_id: str) -> FutureState:
        """
        Retrieve a FutureState by its state_id across all registered models.
        """
        self._require_ready()
        if not state_id.strip():
            raise JanusMissingRequiredFieldError("state_id", engine=_ENGINE_NAME)

        with self._lock:
            entry = self._state_index.get(state_id)
            if entry is None:
                raise JanusFutureStateError(state_id, "FutureState not found.")
            _, state = entry
        return state

    def update_future_state(
        self,
        model_id: str,
        state_id: str,
        updated_label: Optional[str] = None,
        updated_description: Optional[str] = None,
        updated_probability: Optional[float] = None,
        updated_attributes: Optional[dict[str, Any]] = None,
        updated_uncertainty: Optional[UncertaintyProfile] = None,
        updated_risks: Optional[list[RiskAssessment]] = None,
        updated_opportunities: Optional[list[OpportunityAssessment]] = None,
    ) -> FutureState:
        """
        Update individual fields of a FutureState within a named FutureModel.

        Only non-None arguments are applied; existing values are preserved
        for all other fields.  The state_index is refreshed atomically.
        """
        self._require_ready()
        if not model_id.strip():
            raise JanusMissingRequiredFieldError("model_id", engine=_ENGINE_NAME)
        if not state_id.strip():
            raise JanusMissingRequiredFieldError("state_id", engine=_ENGINE_NAME)

        if updated_probability is not None and not (0.0 <= updated_probability <= 1.0):
            raise JanusValidationError(
                f"probability {updated_probability!r} must be in [0.0, 1.0].",
                field="probability",
                engine=_ENGINE_NAME,
            )

        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                raise JanusFutureModelNotFoundError(model_id)

            state_entry = self._state_index.get(state_id)
            if state_entry is None or state_entry[0] != model_id:
                raise JanusFutureStateError(
                    state_id,
                    f"FutureState not found in model '{model_id}'.",
                )

            _, existing_state = state_entry

            # Build an updated FutureState (it's a mutable dataclass).
            if updated_label is not None:
                if not updated_label.strip():
                    raise JanusFutureStateError(state_id, "label cannot be empty.")
                existing_state.label = updated_label
            if updated_description is not None:
                existing_state.description = updated_description
            if updated_probability is not None:
                existing_state.probability = updated_probability
            if updated_attributes is not None:
                existing_state.attributes = updated_attributes
            if updated_uncertainty is not None:
                existing_state.uncertainty = updated_uncertainty
            if updated_risks is not None:
                existing_state.risks = updated_risks
            if updated_opportunities is not None:
                existing_state.opportunities = updated_opportunities

            self._validate_future_state(existing_state)

            # Re-index the updated state.
            self._state_index[state_id] = (model_id, existing_state)
            model.updated_at = datetime.utcnow()

        return existing_state

    def register_future_model(self, model: FutureModel) -> None:
        """
        Register an externally-constructed FutureModel directly.

        Used when a model was built outside the engine (e.g. via
        CounterfactualEngine) and needs to enter the registry.
        """
        self._require_ready()
        self._validate_model_integrity(model)

        with self._lock:
            if model.model_id in self._models:
                raise JanusFutureModelConstructionError(
                    f"FutureModel '{model.model_id}' is already registered.",
                    context={"model_id": model.model_id},
                )
            self._models[model.model_id] = model
            self._index_states(model.model_id, model.future_states)
            self._total_models_created += 1
            self._total_states_created += len(model.future_states)

    def model_exists(self, model_id: str) -> bool:
        """Return True if model_id is registered."""
        self._require_ready()
        with self._lock:
            return model_id in self._models

    def state_exists(self, state_id: str) -> bool:
        """Return True if state_id appears in any registered model."""
        self._require_ready()
        with self._lock:
            return state_id in self._state_index

    def list_models(
        self,
        *,
        horizon: Optional[ForecastHorizon] = None,
    ) -> tuple[FutureModel, ...]:
        """
        Return all registered FutureModels, optionally filtered to those
        that contain at least one FutureState at the given horizon.
        """
        self._require_ready()
        with self._lock:
            models = list(self._models.values())

        if horizon is not None:
            models = [m for m in models if any(s.horizon == horizon for s in m.future_states)]

        return tuple(models)

    def horizon_coverage(self, model_id: str) -> dict[str, int]:
        """
        Return a dict mapping ForecastHorizon value → state count for a model.
        Useful for diagnosing thin or missing horizon coverage.
        """
        self._require_ready()
        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                raise JanusFutureModelNotFoundError(model_id)
            coverage: dict[str, int] = {}
            for state in model.future_states:
                coverage[state.horizon.value] = coverage.get(state.horizon.value, 0) + 1
        return coverage

    def aggregate_risk_level(self, model_id: str) -> Optional[RiskLevel]:
        """
        Return the highest RiskLevel found across all FutureStates in the model.
        Returns None if the model contains no risk assessments.
        """
        self._require_ready()
        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                raise JanusFutureModelNotFoundError(model_id)
            levels: list[RiskLevel] = []
            for state in model.future_states:
                for risk in state.risks:
                    levels.append(risk.level)

        if not levels:
            return None
        # RiskLevel enum members are defined in ascending severity order.
        return max(levels, key=lambda l: l.value)

    def aggregate_opportunity_level(self, model_id: str) -> Optional[OpportunityLevel]:
        """
        Return the highest OpportunityLevel found across all FutureStates.
        Returns None if the model contains no opportunity assessments.
        """
        self._require_ready()
        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                raise JanusFutureModelNotFoundError(model_id)
            levels: list[OpportunityLevel] = []
            for state in model.future_states:
                for opp in state.opportunities:
                    levels.append(opp.level)

        if not levels:
            return None
        return max(levels, key=lambda l: l.value)

    def weighted_probability_summary(self, model_id: str) -> dict[str, float]:
        """
        Return a horizon-keyed dict of average state probabilities for a model.

        Useful for pipeline consumers (ODYSSEY, VEGA) to understand overall
        future-state likelihood without selecting a specific state.
        """
        self._require_ready()
        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                raise JanusFutureModelNotFoundError(model_id)
            horizon_probs: dict[str, list[float]] = {}
            for state in model.future_states:
                key = state.horizon.value
                horizon_probs.setdefault(key, []).append(state.probability)

        return {
            horizon: sum(probs) / len(probs)
            for horizon, probs in horizon_probs.items()
        }

    # ------------------------------------------------------------------
    # Statistics and health
    # ------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        """Return a snapshot of engine statistics for monitoring."""
        self._require_ready()
        with self._lock:
            model_count  = len(self._models)
            state_count  = len(self._state_index)
            horizons_seen: dict[str, int] = {}
            for _, state in self._state_index.values():
                horizons_seen[state.horizon.value] = (
                    horizons_seen.get(state.horizon.value, 0) + 1
                )

        return {
            "engine":               _ENGINE_NAME,
            "version":              _ENGINE_VERSION,
            "initialized":          self._initialized,
            "total_models":         model_count,
            "total_states_indexed": state_count,
            "total_models_created": self._total_models_created,
            "total_states_created": self._total_states_created,
            "total_models_updated": self._total_models_updated,
            "states_by_horizon":    horizons_seen,
            "engine_created_at":    self._created_at.isoformat() if self._created_at else None,
        }

    def health(self) -> dict[str, Any]:
        """Return engine health report."""
        is_healthy = self._initialized and not self._shutdown_flag
        return {
            "engine":        _ENGINE_NAME,
            "version":       _ENGINE_VERSION,
            "healthy":       is_healthy,
            "initialized":   self._initialized,
            "shutdown":      self._shutdown_flag,
            "model_count":   len(self._models)       if is_healthy else 0,
            "state_count":   len(self._state_index)  if is_healthy else 0,
        }
