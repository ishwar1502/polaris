"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/scenario.py

Implements ScenarioEngine — the Scenario Engine for JANUS.

Responsibilities:
  - Scenario creation and registration
  - Scenario retrieval by ID, type, and status
  - Scenario lifecycle management (status transitions)
  - Scenario comparison support
  - Scenario metadata management
  - Scenario statistics and health reporting

Constitutional boundaries enforced:
  - JANUS generates and evaluates scenarios; VEGA selects among them.
  - This engine never selects, approves, or decides on a scenario.
  - All scenarios carry uncertainty (Law 6).
  - JANUS never owns decisions, planning, identity, or knowledge storage.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Optional

from ..models import (
    ConfidenceProfile,
    EvidenceProfile,
    FutureModel,
    Scenario,
    ScenarioBranch,
    ScenarioMetadata,
    ScenarioStatus,
    ScenarioType,
    UncertaintyProfile,
)
from ..exceptions import (
    JanusAlreadyInitializedError,
    JanusApprovalBoundaryViolationError,
    JanusMissingRequiredFieldError,
    JanusNotInitializedError,
    JanusScenarioBranchError,
    JanusScenarioGenerationError,
    JanusScenarioLimitExceededError,
    JanusScenarioNotFoundError,
    JanusScenarioStatusTransitionError,
    JanusShutdownError,
    JanusValidationError,
)
from ..interfaces import IScenarioEngine
from ..schemas import (
    ScenarioArchiveRequest,
    ScenarioArchiveResponse,
    ScenarioGenerateRequest,
    ScenarioGenerateResponse,
    ScenarioGetRequest,
    ScenarioGetResponse,
    ScenarioUpdateStatusRequest,
    ScenarioUpdateStatusResponse,
)

# ---------------------------------------------------------------------------
# Status-transition graph
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: dict[ScenarioStatus, frozenset[ScenarioStatus]] = {
    ScenarioStatus.PENDING:     frozenset({ScenarioStatus.ACTIVE, ScenarioStatus.INVALIDATED}),
    ScenarioStatus.ACTIVE:      frozenset({ScenarioStatus.EVALUATING, ScenarioStatus.ARCHIVED, ScenarioStatus.INVALIDATED}),
    ScenarioStatus.EVALUATING:  frozenset({ScenarioStatus.EVALUATED, ScenarioStatus.ACTIVE, ScenarioStatus.INVALIDATED}),
    ScenarioStatus.EVALUATED:   frozenset({ScenarioStatus.ARCHIVED, ScenarioStatus.INVALIDATED}),
    ScenarioStatus.ARCHIVED:    frozenset(),
    ScenarioStatus.INVALIDATED: frozenset(),
}

# Constitutional: JANUS never approves or selects scenarios.
_APPROVAL_FORBIDDEN_OPERATIONS: frozenset[str] = frozenset({
    "approve_scenario",
    "select_scenario",
    "decide_scenario",
    "commit_scenario",
})

_ENGINE_NAME    = "ScenarioEngine"
_ENGINE_VERSION = "5.1.0"
_DEFAULT_LIMIT  = 10_000


class ScenarioEngine(IScenarioEngine):
    """
    Thread-safe implementation of IScenarioEngine.

    Internal storage uses two dicts, each guarded by a single RLock:
      _scenarios  — scenario_id → Scenario
      _comparisons is maintained externally (ScenarioEvaluationEngine).

    Lifecycle:
        engine = ScenarioEngine()
        engine.initialize()
        # … operational calls …
        engine.shutdown()
    """

    def __init__(self, max_scenarios: int = _DEFAULT_LIMIT) -> None:
        self._max_scenarios  = max_scenarios
        self._scenarios:     dict[str, Scenario] = {}
        self._lock:          threading.RLock = threading.RLock()
        self._initialized:   bool = False
        self._shutdown_flag: bool = False

        # Diagnostics
        self._total_created:  int = 0
        self._total_archived: int = 0
        self._total_invalidated: int = 0
        self._created_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._scenarios.clear()
            self._total_created      = 0
            self._total_archived     = 0
            self._total_invalidated  = 0
            self._created_at         = datetime.utcnow()
            self._initialized        = True

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

    def _require_not_approval(self, operation: str) -> None:
        """Enforce constitutional boundary: JANUS never approves scenarios."""
        if operation in _APPROVAL_FORBIDDEN_OPERATIONS:
            raise JanusApprovalBoundaryViolationError(operation)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_generate_request(self, request: ScenarioGenerateRequest) -> None:
        if not request.decision_context.strip():
            raise JanusMissingRequiredFieldError("decision_context", engine=_ENGINE_NAME)
        if not request.scenario_types:
            raise JanusValidationError(
                "At least one ScenarioType must be specified.",
                engine=_ENGINE_NAME,
            )
        if request.max_branches_per_scenario < 1:
            raise JanusValidationError(
                "max_branches_per_scenario must be >= 1.",
                field="max_branches_per_scenario",
                engine=_ENGINE_NAME,
            )
        if request.metadata is None:
            raise JanusMissingRequiredFieldError("metadata", engine=_ENGINE_NAME)

    def _validate_scenario_integrity(self, scenario: Scenario) -> None:
        """Validate that a Scenario is complete before registration."""
        if not scenario.scenario_id.strip():
            raise JanusScenarioGenerationError("scenario_id is empty.")
        if not scenario.title.strip():
            raise JanusMissingRequiredFieldError("title", engine=_ENGINE_NAME)
        if scenario.future_model is None:
            raise JanusMissingRequiredFieldError("future_model", engine=_ENGINE_NAME)
        if scenario.uncertainty is None:
            raise JanusMissingRequiredFieldError("uncertainty", engine=_ENGINE_NAME)
        if scenario.confidence is None:
            raise JanusMissingRequiredFieldError("confidence", engine=_ENGINE_NAME)
        if scenario.evidence is None:
            raise JanusMissingRequiredFieldError("evidence", engine=_ENGINE_NAME)
        if scenario.metadata is None:
            raise JanusMissingRequiredFieldError("metadata", engine=_ENGINE_NAME)
        if not (0.0 <= scenario.risk_score <= 1.0):
            raise JanusValidationError(
                f"risk_score {scenario.risk_score!r} must be in [0.0, 1.0].",
                field="risk_score",
                engine=_ENGINE_NAME,
            )
        if not (0.0 <= scenario.opportunity_score <= 1.0):
            raise JanusValidationError(
                f"opportunity_score {scenario.opportunity_score!r} must be in [0.0, 1.0].",
                field="opportunity_score",
                engine=_ENGINE_NAME,
            )
        for branch in scenario.branches:
            self._validate_branch(branch)

    def _validate_branch(self, branch: ScenarioBranch) -> None:
        if not branch.branch_id.strip():
            raise JanusScenarioBranchError("", "branch_id is empty.")
        if not branch.label.strip():
            raise JanusScenarioBranchError(branch.branch_id, "label is empty.")
        if not (0.0 <= branch.probability <= 1.0):
            raise JanusScenarioBranchError(
                branch.branch_id,
                f"probability {branch.probability!r} must be in [0.0, 1.0].",
            )
        if branch.future_state is None:
            raise JanusScenarioBranchError(branch.branch_id, "future_state is None.")
        if branch.risk_assessment is None:
            raise JanusScenarioBranchError(branch.branch_id, "risk_assessment is None.")
        if branch.opportunity_assessment is None:
            raise JanusScenarioBranchError(branch.branch_id, "opportunity_assessment is None.")

    def _validate_transition(
        self,
        scenario_id: str,
        current: ScenarioStatus,
        target: ScenarioStatus,
    ) -> None:
        allowed = _VALID_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise JanusScenarioStatusTransitionError(
                scenario_id, current.name, target.name
            )

    # ------------------------------------------------------------------
    # Core scenario construction
    # ------------------------------------------------------------------

    def _build_scenario_from_request(
        self,
        request: ScenarioGenerateRequest,
        scenario_type: ScenarioType,
    ) -> Scenario:
        """
        Construct a bare Scenario skeleton from a generate request.
        Branches and FutureModel are assembled externally or by callers
        integrating with BranchAnalysisEngine / FutureModelingEngine.
        Here we produce the canonical empty-branched scenario that
        downstream integrations populate.
        """
        import uuid
        from ..models import (
            UncertaintyLevel,
            FutureModel,
            EvidenceProfile,
            ConfidenceProfile,
        )

        # Derive a neutral placeholder FutureModel so Scenario.create() succeeds.
        # Real FutureModels are supplied by FutureModelingEngine in production.
        empty_future_model = FutureModel.create(
            title=f"{scenario_type.name} Future Model — {request.decision_context[:60]}",
            description=(
                f"Auto-generated placeholder future model for scenario type "
                f"{scenario_type.name}. Populate via FutureModelingEngine."
            ),
            context=request.decision_context,
            future_states=[],
            uncertainty=UncertaintyProfile.create(
                level=UncertaintyLevel.MODERATE,
                known_risks=[],
                unknown_risk_exposure=0.5,
                volatility_score=0.5,
                external_factors=[],
                market_sensitivity=0.5,
                technology_sensitivity=0.5,
                notes="Placeholder uncertainty profile. Update via UncertaintyEngine.",
            ),
            confidence=ConfidenceProfile.create(
                overall=0.5,
                data_quality=0.5,
                model_fit=0.5,
                signal_strength=0.5,
                notes="Placeholder confidence profile.",
            ),
            evidence=EvidenceProfile.create(
                sources=[],
                patterns_observed=[],
                contradicting_evidence=[],
                evidence_strength=0.5,
            ),
        )

        scenario = Scenario.create(
            title=(
                f"{scenario_type.name.replace('_', ' ').title()} — "
                f"{request.decision_context[:80]}"
            ),
            description=(
                f"Scenario exploring a {scenario_type.name.lower()} future "
                f"for the decision: {request.decision_context}"
            ),
            scenario_type=scenario_type,
            branches=[],
            future_model=empty_future_model,
            uncertainty=empty_future_model.uncertainty,
            confidence=empty_future_model.confidence,
            evidence=empty_future_model.evidence,
            metadata=request.metadata,
            risk_score=0.5,
            opportunity_score=0.5,
        )
        return scenario

    # ------------------------------------------------------------------
    # IScenarioEngine — public interface
    # ------------------------------------------------------------------

    def generate_scenarios(
        self, request: ScenarioGenerateRequest
    ) -> ScenarioGenerateResponse:
        """
        Generate one Scenario per requested ScenarioType and register each.

        Constitutional rule: generated scenarios are never selected here.
        Selection belongs to VEGA.
        """
        self._require_ready()
        self._validate_generate_request(request)

        with self._lock:
            projected_total = len(self._scenarios) + len(request.scenario_types)
            if projected_total > self._max_scenarios:
                raise JanusScenarioLimitExceededError(projected_total, self._max_scenarios)

            generated: list[Scenario] = []
            try:
                for scenario_type in request.scenario_types:
                    scenario = self._build_scenario_from_request(request, scenario_type)
                    self._validate_scenario_integrity(scenario)
                    self._scenarios[scenario.scenario_id] = scenario
                    self._total_created += 1
                    generated.append(scenario)
            except Exception as exc:
                # Roll back any partially registered scenarios.
                for s in generated:
                    self._scenarios.pop(s.scenario_id, None)
                    self._total_created -= 1
                if isinstance(exc, JanusScenarioGenerationError):
                    raise
                raise JanusScenarioGenerationError(
                    str(exc),
                    context={"decision_context": request.decision_context},
                ) from exc

        return ScenarioGenerateResponse(
            scenarios=tuple(generated),
            generation_context=request.decision_context,
            horizon=request.horizon,
            generated_at=datetime.utcnow(),
            engine_version=_ENGINE_VERSION,
        )

    def get_scenario(self, request: ScenarioGetRequest) -> ScenarioGetResponse:
        self._require_ready()
        if not request.scenario_id.strip():
            raise JanusMissingRequiredFieldError("scenario_id", engine=_ENGINE_NAME)

        with self._lock:
            scenario = self._scenarios.get(request.scenario_id)
            if scenario is None:
                raise JanusScenarioNotFoundError(request.scenario_id)
            return ScenarioGetResponse(scenario=scenario, retrieved_at=datetime.utcnow())

    def update_scenario_status(
        self, request: ScenarioUpdateStatusRequest
    ) -> ScenarioUpdateStatusResponse:
        """Transition a Scenario's status through the validated state machine."""
        self._require_ready()
        self._require_not_approval(request.target_status.name.lower())

        if not request.scenario_id.strip():
            raise JanusMissingRequiredFieldError("scenario_id", engine=_ENGINE_NAME)
        if not request.reason.strip():
            raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)
        if not request.updated_by.strip():
            raise JanusMissingRequiredFieldError("updated_by", engine=_ENGINE_NAME)

        with self._lock:
            scenario = self._scenarios.get(request.scenario_id)
            if scenario is None:
                raise JanusScenarioNotFoundError(request.scenario_id)

            previous_status = scenario.status
            self._validate_transition(request.scenario_id, previous_status, request.target_status)

            # Dataclass is mutable (not frozen) — update status and updated_at in-place.
            scenario.status     = request.target_status
            scenario.updated_at = datetime.utcnow()

            if request.target_status == ScenarioStatus.INVALIDATED:
                self._total_invalidated += 1
            elif request.target_status == ScenarioStatus.ARCHIVED:
                self._total_archived += 1

        return ScenarioUpdateStatusResponse(
            scenario_id=request.scenario_id,
            previous_status=previous_status,
            new_status=request.target_status,
            updated_at=scenario.updated_at,
        )

    def archive_scenario(
        self, request: ScenarioArchiveRequest
    ) -> ScenarioArchiveResponse:
        """Archive a Scenario via the validated status transition."""
        self._require_ready()

        if not request.scenario_id.strip():
            raise JanusMissingRequiredFieldError("scenario_id", engine=_ENGINE_NAME)
        if not request.reason.strip():
            raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)
        if not request.archived_by.strip():
            raise JanusMissingRequiredFieldError("archived_by", engine=_ENGINE_NAME)

        update_request = ScenarioUpdateStatusRequest(
            scenario_id=request.scenario_id,
            target_status=ScenarioStatus.ARCHIVED,
            reason=request.reason,
            updated_by=request.archived_by,
        )
        update_response = self.update_scenario_status(update_request)

        return ScenarioArchiveResponse(
            scenario_id=request.scenario_id,
            archived_at=update_response.updated_at,
        )

    def list_scenarios_by_type(
        self, scenario_type: ScenarioType
    ) -> tuple[Scenario, ...]:
        self._require_ready()
        with self._lock:
            return tuple(
                s for s in self._scenarios.values()
                if s.scenario_type == scenario_type
            )

    def list_scenarios_by_status(
        self, status: ScenarioStatus
    ) -> tuple[Scenario, ...]:
        self._require_ready()
        with self._lock:
            return tuple(
                s for s in self._scenarios.values()
                if s.status == status
            )

    # ------------------------------------------------------------------
    # Extended public API (beyond interface minimum)
    # ------------------------------------------------------------------

    def register_scenario(self, scenario: Scenario) -> None:
        """
        Register an externally-constructed Scenario.

        Used when BranchAnalysisEngine and FutureModelingEngine have fully
        populated a Scenario before it enters the ScenarioEngine's registry.
        """
        self._require_ready()
        self._validate_scenario_integrity(scenario)

        with self._lock:
            if len(self._scenarios) >= self._max_scenarios:
                raise JanusScenarioLimitExceededError(
                    len(self._scenarios) + 1, self._max_scenarios
                )
            if scenario.scenario_id in self._scenarios:
                raise JanusScenarioGenerationError(
                    f"Scenario '{scenario.scenario_id}' is already registered.",
                    context={"scenario_id": scenario.scenario_id},
                )
            self._scenarios[scenario.scenario_id] = scenario
            self._total_created += 1

    def update_scenario(self, scenario: Scenario) -> Scenario:
        """
        Replace the stored Scenario with an updated version.

        The scenario_id must match an existing entry.  Status transitions must
        still pass through update_scenario_status(); this method is for
        updating branches, scores, and model references.
        """
        self._require_ready()
        self._validate_scenario_integrity(scenario)

        with self._lock:
            if scenario.scenario_id not in self._scenarios:
                raise JanusScenarioNotFoundError(scenario.scenario_id)
            scenario.updated_at = datetime.utcnow()
            self._scenarios[scenario.scenario_id] = scenario
        return scenario

    def delete_scenario(self, scenario_id: str, deleted_by: str, reason: str) -> None:
        """
        Permanently remove a Scenario from the registry.

        Only ARCHIVED or INVALIDATED scenarios may be deleted.  Active or
        evaluating scenarios must be transitioned first.
        """
        self._require_ready()
        if not scenario_id.strip():
            raise JanusMissingRequiredFieldError("scenario_id", engine=_ENGINE_NAME)
        if not deleted_by.strip():
            raise JanusMissingRequiredFieldError("deleted_by", engine=_ENGINE_NAME)
        if not reason.strip():
            raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)

        with self._lock:
            scenario = self._scenarios.get(scenario_id)
            if scenario is None:
                raise JanusScenarioNotFoundError(scenario_id)
            if scenario.status not in (ScenarioStatus.ARCHIVED, ScenarioStatus.INVALIDATED):
                raise JanusScenarioStatusTransitionError(
                    scenario_id,
                    scenario.status.name,
                    "DELETED (requires ARCHIVED or INVALIDATED first)",
                )
            del self._scenarios[scenario_id]

    def list_scenarios(
        self,
        *,
        scenario_type: Optional[ScenarioType] = None,
        status: Optional[ScenarioStatus] = None,
    ) -> tuple[Scenario, ...]:
        """
        Return all Scenarios, optionally filtered by type and/or status.
        """
        self._require_ready()
        with self._lock:
            results: list[Scenario] = list(self._scenarios.values())

        if scenario_type is not None:
            results = [s for s in results if s.scenario_type == scenario_type]
        if status is not None:
            results = [s for s in results if s.status == status]

        return tuple(results)

    def scenario_exists(self, scenario_id: str) -> bool:
        """Return True if the scenario_id is registered."""
        self._require_ready()
        with self._lock:
            return scenario_id in self._scenarios

    # ------------------------------------------------------------------
    # Statistics and health
    # ------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        """Return a snapshot of engine statistics for monitoring."""
        self._require_ready()
        with self._lock:
            total = len(self._scenarios)
            by_type: dict[str, int] = {}
            by_status: dict[str, int] = {}
            for s in self._scenarios.values():
                by_type[s.scenario_type.name]   = by_type.get(s.scenario_type.name, 0) + 1
                by_status[s.status.name]         = by_status.get(s.status.name, 0) + 1

        return {
            "engine":               _ENGINE_NAME,
            "version":              _ENGINE_VERSION,
            "initialized":          self._initialized,
            "total_registered":     total,
            "total_created":        self._total_created,
            "total_archived":       self._total_archived,
            "total_invalidated":    self._total_invalidated,
            "max_scenarios":        self._max_scenarios,
            "scenarios_by_type":    by_type,
            "scenarios_by_status":  by_status,
            "engine_created_at":    self._created_at.isoformat() if self._created_at else None,
        }

    def health(self) -> dict[str, Any]:
        """Return engine health report."""
        is_healthy = self._initialized and not self._shutdown_flag
        return {
            "engine":       _ENGINE_NAME,
            "version":      _ENGINE_VERSION,
            "healthy":      is_healthy,
            "initialized":  self._initialized,
            "shutdown":     self._shutdown_flag,
            "scenario_count": len(self._scenarios) if is_healthy else 0,
        }
