"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/simulation.py

Implementation of the Outcome Simulation Engine.

Responsibility: Simulate consequence chains from a given decision or event,
producing short-, medium-, and long-term effect chains.

Constitutional rule (Law 3):
    ORION reasons. JANUS simulates. Never merged.

JANUS Law 6:
    All forecasts require uncertainty. No certainty claims. Ever.

JANUS Law 7:
    JANUS cannot create plans. ZENITH owns planning.

Bounded Exploration Law:
    All simulations must support maximum branch depth, maximum simulation
    count, confidence pruning, probability pruning, resource limits, and
    compute budgets. Unlimited simulation growth is forbidden.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .models import (
    OutcomeSimulation,
    ProbabilityDistribution,
    SimulationOutcome,
    SimulationStatus,
    UncertaintyProfile,
)
from .interfaces import IOutcomeSimulationEngine
from .schemas import (
    SimulationRunRequest,
    SimulationRunResponse,
    SimulationResultRequest,
    SimulationResultResponse,
    SimulationAbortRequest,
    SimulationAbortResponse,
)
from .exceptions import (
    JanusNotInitializedError,
    JanusAlreadyInitializedError,
    JanusShutdownError,
    JanusMissingRequiredFieldError,
    JanusInvalidProbabilityError,
    JanusSimulationNotFoundError,
    JanusSimulationExecutionError,
    JanusSimulationAbortedError,
    JanusSimulationStatusError,
    JanusSimulationOutcomeError,
    JanusMissingUncertaintyError,
)


# ---------------------------------------------------------------------------
# Engine Identity
# ---------------------------------------------------------------------------

_ENGINE_NAME: str = "OutcomeSimulationEngine"
_ENGINE_VERSION: str = "5.1.0"

_PROBABILITY_SUM_TOLERANCE: float = 1e-6


# ---------------------------------------------------------------------------
# Bounded Exploration Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationBoundsConfig:
    """
    Bounded Exploration Law configuration for the Outcome Simulation Engine.

    All simulations are governed by these limits. Unlimited simulation
    growth is forbidden.
    """

    max_outcomes_per_simulation: int = 25
    max_branch_depth: int = 6
    max_active_simulations: int = 1_000
    max_total_simulations: int = 100_000
    min_confidence_threshold: float = 0.05
    min_probability_threshold: float = 0.01
    max_compute_budget_units: int = 10_000

    def __post_init__(self) -> None:
        if self.max_outcomes_per_simulation < 1:
            raise ValueError("max_outcomes_per_simulation must be >= 1.")
        if self.max_branch_depth < 1:
            raise ValueError("max_branch_depth must be >= 1.")
        if self.max_active_simulations < 1:
            raise ValueError("max_active_simulations must be >= 1.")
        if self.max_total_simulations < 1:
            raise ValueError("max_total_simulations must be >= 1.")
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError("min_confidence_threshold must be in [0, 1].")
        if not 0.0 <= self.min_probability_threshold <= 1.0:
            raise ValueError("min_probability_threshold must be in [0, 1].")
        if self.max_compute_budget_units < 1:
            raise ValueError("max_compute_budget_units must be >= 1.")


# Legal SimulationStatus transitions.
_LEGAL_TRANSITIONS: dict[SimulationStatus, frozenset[SimulationStatus]] = {
    SimulationStatus.PENDING: frozenset(
        {SimulationStatus.RUNNING, SimulationStatus.ABORTED}
    ),
    SimulationStatus.RUNNING: frozenset(
        {SimulationStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.ABORTED}
    ),
    SimulationStatus.COMPLETED: frozenset({SimulationStatus.ABORTED}),
    SimulationStatus.FAILED: frozenset(),
    SimulationStatus.ABORTED: frozenset(),
}


# ---------------------------------------------------------------------------
# Diagnostics / Health / Statistics / Result Value Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationHealthReport:
    """Health snapshot for the Outcome Simulation Engine."""

    engine_name: str
    engine_version: str
    is_initialized: bool
    is_shutdown: bool
    total_simulations: int
    pending_simulations: int
    running_simulations: int
    completed_simulations: int
    failed_simulations: int
    aborted_simulations: int
    bounds: SimulationBoundsConfig
    healthy: bool
    generated_at: datetime


@dataclass(frozen=True)
class SimulationDiagnosticsReport:
    """Diagnostics snapshot covering integrity of stored simulations."""

    engine_name: str
    engine_version: str
    total_simulations: int
    simulations_missing_outcomes: int
    simulations_with_zero_uncertainty: int
    simulations_exceeding_outcome_limit: int
    average_outcome_count: float
    average_outcome_probability: Optional[float]
    status_distribution: dict[str, int]
    pruned_outcome_total: int
    generated_at: datetime


@dataclass(frozen=True)
class SimulationStatistics:
    """Aggregate statistics across all registered OutcomeSimulations."""

    total_simulations: int
    by_status: dict[str, int]
    total_outcomes: int
    average_outcomes_per_simulation: Optional[float]
    average_most_probable_outcome_probability: Optional[float]
    average_uncertainty_volatility: Optional[float]
    completion_rate: Optional[float]
    abort_rate: Optional[float]
    failure_rate: Optional[float]
    generated_at: datetime


@dataclass(frozen=True)
class SimulationComparisonResult:
    """Structured comparison between two OutcomeSimulations."""

    simulation_id_a: str
    simulation_id_b: str
    triggering_event_a: str
    triggering_event_b: str
    status_a: str
    status_b: str
    outcome_count_a: int
    outcome_count_b: int
    most_probable_label_a: Optional[str]
    most_probable_label_b: Optional[str]
    most_probable_probability_a: Optional[float]
    most_probable_probability_b: Optional[float]
    shared_outcome_labels: tuple[str, ...]
    unique_to_a: tuple[str, ...]
    unique_to_b: tuple[str, ...]
    compared_at: datetime


@dataclass(frozen=True)
class SimulationEvaluation:
    """
    Structured evaluation of a single OutcomeSimulation.

    This is an evaluation for review; it is not a decision and does not
    select or approve any outcome. JANUS simulates and evaluates; VEGA
    decides.
    """

    simulation_id: str
    title: str
    status: str
    triggering_event: str
    outcome_count: int
    most_probable_outcome_label: Optional[str]
    most_probable_outcome_probability: Optional[float]
    average_outcome_probability: Optional[float]
    aggregate_short_term_effects: tuple[str, ...]
    aggregate_medium_term_effects: tuple[str, ...]
    aggregate_long_term_effects: tuple[str, ...]
    aggregate_risk_implications: tuple[str, ...]
    aggregate_opportunity_implications: tuple[str, ...]
    uncertainty_volatility: float
    evaluated_at: datetime


@dataclass(frozen=True)
class MultiPathSimulationResult:
    """
    Result of running a multi-path simulation: one OutcomeSimulation per
    candidate decision path, sharing a common context.
    """

    path_simulation_ids: tuple[str, ...]
    paths: tuple[str, ...]
    bounded_outcome_counts: tuple[int, ...]
    pruned_outcome_counts: tuple[int, ...]
    started_at: datetime


@dataclass(frozen=True)
class ScenarioSimulationResult:
    """
    Result of running an outcome simulation derived from a scenario context
    (e.g., a ScenarioBranch's triggering choice and FutureState attributes).
    """

    simulation_id: str
    scenario_id: str
    branch_id: Optional[str]
    triggering_event: str
    outcome_count: int
    started_at: datetime


@dataclass(frozen=True)
class ForecastSimulationResult:
    """
    Result of running an outcome simulation derived from a Forecast's
    probability distribution: each distribution outcome becomes a candidate
    SimulationOutcome seed.
    """

    simulation_id: str
    forecast_id: str
    triggering_event: str
    seeded_outcome_count: int
    started_at: datetime


# ---------------------------------------------------------------------------
# Outcome Simulation Engine
# ---------------------------------------------------------------------------


class OutcomeSimulationEngine(IOutcomeSimulationEngine):
    """
    Production implementation of the Outcome Simulation Engine.

    Owns:
        - OutcomeSimulation creation, retrieval, and lifecycle management.
        - Multi-path, scenario-derived, and forecast-derived simulation runs.
        - Simulation comparison, evaluation, and statistics.
        - Bounded exploration enforcement (branch depth, simulation count,
          confidence/probability pruning, compute budgets).

    Never owns:
        - Decisions (VEGA).
        - Plans (ZENITH).
        - Reasoning (ORION).

    Thread-safety:
        All mutable state is guarded by a single re-entrant lock. Stored
        OutcomeSimulation instances are treated as immutable; transitions
        produce new instances replacing the prior registry entry.
    """

    def __init__(self, bounds: Optional[SimulationBoundsConfig] = None) -> None:
        self._lock = threading.RLock()
        self._initialized: bool = False
        self._shutdown: bool = False
        self._simulations: dict[str, OutcomeSimulation] = {}
        self._pruned_outcome_counts: dict[str, int] = {}
        self._bounds: SimulationBoundsConfig = bounds or SimulationBoundsConfig()
        self._total_simulations_created: int = 0

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._shutdown:
                raise JanusShutdownError(_ENGINE_NAME)
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._simulations = {}
            self._pruned_outcome_counts = {}
            self._total_simulations_created = 0
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

    @property
    def bounds(self) -> SimulationBoundsConfig:
        return self._bounds

    def _ensure_operational(self) -> None:
        if self._shutdown:
            raise JanusShutdownError(_ENGINE_NAME)
        if not self._initialized:
            raise JanusNotInitializedError(_ENGINE_NAME)

    # -----------------------------------------------------------------
    # Bounded Exploration Enforcement
    # -----------------------------------------------------------------

    def _enforce_simulation_count_bounds(self) -> None:
        active = sum(
            1
            for sim in self._simulations.values()
            if sim.status in (SimulationStatus.PENDING, SimulationStatus.RUNNING)
        )
        if active >= self._bounds.max_active_simulations:
            raise JanusSimulationExecutionError(
                "<pending>",
                (
                    f"Active simulation count {active} would reach or exceed "
                    f"the configured limit of "
                    f"{self._bounds.max_active_simulations}."
                ),
            )
        if self._total_simulations_created >= self._bounds.max_total_simulations:
            raise JanusSimulationExecutionError(
                "<pending>",
                (
                    f"Total simulation count "
                    f"{self._total_simulations_created} would reach or "
                    f"exceed the configured limit of "
                    f"{self._bounds.max_total_simulations}."
                ),
            )

    def _prune_outcomes(
        self, outcomes: tuple[SimulationOutcome, ...]
    ) -> tuple[tuple[SimulationOutcome, ...], int]:
        """
        Apply confidence pruning, probability pruning, and the per-simulation
        outcome cap (branch-depth proxy) to a candidate outcome set.

        Returns the bounded outcome tuple and the number of outcomes pruned.
        """
        survivors: list[SimulationOutcome] = []
        pruned = 0

        for outcome in outcomes:
            if outcome.probability < self._bounds.min_probability_threshold:
                pruned += 1
                continue
            if outcome.confidence.overall < self._bounds.min_confidence_threshold:
                pruned += 1
                continue
            survivors.append(outcome)

        if len(survivors) > self._bounds.max_outcomes_per_simulation:
            survivors.sort(key=lambda o: o.probability, reverse=True)
            overflow = len(survivors) - self._bounds.max_outcomes_per_simulation
            survivors = survivors[: self._bounds.max_outcomes_per_simulation]
            pruned += overflow

        if not survivors:
            # Always retain at least the single most probable outcome so
            # that the simulation remains evaluable; this does not violate
            # pruning bounds since it is the minimum viable set.
            best = max(outcomes, key=lambda o: o.probability)
            survivors = [best]

        return tuple(survivors), pruned

    @staticmethod
    def _enforce_branch_depth(
        outcomes: tuple[SimulationOutcome, ...], max_depth: int
    ) -> None:
        """
        Enforce the maximum branch depth bound. Branch depth is measured as
        the maximum number of chained effects (short + medium + long) across
        all candidate outcomes — each effect level represents one branch
        step in the consequence chain.
        """
        for outcome in outcomes:
            depth = max(
                len(outcome.short_term_effects),
                len(outcome.medium_term_effects),
                len(outcome.long_term_effects),
            )
            if depth > max_depth:
                raise JanusSimulationOutcomeError(
                    outcome.outcome_id,
                    (
                        f"effect-chain depth {depth} exceeds the configured "
                        f"maximum branch depth of {max_depth}."
                    ),
                )

    def _enforce_compute_budget(self, outcome_count: int, num_paths: int = 1) -> None:
        """
        Enforce the compute budget bound. Compute cost is approximated as
        outcome_count * num_paths, representing the total candidate-outcome
        evaluations required for this simulation run.
        """
        cost = outcome_count * max(num_paths, 1)
        if cost > self._bounds.max_compute_budget_units:
            raise JanusSimulationExecutionError(
                "<pending>",
                (
                    f"Estimated compute cost {cost} exceeds the configured "
                    f"compute budget of {self._bounds.max_compute_budget_units} "
                    "units."
                ),
            )

    # -----------------------------------------------------------------
    # Validation Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_run_request(request: SimulationRunRequest) -> None:
        if not request.title.strip():
            raise JanusMissingRequiredFieldError("title", engine=_ENGINE_NAME)
        if not request.description.strip():
            raise JanusMissingRequiredFieldError("description", engine=_ENGINE_NAME)
        if not request.triggering_event.strip():
            raise JanusMissingRequiredFieldError(
                "triggering_event", engine=_ENGINE_NAME
            )
        if not request.candidate_outcomes:
            raise JanusMissingRequiredFieldError(
                "candidate_outcomes", engine=_ENGINE_NAME
            )

        for outcome in request.candidate_outcomes:
            if not 0.0 <= outcome.probability <= 1.0:
                raise JanusInvalidProbabilityError(
                    outcome.probability,
                    field=f"candidate_outcomes[{outcome.outcome_id}].probability",
                    engine=_ENGINE_NAME,
                )

    @staticmethod
    def _validate_uncertainty(
        simulation_id: str, uncertainty: Optional[UncertaintyProfile]
    ) -> None:
        if uncertainty is None:
            raise JanusMissingUncertaintyError(simulation_id, "OutcomeSimulation")

    # -----------------------------------------------------------------
    # Probability Distribution Construction
    # -----------------------------------------------------------------

    @staticmethod
    def _build_probability_distribution(
        label: str,
        outcomes: tuple[SimulationOutcome, ...],
        uncertainty: UncertaintyProfile,
    ) -> ProbabilityDistribution:
        """
        Construct a normalized ProbabilityDistribution over the given
        outcome set, keyed by outcome label.
        """
        total = sum(o.probability for o in outcomes)

        if total <= 0.0:
            uniform_weight = 1.0 / len(outcomes)
            weights = {o.label: uniform_weight for o in outcomes}
        else:
            weights = {o.label: (o.probability / total) for o in outcomes}

        # Correct any floating point drift so the distribution sums to 1.0.
        keys = list(weights.keys())
        drift = 1.0 - sum(weights.values())
        if abs(drift) > _PROBABILITY_SUM_TOLERANCE:
            weights[keys[-1]] += drift

        confidence_values = [o.confidence for o in outcomes]
        aggregate_confidence = confidence_values[0]
        if len(confidence_values) > 1:
            overall = statistics.fmean(c.overall for c in confidence_values)
            data_quality = statistics.fmean(
                c.data_quality for c in confidence_values
            )
            model_fit = statistics.fmean(c.model_fit for c in confidence_values)
            signal_strength = statistics.fmean(
                c.signal_strength for c in confidence_values
            )
            aggregate_confidence = type(confidence_values[0]).create(
                overall=overall,
                data_quality=data_quality,
                model_fit=model_fit,
                signal_strength=signal_strength,
                notes="Aggregated across candidate simulation outcomes.",
            )

        from .models import UncertaintyLevel

        return ProbabilityDistribution.create(
            label=label,
            outcomes=weights,
            uncertainty_level=uncertainty.level,
            confidence=aggregate_confidence,
        )

    # -----------------------------------------------------------------
    # Core Interface: Run Simulation
    # -----------------------------------------------------------------

    def run_simulation(
        self, request: SimulationRunRequest
    ) -> SimulationRunResponse:
        with self._lock:
            self._ensure_operational()

            try:
                self._validate_run_request(request)
            except (
                JanusMissingRequiredFieldError,
                JanusInvalidProbabilityError,
            ) as exc:
                raise JanusSimulationExecutionError(
                    "<pending>", str(exc)
                ) from exc

            self._validate_uncertainty("<pending>", request.uncertainty)
            self._enforce_simulation_count_bounds()

            candidate_outcomes = tuple(request.candidate_outcomes)
            self._enforce_branch_depth(
                candidate_outcomes, self._bounds.max_branch_depth
            )
            self._enforce_compute_budget(len(candidate_outcomes))

            bounded_outcomes, pruned_count = self._prune_outcomes(candidate_outcomes)

            distribution = self._build_probability_distribution(
                label=f"{request.title} — Outcome Distribution",
                outcomes=bounded_outcomes,
                uncertainty=request.uncertainty,
            )

            try:
                simulation = OutcomeSimulation.create(
                    title=request.title,
                    description=request.description,
                    triggering_event=request.triggering_event,
                    outcomes=list(bounded_outcomes),
                    probability_distribution=distribution,
                    uncertainty=request.uncertainty,
                    metadata=request.metadata,
                )
            except ValueError as exc:
                raise JanusSimulationExecutionError("<pending>", str(exc)) from exc

            # Transition PENDING -> RUNNING immediately, then -> COMPLETED,
            # representing a synchronous (non-async) simulation execution
            # model. The result remains available via get_simulation_result.
            running = _with_status(simulation, SimulationStatus.RUNNING)
            completed = _with_status(
                running, SimulationStatus.COMPLETED, completed_at=_utcnow()
            )

            self._simulations[completed.simulation_id] = completed
            self._pruned_outcome_counts[completed.simulation_id] = pruned_count
            self._total_simulations_created += 1

            return SimulationRunResponse(
                simulation=completed,
                started_at=simulation.created_at,
                engine_version=_ENGINE_VERSION,
            )

    # -----------------------------------------------------------------
    # Core Interface: Get Result
    # -----------------------------------------------------------------

    def get_simulation_result(
        self, request: SimulationResultRequest
    ) -> SimulationResultResponse:
        with self._lock:
            self._ensure_operational()

            simulation = self._simulations.get(request.simulation_id)
            if simulation is None:
                raise JanusSimulationNotFoundError(request.simulation_id)

            most_probable: Optional[SimulationOutcome] = (
                simulation.most_probable_outcome
            )

            return SimulationResultResponse(
                simulation=simulation,
                status=simulation.status,
                most_probable_outcome=most_probable,
                retrieved_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Core Interface: Abort
    # -----------------------------------------------------------------

    def abort_simulation(
        self, request: SimulationAbortRequest
    ) -> SimulationAbortResponse:
        with self._lock:
            self._ensure_operational()

            if not request.reason.strip():
                raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)
            if not request.aborted_by.strip():
                raise JanusMissingRequiredFieldError(
                    "aborted_by", engine=_ENGINE_NAME
                )

            simulation = self._simulations.get(request.simulation_id)
            if simulation is None:
                raise JanusSimulationNotFoundError(request.simulation_id)

            target_status = SimulationStatus.ABORTED
            if target_status not in _LEGAL_TRANSITIONS.get(
                simulation.status, frozenset()
            ):
                raise JanusSimulationStatusError(
                    request.simulation_id,
                    simulation.status.name,
                    target_status.name,
                )

            aborted_at = _utcnow()
            updated = _with_status(
                simulation, target_status, completed_at=aborted_at
            )
            self._simulations[request.simulation_id] = updated

            return SimulationAbortResponse(
                simulation_id=request.simulation_id,
                aborted_at=aborted_at,
            )

    def abort_simulation_safe(
        self, request: SimulationAbortRequest
    ) -> SimulationAbortResponse:
        """
        Abort a running simulation and return a confirmation response
        without raising, for callers that need a non-exceptional abort
        confirmation path. Performs the same validation and transition
        as abort_simulation.
        """
        with self._lock:
            self._ensure_operational()

            if not request.reason.strip():
                raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)
            if not request.aborted_by.strip():
                raise JanusMissingRequiredFieldError(
                    "aborted_by", engine=_ENGINE_NAME
                )

            simulation = self._simulations.get(request.simulation_id)
            if simulation is None:
                raise JanusSimulationNotFoundError(request.simulation_id)

            target_status = SimulationStatus.ABORTED
            if target_status not in _LEGAL_TRANSITIONS.get(
                simulation.status, frozenset()
            ):
                raise JanusSimulationStatusError(
                    request.simulation_id,
                    simulation.status.name,
                    target_status.name,
                )

            aborted_at = _utcnow()
            updated = _with_status(
                simulation, target_status, completed_at=aborted_at
            )
            self._simulations[request.simulation_id] = updated

            return SimulationAbortResponse(
                simulation_id=request.simulation_id,
                aborted_at=aborted_at,
            )

    # -----------------------------------------------------------------
    # Core Interface: Most Probable Outcome
    # -----------------------------------------------------------------

    def most_probable_outcome(self, simulation_id: str) -> SimulationOutcome:
        with self._lock:
            self._ensure_operational()

            simulation = self._simulations.get(simulation_id)
            if simulation is None:
                raise JanusSimulationNotFoundError(simulation_id)

            outcome = simulation.most_probable_outcome
            if outcome is None:
                raise JanusSimulationOutcomeError(
                    "<none>",
                    f"OutcomeSimulation '{simulation_id}' has no outcomes.",
                )

            return outcome

    # -----------------------------------------------------------------
    # Core Interface: List by Status
    # -----------------------------------------------------------------

    def list_simulations_by_status(
        self, status: SimulationStatus
    ) -> tuple[OutcomeSimulation, ...]:
        with self._lock:
            self._ensure_operational()

            return tuple(
                sim for sim in self._simulations.values() if sim.status == status
            )

    # -----------------------------------------------------------------
    # Core Interface: Validate Outcome Probabilities
    # -----------------------------------------------------------------

    def validate_outcome_probabilities(self, simulation_id: str) -> bool:
        with self._lock:
            self._ensure_operational()

            simulation = self._simulations.get(simulation_id)
            if simulation is None:
                raise JanusSimulationNotFoundError(simulation_id)

            for outcome in simulation.outcomes:
                if not 0.0 <= outcome.probability <= 1.0:
                    return False

            distribution_total = sum(
                simulation.probability_distribution.outcomes.values()
            )
            if abs(distribution_total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
                return False

            return True

    # -----------------------------------------------------------------
    # Simulation Retrieval / Management
    # -----------------------------------------------------------------

    def get_simulation(self, simulation_id: str) -> OutcomeSimulation:
        """Retrieve an OutcomeSimulation by its simulation_id."""
        with self._lock:
            self._ensure_operational()

            simulation = self._simulations.get(simulation_id)
            if simulation is None:
                raise JanusSimulationNotFoundError(simulation_id)

            return simulation

    def list_all_simulations(self) -> tuple[OutcomeSimulation, ...]:
        """Return all registered OutcomeSimulations."""
        with self._lock:
            self._ensure_operational()
            return tuple(self._simulations.values())

    def list_simulations_for_event(
        self, triggering_event: str
    ) -> tuple[OutcomeSimulation, ...]:
        """Return all OutcomeSimulations registered for a given triggering event."""
        with self._lock:
            self._ensure_operational()

            if not triggering_event.strip():
                raise JanusMissingRequiredFieldError(
                    "triggering_event", engine=_ENGINE_NAME
                )

            return tuple(
                sim
                for sim in self._simulations.values()
                if sim.triggering_event == triggering_event
            )

    def mark_simulation_failed(
        self, simulation_id: str, reason: str
    ) -> OutcomeSimulation:
        """
        Transition a RUNNING (or PENDING) OutcomeSimulation to FAILED.
        Used when downstream evaluation determines the simulation cannot
        be completed.
        """
        with self._lock:
            self._ensure_operational()

            if not reason.strip():
                raise JanusMissingRequiredFieldError("reason", engine=_ENGINE_NAME)

            simulation = self._simulations.get(simulation_id)
            if simulation is None:
                raise JanusSimulationNotFoundError(simulation_id)

            target_status = SimulationStatus.FAILED
            if target_status not in _LEGAL_TRANSITIONS.get(
                simulation.status, frozenset()
            ):
                raise JanusSimulationStatusError(
                    simulation_id, simulation.status.name, target_status.name
                )

            updated = _with_status(
                simulation, target_status, completed_at=_utcnow()
            )
            self._simulations[simulation_id] = updated
            return updated

    # -----------------------------------------------------------------
    # Multi-Path Simulation
    # -----------------------------------------------------------------

    def run_multi_path_simulation(
        self,
        title: str,
        description: str,
        paths: tuple[tuple[str, tuple[SimulationOutcome, ...]], ...],
        uncertainty: UncertaintyProfile,
        metadata,
    ) -> MultiPathSimulationResult:
        """
        Run one OutcomeSimulation per candidate decision path.

        `paths` is a tuple of (path_label, candidate_outcomes) pairs, each
        representing an alternative triggering event / decision path to
        simulate independently. All bounded-exploration limits apply per
        path and across the aggregate set.
        """
        with self._lock:
            self._ensure_operational()

            if not title.strip():
                raise JanusMissingRequiredFieldError("title", engine=_ENGINE_NAME)
            if not paths:
                raise JanusMissingRequiredFieldError("paths", engine=_ENGINE_NAME)

            self._validate_uncertainty("<pending>", uncertainty)

            total_outcomes = sum(len(outcomes) for _, outcomes in paths)
            self._enforce_compute_budget(total_outcomes, num_paths=len(paths))

            simulation_ids: list[str] = []
            path_labels: list[str] = []
            bounded_counts: list[int] = []
            pruned_counts: list[int] = []
            started_at = _utcnow()

            for path_label, candidate_outcomes in paths:
                if not path_label.strip():
                    raise JanusMissingRequiredFieldError(
                        "path_label", engine=_ENGINE_NAME
                    )
                if not candidate_outcomes:
                    raise JanusMissingRequiredFieldError(
                        f"candidate_outcomes for path '{path_label}'",
                        engine=_ENGINE_NAME,
                    )

                for outcome in candidate_outcomes:
                    if not 0.0 <= outcome.probability <= 1.0:
                        raise JanusInvalidProbabilityError(
                            outcome.probability,
                            field=(
                                f"paths[{path_label}]"
                                f"[{outcome.outcome_id}].probability"
                            ),
                            engine=_ENGINE_NAME,
                        )

                self._enforce_simulation_count_bounds()

                self._enforce_branch_depth(
                    tuple(candidate_outcomes), self._bounds.max_branch_depth
                )

                bounded_outcomes, pruned_count = self._prune_outcomes(
                    tuple(candidate_outcomes)
                )

                distribution = self._build_probability_distribution(
                    label=f"{title} — {path_label} Distribution",
                    outcomes=bounded_outcomes,
                    uncertainty=uncertainty,
                )

                try:
                    simulation = OutcomeSimulation.create(
                        title=f"{title} — {path_label}",
                        description=description,
                        triggering_event=path_label,
                        outcomes=list(bounded_outcomes),
                        probability_distribution=distribution,
                        uncertainty=uncertainty,
                        metadata=metadata,
                    )
                except ValueError as exc:
                    raise JanusSimulationExecutionError(
                        "<pending>", str(exc)
                    ) from exc

                running = _with_status(simulation, SimulationStatus.RUNNING)
                completed = _with_status(
                    running, SimulationStatus.COMPLETED, completed_at=_utcnow()
                )

                self._simulations[completed.simulation_id] = completed
                self._pruned_outcome_counts[completed.simulation_id] = pruned_count
                self._total_simulations_created += 1

                simulation_ids.append(completed.simulation_id)
                path_labels.append(path_label)
                bounded_counts.append(len(bounded_outcomes))
                pruned_counts.append(pruned_count)

            return MultiPathSimulationResult(
                path_simulation_ids=tuple(simulation_ids),
                paths=tuple(path_labels),
                bounded_outcome_counts=tuple(bounded_counts),
                pruned_outcome_counts=tuple(pruned_counts),
                started_at=started_at,
            )

    # -----------------------------------------------------------------
    # Scenario Simulation
    # -----------------------------------------------------------------

    def run_scenario_simulation(
        self,
        scenario_id: str,
        branch_id: Optional[str],
        title: str,
        description: str,
        triggering_event: str,
        candidate_outcomes: tuple[SimulationOutcome, ...],
        uncertainty: UncertaintyProfile,
        metadata,
    ) -> ScenarioSimulationResult:
        """
        Run an OutcomeSimulation derived from a Scenario (and optionally a
        specific ScenarioBranch), tagging the resulting simulation's
        triggering_event with the scenario/branch context.

        Does not select the branch or scenario; that remains VEGA's role.
        """
        with self._lock:
            self._ensure_operational()

            if not scenario_id.strip():
                raise JanusMissingRequiredFieldError(
                    "scenario_id", engine=_ENGINE_NAME
                )

            request = SimulationRunRequest(
                title=title,
                description=description,
                triggering_event=triggering_event,
                uncertainty=uncertainty,
                metadata=metadata,
                candidate_outcomes=candidate_outcomes,
            )

            response = self.run_simulation(request)

            return ScenarioSimulationResult(
                simulation_id=response.simulation.simulation_id,
                scenario_id=scenario_id,
                branch_id=branch_id,
                triggering_event=triggering_event,
                outcome_count=len(response.simulation.outcomes),
                started_at=response.started_at,
            )

    # -----------------------------------------------------------------
    # Forecast Simulation
    # -----------------------------------------------------------------

    def run_forecast_simulation(
        self,
        forecast_id: str,
        title: str,
        description: str,
        triggering_event: str,
        probability_distribution: ProbabilityDistribution,
        uncertainty: UncertaintyProfile,
        metadata,
        effect_template: Optional[dict[str, tuple[str, ...]]] = None,
    ) -> ForecastSimulationResult:
        """
        Run an OutcomeSimulation seeded from a Forecast's
        ProbabilityDistribution: each named outcome in the distribution
        becomes a candidate SimulationOutcome, carrying the distribution's
        probability and the forecast's uncertainty/confidence.

        `effect_template` optionally maps outcome label -> (short, medium,
        long) effect descriptions; outcomes without a template entry are
        seeded with empty effect lists.
        """
        with self._lock:
            self._ensure_operational()

            if not forecast_id.strip():
                raise JanusMissingRequiredFieldError(
                    "forecast_id", engine=_ENGINE_NAME
                )
            if not probability_distribution.outcomes:
                raise JanusMissingRequiredFieldError(
                    "probability_distribution.outcomes", engine=_ENGINE_NAME
                )

            template = effect_template or {}

            candidate_outcomes: list[SimulationOutcome] = []
            for label, probability in probability_distribution.outcomes.items():
                short_term, medium_term, long_term = template.get(
                    label, ((), (), ())
                )
                candidate_outcomes.append(
                    SimulationOutcome.create(
                        label=label,
                        description=f"Outcome '{label}' derived from forecast '{forecast_id}'.",
                        probability=probability,
                        short_term_effects=list(short_term),
                        medium_term_effects=list(medium_term),
                        long_term_effects=list(long_term),
                        risk_implications=[],
                        opportunity_implications=[],
                        confidence=probability_distribution.confidence,
                    )
                )

            request = SimulationRunRequest(
                title=title,
                description=description,
                triggering_event=triggering_event,
                uncertainty=uncertainty,
                metadata=metadata,
                candidate_outcomes=tuple(candidate_outcomes),
            )

            response = self.run_simulation(request)

            return ForecastSimulationResult(
                simulation_id=response.simulation.simulation_id,
                forecast_id=forecast_id,
                triggering_event=triggering_event,
                seeded_outcome_count=len(candidate_outcomes),
                started_at=response.started_at,
            )

    # -----------------------------------------------------------------
    # Simulation Comparison
    # -----------------------------------------------------------------

    def compare_simulations(
        self, simulation_id_a: str, simulation_id_b: str
    ) -> SimulationComparisonResult:
        """Produce a structured comparison between two OutcomeSimulations."""
        with self._lock:
            self._ensure_operational()

            sim_a = self._simulations.get(simulation_id_a)
            if sim_a is None:
                raise JanusSimulationNotFoundError(simulation_id_a)

            sim_b = self._simulations.get(simulation_id_b)
            if sim_b is None:
                raise JanusSimulationNotFoundError(simulation_id_b)

            labels_a = {o.label for o in sim_a.outcomes}
            labels_b = {o.label for o in sim_b.outcomes}

            shared = tuple(sorted(labels_a & labels_b))
            unique_a = tuple(sorted(labels_a - labels_b))
            unique_b = tuple(sorted(labels_b - labels_a))

            most_probable_a = sim_a.most_probable_outcome
            most_probable_b = sim_b.most_probable_outcome

            return SimulationComparisonResult(
                simulation_id_a=sim_a.simulation_id,
                simulation_id_b=sim_b.simulation_id,
                triggering_event_a=sim_a.triggering_event,
                triggering_event_b=sim_b.triggering_event,
                status_a=sim_a.status.name,
                status_b=sim_b.status.name,
                outcome_count_a=len(sim_a.outcomes),
                outcome_count_b=len(sim_b.outcomes),
                most_probable_label_a=(
                    most_probable_a.label if most_probable_a else None
                ),
                most_probable_label_b=(
                    most_probable_b.label if most_probable_b else None
                ),
                most_probable_probability_a=(
                    most_probable_a.probability if most_probable_a else None
                ),
                most_probable_probability_b=(
                    most_probable_b.probability if most_probable_b else None
                ),
                shared_outcome_labels=shared,
                unique_to_a=unique_a,
                unique_to_b=unique_b,
                compared_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Simulation Evaluation
    # -----------------------------------------------------------------

    def evaluate_simulation(self, simulation_id: str) -> SimulationEvaluation:
        """
        Produce a structured evaluation of a single OutcomeSimulation,
        aggregating consequence-chain effects across all outcomes.

        This is an evaluation for review; it does not select or approve
        any outcome.
        """
        with self._lock:
            self._ensure_operational()

            simulation = self._simulations.get(simulation_id)
            if simulation is None:
                raise JanusSimulationNotFoundError(simulation_id)

            outcomes = simulation.outcomes
            most_probable = simulation.most_probable_outcome

            probabilities = [o.probability for o in outcomes]
            avg_probability = (
                statistics.fmean(probabilities) if probabilities else None
            )

            short_effects: set[str] = set()
            medium_effects: set[str] = set()
            long_effects: set[str] = set()
            risk_implications: set[str] = set()
            opportunity_implications: set[str] = set()

            for outcome in outcomes:
                short_effects.update(outcome.short_term_effects)
                medium_effects.update(outcome.medium_term_effects)
                long_effects.update(outcome.long_term_effects)
                risk_implications.update(outcome.risk_implications)
                opportunity_implications.update(outcome.opportunity_implications)

            return SimulationEvaluation(
                simulation_id=simulation.simulation_id,
                title=simulation.title,
                status=simulation.status.name,
                triggering_event=simulation.triggering_event,
                outcome_count=len(outcomes),
                most_probable_outcome_label=(
                    most_probable.label if most_probable else None
                ),
                most_probable_outcome_probability=(
                    most_probable.probability if most_probable else None
                ),
                average_outcome_probability=avg_probability,
                aggregate_short_term_effects=tuple(sorted(short_effects)),
                aggregate_medium_term_effects=tuple(sorted(medium_effects)),
                aggregate_long_term_effects=tuple(sorted(long_effects)),
                aggregate_risk_implications=tuple(sorted(risk_implications)),
                aggregate_opportunity_implications=tuple(
                    sorted(opportunity_implications)
                ),
                uncertainty_volatility=simulation.uncertainty.volatility_score,
                evaluated_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def get_statistics(self) -> SimulationStatistics:
        """Return aggregate statistics across all registered simulations."""
        with self._lock:
            self._ensure_operational()

            simulations = list(self._simulations.values())

            by_status: dict[str, int] = {}
            for status in SimulationStatus:
                by_status[status.name] = sum(
                    1 for sim in simulations if sim.status == status
                )

            total_outcomes = sum(len(sim.outcomes) for sim in simulations)
            avg_outcomes = (
                statistics.fmean(len(sim.outcomes) for sim in simulations)
                if simulations
                else None
            )

            most_probable_probs: list[float] = []
            for sim in simulations:
                most_probable = sim.most_probable_outcome
                if most_probable is not None:
                    most_probable_probs.append(most_probable.probability)
            avg_most_probable = (
                statistics.fmean(most_probable_probs)
                if most_probable_probs
                else None
            )

            volatilities = [sim.uncertainty.volatility_score for sim in simulations]
            avg_volatility = (
                statistics.fmean(volatilities) if volatilities else None
            )

            total = len(simulations)
            completion_rate = (
                by_status.get(SimulationStatus.COMPLETED.name, 0) / total
                if total
                else None
            )
            abort_rate = (
                by_status.get(SimulationStatus.ABORTED.name, 0) / total
                if total
                else None
            )
            failure_rate = (
                by_status.get(SimulationStatus.FAILED.name, 0) / total
                if total
                else None
            )

            return SimulationStatistics(
                total_simulations=total,
                by_status=by_status,
                total_outcomes=total_outcomes,
                average_outcomes_per_simulation=avg_outcomes,
                average_most_probable_outcome_probability=avg_most_probable,
                average_uncertainty_volatility=avg_volatility,
                completion_rate=completion_rate,
                abort_rate=abort_rate,
                failure_rate=failure_rate,
                generated_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Health Report
    # -----------------------------------------------------------------

    def check_health(self) -> SimulationHealthReport:
        """Return a health snapshot of the engine and its registry."""
        with self._lock:
            simulations = list(self._simulations.values())

            pending = sum(
                1 for s in simulations if s.status == SimulationStatus.PENDING
            )
            running = sum(
                1 for s in simulations if s.status == SimulationStatus.RUNNING
            )
            completed = sum(
                1 for s in simulations if s.status == SimulationStatus.COMPLETED
            )
            failed = sum(
                1 for s in simulations if s.status == SimulationStatus.FAILED
            )
            aborted = sum(
                1 for s in simulations if s.status == SimulationStatus.ABORTED
            )

            healthy = self._initialized and not self._shutdown

            return SimulationHealthReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                is_initialized=self._initialized,
                is_shutdown=self._shutdown,
                total_simulations=len(simulations),
                pending_simulations=pending,
                running_simulations=running,
                completed_simulations=completed,
                failed_simulations=failed,
                aborted_simulations=aborted,
                bounds=self._bounds,
                healthy=healthy,
                generated_at=_utcnow(),
            )

    # -----------------------------------------------------------------
    # Diagnostics Report
    # -----------------------------------------------------------------

    def run_diagnostics(self) -> SimulationDiagnosticsReport:
        """
        Run a diagnostics pass over all registered simulations, flagging
        integrity concerns without mutating any state.
        """
        with self._lock:
            self._ensure_operational()

            simulations = list(self._simulations.values())

            missing_outcomes = sum(1 for s in simulations if not s.outcomes)

            zero_uncertainty = sum(
                1
                for s in simulations
                if s.uncertainty.volatility_score == 0.0
                and s.uncertainty.unknown_risk_exposure == 0.0
            )

            exceeding_limit = sum(
                1
                for s in simulations
                if len(s.outcomes) > self._bounds.max_outcomes_per_simulation
            )

            outcome_counts = [len(s.outcomes) for s in simulations]
            avg_outcome_count = (
                statistics.fmean(outcome_counts) if outcome_counts else 0.0
            )

            all_probabilities = [
                o.probability for s in simulations for o in s.outcomes
            ]
            avg_probability = (
                statistics.fmean(all_probabilities) if all_probabilities else None
            )

            status_distribution: dict[str, int] = {}
            for s in simulations:
                status_distribution[s.status.name] = (
                    status_distribution.get(s.status.name, 0) + 1
                )

            pruned_total = sum(self._pruned_outcome_counts.values())

            return SimulationDiagnosticsReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                total_simulations=len(simulations),
                simulations_missing_outcomes=missing_outcomes,
                simulations_with_zero_uncertainty=zero_uncertainty,
                simulations_exceeding_outcome_limit=exceeding_limit,
                average_outcome_count=avg_outcome_count,
                average_outcome_probability=avg_probability,
                status_distribution=status_distribution,
                pruned_outcome_total=pruned_total,
                generated_at=_utcnow(),
            )


# ---------------------------------------------------------------------------
# Module-level Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _with_status(
    simulation: OutcomeSimulation,
    status: SimulationStatus,
    completed_at: Optional[datetime] = None,
) -> OutcomeSimulation:
    """Return a copy of the given OutcomeSimulation with a new status."""
    return OutcomeSimulation(
        simulation_id=simulation.simulation_id,
        title=simulation.title,
        description=simulation.description,
        triggering_event=simulation.triggering_event,
        outcomes=simulation.outcomes,
        probability_distribution=simulation.probability_distribution,
        uncertainty=simulation.uncertainty,
        status=status,
        metadata=simulation.metadata,
        created_at=simulation.created_at,
        completed_at=(
            completed_at if completed_at is not None else simulation.completed_at
        ),
    )