"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: janus.py

Defines JanusSubsystem, the top-level coordinator for the JANUS subsystem
of POLARIS.

JanusSubsystem is responsible for:
  - Engine construction and dependency wiring
  - Lifecycle coordination (initialize / shutdown)
  - Rollback on initialization failure
  - Subsystem-level health aggregation
  - Subsystem-level diagnostics aggregation
  - Subsystem-level statistics aggregation
  - Typed engine access for upstream consumers

Constitutional boundary (JANUS Critical Laws 1-8):
  JANUS owns future modeling, scenario analysis, forecasting, branch
  analysis, counterfactual analysis, uncertainty modeling, future risk and
  opportunity analysis, outcome simulation, strategic forecasting, timeline
  projection, probability estimation, scenario evaluation, and scenario
  integrity validation.

  JANUS never owns identity, knowledge storage, learning, decisions, or
  plans. JanusSubsystem does not implement any of those responsibilities;
  it only constructs, wires, and coordinates the JANUS engines that do.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .exceptions import (
    JanusAlreadyInitializedError,
    JanusError,
    JanusNotInitializedError,
    JanusOrchestratorCoordinationError,
    JanusShutdownError,
)
from .interfaces import (
    IBranchAnalysisEngine,
    ICounterfactualEngine,
    IForecastingEngine,
    IFutureModelingEngine,
    IFutureOpportunityEngine,
    IFutureOrchestrator,
    IFutureMemoryInterface,
    IFutureRiskEngine,
    IOutcomeSimulationEngine,
    IProbabilityEngine,
    IScenarioEngine,
    IScenarioEvaluationEngine,
    IScenarioIntegrityEngine,
    IStrategicForecastEngine,
    ITimelineProjectionEngine,
    IUncertaintyEngine,
    JanusEngineLifecycle,
)
from .models import (
    ForecastType,
    FutureAssessmentStatus,
    ProjectionStatus,
    ScenarioStatus,
    ScenarioType,
    SimulationStatus,
)
from .engines.scenario_engine import ScenarioEngine
from .engines.forecasting_engine import ForecastingEngine
from .engines.future_modeling_engine import FutureModelingEngine
from .engines.branch_analysis_engine import BranchAnalysisEngine
from .engines.counterfactual_engine import CounterfactualEngine
from .engines.uncertainty_engine import UncertaintyEngine
from .engines.future_risk_engine import FutureRiskEngine
from .engines.future_opportunity_engine import FutureOpportunityEngine
from .engines.outcome_simulation_engine import OutcomeSimulationEngine
from .engines.strategic_forecast_engine import StrategicForecastEngine
from .engines.timeline_projection_engine import TimelineProjectionEngine
from .engines.probability_engine import ProbabilityEngine
from .engines.scenario_evaluation_engine import ScenarioEvaluationEngine
from .engines.scenario_integrity_engine import ScenarioIntegrityEngine
from .engines.future_memory_interface import FutureMemoryInterface
from .engines.future_orchestrator import FutureOrchestrator


logger = logging.getLogger(__name__)


__all__ = [
    "JanusSubsystem",
    "JanusSubsystemStatus",
    "JanusEngineHealth",
    "JanusSubsystemHealth",
    "JanusEngineDiagnostics",
    "JanusSubsystemDiagnostics",
    "JanusSubsystemStatistics",
    "JanusUnknownEngineError",
]


# ---------------------------------------------------------------------------
# Subsystem-local errors
# ---------------------------------------------------------------------------


class JanusUnknownEngineError(JanusError):
    """
    Raised when an engine is requested from JanusSubsystem by an
    engine_id that does not correspond to any owned JANUS engine.
    """

    def __init__(self, engine_id: str, known_engine_ids: tuple[str, ...]) -> None:
        super().__init__(
            f"Unknown JANUS engine_id '{engine_id}'. "
            f"Known engine_ids: {', '.join(known_engine_ids)}.",
            engine="JanusSubsystem",
            context={"engine_id": engine_id, "known_engine_ids": known_engine_ids},
        )
        self.engine_id = engine_id


# ---------------------------------------------------------------------------
# Subsystem lifecycle status
# ---------------------------------------------------------------------------


class JanusSubsystemStatus(Enum):
    """Lifecycle status of the JANUS subsystem as a whole."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    OPERATIONAL = "operational"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Health reporting models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JanusEngineHealth:
    """Health snapshot for a single JANUS engine."""

    engine_id: str
    engine_name: str
    engine_version: str
    is_initialized: bool
    is_healthy: bool
    detail: str


@dataclass(frozen=True)
class JanusSubsystemHealth:
    """Aggregated health snapshot for the JANUS subsystem."""

    subsystem: str
    subsystem_version: str
    status: JanusSubsystemStatus
    engines: tuple[JanusEngineHealth, ...]
    healthy_engine_count: int
    total_engine_count: int
    generated_at: datetime

    @property
    def is_fully_healthy(self) -> bool:
        return (
            self.status == JanusSubsystemStatus.OPERATIONAL
            and self.healthy_engine_count == self.total_engine_count
        )

    @property
    def is_degraded(self) -> bool:
        return (
            self.status == JanusSubsystemStatus.OPERATIONAL
            and self.healthy_engine_count < self.total_engine_count
        )


# ---------------------------------------------------------------------------
# Diagnostics reporting models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JanusEngineDiagnostics:
    """Diagnostic snapshot for a single JANUS engine."""

    engine_id: str
    engine_name: str
    engine_version: str
    is_initialized: bool
    implementation_class: str
    implementation_module: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JanusSubsystemDiagnostics:
    """Aggregated diagnostics snapshot for the JANUS subsystem."""

    subsystem: str
    subsystem_version: str
    status: JanusSubsystemStatus
    engines: tuple[JanusEngineDiagnostics, ...]
    engine_init_order: tuple[str, ...]
    subsystem_details: dict[str, Any]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Statistics reporting models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JanusSubsystemStatistics:
    """Aggregated statistics snapshot for the JANUS subsystem."""

    subsystem: str
    subsystem_version: str
    status: JanusSubsystemStatus
    engine_count: int
    initialized_engine_count: int
    initialization_count: int
    shutdown_count: int
    uptime_seconds: float
    created_at: datetime
    initialized_at: Optional[datetime]
    shutdown_at: Optional[datetime]
    engine_statistics: dict[str, dict[str, Any]]
    generated_at: datetime


# ---------------------------------------------------------------------------
# JanusSubsystem
# ---------------------------------------------------------------------------


class JanusSubsystem:
    """
    Top-level coordinator for JANUS — the Future Modeling & Scenario
    Intelligence Core of POLARIS.

    JanusSubsystem constructs, wires, and coordinates the sixteen JANUS
    engines:

      Foundation tier:
        - ProbabilityEngine
        - UncertaintyEngine
        - ScenarioIntegrityEngine
        - FutureMemoryInterface

      Core modeling tier:
        - ScenarioEngine
        - BranchAnalysisEngine
        - CounterfactualEngine
        - FutureModelingEngine
        - FutureRiskEngine
        - FutureOpportunityEngine
        - OutcomeSimulationEngine
        - TimelineProjectionEngine
        - StrategicForecastEngine
        - ForecastingEngine
        - ScenarioEvaluationEngine

      Executive tier:
        - FutureOrchestrator

    JanusSubsystem itself never reasons, decides, plans, stores knowledge,
    or owns identity (JANUS Critical Laws 1-8). It exists purely to manage
    the lifecycle and coordination of the JANUS engines listed above.

    Thread safety:
        All lifecycle transitions (initialize / shutdown) are guarded by an
        internal re-entrant lock. Health, diagnostics, and statistics
        aggregation take a brief lock to snapshot subsystem-level state
        before reading engine-level state.
    """

    SUBSYSTEM_NAME: str = "JANUS"
    SUBSYSTEM_VERSION: str = "5.1"

    #: Canonical engine construction / initialization order.
    #: Earlier engines never depend on later engines. Shutdown proceeds in
    #: the reverse of this order so dependents are released before their
    #: dependencies.
    _ENGINE_INIT_ORDER: tuple[str, ...] = (
        "probability_engine",
        "uncertainty_engine",
        "scenario_integrity_engine",
        "future_memory_interface",
        "scenario_engine",
        "branch_analysis_engine",
        "counterfactual_engine",
        "future_modeling_engine",
        "future_risk_engine",
        "future_opportunity_engine",
        "outcome_simulation_engine",
        "timeline_projection_engine",
        "strategic_forecast_engine",
        "forecasting_engine",
        "scenario_evaluation_engine",
        "future_orchestrator",
    )

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """
        Construct the JANUS subsystem and wire all sixteen engines
        according to their constitutional dependency relationships.

        Construction never raises JanusError under normal conditions
        (engines are expected to perform lightweight setup only); resource
        acquisition occurs in initialize().

        Args:
            config: Optional subsystem configuration. May contain an
                "engines" mapping of engine_id -> engine-specific config
                dict, consumed via _engine_config().
        """
        self._config: dict[str, Any] = dict(config or {})
        self._lock = threading.RLock()

        self._status: JanusSubsystemStatus = JanusSubsystemStatus.UNINITIALIZED
        self._created_at: datetime = datetime.now(timezone.utc)
        self._initialized_at: Optional[datetime] = None
        self._shutdown_at: Optional[datetime] = None
        self._initialization_count: int = 0
        self._shutdown_count: int = 0
        self._failed_engine_id: Optional[str] = None
        self._last_error: Optional[str] = None

        # -------------------------------------------------------------
        # Foundation tier
        # -------------------------------------------------------------

        self._probability_engine: IProbabilityEngine = ProbabilityEngine(
            config=self._engine_config("probability_engine"),
        )

        self._uncertainty_engine: IUncertaintyEngine = UncertaintyEngine(
            config=self._engine_config("uncertainty_engine"),
        )

        self._scenario_integrity_engine: IScenarioIntegrityEngine = ScenarioIntegrityEngine(
            uncertainty_engine=self._uncertainty_engine,
            probability_engine=self._probability_engine,
            config=self._engine_config("scenario_integrity_engine"),
        )

        # Reads from ECHO, CHRONOS, ASTRA, CONSTELLATION only.
        # JANUS does not own memory; this interface never writes upstream.
        self._future_memory_interface: IFutureMemoryInterface = FutureMemoryInterface(
            config=self._engine_config("future_memory_interface"),
        )

        # -------------------------------------------------------------
        # Core modeling tier
        # -------------------------------------------------------------

        self._scenario_engine: IScenarioEngine = ScenarioEngine(
            probability_engine=self._probability_engine,
            uncertainty_engine=self._uncertainty_engine,
            integrity_engine=self._scenario_integrity_engine,
            config=self._engine_config("scenario_engine"),
        )

        self._branch_analysis_engine: IBranchAnalysisEngine = BranchAnalysisEngine(
            probability_engine=self._probability_engine,
            integrity_engine=self._scenario_integrity_engine,
            config=self._engine_config("branch_analysis_engine"),
        )

        self._counterfactual_engine: ICounterfactualEngine = CounterfactualEngine(
            scenario_engine=self._scenario_engine,
            uncertainty_engine=self._uncertainty_engine,
            integrity_engine=self._scenario_integrity_engine,
            config=self._engine_config("counterfactual_engine"),
        )

        self._future_modeling_engine: IFutureModelingEngine = FutureModelingEngine(
            uncertainty_engine=self._uncertainty_engine,
            integrity_engine=self._scenario_integrity_engine,
            config=self._engine_config("future_modeling_engine"),
        )

        self._future_risk_engine: IFutureRiskEngine = FutureRiskEngine(
            probability_engine=self._probability_engine,
            uncertainty_engine=self._uncertainty_engine,
            config=self._engine_config("future_risk_engine"),
        )

        self._future_opportunity_engine: IFutureOpportunityEngine = FutureOpportunityEngine(
            probability_engine=self._probability_engine,
            uncertainty_engine=self._uncertainty_engine,
            config=self._engine_config("future_opportunity_engine"),
        )

        self._outcome_simulation_engine: IOutcomeSimulationEngine = OutcomeSimulationEngine(
            probability_engine=self._probability_engine,
            uncertainty_engine=self._uncertainty_engine,
            integrity_engine=self._scenario_integrity_engine,
            config=self._engine_config("outcome_simulation_engine"),
        )

        self._timeline_projection_engine: ITimelineProjectionEngine = TimelineProjectionEngine(
            probability_engine=self._probability_engine,
            uncertainty_engine=self._uncertainty_engine,
            integrity_engine=self._scenario_integrity_engine,
            config=self._engine_config("timeline_projection_engine"),
        )

        self._strategic_forecast_engine: IStrategicForecastEngine = StrategicForecastEngine(
            future_modeling_engine=self._future_modeling_engine,
            uncertainty_engine=self._uncertainty_engine,
            probability_engine=self._probability_engine,
            config=self._engine_config("strategic_forecast_engine"),
        )

        self._forecasting_engine: IForecastingEngine = ForecastingEngine(
            probability_engine=self._probability_engine,
            uncertainty_engine=self._uncertainty_engine,
            integrity_engine=self._scenario_integrity_engine,
            config=self._engine_config("forecasting_engine"),
        )

        self._scenario_evaluation_engine: IScenarioEvaluationEngine = ScenarioEvaluationEngine(
            probability_engine=self._probability_engine,
            scenario_engine=self._scenario_engine,
            config=self._engine_config("scenario_evaluation_engine"),
        )

        # -------------------------------------------------------------
        # Executive tier
        # -------------------------------------------------------------

        self._future_orchestrator: IFutureOrchestrator = FutureOrchestrator(
            scenario_engine=self._scenario_engine,
            forecasting_engine=self._forecasting_engine,
            future_modeling_engine=self._future_modeling_engine,
            branch_analysis_engine=self._branch_analysis_engine,
            counterfactual_engine=self._counterfactual_engine,
            uncertainty_engine=self._uncertainty_engine,
            future_risk_engine=self._future_risk_engine,
            future_opportunity_engine=self._future_opportunity_engine,
            outcome_simulation_engine=self._outcome_simulation_engine,
            strategic_forecast_engine=self._strategic_forecast_engine,
            timeline_projection_engine=self._timeline_projection_engine,
            probability_engine=self._probability_engine,
            scenario_evaluation_engine=self._scenario_evaluation_engine,
            scenario_integrity_engine=self._scenario_integrity_engine,
            future_memory_interface=self._future_memory_interface,
            config=self._engine_config("future_orchestrator"),
        )

        # Ordered map used by lifecycle, health, diagnostics, and
        # statistics aggregation. Iteration order == _ENGINE_INIT_ORDER.
        self._engines: dict[str, JanusEngineLifecycle] = {
            "probability_engine": self._probability_engine,
            "uncertainty_engine": self._uncertainty_engine,
            "scenario_integrity_engine": self._scenario_integrity_engine,
            "future_memory_interface": self._future_memory_interface,
            "scenario_engine": self._scenario_engine,
            "branch_analysis_engine": self._branch_analysis_engine,
            "counterfactual_engine": self._counterfactual_engine,
            "future_modeling_engine": self._future_modeling_engine,
            "future_risk_engine": self._future_risk_engine,
            "future_opportunity_engine": self._future_opportunity_engine,
            "outcome_simulation_engine": self._outcome_simulation_engine,
            "timeline_projection_engine": self._timeline_projection_engine,
            "strategic_forecast_engine": self._strategic_forecast_engine,
            "forecasting_engine": self._forecasting_engine,
            "scenario_evaluation_engine": self._scenario_evaluation_engine,
            "future_orchestrator": self._future_orchestrator,
        }

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _engine_config(self, engine_id: str) -> dict[str, Any]:
        """
        Return the engine-specific configuration sub-dict for engine_id,
        drawn from config["engines"][engine_id]. Returns an empty dict if
        no such configuration is present.
        """
        engines_config = self._config.get("engines", {})
        if not isinstance(engines_config, dict):
            return {}
        engine_config = engines_config.get(engine_id, {})
        if not isinstance(engine_config, dict):
            return {}
        return dict(engine_config)

    # ------------------------------------------------------------------
    # Lifecycle: initialize
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize all JANUS engines in constitutional dependency order.

        On success, the subsystem transitions to OPERATIONAL and all
        sixteen engines are initialized and ready to serve requests.

        On failure of any single engine's initialize(), all previously
        initialized engines in this call are shut down in reverse order
        (rollback), the subsystem transitions to FAILED, and the
        triggering exception is re-raised.

        Raises:
            JanusAlreadyInitializedError: If the subsystem is already
                OPERATIONAL.
            JanusShutdownError: If the subsystem has already been shut
                down.
            JanusOrchestratorCoordinationError: If the subsystem
                previously entered a FAILED state and requires
                reconstruction, or if a non-JanusError exception was
                raised by an engine during initialization.
            JanusError: Whatever JanusError subclass was raised by the
                failing engine, re-raised after rollback completes.
        """
        with self._lock:
            if self._status == JanusSubsystemStatus.OPERATIONAL:
                raise JanusAlreadyInitializedError(self.SUBSYSTEM_NAME)

            if self._status == JanusSubsystemStatus.SHUTDOWN:
                raise JanusShutdownError(self.SUBSYSTEM_NAME)

            if self._status == JanusSubsystemStatus.FAILED:
                raise JanusOrchestratorCoordinationError(
                    "JANUS subsystem previously failed during initialization "
                    "and is in a terminal FAILED state. Construct a new "
                    "JanusSubsystem instance to retry.",
                    failed_engine=self._failed_engine_id,
                )

            self._status = JanusSubsystemStatus.INITIALIZING
            initialized_so_far: list[tuple[str, JanusEngineLifecycle]] = []

            for engine_id in self._ENGINE_INIT_ORDER:
                engine = self._engines[engine_id]

                try:
                    engine.initialize()
                except JanusError as exc:
                    logger.error(
                        "JANUS initialization failed at engine '%s' (%s): %s",
                        engine_id,
                        type(exc).__name__,
                        exc,
                    )
                    self._rollback(initialized_so_far, engine_id)
                    self._status = JanusSubsystemStatus.FAILED
                    self._failed_engine_id = engine_id
                    self._last_error = str(exc)
                    raise
                except Exception as exc:  # noqa: BLE001 - convert to JanusError
                    logger.error(
                        "JANUS initialization failed at engine '%s' with an "
                        "unexpected exception (%s): %s",
                        engine_id,
                        type(exc).__name__,
                        exc,
                    )
                    self._rollback(initialized_so_far, engine_id)
                    self._status = JanusSubsystemStatus.FAILED
                    self._failed_engine_id = engine_id
                    self._last_error = str(exc)
                    raise JanusOrchestratorCoordinationError(
                        f"Engine '{engine_id}' raised an unexpected "
                        f"{type(exc).__name__} during initialization: {exc}",
                        failed_engine=engine_id,
                    ) from exc
                else:
                    initialized_so_far.append((engine_id, engine))

            self._status = JanusSubsystemStatus.OPERATIONAL
            self._initialization_count += 1
            self._initialized_at = datetime.now(timezone.utc)
            self._failed_engine_id = None
            self._last_error = None

            logger.info(
                "JANUS subsystem v%s initialized successfully (%d engines).",
                self.SUBSYSTEM_VERSION,
                len(self._engines),
            )

    def _rollback(
        self,
        initialized_so_far: list[tuple[str, JanusEngineLifecycle]],
        failed_engine_id: str,
    ) -> None:
        """
        Best-effort rollback: shut down every engine that was successfully
        initialized earlier in this initialize() call, in reverse order.

        Shutdown failures during rollback are logged but never raised,
        since the original initialization failure is the authoritative
        error for this call.
        """
        if not initialized_so_far:
            logger.warning(
                "JANUS rollback triggered by engine '%s'; no engines had "
                "completed initialization yet, nothing to roll back.",
                failed_engine_id,
            )
            return

        logger.warning(
            "JANUS rolling back %d previously-initialized engine(s) after "
            "failure in '%s'.",
            len(initialized_so_far),
            failed_engine_id,
        )

        for engine_id, engine in reversed(initialized_so_far):
            try:
                engine.shutdown()
            except JanusError as rollback_exc:
                logger.warning(
                    "Rollback shutdown of engine '%s' raised %s: %s "
                    "(continuing rollback).",
                    engine_id,
                    type(rollback_exc).__name__,
                    rollback_exc,
                )
            except Exception as rollback_exc:  # noqa: BLE001
                logger.warning(
                    "Rollback shutdown of engine '%s' raised unexpected %s: %s "
                    "(continuing rollback).",
                    engine_id,
                    type(rollback_exc).__name__,
                    rollback_exc,
                )

    # ------------------------------------------------------------------
    # Lifecycle: shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Shut down all JANUS engines in reverse constitutional dependency
        order (executive tier first, foundation tier last), so that
        dependents always release resources before their dependencies do.

        Idempotent: calling shutdown() on an already-SHUTDOWN subsystem is
        a no-op.

        Raises:
            JanusNotInitializedError: If the subsystem was never
                successfully initialized.
            JanusOrchestratorCoordinationError: If one or more engines
                raised an error during shutdown. All engines are still
                given the opportunity to shut down regardless of earlier
                failures.
        """
        with self._lock:
            if self._status == JanusSubsystemStatus.SHUTDOWN:
                return

            if self._status == JanusSubsystemStatus.UNINITIALIZED:
                raise JanusNotInitializedError(self.SUBSYSTEM_NAME)

            self._status = JanusSubsystemStatus.SHUTTING_DOWN
            shutdown_errors: list[tuple[str, Exception]] = []

            for engine_id in reversed(self._ENGINE_INIT_ORDER):
                engine = self._engines[engine_id]

                if not self._safe_is_initialized(engine):
                    continue

                try:
                    engine.shutdown()
                except JanusError as exc:
                    logger.error(
                        "JANUS shutdown of engine '%s' raised %s: %s",
                        engine_id,
                        type(exc).__name__,
                        exc,
                    )
                    shutdown_errors.append((engine_id, exc))
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "JANUS shutdown of engine '%s' raised unexpected %s: %s",
                        engine_id,
                        type(exc).__name__,
                        exc,
                    )
                    shutdown_errors.append((engine_id, exc))

            self._status = JanusSubsystemStatus.SHUTDOWN
            self._shutdown_count += 1
            self._shutdown_at = datetime.now(timezone.utc)

            if shutdown_errors:
                failed_ids = ", ".join(engine_id for engine_id, _ in shutdown_errors)
                first_engine_id, first_exc = shutdown_errors[0]
                self._last_error = str(first_exc)
                logger.error(
                    "JANUS subsystem shut down with %d engine error(s): %s",
                    len(shutdown_errors),
                    failed_ids,
                )
                raise JanusOrchestratorCoordinationError(
                    f"One or more engines raised errors during shutdown: "
                    f"{failed_ids}.",
                    failed_engine=first_engine_id,
                ) from first_exc

            logger.info("JANUS subsystem shut down cleanly (%d engines).", len(self._engines))

    @staticmethod
    def _safe_is_initialized(engine: JanusEngineLifecycle) -> bool:
        """Read engine.is_initialized defensively; treat errors as False."""
        try:
            return bool(engine.is_initialized)
        except JanusError:
            return False
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Lifecycle: status
    # ------------------------------------------------------------------

    @property
    def status(self) -> JanusSubsystemStatus:
        """Current lifecycle status of the subsystem."""
        with self._lock:
            return self._status

    @property
    def is_initialized(self) -> bool:
        """True if the subsystem is OPERATIONAL."""
        with self._lock:
            return self._status == JanusSubsystemStatus.OPERATIONAL

    @property
    def is_shutdown(self) -> bool:
        """True if the subsystem has completed shutdown."""
        with self._lock:
            return self._status == JanusSubsystemStatus.SHUTDOWN

    @property
    def subsystem_name(self) -> str:
        return self.SUBSYSTEM_NAME

    @property
    def subsystem_version(self) -> str:
        return self.SUBSYSTEM_VERSION

    # ------------------------------------------------------------------
    # Engine access
    # ------------------------------------------------------------------

    def get_engine(self, engine_id: str) -> JanusEngineLifecycle:
        """
        Retrieve a JANUS engine by its canonical engine_id.

        Args:
            engine_id: One of the values in _ENGINE_INIT_ORDER, e.g.
                "scenario_engine", "future_orchestrator".

        Returns:
            The engine instance, implementing JanusEngineLifecycle and its
            corresponding I*Engine / I*Interface contract.

        Raises:
            JanusUnknownEngineError: If engine_id does not correspond to
                any JANUS engine.
        """
        try:
            return self._engines[engine_id]
        except KeyError:
            raise JanusUnknownEngineError(engine_id, self._ENGINE_INIT_ORDER) from None

    def list_engine_ids(self) -> tuple[str, ...]:
        """Return the canonical engine_ids in construction/initialization order."""
        return self._ENGINE_INIT_ORDER

    @property
    def probability_engine(self) -> IProbabilityEngine:
        """JANUS Probability Engine — probability distributions, confidence, levels."""
        return self._probability_engine

    @property
    def uncertainty_engine(self) -> IUncertaintyEngine:
        """JANUS Uncertainty Engine — uncertainty profiles and Law 6 enforcement."""
        return self._uncertainty_engine

    @property
    def scenario_integrity_engine(self) -> IScenarioIntegrityEngine:
        """JANUS Scenario Integrity Engine — forecast quality and constitutional checks."""
        return self._scenario_integrity_engine

    @property
    def future_memory_interface(self) -> IFutureMemoryInterface:
        """JANUS Future Memory Interface — reads ECHO, CHRONOS, ASTRA, CONSTELLATION."""
        return self._future_memory_interface

    @property
    def scenario_engine(self) -> IScenarioEngine:
        """JANUS Scenario Engine — alternative-future scenario generation."""
        return self._scenario_engine

    @property
    def branch_analysis_engine(self) -> IBranchAnalysisEngine:
        """JANUS Branch Analysis Engine — decision-branch exploration."""
        return self._branch_analysis_engine

    @property
    def counterfactual_engine(self) -> ICounterfactualEngine:
        """JANUS Counterfactual Engine — alternate-reality scenario construction."""
        return self._counterfactual_engine

    @property
    def future_modeling_engine(self) -> IFutureModelingEngine:
        """JANUS Future Modeling Engine — multi-horizon FutureModel construction."""
        return self._future_modeling_engine

    @property
    def future_risk_engine(self) -> IFutureRiskEngine:
        """JANUS Future Risk Engine — future-threat identification and scoring."""
        return self._future_risk_engine

    @property
    def future_opportunity_engine(self) -> IFutureOpportunityEngine:
        """JANUS Future Opportunity Engine — future-opportunity identification and scoring."""
        return self._future_opportunity_engine

    @property
    def outcome_simulation_engine(self) -> IOutcomeSimulationEngine:
        """JANUS Outcome Simulation Engine — consequence-chain simulation."""
        return self._outcome_simulation_engine

    @property
    def timeline_projection_engine(self) -> ITimelineProjectionEngine:
        """JANUS Timeline Projection Engine — future-state projection across time."""
        return self._timeline_projection_engine

    @property
    def strategic_forecast_engine(self) -> IStrategicForecastEngine:
        """JANUS Strategic Forecast Engine — strategic-outcome forecasting."""
        return self._strategic_forecast_engine

    @property
    def forecasting_engine(self) -> IForecastingEngine:
        """JANUS Forecasting Engine — probabilistic forecast generation."""
        return self._forecasting_engine

    @property
    def scenario_evaluation_engine(self) -> IScenarioEvaluationEngine:
        """JANUS Scenario Evaluation Engine — scenario ranking, comparison, dominance."""
        return self._scenario_evaluation_engine

    @property
    def future_orchestrator(self) -> IFutureOrchestrator:
        """JANUS Future Orchestrator — executive coordinator producing FutureAssessments."""
        return self._future_orchestrator

    # ------------------------------------------------------------------
    # Health aggregation
    # ------------------------------------------------------------------

    def check_health(self) -> JanusSubsystemHealth:
        """
        Aggregate health status across all JANUS engines.

        An engine is considered healthy if it reports is_initialized and
        the subsystem as a whole is OPERATIONAL. This method never raises;
        any per-engine error while reading lifecycle properties is
        captured in that engine's JanusEngineHealth.detail field and the
        engine is reported as unhealthy.

        Returns:
            JanusSubsystemHealth aggregating per-engine health alongside
            subsystem-level status.
        """
        with self._lock:
            status = self._status

        engine_healths: list[JanusEngineHealth] = []
        healthy_count = 0

        for engine_id in self._ENGINE_INIT_ORDER:
            engine = self._engines[engine_id]

            try:
                initialized = bool(engine.is_initialized)
                name = engine.engine_name
                version = engine.engine_version
            except JanusError as exc:
                engine_healths.append(
                    JanusEngineHealth(
                        engine_id=engine_id,
                        engine_name=engine_id,
                        engine_version="unknown",
                        is_initialized=False,
                        is_healthy=False,
                        detail=f"Health check raised {type(exc).__name__}: {exc}",
                    )
                )
                continue
            except Exception as exc:  # noqa: BLE001
                engine_healths.append(
                    JanusEngineHealth(
                        engine_id=engine_id,
                        engine_name=engine_id,
                        engine_version="unknown",
                        is_initialized=False,
                        is_healthy=False,
                        detail=f"Health check raised unexpected {type(exc).__name__}: {exc}",
                    )
                )
                continue

            is_healthy = initialized and status == JanusSubsystemStatus.OPERATIONAL

            if is_healthy:
                detail = "operational"
            elif not initialized:
                detail = "not initialized"
            else:
                detail = f"subsystem status is {status.value}, not operational"

            if is_healthy:
                healthy_count += 1

            engine_healths.append(
                JanusEngineHealth(
                    engine_id=engine_id,
                    engine_name=name,
                    engine_version=version,
                    is_initialized=initialized,
                    is_healthy=is_healthy,
                    detail=detail,
                )
            )

        return JanusSubsystemHealth(
            subsystem=self.SUBSYSTEM_NAME,
            subsystem_version=self.SUBSYSTEM_VERSION,
            status=status,
            engines=tuple(engine_healths),
            healthy_engine_count=healthy_count,
            total_engine_count=len(engine_healths),
            generated_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Diagnostics aggregation
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> JanusSubsystemDiagnostics:
        """
        Aggregate diagnostic information across all JANUS engines and the
        subsystem itself.

        Returns:
            JanusSubsystemDiagnostics describing the implementation class,
            module, version, and initialization state of every engine,
            plus subsystem-level lifecycle counters and constitutional
            metadata.
        """
        with self._lock:
            status = self._status
            created_at = self._created_at
            initialized_at = self._initialized_at
            shutdown_at = self._shutdown_at
            initialization_count = self._initialization_count
            shutdown_count = self._shutdown_count
            failed_engine_id = self._failed_engine_id
            last_error = self._last_error
            config_engine_ids = sorted(self._config.get("engines", {}).keys()) if isinstance(
                self._config.get("engines", {}), dict
            ) else []

        engine_diagnostics: list[JanusEngineDiagnostics] = []

        for engine_id in self._ENGINE_INIT_ORDER:
            engine = self._engines[engine_id]
            implementation_class = type(engine).__name__
            implementation_module = type(engine).__module__

            try:
                initialized = bool(engine.is_initialized)
                name = engine.engine_name
                version = engine.engine_version
                details: dict[str, Any] = {}
            except JanusError as exc:
                initialized = False
                name = engine_id
                version = "unknown"
                details = {"lifecycle_error": f"{type(exc).__name__}: {exc}"}
            except Exception as exc:  # noqa: BLE001
                initialized = False
                name = engine_id
                version = "unknown"
                details = {"lifecycle_error": f"{type(exc).__name__}: {exc}"}

            engine_diagnostics.append(
                JanusEngineDiagnostics(
                    engine_id=engine_id,
                    engine_name=name,
                    engine_version=version,
                    is_initialized=initialized,
                    implementation_class=implementation_class,
                    implementation_module=implementation_module,
                    details=details,
                )
            )

        subsystem_details: dict[str, Any] = {
            "created_at": created_at.isoformat(),
            "initialized_at": initialized_at.isoformat() if initialized_at else None,
            "shutdown_at": shutdown_at.isoformat() if shutdown_at else None,
            "initialization_count": initialization_count,
            "shutdown_count": shutdown_count,
            "failed_engine_id": failed_engine_id,
            "last_error": last_error,
            "configured_engine_overrides": config_engine_ids,
            "constitutional_role": "Future Modeling & Scenario Intelligence Core",
            "owns": (
                "future_modeling",
                "scenario_analysis",
                "forecasting",
                "branch_analysis",
                "counterfactual_analysis",
                "uncertainty_modeling",
                "future_risk_analysis",
                "future_opportunity_analysis",
                "outcome_simulation",
                "scenario_evaluation",
                "timeline_projection",
                "strategic_forecasting",
                "probability_estimation",
                "scenario_integrity_validation",
            ),
            "never_owns": (
                "identity",
                "knowledge_storage",
                "learning",
                "decisions",
                "plans",
            ),
        }

        return JanusSubsystemDiagnostics(
            subsystem=self.SUBSYSTEM_NAME,
            subsystem_version=self.SUBSYSTEM_VERSION,
            status=status,
            engines=tuple(engine_diagnostics),
            engine_init_order=self._ENGINE_INIT_ORDER,
            subsystem_details=subsystem_details,
            generated_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Statistics aggregation
    # ------------------------------------------------------------------

    def get_statistics(self) -> JanusSubsystemStatistics:
        """
        Aggregate runtime statistics across all JANUS engines.

        Per-engine statistics are gathered defensively: an engine that is
        not initialized, or that raises a JanusError while reporting
        statistics, contributes a minimal {"initialized": False} (or
        {"initialized": True, "error": ...}) entry rather than failing the
        entire aggregation.

        Returns:
            JanusSubsystemStatistics with subsystem-level lifecycle
            counters, uptime, and a per-engine statistics breakdown.
        """
        with self._lock:
            status = self._status
            created_at = self._created_at
            initialized_at = self._initialized_at
            shutdown_at = self._shutdown_at
            initialization_count = self._initialization_count
            shutdown_count = self._shutdown_count

        initialized_engine_count = sum(
            1 for engine in self._engines.values() if self._safe_is_initialized(engine)
        )

        if initialized_at is not None and status == JanusSubsystemStatus.OPERATIONAL:
            uptime_seconds = (datetime.now(timezone.utc) - initialized_at).total_seconds()
        elif initialized_at is not None and shutdown_at is not None:
            uptime_seconds = (shutdown_at - initialized_at).total_seconds()
        else:
            uptime_seconds = 0.0

        engine_statistics: dict[str, dict[str, Any]] = {
            "probability_engine": self._generic_engine_statistics(self._probability_engine),
            "uncertainty_engine": self._generic_engine_statistics(self._uncertainty_engine),
            "scenario_integrity_engine": self._generic_engine_statistics(
                self._scenario_integrity_engine
            ),
            "future_memory_interface": self._future_memory_statistics(),
            "scenario_engine": self._scenario_engine_statistics(),
            "branch_analysis_engine": self._generic_engine_statistics(
                self._branch_analysis_engine
            ),
            "counterfactual_engine": self._generic_engine_statistics(
                self._counterfactual_engine
            ),
            "future_modeling_engine": self._future_modeling_statistics(),
            "future_risk_engine": self._generic_engine_statistics(self._future_risk_engine),
            "future_opportunity_engine": self._generic_engine_statistics(
                self._future_opportunity_engine
            ),
            "outcome_simulation_engine": self._outcome_simulation_statistics(),
            "timeline_projection_engine": self._timeline_projection_statistics(),
            "strategic_forecast_engine": self._generic_engine_statistics(
                self._strategic_forecast_engine
            ),
            "forecasting_engine": self._forecasting_statistics(),
            "scenario_evaluation_engine": self._generic_engine_statistics(
                self._scenario_evaluation_engine
            ),
            "future_orchestrator": self._future_orchestrator_statistics(),
        }

        return JanusSubsystemStatistics(
            subsystem=self.SUBSYSTEM_NAME,
            subsystem_version=self.SUBSYSTEM_VERSION,
            status=status,
            engine_count=len(self._engines),
            initialized_engine_count=initialized_engine_count,
            initialization_count=initialization_count,
            shutdown_count=shutdown_count,
            uptime_seconds=uptime_seconds,
            created_at=created_at,
            initialized_at=initialized_at,
            shutdown_at=shutdown_at,
            engine_statistics=engine_statistics,
            generated_at=datetime.now(timezone.utc),
        )

    # -- statistics helpers -------------------------------------------

    def _generic_engine_statistics(self, engine: JanusEngineLifecycle) -> dict[str, Any]:
        """
        Minimal statistics common to every engine: identity and lifecycle
        state. Used for engines whose interfaces do not expose
        zero-argument, enum-driven listing methods suitable for safe
        aggregation at the subsystem level.
        """
        initialized = self._safe_is_initialized(engine)
        stats: dict[str, Any] = {"initialized": initialized}

        try:
            stats["engine_name"] = engine.engine_name
            stats["engine_version"] = engine.engine_version
        except JanusError as exc:
            stats["error"] = f"{type(exc).__name__}: {exc}"

        return stats

    def _scenario_engine_statistics(self) -> dict[str, Any]:
        engine = self._scenario_engine
        stats = self._generic_engine_statistics(engine)
        if not stats["initialized"]:
            return stats

        by_status: dict[str, int] = {}
        for scenario_status in ScenarioStatus:
            count = self._safe_count(engine.list_scenarios_by_status, scenario_status)
            if count is not None:
                by_status[scenario_status.name] = count

        by_type: dict[str, int] = {}
        for scenario_type in ScenarioType:
            count = self._safe_count(engine.list_scenarios_by_type, scenario_type)
            if count is not None:
                by_type[scenario_type.name] = count

        stats["scenarios_by_status"] = by_status
        stats["scenarios_by_type"] = by_type
        if by_status:
            stats["total_scenarios"] = sum(by_status.values())

        return stats

    def _forecasting_statistics(self) -> dict[str, Any]:
        engine = self._forecasting_engine
        stats = self._generic_engine_statistics(engine)
        if not stats["initialized"]:
            return stats

        by_type: dict[str, int] = {}
        for forecast_type in ForecastType:
            count = self._safe_count(engine.list_forecasts_by_type, forecast_type)
            if count is not None:
                by_type[forecast_type.name] = count

        stats["forecasts_by_type"] = by_type
        if by_type:
            stats["total_active_forecasts"] = sum(by_type.values())

        return stats

    def _future_modeling_statistics(self) -> dict[str, Any]:
        engine = self._future_modeling_engine
        stats = self._generic_engine_statistics(engine)
        if not stats["initialized"]:
            return stats

        count = self._safe_count(engine.list_future_models)
        if count is not None:
            stats["total_future_models"] = count

        return stats

    def _outcome_simulation_statistics(self) -> dict[str, Any]:
        engine = self._outcome_simulation_engine
        stats = self._generic_engine_statistics(engine)
        if not stats["initialized"]:
            return stats

        by_status: dict[str, int] = {}
        for simulation_status in SimulationStatus:
            count = self._safe_count(engine.list_simulations_by_status, simulation_status)
            if count is not None:
                by_status[simulation_status.name] = count

        stats["simulations_by_status"] = by_status
        if by_status:
            stats["total_simulations"] = sum(by_status.values())

        return stats

    def _timeline_projection_statistics(self) -> dict[str, Any]:
        engine = self._timeline_projection_engine
        stats = self._generic_engine_statistics(engine)
        if not stats["initialized"]:
            return stats

        by_status: dict[str, int] = {}
        for projection_status in ProjectionStatus:
            count = self._safe_count(engine.list_projections_by_status, projection_status)
            if count is not None:
                by_status[projection_status.name] = count

        stats["projections_by_status"] = by_status
        if by_status:
            stats["total_projections"] = sum(by_status.values())

        return stats

    def _future_orchestrator_statistics(self) -> dict[str, Any]:
        engine = self._future_orchestrator
        stats = self._generic_engine_statistics(engine)
        if not stats["initialized"]:
            return stats

        by_status: dict[str, int] = {}
        for assessment_status in FutureAssessmentStatus:
            count = self._safe_count(engine.list_assessments_by_status, assessment_status)
            if count is not None:
                by_status[assessment_status.name] = count

        stats["assessments_by_status"] = by_status
        if by_status:
            stats["total_assessments"] = sum(by_status.values())

        return stats

    def _future_memory_statistics(self) -> dict[str, Any]:
        engine = self._future_memory_interface
        stats = self._generic_engine_statistics(engine)
        if not stats["initialized"]:
            return stats

        try:
            sources = engine.list_available_sources()
        except JanusError as exc:
            stats["error"] = f"{type(exc).__name__}: {exc}"
            return stats

        stats["available_sources"] = tuple(sources)
        stats["available_source_count"] = len(sources)
        return stats

    @staticmethod
    def _safe_count(list_method: Any, *args: Any) -> Optional[int]:
        """
        Invoke a no-arg or single-enum-arg listing method and return the
        length of the resulting sequence, or None if the call could not be
        completed (engine not initialized, shut down, or raised a
        JanusError).
        """
        try:
            result = list_method(*args)
        except JanusNotInitializedError:
            return None
        except JanusShutdownError:
            return None
        except JanusError as exc:
            logger.debug(
                "Statistics call to %r raised %s: %s",
                list_method,
                type(exc).__name__,
                exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Statistics call to %r raised unexpected %s: %s",
                list_method,
                type(exc).__name__,
                exc,
            )
            return None

        try:
            return len(result)
        except TypeError:
            return None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            status = self._status
        return (
            f"JanusSubsystem(version={self.SUBSYSTEM_VERSION!r}, "
            f"status={status.value!r}, engines={len(self._engines)})"
        )
