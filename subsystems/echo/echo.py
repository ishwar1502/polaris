# subsystems/echo/echo.py
"""
ECHO Episodic Memory Core — Root Subsystem Orchestrator.

:class:`EchoSubsystem` is the single entry point for the ECHO Episodic Memory
Core.  It owns the lifecycle of every ECHO engine, wires their dependencies,
enforces initialisation ordering, and exposes the health/diagnostics surface
consumed by the POLARIS system supervisor.

Lifecycle contract
------------------
1. Construct the subsystem (no side-effects).
2. Call :meth:`initialize` exactly once.  All engines are built and started
   in dependency order.  The call is idempotent (a second call is a no-op).
3. Use engine accessors to obtain references to running engines.
4. Call :meth:`shutdown` to stop all engines in reverse-dependency order.
   The call is idempotent.

Thread safety
-------------
:meth:`initialize` and :meth:`shutdown` acquire the subsystem-level
:class:`threading.RLock` so they are safe to call from multiple threads.
All engine accessor properties are read-only after initialisation completes
and therefore require no additional locking.

Engine dependency order (initialisation)
-----------------------------------------
Tier 0 (no inbound deps):
    SignificanceEngine

Tier 1 (depends on Tier 0):
    ExperienceEngine

Tier 2 (depend on Tier 0–1):
    ExperienceRetrievalEngine
    MemoryConsolidationEngine

Tier 3 (depend on Tier 0–2):
    MemoryIntegrityEngine
    EpisodicIndexEngine
    ReflectionEngine

Tier 4 (depend on Tier 0–3):
    ContextReconstructionEngine
    PatternExtractionEngine

Tier 5 (depends on Tier 0–4):
    PersonalHistoryEngine

Engines without concrete v1 implementations (EventEngine, ConversationEngine,
SessionEngine, AchievementEngine, FailureAnalysisEngine) are held as ``None``
in this release.  Their accessor properties return ``None`` and health/
diagnostics reports document them as ``"not_implemented"``.  This preserves
the full public API contract without requiring stub classes.

ECHO Boundary Law
-----------------
ECHO owns experiences, events, conversations, sessions, achievements,
failures, observations, activity history, and personal history.  ECHO does
NOT own knowledge, identity, goals, schedules, relationships, or decisions.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Final

from subsystems.echo.consolidation import MemoryConsolidationEngine
from subsystems.echo.context_reconstruction import ContextReconstructionEngine
from subsystems.echo.episodic_index import EpisodicIndexEngine
from subsystems.echo.experience import ExperienceEngine
from subsystems.echo.integrity import MemoryIntegrityEngine
from subsystems.echo.patterns import PatternExtractionEngine
from subsystems.echo.personal_history import PersonalHistoryEngine
from subsystems.echo.reflection import ReflectionEngine
from subsystems.echo.retrieval import ExperienceRetrievalEngine
from subsystems.echo.significance import SignificanceEngine

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subsystem metadata
# ---------------------------------------------------------------------------

SUBSYSTEM_NAME: Final[str] = "ECHO"
SUBSYSTEM_VERSION: Final[str] = "1.0.0"
SUBSYSTEM_DESCRIPTION: Final[str] = "ECHO Episodic Memory Core — episodic experience storage and recall"

# Engines that are defined in the interface layer but not yet concretely
# implemented.  Listed here so health/diagnostics reports are self-documenting.
_UNIMPLEMENTED_ENGINES: Final[tuple[str, ...]] = (
    "EventEngine",
    "ConversationEngine",
    "SessionEngine",
    "AchievementEngine",
    "FailureAnalysisEngine",
)


# ---------------------------------------------------------------------------
# Subsystem status enum
# ---------------------------------------------------------------------------


class EchoSubsystemStatus(Enum):
    """Lifecycle states of :class:`EchoSubsystem`."""

    CREATED = auto()
    """Constructed but not yet initialised."""

    INITIALIZING = auto()
    """Currently running :meth:`EchoSubsystem.initialize`."""

    RUNNING = auto()
    """All engines are initialised and operational."""

    SHUTTING_DOWN = auto()
    """Currently running :meth:`EchoSubsystem.shutdown`."""

    STOPPED = auto()
    """Shutdown complete.  The subsystem may not be restarted."""

    ERROR = auto()
    """Initialisation or shutdown encountered an unrecoverable error."""


# ---------------------------------------------------------------------------
# EchoSubsystem
# ---------------------------------------------------------------------------


class EchoSubsystem:
    """Root orchestration layer of the ECHO Episodic Memory Core.

    Parameters
    ----------
    min_significance_threshold:
        Minimum significance score [0.0, 1.0] forwarded to
        :class:`~subsystems.echo.experience.ExperienceEngine`.  Controls
        the threshold below which experiences are rejected unless the caller
        passes ``force=True``.  Defaults to ``0.15``.

    Usage
    -----
    ::

        echo = EchoSubsystem()
        echo.initialize()

        exp_engine = echo.experience_engine
        exp = exp_engine.create_experience(
            title="Completed POLARIS milestone",
            experience_type=ExperienceType.ACHIEVEMENT,
            importance=ExperienceImportance.HIGH,
        )

        echo.shutdown()
    """

    def __init__(
        self,
        *,
        min_significance_threshold: float = 0.15,
    ) -> None:
        if not (0.0 <= min_significance_threshold <= 1.0):
            raise ValueError(
                "min_significance_threshold must be in [0.0, 1.0]; "
                f"got {min_significance_threshold!r}."
            )

        self._min_significance_threshold = min_significance_threshold

        # Lifecycle state
        self._status: EchoSubsystemStatus = EchoSubsystemStatus.CREATED
        self._lock: threading.RLock = threading.RLock()
        self._initialized_at: datetime | None = None
        self._shutdown_at: datetime | None = None
        self._init_error: Exception | None = None

        # Engine references — typed as the concrete class for IDE support
        # while exposed through interface-typed accessors.
        self._significance_engine: SignificanceEngine | None = None
        self._experience_engine: ExperienceEngine | None = None
        self._retrieval_engine: ExperienceRetrievalEngine | None = None
        self._consolidation_engine: MemoryConsolidationEngine | None = None
        self._integrity_engine: MemoryIntegrityEngine | None = None
        self._episodic_index_engine: EpisodicIndexEngine | None = None
        self._reflection_engine: ReflectionEngine | None = None
        self._context_reconstruction_engine: ContextReconstructionEngine | None = None
        self._pattern_extraction_engine: PatternExtractionEngine | None = None
        self._personal_history_engine: PersonalHistoryEngine | None = None

        # Engines pending v1 implementation — always None in this release.
        # Type annotations use Any to allow future assignment without touching
        # the accessor surface.
        self._event_engine: Any = None
        self._conversation_engine: Any = None
        self._session_engine: Any = None
        self._achievement_engine: Any = None
        self._failure_analysis_engine: Any = None

        _logger.debug(
            "%s subsystem constructed (min_significance_threshold=%.3f).",
            SUBSYSTEM_NAME,
            min_significance_threshold,
        )

    # ------------------------------------------------------------------
    # Lifecycle — initialize
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialise the ECHO subsystem and all managed engines.

        Engines are constructed and started in strict dependency order so
        that each engine receives only fully-initialised dependencies.

        This method is idempotent: calling it when the subsystem is already
        ``RUNNING`` logs a warning and returns immediately.

        Raises
        ------
        RuntimeError
            If called on a subsystem that has previously been shut down or
            that encountered a fatal error during an earlier initialisation
            attempt.
        Exception
            Any exception raised by an engine's ``initialize()`` method is
            propagated after the subsystem status is set to ``ERROR``.
        """
        with self._lock:
            if self._status is EchoSubsystemStatus.RUNNING:
                _logger.warning(
                    "%s.initialize() called while already RUNNING — no-op.",
                    SUBSYSTEM_NAME,
                )
                return

            if self._status in (
                EchoSubsystemStatus.STOPPED,
                EchoSubsystemStatus.ERROR,
            ):
                raise RuntimeError(
                    f"{SUBSYSTEM_NAME} subsystem cannot be re-initialised from "
                    f"status {self._status.name}.  Create a new instance."
                )

            if self._status is EchoSubsystemStatus.INITIALIZING:
                raise RuntimeError(
                    f"{SUBSYSTEM_NAME} subsystem is already in the middle of "
                    "initialisation (re-entrant call detected)."
                )

            self._status = EchoSubsystemStatus.INITIALIZING
            _logger.info("Initialising %s subsystem v%s …", SUBSYSTEM_NAME, SUBSYSTEM_VERSION)

            try:
                self._build_and_initialize_engines()
            except Exception as exc:
                self._status = EchoSubsystemStatus.ERROR
                self._init_error = exc
                _logger.exception(
                    "%s subsystem initialisation failed: %s", SUBSYSTEM_NAME, exc
                )
                raise

            self._status = EchoSubsystemStatus.RUNNING
            self._initialized_at = datetime.now(timezone.utc)
            _logger.info(
                "%s subsystem initialised successfully at %s.",
                SUBSYSTEM_NAME,
                self._initialized_at.isoformat(),
            )

    def _build_and_initialize_engines(self) -> None:
        """Construct and initialise every managed engine in dependency order.

        Called exclusively from within the subsystem lock during
        :meth:`initialize`.  Any exception here causes the subsystem to
        transition to ``ERROR`` status.
        """
        # ------------------------------------------------------------------
        # Tier 0: SignificanceEngine (no dependencies)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising SignificanceEngine …")
        self._significance_engine = SignificanceEngine()
        self._significance_engine.initialize()

        # ------------------------------------------------------------------
        # Tier 1: ExperienceEngine (depends on SignificanceEngine)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising ExperienceEngine …")
        self._experience_engine = ExperienceEngine(
            significance_engine=self._significance_engine,
            min_significance_threshold=self._min_significance_threshold,
        )
        self._experience_engine.initialize()

        # ------------------------------------------------------------------
        # Tier 2a: ExperienceRetrievalEngine (depends on ExperienceEngine)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising ExperienceRetrievalEngine …")
        self._retrieval_engine = ExperienceRetrievalEngine(
            experience_engine=self._experience_engine,
        )
        self._retrieval_engine.initialize()

        # ------------------------------------------------------------------
        # Tier 2b: MemoryConsolidationEngine
        #          (depends on ExperienceEngine + SignificanceEngine)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising MemoryConsolidationEngine …")
        self._consolidation_engine = MemoryConsolidationEngine(
            experience_engine=self._experience_engine,
            significance_engine=self._significance_engine,
        )
        self._consolidation_engine.initialize()

        # ------------------------------------------------------------------
        # Tier 3a: MemoryIntegrityEngine
        #          (depends on ExperienceEngine + RetrievalEngine)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising MemoryIntegrityEngine …")
        self._integrity_engine = MemoryIntegrityEngine(
            experience_engine=self._experience_engine,
            retrieval_engine=self._retrieval_engine,
        )
        self._integrity_engine.initialize()

        # ------------------------------------------------------------------
        # Tier 3b: EpisodicIndexEngine
        #          (depends on ExperienceEngine + RetrievalEngine)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising EpisodicIndexEngine …")
        self._episodic_index_engine = EpisodicIndexEngine(
            experience_engine=self._experience_engine,
            retrieval_engine=self._retrieval_engine,
        )
        self._episodic_index_engine.initialize()

        # ------------------------------------------------------------------
        # Tier 3c: ReflectionEngine
        #          (depends on ExperienceEngine + SignificanceEngine;
        #           AchievementEngine and FailureEngine are not yet
        #           implemented and are wired as None)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising ReflectionEngine …")
        self._reflection_engine = ReflectionEngine(
            experience_engine=self._experience_engine,
            significance_engine=self._significance_engine,
            achievement_engine=None,
            failure_engine=None,
        )
        self._reflection_engine.initialize()

        # ------------------------------------------------------------------
        # Tier 4a: ContextReconstructionEngine
        #          (depends on ExperienceEngine and optionally on
        #           RetrievalEngine, ReflectionEngine;
        #           SessionEngine and ConversationEngine are not yet
        #           implemented)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising ContextReconstructionEngine …")
        self._context_reconstruction_engine = ContextReconstructionEngine(
            experience_engine=self._experience_engine,
            retrieval_engine=self._retrieval_engine,
            session_engine=None,
            conversation_engine=None,
            achievement_engine=None,
            failure_engine=None,
            reflection_engine=self._reflection_engine,
        )
        self._context_reconstruction_engine.initialize()

        # ------------------------------------------------------------------
        # Tier 4b: PatternExtractionEngine
        #          (depends on ExperienceEngine + RetrievalEngine;
        #           optionally ReflectionEngine, ContextEngine,
        #           EpisodicIndex, IntegrityEngine)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising PatternExtractionEngine …")
        self._pattern_extraction_engine = PatternExtractionEngine(
            experience_engine=self._experience_engine,
            retrieval_engine=self._retrieval_engine,
            reflection_engine=self._reflection_engine,
            context_engine=self._context_reconstruction_engine,
            episodic_index=self._episodic_index_engine,
            integrity_engine=self._integrity_engine,
        )
        self._pattern_extraction_engine.initialize()

        # ------------------------------------------------------------------
        # Tier 5: PersonalHistoryEngine
        #         (depends on ExperienceEngine + all optional engines)
        # ------------------------------------------------------------------
        _logger.debug("ECHO: initialising PersonalHistoryEngine …")
        self._personal_history_engine = PersonalHistoryEngine(
            experience_engine=self._experience_engine,
            retrieval_engine=self._retrieval_engine,
            reflection_engine=self._reflection_engine,
            pattern_engine=self._pattern_extraction_engine,
            context_engine=self._context_reconstruction_engine,
            episodic_index=self._episodic_index_engine,
            integrity_engine=self._integrity_engine,
        )
        self._personal_history_engine.initialize()

        _logger.debug("ECHO: all engine tiers initialised.")

    # ------------------------------------------------------------------
    # Lifecycle — shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shut down the ECHO subsystem and all managed engines.

        Engines are stopped in reverse-dependency order (highest tier first)
        so that an engine never outlives a dependency it relies upon.

        This method is idempotent: calling it on a subsystem that is already
        ``STOPPED`` logs a warning and returns immediately.

        Any exception raised by an individual engine's ``shutdown()`` method
        is caught and logged; shutdown of the remaining engines continues so
        that resource leaks are minimised.  If any engine raised, the
        subsystem transitions to ``ERROR`` rather than ``STOPPED`` after the
        full teardown pass completes.

        Raises
        ------
        RuntimeError
            If the subsystem is in ``INITIALIZING`` state when ``shutdown``
            is called (re-entrant / concurrent misuse).
        """
        with self._lock:
            if self._status is EchoSubsystemStatus.STOPPED:
                _logger.warning(
                    "%s.shutdown() called while already STOPPED — no-op.",
                    SUBSYSTEM_NAME,
                )
                return

            if self._status is EchoSubsystemStatus.CREATED:
                _logger.warning(
                    "%s.shutdown() called before initialize() — marking STOPPED.",
                    SUBSYSTEM_NAME,
                )
                self._status = EchoSubsystemStatus.STOPPED
                self._shutdown_at = datetime.now(timezone.utc)
                return

            if self._status is EchoSubsystemStatus.INITIALIZING:
                raise RuntimeError(
                    f"{SUBSYSTEM_NAME} subsystem cannot be shut down while "
                    "initialisation is in progress."
                )

            if self._status is EchoSubsystemStatus.SHUTTING_DOWN:
                _logger.warning(
                    "%s.shutdown() called while already SHUTTING_DOWN — no-op.",
                    SUBSYSTEM_NAME,
                )
                return

            self._status = EchoSubsystemStatus.SHUTTING_DOWN
            _logger.info("Shutting down %s subsystem …", SUBSYSTEM_NAME)

            shutdown_errors: list[tuple[str, Exception]] = []

            # Shutdown in reverse-dependency order (Tier 5 → Tier 0).
            _shutdown_sequence: list[tuple[str, Any]] = [
                ("PersonalHistoryEngine", self._personal_history_engine),
                ("PatternExtractionEngine", self._pattern_extraction_engine),
                ("ContextReconstructionEngine", self._context_reconstruction_engine),
                ("ReflectionEngine", self._reflection_engine),
                ("EpisodicIndexEngine", self._episodic_index_engine),
                ("MemoryIntegrityEngine", self._integrity_engine),
                ("MemoryConsolidationEngine", self._consolidation_engine),
                ("ExperienceRetrievalEngine", self._retrieval_engine),
                ("ExperienceEngine", self._experience_engine),
                ("SignificanceEngine", self._significance_engine),
            ]

            for engine_name, engine in _shutdown_sequence:
                if engine is None:
                    continue
                try:
                    _logger.debug("ECHO: shutting down %s …", engine_name)
                    engine.shutdown()
                except Exception as exc:  # noqa: BLE001
                    _logger.exception(
                        "ECHO: error shutting down %s: %s", engine_name, exc
                    )
                    shutdown_errors.append((engine_name, exc))

            self._shutdown_at = datetime.now(timezone.utc)

            if shutdown_errors:
                self._status = EchoSubsystemStatus.ERROR
                failed = ", ".join(name for name, _ in shutdown_errors)
                _logger.error(
                    "%s subsystem shutdown completed with errors in: %s",
                    SUBSYSTEM_NAME,
                    failed,
                )
            else:
                self._status = EchoSubsystemStatus.STOPPED
                _logger.info(
                    "%s subsystem shut down cleanly at %s.",
                    SUBSYSTEM_NAME,
                    self._shutdown_at.isoformat(),
                )

    # ------------------------------------------------------------------
    # Internal guard
    # ------------------------------------------------------------------

    def _assert_running(self, caller: str) -> None:
        """Raise :class:`RuntimeError` if the subsystem is not ``RUNNING``."""
        if self._status is not EchoSubsystemStatus.RUNNING:
            raise RuntimeError(
                f"{SUBSYSTEM_NAME}.{caller}() requires the subsystem to be RUNNING; "
                f"current status is {self._status.name}."
            )

    # ------------------------------------------------------------------
    # Engine accessors — implemented engines
    # ------------------------------------------------------------------

    @property
    def significance_engine(self) -> SignificanceEngine:
        """Return the running :class:`~subsystems.echo.significance.SignificanceEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("significance_engine")
        assert self._significance_engine is not None  # guaranteed post-init
        return self._significance_engine

    @property
    def experience_engine(self) -> ExperienceEngine:
        """Return the running :class:`~subsystems.echo.experience.ExperienceEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("experience_engine")
        assert self._experience_engine is not None
        return self._experience_engine

    @property
    def retrieval_engine(self) -> ExperienceRetrievalEngine:
        """Return the running :class:`~subsystems.echo.retrieval.ExperienceRetrievalEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("retrieval_engine")
        assert self._retrieval_engine is not None
        return self._retrieval_engine

    @property
    def consolidation_engine(self) -> MemoryConsolidationEngine:
        """Return the running :class:`~subsystems.echo.consolidation.MemoryConsolidationEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("consolidation_engine")
        assert self._consolidation_engine is not None
        return self._consolidation_engine

    @property
    def integrity_engine(self) -> MemoryIntegrityEngine:
        """Return the running :class:`~subsystems.echo.integrity.MemoryIntegrityEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("integrity_engine")
        assert self._integrity_engine is not None
        return self._integrity_engine

    @property
    def episodic_index_engine(self) -> EpisodicIndexEngine:
        """Return the running :class:`~subsystems.echo.episodic_index.EpisodicIndexEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("episodic_index_engine")
        assert self._episodic_index_engine is not None
        return self._episodic_index_engine

    @property
    def reflection_engine(self) -> ReflectionEngine:
        """Return the running :class:`~subsystems.echo.reflection.ReflectionEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("reflection_engine")
        assert self._reflection_engine is not None
        return self._reflection_engine

    @property
    def context_reconstruction_engine(self) -> ContextReconstructionEngine:
        """Return the running :class:`~subsystems.echo.context_reconstruction.ContextReconstructionEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("context_reconstruction_engine")
        assert self._context_reconstruction_engine is not None
        return self._context_reconstruction_engine

    @property
    def pattern_extraction_engine(self) -> PatternExtractionEngine:
        """Return the running :class:`~subsystems.echo.patterns.PatternExtractionEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("pattern_extraction_engine")
        assert self._pattern_extraction_engine is not None
        return self._pattern_extraction_engine

    @property
    def personal_history_engine(self) -> PersonalHistoryEngine:
        """Return the running :class:`~subsystems.echo.personal_history.PersonalHistoryEngine`.

        Raises
        ------
        RuntimeError
            If the subsystem is not in ``RUNNING`` status.
        """
        self._assert_running("personal_history_engine")
        assert self._personal_history_engine is not None
        return self._personal_history_engine

    # ------------------------------------------------------------------
    # Engine accessors — not-yet-implemented engines
    # ------------------------------------------------------------------

    @property
    def event_engine(self) -> None:
        """Return the EventEngine instance.

        Always returns ``None`` in v1.  EventEngine is not yet implemented.
        """
        return None

    @property
    def conversation_engine(self) -> None:
        """Return the ConversationEngine instance.

        Always returns ``None`` in v1.  ConversationEngine is not yet
        implemented.
        """
        return None

    @property
    def session_engine(self) -> None:
        """Return the SessionEngine instance.

        Always returns ``None`` in v1.  SessionEngine is not yet implemented.
        """
        return None

    @property
    def achievement_engine(self) -> None:
        """Return the AchievementEngine instance.

        Always returns ``None`` in v1.  AchievementEngine is not yet
        implemented.
        """
        return None

    @property
    def failure_analysis_engine(self) -> None:
        """Return the FailureAnalysisEngine instance.

        Always returns ``None`` in v1.  FailureAnalysisEngine is not yet
        implemented.
        """
        return None

    # ------------------------------------------------------------------
    # Status methods
    # ------------------------------------------------------------------

    @property
    def status(self) -> EchoSubsystemStatus:
        """Return the current lifecycle status of the subsystem."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Return ``True`` iff the subsystem is in ``RUNNING`` status."""
        return self._status is EchoSubsystemStatus.RUNNING

    @property
    def initialized_at(self) -> datetime | None:
        """Return the UTC timestamp when the subsystem completed initialisation,
        or ``None`` if it has not yet been initialised."""
        return self._initialized_at

    @property
    def shutdown_at(self) -> datetime | None:
        """Return the UTC timestamp when shutdown completed, or ``None``."""
        return self._shutdown_at

    def uptime_seconds(self) -> float | None:
        """Return the number of seconds the subsystem has been running.

        Returns ``None`` if the subsystem has not been initialised or has
        already been shut down.
        """
        if self._initialized_at is None or self._status is not EchoSubsystemStatus.RUNNING:
            return None
        return (datetime.now(timezone.utc) - self._initialized_at).total_seconds()

    # ------------------------------------------------------------------
    # Subsystem metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the subsystem identifier string ``"ECHO"``."""
        return SUBSYSTEM_NAME

    @property
    def version(self) -> str:
        """Return the subsystem version string."""
        return SUBSYSTEM_VERSION

    @property
    def description(self) -> str:
        """Return a human-readable description of the subsystem."""
        return SUBSYSTEM_DESCRIPTION

    def metadata(self) -> dict[str, Any]:
        """Return a dict of static subsystem metadata.

        Returns
        -------
        dict[str, Any]
            Keys: ``name``, ``version``, ``description``,
            ``implemented_engines``, ``unimplemented_engines``.
        """
        return {
            "name": SUBSYSTEM_NAME,
            "version": SUBSYSTEM_VERSION,
            "description": SUBSYSTEM_DESCRIPTION,
            "implemented_engines": [
                "SignificanceEngine",
                "ExperienceEngine",
                "ExperienceRetrievalEngine",
                "MemoryConsolidationEngine",
                "MemoryIntegrityEngine",
                "EpisodicIndexEngine",
                "ReflectionEngine",
                "ContextReconstructionEngine",
                "PatternExtractionEngine",
                "PersonalHistoryEngine",
            ],
            "unimplemented_engines": list(_UNIMPLEMENTED_ENGINES),
        }

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def health_report(self) -> dict[str, Any]:
        """Return a structured health report for the subsystem.

        The report is safe to call at any lifecycle state; engines that are
        not yet initialised are reported as ``"not_running"``.

        Returns
        -------
        dict[str, Any]
            Top-level keys:

            ``subsystem``
                Name and version identifiers.
            ``status``
                Current :class:`EchoSubsystemStatus` name.
            ``healthy``
                ``True`` iff the subsystem is ``RUNNING`` and all
                implemented engines report ``running: True``.
            ``uptime_seconds``
                Seconds since initialisation completed, or ``None``.
            ``initialized_at``
                ISO-8601 timestamp or ``None``.
            ``engines``
                Per-engine health entries.
        """
        engine_health: dict[str, dict[str, Any]] = {}

        # Collect health snapshots from every implemented engine.
        _implemented: list[tuple[str, Any]] = [
            ("SignificanceEngine", self._significance_engine),
            ("ExperienceEngine", self._experience_engine),
            ("ExperienceRetrievalEngine", self._retrieval_engine),
            ("MemoryConsolidationEngine", self._consolidation_engine),
            ("MemoryIntegrityEngine", self._integrity_engine),
            ("EpisodicIndexEngine", self._episodic_index_engine),
            ("ReflectionEngine", self._reflection_engine),
            ("ContextReconstructionEngine", self._context_reconstruction_engine),
            ("PatternExtractionEngine", self._pattern_extraction_engine),
            ("PersonalHistoryEngine", self._personal_history_engine),
        ]

        all_engines_healthy = True
        for engine_name, engine in _implemented:
            if engine is None:
                engine_health[engine_name] = {"status": "not_running", "running": False}
                all_engines_healthy = False
                continue
            try:
                snap = engine.snapshot()
                running = bool(snap.get("running", False))
                engine_health[engine_name] = {
                    "status": "running" if running else "stopped",
                    "running": running,
                }
                if not running:
                    all_engines_healthy = False
            except AttributeError:
                # Engine does not implement snapshot(); treat as healthy when present
                engine_health[engine_name] = {"status": "running", "running": True}
            except Exception as exc:  # noqa: BLE001
                engine_health[engine_name] = {
                    "status": "error",
                    "running": False,
                    "error": str(exc),
                }
                all_engines_healthy = False

        # Not-yet-implemented engines — excluded from healthy aggregation.
        for engine_name in _UNIMPLEMENTED_ENGINES:
            engine_health[engine_name] = {
                "status": "not_implemented",
                "running": False,
            }

        subsystem_running = self._status is EchoSubsystemStatus.RUNNING
        healthy = subsystem_running and all_engines_healthy

        return {
            "subsystem": {
                "name": SUBSYSTEM_NAME,
                "version": SUBSYSTEM_VERSION,
            },
            "status": self._status.name,
            "healthy": healthy,
            "uptime_seconds": self.uptime_seconds(),
            "initialized_at": (
                self._initialized_at.isoformat() if self._initialized_at else None
            ),
            "shutdown_at": (
                self._shutdown_at.isoformat() if self._shutdown_at else None
            ),
            "engines": engine_health,
        }

    # ------------------------------------------------------------------
    # Diagnostics reporting
    # ------------------------------------------------------------------

    def diagnostics_report(self) -> dict[str, Any]:
        """Return a detailed diagnostics report for the subsystem.

        Extends :meth:`health_report` with per-engine diagnostic snapshots
        containing internal counters, configuration, and store statistics.
        Intended for system operators and integration test assertions — not
        for routine health polling.

        Returns
        -------
        dict[str, Any]
            All keys from :meth:`health_report` plus:

            ``diagnostics``
                Mapping of engine name → full ``snapshot()`` dict (or an
                error descriptor if the snapshot call fails).
            ``metadata``
                Static subsystem metadata from :meth:`metadata`.
        """
        health = self.health_report()

        diagnostics: dict[str, Any] = {}

        _implemented: list[tuple[str, Any]] = [
            ("SignificanceEngine", self._significance_engine),
            ("ExperienceEngine", self._experience_engine),
            ("ExperienceRetrievalEngine", self._retrieval_engine),
            ("MemoryConsolidationEngine", self._consolidation_engine),
            ("MemoryIntegrityEngine", self._integrity_engine),
            ("EpisodicIndexEngine", self._episodic_index_engine),
            ("ReflectionEngine", self._reflection_engine),
            ("ContextReconstructionEngine", self._context_reconstruction_engine),
            ("PatternExtractionEngine", self._pattern_extraction_engine),
            ("PersonalHistoryEngine", self._personal_history_engine),
        ]

        for engine_name, engine in _implemented:
            if engine is None:
                diagnostics[engine_name] = {"status": "not_running"}
                continue
            try:
                diagnostics[engine_name] = engine.snapshot()
            except Exception as exc:  # noqa: BLE001
                diagnostics[engine_name] = {
                    "status": "snapshot_error",
                    "error": str(exc),
                }

        for engine_name in _UNIMPLEMENTED_ENGINES:
            diagnostics[engine_name] = {"status": "not_implemented"}

        return {
            **health,
            "diagnostics": diagnostics,
            "metadata": self.metadata(),
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EchoSubsystem("
            f"status={self._status.name}, "
            f"version={SUBSYSTEM_VERSION!r}"
            f")"
        )