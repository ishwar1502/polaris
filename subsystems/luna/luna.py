"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/luna.py

Orchestration layer for the entire LUNA Semantic Knowledge Core.

LunaSubsystem owns, initialises, and shuts down all LUNA engines in
dependency-safe order.  It is the single entry point through which the
POLARIS Cognitive Substrate interacts with LUNA knowledge services.

Engine ownership:
    FactEngine                  — atomic truth claims
    ConceptEngine               — semantic concept records
    SkillEngine                 — capability / skill records
    KnowledgeDomainEngine       — knowledge domain taxonomy
    ProceduralKnowledgeEngine   — step-by-step procedural knowledge
    ResearchKnowledgeEngine     — research artefacts
    EducationalKnowledgeEngine  — educational / learning content
    KnowledgeValidationEngine   — cross-type validation
    KnowledgeRetrievalEngine    — unified record retrieval
    KnowledgeSynthesisEngine    — higher-order knowledge composition
    KnowledgeEvolutionEngine    — record versioning and change management
    SkillProgressionEngine      — skill progression modelling
    KnowledgeIndexEngine        — fast secondary-index lookups
    KnowledgeIntegrityEngine    — integrity scanning and audit
    SemanticStructureEngine     — hierarchy, relationship, and dependency graph
    LunaEventEngine             — subsystem-internal event bus

Part of the POLARIS Cognitive Substrate:
    ASTRA        → Identity
    ECHO         → Experience
    LUNA         → Knowledge        ← this module
    CHRONOS      → Time
    CONSTELLATION → Relationships
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from subsystems.luna.concepts import ConceptEngine
from subsystems.luna.domains import KnowledgeDomainEngine
from subsystems.luna.education import EducationalKnowledgeEngine
from subsystems.luna.events import LunaEventEngine
from subsystems.luna.evolution import KnowledgeEvolutionEngine
from subsystems.luna.exceptions import (
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.facts import FactEngine
from subsystems.luna.integrity import KnowledgeIntegrityEngine
from subsystems.luna.knowledge_index import KnowledgeIndexEngine
from subsystems.luna.procedures import ProceduralKnowledgeEngine
from subsystems.luna.progression import SkillProgressionEngine
from subsystems.luna.research import ResearchKnowledgeEngine
from subsystems.luna.retrieval import KnowledgeRetrievalEngine
from subsystems.luna.semantic_structure import SemanticStructureEngine
from subsystems.luna.skills import SkillEngine
from subsystems.luna.synthesis import KnowledgeSynthesisEngine
from subsystems.luna.models import KnowledgeType
from subsystems.luna.validation import KnowledgeValidationEngine

logger = logging.getLogger(__name__)

_SUBSYSTEM_VERSION: str = "5.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# STATUS MODEL
# ─────────────────────────────────────────────────────────────────────────────

class LunaSubsystemStatus(Enum):
    """Lifecycle states of the LUNA subsystem."""

    CREATED = "created"
    """Subsystem constructed; no engine has been initialised yet."""

    INITIALIZING = "initializing"
    """Startup sequence is in progress."""

    RUNNING = "running"
    """All engines are online and ready to serve requests."""

    SHUTTING_DOWN = "shutting_down"
    """Shutdown sequence is in progress."""

    STOPPED = "stopped"
    """All engines have been shut down cleanly."""

    FAILED = "failed"
    """A lifecycle transition failed; subsystem is in an undefined state."""


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# LUNA SUBSYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class LunaSubsystem:
    """
    Orchestration layer for the LUNA Semantic Knowledge Core.

    LunaSubsystem owns one instance of every LUNA engine, wires their
    inter-engine dependencies together, and exposes a stable, lifecycle-gated
    API to the rest of the POLARIS Cognitive Substrate.

    Thread safety:
        All lifecycle methods are protected by a single reentrant lock.
        Engine accessors are read-only after startup and therefore safe to call
        from any thread once the subsystem reaches RUNNING status.

    Lifecycle::

        luna = LunaSubsystem()
        luna.initialize()

        facts = luna.fact_engine.list_all()
        ...

        luna.shutdown()
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._status: LunaSubsystemStatus = LunaSubsystemStatus.CREATED
        self._started_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None
        self._failure_reason: Optional[str] = None

        # ── Tier 0: Event bus (no dependencies) ──────────────────────────────
        self._event_engine: LunaEventEngine = LunaEventEngine()

        # ── Tier 1: Leaf knowledge stores (no inter-engine deps) ─────────────
        self._fact_engine: FactEngine = FactEngine()
        self._concept_engine: ConceptEngine = ConceptEngine()
        self._skill_engine: SkillEngine = SkillEngine()
        self._domain_engine: KnowledgeDomainEngine = KnowledgeDomainEngine()
        self._procedure_engine: ProceduralKnowledgeEngine = ProceduralKnowledgeEngine()
        self._research_engine: ResearchKnowledgeEngine = ResearchKnowledgeEngine()
        self._educational_engine: EducationalKnowledgeEngine = EducationalKnowledgeEngine()

        # ── Tier 2: Validation engine (reads from stores) ─────────────────────
        self._validation_engine: KnowledgeValidationEngine = KnowledgeValidationEngine()

        # ── Tier 3: Retrieval + Index (read from stores; retrieval before index) ─
        self._retrieval_engine: KnowledgeRetrievalEngine = KnowledgeRetrievalEngine(
            fact_engine=self._fact_engine,
            concept_engine=self._concept_engine,
            skill_engine=self._skill_engine,
            domain_engine=self._domain_engine,
            procedure_engine=self._procedure_engine,
            research_engine=self._research_engine,
            educational_engine=self._educational_engine,
        )

        self._index_engine: KnowledgeIndexEngine = KnowledgeIndexEngine(
            fact_engine=self._fact_engine,
            concept_engine=self._concept_engine,
            skill_engine=self._skill_engine,
            domain_engine=self._domain_engine,
            procedure_engine=self._procedure_engine,
            research_engine=self._research_engine,
            educational_engine=self._educational_engine,
        )

        # ── Tier 4: Synthesis + Integrity (depend on stores and index) ────────
        self._synthesis_engine: KnowledgeSynthesisEngine = KnowledgeSynthesisEngine(
            fact_engine=self._fact_engine,
            concept_engine=self._concept_engine,
            skill_engine=self._skill_engine,
            domain_engine=self._domain_engine,
            procedure_engine=self._procedure_engine,
            research_engine=self._research_engine,
            educational_engine=self._educational_engine,
        )

        self._integrity_engine: KnowledgeIntegrityEngine = KnowledgeIntegrityEngine(
            fact_engine=self._fact_engine,
            concept_engine=self._concept_engine,
            skill_engine=self._skill_engine,
            domain_engine=self._domain_engine,
            procedure_engine=self._procedure_engine,
            research_engine=self._research_engine,
            educational_engine=self._educational_engine,
            index_engine=self._index_engine,
        )

        # ── Tier 5: Evolution (depends on retrieval + validation + synthesis) ─
        self._evolution_engine: KnowledgeEvolutionEngine = KnowledgeEvolutionEngine(
            retrieval_engine=self._retrieval_engine,
            validation_engine=self._validation_engine,
            synthesis_engine=self._synthesis_engine,
        )

        # ── Tier 6: Skill Progression (depends on skill + education + evolution) ─
        self._progression_engine: SkillProgressionEngine = SkillProgressionEngine(
            skill_engine=self._skill_engine,
            education_engine=self._educational_engine,
            evolution_engine=self._evolution_engine,
        )

        # ── Tier 7: Semantic Structure (depends on concept/domain/index/integrity/retrieval) ─
        self._semantic_structure_engine: SemanticStructureEngine = SemanticStructureEngine(
            concept_engine=self._concept_engine,
            domain_engine=self._domain_engine,
            index_engine=self._index_engine,
            integrity_engine=self._integrity_engine,
            retrieval_engine=self._retrieval_engine,
        )

        logger.debug(
            "LunaSubsystem constructed (version=%s, status=%s)",
            _SUBSYSTEM_VERSION,
            self._status.value,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Start the LUNA subsystem and all owned engines in dependency order.

        Idempotent: if the subsystem is already RUNNING this method returns
        immediately without raising.

        Raises:
            LunaLifecycleError: If any engine fails to initialise, or if the
                subsystem is in a state from which startup is not permitted
                (SHUTTING_DOWN, FAILED).
        """
        with self._lock:
            if self._status is LunaSubsystemStatus.RUNNING:
                return

            if self._status in (
                LunaSubsystemStatus.INITIALIZING,
                LunaSubsystemStatus.SHUTTING_DOWN,
            ):
                raise LunaLifecycleError(
                    phase="startup",
                    reason=f"Cannot initialise from status '{self._status.value}'",
                )

            if self._status is LunaSubsystemStatus.FAILED:
                raise LunaLifecycleError(
                    phase="startup",
                    reason=(
                        "Subsystem is in FAILED state and cannot be restarted. "
                        f"Original failure: {self._failure_reason}"
                    ),
                )

            self._status = LunaSubsystemStatus.INITIALIZING
            logger.info(
                "LunaSubsystem initializing (version=%s)", _SUBSYSTEM_VERSION
            )

            # Startup order mirrors the tier dependency graph defined in __init__.
            _startup_order: list[tuple[str, Any]] = [
                ("LunaEventEngine",               self._event_engine),
                ("FactEngine",                    self._fact_engine),
                ("ConceptEngine",                 self._concept_engine),
                ("SkillEngine",                   self._skill_engine),
                ("KnowledgeDomainEngine",         self._domain_engine),
                ("ProceduralKnowledgeEngine",     self._procedure_engine),
                ("ResearchKnowledgeEngine",       self._research_engine),
                ("EducationalKnowledgeEngine",    self._educational_engine),
                ("KnowledgeValidationEngine",     self._validation_engine),
                ("KnowledgeRetrievalEngine",      self._retrieval_engine),
                ("KnowledgeIndexEngine",          self._index_engine),
                ("KnowledgeSynthesisEngine",      self._synthesis_engine),
                ("KnowledgeIntegrityEngine",      self._integrity_engine),
                ("KnowledgeEvolutionEngine",      self._evolution_engine),
                ("SkillProgressionEngine",        self._progression_engine),
                ("SemanticStructureEngine",       self._semantic_structure_engine),
            ]

            initialized_engines: list[tuple[str, Any]] = []
            try:
                for engine_name, engine in _startup_order:
                    logger.debug("LunaSubsystem: starting %s", engine_name)
                    engine.initialize()
                    initialized_engines.append((engine_name, engine))

                # Wire leaf-engine store dicts into the validation engine so
                # cross-type validation, contradiction detection, stale-record
                # flagging, and confidence checks have live data to work with.
                self._validation_engine.register_store(
                    KnowledgeType.FACT, self._fact_engine._store
                )
                self._validation_engine.register_store(
                    KnowledgeType.CONCEPT, self._concept_engine._store
                )
                self._validation_engine.register_store(
                    KnowledgeType.SKILL, self._skill_engine._store
                )
                self._validation_engine.register_store(
                    KnowledgeType.DOMAIN, self._domain_engine._store
                )
                self._validation_engine.register_store(
                    KnowledgeType.PROCEDURE, self._procedure_engine._store
                )
                self._validation_engine.register_store(
                    KnowledgeType.RESEARCH, self._research_engine._records
                )
                self._validation_engine.register_store(
                    KnowledgeType.EDUCATIONAL, self._educational_engine._records
                )
                logger.debug(
                    "LunaSubsystem: validation engine wired with %d store(s)",
                    7,
                )

                self._started_at = _utcnow()
                self._stopped_at = None
                self._failure_reason = None
                self._status = LunaSubsystemStatus.RUNNING
                logger.info(
                    "LunaSubsystem RUNNING (engines=%d, started_at=%s)",
                    len(initialized_engines),
                    self._started_at.isoformat(),
                )

            except Exception as exc:
                self._status = LunaSubsystemStatus.FAILED
                self._failure_reason = str(exc)
                logger.error(
                    "LunaSubsystem startup failed: %s — attempting partial rollback",
                    exc,
                )
                # Best-effort reverse shutdown of already-started engines
                for rollback_name, rollback_engine in reversed(initialized_engines):
                    try:
                        rollback_engine.shutdown()
                        logger.debug(
                            "LunaSubsystem rollback: shut down %s", rollback_name
                        )
                    except Exception as rollback_exc:
                        logger.warning(
                            "LunaSubsystem rollback: %s shutdown failed: %s",
                            rollback_name,
                            rollback_exc,
                        )
                raise LunaLifecycleError(
                    phase="startup",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        """
        Stop all owned engines in reverse dependency order and mark the
        subsystem STOPPED.

        Idempotent: if the subsystem is already STOPPED this method returns
        immediately.

        Raises:
            LunaLifecycleError: If the subsystem is in INITIALIZING or
                SHUTTING_DOWN state, or if a critical teardown failure occurs.
        """
        with self._lock:
            if self._status is LunaSubsystemStatus.STOPPED:
                return

            if self._status in (
                LunaSubsystemStatus.INITIALIZING,
                LunaSubsystemStatus.SHUTTING_DOWN,
            ):
                raise LunaLifecycleError(
                    phase="shutdown",
                    reason=f"Cannot shut down from status '{self._status.value}'",
                )

            self._status = LunaSubsystemStatus.SHUTTING_DOWN
            logger.info("LunaSubsystem shutting down")

            # Reverse of the startup order
            _shutdown_order: list[tuple[str, Any]] = [
                ("SemanticStructureEngine",       self._semantic_structure_engine),
                ("SkillProgressionEngine",        self._progression_engine),
                ("KnowledgeEvolutionEngine",      self._evolution_engine),
                ("KnowledgeIntegrityEngine",      self._integrity_engine),
                ("KnowledgeSynthesisEngine",      self._synthesis_engine),
                ("KnowledgeIndexEngine",          self._index_engine),
                ("KnowledgeRetrievalEngine",      self._retrieval_engine),
                ("KnowledgeValidationEngine",     self._validation_engine),
                ("EducationalKnowledgeEngine",    self._educational_engine),
                ("ResearchKnowledgeEngine",       self._research_engine),
                ("ProceduralKnowledgeEngine",     self._procedure_engine),
                ("KnowledgeDomainEngine",         self._domain_engine),
                ("SkillEngine",                   self._skill_engine),
                ("ConceptEngine",                 self._concept_engine),
                ("FactEngine",                    self._fact_engine),
                ("LunaEventEngine",               self._event_engine),
            ]

            errors: list[tuple[str, Exception]] = []
            for engine_name, engine in _shutdown_order:
                try:
                    logger.debug("LunaSubsystem: stopping %s", engine_name)
                    engine.shutdown()
                except Exception as exc:
                    logger.error(
                        "LunaSubsystem: %s shutdown error: %s", engine_name, exc
                    )
                    errors.append((engine_name, exc))

            self._stopped_at = _utcnow()

            if errors:
                self._status = LunaSubsystemStatus.FAILED
                self._failure_reason = "; ".join(
                    f"{name}: {exc}" for name, exc in errors
                )
                logger.error(
                    "LunaSubsystem FAILED during shutdown (%d engine error(s)): %s",
                    len(errors),
                    self._failure_reason,
                )
                first_name, first_exc = errors[0]
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine=first_name,
                    reason=self._failure_reason,
                    cause=first_exc,
                ) from first_exc
            else:
                self._status = LunaSubsystemStatus.STOPPED
                logger.info(
                    "LunaSubsystem STOPPED (stopped_at=%s)",
                    self._stopped_at.isoformat(),
                )

    def is_initialized(self) -> bool:
        """Return True if and only if the subsystem status is RUNNING."""
        return self._status is LunaSubsystemStatus.RUNNING

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL GUARD
    # ─────────────────────────────────────────────────────────────────────────

    def _require_running(self, operation: str) -> None:
        """Raise LunaNotInitializedError if the subsystem is not RUNNING."""
        if self._status is not LunaSubsystemStatus.RUNNING:
            raise LunaNotInitializedError(
                operation=operation,
                context={"subsystem_status": self._status.value},
            )

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE ACCESSORS
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def fact_engine(self) -> FactEngine:
        """The LUNA Fact Engine — manages atomic truth claims."""
        self._require_running("fact_engine")
        return self._fact_engine

    @property
    def concept_engine(self) -> ConceptEngine:
        """The LUNA Concept Engine — manages semantic concept records."""
        self._require_running("concept_engine")
        return self._concept_engine

    @property
    def skill_engine(self) -> SkillEngine:
        """The LUNA Skill Engine — manages capability and skill records."""
        self._require_running("skill_engine")
        return self._skill_engine

    @property
    def domain_engine(self) -> KnowledgeDomainEngine:
        """The LUNA Knowledge Domain Engine — manages the domain taxonomy."""
        self._require_running("domain_engine")
        return self._domain_engine

    @property
    def procedure_engine(self) -> ProceduralKnowledgeEngine:
        """The LUNA Procedural Knowledge Engine — manages step-by-step procedures."""
        self._require_running("procedure_engine")
        return self._procedure_engine

    @property
    def research_engine(self) -> ResearchKnowledgeEngine:
        """The LUNA Research Knowledge Engine — manages research artefacts."""
        self._require_running("research_engine")
        return self._research_engine

    @property
    def educational_engine(self) -> EducationalKnowledgeEngine:
        """The LUNA Educational Knowledge Engine — manages educational content."""
        self._require_running("educational_engine")
        return self._educational_engine

    @property
    def validation_engine(self) -> KnowledgeValidationEngine:
        """The LUNA Knowledge Validation Engine — cross-type validation."""
        self._require_running("validation_engine")
        return self._validation_engine

    @property
    def retrieval_engine(self) -> KnowledgeRetrievalEngine:
        """The LUNA Knowledge Retrieval Engine — unified record search and lookup."""
        self._require_running("retrieval_engine")
        return self._retrieval_engine

    @property
    def index_engine(self) -> KnowledgeIndexEngine:
        """The LUNA Knowledge Index Engine — fast secondary-index lookups."""
        self._require_running("index_engine")
        return self._index_engine

    @property
    def synthesis_engine(self) -> KnowledgeSynthesisEngine:
        """The LUNA Knowledge Synthesis Engine — higher-order knowledge composition."""
        self._require_running("synthesis_engine")
        return self._synthesis_engine

    @property
    def integrity_engine(self) -> KnowledgeIntegrityEngine:
        """The LUNA Knowledge Integrity Engine — integrity scanning and audit."""
        self._require_running("integrity_engine")
        return self._integrity_engine

    @property
    def evolution_engine(self) -> KnowledgeEvolutionEngine:
        """The LUNA Knowledge Evolution Engine — versioning and change management."""
        self._require_running("evolution_engine")
        return self._evolution_engine

    @property
    def progression_engine(self) -> SkillProgressionEngine:
        """The LUNA Skill Progression Engine — skill stage and progression models."""
        self._require_running("progression_engine")
        return self._progression_engine

    @property
    def semantic_structure_engine(self) -> SemanticStructureEngine:
        """The LUNA Semantic Structure Engine — hierarchy, relationships, and dependencies."""
        self._require_running("semantic_structure_engine")
        return self._semantic_structure_engine

    @property
    def event_engine(self) -> LunaEventEngine:
        """The LUNA Event Engine — subsystem-internal event bus."""
        self._require_running("event_engine")
        return self._event_engine

    # ─────────────────────────────────────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def status(self) -> LunaSubsystemStatus:
        """The current lifecycle status of the subsystem."""
        return self._status

    # ─────────────────────────────────────────────────────────────────────────
    # HEALTH
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """
        Return a concise health summary for the subsystem and all engines.

        This method is always safe to call regardless of lifecycle state.

        Returns:
            A plain dictionary containing:
                subsystem:          module path identifier
                version:            subsystem version string
                status:             current LunaSubsystemStatus value
                initialized:        True iff status is RUNNING
                started_at:         ISO-8601 string or None
                stopped_at:         ISO-8601 string or None
                failure_reason:     human-readable failure string or None
                engine_count:       total number of owned engines
                engines_healthy:    count of engines reporting "healthy"
                engines:            dict[engine_name → engine health summary]
        """
        with self._lock:
            engine_reports: dict[str, dict[str, Any]] = {
                name: engine.health_report()
                for name, engine in self._engine_registry()
            }

            healthy_count = sum(
                1
                for report in engine_reports.values()
                if report.get("status") == "healthy"
            )

            return {
                "subsystem": "subsystems.luna.luna.LunaSubsystem",
                "version": _SUBSYSTEM_VERSION,
                "status": self._status.value,
                "initialized": self._status is LunaSubsystemStatus.RUNNING,
                "started_at": (
                    self._started_at.isoformat() if self._started_at else None
                ),
                "stopped_at": (
                    self._stopped_at.isoformat() if self._stopped_at else None
                ),
                "failure_reason": self._failure_reason,
                "engine_count": len(engine_reports),
                "engines_healthy": healthy_count,
                "engines": engine_reports,
            }

    def diagnostics_report(self) -> dict[str, Any]:
        """
        Return a verbose diagnostics report for the subsystem and all engines.

        Extends health_report() with per-engine diagnostics details.  Always
        safe to call regardless of lifecycle state.

        Returns:
            All fields from health_report() plus:
                engines_online:     count of engines that report is_initialized()
                engines_offline:    count of engines that are not initialised
                engines:            dict[engine_name → full engine diagnostics]
        """
        with self._lock:
            engine_diags: dict[str, dict[str, Any]] = {
                name: engine.diagnostics_report()
                for name, engine in self._engine_registry()
            }

            engines_online = sum(
                1
                for _, engine in self._engine_registry()
                if engine.is_initialized()
            )
            total = len(list(self._engine_registry()))

            report = self.health_report()
            report.update({
                "engines_online": engines_online,
                "engines_offline": total - engines_online,
                "engines": engine_diags,
            })
            return report

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: ENGINE REGISTRY
    # ─────────────────────────────────────────────────────────────────────────

    def _engine_registry(self) -> list[tuple[str, Any]]:
        """
        Return all owned engines as an ordered (name, engine) list.

        Order matches the startup sequence so that log output is predictable.
        Does NOT invoke lifecycle guards — usable at any lifecycle stage.
        """
        return [
            ("LunaEventEngine",            self._event_engine),
            ("FactEngine",                 self._fact_engine),
            ("ConceptEngine",              self._concept_engine),
            ("SkillEngine",                self._skill_engine),
            ("KnowledgeDomainEngine",      self._domain_engine),
            ("ProceduralKnowledgeEngine",  self._procedure_engine),
            ("ResearchKnowledgeEngine",    self._research_engine),
            ("EducationalKnowledgeEngine", self._educational_engine),
            ("KnowledgeValidationEngine",  self._validation_engine),
            ("KnowledgeRetrievalEngine",   self._retrieval_engine),
            ("KnowledgeIndexEngine",       self._index_engine),
            ("KnowledgeSynthesisEngine",   self._synthesis_engine),
            ("KnowledgeIntegrityEngine",   self._integrity_engine),
            ("KnowledgeEvolutionEngine",   self._evolution_engine),
            ("SkillProgressionEngine",     self._progression_engine),
            ("SemanticStructureEngine",    self._semantic_structure_engine),
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # REPR
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"LunaSubsystem("
            f"status={self._status.value!r}, "
            f"version={_SUBSYSTEM_VERSION!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "LunaSubsystem",
    "LunaSubsystemStatus",
]
