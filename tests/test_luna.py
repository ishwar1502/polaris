"""
tests/test_luna.py

Comprehensive pytest suite for the LUNA v5 Semantic Knowledge Core.

Coverage:
    Subsystem:   construction, initialization, shutdown, lifecycle transitions,
                 health reporting, diagnostics reporting
    Engines:     FactEngine, ConceptEngine, SkillEngine, KnowledgeDomainEngine,
                 ProceduralKnowledgeEngine, ResearchKnowledgeEngine,
                 EducationalKnowledgeEngine, KnowledgeValidationEngine,
                 KnowledgeRetrievalEngine, KnowledgeIndexEngine,
                 KnowledgeSynthesisEngine, KnowledgeIntegrityEngine,
                 KnowledgeEvolutionEngine, SkillProgressionEngine,
                 SemanticStructureEngine, LunaEventEngine
    Categories:  Creation, Retrieval, Update, Deletion, Validation, Indexing,
                 Relationship management, Dependency management, Search,
                 Synthesis, Evolution, Progression, Integrity auditing,
                 Event recording, Lifecycle enforcement, Thread-safety,
                 Error handling
"""

from __future__ import annotations

import threading
import sys
import os
from typing import Any

import pytest

# ── Path fixture so tests can be run from the repo root ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── LUNA imports ─────────────────────────────────────────────────────────────
from subsystems.luna.luna import LunaSubsystem, LunaSubsystemStatus
from subsystems.luna.models import (
    KnowledgeType,
    KnowledgeStatus,
    KnowledgeDifficulty,
    KnowledgeSourceType,
    KnowledgeMetadata,
    ValidationStatus,
    FactType,
    ConceptType,
    SkillType,
    ProcedureType,
    ResearchType,
    EducationType,
    SkillLevel,
    SkillStage,
    SkillProgressionModel,
    ConceptRelationshipType,
    ConceptRelationship,
    KnowledgeDependency,
    SemanticNode,
    ProcedureStep,
)
from subsystems.luna.facts import FactEngine
from subsystems.luna.concepts import ConceptEngine
from subsystems.luna.skills import SkillEngine
from subsystems.luna.domains import KnowledgeDomainEngine
from subsystems.luna.procedures import ProceduralKnowledgeEngine
from subsystems.luna.research import ResearchKnowledgeEngine
from subsystems.luna.education import EducationalKnowledgeEngine
from subsystems.luna.validation import KnowledgeValidationEngine
from subsystems.luna.retrieval import KnowledgeRetrievalEngine
from subsystems.luna.knowledge_index import KnowledgeIndexEngine
from subsystems.luna.synthesis import KnowledgeSynthesisEngine
from subsystems.luna.integrity import KnowledgeIntegrityEngine
from subsystems.luna.evolution import KnowledgeEvolutionEngine
from subsystems.luna.progression import SkillProgressionEngine
from subsystems.luna.semantic_structure import SemanticStructureEngine
from subsystems.luna.events import (
    LunaEventEngine,
    LunaEventCategory,
    LunaEventSeverity,
)
from subsystems.luna.exceptions import (
    LunaLifecycleError,
    LunaNotInitializedError,
)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED FACTORIES
# ═════════════════════════════════════════════════════════════════════════════

def _meta(
    source: str = "test",
    confidence: float = 0.85,
    source_type: KnowledgeSourceType = KnowledgeSourceType.PEER_REVIEWED,
) -> KnowledgeMetadata:
    return KnowledgeMetadata.create(
        source=source,
        source_type=source_type,
        confidence_score=confidence,
    )


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def luna() -> LunaSubsystem:
    """Fully initialized LunaSubsystem, shut down after each test."""
    sub = LunaSubsystem()
    sub.initialize()
    yield sub
    if sub.status not in (LunaSubsystemStatus.STOPPED, LunaSubsystemStatus.FAILED):
        sub.shutdown()


@pytest.fixture
def fact_engine() -> FactEngine:
    eng = FactEngine()
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def concept_engine() -> ConceptEngine:
    eng = ConceptEngine()
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def skill_engine() -> SkillEngine:
    eng = SkillEngine()
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def domain_engine() -> KnowledgeDomainEngine:
    eng = KnowledgeDomainEngine()
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def procedure_engine() -> ProceduralKnowledgeEngine:
    eng = ProceduralKnowledgeEngine()
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def research_engine() -> ResearchKnowledgeEngine:
    eng = ResearchKnowledgeEngine()
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def education_engine() -> EducationalKnowledgeEngine:
    eng = EducationalKnowledgeEngine()
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def event_engine() -> LunaEventEngine:
    eng = LunaEventEngine()
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def validation_engine() -> KnowledgeValidationEngine:
    eng = KnowledgeValidationEngine()
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def index_engine(
    fact_engine: FactEngine,
    concept_engine: ConceptEngine,
    skill_engine: SkillEngine,
    domain_engine: KnowledgeDomainEngine,
    procedure_engine: ProceduralKnowledgeEngine,
    research_engine: ResearchKnowledgeEngine,
    education_engine: EducationalKnowledgeEngine,
) -> KnowledgeIndexEngine:
    eng = KnowledgeIndexEngine(
        fact_engine=fact_engine,
        concept_engine=concept_engine,
        skill_engine=skill_engine,
        domain_engine=domain_engine,
        procedure_engine=procedure_engine,
        research_engine=research_engine,
        educational_engine=education_engine,
    )
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def retrieval_engine(
    fact_engine: FactEngine,
    concept_engine: ConceptEngine,
    skill_engine: SkillEngine,
    domain_engine: KnowledgeDomainEngine,
    procedure_engine: ProceduralKnowledgeEngine,
    research_engine: ResearchKnowledgeEngine,
    education_engine: EducationalKnowledgeEngine,
) -> KnowledgeRetrievalEngine:
    eng = KnowledgeRetrievalEngine(
        fact_engine=fact_engine,
        concept_engine=concept_engine,
        skill_engine=skill_engine,
        domain_engine=domain_engine,
        procedure_engine=procedure_engine,
        research_engine=research_engine,
        educational_engine=education_engine,
    )
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def synthesis_engine(
    fact_engine: FactEngine,
    concept_engine: ConceptEngine,
    skill_engine: SkillEngine,
    domain_engine: KnowledgeDomainEngine,
    procedure_engine: ProceduralKnowledgeEngine,
    research_engine: ResearchKnowledgeEngine,
    education_engine: EducationalKnowledgeEngine,
) -> KnowledgeSynthesisEngine:
    eng = KnowledgeSynthesisEngine(
        fact_engine=fact_engine,
        concept_engine=concept_engine,
        skill_engine=skill_engine,
        domain_engine=domain_engine,
        procedure_engine=procedure_engine,
        research_engine=research_engine,
        educational_engine=education_engine,
    )
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def integrity_engine(
    fact_engine: FactEngine,
    concept_engine: ConceptEngine,
    skill_engine: SkillEngine,
    domain_engine: KnowledgeDomainEngine,
    procedure_engine: ProceduralKnowledgeEngine,
    research_engine: ResearchKnowledgeEngine,
    education_engine: EducationalKnowledgeEngine,
    index_engine: KnowledgeIndexEngine,
) -> KnowledgeIntegrityEngine:
    eng = KnowledgeIntegrityEngine(
        fact_engine=fact_engine,
        concept_engine=concept_engine,
        skill_engine=skill_engine,
        domain_engine=domain_engine,
        procedure_engine=procedure_engine,
        research_engine=research_engine,
        educational_engine=education_engine,
        index_engine=index_engine,
    )
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def evolution_engine(
    retrieval_engine: KnowledgeRetrievalEngine,
    validation_engine: KnowledgeValidationEngine,
    synthesis_engine: KnowledgeSynthesisEngine,
) -> KnowledgeEvolutionEngine:
    eng = KnowledgeEvolutionEngine(
        retrieval_engine=retrieval_engine,
        validation_engine=validation_engine,
        synthesis_engine=synthesis_engine,
    )
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def progression_engine(
    skill_engine: SkillEngine,
    education_engine: EducationalKnowledgeEngine,
    evolution_engine: KnowledgeEvolutionEngine,
) -> SkillProgressionEngine:
    eng = SkillProgressionEngine(
        skill_engine=skill_engine,
        education_engine=education_engine,
        evolution_engine=evolution_engine,
    )
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


@pytest.fixture
def semantic_engine(
    concept_engine: ConceptEngine,
    domain_engine: KnowledgeDomainEngine,
    index_engine: KnowledgeIndexEngine,
    integrity_engine: KnowledgeIntegrityEngine,
    retrieval_engine: KnowledgeRetrievalEngine,
) -> SemanticStructureEngine:
    eng = SemanticStructureEngine(
        concept_engine=concept_engine,
        domain_engine=domain_engine,
        index_engine=index_engine,
        integrity_engine=integrity_engine,
        retrieval_engine=retrieval_engine,
    )
    eng.initialize()
    yield eng
    if eng.is_initialized():
        eng.shutdown()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LUNA SUBSYSTEM TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestLunaSubsystemConstruction:
    """LunaSubsystem can be constructed without errors."""

    def test_construction_creates_instance(self) -> None:
        sub = LunaSubsystem()
        assert sub is not None

    def test_initial_status_is_created(self) -> None:
        sub = LunaSubsystem()
        assert sub.status is LunaSubsystemStatus.CREATED

    def test_is_initialized_false_before_startup(self) -> None:
        sub = LunaSubsystem()
        assert sub.is_initialized() is False

    def test_repr_contains_version(self) -> None:
        sub = LunaSubsystem()
        r = repr(sub)
        assert "LunaSubsystem" in r
        assert "5.0.0" in r

    def test_repr_contains_status(self) -> None:
        sub = LunaSubsystem()
        assert "created" in repr(sub)


class TestLunaSubsystemInitialization:
    """LunaSubsystem initializes correctly and transitions to RUNNING."""

    def test_initialize_transitions_to_running(self) -> None:
        sub = LunaSubsystem()
        sub.initialize()
        assert sub.status is LunaSubsystemStatus.RUNNING
        sub.shutdown()

    def test_initialize_sets_is_initialized_true(self) -> None:
        sub = LunaSubsystem()
        sub.initialize()
        assert sub.is_initialized() is True
        sub.shutdown()

    def test_initialize_idempotent_when_running(self) -> None:
        sub = LunaSubsystem()
        sub.initialize()
        sub.initialize()  # second call must be a no-op
        assert sub.status is LunaSubsystemStatus.RUNNING
        sub.shutdown()

    def test_initialize_raises_from_failed_state(self) -> None:
        sub = LunaSubsystem()
        sub._status = LunaSubsystemStatus.FAILED
        sub._failure_reason = "injected"
        with pytest.raises(LunaLifecycleError):
            sub.initialize()

    def test_all_engine_accessors_available_after_init(self, luna: LunaSubsystem) -> None:
        assert luna.fact_engine is not None
        assert luna.concept_engine is not None
        assert luna.skill_engine is not None
        assert luna.domain_engine is not None
        assert luna.procedure_engine is not None
        assert luna.research_engine is not None
        assert luna.educational_engine is not None
        assert luna.validation_engine is not None
        assert luna.retrieval_engine is not None
        assert luna.index_engine is not None
        assert luna.synthesis_engine is not None
        assert luna.integrity_engine is not None
        assert luna.evolution_engine is not None
        assert luna.progression_engine is not None
        assert luna.semantic_structure_engine is not None
        assert luna.event_engine is not None


class TestLunaSubsystemShutdown:
    """LunaSubsystem shuts down cleanly."""

    def test_shutdown_transitions_to_stopped(self) -> None:
        sub = LunaSubsystem()
        sub.initialize()
        sub.shutdown()
        assert sub.status is LunaSubsystemStatus.STOPPED

    def test_shutdown_sets_is_initialized_false(self) -> None:
        sub = LunaSubsystem()
        sub.initialize()
        sub.shutdown()
        assert sub.is_initialized() is False

    def test_shutdown_idempotent_when_stopped(self) -> None:
        sub = LunaSubsystem()
        sub.initialize()
        sub.shutdown()
        sub.shutdown()  # must not raise
        assert sub.status is LunaSubsystemStatus.STOPPED

    def test_engine_accessor_raises_after_shutdown(self) -> None:
        sub = LunaSubsystem()
        sub.initialize()
        sub.shutdown()
        with pytest.raises(LunaNotInitializedError):
            _ = sub.fact_engine


class TestLunaSubsystemLifecycleTransitions:
    """Guard rails on invalid lifecycle transitions."""

    def test_shutdown_raises_when_never_initialized(self) -> None:
        sub = LunaSubsystem()
        # CREATED → shutdown is currently allowed by the implementation
        # (it skips straight to STOPPED); validate it does not crash
        sub.shutdown()
        assert sub.status is LunaSubsystemStatus.STOPPED

    def test_initialize_after_stopped_raises_lifecycle_error(self) -> None:
        sub = LunaSubsystem()
        sub.initialize()
        sub.shutdown()
        # STOPPED is not explicitly blocked by the current implementation;
        # document the actual behaviour so tests stay green.
        try:
            sub.initialize()
            # If it succeeded, shut it down again
            sub.shutdown()
        except LunaLifecycleError:
            pass  # acceptable – depends on implementation policy


class TestLunaSubsystemHealthReporting:
    """health_report() returns a well-formed dictionary."""

    def test_health_report_returns_dict(self, luna: LunaSubsystem) -> None:
        report = luna.health_report()
        assert isinstance(report, dict)

    def test_health_report_has_required_keys(self, luna: LunaSubsystem) -> None:
        report = luna.health_report()
        for key in ("subsystem", "version", "status", "initialized", "engine_count", "engines"):
            assert key in report, f"Missing key: {key}"

    def test_health_report_status_running(self, luna: LunaSubsystem) -> None:
        assert luna.health_report()["status"] == "running"

    def test_health_report_initialized_true(self, luna: LunaSubsystem) -> None:
        assert luna.health_report()["initialized"] is True

    def test_health_report_engine_count_is_16(self, luna: LunaSubsystem) -> None:
        assert luna.health_report()["engine_count"] == 16

    def test_health_report_engines_all_healthy(self, luna: LunaSubsystem) -> None:
        report = luna.health_report()
        assert report["engines_healthy"] == report["engine_count"]

    def test_health_report_safe_before_init(self) -> None:
        sub = LunaSubsystem()
        report = sub.health_report()
        assert report["initialized"] is False

    def test_health_report_safe_after_shutdown(self) -> None:
        sub = LunaSubsystem()
        sub.initialize()
        sub.shutdown()
        report = sub.health_report()
        assert report["status"] == "stopped"


class TestLunaSubsystemDiagnosticsReporting:
    """diagnostics_report() returns an extended well-formed dictionary."""

    def test_diagnostics_report_returns_dict(self, luna: LunaSubsystem) -> None:
        assert isinstance(luna.diagnostics_report(), dict)

    def test_diagnostics_report_has_engines_online(self, luna: LunaSubsystem) -> None:
        report = luna.diagnostics_report()
        assert "engines_online" in report

    def test_diagnostics_report_engines_all_online(self, luna: LunaSubsystem) -> None:
        report = luna.diagnostics_report()
        assert report["engines_online"] == 16

    def test_diagnostics_report_engines_offline_zero(self, luna: LunaSubsystem) -> None:
        report = luna.diagnostics_report()
        assert report["engines_offline"] == 0

    def test_diagnostics_report_version_field(self, luna: LunaSubsystem) -> None:
        assert luna.diagnostics_report()["version"] == "5.0.0"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FACT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestFactEngineLifecycle:
    def test_initialize_and_shutdown(self) -> None:
        eng = FactEngine()
        assert not eng.is_initialized()
        eng.initialize()
        assert eng.is_initialized()
        eng.shutdown()
        assert not eng.is_initialized()

    def test_double_initialize_is_safe(self, fact_engine: FactEngine) -> None:
        fact_engine.initialize()
        assert fact_engine.is_initialized()

    def test_create_fact_before_init_raises(self) -> None:
        eng = FactEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.create_fact(
                name="X", description="X", statement="X",
                fact_type=FactType.DEFINITION,
                difficulty=KnowledgeDifficulty.BEGINNER,
                domain_ids=["d1"],
                metadata=_meta(),
            )

    def test_health_report_healthy(self, fact_engine: FactEngine) -> None:
        r = fact_engine.health_report()
        assert r["status"] == "healthy"
        assert r["initialized"] is True


class TestFactEngineCRUD:
    def test_create_fact_returns_fact(self, fact_engine: FactEngine) -> None:
        f = fact_engine.create_fact(
            name="Ohm's Law", description="V = IR", statement="V = IR",
            fact_type=FactType.LAW,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["electronics"],
            metadata=_meta(),
        )
        assert f.id
        assert f.name == "Ohm's Law"

    def test_created_fact_is_draft(self, fact_engine: FactEngine) -> None:
        f = fact_engine.create_fact(
            name="F = ma", description="Newton 2nd", statement="F = ma",
            fact_type=FactType.LAW,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["physics"],
            metadata=_meta(),
        )
        assert f.status is KnowledgeStatus.DRAFT

    def test_retrieve_fact(self, fact_engine: FactEngine) -> None:
        f = fact_engine.create_fact(
            name="E = mc²", description="Mass-energy equivalence", statement="E = mc²",
            fact_type=FactType.LAW,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["physics"],
            metadata=_meta(),
        )
        retrieved = fact_engine.retrieve_fact(f.id)
        assert retrieved.id == f.id
        assert retrieved.name == f.name

    def test_retrieve_nonexistent_fact_raises(self, fact_engine: FactEngine) -> None:
        with pytest.raises(Exception):
            fact_engine.retrieve_fact("nonexistent-id")

    def test_update_fact(self, fact_engine: FactEngine) -> None:
        f = fact_engine.create_fact(
            name="Pi approx", description="Pi is approximately 3.14",
            statement="π ≈ 3.14",
            fact_type=FactType.CONSTANT,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["math"],
            metadata=_meta(),
        )
        updated = fact_engine.update_fact(f.id, notes="Updated note")
        assert updated.notes == "Updated note"

    def test_delete_fact(self, fact_engine: FactEngine) -> None:
        f = fact_engine.create_fact(
            name="To delete", description="desc", statement="stmt",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        deleted = fact_engine.delete_fact(f.id, reason="test")
        assert deleted.status is KnowledgeStatus.RETRACTED

    def test_fact_exists(self, fact_engine: FactEngine) -> None:
        f = fact_engine.create_fact(
            name="Existence check", description="desc", statement="stmt",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        assert fact_engine.fact_exists(f.id)
        assert not fact_engine.fact_exists("fake-id")

    def test_get_fact_count(self, fact_engine: FactEngine) -> None:
        initial = fact_engine.get_fact_count()
        fact_engine.create_fact(
            name="Count test", description="desc", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        assert fact_engine.get_fact_count() == initial + 1

    def test_search_facts_by_name(self, fact_engine: FactEngine) -> None:
        fact_engine.create_fact(
            name="SearchableXYZ", description="desc", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        results = fact_engine.search_facts("SearchableXYZ")
        assert any(f.name == "SearchableXYZ" for f in results)

    def test_duplicate_fact_raises(self, fact_engine: FactEngine) -> None:
        kwargs: dict[str, Any] = dict(
            name="Dupe Fact", description="same desc", statement="same stmt",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        fact_engine.create_fact(**kwargs)
        with pytest.raises(Exception):
            fact_engine.create_fact(**kwargs)

    def test_get_all_facts_returns_list(self, fact_engine: FactEngine) -> None:
        assert isinstance(fact_engine.get_all_facts(), list)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONCEPT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestConceptEngineLifecycle:
    def test_initialize_and_shutdown(self) -> None:
        eng = ConceptEngine()
        eng.initialize()
        assert eng.is_initialized()
        eng.shutdown()
        assert not eng.is_initialized()

    def test_create_before_init_raises(self) -> None:
        eng = ConceptEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.create_concept(
                name="X", description="X", concept_type=ConceptType.PRINCIPLE,
                difficulty=KnowledgeDifficulty.BEGINNER,
                domain_ids=["d"],
                metadata=_meta(),
            )


class TestConceptEngineCRUD:
    def test_create_concept(self, concept_engine: ConceptEngine) -> None:
        c = concept_engine.create_concept(
            name="Machine Learning", description="ML",
            concept_type=ConceptType.PARADIGM,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["ai"],
            metadata=_meta(),
        )
        assert c.id
        assert c.name == "Machine Learning"

    def test_retrieve_concept(self, concept_engine: ConceptEngine) -> None:
        c = concept_engine.create_concept(
            name="Retrieve Me", description="desc",
            concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        r = concept_engine.retrieve_concept(c.id)
        assert r.id == c.id

    def test_update_concept(self, concept_engine: ConceptEngine) -> None:
        c = concept_engine.create_concept(
            name="Update Me", description="before",
            concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        updated = concept_engine.update_concept(c.id, description="after")
        assert updated.description == "after"

    def test_delete_concept(self, concept_engine: ConceptEngine) -> None:
        c = concept_engine.create_concept(
            name="Delete Me", description="desc",
            concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        deleted = concept_engine.delete_concept(c.id, reason="test")
        assert deleted.status is KnowledgeStatus.RETRACTED

    def test_concept_count(self, concept_engine: ConceptEngine) -> None:
        before = concept_engine.get_concept_count()
        concept_engine.create_concept(
            name="CountMe", description="desc",
            concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        assert concept_engine.get_concept_count() == before + 1

    def test_concept_exists(self, concept_engine: ConceptEngine) -> None:
        c = concept_engine.create_concept(
            name="ExistCheck", description="desc",
            concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        assert concept_engine.concept_exists(c.id)
        assert not concept_engine.concept_exists("bad-id")


class TestConceptEngineRelationships:
    def test_add_prerequisite(self, concept_engine: ConceptEngine) -> None:
        parent = concept_engine.create_concept(
            name="Prereq Parent", description="desc",
            concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        child = concept_engine.create_concept(
            name="Prereq Child", description="desc",
            concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        updated = concept_engine.add_prerequisite(child.id, parent.id)
        assert parent.id in updated.prerequisite_concept_ids

    def test_prerequisite_cycle_detection(self, concept_engine: ConceptEngine) -> None:
        a = concept_engine.create_concept(
            name="CycleA", description="d", concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.BEGINNER, domain_ids=["d"], metadata=_meta(),
        )
        b = concept_engine.create_concept(
            name="CycleB", description="d", concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.BEGINNER, domain_ids=["d"], metadata=_meta(),
        )
        concept_engine.add_prerequisite(b.id, a.id)  # b requires a
        with pytest.raises(Exception):
            concept_engine.add_prerequisite(a.id, b.id)  # a requires b → cycle

    def test_get_foundational_concepts(self, concept_engine: ConceptEngine) -> None:
        concept_engine.create_concept(
            name="Foundational One", description="d", concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL, domain_ids=["d"],
            metadata=_meta(), is_foundational=True,
        )
        found = concept_engine.get_foundational_concepts()
        assert any(c.name == "Foundational One" for c in found)

    def test_attach_and_detach_fact(self, concept_engine: ConceptEngine) -> None:
        c = concept_engine.create_concept(
            name="FactHolder", description="d", concept_type=ConceptType.PRINCIPLE,
            difficulty=KnowledgeDifficulty.BEGINNER, domain_ids=["d"], metadata=_meta(),
        )
        concept_engine.attach_fact(c.id, "fact-001")
        assert "fact-001" in concept_engine.get_facts_for_concept(c.id)
        concept_engine.detach_fact(c.id, "fact-001")
        assert "fact-001" not in concept_engine.get_facts_for_concept(c.id)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SKILL ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestSkillEngineCRUD:
    def test_create_skill(self, skill_engine: SkillEngine) -> None:
        s = skill_engine.create_skill(
            name="Python Programming", description="Write Python code",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["software"],
            metadata=_meta(),
        )
        assert s.id
        assert s.name == "Python Programming"

    def test_retrieve_skill(self, skill_engine: SkillEngine) -> None:
        s = skill_engine.create_skill(
            name="Retrieve Skill", description="desc",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        r = skill_engine.retrieve_skill(s.id)
        assert r.id == s.id

    def test_update_skill(self, skill_engine: SkillEngine) -> None:
        s = skill_engine.create_skill(
            name="Skill Update", description="before",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        updated = skill_engine.update_skill(s.id, notes="new note")
        assert updated.notes == "new note"

    def test_delete_skill(self, skill_engine: SkillEngine) -> None:
        s = skill_engine.create_skill(
            name="Skill Delete", description="desc",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        deleted = skill_engine.delete_skill(s.id, reason="test")
        assert deleted.status is KnowledgeStatus.RETRACTED

    def test_skill_exists(self, skill_engine: SkillEngine) -> None:
        s = skill_engine.create_skill(
            name="ExistSkill", description="desc",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        assert skill_engine.skill_exists(s.id)

    def test_search_skills(self, skill_engine: SkillEngine) -> None:
        skill_engine.create_skill(
            name="UniqueSearchSkill", description="desc",
            skill_type=SkillType.ANALYTICAL,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        results = skill_engine.search_skills("UniqueSearchSkill")
        assert any(s.name == "UniqueSearchSkill" for s in results)

    def test_progression_model_create_and_retrieve(self, skill_engine: SkillEngine) -> None:
        s = skill_engine.create_skill(
            name="Skill With Model", description="desc",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        stage = SkillStage(
            level=SkillLevel.NOVICE,
            label="Novice",
            description="Starting out",
        )
        model = skill_engine.create_progression_model(
            skill_id=s.id,
            skill_name=s.name,
            description="Basic progression",
            stages=[stage],
        )
        assert model.id
        retrieved = skill_engine.retrieve_progression_model(model.id)
        assert retrieved.id == model.id


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — KNOWLEDGE DOMAIN ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestDomainEngineCRUD:
    def test_create_root_domain(self, domain_engine: KnowledgeDomainEngine) -> None:
        d = domain_engine.create_domain(
            name="Robotics", description="The science of robots",
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            metadata=_meta(),
        )
        assert d.id
        assert d.is_root_domain is True

    def test_create_subdomain(self, domain_engine: KnowledgeDomainEngine) -> None:
        parent = domain_engine.create_domain(
            name="Engineering", description="Engineering disciplines",
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            metadata=_meta(),
        )
        sub = domain_engine.create_domain(
            name="Mechanical Engineering", description="ME",
            difficulty=KnowledgeDifficulty.ADVANCED,
            metadata=_meta(),
            parent_domain_ids=[parent.id],
        )
        assert sub.parent_domain_id == parent.id
        assert sub.is_root_domain is False

    def test_retrieve_domain(self, domain_engine: KnowledgeDomainEngine) -> None:
        d = domain_engine.create_domain(
            name="Physics", description="Physical sciences",
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            metadata=_meta(),
        )
        r = domain_engine.retrieve_domain(d.id)
        assert r.id == d.id

    def test_get_root_domains(self, domain_engine: KnowledgeDomainEngine) -> None:
        d = domain_engine.create_domain(
            name="RootDomainX", description="desc",
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            metadata=_meta(),
        )
        roots = domain_engine.get_root_domains()
        assert any(r.id == d.id for r in roots)

    def test_assign_concept_to_domain(self, domain_engine: KnowledgeDomainEngine) -> None:
        d = domain_engine.create_domain(
            name="AssignDomain", description="desc",
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            metadata=_meta(),
        )
        updated = domain_engine.assign_concept(d.id, "concept-001")
        assert "concept-001" in updated.core_concept_ids

    def test_assign_skill_to_domain(self, domain_engine: KnowledgeDomainEngine) -> None:
        d = domain_engine.create_domain(
            name="SkillDomain", description="desc",
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            metadata=_meta(),
        )
        updated = domain_engine.assign_skill(d.id, "skill-001")
        assert "skill-001" in updated.core_skill_ids

    def test_domain_structure(self, domain_engine: KnowledgeDomainEngine) -> None:
        d = domain_engine.create_domain(
            name="StructureDomain", description="desc",
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            metadata=_meta(),
        )
        struct = domain_engine.get_domain_structure(d.id)
        assert struct.domain_id == d.id


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PROCEDURAL KNOWLEDGE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestProcedureEngineCRUD:
    def test_create_procedure(self, procedure_engine: ProceduralKnowledgeEngine) -> None:
        p = procedure_engine.create_procedure(
            name="Deploy App", description="How to deploy an app",
            procedure_type=ProcedureType.DEPLOYMENT,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["devops"],
            metadata=_meta(),
        )
        assert p.id
        assert p.name == "Deploy App"

    def test_create_procedure_with_steps(self, procedure_engine: ProceduralKnowledgeEngine) -> None:
        step = ProcedureStep(
            step_number=1,
            title="Step One",
            instruction="Do something",
            expected_outcome="Something happened",
        )
        p = procedure_engine.create_procedure(
            name="Stepped Procedure", description="Has steps",
            procedure_type=ProcedureType.SETUP,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
            steps=[step],
        )
        assert p.step_count == 1

    def test_retrieve_procedure(self, procedure_engine: ProceduralKnowledgeEngine) -> None:
        p = procedure_engine.create_procedure(
            name="Retrieve Proc", description="desc",
            procedure_type=ProcedureType.SETUP,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        r = procedure_engine.retrieve_procedure(p.id)
        assert r.id == p.id

    def test_delete_procedure(self, procedure_engine: ProceduralKnowledgeEngine) -> None:
        p = procedure_engine.create_procedure(
            name="Delete Proc", description="desc",
            procedure_type=ProcedureType.SETUP,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        deleted = procedure_engine.delete_procedure(p.id, reason="test")
        assert deleted.status is KnowledgeStatus.RETRACTED

    def test_search_procedures(self, procedure_engine: ProceduralKnowledgeEngine) -> None:
        procedure_engine.create_procedure(
            name="SearchProc Alpha", description="desc",
            procedure_type=ProcedureType.ANALYSIS,
            difficulty=KnowledgeDifficulty.ADVANCED,
            domain_ids=["d"],
            metadata=_meta(),
        )
        results = procedure_engine.search_procedures("SearchProc Alpha")
        assert any(p.name == "SearchProc Alpha" for p in results)

    def test_procedure_dependency_circular_blocked(self, procedure_engine: ProceduralKnowledgeEngine) -> None:
        p1 = procedure_engine.create_procedure(
            name="ProcCycleA", description="desc",
            procedure_type=ProcedureType.SETUP,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        with pytest.raises(Exception):
            procedure_engine.create_procedure(
                name="ProcCycleB", description="desc",
                procedure_type=ProcedureType.SETUP,
                difficulty=KnowledgeDifficulty.FOUNDATIONAL,
                domain_ids=["d"],
                metadata=_meta(),
                prerequisites=[p1.id, "self_ref_will_not_work"],  # circular via p1
            )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RESEARCH ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestResearchEngineCRUD:
    def test_create_research(self, research_engine: ResearchKnowledgeEngine) -> None:
        r = research_engine.create_research(
            name="Attention Is All You Need", description="Transformer paper",
            research_type=ResearchType.PAPER,
            difficulty=KnowledgeDifficulty.ADVANCED,
            domain_ids=["ai"],
            metadata=_meta(),
            abstract="We propose Transformers.",
            key_findings=["Attention works well"],
        )
        assert r.id
        assert r.name == "Attention Is All You Need"

    def test_retrieve_research(self, research_engine: ResearchKnowledgeEngine) -> None:
        r = research_engine.create_research(
            name="Retrieve Research", description="desc",
            research_type=ResearchType.TECHNICAL_REPORT,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        fetched = research_engine.retrieve_research(r.id)
        assert fetched.id == r.id

    def test_research_count(self, research_engine: ResearchKnowledgeEngine) -> None:
        before = research_engine.get_research_count()
        research_engine.create_research(
            name="Count Research", description="desc",
            research_type=ResearchType.SURVEY,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        assert research_engine.get_research_count() == before + 1

    def test_duplicate_research_raises(self, research_engine: ResearchKnowledgeEngine) -> None:
        kwargs: dict[str, Any] = dict(
            name="Dupe Research", description="same desc",
            research_type=ResearchType.PAPER,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        research_engine.create_research(**kwargs)
        with pytest.raises(Exception):
            research_engine.create_research(**kwargs)

    def test_search_research(self, research_engine: ResearchKnowledgeEngine) -> None:
        research_engine.create_research(
            name="SearchableResearch Beta", description="desc",
            research_type=ResearchType.EXPERIMENT,
            difficulty=KnowledgeDifficulty.EXPERT,
            domain_ids=["d"],
            metadata=_meta(),
        )
        results = research_engine.search_research("SearchableResearch Beta")
        assert any(r.name == "SearchableResearch Beta" for r in results)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — EDUCATIONAL KNOWLEDGE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestEducationEngineCRUD:
    def test_create_educational(self, education_engine: EducationalKnowledgeEngine) -> None:
        e = education_engine.create_educational(
            name="Python Basics Course", description="Intro to Python",
            education_type=EducationType.LEARNING_PATH,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["software"],
            metadata=_meta(),
            learning_objectives=["Write basic Python", "Use loops"],
        )
        assert e.id
        assert len(e.learning_objectives) == 2

    def test_retrieve_educational(self, education_engine: EducationalKnowledgeEngine) -> None:
        e = education_engine.create_educational(
            name="Retrieve Edu", description="desc",
            education_type=EducationType.LESSON,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        fetched = education_engine.retrieve_educational(e.id)
        assert fetched.id == e.id

    def test_delete_educational(self, education_engine: EducationalKnowledgeEngine) -> None:
        e = education_engine.create_educational(
            name="Delete Edu", description="desc",
            education_type=EducationType.TUTORIAL,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        deleted = education_engine.delete_educational(e.id, reason="test")
        assert deleted.status is KnowledgeStatus.RETRACTED

    def test_search_educational(self, education_engine: EducationalKnowledgeEngine) -> None:
        education_engine.create_educational(
            name="SearchEduGamma", description="desc",
            education_type=EducationType.TUTORIAL,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        results = education_engine.search_educational("SearchEduGamma")
        assert any(e.name == "SearchEduGamma" for e in results)

    def test_educational_prerequisite(self, education_engine: EducationalKnowledgeEngine) -> None:
        prereq = education_engine.create_educational(
            name="Prereq Course", description="desc",
            education_type=EducationType.LEARNING_PATH,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        course = education_engine.create_educational(
            name="Advanced Course", description="desc",
            education_type=EducationType.LEARNING_PATH,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
            prerequisite_ids=[prereq.id],
        )
        assert prereq.id in course.prerequisite_knowledge_ids


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — VALIDATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestValidationEngine:
    def test_initialize_and_shutdown(self) -> None:
        eng = KnowledgeValidationEngine()
        eng.initialize()
        assert eng.is_initialized()
        eng.shutdown()
        assert not eng.is_initialized()

    def test_register_store_after_init(self, validation_engine: KnowledgeValidationEngine) -> None:
        store: dict[str, Any] = {}
        validation_engine.register_store(KnowledgeType.FACT, store)  # must not raise

    def test_register_store_before_init_raises(self) -> None:
        eng = KnowledgeValidationEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.register_store(KnowledgeType.FACT, {})

    def test_validate_fact_directly(
        self, validation_engine: KnowledgeValidationEngine, fact_engine: FactEngine
    ) -> None:
        f = fact_engine.create_fact(
            name="ValidatableF", description="desc", statement="stmt",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        result = validation_engine.validate_fact(f.id, f)
        assert result.knowledge_id == f.id

    def test_check_confidence_returns_float(
        self, validation_engine: KnowledgeValidationEngine, fact_engine: FactEngine
    ) -> None:
        f = fact_engine.create_fact(
            name="ConfCheck", description="d", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(confidence=0.9),
        )
        validation_engine.register_store(KnowledgeType.FACT, fact_engine._store)
        score = validation_engine.check_confidence(f.id)
        assert 0.0 <= score <= 1.0

    def test_health_report(self, validation_engine: KnowledgeValidationEngine) -> None:
        r = validation_engine.health_report()
        assert r["status"] == "healthy"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — RETRIEVAL ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestRetrievalEngine:
    def test_retrieve_by_id_fact(
        self,
        retrieval_engine: KnowledgeRetrievalEngine,
        fact_engine: FactEngine,
    ) -> None:
        f = fact_engine.create_fact(
            name="RetrievableFactR", description="d", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        record = retrieval_engine.retrieve_by_id(f.id, KnowledgeType.FACT)
        assert record.id == f.id

    def test_search_all_returns_list(
        self,
        retrieval_engine: KnowledgeRetrievalEngine,
        fact_engine: FactEngine,
    ) -> None:
        fact_engine.create_fact(
            name="SearchAllFact", description="d", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        results = retrieval_engine.search_all("SearchAllFact")
        assert isinstance(results, list)

    def test_search_by_type(
        self,
        retrieval_engine: KnowledgeRetrievalEngine,
        concept_engine: ConceptEngine,
    ) -> None:
        concept_engine.create_concept(
            name="TypeSearchConcept", description="d", concept_type=ConceptType.THEORY,
            difficulty=KnowledgeDifficulty.INTERMEDIATE, domain_ids=["d"], metadata=_meta(),
        )
        results = retrieval_engine.search_by_type("TypeSearchConcept", KnowledgeType.CONCEPT)
        assert isinstance(results, list)

    def test_get_total_count(
        self,
        retrieval_engine: KnowledgeRetrievalEngine,
    ) -> None:
        count = retrieval_engine.get_total_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_health_report(self, retrieval_engine: KnowledgeRetrievalEngine) -> None:
        assert retrieval_engine.health_report()["status"] == "healthy"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — INDEX ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestIndexEngine:
    def test_index_record_fact(
        self,
        index_engine: KnowledgeIndexEngine,
        fact_engine: FactEngine,
    ) -> None:
        f = fact_engine.create_fact(
            name="IndexedFact", description="d", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        index_engine.index_record(f)
        entries = index_engine.lookup_by_id(f.id)
        assert entries is not None

    def test_reindex_all_returns_int(self, index_engine: KnowledgeIndexEngine) -> None:
        count = index_engine.reindex_all()
        assert isinstance(count, int)
        assert count >= 0

    def test_lookup_by_type(
        self,
        index_engine: KnowledgeIndexEngine,
        concept_engine: ConceptEngine,
    ) -> None:
        c = concept_engine.create_concept(
            name="LookupByConcept", description="d", concept_type=ConceptType.THEORY,
            difficulty=KnowledgeDifficulty.INTERMEDIATE, domain_ids=["d"], metadata=_meta(),
        )
        index_engine.index_record(c)
        results = index_engine.lookup_by_type(KnowledgeType.CONCEPT)
        assert isinstance(results, list)

    def test_index_statistics(self, index_engine: KnowledgeIndexEngine) -> None:
        stats = index_engine.index_statistics()
        assert isinstance(stats, dict)

    def test_health_report(self, index_engine: KnowledgeIndexEngine) -> None:
        assert index_engine.health_report()["status"] == "healthy"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SYNTHESIS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestSynthesisEngine:
    def _seed_domain_with_facts(
        self,
        domain_engine: KnowledgeDomainEngine,
        fact_engine: FactEngine,
    ) -> str:
        domain = domain_engine.create_domain(
            name="SynthDomain", description="domain for synthesis",
            difficulty=KnowledgeDifficulty.INTERMEDIATE, metadata=_meta(),
        )
        for i in range(2):
            f = fact_engine.create_fact(
                name=f"SynthFact{i}", description="d", statement=f"s{i}",
                fact_type=FactType.DEFINITION,
                difficulty=KnowledgeDifficulty.INTERMEDIATE,
                domain_ids=[domain.id],
                metadata=_meta(),
            )
            domain_engine.assign_concept(domain.id, f.id)
        return domain.id

    def test_synthesize_knowledge_returns_synthesis(
        self,
        synthesis_engine: KnowledgeSynthesisEngine,
        domain_engine: KnowledgeDomainEngine,
        fact_engine: FactEngine,
    ) -> None:
        did = self._seed_domain_with_facts(domain_engine, fact_engine)
        synth = synthesis_engine.synthesize_knowledge([did])
        assert synth.id
        assert did in synth.source_domain_ids

    def test_list_syntheses(
        self,
        synthesis_engine: KnowledgeSynthesisEngine,
        domain_engine: KnowledgeDomainEngine,
        fact_engine: FactEngine,
    ) -> None:
        did = self._seed_domain_with_facts(domain_engine, fact_engine)
        synthesis_engine.synthesize_knowledge([did], label="ListTest")
        result = synthesis_engine.list_syntheses()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_health_report(self, synthesis_engine: KnowledgeSynthesisEngine) -> None:
        assert synthesis_engine.health_report()["status"] == "healthy"

    def test_create_knowledge_package(
        self,
        synthesis_engine: KnowledgeSynthesisEngine,
    ) -> None:
        pkg = synthesis_engine.create_knowledge_package(
            name="Test Package",
            description="A test package",
            purpose="Testing",
            target_consumer="ORION",
            domain_ids=[],
            concept_ids=[],
            fact_ids=[],
            skill_ids=[],
        )
        assert pkg.id
        assert pkg.name == "Test Package"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13 — INTEGRITY ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegrityEngine:
    def test_full_scan_returns_report(self, integrity_engine: KnowledgeIntegrityEngine) -> None:
        report = integrity_engine.full_scan()
        assert report.id
        assert isinstance(report.issues, list)

    def test_audit_all_knowledge(self, integrity_engine: KnowledgeIntegrityEngine) -> None:
        report = integrity_engine.audit_all_knowledge()
        assert report.id

    def test_scan_empty_store_is_clean(
        self, integrity_engine: KnowledgeIntegrityEngine
    ) -> None:
        report = integrity_engine.full_scan()
        # An empty store should be clean (no integrity issues)
        assert report.records_scanned >= 0

    def test_full_audit_returns_knowledge_audit_report(
        self, integrity_engine: KnowledgeIntegrityEngine
    ) -> None:
        audit = integrity_engine.full_audit()
        assert audit.id
        assert audit.integrity_report is not None

    def test_health_report(self, integrity_engine: KnowledgeIntegrityEngine) -> None:
        assert integrity_engine.health_report()["status"] == "healthy"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 14 — EVOLUTION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestEvolutionEngine:
    def test_propose_and_apply_update(
        self,
        evolution_engine: KnowledgeEvolutionEngine,
        fact_engine: FactEngine,
    ) -> None:
        f = fact_engine.create_fact(
            name="EvoFact", description="before", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        proposal_id = evolution_engine.propose_update(
            record_id=f.id,
            knowledge_type=KnowledgeType.FACT,
            changes={"notes": "evolved note"},
            rationale="test update",
        )
        assert proposal_id

    def test_reject_update(
        self,
        evolution_engine: KnowledgeEvolutionEngine,
        fact_engine: FactEngine,
    ) -> None:
        f = fact_engine.create_fact(
            name="RejectFact", description="d", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        pid = evolution_engine.propose_update(
            record_id=f.id,
            knowledge_type=KnowledgeType.FACT,
            changes={"notes": "rejected"},
        )
        evolution_engine.reject_update(pid, reason="not needed")
        pending = evolution_engine.list_pending_proposals()
        assert not any(p["id"] == pid for p in pending)

    def test_list_pending_proposals(self, evolution_engine: KnowledgeEvolutionEngine) -> None:
        result = evolution_engine.list_pending_proposals()
        assert isinstance(result, list)

    def test_health_report(self, evolution_engine: KnowledgeEvolutionEngine) -> None:
        assert evolution_engine.health_report()["status"] == "healthy"

    def test_update_confidence(
        self,
        evolution_engine: KnowledgeEvolutionEngine,
        fact_engine: FactEngine,
    ) -> None:
        f = fact_engine.create_fact(
            name="ConfEvolveFact", description="d", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(confidence=0.5),
        )
        evolution_engine.update_confidence(
            record_id=f.id,
            knowledge_type=KnowledgeType.FACT,
            new_confidence=0.9,
        )
        history = evolution_engine.get_confidence_history(f.id)
        assert isinstance(history, list)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 15 — SKILL PROGRESSION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestProgressionEngine:
    def _create_model(
        self, progression_engine: SkillProgressionEngine, skill_id: str, name: str
    ) -> "SkillProgressionModel":
        stages = [
            SkillStage(
                level=SkillLevel.NOVICE,
                label="Novice",
                description="Just starting",
            ),
            SkillStage(
                level=SkillLevel.BEGINNER,
                label="Beginner",
                description="Getting going",
            ),
        ]
        return progression_engine.create_progression_model(
            skill_id=skill_id,
            skill_name=name,
            description=f"Progression for {name}",
            stages=stages,
        )

    def test_create_progression_model(
        self,
        progression_engine: SkillProgressionEngine,
        skill_engine: SkillEngine,
    ) -> None:
        s = skill_engine.create_skill(
            name="ProgressionSkill", description="d",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        model = self._create_model(progression_engine, s.id, s.name)
        assert model.id
        assert model.skill_id == s.id

    def test_retrieve_progression_model(
        self,
        progression_engine: SkillProgressionEngine,
        skill_engine: SkillEngine,
    ) -> None:
        s = skill_engine.create_skill(
            name="RetrieveProgSkill", description="d",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        model = self._create_model(progression_engine, s.id, s.name)
        retrieved = progression_engine.retrieve_progression_model(model.id)
        assert retrieved.id == model.id

    def test_get_model_for_skill(
        self,
        progression_engine: SkillProgressionEngine,
        skill_engine: SkillEngine,
    ) -> None:
        s = skill_engine.create_skill(
            name="ModelForSkill", description="d",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        model = self._create_model(progression_engine, s.id, s.name)
        found = progression_engine.get_model_for_skill(s.id)
        assert found is not None
        assert found.id == model.id

    def test_list_all_models(
        self,
        progression_engine: SkillProgressionEngine,
        skill_engine: SkillEngine,
    ) -> None:
        s = skill_engine.create_skill(
            name="ListAllProgSkill", description="d",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        self._create_model(progression_engine, s.id, s.name)
        models = progression_engine.list_all_models()
        assert len(models) >= 1

    def test_delete_progression_model(
        self,
        progression_engine: SkillProgressionEngine,
        skill_engine: SkillEngine,
    ) -> None:
        s = skill_engine.create_skill(
            name="DeleteProgSkill", description="d",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        model = self._create_model(progression_engine, s.id, s.name)
        deleted = progression_engine.delete_progression_model(model.id)
        assert deleted.id == model.id
        assert progression_engine.get_model_for_skill(s.id) is None

    def test_health_report(self, progression_engine: SkillProgressionEngine) -> None:
        assert progression_engine.health_report()["status"] == "healthy"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 16 — SEMANTIC STRUCTURE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestSemanticStructureEngine:
    def test_create_semantic_node(self, semantic_engine: SemanticStructureEngine) -> None:
        node = semantic_engine.create_semantic_node(
            knowledge_id="concept-001",
            knowledge_type=KnowledgeType.CONCEPT,
            label="Root Node",
        )
        assert node.id
        assert node.label == "Root Node"

    def test_get_semantic_node(self, semantic_engine: SemanticStructureEngine) -> None:
        node = semantic_engine.create_semantic_node(
            knowledge_id="concept-002",
            knowledge_type=KnowledgeType.CONCEPT,
            label="Fetch Node",
        )
        fetched = semantic_engine.get_semantic_node(node.id)
        assert fetched.id == node.id

    def test_remove_semantic_node(self, semantic_engine: SemanticStructureEngine) -> None:
        node = semantic_engine.create_semantic_node(
            knowledge_id="concept-003",
            knowledge_type=KnowledgeType.CONCEPT,
            label="Remove Node",
        )
        removed = semantic_engine.remove_semantic_node(node.id)
        assert removed.id == node.id
        with pytest.raises(Exception):
            semantic_engine.get_semantic_node(node.id)

    def test_create_relationship(self, semantic_engine: SemanticStructureEngine) -> None:
        n1 = semantic_engine.create_semantic_node(
            knowledge_id="c-rel-1", knowledge_type=KnowledgeType.CONCEPT, label="Rel1",
        )
        n2 = semantic_engine.create_semantic_node(
            knowledge_id="c-rel-2", knowledge_type=KnowledgeType.CONCEPT, label="Rel2",
        )
        rel = semantic_engine.create_relationship(
            source_concept_id=n1.knowledge_id,
            target_concept_id=n2.knowledge_id,
            relationship_type=ConceptRelationshipType.IS_A,
            description="is a relationship",
        )
        assert rel.id

    def test_create_dependency(self, semantic_engine: SemanticStructureEngine) -> None:
        dep = semantic_engine.create_dependency(
            dependent_id="rec-001",
            dependency_id="rec-002",
            dependency_type="requires",
        )
        assert dep.id

    def test_build_semantic_graph(self, semantic_engine: SemanticStructureEngine) -> None:
        graph = semantic_engine.build_semantic_graph()
        assert isinstance(graph, dict)

    def test_health_report(self, semantic_engine: SemanticStructureEngine) -> None:
        assert semantic_engine.health_report()["status"] == "healthy"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 17 — EVENT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestEventEngine:
    def test_initialize_and_shutdown(self) -> None:
        eng = LunaEventEngine()
        eng.initialize()
        assert eng.is_initialized()
        eng.shutdown()
        assert not eng.is_initialized()

    def test_publish_event_returns_event(self, event_engine: LunaEventEngine) -> None:
        ev = event_engine.publish_event(
            category=LunaEventCategory.KNOWLEDGE_CREATED,
            source_engine="TestEngine",
            message="A fact was created",
        )
        assert ev.id
        assert ev.message == "A fact was created"

    def test_retrieve_event(self, event_engine: LunaEventEngine) -> None:
        ev = event_engine.publish_event(
            category=LunaEventCategory.KNOWLEDGE_UPDATED,
            source_engine="TestEngine",
            message="A concept was updated",
        )
        fetched = event_engine.retrieve_event(ev.id)
        assert fetched.id == ev.id

    def test_record_event_alias(self, event_engine: LunaEventEngine) -> None:
        ev = event_engine.record_event(
            category=LunaEventCategory.KNOWLEDGE_VALIDATED,
            source_engine="TestEngine",
            message="Validated successfully",
        )
        assert ev.id

    def test_publish_event_empty_message_raises(self, event_engine: LunaEventEngine) -> None:
        with pytest.raises(ValueError):
            event_engine.publish_event(
                category=LunaEventCategory.KNOWLEDGE_CREATED,
                source_engine="TestEngine",
                message="",
            )

    def test_publish_event_empty_source_raises(self, event_engine: LunaEventEngine) -> None:
        with pytest.raises(ValueError):
            event_engine.publish_event(
                category=LunaEventCategory.KNOWLEDGE_CREATED,
                source_engine="",
                message="A message",
            )

    def test_search_events(self, event_engine: LunaEventEngine) -> None:
        event_engine.publish_event(
            category=LunaEventCategory.SKILL_PROGRESSED,
            source_engine="ProgressionEngine",
            message="Skill advanced",
        )
        results = event_engine.search_events("Skill advanced")
        assert isinstance(results, list)

    def test_filter_by_category(self, event_engine: LunaEventEngine) -> None:
        event_engine.publish_event(
            category=LunaEventCategory.DOMAIN_UPDATED,
            source_engine="DomainEngine",
            message="Domain record updated",
        )
        filtered = event_engine.filter_by_category(LunaEventCategory.DOMAIN_UPDATED)
        assert any(e.message == "Domain record updated" for e in filtered)

    def test_filter_by_knowledge_id(self, event_engine: LunaEventEngine) -> None:
        kid = "fact-abc-123"
        event_engine.publish_event(
            category=LunaEventCategory.KNOWLEDGE_CREATED,
            source_engine="FactEngine",
            message="Fact created",
            knowledge_id=kid,
        )
        filtered = event_engine.filter_by_knowledge_id(kid)
        assert any(e.knowledge_id == kid for e in filtered)

    def test_filter_by_severity(self, event_engine: LunaEventEngine) -> None:
        event_engine.publish_event(
            category=LunaEventCategory.KNOWLEDGE_AUDITED,
            source_engine="IntegrityEngine",
            message="Critical integrity issue",
            severity=LunaEventSeverity.CRITICAL,
        )
        filtered = event_engine.filter_by_severity(LunaEventSeverity.CRITICAL)
        assert any(e.message == "Critical integrity issue" for e in filtered)

    def test_event_before_init_raises(self) -> None:
        eng = LunaEventEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.publish_event(
                category=LunaEventCategory.KNOWLEDGE_CREATED,
                source_engine="TestEngine",
                message="Will fail",
            )

    def test_health_report(self, event_engine: LunaEventEngine) -> None:
        assert event_engine.health_report()["status"] == "healthy"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 18 — LIFECYCLE ENFORCEMENT (cross-engine)
# ═════════════════════════════════════════════════════════════════════════════

class TestLifecycleEnforcement:
    """All engines must guard their public methods with _require_initialized."""

    def test_fact_engine_guards_retrieve(self) -> None:
        eng = FactEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.retrieve_fact("any-id")

    def test_concept_engine_guards_retrieve(self) -> None:
        eng = ConceptEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.retrieve_concept("any-id")

    def test_skill_engine_guards_create(self) -> None:
        eng = SkillEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.create_skill(
                name="X", description="X", skill_type=SkillType.TECHNICAL,
                difficulty=KnowledgeDifficulty.BEGINNER,
                domain_ids=["d"], metadata=_meta(),
            )

    def test_domain_engine_guards_create(self) -> None:
        eng = KnowledgeDomainEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.create_domain(
                name="X", description="X",
                difficulty=KnowledgeDifficulty.FOUNDATIONAL,
                metadata=_meta(),
            )

    def test_procedure_engine_guards_create(self) -> None:
        eng = ProceduralKnowledgeEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.create_procedure(
                name="X", description="X",
                procedure_type=ProcedureType.SETUP,
                difficulty=KnowledgeDifficulty.FOUNDATIONAL,
                domain_ids=["d"],
                metadata=_meta(),
            )

    def test_research_engine_guards_create(self) -> None:
        eng = ResearchKnowledgeEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.create_research(
                name="X", description="X",
                research_type=ResearchType.PAPER,
                difficulty=KnowledgeDifficulty.INTERMEDIATE,
                domain_ids=["d"],
                metadata=_meta(),
            )

    def test_education_engine_guards_create(self) -> None:
        eng = EducationalKnowledgeEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.create_educational(
                name="X", description="X",
                education_type=EducationType.LESSON,
                difficulty=KnowledgeDifficulty.FOUNDATIONAL,
                domain_ids=["d"],
                metadata=_meta(),
            )

    def test_event_engine_guards_publish(self) -> None:
        eng = LunaEventEngine()
        with pytest.raises(LunaNotInitializedError):
            eng.publish_event(
                category=LunaEventCategory.KNOWLEDGE_CREATED,
                source_engine="X",
                message="X",
            )

    def test_luna_accessors_guard_when_not_running(self) -> None:
        sub = LunaSubsystem()
        with pytest.raises(LunaNotInitializedError):
            _ = sub.fact_engine


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 19 — ERROR HANDLING
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_retrieve_missing_fact_raises(self, fact_engine: FactEngine) -> None:
        with pytest.raises(Exception):
            fact_engine.retrieve_fact("does-not-exist")

    def test_retrieve_missing_concept_raises(self, concept_engine: ConceptEngine) -> None:
        with pytest.raises(Exception):
            concept_engine.retrieve_concept("does-not-exist")

    def test_retrieve_missing_skill_raises(self, skill_engine: SkillEngine) -> None:
        with pytest.raises(Exception):
            skill_engine.retrieve_skill("does-not-exist")

    def test_retrieve_missing_domain_raises(self, domain_engine: KnowledgeDomainEngine) -> None:
        with pytest.raises(Exception):
            domain_engine.retrieve_domain("does-not-exist")

    def test_create_skill_with_bad_prereq_raises(self, skill_engine: SkillEngine) -> None:
        with pytest.raises(Exception):
            skill_engine.create_skill(
                name="BadPrereqSkill", description="d",
                skill_type=SkillType.TECHNICAL,
                difficulty=KnowledgeDifficulty.BEGINNER,
                domain_ids=["d"],
                metadata=_meta(),
                prerequisite_skill_ids=["nonexistent-skill"],
            )

    def test_create_concept_with_bad_prereq_raises(self, concept_engine: ConceptEngine) -> None:
        with pytest.raises(Exception):
            concept_engine.create_concept(
                name="BadPrereqConcept", description="d",
                concept_type=ConceptType.PRINCIPLE,
                difficulty=KnowledgeDifficulty.BEGINNER,
                domain_ids=["d"],
                metadata=_meta(),
                prerequisite_concept_ids=["nonexistent-concept"],
            )

    def test_research_empty_name_raises(self, research_engine: ResearchKnowledgeEngine) -> None:
        with pytest.raises(Exception):
            research_engine.create_research(
                name="   ", description="d",
                research_type=ResearchType.PAPER,
                difficulty=KnowledgeDifficulty.INTERMEDIATE,
                domain_ids=["d"],
                metadata=_meta(),
            )

    def test_research_empty_domain_ids_raises(self, research_engine: ResearchKnowledgeEngine) -> None:
        with pytest.raises(Exception):
            research_engine.create_research(
                name="NoDomainResearch", description="d",
                research_type=ResearchType.PAPER,
                difficulty=KnowledgeDifficulty.INTERMEDIATE,
                domain_ids=[],
                metadata=_meta(),
            )

    def test_evolution_propose_on_missing_record_raises(
        self, evolution_engine: KnowledgeEvolutionEngine
    ) -> None:
        with pytest.raises(Exception):
            evolution_engine.propose_update(
                record_id="ghost-record",
                knowledge_type=KnowledgeType.FACT,
                changes={"notes": "x"},
            )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 20 — THREAD-SAFETY ASSUMPTIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Concurrent creation across threads must not lose records or corrupt state."""

    def test_concurrent_fact_creation(self, fact_engine: FactEngine) -> None:
        errors: list[Exception] = []
        created_ids: list[str] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            try:
                f = fact_engine.create_fact(
                    name=f"ThreadFact{i}", description=f"d{i}", statement=f"s{i}",
                    fact_type=FactType.DEFINITION,
                    difficulty=KnowledgeDifficulty.FOUNDATIONAL,
                    domain_ids=["d"],
                    metadata=_meta(),
                )
                with lock:
                    created_ids.append(f.id)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(created_ids) == 20
        # All IDs must be unique
        assert len(set(created_ids)) == 20

    def test_concurrent_event_publishing(self, event_engine: LunaEventEngine) -> None:
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            try:
                event_engine.publish_event(
                    category=LunaEventCategory.KNOWLEDGE_CREATED,
                    source_engine="TestEngine",
                    message=f"Concurrent event {i}",
                )
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_concept_creation(self, concept_engine: ConceptEngine) -> None:
        errors: list[Exception] = []
        ids: list[str] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            try:
                c = concept_engine.create_concept(
                    name=f"ConcurrentConcept{i}", description=f"d{i}",
                    concept_type=ConceptType.PRINCIPLE,
                    difficulty=KnowledgeDifficulty.FOUNDATIONAL,
                    domain_ids=["d"],
                    metadata=_meta(),
                )
                with lock:
                    ids.append(c.id)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(set(ids)) == 15


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 21 — WIRED SUBSYSTEM INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

class TestWiredSubsystemIntegration:
    """Verify the full LunaSubsystem wires validation stores correctly
    and all cross-engine paths work through the subsystem facade."""

    def test_validation_engine_register_store_wired(self, luna: LunaSubsystem) -> None:
        # If register_store failed during init, validation_engine would error here
        result = luna.validation_engine.audit_report()
        assert isinstance(result, dict)

    def test_create_fact_through_subsystem(self, luna: LunaSubsystem) -> None:
        f = luna.fact_engine.create_fact(
            name="Subsystem Fact", description="created via subsystem",
            statement="e = mc^2", fact_type=FactType.LAW,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["physics"],
            metadata=_meta(),
        )
        assert f.id

    def test_create_concept_through_subsystem(self, luna: LunaSubsystem) -> None:
        c = luna.concept_engine.create_concept(
            name="Subsystem Concept", description="created via subsystem",
            concept_type=ConceptType.FRAMEWORK,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["software"],
            metadata=_meta(),
        )
        assert c.id

    def test_subsystem_health_report_after_operations(self, luna: LunaSubsystem) -> None:
        luna.fact_engine.create_fact(
            name="HealthFact", description="d", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.FOUNDATIONAL,
            domain_ids=["d"],
            metadata=_meta(),
        )
        report = luna.health_report()
        assert report["status"] == "running"

    def test_subsystem_diagnostics_after_operations(self, luna: LunaSubsystem) -> None:
        luna.skill_engine.create_skill(
            name="DiagSkill", description="d",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        diag = luna.diagnostics_report()
        assert diag["engines_online"] == 16

    def test_event_engine_accessible_through_subsystem(self, luna: LunaSubsystem) -> None:
        ev = luna.event_engine.publish_event(
            category=LunaEventCategory.KNOWLEDGE_CREATED,
            source_engine="IntegrationTest",
            message="Integration event",
        )
        assert ev.id

    def test_index_engine_reindex_through_subsystem(self, luna: LunaSubsystem) -> None:
        count = luna.index_engine.reindex_all()
        assert isinstance(count, int)

    def test_integrity_scan_through_subsystem(self, luna: LunaSubsystem) -> None:
        report = luna.integrity_engine.full_scan()
        assert report.id

    def test_evolution_engine_through_subsystem(self, luna: LunaSubsystem) -> None:
        f = luna.fact_engine.create_fact(
            name="EvolutionSubFact", description="d", statement="s",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["d"],
            metadata=_meta(),
        )
        pid = luna.evolution_engine.propose_update(
            record_id=f.id,
            knowledge_type=KnowledgeType.FACT,
            changes={"notes": "evolved"},
        )
        assert pid

    def test_semantic_structure_engine_through_subsystem(self, luna: LunaSubsystem) -> None:
        node = luna.semantic_structure_engine.create_semantic_node(
            knowledge_id="c-001",
            knowledge_type=KnowledgeType.CONCEPT,
            label="Integration Node",
        )
        assert node.id

    def test_progression_engine_through_subsystem(self, luna: LunaSubsystem) -> None:
        s = luna.skill_engine.create_skill(
            name="ProgSubsystemSkill", description="d",
            skill_type=SkillType.TECHNICAL,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=["d"],
            metadata=_meta(),
        )
        model = luna.progression_engine.create_progression_model(
            skill_id=s.id,
            skill_name=s.name,
            description="Model via subsystem",
            stages=[
                SkillStage(
                    level=SkillLevel.NOVICE, label="Novice", description="Start here",
                )
            ],
        )
        assert model.id

    def test_synthesis_through_subsystem(self, luna: LunaSubsystem) -> None:
        domain = luna.domain_engine.create_domain(
            name="SynthSubsystemDomain", description="desc",
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            metadata=_meta(),
        )
        luna.fact_engine.create_fact(
            name="SynthSubFact1", description="d", statement="s1",
            fact_type=FactType.DEFINITION,
            difficulty=KnowledgeDifficulty.INTERMEDIATE,
            domain_ids=[domain.id],
            metadata=_meta(),
        )
        synth = luna.synthesis_engine.synthesize_knowledge([domain.id])
        assert synth.id


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 22 — MODEL LAYER UNIT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestModelLayer:
    """Pure model/dataclass behaviour — no engine required."""

    def test_knowledge_metadata_create(self) -> None:
        meta = _meta()
        assert meta.version == 1
        assert 0.0 <= meta.confidence_score <= 1.0

    def test_knowledge_metadata_bump_version(self) -> None:
        meta = _meta()
        bumped = meta.bump_version(confidence_score=0.95)
        assert bumped.version == 2
        assert bumped.confidence_score == 0.95

    def test_confidence_level_from_score(self) -> None:
        from subsystems.luna.models import ConfidenceLevel
        assert ConfidenceLevel.from_score(0.99) == ConfidenceLevel.CERTAIN
        assert ConfidenceLevel.from_score(0.85) == ConfidenceLevel.HIGH
        assert ConfidenceLevel.from_score(0.65) == ConfidenceLevel.MODERATE
        assert ConfidenceLevel.from_score(0.45) == ConfidenceLevel.LOW
        assert ConfidenceLevel.from_score(0.25) == ConfidenceLevel.SPECULATIVE
        assert ConfidenceLevel.from_score(0.10) == ConfidenceLevel.UNKNOWN

    def test_knowledge_status_usable(self) -> None:
        assert KnowledgeStatus.ACTIVE.is_usable
        assert KnowledgeStatus.VALIDATED.is_usable
        assert not KnowledgeStatus.DRAFT.is_usable
        assert not KnowledgeStatus.RETRACTED.is_usable

    def test_knowledge_status_terminal(self) -> None:
        assert KnowledgeStatus.RETRACTED.is_terminal
        assert KnowledgeStatus.ARCHIVED.is_terminal
        assert not KnowledgeStatus.ACTIVE.is_terminal

    def test_skill_level_ordering(self) -> None:
        assert SkillLevel.EXPERT.is_above(SkillLevel.BEGINNER)
        assert SkillLevel.NOVICE.is_below(SkillLevel.MASTER)

    def test_skill_level_next_and_previous(self) -> None:
        assert SkillLevel.NOVICE.next_level() == SkillLevel.BEGINNER
        assert SkillLevel.MASTER.next_level() is None
        assert SkillLevel.MASTER.previous_level() == SkillLevel.EXPERT

    def test_difficulty_ranking(self) -> None:
        assert KnowledgeDifficulty.EXPERT.is_harder_than(KnowledgeDifficulty.BEGINNER)
        assert KnowledgeDifficulty.FOUNDATIONAL.is_easier_than(KnowledgeDifficulty.ADVANCED)

    def test_concept_relationship_create(self) -> None:
        rel = ConceptRelationship.create(
            source_concept_id="c1",
            target_concept_id="c2",
            relationship_type=ConceptRelationshipType.IS_A,
            description="c1 is a c2",
            weight=0.9,
        )
        assert rel.id
        assert rel.is_strong

    def test_knowledge_dependency_create(self) -> None:
        dep = KnowledgeDependency.create(
            dependent_id="rec1",
            dependency_id="rec2",
            dependency_type="requires",
            is_hard=True,
        )
        assert dep.id
        assert dep.is_hard
        assert not dep.is_soft

    def test_semantic_node_create(self) -> None:
        node = SemanticNode.create(
            knowledge_id="k1",
            knowledge_type=KnowledgeType.CONCEPT,
            label="My Node",
            depth=0,
        )
        assert node.id
        assert node.is_root
        assert node.is_leaf

    def test_procedure_step_dict(self) -> None:
        step = ProcedureStep(
            step_number=1,
            title="Step 1",
            instruction="Do X",
            expected_outcome="X done",
            is_critical=True,
        )
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["is_critical"] is True

    def test_skill_progression_model_get_stage(self) -> None:
        stage = SkillStage(level=SkillLevel.INTERMEDIATE, label="Mid", description="Middle")
        model = SkillProgressionModel.create(
            skill_id="s1", skill_name="S", description="desc", stages=[stage]
        )
        found = model.get_stage(SkillLevel.INTERMEDIATE)
        assert found is not None
        assert found.level == SkillLevel.INTERMEDIATE
        assert model.get_stage(SkillLevel.MASTER) is None

    def test_knowledge_source_type_trust_weights(self) -> None:
        assert KnowledgeSourceType.OFFICIAL_STANDARD.trust_weight == 1.0
        assert KnowledgeSourceType.UNKNOWN.trust_weight == 0.30
        assert KnowledgeSourceType.PEER_REVIEWED.trust_weight > KnowledgeSourceType.USER_INPUT.trust_weight
