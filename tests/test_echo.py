# tests/test_echo.py
"""
Comprehensive pytest test suite for the ECHO Episodic Memory Core.

Coverage
--------
* Domain model validation and serialisation round-trips
* Exception hierarchy
* ExperienceEngine   — full lifecycle, significance integration, duplicates
* SignificanceEngine — scoring, classification, thresholds, custom rules
* EventEngine        — recording, deletion, filtering, importance gating
* ConversationEngine — creation, update, participant indexing, retrieval
* SessionEngine      — open/close/reopen, membership, history
* AchievementEngine  — creation, lookup, metrics
* FailureAnalysisEngine — recording, lesson extraction, analysis
* ExperienceRetrievalEngine — semantic, tag, time-range, similarity, recall
* MemoryConsolidationEngine — eligibility, promotion, pruning
* ReflectionEngine   — generation, insights, improvement suggestions
* ContextReconstructionEngine — timeline, session, graph, related memories
* MemoryIntegrityEngine — duplicates, broken refs, audits
* EpisodicIndexEngine   — indexing, rebuild, lookups
* PatternExtractionEngine — discovery, confidence, reporting
* PersonalHistoryEngine — timeline, chapters, narrative, growth trajectory
* EchoSubsystem      — init/shutdown sequence, accessors, health, diagnostics
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on the path when running from tests/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Top-level imports from the ECHO public API
# ---------------------------------------------------------------------------

from subsystems.echo import (
    # Subsystem
    EchoSubsystem,
    EchoSubsystemStatus,
    # Engines
    SignificanceEngine,
    ExperienceEngine,
    ExperienceRetrievalEngine,
    MemoryConsolidationEngine,
    MemoryIntegrityEngine,
    EpisodicIndexEngine,
    ReflectionEngine,
    ContextReconstructionEngine,
    PatternExtractionEngine,
    PersonalHistoryEngine,
    EventEngine,
    ConversationEngine,
    SessionEngine,
    AchievementEngine,
    FailureAnalysisEngine,
    # Models
    Experience,
    ExperienceType,
    ExperienceImportance,
    ExperienceMetadata,
    MemoryTag,
    EventRecord,
    AchievementRecord,
    FailureRecord,
    ObservationRecord,
    # Conversation / session domain types
    ConversationRecord,
    SessionRecord,
    SessionState,
    SessionHistoryEntry,
    # Serialisation
    experience_to_dict,
    experience_from_dict,
    event_record_to_dict,
    event_record_from_dict,
    achievement_record_to_dict,
    achievement_record_from_dict,
    failure_record_to_dict,
    failure_record_from_dict,
    observation_record_to_dict,
    observation_record_from_dict,
    memory_tag_to_dict,
    memory_tag_from_dict,
    experience_metadata_to_dict,
    experience_metadata_from_dict,
    # Exceptions
    EchoError,
    EchoNotInitializedError,
    EchoBoundaryViolationError,
    ExperienceError,
    ExperienceNotFoundError,
    ExperienceValidationError,
    ExperienceDuplicateError,
    ExperienceStorageError,
    BelowSignificanceThresholdError,
    SignificanceError,
    SignificanceScoringError,
    EventError,
    EventNotFoundError,
    EventValidationError,
    AchievementError,
    AchievementNotFoundError,
    AchievementValidationError,
    FailureError,
    FailureNotFoundError,
    FailureValidationError,
    ObservationError,
    ObservationNotFoundError,
    ObservationValidationError,
    MemoryIntegrityError,
    DuplicateExperienceError,
    BrokenReferenceError,
    MemoryCorruptionError,
    ConversationError,
    ConversationNotFoundError,
    ConversationValidationError,
    ConversationDuplicateError,
    SessionError,
    SessionNotFoundError,
    SessionValidationError,
    SessionDuplicateError,
    SessionStateError,
)

from subsystems.echo.consolidation import ConsolidationPolicy, ConsolidationReport
from subsystems.echo.significance import SignificanceScore, SignificanceThresholds


# ===========================================================================
# Helpers
# ===========================================================================

_UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(_UTC)


def _ago(days: int = 0, hours: int = 0) -> datetime:
    return _now() - timedelta(days=days, hours=hours)


def _make_tag(name: str, category: str = "topic") -> MemoryTag:
    return MemoryTag(name=name, category=category)


# ===========================================================================
# Fixtures — individual engines
# ===========================================================================


@pytest.fixture()
def sig_engine() -> SignificanceEngine:
    """Initialised SignificanceEngine, shut down after the test."""
    engine = SignificanceEngine()
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def exp_engine(sig_engine: SignificanceEngine) -> ExperienceEngine:
    engine = ExperienceEngine(significance_engine=sig_engine)
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def retrieval_engine(exp_engine: ExperienceEngine) -> ExperienceRetrievalEngine:
    engine = ExperienceRetrievalEngine(experience_engine=exp_engine)
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def consolidation_engine(
    exp_engine: ExperienceEngine,
    sig_engine: SignificanceEngine,
) -> MemoryConsolidationEngine:
    engine = MemoryConsolidationEngine(
        experience_engine=exp_engine,
        significance_engine=sig_engine,
    )
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def integrity_engine(
    exp_engine: ExperienceEngine,
    retrieval_engine: ExperienceRetrievalEngine,
) -> MemoryIntegrityEngine:
    engine = MemoryIntegrityEngine(
        experience_engine=exp_engine,
        retrieval_engine=retrieval_engine,
    )
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def episodic_index(
    exp_engine: ExperienceEngine,
    retrieval_engine: ExperienceRetrievalEngine,
) -> EpisodicIndexEngine:
    engine = EpisodicIndexEngine(
        experience_engine=exp_engine,
        retrieval_engine=retrieval_engine,
    )
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def reflection_engine(
    exp_engine: ExperienceEngine,
    sig_engine: SignificanceEngine,
) -> ReflectionEngine:
    engine = ReflectionEngine(
        experience_engine=exp_engine,
        significance_engine=sig_engine,
    )
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def context_engine(
    exp_engine: ExperienceEngine,
    retrieval_engine: ExperienceRetrievalEngine,
) -> ContextReconstructionEngine:
    engine = ContextReconstructionEngine(
        experience_engine=exp_engine,
        retrieval_engine=retrieval_engine,
    )
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def pattern_engine(
    exp_engine: ExperienceEngine,
    retrieval_engine: ExperienceRetrievalEngine,
) -> PatternExtractionEngine:
    engine = PatternExtractionEngine(
        experience_engine=exp_engine,
        retrieval_engine=retrieval_engine,
    )
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def personal_history_engine(
    exp_engine: ExperienceEngine,
) -> PersonalHistoryEngine:
    engine = PersonalHistoryEngine(experience_engine=exp_engine)
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def event_engine() -> EventEngine:
    engine = EventEngine()
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def achievement_engine() -> AchievementEngine:
    engine = AchievementEngine()
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def failure_engine() -> FailureAnalysisEngine:
    engine = FailureAnalysisEngine()
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def conversation_engine(
    exp_engine: ExperienceEngine,
    sig_engine: SignificanceEngine,
) -> ConversationEngine:
    engine = ConversationEngine(
        experience_engine=exp_engine,
        significance_engine=sig_engine,
    )
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def session_engine(
    exp_engine: ExperienceEngine,
    sig_engine: SignificanceEngine,
) -> SessionEngine:
    engine = SessionEngine(
        experience_engine=exp_engine,
        significance_engine=sig_engine,
    )
    engine.initialize()
    yield engine
    engine.shutdown()


@pytest.fixture()
def echo() -> EchoSubsystem:
    """Fully initialised EchoSubsystem."""
    subsystem = EchoSubsystem()
    subsystem.initialize()
    yield subsystem
    subsystem.shutdown()


# ---------------------------------------------------------------------------
# Convenience factory: create a HIGH importance experience
# ---------------------------------------------------------------------------


def _create_exp(
    engine: ExperienceEngine,
    *,
    title: str = "Test Experience",
    exp_type: ExperienceType = ExperienceType.EVENT,
    importance: ExperienceImportance = ExperienceImportance.HIGH,
    description: str = "An important test event happened here.",
    outcome: str = "Outcome recorded.",
    tags: list[MemoryTag] | None = None,
    occurred_at: datetime | None = None,
) -> Experience:
    return engine.create_experience(
        title=title,
        experience_type=exp_type,
        importance=importance,
        description=description,
        outcome=outcome,
        tags=tags or [],
        occurred_at=occurred_at or _now(),
        force=True,
    )


# ===========================================================================
# 1. Foundation — Domain Models
# ===========================================================================


class TestMemoryTag:
    def test_valid_creation(self) -> None:
        tag = MemoryTag(name="polaris", category="project")
        assert tag.name == "polaris"
        assert tag.category == "project"

    def test_default_category(self) -> None:
        tag = MemoryTag(name="architecture")
        assert tag.category == "custom"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryTag(name="")

    def test_whitespace_name_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryTag(name="   ")

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryTag(name="test", category="invalid_category")

    def test_frozen(self) -> None:
        tag = MemoryTag(name="immutable")
        with pytest.raises((AttributeError, TypeError)):
            tag.name = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert MemoryTag(name="x", category="topic") == MemoryTag(name="x", category="topic")

    def test_all_valid_categories(self) -> None:
        for cat in ("project", "person", "topic", "goal", "custom"):
            MemoryTag(name="tag", category=cat)


class TestExperienceMetadata:
    def test_defaults(self) -> None:
        meta = ExperienceMetadata()
        assert meta.significance_score == 0.0
        assert meta.consolidated is False
        assert meta.retrieval_count == 0
        assert meta.session_id is None

    def test_invalid_score_raises(self) -> None:
        with pytest.raises(ValueError):
            ExperienceMetadata(significance_score=1.5)

    def test_negative_retrieval_count_raises(self) -> None:
        with pytest.raises(ValueError):
            ExperienceMetadata(retrieval_count=-1)

    def test_record_retrieval(self) -> None:
        meta = ExperienceMetadata()
        meta.record_retrieval()
        assert meta.retrieval_count == 1
        assert meta.last_retrieved_at is not None

    def test_mark_consolidated(self) -> None:
        meta = ExperienceMetadata()
        meta.mark_consolidated()
        assert meta.consolidated is True
        assert meta.consolidation_at is not None


class TestExperience:
    def test_creation(self) -> None:
        exp = Experience(
            title="Test",
            experience_type=ExperienceType.EVENT,
            importance=ExperienceImportance.HIGH,
        )
        assert exp.title == "Test"
        assert exp.experience_id  # UUID generated

    def test_empty_title_raises(self) -> None:
        with pytest.raises(ValueError):
            Experience(
                title="",
                experience_type=ExperienceType.EVENT,
                importance=ExperienceImportance.LOW,
            )

    def test_is_significant(self) -> None:
        for imp in (ExperienceImportance.MEDIUM, ExperienceImportance.HIGH, ExperienceImportance.CRITICAL):
            exp = Experience(title="T", experience_type=ExperienceType.EVENT, importance=imp)
            assert exp.is_significant()
        low = Experience(title="T", experience_type=ExperienceType.EVENT, importance=ExperienceImportance.LOW)
        assert not low.is_significant()

    def test_is_permanent(self) -> None:
        exp = Experience(title="T", experience_type=ExperienceType.ACHIEVEMENT, importance=ExperienceImportance.CRITICAL)
        assert exp.is_permanent()

    def test_add_tag_idempotent(self) -> None:
        exp = Experience(title="T", experience_type=ExperienceType.EVENT, importance=ExperienceImportance.HIGH)
        tag = MemoryTag(name="polaris")
        exp.add_tag(tag)
        exp.add_tag(tag)
        assert len(exp.tags) == 1

    def test_tag_names(self) -> None:
        exp = Experience(title="T", experience_type=ExperienceType.EVENT, importance=ExperienceImportance.HIGH)
        exp.add_tag(MemoryTag(name="alpha"))
        exp.add_tag(MemoryTag(name="beta"))
        assert set(exp.tag_names()) == {"alpha", "beta"}


class TestDomainRecords:
    def test_event_record_validation(self) -> None:
        with pytest.raises(ValueError):
            EventRecord(event_name="")

    def test_achievement_record_validation(self) -> None:
        with pytest.raises(ValueError):
            AchievementRecord(title="")

    def test_failure_record_validation(self) -> None:
        with pytest.raises(ValueError):
            FailureRecord(title="")

    def test_observation_record_validation(self) -> None:
        with pytest.raises(ValueError):
            ObservationRecord(summary="")

    def test_failure_mark_reflected(self) -> None:
        rec = FailureRecord(title="Missed deadline")
        assert not rec.reflection_generated
        rec.mark_reflected()
        assert rec.reflection_generated

    def test_event_record_defaults(self) -> None:
        rec = EventRecord(event_name="ProjectCreated")
        assert rec.importance == ExperienceImportance.MEDIUM
        assert rec.experience_id is None


# ===========================================================================
# 2. Foundation — Serialisation Round-trips
# ===========================================================================


class TestSerialisation:
    def test_experience_round_trip(self) -> None:
        exp = Experience(
            title="Architecture Freeze",
            experience_type=ExperienceType.ACHIEVEMENT,
            importance=ExperienceImportance.CRITICAL,
            description="Froze the v5 spec.",
            outcome="All contracts locked.",
            tags=[MemoryTag(name="polaris", category="project")],
        )
        data = experience_to_dict(exp)
        restored = experience_from_dict(data)
        assert restored.experience_id == exp.experience_id
        assert restored.title == exp.title
        assert restored.experience_type == exp.experience_type
        assert restored.importance == exp.importance
        assert len(restored.tags) == 1
        assert restored.tags[0].name == "polaris"

    def test_event_record_round_trip(self) -> None:
        rec = EventRecord(event_name="GoalCompleted", payload={"goal": "ship v1"})
        data = event_record_to_dict(rec)
        restored = event_record_from_dict(data)
        assert restored.event_id == rec.event_id
        assert restored.event_name == rec.event_name
        assert restored.payload == {"goal": "ship v1"}

    def test_achievement_record_round_trip(self) -> None:
        rec = AchievementRecord(
            title="Shipped POLARIS v1",
            domain="software",
            evidence=["https://example.com/v1"],
            importance=ExperienceImportance.CRITICAL,
        )
        data = achievement_record_to_dict(rec)
        restored = achievement_record_from_dict(data)
        assert restored.achievement_id == rec.achievement_id
        assert restored.title == rec.title
        assert restored.importance == ExperienceImportance.CRITICAL

    def test_failure_record_round_trip(self) -> None:
        rec = FailureRecord(
            title="Missed Deadline",
            contributing_factors=["scope creep"],
            lesson="Break work into smaller units.",
        )
        data = failure_record_to_dict(rec)
        restored = failure_record_from_dict(data)
        assert restored.failure_id == rec.failure_id
        assert restored.contributing_factors == ["scope creep"]
        assert restored.lesson == "Break work into smaller units."

    def test_observation_record_round_trip(self) -> None:
        rec = ObservationRecord(
            summary="User productive past midnight",
            domain="personal",
            detail="Three consecutive nights of late work.",
        )
        data = observation_record_to_dict(rec)
        restored = observation_record_from_dict(data)
        assert restored.observation_id == rec.observation_id
        assert restored.summary == rec.summary

    def test_memory_tag_round_trip(self) -> None:
        tag = MemoryTag(name="echo", category="project")
        data = memory_tag_to_dict(tag)
        restored = memory_tag_from_dict(data)
        assert restored.name == "echo"
        assert restored.category == "project"

    def test_experience_metadata_round_trip(self) -> None:
        meta = ExperienceMetadata(source_subsystem="ORION", significance_score=0.75, retrieval_count=3)
        data = experience_metadata_to_dict(meta)
        restored = experience_metadata_from_dict(data)
        assert restored.source_subsystem == "ORION"
        assert restored.significance_score == pytest.approx(0.75)
        assert restored.retrieval_count == 3


# ===========================================================================
# 3. Foundation — Exception Hierarchy
# ===========================================================================


class TestExceptionHierarchy:
    def test_all_echo_errors_inherit_from_echo_error(self) -> None:
        classes = [
            EchoNotInitializedError,
            EchoBoundaryViolationError,
            ExperienceError,
            ExperienceNotFoundError,
            ExperienceValidationError,
            ExperienceDuplicateError,
            ExperienceStorageError,
            SignificanceError,
            SignificanceScoringError,
            BelowSignificanceThresholdError,
            EventError,
            EventNotFoundError,
            EventValidationError,
            AchievementError,
            AchievementNotFoundError,
            AchievementValidationError,
            FailureError,
            FailureNotFoundError,
            FailureValidationError,
            ObservationError,
            ObservationNotFoundError,
            ObservationValidationError,
            MemoryIntegrityError,
            DuplicateExperienceError,
            BrokenReferenceError,
            MemoryCorruptionError,
        ]
        for cls in classes:
            assert issubclass(cls, EchoError), f"{cls.__name__} does not inherit EchoError"

    def test_not_initialized_error_message(self) -> None:
        err = EchoNotInitializedError("test_op")
        assert "test_op" in str(err)
        assert err.operation == "test_op"

    def test_below_significance_threshold_attributes(self) -> None:
        err = BelowSignificanceThresholdError("exp-1", 0.05, 0.15)
        assert err.score == pytest.approx(0.05)
        assert err.threshold == pytest.approx(0.15)
        assert err.experience_id == "exp-1"

    def test_boundary_violation_attributes(self) -> None:
        err = EchoBoundaryViolationError("goal", "ODYSSEY")
        assert err.attempted_type == "goal"
        assert err.correct_subsystem == "ODYSSEY"

    def test_experience_not_found_attributes(self) -> None:
        err = ExperienceNotFoundError("uid-123")
        assert err.experience_id == "uid-123"

    def test_conversation_exceptions_are_exceptions(self) -> None:
        for cls in (ConversationError, ConversationNotFoundError, ConversationValidationError, ConversationDuplicateError):
            assert issubclass(cls, Exception)

    def test_session_exceptions_are_exceptions(self) -> None:
        for cls in (SessionError, SessionNotFoundError, SessionValidationError, SessionDuplicateError, SessionStateError):
            assert issubclass(cls, Exception)


# ===========================================================================
# 4. SignificanceEngine
# ===========================================================================


class TestSignificanceEngine:
    def test_initialize_and_shutdown_idempotent(self) -> None:
        engine = SignificanceEngine()
        engine.initialize()
        engine.initialize()  # no-op
        engine.shutdown()
        engine.shutdown()  # no-op

    def test_score_returns_float_in_range(self, sig_engine: SignificanceEngine) -> None:
        exp = Experience(
            title="Completed a major milestone",
            experience_type=ExperienceType.ACHIEVEMENT,
            importance=ExperienceImportance.HIGH,
            description="Finished the POLARIS architecture specification.",
            outcome="Spec document delivered.",
        )
        score = sig_engine.score(exp)
        assert 0.0 <= score <= 1.0

    def test_score_experience_returns_significance_score_object(self, sig_engine: SignificanceEngine) -> None:
        exp = Experience(
            title="Important architecture decision",
            experience_type=ExperienceType.EVENT,
            importance=ExperienceImportance.HIGH,
            description="Decided to use event-driven design.",
        )
        result = sig_engine.score_experience(exp)
        assert isinstance(result, SignificanceScore)
        assert 0.0 <= result.score <= 1.0

    def test_classify_returns_experience_importance(self, sig_engine: SignificanceEngine) -> None:
        exp = Experience(
            title="Critical system failure resolved",
            experience_type=ExperienceType.FAILURE,
            importance=ExperienceImportance.CRITICAL,
            description="Production system went down for two hours.",
            outcome="System restored; post-mortem completed.",
        )
        importance = sig_engine.classify(exp)
        assert isinstance(importance, ExperienceImportance)

    def test_is_significant_high_importance(self, sig_engine: SignificanceEngine) -> None:
        exp = Experience(
            title="Shipped POLARIS v1 to production",
            experience_type=ExperienceType.ACHIEVEMENT,
            importance=ExperienceImportance.HIGH,
            description="Released version one of the POLARIS system.",
            outcome="System running in production.",
        )
        # HIGH-importance experiences should generally be considered significant
        result = sig_engine.is_significant(exp)
        assert isinstance(result, bool)

    def test_evaluate_returns_significance_result(self, sig_engine: SignificanceEngine) -> None:
        exp = Experience(
            title="Refactored core module",
            experience_type=ExperienceType.EVENT,
            importance=ExperienceImportance.MEDIUM,
            description="Cleaned up subsystem architecture.",
        )
        result = sig_engine.evaluate(exp)
        assert hasattr(result, "score")
        assert hasattr(result, "importance")
        assert hasattr(result, "eligible_for_storage")

    def test_threshold_getter_and_setter(self, sig_engine: SignificanceEngine) -> None:
        original = sig_engine.get_threshold()
        sig_engine.set_threshold(0.5)
        assert sig_engine.get_threshold() == pytest.approx(0.5)
        sig_engine.set_threshold(original)

    def test_invalid_threshold_raises(self, sig_engine: SignificanceEngine) -> None:
        with pytest.raises((ValueError, SignificanceError)):
            sig_engine.set_threshold(1.5)

    def test_is_eligible_for_storage(self, sig_engine: SignificanceEngine) -> None:
        exp = Experience(
            title="Architecture milestone reached",
            experience_type=ExperienceType.ACHIEVEMENT,
            importance=ExperienceImportance.HIGH,
            description="Completed subsystem interface contracts.",
        )
        result = sig_engine.is_eligible_for_storage(exp)
        assert isinstance(result, bool)

    def test_is_promotion_eligible(self, sig_engine: SignificanceEngine) -> None:
        exp = Experience(
            title="Key design decision",
            experience_type=ExperienceType.EVENT,
            importance=ExperienceImportance.HIGH,
            description="Adopted CQRS pattern for ORION.",
        )
        result = sig_engine.is_promotion_eligible(exp)
        assert isinstance(result, bool)

    def test_eligible_for_consolidation(self, sig_engine: SignificanceEngine) -> None:
        exp = Experience(
            title="Semester completed",
            experience_type=ExperienceType.ACHIEVEMENT,
            importance=ExperienceImportance.HIGH,
        )
        result = sig_engine.eligible_for_consolidation(exp)
        assert isinstance(result, bool)

    def test_score_batch(self, sig_engine: SignificanceEngine) -> None:
        exps = [
            Experience(
                title=f"Event {i}",
                experience_type=ExperienceType.EVENT,
                importance=ExperienceImportance.MEDIUM,
            )
            for i in range(3)
        ]
        scores = sig_engine.score_batch(exps)
        assert len(scores) == 3
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_classify_batch(self, sig_engine: SignificanceEngine) -> None:
        exps = [
            Experience(
                title=f"Achievement {i}",
                experience_type=ExperienceType.ACHIEVEMENT,
                importance=ExperienceImportance.HIGH,
            )
            for i in range(2)
        ]
        classes = sig_engine.classify_batch(exps)
        assert len(classes) == 2
        for c in classes:
            assert isinstance(c, ExperienceImportance)

    def test_rank_experiences(self, sig_engine: SignificanceEngine) -> None:
        exps = [
            Experience(title=f"Exp {i}", experience_type=ExperienceType.EVENT, importance=ExperienceImportance.HIGH)
            for i in range(4)
        ]
        ranked = sig_engine.rank_experiences(exps)
        assert len(ranked) == 4

    def test_register_custom_rule(self, sig_engine: SignificanceEngine) -> None:
        def _boost(exp: Experience) -> float:
            return 0.05 if "polaris" in exp.title.lower() else 0.0

        sig_engine.register_rule("polaris_boost", _boost, max_weight=0.05)
        # Verify it runs without error
        exp = Experience(
            title="POLARIS milestone",
            experience_type=ExperienceType.ACHIEVEMENT,
            importance=ExperienceImportance.HIGH,
        )
        score = sig_engine.score(exp)
        assert 0.0 <= score <= 1.0

    def test_register_duplicate_rule_raises(self, sig_engine: SignificanceEngine) -> None:
        sig_engine.register_rule("unique_rule", lambda e: 0.0)
        with pytest.raises(ValueError):
            sig_engine.register_rule("unique_rule", lambda e: 0.0)

    def test_unregister_rule(self, sig_engine: SignificanceEngine) -> None:
        sig_engine.register_rule("temp_rule", lambda e: 0.0)
        removed = sig_engine.unregister_rule("temp_rule")
        assert removed is True

    def test_unregister_nonexistent_rule(self, sig_engine: SignificanceEngine) -> None:
        removed = sig_engine.unregister_rule("does_not_exist")
        assert removed is False

    def test_operations_raise_when_not_initialized(self) -> None:
        engine = SignificanceEngine()
        exp = Experience(title="T", experience_type=ExperienceType.EVENT, importance=ExperienceImportance.LOW)
        with pytest.raises((EchoNotInitializedError, SignificanceError)):
            engine.score(exp)


# ===========================================================================
# 5. ExperienceEngine
# ===========================================================================


class TestExperienceEngine:
    def test_initialize_idempotent(self, sig_engine: SignificanceEngine) -> None:
        engine = ExperienceEngine(significance_engine=sig_engine)
        engine.initialize()
        engine.initialize()
        engine.shutdown()

    def test_shutdown_idempotent(self, sig_engine: SignificanceEngine) -> None:
        engine = ExperienceEngine(significance_engine=sig_engine)
        engine.initialize()
        engine.shutdown()
        engine.shutdown()

    def test_create_experience_returns_experience(self, exp_engine: ExperienceEngine) -> None:
        exp = _create_exp(exp_engine, title="Architecture Review Completed")
        assert isinstance(exp, Experience)
        assert exp.experience_id
        assert exp.title == "Architecture Review Completed"

    def test_create_experience_populates_metadata(self, exp_engine: ExperienceEngine) -> None:
        exp = _create_exp(exp_engine, title="Metadata Validation Test")
        assert exp.metadata.significance_score >= 0.0

    def test_create_experience_stored_and_retrievable(self, exp_engine: ExperienceEngine) -> None:
        exp = _create_exp(exp_engine, title="Retrievable Experience")
        fetched = exp_engine.get_experience(exp.experience_id)
        assert fetched.experience_id == exp.experience_id

    def test_create_experience_empty_title_raises(self, exp_engine: ExperienceEngine) -> None:
        with pytest.raises(ExperienceValidationError):
            exp_engine.create_experience(
                title="",
                experience_type=ExperienceType.EVENT,
                importance=ExperienceImportance.HIGH,
                force=True,
            )

    def test_create_experience_significance_blocks_low_score(self, exp_engine: ExperienceEngine) -> None:
        engine = ExperienceEngine(
            significance_engine=exp_engine._significance_engine,
            min_significance_threshold=0.99,
        )
        engine.initialize()
        try:
            with pytest.raises(BelowSignificanceThresholdError):
                engine.create_experience(
                    title="Trivial non-event",
                    experience_type=ExperienceType.OBSERVATION,
                    importance=ExperienceImportance.LOW,
                    force=False,
                )
        finally:
            engine.shutdown()

    def test_create_experience_force_bypasses_threshold(self, exp_engine: ExperienceEngine) -> None:
        exp = exp_engine.create_experience(
            title="Force stored low-significance event",
            experience_type=ExperienceType.OBSERVATION,
            importance=ExperienceImportance.LOW,
            force=True,
        )
        assert exp_engine.experience_exists(exp.experience_id)

    def test_boundary_violation_raises(self, exp_engine: ExperienceEngine) -> None:
        with pytest.raises(EchoBoundaryViolationError):
            exp_engine.create_experience(
                title="knowledge about Python",
                experience_type=ExperienceType.EVENT,
                importance=ExperienceImportance.HIGH,
                force=True,
            )

    def test_update_experience_title(self, exp_engine: ExperienceEngine) -> None:
        exp = _create_exp(exp_engine, title="Original Title")
        updated = exp_engine.update_experience(exp.experience_id, title="Updated Title")
        assert updated.title == "Updated Title"

    def test_update_experience_tags(self, exp_engine: ExperienceEngine) -> None:
        exp = _create_exp(exp_engine, title="Tag Update Test")
        new_tags = [MemoryTag(name="echo", category="project")]
        updated = exp_engine.update_experience(exp.experience_id, tags=new_tags)
        assert any(t.name == "echo" for t in updated.tags)

    def test_update_nonexistent_experience_raises(self, exp_engine: ExperienceEngine) -> None:
        with pytest.raises(ExperienceNotFoundError):
            exp_engine.update_experience("nonexistent-uuid", title="New Title")

    def test_delete_experience(self, exp_engine: ExperienceEngine) -> None:
        exp = _create_exp(exp_engine, title="To Be Deleted")
        result = exp_engine.delete_experience(exp.experience_id)
        assert result is True
        assert not exp_engine.experience_exists(exp.experience_id)

    def test_delete_nonexistent_returns_false(self, exp_engine: ExperienceEngine) -> None:
        result = exp_engine.delete_experience("nonexistent-uuid")
        assert result is False

    def test_get_nonexistent_experience_raises(self, exp_engine: ExperienceEngine) -> None:
        with pytest.raises(ExperienceNotFoundError):
            exp_engine.get_experience("nonexistent-uuid")

    def test_experience_exists(self, exp_engine: ExperienceEngine) -> None:
        exp = _create_exp(exp_engine, title="Existence Check")
        assert exp_engine.experience_exists(exp.experience_id)
        assert not exp_engine.experience_exists("fake-uuid")

    def test_store_experience_duplicate_raises(self, exp_engine: ExperienceEngine) -> None:
        exp = _create_exp(exp_engine, title="Duplicate Test")
        with pytest.raises(ExperienceDuplicateError):
            exp_engine.store_experience(exp, force=True)

    def test_query_experiences_by_type(self, exp_engine: ExperienceEngine) -> None:
        _create_exp(exp_engine, title="Achievement One", exp_type=ExperienceType.ACHIEVEMENT)
        _create_exp(exp_engine, title="Event One", exp_type=ExperienceType.EVENT)
        results = exp_engine.query_experiences(experience_type=ExperienceType.ACHIEVEMENT)
        assert all(r.experience_type == ExperienceType.ACHIEVEMENT for r in results)

    def test_count_experiences(self, exp_engine: ExperienceEngine) -> None:
        before = exp_engine.count_experiences()
        _create_exp(exp_engine, title="Count Test One")
        _create_exp(exp_engine, title="Count Test Two")
        assert exp_engine.count_experiences() >= before + 2

    def test_get_recent_experiences(self, exp_engine: ExperienceEngine) -> None:
        _create_exp(exp_engine, title="Recent One")
        _create_exp(exp_engine, title="Recent Two")
        recent = exp_engine.get_recent_experiences(limit=5)
        assert len(recent) >= 1

    def test_snapshot_structure(self, exp_engine: ExperienceEngine) -> None:
        _create_exp(exp_engine, title="Snapshot Subject")
        snap = exp_engine.snapshot()
        assert snap["running"] is True
        assert "total" in snap
        assert "by_type" in snap

    def test_operations_raise_before_initialize(self, sig_engine: SignificanceEngine) -> None:
        engine = ExperienceEngine(significance_engine=sig_engine)
        with pytest.raises(EchoNotInitializedError):
            engine.get_experience("uid")

    def test_operations_raise_after_shutdown(self, sig_engine: SignificanceEngine) -> None:
        engine = ExperienceEngine(significance_engine=sig_engine)
        engine.initialize()
        engine.shutdown()
        with pytest.raises(EchoNotInitializedError):
            engine.get_experience("uid")


# ===========================================================================
# 6. EventEngine
# ===========================================================================


class TestEventEngine:
    def test_initialize_shutdown_idempotent(self) -> None:
        engine = EventEngine()
        engine.initialize()
        engine.initialize()
        engine.shutdown()
        engine.shutdown()

    def test_record_event_returns_event_record(self, event_engine: EventEngine) -> None:
        rec = event_engine.record_event("ProjectCreated", payload={"name": "POLARIS"})
        assert isinstance(rec, EventRecord)
        assert rec.event_name == "ProjectCreated"
        assert rec.payload == {"name": "POLARIS"}

    def test_record_event_stored(self, event_engine: EventEngine) -> None:
        rec = event_engine.record_event("GoalCompleted")
        assert event_engine.event_exists(rec.event_id)

    def test_record_event_empty_name_raises(self, event_engine: EventEngine) -> None:
        with pytest.raises(EventValidationError):
            event_engine.record_event("")

    def test_delete_event(self, event_engine: EventEngine) -> None:
        rec = event_engine.record_event("DeleteMe")
        assert event_engine.delete_event(rec.event_id) is True
        assert not event_engine.event_exists(rec.event_id)

    def test_delete_nonexistent_returns_false(self, event_engine: EventEngine) -> None:
        assert event_engine.delete_event("fake-uuid") is False

    def test_delete_critical_event_raises(self, event_engine: EventEngine) -> None:
        rec = event_engine.record_event("PermanentEvent", importance=ExperienceImportance.CRITICAL)
        with pytest.raises(EventError):
            event_engine.delete_event(rec.event_id)

    def test_get_event(self, event_engine: EventEngine) -> None:
        rec = event_engine.record_event("GetMe")
        fetched = event_engine.get_event(rec.event_id)
        assert fetched.event_id == rec.event_id

    def test_get_nonexistent_event_raises(self, event_engine: EventEngine) -> None:
        with pytest.raises(EventNotFoundError):
            event_engine.get_event("fake-uuid")

    def test_query_events_by_name(self, event_engine: EventEngine) -> None:
        event_engine.record_event("TargetEvent")
        event_engine.record_event("OtherEvent")
        results = event_engine.query_events(event_name="TargetEvent")
        assert all(r.event_name == "TargetEvent" for r in results)

    def test_query_events_by_importance(self, event_engine: EventEngine) -> None:
        event_engine.record_event("HighEvent", importance=ExperienceImportance.HIGH)
        event_engine.record_event("LowEvent", importance=ExperienceImportance.LOW)
        results = event_engine.query_events(importance=ExperienceImportance.HIGH)
        assert all(r.importance == ExperienceImportance.HIGH for r in results)

    def test_query_events_min_importance(self, event_engine: EventEngine) -> None:
        event_engine.record_event("Low", importance=ExperienceImportance.LOW)
        event_engine.record_event("High", importance=ExperienceImportance.HIGH)
        results = event_engine.query_events(min_importance=ExperienceImportance.HIGH)
        for r in results:
            assert r.importance.value >= ExperienceImportance.HIGH.value

    def test_query_events_time_range(self, event_engine: EventEngine) -> None:
        past = _ago(days=5)
        future_bound = _now() + timedelta(seconds=1)
        event_engine.record_event("OldEvent", occurred_at=_ago(days=10))
        event_engine.record_event("RecentEvent", occurred_at=_ago(hours=1))
        results = event_engine.query_events(occurred_after=past, occurred_before=future_bound)
        assert all(r.occurred_at >= past for r in results)

    def test_query_events_by_source_subsystem(self, event_engine: EventEngine) -> None:
        event_engine.record_event("OrionEvent", source_subsystem="ORION")
        event_engine.record_event("OdysseyEvent", source_subsystem="ODYSSEY")
        results = event_engine.query_events(source_subsystem="ORION")
        assert all(r.source_subsystem == "ORION" for r in results)

    def test_update_event_payload(self, event_engine: EventEngine) -> None:
        rec = event_engine.record_event("UpdatePayload", payload={"old": "value"})
        updated = event_engine.update_event_payload(rec.event_id, {"new": "value"})
        assert updated.payload == {"new": "value"}

    def test_get_events_for_experience(self, event_engine: EventEngine) -> None:
        import uuid
        exp_id = str(uuid.uuid4())
        event_engine.record_event("LinkedEvent", experience_id=exp_id)
        event_engine.record_event("UnlinkedEvent")
        results = event_engine.get_events_for_experience(exp_id)
        assert len(results) == 1
        assert results[0].experience_id == exp_id

    def test_get_critical_events(self, event_engine: EventEngine) -> None:
        event_engine.record_event("Critical", importance=ExperienceImportance.CRITICAL)
        crits = event_engine.get_critical_events()
        assert all(r.importance == ExperienceImportance.CRITICAL for r in crits)

    def test_count_events(self, event_engine: EventEngine) -> None:
        before = event_engine.count_events()
        event_engine.record_event("CountA")
        event_engine.record_event("CountB")
        assert event_engine.count_events() == before + 2

    def test_snapshot(self, event_engine: EventEngine) -> None:
        event_engine.record_event("SnapEvent")
        snap = event_engine.snapshot()
        assert snap["running"] is True
        assert snap["total"] >= 1

    def test_operations_raise_before_initialize(self) -> None:
        engine = EventEngine()
        with pytest.raises(EchoNotInitializedError):
            engine.record_event("Test")

    def test_min_importance_threshold_blocks_below(self) -> None:
        engine = EventEngine(min_importance=ExperienceImportance.HIGH)
        engine.initialize()
        try:
            with pytest.raises(EventError):
                engine.record_event("LowEvent", importance=ExperienceImportance.LOW)
        finally:
            engine.shutdown()

    def test_min_importance_force_bypasses(self) -> None:
        engine = EventEngine(min_importance=ExperienceImportance.HIGH)
        engine.initialize()
        try:
            rec = engine.record_event("ForcedLow", importance=ExperienceImportance.LOW, force=True)
            assert rec.event_id
        finally:
            engine.shutdown()


# ===========================================================================
# 7. ConversationEngine
# ===========================================================================


class TestConversationEngine:
    def test_create_conversation_returns_record(self, conversation_engine: ConversationEngine) -> None:
        rec = conversation_engine.create_conversation(
            title="Architecture Discussion",
            summary="Discussed ECHO design decisions.",
            participants=["User", "ORION"],
            force=True,
        )
        assert isinstance(rec, ConversationRecord)
        assert rec.title == "Architecture Discussion"
        assert rec.experience_id is not None

    def test_create_conversation_empty_title_raises(self, conversation_engine: ConversationEngine) -> None:
        with pytest.raises(ConversationValidationError):
            conversation_engine.create_conversation(title="", summary="Some summary.", force=True)

    def test_create_conversation_empty_summary_raises(self, conversation_engine: ConversationEngine) -> None:
        with pytest.raises(ConversationValidationError):
            conversation_engine.create_conversation(title="Valid Title", summary="", force=True)

    def test_get_conversation(self, conversation_engine: ConversationEngine) -> None:
        rec = conversation_engine.create_conversation(
            title="Get Test", summary="A test conversation.", force=True
        )
        fetched = conversation_engine.get_conversation(rec.conversation_id)
        assert fetched.conversation_id == rec.conversation_id

    def test_get_nonexistent_raises(self, conversation_engine: ConversationEngine) -> None:
        with pytest.raises(ConversationNotFoundError):
            conversation_engine.get_conversation("nonexistent")

    def test_update_conversation_title(self, conversation_engine: ConversationEngine) -> None:
        rec = conversation_engine.create_conversation(
            title="Original", summary="Test.", force=True
        )
        updated = conversation_engine.update_conversation(rec.conversation_id, title="Updated")
        assert updated.title == "Updated"

    def test_update_participants(self, conversation_engine: ConversationEngine) -> None:
        rec = conversation_engine.create_conversation(
            title="Participant Test", summary="Participants will change.", participants=["Alice"], force=True
        )
        conversation_engine.update_conversation(rec.conversation_id, participants=["Alice", "Bob"])
        fetched = conversation_engine.get_conversation(rec.conversation_id)
        assert "Bob" in fetched.participants

    def test_participant_index_search(self, conversation_engine: ConversationEngine) -> None:
        conversation_engine.create_conversation(
            title="Alice's Talk",
            summary="Alice discussed things.",
            participants=["Alice"],
            force=True,
        )
        conversation_engine.create_conversation(
            title="Bob's Talk",
            summary="Bob discussed things.",
            participants=["Bob"],
            force=True,
        )
        results = conversation_engine.search_conversations(participant="alice")
        assert all(any("alice" in p.lower() for p in r.participants) for r in results)
        assert len(results) >= 1

    def test_search_by_topic(self, conversation_engine: ConversationEngine) -> None:
        conversation_engine.create_conversation(
            title="Design Session",
            summary="Discussed design.",
            topics=["architecture", "api"],
            force=True,
        )
        results = conversation_engine.search_conversations(topic="architecture")
        assert len(results) >= 1

    def test_search_by_importance(self, conversation_engine: ConversationEngine) -> None:
        conversation_engine.create_conversation(
            title="High Importance Conv",
            summary="Critical conversation.",
            importance=ExperienceImportance.HIGH,
            force=True,
        )
        results = conversation_engine.search_conversations(importance=ExperienceImportance.HIGH)
        assert all(r.importance.value >= ExperienceImportance.HIGH.value for r in results)

    def test_delete_conversation(self, conversation_engine: ConversationEngine) -> None:
        rec = conversation_engine.create_conversation(
            title="Delete Me Conv", summary="Will be removed.", force=True
        )
        result = conversation_engine.delete_conversation(rec.conversation_id)
        assert result is True
        with pytest.raises(ConversationNotFoundError):
            conversation_engine.get_conversation(rec.conversation_id)

    def test_delete_nonexistent_returns_false(self, conversation_engine: ConversationEngine) -> None:
        assert conversation_engine.delete_conversation("fake-id") is False

    def test_count(self, conversation_engine: ConversationEngine) -> None:
        before = conversation_engine.count()
        conversation_engine.create_conversation(title="Count A", summary="Summary A.", force=True)
        assert conversation_engine.count() == before + 1

    def test_operations_raise_before_initialize(
        self, exp_engine: ExperienceEngine, sig_engine: SignificanceEngine
    ) -> None:
        engine = ConversationEngine(experience_engine=exp_engine, significance_engine=sig_engine)
        with pytest.raises(EchoNotInitializedError):
            engine.get_conversation("uid")

    def test_link_experience(
        self, conversation_engine: ConversationEngine, exp_engine: ExperienceEngine
    ) -> None:
        rec = conversation_engine.create_conversation(
            title="Link Test", summary="Will link an experience.", force=True
        )
        exp = _create_exp(exp_engine, title="Experience To Link")
        updated = conversation_engine.link_experience(rec.conversation_id, exp.experience_id)
        assert updated.experience_id == exp.experience_id


# ===========================================================================
# 8. SessionEngine
# ===========================================================================


class TestSessionEngine:
    def test_create_session_returns_record(self, session_engine: SessionEngine) -> None:
        rec = session_engine.create_session(
            title="Architecture Review Session",
            goals=["Review all interfaces"],
            force=True,
        )
        assert isinstance(rec, SessionRecord)
        assert rec.state == SessionState.OPEN
        assert rec.experience_id is not None

    def test_create_session_empty_title_raises(self, session_engine: SessionEngine) -> None:
        with pytest.raises(SessionValidationError):
            session_engine.create_session(title="", force=True)

    def test_close_session(self, session_engine: SessionEngine) -> None:
        rec = session_engine.create_session(title="Session To Close", force=True)
        closed = session_engine.close_session(
            rec.session_id,
            outcomes=["All interfaces reviewed."],
        )
        assert closed.state == SessionState.CLOSED
        assert closed.closed_at is not None

    def test_close_already_closed_raises(self, session_engine: SessionEngine) -> None:
        rec = session_engine.create_session(title="Close Twice", force=True)
        session_engine.close_session(rec.session_id)
        with pytest.raises(SessionStateError):
            session_engine.close_session(rec.session_id)

    def test_reopen_session(self, session_engine: SessionEngine) -> None:
        rec = session_engine.create_session(title="Session To Reopen", force=True)
        session_engine.close_session(rec.session_id)
        reopened = session_engine.reopen_session(rec.session_id, reason="Resumed work")
        assert reopened.state == SessionState.OPEN
        assert reopened.closed_at is None

    def test_reopen_open_session_raises(self, session_engine: SessionEngine) -> None:
        rec = session_engine.create_session(title="Still Open", force=True)
        with pytest.raises(SessionStateError):
            session_engine.reopen_session(rec.session_id)

    def test_add_experience_to_session(
        self, session_engine: SessionEngine, exp_engine: ExperienceEngine
    ) -> None:
        session = session_engine.create_session(title="Session With Exp", force=True)
        exp = _create_exp(exp_engine, title="Member Experience")
        updated = session_engine.add_experience(session.session_id, exp.experience_id)
        assert exp.experience_id in updated.experience_ids

    def test_add_experience_to_closed_session_raises(
        self, session_engine: SessionEngine, exp_engine: ExperienceEngine
    ) -> None:
        session = session_engine.create_session(title="Closed Session", force=True)
        session_engine.close_session(session.session_id)
        exp = _create_exp(exp_engine, title="Late Experience")
        with pytest.raises(SessionStateError):
            session_engine.add_experience(session.session_id, exp.experience_id)

    def test_remove_experience_from_session(
        self, session_engine: SessionEngine, exp_engine: ExperienceEngine
    ) -> None:
        session = session_engine.create_session(title="Session Remove Test", force=True)
        exp = _create_exp(exp_engine, title="Removable Experience")
        session_engine.add_experience(session.session_id, exp.experience_id)
        updated = session_engine.remove_experience(session.session_id, exp.experience_id)
        assert exp.experience_id not in updated.experience_ids

    def test_session_history(self, session_engine: SessionEngine) -> None:
        rec = session_engine.create_session(title="History Test Session", force=True)
        session_engine.close_session(rec.session_id)
        session_engine.reopen_session(rec.session_id)
        history = session_engine.get_session_history(rec.session_id)
        event_names = [h.event for h in history]
        assert "created" in event_names
        assert "closed" in event_names
        assert "reopened" in event_names

    def test_get_session(self, session_engine: SessionEngine) -> None:
        rec = session_engine.create_session(title="Get Session", force=True)
        fetched = session_engine.get_session(rec.session_id)
        assert fetched.session_id == rec.session_id

    def test_get_nonexistent_raises(self, session_engine: SessionEngine) -> None:
        with pytest.raises(SessionNotFoundError):
            session_engine.get_session("fake-id")

    def test_search_sessions_by_state(self, session_engine: SessionEngine) -> None:
        session_engine.create_session(title="Open Session", force=True)
        closed_rec = session_engine.create_session(title="Closed Session", force=True)
        session_engine.close_session(closed_rec.session_id)

        open_sessions = session_engine.search_sessions(state=SessionState.OPEN)
        assert all(s.state == SessionState.OPEN for s in open_sessions)

        closed_sessions = session_engine.search_sessions(state=SessionState.CLOSED)
        assert all(s.state == SessionState.CLOSED for s in closed_sessions)

    def test_find_session_for_experience(
        self, session_engine: SessionEngine, exp_engine: ExperienceEngine
    ) -> None:
        session = session_engine.create_session(title="Lookup Session", force=True)
        exp = _create_exp(exp_engine, title="Lookup Experience")
        session_engine.add_experience(session.session_id, exp.experience_id)
        found = session_engine.find_session_for_experience(exp.experience_id)
        assert found is not None
        assert found.session_id == session.session_id

    def test_count_open(self, session_engine: SessionEngine) -> None:
        before = session_engine.count_open()
        session_engine.create_session(title="New Open", force=True)
        assert session_engine.count_open() == before + 1

    def test_operations_raise_before_initialize(
        self, exp_engine: ExperienceEngine, sig_engine: SignificanceEngine
    ) -> None:
        engine = SessionEngine(experience_engine=exp_engine, significance_engine=sig_engine)
        with pytest.raises(EchoNotInitializedError):
            engine.get_session("uid")


# ===========================================================================
# 9. AchievementEngine
# ===========================================================================


class TestAchievementEngine:
    def test_create_achievement(self, achievement_engine: AchievementEngine) -> None:
        rec = achievement_engine.create_achievement(
            title="Shipped POLARIS v1",
            domain="software",
            description="First public release.",
            evidence=["https://example.com/release"],
            importance=ExperienceImportance.CRITICAL,
        )
        assert isinstance(rec, AchievementRecord)
        assert rec.title == "Shipped POLARIS v1"
        assert rec.domain == "software"

    def test_create_achievement_empty_title_raises(self, achievement_engine: AchievementEngine) -> None:
        with pytest.raises(AchievementValidationError):
            achievement_engine.create_achievement(title="")

    def test_get_achievement(self, achievement_engine: AchievementEngine) -> None:
        rec = achievement_engine.create_achievement(title="Get Me")
        fetched = achievement_engine.get_achievement(rec.achievement_id)
        assert fetched.achievement_id == rec.achievement_id

    def test_get_nonexistent_raises(self, achievement_engine: AchievementEngine) -> None:
        with pytest.raises(AchievementNotFoundError):
            achievement_engine.get_achievement("nonexistent")

    def test_delete_achievement(self, achievement_engine: AchievementEngine) -> None:
        rec = achievement_engine.create_achievement(title="Delete Me")
        result = achievement_engine.delete_achievement(rec.achievement_id)
        assert result is True
        with pytest.raises(AchievementNotFoundError):
            achievement_engine.get_achievement(rec.achievement_id)

    def test_search_achievements_by_domain(self, achievement_engine: AchievementEngine) -> None:
        achievement_engine.create_achievement(title="Ach A", domain="academic")
        achievement_engine.create_achievement(title="Ach B", domain="software")
        results = achievement_engine.search_achievements(domain="academic")
        assert all(r.domain == "academic" for r in results)

    def test_search_achievements_by_keyword(self, achievement_engine: AchievementEngine) -> None:
        achievement_engine.create_achievement(title="Polaris Achievement", description="Built it.")
        results = achievement_engine.search_achievements(keyword="polaris")
        assert len(results) >= 1

    def test_search_achievements_min_importance(self, achievement_engine: AchievementEngine) -> None:
        achievement_engine.create_achievement(title="High Ach", importance=ExperienceImportance.HIGH)
        results = achievement_engine.search_achievements(min_importance=ExperienceImportance.HIGH)
        for r in results:
            assert r.importance.value >= ExperienceImportance.HIGH.value

    def test_update_achievement(self, achievement_engine: AchievementEngine) -> None:
        rec = achievement_engine.create_achievement(title="Updatable")
        updated = achievement_engine.update_achievement(rec.achievement_id, description="Updated desc.")
        assert updated.description == "Updated desc."

    def test_link_experience(
        self, achievement_engine: AchievementEngine, exp_engine: ExperienceEngine
    ) -> None:
        ach = achievement_engine.create_achievement(title="Link Me")
        exp = _create_exp(exp_engine, title="Link Target")
        ach_engine = AchievementEngine(experience_engine=exp_engine)
        ach_engine.initialize()
        try:
            rec2 = ach_engine.create_achievement(title="Linked Achievement")
            linked = ach_engine.link_experience(rec2.achievement_id, exp.experience_id)
            assert linked.experience_id == exp.experience_id
        finally:
            ach_engine.shutdown()

    def test_snapshot(self, achievement_engine: AchievementEngine) -> None:
        achievement_engine.create_achievement(title="Snap Ach")
        snap = achievement_engine.snapshot()
        assert "running" in snap
        assert snap["running"] is True

    def test_operations_raise_before_initialize(self) -> None:
        engine = AchievementEngine()
        with pytest.raises(EchoNotInitializedError):
            engine.get_achievement("uid")


# ===========================================================================
# 10. FailureAnalysisEngine
# ===========================================================================


class TestFailureAnalysisEngine:
    def test_record_failure(self, failure_engine: FailureAnalysisEngine) -> None:
        rec = failure_engine.record_failure(
            title="Sprint Goal Missed",
            domain="software",
            description="Sprint goal not reached due to scope creep.",
            contributing_factors=["Unclear requirements", "Underestimated complexity"],
            lesson="Break features into independently deployable units.",
            importance=ExperienceImportance.HIGH,
        )
        assert isinstance(rec, FailureRecord)
        assert rec.title == "Sprint Goal Missed"
        assert len(rec.contributing_factors) == 2

    def test_record_failure_empty_title_raises(self, failure_engine: FailureAnalysisEngine) -> None:
        with pytest.raises(FailureValidationError):
            failure_engine.record_failure(title="")

    def test_get_failure(self, failure_engine: FailureAnalysisEngine) -> None:
        rec = failure_engine.record_failure(title="Get Failure Test")
        fetched = failure_engine.get_failure(rec.failure_id)
        assert fetched.failure_id == rec.failure_id

    def test_get_nonexistent_raises(self, failure_engine: FailureAnalysisEngine) -> None:
        with pytest.raises(FailureNotFoundError):
            failure_engine.get_failure("nonexistent")

    def test_analyze_failure_returns_bundle(self, failure_engine: FailureAnalysisEngine) -> None:
        rec = failure_engine.record_failure(
            title="Analyzable Failure",
            contributing_factors=["Factor A"],
            lesson="Take note.",
        )
        analysis = failure_engine.analyze_failure(rec.failure_id)
        assert analysis["failure_id"] == rec.failure_id
        assert analysis["factor_count"] == 1
        assert analysis["has_lesson"] is True
        assert "age_days" in analysis

    def test_analyze_failure_no_lesson(self, failure_engine: FailureAnalysisEngine) -> None:
        rec = failure_engine.record_failure(title="No Lesson Yet")
        analysis = failure_engine.analyze_failure(rec.failure_id)
        assert analysis["has_lesson"] is False

    def test_store_lesson(self, failure_engine: FailureAnalysisEngine) -> None:
        rec = failure_engine.record_failure(title="Lesson Store Test")
        updated = failure_engine.store_lesson(rec.failure_id, "Never skip retros.")
        assert updated.lesson == "Never skip retros."

    def test_store_lesson_marks_reflected(self, failure_engine: FailureAnalysisEngine) -> None:
        rec = failure_engine.record_failure(title="Mark Reflected Test")
        updated = failure_engine.store_lesson(
            rec.failure_id, "Document scope changes.", mark_reflected=True
        )
        assert updated.reflection_generated is True

    def test_store_empty_lesson_raises(self, failure_engine: FailureAnalysisEngine) -> None:
        rec = failure_engine.record_failure(title="Empty Lesson")
        with pytest.raises(FailureValidationError):
            failure_engine.store_lesson(rec.failure_id, "")

    def test_search_failures_by_domain(self, failure_engine: FailureAnalysisEngine) -> None:
        failure_engine.record_failure(title="Academic Fail", domain="academic")
        failure_engine.record_failure(title="Software Fail", domain="software")
        results = failure_engine.search_failures(domain="academic")
        assert all(r.domain == "academic" for r in results)

    def test_update_failure(self, failure_engine: FailureAnalysisEngine) -> None:
        rec = failure_engine.record_failure(title="Update Failure Test")
        updated = failure_engine.update_failure(rec.failure_id, description="Updated description.")
        assert updated.description == "Updated description."

    def test_snapshot(self, failure_engine: FailureAnalysisEngine) -> None:
        failure_engine.record_failure(title="Snap Failure")
        snap = failure_engine.snapshot()
        assert "running" in snap
        assert snap["running"] is True

    def test_operations_raise_before_initialize(self) -> None:
        engine = FailureAnalysisEngine()
        with pytest.raises(EchoNotInitializedError):
            engine.get_failure("uid")


# ===========================================================================
# 11. ExperienceRetrievalEngine
# ===========================================================================


class TestExperienceRetrievalEngine:
    def test_initialize_shutdown(self, exp_engine: ExperienceEngine) -> None:
        engine = ExperienceRetrievalEngine(experience_engine=exp_engine)
        engine.initialize()
        engine.shutdown()

    def test_semantic_search_returns_results(self, retrieval_engine: ExperienceRetrievalEngine, exp_engine: ExperienceEngine) -> None:
        _create_exp(exp_engine, title="POLARIS architecture design session")
        _create_exp(exp_engine, title="Architecture decisions reviewed")
        results = retrieval_engine.search_semantic("architecture design")
        assert isinstance(results, list)

    def test_semantic_search_empty_query(self, retrieval_engine: ExperienceRetrievalEngine) -> None:
        results = retrieval_engine.search_semantic("")
        assert isinstance(results, list)

    def test_search_by_tags(self, retrieval_engine: ExperienceRetrievalEngine, exp_engine: ExperienceEngine) -> None:
        _create_exp(exp_engine, title="Tagged Experience", tags=[MemoryTag(name="polaris", category="project")])
        _create_exp(exp_engine, title="Untagged Experience")
        results = retrieval_engine.search_by_tags(["polaris"])
        assert any("Tagged" in r.experience.title for r in results)

    def test_find_by_time_range(self, retrieval_engine: ExperienceRetrievalEngine, exp_engine: ExperienceEngine) -> None:
        recent_ts = _ago(hours=1)
        old_ts = _ago(days=30)
        _create_exp(exp_engine, title="Old Experience", occurred_at=old_ts)
        _create_exp(exp_engine, title="Recent Experience", occurred_at=recent_ts)
        after = _ago(days=2)
        before = _now() + timedelta(seconds=1)
        results = retrieval_engine.find_by_time_range(occurred_after=after, occurred_before=before)
        assert all(r.experience.occurred_at >= after for r in results)

    def test_find_similar(self, retrieval_engine: ExperienceRetrievalEngine, exp_engine: ExperienceEngine) -> None:
        exp = _create_exp(exp_engine, title="Similarity Source Experience")
        _create_exp(exp_engine, title="Similar Experience About Source")
        results = retrieval_engine.find_similar(exp.experience_id)
        assert isinstance(results, list)

    def test_recall_context_returns_bundle(self, retrieval_engine: ExperienceRetrievalEngine, exp_engine: ExperienceEngine) -> None:
        _create_exp(exp_engine, title="Context Recall Experience")
        bundle = retrieval_engine.recall_context("architecture review")
        assert "narrative" in bundle
        assert "query" in bundle
        assert "experiences" in bundle

    def test_get_recent_by_importance(self, retrieval_engine: ExperienceRetrievalEngine, exp_engine: ExperienceEngine) -> None:
        _create_exp(exp_engine, title="High Importance One", importance=ExperienceImportance.HIGH)
        results = retrieval_engine.get_recent_by_importance(
            min_importance=ExperienceImportance.HIGH, limit=10
        )
        assert isinstance(results, list)

    def test_snapshot(self, retrieval_engine: ExperienceRetrievalEngine) -> None:
        snap = retrieval_engine.snapshot()
        assert snap["running"] is True

    def test_operations_raise_before_initialize(self, exp_engine: ExperienceEngine) -> None:
        engine = ExperienceRetrievalEngine(experience_engine=exp_engine)
        with pytest.raises(EchoNotInitializedError):
            engine.search_semantic("test")


# ===========================================================================
# 12. MemoryConsolidationEngine
# ===========================================================================


class TestMemoryConsolidationEngine:
    def test_run_consolidation_cycle_returns_report(
        self, consolidation_engine: MemoryConsolidationEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="High Priority Experience", importance=ExperienceImportance.HIGH)
        report = consolidation_engine.run_consolidation_cycle()
        assert isinstance(report, ConsolidationReport)
        assert report.pruned_count >= 0
        assert report.promoted_count >= 0

    def test_get_consolidation_candidates(
        self, consolidation_engine: MemoryConsolidationEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="Candidate Experience", importance=ExperienceImportance.HIGH)
        candidates = consolidation_engine.get_consolidation_candidates()
        assert isinstance(candidates, list)

    def test_get_prune_candidates(
        self, consolidation_engine: MemoryConsolidationEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(
            exp_engine,
            title="Old Low Importance Experience",
            importance=ExperienceImportance.LOW,
            occurred_at=_ago(days=400),
        )
        candidates = consolidation_engine.get_prune_candidates()
        assert isinstance(candidates, list)

    def test_consolidate_experience(
        self, consolidation_engine: MemoryConsolidationEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(exp_engine, title="Force Consolidate Me", importance=ExperienceImportance.HIGH)
        result = consolidation_engine.consolidate_experience(exp.experience_id)
        assert result.metadata.consolidated is True

    def test_get_cycle_history(
        self, consolidation_engine: MemoryConsolidationEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="History Exp")
        consolidation_engine.run_consolidation_cycle()
        history = consolidation_engine.get_cycle_history(limit=5)
        assert len(history) >= 1

    def test_promote_achievements_policy(
        self, exp_engine: ExperienceEngine, sig_engine: SignificanceEngine
    ) -> None:
        policy = ConsolidationPolicy(promote_achievements=True, prune_low=False)
        engine = MemoryConsolidationEngine(
            experience_engine=exp_engine,
            significance_engine=sig_engine,
            policy=policy,
        )
        engine.initialize()
        try:
            ach = _create_exp(exp_engine, title="Promoted Achievement", exp_type=ExperienceType.ACHIEVEMENT)
            report = engine.run_consolidation_cycle()
            assert isinstance(report, ConsolidationReport)
        finally:
            engine.shutdown()

    def test_snapshot(self, consolidation_engine: MemoryConsolidationEngine) -> None:
        snap = consolidation_engine.snapshot()
        assert snap["running"] is True

    def test_policy_validation(self) -> None:
        with pytest.raises(ValueError):
            ConsolidationPolicy(low_retention_days=0)


# ===========================================================================
# 13. ReflectionEngine
# ===========================================================================


class TestReflectionEngine:
    def test_generate_reflection_returns_experience(
        self, reflection_engine: ReflectionEngine, exp_engine: ExperienceEngine
    ) -> None:
        source = _create_exp(
            exp_engine,
            title="Project scope doubled mid-sprint",
            exp_type=ExperienceType.EVENT,
            importance=ExperienceImportance.HIGH,
            description="Requirements changed significantly after kickoff.",
            outcome="Deadline missed.",
        )
        reflection = reflection_engine.generate_reflection(source.experience_id, force=True)
        assert isinstance(reflection, Experience)
        assert reflection.experience_type == ExperienceType.REFLECTION
        assert "Reflection" in reflection.title

    def test_generate_reflection_nonexistent_source_raises(
        self, reflection_engine: ReflectionEngine
    ) -> None:
        with pytest.raises(ExperienceNotFoundError):
            reflection_engine.generate_reflection("nonexistent-id")

    def test_generate_reflection_low_importance_without_force_raises(
        self, reflection_engine: ReflectionEngine, exp_engine: ExperienceEngine
    ) -> None:
        source = _create_exp(
            exp_engine,
            title="Trivial low event",
            importance=ExperienceImportance.LOW,
        )
        with pytest.raises(ExperienceValidationError):
            reflection_engine.generate_reflection(source.experience_id, force=False)

    def test_get_reflection(
        self, reflection_engine: ReflectionEngine, exp_engine: ExperienceEngine
    ) -> None:
        source = _create_exp(exp_engine, title="Reflection Source", importance=ExperienceImportance.HIGH)
        reflection = reflection_engine.generate_reflection(source.experience_id, force=True)
        fetched = reflection_engine.get_reflection(reflection.experience_id)
        assert fetched.experience_id == reflection.experience_id

    def test_generate_insights_returns_list(
        self, reflection_engine: ReflectionEngine, exp_engine: ExperienceEngine
    ) -> None:
        source = _create_exp(exp_engine, title="Insight Source", importance=ExperienceImportance.HIGH)
        reflection_engine.generate_reflection(source.experience_id, force=True)
        insights = reflection_engine.generate_insights(limit=5)
        assert isinstance(insights, list)

    def test_get_stored_insights_empty_initially(self, reflection_engine: ReflectionEngine) -> None:
        insights = reflection_engine.get_stored_insights()
        assert isinstance(insights, list)

    def test_get_improvement_suggestions_returns_list(
        self, reflection_engine: ReflectionEngine
    ) -> None:
        suggestions = reflection_engine.get_improvement_suggestions()
        assert isinstance(suggestions, list)

    def test_search_reflections(
        self, reflection_engine: ReflectionEngine, exp_engine: ExperienceEngine
    ) -> None:
        source = _create_exp(exp_engine, title="Search Reflection Source", importance=ExperienceImportance.HIGH)
        reflection_engine.generate_reflection(source.experience_id, force=True)
        results = reflection_engine.search_reflections(limit=10)
        assert isinstance(results, list)

    def test_snapshot(self, reflection_engine: ReflectionEngine) -> None:
        snap = reflection_engine.snapshot()
        assert snap["running"] is True

    def test_operations_raise_before_initialize(
        self, exp_engine: ExperienceEngine, sig_engine: SignificanceEngine
    ) -> None:
        engine = ReflectionEngine(experience_engine=exp_engine, significance_engine=sig_engine)
        with pytest.raises(EchoNotInitializedError):
            engine.generate_reflection("uid")


# ===========================================================================
# 14. ContextReconstructionEngine
# ===========================================================================


class TestContextReconstructionEngine:
    def test_reconstruct_timeline_returns_object(
        self, context_engine: ContextReconstructionEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="Timeline Entry A", occurred_at=_ago(days=2))
        _create_exp(exp_engine, title="Timeline Entry B", occurred_at=_ago(days=1))
        timeline = context_engine.reconstruct_timeline(
            min_importance=ExperienceImportance.LOW
        )
        assert hasattr(timeline, "entries")

    def test_reconstruct_timeline_filtered_by_type(
        self, context_engine: ContextReconstructionEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="Achievement Timeline", exp_type=ExperienceType.ACHIEVEMENT)
        _create_exp(exp_engine, title="Event Timeline", exp_type=ExperienceType.EVENT)
        timeline = context_engine.reconstruct_timeline(
            experience_types=[ExperienceType.ACHIEVEMENT]
        )
        for entry in timeline.entries:
            assert entry.experience.experience_type == ExperienceType.ACHIEVEMENT

    def test_reconstruct_session_returns_object(
        self,
        context_engine: ContextReconstructionEngine,
        exp_engine: ExperienceEngine,
        session_engine: SessionEngine,
    ) -> None:
        session = session_engine.create_session(title="Reconstruction Session", force=True)
        exp = _create_exp(exp_engine, title="Session Member")
        session_engine.add_experience(session.session_id, exp.experience_id)
        result = context_engine.reconstruct_session(session.experience_id)
        assert hasattr(result, "experience")

    def test_reconstruct_experience_chain(
        self, context_engine: ContextReconstructionEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(exp_engine, title="Chain Origin Experience")
        chain = context_engine.reconstruct_experience_chain(exp.experience_id)
        assert hasattr(chain, "origin")

    def test_reconstruct_related_memories(
        self, context_engine: ContextReconstructionEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(exp_engine, title="Related Memory Anchor")
        result = context_engine.reconstruct_related_memories(exp.experience_id)
        assert hasattr(result, "anchor")

    def test_generate_context_summary(
        self, context_engine: ContextReconstructionEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="Architecture context experience")
        summary = context_engine.generate_context_summary(query="architecture")
        assert hasattr(summary, "query")

    def test_build_context_graph(
        self, context_engine: ContextReconstructionEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="Graph Node A")
        _create_exp(exp_engine, title="Graph Node B")
        graph = context_engine.build_context_graph(
            min_importance=ExperienceImportance.LOW, max_nodes=20
        )
        assert hasattr(graph, "nodes")
        assert hasattr(graph, "edges")

    def test_snapshot(self, context_engine: ContextReconstructionEngine) -> None:
        snap = context_engine.snapshot()
        assert snap["running"] is True

    def test_operations_raise_before_initialize(self, exp_engine: ExperienceEngine) -> None:
        engine = ContextReconstructionEngine(experience_engine=exp_engine)
        with pytest.raises(EchoNotInitializedError):
            engine.reconstruct_timeline()


# ===========================================================================
# 15. MemoryIntegrityEngine
# ===========================================================================


class TestMemoryIntegrityEngine:
    def test_run_audit_on_clean_store(
        self, integrity_engine: MemoryIntegrityEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="Clean Integrity Test")
        report = integrity_engine.run_audit()
        assert hasattr(report, "violations")
        assert report.is_healthy

    def test_check_experience_integrity(
        self, integrity_engine: MemoryIntegrityEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(exp_engine, title="Individual Integrity Check")
        result = integrity_engine.check_experience_integrity(exp)
        assert isinstance(result, list)

    def test_assert_no_duplicate_passes_for_unique(
        self, integrity_engine: MemoryIntegrityEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(exp_engine, title="Unique Experience No Dup")
        integrity_engine.assert_no_duplicate(exp)  # should not raise

    def test_get_integrity_score_returns_float(
        self, integrity_engine: MemoryIntegrityEngine
    ) -> None:
        score = integrity_engine.get_integrity_score()
        assert 0.0 <= score <= 1.0

    def test_integrity_report_has_violation_count(
        self, integrity_engine: MemoryIntegrityEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="Audit Subject")
        report = integrity_engine.run_audit()
        assert report.violation_count >= 0

    def test_snapshot(self, integrity_engine: MemoryIntegrityEngine) -> None:
        snap = integrity_engine.snapshot()
        assert snap["running"] is True

    def test_operations_raise_before_initialize(
        self, exp_engine: ExperienceEngine, retrieval_engine: ExperienceRetrievalEngine
    ) -> None:
        engine = MemoryIntegrityEngine(
            experience_engine=exp_engine, retrieval_engine=retrieval_engine
        )
        with pytest.raises(EchoNotInitializedError):
            engine.run_audit()


# ===========================================================================
# 16. EpisodicIndexEngine
# ===========================================================================


class TestEpisodicIndexEngine:
    def test_index_experience(
        self, episodic_index: EpisodicIndexEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(
            exp_engine,
            title="Indexed Experience",
            tags=[MemoryTag(name="polaris", category="project")],
        )
        episodic_index.index_experience(exp)
        assert episodic_index.is_indexed(exp.experience_id)

    def test_lookup_by_tag(
        self, episodic_index: EpisodicIndexEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(
            exp_engine,
            title="Lookup Tag Experience",
            tags=[MemoryTag(name="echo-tag", category="topic")],
        )
        episodic_index.index_experience(exp)
        ids = episodic_index.lookup_by_tag("echo-tag")
        assert exp.experience_id in ids

    def test_lookup_by_type(
        self, episodic_index: EpisodicIndexEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(
            exp_engine, title="Achievement Type Lookup", exp_type=ExperienceType.ACHIEVEMENT
        )
        episodic_index.index_experience(exp)
        ids = episodic_index.lookup_by_type(ExperienceType.ACHIEVEMENT)
        assert exp.experience_id in ids

    def test_lookup_by_importance(
        self, episodic_index: EpisodicIndexEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(exp_engine, title="High Importance Lookup", importance=ExperienceImportance.HIGH)
        episodic_index.index_experience(exp)
        ids = episodic_index.lookup_by_importance(ExperienceImportance.HIGH)
        assert exp.experience_id in ids

    def test_remove_experience(
        self, episodic_index: EpisodicIndexEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(exp_engine, title="Remove From Index")
        episodic_index.index_experience(exp)
        episodic_index.remove_experience(exp.experience_id)
        assert not episodic_index.is_indexed(exp.experience_id)

    def test_rebuild_index(
        self, episodic_index: EpisodicIndexEngine, exp_engine: ExperienceEngine
    ) -> None:
        _create_exp(exp_engine, title="Rebuild Index Exp A")
        _create_exp(exp_engine, title="Rebuild Index Exp B")
        report = episodic_index.rebuild_index()
        assert hasattr(report, "indexed_count")
        assert report.indexed_count >= 0

    def test_health_report(self, episodic_index: EpisodicIndexEngine) -> None:
        report = episodic_index.health_report()
        assert hasattr(report, "indexed_count")

    def test_lookup_intersection(
        self, episodic_index: EpisodicIndexEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(
            exp_engine,
            title="Intersection Experience",
            exp_type=ExperienceType.ACHIEVEMENT,
            tags=[MemoryTag(name="polaris-intersection", category="project")],
        )
        episodic_index.index_experience(exp)
        ids = episodic_index.lookup_intersection(
            tags=["polaris-intersection"],
            experience_types=[ExperienceType.ACHIEVEMENT],
        )
        assert exp.experience_id in ids

    def test_get_all_indexed_ids(
        self, episodic_index: EpisodicIndexEngine, exp_engine: ExperienceEngine
    ) -> None:
        exp = _create_exp(exp_engine, title="All IDs Test")
        episodic_index.index_experience(exp)
        all_ids = episodic_index.get_all_indexed_ids()
        assert exp.experience_id in all_ids

    def test_snapshot(self, episodic_index: EpisodicIndexEngine) -> None:
        snap = episodic_index.snapshot()
        assert snap["running"] is True

    def test_operations_raise_before_initialize(
        self, exp_engine: ExperienceEngine, retrieval_engine: ExperienceRetrievalEngine
    ) -> None:
        engine = EpisodicIndexEngine(
            experience_engine=exp_engine, retrieval_engine=retrieval_engine
        )
        with pytest.raises(EchoNotInitializedError):
            engine.lookup_by_tag("tag")


# ===========================================================================
# 17. PatternExtractionEngine
# ===========================================================================


class TestPatternExtractionEngine:
    def _populate_repeating_experiences(
        self, exp_engine: ExperienceEngine, title: str, count: int
    ) -> list[Experience]:
        return [
            _create_exp(
                exp_engine,
                title=title,
                exp_type=ExperienceType.EVENT,
                importance=ExperienceImportance.MEDIUM,
                occurred_at=_ago(days=i * 3),
            )
            for i in range(count)
        ]

    def test_extract_patterns_returns_list(
        self, pattern_engine: PatternExtractionEngine, exp_engine: ExperienceEngine
    ) -> None:
        self._populate_repeating_experiences(exp_engine, "Daily standup completed", 4)
        patterns = pattern_engine.extract_patterns(min_occurrences=2, lookback_days=60)
        assert isinstance(patterns, list)

    def test_get_known_patterns_returns_list(self, pattern_engine: PatternExtractionEngine) -> None:
        known = pattern_engine.get_known_patterns()
        assert isinstance(known, list)

    def test_get_pattern_by_id(
        self, pattern_engine: PatternExtractionEngine, exp_engine: ExperienceEngine
    ) -> None:
        self._populate_repeating_experiences(exp_engine, "Recurring sprint event", 3)
        patterns = pattern_engine.extract_patterns(min_occurrences=2, lookback_days=90)
        if patterns:
            fetched = pattern_engine.get_pattern(patterns[0].pattern_id)
            assert fetched is not None

    def test_audit_patterns_returns_report(
        self, pattern_engine: PatternExtractionEngine, exp_engine: ExperienceEngine
    ) -> None:
        self._populate_repeating_experiences(exp_engine, "Auditable pattern event", 3)
        pattern_engine.extract_patterns(min_occurrences=2, lookback_days=90)
        report = pattern_engine.audit_patterns()
        assert hasattr(report, "total_patterns")

    def test_pattern_health_metrics(self, pattern_engine: PatternExtractionEngine) -> None:
        metrics = pattern_engine.get_pattern_health_metrics()
        assert hasattr(metrics, "total_patterns")

    def test_analyze_tag_cooccurrence(
        self, pattern_engine: PatternExtractionEngine, exp_engine: ExperienceEngine
    ) -> None:
        for i in range(3):
            _create_exp(
                exp_engine,
                title=f"Cooccurrence Exp {i}",
                tags=[MemoryTag(name="alpha", category="topic"), MemoryTag(name="beta", category="topic")],
                occurred_at=_ago(days=i * 5),
            )
        result = pattern_engine.analyze_tag_cooccurrence(lookback_days=90)
        assert isinstance(result, dict)

    def test_snapshot(self, pattern_engine: PatternExtractionEngine) -> None:
        snap = pattern_engine.snapshot()
        assert snap["running"] is True

    def test_operations_raise_before_initialize(
        self, exp_engine: ExperienceEngine, retrieval_engine: ExperienceRetrievalEngine
    ) -> None:
        engine = PatternExtractionEngine(
            experience_engine=exp_engine, retrieval_engine=retrieval_engine
        )
        with pytest.raises(EchoNotInitializedError):
            engine.extract_patterns()


# ===========================================================================
# 18. PersonalHistoryEngine
# ===========================================================================


class TestPersonalHistoryEngine:
    def test_create_chapter(self, personal_history_engine: PersonalHistoryEngine) -> None:
        chapter = personal_history_engine.create_chapter(
            title="POLARIS Build Year One",
            description="The foundational year of POLARIS development.",
        )
        from subsystems.echo.personal_history import HistoryChapter
        assert isinstance(chapter, HistoryChapter)
        assert chapter.title == "POLARIS Build Year One"

    def test_get_chapter(self, personal_history_engine: PersonalHistoryEngine) -> None:
        chapter = personal_history_engine.create_chapter(
            title="Get Chapter Test", description="Test description."
        )
        fetched = personal_history_engine.get_chapter(chapter.chapter_id)
        assert fetched.chapter_id == chapter.chapter_id

    def test_assign_experience_to_chapter(
        self,
        personal_history_engine: PersonalHistoryEngine,
        exp_engine: ExperienceEngine,
    ) -> None:
        chapter = personal_history_engine.create_chapter(
            title="Assignment Chapter", description="Will receive experiences."
        )
        exp = _create_exp(exp_engine, title="Chapter Member Experience")
        personal_history_engine.assign_experience_to_chapter(chapter.chapter_id, exp.experience_id)
        updated = personal_history_engine.get_chapter(chapter.chapter_id)
        assert exp.experience_id in updated.experience_ids

    def test_add_milestone(
        self,
        personal_history_engine: PersonalHistoryEngine,
        exp_engine: ExperienceEngine,
    ) -> None:
        exp = _create_exp(exp_engine, title="Milestone Experience", importance=ExperienceImportance.CRITICAL)
        milestone = personal_history_engine.add_milestone(
            experience_id=exp.experience_id,
            label="Architecture Freeze",
            milestone_type="achievement",
        )
        from subsystems.echo.personal_history import MilestoneRecord
        assert isinstance(milestone, MilestoneRecord)
        assert milestone.label == "Architecture Freeze"

    def test_get_milestone(
        self,
        personal_history_engine: PersonalHistoryEngine,
        exp_engine: ExperienceEngine,
    ) -> None:
        exp = _create_exp(exp_engine, title="Milestone Fetch Exp")
        m = personal_history_engine.add_milestone(
            experience_id=exp.experience_id, label="Fetch Me", milestone_type="achievement"
        )
        fetched = personal_history_engine.get_milestone(m.milestone_id)
        assert fetched.milestone_id == m.milestone_id

    def test_generate_life_timeline(
        self,
        personal_history_engine: PersonalHistoryEngine,
        exp_engine: ExperienceEngine,
    ) -> None:
        _create_exp(exp_engine, title="Timeline A", importance=ExperienceImportance.HIGH, occurred_at=_ago(days=10))
        _create_exp(exp_engine, title="Timeline B", importance=ExperienceImportance.HIGH, occurred_at=_ago(days=5))
        timeline = personal_history_engine.generate_life_timeline(
            min_importance=ExperienceImportance.HIGH
        )
        assert isinstance(timeline, list)

    def test_generate_chapter_narrative(
        self,
        personal_history_engine: PersonalHistoryEngine,
        exp_engine: ExperienceEngine,
    ) -> None:
        chapter = personal_history_engine.create_chapter(
            title="Narrative Chapter", description="Chapter for narrative generation."
        )
        exp = _create_exp(exp_engine, title="Narrative Source Exp")
        personal_history_engine.assign_experience_to_chapter(chapter.chapter_id, exp.experience_id)
        narrative = personal_history_engine.generate_chapter_narrative(chapter.chapter_id)
        from subsystems.echo.personal_history import PersonalNarrative
        assert isinstance(narrative, PersonalNarrative)

    def test_generate_full_narrative(
        self,
        personal_history_engine: PersonalHistoryEngine,
        exp_engine: ExperienceEngine,
    ) -> None:
        _create_exp(exp_engine, title="Full Narrative Exp", importance=ExperienceImportance.HIGH)
        chapter = personal_history_engine.create_chapter(
            title="Full Narrative Chapter", description="Full narrative test."
        )
        exp = _create_exp(exp_engine, title="Chapter Exp")
        personal_history_engine.assign_experience_to_chapter(chapter.chapter_id, exp.experience_id)
        narrative = personal_history_engine.generate_full_narrative()
        from subsystems.echo.personal_history import PersonalNarrative
        assert isinstance(narrative, PersonalNarrative)

    def test_build_growth_trajectory(
        self,
        personal_history_engine: PersonalHistoryEngine,
        exp_engine: ExperienceEngine,
    ) -> None:
        for i in range(3):
            _create_exp(
                exp_engine,
                title=f"Software Achievement {i}",
                exp_type=ExperienceType.ACHIEVEMENT,
                importance=ExperienceImportance.HIGH,
                occurred_at=_ago(days=i * 30),
            )
        trajectory = personal_history_engine.build_growth_trajectory("software")
        from subsystems.echo.personal_history import GrowthTrajectory
        assert isinstance(trajectory, GrowthTrajectory)
        assert trajectory.domain == "software"

    def test_delete_chapter(self, personal_history_engine: PersonalHistoryEngine) -> None:
        chapter = personal_history_engine.create_chapter(
            title="Delete Chapter", description="Will be deleted."
        )
        personal_history_engine.delete_chapter(chapter.chapter_id)
        with pytest.raises(KeyError):
            personal_history_engine.get_chapter(chapter.chapter_id)

    def test_list_chapters(self, personal_history_engine: PersonalHistoryEngine) -> None:
        personal_history_engine.create_chapter(title="Chapter X", description="Desc X.")
        personal_history_engine.create_chapter(title="Chapter Y", description="Desc Y.")
        chapters = personal_history_engine.list_chapters()
        assert len(chapters) >= 2

    def test_snapshot(self, personal_history_engine: PersonalHistoryEngine) -> None:
        snap = personal_history_engine.snapshot()
        assert "running" in snap
        assert snap["running"] is True

    def test_operations_raise_before_initialize(self, exp_engine: ExperienceEngine) -> None:
        engine = PersonalHistoryEngine(experience_engine=exp_engine)
        with pytest.raises(EchoNotInitializedError):
            engine.generate_life_timeline()

    def test_requires_non_none_experience_engine(self) -> None:
        with pytest.raises(ValueError):
            PersonalHistoryEngine(experience_engine=None)  # type: ignore[arg-type]


# ===========================================================================
# 19. EchoSubsystem
# ===========================================================================


class TestEchoSubsystem:
    def test_initialize_sets_status_running(self) -> None:
        subsystem = EchoSubsystem()
        assert subsystem.status == EchoSubsystemStatus.CREATED
        subsystem.initialize()
        assert subsystem.status == EchoSubsystemStatus.RUNNING
        subsystem.shutdown()

    def test_initialize_idempotent(self) -> None:
        subsystem = EchoSubsystem()
        subsystem.initialize()
        subsystem.initialize()  # second call is no-op
        assert subsystem.status == EchoSubsystemStatus.RUNNING
        subsystem.shutdown()

    def test_shutdown_sets_status_stopped(self) -> None:
        subsystem = EchoSubsystem()
        subsystem.initialize()
        subsystem.shutdown()
        assert subsystem.status == EchoSubsystemStatus.STOPPED

    def test_shutdown_idempotent(self) -> None:
        subsystem = EchoSubsystem()
        subsystem.initialize()
        subsystem.shutdown()
        subsystem.shutdown()  # second call is no-op
        assert subsystem.status == EchoSubsystemStatus.STOPPED

    def test_engine_accessors_return_correct_types(self, echo: EchoSubsystem) -> None:
        assert isinstance(echo.significance_engine, SignificanceEngine)
        assert isinstance(echo.experience_engine, ExperienceEngine)
        assert isinstance(echo.retrieval_engine, ExperienceRetrievalEngine)
        assert isinstance(echo.consolidation_engine, MemoryConsolidationEngine)
        assert isinstance(echo.integrity_engine, MemoryIntegrityEngine)
        assert isinstance(echo.episodic_index_engine, EpisodicIndexEngine)
        assert isinstance(echo.reflection_engine, ReflectionEngine)
        assert isinstance(echo.context_reconstruction_engine, ContextReconstructionEngine)
        assert isinstance(echo.pattern_extraction_engine, PatternExtractionEngine)
        assert isinstance(echo.personal_history_engine, PersonalHistoryEngine)

    def test_unimplemented_engine_accessors_return_none(self, echo: EchoSubsystem) -> None:
        assert echo.event_engine is None
        assert echo.conversation_engine is None
        assert echo.session_engine is None
        assert echo.achievement_engine is None
        assert echo.failure_analysis_engine is None

    def test_accessor_raises_when_not_running(self) -> None:
        subsystem = EchoSubsystem()
        with pytest.raises(RuntimeError):
            _ = subsystem.experience_engine

    def test_accessor_raises_after_shutdown(self) -> None:
        subsystem = EchoSubsystem()
        subsystem.initialize()
        subsystem.shutdown()
        with pytest.raises(RuntimeError):
            _ = subsystem.experience_engine

    def test_health_report_structure(self, echo: EchoSubsystem) -> None:
        report = echo.health_report()
        assert "status" in report
        assert "healthy" in report
        assert "engines" in report
        assert "subsystem" in report

    def test_health_report_when_running_is_healthy(self, echo: EchoSubsystem) -> None:
        report = echo.health_report()
        assert report["healthy"] is True
        assert report["status"] == EchoSubsystemStatus.RUNNING.name

    def test_health_report_includes_all_engines(self, echo: EchoSubsystem) -> None:
        report = echo.health_report()
        engines = report["engines"]
        for name in [
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
        ]:
            assert name in engines, f"{name} missing from health report"

    def test_health_report_unimplemented_engines_not_implemented(self, echo: EchoSubsystem) -> None:
        report = echo.health_report()
        engines = report["engines"]
        for name in ["EventEngine", "ConversationEngine", "SessionEngine", "AchievementEngine", "FailureAnalysisEngine"]:
            assert engines[name]["status"] == "not_implemented"

    def test_diagnostics_report_extends_health(self, echo: EchoSubsystem) -> None:
        diag = echo.diagnostics_report()
        assert "diagnostics" in diag
        assert "metadata" in diag
        assert "status" in diag
        assert "engines" in diag

    def test_diagnostics_metadata(self, echo: EchoSubsystem) -> None:
        diag = echo.diagnostics_report()
        meta = diag["metadata"]
        assert meta["name"] == "ECHO"
        assert "version" in meta

    def test_health_report_before_initialize(self) -> None:
        subsystem = EchoSubsystem()
        report = subsystem.health_report()
        assert report["status"] == EchoSubsystemStatus.CREATED.name
        assert report["healthy"] is False

    def test_uptime_seconds_after_init(self, echo: EchoSubsystem) -> None:
        uptime = echo.uptime_seconds()
        assert uptime is not None
        assert uptime >= 0.0

    def test_uptime_seconds_before_init(self) -> None:
        subsystem = EchoSubsystem()
        assert subsystem.uptime_seconds() is None

    def test_subsystem_metadata(self, echo: EchoSubsystem) -> None:
        meta = echo.metadata()
        assert meta["name"] == "ECHO"
        assert len(meta["implemented_engines"]) == 10
        assert len(meta["unimplemented_engines"]) == 5

    def test_name_version_description(self, echo: EchoSubsystem) -> None:
        assert echo.name == "ECHO"
        assert echo.version
        assert echo.description

    def test_invalid_significance_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            EchoSubsystem(min_significance_threshold=1.5)

    def test_end_to_end_experience_through_subsystem(self, echo: EchoSubsystem) -> None:
        exp = echo.experience_engine.create_experience(
            title="End-to-End ECHO Test Experience",
            experience_type=ExperienceType.ACHIEVEMENT,
            importance=ExperienceImportance.HIGH,
            description="Validated the complete ECHO subsystem integration path.",
            outcome="All assertions passed.",
            force=True,
        )
        assert echo.experience_engine.experience_exists(exp.experience_id)

        echo.episodic_index_engine.index_experience(exp)
        ids = echo.episodic_index_engine.lookup_by_type(ExperienceType.ACHIEVEMENT)
        assert exp.experience_id in ids

        report = echo.health_report()
        assert report["healthy"] is True
