# tests/test_astra.py
"""
Comprehensive pytest suite for ASTRA v5 — Digital Twin Core.

Coverage
--------
* Domain models: IdentityProfile, Goal, PreferenceProfile, CapabilityEntry,
  CapabilitySnapshot, DigitalTwinState, EvolutionRecord, ConsistencyReport,
  FutureSelfReference, and all enumerations.
* IdentityEngine — CRUD, validation, thread safety.
* GoalEngine — lifecycle, state transitions, filtering, thread safety.
* PreferenceEngine — all four preference dimensions, validation.
* AstraCapabilityRegistry — registration, removal, queries, snapshots.
* CapabilityManager — upsert_from_evidence, metadata, health.
* ConsistencyEngine — evidence buffer, drift detection, dimension scoring.
* IdentityEvolutionEngine — evidence accumulation, gated evolution, history.
* Serialisation schemas — to_dict / from_dict round-trips.
* Event factories — payload content, types, priorities.
* SubsystemContract compliance (ASTRA lifecycle if present).
* Thread safety under concurrent load.
* Failure / exception handling.

Run with::

    python -m pytest tests/test_astra.py -v

"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ASTRA domain models
# ---------------------------------------------------------------------------
from subsystems.astra.models import (
    CapabilityEntry,
    CapabilitySnapshot,
    ConsistencyReport,
    DigitalTwinState,
    EvidenceItem,
    EvolutionRecord,
    EvolutionTrigger,
    FutureSelfReference,
    Goal,
    GoalPriority,
    GoalState,
    GoalType,
    IdentityProfile,
    PreferenceProfile,
    CommunicationPreferences,
    LearningPreferences,
    DevelopmentPreferences,
    WorkflowPreferences,
)

# ---------------------------------------------------------------------------
# ASTRA engines
# ---------------------------------------------------------------------------
from subsystems.astra.identity import IdentityEngine
from subsystems.astra.goals import GoalEngine
from subsystems.astra.preferences import PreferenceEngine
from subsystems.astra.consistency import ConsistencyEngine
from subsystems.astra.evolution import IdentityEvolutionEngine
from subsystems.astra.capability import (
    AstraCapabilityRegistry,
    CapabilityManager,
    CapabilityNotFoundError,
    CapabilityAlreadyExistsError,
    CapabilityValidationError,
)

# ---------------------------------------------------------------------------
# ASTRA exceptions
# ---------------------------------------------------------------------------
from subsystems.astra.exceptions import (
    AstraError,
    AstraNotInitializedError,
    IdentityNotFoundError,
    IdentityValidationError,
    GoalNotFoundError,
    GoalStateError,
    GoalValidationError,
    PreferenceValidationError,
    EvolutionError,
    InsufficientEvidenceError,
)

# ---------------------------------------------------------------------------
# ASTRA events
# ---------------------------------------------------------------------------
from subsystems.astra import events as astra_events
from subsystems.astra.events import (
    ET_IDENTITY_UPDATED,
    ET_GOAL_CREATED,
    ET_GOAL_UPDATED,
    ET_GOAL_REMOVED,
    ET_PREFERENCE_CHANGED,
    ET_DIGITAL_TWIN_UPDATED,
    ET_CONSISTENCY_CHECK_COMPLETED,
    ET_IDENTITY_EVOLVED,
    ASTRA_SOURCE,
)

# ---------------------------------------------------------------------------
# Serialisation schemas
# ---------------------------------------------------------------------------
from subsystems.astra import schemas
from subsystems.astra.schemas import (
    identity_profile_to_dict,
    identity_profile_from_dict,
    goal_to_dict,
    goal_from_dict,
    preference_profile_to_dict,
    preference_profile_from_dict,
    capability_snapshot_to_dict,
    capability_snapshot_from_dict,
    future_self_ref_to_dict,
    future_self_ref_from_dict,
    digital_twin_state_to_dict,
    evolution_record_to_dict,
    consistency_report_to_dict,
)

# ---------------------------------------------------------------------------
# Event primitives
# ---------------------------------------------------------------------------
from core.events.event import EventPriority


# ===========================================================================
# Helpers / fixtures
# ===========================================================================


def _make_identity(name: str = "Alex", **kwargs: Any) -> IdentityProfile:
    return IdentityProfile(name=name, **kwargs)


def _make_goal(
    title: str = "Build POLARIS",
    goal_type: GoalType = GoalType.PROJECT,
    **kwargs: Any,
) -> Goal:
    return Goal(title=title, goal_type=goal_type, **kwargs)


def _make_capability(
    name: str = "python_arch",
    domain: str = "programming",
    confidence: float = 0.8,
    evidence_count: int = 5,
) -> CapabilityEntry:
    return CapabilityEntry(
        name=name, domain=domain, confidence=confidence, evidence_count=evidence_count
    )


def _make_twin(
    identity: IdentityProfile | None = None,
    goals: list[Goal] | None = None,
) -> DigitalTwinState:
    identity = identity or _make_identity()
    goals = goals or []
    return DigitalTwinState(
        identity=identity,
        goals=goals,
        preferences=PreferenceProfile(),
        capabilities=CapabilitySnapshot(),
    )


# ===========================================================================
# Section 1: Domain Models
# ===========================================================================


class TestIdentityProfileModel:
    def test_create_minimal(self) -> None:
        p = IdentityProfile(name="Sam")
        assert p.name == "Sam"
        assert p.version == 1
        assert p.interests == []
        assert p.core_identity_tags == []

    def test_create_full(self) -> None:
        p = IdentityProfile(
            name="Sam",
            background="Engineer",
            education="BSc CS",
            career_direction="Robotics",
            interests=["AI", "Systems"],
            core_identity_tags=["builder"],
        )
        assert p.interests == ["AI", "Systems"]
        assert "builder" in p.core_identity_tags

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            IdentityProfile(name="")

    def test_whitespace_name_raises(self) -> None:
        with pytest.raises(ValueError):
            IdentityProfile(name="   ")

    def test_invalid_version_raises(self) -> None:
        with pytest.raises(ValueError):
            IdentityProfile(name="Sam", version=0)

    def test_bump_version(self) -> None:
        p = IdentityProfile(name="Sam")
        before = p.updated_at
        p.bump_version()
        assert p.version == 2
        assert p.updated_at >= before

    def test_timestamps_are_utc(self) -> None:
        p = IdentityProfile(name="Sam")
        assert p.created_at.tzinfo is not None
        assert p.updated_at.tzinfo is not None


class TestGoalModel:
    def test_create_minimal(self) -> None:
        g = _make_goal()
        assert g.title == "Build POLARIS"
        assert g.state == GoalState.ACTIVE
        assert g.priority == GoalPriority.MEDIUM
        assert g.progress_pct == 0
        assert g.goal_id  # non-empty UUID

    def test_empty_title_raises(self) -> None:
        with pytest.raises(ValueError):
            Goal(title="", goal_type=GoalType.PROJECT)

    def test_invalid_progress_raises(self) -> None:
        with pytest.raises(ValueError):
            Goal(title="X", goal_type=GoalType.DAILY, progress_pct=101)

    def test_negative_progress_raises(self) -> None:
        with pytest.raises(ValueError):
            Goal(title="X", goal_type=GoalType.DAILY, progress_pct=-1)

    def test_is_terminal_active(self) -> None:
        g = _make_goal()
        assert not g.is_terminal()

    def test_is_terminal_completed(self) -> None:
        g = _make_goal(state=GoalState.COMPLETED)
        assert g.is_terminal()

    def test_is_terminal_abandoned(self) -> None:
        g = _make_goal(state=GoalState.ABANDONED)
        assert g.is_terminal()

    def test_mark_updated(self) -> None:
        g = _make_goal()
        before = g.updated_at
        g.mark_updated()
        assert g.updated_at >= before

    def test_unique_goal_ids(self) -> None:
        g1 = _make_goal()
        g2 = _make_goal()
        assert g1.goal_id != g2.goal_id

    def test_all_goal_types(self) -> None:
        for gtype in GoalType:
            g = Goal(title="T", goal_type=gtype)
            assert g.goal_type == gtype

    def test_all_goal_priorities(self) -> None:
        for pri in GoalPriority:
            g = Goal(title="T", goal_type=GoalType.DAILY, priority=pri)
            assert g.priority == pri


class TestCapabilityEntryModel:
    def test_create_valid(self) -> None:
        entry = _make_capability()
        assert entry.name == "python_arch"
        assert entry.confidence == 0.8

    def test_confidence_out_of_range_low(self) -> None:
        with pytest.raises(ValueError):
            CapabilityEntry(name="x", domain="d", confidence=-0.1)

    def test_confidence_out_of_range_high(self) -> None:
        with pytest.raises(ValueError):
            CapabilityEntry(name="x", domain="d", confidence=1.1)

    def test_negative_evidence_count_raises(self) -> None:
        with pytest.raises(ValueError):
            CapabilityEntry(name="x", domain="d", confidence=0.5, evidence_count=-1)

    def test_boundary_confidence_zero(self) -> None:
        e = CapabilityEntry(name="x", domain="d", confidence=0.0)
        assert e.confidence == 0.0

    def test_boundary_confidence_one(self) -> None:
        e = CapabilityEntry(name="x", domain="d", confidence=1.0)
        assert e.confidence == 1.0


class TestCapabilitySnapshotModel:
    def test_all_entries(self) -> None:
        s = _make_capability()
        g = _make_capability(name="g", confidence=0.4)
        snap = CapabilitySnapshot(strengths=[s], growth_areas=[g])
        all_e = snap.all_entries()
        assert len(all_e) == 2

    def test_empty_snapshot(self) -> None:
        snap = CapabilitySnapshot()
        assert snap.all_entries() == []


class TestDigitalTwinStateModel:
    def test_create_valid(self) -> None:
        twin = _make_twin()
        assert twin.consistency_score == 1.0
        assert twin.twin_version == 1

    def test_invalid_consistency_score(self) -> None:
        with pytest.raises(ValueError):
            _make_twin()
            DigitalTwinState(
                identity=_make_identity(),
                goals=[],
                preferences=PreferenceProfile(),
                capabilities=CapabilitySnapshot(),
                consistency_score=1.5,
            )

    def test_active_goals_filter(self) -> None:
        active = _make_goal(state=GoalState.ACTIVE)
        completed = _make_goal(title="Done", state=GoalState.COMPLETED)
        twin = DigitalTwinState(
            identity=_make_identity(),
            goals=[active, completed],
            preferences=PreferenceProfile(),
            capabilities=CapabilitySnapshot(),
        )
        assert len(twin.active_goals) == 1
        assert twin.active_goals[0].state == GoalState.ACTIVE

    def test_is_consistent_above_threshold(self) -> None:
        twin = DigitalTwinState(
            identity=_make_identity(),
            goals=[],
            preferences=PreferenceProfile(),
            capabilities=CapabilitySnapshot(),
            consistency_score=0.8,
        )
        assert twin.is_consistent

    def test_is_consistent_below_threshold(self) -> None:
        twin = DigitalTwinState(
            identity=_make_identity(),
            goals=[],
            preferences=PreferenceProfile(),
            capabilities=CapabilitySnapshot(),
            consistency_score=0.5,
        )
        assert not twin.is_consistent


class TestEvolutionRecordModel:
    def test_create_factory(self) -> None:
        record = EvolutionRecord.create(
            trigger=EvolutionTrigger.MANUAL_UPDATE,
            changed_fields=["name", "background"],
            previous_snapshot={"name": "Old"},
            new_snapshot={"name": "New"},
            confidence=0.9,
            evidence_count=5,
            notes="Test evolution",
        )
        assert record.trigger == EvolutionTrigger.MANUAL_UPDATE
        assert "name" in record.changed_fields
        assert record.confidence == 0.9
        assert record.evidence_count == 5
        assert record.record_id  # UUID present
        assert record.evolved_at.tzinfo is not None

    def test_immutable_frozen(self) -> None:
        record = EvolutionRecord.create(
            trigger=EvolutionTrigger.EVIDENCE_THRESHOLD,
            changed_fields=["interests"],
            previous_snapshot={},
            new_snapshot={},
            confidence=0.7,
            evidence_count=3,
        )
        with pytest.raises((AttributeError, TypeError)):
            record.confidence = 0.5  # type: ignore[misc]

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError):
            EvolutionRecord(
                record_id=str(uuid.uuid4()),
                trigger=EvolutionTrigger.MANUAL_UPDATE,
                changed_fields=("name",),
                previous_snapshot={},
                new_snapshot={},
                confidence=1.5,
                evidence_count=1,
                notes="",
                evolved_at=datetime.now(timezone.utc),
            )

    def test_negative_evidence_count_raises(self) -> None:
        with pytest.raises(ValueError):
            EvolutionRecord(
                record_id=str(uuid.uuid4()),
                trigger=EvolutionTrigger.MANUAL_UPDATE,
                changed_fields=("name",),
                previous_snapshot={},
                new_snapshot={},
                confidence=0.8,
                evidence_count=-1,
                notes="",
                evolved_at=datetime.now(timezone.utc),
            )


class TestFutureSelfReferenceModel:
    def test_create_valid(self) -> None:
        ref = FutureSelfReference(
            horizon_label="1-year",
            horizon_years=1.0,
            headline_identity="Engineer at a startup",
        )
        assert ref.horizon_years == 1.0
        assert ref.confidence == 0.5

    def test_zero_horizon_raises(self) -> None:
        with pytest.raises(ValueError):
            FutureSelfReference(
                horizon_label="x", horizon_years=0, headline_identity="X"
            )

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError):
            FutureSelfReference(
                horizon_label="x",
                horizon_years=1,
                headline_identity="X",
                confidence=1.5,
            )


class TestEvidenceItemModel:
    def test_create_valid(self) -> None:
        item = EvidenceItem(
            source="ECHO",
            signal_type="behavior_observation",
            weight=0.7,
            description="User demonstrated consistent late-night coding.",
        )
        assert item.weight == 0.7

    def test_invalid_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            EvidenceItem(source="X", signal_type="Y", weight=1.5, description="Z")


class TestConsistencyReportModel:
    def test_is_stable_no_drift(self) -> None:
        report = ConsistencyReport(
            overall_score=0.9,
            dimension_scores={"identity": 0.9},
            drift_detected=False,
            drift_fields=[],
            evidence_items=[],
            recommendation="Stable.",
        )
        assert report.is_stable

    def test_is_stable_drift_detected(self) -> None:
        report = ConsistencyReport(
            overall_score=0.9,
            dimension_scores={},
            drift_detected=True,
            drift_fields=["goals"],
            evidence_items=[],
            recommendation="Drift.",
        )
        assert not report.is_stable

    def test_invalid_score_raises(self) -> None:
        with pytest.raises(ValueError):
            ConsistencyReport(
                overall_score=1.2,
                dimension_scores={},
                drift_detected=False,
                drift_fields=[],
                evidence_items=[],
                recommendation="",
            )


# ===========================================================================
# Section 2: Identity Engine
# ===========================================================================


class TestIdentityEngine:
    def test_initial_no_identity(self) -> None:
        engine = IdentityEngine()
        assert not engine.has_identity()

    def test_create_identity(self) -> None:
        engine = IdentityEngine()
        profile = engine.update_identity({"name": "Alex"})
        assert profile.name == "Alex"
        assert engine.has_identity()

    def test_get_identity_raises_when_none(self) -> None:
        engine = IdentityEngine()
        with pytest.raises(IdentityNotFoundError):
            engine.get_identity()

    def test_create_without_name_raises(self) -> None:
        engine = IdentityEngine()
        with pytest.raises(IdentityValidationError):
            engine.update_identity({"background": "Engineer"})

    def test_create_with_empty_name_raises(self) -> None:
        engine = IdentityEngine()
        with pytest.raises(IdentityValidationError):
            engine.update_identity({"name": "  "})

    def test_update_existing_identity(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex"})
        profile = engine.update_identity({"background": "Engineer"})
        assert profile.background == "Engineer"
        assert profile.name == "Alex"

    def test_update_increments_version(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex"})
        v1 = engine.get_identity().version
        engine.update_identity({"background": "Updated"})
        v2 = engine.get_identity().version
        assert v2 > v1

    def test_update_interests(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex"})
        engine.update_identity({"interests": ["AI", "Robotics"]})
        assert engine.get_identity().interests == ["AI", "Robotics"]

    def test_interests_must_be_list(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex"})
        with pytest.raises(IdentityValidationError):
            engine.update_identity({"interests": "AI"})

    def test_update_core_identity_tags(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex", "core_identity_tags": ["builder"]})
        assert "builder" in engine.get_identity().core_identity_tags

    def test_core_identity_tags_must_be_list(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex"})
        with pytest.raises(IdentityValidationError):
            engine.update_identity({"core_identity_tags": "builder"})

    def test_metadata_update_merges(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex", "metadata": {"key1": "val1"}})
        engine.update_identity({"metadata": {"key2": "val2"}})
        meta = engine.get_identity().metadata
        assert "key1" in meta
        assert "key2" in meta

    def test_metadata_must_be_dict(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex"})
        with pytest.raises(IdentityValidationError):
            engine.update_identity({"metadata": ["not", "a", "dict"]})

    def test_unknown_field_raises(self) -> None:
        engine = IdentityEngine()
        with pytest.raises(IdentityValidationError) as exc_info:
            engine.update_identity({"name": "Alex", "invalid_field": "x"})
        assert exc_info.value.field == "invalid_field"

    def test_initial_profile_injected(self) -> None:
        profile = _make_identity(name="Injected")
        engine = IdentityEngine(initial_profile=profile)
        assert engine.has_identity()
        assert engine.get_identity().name == "Injected"

    def test_get_changed_fields_new_profile(self) -> None:
        engine = IdentityEngine()
        changed = engine.get_changed_fields({"name": "Alex", "background": "Eng"})
        assert "name" in changed
        assert "background" in changed

    def test_get_changed_fields_existing_profile(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex", "background": "Old"})
        changed = engine.get_changed_fields({"background": "New"})
        assert "background" in changed

    def test_get_changed_fields_no_change(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex"})
        changed = engine.get_changed_fields({"name": "Alex"})
        assert changed == []

    def test_thread_safety_concurrent_updates(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Base"})
        errors: list[Exception] = []

        def update_bg(i: int) -> None:
            try:
                engine.update_identity({"background": f"bg-{i}"})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=update_bg, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert engine.has_identity()

    def test_update_career_direction(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex"})
        engine.update_identity({"career_direction": "Robotics Entrepreneur"})
        assert engine.get_identity().career_direction == "Robotics Entrepreneur"

    def test_update_education(self) -> None:
        engine = IdentityEngine()
        engine.update_identity({"name": "Alex"})
        engine.update_identity({"education": "MSc Mechatronics"})
        assert engine.get_identity().education == "MSc Mechatronics"


# ===========================================================================
# Section 3: Goal Engine
# ===========================================================================


class TestGoalEngine:
    def test_create_goal(self) -> None:
        engine = GoalEngine()
        goal = engine.create_goal("Build POLARIS", GoalType.PROJECT)
        assert goal.title == "Build POLARIS"
        assert goal.state == GoalState.ACTIVE
        assert engine.get_goal_count() == 1

    def test_create_goal_empty_title_raises(self) -> None:
        engine = GoalEngine()
        with pytest.raises(GoalValidationError):
            engine.create_goal("", GoalType.PROJECT)

    def test_create_goal_whitespace_title_raises(self) -> None:
        engine = GoalEngine()
        with pytest.raises(GoalValidationError):
            engine.create_goal("   ", GoalType.DAILY)

    def test_get_goal_by_id(self) -> None:
        engine = GoalEngine()
        goal = engine.create_goal("Test Goal", GoalType.LIFE)
        fetched = engine.get_goal(goal.goal_id)
        assert fetched.goal_id == goal.goal_id

    def test_get_goal_not_found_raises(self) -> None:
        engine = GoalEngine()
        with pytest.raises(GoalNotFoundError):
            engine.get_goal(str(uuid.uuid4()))

    def test_get_all_goals(self) -> None:
        engine = GoalEngine()
        engine.create_goal("G1", GoalType.DAILY)
        engine.create_goal("G2", GoalType.CAREER)
        goals = engine.get_goals()
        assert len(goals) == 2

    def test_filter_goals_by_state(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("Active G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"state": GoalState.PAUSED})
        active = engine.get_goals(state=GoalState.ACTIVE)
        paused = engine.get_goals(state=GoalState.PAUSED)
        assert len(active) == 0
        assert len(paused) == 1

    def test_filter_goals_by_type(self) -> None:
        engine = GoalEngine()
        engine.create_goal("Daily", GoalType.DAILY)
        engine.create_goal("Life", GoalType.LIFE)
        daily = engine.get_goals(goal_type=GoalType.DAILY)
        assert len(daily) == 1

    def test_filter_goals_by_priority(self) -> None:
        engine = GoalEngine()
        engine.create_goal("High", GoalType.PROJECT, priority=GoalPriority.HIGH)
        engine.create_goal("Low", GoalType.PROJECT, priority=GoalPriority.LOW)
        high = engine.get_goals(priority=GoalPriority.HIGH)
        assert len(high) == 1

    def test_goals_sorted_by_priority(self) -> None:
        engine = GoalEngine()
        engine.create_goal("Low", GoalType.PROJECT, priority=GoalPriority.LOW)
        engine.create_goal("Critical", GoalType.PROJECT, priority=GoalPriority.CRITICAL)
        engine.create_goal("Medium", GoalType.PROJECT, priority=GoalPriority.MEDIUM)
        goals = engine.get_goals()
        priorities = [g.priority.value for g in goals]
        assert priorities == sorted(priorities, reverse=True)

    def test_update_goal_title(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("Old Title", GoalType.DAILY)
        updated = engine.update_goal(g.goal_id, {"title": "New Title"})
        assert updated.title == "New Title"

    def test_update_goal_progress(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"progress_pct": 75})
        assert engine.get_goal(g.goal_id).progress_pct == 75

    def test_update_goal_invalid_progress_raises(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        with pytest.raises(GoalValidationError):
            engine.update_goal(g.goal_id, {"progress_pct": 150})

    def test_update_goal_state_active_to_paused(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"state": GoalState.PAUSED})
        assert engine.get_goal(g.goal_id).state == GoalState.PAUSED

    def test_update_goal_state_paused_to_active(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"state": GoalState.PAUSED})
        engine.update_goal(g.goal_id, {"state": GoalState.ACTIVE})
        assert engine.get_goal(g.goal_id).state == GoalState.ACTIVE

    def test_update_goal_state_active_to_completed(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"state": GoalState.COMPLETED})
        updated = engine.get_goal(g.goal_id)
        assert updated.state == GoalState.COMPLETED
        assert updated.completed_at is not None

    def test_update_terminal_goal_raises(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"state": GoalState.COMPLETED})
        with pytest.raises(GoalStateError):
            engine.update_goal(g.goal_id, {"description": "Cannot update terminal."})

    def test_invalid_state_transition_raises(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        with pytest.raises(GoalStateError):
            # ACTIVE cannot go directly to COMPLETED via an invalid path? No,
            # ACTIVE -> COMPLETED is valid, use DEFERRED -> COMPLETED
            engine.update_goal(g.goal_id, {"state": GoalState.DEFERRED})
            engine.update_goal(g.goal_id, {"state": GoalState.COMPLETED})

    def test_invalid_transition_from_active_to_active_not_in_table(self) -> None:
        # Test a genuinely invalid transition: COMPLETED -> ACTIVE
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"state": GoalState.COMPLETED})
        with pytest.raises(GoalStateError):
            engine.update_goal(g.goal_id, {"state": GoalState.ACTIVE})

    def test_remove_goal(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("To Remove", GoalType.DAILY)
        removed = engine.remove_goal(g.goal_id)
        assert removed.goal_id == g.goal_id
        assert engine.get_goal_count() == 0

    def test_remove_nonexistent_goal_raises(self) -> None:
        engine = GoalEngine()
        with pytest.raises(GoalNotFoundError):
            engine.remove_goal(str(uuid.uuid4()))

    def test_get_active_goals(self) -> None:
        engine = GoalEngine()
        engine.create_goal("Active", GoalType.PROJECT)
        g2 = engine.create_goal("To Pause", GoalType.DAILY)
        engine.update_goal(g2.goal_id, {"state": GoalState.PAUSED})
        active = engine.get_active_goals()
        assert len(active) == 1
        assert active[0].title == "Active"

    def test_create_child_goal(self) -> None:
        engine = GoalEngine()
        parent = engine.create_goal("Parent", GoalType.LIFE)
        child = engine.create_goal(
            "Child", GoalType.PROJECT, parent_goal_id=parent.goal_id
        )
        assert child.parent_goal_id == parent.goal_id

    def test_create_child_invalid_parent_raises(self) -> None:
        engine = GoalEngine()
        with pytest.raises(GoalValidationError):
            engine.create_goal(
                "Child", GoalType.PROJECT, parent_goal_id=str(uuid.uuid4())
            )

    def test_update_unknown_field_raises(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        with pytest.raises(GoalValidationError):
            engine.update_goal(g.goal_id, {"nonexistent_field": "x"})

    def test_update_motivation(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"motivation": "To learn AI"})
        assert engine.get_goal(g.goal_id).motivation == "To learn AI"

    def test_update_success_criteria(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"success_criteria": ["Ship MVP"]})
        assert "Ship MVP" in engine.get_goal(g.goal_id).success_criteria

    def test_success_criteria_must_be_list(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        with pytest.raises(GoalValidationError):
            engine.update_goal(g.goal_id, {"success_criteria": "not a list"})

    def test_update_tags(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"tags": ["polaris", "ai"]})
        assert "polaris" in engine.get_goal(g.goal_id).tags

    def test_update_tags_must_be_list(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        with pytest.raises(GoalValidationError):
            engine.update_goal(g.goal_id, {"tags": "not a list"})

    def test_update_goal_string_state(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"state": "PAUSED"})
        assert engine.get_goal(g.goal_id).state == GoalState.PAUSED

    def test_update_goal_invalid_string_state(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        with pytest.raises(GoalValidationError):
            engine.update_goal(g.goal_id, {"state": "INVALID_STATE"})

    def test_get_changed_fields(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("Original", GoalType.PROJECT)
        changed = engine.get_changed_fields(g.goal_id, {"title": "New Title"})
        assert "title" in changed

    def test_thread_safety_concurrent_creates(self) -> None:
        engine = GoalEngine()
        errors: list[Exception] = []

        def create_goal(i: int) -> None:
            try:
                engine.create_goal(f"Goal {i}", GoalType.PROJECT)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=create_goal, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert engine.get_goal_count() == 30


# ===========================================================================
# Section 4: Preference Engine
# ===========================================================================


class TestPreferenceEngine:
    def test_default_preferences(self) -> None:
        engine = PreferenceEngine()
        prefs = engine.get_preferences()
        assert isinstance(prefs, PreferenceProfile)
        assert prefs.version == 1

    def test_update_communication_tone(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences({"communication": {"preferred_tone": "direct"}})
        assert engine.get_preferences().communication.preferred_tone == "direct"

    def test_update_communication_detail_level(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences(
            {"communication": {"preferred_detail_level": "brief"}}
        )
        assert engine.get_preferences().communication.preferred_detail_level == "brief"

    def test_invalid_communication_tone_raises(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences({"communication": {"preferred_tone": "angry"}})

    def test_invalid_detail_level_raises(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences(
                {"communication": {"preferred_detail_level": "extreme"}}
            )

    def test_update_learning_approach(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences(
            {"learning": {"preferred_approach": "examples_first"}}
        )
        assert (
            engine.get_preferences().learning.preferred_approach == "examples_first"
        )

    def test_invalid_learning_approach_raises(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences(
                {"learning": {"preferred_approach": "memorisation"}}
            )

    def test_update_learning_pace(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences(
            {"learning": {"preferred_learning_pace": "intensive"}}
        )
        assert (
            engine.get_preferences().learning.preferred_learning_pace == "intensive"
        )

    def test_invalid_learning_pace_raises(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences(
                {"learning": {"preferred_learning_pace": "turbo"}}
            )

    def test_update_development_languages(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences(
            {"development": {"preferred_languages": ["Python", "Rust"]}}
        )
        assert "Python" in engine.get_preferences().development.preferred_languages

    def test_development_languages_must_be_list(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences(
                {"development": {"preferred_languages": "Python"}}
            )

    def test_update_testing_discipline(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences({"development": {"testing_discipline": "tdd"}})
        assert engine.get_preferences().development.testing_discipline == "tdd"

    def test_invalid_testing_discipline_raises(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences(
                {"development": {"testing_discipline": "haphazard"}}
            )

    def test_update_documentation_style(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences(
            {"development": {"documentation_style": "comprehensive"}}
        )
        assert (
            engine.get_preferences().development.documentation_style == "comprehensive"
        )

    def test_invalid_documentation_style_raises(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences(
                {"development": {"documentation_style": "random"}}
            )

    def test_update_workflow_session_length(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences(
            {"workflow": {"preferred_session_length_hours": 4.5}}
        )
        assert (
            engine.get_preferences().workflow.preferred_session_length_hours == 4.5
        )

    def test_workflow_negative_session_length_raises(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences(
                {"workflow": {"preferred_session_length_hours": -1.0}}
            )

    def test_update_workflow_peak_hours(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences({"workflow": {"peak_hours": ["22:00", "23:00"]}})
        assert "22:00" in engine.get_preferences().workflow.peak_hours

    def test_peak_hours_must_be_list(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences({"workflow": {"peak_hours": "22:00"}})

    def test_unknown_dimension_raises(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences({"unknown_dim": {}})

    def test_unknown_communication_field_raises(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences({"communication": {"secret_field": "x"}})

    def test_update_increments_version(self) -> None:
        engine = PreferenceEngine()
        v1 = engine.get_preferences().version
        engine.update_preferences({"communication": {"preferred_tone": "direct"}})
        v2 = engine.get_preferences().version
        assert v2 > v1

    def test_update_metadata(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences({"metadata": {"source": "onboarding"}})
        assert engine.get_preferences().metadata.get("source") == "onboarding"

    def test_metadata_must_be_dict(self) -> None:
        engine = PreferenceEngine()
        with pytest.raises(PreferenceValidationError):
            engine.update_preferences({"metadata": "not a dict"})

    def test_multiple_dimensions_in_one_call(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences(
            {
                "communication": {"preferred_tone": "casual"},
                "learning": {"depth_over_breadth": True},
            }
        )
        prefs = engine.get_preferences()
        assert prefs.communication.preferred_tone == "casual"
        assert prefs.learning.depth_over_breadth is True

    def test_get_changed_dimensions(self) -> None:
        engine = PreferenceEngine()
        changed = engine.get_changed_dimensions({"communication": {}, "workflow": {}})
        assert "communication" in changed
        assert "workflow" in changed

    def test_thread_safety(self) -> None:
        engine = PreferenceEngine()
        errors: list[Exception] = []

        def update(i: int) -> None:
            try:
                engine.update_preferences(
                    {"workflow": {"preferred_session_length_hours": float(i % 5 + 1)}}
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=update, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# Section 5: AstraCapabilityRegistry
# ===========================================================================


class TestAstraCapabilityRegistry:
    def test_register_and_get(self) -> None:
        reg = AstraCapabilityRegistry()
        entry = _make_capability()
        reg.register(entry)
        fetched = reg.get(name="python_arch", domain="programming")
        assert fetched.name == "python_arch"

    def test_register_duplicate_raises(self) -> None:
        reg = AstraCapabilityRegistry()
        entry = _make_capability()
        reg.register(entry)
        with pytest.raises(CapabilityAlreadyExistsError):
            reg.register(_make_capability())

    def test_register_non_entry_raises(self) -> None:
        reg = AstraCapabilityRegistry()
        with pytest.raises(TypeError):
            reg.register("not an entry")  # type: ignore[arg-type]

    def test_register_empty_name_raises(self) -> None:
        reg = AstraCapabilityRegistry()
        with pytest.raises(CapabilityValidationError):
            reg.register(CapabilityEntry(name="", domain="d", confidence=0.5))

    def test_register_empty_domain_raises(self) -> None:
        reg = AstraCapabilityRegistry()
        with pytest.raises(CapabilityValidationError):
            reg.register(CapabilityEntry(name="n", domain="", confidence=0.5))

    def test_remove_capability(self) -> None:
        reg = AstraCapabilityRegistry()
        entry = _make_capability()
        reg.register(entry)
        removed = reg.remove(name="python_arch", domain="programming")
        assert removed.name == "python_arch"
        assert not reg.has(name="python_arch", domain="programming")

    def test_remove_nonexistent_raises(self) -> None:
        reg = AstraCapabilityRegistry()
        with pytest.raises(CapabilityNotFoundError):
            reg.remove(name="nonexistent", domain="programming")

    def test_update_capability_confidence(self) -> None:
        reg = AstraCapabilityRegistry()
        reg.register(_make_capability())
        updated = reg.update_capability(
            name="python_arch", domain="programming", confidence=0.95
        )
        assert updated.confidence == 0.95

    def test_update_capability_invalid_confidence_raises(self) -> None:
        reg = AstraCapabilityRegistry()
        reg.register(_make_capability())
        with pytest.raises(CapabilityValidationError):
            reg.update_capability(
                name="python_arch", domain="programming", confidence=2.0
            )

    def test_update_capability_negative_evidence_raises(self) -> None:
        reg = AstraCapabilityRegistry()
        reg.register(_make_capability())
        with pytest.raises(CapabilityValidationError):
            reg.update_capability(
                name="python_arch", domain="programming", evidence_count=-5
            )

    def test_update_nonexistent_raises(self) -> None:
        reg = AstraCapabilityRegistry()
        with pytest.raises(CapabilityNotFoundError):
            reg.update_capability(name="missing", domain="d", confidence=0.5)

    def test_has_returns_false_unknown(self) -> None:
        reg = AstraCapabilityRegistry()
        assert not reg.has(name="unknown", domain="d")

    def test_list_all(self) -> None:
        reg = AstraCapabilityRegistry()
        reg.register(_make_capability(name="a"))
        reg.register(_make_capability(name="b"))
        assert len(reg.list_all()) == 2

    def test_list_by_domain(self) -> None:
        reg = AstraCapabilityRegistry()
        reg.register(_make_capability(name="a", domain="programming"))
        reg.register(_make_capability(name="b", domain="design"))
        prog = reg.list_by_domain("programming")
        assert len(prog) == 1 and prog[0].name == "a"

    def test_strengths_and_growth_areas(self) -> None:
        reg = AstraCapabilityRegistry(strength_threshold=0.7)
        reg.register(_make_capability(name="strong", confidence=0.9))
        reg.register(_make_capability(name="weak", confidence=0.4))
        assert len(reg.strengths()) == 1
        assert len(reg.growth_areas()) == 1

    def test_snapshot(self) -> None:
        reg = AstraCapabilityRegistry(strength_threshold=0.7)
        reg.register(_make_capability(name="strong", confidence=0.9))
        reg.register(_make_capability(name="weak", confidence=0.3))
        snap = reg.snapshot()
        assert len(snap.strengths) == 1
        assert len(snap.growth_areas) == 1
        assert isinstance(snap.snapshot_at, datetime)

    def test_health_summary(self) -> None:
        reg = AstraCapabilityRegistry(strength_threshold=0.7)
        reg.register(_make_capability(name="a", confidence=0.9, evidence_count=10))
        reg.register(_make_capability(name="b", confidence=0.4, evidence_count=2))
        summary = reg.health_summary()
        assert summary["total_capabilities"] == 2
        assert summary["strengths_count"] == 1
        assert summary["growth_areas_count"] == 1
        assert 0.0 <= summary["average_confidence"] <= 1.0

    def test_len(self) -> None:
        reg = AstraCapabilityRegistry()
        reg.register(_make_capability())
        assert len(reg) == 1

    def test_contains_tuple(self) -> None:
        reg = AstraCapabilityRegistry()
        reg.register(_make_capability())
        assert ("python_arch", "programming") in reg
        assert ("missing", "programming") not in reg

    def test_invalid_strength_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            AstraCapabilityRegistry(strength_threshold=0.0)

    def test_case_insensitive_lookup(self) -> None:
        reg = AstraCapabilityRegistry()
        reg.register(CapabilityEntry(name="PythonArch", domain="Programming", confidence=0.8))
        assert reg.has(name="pythonarch", domain="programming")

    def test_thread_safety_concurrent_register(self) -> None:
        reg = AstraCapabilityRegistry()
        errors: list[Exception] = []

        def register_entry(i: int) -> None:
            try:
                reg.register(CapabilityEntry(name=f"cap_{i}", domain="d", confidence=0.5))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=register_entry, args=(i,)) for i in range(30)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(reg) == 30


# ===========================================================================
# Section 6: CapabilityManager
# ===========================================================================


class TestCapabilityManager:
    def test_register_capability(self) -> None:
        mgr = CapabilityManager()
        entry = mgr.register_capability(
            name="sys_thinking", domain="cognitive", confidence=0.85
        )
        assert entry.name == "sys_thinking"
        assert mgr.has_capability(name="sys_thinking", domain="cognitive")

    def test_register_duplicate_raises(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(name="x", domain="d", confidence=0.5)
        with pytest.raises(CapabilityAlreadyExistsError):
            mgr.register_capability(name="x", domain="d", confidence=0.6)

    def test_unregister_capability(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(name="x", domain="d", confidence=0.5)
        removed = mgr.unregister_capability(name="x", domain="d")
        assert removed.name == "x"
        assert not mgr.has_capability(name="x", domain="d")

    def test_unregister_nonexistent_raises(self) -> None:
        mgr = CapabilityManager()
        with pytest.raises(CapabilityNotFoundError):
            mgr.unregister_capability(name="missing", domain="d")

    def test_upsert_creates_new(self) -> None:
        mgr = CapabilityManager()
        entry = mgr.upsert_from_evidence(
            name="arch_design", domain="engineering", confidence_delta=0.3
        )
        assert mgr.has_capability(name="arch_design", domain="engineering")
        assert entry.confidence == 0.3
        assert entry.evidence_count == 1

    def test_upsert_updates_existing(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(
            name="arch_design", domain="engineering", confidence=0.5, evidence_count=2
        )
        entry = mgr.upsert_from_evidence(
            name="arch_design", domain="engineering", confidence_delta=0.2
        )
        assert abs(entry.confidence - 0.7) < 0.001
        assert entry.evidence_count == 3

    def test_upsert_clamps_to_1(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(name="x", domain="d", confidence=0.9)
        entry = mgr.upsert_from_evidence(
            name="x", domain="d", confidence_delta=0.5
        )
        assert entry.confidence <= 1.0

    def test_upsert_clamps_to_0(self) -> None:
        mgr = CapabilityManager()
        entry = mgr.upsert_from_evidence(
            name="x", domain="d", confidence_delta=-0.5
        )
        assert entry.confidence >= 0.0

    def test_query_capability_returns_none_for_missing(self) -> None:
        mgr = CapabilityManager()
        result = mgr.query_capability(name="missing", domain="d")
        assert result is None

    def test_query_capability_returns_entry(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(name="x", domain="d", confidence=0.5)
        result = mgr.query_capability(name="x", domain="d")
        assert result is not None
        assert result.name == "x"

    def test_metadata_stored_and_retrieved(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(
            name="x", domain="d", confidence=0.5, metadata={"source": "test"}
        )
        meta = mgr.get_metadata(name="x", domain="d")
        assert meta.get("source") == "test"

    def test_metadata_empty_for_untracked(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(name="x", domain="d", confidence=0.5)
        meta = mgr.get_metadata(name="x", domain="d")
        assert meta == {}

    def test_snapshot_returns_correct_type(self) -> None:
        mgr = CapabilityManager()
        snap = mgr.snapshot()
        assert isinstance(snap, CapabilitySnapshot)

    def test_health_returns_summary(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(name="x", domain="d", confidence=0.8)
        health = mgr.health()
        assert "total_capabilities" in health
        assert health["total_capabilities"] == 1

    def test_list_all(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(name="a", domain="d", confidence=0.5)
        mgr.register_capability(name="b", domain="d", confidence=0.6)
        assert len(mgr.list_all()) == 2

    def test_list_by_domain(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(name="a", domain="eng", confidence=0.5)
        mgr.register_capability(name="b", domain="design", confidence=0.6)
        eng = mgr.list_by_domain("eng")
        assert len(eng) == 1

    def test_strengths_and_growth_areas(self) -> None:
        mgr = CapabilityManager(strength_threshold=0.7)
        mgr.register_capability(name="strong", domain="d", confidence=0.9)
        mgr.register_capability(name="growing", domain="d", confidence=0.4)
        assert len(mgr.strengths()) == 1
        assert len(mgr.growth_areas()) == 1

    def test_len(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(name="x", domain="d", confidence=0.5)
        assert len(mgr) == 1

    def test_registry_property(self) -> None:
        mgr = CapabilityManager()
        assert isinstance(mgr.registry, AstraCapabilityRegistry)

    def test_metadata_cleared_on_unregister(self) -> None:
        mgr = CapabilityManager()
        mgr.register_capability(
            name="x", domain="d", confidence=0.5, metadata={"k": "v"}
        )
        mgr.unregister_capability(name="x", domain="d")
        # re-register without metadata
        mgr.register_capability(name="x", domain="d", confidence=0.5)
        assert mgr.get_metadata(name="x", domain="d") == {}


# ===========================================================================
# Section 7: Consistency Engine
# ===========================================================================


class TestConsistencyEngine:
    def test_record_evidence(self) -> None:
        engine = ConsistencyEngine()
        engine.record_evidence(
            source="ECHO",
            signal_type="behavior_observation",
            weight=0.5,
            description="Late-night coding again.",
        )
        assert engine.get_evidence_count() == 1

    def test_invalid_evidence_weight_raises(self) -> None:
        engine = ConsistencyEngine()
        with pytest.raises(ValueError):
            engine.record_evidence(
                source="X", signal_type="Y", weight=1.5, description="Z"
            )

    def test_evidence_buffer_trims_at_max(self) -> None:
        engine = ConsistencyEngine()
        for i in range(250):
            engine.record_evidence(
                source="src", signal_type="test", weight=0.5, description=f"item {i}"
            )
        assert engine.get_evidence_count() <= 200

    def test_clear_evidence(self) -> None:
        engine = ConsistencyEngine()
        engine.record_evidence(source="X", signal_type="Y", weight=0.5, description="Z")
        count = engine.clear_evidence()
        assert count == 1
        assert engine.get_evidence_count() == 0

    def test_run_check_no_evidence_stable(self) -> None:
        engine = ConsistencyEngine()
        identity = _make_identity()
        report = engine.run_consistency_check(
            identity=identity,
            goals=[],
            preferences=PreferenceProfile(),
        )
        assert isinstance(report, ConsistencyReport)
        assert report.overall_score >= 0.0

    def test_run_check_returns_consistency_report(self) -> None:
        engine = ConsistencyEngine()
        report = engine.run_consistency_check(
            identity=_make_identity(),
            goals=[],
            preferences=PreferenceProfile(),
        )
        assert "identity" in report.dimension_scores
        assert "goals" in report.dimension_scores
        assert "preferences" in report.dimension_scores

    def test_run_check_no_identity(self) -> None:
        engine = ConsistencyEngine()
        report = engine.run_consistency_check(
            identity=None, goals=[], preferences=PreferenceProfile()
        )
        assert report.dimension_scores["identity"] == 1.0

    def test_drift_detected_high_identity_signals(self) -> None:
        engine = ConsistencyEngine()
        for _ in range(10):
            engine.record_evidence(
                source="AURORA",
                signal_type="identity_instability",
                weight=0.9,
                description="Major shift in self-perception.",
            )
        report = engine.run_consistency_check(
            identity=_make_identity(),
            goals=[],
            preferences=PreferenceProfile(),
        )
        assert report.drift_detected or report.overall_score < 1.0

    def test_goal_abandonment_affects_score(self) -> None:
        engine = ConsistencyEngine()
        goals = [
            Goal(title=f"G{i}", goal_type=GoalType.PROJECT, state=GoalState.ABANDONED)
            for i in range(5)
        ]
        report = engine.run_consistency_check(
            identity=_make_identity(),
            goals=goals,
            preferences=PreferenceProfile(),
        )
        assert report.dimension_scores["goals"] < 1.0

    def test_preference_contradiction_affects_score(self) -> None:
        engine = ConsistencyEngine()
        for _ in range(5):
            engine.record_evidence(
                source="ECHO",
                signal_type="preference_pattern",
                weight=0.8,
                description="User shows contradiction in workflow preference",
            )
        report = engine.run_consistency_check(
            identity=_make_identity(),
            goals=[],
            preferences=PreferenceProfile(),
        )
        assert report is not None

    def test_external_signals_incorporated(self) -> None:
        engine = ConsistencyEngine()
        external = [
            {
                "source": "AURORA",
                "signal_type": "emotion_shift",
                "weight": 0.8,
                "description": "High emotional volatility.",
            }
        ]
        report = engine.run_consistency_check(
            identity=_make_identity(),
            goals=[],
            preferences=PreferenceProfile(),
            external_signals=external,
        )
        assert len(report.evidence_items) >= 1

    def test_get_last_report_initially_none(self) -> None:
        engine = ConsistencyEngine()
        assert engine.get_last_report() is None

    def test_get_last_report_after_check(self) -> None:
        engine = ConsistencyEngine()
        engine.run_consistency_check(
            identity=_make_identity(),
            goals=[],
            preferences=PreferenceProfile(),
        )
        assert engine.get_last_report() is not None

    def test_thread_safety_concurrent_evidence(self) -> None:
        engine = ConsistencyEngine()
        errors: list[Exception] = []

        def add_evidence() -> None:
            try:
                engine.record_evidence(
                    source="X", signal_type="Y", weight=0.5, description="concurrent"
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_evidence) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# Section 8: Identity Evolution Engine
# ===========================================================================


class TestIdentityEvolutionEngine:
    def test_accumulate_evidence(self) -> None:
        engine = IdentityEvolutionEngine()
        engine.accumulate_evidence(
            source="ECHO",
            signal_type="career_shift",
            weight=0.8,
            description="Shifted toward ML engineering.",
        )
        assert engine.get_pending_evidence_count() == 1

    def test_invalid_weight_raises(self) -> None:
        engine = IdentityEvolutionEngine()
        with pytest.raises(ValueError):
            engine.accumulate_evidence(
                source="X", signal_type="Y", weight=2.0, description="bad"
            )

    def test_clear_pending_evidence(self) -> None:
        engine = IdentityEvolutionEngine()
        engine.accumulate_evidence(
            source="X", signal_type="Y", weight=0.5, description="Z"
        )
        count = engine.clear_pending_evidence()
        assert count == 1
        assert engine.get_pending_evidence_count() == 0

    def test_compute_confidence_no_evidence(self) -> None:
        engine = IdentityEvolutionEngine()
        assert engine.compute_confidence() == 0.0

    def test_compute_confidence_with_evidence(self) -> None:
        engine = IdentityEvolutionEngine()
        engine.accumulate_evidence(
            source="X", signal_type="Y", weight=0.8, description="A"
        )
        engine.accumulate_evidence(
            source="X", signal_type="Y", weight=0.6, description="B"
        )
        confidence = engine.compute_confidence()
        assert 0.0 < confidence <= 1.0

    def test_evolve_identity_insufficient_evidence_raises(self) -> None:
        engine = IdentityEvolutionEngine(min_evidence_count=3, min_confidence=0.65)
        identity = _make_identity()
        with pytest.raises(InsufficientEvidenceError):
            engine.evolve_identity(identity, {"background": "Changed"})

    def test_evolve_identity_low_confidence_raises(self) -> None:
        engine = IdentityEvolutionEngine(min_evidence_count=1, min_confidence=0.9)
        engine.accumulate_evidence(
            source="X", signal_type="Y", weight=0.1, description="low"
        )
        identity = _make_identity()
        with pytest.raises(InsufficientEvidenceError):
            engine.evolve_identity(identity, {"background": "Changed"})

    def test_evolve_identity_forced(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        updated, record = engine.evolve_identity(
            identity,
            {"background": "New background"},
            force=True,
        )
        assert updated.background == "New background"
        assert isinstance(record, EvolutionRecord)

    def test_evolve_identity_with_sufficient_evidence(self) -> None:
        engine = IdentityEvolutionEngine(min_evidence_count=3, min_confidence=0.65)
        for _ in range(4):
            engine.accumulate_evidence(
                source="ECHO", signal_type="career_shift", weight=0.8, description="Sig"
            )
        identity = _make_identity()
        updated, record = engine.evolve_identity(
            identity, {"career_direction": "ML Engineering"}
        )
        assert updated.career_direction == "ML Engineering"

    def test_evolve_clears_pending_evidence(self) -> None:
        engine = IdentityEvolutionEngine(min_evidence_count=1, min_confidence=0.5)
        engine.accumulate_evidence(source="X", signal_type="Y", weight=0.7, description="Z")
        identity = _make_identity()
        engine.evolve_identity(identity, {"background": "New"})
        assert engine.get_pending_evidence_count() == 0

    def test_evolution_logged(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        engine.evolve_identity(identity, {"background": "Evolved"}, force=True)
        history = engine.get_evolution_history()
        assert len(history) == 1

    def test_evolution_log_accumulates(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        engine.evolve_identity(identity, {"background": "First"}, force=True)
        engine.evolve_identity(identity, {"career_direction": "Second"}, force=True)
        assert engine.get_evolution_count() == 2

    def test_evolution_record_immutable(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        engine.evolve_identity(identity, {"background": "Test"}, force=True)
        history = engine.get_evolution_history()
        assert isinstance(history[0], EvolutionRecord)

    def test_evolve_none_identity_raises(self) -> None:
        engine = IdentityEvolutionEngine()
        with pytest.raises(IdentityNotFoundError):
            engine.evolve_identity(None, {"background": "X"}, force=True)  # type: ignore[arg-type]

    def test_evolve_no_valid_fields_raises(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        with pytest.raises(EvolutionError):
            engine.evolve_identity(identity, {"nonexistent_field": "X"}, force=True)

    def test_evolve_invalid_field_raises(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        with pytest.raises(EvolutionError):
            engine.evolve_identity(identity, {"version": 99}, force=True)

    def test_evolve_trigger_manual(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        _, record = engine.evolve_identity(
            identity,
            {"background": "X"},
            trigger=EvolutionTrigger.MANUAL_UPDATE,
            force=True,
        )
        assert record.trigger == EvolutionTrigger.MANUAL_UPDATE

    def test_forced_evolution_sets_confidence_one(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        _, record = engine.evolve_identity(
            identity, {"background": "X"}, force=True
        )
        assert record.confidence == 1.0

    def test_evolution_with_notes(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        _, record = engine.evolve_identity(
            identity,
            {"background": "Updated"},
            notes="User confirmed career pivot.",
            force=True,
        )
        assert "career" in record.notes

    def test_thread_safety_accumulate(self) -> None:
        engine = IdentityEvolutionEngine()
        errors: list[Exception] = []

        def accumulate() -> None:
            try:
                engine.accumulate_evidence(
                    source="X", signal_type="Y", weight=0.5, description="concurrent"
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=accumulate) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert engine.get_pending_evidence_count() == 30


# ===========================================================================
# Section 9: Serialisation Schemas
# ===========================================================================


class TestSerialisationSchemas:
    def test_identity_profile_round_trip(self) -> None:
        original = IdentityProfile(
            name="Sam",
            background="Engineer",
            education="MSc",
            career_direction="AI",
            interests=["ML", "Systems"],
            core_identity_tags=["builder"],
        )
        d = identity_profile_to_dict(original)
        restored = identity_profile_from_dict(d)
        assert restored.name == original.name
        assert restored.background == original.background
        assert restored.interests == original.interests
        assert restored.core_identity_tags == original.core_identity_tags
        assert restored.version == original.version

    def test_goal_round_trip(self) -> None:
        original = Goal(
            title="Build POLARIS",
            goal_type=GoalType.PROJECT,
            priority=GoalPriority.HIGH,
            description="Full description",
            motivation="Learn AI",
            success_criteria=["Ship v1"],
            tags=["polaris"],
        )
        d = goal_to_dict(original)
        restored = goal_from_dict(d)
        assert restored.title == original.title
        assert restored.goal_type == original.goal_type
        assert restored.priority == original.priority
        assert restored.success_criteria == original.success_criteria
        assert restored.tags == original.tags

    def test_goal_completed_at_preserved(self) -> None:
        from datetime import timezone as tz
        completed_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=tz.utc)
        original = Goal(
            title="Done",
            goal_type=GoalType.DAILY,
            state=GoalState.COMPLETED,
            completed_at=completed_at,
        )
        d = goal_to_dict(original)
        restored = goal_from_dict(d)
        assert restored.completed_at is not None

    def test_preference_profile_round_trip(self) -> None:
        original = PreferenceProfile()
        original.communication.preferred_tone = "direct"
        original.learning.depth_over_breadth = True
        original.development.preferred_languages = ["Python", "Rust"]
        original.workflow.peak_hours = ["22:00"]
        d = preference_profile_to_dict(original)
        restored = preference_profile_from_dict(d)
        assert restored.communication.preferred_tone == "direct"
        assert restored.learning.depth_over_breadth is True
        assert "Python" in restored.development.preferred_languages
        assert "22:00" in restored.workflow.peak_hours

    def test_capability_snapshot_round_trip(self) -> None:
        strong = _make_capability(name="strong", confidence=0.9)
        weak = _make_capability(name="weak", confidence=0.3)
        original = CapabilitySnapshot(strengths=[strong], growth_areas=[weak])
        d = capability_snapshot_to_dict(original)
        restored = capability_snapshot_from_dict(d)
        assert len(restored.strengths) == 1
        assert len(restored.growth_areas) == 1

    def test_future_self_ref_round_trip(self) -> None:
        original = FutureSelfReference(
            horizon_label="1-year",
            horizon_years=1.0,
            headline_identity="ML Engineer at a startup",
            projected_goals=["Ship POLARIS"],
            growth_trajectory={"ml": "accelerating"},
            confidence=0.7,
        )
        d = future_self_ref_to_dict(original)
        restored = future_self_ref_from_dict(d)
        assert restored.horizon_label == original.horizon_label
        assert restored.headline_identity == original.headline_identity
        assert restored.confidence == original.confidence

    def test_digital_twin_state_serialisation(self) -> None:
        twin = DigitalTwinState(
            identity=_make_identity(),
            goals=[_make_goal()],
            preferences=PreferenceProfile(),
            capabilities=CapabilitySnapshot(),
            consistency_score=0.9,
        )
        d = digital_twin_state_to_dict(twin)
        assert "identity" in d
        assert "goals" in d
        assert "preferences" in d
        assert "capabilities" in d
        assert d["consistency_score"] == 0.9
        assert len(d["goals"]) == 1

    def test_evolution_record_serialisation(self) -> None:
        record = EvolutionRecord.create(
            trigger=EvolutionTrigger.EVIDENCE_THRESHOLD,
            changed_fields=["career_direction"],
            previous_snapshot={"career_direction": "Old"},
            new_snapshot={"career_direction": "New"},
            confidence=0.85,
            evidence_count=4,
            notes="Consistent signal pattern.",
        )
        d = evolution_record_to_dict(record)
        assert d["trigger"] == "EVIDENCE_THRESHOLD"
        assert "career_direction" in d["changed_fields"]
        assert d["confidence"] == 0.85

    def test_consistency_report_serialisation(self) -> None:
        report = ConsistencyReport(
            overall_score=0.85,
            dimension_scores={"identity": 0.9, "goals": 0.8},
            drift_detected=False,
            drift_fields=[],
            evidence_items=[
                EvidenceItem(
                    source="test",
                    signal_type="observation",
                    weight=0.5,
                    description="Test",
                )
            ],
            recommendation="Stable.",
        )
        d = consistency_report_to_dict(report)
        assert d["overall_score"] == 0.85
        assert not d["drift_detected"]
        assert len(d["evidence_items"]) == 1


# ===========================================================================
# Section 10: Event Factories
# ===========================================================================


class TestEventFactories:
    def test_identity_updated_event(self) -> None:
        event = astra_events.identity_updated(
            changed_fields=["name", "background"],
            identity_version=2,
        )
        assert event.event_type == ET_IDENTITY_UPDATED
        assert event.source == ASTRA_SOURCE
        assert event.payload["changed_fields"] == ["name", "background"]
        assert event.payload["identity_version"] == 2

    def test_goal_created_event(self) -> None:
        event = astra_events.goal_created(
            goal_id="abc-123",
            title="Build POLARIS",
            goal_type="PROJECT",
            priority="HIGH",
        )
        assert event.event_type == ET_GOAL_CREATED
        assert event.payload["goal_id"] == "abc-123"
        assert event.payload["title"] == "Build POLARIS"

    def test_goal_updated_event(self) -> None:
        event = astra_events.goal_updated(
            goal_id="abc-123",
            changed_fields=["state"],
            new_state="PAUSED",
        )
        assert event.event_type == ET_GOAL_UPDATED
        assert event.payload["new_state"] == "PAUSED"
        assert "state" in event.payload["changed_fields"]

    def test_goal_removed_event(self) -> None:
        event = astra_events.goal_removed(goal_id="abc-123", title="To Remove")
        assert event.event_type == ET_GOAL_REMOVED
        assert event.payload["title"] == "To Remove"

    def test_preference_changed_event(self) -> None:
        event = astra_events.preference_changed(
            changed_dimensions=["communication"],
            preference_version=3,
        )
        assert event.event_type == ET_PREFERENCE_CHANGED
        assert event.payload["preference_version"] == 3

    def test_digital_twin_updated_event(self) -> None:
        event = astra_events.digital_twin_updated(
            twin_version=5, consistency_score=0.9, active_goal_count=3
        )
        assert event.event_type == ET_DIGITAL_TWIN_UPDATED
        assert event.payload["twin_version"] == 5
        assert event.payload["consistency_score"] == 0.9
        assert event.payload["active_goal_count"] == 3

    def test_consistency_check_completed_event(self) -> None:
        event = astra_events.consistency_check_completed(
            overall_score=0.85,
            drift_detected=False,
            drift_fields=[],
        )
        assert event.event_type == ET_CONSISTENCY_CHECK_COMPLETED
        assert event.payload["overall_score"] == 0.85
        assert not event.payload["drift_detected"]

    def test_identity_evolved_event(self) -> None:
        event = astra_events.identity_evolved(
            record_id="rec-001",
            trigger="EVIDENCE_THRESHOLD",
            changed_fields=["career_direction"],
            confidence=0.87,
            evidence_count=5,
        )
        assert event.event_type == ET_IDENTITY_EVOLVED
        assert event.priority == EventPriority.HIGH
        assert event.payload["record_id"] == "rec-001"
        assert event.payload["confidence"] == 0.87

    def test_event_source_is_astra(self) -> None:
        for factory, kwargs in [
            (astra_events.goal_removed, {"goal_id": "x", "title": "T"}),
            (
                astra_events.preference_changed,
                {"changed_dimensions": [], "preference_version": 1},
            ),
        ]:
            event = factory(**kwargs)
            assert event.source == ASTRA_SOURCE

    def test_identity_evolved_is_high_priority(self) -> None:
        event = astra_events.identity_evolved(
            record_id="r",
            trigger="MANUAL_UPDATE",
            changed_fields=["name"],
            confidence=1.0,
            evidence_count=0,
        )
        assert event.priority == EventPriority.HIGH

    def test_goal_created_is_normal_priority(self) -> None:
        event = astra_events.goal_created(
            goal_id="x", title="T", goal_type="DAILY", priority="LOW"
        )
        assert event.priority == EventPriority.NORMAL

    def test_metadata_passthrough(self) -> None:
        event = astra_events.goal_removed(
            goal_id="x", title="T", metadata={"audit": "test"}
        )
        assert event.metadata.get("audit") == "test"

    def test_event_has_event_id(self) -> None:
        event = astra_events.goal_created(
            goal_id="x", title="T", goal_type="PROJECT", priority="MEDIUM"
        )
        assert event.event_id
        # Should be a UUID-like string
        assert len(event.event_id) > 0

    def test_events_have_unique_ids(self) -> None:
        e1 = astra_events.goal_removed(goal_id="a", title="A")
        e2 = astra_events.goal_removed(goal_id="b", title="B")
        assert e1.event_id != e2.event_id


# ===========================================================================
# Section 11: Exception Hierarchy
# ===========================================================================


class TestExceptionHierarchy:
    def test_astra_not_initialized_is_astra_error(self) -> None:
        exc = AstraNotInitializedError("test_op")
        assert isinstance(exc, AstraError)
        assert "test_op" in str(exc)

    def test_identity_not_found_is_astra_error(self) -> None:
        exc = IdentityNotFoundError()
        assert isinstance(exc, AstraError)

    def test_identity_validation_error_has_field(self) -> None:
        exc = IdentityValidationError("Bad name", field="name")
        assert exc.field == "name"
        assert isinstance(exc, AstraError)

    def test_goal_not_found_has_goal_id(self) -> None:
        exc = GoalNotFoundError("goal-123")
        assert exc.goal_id == "goal-123"
        assert isinstance(exc, AstraError)

    def test_goal_validation_error_has_field(self) -> None:
        exc = GoalValidationError("Bad title", field="title")
        assert exc.field == "title"

    def test_goal_state_error(self) -> None:
        exc = GoalStateError("gid", "COMPLETED", "ACTIVE")
        assert exc.goal_id == "gid"
        assert exc.current_state == "COMPLETED"
        assert exc.attempted_state == "ACTIVE"

    def test_preference_validation_error_has_field(self) -> None:
        exc = PreferenceValidationError("Bad tone", field="preferred_tone")
        assert exc.field == "preferred_tone"

    def test_insufficient_evidence_error(self) -> None:
        exc = InsufficientEvidenceError(required=5, available=2, confidence=0.4)
        assert exc.required == 5
        assert exc.available == 2
        assert isinstance(exc, EvolutionError)
        assert isinstance(exc, AstraError)

    def test_evolution_error_with_confidence(self) -> None:
        exc = EvolutionError("Not enough evidence.", confidence=0.3)
        assert exc.confidence == 0.3

    def test_capability_not_found_error(self) -> None:
        exc = CapabilityNotFoundError("arch_design", "engineering")
        assert exc.name == "arch_design"
        assert exc.domain == "engineering"

    def test_capability_already_exists_error(self) -> None:
        exc = CapabilityAlreadyExistsError("arch_design", "engineering")
        assert exc.name == "arch_design"
        assert exc.domain == "engineering"

    def test_capability_validation_error(self) -> None:
        exc = CapabilityValidationError("Bad confidence", field="confidence")
        assert exc.field == "confidence"


# ===========================================================================
# Section 12: Integration workflows
# ===========================================================================


class TestIdentityWorkflow:
    """End-to-end identity creation + evolution workflow."""

    def test_create_then_evolve_identity(self) -> None:
        id_engine = IdentityEngine()
        evo_engine = IdentityEvolutionEngine()

        # Create identity
        id_engine.update_identity(
            {"name": "Alex", "career_direction": "Software Engineer"}
        )

        # Accumulate evidence for career pivot
        for _ in range(4):
            evo_engine.accumulate_evidence(
                source="ECHO",
                signal_type="career_shift",
                weight=0.8,
                description="Repeated interest in robotics projects.",
                related_fields=["career_direction"],
            )

        identity = id_engine.get_identity()
        updated, record = evo_engine.evolve_identity(
            identity, {"career_direction": "Robotics Entrepreneur"}
        )

        assert updated.career_direction == "Robotics Entrepreneur"
        assert record.evidence_count == 4
        assert "career_direction" in record.changed_fields

    def test_identity_version_increments_through_evolution(self) -> None:
        id_engine = IdentityEngine()
        id_engine.update_identity({"name": "Sam"})
        v1 = id_engine.get_identity().version

        evo_engine = IdentityEvolutionEngine()
        identity = id_engine.get_identity()
        evo_engine.evolve_identity(identity, {"background": "Evolved"}, force=True)

        assert identity.version > v1


class TestGoalWorkflow:
    """End-to-end goal lifecycle."""

    def test_full_goal_lifecycle(self) -> None:
        engine = GoalEngine()

        # Create
        goal = engine.create_goal(
            "Launch POLARIS",
            GoalType.PROJECT,
            priority=GoalPriority.CRITICAL,
            motivation="Build digital partner",
        )
        assert goal.state == GoalState.ACTIVE

        # Progress
        engine.update_goal(goal.goal_id, {"progress_pct": 50})
        assert engine.get_goal(goal.goal_id).progress_pct == 50

        # Pause
        engine.update_goal(goal.goal_id, {"state": GoalState.PAUSED})
        assert engine.get_goal(goal.goal_id).state == GoalState.PAUSED

        # Resume
        engine.update_goal(goal.goal_id, {"state": GoalState.ACTIVE})
        engine.update_goal(goal.goal_id, {"progress_pct": 100})

        # Complete
        engine.update_goal(goal.goal_id, {"state": GoalState.COMPLETED})
        completed = engine.get_goal(goal.goal_id)
        assert completed.state == GoalState.COMPLETED
        assert completed.completed_at is not None

    def test_goal_hierarchy(self) -> None:
        engine = GoalEngine()
        parent = engine.create_goal("Master AI", GoalType.LIFE)
        child1 = engine.create_goal(
            "Complete POLARIS", GoalType.PROJECT, parent_goal_id=parent.goal_id
        )
        child2 = engine.create_goal(
            "Study ML", GoalType.PROJECT, parent_goal_id=parent.goal_id
        )

        all_goals = engine.get_goals()
        child_goals = [g for g in all_goals if g.parent_goal_id == parent.goal_id]
        assert len(child_goals) == 2


class TestConsistencyAndEvolutionIntegration:
    """Combined consistency check and evolution workflow."""

    def test_consistency_before_evolution_gate(self) -> None:
        consistency = ConsistencyEngine()
        evolution = IdentityEvolutionEngine(min_evidence_count=2, min_confidence=0.7)

        identity = _make_identity(career_direction="Software Engineer")

        # Record signals in consistency engine
        for _ in range(3):
            consistency.record_evidence(
                source="AURORA",
                signal_type="identity_change",
                weight=0.8,
                description="User self-described as roboticist.",
            )
            evolution.accumulate_evidence(
                source="AURORA",
                signal_type="career_shift",
                weight=0.8,
                description="User self-described as roboticist.",
            )

        report = consistency.run_consistency_check(
            identity=identity, goals=[], preferences=PreferenceProfile()
        )

        assert isinstance(report, ConsistencyReport)

        # Evolution should now be possible
        updated, record = evolution.evolve_identity(
            identity, {"career_direction": "Robotics Engineer"}
        )
        assert updated.career_direction == "Robotics Engineer"


class TestCapabilityWorkflow:
    """Capability registry and manager workflow."""

    def test_register_observe_promote(self) -> None:
        mgr = CapabilityManager(strength_threshold=0.7)

        # Start as growth area
        mgr.upsert_from_evidence(
            name="systems_design", domain="engineering", confidence_delta=0.4
        )
        assert len(mgr.growth_areas()) == 1
        assert len(mgr.strengths()) == 0

        # Accumulate evidence pushing to strength
        for _ in range(4):
            mgr.upsert_from_evidence(
                name="systems_design", domain="engineering", confidence_delta=0.1
            )

        cap = mgr.query_capability(name="systems_design", domain="engineering")
        assert cap is not None
        assert cap.confidence >= 0.7


class TestDigitalTwinStateIntegration:
    """Building a DigitalTwinState from engine outputs."""

    def test_build_digital_twin_from_engines(self) -> None:
        id_engine = IdentityEngine()
        id_engine.update_identity(
            {
                "name": "Alex",
                "career_direction": "AI Developer",
                "interests": ["AI", "Robotics"],
                "core_identity_tags": ["builder", "systems-thinker"],
            }
        )

        goal_engine = GoalEngine()
        goal_engine.create_goal("Build POLARIS", GoalType.PROJECT, priority=GoalPriority.CRITICAL)
        goal_engine.create_goal("Master ML", GoalType.CAREER)

        cap_mgr = CapabilityManager(strength_threshold=0.7)
        cap_mgr.register_capability(
            name="python", domain="programming", confidence=0.9, evidence_count=20
        )

        consistency = ConsistencyEngine()
        report = consistency.run_consistency_check(
            identity=id_engine.get_identity(),
            goals=goal_engine.get_goals(),
            preferences=PreferenceProfile(),
        )

        twin = DigitalTwinState(
            identity=id_engine.get_identity(),
            goals=goal_engine.get_goals(),
            preferences=PreferenceProfile(),
            capabilities=cap_mgr.snapshot(),
            consistency_score=report.overall_score,
        )

        assert twin.identity.name == "Alex"
        assert len(twin.active_goals) == 2
        assert len(twin.capabilities.strengths) == 1
        assert twin.is_consistent

    def test_digital_twin_serialisation_round_trip(self) -> None:
        twin = DigitalTwinState(
            identity=_make_identity(
                name="Sam",
                interests=["AI"],
                core_identity_tags=["builder"],
            ),
            goals=[_make_goal()],
            preferences=PreferenceProfile(),
            capabilities=CapabilitySnapshot(
                strengths=[_make_capability()], growth_areas=[]
            ),
            consistency_score=0.92,
        )

        d = digital_twin_state_to_dict(twin)
        assert d["identity"]["name"] == "Sam"
        assert len(d["goals"]) == 1
        assert d["consistency_score"] == 0.92
        assert len(d["capabilities"]["strengths"]) == 1


# ===========================================================================
# Section 13: Edge cases and failure paths
# ===========================================================================


class TestEdgeCases:
    def test_identity_engine_update_only_name(self) -> None:
        engine = IdentityEngine()
        profile = engine.update_identity({"name": "X"})
        assert profile.name == "X"
        assert profile.background == ""

    def test_goal_engine_empty_after_remove_all(self) -> None:
        engine = GoalEngine()
        g1 = engine.create_goal("G1", GoalType.DAILY)
        g2 = engine.create_goal("G2", GoalType.DAILY)
        engine.remove_goal(g1.goal_id)
        engine.remove_goal(g2.goal_id)
        assert engine.get_goal_count() == 0

    def test_consistency_engine_clears_and_recounts(self) -> None:
        engine = ConsistencyEngine()
        engine.record_evidence(source="X", signal_type="Y", weight=0.5, description="Z")
        engine.clear_evidence()
        engine.record_evidence(source="X", signal_type="Y", weight=0.5, description="Z2")
        assert engine.get_evidence_count() == 1

    def test_evolution_engine_multiple_evolutions(self) -> None:
        engine = IdentityEvolutionEngine()
        identity = _make_identity()
        for i in range(5):
            engine.evolve_identity(
                identity, {"background": f"Version {i}"}, force=True
            )
        assert engine.get_evolution_count() == 5
        assert engine.get_evolution_history()[0].new_snapshot["background"] == "Version 0"

    def test_capability_registry_returns_copy_not_original(self) -> None:
        reg = AstraCapabilityRegistry()
        reg.register(_make_capability())
        entry = reg.get(name="python_arch", domain="programming")
        entry.confidence = 0.0  # Mutate the returned copy
        original = reg.get(name="python_arch", domain="programming")
        assert original.confidence == 0.8  # Registry unaffected

    def test_goal_engine_deferred_state(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"state": GoalState.DEFERRED})
        assert engine.get_goal(g.goal_id).state == GoalState.DEFERRED

    def test_goal_engine_deferred_back_to_active(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT)
        engine.update_goal(g.goal_id, {"state": GoalState.DEFERRED})
        engine.update_goal(g.goal_id, {"state": GoalState.ACTIVE})
        assert engine.get_goal(g.goal_id).state == GoalState.ACTIVE

    def test_goal_engine_metadata_merged(self) -> None:
        engine = GoalEngine()
        g = engine.create_goal("G", GoalType.PROJECT, metadata={"k1": "v1"})
        engine.update_goal(g.goal_id, {"metadata": {"k2": "v2"}})
        goal = engine.get_goal(g.goal_id)
        assert "k1" in goal.metadata
        assert "k2" in goal.metadata

    def test_capability_manager_upsert_notes(self) -> None:
        mgr = CapabilityManager()
        mgr.upsert_from_evidence(
            name="arch", domain="eng", confidence_delta=0.5, notes="First observation"
        )
        mgr.upsert_from_evidence(
            name="arch", domain="eng", confidence_delta=0.1, notes="Updated note"
        )
        cap = mgr.query_capability(name="arch", domain="eng")
        assert cap is not None
        assert cap.notes == "Updated note"

    def test_capability_manager_upsert_metadata_merges(self) -> None:
        mgr = CapabilityManager()
        mgr.upsert_from_evidence(
            name="x", domain="d", confidence_delta=0.3, metadata={"source": "obs1"}
        )
        mgr.upsert_from_evidence(
            name="x", domain="d", confidence_delta=0.1, metadata={"extra": "data"}
        )
        meta = mgr.get_metadata(name="x", domain="d")
        assert "source" in meta
        assert "extra" in meta

    def test_preference_engine_deep_work_false(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences({"workflow": {"deep_work_sessions": False}})
        assert engine.get_preferences().workflow.deep_work_sessions is False

    def test_preference_engine_prefers_examples_false(self) -> None:
        engine = PreferenceEngine()
        engine.update_preferences({"communication": {"prefers_examples": False}})
        assert engine.get_preferences().communication.prefers_examples is False

    def test_evolution_record_fields_are_tuple(self) -> None:
        record = EvolutionRecord.create(
            trigger=EvolutionTrigger.MANUAL_UPDATE,
            changed_fields=["name"],
            previous_snapshot={"name": "Old"},
            new_snapshot={"name": "New"},
            confidence=1.0,
            evidence_count=0,
        )
        assert isinstance(record.changed_fields, tuple)

    def test_future_self_reference_negative_horizon_raises(self) -> None:
        with pytest.raises(ValueError):
            FutureSelfReference(
                horizon_label="x", horizon_years=-1.0, headline_identity="X"
            )

    def test_goal_engine_filter_combined(self) -> None:
        engine = GoalEngine()
        engine.create_goal("A", GoalType.PROJECT, priority=GoalPriority.HIGH)
        engine.create_goal("B", GoalType.DAILY, priority=GoalPriority.HIGH)
        engine.create_goal("C", GoalType.PROJECT, priority=GoalPriority.LOW)

        result = engine.get_goals(goal_type=GoalType.PROJECT, priority=GoalPriority.HIGH)
        assert len(result) == 1
        assert result[0].title == "A"