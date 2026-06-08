# subsystems/astra/schemas.py
"""
ASTRA v5 Serialisation Schemas.

Provides to_dict / from_dict round-trip serialisation for all ASTRA domain
models.  These functions are the canonical serialisation layer used by the
Memory Gateway integration and the public API response formatting.

All functions are pure (no side effects) and raise :class:`ValueError` on
invalid input rather than failing silently.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from subsystems.astra.models import (
    CapabilityEntry,
    CapabilitySnapshot,
    CommunicationPreferences,
    ConsistencyReport,
    DevelopmentPreferences,
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
    LearningPreferences,
    PreferenceProfile,
    WorkflowPreferences,
)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# IdentityProfile
# ---------------------------------------------------------------------------

def identity_profile_to_dict(profile: IdentityProfile) -> dict[str, Any]:
    """Serialise an :class:`IdentityProfile` to a plain dictionary."""
    return {
        "name": profile.name,
        "background": profile.background,
        "education": profile.education,
        "career_direction": profile.career_direction,
        "interests": list(profile.interests),
        "core_identity_tags": list(profile.core_identity_tags),
        "created_at": _dt_to_str(profile.created_at),
        "updated_at": _dt_to_str(profile.updated_at),
        "version": profile.version,
        "metadata": dict(profile.metadata),
    }


def identity_profile_from_dict(data: dict[str, Any]) -> IdentityProfile:
    """Deserialise an :class:`IdentityProfile` from a dictionary."""
    return IdentityProfile(
        name=data["name"],
        background=data.get("background", ""),
        education=data.get("education", ""),
        career_direction=data.get("career_direction", ""),
        interests=list(data.get("interests", [])),
        core_identity_tags=list(data.get("core_identity_tags", [])),
        created_at=_str_to_dt(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
        updated_at=_str_to_dt(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        version=data.get("version", 1),
        metadata=dict(data.get("metadata", {})),
    )


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

def goal_to_dict(goal: Goal) -> dict[str, Any]:
    """Serialise a :class:`Goal` to a plain dictionary."""
    return {
        "goal_id": goal.goal_id,
        "title": goal.title,
        "goal_type": goal.goal_type.name,
        "state": goal.state.name,
        "priority": goal.priority.name,
        "description": goal.description,
        "motivation": goal.motivation,
        "success_criteria": list(goal.success_criteria),
        "tags": list(goal.tags),
        "parent_goal_id": goal.parent_goal_id,
        "created_at": _dt_to_str(goal.created_at),
        "updated_at": _dt_to_str(goal.updated_at),
        "target_date": _dt_to_str(goal.target_date) if goal.target_date else None,
        "completed_at": _dt_to_str(goal.completed_at) if goal.completed_at else None,
        "progress_pct": goal.progress_pct,
        "metadata": dict(goal.metadata),
    }


def goal_from_dict(data: dict[str, Any]) -> Goal:
    """Deserialise a :class:`Goal` from a dictionary."""
    return Goal(
        goal_id=data.get("goal_id", ""),
        title=data["title"],
        goal_type=GoalType[data["goal_type"]],
        state=GoalState[data.get("state", "ACTIVE")],
        priority=GoalPriority[data.get("priority", "MEDIUM")],
        description=data.get("description", ""),
        motivation=data.get("motivation", ""),
        success_criteria=list(data.get("success_criteria", [])),
        tags=list(data.get("tags", [])),
        parent_goal_id=data.get("parent_goal_id"),
        created_at=_str_to_dt(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
        updated_at=_str_to_dt(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        target_date=_str_to_dt(data["target_date"]) if data.get("target_date") else None,
        completed_at=_str_to_dt(data["completed_at"]) if data.get("completed_at") else None,
        progress_pct=data.get("progress_pct", 0),
        metadata=dict(data.get("metadata", {})),
    )


# ---------------------------------------------------------------------------
# PreferenceProfile
# ---------------------------------------------------------------------------

def preference_profile_to_dict(profile: PreferenceProfile) -> dict[str, Any]:
    """Serialise a :class:`PreferenceProfile` to a plain dictionary."""
    c = profile.communication
    l = profile.learning
    d = profile.development
    w = profile.workflow
    return {
        "communication": {
            "preferred_detail_level": c.preferred_detail_level,
            "prefers_examples": c.prefers_examples,
            "prefers_analogies": c.prefers_analogies,
            "preferred_tone": c.preferred_tone,
            "structured_output_preferred": c.structured_output_preferred,
        },
        "learning": {
            "preferred_approach": l.preferred_approach,
            "prefers_big_picture_first": l.prefers_big_picture_first,
            "learns_by_building": l.learns_by_building,
            "preferred_learning_pace": l.preferred_learning_pace,
            "depth_over_breadth": l.depth_over_breadth,
        },
        "development": {
            "preferred_languages": list(d.preferred_languages),
            "architecture_first": d.architecture_first,
            "prefers_modularity": d.prefers_modularity,
            "testing_discipline": d.testing_discipline,
            "documentation_style": d.documentation_style,
        },
        "workflow": {
            "deep_work_sessions": w.deep_work_sessions,
            "preferred_session_length_hours": w.preferred_session_length_hours,
            "async_communication_preferred": w.async_communication_preferred,
            "batch_decisions": w.batch_decisions,
            "peak_hours": list(w.peak_hours),
        },
        "updated_at": _dt_to_str(profile.updated_at),
        "version": profile.version,
        "metadata": dict(profile.metadata),
    }


def preference_profile_from_dict(data: dict[str, Any]) -> PreferenceProfile:
    """Deserialise a :class:`PreferenceProfile` from a dictionary."""
    c_data = data.get("communication", {})
    l_data = data.get("learning", {})
    d_data = data.get("development", {})
    w_data = data.get("workflow", {})

    comm = CommunicationPreferences(
        preferred_detail_level=c_data.get("preferred_detail_level", "detailed"),
        prefers_examples=c_data.get("prefers_examples", True),
        prefers_analogies=c_data.get("prefers_analogies", True),
        preferred_tone=c_data.get("preferred_tone", "collaborative"),
        structured_output_preferred=c_data.get("structured_output_preferred", True),
    )
    learn = LearningPreferences(
        preferred_approach=l_data.get("preferred_approach", "systems_first"),
        prefers_big_picture_first=l_data.get("prefers_big_picture_first", True),
        learns_by_building=l_data.get("learns_by_building", True),
        preferred_learning_pace=l_data.get("preferred_learning_pace", "fast"),
        depth_over_breadth=l_data.get("depth_over_breadth", False),
    )
    dev = DevelopmentPreferences(
        preferred_languages=list(d_data.get("preferred_languages", [])),
        architecture_first=d_data.get("architecture_first", True),
        prefers_modularity=d_data.get("prefers_modularity", True),
        testing_discipline=d_data.get("testing_discipline", "pragmatic"),
        documentation_style=d_data.get("documentation_style", "inline"),
    )
    work = WorkflowPreferences(
        deep_work_sessions=w_data.get("deep_work_sessions", True),
        preferred_session_length_hours=w_data.get("preferred_session_length_hours", 3.0),
        async_communication_preferred=w_data.get("async_communication_preferred", True),
        batch_decisions=w_data.get("batch_decisions", False),
        peak_hours=list(w_data.get("peak_hours", [])),
    )

    return PreferenceProfile(
        communication=comm,
        learning=learn,
        development=dev,
        workflow=work,
        updated_at=_str_to_dt(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        version=data.get("version", 1),
        metadata=dict(data.get("metadata", {})),
    )


# ---------------------------------------------------------------------------
# CapabilitySnapshot
# ---------------------------------------------------------------------------

def _cap_entry_to_dict(entry: CapabilityEntry) -> dict[str, Any]:
    return {
        "name": entry.name,
        "domain": entry.domain,
        "confidence": entry.confidence,
        "evidence_count": entry.evidence_count,
        "notes": entry.notes,
        "observed_at": _dt_to_str(entry.observed_at),
    }


def _cap_entry_from_dict(data: dict[str, Any]) -> CapabilityEntry:
    return CapabilityEntry(
        name=data["name"],
        domain=data["domain"],
        confidence=data["confidence"],
        evidence_count=data.get("evidence_count", 0),
        notes=data.get("notes", ""),
        observed_at=_str_to_dt(data["observed_at"]) if "observed_at" in data else datetime.now(timezone.utc),
    )


def capability_snapshot_to_dict(snapshot: CapabilitySnapshot) -> dict[str, Any]:
    """Serialise a :class:`CapabilitySnapshot`."""
    return {
        "strengths": [_cap_entry_to_dict(e) for e in snapshot.strengths],
        "growth_areas": [_cap_entry_to_dict(e) for e in snapshot.growth_areas],
        "snapshot_at": _dt_to_str(snapshot.snapshot_at),
    }


def capability_snapshot_from_dict(data: dict[str, Any]) -> CapabilitySnapshot:
    """Deserialise a :class:`CapabilitySnapshot`."""
    return CapabilitySnapshot(
        strengths=[_cap_entry_from_dict(e) for e in data.get("strengths", [])],
        growth_areas=[_cap_entry_from_dict(e) for e in data.get("growth_areas", [])],
        snapshot_at=_str_to_dt(data["snapshot_at"]) if "snapshot_at" in data else datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# FutureSelfReference
# ---------------------------------------------------------------------------

def future_self_ref_to_dict(ref: FutureSelfReference) -> dict[str, Any]:
    """Serialise a :class:`FutureSelfReference`."""
    return {
        "horizon_label": ref.horizon_label,
        "horizon_years": ref.horizon_years,
        "headline_identity": ref.headline_identity,
        "projected_goals": list(ref.projected_goals),
        "growth_trajectory": dict(ref.growth_trajectory),
        "confidence": ref.confidence,
        "last_updated": _dt_to_str(ref.last_updated),
        "metadata": dict(ref.metadata),
    }


def future_self_ref_from_dict(data: dict[str, Any]) -> FutureSelfReference:
    """Deserialise a :class:`FutureSelfReference`."""
    return FutureSelfReference(
        horizon_label=data["horizon_label"],
        horizon_years=data["horizon_years"],
        headline_identity=data["headline_identity"],
        projected_goals=list(data.get("projected_goals", [])),
        growth_trajectory=dict(data.get("growth_trajectory", {})),
        confidence=data.get("confidence", 0.5),
        last_updated=_str_to_dt(data["last_updated"]) if "last_updated" in data else datetime.now(timezone.utc),
        metadata=dict(data.get("metadata", {})),
    )


# ---------------------------------------------------------------------------
# DigitalTwinState
# ---------------------------------------------------------------------------

def digital_twin_state_to_dict(twin: DigitalTwinState) -> dict[str, Any]:
    """Serialise a :class:`DigitalTwinState` to a plain dictionary."""
    return {
        "identity": identity_profile_to_dict(twin.identity),
        "goals": [goal_to_dict(g) for g in twin.goals],
        "preferences": preference_profile_to_dict(twin.preferences),
        "capabilities": capability_snapshot_to_dict(twin.capabilities),
        "growth_indicators": dict(twin.growth_indicators),
        "future_self_refs": [future_self_ref_to_dict(r) for r in twin.future_self_refs],
        "consistency_score": twin.consistency_score,
        "generated_at": _dt_to_str(twin.generated_at),
        "twin_version": twin.twin_version,
        "metadata": dict(twin.metadata),
    }


# ---------------------------------------------------------------------------
# EvolutionRecord
# ---------------------------------------------------------------------------

def evolution_record_to_dict(record: EvolutionRecord) -> dict[str, Any]:
    """Serialise an :class:`EvolutionRecord`."""
    return {
        "record_id": record.record_id,
        "trigger": record.trigger.name,
        "changed_fields": list(record.changed_fields),
        "previous_snapshot": dict(record.previous_snapshot),
        "new_snapshot": dict(record.new_snapshot),
        "confidence": record.confidence,
        "evidence_count": record.evidence_count,
        "notes": record.notes,
        "evolved_at": _dt_to_str(record.evolved_at),
    }


# ---------------------------------------------------------------------------
# ConsistencyReport
# ---------------------------------------------------------------------------

def consistency_report_to_dict(report: ConsistencyReport) -> dict[str, Any]:
    """Serialise a :class:`ConsistencyReport`."""
    return {
        "overall_score": report.overall_score,
        "dimension_scores": dict(report.dimension_scores),
        "drift_detected": report.drift_detected,
        "drift_fields": list(report.drift_fields),
        "evidence_items": [
            {
                "source": e.source,
                "signal_type": e.signal_type,
                "weight": e.weight,
                "description": e.description,
                "observed_at": _dt_to_str(e.observed_at),
            }
            for e in report.evidence_items
        ],
        "recommendation": report.recommendation,
        "checked_at": _dt_to_str(report.checked_at),
    }