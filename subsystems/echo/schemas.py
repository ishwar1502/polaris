# subsystems/echo/schemas.py
"""
ECHO v1 Serialisation Schemas.

Provides ``to_dict`` / ``from_dict`` round-trip serialisation for all ECHO
domain models.  These functions are the canonical serialisation layer used
by the Memory Gateway integration, the public API response formatting, and
the Memory Consolidation Engine's persistence pipeline.

All functions are pure (no side effects) and raise :class:`ValueError` on
invalid or missing required input rather than failing silently.

Functions
---------
Experience
    :func:`experience_to_dict` / :func:`experience_from_dict`

EventRecord
    :func:`event_record_to_dict` / :func:`event_record_from_dict`

AchievementRecord
    :func:`achievement_record_to_dict` / :func:`achievement_record_from_dict`

FailureRecord
    :func:`failure_record_to_dict` / :func:`failure_record_from_dict`

ObservationRecord
    :func:`observation_record_to_dict` / :func:`observation_record_from_dict`

Supporting types
    :func:`memory_tag_to_dict` / :func:`memory_tag_from_dict`
    :func:`experience_metadata_to_dict` / :func:`experience_metadata_from_dict`
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from subsystems.echo.models import (
    AchievementRecord,
    EventRecord,
    Experience,
    ExperienceImportance,
    ExperienceMetadata,
    ExperienceType,
    FailureRecord,
    MemoryTag,
    ObservationRecord,
)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _dt_to_str(dt: datetime) -> str:
    """Serialise a :class:`datetime` to an ISO-8601 string."""
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    """Deserialise an ISO-8601 string to a timezone-aware :class:`datetime`."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _opt_dt_to_str(dt: datetime | None) -> str | None:
    """Serialise an optional :class:`datetime`."""
    return _dt_to_str(dt) if dt is not None else None


def _opt_str_to_dt(s: str | None) -> datetime | None:
    """Deserialise an optional ISO-8601 string."""
    return _str_to_dt(s) if s is not None else None


# ---------------------------------------------------------------------------
# MemoryTag
# ---------------------------------------------------------------------------


def memory_tag_to_dict(tag: MemoryTag) -> dict[str, Any]:
    """Serialise a :class:`MemoryTag` to a plain dictionary."""
    return {
        "name": tag.name,
        "category": tag.category,
    }


def memory_tag_from_dict(data: dict[str, Any]) -> MemoryTag:
    """Deserialise a :class:`MemoryTag` from a dictionary."""
    return MemoryTag(
        name=data["name"],
        category=data.get("category", "custom"),
    )


# ---------------------------------------------------------------------------
# ExperienceMetadata
# ---------------------------------------------------------------------------


def experience_metadata_to_dict(meta: ExperienceMetadata) -> dict[str, Any]:
    """Serialise an :class:`ExperienceMetadata` to a plain dictionary."""
    return {
        "source_subsystem": meta.source_subsystem,
        "session_id": meta.session_id,
        "related_experience_ids": list(meta.related_experience_ids),
        "project_refs": list(meta.project_refs),
        "significance_score": meta.significance_score,
        "consolidated": meta.consolidated,
        "consolidation_at": _opt_dt_to_str(meta.consolidation_at),
        "retrieval_count": meta.retrieval_count,
        "last_retrieved_at": _opt_dt_to_str(meta.last_retrieved_at),
    }


def experience_metadata_from_dict(data: dict[str, Any]) -> ExperienceMetadata:
    """Deserialise an :class:`ExperienceMetadata` from a dictionary."""
    return ExperienceMetadata(
        source_subsystem=data.get("source_subsystem", "ECHO_API"),
        session_id=data.get("session_id"),
        related_experience_ids=list(data.get("related_experience_ids", [])),
        project_refs=list(data.get("project_refs", [])),
        significance_score=float(data.get("significance_score", 0.0)),
        consolidated=bool(data.get("consolidated", False)),
        consolidation_at=_opt_str_to_dt(data.get("consolidation_at")),
        retrieval_count=int(data.get("retrieval_count", 0)),
        last_retrieved_at=_opt_str_to_dt(data.get("last_retrieved_at")),
    )


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------


def experience_to_dict(exp: Experience) -> dict[str, Any]:
    """Serialise an :class:`Experience` to a plain dictionary."""
    return {
        "experience_id": exp.experience_id,
        "title": exp.title,
        "experience_type": exp.experience_type.name,
        "importance": exp.importance.name,
        "description": exp.description,
        "context": exp.context,
        "outcome": exp.outcome,
        "tags": [memory_tag_to_dict(t) for t in exp.tags],
        "occurred_at": _dt_to_str(exp.occurred_at),
        "recorded_at": _dt_to_str(exp.recorded_at),
        "metadata": experience_metadata_to_dict(exp.metadata),
        "extra": dict(exp.extra),
    }


def experience_from_dict(data: dict[str, Any]) -> Experience:
    """Deserialise an :class:`Experience` from a dictionary."""
    return Experience(
        experience_id=data["experience_id"],
        title=data["title"],
        experience_type=ExperienceType[data["experience_type"]],
        importance=ExperienceImportance[data["importance"]],
        description=data.get("description", ""),
        context=data.get("context", ""),
        outcome=data.get("outcome", ""),
        tags=[memory_tag_from_dict(t) for t in data.get("tags", [])],
        occurred_at=_str_to_dt(data["occurred_at"]),
        recorded_at=_str_to_dt(data["recorded_at"]),
        metadata=experience_metadata_from_dict(data.get("metadata", {})),
        extra=dict(data.get("extra", {})),
    )


# ---------------------------------------------------------------------------
# EventRecord
# ---------------------------------------------------------------------------


def event_record_to_dict(event: EventRecord) -> dict[str, Any]:
    """Serialise an :class:`EventRecord` to a plain dictionary."""
    return {
        "event_id": event.event_id,
        "event_name": event.event_name,
        "experience_id": event.experience_id,
        "payload": dict(event.payload),
        "importance": event.importance.name,
        "source_subsystem": event.source_subsystem,
        "occurred_at": _dt_to_str(event.occurred_at),
        "recorded_at": _dt_to_str(event.recorded_at),
    }


def event_record_from_dict(data: dict[str, Any]) -> EventRecord:
    """Deserialise an :class:`EventRecord` from a dictionary."""
    return EventRecord(
        event_id=data["event_id"],
        event_name=data["event_name"],
        experience_id=data.get("experience_id"),
        payload=dict(data.get("payload", {})),
        importance=ExperienceImportance[data.get("importance", "MEDIUM")],
        source_subsystem=data.get("source_subsystem", "ECHO_API"),
        occurred_at=_str_to_dt(data["occurred_at"]),
        recorded_at=_str_to_dt(data["recorded_at"]),
    )


# ---------------------------------------------------------------------------
# AchievementRecord
# ---------------------------------------------------------------------------


def achievement_record_to_dict(ach: AchievementRecord) -> dict[str, Any]:
    """Serialise an :class:`AchievementRecord` to a plain dictionary."""
    return {
        "achievement_id": ach.achievement_id,
        "title": ach.title,
        "experience_id": ach.experience_id,
        "domain": ach.domain,
        "description": ach.description,
        "evidence": list(ach.evidence),
        "importance": ach.importance.name,
        "tags": [memory_tag_to_dict(t) for t in ach.tags],
        "achieved_at": _dt_to_str(ach.achieved_at),
        "recorded_at": _dt_to_str(ach.recorded_at),
    }


def achievement_record_from_dict(data: dict[str, Any]) -> AchievementRecord:
    """Deserialise an :class:`AchievementRecord` from a dictionary."""
    return AchievementRecord(
        achievement_id=data["achievement_id"],
        title=data["title"],
        experience_id=data.get("experience_id"),
        domain=data.get("domain", "general"),
        description=data.get("description", ""),
        evidence=list(data.get("evidence", [])),
        importance=ExperienceImportance[data.get("importance", "HIGH")],
        tags=[memory_tag_from_dict(t) for t in data.get("tags", [])],
        achieved_at=_str_to_dt(data["achieved_at"]),
        recorded_at=_str_to_dt(data["recorded_at"]),
    )


# ---------------------------------------------------------------------------
# FailureRecord
# ---------------------------------------------------------------------------


def failure_record_to_dict(fail: FailureRecord) -> dict[str, Any]:
    """Serialise a :class:`FailureRecord` to a plain dictionary."""
    return {
        "failure_id": fail.failure_id,
        "title": fail.title,
        "experience_id": fail.experience_id,
        "domain": fail.domain,
        "description": fail.description,
        "contributing_factors": list(fail.contributing_factors),
        "lesson": fail.lesson,
        "importance": fail.importance.name,
        "tags": [memory_tag_to_dict(t) for t in fail.tags],
        "reflection_generated": fail.reflection_generated,
        "failed_at": _dt_to_str(fail.failed_at),
        "recorded_at": _dt_to_str(fail.recorded_at),
    }


def failure_record_from_dict(data: dict[str, Any]) -> FailureRecord:
    """Deserialise a :class:`FailureRecord` from a dictionary."""
    return FailureRecord(
        failure_id=data["failure_id"],
        title=data["title"],
        experience_id=data.get("experience_id"),
        domain=data.get("domain", "general"),
        description=data.get("description", ""),
        contributing_factors=list(data.get("contributing_factors", [])),
        lesson=data.get("lesson", ""),
        importance=ExperienceImportance[data.get("importance", "MEDIUM")],
        tags=[memory_tag_from_dict(t) for t in data.get("tags", [])],
        reflection_generated=bool(data.get("reflection_generated", False)),
        failed_at=_str_to_dt(data["failed_at"]),
        recorded_at=_str_to_dt(data["recorded_at"]),
    )


# ---------------------------------------------------------------------------
# ObservationRecord
# ---------------------------------------------------------------------------


def observation_record_to_dict(obs: ObservationRecord) -> dict[str, Any]:
    """Serialise an :class:`ObservationRecord` to a plain dictionary."""
    return {
        "observation_id": obs.observation_id,
        "summary": obs.summary,
        "experience_id": obs.experience_id,
        "domain": obs.domain,
        "detail": obs.detail,
        "evidence_refs": list(obs.evidence_refs),
        "importance": obs.importance.name,
        "observed_at": _dt_to_str(obs.observed_at),
        "recorded_at": _dt_to_str(obs.recorded_at),
    }


def observation_record_from_dict(data: dict[str, Any]) -> ObservationRecord:
    """Deserialise an :class:`ObservationRecord` from a dictionary."""
    return ObservationRecord(
        observation_id=data["observation_id"],
        summary=data["summary"],
        experience_id=data.get("experience_id"),
        domain=data.get("domain", "general"),
        detail=data.get("detail", ""),
        evidence_refs=list(data.get("evidence_refs", [])),
        importance=ExperienceImportance[data.get("importance", "LOW")],
        observed_at=_str_to_dt(data["observed_at"]),
        recorded_at=_str_to_dt(data["recorded_at"]),
    )