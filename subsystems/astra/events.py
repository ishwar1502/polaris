# subsystems/astra/events.py
"""
ASTRA v5 Event Definitions.

All events published by ASTRA are constructed via the factory functions in
this module, which ensure consistent event types, source identifiers, and
payload schemas.

Event type namespace: ``polaris.astra.*``

Published events
----------------
* :func:`identity_updated`          — identity profile changed
* :func:`goal_created`              — new goal registered
* :func:`goal_updated`              — goal metadata or state changed
* :func:`goal_removed`              — goal permanently removed
* :func:`preference_changed`        — preference profile updated
* :func:`digital_twin_updated`      — new digital twin generated
* :func:`consistency_check_completed` — consistency analysis finished
* :func:`identity_evolved`          — evidence-driven identity evolution
"""

from __future__ import annotations

from typing import Any

from core.events.event import Event, EventPriority

# ASTRA subsystem source identifier
ASTRA_SOURCE: str = "polaris.astra"

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

ET_IDENTITY_UPDATED: str = "polaris.astra.identity_updated"
ET_GOAL_CREATED: str = "polaris.astra.goal_created"
ET_GOAL_UPDATED: str = "polaris.astra.goal_updated"
ET_GOAL_REMOVED: str = "polaris.astra.goal_removed"
ET_PREFERENCE_CHANGED: str = "polaris.astra.preference_changed"
ET_DIGITAL_TWIN_UPDATED: str = "polaris.astra.digital_twin_updated"
ET_CONSISTENCY_CHECK_COMPLETED: str = "polaris.astra.consistency_check_completed"
ET_IDENTITY_EVOLVED: str = "polaris.astra.identity_evolved"


# ---------------------------------------------------------------------------
# Event factory functions
# ---------------------------------------------------------------------------


def identity_updated(
    *,
    changed_fields: list[str],
    identity_version: int,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create an ``identity_updated`` event.

    Parameters
    ----------
    changed_fields:
        List of field names that changed.
    identity_version:
        New version of the identity profile.
    metadata:
        Optional additional annotations.
    """
    return Event.create(
        event_type=ET_IDENTITY_UPDATED,
        source=ASTRA_SOURCE,
        payload={
            "changed_fields": changed_fields,
            "identity_version": identity_version,
        },
        priority=EventPriority.NORMAL,
        metadata=metadata or {},
    )


def goal_created(
    *,
    goal_id: str,
    title: str,
    goal_type: str,
    priority: str,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create a ``goal_created`` event.

    Parameters
    ----------
    goal_id:
        UUID of the newly created goal.
    title:
        Human-readable goal title.
    goal_type:
        :class:`~subsystems.astra.models.GoalType` name.
    priority:
        :class:`~subsystems.astra.models.GoalPriority` name.
    metadata:
        Optional additional annotations.
    """
    return Event.create(
        event_type=ET_GOAL_CREATED,
        source=ASTRA_SOURCE,
        payload={
            "goal_id": goal_id,
            "title": title,
            "goal_type": goal_type,
            "priority": priority,
        },
        priority=EventPriority.NORMAL,
        metadata=metadata or {},
    )


def goal_updated(
    *,
    goal_id: str,
    changed_fields: list[str],
    new_state: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create a ``goal_updated`` event.

    Parameters
    ----------
    goal_id:
        UUID of the updated goal.
    changed_fields:
        Fields that changed.
    new_state:
        New :class:`~subsystems.astra.models.GoalState` name if state changed.
    metadata:
        Optional additional annotations.
    """
    return Event.create(
        event_type=ET_GOAL_UPDATED,
        source=ASTRA_SOURCE,
        payload={
            "goal_id": goal_id,
            "changed_fields": changed_fields,
            "new_state": new_state,
        },
        priority=EventPriority.NORMAL,
        metadata=metadata or {},
    )


def goal_removed(
    *,
    goal_id: str,
    title: str,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create a ``goal_removed`` event.

    Parameters
    ----------
    goal_id:
        UUID of the removed goal.
    title:
        Title of the removed goal (for audit purposes).
    metadata:
        Optional additional annotations.
    """
    return Event.create(
        event_type=ET_GOAL_REMOVED,
        source=ASTRA_SOURCE,
        payload={
            "goal_id": goal_id,
            "title": title,
        },
        priority=EventPriority.NORMAL,
        metadata=metadata or {},
    )


def preference_changed(
    *,
    changed_dimensions: list[str],
    preference_version: int,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create a ``preference_changed`` event.

    Parameters
    ----------
    changed_dimensions:
        Preference dimensions that changed (e.g. ``["communication", "learning"]``).
    preference_version:
        New version of the preference profile.
    metadata:
        Optional additional annotations.
    """
    return Event.create(
        event_type=ET_PREFERENCE_CHANGED,
        source=ASTRA_SOURCE,
        payload={
            "changed_dimensions": changed_dimensions,
            "preference_version": preference_version,
        },
        priority=EventPriority.NORMAL,
        metadata=metadata or {},
    )


def digital_twin_updated(
    *,
    twin_version: int,
    consistency_score: float,
    active_goal_count: int,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create a ``digital_twin_updated`` event.

    Parameters
    ----------
    twin_version:
        New twin generation number.
    consistency_score:
        Current consistency score (0.0-1.0).
    active_goal_count:
        Number of active goals in the twin.
    metadata:
        Optional additional annotations.
    """
    return Event.create(
        event_type=ET_DIGITAL_TWIN_UPDATED,
        source=ASTRA_SOURCE,
        payload={
            "twin_version": twin_version,
            "consistency_score": consistency_score,
            "active_goal_count": active_goal_count,
        },
        priority=EventPriority.NORMAL,
        metadata=metadata or {},
    )


def consistency_check_completed(
    *,
    overall_score: float,
    drift_detected: bool,
    drift_fields: list[str],
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create a ``consistency_check_completed`` event.

    Parameters
    ----------
    overall_score:
        Computed consistency score (0.0-1.0).
    drift_detected:
        Whether identity drift was detected.
    drift_fields:
        Fields where drift was observed.
    metadata:
        Optional additional annotations.
    """
    return Event.create(
        event_type=ET_CONSISTENCY_CHECK_COMPLETED,
        source=ASTRA_SOURCE,
        payload={
            "overall_score": overall_score,
            "drift_detected": drift_detected,
            "drift_fields": drift_fields,
        },
        priority=EventPriority.NORMAL,
        metadata=metadata or {},
    )


def identity_evolved(
    *,
    record_id: str,
    trigger: str,
    changed_fields: list[str],
    confidence: float,
    evidence_count: int,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create an ``identity_evolved`` event.

    Parameters
    ----------
    record_id:
        UUID of the :class:`~subsystems.astra.models.EvolutionRecord`.
    trigger:
        :class:`~subsystems.astra.models.EvolutionTrigger` name.
    changed_fields:
        Fields that evolved.
    confidence:
        Confidence in the evolution.
    evidence_count:
        Evidence items that justified the evolution.
    metadata:
        Optional additional annotations.
    """
    return Event.create(
        event_type=ET_IDENTITY_EVOLVED,
        source=ASTRA_SOURCE,
        payload={
            "record_id": record_id,
            "trigger": trigger,
            "changed_fields": changed_fields,
            "confidence": confidence,
            "evidence_count": evidence_count,
        },
        priority=EventPriority.HIGH,
        metadata=metadata or {},
    )
    