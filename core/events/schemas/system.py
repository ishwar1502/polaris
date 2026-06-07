# core/events/schemas/system.py
"""
Canonical system-level event type constants and payload schemas for
the POLARIS v5 runtime.

These events are emitted by the runtime itself (registry, lifecycle manager,
health monitor) and consumed by observability subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from core.events.event import EventType
from core.events.exceptions import EventValidationError


# ---------------------------------------------------------------------------
# System event type constants
# ---------------------------------------------------------------------------


class SystemEvents:
    """Namespace of canonical :class:`~core.events.event.EventType` constants
    for runtime-level events.

    Naming convention: ``polaris.runtime.<domain>.<verb>``
    """

    # Subsystem lifecycle
    SUBSYSTEM_REGISTERED: Final[EventType] = EventType(
        "polaris.runtime.subsystem.registered"
    )
    SUBSYSTEM_UNREGISTERED: Final[EventType] = EventType(
        "polaris.runtime.subsystem.unregistered"
    )
    SUBSYSTEM_INITIALIZED: Final[EventType] = EventType(
        "polaris.runtime.subsystem.initialized"
    )
    SUBSYSTEM_STARTED: Final[EventType] = EventType(
        "polaris.runtime.subsystem.started"
    )
    SUBSYSTEM_PAUSED: Final[EventType] = EventType(
        "polaris.runtime.subsystem.paused"
    )
    SUBSYSTEM_RESUMED: Final[EventType] = EventType(
        "polaris.runtime.subsystem.resumed"
    )
    SUBSYSTEM_STOPPED: Final[EventType] = EventType(
        "polaris.runtime.subsystem.stopped"
    )
    SUBSYSTEM_FAILED: Final[EventType] = EventType(
        "polaris.runtime.subsystem.failed"
    )

    # Health
    SUBSYSTEM_HEALTH_CHANGED: Final[EventType] = EventType(
        "polaris.runtime.health.changed"
    )

    # Bus
    BUS_STARTED: Final[EventType] = EventType(
        "polaris.runtime.bus.started"
    )
    BUS_STOPPED: Final[EventType] = EventType(
        "polaris.runtime.bus.stopped"
    )

    # Dependency
    DEPENDENCY_SATISFIED: Final[EventType] = EventType(
        "polaris.runtime.dependency.satisfied"
    )
    DEPENDENCY_FAILED: Final[EventType] = EventType(
        "polaris.runtime.dependency.failed"
    )


# ---------------------------------------------------------------------------
# Payload schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsystemRegisteredPayload:
    """Payload for :attr:`SystemEvents.SUBSYSTEM_REGISTERED`.

    Attributes
    ----------
    subsystem_id:
        The registered subsystem's id.
    version:
        The subsystem's declared version string.
    capability_count:
        Number of capabilities declared by the subsystem.
    """

    subsystem_id: str
    version: str
    capability_count: int

    def __post_init__(self) -> None:
        if self.capability_count < 0:
            raise EventValidationError(
                "SubsystemRegisteredPayload.capability_count cannot be negative.",
                field="capability_count",
                invalid_value=str(self.capability_count),
            )


@dataclass(frozen=True)
class SubsystemUnregisteredPayload:
    """Payload for :attr:`SystemEvents.SUBSYSTEM_UNREGISTERED`.

    Attributes
    ----------
    subsystem_id:
        The unregistered subsystem's id.
    forced:
        Whether the unregistration was forced (``force_stop=True``).
    """

    subsystem_id: str
    forced: bool = False


@dataclass(frozen=True)
class SubsystemLifecyclePayload:
    """Payload for subsystem lifecycle transition events.

    Used by: ``INITIALIZED``, ``STARTED``, ``PAUSED``, ``RESUMED``,
    ``STOPPED``, ``FAILED``.

    Attributes
    ----------
    subsystem_id:
        The subsystem that transitioned.
    from_state:
        Previous lifecycle state name.
    to_state:
        New lifecycle state name.
    reason:
        Optional human-readable reason for the transition.
    """

    subsystem_id: str
    from_state: str
    to_state: str
    reason: str = ""


@dataclass(frozen=True)
class SubsystemHealthChangedPayload:
    """Payload for :attr:`SystemEvents.SUBSYSTEM_HEALTH_CHANGED`.

    Attributes
    ----------
    subsystem_id:
        The subsystem whose health changed.
    previous_status:
        Previous :class:`~core.contracts.health.HealthStatus` name.
    current_status:
        New :class:`~core.contracts.health.HealthStatus` name.
    message:
        Human-readable description from the health report.
    failed_check_count:
        Number of individual checks that failed.
    """

    subsystem_id: str
    previous_status: str
    current_status: str
    message: str
    failed_check_count: int = 0