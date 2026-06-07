# core/lifecycle/events.py
"""
POLARIS v5 Lifecycle Manager — Event type constants and payloads.

Defines the canonical :class:`LifecycleEvents` namespace and the
:class:`LifecycleEventPayload` dataclass used for all lifecycle events
emitted by the :class:`~core.lifecycle.manager.LifecycleManager`.

All lifecycle event types follow the naming convention::

    polaris.lifecycle.<verb>

and integrate directly with the POLARIS Event Bus via
:class:`~core.events.event.Event`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from core.events.event import EventType


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------


class LifecycleEvents:
    """Namespace of canonical :class:`~core.events.event.EventType` constants
    for lifecycle manager events.

    Naming convention: ``polaris.lifecycle.<verb>``
    """

    MODULE_LOADED: Final[EventType] = EventType("polaris.lifecycle.loaded")
    """Emitted after a module has been successfully loaded (state → LOADED)."""

    MODULE_INITIALIZED: Final[EventType] = EventType("polaris.lifecycle.initialized")
    """Emitted after a module has been successfully initialized (state → INITIALIZED)."""

    MODULE_STARTED: Final[EventType] = EventType("polaris.lifecycle.started")
    """Emitted after a module has transitioned to RUNNING."""

    MODULE_PAUSED: Final[EventType] = EventType("polaris.lifecycle.paused")
    """Emitted after a module has been paused."""

    MODULE_RESUMED: Final[EventType] = EventType("polaris.lifecycle.resumed")
    """Emitted after a module has been resumed (PAUSED → RUNNING)."""

    MODULE_RECOVERED: Final[EventType] = EventType("polaris.lifecycle.recovered")
    """Emitted after a module has successfully recovered from FAILED → RUNNING."""

    MODULE_STOPPED: Final[EventType] = EventType("polaris.lifecycle.stopped")
    """Emitted after a module has been stopped (state → STOPPED)."""

    MODULE_FAILED: Final[EventType] = EventType("polaris.lifecycle.failed")
    """Emitted after a module transitions to FAILED state."""

    MODULE_UNLOADED: Final[EventType] = EventType("polaris.lifecycle.unloaded")
    """Emitted after a module has been fully unloaded (state → UNLOADED)."""

    # Convenience tuple of all event types for subscription patterns.
    ALL: Final[tuple[EventType, ...]] = (
        MODULE_LOADED,
        MODULE_INITIALIZED,
        MODULE_STARTED,
        MODULE_PAUSED,
        MODULE_RESUMED,
        MODULE_RECOVERED,
        MODULE_STOPPED,
        MODULE_FAILED,
        MODULE_UNLOADED,
    )


# ---------------------------------------------------------------------------
# Event payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleEventPayload:
    """Payload carried by all events emitted by the Lifecycle Manager.

    Attributes
    ----------
    module_id:
        Identifier of the module that changed state.
    from_state:
        State name *before* the transition.  May be ``None`` for the very
        first event emitted for a module (initial state).
    to_state:
        State name *after* the transition.
    reason:
        Optional human-readable explanation for the transition.
    extra:
        Optional dictionary of additional context (error message, recovery
        attempt count, etc.).
    """

    module_id: str
    from_state: str | None
    to_state: str
    reason: str = ""
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Validate required string fields
        if not self.module_id or not self.module_id.strip():
            raise ValueError(
                "LifecycleEventPayload.module_id must be a non-empty string."
            )
        if not self.to_state or not self.to_state.strip():
            raise ValueError(
                "LifecycleEventPayload.to_state must be a non-empty string."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "module_id": self.module_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
            "extra": self.extra or {},
        }


__all__ = [
    "LifecycleEvents",
    "LifecycleEventPayload",
]