# core/events/event.py
"""
Core event primitives for the POLARIS v5 Event Bus.

Events are the sole communication channel between subsystems.
Every piece of inter-subsystem data must be wrapped in an :class:`Event`
and routed through the bus.  Events are **immutable** once created.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, unique
from typing import Any, Final, NewType

from core.events.exceptions import EventValidationError


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EventType = NewType("EventType", str)
"""Dot-namespaced event type string, e.g. ``polaris.memory.record_stored``.

Format: ``<subsystem_ns>.<domain>.<verb>`` (all lowercase, dots as separators).
"""

EventId = NewType("EventId", str)
"""UUID-4 string uniquely identifying a single event instance."""


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------


@unique
class EventPriority(Enum):
    """Delivery priority for an :class:`Event`.

    The dispatcher processes higher-priority events before lower-priority
    ones within the same dispatch cycle.

    Ordering: ``CRITICAL > HIGH > NORMAL > LOW``
    """

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

    def __lt__(self, other: "EventPriority") -> bool:
        if not isinstance(other, EventPriority):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: "EventPriority") -> bool:
        if not isinstance(other, EventPriority):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: "EventPriority") -> bool:
        if not isinstance(other, EventPriority):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: "EventPriority") -> bool:
        if not isinstance(other, EventPriority):
            return NotImplemented
        return self.value >= other.value


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_EVENT_TYPE_MAX_LEN: Final[int] = 256
_SOURCE_MAX_LEN: Final[int] = 256
_METADATA_MAX_KEYS: Final[int] = 64
_SENTINEL: Final[object] = object()


def _validate_event_type(value: str) -> None:
    """Assert that *value* is a valid :class:`EventType` string.

    Raises
    ------
    EventValidationError
        If the format is invalid.
    """
    if not value or not value.strip():
        raise EventValidationError(
            "event_type must be a non-empty string.",
            field="event_type",
            invalid_value=repr(value),
        )
    if len(value) > _EVENT_TYPE_MAX_LEN:
        raise EventValidationError(
            f"event_type exceeds maximum length of {_EVENT_TYPE_MAX_LEN}.",
            field="event_type",
            invalid_value=value[:64] + "...",
        )
    parts = value.split(".")
    if len(parts) < 2:
        raise EventValidationError(
            "event_type must have at least two dot-separated segments "
            "(e.g. 'polaris.subsystem.verb').",
            field="event_type",
            invalid_value=value,
        )
    for part in parts:
        if not part or not part.replace("_", "").replace("-", "").isalnum():
            raise EventValidationError(
                f"event_type segment {part!r} contains invalid characters. "
                "Segments must be alphanumeric (underscores and hyphens allowed).",
                field="event_type",
                invalid_value=value,
            )


def _validate_source(value: str) -> None:
    """Assert that *value* is a non-empty subsystem source identifier.

    Raises
    ------
    EventValidationError
    """
    if not value or not value.strip():
        raise EventValidationError(
            "source must be a non-empty string.",
            field="source",
            invalid_value=repr(value),
        )
    if len(value) > _SOURCE_MAX_LEN:
        raise EventValidationError(
            f"source exceeds maximum length of {_SOURCE_MAX_LEN}.",
            field="source",
            invalid_value=value[:64] + "...",
        )


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """Immutable message transmitted through the POLARIS Event Bus.

    Attributes
    ----------
    event_type:
        Dot-namespaced type string that identifies what occurred
        (e.g. ``polaris.memory.record_stored``).
    source:
        :class:`~core.types.identifiers.SubsystemId` (or any string id) of
        the subsystem that produced this event.
    payload:
        Arbitrary event data.  Must be serialisable if persistence is
        required; the bus itself imposes no constraints here.
    event_id:
        UUID-4 string; auto-generated if not supplied.
    timestamp:
        UTC :class:`~datetime.datetime` of creation; auto-set if not supplied.
    priority:
        Delivery priority; defaults to :attr:`EventPriority.NORMAL`.
    metadata:
        Optional key-value annotations (correlation ids, trace ids, etc.).
        Stored as a frozen mapping (``tuple[tuple[str,Any],...]``) to
        preserve immutability; use :meth:`get_metadata` for dict access.
    correlation_id:
        Optional id linking this event to a causal chain or transaction.
    causation_id:
        Optional :attr:`event_id` of the event that directly caused this one.
    """

    event_type: EventType
    source: str
    payload: Any
    event_id: EventId = field(
        default_factory=lambda: EventId(str(uuid.uuid4()))
    )
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    priority: EventPriority = EventPriority.NORMAL
    # Stored as tuple-of-pairs to remain hashable and frozen.
    _metadata_raw: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    correlation_id: str | None = None
    causation_id: EventId | None = None

    def __post_init__(self) -> None:
        # Validate fields that have domain constraints.
        _validate_event_type(self.event_type)
        _validate_source(self.source)
        if self.timestamp.tzinfo is None:
            raise EventValidationError(
                "Event.timestamp must be timezone-aware.",
                field="timestamp",
                invalid_value=str(self.timestamp),
            )
        if not self.event_id or not self.event_id.strip():
            raise EventValidationError(
                "Event.event_id must be a non-empty string.",
                field="event_id",
                invalid_value=repr(self.event_id),
            )
        if len(self._metadata_raw) > _METADATA_MAX_KEYS:
            raise EventValidationError(
                f"Event metadata exceeds maximum of {_METADATA_MAX_KEYS} keys.",
                field="metadata",
                invalid_value=str(len(self._metadata_raw)),
            )

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the event metadata as a plain dictionary (copy)."""
        return dict(self._metadata_raw)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve a single metadata value.

        Parameters
        ----------
        key:
            Metadata key to look up.
        default:
            Value to return if *key* is absent.
        """
        return dict(self._metadata_raw).get(key, default)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        source: str,
        payload: Any = None,
        priority: EventPriority = EventPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> "Event":
        """Validated factory for :class:`Event`.

        Parameters
        ----------
        event_type:
            Dot-namespaced event type string.
        source:
            Identifier of the originating subsystem.
        payload:
            Arbitrary event data.
        priority:
            :class:`EventPriority`; defaults to ``NORMAL``.
        metadata:
            Optional key-value annotations.
        correlation_id:
            Optional correlation chain identifier.
        causation_id:
            Optional id of the causing event.

        Returns
        -------
        Event
            A fully validated, immutable event instance.
        """
        raw_meta: tuple[tuple[str, Any], ...] = tuple(
            (k, v) for k, v in (metadata or {}).items()
        )
        return cls(
            event_type=EventType(event_type),
            source=source,
            payload=payload,
            priority=priority,
            _metadata_raw=raw_meta,
            correlation_id=correlation_id,
            causation_id=EventId(causation_id) if causation_id else None,
        )

    def derive(
        self,
        *,
        event_type: str | None = None,
        source: str | None = None,
        payload: Any = _SENTINEL,
        priority: EventPriority | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Event":
        """Create a causally derived child event from this one.

        The child's :attr:`causation_id` is set to this event's
        :attr:`event_id`, and :attr:`correlation_id` is inherited unless
        explicitly overridden via *metadata*.

        Parameters
        ----------
        event_type:
            Override event type; defaults to this event's type.
        source:
            Override source; defaults to this event's source.
        payload:
            Override payload; defaults to this event's payload.
        priority:
            Override priority; defaults to this event's priority.
        metadata:
            Override metadata; defaults to this event's metadata.

        Returns
        -------
        Event
            New child event with causal linkage.
        """
        effective_payload = self.payload if payload is Event.derive.__kwdefaults__ else payload  # type: ignore[attr-defined]
        # Simpler sentinel check:
        if payload is _SENTINEL:
            effective_payload = self.payload
        else:
            effective_payload = payload

        merged_meta = {**self.metadata, **(metadata or {})}
        return Event.create(
            event_type=event_type or self.event_type,
            source=source or self.source,
            payload=effective_payload,
            priority=priority or self.priority,
            metadata=merged_meta,
            correlation_id=self.correlation_id,
            causation_id=self.event_id,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise this event to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-serialisable representation (where payload allows it).
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "payload": self.payload,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Event("
            f"id={self.event_id[:8]}…, "
            f"type={self.event_type!r}, "
            f"source={self.source!r}, "
            f"priority={self.priority.name})"
        )


# ---------------------------------------------------------------------------
# Convenience type alias for event type filters
# ---------------------------------------------------------------------------

EventFilter = NewType("EventFilter", str)
"""Glob-style event type pattern used by subscribers.

Wildcards:
* ``*`` — match any single segment.
* ``**`` — match any sequence of segments.

Examples:
* ``polaris.memory.*`` — all memory events.
* ``polaris.**`` — all polaris events.
* ``polaris.memory.record_stored`` — exact match.
"""