# core/events/publisher.py
"""
Publisher abstract base class for the POLARIS v5 Event Bus.

Every subsystem that emits events must extend :class:`Publisher` and call
:meth:`publish` to submit events.  The publisher holds a weak reference to
the bus so the bus can be garbage-collected independently.
"""

from __future__ import annotations

import abc
import logging
import weakref
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.events.event import Event, EventFilter, EventPriority, EventType
from core.events.exceptions import EventBusError, PublisherError

if TYPE_CHECKING:
    from core.events.bus import EventBus

_logger = logging.getLogger(__name__)


class Publisher(abc.ABC):
    """Abstract base class for all POLARIS event publishers.

    A publisher represents any subsystem component that emits events.
    Concrete publishers must supply a :attr:`publisher_id` and may
    override :meth:`_on_publish_success` and :meth:`_on_publish_failure`
    for observability hooks.

    Parameters
    ----------
    publisher_id:
        Unique identifier for this publisher, typically the owning
        subsystem's :class:`~core.types.identifiers.SubsystemId`.
    bus:
        The :class:`~core.events.bus.EventBus` to publish events on.
        Stored as a ``weakref`` to prevent circular retention.
    """

    def __init__(self, publisher_id: str, bus: "EventBus") -> None:
        if not publisher_id or not publisher_id.strip():
            raise PublisherError(
                "publisher_id must be a non-empty string.",
                publisher_id=publisher_id,
            )
        self._publisher_id = publisher_id
        self._bus_ref: weakref.ref["EventBus"] = weakref.ref(bus)
        self._publish_count: int = 0
        self._error_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def publisher_id(self) -> str:
        """Unique identifier for this publisher."""
        return self._publisher_id

    @property
    def publish_count(self) -> int:
        """Total number of events successfully published."""
        return self._publish_count

    @property
    def error_count(self) -> int:
        """Total number of publish failures recorded."""
        return self._error_count

    # ------------------------------------------------------------------
    # Core publish API
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Submit *event* to the connected :class:`~core.events.bus.EventBus`.

        Parameters
        ----------
        event:
            The fully constructed :class:`~core.events.event.Event` to emit.

        Raises
        ------
        PublisherError
            If the bus has been garbage-collected or the publish call fails.
        TypeError
            If *event* is not an :class:`~core.events.event.Event` instance.
        """
        if not isinstance(event, Event):
            raise TypeError(
                f"publish() expects an Event instance, "
                f"got {type(event).__name__!r}."
            )
        bus = self._bus_ref()
        if bus is None:
            self._error_count += 1
            raise PublisherError(
                f"Publisher {self._publisher_id!r}: the EventBus has been "
                "garbage-collected; cannot publish.",
                publisher_id=self._publisher_id,
            )
        try:
            bus.publish(event)
            self._publish_count += 1
            self._on_publish_success(event)
            _logger.debug(
                "Publisher %r: emitted %r (priority=%s).",
                self._publisher_id,
                event.event_type,
                event.priority.name,
            )
        except Exception as exc:
            self._error_count += 1
            self._on_publish_failure(event, exc)
            raise PublisherError(
                f"Publisher {self._publisher_id!r}: failed to publish "
                f"event {event.event_type!r}: {exc}",
                publisher_id=self._publisher_id,
            ) from exc

    def emit(
        self,
        *,
        event_type: str,
        payload: Any = None,
        priority: EventPriority = EventPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> Event:
        """Convenience factory + publish in a single call.

        Constructs an :class:`~core.events.event.Event` from *event_type*
        and *payload*, stamping this publisher's id as the ``source``, then
        submits it to the bus.

        Parameters
        ----------
        event_type:
            Dot-namespaced event type string.
        payload:
            Arbitrary event data.
        priority:
            :class:`~core.events.event.EventPriority`; defaults to ``NORMAL``.
        metadata:
            Optional key-value annotations.
        correlation_id:
            Optional correlation chain id.
        causation_id:
            Optional id of the causing event.

        Returns
        -------
        Event
            The constructed and published event.
        """
        event = Event.create(
            event_type=event_type,
            source=self._publisher_id,
            payload=payload,
            priority=priority,
            metadata=metadata,
            correlation_id=correlation_id,
            causation_id=causation_id,
        )
        self.publish(event)
        return event

    # ------------------------------------------------------------------
    # Hooks (optional override)
    # ------------------------------------------------------------------

    def _on_publish_success(self, event: Event) -> None:
        """Called after each successful publish.  Override to add telemetry.

        Parameters
        ----------
        event:
            The event that was successfully dispatched.
        """

    def _on_publish_failure(self, event: Event, exc: Exception) -> None:
        """Called when a publish attempt raises an exception.

        Parameters
        ----------
        event:
            The event that failed to dispatch.
        exc:
            The exception raised during dispatch.
        """
        _logger.warning(
            "Publisher %r: publish failure for %r: %s",
            self._publisher_id,
            event.event_type,
            exc,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{type(self).__name__}("
            f"publisher_id={self._publisher_id!r}, "
            f"published={self._publish_count}, "
            f"errors={self._error_count})"
        )