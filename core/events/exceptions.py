# core/events/exceptions.py
"""
Exception hierarchy for the POLARIS v5 Event Bus.

All event-bus exceptions inherit from :class:`EventBusError` so callers
can catch the entire taxonomy with a single clause.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.events.event import Event, EventType


class EventBusError(Exception):
    """Root exception for all POLARIS Event Bus errors."""


class EventDispatchError(EventBusError):
    """Raised when the dispatcher cannot route or deliver an event.

    Attributes
    ----------
    event:
        The :class:`~core.events.event.Event` that could not be dispatched.
    subscriber_id:
        Optional identifier of the subscriber that failed during delivery.
    original_exception:
        The underlying exception raised by the subscriber, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        event: "Event | None" = None,
        subscriber_id: str | None = None,
        original_exception: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.event = event
        self.subscriber_id = subscriber_id
        self.original_exception = original_exception

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EventDispatchError("
            f"subscriber_id={self.subscriber_id!r}, "
            f"event_type={self.event.event_type if self.event else None!r}, "
            f"message={str(self)!r})"
        )


class SubscriptionError(EventBusError):
    """Raised when a subscription operation is invalid or violates policy.

    Attributes
    ----------
    subscriber_id:
        Identifier of the subscriber involved.
    event_type:
        The event type that was the target of the subscription, if applicable.
    """

    def __init__(
        self,
        message: str,
        *,
        subscriber_id: str | None = None,
        event_type: "EventType | None" = None,
    ) -> None:
        super().__init__(message)
        self.subscriber_id = subscriber_id
        self.event_type = event_type

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SubscriptionError("
            f"subscriber_id={self.subscriber_id!r}, "
            f"event_type={self.event_type!r}, "
            f"message={str(self)!r})"
        )


class EventValidationError(EventBusError):
    """Raised when an :class:`~core.events.event.Event` fails schema or
    semantic validation before being admitted to the bus.

    Attributes
    ----------
    field:
        The field or attribute that triggered the validation failure.
    invalid_value:
        The value that was rejected, serialised to a string for safety.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        invalid_value: str | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.invalid_value = invalid_value

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EventValidationError("
            f"field={self.field!r}, "
            f"invalid_value={self.invalid_value!r}, "
            f"message={str(self)!r})"
        )


class PublisherError(EventBusError):
    """Raised when a publisher fails to emit an event to the bus.

    Attributes
    ----------
    publisher_id:
        Identifier of the publisher that raised the error.
    """

    def __init__(
        self,
        message: str,
        *,
        publisher_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.publisher_id = publisher_id