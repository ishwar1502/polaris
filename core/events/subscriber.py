# core/events/subscriber.py
"""
Subscriber abstract base class and subscription descriptor for the
POLARIS v5 Event Bus.

Subscribers are the consumers of the event stream.  Each subscriber
declares which event types it wishes to receive via :attr:`subscriptions`,
and the dispatcher calls :meth:`handle_event` for each matched event.
"""

from __future__ import annotations

import abc
import fnmatch
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Final

from core.events.event import Event, EventFilter, EventPriority, EventType
from core.events.exceptions import SubscriptionError

_logger = logging.getLogger(__name__)

_WILDCARD_ALL: Final[str] = "**"


# ---------------------------------------------------------------------------
# Subscription descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Subscription:
    """Descriptor binding a subscriber to a set of event type patterns.

    Attributes
    ----------
    subscriber_id:
        Identifier of the owning subscriber.
    event_filter:
        Glob-style pattern string (see :class:`~core.events.event.EventFilter`).
        Use ``**`` to subscribe to every event type.
    min_priority:
        Minimum :class:`~core.events.event.EventPriority` the subscriber
        wishes to receive.  Events with lower priority are skipped.
    """

    subscriber_id: str
    event_filter: EventFilter
    min_priority: EventPriority = EventPriority.LOW

    def __post_init__(self) -> None:
        if not self.subscriber_id or not self.subscriber_id.strip():
            raise SubscriptionError(
                "Subscription.subscriber_id must be non-empty.",
                subscriber_id=self.subscriber_id,
            )
        if not self.event_filter or not self.event_filter.strip():
            raise SubscriptionError(
                "Subscription.event_filter must be non-empty.",
                subscriber_id=self.subscriber_id,
            )

    def matches(self, event: Event) -> bool:
        """Return whether this subscription matches *event*.

        Performs glob-style pattern matching on :attr:`Event.event_type`
        and checks :attr:`Event.priority` against :attr:`min_priority`.

        Parameters
        ----------
        event:
            The event to test.

        Returns
        -------
        bool
            ``True`` if the subscription pattern matches and the event's
            priority meets the minimum threshold.
        """
        if event.priority < self.min_priority:
            return False
        if self.event_filter == _WILDCARD_ALL:
            return True
        # Translate dot-namespaced wildcard to fnmatch glob.
        # Replace '**' with '*' for fnmatch, handle '.*.' separators.
        pattern = self.event_filter.replace("**", "*")
        return fnmatch.fnmatchcase(event.event_type, pattern)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Subscription("
            f"subscriber_id={self.subscriber_id!r}, "
            f"filter={self.event_filter!r}, "
            f"min_priority={self.min_priority.name})"
        )


# ---------------------------------------------------------------------------
# Subscriber ABC
# ---------------------------------------------------------------------------


class Subscriber(abc.ABC):
    """Abstract base class for all POLARIS event subscribers.

    A subscriber declares one or more :class:`Subscription` descriptors and
    implements :meth:`handle_event` to process matched events.  The bus calls
    this method for every event matching the subscriber's subscriptions.

    Parameters
    ----------
    subscriber_id:
        Unique identifier for this subscriber.  Must be non-empty.
    """

    def __init__(self, subscriber_id: str) -> None:
        if not subscriber_id or not subscriber_id.strip():
            raise SubscriptionError(
                "subscriber_id must be a non-empty string.",
                subscriber_id=subscriber_id,
            )
        self._subscriber_id = subscriber_id
        self._handled_count: int = 0
        self._error_count: int = 0
        self._last_event_at: datetime | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def subscriber_id(self) -> str:
        """Unique identifier for this subscriber."""
        return self._subscriber_id

    @property
    def handled_count(self) -> int:
        """Total number of events successfully handled."""
        return self._handled_count

    @property
    def error_count(self) -> int:
        """Total number of events that raised during handling."""
        return self._error_count

    @property
    def last_event_at(self) -> datetime | None:
        """UTC timestamp of the last event handled, or ``None``."""
        return self._last_event_at

    @property
    @abc.abstractmethod
    def subscriptions(self) -> list[Subscription]:
        """Ordered list of :class:`Subscription` descriptors.

        The dispatcher iterates this list and delivers the event to this
        subscriber if **any** subscription matches.  Concrete subscribers
        must implement this property.

        Returns
        -------
        list[Subscription]
            Non-empty list of subscriptions.
        """

    # ------------------------------------------------------------------
    # Dispatch entry point (called by Dispatcher — not by user code)
    # ------------------------------------------------------------------

    def _dispatch(self, event: Event) -> None:
        """Internal entry point called by the :class:`~core.events.dispatcher.Dispatcher`.

        Records metrics and delegates to :meth:`handle_event`.  Should not
        be overridden by application code.

        Parameters
        ----------
        event:
            The event being delivered.
        """
        try:
            self.handle_event(event)
            self._handled_count += 1
            self._last_event_at = datetime.now(timezone.utc)
        except Exception as exc:
            self._error_count += 1
            _logger.exception(
                "Subscriber %r raised during handle_event for %r: %s",
                self._subscriber_id,
                event.event_type,
                exc,
            )
            raise  # Re-raise so Dispatcher can record the failure.

    # ------------------------------------------------------------------
    # Abstract handler
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def handle_event(self, event: Event) -> None:
        """Process a matched event.

        This method is called by the dispatcher for every event that
        matches at least one of this subscriber's :attr:`subscriptions`.

        **Implementations must not raise.**  Exceptions are caught by the
        dispatcher's failure-isolation layer, logged, and the bus continues
        delivering to other subscribers.

        Parameters
        ----------
        event:
            The matched event.
        """

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    def matches_any(self, event: Event) -> bool:
        """Return whether *event* matches at least one subscription.

        Parameters
        ----------
        event:
            Event to test.

        Returns
        -------
        bool
            ``True`` if any subscription in :attr:`subscriptions` matches.
        """
        return any(sub.matches(event) for sub in self.subscriptions)

    # ------------------------------------------------------------------
    # Factory helpers for concrete subscriber construction
    # ------------------------------------------------------------------

    @classmethod
    def make_subscription(
        cls,
        subscriber_id: str,
        event_filter: str,
        *,
        min_priority: EventPriority = EventPriority.LOW,
    ) -> Subscription:
        """Convenience factory for :class:`Subscription`.

        Parameters
        ----------
        subscriber_id:
            Owning subscriber's identifier.
        event_filter:
            Glob-style event type pattern.
        min_priority:
            Minimum accepted priority.
        """
        return Subscription(
            subscriber_id=subscriber_id,
            event_filter=EventFilter(event_filter),
            min_priority=min_priority,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{type(self).__name__}("
            f"subscriber_id={self._subscriber_id!r}, "
            f"subscriptions={len(self.subscriptions)}, "
            f"handled={self._handled_count})"
        )


# ---------------------------------------------------------------------------
# Functional subscriber (lambda / callable adapter)
# ---------------------------------------------------------------------------


class FunctionalSubscriber(Subscriber):
    """Concrete subscriber that delegates :meth:`handle_event` to a callable.

    This adapter allows subscribing a plain function or lambda to the bus
    without defining a full :class:`Subscriber` subclass.

    Parameters
    ----------
    subscriber_id:
        Unique identifier.
    handler:
        Callable that accepts a single :class:`~core.events.event.Event` arg.
    event_filters:
        One or more glob-style event type patterns.
    min_priority:
        Minimum priority; defaults to ``LOW``.
    """

    def __init__(
        self,
        subscriber_id: str,
        handler: Callable[[Event], None],
        event_filters: list[str],
        *,
        min_priority: EventPriority = EventPriority.LOW,
    ) -> None:
        super().__init__(subscriber_id)
        if not callable(handler):
            raise SubscriptionError(
                "FunctionalSubscriber.handler must be callable.",
                subscriber_id=subscriber_id,
            )
        if not event_filters:
            raise SubscriptionError(
                "FunctionalSubscriber requires at least one event_filter.",
                subscriber_id=subscriber_id,
            )
        self._handler = handler
        self._subscriptions: list[Subscription] = [
            Subscription(
                subscriber_id=subscriber_id,
                event_filter=EventFilter(f),
                min_priority=min_priority,
            )
            for f in event_filters
        ]

    @property
    def subscriptions(self) -> list[Subscription]:
        """Return the list of subscriptions bound at construction."""
        return list(self._subscriptions)

    def handle_event(self, event: Event) -> None:
        """Delegate to the wrapped callable.

        Parameters
        ----------
        event:
            The matched event.
        """
        self._handler(event)