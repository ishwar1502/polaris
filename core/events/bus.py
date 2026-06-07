# core/events/bus.py
"""
POLARIS v5 Event Bus — the nervous system of the runtime.

:class:`EventBus` is the single integration point for all inter-subsystem
communication.  Subsystems never hold direct references to each other;
they publish events and declare subscriptions through the bus.

Architecture
------------
::

    Publisher → EventBus.publish()
                    ↓ validation
                    ↓ history buffer (append)
                    ↓ Dispatcher.dispatch(event, subscriber_snapshot)
                         ↓ for each matching Subscriber → handle_event()

Thread safety
-------------
All mutations to the subscriber registry and history buffer are protected
by a :class:`threading.RLock`.  The subscriber list is snapshotted before
dispatch so that subscribe/unsubscribe calls during delivery do not
affect the in-progress dispatch cycle.
"""

from __future__ import annotations

import collections
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Final, Iterator, Sequence

from core.events.dispatcher import DispatchResult, Dispatcher
from core.events.event import Event, EventFilter, EventPriority, EventType
from core.events.exceptions import (
    EventBusError,
    EventDispatchError,
    EventValidationError,
    SubscriptionError,
)
from core.events.subscriber import FunctionalSubscriber, Subscriber, Subscription

_logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_CAPACITY: Final[int] = 4096
_MIN_HISTORY_CAPACITY: Final[int] = 0
_MAX_HISTORY_CAPACITY: Final[int] = 1_000_000


# ---------------------------------------------------------------------------
# Bus statistics
# ---------------------------------------------------------------------------


@dataclass
class BusStatistics:
    """Snapshot of EventBus runtime statistics.

    Attributes
    ----------
    total_published:
        Events admitted to the bus since creation or last reset.
    total_dispatched:
        Events routed by the dispatcher.
    total_dispatch_failures:
        Cumulative subscriber-level dispatch failures.
    subscriber_count:
        Current number of registered subscribers.
    history_size:
        Current number of events retained in the history buffer.
    history_capacity:
        Maximum history buffer capacity.
    captured_at:
        UTC timestamp of this snapshot.
    """

    total_published: int
    total_dispatched: int
    total_dispatch_failures: int
    subscriber_count: int
    history_size: int
    history_capacity: int
    captured_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Thread-safe central event bus for the POLARIS v5 runtime.

    Parameters
    ----------
    name:
        Human-readable identifier for this bus instance (used in logs).
    history_capacity:
        Maximum number of events to retain in the circular history buffer.
        Set to ``0`` to disable history.  Defaults to
        :data:`_DEFAULT_HISTORY_CAPACITY`.
    dispatcher:
        Optional custom :class:`~core.events.dispatcher.Dispatcher`.
        A default instance is created if not provided.
    validate_on_publish:
        If ``True`` (default), events are re-validated by the bus before
        dispatch.  Disable only in performance-critical hot paths where
        events are already known to be valid.

    Raises
    ------
    ValueError
        If *history_capacity* is outside the permitted range.
    """

    def __init__(
        self,
        name: str = "polaris.event_bus",
        *,
        history_capacity: int = _DEFAULT_HISTORY_CAPACITY,
        dispatcher: Dispatcher | None = None,
        validate_on_publish: bool = True,
    ) -> None:
        if not (_MIN_HISTORY_CAPACITY <= history_capacity <= _MAX_HISTORY_CAPACITY):
            raise ValueError(
                f"history_capacity must be between {_MIN_HISTORY_CAPACITY} "
                f"and {_MAX_HISTORY_CAPACITY}, got {history_capacity}."
            )
        self._name = name
        self._history_capacity = history_capacity
        self._validate_on_publish = validate_on_publish
        self._dispatcher = dispatcher or Dispatcher()

        # Subscriber registry: subscriber_id → Subscriber
        self._subscribers: dict[str, Subscriber] = {}

        # Circular event history (deque with maxlen acts as a ring buffer).
        self._history: deque[Event] = deque(
            maxlen=history_capacity if history_capacity > 0 else None
        )
        self._history_enabled: bool = history_capacity > 0

        # Counters
        self._total_published: int = 0

        # Single reentrant lock guards subscribers + history + counters.
        self._lock: threading.RLock = threading.RLock()

        _logger.info(
            "EventBus %r created (history_capacity=%d).",
            name,
            history_capacity,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable bus identifier."""
        return self._name

    @property
    def history_capacity(self) -> int:
        """Maximum size of the event history buffer."""
        return self._history_capacity

    # ------------------------------------------------------------------
    # Subscriber management
    # ------------------------------------------------------------------

    def register_subscriber(self, subscriber: Subscriber) -> None:
        """Register a :class:`~core.events.subscriber.Subscriber` with the bus.

        The subscriber must not already be registered (by id).  Its
        :attr:`~core.events.subscriber.Subscriber.subscriptions` are used
        by the dispatcher at delivery time.

        Parameters
        ----------
        subscriber:
            The subscriber instance to register.

        Raises
        ------
        TypeError
            If *subscriber* is not a :class:`~core.events.subscriber.Subscriber`.
        SubscriptionError
            If a subscriber with the same id is already registered.
        SubscriptionError
            If the subscriber declares no subscriptions.
        """
        if not isinstance(subscriber, Subscriber):
            raise TypeError(
                f"register_subscriber() expects a Subscriber instance, "
                f"got {type(subscriber).__name__!r}."
            )
        subs = subscriber.subscriptions
        if not subs:
            raise SubscriptionError(
                f"Subscriber {subscriber.subscriber_id!r} declares no "
                "subscriptions and cannot be registered.",
                subscriber_id=subscriber.subscriber_id,
            )
        with self._lock:
            if subscriber.subscriber_id in self._subscribers:
                raise SubscriptionError(
                    f"Subscriber {subscriber.subscriber_id!r} is already "
                    "registered. Call unregister_subscriber() first.",
                    subscriber_id=subscriber.subscriber_id,
                )
            self._subscribers[subscriber.subscriber_id] = subscriber
        _logger.debug(
            "EventBus %r: subscriber %r registered (%d subscriptions).",
            self._name,
            subscriber.subscriber_id,
            len(subs),
        )

    def unregister_subscriber(self, subscriber_id: str) -> Subscriber:
        """Remove a subscriber from the bus.

        Parameters
        ----------
        subscriber_id:
            Identifier of the subscriber to remove.

        Returns
        -------
        Subscriber
            The removed subscriber instance.

        Raises
        ------
        SubscriptionError
            If *subscriber_id* is not registered.
        """
        with self._lock:
            if subscriber_id not in self._subscribers:
                raise SubscriptionError(
                    f"Subscriber {subscriber_id!r} is not registered.",
                    subscriber_id=subscriber_id,
                )
            removed = self._subscribers.pop(subscriber_id)
        _logger.debug(
            "EventBus %r: subscriber %r unregistered.",
            self._name,
            subscriber_id,
        )
        return removed

    def subscribe(
        self,
        subscriber_id: str,
        handler: Callable[[Event], None],
        event_filters: list[str] | str,
        *,
        min_priority: EventPriority = EventPriority.LOW,
    ) -> FunctionalSubscriber:
        """Register a plain callable as a subscriber.

        This is the ergonomic API for one-liner subscriptions; it wraps the
        callable in a :class:`~core.events.subscriber.FunctionalSubscriber`
        and calls :meth:`register_subscriber`.

        Parameters
        ----------
        subscriber_id:
            Unique identifier for this subscription.
        handler:
            A callable ``(event: Event) -> None``.
        event_filters:
            One or more glob-style event type patterns (string or list).
        min_priority:
            Minimum priority to receive.

        Returns
        -------
        FunctionalSubscriber
            The created and registered subscriber wrapper.

        Raises
        ------
        SubscriptionError
            If *subscriber_id* is already registered.
        """
        if isinstance(event_filters, str):
            event_filters = [event_filters]
        fs = FunctionalSubscriber(
            subscriber_id=subscriber_id,
            handler=handler,
            event_filters=event_filters,
            min_priority=min_priority,
        )
        self.register_subscriber(fs)
        return fs

    def unsubscribe(self, subscriber_id: str) -> None:
        """Convenience alias for :meth:`unregister_subscriber`.

        Parameters
        ----------
        subscriber_id:
            Identifier to remove.
        """
        self.unregister_subscriber(subscriber_id)

    def get_subscribers(
        self,
        *,
        event_type: str | None = None,
    ) -> list[Subscriber]:
        """Return a snapshot of registered subscribers.

        Parameters
        ----------
        event_type:
            If provided, return only subscribers that would match an event
            of this type (with default priority).  Useful for introspection.

        Returns
        -------
        list[Subscriber]
            Matching subscribers (unordered).
        """
        with self._lock:
            all_subs = list(self._subscribers.values())

        if event_type is None:
            return all_subs

        # Filter by creating a probe event and testing subscriptions.
        probe = Event.create(
            event_type=event_type,
            source="bus.introspection",
            payload=None,
            priority=EventPriority.LOW,
        )
        return [s for s in all_subs if s.matches_any(probe)]

    def has_subscriber(self, subscriber_id: str) -> bool:
        """Return whether *subscriber_id* is currently registered.

        Parameters
        ----------
        subscriber_id:
            Identifier to check.
        """
        with self._lock:
            return subscriber_id in self._subscribers

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> DispatchResult:
        """Admit *event* to the bus and deliver it to all matching subscribers.

        Workflow:

        1. Type check.
        2. Optional validation pass.
        3. Append to history buffer (if enabled).
        4. Snapshot subscriber list (thread-safe).
        5. Delegate to :class:`~core.events.dispatcher.Dispatcher`.

        Parameters
        ----------
        event:
            The event to publish.

        Returns
        -------
        DispatchResult
            Summary of the dispatch cycle.

        Raises
        ------
        TypeError
            If *event* is not an :class:`~core.events.event.Event`.
        EventValidationError
            If *validate_on_publish* is ``True`` and the event fails
            validation (unusual — events are validated at construction).
        """
        if not isinstance(event, Event):
            raise TypeError(
                f"publish() expects an Event instance, "
                f"got {type(event).__name__!r}."
            )

        if self._validate_on_publish:
            self._validate_event(event)

        with self._lock:
            if self._history_enabled:
                self._history.append(event)
            self._total_published += 1
            subscriber_snapshot = list(self._subscribers.values())

        _logger.debug(
            "EventBus %r: publishing %r (priority=%s, subscribers=%d).",
            self._name,
            event.event_type,
            event.priority.name,
            len(subscriber_snapshot),
        )

        result = self._dispatcher.dispatch(event, subscriber_snapshot)

        if result.had_failures:
            _logger.warning(
                "EventBus %r: event %r had %d delivery failure(s).",
                self._name,
                event.event_type,
                result.failure_count,
            )

        return result

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_event_history(
        self,
        *,
        event_type: str | None = None,
        source: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
        priority: EventPriority | None = None,
    ) -> list[Event]:
        """Retrieve events from the history buffer with optional filtering.

        All filters are applied with AND semantics (all must match).

        Parameters
        ----------
        event_type:
            Glob-style pattern to filter by event type.
        source:
            Exact match filter on ``event.source``.
        since:
            Return only events with ``timestamp >= since``.
        limit:
            Maximum number of events to return (most-recent first if
            filtering would otherwise truncate).
        priority:
            Return only events with exactly this priority.

        Returns
        -------
        list[Event]
            Matching events in chronological order (oldest first).
        """
        with self._lock:
            snapshot: list[Event] = list(self._history)

        results: list[Event] = []
        for event in snapshot:
            if event_type is not None:
                import fnmatch as _fnmatch
                pattern = event_type.replace("**", "*")
                if not _fnmatch.fnmatchcase(event.event_type, pattern):
                    continue
            if source is not None and event.source != source:
                continue
            if since is not None and event.timestamp < since:
                continue
            if priority is not None and event.priority is not priority:
                continue
            results.append(event)

        if limit is not None:
            results = results[-limit:]

        return results

    def clear_history(self) -> int:
        """Remove all events from the history buffer.

        Returns
        -------
        int
            Number of events cleared.
        """
        with self._lock:
            count = len(self._history)
            self._history.clear()
        _logger.debug(
            "EventBus %r: history cleared (%d events removed).",
            self._name,
            count,
        )
        return count

    def history_snapshot(self) -> list[Event]:
        """Return a full copy of the current history buffer.

        Returns
        -------
        list[Event]
            All retained events in chronological order.
        """
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> BusStatistics:
        """Capture a statistics snapshot.

        Returns
        -------
        BusStatistics
            Current runtime counters for this bus instance.
        """
        with self._lock:
            return BusStatistics(
                total_published=self._total_published,
                total_dispatched=self._dispatcher.total_dispatched,
                total_dispatch_failures=self._dispatcher.total_failures,
                subscriber_count=len(self._subscribers),
                history_size=len(self._history),
                history_capacity=self._history_capacity,
            )

    # ------------------------------------------------------------------
    # Validation (internal)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_event(event: Event) -> None:
        """Perform bus-level semantic validation on *event*.

        The :class:`~core.events.event.Event` constructor already runs
        structural validation.  This method adds any additional bus-level
        policy checks.

        Parameters
        ----------
        event:
            Event to validate.

        Raises
        ------
        EventValidationError
            If the event fails bus-level policy.
        """
        # Bus policy: CRITICAL events must carry a source identifier.
        if event.priority is EventPriority.CRITICAL and not event.source.strip():
            raise EventValidationError(
                "CRITICAL events must specify a non-empty source.",
                field="source",
                invalid_value=repr(event.source),
            )

    # ------------------------------------------------------------------
    # Iteration and containment
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of currently registered subscribers."""
        with self._lock:
            return len(self._subscribers)

    def __contains__(self, subscriber_id: object) -> bool:
        """Support ``in`` operator for subscriber id membership tests."""
        if not isinstance(subscriber_id, str):
            return False
        with self._lock:
            return subscriber_id in self._subscribers

    def __iter__(self) -> Iterator[Subscriber]:
        """Iterate over a snapshot of currently registered subscribers."""
        return iter(self.get_subscribers())

    def __repr__(self) -> str:  # pragma: no cover
        with self._lock:
            return (
                f"EventBus("
                f"name={self._name!r}, "
                f"subscribers={len(self._subscribers)}, "
                f"published={self._total_published}, "
                f"history={len(self._history)}/{self._history_capacity})"
            )