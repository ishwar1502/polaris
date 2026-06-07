# core/events/dispatcher.py
"""
Event dispatcher for the POLARIS v5 Event Bus.

The dispatcher is the routing engine: it receives a single event and
delivers it to every subscriber whose subscriptions match, in priority
order.  Subscriber failures are isolated — a crash in one handler never
prevents delivery to others.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

from core.events.event import Event, EventPriority
from core.events.exceptions import EventDispatchError
from core.events.subscriber import Subscriber

_logger = logging.getLogger(__name__)

_MAX_DISPATCH_ERRORS_PER_EVENT: Final[int] = 32


# ---------------------------------------------------------------------------
# Dispatch result
# ---------------------------------------------------------------------------


@dataclass
class DispatchResult:
    """Summary of a single event's dispatch cycle.

    Attributes
    ----------
    event:
        The event that was dispatched.
    delivered_to:
        Ordered list of subscriber ids that successfully handled the event.
    failed_deliveries:
        Mapping of subscriber id → exception for failed handlers.
    dispatched_at:
        UTC timestamp of the dispatch call.
    """

    event: Event
    delivered_to: list[str] = field(default_factory=list)
    failed_deliveries: dict[str, Exception] = field(default_factory=dict)
    dispatched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def total_subscribers(self) -> int:
        """Total number of subscribers that were attempted."""
        return len(self.delivered_to) + len(self.failed_deliveries)

    @property
    def success_count(self) -> int:
        """Number of successful deliveries."""
        return len(self.delivered_to)

    @property
    def failure_count(self) -> int:
        """Number of failed deliveries."""
        return len(self.failed_deliveries)

    @property
    def had_failures(self) -> bool:
        """``True`` if at least one subscriber failed during delivery."""
        return bool(self.failed_deliveries)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DispatchResult("
            f"event_type={self.event.event_type!r}, "
            f"delivered={self.success_count}, "
            f"failed={self.failure_count})"
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class Dispatcher:
    """Thread-safe event router that delivers events to matched subscribers.

    The dispatcher maintains no persistent subscriber state of its own; it
    operates on whatever subscriber list the :class:`~core.events.bus.EventBus`
    passes at dispatch time.  This keeps the two concerns cleanly separated.

    Design decisions
    ----------------
    * Subscribers are called in **priority-descending** order of their
      highest-matching subscription's ``min_priority`` setting — higher
      thresholds get delivery first.
    * A failed subscriber never blocks delivery to subsequent subscribers.
    * Dispatch errors are recorded in :class:`DispatchResult` and optionally
      re-raised in aggregate if ``raise_on_failure`` is ``True``.

    Parameters
    ----------
    raise_on_failure:
        If ``True``, the dispatcher raises an :class:`EventDispatchError`
        after completing all deliveries when at least one subscriber failed.
        Defaults to ``False`` (fail-silent for maximum bus resilience).
    """

    def __init__(self, *, raise_on_failure: bool = False) -> None:
        self._raise_on_failure = raise_on_failure
        self._total_dispatched: int = 0
        self._total_failures: int = 0
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_dispatched(self) -> int:
        """Total number of events dispatched (since creation or reset)."""
        with self._lock:
            return self._total_dispatched

    @property
    def total_failures(self) -> int:
        """Total cumulative subscriber failures across all dispatch calls."""
        with self._lock:
            return self._total_failures

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        event: Event,
        subscribers: list[Subscriber],
    ) -> DispatchResult:
        """Deliver *event* to every subscriber in *subscribers* that matches.

        The method iterates subscribers in the order determined by
        :meth:`_sort_subscribers`, calls each matching subscriber's
        ``_dispatch`` method inside a try/except, and accumulates successes
        and failures into a :class:`DispatchResult`.

        Parameters
        ----------
        event:
            The event to deliver.
        subscribers:
            Snapshot of all currently registered subscribers.  The caller
            is responsible for providing a thread-safe copy.

        Returns
        -------
        DispatchResult
            Summary of the dispatch cycle.

        Raises
        ------
        EventDispatchError
            If ``raise_on_failure=True`` and at least one subscriber raised.
        """
        result = DispatchResult(event=event)
        ordered = self._sort_subscribers(subscribers, event)

        for subscriber in ordered:
            if not subscriber.matches_any(event):
                continue
            try:
                subscriber._dispatch(event)
                result.delivered_to.append(subscriber.subscriber_id)
                _logger.debug(
                    "Dispatcher: delivered %r to subscriber %r.",
                    event.event_type,
                    subscriber.subscriber_id,
                )
            except Exception as exc:  # noqa: BLE001
                result.failed_deliveries[subscriber.subscriber_id] = exc
                _logger.warning(
                    "Dispatcher: subscriber %r failed for event %r: %s",
                    subscriber.subscriber_id,
                    event.event_type,
                    exc,
                )
                if len(result.failed_deliveries) >= _MAX_DISPATCH_ERRORS_PER_EVENT:
                    _logger.error(
                        "Dispatcher: reached max per-event failure limit (%d); "
                        "aborting remaining deliveries for event %r.",
                        _MAX_DISPATCH_ERRORS_PER_EVENT,
                        event.event_type,
                    )
                    break

        with self._lock:
            self._total_dispatched += 1
            self._total_failures += result.failure_count

        if result.had_failures and self._raise_on_failure:
            first_sid, first_exc = next(iter(result.failed_deliveries.items()))
            raise EventDispatchError(
                f"Dispatch of event {event.event_type!r} had "
                f"{result.failure_count} subscriber failure(s). "
                f"First failure from {first_sid!r}: {first_exc}",
                event=event,
                subscriber_id=first_sid,
                original_exception=first_exc,
            )

        return result

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_subscribers(
        subscribers: list[Subscriber],
        event: Event,
    ) -> list[Subscriber]:
        """Return *subscribers* ordered by effective priority (descending).

        A subscriber's effective priority for a given event is the highest
        ``min_priority`` among its matching subscriptions.  Subscribers
        with higher thresholds (i.e. those that opted into higher-priority
        events specifically) are served first.

        Parameters
        ----------
        subscribers:
            Full subscriber list.
        event:
            The event being dispatched (used to resolve matching subs).

        Returns
        -------
        list[Subscriber]
            Sorted copy of *subscribers*.
        """
        def _effective_priority(sub: Subscriber) -> int:
            best = EventPriority.LOW
            for s in sub.subscriptions:
                if s.matches(event) and s.min_priority > best:
                    best = s.min_priority
            return best.value

        return sorted(subscribers, key=_effective_priority, reverse=True)

    def reset_stats(self) -> None:
        """Reset cumulative dispatch statistics to zero."""
        with self._lock:
            self._total_dispatched = 0
            self._total_failures = 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Dispatcher("
            f"dispatched={self._total_dispatched}, "
            f"failures={self._total_failures}, "
            f"raise_on_failure={self._raise_on_failure})"
        )