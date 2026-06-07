# core/events/__init__.py
"""
POLARIS v5 Event Bus package.

Re-exports the complete public surface of the event system for convenient
single-import access.

Usage
-----
.. code-block:: python

    from core.events import EventBus, Event, EventPriority, Subscriber

    bus = EventBus()

    received: list[Event] = []
    bus.subscribe("logger", received.append, "**")

    bus.publish(Event.create(
        event_type="polaris.memory.record_stored",
        source="polaris.core.memory",
        payload={"record_id": "abc123"},
    ))
"""

from __future__ import annotations

from core.events.bus import BusStatistics, EventBus
from core.events.dispatcher import DispatchResult, Dispatcher
from core.events.event import Event, EventFilter, EventId, EventPriority, EventType
from core.events.exceptions import (
    EventBusError,
    EventDispatchError,
    EventValidationError,
    PublisherError,
    SubscriptionError,
)
from core.events.publisher import Publisher
from core.events.subscriber import FunctionalSubscriber, Subscriber, Subscription

__all__ = [
    # bus
    "BusStatistics",
    "EventBus",
    # dispatcher
    "DispatchResult",
    "Dispatcher",
    # event
    "Event",
    "EventFilter",
    "EventId",
    "EventPriority",
    "EventType",
    # exceptions
    "EventBusError",
    "EventDispatchError",
    "EventValidationError",
    "PublisherError",
    "SubscriptionError",
    # publisher
    "Publisher",
    # subscriber
    "FunctionalSubscriber",
    "Subscriber",
    "Subscription",
]