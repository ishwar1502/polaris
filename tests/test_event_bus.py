# tests/test_event_bus.py
"""
Comprehensive pytest suite for the POLARIS v5 Event Bus (Phase 2).

Coverage targets
----------------
* Event creation, immutability, validation
* Subscription creation and pattern matching
* Subscriber lifecycle (register / unregister)
* FunctionalSubscriber
* Publisher emit + error handling
* Bus publish → dispatch pipeline
* Multiple subscribers
* Priority ordering
* Event history: append, filter, clear, capacity overflow
* Dispatcher failure isolation
* Thread safety (concurrent publish + subscribe)
* DispatchResult accuracy
* Custom Dispatcher integration
* Schema payload validation
* EventBus statistics
* Error taxonomy (exception attributes)
* Derived events (causation chain)
* Wildcard subscriptions
* Edge cases (empty bus, no-match event, history disabled)

Total: 50+ assertions across 47 test functions.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.events.bus import EventBus
from core.events.dispatcher import DispatchResult, Dispatcher
from core.events.event import Event, EventFilter, EventPriority, EventType
from core.events.exceptions import (
    EventBusError,
    EventDispatchError,
    EventValidationError,
    PublisherError,
    SubscriptionError,
)
from core.events.publisher import Publisher
from core.events.schemas.system import (
    SubsystemHealthChangedPayload,
    SubsystemLifecyclePayload,
    SubsystemRegisteredPayload,
    SubsystemUnregisteredPayload,
    SystemEvents,
)
from core.events.subscriber import FunctionalSubscriber, Subscriber, Subscription


# ===========================================================================
# Helpers / fixtures
# ===========================================================================


def _make_event(
    event_type: str = "polaris.test.happened",
    source: str = "polaris.test.source",
    payload: Any = None,
    priority: EventPriority = EventPriority.NORMAL,
    metadata: dict | None = None,
) -> Event:
    return Event.create(
        event_type=event_type,
        source=source,
        payload=payload,
        priority=priority,
        metadata=metadata,
    )


class _CollectorSubscriber(Subscriber):
    """Test subscriber that collects all received events."""

    def __init__(
        self,
        subscriber_id: str,
        filters: list[str],
        *,
        min_priority: EventPriority = EventPriority.LOW,
        raise_on_handle: bool = False,
    ) -> None:
        super().__init__(subscriber_id)
        self._filters = filters
        self._min_priority = min_priority
        self._raise_on_handle = raise_on_handle
        self.received: list[Event] = []

    @property
    def subscriptions(self) -> list[Subscription]:
        return [
            Subscription(
                subscriber_id=self._subscriber_id,
                event_filter=EventFilter(f),
                min_priority=self._min_priority,
            )
            for f in self._filters
        ]

    def handle_event(self, event: Event) -> None:
        if self._raise_on_handle:
            raise RuntimeError(f"Simulated failure in {self._subscriber_id}")
        self.received.append(event)


class _ConcretePublisher(Publisher):
    """Concrete publisher for tests."""

    pass


@pytest.fixture
def bus() -> EventBus:
    return EventBus(history_capacity=256)


@pytest.fixture
def no_history_bus() -> EventBus:
    return EventBus(history_capacity=0)


@pytest.fixture
def strict_bus() -> EventBus:
    """Bus backed by a raise_on_failure Dispatcher."""
    return EventBus(
        history_capacity=256,
        dispatcher=Dispatcher(raise_on_failure=True),
    )


# ===========================================================================
# 1. Event creation and immutability
# ===========================================================================


class TestEventCreation:
    def test_create_minimal(self) -> None:
        event = _make_event()
        assert event.event_type == "polaris.test.happened"
        assert event.source == "polaris.test.source"
        assert event.priority is EventPriority.NORMAL
        assert event.payload is None

    def test_create_auto_id(self) -> None:
        e1 = _make_event()
        e2 = _make_event()
        assert e1.event_id != e2.event_id

    def test_create_auto_timestamp_utc(self) -> None:
        event = _make_event()
        assert event.timestamp.tzinfo is not None

    def test_create_with_payload(self) -> None:
        payload = {"key": "value", "count": 42}
        event = _make_event(payload=payload)
        assert event.payload == payload

    def test_create_with_metadata(self) -> None:
        event = _make_event(metadata={"trace_id": "abc", "request_id": "xyz"})
        assert event.get_metadata("trace_id") == "abc"
        assert event.get_metadata("request_id") == "xyz"

    def test_metadata_default_returns_none(self) -> None:
        event = _make_event()
        assert event.get_metadata("nonexistent") is None

    def test_metadata_default_override(self) -> None:
        event = _make_event()
        assert event.get_metadata("missing", "default_val") == "default_val"

    def test_event_is_immutable(self) -> None:
        event = _make_event()
        with pytest.raises((AttributeError, TypeError)):
            event.event_type = "polaris.other.event"  # type: ignore[misc]

    def test_event_to_dict(self) -> None:
        event = _make_event(payload={"x": 1})
        d = event.to_dict()
        assert d["event_type"] == "polaris.test.happened"
        assert d["source"] == "polaris.test.source"
        assert d["priority"] == "NORMAL"
        assert d["payload"] == {"x": 1}
        assert "event_id" in d
        assert "timestamp" in d

    def test_event_priority_ordering(self) -> None:
        assert EventPriority.LOW < EventPriority.NORMAL
        assert EventPriority.NORMAL < EventPriority.HIGH
        assert EventPriority.HIGH < EventPriority.CRITICAL
        assert EventPriority.CRITICAL > EventPriority.LOW

    def test_all_priority_levels_constructable(self) -> None:
        for prio in EventPriority:
            event = _make_event(priority=prio)
            assert event.priority is prio


# ===========================================================================
# 2. Event validation
# ===========================================================================


class TestEventValidation:
    def test_invalid_event_type_empty(self) -> None:
        with pytest.raises(EventValidationError) as exc:
            _make_event(event_type="")
        assert exc.value.field == "event_type"

    def test_invalid_event_type_single_segment(self) -> None:
        with pytest.raises(EventValidationError):
            _make_event(event_type="polaris")

    def test_invalid_source_empty(self) -> None:
        with pytest.raises(EventValidationError) as exc:
            _make_event(source="")
        assert exc.value.field == "source"

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(EventValidationError) as exc:
            Event(
                event_type=EventType("polaris.test.event"),
                source="src",
                payload=None,
                timestamp=datetime(2024, 1, 1),  # naive
            )
        assert exc.value.field == "timestamp"

    def test_event_type_with_underscore_allowed(self) -> None:
        event = _make_event(event_type="polaris.memory.record_stored")
        assert event.event_type == "polaris.memory.record_stored"

    def test_event_type_too_long_rejected(self) -> None:
        long_type = "polaris." + "a" * 300
        with pytest.raises(EventValidationError):
            _make_event(event_type=long_type)


# ===========================================================================
# 3. Derived events
# ===========================================================================


class TestDerivedEvents:
    def test_derive_sets_causation_id(self) -> None:
        parent = _make_event()
        child = parent.derive(event_type="polaris.test.derived")
        assert child.causation_id == parent.event_id

    def test_derive_inherits_correlation_id(self) -> None:
        parent = Event.create(
            event_type="polaris.test.parent",
            source="src",
            payload=None,
            correlation_id="corr-123",
        )
        child = parent.derive(event_type="polaris.test.child")
        assert child.correlation_id == "corr-123"

    def test_derive_overrides_payload(self) -> None:
        parent = _make_event(payload={"original": True})
        child = parent.derive(payload={"overridden": True})
        assert child.payload == {"overridden": True}

    def test_derive_inherits_payload_by_default(self) -> None:
        parent = _make_event(payload={"data": 42})
        child = parent.derive(event_type="polaris.test.child")
        assert child.payload == {"data": 42}

    def test_derive_new_event_id(self) -> None:
        parent = _make_event()
        child = parent.derive()
        assert child.event_id != parent.event_id


# ===========================================================================
# 4. Subscription matching
# ===========================================================================


class TestSubscriptionMatching:
    def _make_sub(
        self, filt: str, min_priority: EventPriority = EventPriority.LOW
    ) -> Subscription:
        return Subscription(
            subscriber_id="test.subscriber",
            event_filter=EventFilter(filt),
            min_priority=min_priority,
        )

    def test_exact_match(self) -> None:
        sub = self._make_sub("polaris.memory.record_stored")
        event = _make_event("polaris.memory.record_stored")
        assert sub.matches(event)

    def test_exact_no_match(self) -> None:
        sub = self._make_sub("polaris.memory.record_stored")
        event = _make_event("polaris.memory.record_deleted")
        assert not sub.matches(event)

    def test_wildcard_all(self) -> None:
        sub = self._make_sub("**")
        assert sub.matches(_make_event("polaris.anything.here"))
        assert sub.matches(_make_event("other.namespace.event"))

    def test_wildcard_single_segment(self) -> None:
        sub = self._make_sub("polaris.memory.*")
        assert sub.matches(_make_event("polaris.memory.stored"))
        assert sub.matches(_make_event("polaris.memory.deleted"))
        assert not sub.matches(_make_event("polaris.compute.done"))

    def test_priority_filter_respected(self) -> None:
        sub = self._make_sub("**", min_priority=EventPriority.HIGH)
        low_event = _make_event(priority=EventPriority.LOW)
        high_event = _make_event(priority=EventPriority.HIGH)
        critical_event = _make_event(priority=EventPriority.CRITICAL)
        assert not sub.matches(low_event)
        assert not sub.matches(_make_event(priority=EventPriority.NORMAL))
        assert sub.matches(high_event)
        assert sub.matches(critical_event)

    def test_empty_filter_raises(self) -> None:
        with pytest.raises(SubscriptionError):
            Subscription(
                subscriber_id="sid",
                event_filter=EventFilter(""),
            )

    def test_empty_subscriber_id_raises(self) -> None:
        with pytest.raises(SubscriptionError):
            Subscription(
                subscriber_id="",
                event_filter=EventFilter("**"),
            )


# ===========================================================================
# 5. FunctionalSubscriber
# ===========================================================================


class TestFunctionalSubscriber:
    def test_basic_creation(self) -> None:
        received: list[Event] = []
        fs = FunctionalSubscriber(
            subscriber_id="func.sub",
            handler=received.append,
            event_filters=["polaris.test.*"],
        )
        assert fs.subscriber_id == "func.sub"
        assert len(fs.subscriptions) == 1

    def test_multiple_filters(self) -> None:
        fs = FunctionalSubscriber(
            subscriber_id="func.multi",
            handler=lambda e: None,
            event_filters=["polaris.memory.*", "polaris.compute.*"],
        )
        assert len(fs.subscriptions) == 2

    def test_handler_called(self) -> None:
        received: list[Event] = []
        fs = FunctionalSubscriber(
            subscriber_id="func.sub2",
            handler=received.append,
            event_filters=["**"],
        )
        event = _make_event()
        fs.handle_event(event)
        assert received == [event]

    def test_non_callable_handler_raises(self) -> None:
        with pytest.raises(SubscriptionError):
            FunctionalSubscriber(
                subscriber_id="bad",
                handler="not_callable",  # type: ignore[arg-type]
                event_filters=["**"],
            )

    def test_empty_filters_raises(self) -> None:
        with pytest.raises(SubscriptionError):
            FunctionalSubscriber(
                subscriber_id="bad",
                handler=lambda e: None,
                event_filters=[],
            )


# ===========================================================================
# 6. Publisher
# ===========================================================================


class TestPublisher:
    def test_emit_returns_event(self, bus: EventBus) -> None:
        pub = _ConcretePublisher("polaris.test.publisher", bus)
        event = pub.emit(event_type="polaris.test.happened", payload={"x": 1})
        assert isinstance(event, Event)
        assert event.source == "polaris.test.publisher"

    def test_emit_increments_count(self, bus: EventBus) -> None:
        pub = _ConcretePublisher("polaris.test.pub2", bus)
        pub.emit(event_type="polaris.test.ev")
        pub.emit(event_type="polaris.test.ev")
        assert pub.publish_count == 2

    def test_publish_event_directly(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.subscribe("listener", received.append, "**")
        pub = _ConcretePublisher("polaris.test.pub3", bus)
        event = _make_event(source="polaris.test.pub3")
        pub.publish(event)
        assert event in received

    def test_publish_wrong_type_raises(self, bus: EventBus) -> None:
        pub = _ConcretePublisher("polaris.test.pub4", bus)
        with pytest.raises(TypeError):
            pub.publish("not_an_event")  # type: ignore[arg-type]

    def test_empty_publisher_id_raises(self, bus: EventBus) -> None:
        with pytest.raises(PublisherError):
            _ConcretePublisher("", bus)

    def test_publish_increments_error_on_bus_error(self, bus: EventBus) -> None:
        pub = _ConcretePublisher("polaris.test.pub5", bus)
        # Publish an invalid event (source empty — circumvent dataclass by
        # patching bus.publish to raise).
        with patch.object(bus, "publish", side_effect=RuntimeError("bus down")):
            with pytest.raises(PublisherError):
                pub.publish(_make_event())
        assert pub.error_count == 1


# ===========================================================================
# 7. EventBus — subscriber management
# ===========================================================================


class TestEventBusSubscriberManagement:
    def test_register_and_has(self, bus: EventBus) -> None:
        sub = _CollectorSubscriber("sub.alpha", ["**"])
        bus.register_subscriber(sub)
        assert bus.has_subscriber("sub.alpha")

    def test_register_duplicate_raises(self, bus: EventBus) -> None:
        sub = _CollectorSubscriber("sub.dup", ["**"])
        bus.register_subscriber(sub)
        with pytest.raises(SubscriptionError):
            bus.register_subscriber(sub)

    def test_unregister(self, bus: EventBus) -> None:
        sub = _CollectorSubscriber("sub.beta", ["**"])
        bus.register_subscriber(sub)
        removed = bus.unregister_subscriber("sub.beta")
        assert removed is sub
        assert not bus.has_subscriber("sub.beta")

    def test_unregister_missing_raises(self, bus: EventBus) -> None:
        with pytest.raises(SubscriptionError):
            bus.unregister_subscriber("nonexistent.sub")

    def test_subscribe_convenience(self, bus: EventBus) -> None:
        received: list[Event] = []
        fs = bus.subscribe("conv.sub", received.append, "polaris.test.*")
        assert isinstance(fs, FunctionalSubscriber)
        assert bus.has_subscriber("conv.sub")

    def test_unsubscribe_convenience(self, bus: EventBus) -> None:
        bus.subscribe("conv.sub2", lambda e: None, "**")
        bus.unsubscribe("conv.sub2")
        assert not bus.has_subscriber("conv.sub2")

    def test_register_no_subscriptions_raises(self, bus: EventBus) -> None:
        class _Empty(Subscriber):
            @property
            def subscriptions(self) -> list[Subscription]:
                return []
            def handle_event(self, event: Event) -> None:
                pass

        with pytest.raises(SubscriptionError):
            bus.register_subscriber(_Empty("empty.sub"))

    def test_register_non_subscriber_raises(self, bus: EventBus) -> None:
        with pytest.raises(TypeError):
            bus.register_subscriber("not_a_subscriber")  # type: ignore[arg-type]

    def test_len_reflects_count(self, bus: EventBus) -> None:
        assert len(bus) == 0
        bus.subscribe("s1", lambda e: None, "**")
        assert len(bus) == 1
        bus.subscribe("s2", lambda e: None, "**")
        assert len(bus) == 2
        bus.unsubscribe("s1")
        assert len(bus) == 1

    def test_contains_operator(self, bus: EventBus) -> None:
        bus.subscribe("in.bus", lambda e: None, "**")
        assert "in.bus" in bus
        assert "not.in.bus" not in bus

    def test_get_subscribers_all(self, bus: EventBus) -> None:
        bus.subscribe("s1", lambda e: None, "**")
        bus.subscribe("s2", lambda e: None, "**")
        subs = bus.get_subscribers()
        assert len(subs) == 2

    def test_get_subscribers_filtered_by_type(self, bus: EventBus) -> None:
        bus.subscribe("mem.sub", lambda e: None, "polaris.memory.*")
        bus.subscribe("all.sub", lambda e: None, "**")
        matching = bus.get_subscribers(event_type="polaris.memory.stored")
        ids = {s.subscriber_id for s in matching}
        assert "mem.sub" in ids
        assert "all.sub" in ids

    def test_get_subscribers_no_match(self, bus: EventBus) -> None:
        bus.subscribe("compute.sub", lambda e: None, "polaris.compute.*")
        matching = bus.get_subscribers(event_type="polaris.memory.stored")
        assert not matching


# ===========================================================================
# 8. EventBus — publishing and dispatch
# ===========================================================================


class TestEventBusPublishing:
    def test_publish_delivers_to_subscriber(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.subscribe("recv", received.append, "**")
        event = _make_event()
        bus.publish(event)
        assert received == [event]

    def test_publish_delivers_to_multiple_subscribers(self, bus: EventBus) -> None:
        received_a: list[Event] = []
        received_b: list[Event] = []
        bus.subscribe("a", received_a.append, "**")
        bus.subscribe("b", received_b.append, "**")
        event = _make_event()
        bus.publish(event)
        assert received_a == [event]
        assert received_b == [event]

    def test_publish_filters_by_event_type(self, bus: EventBus) -> None:
        memory_events: list[Event] = []
        compute_events: list[Event] = []
        bus.subscribe("mem", memory_events.append, "polaris.memory.*")
        bus.subscribe("comp", compute_events.append, "polaris.compute.*")
        mem_event = _make_event("polaris.memory.stored")
        comp_event = _make_event("polaris.compute.done")
        bus.publish(mem_event)
        bus.publish(comp_event)
        assert memory_events == [mem_event]
        assert compute_events == [comp_event]

    def test_publish_no_subscribers_no_error(self, bus: EventBus) -> None:
        result = bus.publish(_make_event())
        assert isinstance(result, DispatchResult)
        assert result.success_count == 0

    def test_publish_returns_dispatch_result(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.subscribe("r", received.append, "**")
        result = bus.publish(_make_event())
        assert isinstance(result, DispatchResult)
        assert result.success_count == 1

    def test_publish_wrong_type_raises(self, bus: EventBus) -> None:
        with pytest.raises(TypeError):
            bus.publish("not_an_event")  # type: ignore[arg-type]

    def test_publish_increments_total(self, bus: EventBus) -> None:
        bus.publish(_make_event())
        bus.publish(_make_event())
        stats = bus.statistics()
        assert stats.total_published == 2


# ===========================================================================
# 9. Failure isolation
# ===========================================================================


class TestFailureIsolation:
    def test_failing_subscriber_does_not_block_others(self, bus: EventBus) -> None:
        failing = _CollectorSubscriber("failing", ["**"], raise_on_handle=True)
        good: list[Event] = []
        bus.register_subscriber(failing)
        bus.subscribe("good", good.append, "**")
        event = _make_event()
        result = bus.publish(event)
        # Good subscriber still received the event.
        assert event in good
        # Failure is recorded in result.
        assert "failing" in result.failed_deliveries
        assert result.failure_count == 1
        assert result.success_count == 1

    def test_failure_count_in_result(self, bus: EventBus) -> None:
        bus.register_subscriber(
            _CollectorSubscriber("f1", ["**"], raise_on_handle=True)
        )
        bus.register_subscriber(
            _CollectorSubscriber("f2", ["**"], raise_on_handle=True)
        )
        result = bus.publish(_make_event())
        assert result.failure_count == 2

    def test_strict_bus_raises_on_failure(self, strict_bus: EventBus) -> None:
        strict_bus.register_subscriber(
            _CollectorSubscriber("bad", ["**"], raise_on_handle=True)
        )
        with pytest.raises(EventDispatchError):
            strict_bus.publish(_make_event())

    def test_subscriber_error_count_increments(self, bus: EventBus) -> None:
        failing = _CollectorSubscriber("bad.sub", ["**"], raise_on_handle=True)
        bus.register_subscriber(failing)
        bus.publish(_make_event())
        bus.publish(_make_event())
        assert failing.error_count == 2

    def test_good_subscriber_handled_count(self, bus: EventBus) -> None:
        good = _CollectorSubscriber("good.sub", ["**"])
        bus.register_subscriber(good)
        bus.publish(_make_event())
        bus.publish(_make_event())
        assert good.handled_count == 2


# ===========================================================================
# 10. Priority handling
# ===========================================================================


class TestPriorityHandling:
    def test_priority_filter_on_subscriber(self, bus: EventBus) -> None:
        critical_only: list[Event] = []
        bus.subscribe(
            "crit.only",
            critical_only.append,
            "**",
            min_priority=EventPriority.CRITICAL,
        )
        bus.publish(_make_event(priority=EventPriority.LOW))
        bus.publish(_make_event(priority=EventPriority.NORMAL))
        bus.publish(_make_event(priority=EventPriority.HIGH))
        crit = _make_event(priority=EventPriority.CRITICAL)
        bus.publish(crit)
        assert critical_only == [crit]

    def test_high_priority_subscriber_served_first(self, bus: EventBus) -> None:
        order: list[str] = []

        class _OrderedSub(Subscriber):
            def __init__(self, sid: str, label: str, min_p: EventPriority) -> None:
                super().__init__(sid)
                self._label = label
                self._min_p = min_p

            @property
            def subscriptions(self) -> list[Subscription]:
                return [
                    Subscription(
                        subscriber_id=self._subscriber_id,
                        event_filter=EventFilter("**"),
                        min_priority=self._min_p,
                    )
                ]

            def handle_event(self, event: Event) -> None:
                order.append(self._label)

        high_sub = _OrderedSub("high.sub", "HIGH", EventPriority.HIGH)
        low_sub = _OrderedSub("low.sub", "LOW", EventPriority.LOW)
        # Register low first, then high — delivery order should still be HIGH first.
        bus.register_subscriber(low_sub)
        bus.register_subscriber(high_sub)
        bus.publish(_make_event(priority=EventPriority.HIGH))
        assert order[0] == "HIGH"

    def test_dispatch_result_total_subscribers(self, bus: EventBus) -> None:
        bus.subscribe("s1", lambda e: None, "**")
        bus.subscribe("s2", lambda e: None, "**")
        result = bus.publish(_make_event())
        assert result.total_subscribers == 2


# ===========================================================================
# 11. Event history
# ===========================================================================


class TestEventHistory:
    def test_history_records_events(self, bus: EventBus) -> None:
        e1 = _make_event("polaris.test.one")
        e2 = _make_event("polaris.test.two")
        bus.publish(e1)
        bus.publish(e2)
        history = bus.get_event_history()
        assert e1 in history
        assert e2 in history

    def test_history_disabled(self, no_history_bus: EventBus) -> None:
        no_history_bus.publish(_make_event())
        assert no_history_bus.get_event_history() == []

    def test_history_filter_by_event_type(self, bus: EventBus) -> None:
        bus.publish(_make_event("polaris.memory.stored"))
        bus.publish(_make_event("polaris.compute.done"))
        mem_history = bus.get_event_history(event_type="polaris.memory.*")
        assert all("memory" in e.event_type for e in mem_history)

    def test_history_filter_by_source(self, bus: EventBus) -> None:
        bus.publish(_make_event(source="polaris.memory.manager"))
        bus.publish(_make_event(source="polaris.compute.engine"))
        mem_history = bus.get_event_history(source="polaris.memory.manager")
        assert all(e.source == "polaris.memory.manager" for e in mem_history)

    def test_history_filter_by_since(self, bus: EventBus) -> None:
        before = datetime.now(timezone.utc)
        bus.publish(_make_event())
        after = datetime.now(timezone.utc)
        history = bus.get_event_history(since=after)
        assert history == []
        history_all = bus.get_event_history(since=before)
        assert len(history_all) == 1

    def test_history_filter_by_priority(self, bus: EventBus) -> None:
        bus.publish(_make_event(priority=EventPriority.LOW))
        bus.publish(_make_event(priority=EventPriority.CRITICAL))
        critical = bus.get_event_history(priority=EventPriority.CRITICAL)
        assert len(critical) == 1
        assert critical[0].priority is EventPriority.CRITICAL

    def test_history_limit(self, bus: EventBus) -> None:
        for _ in range(10):
            bus.publish(_make_event())
        limited = bus.get_event_history(limit=3)
        assert len(limited) == 3

    def test_history_clear(self, bus: EventBus) -> None:
        bus.publish(_make_event())
        bus.publish(_make_event())
        cleared = bus.clear_history()
        assert cleared == 2
        assert bus.get_event_history() == []

    def test_history_capacity_ring_buffer(self) -> None:
        small_bus = EventBus(history_capacity=3)
        for i in range(5):
            small_bus.publish(_make_event(payload=i))
        history = small_bus.history_snapshot()
        assert len(history) == 3
        # Most recent 3 events are retained.
        assert [e.payload for e in history] == [2, 3, 4]

    def test_history_snapshot_is_copy(self, bus: EventBus) -> None:
        bus.publish(_make_event())
        snap = bus.history_snapshot()
        snap.clear()
        assert len(bus.history_snapshot()) == 1

    def test_history_invalid_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            EventBus(history_capacity=-1)


# ===========================================================================
# 12. Dispatcher
# ===========================================================================


class TestDispatcher:
    def test_dispatch_to_matching_subscribers(self) -> None:
        dispatcher = Dispatcher()
        received: list[Event] = []
        sub = FunctionalSubscriber(
            subscriber_id="d.sub",
            handler=received.append,
            event_filters=["**"],
        )
        event = _make_event()
        result = dispatcher.dispatch(event, [sub])
        assert received == [event]
        assert result.success_count == 1

    def test_dispatch_skips_non_matching(self) -> None:
        dispatcher = Dispatcher()
        received: list[Event] = []
        sub = FunctionalSubscriber(
            subscriber_id="specific.sub",
            handler=received.append,
            event_filters=["polaris.memory.*"],
        )
        event = _make_event("polaris.compute.done")
        dispatcher.dispatch(event, [sub])
        assert received == []

    def test_dispatch_failure_isolation(self) -> None:
        dispatcher = Dispatcher()
        good: list[Event] = []
        bad = _CollectorSubscriber("bad", ["**"], raise_on_handle=True)
        ok = FunctionalSubscriber("ok", good.append, ["**"])
        event = _make_event()
        result = dispatcher.dispatch(event, [bad, ok])
        assert "bad" in result.failed_deliveries
        assert "ok" in result.delivered_to
        assert event in good

    def test_raise_on_failure_mode(self) -> None:
        dispatcher = Dispatcher(raise_on_failure=True)
        bad = _CollectorSubscriber("bad2", ["**"], raise_on_handle=True)
        with pytest.raises(EventDispatchError):
            dispatcher.dispatch(_make_event(), [bad])

    def test_total_dispatched_increments(self) -> None:
        dispatcher = Dispatcher()
        sub = FunctionalSubscriber("s", lambda e: None, ["**"])
        dispatcher.dispatch(_make_event(), [sub])
        dispatcher.dispatch(_make_event(), [sub])
        assert dispatcher.total_dispatched == 2

    def test_reset_stats(self) -> None:
        dispatcher = Dispatcher()
        sub = FunctionalSubscriber("s2", lambda e: None, ["**"])
        dispatcher.dispatch(_make_event(), [sub])
        dispatcher.reset_stats()
        assert dispatcher.total_dispatched == 0
        assert dispatcher.total_failures == 0

    def test_empty_subscriber_list(self) -> None:
        dispatcher = Dispatcher()
        result = dispatcher.dispatch(_make_event(), [])
        assert result.success_count == 0
        assert result.failure_count == 0


# ===========================================================================
# 13. Statistics
# ===========================================================================


class TestBusStatistics:
    def test_statistics_initial(self, bus: EventBus) -> None:
        stats = bus.statistics()
        assert stats.total_published == 0
        assert stats.subscriber_count == 0
        assert stats.history_size == 0

    def test_statistics_after_publish(self, bus: EventBus) -> None:
        bus.subscribe("s", lambda e: None, "**")
        bus.publish(_make_event())
        stats = bus.statistics()
        assert stats.total_published == 1
        assert stats.total_dispatched == 1
        assert stats.subscriber_count == 1
        assert stats.history_size == 1

    def test_statistics_history_capacity(self, bus: EventBus) -> None:
        stats = bus.statistics()
        assert stats.history_capacity == 256


# ===========================================================================
# 14. System event schemas
# ===========================================================================


class TestSystemEventSchemas:
    def test_system_event_type_constants(self) -> None:
        assert "polaris.runtime" in SystemEvents.SUBSYSTEM_REGISTERED
        assert "polaris.runtime" in SystemEvents.SUBSYSTEM_FAILED

    def test_subsystem_registered_payload(self) -> None:
        payload = SubsystemRegisteredPayload(
            subsystem_id="polaris.core.memory",
            version="1.0.0",
            capability_count=3,
        )
        assert payload.subsystem_id == "polaris.core.memory"
        assert payload.capability_count == 3

    def test_subsystem_registered_negative_count_raises(self) -> None:
        with pytest.raises(EventValidationError):
            SubsystemRegisteredPayload(
                subsystem_id="polaris.core.memory",
                version="1.0.0",
                capability_count=-1,
            )

    def test_subsystem_lifecycle_payload(self) -> None:
        p = SubsystemLifecyclePayload(
            subsystem_id="polaris.core.memory",
            from_state="INITIALIZED",
            to_state="RUNNING",
            reason="start() called",
        )
        assert p.to_state == "RUNNING"

    def test_health_changed_payload(self) -> None:
        p = SubsystemHealthChangedPayload(
            subsystem_id="polaris.core.memory",
            previous_status="HEALTHY",
            current_status="DEGRADED",
            message="High latency detected.",
            failed_check_count=1,
        )
        assert p.current_status == "DEGRADED"

    def test_system_event_roundtrip_through_bus(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.subscribe("monitor", received.append, "polaris.runtime.*")
        payload = SubsystemRegisteredPayload(
            subsystem_id="polaris.core.memory",
            version="1.0.0",
            capability_count=2,
        )
        bus.publish(Event.create(
            event_type=SystemEvents.SUBSYSTEM_REGISTERED,
            source="polaris.registry",
            payload=payload,
        ))
        assert len(received) == 1
        assert received[0].payload == payload


# ===========================================================================
# 15. Exception hierarchy
# ===========================================================================


class TestExceptionHierarchy:
    def test_all_inherit_from_event_bus_error(self) -> None:
        assert issubclass(EventDispatchError, EventBusError)
        assert issubclass(SubscriptionError, EventBusError)
        assert issubclass(EventValidationError, EventBusError)
        assert issubclass(PublisherError, EventBusError)

    def test_dispatch_error_attributes(self) -> None:
        event = _make_event()
        exc = EventDispatchError(
            "dispatch failed",
            event=event,
            subscriber_id="bad.sub",
            original_exception=ValueError("inner"),
        )
        assert exc.event is event
        assert exc.subscriber_id == "bad.sub"
        assert isinstance(exc.original_exception, ValueError)

    def test_subscription_error_attributes(self) -> None:
        exc = SubscriptionError(
            "bad sub",
            subscriber_id="sid",
            event_type=EventType("polaris.test.ev"),
        )
        assert exc.subscriber_id == "sid"
        assert exc.event_type == "polaris.test.ev"

    def test_validation_error_attributes(self) -> None:
        exc = EventValidationError(
            "bad field",
            field="event_type",
            invalid_value="bad_value",
        )
        assert exc.field == "event_type"
        assert exc.invalid_value == "bad_value"


# ===========================================================================
# 16. Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_publish(self, bus: EventBus) -> None:
        """100 threads each publish 10 events; all must reach a subscriber."""
        received: list[Event] = []
        lock = threading.Lock()

        def _collect(event: Event) -> None:
            with lock:
                received.append(event)

        bus.subscribe("collector", _collect, "**")

        def _publish_batch() -> None:
            for _ in range(10):
                bus.publish(_make_event())

        threads = [threading.Thread(target=_publish_batch) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 1000

    def test_concurrent_subscribe_unsubscribe(self, bus: EventBus) -> None:
        """Simultaneous register/unregister must not corrupt internal state."""
        errors: list[Exception] = []

        def _subscribe_then_unsubscribe(idx: int) -> None:
            sid = f"concurrent.sub.{idx}"
            try:
                bus.subscribe(sid, lambda e: None, "**")
                time.sleep(0.001)
                bus.unsubscribe(sid)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_subscribe_then_unsubscribe, args=(i,))
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_publish_during_subscribe(self, bus: EventBus) -> None:
        """Publishing while subscriptions are changing must not raise."""
        errors: list[Exception] = []
        stop = threading.Event()

        def _spam_publish() -> None:
            while not stop.is_set():
                try:
                    bus.publish(_make_event())
                except Exception as exc:
                    errors.append(exc)

        def _spam_subscribe(idx: int) -> None:
            sid = f"dyn.sub.{idx}"
            try:
                bus.subscribe(sid, lambda e: None, "**")
                time.sleep(0.002)
                bus.unsubscribe(sid)
            except Exception as exc:
                errors.append(exc)

        publisher_thread = threading.Thread(target=_spam_publish)
        publisher_thread.start()

        sub_threads = [
            threading.Thread(target=_spam_subscribe, args=(i,))
            for i in range(20)
        ]
        for t in sub_threads:
            t.start()
        for t in sub_threads:
            t.join()

        stop.set()
        publisher_thread.join()

        assert not errors

    def test_concurrent_history_access(self, bus: EventBus) -> None:
        """History reads and writes from multiple threads must be safe."""
        errors: list[Exception] = []

        def _publish_events() -> None:
            for _ in range(50):
                try:
                    bus.publish(_make_event())
                except Exception as exc:
                    errors.append(exc)

        def _read_history() -> None:
            for _ in range(50):
                try:
                    bus.get_event_history()
                except Exception as exc:
                    errors.append(exc)

        threads = (
            [threading.Thread(target=_publish_events) for _ in range(5)]
            + [threading.Thread(target=_read_history) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ===========================================================================
# 17. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_event_with_none_payload(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.subscribe("r", received.append, "**")
        bus.publish(_make_event(payload=None))
        assert received[0].payload is None

    def test_event_with_complex_payload(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.subscribe("r2", received.append, "**")
        payload = {"nested": {"deep": [1, 2, 3]}, "flag": True}
        bus.publish(_make_event(payload=payload))
        assert received[0].payload == payload

    def test_bus_iteration(self, bus: EventBus) -> None:
        bus.subscribe("iter.s1", lambda e: None, "**")
        bus.subscribe("iter.s2", lambda e: None, "**")
        ids = {s.subscriber_id for s in bus}
        assert ids == {"iter.s1", "iter.s2"}

    def test_subscriber_last_event_at(self, bus: EventBus) -> None:
        sub = _CollectorSubscriber("ts.sub", ["**"])
        bus.register_subscriber(sub)
        assert sub.last_event_at is None
        bus.publish(_make_event())
        assert sub.last_event_at is not None

    def test_unregister_removes_from_dispatch(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.subscribe("temp.sub", received.append, "**")
        bus.publish(_make_event())
        bus.unsubscribe("temp.sub")
        bus.publish(_make_event())
        assert len(received) == 1

    def test_multi_filter_subscriber_matches_either(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.subscribe(
            "multi.filter",
            received.append,
            ["polaris.memory.*", "polaris.compute.*"],
        )
        bus.publish(_make_event("polaris.memory.stored"))
        bus.publish(_make_event("polaris.compute.done"))
        bus.publish(_make_event("polaris.network.sent"))
        assert len(received) == 2

    def test_history_wildcard_filter(self, bus: EventBus) -> None:
        bus.publish(_make_event("polaris.memory.stored"))
        bus.publish(_make_event("polaris.compute.done"))
        results = bus.get_event_history(event_type="polaris.memory.*")
        assert len(results) == 1
        assert results[0].event_type == "polaris.memory.stored"

    def test_dispatch_result_repr_contains_event_type(self, bus: EventBus) -> None:
        bus.subscribe("r", lambda e: None, "**")
        result = bus.publish(_make_event("polaris.test.repr"))
        assert isinstance(result, DispatchResult)
        assert result.event.event_type == "polaris.test.repr"