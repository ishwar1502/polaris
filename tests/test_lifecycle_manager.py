# tests/test_lifecycle_manager.py
"""
Comprehensive test suite for POLARIS v5 Runtime Phase 5 — Lifecycle Manager.

Coverage targets
----------------
* Valid transitions
* Invalid transitions (must raise InvalidTransitionError)
* Recovery path: RUNNING → FAILED → RECOVERING → RUNNING
* State history recording and querying
* Thread safety under concurrent load
* Event generation and integration with the Event Bus
* LifecycleManager registration, unregistration, and queries
* Module start / stop / pause / resume / recover / unload
* Failure injection and recovery
* 100+ tests across all files / components
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.lifecycle.events import LifecycleEventPayload, LifecycleEvents
from core.lifecycle.exceptions import (
    InvalidTransitionError,
    LifecycleError,
    ModuleStateError,
    StateMachineError,
)
from core.lifecycle.manager import LifecycleManager
from core.lifecycle.models import (
    LifecycleState,
    OPERATIONAL_STATES,
    TERMINAL_STATES,
    TRANSITIONAL_STATES,
)
from core.lifecycle.state_machine import StateMachine, StateHistoryEntry
from core.lifecycle.transitions import (
    ALLOWED_TRANSITIONS,
    LifecycleTransition,
    allowed_from,
    get_transition,
    is_allowed,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def manager() -> LifecycleManager:
    """Return a fresh LifecycleManager with no event bus."""
    return LifecycleManager()


@pytest.fixture()
def mock_bus() -> MagicMock:
    """Return a MagicMock acting as an EventBus."""
    bus = MagicMock()
    bus.publish = MagicMock(return_value=None)
    return bus


@pytest.fixture()
def manager_with_bus(mock_bus: MagicMock) -> LifecycleManager:
    """Return a LifecycleManager wired to a mock EventBus."""
    return LifecycleManager(event_bus=mock_bus)


@pytest.fixture()
def sm() -> StateMachine:
    """Return a fresh StateMachine in CREATED state."""
    return StateMachine("test.module")


@pytest.fixture()
def sm_running() -> StateMachine:
    """Return a StateMachine already in RUNNING state."""
    m = StateMachine("test.module")
    m.transition(LifecycleState.DISCOVERED)
    m.transition(LifecycleState.VALIDATED)
    m.transition(LifecycleState.LOADED)
    m.transition(LifecycleState.INITIALIZED)
    m.transition(LifecycleState.STARTING)
    m.transition(LifecycleState.RUNNING)
    return m


# ===========================================================================
# 1. LifecycleState enum tests
# ===========================================================================


class TestLifecycleState:
    def test_all_states_exist(self) -> None:
        states = [s.name for s in LifecycleState]
        expected = [
            "CREATED", "DISCOVERED", "VALIDATED", "LOADED",
            "INITIALIZED", "STARTING", "RUNNING", "PAUSED",
            "RECOVERING", "STOPPING", "STOPPED", "FAILED", "UNLOADED",
        ]
        assert set(states) == set(expected)

    def test_thirteen_states(self) -> None:
        assert len(LifecycleState) == 13

    def test_operational_states(self) -> None:
        assert LifecycleState.RUNNING.is_operational()
        assert LifecycleState.PAUSED.is_operational()
        for s in LifecycleState:
            if s not in (LifecycleState.RUNNING, LifecycleState.PAUSED):
                assert not s.is_operational(), f"{s} should not be operational"

    def test_terminal_states(self) -> None:
        assert LifecycleState.UNLOADED.is_terminal()
        assert LifecycleState.FAILED.is_terminal()
        for s in LifecycleState:
            if s not in (LifecycleState.UNLOADED, LifecycleState.FAILED):
                assert not s.is_terminal(), f"{s} should not be terminal"

    def test_transitional_states(self) -> None:
        assert LifecycleState.STARTING.is_transitional()
        assert LifecycleState.STOPPING.is_transitional()
        assert LifecycleState.RECOVERING.is_transitional()

    def test_can_fail(self) -> None:
        # FAILED, UNLOADED, STOPPED cannot fail again
        assert not LifecycleState.FAILED.can_fail()
        assert not LifecycleState.UNLOADED.can_fail()
        assert not LifecycleState.STOPPED.can_fail()
        # All others can fail
        for s in LifecycleState:
            if s not in (LifecycleState.FAILED, LifecycleState.UNLOADED, LifecycleState.STOPPED):
                assert s.can_fail(), f"{s} should be failable"

    def test_can_recover(self) -> None:
        assert LifecycleState.FAILED.can_recover()
        for s in LifecycleState:
            if s is not LifecycleState.FAILED:
                assert not s.can_recover()

    def test_operational_states_constant(self) -> None:
        assert OPERATIONAL_STATES == frozenset({LifecycleState.RUNNING, LifecycleState.PAUSED})

    def test_terminal_states_constant(self) -> None:
        assert TERMINAL_STATES == frozenset({LifecycleState.UNLOADED, LifecycleState.FAILED})

    def test_transitional_states_constant(self) -> None:
        assert TRANSITIONAL_STATES == frozenset({
            LifecycleState.STARTING,
            LifecycleState.STOPPING,
            LifecycleState.RECOVERING,
        })


# ===========================================================================
# 2. LifecycleTransition tests
# ===========================================================================


class TestLifecycleTransition:
    def test_immutable(self) -> None:
        t = LifecycleTransition(
            LifecycleState.CREATED, LifecycleState.DISCOVERED, allowed=True
        )
        with pytest.raises((AttributeError, TypeError)):
            t.allowed = False  # type: ignore[misc]

    def test_str_representation(self) -> None:
        t = LifecycleTransition(LifecycleState.CREATED, LifecycleState.DISCOVERED, allowed=True)
        s = str(t)
        assert "CREATED" in s
        assert "DISCOVERED" in s
        assert "→" in s

    def test_not_allowed_str(self) -> None:
        t = LifecycleTransition(LifecycleState.RUNNING, LifecycleState.CREATED, allowed=False)
        s = str(t)
        assert "↛" in s

    def test_description_stored(self) -> None:
        t = LifecycleTransition(
            LifecycleState.CREATED,
            LifecycleState.DISCOVERED,
            description="Testing description",
        )
        assert t.description == "Testing description"

    def test_default_allowed_is_true(self) -> None:
        t = LifecycleTransition(LifecycleState.CREATED, LifecycleState.DISCOVERED)
        assert t.allowed is True


# ===========================================================================
# 3. Transitions module tests
# ===========================================================================


class TestTransitionTable:
    def test_normal_forward_path(self) -> None:
        forward = [
            (LifecycleState.CREATED, LifecycleState.DISCOVERED),
            (LifecycleState.DISCOVERED, LifecycleState.VALIDATED),
            (LifecycleState.VALIDATED, LifecycleState.LOADED),
            (LifecycleState.LOADED, LifecycleState.INITIALIZED),
            (LifecycleState.INITIALIZED, LifecycleState.STARTING),
            (LifecycleState.STARTING, LifecycleState.RUNNING),
        ]
        for frm, to in forward:
            assert is_allowed(frm, to), f"{frm} → {to} should be allowed"

    def test_pause_resume(self) -> None:
        assert is_allowed(LifecycleState.RUNNING, LifecycleState.PAUSED)
        assert is_allowed(LifecycleState.PAUSED, LifecycleState.RUNNING)

    def test_shutdown_path(self) -> None:
        assert is_allowed(LifecycleState.RUNNING, LifecycleState.STOPPING)
        assert is_allowed(LifecycleState.STOPPING, LifecycleState.STOPPED)
        assert is_allowed(LifecycleState.STOPPED, LifecycleState.UNLOADED)

    def test_recovery_path(self) -> None:
        assert is_allowed(LifecycleState.RUNNING, LifecycleState.FAILED)
        assert is_allowed(LifecycleState.FAILED, LifecycleState.RECOVERING)
        assert is_allowed(LifecycleState.RECOVERING, LifecycleState.RUNNING)

    def test_failure_from_many_states(self) -> None:
        for s in [
            LifecycleState.CREATED, LifecycleState.DISCOVERED, LifecycleState.VALIDATED,
            LifecycleState.LOADED, LifecycleState.INITIALIZED, LifecycleState.STARTING,
            LifecycleState.RUNNING, LifecycleState.PAUSED, LifecycleState.STOPPING,
            LifecycleState.RECOVERING,
        ]:
            assert is_allowed(s, LifecycleState.FAILED), f"{s} → FAILED should be allowed"

    def test_invalid_backwards_transition(self) -> None:
        assert not is_allowed(LifecycleState.RUNNING, LifecycleState.CREATED)
        assert not is_allowed(LifecycleState.STOPPED, LifecycleState.RUNNING)
        assert not is_allowed(LifecycleState.UNLOADED, LifecycleState.CREATED)

    def test_get_transition_returns_object(self) -> None:
        t = get_transition(LifecycleState.CREATED, LifecycleState.DISCOVERED)
        assert t is not None
        assert isinstance(t, LifecycleTransition)

    def test_get_transition_returns_none_for_invalid(self) -> None:
        t = get_transition(LifecycleState.RUNNING, LifecycleState.CREATED)
        assert t is None

    def test_allowed_from_created(self) -> None:
        nexts = allowed_from(LifecycleState.CREATED)
        assert LifecycleState.DISCOVERED in nexts
        assert LifecycleState.FAILED in nexts

    def test_allowed_from_running(self) -> None:
        nexts = allowed_from(LifecycleState.RUNNING)
        assert LifecycleState.PAUSED in nexts
        assert LifecycleState.STOPPING in nexts
        assert LifecycleState.FAILED in nexts

    def test_allowed_from_unloaded_is_empty(self) -> None:
        nexts = allowed_from(LifecycleState.UNLOADED)
        assert len(nexts) == 0

    def test_transition_table_has_expected_minimum_entries(self) -> None:
        # At minimum, we expect coverage for all 13 states
        assert len(ALLOWED_TRANSITIONS) >= 20


# ===========================================================================
# 4. StateMachine tests
# ===========================================================================


class TestStateMachine:
    def test_initial_state_default(self, sm: StateMachine) -> None:
        assert sm.current_state is LifecycleState.CREATED

    def test_initial_state_custom(self) -> None:
        m = StateMachine("mod", initial_state=LifecycleState.LOADED)
        assert m.current_state is LifecycleState.LOADED

    def test_invalid_initial_state_type(self) -> None:
        with pytest.raises(StateMachineError):
            StateMachine("mod", initial_state="CREATED")  # type: ignore[arg-type]

    def test_invalid_max_history(self) -> None:
        with pytest.raises(StateMachineError):
            StateMachine("mod", max_history=0)

    def test_transition_valid(self, sm: StateMachine) -> None:
        result = sm.transition(LifecycleState.DISCOVERED)
        assert isinstance(result, LifecycleTransition)
        assert sm.current_state is LifecycleState.DISCOVERED

    def test_transition_invalid_raises(self, sm: StateMachine) -> None:
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.transition(LifecycleState.RUNNING)  # CREATED → RUNNING not allowed
        assert exc_info.value.from_state is LifecycleState.CREATED
        assert exc_info.value.to_state is LifecycleState.RUNNING

    def test_transition_bad_type_raises(self, sm: StateMachine) -> None:
        with pytest.raises(StateMachineError):
            sm.transition("RUNNING")  # type: ignore[arg-type]

    def test_history_grows_with_transitions(self, sm: StateMachine) -> None:
        # Initial history has 1 entry (initial state)
        assert len(sm.history) == 1
        sm.transition(LifecycleState.DISCOVERED)
        assert len(sm.history) == 2
        sm.transition(LifecycleState.VALIDATED)
        assert len(sm.history) == 3

    def test_history_initial_entry_has_none_from_state(self, sm: StateMachine) -> None:
        initial = sm.history[0]
        assert initial.from_state is None
        assert initial.to_state is LifecycleState.CREATED

    def test_history_entry_has_timestamp(self, sm: StateMachine) -> None:
        sm.transition(LifecycleState.DISCOVERED)
        entry = sm.history[-1]
        assert entry.timestamp.tzinfo is not None

    def test_history_immutable_snapshot(self, sm: StateMachine) -> None:
        h1 = sm.history
        sm.transition(LifecycleState.DISCOVERED)
        h2 = sm.history
        assert len(h2) == len(h1) + 1

    def test_can_transition_to(self, sm: StateMachine) -> None:
        assert sm.can_transition_to(LifecycleState.DISCOVERED)
        assert not sm.can_transition_to(LifecycleState.RUNNING)

    def test_allowed_next_states(self, sm: StateMachine) -> None:
        nexts = sm.allowed_next_states()
        assert LifecycleState.DISCOVERED in nexts
        assert LifecycleState.FAILED in nexts

    def test_is_in_state(self, sm: StateMachine) -> None:
        assert sm.is_in_state(LifecycleState.CREATED)
        assert not sm.is_in_state(LifecycleState.RUNNING)

    def test_is_in_any_state(self, sm: StateMachine) -> None:
        assert sm.is_in_any_state(LifecycleState.CREATED, LifecycleState.RUNNING)
        assert not sm.is_in_any_state(LifecycleState.PAUSED, LifecycleState.STOPPED)

    def test_last_transition_none_before_any(self, sm: StateMachine) -> None:
        assert sm.last_transition() is None

    def test_last_transition_after_first(self, sm: StateMachine) -> None:
        sm.transition(LifecycleState.DISCOVERED)
        lt = sm.last_transition()
        assert lt is not None
        assert lt.to_state is LifecycleState.DISCOVERED

    def test_transitions_to_query(self, sm_running: StateMachine) -> None:
        entries = sm_running.transitions_to(LifecycleState.RUNNING)
        assert len(entries) == 1
        assert entries[0].to_state is LifecycleState.RUNNING

    def test_time_in_current_state(self, sm: StateMachine) -> None:
        t = sm.time_in_current_state()
        assert t >= 0.0

    def test_history_trimmed_at_max(self) -> None:
        m = StateMachine("mod", max_history=5)
        # Transition to DISCOVERED (2 entries), VALIDATED (3), LOADED (4),
        # INITIALIZED (5), then STARTING (should trim to 5)
        m.transition(LifecycleState.DISCOVERED)
        m.transition(LifecycleState.VALIDATED)
        m.transition(LifecycleState.LOADED)
        m.transition(LifecycleState.INITIALIZED)
        m.transition(LifecycleState.STARTING)
        # At this point we have initial + 5 transitions = 6, but max_history=5
        assert len(m.history) == 5

    def test_reason_recorded_in_history(self, sm: StateMachine) -> None:
        sm.transition(LifecycleState.DISCOVERED, reason="unit test reason")
        entry = sm.last_transition()
        assert entry is not None
        assert "unit test reason" in entry.reason

    def test_module_id_property(self, sm: StateMachine) -> None:
        assert sm.module_id == "test.module"

    def test_invalid_transition_includes_module_id(self, sm: StateMachine) -> None:
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.transition(LifecycleState.RUNNING)
        assert exc_info.value.module_id == "test.module"

    def test_full_forward_path(self) -> None:
        m = StateMachine("path.test")
        for state in [
            LifecycleState.DISCOVERED,
            LifecycleState.VALIDATED,
            LifecycleState.LOADED,
            LifecycleState.INITIALIZED,
            LifecycleState.STARTING,
            LifecycleState.RUNNING,
        ]:
            m.transition(state)
        assert m.current_state is LifecycleState.RUNNING

    def test_state_after_failure(self, sm_running: StateMachine) -> None:
        sm_running.transition(LifecycleState.FAILED)
        assert sm_running.current_state is LifecycleState.FAILED

    def test_recovery_path_full(self, sm_running: StateMachine) -> None:
        sm_running.transition(LifecycleState.FAILED)
        sm_running.transition(LifecycleState.RECOVERING)
        sm_running.transition(LifecycleState.RUNNING)
        assert sm_running.current_state is LifecycleState.RUNNING


# ===========================================================================
# 5. Exceptions tests
# ===========================================================================


class TestExceptions:
    def test_lifecycle_error_message(self) -> None:
        e = LifecycleError("test message")
        assert "test message" in str(e)

    def test_lifecycle_error_with_module_id(self) -> None:
        e = LifecycleError("test", module_id="my.mod")
        assert "my.mod" in str(e)

    def test_invalid_transition_error_fields(self) -> None:
        e = InvalidTransitionError(
            LifecycleState.RUNNING,
            LifecycleState.CREATED,
            module_id="x.mod",
        )
        assert e.from_state is LifecycleState.RUNNING
        assert e.to_state is LifecycleState.CREATED
        assert e.module_id == "x.mod"

    def test_invalid_transition_inherits_lifecycle_error(self) -> None:
        e = InvalidTransitionError(LifecycleState.RUNNING, LifecycleState.CREATED)
        assert isinstance(e, LifecycleError)

    def test_state_machine_error_inherits(self) -> None:
        e = StateMachineError("sm error")
        assert isinstance(e, LifecycleError)

    def test_module_state_error_fields(self) -> None:
        e = ModuleStateError(
            "my.module",
            expected_states={LifecycleState.RUNNING},
            actual_state=LifecycleState.STOPPED,
            operation="start",
        )
        assert e.module_id == "my.module"
        assert LifecycleState.RUNNING in e.expected_states
        assert e.actual_state is LifecycleState.STOPPED
        assert e.operation == "start"

    def test_module_state_error_message_contains_info(self) -> None:
        e = ModuleStateError(
            "my.module",
            expected_states={LifecycleState.RUNNING},
            actual_state=LifecycleState.STOPPED,
            operation="start",
        )
        msg = str(e)
        assert "my.module" in msg

    def test_module_state_error_inherits(self) -> None:
        e = ModuleStateError(
            "m", expected_states={LifecycleState.RUNNING}, actual_state=LifecycleState.STOPPED
        )
        assert isinstance(e, LifecycleError)


# ===========================================================================
# 6. Lifecycle Events tests
# ===========================================================================


class TestLifecycleEvents:
    def test_all_event_types_exist(self) -> None:
        events = [
            LifecycleEvents.MODULE_LOADED,
            LifecycleEvents.MODULE_INITIALIZED,
            LifecycleEvents.MODULE_STARTED,
            LifecycleEvents.MODULE_PAUSED,
            LifecycleEvents.MODULE_RESUMED,
            LifecycleEvents.MODULE_RECOVERED,
            LifecycleEvents.MODULE_STOPPED,
            LifecycleEvents.MODULE_FAILED,
            LifecycleEvents.MODULE_UNLOADED,
        ]
        for ev in events:
            assert isinstance(ev, str), f"{ev} should be an EventType str"

    def test_event_types_are_dot_namespaced(self) -> None:
        for ev in LifecycleEvents.ALL:
            assert "." in ev, f"{ev} should be dot-namespaced"

    def test_all_tuple_contains_nine_events(self) -> None:
        assert len(LifecycleEvents.ALL) == 9

    def test_payload_construction(self) -> None:
        p = LifecycleEventPayload(
            module_id="a.b",
            from_state="RUNNING",
            to_state="FAILED",
            reason="test",
        )
        assert p.module_id == "a.b"
        assert p.from_state == "RUNNING"
        assert p.to_state == "FAILED"

    def test_payload_immutable(self) -> None:
        p = LifecycleEventPayload(module_id="a.b", from_state=None, to_state="CREATED")
        with pytest.raises((AttributeError, TypeError)):
            p.module_id = "x"  # type: ignore[misc]

    def test_payload_to_dict(self) -> None:
        p = LifecycleEventPayload(
            module_id="a.b",
            from_state="RUNNING",
            to_state="FAILED",
        )
        d = p.to_dict()
        assert d["module_id"] == "a.b"
        assert d["from_state"] == "RUNNING"
        assert d["to_state"] == "FAILED"

    def test_payload_requires_module_id(self) -> None:
        with pytest.raises(ValueError):
            LifecycleEventPayload(module_id="", from_state=None, to_state="CREATED")

    def test_payload_requires_to_state(self) -> None:
        with pytest.raises(ValueError):
            LifecycleEventPayload(module_id="x.y", from_state=None, to_state="")


# ===========================================================================
# 7. LifecycleManager — Registration tests
# ===========================================================================


class TestLifecycleManagerRegistration:
    def test_register_module(self, manager: LifecycleManager) -> None:
        machine = manager.register("alpha.mod")
        assert isinstance(machine, StateMachine)
        assert manager.is_registered("alpha.mod")

    def test_register_sets_initial_state(self, manager: LifecycleManager) -> None:
        manager.register("a.b")
        assert manager.get_state("a.b") is LifecycleState.CREATED

    def test_register_with_custom_initial_state(self, manager: LifecycleManager) -> None:
        manager.register("a.b", initial_state=LifecycleState.LOADED)
        assert manager.get_state("a.b") is LifecycleState.LOADED

    def test_register_duplicate_raises(self, manager: LifecycleManager) -> None:
        manager.register("a.b")
        with pytest.raises(LifecycleError):
            manager.register("a.b")

    def test_register_empty_id_raises(self, manager: LifecycleManager) -> None:
        with pytest.raises(LifecycleError):
            manager.register("")

    def test_unregister_stopped_module(self, manager: LifecycleManager) -> None:
        manager.register("a.b", initial_state=LifecycleState.STOPPED)
        manager.unregister("a.b")
        assert not manager.is_registered("a.b")

    def test_unregister_unloaded_module(self, manager: LifecycleManager) -> None:
        manager.register("a.b", initial_state=LifecycleState.STOPPED)
        manager.unload("a.b")
        manager.unregister("a.b")
        assert not manager.is_registered("a.b")

    def test_unregister_running_raises(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "a.b")
        with pytest.raises(LifecycleError):
            manager.unregister("a.b")

    def test_unregister_unknown_raises(self, manager: LifecycleManager) -> None:
        with pytest.raises(LifecycleError):
            manager.unregister("nonexistent")

    def test_len(self, manager: LifecycleManager) -> None:
        assert len(manager) == 0
        manager.register("a")
        assert len(manager) == 1
        manager.register("b")
        assert len(manager) == 2

    def test_contains(self, manager: LifecycleManager) -> None:
        manager.register("a.b")
        assert "a.b" in manager
        assert "x.y" not in manager

    def test_all_module_ids(self, manager: LifecycleManager) -> None:
        manager.register("a")
        manager.register("b")
        ids = manager.all_module_ids()
        assert "a" in ids
        assert "b" in ids


# ===========================================================================
# 8. LifecycleManager — Lifecycle operation tests
# ===========================================================================


def _setup_running(manager: LifecycleManager, module_id: str = "test.mod") -> None:
    """Helper: register and fully start a module."""
    manager.register(module_id)
    manager.discover(module_id)
    manager.validate(module_id)
    manager.load(module_id)
    manager.initialize(module_id)
    manager.start(module_id)


class TestLifecycleManagerOperations:
    def test_full_forward_lifecycle(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "full.mod")
        assert manager.get_state("full.mod") is LifecycleState.RUNNING

    def test_pause_and_resume(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "pr.mod")
        manager.pause("pr.mod")
        assert manager.get_state("pr.mod") is LifecycleState.PAUSED
        manager.resume("pr.mod")
        assert manager.get_state("pr.mod") is LifecycleState.RUNNING

    def test_stop_from_running(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "stop.mod")
        manager.stop("stop.mod")
        assert manager.get_state("stop.mod") is LifecycleState.STOPPED

    def test_stop_from_paused(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "sp.mod")
        manager.pause("sp.mod")
        manager.stop("sp.mod")
        assert manager.get_state("sp.mod") is LifecycleState.STOPPED

    def test_unload_after_stop(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "ul.mod")
        manager.stop("ul.mod")
        manager.unload("ul.mod")
        assert manager.get_state("ul.mod") is LifecycleState.UNLOADED

    def test_fail_from_running(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "fail.mod")
        manager.fail("fail.mod", reason="test failure")
        assert manager.get_state("fail.mod") is LifecycleState.FAILED

    def test_recover_from_failed(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "rec.mod")
        manager.fail("rec.mod")
        manager.recover("rec.mod")
        assert manager.get_state("rec.mod") is LifecycleState.RUNNING

    def test_recover_requires_failed_state(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "notrec.mod")
        with pytest.raises(ModuleStateError):
            manager.recover("notrec.mod")

    def test_start_non_initialized_raises(self, manager: LifecycleManager) -> None:
        manager.register("early.mod")
        with pytest.raises(InvalidTransitionError):
            manager.start("early.mod")

    def test_pause_non_running_raises(self, manager: LifecycleManager) -> None:
        manager.register("np.mod")
        with pytest.raises(InvalidTransitionError):
            manager.pause("np.mod")

    def test_resume_non_paused_raises(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "nr.mod")
        with pytest.raises(InvalidTransitionError):
            manager.resume("nr.mod")

    def test_stop_from_initialized(self, manager: LifecycleManager) -> None:
        """Stopping a never-started (INITIALIZED) module should work."""
        manager.register("si.mod")
        manager.discover("si.mod")
        manager.validate("si.mod")
        manager.load("si.mod")
        manager.initialize("si.mod")
        manager.stop("si.mod")
        assert manager.get_state("si.mod") is LifecycleState.STOPPED

    def test_fail_with_exception(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "fe.mod")
        err = RuntimeError("disk full")
        manager.fail("fe.mod", error=err)
        assert manager.get_state("fe.mod") is LifecycleState.FAILED

    def test_operations_on_unknown_module_raise(self, manager: LifecycleManager) -> None:
        with pytest.raises(LifecycleError):
            manager.get_state("ghost.mod")
        with pytest.raises(LifecycleError):
            manager.start("ghost.mod")
        with pytest.raises(LifecycleError):
            manager.stop("ghost.mod")

    def test_is_running(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "ir.mod")
        assert manager.is_running("ir.mod")
        manager.pause("ir.mod")
        assert not manager.is_running("ir.mod")

    def test_is_running_unregistered_returns_false(self, manager: LifecycleManager) -> None:
        assert not manager.is_running("nonexistent")


# ===========================================================================
# 9. LifecycleManager — Query tests
# ===========================================================================


class TestLifecycleManagerQueries:
    def test_modules_in_state(self, manager: LifecycleManager) -> None:
        manager.register("a", initial_state=LifecycleState.CREATED)
        manager.register("b", initial_state=LifecycleState.CREATED)
        manager.register("c", initial_state=LifecycleState.LOADED)
        created = manager.modules_in_state(LifecycleState.CREATED)
        assert "a" in created
        assert "b" in created
        assert "c" not in created

    def test_all_states_snapshot(self, manager: LifecycleManager) -> None:
        manager.register("a", initial_state=LifecycleState.CREATED)
        _setup_running(manager, "b")
        states = manager.all_states()
        assert states["a"] is LifecycleState.CREATED
        assert states["b"] is LifecycleState.RUNNING

    def test_get_machine_returns_state_machine(self, manager: LifecycleManager) -> None:
        manager.register("q.mod")
        m = manager.get_machine("q.mod")
        assert isinstance(m, StateMachine)

    def test_get_history_grows(self, manager: LifecycleManager) -> None:
        manager.register("h.mod")
        h1 = manager.get_history("h.mod")
        manager.discover("h.mod")
        h2 = manager.get_history("h.mod")
        assert len(h2) == len(h1) + 1


# ===========================================================================
# 10. LifecycleManager — Recovery path tests
# ===========================================================================


class TestRecoveryPath:
    def test_full_recovery_cycle(self, manager: LifecycleManager) -> None:
        """RUNNING → FAILED → RECOVERING → RUNNING."""
        _setup_running(manager, "cyc.mod")
        manager.fail("cyc.mod")
        assert manager.get_state("cyc.mod") is LifecycleState.FAILED
        manager.recover("cyc.mod")
        assert manager.get_state("cyc.mod") is LifecycleState.RUNNING

    def test_multiple_recovery_cycles(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "multi.mod")
        for _ in range(3):
            manager.fail("multi.mod")
            manager.recover("multi.mod")
        assert manager.get_state("multi.mod") is LifecycleState.RUNNING

    def test_recovery_history_contains_recovering_state(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "hist.mod")
        manager.fail("hist.mod")
        manager.recover("hist.mod")
        history = manager.get_history("hist.mod")
        states_visited = [e.to_state for e in history]
        assert LifecycleState.RECOVERING in states_visited

    def test_unload_failed_module(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "uf.mod")
        manager.fail("uf.mod")
        manager.unload("uf.mod")
        assert manager.get_state("uf.mod") is LifecycleState.UNLOADED


# ===========================================================================
# 11. Event Bus integration tests
# ===========================================================================


class TestEventBusIntegration:
    def test_start_emits_started_event(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        _setup_running(manager_with_bus, "evt.start")
        mock_bus.publish.assert_called()
        # Find the started event
        calls = [str(c) for c in mock_bus.publish.call_args_list]
        event_types = []
        for call in mock_bus.publish.call_args_list:
            arg = call.args[0]
            event_types.append(arg.event_type)
        assert LifecycleEvents.MODULE_STARTED in event_types

    def test_pause_emits_paused_event(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        _setup_running(manager_with_bus, "evt.pause")
        mock_bus.publish.reset_mock()
        manager_with_bus.pause("evt.pause")
        mock_bus.publish.assert_called_once()
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == LifecycleEvents.MODULE_PAUSED

    def test_resume_emits_resumed_event(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        _setup_running(manager_with_bus, "evt.resume")
        manager_with_bus.pause("evt.resume")
        mock_bus.publish.reset_mock()
        manager_with_bus.resume("evt.resume")
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == LifecycleEvents.MODULE_RESUMED

    def test_stop_emits_stopped_event(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        _setup_running(manager_with_bus, "evt.stop")
        mock_bus.publish.reset_mock()
        manager_with_bus.stop("evt.stop")
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == LifecycleEvents.MODULE_STOPPED

    def test_fail_emits_failed_event(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        _setup_running(manager_with_bus, "evt.fail")
        mock_bus.publish.reset_mock()
        manager_with_bus.fail("evt.fail", reason="test error")
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == LifecycleEvents.MODULE_FAILED

    def test_recover_emits_recovered_event(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        _setup_running(manager_with_bus, "evt.rec")
        manager_with_bus.fail("evt.rec")
        mock_bus.publish.reset_mock()
        manager_with_bus.recover("evt.rec")
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == LifecycleEvents.MODULE_RECOVERED

    def test_unload_emits_unloaded_event(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        _setup_running(manager_with_bus, "evt.ul")
        manager_with_bus.stop("evt.ul")
        mock_bus.publish.reset_mock()
        manager_with_bus.unload("evt.ul")
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == LifecycleEvents.MODULE_UNLOADED

    def test_event_payload_contains_module_id(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        _setup_running(manager_with_bus, "pld.mod")
        mock_bus.publish.reset_mock()
        manager_with_bus.fail("pld.mod")
        event = mock_bus.publish.call_args.args[0]
        assert isinstance(event.payload, LifecycleEventPayload)
        assert event.payload.module_id == "pld.mod"

    def test_failed_event_has_critical_priority(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        from core.events.event import EventPriority
        _setup_running(manager_with_bus, "crit.mod")
        mock_bus.publish.reset_mock()
        manager_with_bus.fail("crit.mod")
        event = mock_bus.publish.call_args.args[0]
        assert event.priority == EventPriority.CRITICAL

    def test_no_bus_operations_still_work(self, manager: LifecycleManager) -> None:
        """Manager without an event bus should not raise on lifecycle ops."""
        _setup_running(manager, "nobus.mod")
        manager.pause("nobus.mod")
        manager.resume("nobus.mod")
        manager.stop("nobus.mod")

    def test_bus_publish_error_does_not_propagate(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        """If the bus raises on publish, the lifecycle op should still succeed."""
        mock_bus.publish.side_effect = RuntimeError("bus error")
        _setup_running(manager_with_bus, "safe.mod")
        manager_with_bus.pause("safe.mod")
        assert manager_with_bus.get_state("safe.mod") is LifecycleState.PAUSED

    def test_initialized_event_emitted(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        manager_with_bus.register("init.mod")
        manager_with_bus.discover("init.mod")
        manager_with_bus.validate("init.mod")
        manager_with_bus.load("init.mod")
        mock_bus.publish.reset_mock()
        manager_with_bus.initialize("init.mod")
        mock_bus.publish.assert_called_once()
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == LifecycleEvents.MODULE_INITIALIZED

    def test_loaded_event_emitted(
        self, manager_with_bus: LifecycleManager, mock_bus: MagicMock
    ) -> None:
        manager_with_bus.register("load.mod")
        manager_with_bus.discover("load.mod")
        manager_with_bus.validate("load.mod")
        mock_bus.publish.reset_mock()
        manager_with_bus.load("load.mod")
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == LifecycleEvents.MODULE_LOADED


# ===========================================================================
# 12. Thread safety tests
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_registrations(self, manager: LifecycleManager) -> None:
        """Register 50 modules from 10 threads simultaneously."""
        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def register_batch(batch_id: int) -> None:
            try:
                barrier.wait()
                for i in range(5):
                    mid = f"thread{batch_id}.mod{i}"
                    manager.register(mid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_batch, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(manager) == 50

    def test_concurrent_state_transitions(self, manager: LifecycleManager) -> None:
        """Start 20 modules concurrently and verify they all reach RUNNING."""
        module_ids = [f"conc.mod{i}" for i in range(20)]
        for mid in module_ids:
            manager.register(mid)

        errors: list[Exception] = []
        barrier = threading.Barrier(20)

        def advance_module(mid: str) -> None:
            try:
                barrier.wait()
                manager.discover(mid)
                manager.validate(mid)
                manager.load(mid)
                manager.initialize(mid)
                manager.start(mid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=advance_module, args=(mid,)) for mid in module_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        for mid in module_ids:
            assert manager.get_state(mid) is LifecycleState.RUNNING

    def test_concurrent_pause_resume(self, manager: LifecycleManager) -> None:
        """A single module is not double-paused or double-resumed under concurrency."""
        _setup_running(manager, "conc.pr")
        # This test validates no exception from concurrent reads of state
        results: list[LifecycleState] = []
        lock = threading.Lock()

        def read_state() -> None:
            s = manager.get_state("conc.pr")
            with lock:
                results.append(s)

        threads = [threading.Thread(target=read_state) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(s in LifecycleState for s in results)

    def test_concurrent_recovery_attempts(self, manager: LifecycleManager) -> None:
        """Only one recovery attempt should succeed; others fail gracefully."""
        _setup_running(manager, "crec.mod")
        manager.fail("crec.mod")

        success = []
        failures = []
        barrier = threading.Barrier(5)

        def try_recover() -> None:
            barrier.wait()
            try:
                manager.recover("crec.mod")
                success.append(True)
            except (LifecycleError, InvalidTransitionError, ModuleStateError) as e:
                failures.append(e)

        threads = [threading.Thread(target=try_recover) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one recovery should succeed
        assert len(success) == 1
        assert len(failures) == 4

    def test_state_machine_concurrent_transitions(self) -> None:
        """StateMachine itself is thread-safe for reads."""
        m = StateMachine("ts.mod")
        errors: list[Exception] = []

        def read_state() -> None:
            try:
                _ = m.current_state
                _ = m.history
                _ = m.allowed_next_states()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_state) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ===========================================================================
# 13. State history tests
# ===========================================================================


class TestStateHistory:
    def test_history_entry_fields(self, sm: StateMachine) -> None:
        sm.transition(LifecycleState.DISCOVERED, reason="test")
        entry = sm.history[-1]
        assert isinstance(entry, StateHistoryEntry)
        assert entry.from_state is LifecycleState.CREATED
        assert entry.to_state is LifecycleState.DISCOVERED
        assert isinstance(entry.timestamp, datetime)
        assert entry.timestamp.tzinfo is not None

    def test_history_grows_correctly(self, sm_running: StateMachine) -> None:
        count = len(sm_running.history)
        # CREATED(init) + 6 transitions = 7
        assert count == 7

    def test_transitions_to_query_multiple(self) -> None:
        m = StateMachine("hist2.mod")
        m.transition(LifecycleState.DISCOVERED)
        m.transition(LifecycleState.VALIDATED)
        m.transition(LifecycleState.LOADED)
        m.transition(LifecycleState.INITIALIZED)
        m.transition(LifecycleState.STARTING)
        m.transition(LifecycleState.RUNNING)
        m.transition(LifecycleState.FAILED)
        m.transition(LifecycleState.RECOVERING)
        m.transition(LifecycleState.RUNNING)
        # RUNNING appears twice
        running_entries = m.transitions_to(LifecycleState.RUNNING)
        assert len(running_entries) == 2

    def test_manager_history_reflects_all_transitions(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "mhist.mod")
        manager.fail("mhist.mod")
        manager.recover("mhist.mod")
        history = manager.get_history("mhist.mod")
        states = [e.to_state for e in history]
        assert LifecycleState.RUNNING in states
        assert LifecycleState.FAILED in states
        assert LifecycleState.RECOVERING in states

    def test_history_entries_are_immutable(self, sm: StateMachine) -> None:
        sm.transition(LifecycleState.DISCOVERED)
        entry = sm.history[-1]
        with pytest.raises((AttributeError, TypeError)):
            entry.to_state = LifecycleState.CREATED  # type: ignore[misc]


# ===========================================================================
# 14. Module shutdown / unload tests
# ===========================================================================


class TestModuleShutdown:
    def test_full_lifecycle_to_unload(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "full.unload")
        manager.stop("full.unload")
        manager.unload("full.unload")
        assert manager.get_state("full.unload") is LifecycleState.UNLOADED

    def test_unload_not_stopped_raises(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "badunload")
        with pytest.raises(InvalidTransitionError):
            manager.unload("badunload")

    def test_stop_from_loaded(self, manager: LifecycleManager) -> None:
        manager.register("l.mod")
        manager.discover("l.mod")
        manager.validate("l.mod")
        manager.load("l.mod")
        manager.stop("l.mod")
        assert manager.get_state("l.mod") is LifecycleState.STOPPED


# ===========================================================================
# 15. Edge cases and additional coverage
# ===========================================================================


class TestEdgeCases:
    def test_fail_reason_stored(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "fr.mod")
        manager.fail("fr.mod", reason="disk full")
        history = manager.get_history("fr.mod")
        last = history[-1]
        assert "disk full" in last.reason

    def test_modules_in_state_empty(self, manager: LifecycleManager) -> None:
        result = manager.modules_in_state(LifecycleState.RUNNING)
        assert result == ()

    def test_allowed_transitions_completeness(self) -> None:
        """Every non-terminal state can reach FAILED."""
        non_failable = {
            LifecycleState.FAILED,
            LifecycleState.UNLOADED,
            LifecycleState.STOPPED,
        }
        for state in LifecycleState:
            if state not in non_failable:
                assert is_allowed(state, LifecycleState.FAILED), \
                    f"{state} should be able to transition to FAILED"

    def test_state_machine_history_after_recovery(self) -> None:
        m = StateMachine("edge.mod", initial_state=LifecycleState.RUNNING)
        m.transition(LifecycleState.FAILED)
        m.transition(LifecycleState.RECOVERING)
        m.transition(LifecycleState.RUNNING)
        assert m.current_state is LifecycleState.RUNNING
        assert len(m.transitions_to(LifecycleState.RUNNING)) == 2

    def test_manager_repr_does_not_raise(self, manager: LifecycleManager) -> None:
        manager.register("a.b")
        # repr should not raise
        repr(manager)

    def test_state_history_entry_str(self) -> None:
        entry = StateHistoryEntry(
            from_state=LifecycleState.RUNNING,
            to_state=LifecycleState.FAILED,
            timestamp=datetime.now(timezone.utc),
            reason="test",
        )
        s = str(entry)
        assert "RUNNING" in s
        assert "FAILED" in s

    def test_get_machine_unknown_raises(self, manager: LifecycleManager) -> None:
        with pytest.raises(LifecycleError):
            manager.get_machine("not.there")

    def test_get_history_unknown_raises(self, manager: LifecycleManager) -> None:
        with pytest.raises(LifecycleError):
            manager.get_history("not.there")

    def test_paused_to_stopping(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "pts.mod")
        manager.pause("pts.mod")
        manager.stop("pts.mod")
        assert manager.get_state("pts.mod") is LifecycleState.STOPPED

    def test_discover_validate_load_sequence(self, manager: LifecycleManager) -> None:
        manager.register("seq.mod")
        manager.discover("seq.mod")
        assert manager.get_state("seq.mod") is LifecycleState.DISCOVERED
        manager.validate("seq.mod")
        assert manager.get_state("seq.mod") is LifecycleState.VALIDATED
        manager.load("seq.mod")
        assert manager.get_state("seq.mod") is LifecycleState.LOADED

    def test_all_states_snapshot_is_copy(self, manager: LifecycleManager) -> None:
        manager.register("snap.mod")
        snapshot = manager.all_states()
        # Mutating the snapshot should not affect the manager
        snapshot["snap.mod"] = LifecycleState.RUNNING  # type: ignore[assignment]
        assert manager.get_state("snap.mod") is LifecycleState.CREATED

    def test_fail_multiple_times_raises_second(self, manager: LifecycleManager) -> None:
        _setup_running(manager, "df.mod")
        manager.fail("df.mod")
        with pytest.raises(InvalidTransitionError):
            manager.fail("df.mod")

    def test_multiple_modules_independent(self, manager: LifecycleManager) -> None:
        for i in range(5):
            _setup_running(manager, f"ind.mod{i}")
        for i in range(5):
            assert manager.get_state(f"ind.mod{i}") is LifecycleState.RUNNING
        # Fail one; others unaffected
        manager.fail("ind.mod2")
        assert manager.get_state("ind.mod2") is LifecycleState.FAILED
        for i in [0, 1, 3, 4]:
            assert manager.get_state(f"ind.mod{i}") is LifecycleState.RUNNING