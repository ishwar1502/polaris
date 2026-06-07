# core/lifecycle/manager.py
"""
POLARIS v5 Lifecycle Manager — Central controller.

:class:`LifecycleManager` is the authoritative controller for all module
lifecycle state transitions.  Where the Module Loader *loads* modules (imports
their classes, instantiates them, etc.), the Lifecycle Manager *governs* them
by tracking their state, enforcing valid transitions, triggering recovery, and
emitting lifecycle events onto the Event Bus.

Design principles
-----------------
* **Single source of truth** — every module's current state is held exactly
  once, inside the manager's :class:`~core.lifecycle.state_machine.StateMachine`
  registry.
* **Thread-safe** — a single reentrant lock (``threading.RLock``) serialises
  all mutating operations.  Read-only queries also acquire the lock to avoid
  torn reads.
* **Event-driven** — every state change publishes a corresponding lifecycle
  event on the injected :class:`~core.events.bus.EventBus` (if one was
  provided).  Lifecycle operations remain functional even when no bus is
  available.
* **Non-invasive** — the manager does not own module instances; it only tracks
  their state.  Module objects are registered by id and may hold any type.

Usage
-----
.. code-block:: python

    from core.lifecycle.manager import LifecycleManager
    from core.lifecycle.models import LifecycleState

    manager = LifecycleManager()

    manager.register("my.module")
    manager.initialize("my.module")
    manager.start("my.module")
    assert manager.get_state("my.module") is LifecycleState.RUNNING

    manager.pause("my.module")
    manager.resume("my.module")
    manager.stop("my.module")
    manager.unload("my.module")
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from core.lifecycle.events import LifecycleEventPayload, LifecycleEvents
from core.lifecycle.exceptions import (
    InvalidTransitionError,
    LifecycleError,
    ModuleStateError,
    StateMachineError,
)
from core.lifecycle.models import LifecycleState
from core.lifecycle.state_machine import StateMachine, StateHistoryEntry
from core.lifecycle.transitions import LifecycleTransition

_logger = logging.getLogger(__name__)

_SOURCE: str = "polaris.lifecycle.manager"


# ---------------------------------------------------------------------------
# LifecycleManager
# ---------------------------------------------------------------------------


class LifecycleManager:
    """Thread-safe controller for all module lifecycle state transitions.

    Parameters
    ----------
    event_bus:
        Optional :class:`~core.events.bus.EventBus` instance.  When provided,
        all state transitions publish a corresponding lifecycle event.  If
        ``None``, lifecycle operations work normally but emit no events.
    max_history:
        Maximum number of history entries to retain per module's state
        machine.  Defaults to 1024.

    Example
    -------
    .. code-block:: python

        from core.lifecycle.manager import LifecycleManager

        manager = LifecycleManager()
        manager.register("my_module")
        manager.start("my_module")      # CREATED → DISCOVERED → ... → RUNNING
    """

    def __init__(
        self,
        event_bus: Any = None,
        *,
        max_history: int = 1024,
    ) -> None:
        self._event_bus = event_bus
        self._max_history = max_history
        self._lock = threading.RLock()
        # module_id → StateMachine
        self._machines: dict[str, StateMachine] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        module_id: str,
        *,
        initial_state: LifecycleState = LifecycleState.CREATED,
        reason: str = "",
    ) -> StateMachine:
        """Register a module with the lifecycle manager.

        Creates a :class:`~core.lifecycle.state_machine.StateMachine` for the
        module and makes it trackable by the manager.

        Parameters
        ----------
        module_id:
            Unique identifier for the module.
        initial_state:
            Starting state; defaults to :attr:`~LifecycleState.CREATED`.
        reason:
            Optional reason for the initial state.

        Returns
        -------
        StateMachine
            The newly created state machine for this module.

        Raises
        ------
        LifecycleError
            If *module_id* is already registered.
        StateMachineError
            If *initial_state* is not valid.
        """
        if not module_id or not module_id.strip():
            raise LifecycleError(
                "module_id must be a non-empty string.", module_id=module_id
            )

        with self._lock:
            if module_id in self._machines:
                raise LifecycleError(
                    f"Module '{module_id}' is already registered.",
                    module_id=module_id,
                )
            machine = StateMachine(
                module_id=module_id,
                initial_state=initial_state,
                max_history=self._max_history,
            )
            self._machines[module_id] = machine
            _logger.debug("Registered module '%s' in state %s.", module_id, initial_state.name)
            return machine

    def unregister(self, module_id: str) -> None:
        """Remove a module from the lifecycle manager.

        The module must be in the :attr:`~LifecycleState.UNLOADED` or
        :attr:`~LifecycleState.FAILED` state before it can be unregistered.

        Parameters
        ----------
        module_id:
            The module to remove.

        Raises
        ------
        LifecycleError
            If *module_id* is not registered, or is not in an unregisterable
            state.
        """
        with self._lock:
            machine = self._get_machine(module_id)
            state = machine.current_state
            if state not in (
                LifecycleState.UNLOADED,
                LifecycleState.FAILED,
                LifecycleState.STOPPED,
            ):
                raise LifecycleError(
                    f"Module '{module_id}' cannot be unregistered from state "
                    f"{state.name}; must be UNLOADED, FAILED, or STOPPED.",
                    module_id=module_id,
                )
            del self._machines[module_id]
            _logger.debug("Unregistered module '%s'.", module_id)

    def is_registered(self, module_id: str) -> bool:
        """Return ``True`` if *module_id* has been registered."""
        with self._lock:
            return module_id in self._machines

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def discover(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Transition module from CREATED to DISCOVERED.

        Parameters
        ----------
        module_id:
            Target module.
        reason:
            Optional reason string.

        Returns
        -------
        StateMachine
            The module's state machine after transition.
        """
        return self._transition(
            module_id,
            LifecycleState.DISCOVERED,
            event_type=None,  # no specific event for this step
            reason=reason or "Module discovered.",
        )

    def validate(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Transition module from DISCOVERED to VALIDATED."""
        return self._transition(
            module_id,
            LifecycleState.VALIDATED,
            event_type=None,
            reason=reason or "Module validated.",
        )

    def initialize(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Transition module to INITIALIZED.

        Expects the module to be in LOADED state.

        Returns
        -------
        StateMachine
            The module's state machine after the transition.

        Raises
        ------
        InvalidTransitionError
            If the module is not in an appropriate state.
        """
        machine = self._transition(
            module_id,
            LifecycleState.INITIALIZED,
            event_type=LifecycleEvents.MODULE_INITIALIZED,
            reason=reason or "Module initialized.",
        )
        return machine

    def load(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Transition module from VALIDATED to LOADED.

        Emits a :attr:`~LifecycleEvents.MODULE_LOADED` event.
        """
        return self._transition(
            module_id,
            LifecycleState.LOADED,
            event_type=LifecycleEvents.MODULE_LOADED,
            reason=reason or "Module class loaded.",
        )

    def start(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Transition module from INITIALIZED → STARTING → RUNNING.

        This helper performs two transitions atomically (under the lock) to
        move the module through the transient STARTING state into RUNNING.

        Emits a :attr:`~LifecycleEvents.MODULE_STARTED` event on arrival at
        RUNNING.

        Returns
        -------
        StateMachine
            The module's state machine after the RUNNING transition.
        """
        with self._lock:
            machine = self._get_machine(module_id)
            # INITIALIZED → STARTING
            self._do_transition(machine, LifecycleState.STARTING, reason="Starting module.")
            # STARTING → RUNNING
            self._do_transition(
                machine,
                LifecycleState.RUNNING,
                reason=reason or "Module started successfully.",
                event_type=LifecycleEvents.MODULE_STARTED,
            )
            return machine

    def pause(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Transition module from RUNNING to PAUSED.

        Emits a :attr:`~LifecycleEvents.MODULE_PAUSED` event.
        """
        return self._transition(
            module_id,
            LifecycleState.PAUSED,
            event_type=LifecycleEvents.MODULE_PAUSED,
            reason=reason or "Module paused.",
        )

    def resume(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Transition module from PAUSED to RUNNING.

        Emits a :attr:`~LifecycleEvents.MODULE_RESUMED` event.
        """
        return self._transition(
            module_id,
            LifecycleState.RUNNING,
            event_type=LifecycleEvents.MODULE_RESUMED,
            reason=reason or "Module resumed.",
        )

    def stop(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Transition module from RUNNING/PAUSED → STOPPING → STOPPED.

        This helper performs two transitions atomically (under the lock).

        Emits a :attr:`~LifecycleEvents.MODULE_STOPPED` event on arrival at
        STOPPED.

        Returns
        -------
        StateMachine
            The module's state machine after the STOPPED transition.
        """
        with self._lock:
            machine = self._get_machine(module_id)
            current = machine.current_state

            # Handle modules that are INITIALIZED or LOADED — allow direct
            # stop without going through STOPPING
            if current in (LifecycleState.INITIALIZED, LifecycleState.LOADED):
                self._do_transition(
                    machine,
                    LifecycleState.STOPPED,
                    reason=reason or "Module stopped (never started).",
                    event_type=LifecycleEvents.MODULE_STOPPED,
                )
                return machine

            # Normal path: RUNNING/PAUSED → STOPPING → STOPPED
            self._do_transition(machine, LifecycleState.STOPPING, reason="Stopping module.")
            self._do_transition(
                machine,
                LifecycleState.STOPPED,
                reason=reason or "Module stopped successfully.",
                event_type=LifecycleEvents.MODULE_STOPPED,
            )
            return machine

    def unload(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Transition module from STOPPED/FAILED to UNLOADED.

        Emits a :attr:`~LifecycleEvents.MODULE_UNLOADED` event.
        """
        return self._transition(
            module_id,
            LifecycleState.UNLOADED,
            event_type=LifecycleEvents.MODULE_UNLOADED,
            reason=reason or "Module unloaded.",
        )

    def fail(
        self,
        module_id: str,
        *,
        reason: str = "",
        error: Exception | None = None,
    ) -> StateMachine:
        """Transition module to FAILED state.

        This operation is valid from most non-terminal states.

        Emits a :attr:`~LifecycleEvents.MODULE_FAILED` event.

        Parameters
        ----------
        module_id:
            Target module.
        reason:
            Human-readable failure reason.
        error:
            Optional originating exception; its message is appended to the
            reason if provided.
        """
        extra: dict[str, Any] = {}
        full_reason = reason or "Module failed."
        if error is not None:
            extra["error"] = str(error)
            extra["error_type"] = type(error).__name__
            if not reason:
                full_reason = f"Module failed: {error}"

        return self._transition(
            module_id,
            LifecycleState.FAILED,
            event_type=LifecycleEvents.MODULE_FAILED,
            reason=full_reason,
            extra=extra,
        )

    def recover(self, module_id: str, *, reason: str = "") -> StateMachine:
        """Initiate recovery: FAILED → RECOVERING → RUNNING.

        This helper performs two transitions atomically.

        Emits a :attr:`~LifecycleEvents.MODULE_RECOVERED` event on arrival at
        RUNNING.

        Returns
        -------
        StateMachine
            The module's state machine after the RUNNING transition.

        Raises
        ------
        ModuleStateError
            If the module is not in the FAILED state.
        """
        with self._lock:
            machine = self._get_machine(module_id)
            if machine.current_state is not LifecycleState.FAILED:
                raise ModuleStateError(
                    module_id,
                    expected_states={LifecycleState.FAILED},
                    actual_state=machine.current_state,
                    operation="recover",
                )
            # FAILED → RECOVERING
            self._do_transition(
                machine,
                LifecycleState.RECOVERING,
                reason="Recovery initiated.",
            )
            # RECOVERING → RUNNING
            self._do_transition(
                machine,
                LifecycleState.RUNNING,
                reason=reason or "Module recovered successfully.",
                event_type=LifecycleEvents.MODULE_RECOVERED,
            )
            return machine

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_state(self, module_id: str) -> LifecycleState:
        """Return the current :class:`LifecycleState` of *module_id*.

        Raises
        ------
        LifecycleError
            If *module_id* is not registered.
        """
        with self._lock:
            return self._get_machine(module_id).current_state

    def get_machine(self, module_id: str) -> StateMachine:
        """Return the :class:`StateMachine` for *module_id*.

        Raises
        ------
        LifecycleError
            If *module_id* is not registered.
        """
        with self._lock:
            return self._get_machine(module_id)

    def get_history(self, module_id: str) -> tuple[StateHistoryEntry, ...]:
        """Return the transition history for *module_id*.

        Returns
        -------
        tuple[StateHistoryEntry, ...]
            Ordered history, oldest first.

        Raises
        ------
        LifecycleError
            If *module_id* is not registered.
        """
        with self._lock:
            return self._get_machine(module_id).history

    def modules_in_state(
        self, state: LifecycleState
    ) -> tuple[str, ...]:
        """Return the ids of all modules currently in *state*.

        Parameters
        ----------
        state:
            The state to filter by.
        """
        with self._lock:
            return tuple(
                mid for mid, m in self._machines.items()
                if m.current_state is state
            )

    def all_module_ids(self) -> tuple[str, ...]:
        """Return the ids of all registered modules."""
        with self._lock:
            return tuple(self._machines.keys())

    def all_states(self) -> dict[str, LifecycleState]:
        """Return a snapshot dict mapping module_id → current state."""
        with self._lock:
            return {mid: m.current_state for mid, m in self._machines.items()}

    def is_running(self, module_id: str) -> bool:
        """Return ``True`` if *module_id* is in the RUNNING state."""
        with self._lock:
            if module_id not in self._machines:
                return False
            return self._machines[module_id].current_state is LifecycleState.RUNNING

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_machine(self, module_id: str) -> StateMachine:
        """Retrieve the machine for *module_id* or raise :class:`LifecycleError`."""
        machine = self._machines.get(module_id)
        if machine is None:
            raise LifecycleError(
                f"Module '{module_id}' is not registered with the LifecycleManager.",
                module_id=module_id,
            )
        return machine

    def _do_transition(
        self,
        machine: StateMachine,
        to_state: LifecycleState,
        *,
        reason: str = "",
        event_type: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Execute one transition on *machine* and optionally publish an event.

        Called with ``self._lock`` already held.

        Parameters
        ----------
        machine:
            The state machine to transition.
        to_state:
            Target state.
        reason:
            Human-readable reason.
        event_type:
            If set, publish a lifecycle event of this type.
        extra:
            Extra payload data for the event.
        """
        from_state = machine.current_state
        transition = machine.transition(to_state, reason=reason)

        if event_type and self._event_bus is not None:
            self._publish_event(
                event_type=event_type,
                module_id=machine.module_id,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                extra=extra,
            )

    def _transition(
        self,
        module_id: str,
        to_state: LifecycleState,
        *,
        event_type: str | None,
        reason: str = "",
        extra: dict[str, Any] | None = None,
    ) -> StateMachine:
        """Thread-safe single-step transition helper."""
        with self._lock:
            machine = self._get_machine(module_id)
            self._do_transition(
                machine,
                to_state,
                reason=reason,
                event_type=event_type,
                extra=extra,
            )
            return machine

    def _publish_event(
        self,
        *,
        event_type: str,
        module_id: str,
        from_state: LifecycleState,
        to_state: LifecycleState,
        reason: str,
        extra: dict[str, Any] | None,
    ) -> None:
        """Publish a lifecycle event to the Event Bus.

        Errors during publishing are logged but do not propagate — the
        lifecycle operation has already completed.
        """
        try:
            from core.events.event import Event, EventPriority

            payload = LifecycleEventPayload(
                module_id=module_id,
                from_state=from_state.name,
                to_state=to_state.name,
                reason=reason,
                extra=extra,
            )

            # Use CRITICAL priority for FAILED events, HIGH for operational
            # changes, NORMAL for others.
            if to_state is LifecycleState.FAILED:
                priority = EventPriority.CRITICAL
            elif to_state in (LifecycleState.RUNNING, LifecycleState.STOPPED):
                priority = EventPriority.HIGH
            else:
                priority = EventPriority.NORMAL

            event = Event.create(
                event_type=event_type,
                source=_SOURCE,
                payload=payload,
                priority=priority,
            )
            self._event_bus.publish(event)

        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "Failed to publish lifecycle event '%s' for module '%s': %s",
                event_type,
                module_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of registered modules."""
        with self._lock:
            return len(self._machines)

    def __contains__(self, module_id: str) -> bool:
        """Support ``module_id in manager`` membership test."""
        with self._lock:
            return module_id in self._machines

    def __repr__(self) -> str:  # pragma: no cover
        with self._lock:
            return (
                f"LifecycleManager("
                f"modules={len(self._machines)}, "
                f"bus={'yes' if self._event_bus else 'no'})"
            )


__all__ = ["LifecycleManager"]