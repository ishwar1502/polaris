# core/lifecycle/models.py
"""
POLARIS v5 Lifecycle Manager — State definitions.

:class:`LifecycleState` is the complete 13-state enumeration that describes
every phase a managed module can occupy, from construction through unloading.

State ordering (forward path)
-----------------------------
::

    CREATED → DISCOVERED → VALIDATED → LOADED → INITIALIZED
            → STARTING → RUNNING ←→ PAUSED
            → RECOVERING (from FAILED)
            → STOPPING → STOPPED → UNLOADED
    Any operational/transitional state → FAILED

Failure recovery path
---------------------
::

    RUNNING → FAILED → RECOVERING → RUNNING
"""

from __future__ import annotations

from enum import Enum, auto, unique
from typing import Final


@unique
class LifecycleState(Enum):
    """All states a POLARIS v5 lifecycle-managed module may occupy.

    The :class:`~core.lifecycle.state_machine.StateMachine` enforces strict
    ordering; no transition that is not listed in
    :mod:`core.lifecycle.transitions` will be accepted.
    """

    # ------------------------------------------------------------------
    # Pre-operational states
    # ------------------------------------------------------------------

    CREATED = auto()
    """Module object has been constructed but no lifecycle processing has
    begun.  Entry point for newly instantiated module descriptors."""

    DISCOVERED = auto()
    """Module has been found on disk / in the registry and its manifest has
    been read; structural validation has not yet been attempted."""

    VALIDATED = auto()
    """Manifest has passed all structural and semantic checks; dependency
    resolution has succeeded against the discovery set."""

    LOADED = auto()
    """Module class has been successfully imported into the Python runtime;
    the class object is held in memory."""

    INITIALIZED = auto()
    """Module instance has been constructed and :meth:`initialize` completed
    successfully; the module is ready to start but not yet running."""

    # ------------------------------------------------------------------
    # Transitional startup state
    # ------------------------------------------------------------------

    STARTING = auto()
    """Module's :meth:`start` hook is currently executing.  This is a
    transient state — the module will move to :attr:`RUNNING` on success
    or :attr:`FAILED` on error."""

    # ------------------------------------------------------------------
    # Operational states
    # ------------------------------------------------------------------

    RUNNING = auto()
    """Module is fully operational and serving requests."""

    PAUSED = auto()
    """Module has been temporarily suspended.  It retains all internal state
    and can resume without re-initialization."""

    # ------------------------------------------------------------------
    # Recovery state
    # ------------------------------------------------------------------

    RECOVERING = auto()
    """Module is actively recovering from a :attr:`FAILED` state.  A
    successful recovery transitions back to :attr:`RUNNING`."""

    # ------------------------------------------------------------------
    # Shutdown states
    # ------------------------------------------------------------------

    STOPPING = auto()
    """Module's :meth:`stop` hook is currently executing.  Transitions to
    :attr:`STOPPED` on success or :attr:`FAILED` on error."""

    STOPPED = auto()
    """Module has been gracefully shut down.  A stopped module may be
    unloaded but cannot be restarted without reloading."""

    # ------------------------------------------------------------------
    # Error state
    # ------------------------------------------------------------------

    FAILED = auto()
    """Module entered an unrecoverable error state during any lifecycle
    phase.  Recovery to :attr:`RECOVERING` → :attr:`RUNNING` is possible."""

    # ------------------------------------------------------------------
    # Terminal state
    # ------------------------------------------------------------------

    UNLOADED = auto()
    """Module has been fully removed from the runtime.  All references are
    released.  This is a permanent terminal state for a given instance."""

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    def is_operational(self) -> bool:
        """Return ``True`` if the module is in an active operational state.

        Operational states are :attr:`RUNNING` and :attr:`PAUSED`.
        """
        return self in _OPERATIONAL_STATES

    def is_terminal(self) -> bool:
        """Return ``True`` if the state has no valid forward transition
        under normal (non-error) circumstances.

        Terminal states are :attr:`UNLOADED` and :attr:`FAILED`.
        """
        return self in _TERMINAL_STATES

    def is_transitional(self) -> bool:
        """Return ``True`` if the state is a transient transition state
        (:attr:`STARTING`, :attr:`STOPPING`, :attr:`RECOVERING`).
        """
        return self in _TRANSITIONAL_STATES

    def can_fail(self) -> bool:
        """Return ``True`` if a module in this state may transition to
        :attr:`FAILED`.
        """
        return self not in _NON_FAILABLE_STATES

    def can_recover(self) -> bool:
        """Return ``True`` if a module in this state may transition to
        :attr:`RECOVERING`.
        """
        return self is LifecycleState.FAILED


# ---------------------------------------------------------------------------
# State sets
# ---------------------------------------------------------------------------

_OPERATIONAL_STATES: Final[frozenset[LifecycleState]] = frozenset({
    LifecycleState.RUNNING,
    LifecycleState.PAUSED,
})

_TERMINAL_STATES: Final[frozenset[LifecycleState]] = frozenset({
    LifecycleState.UNLOADED,
    LifecycleState.FAILED,
})

_TRANSITIONAL_STATES: Final[frozenset[LifecycleState]] = frozenset({
    LifecycleState.STARTING,
    LifecycleState.STOPPING,
    LifecycleState.RECOVERING,
})

_NON_FAILABLE_STATES: Final[frozenset[LifecycleState]] = frozenset({
    LifecycleState.FAILED,
    LifecycleState.UNLOADED,
    LifecycleState.STOPPED,
})

# Public aliases
OPERATIONAL_STATES: Final[frozenset[LifecycleState]] = _OPERATIONAL_STATES
TERMINAL_STATES: Final[frozenset[LifecycleState]] = _TERMINAL_STATES
TRANSITIONAL_STATES: Final[frozenset[LifecycleState]] = _TRANSITIONAL_STATES


__all__ = [
    "LifecycleState",
    "OPERATIONAL_STATES",
    "TERMINAL_STATES",
    "TRANSITIONAL_STATES",
]