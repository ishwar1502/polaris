# core/contracts/lifecycle.py
"""
Lifecycle state machine for POLARIS v5 subsystems.

Defines the canonical set of lifecycle states, the valid transition graph,
and validation helpers used by :class:`~core.contracts.subsystem.SubsystemContract`
to enforce runtime invariants.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, unique
from typing import TYPE_CHECKING, Final, FrozenSet

if TYPE_CHECKING:
    pass  # Forward references only


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LifecycleError(Exception):
    """Raised when a subsystem attempts an illegal lifecycle transition.

    Attributes
    ----------
    current_state:
        The state the subsystem was in when the violation occurred.
    attempted_transition:
        The method / transition that was attempted.
    """

    def __init__(
        self,
        message: str,
        *,
        current_state: "LifecycleState",
        attempted_transition: str,
    ) -> None:
        super().__init__(message)
        self.current_state = current_state
        self.attempted_transition = attempted_transition

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LifecycleError(current_state={self.current_state!r}, "
            f"attempted_transition={self.attempted_transition!r}, "
            f"message={str(self)!r})"
        )


# ---------------------------------------------------------------------------
# State enumeration
# ---------------------------------------------------------------------------


@unique
class LifecycleState(Enum):
    """All possible states a POLARIS subsystem may occupy.

    The valid transition graph is enforced by :data:`VALID_TRANSITIONS`.
    """

    CREATED = auto()
    """Subsystem object has been instantiated; not yet initialised."""

    INITIALIZED = auto()
    """``initialize()`` completed successfully; ready to start."""

    STARTING = auto()
    """``start()`` is executing; subsystem is not yet fully operational."""

    RUNNING = auto()
    """Subsystem is fully operational and serving requests."""

    PAUSED = auto()
    """Subsystem is temporarily suspended; recoverable without re-init."""

    RECOVERING = auto()
    """Subsystem is attempting self-recovery after a transient failure."""

    STOPPING = auto()
    """``stop()`` is executing; subsystem is draining resources."""

    STOPPED = auto()
    """Subsystem has cleanly shut down; terminal state (restartable via
    re-initialisation at the registry level)."""

    FAILED = auto()
    """Subsystem entered an unrecoverable error state."""


# ---------------------------------------------------------------------------
# Transition graph
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: Final[dict[LifecycleState, FrozenSet[LifecycleState]]] = {
    LifecycleState.CREATED: frozenset({
        LifecycleState.INITIALIZED,
        LifecycleState.FAILED,
    }),
    LifecycleState.INITIALIZED: frozenset({
        LifecycleState.STARTING,
        LifecycleState.FAILED,
    }),
    LifecycleState.STARTING: frozenset({
        LifecycleState.RUNNING,
        LifecycleState.FAILED,
    }),
    LifecycleState.RUNNING: frozenset({
        LifecycleState.PAUSED,
        LifecycleState.RECOVERING,
        LifecycleState.STOPPING,
        LifecycleState.FAILED,
    }),
    LifecycleState.PAUSED: frozenset({
        LifecycleState.RUNNING,   # resume
        LifecycleState.STOPPING,
        LifecycleState.FAILED,
    }),
    LifecycleState.RECOVERING: frozenset({
        LifecycleState.RUNNING,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    }),
    LifecycleState.STOPPING: frozenset({
        LifecycleState.STOPPED,
        LifecycleState.FAILED,
    }),
    LifecycleState.STOPPED: frozenset({
        # Terminal; a stopped subsystem may only be re-registered fresh.
    }),
    LifecycleState.FAILED: frozenset({
        LifecycleState.RECOVERING,
        LifecycleState.STOPPING,
    }),
}
"""Adjacency map of permitted state transitions.

Keys are source states; values are the set of reachable target states.
Any transition not present in the value set is illegal.
"""

# Derive terminal states automatically from the graph
TERMINAL_STATES: Final[FrozenSet[LifecycleState]] = frozenset(
    state for state, targets in VALID_TRANSITIONS.items() if not targets
)


# ---------------------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LifecycleTransition:
    """Immutable record of a single state transition event.

    Attributes
    ----------
    from_state:
        The state the subsystem was in before the transition.
    to_state:
        The state entered after the transition.
    timestamp:
        UTC timestamp of the transition.
    reason:
        Optional human-readable justification for the transition.
    """

    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: datetime
    reason: str = ""

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("LifecycleTransition.timestamp must be timezone-aware.")


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


@dataclass
class LifecycleMachine:
    """Thread-safe state machine governing a single subsystem's lifecycle.

    This class is owned exclusively by a
    :class:`~core.contracts.subsystem.SubsystemContract` instance and must
    not be shared across subsystems.

    Parameters
    ----------
    initial_state:
        The state to begin in; defaults to :attr:`LifecycleState.CREATED`.
    """

    _state: LifecycleState = field(
        default=LifecycleState.CREATED, init=False
    )
    _history: list[LifecycleTransition] = field(default_factory=list, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    @property
    def state(self) -> LifecycleState:
        """Current lifecycle state (thread-safe read)."""
        with self._lock:
            return self._state

    @property
    def history(self) -> list[LifecycleTransition]:
        """Ordered list of all transitions that have occurred (copy)."""
        with self._lock:
            return list(self._history)

    @property
    def is_terminal(self) -> bool:
        """``True`` if the current state has no valid outgoing transitions."""
        return self.state in TERMINAL_STATES

    def transition(
        self,
        target: LifecycleState,
        *,
        reason: str = "",
    ) -> LifecycleTransition:
        """Attempt a transition to *target*.

        Parameters
        ----------
        target:
            The desired next state.
        reason:
            Optional human-readable description of why the transition
            is occurring (stored in the history).

        Returns
        -------
        LifecycleTransition
            The recorded transition event.

        Raises
        ------
        LifecycleError
            If the transition from the current state to *target* is not
            permitted by :data:`VALID_TRANSITIONS`.
        """
        with self._lock:
            current = self._state
            allowed = VALID_TRANSITIONS.get(current, frozenset())
            if target not in allowed:
                raise LifecycleError(
                    f"Illegal transition {current.name} → {target.name}. "
                    f"Permitted targets from {current.name}: "
                    f"{sorted(s.name for s in allowed) or '(none — terminal state)'}.",
                    current_state=current,
                    attempted_transition=target.name,
                )
            record = LifecycleTransition(
                from_state=current,
                to_state=target,
                timestamp=datetime.now(timezone.utc),
                reason=reason,
            )
            self._state = target
            self._history.append(record)
            return record

    def assert_in(
        self,
        *expected: LifecycleState,
        operation: str = "operation",
    ) -> None:
        """Assert that the current state is one of *expected*.

        Parameters
        ----------
        *expected:
            One or more states that are acceptable for the operation.
        operation:
            Name of the operation requiring this precondition (used in
            the error message).

        Raises
        ------
        LifecycleError
            If the current state is not among *expected*.
        """
        current = self.state
        if current not in expected:
            raise LifecycleError(
                f"Cannot perform '{operation}' in state {current.name}. "
                f"Required: {sorted(s.name for s in expected)}.",
                current_state=current,
                attempted_transition=operation,
            )

    def can_transition_to(self, target: LifecycleState) -> bool:
        """Return whether a transition to *target* is currently valid.

        Parameters
        ----------
        target:
            Candidate next state.

        Returns
        -------
        bool
            ``True`` if the transition is permitted; ``False`` otherwise.
        """
        with self._lock:
            return target in VALID_TRANSITIONS.get(self._state, frozenset())

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LifecycleMachine(state={self._state.name}, "
            f"transitions={len(self._history)})"
        )