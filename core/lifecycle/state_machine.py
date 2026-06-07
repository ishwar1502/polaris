# core/lifecycle/state_machine.py
"""
POLARIS v5 Lifecycle Manager — Per-module State Machine.

:class:`StateMachine` tracks the lifecycle state of a single module,
enforces the allowed transition table, and maintains a full history of
every state change that has occurred.

Thread safety
-------------
All public methods acquire the internal ``threading.RLock`` before reading
or mutating shared state, so concurrent calls from multiple threads are safe.
Each :class:`StateMachine` instance carries its own lock, independent of any
external lock held by the :class:`~core.lifecycle.manager.LifecycleManager`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final, Sequence

from core.lifecycle.exceptions import InvalidTransitionError, StateMachineError
from core.lifecycle.models import LifecycleState
from core.lifecycle.transitions import (
    LifecycleTransition,
    get_transition,
    is_allowed,
    allowed_from,
)


# ---------------------------------------------------------------------------
# History entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateHistoryEntry:
    """One record in a :class:`StateMachine`'s transition history.

    Attributes
    ----------
    from_state:
        The state *before* the transition.  ``None`` for the initial entry.
    to_state:
        The state *after* the transition.
    timestamp:
        UTC time at which the transition was recorded.
    reason:
        Optional human-readable description of why the transition occurred.
    """

    from_state: LifecycleState | None
    to_state: LifecycleState
    timestamp: datetime
    reason: str = ""

    def __str__(self) -> str:  # pragma: no cover
        src = self.from_state.name if self.from_state else "<init>"
        return (
            f"{self.timestamp.isoformat()} "
            f"{src} → {self.to_state.name}"
            + (f" ({self.reason})" if self.reason else "")
        )


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

_MAX_HISTORY: Final[int] = 1024


class StateMachine:
    """Thread-safe finite state machine for a single module's lifecycle.

    Parameters
    ----------
    module_id:
        Identifier of the module this machine is tracking.
    initial_state:
        Starting state.  Defaults to :attr:`~LifecycleState.CREATED`.
    max_history:
        Maximum number of history entries to retain.  Oldest entries are
        dropped once this limit is reached.  Defaults to 1024.

    Raises
    ------
    StateMachineError
        If *initial_state* is not a valid :class:`LifecycleState`.
    """

    def __init__(
        self,
        module_id: str,
        initial_state: LifecycleState = LifecycleState.CREATED,
        max_history: int = _MAX_HISTORY,
    ) -> None:
        if not isinstance(initial_state, LifecycleState):
            raise StateMachineError(
                f"initial_state must be a LifecycleState, got {type(initial_state)!r}.",
                module_id=module_id,
            )
        if max_history < 1:
            raise StateMachineError(
                f"max_history must be at least 1, got {max_history!r}.",
                module_id=module_id,
            )

        self._module_id = module_id
        self._current_state: LifecycleState = initial_state
        self._max_history = max_history
        self._lock = threading.RLock()

        # Seed the history with the initial state entry.
        self._history: list[StateHistoryEntry] = [
            StateHistoryEntry(
                from_state=None,
                to_state=initial_state,
                timestamp=datetime.now(timezone.utc),
                reason="initial state",
            )
        ]

    # ------------------------------------------------------------------
    # Public read methods (thread-safe)
    # ------------------------------------------------------------------

    @property
    def module_id(self) -> str:
        """The id of the module this machine is tracking."""
        return self._module_id

    @property
    def current_state(self) -> LifecycleState:
        """Current :class:`LifecycleState` (snapshot; may change immediately
        after reading in concurrent contexts).
        """
        with self._lock:
            return self._current_state

    @property
    def history(self) -> tuple[StateHistoryEntry, ...]:
        """Immutable snapshot of the full transition history, oldest first."""
        with self._lock:
            return tuple(self._history)

    def can_transition_to(self, to_state: LifecycleState) -> bool:
        """Return ``True`` if the current state may transition to *to_state*.

        Parameters
        ----------
        to_state:
            Desired next state.
        """
        with self._lock:
            return is_allowed(self._current_state, to_state)

    def allowed_next_states(self) -> frozenset[LifecycleState]:
        """Return the set of states reachable from the current state."""
        with self._lock:
            return allowed_from(self._current_state)

    def is_in_state(self, state: LifecycleState) -> bool:
        """Return ``True`` if the machine is currently in *state*."""
        with self._lock:
            return self._current_state is state

    def is_in_any_state(self, *states: LifecycleState) -> bool:
        """Return ``True`` if the machine is currently in any of *states*."""
        with self._lock:
            return self._current_state in states

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def transition(
        self,
        to_state: LifecycleState,
        *,
        reason: str = "",
    ) -> LifecycleTransition:
        """Execute a state transition.

        Validates the requested transition against the allowed table, updates
        the current state, and appends a history entry.

        Parameters
        ----------
        to_state:
            The state to transition to.
        reason:
            Optional human-readable explanation for this transition.

        Returns
        -------
        LifecycleTransition
            The :class:`LifecycleTransition` descriptor that was applied.

        Raises
        ------
        InvalidTransitionError
            If the transition ``current_state → to_state`` is not in the
            allowed table.
        StateMachineError
            If *to_state* is not a :class:`LifecycleState` instance.
        """
        if not isinstance(to_state, LifecycleState):
            raise StateMachineError(
                f"to_state must be a LifecycleState, got {type(to_state)!r}.",
                module_id=self._module_id,
            )

        with self._lock:
            transition = get_transition(self._current_state, to_state)
            if transition is None:
                raise InvalidTransitionError(
                    self._current_state,
                    to_state,
                    module_id=self._module_id,
                )

            prev_state = self._current_state
            self._current_state = to_state

            entry = StateHistoryEntry(
                from_state=prev_state,
                to_state=to_state,
                timestamp=datetime.now(timezone.utc),
                reason=reason or transition.description,
            )
            self._history.append(entry)

            # Trim history if needed
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            return transition

    # ------------------------------------------------------------------
    # History query helpers
    # ------------------------------------------------------------------

    def last_transition(self) -> StateHistoryEntry | None:
        """Return the most recent history entry, or ``None`` if only the
        initial entry exists (i.e. no transitions have been made yet).
        """
        with self._lock:
            if len(self._history) <= 1:
                return None
            return self._history[-1]

    def transitions_to(
        self, state: LifecycleState
    ) -> tuple[StateHistoryEntry, ...]:
        """Return all history entries where ``to_state == state``."""
        with self._lock:
            return tuple(e for e in self._history if e.to_state is state)

    def time_in_current_state(self) -> float:
        """Return the number of seconds the machine has been in its current
        state (as a float).
        """
        with self._lock:
            if not self._history:
                return 0.0
            last = self._history[-1]
            elapsed = datetime.now(timezone.utc) - last.timestamp
            return elapsed.total_seconds()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"StateMachine("
            f"module_id={self._module_id!r}, "
            f"state={self._current_state.name})"
        )

    def __str__(self) -> str:  # pragma: no cover
        return f"StateMachine[{self._module_id}:{self._current_state.name}]"


__all__ = [
    "StateMachine",
    "StateHistoryEntry",
]
