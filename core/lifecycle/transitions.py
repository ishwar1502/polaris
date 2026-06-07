# core/lifecycle/transitions.py
"""
POLARIS v5 Lifecycle Manager — Transition table.

:class:`LifecycleTransition` is an immutable descriptor for a single valid
state transition.  The module-level constant :data:`ALLOWED_TRANSITIONS`
is the authoritative table of every permitted move in the lifecycle state
machine.

Adding a new permitted transition requires only adding an entry to
:data:`ALLOWED_TRANSITIONS`; the state machine consults this table at
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, FrozenSet, Mapping

from core.lifecycle.models import LifecycleState


# ---------------------------------------------------------------------------
# LifecycleTransition dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleTransition:
    """Immutable descriptor of a single valid lifecycle state transition.

    Instances are created once at module load time and never mutated; the
    ``frozen=True`` flag on the dataclass enforces this.

    Attributes
    ----------
    from_state:
        The source :class:`~core.lifecycle.models.LifecycleState`.
    to_state:
        The destination :class:`~core.lifecycle.models.LifecycleState`.
    allowed:
        Whether this transition is permitted.  Always ``True`` for entries
        in the live table; included for clarity and future extension.
    description:
        Human-readable explanation of when/why this transition occurs.
    """

    from_state: LifecycleState
    to_state: LifecycleState
    allowed: bool = True
    description: str = ""

    def __str__(self) -> str:
        arrow = "→" if self.allowed else "↛"
        return f"{self.from_state.name} {arrow} {self.to_state.name}"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LifecycleTransition("
            f"from_state={self.from_state!r}, "
            f"to_state={self.to_state!r}, "
            f"allowed={self.allowed!r})"
        )


# ---------------------------------------------------------------------------
# Allowed transition table
# ---------------------------------------------------------------------------

S = LifecycleState  # local alias for readability


def _t(
    from_state: LifecycleState,
    to_state: LifecycleState,
    description: str = "",
) -> LifecycleTransition:
    """Shorthand factory for an allowed :class:`LifecycleTransition`."""
    return LifecycleTransition(
        from_state=from_state,
        to_state=to_state,
        allowed=True,
        description=description,
    )


#: The complete set of permitted lifecycle state transitions.
#: Keyed by ``(from_state, to_state)`` for O(1) look-up.
ALLOWED_TRANSITIONS: Final[
    Mapping[tuple[LifecycleState, LifecycleState], LifecycleTransition]
] = {
    # ------------------------------------------------------------------
    # Normal forward progression
    # ------------------------------------------------------------------
    (S.CREATED, S.DISCOVERED): _t(
        S.CREATED, S.DISCOVERED,
        "Module has been found and its manifest read."
    ),
    (S.DISCOVERED, S.VALIDATED): _t(
        S.DISCOVERED, S.VALIDATED,
        "Manifest validation and dependency resolution succeeded."
    ),
    (S.VALIDATED, S.LOADED): _t(
        S.VALIDATED, S.LOADED,
        "Module class imported successfully into the Python runtime."
    ),
    (S.LOADED, S.INITIALIZED): _t(
        S.LOADED, S.INITIALIZED,
        "Module instance constructed and initialize() completed."
    ),
    (S.INITIALIZED, S.STARTING): _t(
        S.INITIALIZED, S.STARTING,
        "start() hook has been invoked; module is starting up."
    ),
    (S.STARTING, S.RUNNING): _t(
        S.STARTING, S.RUNNING,
        "start() hook completed successfully; module is now running."
    ),
    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------
    (S.RUNNING, S.PAUSED): _t(
        S.RUNNING, S.PAUSED,
        "Module temporarily suspended via pause()."
    ),
    (S.PAUSED, S.RUNNING): _t(
        S.PAUSED, S.RUNNING,
        "Module resumed from paused state via resume()."
    ),
    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    (S.RUNNING, S.STOPPING): _t(
        S.RUNNING, S.STOPPING,
        "stop() hook has been invoked; module is shutting down."
    ),
    (S.PAUSED, S.STOPPING): _t(
        S.PAUSED, S.STOPPING,
        "stop() invoked on a paused module."
    ),
    (S.STOPPING, S.STOPPED): _t(
        S.STOPPING, S.STOPPED,
        "stop() hook completed; module is fully stopped."
    ),
    (S.STOPPED, S.UNLOADED): _t(
        S.STOPPED, S.UNLOADED,
        "Module removed from runtime; all references released."
    ),
    # Also allow INITIALIZED → STOPPED for modules that were never started
    (S.INITIALIZED, S.STOPPED): _t(
        S.INITIALIZED, S.STOPPED,
        "Module stopped before it was ever started."
    ),
    # And LOADED → STOPPED for early abort
    (S.LOADED, S.STOPPED): _t(
        S.LOADED, S.STOPPED,
        "Module stopped before initialization."
    ),
    # ------------------------------------------------------------------
    # Failure transitions — any non-terminal operational state → FAILED
    # ------------------------------------------------------------------
    (S.CREATED, S.FAILED): _t(
        S.CREATED, S.FAILED,
        "Module failed during creation phase."
    ),
    (S.DISCOVERED, S.FAILED): _t(
        S.DISCOVERED, S.FAILED,
        "Module failed during discovery/manifest reading."
    ),
    (S.VALIDATED, S.FAILED): _t(
        S.VALIDATED, S.FAILED,
        "Module failed during or after validation."
    ),
    (S.LOADED, S.FAILED): _t(
        S.LOADED, S.FAILED,
        "Module failed during or after loading."
    ),
    (S.INITIALIZED, S.FAILED): _t(
        S.INITIALIZED, S.FAILED,
        "Module failed during or after initialization."
    ),
    (S.STARTING, S.FAILED): _t(
        S.STARTING, S.FAILED,
        "Module failed while starting up."
    ),
    (S.RUNNING, S.FAILED): _t(
        S.RUNNING, S.FAILED,
        "Module failed during normal operation."
    ),
    (S.PAUSED, S.FAILED): _t(
        S.PAUSED, S.FAILED,
        "Module failed while paused."
    ),
    (S.STOPPING, S.FAILED): _t(
        S.STOPPING, S.FAILED,
        "Module failed during shutdown."
    ),
    (S.RECOVERING, S.FAILED): _t(
        S.RECOVERING, S.FAILED,
        "Module failed during recovery attempt."
    ),
    # ------------------------------------------------------------------
    # Recovery path: FAILED → RECOVERING → RUNNING
    # ------------------------------------------------------------------
    (S.FAILED, S.RECOVERING): _t(
        S.FAILED, S.RECOVERING,
        "Recovery sequence initiated for failed module."
    ),
    (S.RECOVERING, S.RUNNING): _t(
        S.RECOVERING, S.RUNNING,
        "Module successfully recovered and is running again."
    ),
    # ------------------------------------------------------------------
    # Unload paths for failed / recovering modules
    # ------------------------------------------------------------------
    (S.FAILED, S.UNLOADED): _t(
        S.FAILED, S.UNLOADED,
        "Failed module forcibly unloaded."
    ),
}


def is_allowed(
    from_state: LifecycleState,
    to_state: LifecycleState,
) -> bool:
    """Return ``True`` if the ``from_state → to_state`` transition is in the
    allowed table.

    Parameters
    ----------
    from_state:
        Current state.
    to_state:
        Desired state.
    """
    return (from_state, to_state) in ALLOWED_TRANSITIONS


def get_transition(
    from_state: LifecycleState,
    to_state: LifecycleState,
) -> LifecycleTransition | None:
    """Return the :class:`LifecycleTransition` for the given pair, or
    ``None`` if the transition is not permitted.

    Parameters
    ----------
    from_state:
        Current state.
    to_state:
        Desired state.
    """
    return ALLOWED_TRANSITIONS.get((from_state, to_state))


def allowed_from(state: LifecycleState) -> FrozenSet[LifecycleState]:
    """Return the set of states reachable from *state* in one step.

    Parameters
    ----------
    state:
        Source state.
    """
    return frozenset(
        to for (frm, to) in ALLOWED_TRANSITIONS if frm is state
    )


__all__ = [
    "LifecycleTransition",
    "ALLOWED_TRANSITIONS",
    "is_allowed",
    "get_transition",
    "allowed_from",
]