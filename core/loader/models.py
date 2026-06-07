# core/loader/models.py
"""
Domain models for the POLARIS v5 Module Loader.

This module defines the :class:`ModuleState` enumeration that describes the
full lifecycle of a loadable module, plus module-level constants used across
the loader subsystem.

Module State Machine
--------------------
The following transitions are valid::

    DISCOVERED → VALIDATED → LOADED → INITIALIZED → RUNNING
                                                   ↓
                                              PAUSED ←→ RUNNING
                                                   ↓
                                              STOPPED
    Any operational state → FAILED

A module enters ``FAILED`` when any loader or runtime operation raises an
unhandled exception.
"""

from __future__ import annotations

from enum import Enum, auto, unique
from typing import Final


# ---------------------------------------------------------------------------
# Module state enumeration
# ---------------------------------------------------------------------------


@unique
class ModuleState(Enum):
    """All possible states a POLARIS v5 module may occupy.

    The loader enforces strict ordering; a module cannot skip states.
    """

    DISCOVERED = auto()
    """Module has been found on disk; its manifest has been read but not yet
    validated."""

    VALIDATED = auto()
    """Manifest has passed all structural and semantic validation checks;
    dependencies have been resolved against the discovery set."""

    LOADED = auto()
    """Module class has been imported into the Python runtime via
    :func:`importlib.import_module`; the class object is held in memory."""

    INITIALIZED = auto()
    """Module instance has been constructed and :meth:`initialize` has
    completed successfully."""

    RUNNING = auto()
    """Module is fully operational and serving requests."""

    PAUSED = auto()
    """Module has been temporarily suspended; recoverable without
    re-initialization."""

    STOPPED = auto()
    """Module has been gracefully shut down.  Terminal state for a given
    instance; the module must be unloaded and reloaded to restart."""

    FAILED = auto()
    """Module entered an unrecoverable error state during loading,
    initialization, starting, or operation."""

    def is_operational(self) -> bool:
        """Return ``True`` if the module is in an active operational state.

        Operational states are :attr:`RUNNING` and :attr:`PAUSED`.
        """
        return self in _OPERATIONAL_STATES

    def is_terminal(self) -> bool:
        """Return ``True`` if the state has no valid forward transition under
        normal (non-error) circumstances.

        Terminal states are :attr:`STOPPED` and :attr:`FAILED`.
        """
        return self in _TERMINAL_STATES

    def can_start(self) -> bool:
        """Return ``True`` if a module in this state may be started."""
        return self is ModuleState.INITIALIZED

    def can_stop(self) -> bool:
        """Return ``True`` if a module in this state may be stopped."""
        return self in _STOPPABLE_STATES


# ---------------------------------------------------------------------------
# State sets (derived from the enum, kept in sync automatically)
# ---------------------------------------------------------------------------

_OPERATIONAL_STATES: Final[frozenset[ModuleState]] = frozenset({
    ModuleState.RUNNING,
    ModuleState.PAUSED,
})

_TERMINAL_STATES: Final[frozenset[ModuleState]] = frozenset({
    ModuleState.STOPPED,
    ModuleState.FAILED,
})

_STOPPABLE_STATES: Final[frozenset[ModuleState]] = frozenset({
    ModuleState.RUNNING,
    ModuleState.PAUSED,
    ModuleState.INITIALIZED,
    ModuleState.LOADED,
})

# Public aliases so external code can reference the sets without importing
# the private names.
OPERATIONAL_STATES: Final[frozenset[ModuleState]] = _OPERATIONAL_STATES
TERMINAL_STATES: Final[frozenset[ModuleState]] = _TERMINAL_STATES
STOPPABLE_STATES: Final[frozenset[ModuleState]] = _STOPPABLE_STATES


__all__ = [
    "ModuleState",
    "OPERATIONAL_STATES",
    "TERMINAL_STATES",
    "STOPPABLE_STATES",
]