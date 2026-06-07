# core/lifecycle/__init__.py
"""
POLARIS v5 Lifecycle Manager — Runtime Phase 5.

The Lifecycle Manager governs operational state transitions for all modules.
Where the Module Loader *loads* modules, the Lifecycle Manager *controls* them.

Public API
----------
- :class:`LifecycleState`      — Full 13-state enumeration.
- :class:`LifecycleTransition` — Immutable transition descriptor.
- :class:`StateMachine`        — Per-module state machine.
- :class:`LifecycleManager`    — Central controller for all module lifecycles.
- :class:`LifecycleError`      — Base lifecycle exception.
- :class:`InvalidTransitionError` — Rejected state transition.
- :class:`StateMachineError`   — State machine consistency error.
- :class:`ModuleStateError`    — Module not in expected state.
"""

from core.lifecycle.exceptions import (
    InvalidTransitionError,
    LifecycleError,
    ModuleStateError,
    StateMachineError,
)
from core.lifecycle.manager import LifecycleManager
from core.lifecycle.models import LifecycleState
from core.lifecycle.state_machine import StateMachine
from core.lifecycle.transitions import LifecycleTransition

__all__ = [
    "LifecycleState",
    "LifecycleTransition",
    "StateMachine",
    "LifecycleManager",
    "LifecycleError",
    "InvalidTransitionError",
    "StateMachineError",
    "ModuleStateError",
]