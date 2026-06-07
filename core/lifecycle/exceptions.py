# core/lifecycle/exceptions.py
"""
Exception hierarchy for the POLARIS v5 Lifecycle Manager.

All lifecycle-related exceptions inherit from :class:`LifecycleError` so
callers can catch either specific exceptions or the entire family with a
single ``except LifecycleError`` clause.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class LifecycleError(Exception):
    """Base class for all POLARIS v5 Lifecycle Manager exceptions.

    Parameters
    ----------
    message:
        Human-readable description of the error.
    module_id:
        Optional id of the module involved in the error.
    """

    def __init__(self, message: str, *, module_id: str | None = None) -> None:
        super().__init__(message)
        self.module_id = module_id
        self.message = message

    def __str__(self) -> str:
        if self.module_id:
            return f"[module={self.module_id!r}] {self.message}"
        return self.message

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"module_id={self.module_id!r})"
        )


# ---------------------------------------------------------------------------
# Specific exceptions
# ---------------------------------------------------------------------------


class InvalidTransitionError(LifecycleError):
    """Raised when a requested lifecycle state transition is not permitted.

    Parameters
    ----------
    from_state:
        The state the module is currently in.
    to_state:
        The state that was requested.
    module_id:
        Optional id of the module.
    """

    def __init__(
        self,
        from_state: Any,
        to_state: Any,
        *,
        module_id: str | None = None,
    ) -> None:
        message = (
            f"Invalid lifecycle transition: {from_state!s} → {to_state!s}. "
            f"This transition is not permitted."
        )
        super().__init__(message, module_id=module_id)
        self.from_state = from_state
        self.to_state = to_state

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"InvalidTransitionError("
            f"from_state={self.from_state!r}, "
            f"to_state={self.to_state!r}, "
            f"module_id={self.module_id!r})"
        )


class StateMachineError(LifecycleError):
    """Raised when the state machine itself is in an inconsistent or invalid
    state — e.g. a machine with no current state, or corrupted history.

    Parameters
    ----------
    message:
        Human-readable description of the inconsistency.
    module_id:
        Optional id of the module whose machine is affected.
    """

    def __init__(self, message: str, *, module_id: str | None = None) -> None:
        super().__init__(message, module_id=module_id)


class ModuleStateError(LifecycleError):
    """Raised when a lifecycle operation is attempted on a module that is not
    in the required state for that operation.

    Parameters
    ----------
    module_id:
        The id of the module.
    expected_states:
        The state(s) that would have been acceptable.
    actual_state:
        The state the module is actually in.
    operation:
        Human-readable name of the attempted operation.
    """

    def __init__(
        self,
        module_id: str,
        *,
        expected_states: Any,
        actual_state: Any,
        operation: str = "",
    ) -> None:
        op_str = f" for operation '{operation}'" if operation else ""
        message = (
            f"Module '{module_id}' is in state {actual_state!s}{op_str}; "
            f"expected one of {expected_states!s}."
        )
        super().__init__(message, module_id=module_id)
        self.expected_states = expected_states
        self.actual_state = actual_state
        self.operation = operation

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ModuleStateError("
            f"module_id={self.module_id!r}, "
            f"expected_states={self.expected_states!r}, "
            f"actual_state={self.actual_state!r}, "
            f"operation={self.operation!r})"
        )


__all__ = [
    "LifecycleError",
    "InvalidTransitionError",
    "StateMachineError",
    "ModuleStateError",
]