# subsystems/astra/exceptions.py
"""
ASTRA v5 Exception Hierarchy.

All exceptions raised by ASTRA subsystem code inherit from
:class:`AstraError` so callers can catch the entire ASTRA error taxonomy
with a single clause while still distinguishing specific failure modes.
"""

from __future__ import annotations


class AstraError(Exception):
    """Root exception for all ASTRA subsystem errors."""


class AstraNotInitializedError(AstraError):
    """Raised when an ASTRA operation is called before the subsystem is running."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"ASTRA is not running. Cannot execute '{operation}'. "
            "Call initialize() and start() first."
        )
        self.operation = operation


class IdentityNotFoundError(AstraError):
    """Raised when the identity profile has not been created yet."""

    def __init__(self) -> None:
        super().__init__(
            "No identity profile exists. Call update_identity() to create one."
        )


class IdentityValidationError(AstraError):
    """Raised when identity data fails validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class GoalNotFoundError(AstraError):
    """Raised when a goal_id does not correspond to any known goal."""

    def __init__(self, goal_id: str) -> None:
        super().__init__(f"Goal '{goal_id}' not found.")
        self.goal_id = goal_id


class GoalValidationError(AstraError):
    """Raised when goal data fails validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class GoalStateError(AstraError):
    """Raised when a goal state transition is invalid."""

    def __init__(
        self,
        goal_id: str,
        current_state: str,
        attempted_state: str,
    ) -> None:
        super().__init__(
            f"Cannot transition goal '{goal_id}' from {current_state} "
            f"to {attempted_state}."
        )
        self.goal_id = goal_id
        self.current_state = current_state
        self.attempted_state = attempted_state


class PreferenceValidationError(AstraError):
    """Raised when preference data fails validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class DigitalTwinError(AstraError):
    """Raised when the Digital Twin Engine encounters an irrecoverable error."""


class ConsistencyError(AstraError):
    """Raised when the Consistency Engine detects a critical integrity violation."""


class EvolutionError(AstraError):
    """Raised when an identity evolution attempt fails validation."""

    def __init__(self, message: str, *, confidence: float | None = None) -> None:
        super().__init__(message)
        self.confidence = confidence


class InsufficientEvidenceError(EvolutionError):
    """Raised when evidence accumulation does not yet justify an evolution."""

    def __init__(
        self,
        required: int,
        available: int,
        confidence: float,
    ) -> None:
        super().__init__(
            f"Insufficient evidence for identity evolution: "
            f"required={required}, available={available}, confidence={confidence:.2f}.",
            confidence=confidence,
        )
        self.required = required
        self.available = available