# core/memory/exceptions.py
"""
Exception hierarchy for the POLARIS v5 Memory Gateway.

All memory-subsystem failures raise a subclass of :class:`MemoryError` so
callers can catch the entire family with a single ``except MemoryError``.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class MemoryError(Exception):
    """Root exception for all Memory Gateway failures.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    memory_type:
        Optional :class:`~core.memory.models.MemoryType` name involved in the
        failing operation.
    operation:
        Optional operation name (``"store"``, ``"retrieve"``, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        memory_type: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.memory_type = memory_type
        self.operation = operation

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"memory_type={self.memory_type!r}, "
            f"operation={self.operation!r})"
        )


# ---------------------------------------------------------------------------
# Subclasses
# ---------------------------------------------------------------------------


class MemoryProviderError(MemoryError):
    """Raised when a :class:`~core.memory.provider.MemoryProvider` operation
    fails at runtime (e.g. backend unavailable, I/O error).

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    provider_id:
        Optional identifier of the provider that failed.
    memory_type:
        Optional memory type being served by the failing provider.
    operation:
        Optional operation name.
    cause:
        Optional underlying exception that triggered this error.
    """

    def __init__(
        self,
        message: str,
        *,
        provider_id: str | None = None,
        memory_type: str | None = None,
        operation: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, memory_type=memory_type, operation=operation)
        self.provider_id = provider_id
        self.cause = cause

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"provider_id={self.provider_id!r}, "
            f"memory_type={self.memory_type!r}, "
            f"operation={self.operation!r})"
        )


class MemoryRoutingError(MemoryError):
    """Raised when the :class:`~core.memory.router.MemoryRouter` cannot route
    a request to an appropriate provider.

    This typically indicates that no provider has been registered for the
    requested :class:`~core.memory.models.MemoryType`, or that the registered
    provider is not available.

    Parameters
    ----------
    message:
        Human-readable description of the routing failure.
    memory_type:
        Memory type for which routing failed.
    operation:
        Optional operation name that triggered the routing attempt.
    """

    def __init__(
        self,
        message: str,
        *,
        memory_type: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message, memory_type=memory_type, operation=operation)


class MemoryValidationError(MemoryError):
    """Raised when a :class:`~core.memory.request.MemoryRequest` or an
    argument to a Gateway method fails validation.

    Parameters
    ----------
    message:
        Human-readable description of the validation failure.
    field:
        Name of the field that failed validation.
    invalid_value:
        String representation of the offending value.
    memory_type:
        Optional memory type context.
    operation:
        Optional operation name context.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        invalid_value: Any = None,
        memory_type: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message, memory_type=memory_type, operation=operation)
        self.field = field
        self.invalid_value = invalid_value

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"field={self.field!r}, "
            f"invalid_value={self.invalid_value!r})"
        )