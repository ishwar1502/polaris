# core/loader/exceptions.py
"""
Exception hierarchy for the POLARIS v5 Module Loader.

All module-loader failures are subclasses of :class:`ModuleLoaderError` so
callers can catch the entire loader error taxonomy with a single
``except ModuleLoaderError``.

Hierarchy
---------
ModuleLoaderError
├── ModuleDiscoveryError
├── DependencyResolutionError
│   └── CircularDependencyError
└── ModuleValidationError
"""

from __future__ import annotations

from typing import Sequence


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ModuleLoaderError(Exception):
    """Root exception for all POLARIS v5 Module Loader failures.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    module_id:
        Optional module identifier involved in the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        module_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.module_id = module_id

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"module_id={self.module_id!r})"
        )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class ModuleDiscoveryError(ModuleLoaderError):
    """Raised when the discovery phase cannot locate or read a module manifest.

    Parameters
    ----------
    message:
        Human-readable description of the discovery failure.
    module_path:
        Optional filesystem path where discovery failed.
    module_id:
        Optional module identifier, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        module_path: str | None = None,
        module_id: str | None = None,
    ) -> None:
        super().__init__(message, module_id=module_id)
        self.module_path = module_path

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"module_path={self.module_path!r}, "
            f"module_id={self.module_id!r})"
        )


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


class DependencyResolutionError(ModuleLoaderError):
    """Raised when one or more declared dependencies cannot be satisfied.

    Parameters
    ----------
    message:
        Human-readable description of the resolution failure.
    module_id:
        The module whose dependencies could not be resolved.
    missing:
        Collection of dependency IDs that are absent from the registry.
    """

    def __init__(
        self,
        message: str,
        *,
        module_id: str | None = None,
        missing: Sequence[str] | None = None,
    ) -> None:
        super().__init__(message, module_id=module_id)
        self.missing: list[str] = list(missing) if missing else []

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"module_id={self.module_id!r}, "
            f"missing={self.missing!r})"
        )


class CircularDependencyError(DependencyResolutionError):
    """Raised when a circular dependency chain is detected in the module graph.

    Parameters
    ----------
    message:
        Human-readable description of the cycle.
    module_id:
        A module involved in the cycle (the one at which detection triggered).
    cycle:
        Ordered sequence of module IDs that form the detected cycle.
    missing:
        Unused; retained for API compatibility with parent class.
    """

    def __init__(
        self,
        message: str,
        *,
        module_id: str | None = None,
        cycle: Sequence[str] | None = None,
        missing: Sequence[str] | None = None,
    ) -> None:
        super().__init__(message, module_id=module_id, missing=missing)
        self.cycle: list[str] = list(cycle) if cycle else []

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"module_id={self.module_id!r}, "
            f"cycle={self.cycle!r})"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ModuleValidationError(ModuleLoaderError):
    """Raised when a module manifest or module class fails validation.

    Parameters
    ----------
    message:
        Human-readable description of the validation failure.
    module_id:
        Optional module identifier that failed validation.
    field:
        Optional name of the manifest field that is invalid.
    invalid_value:
        Optional string representation of the offending value.
    """

    def __init__(
        self,
        message: str,
        *,
        module_id: str | None = None,
        field: str | None = None,
        invalid_value: object = None,
    ) -> None:
        super().__init__(message, module_id=module_id)
        self.field = field
        self.invalid_value = invalid_value

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"module_id={self.module_id!r}, "
            f"field={self.field!r}, "
            f"invalid_value={self.invalid_value!r})"
        )


__all__ = [
    "ModuleLoaderError",
    "ModuleDiscoveryError",
    "DependencyResolutionError",
    "CircularDependencyError",
    "ModuleValidationError",
]