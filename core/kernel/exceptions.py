# core/kernel/exceptions.py
"""
POLARIS v5 Runtime Kernel — Exception hierarchy.

All kernel-level exceptions inherit from :class:`KernelError`, which
itself inherits from :class:`Exception`.  This allows callers to catch
the complete kernel error taxonomy with a single ``except KernelError`` clause.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base kernel exception
# ---------------------------------------------------------------------------


class KernelError(Exception):
    """Base exception for all POLARIS v5 Runtime Kernel errors.

    Attributes
    ----------
    kernel_state:
        Optional string describing the kernel state at the time of the error.
    """

    def __init__(
        self,
        message: str,
        *,
        kernel_state: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kernel_state = kernel_state

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{type(self).__name__}("
            f"message={str(self)!r}, "
            f"kernel_state={self.kernel_state!r})"
        )


# ---------------------------------------------------------------------------
# Specific kernel exceptions
# ---------------------------------------------------------------------------


class BootstrapError(KernelError):
    """Raised when the kernel bootstrap sequence fails.

    Bootstrap failures are unrecoverable — the kernel cannot proceed to
    startup without completing the bootstrap phase.

    Attributes
    ----------
    failed_component:
        The name of the component that caused the bootstrap failure.
    """

    def __init__(
        self,
        message: str,
        *,
        kernel_state: str | None = None,
        failed_component: str | None = None,
    ) -> None:
        super().__init__(message, kernel_state=kernel_state)
        self.failed_component = failed_component


class StartupError(KernelError):
    """Raised when the kernel startup sequence fails.

    Startup failures transition the kernel to FAILED state.  Recovery may
    be attempted depending on configuration.

    Attributes
    ----------
    failed_module:
        Optional module id that caused the startup failure.
    """

    def __init__(
        self,
        message: str,
        *,
        kernel_state: str | None = None,
        failed_module: str | None = None,
    ) -> None:
        super().__init__(message, kernel_state=kernel_state)
        self.failed_module = failed_module


class ShutdownError(KernelError):
    """Raised when the kernel shutdown sequence encounters an error.

    Shutdown errors are logged but the kernel attempts to proceed with
    shutdown regardless — resource cleanup must be best-effort.

    Attributes
    ----------
    failed_module:
        Optional module id that caused the shutdown failure.
    """

    def __init__(
        self,
        message: str,
        *,
        kernel_state: str | None = None,
        failed_module: str | None = None,
    ) -> None:
        super().__init__(message, kernel_state=kernel_state)
        self.failed_module = failed_module


class RecoveryError(KernelError):
    """Raised when the kernel recovery sequence fails.

    A :class:`RecoveryError` typically indicates that the kernel has entered
    an unrecoverable FAILED state.

    Attributes
    ----------
    recovery_attempt:
        Which recovery attempt (1-based) failed.
    """

    def __init__(
        self,
        message: str,
        *,
        kernel_state: str | None = None,
        recovery_attempt: int = 1,
    ) -> None:
        super().__init__(message, kernel_state=kernel_state)
        self.recovery_attempt = recovery_attempt


__all__ = [
    "KernelError",
    "BootstrapError",
    "StartupError",
    "ShutdownError",
    "RecoveryError",
]