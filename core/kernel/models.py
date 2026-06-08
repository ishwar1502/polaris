# core/kernel/models.py
"""
POLARIS v5 Runtime Kernel — Domain models.

Defines the :class:`RuntimeState` enumeration, :class:`KernelConfiguration`
immutable dataclass, and :class:`RuntimeHealthReport` immutable snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, unique
from typing import Any, Final


# ---------------------------------------------------------------------------
# RuntimeState enumeration
# ---------------------------------------------------------------------------


@unique
class RuntimeState(Enum):
    """All states the POLARIS v5 Runtime Kernel may occupy.

    State ordering (forward path)
    -----------------------------
    ::

        CREATED → BOOTSTRAPPING → STARTING → RUNNING
                                           ↘ DEGRADED ↔ RECOVERING
                                → STOPPING → STOPPED
        Any state → FAILED

    Recovery path
    -------------
    ::

        FAILED → RECOVERING → RUNNING
    """

    CREATED = auto()
    """Kernel object has been instantiated; no initialization has begun."""

    BOOTSTRAPPING = auto()
    """Bootstrap sequence is in progress — initializing runtime components."""

    STARTING = auto()
    """Startup sequence is in progress — starting modules."""

    RUNNING = auto()
    """Kernel is fully operational; all required modules are running."""

    DEGRADED = auto()
    """Kernel is operational but one or more modules have failed or degraded."""

    RECOVERING = auto()
    """Kernel is attempting automatic recovery from a failed state."""

    STOPPING = auto()
    """Shutdown sequence is in progress — stopping and unloading modules."""

    STOPPED = auto()
    """Kernel has cleanly shut down.  Terminal state."""

    FAILED = auto()
    """Kernel entered an unrecoverable error state."""

    def is_operational(self) -> bool:
        """Return ``True`` if the kernel is in an active operational state."""
        return self in _OPERATIONAL_STATES

    def is_terminal(self) -> bool:
        """Return ``True`` if the kernel cannot transition further."""
        return self in _TERMINAL_STATES


# Derived sets (populated after class definition)
_OPERATIONAL_STATES: Final[frozenset[RuntimeState]] = frozenset({
    RuntimeState.RUNNING,
    RuntimeState.DEGRADED,
    RuntimeState.RECOVERING,
})

_TERMINAL_STATES: Final[frozenset[RuntimeState]] = frozenset({
    RuntimeState.STOPPED,
    RuntimeState.FAILED,
})

OPERATIONAL_STATES: Final[frozenset[RuntimeState]] = _OPERATIONAL_STATES
TERMINAL_STATES: Final[frozenset[RuntimeState]] = _TERMINAL_STATES


# ---------------------------------------------------------------------------
# KernelConfiguration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelConfiguration:
    """Immutable configuration descriptor for the Runtime Kernel.

    Attributes
    ----------
    environment:
        Deployment environment label (e.g. ``"production"``, ``"development"``).
    auto_start:
        If ``True``, calling :meth:`~core.kernel.kernel.RuntimeKernel.bootstrap`
        will automatically proceed to startup.  Defaults to ``False``.
    enable_recovery:
        If ``True``, the kernel will attempt automatic recovery when it
        transitions to :attr:`RuntimeState.FAILED`.  Defaults to ``True``.
    health_interval:
        Interval in seconds between automatic health checks.  Set to ``0``
        to disable periodic health monitoring.  Defaults to ``30``.
    max_recovery_attempts:
        Maximum number of consecutive recovery attempts before the kernel
        permanently enters :attr:`RuntimeState.FAILED`.  Defaults to ``3``.
    shutdown_timeout:
        Maximum seconds to wait for a graceful shutdown before forcing
        module termination.  Defaults to ``30``.
    metadata:
        Arbitrary key-value metadata attached to this configuration (e.g.
        deployment tags, build version).
    """

    environment: str = "development"
    auto_start: bool = False
    enable_recovery: bool = True
    health_interval: float = 30.0
    max_recovery_attempts: int = 3
    shutdown_timeout: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.environment or not self.environment.strip():
            raise ValueError("KernelConfiguration.environment must be non-empty.")
        if self.health_interval < 0:
            raise ValueError("KernelConfiguration.health_interval cannot be negative.")
        if self.max_recovery_attempts < 0:
            raise ValueError(
                "KernelConfiguration.max_recovery_attempts cannot be negative."
            )
        if self.shutdown_timeout < 0:
            raise ValueError("KernelConfiguration.shutdown_timeout cannot be negative.")


# ---------------------------------------------------------------------------
# RuntimeHealthReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeHealthReport:
    """Immutable snapshot of the kernel's health at a point in time.

    Attributes
    ----------
    runtime_state:
        The current :class:`RuntimeState` of the kernel.
    loaded_modules:
        Tuple of module ids that are currently in LOADED / INITIALIZED state.
    running_modules:
        Tuple of module ids that are currently RUNNING.
    failed_modules:
        Tuple of module ids that are in a FAILED state.
    memory_status:
        String describing the memory gateway status (e.g. ``"ok"``,
        ``"degraded"``, ``"unavailable"``).
    event_bus_status:
        String describing the event bus status.
    timestamp:
        UTC timestamp when this report was generated.
    metadata:
        Additional diagnostic key-value pairs.
    """

    runtime_state: RuntimeState
    loaded_modules: tuple[str, ...]
    running_modules: tuple[str, ...]
    failed_modules: tuple[str, ...]
    memory_status: str
    event_bus_status: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("RuntimeHealthReport.timestamp must be timezone-aware.")

    @property
    def is_healthy(self) -> bool:
        """``True`` if the kernel is RUNNING with no failed modules."""
        return (
            self.runtime_state is RuntimeState.RUNNING
            and len(self.failed_modules) == 0
        )

    @property
    def is_operational(self) -> bool:
        """``True`` if the kernel is in any operational state."""
        return self.runtime_state.is_operational()

    def to_dict(self) -> dict[str, Any]:
        """Serialise this report to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-serialisable representation.
        """
        return {
            "runtime_state": self.runtime_state.name,
            "loaded_modules": list(self.loaded_modules),
            "running_modules": list(self.running_modules),
            "failed_modules": list(self.failed_modules),
            "memory_status": self.memory_status,
            "event_bus_status": self.event_bus_status,
            "timestamp": self.timestamp.isoformat(),
            "is_healthy": self.is_healthy,
            "is_operational": self.is_operational,
            "metadata": self.metadata,
        }


__all__ = [
    "RuntimeState",
    "OPERATIONAL_STATES",
    "TERMINAL_STATES",
    "KernelConfiguration",
    "RuntimeHealthReport",
]