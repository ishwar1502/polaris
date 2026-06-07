# core/contracts/health.py
"""
Health reporting primitives for POLARIS v5 subsystems.

Every subsystem exposes a :meth:`~core.contracts.subsystem.SubsystemContract.health`
method that returns a :class:`HealthReport`.  The runtime aggregates these
reports to drive observability, auto-recovery, and alerting pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, unique
from typing import Any


# ---------------------------------------------------------------------------
# Health status enumeration
# ---------------------------------------------------------------------------


@unique
class HealthStatus(Enum):
    """Coarse-grained health classification for a POLARIS subsystem.

    The ordering from best to worst is:
    ``HEALTHY`` > ``DEGRADED`` > ``UNHEALTHY`` > ``FAILED``
    """

    HEALTHY = auto()
    """All internal checks pass; the subsystem is fully operational."""

    DEGRADED = auto()
    """The subsystem is operational but one or more non-critical checks
    have failed or performance is below baseline thresholds."""

    UNHEALTHY = auto()
    """The subsystem is unable to serve requests reliably; intervention
    may be required but automatic recovery is still possible."""

    FAILED = auto()
    """The subsystem has entered an unrecoverable error state and must
    be stopped and restarted at the registry level."""

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    def is_worse_than(self, other: "HealthStatus") -> bool:
        """Return ``True`` if this status is worse than *other*.

        The severity ordering is ``HEALTHY < DEGRADED < UNHEALTHY < FAILED``.

        Parameters
        ----------
        other:
            Status to compare against.
        """
        _SEVERITY: dict[HealthStatus, int] = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNHEALTHY: 2,
            HealthStatus.FAILED: 3,
        }
        return _SEVERITY[self] > _SEVERITY[other]

    def is_better_than(self, other: "HealthStatus") -> bool:
        """Return ``True`` if this status is better than *other*.

        Parameters
        ----------
        other:
            Status to compare against.
        """
        return other.is_worse_than(self)

    def is_operational(self) -> bool:
        """Return ``True`` if the status does not indicate a hard failure.

        Both ``HEALTHY`` and ``DEGRADED`` are considered operational.
        """
        return self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


# ---------------------------------------------------------------------------
# Health check record (individual check result)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """Result of a single named health check.

    Attributes
    ----------
    name:
        Short, human-readable label for the check (e.g. ``"db-connection"``).
    passed:
        Whether the check succeeded.
    message:
        Detailed description of the outcome or failure reason.
    latency_ms:
        Optional time taken to execute the check, in milliseconds.
    metadata:
        Arbitrary additional data from the check (e.g. error codes).
    """

    name: str
    passed: bool
    message: str = ""
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("HealthCheckResult.name must be a non-empty string.")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("HealthCheckResult.latency_ms cannot be negative.")


# ---------------------------------------------------------------------------
# Aggregate health report
# ---------------------------------------------------------------------------


@dataclass
class HealthReport:
    """Aggregate health snapshot emitted by a subsystem's ``health()`` method.

    Attributes
    ----------
    status:
        Overall coarse-grained health status.
    message:
        Human-readable summary of the health state.
    timestamp:
        UTC timestamp at which the report was generated.
    checks:
        Ordered list of individual :class:`HealthCheckResult` instances
        that informed the overall *status*.
    metadata:
        Arbitrary key-value pairs for observability tooling (e.g. build
        version, resource utilisation counters).
    """

    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checks: list[HealthCheckResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("HealthReport.timestamp must be timezone-aware.")
        if not self.message:
            raise ValueError("HealthReport.message must be a non-empty string.")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def failed_checks(self) -> list[HealthCheckResult]:
        """All :class:`HealthCheckResult` entries where ``passed`` is ``False``."""
        return [c for c in self.checks if not c.passed]

    @property
    def passed_checks(self) -> list[HealthCheckResult]:
        """All :class:`HealthCheckResult` entries where ``passed`` is ``True``."""
        return [c for c in self.checks if c.passed]

    @property
    def check_count(self) -> int:
        """Total number of checks recorded in this report."""
        return len(self.checks)

    @property
    def is_operational(self) -> bool:
        """Delegate to :meth:`HealthStatus.is_operational`."""
        return self.status.is_operational()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def healthy(
        cls,
        message: str = "All checks passed.",
        *,
        checks: list[HealthCheckResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "HealthReport":
        """Convenience factory for a :attr:`HealthStatus.HEALTHY` report.

        Parameters
        ----------
        message:
            Summary message.
        checks:
            Optional list of individual check results.
        metadata:
            Optional extra metadata.
        """
        return cls(
            status=HealthStatus.HEALTHY,
            message=message,
            checks=checks or [],
            metadata=metadata or {},
        )

    @classmethod
    def degraded(
        cls,
        message: str,
        *,
        checks: list[HealthCheckResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "HealthReport":
        """Convenience factory for a :attr:`HealthStatus.DEGRADED` report.

        Parameters
        ----------
        message:
            Summary of what is degraded.
        checks:
            Optional list of individual check results.
        metadata:
            Optional extra metadata.
        """
        return cls(
            status=HealthStatus.DEGRADED,
            message=message,
            checks=checks or [],
            metadata=metadata or {},
        )

    @classmethod
    def unhealthy(
        cls,
        message: str,
        *,
        checks: list[HealthCheckResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "HealthReport":
        """Convenience factory for a :attr:`HealthStatus.UNHEALTHY` report.

        Parameters
        ----------
        message:
            Summary of what has failed.
        checks:
            Optional list of individual check results.
        metadata:
            Optional extra metadata.
        """
        return cls(
            status=HealthStatus.UNHEALTHY,
            message=message,
            checks=checks or [],
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        message: str,
        *,
        checks: list[HealthCheckResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "HealthReport":
        """Convenience factory for a :attr:`HealthStatus.FAILED` report.

        Parameters
        ----------
        message:
            Summary of the fatal failure.
        checks:
            Optional list of individual check results.
        metadata:
            Optional extra metadata.
        """
        return cls(
            status=HealthStatus.FAILED,
            message=message,
            checks=checks or [],
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise this report to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-serialisable representation (timestamps as ISO-8601 strings).
        """
        return {
            "status": self.status.name,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "metadata": c.metadata,
                }
                for c in self.checks
            ],
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"HealthReport(status={self.status.name}, "
            f"checks={self.check_count}, "
            f"failed={len(self.failed_checks)}, "
            f"ts={self.timestamp.isoformat()})"
        )