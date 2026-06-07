# core/contracts/subsystem.py
"""
Core subsystem contract for POLARIS v5.

Every one of the 30 POLARIS subsystems **must** inherit from
:class:`SubsystemContract` and implement all abstract methods.  The contract
owns lifecycle enforcement: no subsystem can bypass the state machine, and
no method can be called out of order.

Public surface
--------------
* :class:`SubsystemMetadata` — static descriptor attached to every subsystem.
* :class:`SubsystemContract` — abstract base class every subsystem inherits.
* Exception hierarchy re-exported for convenience.
"""

from __future__ import annotations

import abc
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from core.contracts.capability import Capability
from core.contracts.health import HealthReport, HealthStatus
from core.contracts.lifecycle import (
    LifecycleError,
    LifecycleMachine,
    LifecycleState,
    LifecycleTransition,
)
from core.types.identifiers import (
    CapabilityId,
    SubsystemId,
    VersionString,
    make_version,
)

_logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RuntimeErrorBase(Exception):
    """Root exception for all POLARIS runtime errors.

    All domain-specific exceptions inherit from this base so callers can
    catch the entire POLARIS error taxonomy with a single clause.
    """


class RegistrationError(RuntimeErrorBase):
    """Raised during subsystem registration/deregistration failures."""

    def __init__(self, message: str, *, subsystem_id: SubsystemId) -> None:
        super().__init__(message)
        self.subsystem_id = subsystem_id


class DependencyError(RuntimeErrorBase):
    """Raised when a dependency declared by a subsystem cannot be satisfied.

    Attributes
    ----------
    subsystem_id:
        The subsystem whose dependency is unsatisfied.
    missing:
        The set of :class:`~core.types.identifiers.SubsystemId` values that
        could not be resolved.
    """

    def __init__(
        self,
        message: str,
        *,
        subsystem_id: SubsystemId,
        missing: frozenset[SubsystemId],
    ) -> None:
        super().__init__(message)
        self.subsystem_id = subsystem_id
        self.missing = missing


# Re-export lifecycle and capability exceptions at package level
__all__ = [
    "RuntimeErrorBase",
    "RegistrationError",
    "DependencyError",
    "LifecycleError",
    "SubsystemMetadata",
    "SubsystemContract",
]


# ---------------------------------------------------------------------------
# Subsystem metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsystemMetadata:
    """Static, immutable descriptor attached to every subsystem.

    Attributes
    ----------
    id:
        Globally unique :class:`~core.types.identifiers.SubsystemId`.
    name:
        Human-readable display name (e.g. ``"Memory Manager"``).
    version:
        :class:`~core.types.identifiers.VersionString` (SemVer 2.0.0).
    description:
        One-paragraph description of the subsystem's responsibilities.
    dependencies:
        Frozenset of :class:`~core.types.identifiers.SubsystemId` values
        that this subsystem requires to be ``RUNNING`` before it may start.
    capabilities:
        Tuple of :class:`~core.contracts.capability.Capability` descriptors
        declared by this subsystem.
    """

    id: SubsystemId
    name: str
    version: VersionString
    description: str
    dependencies: frozenset[SubsystemId] = field(default_factory=frozenset)
    capabilities: tuple[Capability, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("SubsystemMetadata.name must be a non-empty string.")
        if not self.description or not self.description.strip():
            raise ValueError(
                "SubsystemMetadata.description must be a non-empty string."
            )
        # Validate version
        make_version(self.version)
        # All declared capabilities must be owned by this subsystem.
        for cap in self.capabilities:
            if cap.owner != self.id:
                raise ValueError(
                    f"Capability {cap.id!r} is owned by {cap.owner!r}, "
                    f"not by this subsystem ({self.id!r})."
                )
        # A subsystem must not declare itself as its own dependency.
        if self.id in self.dependencies:
            raise ValueError(
                f"Subsystem {self.id!r} cannot depend on itself."
            )

    @property
    def capability_ids(self) -> frozenset[CapabilityId]:
        """Set of :class:`~core.types.identifiers.CapabilityId` declared
        by this subsystem."""
        return frozenset(c.id for c in self.capabilities)


# ---------------------------------------------------------------------------
# Lifecycle guard decorator
# ---------------------------------------------------------------------------


def _requires_state(*states: LifecycleState, operation: str = "") -> Callable[[F], F]:
    """Method decorator that asserts the lifecycle machine is in one of
    *states* before the decorated method body executes.

    This decorator is **internal** to :class:`SubsystemContract` and operates
    on instance methods that have access to ``self._lifecycle``.

    Parameters
    ----------
    *states:
        Acceptable states for the decorated method.
    operation:
        Human-readable label used in error messages; defaults to the method
        name if not provided.
    """
    def decorator(func: F) -> F:
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(self: "SubsystemContract", *args: Any, **kwargs: Any) -> Any:
            self._lifecycle.assert_in(*states, operation=op_name)
            return func(self, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class SubsystemContract(abc.ABC):
    """Abstract base class that every POLARIS subsystem must inherit.

    Enforcement rules
    -----------------
    The following invariants are enforced at the framework level.
    Violations raise :class:`~core.contracts.lifecycle.LifecycleError`.

    * ``start()`` may only be called when state is ``INITIALIZED``.
    * ``pause()`` may only be called when state is ``RUNNING``.
    * ``resume()`` may only be called when state is ``PAUSED``.
    * ``stop()`` may only be called when state is ``RUNNING``, ``PAUSED``,
      ``RECOVERING``, or ``STOPPING`` — **not** when already ``STOPPED``.
    * ``initialize()`` may only be called when state is ``CREATED``.

    Concrete subclasses implement ``_do_initialize``, ``_do_start``,
    ``_do_pause``, ``_do_resume``, and ``_do_stop``.  The public methods on
    this class manage state transitions and call the private hooks.

    Parameters
    ----------
    metadata:
        The static :class:`SubsystemMetadata` descriptor for this subsystem.
    """

    def __init__(self, metadata: SubsystemMetadata) -> None:
        if not isinstance(metadata, SubsystemMetadata):
            raise TypeError(
                f"metadata must be a SubsystemMetadata instance, "
                f"got {type(metadata).__name__!r}."
            )
        self._metadata = metadata
        self._lifecycle = LifecycleMachine()
        _logger.debug(
            "Subsystem %r instantiated (state=%s).",
            metadata.id,
            self._lifecycle.state.name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> SubsystemMetadata:
        """The immutable :class:`SubsystemMetadata` descriptor."""
        return self._metadata

    @property
    def state(self) -> LifecycleState:
        """Current :class:`~core.contracts.lifecycle.LifecycleState`
        (thread-safe)."""
        return self._lifecycle.state

    @property
    def lifecycle_history(self) -> list[LifecycleTransition]:
        """Ordered list of all lifecycle transitions recorded so far."""
        return self._lifecycle.history

    @property
    def is_running(self) -> bool:
        """``True`` if the subsystem is in the ``RUNNING`` state."""
        return self._lifecycle.state is LifecycleState.RUNNING

    @property
    def is_healthy(self) -> bool:
        """``True`` if both the lifecycle state is ``RUNNING`` **and** the
        :meth:`health` report indicates an operational status."""
        if not self.is_running:
            return False
        report = self.health()
        return report.is_operational

    # ------------------------------------------------------------------
    # Public lifecycle methods (enforce contracts, delegate to hooks)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialise the subsystem.

        Transitions: ``CREATED`` → ``INITIALIZED`` (success)
                     ``CREATED`` → ``FAILED``       (exception in hook)

        The subsystem must perform all one-time setup here (e.g. loading
        configuration, allocating resources).  The method must be idempotent
        with respect to the object state — calling it twice without an
        intervening ``stop()`` is a lifecycle violation.

        Raises
        ------
        LifecycleError
            If the current state is not ``CREATED``.
        """
        self._lifecycle.assert_in(
            LifecycleState.CREATED, operation="initialize"
        )
        _logger.info("Subsystem %r: initializing.", self._metadata.id)
        try:
            self._do_initialize()
            self._lifecycle.transition(
                LifecycleState.INITIALIZED, reason="initialize() succeeded"
            )
            _logger.info("Subsystem %r: initialized.", self._metadata.id)
        except Exception:
            self._lifecycle.transition(
                LifecycleState.FAILED, reason="initialize() raised an exception"
            )
            _logger.exception(
                "Subsystem %r: initialize() failed; state → FAILED.",
                self._metadata.id,
            )
            raise

    def start(self) -> None:
        """Start the subsystem.

        Transitions: ``INITIALIZED`` → ``STARTING`` → ``RUNNING`` (success)
                     ``INITIALIZED`` → ``STARTING`` → ``FAILED`` (exception)

        The subsystem should begin serving requests after this call returns.

        Raises
        ------
        LifecycleError
            If the current state is not ``INITIALIZED``.
        """
        self._lifecycle.assert_in(
            LifecycleState.INITIALIZED, operation="start"
        )
        _logger.info("Subsystem %r: starting.", self._metadata.id)
        self._lifecycle.transition(
            LifecycleState.STARTING, reason="start() called"
        )
        try:
            self._do_start()
            self._lifecycle.transition(
                LifecycleState.RUNNING, reason="start() succeeded"
            )
            _logger.info("Subsystem %r: running.", self._metadata.id)
        except Exception:
            self._lifecycle.transition(
                LifecycleState.FAILED, reason="start() raised an exception"
            )
            _logger.exception(
                "Subsystem %r: start() failed; state → FAILED.",
                self._metadata.id,
            )
            raise

    def pause(self) -> None:
        """Temporarily suspend the subsystem.

        Transitions: ``RUNNING`` → ``PAUSED`` (success)
                     ``RUNNING`` → ``FAILED`` (exception in hook)

        The subsystem should cease accepting new work but retain all internal
        state so that :meth:`resume` can restore full operation cheaply.

        Raises
        ------
        LifecycleError
            If the current state is not ``RUNNING``.
        """
        self._lifecycle.assert_in(
            LifecycleState.RUNNING, operation="pause"
        )
        _logger.info("Subsystem %r: pausing.", self._metadata.id)
        try:
            self._do_pause()
            self._lifecycle.transition(
                LifecycleState.PAUSED, reason="pause() succeeded"
            )
            _logger.info("Subsystem %r: paused.", self._metadata.id)
        except Exception:
            self._lifecycle.transition(
                LifecycleState.FAILED, reason="pause() raised an exception"
            )
            _logger.exception(
                "Subsystem %r: pause() failed; state → FAILED.",
                self._metadata.id,
            )
            raise

    def resume(self) -> None:
        """Resume a paused subsystem.

        Transitions: ``PAUSED`` → ``RUNNING`` (success)
                     ``PAUSED`` → ``FAILED`` (exception in hook)

        Raises
        ------
        LifecycleError
            If the current state is not ``PAUSED``.
        """
        self._lifecycle.assert_in(
            LifecycleState.PAUSED, operation="resume"
        )
        _logger.info("Subsystem %r: resuming.", self._metadata.id)
        try:
            self._do_resume()
            self._lifecycle.transition(
                LifecycleState.RUNNING, reason="resume() succeeded"
            )
            _logger.info("Subsystem %r: running (resumed).", self._metadata.id)
        except Exception:
            self._lifecycle.transition(
                LifecycleState.FAILED, reason="resume() raised an exception"
            )
            _logger.exception(
                "Subsystem %r: resume() failed; state → FAILED.",
                self._metadata.id,
            )
            raise

    def stop(self) -> None:
        """Gracefully shut down the subsystem.

        Transitions: ``RUNNING``    → ``STOPPING`` → ``STOPPED`` (success)
                     ``PAUSED``     → ``STOPPING`` → ``STOPPED`` (success)
                     ``RECOVERING`` → ``STOPPING`` → ``STOPPED`` (success)
                     ``STOPPING``   → ``STOPPED``               (idempotent)
                     ``*``         → ``FAILED``                 (exception)

        Raises
        ------
        LifecycleError
            If the subsystem is in a state from which ``stop()`` is not
            permitted (e.g. already ``STOPPED`` or ``CREATED``).
        """
        self._lifecycle.assert_in(
            LifecycleState.RUNNING,
            LifecycleState.PAUSED,
            LifecycleState.RECOVERING,
            LifecycleState.STOPPING,
            operation="stop",
        )
        if self._lifecycle.state is LifecycleState.STOPPING:
            # Already draining; transition directly to STOPPED.
            _logger.debug(
                "Subsystem %r: stop() called while STOPPING; "
                "transitioning directly to STOPPED.",
                self._metadata.id,
            )
            self._lifecycle.transition(
                LifecycleState.STOPPED, reason="stop() called during STOPPING"
            )
            return

        _logger.info("Subsystem %r: stopping.", self._metadata.id)
        self._lifecycle.transition(
            LifecycleState.STOPPING, reason="stop() called"
        )
        try:
            self._do_stop()
            self._lifecycle.transition(
                LifecycleState.STOPPED, reason="stop() succeeded"
            )
            _logger.info("Subsystem %r: stopped.", self._metadata.id)
        except Exception:
            self._lifecycle.transition(
                LifecycleState.FAILED, reason="stop() raised an exception"
            )
            _logger.exception(
                "Subsystem %r: stop() failed; state → FAILED.",
                self._metadata.id,
            )
            raise

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def health(self) -> HealthReport:
        """Return the current health status of this subsystem.

        Implementations **must** never raise; if health probing itself fails,
        return a :class:`~core.contracts.health.HealthReport` with status
        :attr:`~core.contracts.health.HealthStatus.UNHEALTHY` or
        :attr:`~core.contracts.health.HealthStatus.FAILED`.

        Returns
        -------
        HealthReport
            Current health snapshot.
        """

    # ------------------------------------------------------------------
    # Abstract lifecycle hooks (implemented by concrete subsystems)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _do_initialize(self) -> None:
        """Implementation hook called by :meth:`initialize`.

        Perform one-time resource allocation, configuration loading, and
        dependency wiring here.  Raise any exception to signal failure;
        the framework will transition to ``FAILED``.
        """

    @abc.abstractmethod
    def _do_start(self) -> None:
        """Implementation hook called by :meth:`start`.

        Activate background threads, open network connections, etc.
        Raise any exception to signal failure.
        """

    @abc.abstractmethod
    def _do_pause(self) -> None:
        """Implementation hook called by :meth:`pause`.

        Suspend processing without releasing resources.  Must be
        reversible by :meth:`_do_resume`.
        """

    @abc.abstractmethod
    def _do_resume(self) -> None:
        """Implementation hook called by :meth:`resume`.

        Restore full operation after a :meth:`pause`."""

    @abc.abstractmethod
    def _do_stop(self) -> None:
        """Implementation hook called by :meth:`stop`.

        Release all resources, join threads, close connections.
        Must be safe to call from any operational state.
        """

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{type(self).__name__}("
            f"id={self._metadata.id!r}, "
            f"state={self._lifecycle.state.name})"
        )