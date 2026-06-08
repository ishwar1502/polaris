# core/kernel/kernel.py
"""
POLARIS v5 Runtime Kernel — Central Orchestrator.

:class:`RuntimeKernel` is the operational heart of POLARIS v5.  It coordinates
all runtime subsystems through a well-defined lifecycle:

    CREATED → BOOTSTRAPPING → STARTING → RUNNING
                                       ↘ DEGRADED ↔ RECOVERING
                           → STOPPING → STOPPED
    Any state → FAILED

The kernel is the single entry point for:
* Bootstrapping all runtime components
* Starting and stopping the runtime
* Module registration, loading, and lifecycle management
* Health monitoring and reporting
* Automatic recovery from failures

Thread Safety
-------------
All state mutations are protected by a :class:`threading.RLock`.  The kernel
is safe for concurrent use from multiple threads.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from core.kernel.bootstrap import BootstrapManager
from core.kernel.exceptions import (
    BootstrapError,
    KernelError,
    RecoveryError,
    ShutdownError,
    StartupError,
)
from core.kernel.health import HealthMonitor
from core.kernel.models import (
    KernelConfiguration,
    RuntimeHealthReport,
    RuntimeState,
)
from core.kernel.shutdown import ShutdownManager
from core.kernel.startup import StartupManager

_logger = logging.getLogger(__name__)

# Valid kernel state transitions
_VALID_TRANSITIONS: dict[RuntimeState, frozenset[RuntimeState]] = {
    RuntimeState.CREATED: frozenset({
        RuntimeState.BOOTSTRAPPING,
        RuntimeState.FAILED,
    }),
    RuntimeState.BOOTSTRAPPING: frozenset({
        RuntimeState.STARTING,
        RuntimeState.FAILED,
    }),
    RuntimeState.STARTING: frozenset({
        RuntimeState.RUNNING,
        RuntimeState.DEGRADED,
        RuntimeState.STOPPING,
        RuntimeState.FAILED,
    }),
    RuntimeState.RUNNING: frozenset({
        RuntimeState.DEGRADED,
        RuntimeState.STOPPING,
        RuntimeState.FAILED,
    }),
    RuntimeState.DEGRADED: frozenset({
        RuntimeState.RUNNING,
        RuntimeState.RECOVERING,
        RuntimeState.STOPPING,
        RuntimeState.FAILED,
    }),
    RuntimeState.RECOVERING: frozenset({
        RuntimeState.RUNNING,
        RuntimeState.DEGRADED,
        RuntimeState.STOPPING,
        RuntimeState.FAILED,
    }),
    RuntimeState.STOPPING: frozenset({
        RuntimeState.STOPPED,
        RuntimeState.FAILED,
    }),
    RuntimeState.STOPPED: frozenset(),  # terminal
    RuntimeState.FAILED: frozenset({
        RuntimeState.RECOVERING,
        RuntimeState.STOPPING,
    }),
}


class RuntimeKernel:
    """Central orchestrator for the POLARIS v5 cognitive runtime.

    The kernel coordinates all runtime subsystems through their complete
    lifecycle.  It is designed to be instantiated once per process and
    shared across all subsystems.

    Parameters
    ----------
    config:
        Immutable :class:`KernelConfiguration` governing kernel behavior.
        If ``None``, a default configuration is used.

    Examples
    --------
    .. code-block:: python

        from core.kernel import RuntimeKernel, KernelConfiguration

        config = KernelConfiguration(environment="production")
        kernel = RuntimeKernel(config)

        kernel.bootstrap()
        kernel.start()

        report = kernel.health()
        print(report.running_modules)

        kernel.stop()
    """

    def __init__(self, config: KernelConfiguration | None = None) -> None:
        self._config = config or KernelConfiguration()
        self._state = RuntimeState.CREATED
        self._lock = threading.RLock()
        self._components: dict[str, Any] = {}
        self._bootstrap_manager: BootstrapManager | None = None
        self._startup_manager: StartupManager | None = None
        self._shutdown_manager: ShutdownManager | None = None
        self._health_monitor: HealthMonitor | None = None
        self._recovery_attempts = 0
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None

        _logger.info(
            "RuntimeKernel created [environment=%s].",
            self._config.environment,
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> RuntimeState:
        """Current :class:`RuntimeState` of the kernel (thread-safe)."""
        with self._lock:
            return self._state

    @property
    def config(self) -> KernelConfiguration:
        """The immutable :class:`KernelConfiguration`."""
        return self._config

    @property
    def components(self) -> dict[str, Any]:
        """Read-only snapshot of bootstrapped components."""
        with self._lock:
            return dict(self._components)

    @property
    def is_running(self) -> bool:
        """``True`` if the kernel is in RUNNING or DEGRADED state."""
        return self._state in (RuntimeState.RUNNING, RuntimeState.DEGRADED)

    # ------------------------------------------------------------------
    # Primary lifecycle operations
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Bootstrap the runtime kernel.

        Initializes all runtime components in dependency order:
        Event Bus → Memory Gateway → Registry → Module Loader → Lifecycle Manager.

        After successful bootstrap, the kernel transitions to ``STARTING`` state.
        If :attr:`~KernelConfiguration.auto_start` is ``True``, startup is
        automatically initiated.

        Raises
        ------
        KernelError
            If the kernel is not in ``CREATED`` state.
        BootstrapError
            If any component fails to initialize.
        """
        with self._lock:
            self._assert_state(RuntimeState.CREATED, operation="bootstrap")
            self._transition(RuntimeState.BOOTSTRAPPING)

        _logger.info("Bootstrapping kernel.")

        try:
            bootstrap_mgr = BootstrapManager(self._config)
            components = bootstrap_mgr.bootstrap()

            with self._lock:
                self._components = components
                self._bootstrap_manager = bootstrap_mgr
                self._startup_manager = StartupManager(components)
                self._shutdown_manager = ShutdownManager(components)
                self._health_monitor = HealthMonitor(components)
                self._transition(RuntimeState.STARTING)

            self._publish_event("polaris.kernel.bootstrapped", priority="HIGH")
            _logger.info("Kernel bootstrapped successfully.")

        except BootstrapError:
            self._transition_to_failed("Bootstrap failed.")
            raise
        except Exception as exc:
            self._transition_to_failed(f"Bootstrap error: {exc}")
            raise BootstrapError(
                f"Unexpected error during bootstrap: {exc}",
            ) from exc

        if self._config.auto_start:
            self.start()

    def start(self) -> None:
        """Start the kernel runtime.

        Starts all registered modules.  The kernel must be in ``STARTING``
        state (after :meth:`bootstrap`) to call this method.

        Raises
        ------
        KernelError
            If the kernel is not in ``STARTING`` state.
        StartupError
            If module startup fails.
        """
        with self._lock:
            self._assert_state(RuntimeState.STARTING, operation="start")

        _logger.info("Starting kernel runtime.")

        try:
            startup_mgr = self._startup_manager
            if startup_mgr is None:
                raise StartupError("StartupManager not available; call bootstrap() first.")

            startup_mgr.start()

            with self._lock:
                self._started_at = datetime.now(timezone.utc)
                # Check for failed modules → DEGRADED vs RUNNING
                if self._has_failed_modules():
                    self._transition(RuntimeState.DEGRADED)
                    _logger.warning("Kernel started in DEGRADED state — some modules failed.")
                else:
                    self._transition(RuntimeState.RUNNING)
                    _logger.info("Kernel running.")

            self._publish_event("polaris.kernel.started", priority="HIGH")

        except StartupError:
            self._transition_to_failed("Startup failed.")
            raise
        except Exception as exc:
            self._transition_to_failed(f"Startup error: {exc}")
            raise StartupError(
                f"Unexpected error during startup: {exc}",
            ) from exc

    def stop(self) -> None:
        """Gracefully stop the kernel runtime.

        Stops all running modules, unloads them, and releases resources.
        Publishes ``KernelStopped`` event.

        Raises
        ------
        KernelError
            If the kernel cannot be stopped from its current state.
        """
        with self._lock:
            current = self._state
            if current is RuntimeState.STOPPED:
                _logger.debug("Kernel already stopped; stop() is a no-op.")
                return
            if current is RuntimeState.CREATED:
                raise KernelError(
                    "Cannot stop kernel that has not been bootstrapped.",
                    kernel_state=current.name,
                )
            self._transition(RuntimeState.STOPPING)

        _logger.info("Stopping kernel runtime.")

        try:
            shutdown_mgr = self._shutdown_manager
            if shutdown_mgr is not None:
                shutdown_mgr.shutdown()

            with self._lock:
                self._stopped_at = datetime.now(timezone.utc)
                self._transition(RuntimeState.STOPPED)

            self._publish_event("polaris.kernel.stopped", priority="HIGH")
            _logger.info("Kernel stopped.")

        except ShutdownError as exc:
            _logger.error("Shutdown error: %s", exc)
            # Still attempt to reach STOPPED
            with self._lock:
                if self._state is RuntimeState.STOPPING:
                    self._stopped_at = datetime.now(timezone.utc)
                    self._transition(RuntimeState.STOPPED)
        except Exception as exc:
            _logger.error("Unexpected shutdown error: %s", exc)
            with self._lock:
                if self._state is RuntimeState.STOPPING:
                    self._transition(RuntimeState.FAILED)

    def restart(self) -> None:
        """Restart the kernel runtime.

        Performs a graceful stop followed by a fresh bootstrap and start.

        Raises
        ------
        KernelError
            If the kernel cannot be restarted from its current state.
        """
        with self._lock:
            current = self._state
            if current not in (
                RuntimeState.RUNNING,
                RuntimeState.DEGRADED,
                RuntimeState.FAILED,
            ):
                raise KernelError(
                    f"Cannot restart kernel from state {current.name}.",
                    kernel_state=current.name,
                )

        _logger.info("Restarting kernel.")

        # Stop if operational
        if self._state in (RuntimeState.RUNNING, RuntimeState.DEGRADED):
            self.stop()

        # Reset to CREATED for fresh bootstrap
        with self._lock:
            self._state = RuntimeState.CREATED
            self._components = {}
            self._bootstrap_manager = None
            self._startup_manager = None
            self._shutdown_manager = None
            self._health_monitor = None
            self._recovery_attempts = 0
            self._started_at = None
            self._stopped_at = None

        self.bootstrap()

    def recover(self) -> None:
        """Attempt recovery from a FAILED or DEGRADED state.

        The kernel transitions to ``RECOVERING``, attempts to restart failed
        modules, then transitions to ``RUNNING`` or ``DEGRADED``.

        Raises
        ------
        KernelError
            If the kernel is not in a recoverable state.
        RecoveryError
            If recovery fails and max attempts are exceeded.
        """
        with self._lock:
            current = self._state
            if current not in (RuntimeState.FAILED, RuntimeState.DEGRADED):
                raise KernelError(
                    f"Cannot recover from state {current.name}; "
                    "kernel must be in FAILED or DEGRADED state.",
                    kernel_state=current.name,
                )

            if (
                self._config.enable_recovery is False
                or self._recovery_attempts >= self._config.max_recovery_attempts
            ):
                raise RecoveryError(
                    f"Recovery disabled or max attempts "
                    f"({self._config.max_recovery_attempts}) reached.",
                    kernel_state=current.name,
                    recovery_attempt=self._recovery_attempts,
                )

            self._recovery_attempts += 1
            attempt = self._recovery_attempts
            self._transition(RuntimeState.RECOVERING)

        _logger.info(
            "Kernel recovery attempt %d/%d starting.",
            attempt,
            self._config.max_recovery_attempts,
        )

        try:
            self._recover_failed_modules()

            with self._lock:
                if self._has_failed_modules():
                    self._transition(RuntimeState.DEGRADED)
                    _logger.warning("Recovery partial — kernel in DEGRADED state.")
                else:
                    self._recovery_attempts = 0  # reset on full success
                    self._transition(RuntimeState.RUNNING)
                    _logger.info("Kernel recovered to RUNNING state.")

            self._publish_event("polaris.kernel.recovered", priority="HIGH")

        except Exception as exc:
            _logger.error("Recovery attempt %d failed: %s", attempt, exc)
            with self._lock:
                self._transition(RuntimeState.FAILED)
            self._publish_event("polaris.kernel.failed", priority="CRITICAL")
            raise RecoveryError(
                f"Recovery attempt {attempt} failed: {exc}",
                kernel_state=RuntimeState.FAILED.name,
                recovery_attempt=attempt,
            ) from exc

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> RuntimeHealthReport:
        """Return the current health of the kernel.

        This method never raises — if health probing fails, it returns
        a report reflecting the current (potentially failed) state.

        Returns
        -------
        RuntimeHealthReport
            Immutable health snapshot.
        """
        with self._lock:
            current_state = self._state
            monitor = self._health_monitor

        if monitor is None:
            return RuntimeHealthReport(
                runtime_state=current_state,
                loaded_modules=(),
                running_modules=(),
                failed_modules=(),
                memory_status="unavailable",
                event_bus_status="unavailable",
                timestamp=datetime.now(timezone.utc),
            )

        try:
            return monitor.check(current_state)
        except Exception as exc:
            _logger.warning("Health check error: %s", exc)
            return RuntimeHealthReport(
                runtime_state=current_state,
                loaded_modules=(),
                running_modules=(),
                failed_modules=(),
                memory_status="error",
                event_bus_status="error",
                timestamp=datetime.now(timezone.utc),
                metadata={"health_check_error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Module management
    # ------------------------------------------------------------------

    def register_module(
        self,
        manifest: Any,
        module_path: str | None = None,
    ) -> None:
        """Register a module manifest with the kernel.

        The module is discovered and added to the loader's registry.

        Parameters
        ----------
        manifest:
            :class:`~core.loader.manifest.ModuleManifest` descriptor.
        module_path:
            Optional Python import path (e.g. ``"myproject.MyModule"``).

        Raises
        ------
        KernelError
            If the kernel has not been bootstrapped.
        """
        loader = self._get_component("module_loader")
        try:
            loader.discover(manifest, module_path=module_path)
            _logger.info("Module %r registered.", manifest.id)
        except Exception as exc:
            raise KernelError(
                f"Failed to register module {manifest.id!r}: {exc}",
                kernel_state=self._state.name,
            ) from exc

    def load_module(self, module_id: str) -> None:
        """Load a discovered module.

        Parameters
        ----------
        module_id:
            Module identifier to load.

        Raises
        ------
        KernelError
            If the kernel is not bootstrapped or loading fails.
        """
        loader = self._get_component("module_loader")
        try:
            loader.load(module_id)
            _logger.info("Module %r loaded.", module_id)
        except Exception as exc:
            raise KernelError(
                f"Failed to load module {module_id!r}: {exc}",
                kernel_state=self._state.name,
            ) from exc

    def start_module(self, module_id: str) -> None:
        """Start a loaded module.

        Parameters
        ----------
        module_id:
            Module identifier to start.

        Raises
        ------
        KernelError
            If the kernel is not running or startup fails.
        StartupError
            If the module fails to start.
        """
        startup_mgr = self._startup_manager
        if startup_mgr is None:
            raise KernelError(
                "Cannot start module — kernel not bootstrapped.",
                kernel_state=self._state.name,
            )
        startup_mgr.start_module(module_id)

    def stop_module(self, module_id: str) -> None:
        """Stop a running module.

        Parameters
        ----------
        module_id:
            Module identifier to stop.

        Raises
        ------
        KernelError
            If the kernel is not bootstrapped.
        ShutdownError
            If the module fails to stop.
        """
        shutdown_mgr = self._shutdown_manager
        if shutdown_mgr is None:
            raise KernelError(
                "Cannot stop module — kernel not bootstrapped.",
                kernel_state=self._state.name,
            )
        shutdown_mgr.stop_module(module_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_state(self, *expected: RuntimeState, operation: str) -> None:
        """Assert the kernel is in one of the expected states."""
        if self._state not in expected:
            raise KernelError(
                f"Cannot perform '{operation}' in state {self._state.name}. "
                f"Required: {[s.name for s in expected]}.",
                kernel_state=self._state.name,
            )

    def _transition(self, target: RuntimeState) -> None:
        """Perform a state transition (must be called under lock)."""
        current = self._state
        allowed = _VALID_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise KernelError(
                f"Invalid kernel transition: {current.name} → {target.name}. "
                f"Allowed: {[s.name for s in allowed] or '(terminal)'}.",
                kernel_state=current.name,
            )
        _logger.debug("Kernel state: %s → %s", current.name, target.name)
        self._state = target

    def _transition_to_failed(self, reason: str) -> None:
        """Attempt to transition to FAILED state (best-effort)."""
        with self._lock:
            try:
                current = self._state
                allowed = _VALID_TRANSITIONS.get(current, frozenset())
                if RuntimeState.FAILED in allowed:
                    _logger.error("Kernel transitioning to FAILED: %s", reason)
                    self._state = RuntimeState.FAILED
                    self._publish_event("polaris.kernel.failed", priority="CRITICAL")
            except Exception as exc:
                _logger.critical("Could not transition to FAILED: %s", exc)

    def _get_component(self, name: str) -> Any:
        """Retrieve a bootstrapped component, raising KernelError if missing."""
        with self._lock:
            component = self._components.get(name)
        if component is None:
            raise KernelError(
                f"Component {name!r} not available. Has the kernel been bootstrapped?",
                kernel_state=self._state.name,
            )
        return component

    def _has_failed_modules(self) -> bool:
        """Check if any modules are in FAILED state."""
        loader = self._components.get("module_loader")
        if loader is None:
            return False
        try:
            from core.loader.models import ModuleState
            descriptors = loader.all_descriptors()
            return any(d.state is ModuleState.FAILED for d in descriptors)
        except Exception:
            return False

    def _recover_failed_modules(self) -> None:
        """Attempt to restart failed modules."""
        loader = self._components.get("module_loader")
        if loader is None:
            return

        try:
            from core.loader.models import ModuleState
            descriptors = loader.all_descriptors()
            failed = [d for d in descriptors if d.state is ModuleState.FAILED]

            for descriptor in failed:
                module_id = descriptor.manifest.id
                _logger.info("Attempting to recover module %r.", module_id)
                try:
                    loader.stop(module_id)
                except Exception:
                    pass
                try:
                    loader.unload(module_id)
                    loader.load(module_id)
                    loader.initialize(module_id)
                    loader.start(module_id)
                    _logger.info("Module %r recovered.", module_id)
                except Exception as exc:
                    _logger.error("Failed to recover module %r: %s", module_id, exc)
        except Exception as exc:
            _logger.error("Module recovery scan failed: %s", exc)

    def _publish_event(
        self,
        event_type: str,
        priority: str = "NORMAL",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Publish a kernel lifecycle event (best-effort)."""
        event_bus = self._components.get("event_bus")
        if event_bus is None:
            return
        try:
            from core.events.event import Event, EventPriority
            p = getattr(EventPriority, priority, EventPriority.NORMAL)
            event = Event.create(
                event_type=event_type,
                source="polaris.kernel",
                payload=payload or {"kernel_state": self._state.name},
                priority=p,
            )
            event_bus.publish(event)
        except Exception as exc:
            _logger.warning("Failed to publish kernel event %r: %s", event_type, exc)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RuntimeKernel("
            f"state={self._state.name}, "
            f"env={self._config.environment!r})"
        )

    def __enter__(self) -> "RuntimeKernel":
        """Context manager entry — bootstraps and starts the kernel."""
        self.bootstrap()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit — stops the kernel."""
        try:
            if self._state not in (RuntimeState.STOPPED, RuntimeState.CREATED):
                self.stop()
        except Exception as exc:
            _logger.error("Error during context manager exit: %s", exc)
        return False


__all__ = ["RuntimeKernel"]