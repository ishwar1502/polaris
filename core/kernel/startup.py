# core/kernel/startup.py
"""
POLARIS v5 Runtime Kernel — Startup Manager.

:class:`StartupManager` is responsible for starting the runtime after
bootstrap has completed:

1. Publish KernelStarted event
2. Start all registered modules via the Module Loader
3. Verify that required modules transitioned to RUNNING
4. Update kernel state

The startup manager is non-destructive — if startup fails, it records
the failure and allows the kernel to attempt recovery.
"""

from __future__ import annotations

import logging
from typing import Any

from core.kernel.exceptions import StartupError

_logger = logging.getLogger(__name__)


class StartupManager:
    """Manages the kernel startup sequence.

    Parameters
    ----------
    components:
        Dictionary of bootstrapped runtime components (from BootstrapManager).
    """

    def __init__(self, components: dict[str, Any]) -> None:
        self._components = components

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Execute the kernel startup sequence.

        Steps:
        1. Verify all components are available.
        2. Load all discovered modules.
        3. Start all loaded modules.
        4. Verify startup success.
        5. Publish KernelStarted event.

        Raises
        ------
        StartupError
            If startup fails at any step.
        """
        _logger.info("Kernel startup sequence starting.")

        try:
            self._verify_components()
        except StartupError:
            raise
        except Exception as exc:
            raise StartupError(
                f"Component verification failed during startup: {exc}",
            ) from exc

        try:
            self._start_modules()
        except StartupError:
            raise
        except Exception as exc:
            raise StartupError(
                f"Module startup failed: {exc}",
            ) from exc

        self._publish_kernel_started()
        _logger.info("Kernel startup sequence complete.")

    def start_module(self, module_id: str) -> None:
        """Start a single module by id.

        Parameters
        ----------
        module_id:
            The module to start.

        Raises
        ------
        StartupError
            If the module cannot be started.
        """
        loader = self._components.get("module_loader")
        if loader is None:
            raise StartupError(
                "Cannot start module — module_loader not available.",
                failed_module=module_id,
            )

        try:
            loader.start(module_id)
            _logger.info("Module %r started.", module_id)
        except Exception as exc:
            raise StartupError(
                f"Failed to start module {module_id!r}: {exc}",
                failed_module=module_id,
            ) from exc

    def verify_startup_success(self) -> bool:
        """Verify that the startup sequence succeeded.

        Returns
        -------
        bool
            ``True`` if all modules that should be running are running.
        """
        loader = self._components.get("module_loader")
        if loader is None:
            return False

        try:
            all_descriptors = loader.all_descriptors()
            from core.loader.models import ModuleState
            failed = [
                d for d in all_descriptors
                if d.state is ModuleState.FAILED
            ]
            return len(failed) == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _verify_components(self) -> None:
        """Verify that all required components are present."""
        required = ["event_bus", "module_loader", "lifecycle_manager"]
        missing = [name for name in required if self._components.get(name) is None]
        if missing:
            raise StartupError(
                f"Missing required components for startup: {missing}",
            )

    def _start_modules(self) -> None:
        """Start all modules via the module loader."""
        loader = self._components.get("module_loader")
        if loader is None:
            _logger.warning("No module_loader available; skipping module startup.")
            return

        try:
            all_descriptors = loader.all_descriptors()
            _logger.info(
                "Starting %d discovered modules.", len(all_descriptors)
            )

            from core.loader.models import ModuleState

            for descriptor in all_descriptors:
                if descriptor.state is ModuleState.INITIALIZED:
                    try:
                        loader.start(descriptor.manifest.id)
                        _logger.debug("Module %r started.", descriptor.manifest.id)
                    except Exception as exc:
                        _logger.error(
                            "Failed to start module %r: %s",
                            descriptor.manifest.id,
                            exc,
                        )
                        raise StartupError(
                            f"Module {descriptor.manifest.id!r} failed to start: {exc}",
                            failed_module=descriptor.manifest.id,
                        ) from exc
        except StartupError:
            raise
        except Exception as exc:
            raise StartupError(
                f"Error during module startup: {exc}",
            ) from exc

    def _publish_kernel_started(self) -> None:
        """Publish the KernelStarted event to the event bus."""
        event_bus = self._components.get("event_bus")
        if event_bus is None:
            return

        try:
            from core.events.event import Event, EventPriority
            event = Event.create(
                event_type="polaris.kernel.started",
                source="polaris.kernel",
                payload={"environment": "runtime"},
                priority=EventPriority.HIGH,
            )
            event_bus.publish(event)
            _logger.debug("KernelStarted event published.")
        except Exception as exc:
            _logger.warning("Failed to publish KernelStarted event: %s", exc)

    def __repr__(self) -> str:  # pragma: no cover
        return f"StartupManager(components={list(self._components.keys())})"


__all__ = ["StartupManager"]