# core/kernel/shutdown.py
"""
POLARIS v5 Runtime Kernel — Shutdown Manager.

:class:`ShutdownManager` is responsible for an orderly kernel shutdown:

1. Publish KernelStopping event
2. Stop all running modules (in reverse dependency order)
3. Unload all modules
4. Release resources
5. Publish KernelStopped event

Shutdown is best-effort: errors during individual module shutdown are
logged but do not prevent the sequence from continuing.
"""

from __future__ import annotations

import logging
from typing import Any

from core.kernel.exceptions import ShutdownError

_logger = logging.getLogger(__name__)


class ShutdownManager:
    """Manages the kernel shutdown sequence.

    Parameters
    ----------
    components:
        Dictionary of bootstrapped runtime components.
    """

    def __init__(self, components: dict[str, Any]) -> None:
        self._components = components

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Execute the kernel shutdown sequence.

        Performs a graceful, ordered shutdown:
        1. Publish pre-shutdown event.
        2. Stop all running modules.
        3. Unload all modules.
        4. Release resources.
        5. Publish post-shutdown event.

        Raises
        ------
        ShutdownError
            If shutdown fails in a way that leaves the system in an
            inconsistent state.  Individual module failures are logged
            but do not raise.
        """
        _logger.info("Kernel shutdown sequence starting.")

        self._publish_kernel_stopping()

        errors: list[str] = []

        try:
            self._stop_modules(errors)
        except Exception as exc:
            _logger.error("Critical error during module stop phase: %s", exc)
            errors.append(str(exc))

        try:
            self._unload_modules(errors)
        except Exception as exc:
            _logger.error("Critical error during module unload phase: %s", exc)
            errors.append(str(exc))

        self._release_resources()
        self._publish_kernel_stopped()

        if errors:
            _logger.warning(
                "Shutdown completed with %d error(s): %s", len(errors), errors
            )

        _logger.info("Kernel shutdown sequence complete.")

    def stop_module(self, module_id: str) -> None:
        """Stop a single module by id.

        Parameters
        ----------
        module_id:
            The module to stop.

        Raises
        ------
        ShutdownError
            If the module cannot be stopped.
        """
        loader = self._components.get("module_loader")
        if loader is None:
            raise ShutdownError(
                "Cannot stop module — module_loader not available.",
                failed_module=module_id,
            )

        try:
            loader.stop(module_id)
            _logger.info("Module %r stopped.", module_id)
        except Exception as exc:
            raise ShutdownError(
                f"Failed to stop module {module_id!r}: {exc}",
                failed_module=module_id,
            ) from exc

    def unload_module(self, module_id: str) -> None:
        """Unload a single module by id.

        Parameters
        ----------
        module_id:
            The module to unload.

        Raises
        ------
        ShutdownError
            If the module cannot be unloaded.
        """
        loader = self._components.get("module_loader")
        if loader is None:
            raise ShutdownError(
                "Cannot unload module — module_loader not available.",
                failed_module=module_id,
            )

        try:
            loader.unload(module_id)
            _logger.info("Module %r unloaded.", module_id)
        except Exception as exc:
            raise ShutdownError(
                f"Failed to unload module {module_id!r}: {exc}",
                failed_module=module_id,
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stop_modules(self, errors: list[str]) -> None:
        """Stop all running modules."""
        loader = self._components.get("module_loader")
        if loader is None:
            _logger.warning("No module_loader; skipping module stop.")
            return

        try:
            loader.shutdown_all()
            _logger.info("All modules stopped via shutdown_all().")
        except Exception as exc:
            msg = f"shutdown_all() failed: {exc}"
            _logger.error(msg)
            errors.append(msg)
            # Attempt individual stops as fallback
            self._stop_modules_individually(loader, errors)

    def _stop_modules_individually(self, loader: Any, errors: list[str]) -> None:
        """Attempt to stop modules individually as a fallback."""
        try:
            from core.loader.models import ModuleState, STOPPABLE_STATES
            descriptors = loader.all_descriptors()
            for descriptor in reversed(descriptors):
                if descriptor.state in STOPPABLE_STATES:
                    try:
                        loader.stop(descriptor.manifest.id)
                    except Exception as exc:
                        msg = f"Failed to stop module {descriptor.manifest.id!r}: {exc}"
                        _logger.error(msg)
                        errors.append(msg)
        except Exception as exc:
            errors.append(f"Individual stop fallback failed: {exc}")

    def _unload_modules(self, errors: list[str]) -> None:
        """Unload all modules."""
        loader = self._components.get("module_loader")
        if loader is None:
            return

        try:
            from core.loader.models import ModuleState
            descriptors = loader.all_descriptors()
            for descriptor in reversed(descriptors):
                if descriptor.state not in (
                    ModuleState.FAILED,
                ):
                    try:
                        loader.unload(descriptor.manifest.id)
                    except Exception as exc:
                        msg = f"Failed to unload module {descriptor.manifest.id!r}: {exc}"
                        _logger.warning(msg)
                        errors.append(msg)
        except Exception as exc:
            errors.append(f"Unload phase failed: {exc}")

    def _release_resources(self) -> None:
        """Release all runtime resources (best-effort)."""
        _logger.debug("Releasing runtime resources.")
        # Event bus does not require explicit teardown in current implementation.
        # Memory registry providers are managed externally.
        # This hook exists for future resource cleanup needs.

    def _publish_kernel_stopping(self) -> None:
        """Publish a pre-shutdown event."""
        self._publish_event("polaris.kernel.stopping")

    def _publish_kernel_stopped(self) -> None:
        """Publish the KernelStopped event."""
        self._publish_event("polaris.kernel.stopped")

    def _publish_event(self, event_type: str) -> None:
        """Publish a kernel lifecycle event."""
        event_bus = self._components.get("event_bus")
        if event_bus is None:
            return
        try:
            from core.events.event import Event, EventPriority
            event = Event.create(
                event_type=event_type,
                source="polaris.kernel",
                payload={"phase": "shutdown"},
                priority=EventPriority.HIGH,
            )
            event_bus.publish(event)
        except Exception as exc:
            _logger.warning("Failed to publish event %r: %s", event_type, exc)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ShutdownManager(components={list(self._components.keys())})"


__all__ = ["ShutdownManager"]