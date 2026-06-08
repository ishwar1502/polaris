# core/kernel/health.py
"""
POLARIS v5 Runtime Kernel — Health Monitor.

:class:`HealthMonitor` aggregates health information from all runtime
components and produces :class:`~core.kernel.models.RuntimeHealthReport`
snapshots.

It queries:
* Module Loader — which modules are loaded, running, failed
* Memory Gateway — provider availability
* Event Bus — statistics and availability
* Lifecycle Manager — module state overview
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from core.kernel.models import RuntimeHealthReport, RuntimeState

_logger = logging.getLogger(__name__)


class HealthMonitor:
    """Aggregates runtime component health into a :class:`RuntimeHealthReport`.

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

    def check(self, runtime_state: RuntimeState) -> RuntimeHealthReport:
        """Generate a health report for the current kernel state.

        Parameters
        ----------
        runtime_state:
            The kernel's current :class:`RuntimeState`.

        Returns
        -------
        RuntimeHealthReport
            Immutable health snapshot.
        """
        loaded_modules = self._get_loaded_modules()
        running_modules = self._get_running_modules()
        failed_modules = self._get_failed_modules()
        memory_status = self._check_memory_status()
        event_bus_status = self._check_event_bus_status()
        metadata = self._collect_metadata()

        return RuntimeHealthReport(
            runtime_state=runtime_state,
            loaded_modules=tuple(loaded_modules),
            running_modules=tuple(running_modules),
            failed_modules=tuple(failed_modules),
            memory_status=memory_status,
            event_bus_status=event_bus_status,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private component probes
    # ------------------------------------------------------------------

    def _get_loaded_modules(self) -> list[str]:
        """Get list of modules in LOADED or INITIALIZED state."""
        loader = self._components.get("module_loader")
        if loader is None:
            return []

        try:
            from core.loader.models import ModuleState
            descriptors = loader.all_descriptors()
            return [
                d.manifest.id
                for d in descriptors
                if d.state in (ModuleState.LOADED, ModuleState.INITIALIZED)
            ]
        except Exception as exc:
            _logger.warning("Failed to query loaded modules: %s", exc)
            return []

    def _get_running_modules(self) -> list[str]:
        """Get list of currently RUNNING modules."""
        loader = self._components.get("module_loader")
        if loader is None:
            return []

        try:
            from core.loader.models import ModuleState
            descriptors = loader.all_descriptors()
            return [
                d.manifest.id
                for d in descriptors
                if d.state is ModuleState.RUNNING
            ]
        except Exception as exc:
            _logger.warning("Failed to query running modules: %s", exc)
            return []

    def _get_failed_modules(self) -> list[str]:
        """Get list of modules in FAILED state."""
        loader = self._components.get("module_loader")
        if loader is None:
            return []

        try:
            from core.loader.models import ModuleState
            descriptors = loader.all_descriptors()
            return [
                d.manifest.id
                for d in descriptors
                if d.state is ModuleState.FAILED
            ]
        except Exception as exc:
            _logger.warning("Failed to query failed modules: %s", exc)
            return []

    def _check_memory_status(self) -> str:
        """Probe memory gateway availability."""
        memory_registry = self._components.get("memory_registry")
        if memory_registry is None:
            return "unavailable"

        try:
            provider_count = len(list(memory_registry))
            if provider_count == 0:
                return "no_providers"
            return f"ok ({provider_count} providers)"
        except Exception as exc:
            _logger.warning("Memory status check failed: %s", exc)
            return "error"

    def _check_event_bus_status(self) -> str:
        """Probe event bus availability and statistics."""
        event_bus = self._components.get("event_bus")
        if event_bus is None:
            return "unavailable"

        try:
            stats = event_bus.statistics()
            return (
                f"ok (published={stats.total_published}, "
                f"subscribers={stats.subscriber_count})"
            )
        except Exception as exc:
            _logger.warning("Event bus status check failed: %s", exc)
            return "error"

    def _collect_metadata(self) -> dict[str, Any]:
        """Collect miscellaneous diagnostic metadata."""
        metadata: dict[str, Any] = {}

        loader = self._components.get("module_loader")
        if loader is not None:
            try:
                descriptors = loader.all_descriptors()
                metadata["total_modules"] = len(descriptors)
            except Exception:
                pass

        lifecycle_manager = self._components.get("lifecycle_manager")
        if lifecycle_manager is not None:
            try:
                metadata["lifecycle_manager_modules"] = len(lifecycle_manager)
            except Exception:
                pass

        return metadata

    def __repr__(self) -> str:  # pragma: no cover
        return f"HealthMonitor(components={list(self._components.keys())})"


__all__ = ["HealthMonitor"]