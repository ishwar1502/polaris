# core/kernel/bootstrap.py
"""
POLARIS v5 Runtime Kernel — Bootstrap Manager.

:class:`BootstrapManager` is responsible for initializing all runtime
components in the correct dependency order:

1. Event Bus
2. Memory Registry + Gateway
3. Subsystem Registry
4. Module Loader
5. Lifecycle Manager

Bootstrap validates that all required components are available and wires
them together before the kernel can proceed to startup.
"""

from __future__ import annotations

import logging
from typing import Any

from core.kernel.exceptions import BootstrapError
from core.kernel.models import KernelConfiguration

_logger = logging.getLogger(__name__)


class BootstrapManager:
    """Initializes and wires all POLARIS v5 runtime components.

    :class:`BootstrapManager` is owned by :class:`~core.kernel.kernel.RuntimeKernel`
    and called once during the ``BOOTSTRAPPING`` phase.  It validates that all
    dependencies are importable and returns a fully wired component graph.

    Parameters
    ----------
    config:
        The kernel configuration governing bootstrap behavior.
    """

    def __init__(self, config: KernelConfiguration) -> None:
        self._config = config
        self._components: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bootstrap(self) -> dict[str, Any]:
        """Run the full bootstrap sequence.

        Returns a dictionary of initialized runtime components keyed by
        component name:

        * ``"event_bus"`` — :class:`~core.events.bus.EventBus`
        * ``"memory_registry"`` — :class:`~core.memory.registry.MemoryRegistry`
        * ``"memory_gateway"`` — :class:`~core.memory.gateway.MemoryGateway`
        * ``"subsystem_registry"`` — :class:`~core.registry.registry.SubsystemRegistry`
        * ``"module_loader"`` — :class:`~core.loader.loader.ModuleLoader`
        * ``"lifecycle_manager"`` — :class:`~core.lifecycle.manager.LifecycleManager`

        Returns
        -------
        dict[str, Any]
            Wired component graph.

        Raises
        ------
        BootstrapError
            If any component fails to initialize.
        """
        _logger.info(
            "Bootstrap starting [environment=%s].", self._config.environment
        )

        try:
            self._init_event_bus()
        except Exception as exc:
            raise BootstrapError(
                f"Failed to initialize Event Bus: {exc}",
                failed_component="event_bus",
            ) from exc

        try:
            self._init_memory()
        except Exception as exc:
            raise BootstrapError(
                f"Failed to initialize Memory Gateway: {exc}",
                failed_component="memory_gateway",
            ) from exc

        try:
            self._init_registry()
        except Exception as exc:
            raise BootstrapError(
                f"Failed to initialize Subsystem Registry: {exc}",
                failed_component="subsystem_registry",
            ) from exc

        try:
            self._init_loader()
        except Exception as exc:
            raise BootstrapError(
                f"Failed to initialize Module Loader: {exc}",
                failed_component="module_loader",
            ) from exc

        try:
            self._init_lifecycle_manager()
        except Exception as exc:
            raise BootstrapError(
                f"Failed to initialize Lifecycle Manager: {exc}",
                failed_component="lifecycle_manager",
            ) from exc

        self._validate_dependencies()

        _logger.info("Bootstrap complete — %d components initialized.", len(self._components))
        return dict(self._components)

    def validate_dependencies(self) -> None:
        """Validate that all required components are present and wired.

        Raises
        ------
        BootstrapError
            If any required component is missing.
        """
        self._validate_dependencies()

    # ------------------------------------------------------------------
    # Private initialization steps
    # ------------------------------------------------------------------

    def _init_event_bus(self) -> None:
        """Initialize the Event Bus."""
        from core.events.bus import EventBus
        event_bus = EventBus()
        self._components["event_bus"] = event_bus
        _logger.debug("Event Bus initialized.")

    def _init_memory(self) -> None:
        """Initialize the Memory Registry and Gateway."""
        from core.memory.registry import MemoryRegistry
        from core.memory.gateway import MemoryGateway

        memory_registry = MemoryRegistry()
        memory_gateway = MemoryGateway(memory_registry)
        self._components["memory_registry"] = memory_registry
        self._components["memory_gateway"] = memory_gateway
        _logger.debug("Memory Gateway initialized.")

    def _init_registry(self) -> None:
        """Initialize the Subsystem Registry."""
        from core.registry.registry import SubsystemRegistry
        subsystem_registry = SubsystemRegistry()
        self._components["subsystem_registry"] = subsystem_registry
        _logger.debug("Subsystem Registry initialized.")

    def _init_loader(self) -> None:
        """Initialize the Module Loader with event bus and registry."""
        from core.loader.loader import ModuleLoader
        from core.loader.discovery import ModuleDiscovery

        event_bus = self._components.get("event_bus")
        subsystem_registry = self._components.get("subsystem_registry")

        discovery = ModuleDiscovery()
        module_loader = ModuleLoader(
            discovery=discovery,
            event_bus=event_bus,
            registry=subsystem_registry,
        )
        self._components["module_loader"] = module_loader
        _logger.debug("Module Loader initialized.")

    def _init_lifecycle_manager(self) -> None:
        """Initialize the Lifecycle Manager with event bus."""
        from core.lifecycle.manager import LifecycleManager

        event_bus = self._components.get("event_bus")
        lifecycle_manager = LifecycleManager(event_bus=event_bus)
        self._components["lifecycle_manager"] = lifecycle_manager
        _logger.debug("Lifecycle Manager initialized.")

    def _validate_dependencies(self) -> None:
        """Validate all required components exist."""
        required = [
            "event_bus",
            "memory_registry",
            "memory_gateway",
            "subsystem_registry",
            "module_loader",
            "lifecycle_manager",
        ]
        missing = [name for name in required if name not in self._components]
        if missing:
            raise BootstrapError(
                f"Bootstrap incomplete — missing components: {missing}",
                failed_component=missing[0] if missing else None,
            )

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    def get_component(self, name: str) -> Any:
        """Retrieve a bootstrapped component by name.

        Parameters
        ----------
        name:
            Component name (e.g. ``"event_bus"``).

        Returns
        -------
        Any
            The component instance.

        Raises
        ------
        KeyError
            If the named component does not exist.
        """
        if name not in self._components:
            raise KeyError(f"No bootstrapped component named {name!r}.")
        return self._components[name]

    @property
    def components(self) -> dict[str, Any]:
        """Read-only view of all bootstrapped components."""
        return dict(self._components)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BootstrapManager("
            f"environment={self._config.environment!r}, "
            f"components={list(self._components.keys())})"
        )


__all__ = ["BootstrapManager"]