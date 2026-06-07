# core/loader/loader.py
"""
Module Loader — the POLARIS v5 runtime boot mechanism.

:class:`ModuleLoader` orchestrates the complete lifecycle of every POLARIS
subsystem module: discovery, validation, dependency resolution, loading,
initialization, start, pause, resume, stop, and unloading.

Loader rules (enforced)
-----------------------
* A module **cannot start** unless it is ``LOADED``, ``INITIALIZED``, and
  all of its declared dependencies are ``RUNNING``.
* A module **cannot be unloaded** while dependent modules are still
  ``RUNNING`` (or otherwise not yet stopped/failed).
* All mutating operations are serialised through a reentrant lock
  (``threading.RLock``) so that concurrent calls from multiple threads are
  safe.

Integration
-----------
The loader integrates with the existing POLARIS runtime components:

* :mod:`core.contracts` — subsystem lifecycle enforcement
* :mod:`core.events` — publishes lifecycle change events on the event bus
* :mod:`core.registry` — registers / deregisters subsystem instances
* :mod:`core.memory` — no direct integration; memory subsystems are ordinary
  loadable modules

Usage
-----
.. code-block:: python

    from core.loader import ModuleLoader
    from core.loader.manifest import ModuleManifest

    loader = ModuleLoader()

    manifest = ModuleManifest(
        id="polaris.memory.echo",
        name="Echo Memory",
        version="1.0.0",
        description="Episodic memory subsystem.",
    )
    loader.discover(manifest, module_path="myproject.echo.EchoSubsystem")
    loader.validate()
    loader.load_all()
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Callable, Optional, Type

from core.loader.dependency import DependencyGraph
from core.loader.discovery import ModuleDiscovery
from core.loader.exceptions import (
    CircularDependencyError,
    DependencyResolutionError,
    ModuleLoaderError,
    ModuleValidationError,
)
from core.loader.manifest import ModuleDescriptor, ModuleManifest
from core.loader.models import ModuleState

_logger = logging.getLogger(__name__)


class ModuleLoader:
    """Runtime boot mechanism for POLARIS v5 modules.

    The loader owns the full module lifecycle from discovery through shutdown.
    It maintains a private :class:`~core.loader.discovery.ModuleDiscovery`
    registry and a :class:`~core.loader.dependency.DependencyGraph` that is
    rebuilt whenever new modules are discovered.

    Parameters
    ----------
    discovery:
        Optional pre-configured :class:`~core.loader.discovery.ModuleDiscovery`
        instance.  If not supplied a fresh one is created.
    event_bus:
        Optional event bus for publishing lifecycle events.  If ``None``,
        lifecycle events are silently suppressed.
    registry:
        Optional subsystem registry.  If supplied, module instances that
        implement :class:`~core.contracts.subsystem.SubsystemContract` are
        automatically registered / deregistered.

    Thread safety
    -------------
    All public methods are serialised through a single :class:`threading.RLock`.
    """

    def __init__(
        self,
        discovery: ModuleDiscovery | None = None,
        event_bus: Any | None = None,
        registry: Any | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._discovery: ModuleDiscovery = discovery or ModuleDiscovery()
        self._event_bus = event_bus
        self._registry = registry
        # Dependency graph is rebuilt lazily after each discover() call.
        self._graph: DependencyGraph | None = None
        self._graph_dirty: bool = True  # rebuild on next validate()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(
        self,
        manifest: ModuleManifest,
        *,
        module_path: str,
    ) -> ModuleDescriptor:
        """Register a module manifest for later loading.

        This is the primary entry-point for adding a module to the loader.
        The manifest is validated and a :class:`~core.loader.manifest.ModuleDescriptor`
        in :attr:`~ModuleState.DISCOVERED` state is created.

        Parameters
        ----------
        manifest:
            Fully constructed :class:`~core.loader.manifest.ModuleManifest`.
        module_path:
            Fully-qualified Python import path to the module class,
            e.g. ``"myproject.subsystems.EchoSubsystem"``.

        Returns
        -------
        ModuleDescriptor
            The newly created descriptor.

        Raises
        ------
        ModuleLoaderError
            If a module with the same ID is already registered.
        """
        with self._lock:
            descriptor = self._discovery.register_manifest(manifest, module_path)
            self._graph_dirty = True
            _logger.info(
                "Loader: discovered module %r (path=%s).",
                manifest.id,
                module_path,
            )
            return descriptor

    def discover_all(
        self,
        modules: list[tuple[ModuleManifest, str]],
    ) -> list[ModuleDescriptor]:
        """Register multiple modules in one call.

        Parameters
        ----------
        modules:
            List of ``(manifest, module_path)`` pairs.

        Returns
        -------
        list[ModuleDescriptor]
            Descriptors for all successfully registered modules.
        """
        with self._lock:
            descriptors: list[ModuleDescriptor] = []
            for manifest, module_path in modules:
                descriptor = self.discover(manifest, module_path=module_path)
                descriptors.append(descriptor)
            return descriptors

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate the dependency graph of all discovered modules.

        Checks:
        * No duplicate module IDs.
        * No missing dependency declarations.
        * No circular dependencies.

        Transitions successfully validated modules from
        :attr:`~ModuleState.DISCOVERED` → :attr:`~ModuleState.VALIDATED`.

        Raises
        ------
        DependencyResolutionError
            If any module declares a dependency that is not registered.
        CircularDependencyError
            If the dependency graph contains a cycle.
        ModuleValidationError
            If duplicate module IDs are found.
        """
        with self._lock:
            manifests = self._discovery.all_manifests()
            if not manifests:
                _logger.debug("Loader.validate: no modules to validate.")
                return

            graph = DependencyGraph(manifests)
            graph.validate()
            self._graph = graph
            self._graph_dirty = False

            # Transition all DISCOVERED → VALIDATED
            for descriptor in self._discovery.all_descriptors():
                if descriptor.state is ModuleState.DISCOVERED:
                    descriptor.state = ModuleState.VALIDATED
                    _logger.debug("Module %r: DISCOVERED → VALIDATED.", descriptor.id)

            _logger.info(
                "Loader.validate: %d modules validated.", len(manifests)
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, module_id: str) -> ModuleDescriptor:
        """Import the Python class for a single module.

        The module must be in :attr:`~ModuleState.VALIDATED` state.
        On success the descriptor transitions to :attr:`~ModuleState.LOADED`.

        Parameters
        ----------
        module_id:
            ID of the module to load.

        Returns
        -------
        ModuleDescriptor
            The updated descriptor.

        Raises
        ------
        ModuleLoaderError
            If the module is not in ``VALIDATED`` state, or if the import
            fails.
        """
        with self._lock:
            descriptor = self._get_descriptor(module_id)
            if descriptor.state is ModuleState.LOADED:
                _logger.debug("Module %r already loaded; skipping.", module_id)
                return descriptor
            if descriptor.state is not ModuleState.VALIDATED:
                raise ModuleLoaderError(
                    f"Cannot load module {module_id!r}: "
                    f"expected VALIDATED state, got {descriptor.state.name}.",
                    module_id=module_id,
                )
            self._import_module_class(descriptor)
            descriptor.state = ModuleState.LOADED
            _logger.info("Module %r: VALIDATED → LOADED.", module_id)
            return descriptor

    def _import_module_class(self, descriptor: ModuleDescriptor) -> None:
        """Resolve ``module_path`` to a Python class and store it."""
        module_path = descriptor.module_path
        # module_path may be "some.package.ClassName"
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            import_path, class_name = parts
        else:
            # No dot — treat as a module without a specific class.
            import_path, class_name = module_path, None

        try:
            mod = importlib.import_module(import_path)
        except ImportError as exc:
            descriptor.state = ModuleState.FAILED
            descriptor.error = exc
            raise ModuleLoaderError(
                f"Failed to import module {descriptor.id!r} "
                f"from {import_path!r}: {exc}",
                module_id=descriptor.id,
            ) from exc

        if class_name:
            try:
                cls = getattr(mod, class_name)
            except AttributeError as exc:
                descriptor.state = ModuleState.FAILED
                descriptor.error = exc
                raise ModuleLoaderError(
                    f"Module {descriptor.id!r}: class {class_name!r} "
                    f"not found in {import_path!r}.",
                    module_id=descriptor.id,
                ) from exc
            descriptor.module_class = cls
        else:
            descriptor.module_class = mod  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, module_id: str) -> ModuleDescriptor:
        """Instantiate and initialize a single module.

        The module must be in :attr:`~ModuleState.LOADED` state.
        On success transitions to :attr:`~ModuleState.INITIALIZED`.

        Parameters
        ----------
        module_id:
            ID of the module to initialize.

        Returns
        -------
        ModuleDescriptor
            The updated descriptor.

        Raises
        ------
        ModuleLoaderError
            If the module is not ``LOADED``, or if instantiation /
            initialization raises an exception.
        """
        with self._lock:
            descriptor = self._get_descriptor(module_id)
            if descriptor.state is ModuleState.INITIALIZED:
                _logger.debug(
                    "Module %r already initialized; skipping.", module_id
                )
                return descriptor
            if descriptor.state is not ModuleState.LOADED:
                raise ModuleLoaderError(
                    f"Cannot initialize module {module_id!r}: "
                    f"expected LOADED state, got {descriptor.state.name}.",
                    module_id=module_id,
                )
            self._instantiate_and_init(descriptor)
            return descriptor

    def _instantiate_and_init(self, descriptor: ModuleDescriptor) -> None:
        """Instantiate the module class and call initialize() if present."""
        cls = descriptor.module_class
        if cls is None:
            raise ModuleLoaderError(
                f"Module {descriptor.id!r}: module_class is None; "
                "load() must be called before initialize().",
                module_id=descriptor.id,
            )
        try:
            instance = cls()
        except Exception as exc:
            descriptor.state = ModuleState.FAILED
            descriptor.error = exc
            raise ModuleLoaderError(
                f"Failed to instantiate {descriptor.id!r}: {exc}",
                module_id=descriptor.id,
            ) from exc

        # Call initialize() if the instance has it (SubsystemContract API).
        if hasattr(instance, "initialize") and callable(instance.initialize):
            try:
                instance.initialize()
            except Exception as exc:
                descriptor.state = ModuleState.FAILED
                descriptor.error = exc
                raise ModuleLoaderError(
                    f"initialize() failed for {descriptor.id!r}: {exc}",
                    module_id=descriptor.id,
                ) from exc

        descriptor.instance = instance
        descriptor.state = ModuleState.INITIALIZED

        # Optionally register in the subsystem registry.
        if self._registry is not None and hasattr(instance, "metadata"):
            try:
                self._registry.register(instance)
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "Could not register %r in subsystem registry: %s",
                    descriptor.id,
                    exc,
                )

        _logger.info("Module %r: LOADED → INITIALIZED.", descriptor.id)

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self, module_id: str) -> ModuleDescriptor:
        """Start a single module.

        Pre-conditions (all must hold):

        1. Module is in :attr:`~ModuleState.INITIALIZED` state.
        2. All declared dependencies are currently :attr:`~ModuleState.RUNNING`.

        On success transitions to :attr:`~ModuleState.RUNNING`.

        Parameters
        ----------
        module_id:
            ID of the module to start.

        Returns
        -------
        ModuleDescriptor
            The updated descriptor.

        Raises
        ------
        DependencyResolutionError
            If any declared dependency is not ``RUNNING``.
        ModuleLoaderError
            If the module is not in ``INITIALIZED`` state, or if
            :meth:`start` raises an exception.
        """
        with self._lock:
            descriptor = self._get_descriptor(module_id)
            if descriptor.state is ModuleState.RUNNING:
                _logger.debug(
                    "Module %r already running; skipping.", module_id
                )
                return descriptor
            if descriptor.state is not ModuleState.INITIALIZED:
                raise ModuleLoaderError(
                    f"Cannot start module {module_id!r}: "
                    f"expected INITIALIZED state, got {descriptor.state.name}.",
                    module_id=module_id,
                )
            self._assert_dependencies_running(descriptor)
            self._do_start(descriptor)
            return descriptor

    def _assert_dependencies_running(self, descriptor: ModuleDescriptor) -> None:
        """Raise DependencyResolutionError if any dependency is not RUNNING."""
        not_running: list[str] = []
        for dep_id in descriptor.manifest.dependencies:
            dep_desc = self._discovery.get_descriptor(dep_id)
            if dep_desc.state is not ModuleState.RUNNING:
                not_running.append(dep_id)
        if not_running:
            raise DependencyResolutionError(
                f"Module {descriptor.id!r} cannot start: "
                f"dependencies not RUNNING: {not_running}.",
                module_id=descriptor.id,
                missing=not_running,
            )

    def _do_start(self, descriptor: ModuleDescriptor) -> None:
        """Call start() on the instance and transition state."""
        instance = descriptor.instance
        if instance is not None and hasattr(instance, "start") and callable(instance.start):
            try:
                instance.start()
            except Exception as exc:
                descriptor.state = ModuleState.FAILED
                descriptor.error = exc
                raise ModuleLoaderError(
                    f"start() failed for {descriptor.id!r}: {exc}",
                    module_id=descriptor.id,
                ) from exc

        descriptor.state = ModuleState.RUNNING
        self._publish_event("module.started", descriptor)
        _logger.info("Module %r: INITIALIZED → RUNNING.", descriptor.id)

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self, module_id: str) -> ModuleDescriptor:
        """Stop a single module.

        The module must be in an operational state
        (:attr:`~ModuleState.RUNNING` or :attr:`~ModuleState.PAUSED`).
        On success transitions to :attr:`~ModuleState.STOPPED`.

        Parameters
        ----------
        module_id:
            ID of the module to stop.

        Returns
        -------
        ModuleDescriptor
            The updated descriptor.

        Raises
        ------
        ModuleLoaderError
            If the module is not in a stoppable state.
        """
        with self._lock:
            descriptor = self._get_descriptor(module_id)
            if descriptor.state is ModuleState.STOPPED:
                _logger.debug(
                    "Module %r already stopped; skipping.", module_id
                )
                return descriptor
            if descriptor.state not in (
                ModuleState.RUNNING,
                ModuleState.PAUSED,
                ModuleState.INITIALIZED,
                ModuleState.LOADED,
            ):
                raise ModuleLoaderError(
                    f"Cannot stop module {module_id!r}: "
                    f"current state {descriptor.state.name} is not stoppable.",
                    module_id=module_id,
                )
            self._do_stop(descriptor)
            return descriptor

    def _do_stop(self, descriptor: ModuleDescriptor) -> None:
        """Call stop() on the instance and transition state."""
        instance = descriptor.instance
        if instance is not None and hasattr(instance, "stop") and callable(instance.stop):
            try:
                instance.stop()
            except Exception as exc:
                descriptor.state = ModuleState.FAILED
                descriptor.error = exc
                _logger.exception(
                    "Module %r: stop() raised; state → FAILED.", descriptor.id
                )
                raise ModuleLoaderError(
                    f"stop() failed for {descriptor.id!r}: {exc}",
                    module_id=descriptor.id,
                ) from exc

        descriptor.state = ModuleState.STOPPED
        self._publish_event("module.stopped", descriptor)
        _logger.info("Module %r: → STOPPED.", descriptor.id)

    # ------------------------------------------------------------------
    # Unloading
    # ------------------------------------------------------------------

    def unload(self, module_id: str) -> None:
        """Unload a module, releasing its class and instance references.

        Pre-condition: no dependent modules may be ``RUNNING`` at the time
        this is called.

        Parameters
        ----------
        module_id:
            ID of the module to unload.

        Raises
        ------
        ModuleLoaderError
            If dependent modules are still running, or if the module is not
            in an unloadable state (i.e. it must be ``STOPPED``,
            ``FAILED``, or ``VALIDATED``/``DISCOVERED`` — i.e. never
            successfully initialized).
        """
        with self._lock:
            descriptor = self._get_descriptor(module_id)
            # Check that no dependent modules are running.
            self._assert_no_running_dependents(module_id)

            # Must not be in an operational state.
            if descriptor.state in (ModuleState.RUNNING, ModuleState.PAUSED):
                raise ModuleLoaderError(
                    f"Cannot unload module {module_id!r}: "
                    f"module is still {descriptor.state.name}. "
                    "Stop it first.",
                    module_id=module_id,
                )

            # Deregister from the subsystem registry if applicable.
            if (
                self._registry is not None
                and descriptor.instance is not None
                and hasattr(descriptor.instance, "metadata")
            ):
                try:
                    self._registry.unregister(module_id)
                except Exception as exc:  # noqa: BLE001
                    _logger.warning(
                        "Could not deregister %r from registry: %s",
                        module_id,
                        exc,
                    )

            # Release references.
            descriptor.module_class = None
            descriptor.instance = None
            descriptor.error = None
            descriptor.state = ModuleState.DISCOVERED  # reset to ground state

            _logger.info("Module %r: unloaded; state reset to DISCOVERED.", module_id)

    def _assert_no_running_dependents(self, module_id: str) -> None:
        """Raise ModuleLoaderError if any dependent is still RUNNING."""
        if self._graph is None:
            return
        if module_id not in self._graph:
            return
        running_dependents = [
            dep_id
            for dep_id in self._graph.dependents_of(module_id)
            if self._discovery.is_registered(dep_id)
            and self._discovery.get_descriptor(dep_id).state is ModuleState.RUNNING
        ]
        if running_dependents:
            raise ModuleLoaderError(
                f"Cannot unload module {module_id!r}: "
                f"dependent modules are still RUNNING: {running_dependents}.",
                module_id=module_id,
            )

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Discover → validate → load → initialize → start all registered modules.

        Modules are processed in topologically sorted order so that
        dependencies are fully started before their dependents.

        Raises
        ------
        DependencyResolutionError
            If dependency resolution fails.
        CircularDependencyError
            If a cycle is detected.
        ModuleLoaderError
            If any individual module operation fails.
        """
        with self._lock:
            # Re-validate to ensure graph is current.
            self.validate()
            if self._graph is None:
                return
            order = self._graph.topological_order()
            _logger.info(
                "load_all: processing %d modules in order %s.",
                len(order),
                order,
            )
            for module_id in order:
                descriptor = self._get_descriptor(module_id)
                if descriptor.state is ModuleState.VALIDATED:
                    self.load(module_id)
                if descriptor.state is ModuleState.LOADED:
                    self.initialize(module_id)
                if descriptor.state is ModuleState.INITIALIZED:
                    self.start(module_id)

    def shutdown_all(self) -> None:
        """Stop all RUNNING modules in reverse topological order.

        Modules are stopped in reverse dependency order so that dependents
        are stopped before their dependencies.

        Stop failures are logged but do not abort the shutdown sequence.
        """
        with self._lock:
            if self._graph is None:
                # No graph — stop all running modules in arbitrary order.
                order = [d.id for d in self._discovery.all_descriptors()]
            else:
                try:
                    order = list(reversed(self._graph.topological_order()))
                except Exception:  # noqa: BLE001
                    order = [d.id for d in self._discovery.all_descriptors()]

            _logger.info(
                "shutdown_all: stopping up to %d modules.", len(order)
            )
            for module_id in order:
                if not self._discovery.is_registered(module_id):
                    continue
                descriptor = self._get_descriptor(module_id)
                if descriptor.state in (ModuleState.RUNNING, ModuleState.PAUSED):
                    try:
                        self._do_stop(descriptor)
                    except Exception as exc:  # noqa: BLE001
                        _logger.error(
                            "shutdown_all: failed to stop %r: %s",
                            module_id,
                            exc,
                        )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_descriptor(self, module_id: str) -> ModuleDescriptor:
        """Return the descriptor for *module_id*.

        Raises
        ------
        ModuleLoaderError
            If the module is not registered.
        """
        with self._lock:
            return self._get_descriptor(module_id)

    def all_descriptors(self) -> list[ModuleDescriptor]:
        """Return a snapshot of all registered descriptors (sorted by ID)."""
        with self._lock:
            return self._discovery.all_descriptors()

    def descriptors_in_state(self, state: ModuleState) -> list[ModuleDescriptor]:
        """Return all descriptors currently in *state*.

        Parameters
        ----------
        state:
            The target :class:`~core.loader.models.ModuleState`.

        Returns
        -------
        list[ModuleDescriptor]
            Matching descriptors sorted by module ID.
        """
        with self._lock:
            return [
                d for d in self._discovery.all_descriptors()
                if d.state is state
            ]

    def is_running(self, module_id: str) -> bool:
        """Return ``True`` if *module_id* is currently ``RUNNING``."""
        with self._lock:
            try:
                return self._get_descriptor(module_id).state is ModuleState.RUNNING
            except ModuleLoaderError:
                return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_descriptor(self, module_id: str) -> ModuleDescriptor:
        """Retrieve a descriptor, raising ModuleLoaderError if absent."""
        try:
            return self._discovery.get_descriptor(module_id)
        except Exception as exc:
            raise ModuleLoaderError(
                f"Module {module_id!r} is not registered with the loader.",
                module_id=module_id,
            ) from exc

    def _publish_event(self, event_type: str, descriptor: ModuleDescriptor) -> None:
        """Publish a lifecycle event on the event bus (best-effort)."""
        if self._event_bus is None:
            return
        try:
            from core.events.event import Event  # local import to avoid circularity
            event = Event.create(
                event_type=f"polaris.loader.{event_type}",
                source="polaris.core.loader",
                payload={
                    "module_id": descriptor.id,
                    "module_name": descriptor.name,
                    "module_version": descriptor.version,
                    "state": descriptor.state.name,
                },
            )
            self._event_bus.publish(event)
        except Exception as exc:  # noqa: BLE001
            _logger.debug(
                "Failed to publish event %r for module %r: %s",
                event_type,
                descriptor.id,
                exc,
            )


__all__ = ["ModuleLoader"]