# core/loader/__init__.py
"""
POLARIS v5 Module Loader — Phase 4.

The module loader is the runtime boot mechanism responsible for discovering,
validating, loading, initializing, starting, stopping, and unloading POLARIS
subsystem modules.

Public surface
--------------
All consumers should import from this package rather than from internal
modules directly.

.. code-block:: python

    from core.loader import (
        ModuleLoader,
        ModuleManifest,
        ModuleDescriptor,
        ModuleState,
        DependencyGraph,
        ModuleDiscovery,
    )
"""

from __future__ import annotations

from core.loader.dependency import DependencyGraph
from core.loader.discovery import ModuleDiscovery
from core.loader.exceptions import (
    CircularDependencyError,
    DependencyResolutionError,
    ModuleDiscoveryError,
    ModuleLoaderError,
    ModuleValidationError,
)
from core.loader.loader import ModuleLoader
from core.loader.manifest import ModuleDescriptor, ModuleManifest
from core.loader.models import (
    ModuleState,
    OPERATIONAL_STATES,
    TERMINAL_STATES,
    STOPPABLE_STATES,
)

__all__ = [
    # Core loader
    "ModuleLoader",
    # Discovery
    "ModuleDiscovery",
    # Dependency graph
    "DependencyGraph",
    # Manifest / descriptor
    "ModuleManifest",
    "ModuleDescriptor",
    # States
    "ModuleState",
    "OPERATIONAL_STATES",
    "TERMINAL_STATES",
    "STOPPABLE_STATES",
    # Exceptions
    "ModuleLoaderError",
    "ModuleDiscoveryError",
    "DependencyResolutionError",
    "CircularDependencyError",
    "ModuleValidationError",
]