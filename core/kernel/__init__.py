# core/kernel/__init__.py
"""
POLARIS v5 Runtime Kernel — Phase 6.

The Runtime Kernel is the central orchestrator of the POLARIS v5 cognitive
architecture.  It coordinates all runtime components:

* Contracts & Registry
* Event Bus
* Memory Gateway
* Module Loader
* Lifecycle Manager

Public surface
--------------
All consumers should import exclusively from this package.

.. code-block:: python

    from core.kernel import RuntimeKernel, KernelConfiguration, RuntimeState

    config = KernelConfiguration(environment="production")
    kernel = RuntimeKernel(config)
    kernel.bootstrap()
    kernel.start()
    report = kernel.health()
"""

from __future__ import annotations

from core.kernel.bootstrap import BootstrapManager
from core.kernel.exceptions import (
    BootstrapError,
    KernelError,
    RecoveryError,
    ShutdownError,
    StartupError,
)
from core.kernel.health import HealthMonitor
from core.kernel.kernel import RuntimeKernel
from core.kernel.models import KernelConfiguration, RuntimeHealthReport, RuntimeState
from core.kernel.shutdown import ShutdownManager
from core.kernel.startup import StartupManager

__all__ = [
    # Kernel — primary public API
    "RuntimeKernel",
    # Managers
    "BootstrapManager",
    "StartupManager",
    "ShutdownManager",
    "HealthMonitor",
    # Models
    "KernelConfiguration",
    "RuntimeHealthReport",
    "RuntimeState",
    # Exceptions
    "KernelError",
    "BootstrapError",
    "StartupError",
    "ShutdownError",
    "RecoveryError",
]