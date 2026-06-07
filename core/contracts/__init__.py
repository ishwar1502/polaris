# core/contracts/__init__.py
"""Contract primitives for the POLARIS v5 runtime."""

from core.contracts.capability import (
    Capability,
    CapabilityAlreadyRegisteredError,
    CapabilityError,
    CapabilityNotFoundError,
    CapabilityRegistry,
)
from core.contracts.health import (
    HealthCheckResult,
    HealthReport,
    HealthStatus,
)
from core.contracts.lifecycle import (
    LifecycleError,
    LifecycleMachine,
    LifecycleState,
    LifecycleTransition,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
)
from core.contracts.subsystem import (
    DependencyError,
    LifecycleError,
    RegistrationError,
    RuntimeErrorBase,
    SubsystemContract,
    SubsystemMetadata,
)

__all__ = [
    # capability
    "Capability",
    "CapabilityAlreadyRegisteredError",
    "CapabilityError",
    "CapabilityNotFoundError",
    "CapabilityRegistry",
    # health
    "HealthCheckResult",
    "HealthReport",
    "HealthStatus",
    # lifecycle
    "LifecycleError",
    "LifecycleMachine",
    "LifecycleState",
    "LifecycleTransition",
    "TERMINAL_STATES",
    "VALID_TRANSITIONS",
    # subsystem
    "DependencyError",
    "RegistrationError",
    "RuntimeErrorBase",
    "SubsystemContract",
    "SubsystemMetadata",
]