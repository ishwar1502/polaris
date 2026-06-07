# core/memory/__init__.py
"""
POLARIS v5 Memory Gateway — Phase 3.

Public surface
--------------
All subsystems should import exclusively from this package (or from
``core.memory.gateway``) rather than from internal modules.

Typical usage::

    from core.memory import MemoryGateway, MemoryType, MemoryRegistry
    from core.memory.provider import MemoryProvider   # for implementors only

    registry = MemoryRegistry()
    gateway  = MemoryGateway(registry)
"""

from core.memory.exceptions import (
    MemoryError,
    MemoryProviderError,
    MemoryRoutingError,
    MemoryValidationError,
)
from core.memory.gateway import MemoryGateway
from core.memory.models import MemoryType, ROUTING_TABLE
from core.memory.provider import MemoryProvider
from core.memory.registry import MemoryRegistry
from core.memory.request import MemoryRequest
from core.memory.response import MemoryResponse
from core.memory.router import MemoryRouter

__all__ = [
    # Gateway — primary public API
    "MemoryGateway",
    # Enumerations / models
    "MemoryType",
    "ROUTING_TABLE",
    # Request / response primitives
    "MemoryRequest",
    "MemoryResponse",
    # Infrastructure (for provider implementors and tests)
    "MemoryProvider",
    "MemoryRegistry",
    "MemoryRouter",
    # Exceptions
    "MemoryError",
    "MemoryProviderError",
    "MemoryRoutingError",
    "MemoryValidationError",
]