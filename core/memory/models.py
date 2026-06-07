# core/memory/models.py
"""
Shared domain models for the POLARIS v5 Memory Gateway.

This module defines the :class:`MemoryType` enumeration that categorises the
four constitutional memory systems, plus module-level constants used across
the memory subsystem.
"""

from __future__ import annotations

from enum import Enum, unique
from typing import Final


# ---------------------------------------------------------------------------
# MemoryType
# ---------------------------------------------------------------------------


@unique
class MemoryType(Enum):
    """The four constitutional memory systems of POLARIS v5.

    Each value maps to exactly one backing subsystem and, therefore, to
    exactly one registered :class:`~core.memory.provider.MemoryProvider`.

    Routing table
    -------------
    ``EXPERIENCE``   → ECHO subsystem
    ``KNOWLEDGE``    → LUNA subsystem
    ``TEMPORAL``     → CHRONOS subsystem
    ``RELATIONSHIP`` → CONSTELLATION subsystem
    """

    EXPERIENCE = "experience"
    """Episodic / experiential memory — backed by ECHO."""

    KNOWLEDGE = "knowledge"
    """Semantic / factual knowledge — backed by LUNA."""

    TEMPORAL = "temporal"
    """Time-ordered event memory — backed by CHRONOS."""

    RELATIONSHIP = "relationship"
    """Graph-structured relationship memory — backed by CONSTELLATION."""

    def __str__(self) -> str:  # pragma: no cover
        return self.value


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_KEY_MAX_LEN: Final[int] = 512
"""Maximum byte-length of a memory record key."""

_PROVIDER_ID_MAX_LEN: Final[int] = 256
"""Maximum length of a provider identifier string."""

_METADATA_MAX_KEYS: Final[int] = 64
"""Maximum number of entries in a request/response metadata dict."""

_SEARCH_QUERY_MAX_LEN: Final[int] = 4096
"""Maximum length of a search query string."""

# Canonical provider IDs used by the routing table.
PROVIDER_ID_ECHO: Final[str] = "echo"
PROVIDER_ID_LUNA: Final[str] = "luna"
PROVIDER_ID_CHRONOS: Final[str] = "chronos"
PROVIDER_ID_CONSTELLATION: Final[str] = "constellation"

# Routing table: MemoryType → canonical provider id.
ROUTING_TABLE: Final[dict[MemoryType, str]] = {
    MemoryType.EXPERIENCE: PROVIDER_ID_ECHO,
    MemoryType.KNOWLEDGE: PROVIDER_ID_LUNA,
    MemoryType.TEMPORAL: PROVIDER_ID_CHRONOS,
    MemoryType.RELATIONSHIP: PROVIDER_ID_CONSTELLATION,
}