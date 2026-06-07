# core/types/identifiers.py
"""
Strongly typed identifier primitives for the POLARIS v5 runtime.

All identifiers are NewType wrappers over str, enforcing domain separation
at the type-checker level without introducing runtime overhead.
"""

from __future__ import annotations

import re
import uuid
from typing import NewType, Final

# ---------------------------------------------------------------------------
# Raw NewType identifiers
# ---------------------------------------------------------------------------

SubsystemId = NewType("SubsystemId", str)
"""Unique, stable identifier for a registered subsystem.

Format: reverse-domain style, e.g. ``polaris.core.memory``
"""

CapabilityId = NewType("CapabilityId", str)
"""Unique identifier for a declared capability.

Format: ``<subsystem_id>/<capability_slug>``, e.g.
``polaris.core.memory/vector-search``
"""

VersionString = NewType("VersionString", str)
"""Semantic version string conforming to SemVer 2.0.0.

Examples: ``1.0.0``, ``2.3.1-alpha.1``, ``0.0.1+build.42``
"""

# ---------------------------------------------------------------------------
# Validation patterns
# ---------------------------------------------------------------------------

_SUBSYSTEM_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*){1,8}$"
)
"""Dot-separated, lowercase alphanumeric segments.  2–9 segments total."""

_CAPABILITY_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*){1,8}/[a-z][a-z0-9\-]{0,63}$"
)
"""``<subsystem_id>/<slug>`` where slug is kebab-case, max 64 chars."""

_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)
"""Full SemVer 2.0.0 regular expression."""


# ---------------------------------------------------------------------------
# Factory / validation helpers
# ---------------------------------------------------------------------------


def make_subsystem_id(value: str) -> SubsystemId:
    """Create and validate a :class:`SubsystemId`.

    Parameters
    ----------
    value:
        Raw string to validate and wrap.

    Returns
    -------
    SubsystemId
        The validated identifier.

    Raises
    ------
    ValueError
        If *value* does not conform to the expected format.
    """
    if not _SUBSYSTEM_ID_RE.fullmatch(value):
        raise ValueError(
            f"Invalid SubsystemId {value!r}. "
            "Expected dot-separated lowercase alphanumeric segments "
            "(2–9 segments, e.g. 'polaris.core.memory')."
        )
    return SubsystemId(value)


def make_capability_id(subsystem_id: SubsystemId, slug: str) -> CapabilityId:
    """Create and validate a :class:`CapabilityId` from its constituent parts.

    Parameters
    ----------
    subsystem_id:
        The owning subsystem's identifier (already validated).
    slug:
        Kebab-case capability name (e.g. ``vector-search``).

    Returns
    -------
    CapabilityId
        The validated composite identifier.

    Raises
    ------
    ValueError
        If the resulting identifier does not conform to the expected format.
    """
    raw = f"{subsystem_id}/{slug}"
    if not _CAPABILITY_ID_RE.fullmatch(raw):
        raise ValueError(
            f"Invalid CapabilityId {raw!r}. "
            "Expected '<subsystem_id>/<kebab-slug>' where slug is "
            "lowercase alphanumeric with hyphens, max 64 chars."
        )
    return CapabilityId(raw)


def make_version(value: str) -> VersionString:
    """Create and validate a :class:`VersionString`.

    Parameters
    ----------
    value:
        Raw version string to validate.

    Returns
    -------
    VersionString
        The validated version string.

    Raises
    ------
    ValueError
        If *value* does not conform to SemVer 2.0.0.
    """
    if not _VERSION_RE.fullmatch(value):
        raise ValueError(
            f"Invalid VersionString {value!r}. Must conform to SemVer 2.0.0."
        )
    return VersionString(value)


def generate_subsystem_id(namespace: str, name: str) -> SubsystemId:
    """Generate a deterministic :class:`SubsystemId` from a namespace + name.

    Parameters
    ----------
    namespace:
        Top-level namespace, e.g. ``polaris``.
    name:
        Subsystem name component, e.g. ``memory``.

    Returns
    -------
    SubsystemId
        Validated identifier in the form ``<namespace>.<name>``.
    """
    return make_subsystem_id(f"{namespace}.{name}")


def parse_capability_id(capability_id: CapabilityId) -> tuple[SubsystemId, str]:
    """Decompose a :class:`CapabilityId` into its subsystem and slug parts.

    Parameters
    ----------
    capability_id:
        A previously validated capability identifier.

    Returns
    -------
    tuple[SubsystemId, str]
        ``(subsystem_id, slug)``

    Raises
    ------
    ValueError
        If *capability_id* cannot be split (malformed).
    """
    parts = capability_id.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse CapabilityId {capability_id!r}: "
            "missing '/' separator."
        )
    return SubsystemId(parts[0]), parts[1]