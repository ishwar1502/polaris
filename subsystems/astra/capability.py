# subsystems/astra/capability.py
"""
ASTRA v5 Capability Module.

Provides :class:`AstraCapabilityRegistry` and :class:`CapabilityManager` for
managing assessed user capabilities within the ASTRA Digital Twin Core.

ASTRA's capability model is distinct from the runtime
:class:`~core.contracts.capability.CapabilityRegistry`.  The runtime registry
tracks what *subsystems* can do; ASTRA's registry models what the *user* is
capable of — strengths, growth areas, and assessed competencies.

All mutations are thread-safe via an internal :class:`threading.RLock`.
"""

from __future__ import annotations

import copy
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from subsystems.astra.exceptions import AstraError
from subsystems.astra.models import (
    CapabilityEntry,
    CapabilitySnapshot,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CapabilityNotFoundError(AstraError):
    """Raised when a capability name is not found in the registry."""

    def __init__(self, name: str, domain: str | None = None) -> None:
        key = f"{domain}/{name}" if domain else name
        super().__init__(f"Capability '{key}' not found.")
        self.name = name
        self.domain = domain


class CapabilityAlreadyExistsError(AstraError):
    """Raised when attempting to register a duplicate capability."""

    def __init__(self, name: str, domain: str) -> None:
        super().__init__(
            f"Capability '{domain}/{name}' already registered. "
            "Use update_capability() to modify existing entries."
        )
        self.name = name
        self.domain = domain


class CapabilityValidationError(AstraError):
    """Raised when capability data fails validation."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


# ---------------------------------------------------------------------------
# Key helper
# ---------------------------------------------------------------------------

def _cap_key(name: str, domain: str) -> str:
    """Canonical dictionary key: ``domain::name``."""
    return f"{domain.lower()}::{name.lower()}"


# ---------------------------------------------------------------------------
# AstraCapabilityRegistry
# ---------------------------------------------------------------------------


class AstraCapabilityRegistry:
    """Thread-safe registry of assessed user capabilities.

    Stores :class:`~subsystems.astra.models.CapabilityEntry` instances keyed
    by ``(domain, name)``.  Entries are partitioned into *strengths* (high
    confidence, evidence-backed) and *growth_areas* (developing, not yet
    consolidated).

    The threshold between strength and growth area is configurable via
    *strength_threshold*.

    Parameters
    ----------
    strength_threshold:
        Minimum confidence required to classify an entry as a strength.
        Defaults to 0.7.
    """

    def __init__(self, strength_threshold: float = 0.7) -> None:
        if not (0.0 < strength_threshold <= 1.0):
            raise ValueError("strength_threshold must be between 0.0 (exclusive) and 1.0.")
        self._lock: threading.RLock = threading.RLock()
        self._entries: dict[str, CapabilityEntry] = {}
        self._strength_threshold = strength_threshold

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, entry: CapabilityEntry) -> None:
        """Register a new capability entry.

        Parameters
        ----------
        entry:
            The :class:`~subsystems.astra.models.CapabilityEntry` to register.

        Raises
        ------
        CapabilityAlreadyExistsError
            If a capability with the same name and domain already exists.
        CapabilityValidationError
            If the entry fails validation.
        TypeError
            If *entry* is not a :class:`~subsystems.astra.models.CapabilityEntry`.
        """
        if not isinstance(entry, CapabilityEntry):
            raise TypeError(
                f"Expected a CapabilityEntry instance, got {type(entry).__name__!r}."
            )
        self._validate_entry(entry)
        key = _cap_key(entry.name, entry.domain)
        with self._lock:
            if key in self._entries:
                raise CapabilityAlreadyExistsError(entry.name, entry.domain)
            self._entries[key] = copy.copy(entry)
        _logger.debug("Registered capability '%s/%s'.", entry.domain, entry.name)

    def remove(self, *, name: str, domain: str) -> CapabilityEntry:
        """Remove a capability entry from the registry.

        Parameters
        ----------
        name:
            Capability name.
        domain:
            Domain the capability belongs to.

        Returns
        -------
        CapabilityEntry
            The removed entry.

        Raises
        ------
        CapabilityNotFoundError
            If no matching entry is found.
        """
        key = _cap_key(name, domain)
        with self._lock:
            if key not in self._entries:
                raise CapabilityNotFoundError(name, domain)
            removed = self._entries.pop(key)
        _logger.debug("Removed capability '%s/%s'.", domain, name)
        return removed

    def update_capability(
        self,
        *,
        name: str,
        domain: str,
        confidence: float | None = None,
        evidence_count: int | None = None,
        notes: str | None = None,
    ) -> CapabilityEntry:
        """Update fields on an existing capability entry.

        Parameters
        ----------
        name:
            Capability name.
        domain:
            Capability domain.
        confidence:
            New confidence value (0.0–1.0) if updating.
        evidence_count:
            New evidence count if updating.
        notes:
            New notes text if updating.

        Returns
        -------
        CapabilityEntry
            The updated entry.

        Raises
        ------
        CapabilityNotFoundError
            If no matching entry is found.
        CapabilityValidationError
            If the new values are invalid.
        """
        key = _cap_key(name, domain)
        with self._lock:
            if key not in self._entries:
                raise CapabilityNotFoundError(name, domain)
            entry = self._entries[key]
            if confidence is not None:
                if not (0.0 <= confidence <= 1.0):
                    raise CapabilityValidationError(
                        "confidence must be between 0.0 and 1.0.", field="confidence"
                    )
                entry.confidence = confidence
            if evidence_count is not None:
                if evidence_count < 0:
                    raise CapabilityValidationError(
                        "evidence_count cannot be negative.", field="evidence_count"
                    )
                entry.evidence_count = evidence_count
            if notes is not None:
                entry.notes = notes
            entry.observed_at = datetime.now(timezone.utc)
            updated = copy.copy(entry)
        _logger.debug("Updated capability '%s/%s'.", domain, name)
        return updated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, *, name: str, domain: str) -> CapabilityEntry:
        """Return the entry matching (name, domain).

        Raises
        ------
        CapabilityNotFoundError
            If no matching entry is found.
        """
        key = _cap_key(name, domain)
        with self._lock:
            if key not in self._entries:
                raise CapabilityNotFoundError(name, domain)
            return copy.copy(self._entries[key])

    def has(self, *, name: str, domain: str) -> bool:
        """Return whether a capability is registered."""
        key = _cap_key(name, domain)
        with self._lock:
            return key in self._entries

    def list_all(self) -> list[CapabilityEntry]:
        """Return a snapshot of all registered capability entries."""
        with self._lock:
            return [copy.copy(e) for e in self._entries.values()]

    def list_by_domain(self, domain: str) -> list[CapabilityEntry]:
        """Return all capabilities in a given domain."""
        domain_lower = domain.lower()
        with self._lock:
            return [
                copy.copy(e)
                for e in self._entries.values()
                if e.domain.lower() == domain_lower
            ]

    def strengths(self) -> list[CapabilityEntry]:
        """Return entries whose confidence meets or exceeds the strength threshold."""
        with self._lock:
            return [
                copy.copy(e)
                for e in self._entries.values()
                if e.confidence >= self._strength_threshold
            ]

    def growth_areas(self) -> list[CapabilityEntry]:
        """Return entries whose confidence is below the strength threshold."""
        with self._lock:
            return [
                copy.copy(e)
                for e in self._entries.values()
                if e.confidence < self._strength_threshold
            ]

    def snapshot(self) -> CapabilitySnapshot:
        """Generate a :class:`~subsystems.astra.models.CapabilitySnapshot`."""
        with self._lock:
            strengths = [
                copy.copy(e)
                for e in self._entries.values()
                if e.confidence >= self._strength_threshold
            ]
            growth = [
                copy.copy(e)
                for e in self._entries.values()
                if e.confidence < self._strength_threshold
            ]
        return CapabilitySnapshot(
            strengths=sorted(strengths, key=lambda e: e.confidence, reverse=True),
            growth_areas=sorted(growth, key=lambda e: e.confidence, reverse=True),
            snapshot_at=datetime.now(timezone.utc),
        )

    def health_summary(self) -> dict[str, Any]:
        """Return a dictionary summarising the capability health state."""
        with self._lock:
            all_entries = list(self._entries.values())
        total = len(all_entries)
        strong = sum(1 for e in all_entries if e.confidence >= self._strength_threshold)
        avg_confidence = (
            sum(e.confidence for e in all_entries) / total if total else 0.0
        )
        avg_evidence = (
            sum(e.evidence_count for e in all_entries) / total if total else 0.0
        )
        domains = sorted({e.domain for e in all_entries})
        return {
            "total_capabilities": total,
            "strengths_count": strong,
            "growth_areas_count": total - strong,
            "average_confidence": round(avg_confidence, 4),
            "average_evidence_count": round(avg_evidence, 2),
            "domains": domains,
            "strength_threshold": self._strength_threshold,
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, tuple) and len(item) == 2:
            name, domain = item
            return self.has(name=name, domain=domain)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_entry(entry: CapabilityEntry) -> None:
        if not entry.name or not entry.name.strip():
            raise CapabilityValidationError(
                "CapabilityEntry.name must be a non-empty string.", field="name"
            )
        if not entry.domain or not entry.domain.strip():
            raise CapabilityValidationError(
                "CapabilityEntry.domain must be a non-empty string.", field="domain"
            )


# ---------------------------------------------------------------------------
# CapabilityManager
# ---------------------------------------------------------------------------


class CapabilityManager:
    """High-level manager for ASTRA user capabilities.

    Wraps :class:`AstraCapabilityRegistry` and adds higher-level operations:
    evidence-based capability upsert, domain discovery, metadata annotations,
    and integration helpers for the Digital Twin Engine.

    Parameters
    ----------
    registry:
        Optional pre-built registry.  A new one is created if not provided.
    strength_threshold:
        Passed to the registry when creating a fresh one.  Ignored when an
        existing *registry* is supplied.
    """

    def __init__(
        self,
        registry: AstraCapabilityRegistry | None = None,
        strength_threshold: float = 0.7,
    ) -> None:
        self._registry = registry or AstraCapabilityRegistry(
            strength_threshold=strength_threshold
        )
        self._lock: threading.RLock = threading.RLock()
        self._metadata: dict[str, dict[str, Any]] = {}  # key → metadata dict

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def register_capability(
        self,
        *,
        name: str,
        domain: str,
        confidence: float,
        evidence_count: int = 0,
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CapabilityEntry:
        """Register a new capability.

        Parameters
        ----------
        name:
            Canonical capability name.
        domain:
            Domain category (e.g. ``"programming"``, ``"design"``).
        confidence:
            0.0–1.0 confidence in the assessment.
        evidence_count:
            Number of supporting evidence instances.
        notes:
            Human-readable rationale.
        metadata:
            Arbitrary annotations stored alongside the entry.

        Returns
        -------
        CapabilityEntry
            The registered entry.

        Raises
        ------
        CapabilityAlreadyExistsError
            If already registered.
        CapabilityValidationError
            If validation fails.
        """
        entry = CapabilityEntry(
            name=name,
            domain=domain,
            confidence=confidence,
            evidence_count=evidence_count,
            notes=notes,
        )
        self._registry.register(entry)
        if metadata:
            key = _cap_key(name, domain)
            with self._lock:
                self._metadata[key] = dict(metadata)
        _logger.info("Capability registered: %s/%s (confidence=%.2f).", domain, name, confidence)
        return entry

    def unregister_capability(self, *, name: str, domain: str) -> CapabilityEntry:
        """Remove a capability.

        Returns
        -------
        CapabilityEntry
            The removed entry.
        """
        removed = self._registry.remove(name=name, domain=domain)
        key = _cap_key(name, domain)
        with self._lock:
            self._metadata.pop(key, None)
        _logger.info("Capability unregistered: %s/%s.", domain, name)
        return removed

    def upsert_from_evidence(
        self,
        *,
        name: str,
        domain: str,
        confidence_delta: float,
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CapabilityEntry:
        """Record evidence for a capability, creating or updating as needed.

        If the capability is not yet registered, it is created with
        ``confidence = max(0.0, confidence_delta)`` and ``evidence_count=1``.

        If it already exists, the confidence is clamped-updated and the
        evidence count is incremented.

        Parameters
        ----------
        name:
            Capability name.
        domain:
            Capability domain.
        confidence_delta:
            Amount to adjust confidence by (-1.0 to +1.0).
        notes:
            Optional notes for the update.
        metadata:
            Optional metadata to merge in.
        """
        try:
            existing = self._registry.get(name=name, domain=domain)
            new_confidence = max(0.0, min(1.0, existing.confidence + confidence_delta))
            updated = self._registry.update_capability(
                name=name,
                domain=domain,
                confidence=new_confidence,
                evidence_count=existing.evidence_count + 1,
                notes=notes or existing.notes,
            )
            if metadata:
                key = _cap_key(name, domain)
                with self._lock:
                    self._metadata.setdefault(key, {}).update(metadata)
            return updated
        except CapabilityNotFoundError:
            return self.register_capability(
                name=name,
                domain=domain,
                confidence=max(0.0, min(1.0, confidence_delta)),
                evidence_count=1,
                notes=notes,
                metadata=metadata,
            )

    def query_capability(self, *, name: str, domain: str) -> CapabilityEntry | None:
        """Return the capability entry or None if not registered."""
        try:
            return self._registry.get(name=name, domain=domain)
        except CapabilityNotFoundError:
            return None

    def get_metadata(self, *, name: str, domain: str) -> dict[str, Any]:
        """Return the metadata annotations for a capability."""
        key = _cap_key(name, domain)
        with self._lock:
            return dict(self._metadata.get(key, {}))

    # ------------------------------------------------------------------
    # Snapshot / health
    # ------------------------------------------------------------------

    def snapshot(self) -> CapabilitySnapshot:
        """Return a :class:`~subsystems.astra.models.CapabilitySnapshot`."""
        return self._registry.snapshot()

    def health(self) -> dict[str, Any]:
        """Return capability health summary."""
        return self._registry.health_summary()

    # ------------------------------------------------------------------
    # Passthrough read queries
    # ------------------------------------------------------------------

    def list_all(self) -> list[CapabilityEntry]:
        """Return all registered capabilities."""
        return self._registry.list_all()

    def list_by_domain(self, domain: str) -> list[CapabilityEntry]:
        """Return capabilities in *domain*."""
        return self._registry.list_by_domain(domain)

    def strengths(self) -> list[CapabilityEntry]:
        """Return entries meeting the strength threshold."""
        return self._registry.strengths()

    def growth_areas(self) -> list[CapabilityEntry]:
        """Return entries below the strength threshold."""
        return self._registry.growth_areas()

    def has_capability(self, *, name: str, domain: str) -> bool:
        """Return True if the capability is registered."""
        return self._registry.has(name=name, domain=domain)

    @property
    def registry(self) -> AstraCapabilityRegistry:
        """Direct access to the underlying registry (read-only intent)."""
        return self._registry

    def __len__(self) -> int:
        return len(self._registry)