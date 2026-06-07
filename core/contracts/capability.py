# core/contracts/capability.py
"""
Capability primitives and in-process capability registry for POLARIS v5.

A *capability* is a named, versioned contract that a subsystem declares it
can fulfil.  Other subsystems may query the registry to discover which peers
provide a required capability before establishing a dependency.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterator

from core.types.identifiers import (
    CapabilityId,
    SubsystemId,
    VersionString,
    make_capability_id,
    make_version,
    parse_capability_id,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CapabilityError(Exception):
    """Base for all capability-related errors.

    Attributes
    ----------
    capability_id:
        The :class:`~core.types.identifiers.CapabilityId` involved in the
        error, if applicable.
    """

    def __init__(
        self,
        message: str,
        *,
        capability_id: CapabilityId | None = None,
    ) -> None:
        super().__init__(message)
        self.capability_id = capability_id


class CapabilityAlreadyRegisteredError(CapabilityError):
    """Raised when a capability with the same :class:`CapabilityId` is
    registered a second time without first unregistering it."""


class CapabilityNotFoundError(CapabilityError):
    """Raised when a requested :class:`CapabilityId` does not exist in the
    registry."""


# ---------------------------------------------------------------------------
# Capability dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Capability:
    """Immutable descriptor for a single subsystem capability.

    Attributes
    ----------
    id:
        Globally unique :class:`~core.types.identifiers.CapabilityId`.
    name:
        Short, human-readable name (e.g. ``"Vector Search"``).
    description:
        One-paragraph description of what this capability provides.
    version:
        :class:`~core.types.identifiers.VersionString` (SemVer 2.0.0).
    owner:
        :class:`~core.types.identifiers.SubsystemId` of the subsystem that
        declares this capability.
    tags:
        Optional frozenset of classification tags for discoverability
        (e.g. ``frozenset({"search", "embedding"})``).
    """

    id: CapabilityId
    name: str
    description: str
    version: VersionString
    owner: SubsystemId
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Capability.name must be a non-empty string.")
        if not self.description or not self.description.strip():
            raise ValueError("Capability.description must be a non-empty string.")
        # Validate that the id's subsystem part matches the declared owner.
        id_owner, _ = parse_capability_id(self.id)
        if id_owner != self.owner:
            raise ValueError(
                f"Capability.id owner segment {id_owner!r} does not match "
                f"Capability.owner {self.owner!r}."
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        owner: SubsystemId,
        slug: str,
        name: str,
        description: str,
        version: str,
        tags: frozenset[str] | set[str] | None = None,
    ) -> "Capability":
        """Validated factory method for :class:`Capability`.

        Parameters
        ----------
        owner:
            :class:`~core.types.identifiers.SubsystemId` of the declaring
            subsystem.
        slug:
            Kebab-case capability identifier (e.g. ``vector-search``).
        name:
            Human-readable display name.
        description:
            Prose description of the capability's contract.
        version:
            SemVer 2.0.0 version string.
        tags:
            Optional classification tags.

        Returns
        -------
        Capability
            A fully validated :class:`Capability` instance.
        """
        cap_id = make_capability_id(owner, slug)
        ver = make_version(version)
        return cls(
            id=cap_id,
            name=name,
            description=description,
            version=ver,
            owner=owner,
            tags=frozenset(tags) if tags else frozenset(),
        )


# ---------------------------------------------------------------------------
# Capability registry
# ---------------------------------------------------------------------------


class CapabilityRegistry:
    """Thread-safe, in-process registry of all declared :class:`Capability`
    instances across the POLARIS runtime.

    Lifecycle
    ---------
    * ``register()`` — add a capability (idempotency-checked).
    * ``unregister()`` — remove a capability.
    * ``get()`` — exact lookup by :class:`~core.types.identifiers.CapabilityId`.
    * ``list()`` — iterate all registered capabilities (with optional filtering).
    * ``list_by_owner()`` — iterate capabilities belonging to one subsystem.
    * ``list_by_tag()`` — iterate capabilities carrying a specific tag.

    Thread safety
    -------------
    All mutations are protected by a :class:`threading.RLock`.  Read
    operations (``get``, ``list*``) return copies so callers cannot hold
    references into the internal store.
    """

    def __init__(self) -> None:
        self._store: dict[CapabilityId, Capability] = {}
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def register(self, capability: Capability) -> None:
        """Register a :class:`Capability` in the registry.

        Parameters
        ----------
        capability:
            The capability descriptor to register.

        Raises
        ------
        CapabilityAlreadyRegisteredError
            If a capability with the same :attr:`Capability.id` is already
            registered.
        TypeError
            If *capability* is not a :class:`Capability` instance.
        """
        if not isinstance(capability, Capability):
            raise TypeError(
                f"Expected a Capability instance, got {type(capability).__name__!r}."
            )
        with self._lock:
            if capability.id in self._store:
                raise CapabilityAlreadyRegisteredError(
                    f"Capability {capability.id!r} is already registered. "
                    "Call unregister() first if you intend to replace it.",
                    capability_id=capability.id,
                )
            self._store[capability.id] = capability

    def unregister(self, capability_id: CapabilityId) -> Capability:
        """Remove a capability from the registry.

        Parameters
        ----------
        capability_id:
            The identifier of the capability to remove.

        Returns
        -------
        Capability
            The removed capability descriptor.

        Raises
        ------
        CapabilityNotFoundError
            If *capability_id* is not registered.
        """
        with self._lock:
            if capability_id not in self._store:
                raise CapabilityNotFoundError(
                    f"Capability {capability_id!r} is not registered.",
                    capability_id=capability_id,
                )
            return self._store.pop(capability_id)

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get(self, capability_id: CapabilityId) -> Capability:
        """Retrieve a capability by its identifier.

        Parameters
        ----------
        capability_id:
            Exact :class:`~core.types.identifiers.CapabilityId` to look up.

        Returns
        -------
        Capability
            The matching capability descriptor.

        Raises
        ------
        CapabilityNotFoundError
            If *capability_id* is not registered.
        """
        with self._lock:
            if capability_id not in self._store:
                raise CapabilityNotFoundError(
                    f"Capability {capability_id!r} is not registered.",
                    capability_id=capability_id,
                )
            return self._store[capability_id]

    def has(self, capability_id: CapabilityId) -> bool:
        """Return whether *capability_id* is currently registered.

        Parameters
        ----------
        capability_id:
            Identifier to probe.
        """
        with self._lock:
            return capability_id in self._store

    def list(self) -> list[Capability]:
        """Return a snapshot of all registered capabilities.

        Returns
        -------
        list[Capability]
            Unordered list of all current capability descriptors.
        """
        with self._lock:
            return list(self._store.values())

    def list_by_owner(self, owner: SubsystemId) -> list[Capability]:
        """Return all capabilities owned by *owner*.

        Parameters
        ----------
        owner:
            :class:`~core.types.identifiers.SubsystemId` to filter by.

        Returns
        -------
        list[Capability]
            All capabilities whose :attr:`Capability.owner` matches *owner*.
        """
        with self._lock:
            return [c for c in self._store.values() if c.owner == owner]

    def list_by_tag(self, tag: str) -> list[Capability]:
        """Return all capabilities carrying *tag*.

        Parameters
        ----------
        tag:
            Tag string to filter by (exact match).

        Returns
        -------
        list[Capability]
            All capabilities whose :attr:`Capability.tags` contains *tag*.
        """
        with self._lock:
            return [c for c in self._store.values() if tag in c.tags]

    def __iter__(self) -> Iterator[Capability]:
        """Iterate over a snapshot of all registered capabilities."""
        return iter(self.list())

    def __len__(self) -> int:
        """Return the number of currently registered capabilities."""
        with self._lock:
            return len(self._store)

    def __contains__(self, capability_id: object) -> bool:
        """Support ``in`` operator for :class:`CapabilityId` membership tests."""
        if not isinstance(capability_id, str):
            return False
        with self._lock:
            return capability_id in self._store

    def __repr__(self) -> str:  # pragma: no cover
        with self._lock:
            return f"CapabilityRegistry(count={len(self._store)})"