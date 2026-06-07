# core/registry/registry.py
"""
Central subsystem registry for the POLARIS v5 runtime.

:class:`SubsystemRegistry` is the authoritative store of all subsystems that
have been admitted into the runtime.  It enforces registration contracts,
validates dependency graphs, and provides capability-based discovery.

Thread safety
-------------
All public methods are protected by a single :class:`threading.RLock`.
The lock is reentrant to allow registry inspection from within callback hooks
that themselves call registry methods.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

from core.contracts.capability import Capability, CapabilityId, CapabilityRegistry
from core.contracts.health import HealthReport, HealthStatus
from core.contracts.lifecycle import LifecycleState
from core.contracts.subsystem import (
    DependencyError,
    LifecycleError,
    RegistrationError,
    SubsystemContract,
)
from core.types.identifiers import SubsystemId

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registration record
# ---------------------------------------------------------------------------


@dataclass
class RegistrationRecord:
    """Internal metadata held by the registry for each admitted subsystem.

    Attributes
    ----------
    subsystem:
        The live :class:`~core.contracts.subsystem.SubsystemContract` instance.
    registered_at:
        UTC timestamp of initial registration.
    """

    subsystem: SubsystemContract
    registered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def subsystem_id(self) -> SubsystemId:
        """Shortcut to :attr:`SubsystemMetadata.id`."""
        return self.subsystem.metadata.id


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SubsystemRegistry:
    """Thread-safe central registry for all POLARIS v5 subsystems.

    Responsibilities
    ----------------
    * **Registration** — admit a :class:`~core.contracts.subsystem.SubsystemContract`
      instance into the runtime; reject duplicates.
    * **Unregistration** — remove a subsystem, enforcing that it is
      ``STOPPED`` before removal (or optionally forcing shutdown).
    * **Lookup** — retrieve a live subsystem by
      :class:`~core.types.identifiers.SubsystemId`.
    * **Dependency verification** — assert that all declared dependencies of
      a subsystem are registered and in an acceptable state before the
      subsystem is allowed to start.
    * **Capability discovery** — answer queries about which subsystems
      expose a given :class:`~core.contracts.capability.Capability`.
    * **Health aggregation** — collect :class:`~core.contracts.health.HealthReport`
      snapshots from every registered subsystem.

    Usage
    -----
    .. code-block:: python

        registry = SubsystemRegistry()
        registry.register(my_subsystem)
        registry.verify_dependencies(my_subsystem.metadata.id)
        my_subsystem.initialize()
        my_subsystem.start()

    Parameters
    ----------
    name:
        Optional human-readable name for this registry instance (used in
        log messages and ``repr``).
    """

    def __init__(self, name: str = "polaris.runtime.registry") -> None:
        self._name = name
        self._records: dict[SubsystemId, RegistrationRecord] = {}
        self._capability_registry: CapabilityRegistry = CapabilityRegistry()
        self._lock: threading.RLock = threading.RLock()
        _logger.debug("SubsystemRegistry %r created.", name)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, subsystem: SubsystemContract) -> RegistrationRecord:
        """Admit *subsystem* into the registry.

        The subsystem must be in the ``CREATED`` state and must not already
        be registered under the same
        :class:`~core.types.identifiers.SubsystemId`.

        Parameters
        ----------
        subsystem:
            A fully constructed but not yet initialised subsystem instance.

        Returns
        -------
        RegistrationRecord
            The internal record created for this subsystem.

        Raises
        ------
        TypeError
            If *subsystem* is not a :class:`~core.contracts.subsystem.SubsystemContract`.
        RegistrationError
            If a subsystem with the same id is already registered.
        LifecycleError
            If *subsystem* is not in the ``CREATED`` state.
        """
        if not isinstance(subsystem, SubsystemContract):
            raise TypeError(
                f"subsystem must be a SubsystemContract instance, "
                f"got {type(subsystem).__name__!r}."
            )

        meta = subsystem.metadata
        sid = meta.id

        with self._lock:
            if sid in self._records:
                raise RegistrationError(
                    f"Subsystem {sid!r} is already registered. "
                    "Unregister it first before registering a replacement.",
                    subsystem_id=sid,
                )

            if subsystem.state is not LifecycleState.CREATED:
                raise LifecycleError(
                    f"Subsystem {sid!r} must be in state CREATED to be "
                    f"registered, but is in {subsystem.state.name}.",
                    current_state=subsystem.state,
                    attempted_transition="register",
                )

            record = RegistrationRecord(subsystem=subsystem)
            self._records[sid] = record

            # Register all capabilities declared by this subsystem.
            for cap in meta.capabilities:
                self._capability_registry.register(cap)
                _logger.debug(
                    "Capability %r registered (owner=%r).", cap.id, sid
                )

        _logger.info(
            "Subsystem %r registered (version=%s, capabilities=%d).",
            sid,
            meta.version,
            len(meta.capabilities),
        )
        return record

    def unregister(
        self,
        subsystem_id: SubsystemId,
        *,
        force_stop: bool = False,
    ) -> SubsystemContract:
        """Remove a subsystem from the registry.

        By default, the subsystem must already be in the ``STOPPED`` or
        ``FAILED`` state.  Set *force_stop* to ``True`` to automatically
        call :meth:`~core.contracts.subsystem.SubsystemContract.stop` first
        if the subsystem is still operational.

        Parameters
        ----------
        subsystem_id:
            Identifier of the subsystem to remove.
        force_stop:
            If ``True`` and the subsystem is not yet stopped, attempt a
            graceful shutdown before unregistering.

        Returns
        -------
        SubsystemContract
            The removed subsystem instance.

        Raises
        ------
        RegistrationError
            If *subsystem_id* is not registered.
        LifecycleError
            If *force_stop* is ``False`` and the subsystem is not in a
            terminal or failed state.
        """
        with self._lock:
            if subsystem_id not in self._records:
                raise RegistrationError(
                    f"Subsystem {subsystem_id!r} is not registered.",
                    subsystem_id=subsystem_id,
                )

            record = self._records[subsystem_id]
            subsystem = record.subsystem
            current_state = subsystem.state

            _stoppable = {
                LifecycleState.RUNNING,
                LifecycleState.PAUSED,
                LifecycleState.RECOVERING,
                LifecycleState.STOPPING,
            }
            _terminal = {LifecycleState.STOPPED, LifecycleState.FAILED}

            if current_state in _stoppable:
                if not force_stop:
                    raise LifecycleError(
                        f"Cannot unregister subsystem {subsystem_id!r} while it "
                        f"is in state {current_state.name}. "
                        "Call stop() first, or pass force_stop=True.",
                        current_state=current_state,
                        attempted_transition="unregister",
                    )
                _logger.warning(
                    "force_stop=True: stopping subsystem %r (state=%s) "
                    "before unregistering.",
                    subsystem_id,
                    current_state.name,
                )
                try:
                    subsystem.stop()
                except Exception:
                    _logger.exception(
                        "Subsystem %r raised during forced stop; "
                        "proceeding with unregistration.",
                        subsystem_id,
                    )

            # Remove capabilities.
            for cap_id in subsystem.metadata.capability_ids:
                if self._capability_registry.has(cap_id):
                    self._capability_registry.unregister(cap_id)
                    _logger.debug(
                        "Capability %r unregistered (owner=%r).",
                        cap_id,
                        subsystem_id,
                    )

            del self._records[subsystem_id]

        _logger.info("Subsystem %r unregistered.", subsystem_id)
        return subsystem

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, subsystem_id: SubsystemId) -> SubsystemContract:
        """Retrieve a registered subsystem by its identifier.

        Parameters
        ----------
        subsystem_id:
            Identifier to look up.

        Returns
        -------
        SubsystemContract
            The live subsystem instance.

        Raises
        ------
        RegistrationError
            If *subsystem_id* is not registered.
        """
        with self._lock:
            if subsystem_id not in self._records:
                raise RegistrationError(
                    f"Subsystem {subsystem_id!r} is not registered.",
                    subsystem_id=subsystem_id,
                )
            return self._records[subsystem_id].subsystem

    def get_record(self, subsystem_id: SubsystemId) -> RegistrationRecord:
        """Retrieve the full :class:`RegistrationRecord` for a subsystem.

        Parameters
        ----------
        subsystem_id:
            Identifier to look up.

        Returns
        -------
        RegistrationRecord
            The registration record (includes timestamps).

        Raises
        ------
        RegistrationError
            If *subsystem_id* is not registered.
        """
        with self._lock:
            if subsystem_id not in self._records:
                raise RegistrationError(
                    f"Subsystem {subsystem_id!r} is not registered.",
                    subsystem_id=subsystem_id,
                )
            return self._records[subsystem_id]

    def has(self, subsystem_id: SubsystemId) -> bool:
        """Return whether *subsystem_id* is currently registered.

        Parameters
        ----------
        subsystem_id:
            Identifier to probe.
        """
        with self._lock:
            return subsystem_id in self._records

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_subsystems(self) -> list[SubsystemContract]:
        """Return a snapshot of all currently registered subsystems.

        Returns
        -------
        list[SubsystemContract]
            Unordered list of all registered subsystem instances.
        """
        with self._lock:
            return [r.subsystem for r in self._records.values()]

    def list_records(self) -> list[RegistrationRecord]:
        """Return a snapshot of all :class:`RegistrationRecord` entries.

        Returns
        -------
        list[RegistrationRecord]
            Copy of the internal record list.
        """
        with self._lock:
            return list(self._records.values())

    def list_by_state(self, state: LifecycleState) -> list[SubsystemContract]:
        """Return all subsystems currently in *state*.

        Parameters
        ----------
        state:
            :class:`~core.contracts.lifecycle.LifecycleState` to filter by.

        Returns
        -------
        list[SubsystemContract]
            All subsystems in the given state.
        """
        with self._lock:
            return [
                r.subsystem
                for r in self._records.values()
                if r.subsystem.state is state
            ]

    # ------------------------------------------------------------------
    # Dependency verification
    # ------------------------------------------------------------------

    def verify_dependencies(self, subsystem_id: SubsystemId) -> None:
        """Verify that all declared dependencies of *subsystem_id* are
        registered and in the ``RUNNING`` state.

        This method should be called **before** invoking
        :meth:`~core.contracts.subsystem.SubsystemContract.start` to ensure
        the dependency graph is satisfied.

        Parameters
        ----------
        subsystem_id:
            The subsystem whose dependencies to verify.

        Raises
        ------
        RegistrationError
            If *subsystem_id* is not registered.
        DependencyError
            If one or more declared dependencies are not registered, or are
            not in the ``RUNNING`` state.
        """
        with self._lock:
            if subsystem_id not in self._records:
                raise RegistrationError(
                    f"Subsystem {subsystem_id!r} is not registered.",
                    subsystem_id=subsystem_id,
                )

            meta = self._records[subsystem_id].subsystem.metadata
            not_registered: set[SubsystemId] = set()
            not_running: set[SubsystemId] = set()

            for dep_id in meta.dependencies:
                if dep_id not in self._records:
                    not_registered.add(dep_id)
                elif self._records[dep_id].subsystem.state is not LifecycleState.RUNNING:
                    not_running.add(dep_id)

            missing = not_registered | not_running
            if missing:
                details: list[str] = []
                if not_registered:
                    details.append(
                        f"not registered: {sorted(not_registered)}"
                    )
                if not_running:
                    states = {
                        dep: self._records[dep].subsystem.state.name
                        for dep in not_running
                    }
                    details.append(f"not RUNNING: {states}")
                raise DependencyError(
                    f"Subsystem {subsystem_id!r} has unsatisfied dependencies. "
                    + "; ".join(details) + ".",
                    subsystem_id=subsystem_id,
                    missing=frozenset(missing),
                )

        _logger.debug(
            "Subsystem %r: all dependencies satisfied.", subsystem_id
        )

    def build_dependency_order(
        self,
        subsystem_ids: list[SubsystemId] | None = None,
    ) -> list[SubsystemId]:
        """Return a topologically sorted list of subsystem ids such that each
        subsystem appears after all of its dependencies.

        Parameters
        ----------
        subsystem_ids:
            Optional subset of registered subsystem ids to include.
            Defaults to all registered subsystems.

        Returns
        -------
        list[SubsystemId]
            Dependency-ordered list of subsystem identifiers.

        Raises
        ------
        DependencyError
            If the dependency graph contains a cycle.
        RegistrationError
            If any id in *subsystem_ids* is not registered.
        """
        with self._lock:
            target_ids: list[SubsystemId]
            if subsystem_ids is None:
                target_ids = list(self._records.keys())
            else:
                for sid in subsystem_ids:
                    if sid not in self._records:
                        raise RegistrationError(
                            f"Subsystem {sid!r} is not registered.",
                            subsystem_id=sid,
                        )
                target_ids = list(subsystem_ids)

            # Kahn's algorithm for topological sort.
            all_deps: dict[SubsystemId, set[SubsystemId]] = {}
            for sid in target_ids:
                meta = self._records[sid].subsystem.metadata
                # Only consider dependencies that are in the target set.
                all_deps[sid] = {
                    d for d in meta.dependencies if d in set(target_ids)
                }

            in_degree: dict[SubsystemId, int] = {
                sid: len(deps) for sid, deps in all_deps.items()
            }
            queue: list[SubsystemId] = [
                sid for sid, deg in in_degree.items() if deg == 0
            ]
            result: list[SubsystemId] = []

            while queue:
                node = queue.pop(0)
                result.append(node)
                for sid in target_ids:
                    if node in all_deps[sid]:
                        all_deps[sid].discard(node)
                        in_degree[sid] -= 1
                        if in_degree[sid] == 0:
                            queue.append(sid)

            if len(result) != len(target_ids):
                cycle_nodes = sorted(
                    set(target_ids) - set(result)
                )
                raise DependencyError(
                    f"Circular dependency detected among subsystems: "
                    f"{cycle_nodes}.",
                    subsystem_id=SubsystemId(cycle_nodes[0] if cycle_nodes else ""),
                    missing=frozenset(
                        SubsystemId(n) for n in cycle_nodes
                    ),
                )

            return result

    # ------------------------------------------------------------------
    # Capability queries
    # ------------------------------------------------------------------

    def get_capability(self, capability_id: CapabilityId) -> Capability:
        """Retrieve a :class:`~core.contracts.capability.Capability` by id.

        Parameters
        ----------
        capability_id:
            The capability identifier to resolve.

        Returns
        -------
        Capability
            The matching capability descriptor.

        Raises
        ------
        CapabilityNotFoundError
            If the capability is not registered.
        """
        return self._capability_registry.get(capability_id)

    def list_capabilities(self) -> list[Capability]:
        """Return a snapshot of all capabilities declared by registered
        subsystems.

        Returns
        -------
        list[Capability]
            All currently registered capability descriptors.
        """
        return self._capability_registry.list()

    def find_subsystems_with_capability(
        self, capability_id: CapabilityId
    ) -> list[SubsystemContract]:
        """Find all subsystems that declare *capability_id*.

        Parameters
        ----------
        capability_id:
            The capability to search for.

        Returns
        -------
        list[SubsystemContract]
            Subsystems whose metadata declares the given capability.
            Returns an empty list if none match.
        """
        with self._lock:
            matches: list[SubsystemContract] = []
            for record in self._records.values():
                if capability_id in record.subsystem.metadata.capability_ids:
                    matches.append(record.subsystem)
            return matches

    def find_subsystems_by_tag(self, tag: str) -> list[SubsystemContract]:
        """Find all subsystems that declare at least one capability with
        *tag*.

        Parameters
        ----------
        tag:
            Tag string to filter by (exact match).

        Returns
        -------
        list[SubsystemContract]
            Matching subsystem instances.
        """
        tagged_caps = self._capability_registry.list_by_tag(tag)
        owners = {cap.owner for cap in tagged_caps}
        with self._lock:
            return [
                self._records[oid].subsystem
                for oid in owners
                if oid in self._records
            ]

    # ------------------------------------------------------------------
    # Health aggregation
    # ------------------------------------------------------------------

    def collect_health(self) -> dict[SubsystemId, HealthReport]:
        """Collect :class:`~core.contracts.health.HealthReport` from every
        registered subsystem.

        Health probes are executed sequentially.  Exceptions raised by a
        subsystem's ``health()`` method are caught and translated into a
        :attr:`~core.contracts.health.HealthStatus.FAILED` report so that a
        single broken subsystem does not prevent the rest from reporting.

        Returns
        -------
        dict[SubsystemId, HealthReport]
            Mapping of subsystem id → health report.
        """
        snapshot = self.list_records()
        reports: dict[SubsystemId, HealthReport] = {}
        for record in snapshot:
            sid = record.subsystem_id
            try:
                reports[sid] = record.subsystem.health()
            except Exception as exc:  # noqa: BLE001
                _logger.exception(
                    "Subsystem %r raised during health probe.", sid
                )
                reports[sid] = HealthReport.failed(
                    message=f"health() raised an unexpected exception: {exc}",
                    metadata={"exception_type": type(exc).__name__},
                )
        return reports

    def aggregate_health(self) -> HealthStatus:
        """Compute the worst-case :class:`~core.contracts.health.HealthStatus`
        across all registered subsystems.

        Returns
        -------
        HealthStatus
            The most severe status observed.  Returns
            :attr:`~core.contracts.health.HealthStatus.HEALTHY` if no
            subsystems are registered.
        """
        reports = self.collect_health()
        if not reports:
            return HealthStatus.HEALTHY
        worst = HealthStatus.HEALTHY
        for report in reports.values():
            if report.status.is_worse_than(worst):
                worst = report.status
        return worst

    # ------------------------------------------------------------------
    # Iteration and containment
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[SubsystemContract]:
        """Iterate over a snapshot of registered subsystems."""
        return iter(self.list_subsystems())

    def __len__(self) -> int:
        """Return the number of currently registered subsystems."""
        with self._lock:
            return len(self._records)

    def __contains__(self, subsystem_id: object) -> bool:
        """Support ``in`` operator for :class:`~core.types.identifiers.SubsystemId`
        membership tests."""
        if not isinstance(subsystem_id, str):
            return False
        with self._lock:
            return subsystem_id in self._records

    def __repr__(self) -> str:  # pragma: no cover
        with self._lock:
            return (
                f"SubsystemRegistry(name={self._name!r}, "
                f"subsystems={len(self._records)}, "
                f"capabilities={len(self._capability_registry)})"
            )