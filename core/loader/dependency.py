# core/loader/dependency.py
"""
Dependency graph for the POLARIS v5 Module Loader.

:class:`DependencyGraph` builds a directed acyclic graph (DAG) from the
declared dependencies of a collection of :class:`~core.loader.manifest.ModuleManifest`
objects, validates it for common errors, and produces a topologically-sorted
load order via Kahn's algorithm.

Errors detected
---------------
* **Missing dependency** — a module declares a dependency whose ID does not
  appear in the discovery set.
* **Duplicate module** — two manifests share the same ``id``.
* **Circular dependency** — the dependency graph contains a cycle, making a
  valid topological order impossible.
* **Self-dependency** — a module lists itself as a dependency
  (caught by :class:`~core.loader.manifest.ModuleManifest` validation, re-checked
  here for defence-in-depth).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterator

from core.loader.exceptions import (
    CircularDependencyError,
    DependencyResolutionError,
    ModuleValidationError,
)
from core.loader.manifest import ModuleManifest

_logger = logging.getLogger(__name__)


class DependencyGraph:
    """Directed acyclic graph of module dependencies.

    Parameters
    ----------
    manifests:
        Iterable of :class:`~core.loader.manifest.ModuleManifest` objects to
        build the graph from.  Each manifest's ``dependencies`` field is used
        to construct directed edges (dependency → dependent).

    Raises
    ------
    ModuleValidationError
        If two manifests share the same ``id``.

    Notes
    -----
    The graph is built eagerly at construction time.  Call
    :meth:`validate` to perform full correctness checks, and
    :meth:`topological_order` to obtain the validated load sequence.
    """

    def __init__(self, manifests: list[ModuleManifest]) -> None:
        # Map module_id → manifest
        self._manifests: dict[str, ModuleManifest] = {}
        for manifest in manifests:
            if manifest.id in self._manifests:
                raise ModuleValidationError(
                    f"Duplicate module id {manifest.id!r}; "
                    "each module must have a unique identifier.",
                    module_id=manifest.id,
                    field="id",
                    invalid_value=manifest.id,
                )
            self._manifests[manifest.id] = manifest

        # Adjacency list: module_id → set of module_ids that depend on it
        # (reverse edges, used for Kahn's algorithm)
        self._dependents: dict[str, set[str]] = {
            mid: set() for mid in self._manifests
        }
        # In-degree: module_id → number of unsatisfied dependencies
        self._in_degree: dict[str, int] = {
            mid: 0 for mid in self._manifests
        }

        # Build graph edges
        for manifest in manifests:
            for dep_id in manifest.dependencies:
                if dep_id in self._manifests:
                    # dep_id must finish before manifest.id can start
                    self._dependents[dep_id].add(manifest.id)
                    self._in_degree[manifest.id] += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def module_ids(self) -> frozenset[str]:
        """Set of all module IDs registered in this graph."""
        return frozenset(self._manifests)

    def manifest(self, module_id: str) -> ModuleManifest:
        """Return the manifest for *module_id*.

        Raises
        ------
        KeyError
            If *module_id* is not in the graph.
        """
        return self._manifests[module_id]

    def dependencies_of(self, module_id: str) -> frozenset[str]:
        """Return the direct dependencies of *module_id*.

        Raises
        ------
        KeyError
            If *module_id* is not in the graph.
        """
        return frozenset(self._manifests[module_id].dependencies)

    def dependents_of(self, module_id: str) -> frozenset[str]:
        """Return the modules that directly depend on *module_id*.

        Raises
        ------
        KeyError
            If *module_id* is not in the graph.
        """
        return frozenset(self._dependents.get(module_id, set()))

    def validate(self) -> None:
        """Validate the dependency graph.

        Checks performed, in order:
        1. **Missing dependencies** — every declared dependency ID must be
           present in the graph.
        2. **Self-dependencies** — a module cannot depend on itself.
        3. **Circular dependencies** — the graph must be acyclic.

        Raises
        ------
        DependencyResolutionError
            If any declared dependency is not present in the discovery set.
        CircularDependencyError
            If a cycle is detected in the dependency graph.
        ModuleValidationError
            If a module declares itself as a dependency.
        """
        self._check_missing_dependencies()
        self._check_circular_dependencies()

    def topological_order(self) -> list[str]:
        """Return module IDs in a valid load order (dependencies first).

        Uses Kahn's algorithm (BFS-based topological sort) on the dependency
        graph.  All declared dependencies are guaranteed to appear before
        their dependents in the returned list.

        Calling :meth:`validate` before this method is strongly recommended
        but not mandatory.  If the graph contains missing dependencies the
        missing modules will simply be absent from the sort result; if it
        contains cycles a :class:`~core.loader.exceptions.CircularDependencyError`
        will be raised.

        Returns
        -------
        list[str]
            Ordered list of module IDs; ready to be loaded left-to-right.

        Raises
        ------
        CircularDependencyError
            If a cycle prevents topological sorting.
        """
        return self._kahn_sort()

    def transitive_dependencies(self, module_id: str) -> frozenset[str]:
        """Return all transitive dependencies of *module_id*.

        Performs a breadth-first traversal of the dependency graph starting
        from *module_id*.  The result includes direct and indirect
        dependencies but **not** *module_id* itself.

        Parameters
        ----------
        module_id:
            The module whose transitive closure to compute.

        Returns
        -------
        frozenset[str]
            All module IDs that *module_id* depends on (directly or
            transitively).

        Raises
        ------
        KeyError
            If *module_id* is not in the graph.
        """
        visited: set[str] = set()
        queue: deque[str] = deque(self._manifests[module_id].dependencies)
        while queue:
            dep = queue.popleft()
            if dep in visited:
                continue
            visited.add(dep)
            if dep in self._manifests:
                queue.extend(self._manifests[dep].dependencies)
        return frozenset(visited)

    def __len__(self) -> int:
        """Return the number of modules in the graph."""
        return len(self._manifests)

    def __contains__(self, module_id: object) -> bool:
        """Support ``in`` operator for module ID membership tests."""
        return module_id in self._manifests

    def __iter__(self) -> Iterator[str]:
        """Iterate over module IDs in insertion order."""
        return iter(self._manifests)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_missing_dependencies(self) -> None:
        """Raise :class:`DependencyResolutionError` for any undeclared dep."""
        for manifest in self._manifests.values():
            missing = [
                dep for dep in manifest.dependencies
                if dep not in self._manifests
            ]
            if missing:
                raise DependencyResolutionError(
                    f"Module {manifest.id!r} declares "
                    f"{len(missing)} missing "
                    f"{'dependency' if len(missing) == 1 else 'dependencies'}: "
                    f"{missing}.",
                    module_id=manifest.id,
                    missing=missing,
                )

    def _check_circular_dependencies(self) -> None:
        """Raise :class:`CircularDependencyError` if any cycle exists.

        Uses Kahn's algorithm: if the topological sort cannot consume all
        nodes the remaining nodes form one or more cycles.
        """
        # Run a dry-run sort; if nodes remain, we have a cycle.
        try:
            order = self._kahn_sort(raise_on_cycle=True)
        except CircularDependencyError:
            raise
        # Extra safety: if the sort succeeded but length mismatches, something
        # is deeply wrong.
        assert len(order) == len(self._manifests), (
            "BUG: topological sort returned wrong number of nodes."
        )

    def _kahn_sort(self, *, raise_on_cycle: bool = True) -> list[str]:
        """Kahn's algorithm — BFS topological sort.

        Parameters
        ----------
        raise_on_cycle:
            If ``True`` (default), raise :class:`CircularDependencyError`
            when a cycle is detected.  If ``False``, return only the
            successfully sorted nodes.
        """
        # Work on copies so the real in-degree table is not mutated.
        in_degree = dict(self._in_degree)
        dependents = {mid: set(deps) for mid, deps in self._dependents.items()}

        # Seed the queue with all nodes that have no incoming edges (no deps).
        queue: deque[str] = deque(
            sorted(mid for mid, deg in in_degree.items() if deg == 0)
        )
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for dependent in sorted(dependents.get(node, set())):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self._manifests):
            # Cycle detected: remaining nodes are the culprits.
            cycle_nodes = [
                mid for mid in self._manifests if mid not in set(order)
            ]
            if raise_on_cycle:
                # Try to surface the shortest cycle for a clear error message.
                cycle = self._find_cycle(cycle_nodes)
                raise CircularDependencyError(
                    f"Circular dependency detected among modules: "
                    f"{cycle}. "
                    "Remove the cycle before loading.",
                    module_id=cycle[0] if cycle else cycle_nodes[0],
                    cycle=cycle or cycle_nodes,
                )

        return order

    def _find_cycle(self, candidates: list[str]) -> list[str]:
        """Return a list of module IDs forming a cycle among *candidates*.

        Uses DFS with colouring (white/grey/black) to detect back-edges.
        Returns the cycle path, or an empty list if none is found (should
        not happen after Kahn's detects a cycle, but kept for safety).
        """
        WHITE, GREY, BLACK = 0, 1, 2
        color: dict[str, int] = {mid: WHITE for mid in candidates}
        stack: list[str] = []
        cycle_found: list[str] = []

        def dfs(node: str) -> bool:
            color[node] = GREY
            stack.append(node)
            for dep in self._manifests.get(node, ModuleManifest(  # type: ignore[call-arg]
                id=node, name=node, version="0.0.0", description="-"
            )).dependencies:
                if dep not in color:
                    continue
                if color[dep] == GREY:
                    # Back-edge found; extract cycle from stack.
                    idx = stack.index(dep)
                    cycle_found.extend(stack[idx:])
                    cycle_found.append(dep)  # close the cycle
                    return True
                if color[dep] == WHITE:
                    if dfs(dep):
                        return True
            color[node] = BLACK
            stack.pop()
            return False

        for node in candidates:
            if color[node] == WHITE:
                if dfs(node):
                    break

        return cycle_found


__all__ = ["DependencyGraph"]