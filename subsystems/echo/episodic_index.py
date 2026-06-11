# subsystems/echo/episodic_index.py
"""
ECHO v1 Episodic Index Engine.

Implements :class:`EpisodicIndexEngine` — the fast-lookup index layer for
ECHO's in-process experience store.  The engine maintains a suite of
in-memory inverted and forward indices that allow O(1)–O(k) retrieval
across every structurally meaningful dimension of an :class:`Experience`.

Index Dimensions
----------------
* **Tag index**              — ``tag_name → set[experience_id]``
* **Experience-type index**  — ``ExperienceType → set[experience_id]``
* **Session index**          — ``session_id → set[experience_id]``
* **Conversation index**     — tracks CONVERSATION-typed experiences by ID
* **Project-reference index**— ``project_ref → set[experience_id]``
* **Temporal index**         — ISO-8601 date bucket (``YYYY-MM-DD``) →
                               ordered list of ``(occurred_at, experience_id)``
* **Importance-tier index**  — ``ExperienceImportance → set[experience_id]``
* **Related-memory index**   — ``experience_id → set[related_experience_id]``
                               (bidirectional: both directions indexed)
* **Forward ID index**       — ``experience_id → Experience`` (canonical copy)

Integration
-----------
Requires:
* :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
  for full-store enumeration during rebuild operations and for event-driven
  index updates.
* :class:`~subsystems.echo.interfaces.ExperienceRetrievalEngineInterface`
  for cross-engine query delegation during health checks.
* :class:`~subsystems.echo.integrity.MemoryIntegrityEngine`
  for broken-reference detection during health reporting.

Thread Safety
-------------
All public methods serialise access via ``self._lock`` (a
:class:`threading.RLock`).  Mutations are atomic at the Python GIL level
and additionally guarded by the reentrant lock so callers can safely call
index methods from multiple threads without data races.

Lifecycle
---------
The engine follows the two-phase ECHO lifecycle pattern:

1. :meth:`initialize` — allocates internal structures and performs an
   initial index build from the experience store.
2. :meth:`shutdown`   — clears all index structures and marks the engine
   as stopped.

Any public method called outside the ``running`` state raises
:class:`~subsystems.echo.exceptions.EchoNotInitializedError`.

Architecture Note
-----------------
v1 is a pure in-memory implementation.  All index state is rebuilt from
the experience store on startup and discarded on shutdown.  Future
versions will persist the index to the Memory Gateway layer and apply
incremental WAL-style updates.
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from subsystems.echo.exceptions import EchoNotInitializedError
from subsystems.echo.interfaces import (
    ExperienceEngineInterface,
    ExperienceRetrievalEngineInterface,
)
from subsystems.echo.models import (
    Experience,
    ExperienceImportance,
    ExperienceType,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal type aliases
# ---------------------------------------------------------------------------

# Temporal index entries: (iso_timestamp_str, experience_id) — kept sorted
# by the first element so bisect operations work on the string-comparable
# ISO-8601 format (``YYYY-MM-DDTHH:MM:SS+00:00``).
_TemporalEntry = tuple[str, str]


# ---------------------------------------------------------------------------
# Index Health Report
# ---------------------------------------------------------------------------


@dataclass
class IndexHealthReport:
    """Structured diagnostic produced by :meth:`EpisodicIndexEngine.health_report`.

    Attributes
    ----------
    total_indexed:
        Total number of experience IDs in the forward index.
    index_counts:
        Mapping of index name → total entry count (across all keys in that
        index).
    stale_ids:
        Experience IDs present in the forward index but no longer found in
        the backing experience store.  Non-empty indicates drift.
    missing_ids:
        Experience IDs found in the store but absent from the forward index.
        Non-empty indicates the index needs a rebuild.
    broken_related_refs:
        ``(source_id, target_id)`` pairs where a related-memory link in the
        index references a non-existent experience.
    tag_cardinality:
        Number of distinct tag names currently indexed.
    type_cardinality:
        Number of distinct :class:`ExperienceType` values with at least one
        indexed experience.
    session_cardinality:
        Number of distinct session IDs in the session index.
    project_cardinality:
        Number of distinct project reference strings in the project index.
    importance_distribution:
        Mapping of :attr:`ExperienceImportance.name` → count of indexed IDs.
    temporal_bucket_count:
        Number of distinct date buckets (``YYYY-MM-DD``) in the temporal index.
    is_healthy:
        ``True`` when ``stale_ids`` and ``missing_ids`` are both empty.
    generated_at:
        UTC timestamp of report generation.
    elapsed_seconds:
        Wall-clock time the health check took to complete.
    errors:
        Non-fatal errors encountered during the health check.
    """

    total_indexed: int = 0
    index_counts: dict[str, int] = field(default_factory=dict)
    stale_ids: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    broken_related_refs: list[tuple[str, str]] = field(default_factory=list)
    tag_cardinality: int = 0
    type_cardinality: int = 0
    session_cardinality: int = 0
    project_cardinality: int = 0
    importance_distribution: dict[str, int] = field(default_factory=dict)
    temporal_bucket_count: int = 0
    is_healthy: bool = True
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a single-line human-readable summary."""
        status = "HEALTHY" if self.is_healthy else "DEGRADED"
        return (
            f"IndexHealthReport [{status}] "
            f"indexed={self.total_indexed} "
            f"stale={len(self.stale_ids)} "
            f"missing={len(self.missing_ids)} "
            f"broken_refs={len(self.broken_related_refs)} "
            f"elapsed={self.elapsed_seconds:.3f}s"
        )

    @property
    def indexed_count(self) -> int:
        """Alias for :attr:`total_indexed` — number of experience IDs in the forward index."""
        return self.total_indexed


# ---------------------------------------------------------------------------
# Index Rebuild Report
# ---------------------------------------------------------------------------


@dataclass
class IndexRebuildReport:
    """Structured result produced by :meth:`EpisodicIndexEngine.rebuild_index`.

    Attributes
    ----------
    experiences_indexed:
        Number of :class:`Experience` records processed in this rebuild.
    index_entries_written:
        Total index entries written across all index structures.
    duration_seconds:
        Wall-clock time the rebuild took.
    rebuilt_at:
        UTC timestamp of rebuild completion.
    errors:
        Non-fatal errors encountered while indexing individual records.
    """

    experiences_indexed: int = 0
    index_entries_written: int = 0
    duration_seconds: float = 0.0
    rebuilt_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"IndexRebuildReport("
            f"indexed={self.experiences_indexed}, "
            f"entries={self.index_entries_written}, "
            f"duration={self.duration_seconds:.3f}s)"
        )

    @property
    def indexed_count(self) -> int:
        """Alias for :attr:`experiences_indexed` — number of records processed."""
        return self.experiences_indexed


# ---------------------------------------------------------------------------
# EpisodicIndexEngine
# ---------------------------------------------------------------------------


class EpisodicIndexEngine:
    """Fast-lookup index layer for ECHO's episodic experience store.

    Maintains nine independent in-memory indices across all structurally
    meaningful dimensions of an :class:`~subsystems.echo.models.Experience`.
    Provides O(1) lookup for ID-based queries and O(k) set-intersection for
    multi-dimensional filter queries where k is the size of the candidate set.

    Parameters
    ----------
    experience_engine:
        Live :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        implementation used for store enumeration and existence checks.
    retrieval_engine:
        Live :class:`~subsystems.echo.interfaces.ExperienceRetrievalEngineInterface`
        implementation used for cross-engine health checks.

    Usage
    -----
    ::

        index = EpisodicIndexEngine(
            experience_engine=exp_engine,
            retrieval_engine=ret_engine,
        )
        index.initialize()   # builds initial index from store

        # After an experience is created / updated:
        index.index_experience(experience)

        # Fast lookups:
        ids = index.lookup_by_tag("polaris")
        ids = index.lookup_by_type(ExperienceType.SESSION)
        ids = index.lookup_by_session("session-uuid")
        ids = index.lookup_by_project("polaris-v1")
        ids = index.lookup_by_importance(ExperienceImportance.CRITICAL)
        ids = index.lookup_temporal_range(start_dt, end_dt)
        ids = index.lookup_related(experience_id)

        # Compound:
        ids = index.lookup_intersection(
            tags=["polaris", "architecture"],
            experience_types=[ExperienceType.SESSION],
        )

        report = index.health_report()
        rebuild = index.rebuild_index()

        index.shutdown()
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        experience_engine: ExperienceEngineInterface,
        retrieval_engine: ExperienceRetrievalEngineInterface,
    ) -> None:
        self._experience_engine = experience_engine
        self._retrieval_engine = retrieval_engine
        self._lock = threading.RLock()
        self._running: bool = False

        # Forward index: experience_id → Experience
        self._id_index: dict[str, Experience] = {}

        # Inverted tag index: tag_name (lowercased) → set[experience_id]
        self._tag_index: dict[str, set[str]] = defaultdict(set)

        # Type index: ExperienceType → set[experience_id]
        self._type_index: dict[ExperienceType, set[str]] = defaultdict(set)

        # Session index: session_id → set[experience_id]
        # Populated from metadata.session_id on each experience.
        self._session_index: dict[str, set[str]] = defaultdict(set)

        # Conversation index: set of experience_ids whose type == CONVERSATION
        # A separate fast-lookup set so conversation queries need no type scan.
        self._conversation_ids: set[str] = set()

        # Project-reference index: project_ref → set[experience_id]
        self._project_index: dict[str, set[str]] = defaultdict(set)

        # Temporal index: date_bucket (YYYY-MM-DD) → sorted list of
        # (occurred_at_isoformat, experience_id) tuples.
        # Sorted insertion via bisect.insort so range queries are O(log n + k).
        self._temporal_index: dict[str, list[_TemporalEntry]] = defaultdict(list)

        # Importance-tier index: ExperienceImportance → set[experience_id]
        self._importance_index: dict[ExperienceImportance, set[str]] = defaultdict(set)

        # Related-memory index: experience_id → set[related_experience_id]
        # Both directions are indexed (if A relates to B, B's set also contains A).
        self._related_index: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine and build the initial index from the store.

        Fetches all experiences from the :class:`ExperienceEngineInterface`
        and indexes them.  Safe to call exactly once; calling a second time
        without an intervening :meth:`shutdown` is a no-op (logged as a
        warning).

        Raises
        ------
        EchoError
            If the experience store cannot be enumerated.
        """
        with self._lock:
            if self._running:
                _logger.warning(
                    "EpisodicIndexEngine.initialize() called while already running; "
                    "ignoring."
                )
                return

            _logger.info("EpisodicIndexEngine: initialising.")
            self._running = True

        # Build the initial index outside the lock so that long stores don't
        # block other lifecycle calls; rebuild_index re-acquires internally.
        report = self.rebuild_index()
        _logger.info(
            "EpisodicIndexEngine: initialised. %s", report
        )

    def shutdown(self) -> None:
        """Release all index structures and stop the engine.

        After this call the engine is inert.  All index data is cleared so
        that any references held externally do not prevent GC.  The engine
        may be re-initialised by calling :meth:`initialize` again.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._clear_all_indices()
            _logger.info("EpisodicIndexEngine: shut down; all indices cleared.")

    # ------------------------------------------------------------------
    # Single-experience index mutation
    # ------------------------------------------------------------------

    def index_experience(self, experience: Experience) -> None:
        """Add or update a single experience in all index structures.

        Idempotent: if the experience was previously indexed, the old entries
        are removed before the new ones are inserted.  Call this immediately
        after any create or update operation on the experience store.

        Parameters
        ----------
        experience:
            The :class:`~subsystems.echo.models.Experience` to index.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("index_experience")

        with self._lock:
            eid = experience.experience_id
            # Remove stale entries if this experience was previously indexed.
            if eid in self._id_index:
                self._remove_from_indices(self._id_index[eid])
            self._add_to_indices(experience)

    def remove_experience(self, experience_id: str) -> None:
        """Remove a single experience from all index structures.

        No-op if the experience is not currently indexed.

        Parameters
        ----------
        experience_id:
            UUID of the experience to deindex.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("remove_experience")

        with self._lock:
            if experience_id not in self._id_index:
                return
            self._remove_from_indices(self._id_index[experience_id])

    # ------------------------------------------------------------------
    # Tag Index Lookups
    # ------------------------------------------------------------------

    def lookup_by_tag(
        self,
        tag_name: str,
        *,
        category: str | None = None,
    ) -> frozenset[str]:
        """Return all experience IDs carrying the given tag name.

        Tag matching is case-insensitive exact-match on the normalised name.

        Parameters
        ----------
        tag_name:
            The tag name to look up (e.g. ``"polaris"``).
        category:
            Optional MemoryTag category filter.  When supplied, only
            experiences whose tag with *tag_name* also has this category are
            returned.

        Returns
        -------
        frozenset[str]
            Set of matching experience IDs.  Empty if the tag is unknown.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_by_tag")

        with self._lock:
            key = tag_name.strip().lower()
            candidates = frozenset(self._tag_index.get(key, set()))

        if category is None or not candidates:
            return candidates

        # Filter by category: re-inspect the live experience objects.
        with self._lock:
            result: set[str] = set()
            for eid in candidates:
                exp = self._id_index.get(eid)
                if exp is None:
                    continue
                for tag in exp.tags:
                    if tag.name.strip().lower() == key and tag.category == category:
                        result.add(eid)
                        break
            return frozenset(result)

    def lookup_by_tags(
        self,
        tag_names: list[str],
        *,
        require_all: bool = True,
    ) -> frozenset[str]:
        """Return experience IDs matching a list of tag names.

        Parameters
        ----------
        tag_names:
            Tags to match against.
        require_all:
            If ``True`` (default), return only IDs that carry **all** listed
            tags (set intersection).  If ``False``, return IDs that carry
            **any** listed tag (set union).

        Returns
        -------
        frozenset[str]
            Matching experience IDs.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_by_tags")

        if not tag_names:
            return frozenset()

        with self._lock:
            sets = [
                frozenset(self._tag_index.get(n.strip().lower(), set()))
                for n in tag_names
            ]

        if require_all:
            result = sets[0]
            for s in sets[1:]:
                result = result & s
            return result
        else:
            result = sets[0]
            for s in sets[1:]:
                result = result | s
            return result

    # ------------------------------------------------------------------
    # Experience-Type Index Lookups
    # ------------------------------------------------------------------

    def lookup_by_type(
        self,
        experience_type: ExperienceType,
    ) -> frozenset[str]:
        """Return all experience IDs of the given type.

        Parameters
        ----------
        experience_type:
            :class:`~subsystems.echo.models.ExperienceType` to filter by.

        Returns
        -------
        frozenset[str]
            Set of matching experience IDs.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_by_type")

        with self._lock:
            return frozenset(self._type_index.get(experience_type, set()))

    def lookup_by_types(
        self,
        experience_types: list[ExperienceType],
    ) -> frozenset[str]:
        """Return all experience IDs whose type is in *experience_types*.

        Parameters
        ----------
        experience_types:
            List of :class:`~subsystems.echo.models.ExperienceType` values.

        Returns
        -------
        frozenset[str]
            Union of all matched experience IDs.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_by_types")

        with self._lock:
            result: set[str] = set()
            for et in experience_types:
                result |= self._type_index.get(et, set())
            return frozenset(result)

    # ------------------------------------------------------------------
    # Session Index Lookups
    # ------------------------------------------------------------------

    def lookup_by_session(self, session_id: str) -> frozenset[str]:
        """Return all experience IDs that belong to *session_id*.

        Parameters
        ----------
        session_id:
            UUID of the session experience.

        Returns
        -------
        frozenset[str]
            Set of experience IDs whose ``metadata.session_id`` equals
            *session_id*.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_by_session")

        with self._lock:
            return frozenset(self._session_index.get(session_id, set()))

    def list_session_ids(self) -> list[str]:
        """Return all session IDs currently in the index.

        Returns
        -------
        list[str]
            Sorted list of distinct session UUIDs.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("list_session_ids")

        with self._lock:
            return sorted(self._session_index.keys())

    # ------------------------------------------------------------------
    # Conversation Index Lookups
    # ------------------------------------------------------------------

    def lookup_conversations(self) -> frozenset[str]:
        """Return all experience IDs of type ``CONVERSATION``.

        Returns
        -------
        frozenset[str]
            Set of CONVERSATION experience IDs.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_conversations")

        with self._lock:
            return frozenset(self._conversation_ids)

    # ------------------------------------------------------------------
    # Project-Reference Index Lookups
    # ------------------------------------------------------------------

    def lookup_by_project(self, project_ref: str) -> frozenset[str]:
        """Return all experience IDs tagged with *project_ref*.

        Parameters
        ----------
        project_ref:
            Project reference string as stored in
            :attr:`~subsystems.echo.models.ExperienceMetadata.project_refs`
            (e.g. ``"polaris-v1"``).

        Returns
        -------
        frozenset[str]
            Set of matching experience IDs.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_by_project")

        with self._lock:
            key = project_ref.strip().lower()
            return frozenset(self._project_index.get(key, set()))

    def list_project_refs(self) -> list[str]:
        """Return all project reference strings currently in the index.

        Returns
        -------
        list[str]
            Sorted list of distinct project reference strings.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("list_project_refs")

        with self._lock:
            return sorted(self._project_index.keys())

    # ------------------------------------------------------------------
    # Temporal Index Lookups
    # ------------------------------------------------------------------

    def lookup_temporal_range(
        self,
        start: datetime,
        end: datetime,
    ) -> frozenset[str]:
        """Return all experience IDs whose ``occurred_at`` falls in [start, end].

        Parameters
        ----------
        start:
            Inclusive lower bound (UTC).
        end:
            Inclusive upper bound (UTC).

        Returns
        -------
        frozenset[str]
            Set of matching experience IDs, unordered.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ValueError
            If *start* is after *end*.
        """
        self._assert_running("lookup_temporal_range")

        if start > end:
            raise ValueError(
                f"lookup_temporal_range: start ({start.isoformat()}) must be "
                f"<= end ({end.isoformat()})."
            )

        start_iso = _to_utc_iso(start)
        end_iso = _to_utc_iso(end)
        start_bucket = start.strftime("%Y-%m-%d")
        end_bucket = end.strftime("%Y-%m-%d")

        result: set[str] = set()

        with self._lock:
            # Collect all date buckets that overlap the range.
            for bucket, entries in self._temporal_index.items():
                if bucket < start_bucket or bucket > end_bucket:
                    continue
                # Within the bucket use bisect on the ISO timestamps.
                lo = bisect.bisect_left(entries, (start_iso, ""))
                hi = bisect.bisect_right(entries, (end_iso, "\xff\xff\xff\xff"))
                for ts_iso, eid in entries[lo:hi]:
                    result.add(eid)

        return frozenset(result)

    def lookup_by_date_bucket(self, date_bucket: str) -> frozenset[str]:
        """Return experience IDs for a specific date bucket (``YYYY-MM-DD``).

        Parameters
        ----------
        date_bucket:
            Date string in ``YYYY-MM-DD`` format.

        Returns
        -------
        frozenset[str]
            Set of experience IDs that occurred on that date (UTC).

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_by_date_bucket")

        with self._lock:
            entries = self._temporal_index.get(date_bucket, [])
            return frozenset(eid for _, eid in entries)

    def list_date_buckets(self) -> list[str]:
        """Return all date buckets that have at least one experience indexed.

        Returns
        -------
        list[str]
            Sorted list of ``YYYY-MM-DD`` strings.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("list_date_buckets")

        with self._lock:
            return sorted(self._temporal_index.keys())

    # ------------------------------------------------------------------
    # Importance-Tier Index Lookups
    # ------------------------------------------------------------------

    def lookup_by_importance(
        self,
        importance: ExperienceImportance,
    ) -> frozenset[str]:
        """Return all experience IDs at exactly *importance* tier.

        Parameters
        ----------
        importance:
            :class:`~subsystems.echo.models.ExperienceImportance` tier.

        Returns
        -------
        frozenset[str]
            Set of matching experience IDs.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_by_importance")

        with self._lock:
            return frozenset(self._importance_index.get(importance, set()))

    def lookup_by_min_importance(
        self,
        min_importance: ExperienceImportance,
    ) -> frozenset[str]:
        """Return experience IDs at or above *min_importance*.

        Parameters
        ----------
        min_importance:
            Minimum :class:`~subsystems.echo.models.ExperienceImportance` tier.

        Returns
        -------
        frozenset[str]
            Union of all IDs at ``min_importance`` or higher.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_by_min_importance")

        min_value = min_importance.value
        with self._lock:
            result: set[str] = set()
            for imp, ids in self._importance_index.items():
                if imp.value >= min_value:
                    result |= ids
            return frozenset(result)

    # ------------------------------------------------------------------
    # Related-Memory Index Lookups
    # ------------------------------------------------------------------

    def lookup_related(self, experience_id: str) -> frozenset[str]:
        """Return all experience IDs related to *experience_id*.

        Returns both directions: if A→B and C→A, returns {B, C}.

        Parameters
        ----------
        experience_id:
            UUID of the anchor experience.

        Returns
        -------
        frozenset[str]
            Set of related experience IDs.  Empty if none are indexed.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("lookup_related")

        with self._lock:
            return frozenset(self._related_index.get(experience_id, set()))

    def lookup_related_chain(
        self,
        experience_id: str,
        *,
        depth: int = 2,
    ) -> frozenset[str]:
        """Return experience IDs reachable from *experience_id* within *depth* hops.

        Performs a breadth-first traversal of the related-memory index.  The
        anchor experience itself is excluded from the result.

        Parameters
        ----------
        experience_id:
            UUID of the starting experience.
        depth:
            Number of hops to follow.  ``depth=1`` is equivalent to
            :meth:`lookup_related`.

        Returns
        -------
        frozenset[str]
            All reachable experience IDs within *depth* hops, excluding the
            anchor.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ValueError
            If *depth* is less than 1.
        """
        self._assert_running("lookup_related_chain")

        if depth < 1:
            raise ValueError(f"lookup_related_chain: depth must be >= 1, got {depth}.")

        with self._lock:
            visited: set[str] = set()
            frontier: set[str] = {experience_id}
            for _ in range(depth):
                next_frontier: set[str] = set()
                for anchor in frontier:
                    neighbours = self._related_index.get(anchor, set())
                    for n in neighbours:
                        if n not in visited and n != experience_id:
                            next_frontier.add(n)
                visited |= frontier
                frontier = next_frontier - visited

            visited.discard(experience_id)
            return frozenset(visited)

    # ------------------------------------------------------------------
    # Compound Intersection Lookup
    # ------------------------------------------------------------------

    def lookup_intersection(
        self,
        *,
        tags: list[str] | None = None,
        experience_types: list[ExperienceType] | None = None,
        session_id: str | None = None,
        project_refs: list[str] | None = None,
        min_importance: ExperienceImportance | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> frozenset[str]:
        """Return experience IDs satisfying ALL supplied filter criteria.

        Any criterion left as ``None`` is treated as unconstrained (matches
        all).  At least one criterion must be supplied.

        Parameters
        ----------
        tags:
            Experience must carry ALL listed tag names.
        experience_types:
            Experience type must be one of the listed values.
        session_id:
            Experience must belong to this session.
        project_refs:
            Experience must carry ALL listed project references.
        min_importance:
            Experience importance must be at or above this tier.
        start:
            ``occurred_at`` must be >= *start* (UTC).
        end:
            ``occurred_at`` must be <= *end* (UTC).

        Returns
        -------
        frozenset[str]
            Experience IDs satisfying all supplied criteria.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ValueError
            If no criteria are supplied, or if *start* > *end*.
        """
        self._assert_running("lookup_intersection")

        criteria_supplied = any(
            v is not None
            for v in (tags, experience_types, session_id, project_refs,
                      min_importance, start, end)
        )
        if not criteria_supplied:
            raise ValueError(
                "lookup_intersection: at least one filter criterion must be supplied."
            )

        # Build candidate sets per criterion, then intersect.
        candidate_sets: list[frozenset[str]] = []

        if tags:
            candidate_sets.append(self.lookup_by_tags(tags, require_all=True))

        if experience_types:
            candidate_sets.append(self.lookup_by_types(experience_types))

        if session_id is not None:
            candidate_sets.append(self.lookup_by_session(session_id))

        if project_refs:
            # All project refs must be present (intersection).
            proj_sets = [self.lookup_by_project(ref) for ref in project_refs]
            combined = proj_sets[0]
            for ps in proj_sets[1:]:
                combined = combined & ps
            candidate_sets.append(combined)

        if min_importance is not None:
            candidate_sets.append(self.lookup_by_min_importance(min_importance))

        if start is not None and end is not None:
            candidate_sets.append(self.lookup_temporal_range(start, end))
        elif start is not None:
            # No upper bound: temporal range from start to far future.
            candidate_sets.append(
                self.lookup_temporal_range(start, datetime.max.replace(tzinfo=timezone.utc))
            )
        elif end is not None:
            # No lower bound: temporal range from far past to end.
            candidate_sets.append(
                self.lookup_temporal_range(datetime.min.replace(tzinfo=timezone.utc), end)
            )

        if not candidate_sets:
            return frozenset()

        result = candidate_sets[0]
        for s in candidate_sets[1:]:
            result = result & s
        return result

    # ------------------------------------------------------------------
    # Forward ID Lookup
    # ------------------------------------------------------------------

    def get_indexed_experience(self, experience_id: str) -> Experience | None:
        """Return the cached :class:`Experience` from the forward index.

        This returns the engine's copy of the experience as of the last
        :meth:`index_experience` call.  Callers that need the freshest
        version from the store should use the ExperienceEngine directly.

        Parameters
        ----------
        experience_id:
            UUID to look up.

        Returns
        -------
        Experience | None
            The cached experience, or ``None`` if not indexed.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("get_indexed_experience")

        with self._lock:
            return self._id_index.get(experience_id)

    def get_all_indexed_ids(self) -> frozenset[str]:
        """Return all experience IDs currently in the forward index.

        Returns
        -------
        frozenset[str]
            Complete set of indexed IDs.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("get_all_indexed_ids")

        with self._lock:
            return frozenset(self._id_index.keys())

    def is_indexed(self, experience_id: str) -> bool:
        """Return ``True`` if *experience_id* is present in the forward index.

        Parameters
        ----------
        experience_id:
            UUID to check.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("is_indexed")

        with self._lock:
            return experience_id in self._id_index

    # ------------------------------------------------------------------
    # Index Rebuild
    # ------------------------------------------------------------------

    def rebuild_index(self) -> IndexRebuildReport:
        """Perform a full index rebuild by re-enumerating the experience store.

        Clears all existing index structures, then fetches every experience
        from the :class:`ExperienceEngineInterface` and re-indexes them.
        Safe to call on a running engine — the index is unavailable for
        the duration of the rebuild (callers blocked by the lock).

        Returns
        -------
        IndexRebuildReport
            Summary of the rebuild operation.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("rebuild_index")

        t0 = time.monotonic()
        errors: list[str] = []
        entries_written = 0

        with self._lock:
            self._clear_all_indices()

            try:
                all_experiences = self._experience_engine.list_experiences()
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "EpisodicIndexEngine: failed to enumerate experience store "
                    "during rebuild: %s", exc
                )
                errors.append(f"Store enumeration failed: {exc}")
                all_experiences = []

            for exp in all_experiences:
                try:
                    written = self._add_to_indices(exp)
                    entries_written += written
                except Exception as exc:  # noqa: BLE001
                    msg = (
                        f"Failed to index experience "
                        f"'{exp.experience_id}': {exc}"
                    )
                    _logger.warning("EpisodicIndexEngine: %s", msg)
                    errors.append(msg)

            report = IndexRebuildReport(
                experiences_indexed=len(all_experiences),
                index_entries_written=entries_written,
                duration_seconds=time.monotonic() - t0,
                rebuilt_at=datetime.now(timezone.utc),
                errors=errors,
            )

        _logger.info("EpisodicIndexEngine rebuild complete: %s", report)
        return report

    # ------------------------------------------------------------------
    # Health Reporting
    # ------------------------------------------------------------------

    def health_report(self) -> IndexHealthReport:
        """Generate a comprehensive index health report.

        Compares the forward index against the live experience store to
        detect drift (stale or missing entries).  Also validates all
        related-memory links for broken references.

        Returns
        -------
        IndexHealthReport
            Structured health report.  :attr:`IndexHealthReport.is_healthy`
            is ``True`` only when no stale IDs, missing IDs, or broken
            related references are detected.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("health_report")

        t0 = time.monotonic()
        errors: list[str] = []

        with self._lock:
            indexed_ids = set(self._id_index.keys())

        # Fetch live IDs from the store (outside lock to avoid deadlock with
        # experience engine callbacks that may re-enter index methods).
        try:
            live_experiences = self._experience_engine.list_experiences()
            live_ids = {e.experience_id for e in live_experiences}
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "EpisodicIndexEngine.health_report: cannot enumerate store: %s", exc
            )
            errors.append(f"Store enumeration failed: {exc}")
            live_ids = set()

        stale_ids = sorted(indexed_ids - live_ids)
        missing_ids = sorted(live_ids - indexed_ids)

        with self._lock:
            # Broken related-memory references.
            broken_refs: list[tuple[str, str]] = []
            for src_id, targets in self._related_index.items():
                for tgt_id in targets:
                    if tgt_id not in self._id_index:
                        broken_refs.append((src_id, tgt_id))

            # Index entry counts.
            index_counts = {
                "id_index": len(self._id_index),
                "tag_index": sum(len(v) for v in self._tag_index.values()),
                "type_index": sum(len(v) for v in self._type_index.values()),
                "session_index": sum(len(v) for v in self._session_index.values()),
                "conversation_ids": len(self._conversation_ids),
                "project_index": sum(len(v) for v in self._project_index.values()),
                "temporal_index": sum(
                    len(v) for v in self._temporal_index.values()
                ),
                "importance_index": sum(
                    len(v) for v in self._importance_index.values()
                ),
                "related_index": sum(
                    len(v) for v in self._related_index.values()
                ),
            }

            importance_distribution = {
                imp.name: len(ids)
                for imp, ids in self._importance_index.items()
            }

            report = IndexHealthReport(
                total_indexed=len(self._id_index),
                index_counts=index_counts,
                stale_ids=stale_ids,
                missing_ids=missing_ids,
                broken_related_refs=broken_refs,
                tag_cardinality=len(self._tag_index),
                type_cardinality=sum(
                    1 for v in self._type_index.values() if v
                ),
                session_cardinality=len(self._session_index),
                project_cardinality=len(self._project_index),
                importance_distribution=importance_distribution,
                temporal_bucket_count=len(self._temporal_index),
                is_healthy=(
                    not stale_ids and not missing_ids and not broken_refs
                ),
                generated_at=datetime.now(timezone.utc),
                elapsed_seconds=time.monotonic() - t0,
                errors=errors,
            )

        _logger.info("EpisodicIndexEngine: %s", report.summary())
        return report

    # ------------------------------------------------------------------
    # Diagnostic Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a lightweight diagnostic snapshot of the engine's state.

        Returns
        -------
        dict[str, Any]
            Contains ``running``, ``engine``, ``total_indexed``,
            and index size counters for quick monitoring.
        """
        with self._lock:
            return {
                "engine": "EpisodicIndexEngine",
                "running": self._running,
                "total_indexed": len(self._id_index),
                "tag_keys": len(self._tag_index),
                "type_keys": len(self._type_index),
                "session_keys": len(self._session_index),
                "conversation_count": len(self._conversation_ids),
                "project_keys": len(self._project_index),
                "temporal_bucket_keys": len(self._temporal_index),
                "importance_keys": len(self._importance_index),
                "related_keys": len(self._related_index),
            }

    # ------------------------------------------------------------------
    # Internal helpers — index mutation
    # ------------------------------------------------------------------

    def _add_to_indices(self, experience: Experience) -> int:
        """Index *experience* into all structures.  Returns count of entries written.

        Must be called with ``self._lock`` held.
        """
        eid = experience.experience_id
        entries = 0

        # Forward index.
        self._id_index[eid] = experience
        entries += 1

        # Tag index.
        for tag in experience.tags:
            key = tag.name.strip().lower()
            self._tag_index[key].add(eid)
            entries += 1

        # Type index.
        self._type_index[experience.experience_type].add(eid)
        entries += 1

        # Conversation shortcut.
        if experience.experience_type == ExperienceType.CONVERSATION:
            self._conversation_ids.add(eid)
            entries += 1

        # Session index.
        if experience.metadata.session_id:
            self._session_index[experience.metadata.session_id].add(eid)
            entries += 1

        # Project-reference index.
        for proj_ref in experience.metadata.project_refs:
            key = proj_ref.strip().lower()
            if key:
                self._project_index[key].add(eid)
                entries += 1

        # Temporal index.
        bucket = experience.occurred_at.strftime("%Y-%m-%d")
        entry: _TemporalEntry = (_to_utc_iso(experience.occurred_at), eid)
        bucket_list = self._temporal_index[bucket]
        # bisect.insort keeps the list sorted by (iso_ts, eid).
        bisect.insort(bucket_list, entry)
        entries += 1

        # Importance-tier index.
        importance_key = (
            experience.importance
            if isinstance(experience.importance, ExperienceImportance)
            else ExperienceImportance(experience.importance)
        )
        self._importance_index[importance_key].add(eid)
        entries += 1

        # Related-memory index — bidirectional.
        for related_id in experience.metadata.related_experience_ids:
            if related_id == eid:
                continue  # skip self-referential links silently
            self._related_index[eid].add(related_id)
            self._related_index[related_id].add(eid)
            entries += 2

        return entries

    def _remove_from_indices(self, experience: Experience) -> None:
        """Remove *experience* from all index structures.

        Must be called with ``self._lock`` held.
        """
        eid = experience.experience_id

        # Forward index.
        self._id_index.pop(eid, None)

        # Tag index.
        for tag in experience.tags:
            key = tag.name.strip().lower()
            self._tag_index[key].discard(eid)
            if not self._tag_index[key]:
                del self._tag_index[key]

        # Type index.
        self._type_index[experience.experience_type].discard(eid)

        # Conversation shortcut.
        self._conversation_ids.discard(eid)

        # Session index.
        if experience.metadata.session_id:
            sid = experience.metadata.session_id
            self._session_index[sid].discard(eid)
            if not self._session_index[sid]:
                del self._session_index[sid]

        # Project-reference index.
        for proj_ref in experience.metadata.project_refs:
            key = proj_ref.strip().lower()
            if key:
                self._project_index[key].discard(eid)
                if not self._project_index[key]:
                    del self._project_index[key]

        # Temporal index.
        bucket = experience.occurred_at.strftime("%Y-%m-%d")
        iso_ts = _to_utc_iso(experience.occurred_at)
        entry: _TemporalEntry = (iso_ts, eid)
        bucket_list = self._temporal_index.get(bucket)
        if bucket_list is not None:
            pos = bisect.bisect_left(bucket_list, entry)
            if pos < len(bucket_list) and bucket_list[pos] == entry:
                bucket_list.pop(pos)
            if not bucket_list:
                del self._temporal_index[bucket]

        # Importance-tier index.
        importance_key = (
            experience.importance
            if isinstance(experience.importance, ExperienceImportance)
            else ExperienceImportance(experience.importance)
        )
        self._importance_index[importance_key].discard(eid)

        # Related-memory index — bidirectional removal.
        # Remove the outbound edges from eid's own set.
        related_targets = self._related_index.pop(eid, set())
        # Remove eid from every target's inbound set.
        for related_id in related_targets:
            self._related_index[related_id].discard(eid)
            if not self._related_index[related_id]:
                del self._related_index[related_id]
        # Remove eid from any set where it appeared as an inbound edge.
        for src_id, targets in list(self._related_index.items()):
            targets.discard(eid)
            if not targets:
                del self._related_index[src_id]

    def _clear_all_indices(self) -> None:
        """Wipe every index structure.  Must be called with ``self._lock`` held."""
        self._id_index.clear()
        self._tag_index.clear()
        self._type_index.clear()
        self._session_index.clear()
        self._conversation_ids.clear()
        self._project_index.clear()
        self._temporal_index.clear()
        self._importance_index.clear()
        self._related_index.clear()

    # ------------------------------------------------------------------
    # Lifecycle guard
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if the engine is not running."""
        if not self._running:
            raise EchoNotInitializedError(operation)


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------


def _to_utc_iso(dt: datetime) -> str:
    """Normalise *dt* to UTC and return an ISO-8601 string suitable for
    lexicographic comparison in the temporal index.

    Naïve datetimes are assumed to be UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()