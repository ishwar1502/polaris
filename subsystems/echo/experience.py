# subsystems/echo/experience.py
"""
ECHO v1 Experience Engine.

Implements :class:`ExperienceEngine` — the production implementation of
:class:`~subsystems.echo.interfaces.ExperienceEngineInterface`.

The Experience Engine is ECHO's primary storage and lifecycle manager.
It creates, stores, updates, deletes, and retrieves :class:`Experience`
objects.  Every write operation consults the Significance Engine to
ensure that only meaningful experiences enter ECHO's store (unless the
caller explicitly forces storage).

Thread safety
-------------
All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
before modifying internal state.  This makes the engine safe to use from
multiple threads simultaneously, as expected in the POLARIS runtime.

Architecture notes
------------------
* In v1, the backing store is an in-process ``dict[str, Experience]``
  keyed by ``experience_id``.  The Memory Gateway integration for
  persistent storage is wired in a future iteration.
* The engine integrates with the Event Bus to publish domain events after
  successful lifecycle operations.
* The Significance Engine reference is injected at construction time via
  the ``significance_engine`` parameter, so the dependency is explicit
  and testable.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from subsystems.echo.exceptions import (
    EchoBoundaryViolationError,
    EchoNotInitializedError,
    ExperienceDuplicateError,
    ExperienceNotFoundError,
    ExperienceStorageError,
    ExperienceValidationError,
)
from subsystems.echo.interfaces import ExperienceEngineInterface, SignificanceEngineInterface
from subsystems.echo.models import (
    AchievementRecord,
    EventRecord,
    Experience,
    ExperienceImportance,
    ExperienceMetadata,
    ExperienceType,
    FailureRecord,
    MemoryTag,
    ObservationRecord,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ECHO boundary — data types this engine refuses to own
# ---------------------------------------------------------------------------

_BOUNDARY_VIOLATIONS: dict[str, str] = {
    "knowledge": "LUNA",
    "fact": "LUNA",
    "identity": "ASTRA",
    "goal": "ODYSSEY",
    "schedule": "CHRONOS",
    "relationship": "ASTRA",
    "decision": "ORION",
    "plan": "ODYSSEY",
}


class ExperienceEngine(ExperienceEngineInterface):
    """Production implementation of the ECHO Experience Engine.

    Parameters
    ----------
    significance_engine:
        A running :class:`~subsystems.echo.interfaces.SignificanceEngineInterface`
        used to score every candidate experience before storage.
    min_significance_threshold:
        Minimum score [0.0, 1.0] required for automatic storage.  Defaults
        to ``0.15`` — a deliberately low floor so that callers retain
        control while preventing pure noise.

    Usage
    -----
    ::

        sig_engine = SignificanceEngine()
        sig_engine.initialize()

        eng = ExperienceEngine(significance_engine=sig_engine)
        eng.initialize()

        exp = eng.create_experience(
            title="Completed POLARIS architecture freeze",
            experience_type=ExperienceType.ACHIEVEMENT,
            importance=ExperienceImportance.HIGH,
            description="Finalised the v5 frozen specification.",
            source_subsystem="ECHO_API",
        )

        eng.shutdown()
        sig_engine.shutdown()
    """

    def __init__(
        self,
        significance_engine: SignificanceEngineInterface,
        *,
        min_significance_threshold: float = 0.15,
    ) -> None:
        if not (0.0 <= min_significance_threshold <= 1.0):
            raise ValueError(
                "min_significance_threshold must be in [0.0, 1.0]; "
                f"got {min_significance_threshold!r}."
            )

        self._significance_engine = significance_engine
        self._min_significance_threshold = min_significance_threshold

        # In-process store: experience_id → Experience
        self._store: dict[str, Experience] = {}
        self._lock = threading.RLock()
        self._running = False

        _logger.debug("ExperienceEngine constructed (threshold=%.3f).", min_significance_threshold)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Sets the running flag and logs readiness.  In a future iteration
        this will connect to the Memory Gateway persistence layer.
        """
        with self._lock:
            if self._running:
                _logger.warning("ExperienceEngine.initialize() called while already running.")
                return
            self._running = True
            _logger.info("ExperienceEngine initialised.")

    def shutdown(self) -> None:
        """Release resources and mark the engine as stopped."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            _logger.info(
                "ExperienceEngine shut down. Store contained %d experience(s).",
                len(self._store),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if not running."""
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _validate_title(self, title: str, field: str = "title") -> None:
        """Raise :class:`ExperienceValidationError` if title is blank."""
        if not title or not title.strip():
            raise ExperienceValidationError(
                f"Experience.{field} must be a non-empty string.",
                field=field,
            )

    def _check_boundary(self, title: str, description: str) -> None:
        """Raise :class:`EchoBoundaryViolationError` for disallowed data types.

        Performs a lightweight keyword scan of the title and description.
        This is a best-effort guard; the full boundary enforcement layer
        will be added in a future iteration.
        """
        combined = (title + " " + description).lower()
        for keyword, owner in _BOUNDARY_VIOLATIONS.items():
            if combined.startswith(keyword + " ") or f" {keyword} " in combined:
                raise EchoBoundaryViolationError(keyword, owner)

    def _apply_significance(
        self,
        experience: Experience,
        force: bool,
    ) -> Experience:
        """Consult the Significance Engine and update the experience in place.

        If ``force=True`` the score is computed but the threshold check is
        skipped.  The experience's ``metadata.significance_score`` and its
        ``importance`` tier are updated from the engine's evaluation.

        Returns
        -------
        Experience
            The same object, with updated metadata and importance.

        Raises
        ------
        BelowSignificanceThresholdError
            If score < threshold and ``force=False``.
        """
        from subsystems.echo.exceptions import BelowSignificanceThresholdError

        result = self._significance_engine.evaluate(experience)

        # Update metadata with the computed score
        experience.metadata.significance_score = result.score

        # Override the caller-supplied importance with the engine's classification
        experience.importance = result.importance

        if not force and not result.eligible_for_storage:
            raise BelowSignificanceThresholdError(
                experience.experience_id,
                result.score,
                self._min_significance_threshold,
            )

        return experience

    def _integrity_check_on_store(self, experience: Experience) -> None:
        """Guard against duplicate experiences on insertion."""
        if experience.experience_id in self._store:
            raise ExperienceDuplicateError(experience.experience_id)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_experience(
        self,
        title: str,
        experience_type: ExperienceType,
        importance: ExperienceImportance,
        *,
        description: str = "",
        context: str = "",
        outcome: str = "",
        tags: list[MemoryTag] | None = None,
        occurred_at: datetime | None = None,
        source_subsystem: str = "ECHO_API",
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Experience:
        """Create and persist a new :class:`Experience`.

        See :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        for full parameter documentation.
        """
        self._assert_running("create_experience")
        self._validate_title(title)
        self._check_boundary(title, description)

        now = datetime.now(timezone.utc)
        metadata = ExperienceMetadata(
            source_subsystem=source_subsystem,
            session_id=session_id,
        )
        experience = Experience(
            title=title.strip(),
            experience_type=experience_type,
            importance=importance,
            description=description,
            context=context,
            outcome=outcome,
            tags=list(tags) if tags else [],
            occurred_at=occurred_at if occurred_at is not None else now,
            recorded_at=now,
            metadata=metadata,
            extra=dict(extra) if extra else {},
        )

        with self._lock:
            self._integrity_check_on_store(experience)
            experience = self._apply_significance(experience, force)
            self._store[experience.experience_id] = experience

        _logger.info(
            "ExperienceEngine: stored experience '%s' (id=%s, importance=%s, score=%.3f).",
            experience.title,
            experience.experience_id,
            experience.importance.name,
            experience.metadata.significance_score,
        )
        self._publish_experience_stored(experience)
        return experience

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_experience(
        self,
        experience: Experience,
        *,
        force: bool = False,
    ) -> Experience:
        """Persist an already-constructed :class:`Experience`.

        See :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        for full parameter documentation.
        """
        self._assert_running("store_experience")

        if experience is None:
            raise ExperienceValidationError("experience must not be None.", field="experience")
        if not experience.title or not experience.title.strip():
            raise ExperienceValidationError("Experience.title must be non-empty.", field="title")

        with self._lock:
            self._integrity_check_on_store(experience)
            experience = self._apply_significance(experience, force)
            self._store[experience.experience_id] = experience

        _logger.info(
            "ExperienceEngine: stored pre-built experience '%s' (id=%s).",
            experience.title,
            experience.experience_id,
        )
        self._publish_experience_stored(experience)
        return experience

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_experience(
        self,
        experience_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        context: str | None = None,
        outcome: str | None = None,
        importance: ExperienceImportance | None = None,
        tags: list[MemoryTag] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Experience:
        """Update mutable fields of an existing :class:`Experience`.

        See :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        for full parameter documentation.
        """
        self._assert_running("update_experience")

        with self._lock:
            if experience_id not in self._store:
                raise ExperienceNotFoundError(experience_id)

            exp = self._store[experience_id]

            if title is not None:
                self._validate_title(title)
                exp.title = title.strip()

            if description is not None:
                exp.description = description

            if context is not None:
                exp.context = context

            if outcome is not None:
                exp.outcome = outcome

            if importance is not None:
                exp.importance = importance

            if tags is not None:
                exp.tags = list(tags)

            if extra is not None:
                exp.extra = dict(extra)

        _logger.debug(
            "ExperienceEngine: updated experience id=%s.",
            experience_id,
        )
        return exp

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_experience(
        self,
        experience_id: str,
        *,
        cascade: bool = False,
    ) -> bool:
        """Remove an experience from ECHO's store.

        CRITICAL experiences are protected and cannot be deleted.

        See :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        for full parameter documentation.
        """
        from subsystems.echo.exceptions import ExperienceError

        self._assert_running("delete_experience")

        with self._lock:
            exp = self._store.get(experience_id)
            if exp is None:
                return False  # idempotent

            if exp.is_permanent():
                raise ExperienceError(
                    f"Experience '{experience_id}' is CRITICAL and cannot be deleted. "
                    "CRITICAL experiences are permanent records."
                )

            del self._store[experience_id]

            if cascade:
                # In v1 there is no separate event/achievement/failure store;
                # cascade is a no-op here but the parameter is reserved for
                # the Memory Gateway integration in a future iteration.
                _logger.debug(
                    "ExperienceEngine: cascade delete requested for id=%s "
                    "(no child stores in v1).",
                    experience_id,
                )

        _logger.info("ExperienceEngine: deleted experience id=%s.", experience_id)
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_experience(self, experience_id: str) -> Experience:
        """Retrieve a single experience by UUID.

        Increments the retrieval counter on the returned experience.
        """
        self._assert_running("get_experience")

        with self._lock:
            exp = self._store.get(experience_id)
            if exp is None:
                raise ExperienceNotFoundError(experience_id)
            exp.metadata.record_retrieval()

        return exp

    def query_experiences(
        self,
        *,
        experience_type: ExperienceType | None = None,
        importance: ExperienceImportance | None = None,
        min_importance: ExperienceImportance | None = None,
        tags: list[str] | None = None,
        source_subsystem: str | None = None,
        session_id: str | None = None,
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        consolidated: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Experience]:
        """Return experiences matching the supplied filter criteria.

        See :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        for full parameter documentation.
        """
        self._assert_running("query_experiences")

        if limit < 0:
            raise ExperienceValidationError("limit must be >= 0.", field="limit")
        if offset < 0:
            raise ExperienceValidationError("offset must be >= 0.", field="offset")

        tag_set = set(tags) if tags else None

        with self._lock:
            results: list[Experience] = []

            for exp in self._store.values():
                # Type filter
                if experience_type is not None and exp.experience_type != experience_type:
                    continue

                # Exact importance filter
                if importance is not None and exp.importance != importance:
                    continue

                # Minimum importance filter (using the .value attribute of the enum)
                if (
                    min_importance is not None
                    and exp.importance.value < min_importance.value
                ):
                    continue

                # Tag filter (all supplied tags must be present)
                if tag_set and not tag_set.issubset(set(exp.tag_names())):
                    continue

                # Source subsystem filter
                if (
                    source_subsystem is not None
                    and exp.metadata.source_subsystem != source_subsystem
                ):
                    continue

                # Session filter
                if session_id is not None and exp.metadata.session_id != session_id:
                    continue

                # Time range filters
                if occurred_after is not None and exp.occurred_at < occurred_after:
                    continue
                if occurred_before is not None and exp.occurred_at > occurred_before:
                    continue

                # Consolidation filter
                if consolidated is not None and exp.metadata.consolidated != consolidated:
                    continue

                results.append(exp)

        # Sort by occurred_at descending (most recent first)
        results.sort(key=lambda e: e.occurred_at, reverse=True)

        # Pagination
        return results[offset : offset + limit]

    def count_experiences(
        self,
        *,
        experience_type: ExperienceType | None = None,
        importance: ExperienceImportance | None = None,
        consolidated: bool | None = None,
    ) -> int:
        """Return the count of stored experiences matching the given filters.

        See :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        for full parameter documentation.
        """
        self._assert_running("count_experiences")

        with self._lock:
            count = 0
            for exp in self._store.values():
                if experience_type is not None and exp.experience_type != experience_type:
                    continue
                if importance is not None and exp.importance != importance:
                    continue
                if consolidated is not None and exp.metadata.consolidated != consolidated:
                    continue
                count += 1

        return count

    def experience_exists(self, experience_id: str) -> bool:
        """Return whether an experience with the given UUID exists."""
        self._assert_running("experience_exists")

        with self._lock:
            return experience_id in self._store

    # ------------------------------------------------------------------
    # Bulk / Utility
    # ------------------------------------------------------------------

    def get_recent_experiences(
        self,
        limit: int = 20,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
    ) -> list[Experience]:
        """Return the most recently recorded experiences.

        See :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        for full parameter documentation.
        """
        self._assert_running("get_recent_experiences")

        with self._lock:
            filtered = [
                exp
                for exp in self._store.values()
                if exp.importance.value >= min_importance.value
            ]

        filtered.sort(key=lambda e: e.recorded_at, reverse=True)
        return filtered[:limit]

    def get_significant_experiences(
        self,
        limit: int = 50,
    ) -> list[Experience]:
        """Return experiences above the LOW importance threshold.

        See :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        for full parameter documentation.
        """
        self._assert_running("get_significant_experiences")

        with self._lock:
            significant = [
                exp
                for exp in self._store.values()
                if exp.is_significant()
            ]

        # Sort by significance_score descending, then recency as tiebreaker
        significant.sort(
            key=lambda e: (e.metadata.significance_score, e.occurred_at.timestamp()),
            reverse=True,
        )
        return significant[:limit]

    # ------------------------------------------------------------------
    # Domain event publishing
    # ------------------------------------------------------------------

    def _publish_experience_stored(self, experience: Experience) -> None:
        """Emit the ``ExperienceStored`` domain event.

        In v1, this logs at DEBUG level.  The Event Bus integration will
        be wired in a future iteration once the bus interface is available
        in this subsystem's dependency set.
        """
        _logger.debug(
            "EVENT ExperienceStored: id=%s title=%r importance=%s",
            experience.experience_id,
            experience.title,
            experience.importance.name,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of the engine's current state.

        Useful for health checks, debugging, and test assertions.

        Returns
        -------
        dict[str, Any]
            Contains ``total``, ``by_type``, ``by_importance``,
            ``consolidated_count``, and ``running``.
        """
        with self._lock:
            total = len(self._store)
            by_type: dict[str, int] = {}
            by_importance: dict[str, int] = {}
            consolidated_count = 0

            for exp in self._store.values():
                type_key = exp.experience_type.name
                imp_key = exp.importance.name
                by_type[type_key] = by_type.get(type_key, 0) + 1
                by_importance[imp_key] = by_importance.get(imp_key, 0) + 1
                if exp.metadata.consolidated:
                    consolidated_count += 1

        return {
            "running": self._running,
            "total": total,
            "by_type": by_type,
            "by_importance": by_importance,
            "consolidated_count": consolidated_count,
            "threshold": self._min_significance_threshold,
        }