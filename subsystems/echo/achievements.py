# subsystems/echo/achievement.py
"""
ECHO v1 Achievement Engine.

Implements :class:`AchievementEngine` — the production engine responsible
for creating, storing, updating, deleting, and retrieving
:class:`~subsystems.echo.models.AchievementRecord` objects within ECHO.

Achievement records represent completed accomplishments: shipped features,
passed exams, finished semesters, released versions, built new tools.  They
are first-class citizens of ECHO and link back to parent
:class:`~subsystems.echo.models.Experience` objects when one exists.

Thread safety
-------------
All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
before reading or modifying internal state.  The engine is safe for
concurrent use across multiple threads as required by the POLARIS runtime.

Architecture notes
------------------
* In v1, the backing store is an in-process ``dict[str, AchievementRecord]``
  keyed by ``achievement_id``.
* The Experience Engine reference is injected at construction time.
  Achievement records may optionally be linked to an existing Experience,
  but a corresponding Experience is not required.
* Domain events are published (logged at DEBUG in v1) after every successful
  lifecycle mutation.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from subsystems.echo.exceptions import (
    AchievementNotFoundError,
    AchievementValidationError,
    EchoNotInitializedError,
    ExperienceNotFoundError,
)
from subsystems.echo.models import (
    AchievementRecord,
    ExperienceImportance,
    MemoryTag,
)

_logger = logging.getLogger(__name__)


class AchievementEngine:
    """Production implementation of the ECHO Achievement Engine.

    Tracks accomplishments throughout the user's journey.  Each
    :class:`~subsystems.echo.models.AchievementRecord` captures what was
    accomplished, the domain it belongs to, observable evidence, and an
    optional link to a parent :class:`~subsystems.echo.models.Experience`.

    Parameters
    ----------
    experience_engine:
        Optional reference to the running ExperienceEngine.  When supplied,
        :meth:`link_experience` can validate that the target experience exists
        before creating the association.  Pass ``None`` to operate in
        standalone mode (no cross-engine validation).

    Usage
    -----
    ::

        engine = AchievementEngine()
        engine.initialize()

        record = engine.create_achievement(
            title="Released POLARIS v1",
            domain="software",
            description="Shipped the first public release.",
            evidence=["https://github.com/org/polaris/releases/v1"],
            importance=ExperienceImportance.CRITICAL,
        )

        engine.shutdown()
    """

    def __init__(
        self,
        experience_engine: Any | None = None,
    ) -> None:
        self._experience_engine = experience_engine
        self._store: dict[str, AchievementRecord] = {}
        self._lock = threading.RLock()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Must be called before any other method.  Idempotent: calling
        ``initialize()`` on an already-running engine is a no-op.
        """
        with self._lock:
            if self._running:
                return
            self._store.clear()
            self._running = True
            _logger.info("AchievementEngine initialised.")

    def shutdown(self) -> None:
        """Release all resources held by this engine.

        Flushes internal state and marks the engine as stopped.  Calling
        any public method after ``shutdown()`` will raise
        :class:`~subsystems.echo.exceptions.EchoNotInitializedError`.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            _logger.info(
                "AchievementEngine shut down. Stored %d achievement(s).",
                len(self._store),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _validate_title(self, title: str, field: str = "title") -> None:
        if not title or not title.strip():
            raise AchievementValidationError(
                f"AchievementRecord.{field} must be a non-empty string.",
                field=field,
            )

    def _publish_event(self, event_name: str, record: AchievementRecord) -> None:
        _logger.debug(
            "EVENT %s: achievement_id=%s title=%r domain=%s importance=%s",
            event_name,
            record.achievement_id,
            record.title,
            record.domain,
            record.importance.name,
        )

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_achievement(
        self,
        title: str,
        *,
        domain: str = "general",
        description: str = "",
        evidence: list[str] | None = None,
        importance: ExperienceImportance = ExperienceImportance.HIGH,
        tags: list[MemoryTag] | None = None,
        experience_id: str | None = None,
        achieved_at: datetime | None = None,
    ) -> AchievementRecord:
        """Create and persist a new :class:`~subsystems.echo.models.AchievementRecord`.

        Parameters
        ----------
        title:
            Short, human-readable label for the achievement.
        domain:
            Broad domain category (e.g. ``"software"``, ``"academic"``,
            ``"personal"``).  Defaults to ``"general"``.
        description:
            Narrative of what was accomplished and why it matters.
        evidence:
            Observable proof items (e.g. commit hash, exam result, URL).
        importance:
            :class:`~subsystems.echo.models.ExperienceImportance` tier.
            Defaults to HIGH since achievements are inherently significant.
        tags:
            Optional list of :class:`~subsystems.echo.models.MemoryTag`
            objects for indexing.
        experience_id:
            Optional UUID of the parent Experience to link this record to.
        achieved_at:
            UTC timestamp of the achievement moment.  Defaults to now.

        Returns
        -------
        AchievementRecord
            The created and stored achievement record.

        Raises
        ------
        AchievementValidationError
            If ``title`` is empty.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("create_achievement")
        self._validate_title(title)

        record = AchievementRecord(
            title=title.strip(),
            experience_id=experience_id,
            domain=domain or "general",
            description=description,
            evidence=list(evidence) if evidence else [],
            importance=importance,
            tags=list(tags) if tags else [],
            achieved_at=achieved_at or datetime.now(timezone.utc),
            recorded_at=datetime.now(timezone.utc),
        )

        return self.store_achievement(record)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_achievement(self, record: AchievementRecord) -> AchievementRecord:
        """Persist an already-constructed :class:`~subsystems.echo.models.AchievementRecord`.

        Parameters
        ----------
        record:
            The achievement record to persist.  Its ``achievement_id`` must
            not already exist in the store.

        Returns
        -------
        AchievementRecord
            The stored record (mutated ``recorded_at`` if not already set).

        Raises
        ------
        AchievementValidationError
            If a record with the same ``achievement_id`` already exists.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("store_achievement")

        if not isinstance(record, AchievementRecord):
            raise AchievementValidationError(
                "store_achievement requires an AchievementRecord instance."
            )

        with self._lock:
            if record.achievement_id in self._store:
                raise AchievementValidationError(
                    f"Achievement '{record.achievement_id}' already exists. "
                    "Use update_achievement() to modify it.",
                    field="achievement_id",
                )
            self._store[record.achievement_id] = record

        self._publish_event("AchievementStored", record)
        _logger.info(
            "Stored achievement '%s' (%s).",
            record.title,
            record.achievement_id,
        )
        return record

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_achievement(
        self,
        achievement_id: str,
        *,
        title: str | None = None,
        domain: str | None = None,
        description: str | None = None,
        evidence: list[str] | None = None,
        importance: ExperienceImportance | None = None,
        tags: list[MemoryTag] | None = None,
    ) -> AchievementRecord:
        """Update mutable fields of an existing achievement record.

        Only fields supplied as non-``None`` are modified.  ``achievement_id``,
        ``experience_id``, and ``recorded_at`` are immutable.

        Parameters
        ----------
        achievement_id:
            UUID of the achievement to update.
        title:
            New title, or ``None`` to leave unchanged.
        domain:
            New domain, or ``None`` to leave unchanged.
        description:
            New description, or ``None`` to leave unchanged.
        evidence:
            Replacement evidence list, or ``None`` to leave unchanged.
        importance:
            New importance tier, or ``None`` to leave unchanged.
        tags:
            Replacement tag list, or ``None`` to leave unchanged.

        Returns
        -------
        AchievementRecord
            The updated record.

        Raises
        ------
        AchievementNotFoundError
            If no record with ``achievement_id`` exists.
        AchievementValidationError
            If the new ``title`` is empty.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("update_achievement")

        with self._lock:
            record = self._store.get(achievement_id)
            if record is None:
                raise AchievementNotFoundError(achievement_id)

            if title is not None:
                self._validate_title(title)
                record.title = title.strip()
            if domain is not None:
                record.domain = domain
            if description is not None:
                record.description = description
            if evidence is not None:
                record.evidence = list(evidence)
            if importance is not None:
                record.importance = importance
            if tags is not None:
                record.tags = list(tags)

        self._publish_event("AchievementUpdated", record)
        _logger.debug("Updated achievement '%s'.", achievement_id)
        return record

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_achievement(self, achievement_id: str) -> bool:
        """Remove an achievement record from the store.

        CRITICAL importance achievements are protected and cannot be deleted.

        Parameters
        ----------
        achievement_id:
            UUID of the achievement to delete.

        Returns
        -------
        bool
            ``True`` if the record was found and removed; ``False`` if it
            did not exist (idempotent delete).

        Raises
        ------
        AchievementValidationError
            If the achievement is CRITICAL and deletion is therefore blocked.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("delete_achievement")

        with self._lock:
            record = self._store.get(achievement_id)
            if record is None:
                return False
            if record.importance == ExperienceImportance.CRITICAL:
                raise AchievementValidationError(
                    f"Achievement '{achievement_id}' is CRITICAL and cannot be deleted.",
                    field="importance",
                )
            del self._store[achievement_id]

        _logger.info("Deleted achievement '%s'.", achievement_id)
        return True

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_achievement(self, achievement_id: str) -> AchievementRecord:
        """Retrieve a single achievement by its UUID.

        Parameters
        ----------
        achievement_id:
            UUID of the desired achievement.

        Returns
        -------
        AchievementRecord
            The matching record.

        Raises
        ------
        AchievementNotFoundError
            If no record with this UUID exists.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("get_achievement")

        with self._lock:
            record = self._store.get(achievement_id)
            if record is None:
                raise AchievementNotFoundError(achievement_id)
            return record

    def search_achievements(
        self,
        *,
        domain: str | None = None,
        importance: ExperienceImportance | None = None,
        min_importance: ExperienceImportance | None = None,
        tags: list[str] | None = None,
        experience_id: str | None = None,
        achieved_after: datetime | None = None,
        achieved_before: datetime | None = None,
        keyword: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AchievementRecord]:
        """Return achievement records matching the supplied filter criteria.

        All filter parameters are optional and combined with logical AND.
        Results are ordered by ``achieved_at`` descending (most recent first).

        Parameters
        ----------
        domain:
            Filter by domain category (exact match).
        importance:
            Filter by exact :class:`~subsystems.echo.models.ExperienceImportance` tier.
        min_importance:
            Filter records at or above this importance tier.
        tags:
            Filter records that carry ALL of the supplied tag names.
        experience_id:
            Filter records linked to a specific parent Experience UUID.
        achieved_after:
            Only return records achieved after this UTC datetime.
        achieved_before:
            Only return records achieved before this UTC datetime.
        keyword:
            Case-insensitive substring match against ``title`` and
            ``description``.
        limit:
            Maximum number of results.
        offset:
            Number of results to skip (for pagination).

        Returns
        -------
        list[AchievementRecord]
            Matching records, ordered by ``achieved_at`` descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("search_achievements")

        if limit < 0:
            raise AchievementValidationError("limit must be >= 0.", field="limit")
        if offset < 0:
            raise AchievementValidationError("offset must be >= 0.", field="offset")

        tag_set = set(tags) if tags else None
        kw = keyword.lower() if keyword else None

        with self._lock:
            results: list[AchievementRecord] = []
            for record in self._store.values():
                if domain is not None and record.domain != domain:
                    continue
                if importance is not None and record.importance != importance:
                    continue
                if (
                    min_importance is not None
                    and record.importance.value < min_importance.value
                ):
                    continue
                if experience_id is not None and record.experience_id != experience_id:
                    continue
                if (
                    tag_set is not None
                    and not tag_set.issubset({t.name for t in record.tags})
                ):
                    continue
                if achieved_after is not None and record.achieved_at < achieved_after:
                    continue
                if achieved_before is not None and record.achieved_at > achieved_before:
                    continue
                if kw is not None and (
                    kw not in record.title.lower()
                    and kw not in record.description.lower()
                ):
                    continue
                results.append(record)

        results.sort(key=lambda r: r.achieved_at, reverse=True)
        return results[offset : offset + limit]

    # ------------------------------------------------------------------
    # Link to Experience
    # ------------------------------------------------------------------

    def link_experience(
        self,
        achievement_id: str,
        experience_id: str,
    ) -> AchievementRecord:
        """Associate an achievement with a parent Experience.

        If an ``experience_engine`` was injected at construction time, this
        method validates that the target Experience exists before writing
        the link.  Without an injected engine, the link is written blindly.

        Parameters
        ----------
        achievement_id:
            UUID of the achievement to update.
        experience_id:
            UUID of the Experience to link.

        Returns
        -------
        AchievementRecord
            The updated record.

        Raises
        ------
        AchievementNotFoundError
            If no achievement with ``achievement_id`` exists.
        ExperienceNotFoundError
            If the Experience engine is available and the target Experience
            does not exist.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("link_experience")

        # Validate target experience exists if we have the engine reference.
        if self._experience_engine is not None:
            if not self._experience_engine.experience_exists(experience_id):
                raise ExperienceNotFoundError(experience_id)

        with self._lock:
            record = self._store.get(achievement_id)
            if record is None:
                raise AchievementNotFoundError(achievement_id)
            record.experience_id = experience_id

        self._publish_event("AchievementLinked", record)
        _logger.debug(
            "Linked achievement '%s' to experience '%s'.",
            achievement_id,
            experience_id,
        )
        return record

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of the engine's current state.

        Returns
        -------
        dict[str, Any]
            Contains ``running``, ``total``, ``by_domain``,
            ``by_importance``, and ``linked_count``.
        """
        with self._lock:
            total = len(self._store)
            by_domain: dict[str, int] = {}
            by_importance: dict[str, int] = {}
            linked_count = 0

            for record in self._store.values():
                by_domain[record.domain] = by_domain.get(record.domain, 0) + 1
                imp_key = record.importance.name
                by_importance[imp_key] = by_importance.get(imp_key, 0) + 1
                if record.experience_id is not None:
                    linked_count += 1

        return {
            "running": self._running,
            "total": total,
            "by_domain": by_domain,
            "by_importance": by_importance,
            "linked_count": linked_count,
        }