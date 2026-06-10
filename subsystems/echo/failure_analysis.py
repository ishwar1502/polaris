# subsystems/echo/failure_analysis.py
"""
ECHO v1 Failure Analysis Engine.

Implements :class:`FailureAnalysisEngine` — the production engine responsible
for recording, storing, analysing, and retrieving
:class:`~subsystems.echo.models.FailureRecord` objects within ECHO.

Failures are first-class citizens of ECHO.  Recording failures objectively
and honestly is fundamental to the learning loop: without an accurate account
of what did not work, the Reflection Engine cannot produce meaningful lessons
and ASTRA cannot track growth trajectories.

This engine purposefully avoids subjective language.  Failures are facts, not
criticism.  Every :class:`~subsystems.echo.models.FailureRecord` captures
what happened, contributing factors, outcomes, and lessons learned.

Thread safety
-------------
All public methods acquire ``self._lock`` (a :class:`threading.RLock`) before
reading or modifying internal state.  Safe for concurrent use across multiple
threads as required by the POLARIS runtime.

Architecture notes
------------------
* In v1, the backing store is an in-process ``dict[str, FailureRecord]``
  keyed by ``failure_id``.
* Lesson records are stored as annotations on the :class:`FailureRecord`
  ``lesson`` field and are accessible via :meth:`store_lesson`.
* The engine integrates with the Experience Engine when injected, allowing
  failure records to be validated against parent experiences.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from subsystems.echo.exceptions import (
    EchoNotInitializedError,
    ExperienceNotFoundError,
    FailureNotFoundError,
    FailureValidationError,
)
from subsystems.echo.models import (
    ExperienceImportance,
    FailureRecord,
    MemoryTag,
)

_logger = logging.getLogger(__name__)


class FailureAnalysisEngine:
    """Production implementation of the ECHO Failure Analysis Engine.

    Tracks failures objectively to support learning.  Each
    :class:`~subsystems.echo.models.FailureRecord` captures the title,
    domain, description, contributing factors, lesson extracted, and an
    optional link to a parent :class:`~subsystems.echo.models.Experience`.

    Parameters
    ----------
    experience_engine:
        Optional reference to the running ExperienceEngine.  When supplied,
        :meth:`record_failure` and :meth:`analyze_failure` can validate that
        the referenced Experience exists.  Pass ``None`` for standalone mode.

    Usage
    -----
    ::

        engine = FailureAnalysisEngine()
        engine.initialize()

        record = engine.record_failure(
            title="Missed Sprint Deadline",
            domain="software",
            description="Sprint goal not reached due to scope creep.",
            contributing_factors=["Unclear requirements", "Underestimated complexity"],
            lesson="Break features into smaller, independently deployable units.",
            importance=ExperienceImportance.HIGH,
        )

        analysis = engine.analyze_failure(record.failure_id)
        engine.shutdown()
    """

    def __init__(
        self,
        experience_engine: Any | None = None,
    ) -> None:
        self._experience_engine = experience_engine
        self._store: dict[str, FailureRecord] = {}
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
            _logger.info("FailureAnalysisEngine initialised.")

    def shutdown(self) -> None:
        """Release all resources held by this engine.

        Flushes internal state and marks the engine as stopped.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            _logger.info(
                "FailureAnalysisEngine shut down. Stored %d failure record(s).",
                len(self._store),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _validate_title(self, title: str) -> None:
        if not title or not title.strip():
            raise FailureValidationError(
                "FailureRecord.title must be a non-empty string.",
                field="title",
            )

    def _publish_event(self, event_name: str, record: FailureRecord) -> None:
        _logger.debug(
            "EVENT %s: failure_id=%s title=%r domain=%s importance=%s",
            event_name,
            record.failure_id,
            record.title,
            record.domain,
            record.importance.name,
        )

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record_failure(
        self,
        title: str,
        *,
        domain: str = "general",
        description: str = "",
        contributing_factors: list[str] | None = None,
        lesson: str = "",
        importance: ExperienceImportance = ExperienceImportance.MEDIUM,
        tags: list[MemoryTag] | None = None,
        experience_id: str | None = None,
        failed_at: datetime | None = None,
    ) -> FailureRecord:
        """Create and persist a new :class:`~subsystems.echo.models.FailureRecord`.

        Parameters
        ----------
        title:
            Short, objective label for the failure (e.g. ``"Missed Deadline"``).
        domain:
            Broad domain category (e.g. ``"software"``, ``"academic"``,
            ``"personal"``).  Defaults to ``"general"``.
        description:
            Objective account of what failed and the immediate circumstances.
        contributing_factors:
            List of identified root-cause or contributing factors.
        lesson:
            Initial lesson extracted at record time.  May be refined later by
            the Reflection Engine via :meth:`store_lesson`.
        importance:
            :class:`~subsystems.echo.models.ExperienceImportance` tier.
            Defaults to MEDIUM; callers should escalate critical failures.
        tags:
            Optional :class:`~subsystems.echo.models.MemoryTag` list for indexing.
        experience_id:
            Optional UUID of the parent Experience to link this record to.
        failed_at:
            UTC timestamp of the failure moment.  Defaults to now.

        Returns
        -------
        FailureRecord
            The created and stored failure record.

        Raises
        ------
        FailureValidationError
            If ``title`` is empty.
        ExperienceNotFoundError
            If ``experience_id`` is supplied and the experience does not exist
            (only checked when an experience_engine is injected).
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("record_failure")
        self._validate_title(title)

        if experience_id is not None and self._experience_engine is not None:
            if not self._experience_engine.experience_exists(experience_id):
                raise ExperienceNotFoundError(experience_id)

        record = FailureRecord(
            title=title.strip(),
            experience_id=experience_id,
            domain=domain or "general",
            description=description,
            contributing_factors=list(contributing_factors) if contributing_factors else [],
            lesson=lesson,
            importance=importance,
            tags=list(tags) if tags else [],
            reflection_generated=False,
            failed_at=failed_at or datetime.now(timezone.utc),
            recorded_at=datetime.now(timezone.utc),
        )

        with self._lock:
            self._store[record.failure_id] = record

        self._publish_event("FailureRecorded", record)
        _logger.info(
            "Recorded failure '%s' (%s).",
            record.title,
            record.failure_id,
        )
        return record

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_failure(
        self,
        failure_id: str,
        *,
        title: str | None = None,
        domain: str | None = None,
        description: str | None = None,
        contributing_factors: list[str] | None = None,
        lesson: str | None = None,
        importance: ExperienceImportance | None = None,
        tags: list[MemoryTag] | None = None,
    ) -> FailureRecord:
        """Update mutable fields of an existing failure record.

        Only fields supplied as non-``None`` are modified.  ``failure_id``,
        ``experience_id``, and ``recorded_at`` are immutable.

        Parameters
        ----------
        failure_id:
            UUID of the failure record to update.
        title:
            New title, or ``None`` to leave unchanged.
        domain:
            New domain, or ``None`` to leave unchanged.
        description:
            New description, or ``None`` to leave unchanged.
        contributing_factors:
            Replacement factor list, or ``None`` to leave unchanged.
        lesson:
            Updated lesson text, or ``None`` to leave unchanged.
        importance:
            New importance tier, or ``None`` to leave unchanged.
        tags:
            Replacement tag list, or ``None`` to leave unchanged.

        Returns
        -------
        FailureRecord
            The updated record.

        Raises
        ------
        FailureNotFoundError
            If no record with ``failure_id`` exists.
        FailureValidationError
            If the new ``title`` is empty.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("update_failure")

        with self._lock:
            record = self._store.get(failure_id)
            if record is None:
                raise FailureNotFoundError(failure_id)

            if title is not None:
                self._validate_title(title)
                record.title = title.strip()
            if domain is not None:
                record.domain = domain
            if description is not None:
                record.description = description
            if contributing_factors is not None:
                record.contributing_factors = list(contributing_factors)
            if lesson is not None:
                record.lesson = lesson
            if importance is not None:
                record.importance = importance
            if tags is not None:
                record.tags = list(tags)

        self._publish_event("FailureUpdated", record)
        _logger.debug("Updated failure record '%s'.", failure_id)
        return record

    # ------------------------------------------------------------------
    # Analyse
    # ------------------------------------------------------------------

    def analyze_failure(self, failure_id: str) -> dict[str, Any]:
        """Produce an analysis summary for a failure record.

        Builds a structured analysis bundle from the stored record, suitable
        for consumption by the Reflection Engine and ASTRA's growth tracker.

        Parameters
        ----------
        failure_id:
            UUID of the failure record to analyse.

        Returns
        -------
        dict[str, Any]
            Analysis bundle containing:

            * ``failure_id`` — record UUID
            * ``title`` — failure title
            * ``domain`` — domain category
            * ``description`` — objective description
            * ``contributing_factors`` — list of causal factors
            * ``factor_count`` — number of identified factors
            * ``lesson`` — extracted lesson (may be empty if not yet set)
            * ``has_lesson`` — whether a lesson has been recorded
            * ``reflection_generated`` — whether the Reflection Engine has
              already processed this failure
            * ``importance`` — importance tier name
            * ``tags`` — tag name list
            * ``experience_id`` — linked experience UUID or ``None``
            * ``failed_at`` — UTC timestamp string
            * ``recorded_at`` — UTC timestamp string
            * ``age_days`` — days since the failure occurred (float)

        Raises
        ------
        FailureNotFoundError
            If no record with ``failure_id`` exists.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("analyze_failure")

        with self._lock:
            record = self._store.get(failure_id)
            if record is None:
                raise FailureNotFoundError(failure_id)

        now = datetime.now(timezone.utc)
        age_days = (now - record.failed_at).total_seconds() / 86_400.0

        return {
            "failure_id": record.failure_id,
            "title": record.title,
            "domain": record.domain,
            "description": record.description,
            "contributing_factors": list(record.contributing_factors),
            "factor_count": len(record.contributing_factors),
            "lesson": record.lesson,
            "has_lesson": bool(record.lesson and record.lesson.strip()),
            "reflection_generated": record.reflection_generated,
            "importance": record.importance.name,
            "tags": [t.name for t in record.tags],
            "experience_id": record.experience_id,
            "failed_at": record.failed_at.isoformat(),
            "recorded_at": record.recorded_at.isoformat(),
            "age_days": round(age_days, 2),
        }

    # ------------------------------------------------------------------
    # Store lesson
    # ------------------------------------------------------------------

    def store_lesson(
        self,
        failure_id: str,
        lesson: str,
        *,
        mark_reflected: bool = False,
    ) -> FailureRecord:
        """Store or update the lesson learned from a failure.

        Typically called by the Reflection Engine after it has generated a
        formal :class:`~subsystems.echo.models.Experience` of type
        ``REFLECTION`` from this failure.

        Parameters
        ----------
        failure_id:
            UUID of the failure record to update.
        lesson:
            The lesson text to store.
        mark_reflected:
            If ``True``, set ``FailureRecord.reflection_generated = True``
            to indicate the Reflection Engine has processed this failure.

        Returns
        -------
        FailureRecord
            The updated record.

        Raises
        ------
        FailureNotFoundError
            If no record with ``failure_id`` exists.
        FailureValidationError
            If ``lesson`` is empty.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("store_lesson")

        if not lesson or not lesson.strip():
            raise FailureValidationError(
                "lesson must be a non-empty string.",
                field="lesson",
            )

        with self._lock:
            record = self._store.get(failure_id)
            if record is None:
                raise FailureNotFoundError(failure_id)
            record.lesson = lesson.strip()
            if mark_reflected:
                record.mark_reflected()

        self._publish_event("LessonStored", record)
        _logger.debug(
            "Stored lesson for failure '%s' (reflected=%s).",
            failure_id,
            record.reflection_generated,
        )
        return record

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_failure(self, failure_id: str) -> FailureRecord:
        """Retrieve a single failure record by its UUID.

        Parameters
        ----------
        failure_id:
            UUID of the desired record.

        Returns
        -------
        FailureRecord
            The matching record.

        Raises
        ------
        FailureNotFoundError
            If no record with this UUID exists.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("get_failure")

        with self._lock:
            record = self._store.get(failure_id)
            if record is None:
                raise FailureNotFoundError(failure_id)
            return record

    def search_failures(
        self,
        *,
        domain: str | None = None,
        importance: ExperienceImportance | None = None,
        min_importance: ExperienceImportance | None = None,
        tags: list[str] | None = None,
        experience_id: str | None = None,
        has_lesson: bool | None = None,
        reflection_generated: bool | None = None,
        failed_after: datetime | None = None,
        failed_before: datetime | None = None,
        keyword: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FailureRecord]:
        """Return failure records matching the supplied filter criteria.

        All filter parameters are optional and combined with logical AND.
        Results are ordered by ``failed_at`` descending (most recent first).

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
        has_lesson:
            ``True`` to return only records with a non-empty lesson;
            ``False`` for records without a lesson.
        reflection_generated:
            ``True`` for records that have already been reflected upon;
            ``False`` for unprocessed failures.
        failed_after:
            Only return records that failed after this UTC datetime.
        failed_before:
            Only return records that failed before this UTC datetime.
        keyword:
            Case-insensitive substring match against ``title`` and
            ``description``.
        limit:
            Maximum number of results.
        offset:
            Number of results to skip (for pagination).

        Returns
        -------
        list[FailureRecord]
            Matching records, ordered by ``failed_at`` descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("search_failures")

        if limit < 0:
            raise FailureValidationError("limit must be >= 0.", field="limit")
        if offset < 0:
            raise FailureValidationError("offset must be >= 0.", field="offset")

        tag_set = set(tags) if tags else None
        kw = keyword.lower() if keyword else None

        with self._lock:
            results: list[FailureRecord] = []
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
                if has_lesson is not None:
                    lesson_present = bool(record.lesson and record.lesson.strip())
                    if lesson_present != has_lesson:
                        continue
                if (
                    reflection_generated is not None
                    and record.reflection_generated != reflection_generated
                ):
                    continue
                if failed_after is not None and record.failed_at < failed_after:
                    continue
                if failed_before is not None and record.failed_at > failed_before:
                    continue
                if kw is not None and (
                    kw not in record.title.lower()
                    and kw not in record.description.lower()
                ):
                    continue
                results.append(record)

        results.sort(key=lambda r: r.failed_at, reverse=True)
        return results[offset : offset + limit]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of the engine's current state.

        Returns
        -------
        dict[str, Any]
            Contains ``running``, ``total``, ``by_domain``,
            ``by_importance``, ``with_lesson``, ``reflected_count``,
            and ``unprocessed_count``.
        """
        with self._lock:
            total = len(self._store)
            by_domain: dict[str, int] = {}
            by_importance: dict[str, int] = {}
            with_lesson = 0
            reflected_count = 0

            for record in self._store.values():
                by_domain[record.domain] = by_domain.get(record.domain, 0) + 1
                imp_key = record.importance.name
                by_importance[imp_key] = by_importance.get(imp_key, 0) + 1
                if record.lesson and record.lesson.strip():
                    with_lesson += 1
                if record.reflection_generated:
                    reflected_count += 1

        return {
            "running": self._running,
            "total": total,
            "by_domain": by_domain,
            "by_importance": by_importance,
            "with_lesson": with_lesson,
            "reflected_count": reflected_count,
            "unprocessed_count": total - reflected_count,
        }