# subsystems/echo/events.py
"""
ECHO v1 Event Engine.

Implements :class:`EventEngine` — the production manager for
:class:`~subsystems.echo.models.EventRecord` objects within the ECHO
Episodic Memory Core.

The Event Engine is responsible for capturing discrete events — the atomic
building blocks of ECHO's episodic history.  Events are distinct from full
:class:`~subsystems.echo.models.Experience` objects: they record a single,
concrete happening (e.g. *"ProjectCreated"*, *"GoalCompleted"*) without
narrative elaboration.  Multiple events may compose a session or be grouped
under a parent experience.

Design Principles
-----------------
* **Thread-safe**: All public methods serialise concurrent access via
  ``self._lock`` (a :class:`threading.RLock`).
* **Lifecycle-gated**: Every public operation guards against calls made
  before :meth:`initialize` or after :meth:`shutdown`.
* **Experience-linked**: Events may be associated with a parent
  :class:`~subsystems.echo.models.Experience` via ``experience_id``.
  The engine does not validate that the experience exists — that is the
  caller's responsibility.
* **Filterable**: :meth:`query_events` supports multi-dimensional
  filtering over type, importance, subsystem, experience, and time range.
* **Importance-gated write path**: A configurable minimum importance
  threshold prevents trivial events from bloating the store.  Callers
  may bypass the gate with ``force=True``.
* **Domain events**: Successful lifecycle operations emit ``EventRecorded``
  and ``EventDeleted`` domain events (logged at DEBUG in v1; Event Bus
  integration is reserved for a future iteration).

ECHO Boundary Law
-----------------
The Event Engine owns :class:`~subsystems.echo.models.EventRecord` objects.
It does NOT own experiences, sessions, achievements, or failures; those
belong to their respective engines.

Publish/Subscribe (ECHO domain)
--------------------------------
Publishes:
    * ``polaris.echo.event.recorded``
    * ``polaris.echo.event.deleted``

Subscribes (future):
    * ``polaris.runtime.subsystem.started``  — auto-record subsystem events
    * ``polaris.runtime.subsystem.stopped``
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from subsystems.echo.exceptions import (
    EchoNotInitializedError,
    EventError,
    EventNotFoundError,
    EventValidationError,
)
from subsystems.echo.models import (
    EventRecord,
    ExperienceImportance,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum importance threshold (default)
# ---------------------------------------------------------------------------

_DEFAULT_MIN_IMPORTANCE: ExperienceImportance = ExperienceImportance.LOW

# Ordinal map for importance comparison
_IMPORTANCE_VALUE: dict[ExperienceImportance, int] = {
    ExperienceImportance.LOW:      1,
    ExperienceImportance.MEDIUM:   2,
    ExperienceImportance.HIGH:     3,
    ExperienceImportance.CRITICAL: 4,
}


class EventEngine:
    """Production Event Engine for the ECHO Episodic Memory Core.

    Manages the full lifecycle of :class:`~subsystems.echo.models.EventRecord`
    objects: recording, querying, updating payloads, linking to experiences,
    and deletion.

    Parameters
    ----------
    min_importance:
        Minimum :class:`~subsystems.echo.models.ExperienceImportance` tier an
        event must carry before being accepted into the store without
        ``force=True``.  Defaults to :attr:`ExperienceImportance.LOW`
        (accepts everything).

    Thread Safety
    -------------
    All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
    before reading or modifying internal state.

    Examples
    --------
    ::

        engine = EventEngine()
        engine.initialize()

        record = engine.record_event(
            event_name="ProjectCreated",
            payload={"project": "POLARIS"},
            importance=ExperienceImportance.HIGH,
            source_subsystem="ORION",
        )

        engine.shutdown()
    """

    def __init__(
        self,
        *,
        min_importance: ExperienceImportance = _DEFAULT_MIN_IMPORTANCE,
    ) -> None:
        self._min_importance: ExperienceImportance = min_importance
        # In-process store: event_id → EventRecord
        self._store: dict[str, EventRecord] = {}
        # Secondary index: experience_id → [event_id, ...]
        self._experience_index: dict[str, list[str]] = {}
        self._lock: threading.RLock = threading.RLock()
        self._running: bool = False

        _logger.debug(
            "EventEngine constructed (min_importance=%s).",
            min_importance.name,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Sets the running flag and prepares internal state.  Idempotent:
        calling ``initialize()`` on an already-running engine is a no-op.

        Raises
        ------
        EventError
            If initialisation fails.
        """
        with self._lock:
            if self._running:
                _logger.warning(
                    "EventEngine.initialize() called while already running."
                )
                return
            try:
                self._running = True
                _logger.info(
                    "EventEngine initialised (min_importance=%s).",
                    self._min_importance.name,
                )
            except Exception as exc:
                raise EventError(
                    f"EventEngine initialisation failed: {exc}"
                ) from exc

    def shutdown(self) -> None:
        """Release all resources held by this engine.

        After shutdown, any public operation raises
        :class:`~subsystems.echo.exceptions.EchoNotInitializedError`.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            _logger.info(
                "EventEngine shut down.  Store contained %d event(s).",
                len(self._store),
            )

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _require_running(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if the engine is not running."""
        if not self._running:
            raise EchoNotInitializedError(f"EventEngine.{operation}")

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_event_name(self, event_name: str) -> None:
        """Raise :class:`EventValidationError` if ``event_name`` is blank."""
        if not event_name or not event_name.strip():
            raise EventValidationError(
                "EventRecord.event_name must be a non-empty string.",
                field="event_name",
            )

    def _importance_passes_threshold(
        self,
        importance: ExperienceImportance,
    ) -> bool:
        """Return ``True`` if ``importance`` meets the configured minimum."""
        return (
            _IMPORTANCE_VALUE[importance]
            >= _IMPORTANCE_VALUE[self._min_importance]
        )

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------

    def _index_add(self, event: EventRecord) -> None:
        """Add an event to the experience secondary index (if linked)."""
        if event.experience_id is None:
            return
        bucket = self._experience_index.setdefault(event.experience_id, [])
        if event.event_id not in bucket:
            bucket.append(event.event_id)

    def _index_remove(self, event: EventRecord) -> None:
        """Remove an event from the experience secondary index."""
        if event.experience_id is None:
            return
        bucket = self._experience_index.get(event.experience_id)
        if bucket is None:
            return
        try:
            bucket.remove(event.event_id)
        except ValueError:
            pass
        if not bucket:
            del self._experience_index[event.experience_id]

    # ------------------------------------------------------------------
    # Record (create)
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_name: str,
        *,
        payload: dict[str, Any] | None = None,
        importance: ExperienceImportance = ExperienceImportance.MEDIUM,
        source_subsystem: str = "ECHO_API",
        experience_id: str | None = None,
        occurred_at: datetime | None = None,
        force: bool = False,
    ) -> EventRecord:
        """Record a new discrete event into ECHO's event store.

        Parameters
        ----------
        event_name:
            Short label for the event (e.g. ``"ProjectCreated"``).
            Must be a non-empty string.
        payload:
            Structured key-value data describing the event.  Defaults to
            an empty dict if not supplied.
        importance:
            :class:`~subsystems.echo.models.ExperienceImportance` tier for
            this event.  Defaults to ``MEDIUM``.
        source_subsystem:
            Originating POLARIS subsystem (e.g. ``"ORION"``, ``"ODYSSEY"``).
        experience_id:
            Optional UUID of a parent :class:`~subsystems.echo.models.Experience`
            that this event belongs to.  The caller is responsible for
            ensuring the referenced experience exists.
        occurred_at:
            UTC timestamp of when this event occurred.  Defaults to now.
        force:
            If ``True``, bypass the importance threshold gate and store
            regardless of the event's importance tier.

        Returns
        -------
        EventRecord
            The recorded event with all fields populated.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        EventValidationError
            If ``event_name`` is blank or other field validation fails.
        EventError
            If the event is below the importance threshold and
            ``force=False``.
        """
        with self._lock:
            self._require_running("record_event")
            self._validate_event_name(event_name)

            if not force and not self._importance_passes_threshold(importance):
                raise EventError(
                    f"Event '{event_name}' has importance {importance.name}, "
                    f"which is below the configured minimum "
                    f"{self._min_importance.name}.  Pass force=True to store "
                    "regardless."
                )

            now = datetime.now(timezone.utc)
            event = EventRecord(
                event_name=event_name.strip(),
                experience_id=experience_id,
                payload=dict(payload) if payload else {},
                importance=importance,
                source_subsystem=source_subsystem,
                occurred_at=occurred_at if occurred_at is not None else now,
                recorded_at=now,
            )

            self._store[event.event_id] = event
            self._index_add(event)

        _logger.info(
            "EventEngine: recorded event '%s' (id=%s, importance=%s).",
            event.event_name,
            event.event_id,
            event.importance.name,
        )
        self._publish_event_recorded(event)
        return event

    # ------------------------------------------------------------------
    # Store (persist pre-built EventRecord)
    # ------------------------------------------------------------------

    def store_event(
        self,
        event: EventRecord,
        *,
        force: bool = False,
    ) -> EventRecord:
        """Persist an already-constructed :class:`EventRecord`.

        Unlike :meth:`record_event`, this method accepts a pre-built domain
        object.  The importance threshold check still applies unless
        ``force=True``.

        Parameters
        ----------
        event:
            The :class:`EventRecord` to persist.  Must not be ``None``.
        force:
            Bypass the importance threshold if ``True``.

        Returns
        -------
        EventRecord
            The stored event.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        EventValidationError
            If ``event`` is ``None`` or has an empty ``event_name``.
        EventError
            If the event is below the importance threshold and
            ``force=False``, or if an event with the same ``event_id``
            already exists.
        """
        with self._lock:
            self._require_running("store_event")

            if event is None:
                raise EventValidationError(
                    "event must not be None.", field="event"
                )
            self._validate_event_name(event.event_name)

            if event.event_id in self._store:
                raise EventError(
                    f"An event with id '{event.event_id}' already exists. "
                    "Use update_event_payload() if you intend to modify it."
                )

            if not force and not self._importance_passes_threshold(event.importance):
                raise EventError(
                    f"Event '{event.event_name}' has importance "
                    f"{event.importance.name}, which is below the configured "
                    f"minimum {self._min_importance.name}.  Pass force=True "
                    "to store regardless."
                )

            self._store[event.event_id] = event
            self._index_add(event)

        _logger.info(
            "EventEngine: stored pre-built event '%s' (id=%s).",
            event.event_name,
            event.event_id,
        )
        self._publish_event_recorded(event)
        return event

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_event_payload(
        self,
        event_id: str,
        payload: dict[str, Any],
    ) -> EventRecord:
        """Replace the payload of an existing :class:`EventRecord`.

        Only the ``payload`` field is mutable after creation.  All other
        fields (name, importance, timestamps, ids) are immutable.

        Parameters
        ----------
        event_id:
            UUID of the event to update.
        payload:
            New payload dict to replace the existing one.

        Returns
        -------
        EventRecord
            The updated event record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        EventNotFoundError
            If no event with ``event_id`` exists.
        EventValidationError
            If ``payload`` is not a dict.
        """
        with self._lock:
            self._require_running("update_event_payload")

            if not isinstance(payload, dict):
                raise EventValidationError(
                    "payload must be a dict.", field="payload"
                )

            event = self._store.get(event_id)
            if event is None:
                raise EventNotFoundError(event_id)

            event.payload = dict(payload)

        _logger.debug(
            "EventEngine: updated payload for event id=%s.", event_id
        )
        return event

    def link_event_to_experience(
        self,
        event_id: str,
        experience_id: str,
    ) -> EventRecord:
        """Associate an existing event with a parent experience.

        If the event is already linked to the supplied ``experience_id``
        this is a no-op.  Linking to a different experience replaces the
        existing link and updates the secondary index accordingly.

        Parameters
        ----------
        event_id:
            UUID of the event to link.
        experience_id:
            UUID of the parent :class:`~subsystems.echo.models.Experience`.

        Returns
        -------
        EventRecord
            The updated event record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        EventNotFoundError
            If no event with ``event_id`` exists.
        EventValidationError
            If ``experience_id`` is blank.
        """
        with self._lock:
            self._require_running("link_event_to_experience")

            if not experience_id or not experience_id.strip():
                raise EventValidationError(
                    "experience_id must be a non-empty string.",
                    field="experience_id",
                )

            event = self._store.get(event_id)
            if event is None:
                raise EventNotFoundError(event_id)

            if event.experience_id == experience_id:
                return event

            # Remove old index entry before relinking.
            self._index_remove(event)
            event.experience_id = experience_id
            self._index_add(event)

        _logger.debug(
            "EventEngine: linked event id=%s → experience id=%s.",
            event_id,
            experience_id,
        )
        return event

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_event(self, event_id: str) -> bool:
        """Remove an event from the store.

        CRITICAL importance events are protected and cannot be deleted.

        Parameters
        ----------
        event_id:
            UUID of the event to delete.

        Returns
        -------
        bool
            ``True`` if the event was found and removed; ``False`` if it
            did not exist (idempotent delete).

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        EventError
            If the event carries CRITICAL importance.
        """
        with self._lock:
            self._require_running("delete_event")

            event = self._store.get(event_id)
            if event is None:
                return False

            if event.importance == ExperienceImportance.CRITICAL:
                raise EventError(
                    f"Event '{event_id}' carries CRITICAL importance and "
                    "cannot be deleted.  CRITICAL events are permanent records."
                )

            self._index_remove(event)
            del self._store[event_id]

        _logger.info("EventEngine: deleted event id=%s.", event_id)
        self._publish_event_deleted(event_id)
        return True

    # ------------------------------------------------------------------
    # Query — single record
    # ------------------------------------------------------------------

    def get_event(self, event_id: str) -> EventRecord:
        """Retrieve a single :class:`EventRecord` by UUID.

        Parameters
        ----------
        event_id:
            UUID of the desired event.

        Returns
        -------
        EventRecord
            The matching event.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        EventNotFoundError
            If no event with this UUID exists.
        """
        with self._lock:
            self._require_running("get_event")
            event = self._store.get(event_id)
            if event is None:
                raise EventNotFoundError(event_id)
            return event

    def event_exists(self, event_id: str) -> bool:
        """Return whether an event with the given UUID exists.

        Parameters
        ----------
        event_id:
            UUID to check.

        Returns
        -------
        bool
            ``True`` if the event is in the store.
        """
        with self._lock:
            self._require_running("event_exists")
            return event_id in self._store

    # ------------------------------------------------------------------
    # Query — by experience
    # ------------------------------------------------------------------

    def get_events_for_experience(
        self,
        experience_id: str,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        limit: int = 100,
    ) -> list[EventRecord]:
        """Return all events linked to a specific experience.

        Results are ordered by ``occurred_at`` descending (most recent first).

        Parameters
        ----------
        experience_id:
            UUID of the parent :class:`~subsystems.echo.models.Experience`.
        min_importance:
            Exclude events below this importance tier.
        limit:
            Maximum number of results to return.

        Returns
        -------
        list[EventRecord]
            Events linked to the experience, newest first.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        EventValidationError
            If ``experience_id`` is blank.
        """
        with self._lock:
            self._require_running("get_events_for_experience")

            if not experience_id or not experience_id.strip():
                raise EventValidationError(
                    "experience_id must be a non-empty string.",
                    field="experience_id",
                )

            min_val = _IMPORTANCE_VALUE[min_importance]
            event_ids = self._experience_index.get(experience_id, [])
            results: list[EventRecord] = []

            for eid in event_ids:
                ev = self._store.get(eid)
                if ev is None:
                    continue
                if _IMPORTANCE_VALUE[ev.importance] < min_val:
                    continue
                results.append(ev)

        results.sort(key=lambda e: e.occurred_at, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Query — filtered
    # ------------------------------------------------------------------

    def query_events(
        self,
        *,
        event_name: str | None = None,
        importance: ExperienceImportance | None = None,
        min_importance: ExperienceImportance | None = None,
        source_subsystem: str | None = None,
        experience_id: str | None = None,
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EventRecord]:
        """Return events matching the supplied filter criteria.

        All filter parameters are optional and combined with logical AND.
        Results are ordered by ``occurred_at`` descending (most recent first).

        Parameters
        ----------
        event_name:
            Exact event name match (case-sensitive).
        importance:
            Exact importance tier filter.
        min_importance:
            Minimum importance tier (inclusive).
        source_subsystem:
            Filter by originating subsystem identifier.
        experience_id:
            Filter events linked to a specific parent experience.
        occurred_after:
            Only return events that occurred after this UTC datetime.
        occurred_before:
            Only return events that occurred before this UTC datetime.
        limit:
            Maximum number of results to return.
        offset:
            Number of results to skip (for pagination).

        Returns
        -------
        list[EventRecord]
            Matching events, ordered by ``occurred_at`` descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        EventValidationError
            If ``limit`` or ``offset`` are negative.
        """
        with self._lock:
            self._require_running("query_events")

            if limit < 0:
                raise EventValidationError(
                    "limit must be >= 0.", field="limit"
                )
            if offset < 0:
                raise EventValidationError(
                    "offset must be >= 0.", field="offset"
                )

            min_val = (
                _IMPORTANCE_VALUE[min_importance]
                if min_importance is not None
                else None
            )

            results: list[EventRecord] = []

            for ev in self._store.values():
                if event_name is not None and ev.event_name != event_name:
                    continue
                if importance is not None and ev.importance != importance:
                    continue
                if min_val is not None and _IMPORTANCE_VALUE[ev.importance] < min_val:
                    continue
                if (
                    source_subsystem is not None
                    and ev.source_subsystem != source_subsystem
                ):
                    continue
                if experience_id is not None and ev.experience_id != experience_id:
                    continue
                if occurred_after is not None and ev.occurred_at < occurred_after:
                    continue
                if occurred_before is not None and ev.occurred_at > occurred_before:
                    continue

                results.append(ev)

        results.sort(key=lambda e: e.occurred_at, reverse=True)
        return results[offset : offset + limit]

    def count_events(
        self,
        *,
        event_name: str | None = None,
        importance: ExperienceImportance | None = None,
        source_subsystem: str | None = None,
        experience_id: str | None = None,
    ) -> int:
        """Return the count of stored events matching the given filters.

        Parameters
        ----------
        event_name:
            Exact event name match.
        importance:
            Exact importance tier filter.
        source_subsystem:
            Filter by originating subsystem.
        experience_id:
            Count only events linked to this experience.

        Returns
        -------
        int
            Number of matching event records.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        with self._lock:
            self._require_running("count_events")

            count = 0
            for ev in self._store.values():
                if event_name is not None and ev.event_name != event_name:
                    continue
                if importance is not None and ev.importance != importance:
                    continue
                if (
                    source_subsystem is not None
                    and ev.source_subsystem != source_subsystem
                ):
                    continue
                if experience_id is not None and ev.experience_id != experience_id:
                    continue
                count += 1

            return count

    # ------------------------------------------------------------------
    # Bulk / Utility
    # ------------------------------------------------------------------

    def get_recent_events(
        self,
        limit: int = 20,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
    ) -> list[EventRecord]:
        """Return the most recently recorded events.

        Parameters
        ----------
        limit:
            Maximum number of results.
        min_importance:
            Exclude events below this importance tier.

        Returns
        -------
        list[EventRecord]
            Most recently recorded events, newest first.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        with self._lock:
            self._require_running("get_recent_events")
            min_val = _IMPORTANCE_VALUE[min_importance]
            filtered = [
                ev
                for ev in self._store.values()
                if _IMPORTANCE_VALUE[ev.importance] >= min_val
            ]

        filtered.sort(key=lambda e: e.recorded_at, reverse=True)
        return filtered[:limit]

    def get_critical_events(self) -> list[EventRecord]:
        """Return all events with CRITICAL importance.

        CRITICAL events are permanent records and cannot be deleted.

        Returns
        -------
        list[EventRecord]
            All CRITICAL events, ordered by ``occurred_at`` descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        with self._lock:
            self._require_running("get_critical_events")
            critical = [
                ev
                for ev in self._store.values()
                if ev.importance == ExperienceImportance.CRITICAL
            ]

        critical.sort(key=lambda e: e.occurred_at, reverse=True)
        return critical

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def get_min_importance(self) -> ExperienceImportance:
        """Return the current minimum importance threshold.

        Returns
        -------
        ExperienceImportance
            The current minimum importance for event acceptance.
        """
        with self._lock:
            return self._min_importance

    def set_min_importance(self, importance: ExperienceImportance) -> None:
        """Update the minimum importance threshold for new events.

        This setting does not retroactively affect already-stored events.

        Parameters
        ----------
        importance:
            New minimum :class:`~subsystems.echo.models.ExperienceImportance`
            tier.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        with self._lock:
            self._require_running("set_min_importance")
            self._min_importance = importance
            _logger.info(
                "EventEngine: minimum importance threshold updated to %s.",
                importance.name,
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of the engine's current state.

        Returns
        -------
        dict[str, Any]
            Contains ``running``, ``total``, ``by_name``, ``by_importance``,
            ``by_subsystem``, ``linked_count``, and ``min_importance``.
        """
        with self._lock:
            total = len(self._store)
            by_name: dict[str, int] = {}
            by_importance: dict[str, int] = {}
            by_subsystem: dict[str, int] = {}
            linked_count = 0

            for ev in self._store.values():
                by_name[ev.event_name] = by_name.get(ev.event_name, 0) + 1
                by_importance[ev.importance.name] = (
                    by_importance.get(ev.importance.name, 0) + 1
                )
                by_subsystem[ev.source_subsystem] = (
                    by_subsystem.get(ev.source_subsystem, 0) + 1
                )
                if ev.experience_id is not None:
                    linked_count += 1

            return {
                "running": self._running,
                "total": total,
                "by_name": by_name,
                "by_importance": by_importance,
                "by_subsystem": by_subsystem,
                "linked_count": linked_count,
                "min_importance": self._min_importance.name,
            }

    # ------------------------------------------------------------------
    # Domain event publishing
    # ------------------------------------------------------------------

    def _publish_event_recorded(self, event: EventRecord) -> None:
        """Emit the ``polaris.echo.event.recorded`` domain event.

        In v1, this logs at DEBUG level.  The Event Bus integration will
        be wired in a future iteration.
        """
        _logger.debug(
            "EVENT polaris.echo.event.recorded: id=%s name=%r importance=%s",
            event.event_id,
            event.event_name,
            event.importance.name,
        )

    def _publish_event_deleted(self, event_id: str) -> None:
        """Emit the ``polaris.echo.event.deleted`` domain event.

        In v1, this logs at DEBUG level.  The Event Bus integration will
        be wired in a future iteration.
        """
        _logger.debug(
            "EVENT polaris.echo.event.deleted: id=%s",
            event_id,
        )