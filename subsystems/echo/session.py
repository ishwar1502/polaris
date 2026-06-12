# subsystems/echo/session.py
"""
ECHO v1 Session Engine.

Implements :class:`SessionEngine` — the production manager for
:class:`SessionRecord` objects within the ECHO Episodic Memory Core.

A session is a bounded work or activity period that groups related
:class:`~subsystems.echo.models.Experience` objects under a single
meaningful label — for example, an *Architecture Review Session*, a
*POLARIS Audit Session*, or a *Semester Planning Session*.

Sessions are not time-boxes imposed from the outside.  They are episodic
containers that give ECHO's memory a hierarchical structure: experiences
belong to sessions, and sessions belong to personal history.

Design Principles
-----------------
* **Thread-safe**: All public methods serialise concurrent access via
  ``self._lock`` (a :class:`threading.RLock`).
* **Lifecycle-gated**: Every public operation guards against calls made
  before :meth:`initialize` or after :meth:`shutdown`.
* **Experience-linked**: Sessions are backed by an
  :class:`~subsystems.echo.models.Experience` of type
  :attr:`~subsystems.echo.models.ExperienceType.SESSION`.  Member
  experiences carry the session's ``experience_id`` in their
  :attr:`~subsystems.echo.models.ExperienceMetadata.session_id` field.
* **Lifecycle management**: Sessions can be opened, closed, and reopened.
  A closed session is a finished episodic unit; reopening signals that
  the context is being revisited.
* **History tracking**: Every state change (open → closed → reopened) is
  recorded in a lightweight audit trail for Context Reconstruction.

ECHO Boundary Law
-----------------
The Session Engine owns :class:`SessionRecord` objects.  It does NOT own
knowledge, identity, goals, schedules, relationships, or decisions.

Publish / Subscribe (ECHO domain)
----------------------------------
Publishes (logged at DEBUG in v1; Event Bus integration reserved for future):
    * ``polaris.echo.session.created``
    * ``polaris.echo.session.closed``
    * ``polaris.echo.session.reopened``
    * ``polaris.echo.session.experience_added``
    * ``polaris.echo.session.experience_removed``
    * ``polaris.echo.session.updated``
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, unique
from typing import Any

from subsystems.echo.exceptions import (
    EchoNotInitializedError,
    ExperienceNotFoundError,
)
from subsystems.echo.interfaces import (
    ExperienceEngineInterface,
    SignificanceEngineInterface,
)
from subsystems.echo.models import (
    Experience,
    ExperienceImportance,
    ExperienceType,
    MemoryTag,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session state enumeration
# ---------------------------------------------------------------------------


@unique
class SessionState(Enum):
    """All states a :class:`SessionRecord` may occupy.

    State ordering::

        OPEN → CLOSED
        CLOSED → OPEN  (via reopen)
    """

    OPEN = auto()
    """Session is active; new experiences can be added."""

    CLOSED = auto()
    """Session has concluded; its episodic record is finalised."""


# ---------------------------------------------------------------------------
# Session-specific exceptions
# ---------------------------------------------------------------------------


class SessionError(Exception):
    """Base exception for all Session Engine errors."""


class SessionNotFoundError(SessionError):
    """Raised when a session_id does not correspond to any stored record."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' not found.")
        self.session_id = session_id


class SessionValidationError(SessionError):
    """Raised when session data fails domain validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class SessionDuplicateError(SessionError):
    """Raised when a session with the same session_id already exists."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"A session with id '{session_id}' already exists. "
            "Use update semantics if you intend to modify it."
        )
        self.session_id = session_id


class SessionStateError(SessionError):
    """Raised when a state transition is not permitted in the current state."""

    def __init__(self, session_id: str, current: SessionState, attempted: str) -> None:
        super().__init__(
            f"Session '{session_id}' is {current.name}; cannot perform '{attempted}'."
        )
        self.session_id = session_id
        self.current_state = current
        self.attempted_operation = attempted


# ---------------------------------------------------------------------------
# History entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionHistoryEntry:
    """Immutable audit record for a single session state change.

    Attributes
    ----------
    event:
        Short label describing the change (e.g. ``"created"``, ``"closed"``,
        ``"reopened"``, ``"experience_added"``, ``"experience_removed"``).
    timestamp:
        UTC timestamp of the change.
    detail:
        Optional free-text detail about the change.
    """

    event: str
    timestamp: datetime
    detail: str = ""


# ---------------------------------------------------------------------------
# Domain model: SessionRecord
# ---------------------------------------------------------------------------


@dataclass
class SessionRecord:
    """Domain record for a bounded episodic session.

    A :class:`SessionRecord` groups related experiences under a meaningful
    label.  It tracks the goals the session was started to accomplish, the
    discussions held, the decisions made, and the outcomes achieved.

    Attributes
    ----------
    title:
        Short, human-readable label
        (e.g. ``"Architecture Review Session — June 2026"``).
    session_id:
        UUID-4 string uniquely identifying this record.
    experience_id:
        UUID of the backing :class:`~subsystems.echo.models.Experience`.
        Populated automatically by :meth:`SessionEngine.create_session`.
    state:
        Current :class:`SessionState` of this session.
    goals:
        Objectives the session was started to accomplish.
    discussions:
        Key topics or threads discussed during the session.
    decisions:
        Decisions reached during this session.
    outcomes:
        Concrete outcomes or action items produced.
    experience_ids:
        Ordered list of :attr:`~subsystems.echo.models.Experience.experience_id`
        values for experiences that belong to this session.
    importance:
        :class:`~subsystems.echo.models.ExperienceImportance` tier.
    tags:
        :class:`~subsystems.echo.models.MemoryTag` list for indexing.
    opened_at:
        UTC timestamp of when the session was first created.
    closed_at:
        UTC timestamp of when the session was last closed, or ``None`` if open.
    recorded_at:
        UTC timestamp of when ECHO stored this record.
    history:
        Ordered audit trail of :class:`SessionHistoryEntry` objects.
    extra:
        Extensible key-value store for engine-specific annotations.
    """

    title: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experience_id: str | None = None
    state: SessionState = SessionState.OPEN
    goals: list[str] = field(default_factory=list)
    discussions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    experience_ids: list[str] = field(default_factory=list)
    importance: ExperienceImportance = ExperienceImportance.MEDIUM
    tags: list[MemoryTag] = field(default_factory=list)
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: list[SessionHistoryEntry] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise SessionValidationError(
                "SessionRecord.title must be a non-empty string.",
                field="title",
            )

    @property
    def is_open(self) -> bool:
        """``True`` if this session is in the OPEN state."""
        return self.state is SessionState.OPEN

    @property
    def is_closed(self) -> bool:
        """``True`` if this session is in the CLOSED state."""
        return self.state is SessionState.CLOSED

    def _append_history(self, event: str, detail: str = "") -> None:
        """Append a :class:`SessionHistoryEntry` to the audit trail."""
        self.history.append(
            SessionHistoryEntry(
                event=event,
                timestamp=datetime.now(timezone.utc),
                detail=detail,
            )
        )


# ---------------------------------------------------------------------------
# Session Engine
# ---------------------------------------------------------------------------


class SessionEngine:
    """Production Session Engine for the ECHO Episodic Memory Core.

    Manages the full lifecycle of :class:`SessionRecord` objects: creation,
    closure, reopening, experience membership, retrieval, and search.

    Parameters
    ----------
    experience_engine:
        A running :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        used to back every session with an
        :class:`~subsystems.echo.models.Experience`.
    significance_engine:
        A running :class:`~subsystems.echo.interfaces.SignificanceEngineInterface`
        consulted when computing session importance.
    min_significance_threshold:
        Minimum significance score [0.0, 1.0] required for automatic storage.
        Defaults to ``0.15``.

    Thread Safety
    -------------
    All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
    before reading or modifying internal state.

    Examples
    --------
    ::

        engine = SessionEngine(
            experience_engine=exp_engine,
            significance_engine=sig_engine,
        )
        engine.initialize()

        session = engine.create_session(
            title="Architecture Review Session — June 2026",
            goals=["Audit POLARIS subsystem contracts"],
            importance=ExperienceImportance.HIGH,
        )

        engine.add_experience(session.session_id, some_experience_id)

        engine.close_session(
            session.session_id,
            outcomes=["All contracts reviewed; ECHO v1 spec frozen."],
        )

        engine.shutdown()
    """

    def __init__(
        self,
        experience_engine: ExperienceEngineInterface,
        significance_engine: SignificanceEngineInterface,
        *,
        min_significance_threshold: float = 0.15,
    ) -> None:
        if not (0.0 <= min_significance_threshold <= 1.0):
            raise ValueError(
                "min_significance_threshold must be in [0.0, 1.0]; "
                f"got {min_significance_threshold!r}."
            )
        self._experience_engine = experience_engine
        self._significance_engine = significance_engine
        self._min_significance_threshold = min_significance_threshold

        # Primary store: session_id → SessionRecord
        self._store: dict[str, SessionRecord] = {}
        # Secondary index: experience_id (of a member) → session_id
        self._member_index: dict[str, str] = {}
        # Secondary index: backing experience_id → session_id
        self._backing_experience_index: dict[str, str] = {}

        self._lock: threading.RLock = threading.RLock()
        self._running: bool = False

        _logger.debug(
            "SessionEngine constructed (threshold=%.3f).",
            min_significance_threshold,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Sets the running flag and prepares internal state.  Idempotent —
        calling ``initialize()`` on an already-running engine is a no-op
        (logged as a warning).
        """
        with self._lock:
            if self._running:
                _logger.warning(
                    "SessionEngine.initialize() called while already running."
                )
                return
            self._running = True
            _logger.info("SessionEngine initialised.")

    def shutdown(self) -> None:
        """Release resources and mark the engine as stopped.

        Idempotent — calling ``shutdown()`` on an already-stopped engine
        is a no-op.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            _logger.info(
                "SessionEngine shut down. Store contained %d session(s).",
                len(self._store),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if not running."""
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _publish_event(self, event_name: str, payload: dict[str, Any]) -> None:
        """Emit a domain event (logged at DEBUG in v1)."""
        _logger.debug("ECHO domain event: %s — %s", event_name, payload)

    def _build_description(self, record: SessionRecord) -> str:
        """Compose the description text for the backing Experience."""
        parts: list[str] = []
        if record.goals:
            parts.append(f"Goals: {', '.join(record.goals)}")
        if record.discussions:
            parts.append(f"Discussions: {', '.join(record.discussions)}")
        if record.decisions:
            parts.append(f"Decisions: {', '.join(record.decisions)}")
        return "  ".join(parts) if parts else record.title

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_session(
        self,
        title: str,
        *,
        goals: list[str] | None = None,
        discussions: list[str] | None = None,
        decisions: list[str] | None = None,
        outcomes: list[str] | None = None,
        importance: ExperienceImportance = ExperienceImportance.MEDIUM,
        tags: list[MemoryTag] | None = None,
        opened_at: datetime | None = None,
        source_subsystem: str = "ECHO_API",
        extra: dict[str, Any] | None = None,
        force: bool = False,
    ) -> SessionRecord:
        """Create and persist a new :class:`SessionRecord`.

        A backing :class:`~subsystems.echo.models.Experience` of type
        :attr:`~subsystems.echo.models.ExperienceType.SESSION` is created
        via the Experience Engine.

        Parameters
        ----------
        title:
            Short, human-readable label for this session.
        goals:
            Objectives the session was started to accomplish.
        discussions:
            Key topics or threads to be tracked during the session.
        decisions:
            Initial decisions to record (can be added/updated later).
        outcomes:
            Initial outcomes (can be added/updated via :meth:`close_session`).
        importance:
            Caller-suggested importance tier.
        tags:
            Optional :class:`~subsystems.echo.models.MemoryTag` list.
        opened_at:
            UTC timestamp of when the session started.  Defaults to now.
        source_subsystem:
            Which POLARIS subsystem is opening this session.
        extra:
            Extensible key-value store for engine-specific annotations.
        force:
            Bypass the Significance Engine threshold gate if ``True``.

        Returns
        -------
        SessionRecord
            The created and stored session record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SessionValidationError
            If required fields are invalid.
        """
        self._assert_running("create_session")
        if not title or not title.strip():
            raise SessionValidationError(
                "SessionRecord.title must be a non-empty string.",
                field="title",
            )

        opened_at = opened_at or datetime.now(timezone.utc)
        goals = goals or []
        discussions = discussions or []
        decisions = decisions or []
        outcomes = outcomes or []
        tags = tags or []
        extra = extra or {}

        record = SessionRecord(
            title=title,
            state=SessionState.OPEN,
            goals=list(goals),
            discussions=list(discussions),
            decisions=list(decisions),
            outcomes=list(outcomes),
            importance=importance,
            tags=list(tags),
            opened_at=opened_at,
            recorded_at=datetime.now(timezone.utc),
            extra=dict(extra),
        )
        record._append_history("created", f"Session '{title}' opened.")

        # Build description for the backing Experience
        description = self._build_description(record)
        outcome_text = "; ".join(outcomes) if outcomes else ""

        experience = self._experience_engine.create_experience(
            title=title,
            experience_type=ExperienceType.SESSION,
            importance=importance,
            description=description,
            outcome=outcome_text,
            tags=list(tags),
            occurred_at=opened_at,
            source_subsystem=source_subsystem,
            extra=dict(extra),
            force=force,
        )
        record.experience_id = experience.experience_id

        with self._lock:
            self._store[record.session_id] = record
            self._backing_experience_index[experience.experience_id] = record.session_id

        self._publish_event(
            "polaris.echo.session.created",
            {
                "session_id": record.session_id,
                "experience_id": experience.experience_id,
                "title": title,
            },
        )
        _logger.info(
            "SessionEngine: created session '%s' (id=%s).",
            title,
            record.session_id,
        )
        return record

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close_session(
        self,
        session_id: str,
        *,
        outcomes: list[str] | None = None,
        decisions: list[str] | None = None,
        discussions: list[str] | None = None,
        closed_at: datetime | None = None,
    ) -> SessionRecord:
        """Close an open session, marking it as a completed episodic unit.

        Closing a session finalises its record and updates the backing
        :class:`~subsystems.echo.models.Experience` with outcome information.

        Parameters
        ----------
        session_id:
            UUID of the session to close.
        outcomes:
            Final outcomes to record.  Merged with any existing outcomes.
        decisions:
            Final decisions to record.  Merged with existing decisions.
        discussions:
            Final discussion items to record.  Merged with existing items.
        closed_at:
            UTC timestamp of closure.  Defaults to now.

        Returns
        -------
        SessionRecord
            The updated, closed session record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SessionNotFoundError
            If no session with ``session_id`` exists.
        SessionStateError
            If the session is already CLOSED.
        """
        self._assert_running("close_session")

        closed_at = closed_at or datetime.now(timezone.utc)

        with self._lock:
            record = self._store.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)
            if record.state is SessionState.CLOSED:
                raise SessionStateError(session_id, record.state, "close_session")

            if outcomes:
                record.outcomes.extend(outcomes)
            if decisions:
                record.decisions.extend(decisions)
            if discussions:
                record.discussions.extend(discussions)

            record.state = SessionState.CLOSED
            record.closed_at = closed_at
            record._append_history(
                "closed",
                f"Session closed at {closed_at.isoformat()}.",
            )

        # Mirror outcome/decision updates to the backing Experience
        if record.experience_id is not None:
            try:
                self._experience_engine.update_experience(
                    record.experience_id,
                    outcome="; ".join(record.outcomes),
                    description=self._build_description(record),
                )
            except ExperienceNotFoundError:
                _logger.warning(
                    "SessionEngine: backing experience '%s' not found during "
                    "close of session '%s'. Skipping experience sync.",
                    record.experience_id,
                    session_id,
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "SessionEngine: failed to sync experience during close "
                    "of session '%s': %s",
                    session_id,
                    exc,
                )

        self._publish_event(
            "polaris.echo.session.closed",
            {"session_id": session_id, "closed_at": closed_at.isoformat()},
        )
        _logger.info("SessionEngine: closed session '%s'.", session_id)
        return record

    # ------------------------------------------------------------------
    # Reopen
    # ------------------------------------------------------------------

    def reopen_session(
        self,
        session_id: str,
        *,
        reason: str = "",
    ) -> SessionRecord:
        """Reopen a closed session, allowing further experience additions.

        Reopening signals that the episodic context is being revisited —
        for example, when a project phase is resumed after a pause.

        Parameters
        ----------
        session_id:
            UUID of the session to reopen.
        reason:
            Optional rationale for reopening, recorded in the audit trail.

        Returns
        -------
        SessionRecord
            The updated, reopened session record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SessionNotFoundError
            If no session with ``session_id`` exists.
        SessionStateError
            If the session is already OPEN.
        """
        self._assert_running("reopen_session")

        with self._lock:
            record = self._store.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)
            if record.state is SessionState.OPEN:
                raise SessionStateError(session_id, record.state, "reopen_session")

            record.state = SessionState.OPEN
            record.closed_at = None
            detail = f"Session reopened. Reason: {reason}" if reason else "Session reopened."
            record._append_history("reopened", detail)

        self._publish_event(
            "polaris.echo.session.reopened",
            {"session_id": session_id, "reason": reason},
        )
        _logger.info("SessionEngine: reopened session '%s'.", session_id)
        return record

    # ------------------------------------------------------------------
    # Experience membership
    # ------------------------------------------------------------------

    def add_experience(
        self,
        session_id: str,
        experience_id: str,
    ) -> SessionRecord:
        """Add an experience to a session's membership list.

        Parameters
        ----------
        session_id:
            UUID of the session.
        experience_id:
            UUID of the :class:`~subsystems.echo.models.Experience` to add.

        Returns
        -------
        SessionRecord
            The updated session record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SessionNotFoundError
            If no session with ``session_id`` exists.
        SessionStateError
            If the session is CLOSED (use reopen first).
        SessionValidationError
            If ``experience_id`` is blank.
        """
        self._assert_running("add_experience")

        if not experience_id or not experience_id.strip():
            raise SessionValidationError(
                "experience_id must be a non-empty string.",
                field="experience_id",
            )

        with self._lock:
            record = self._store.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)
            if record.state is SessionState.CLOSED:
                raise SessionStateError(session_id, record.state, "add_experience")

            if experience_id not in record.experience_ids:
                record.experience_ids.append(experience_id)
                self._member_index[experience_id] = session_id
                record._append_history(
                    "experience_added",
                    f"Experience '{experience_id}' added to session.",
                )

        self._publish_event(
            "polaris.echo.session.experience_added",
            {"session_id": session_id, "experience_id": experience_id},
        )
        _logger.debug(
            "SessionEngine: added experience '%s' to session '%s'.",
            experience_id,
            session_id,
        )
        return record

    def remove_experience(
        self,
        session_id: str,
        experience_id: str,
    ) -> SessionRecord:
        """Remove an experience from a session's membership list.

        This does NOT delete the experience from the Experience Engine —
        it only removes the membership association.

        Parameters
        ----------
        session_id:
            UUID of the session.
        experience_id:
            UUID of the :class:`~subsystems.echo.models.Experience` to remove.

        Returns
        -------
        SessionRecord
            The updated session record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SessionNotFoundError
            If no session with ``session_id`` exists.
        SessionValidationError
            If ``experience_id`` is blank.
        """
        self._assert_running("remove_experience")

        if not experience_id or not experience_id.strip():
            raise SessionValidationError(
                "experience_id must be a non-empty string.",
                field="experience_id",
            )

        with self._lock:
            record = self._store.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)

            if experience_id in record.experience_ids:
                record.experience_ids.remove(experience_id)
                self._member_index.pop(experience_id, None)
                record._append_history(
                    "experience_removed",
                    f"Experience '{experience_id}' removed from session.",
                )

        self._publish_event(
            "polaris.echo.session.experience_removed",
            {"session_id": session_id, "experience_id": experience_id},
        )
        _logger.debug(
            "SessionEngine: removed experience '%s' from session '%s'.",
            experience_id,
            session_id,
        )
        return record

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> SessionRecord:
        """Retrieve a single :class:`SessionRecord` by its UUID.

        Parameters
        ----------
        session_id:
            UUID of the session to retrieve.

        Returns
        -------
        SessionRecord
            The stored record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SessionNotFoundError
            If no session with ``session_id`` exists.
        """
        self._assert_running("get_session")

        with self._lock:
            record = self._store.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)
            return record

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_sessions(
        self,
        *,
        state: SessionState | None = None,
        importance: ExperienceImportance | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        tag_name: str | None = None,
        goal_keyword: str | None = None,
        limit: int | None = None,
    ) -> list[SessionRecord]:
        """Search stored sessions by one or more filter criteria.

        All supplied criteria are ANDed together.  Omitting a criterion
        means "no filter on that dimension".

        Parameters
        ----------
        state:
            Filter to sessions in this :class:`SessionState`.
        importance:
            Filter to sessions at this importance tier or above.
        since:
            Filter to sessions that opened at or after this timestamp.
        until:
            Filter to sessions that opened at or before this timestamp.
        tag_name:
            Filter to sessions that carry a tag with this name
            (case-insensitive exact match).
        goal_keyword:
            Filter to sessions whose goals contain this keyword
            (case-insensitive substring match against each goal string).
        limit:
            Maximum number of results to return, ordered by most recent first.

        Returns
        -------
        list[SessionRecord]
            Matching session records, sorted by ``opened_at`` descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("search_sessions")

        with self._lock:
            candidates: list[SessionRecord] = list(self._store.values())

        results: list[SessionRecord] = []

        for record in candidates:
            if state is not None and record.state is not state:
                continue

            if importance is not None and record.importance.value < importance.value:
                continue

            if since is not None:
                ts = record.opened_at
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < since:
                    continue

            if until is not None:
                ts = record.opened_at
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > until:
                    continue

            if tag_name is not None:
                term = tag_name.lower()
                if not any(t.name.lower() == term for t in record.tags):
                    continue

            if goal_keyword is not None:
                term = goal_keyword.lower()
                if not any(term in g.lower() for g in record.goals):
                    continue

            results.append(record)

        # Sort by opened_at descending (most recent first)
        results.sort(
            key=lambda r: r.opened_at.replace(tzinfo=timezone.utc)
            if r.opened_at.tzinfo is None
            else r.opened_at,
            reverse=True,
        )

        if limit is not None and limit > 0:
            results = results[:limit]

        return results

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_session_history(self, session_id: str) -> list[SessionHistoryEntry]:
        """Return the full audit trail for a session.

        Parameters
        ----------
        session_id:
            UUID of the session whose history to retrieve.

        Returns
        -------
        list[SessionHistoryEntry]
            Ordered list of :class:`SessionHistoryEntry` objects, oldest first.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SessionNotFoundError
            If no session with ``session_id`` exists.
        """
        self._assert_running("get_session_history")

        with self._lock:
            record = self._store.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)
            return list(record.history)

    # ------------------------------------------------------------------
    # Read-only statistics
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of stored session records."""
        with self._lock:
            return len(self._store)

    def count_open(self) -> int:
        """Return the number of currently open sessions."""
        with self._lock:
            return sum(
                1 for r in self._store.values() if r.state is SessionState.OPEN
            )

    def is_running(self) -> bool:
        """Return ``True`` if the engine has been initialised and not shut down."""
        return self._running

    def find_session_for_experience(self, experience_id: str) -> SessionRecord | None:
        """Return the session that contains a given member experience, or ``None``.

        Parameters
        ----------
        experience_id:
            UUID of the member experience to look up.

        Returns
        -------
        SessionRecord | None
            The containing session, or ``None`` if the experience is not a
            member of any session.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("find_session_for_experience")

        with self._lock:
            session_id = self._member_index.get(experience_id)
            if session_id is None:
                return None
            return self._store.get(session_id)


__all__ = [
    "SessionEngine",
    "SessionRecord",
    "SessionHistoryEntry",
    "SessionState",
    "SessionError",
    "SessionNotFoundError",
    "SessionValidationError",
    "SessionDuplicateError",
    "SessionStateError",
]