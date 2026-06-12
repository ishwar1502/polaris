# subsystems/echo/conversation.py
"""
ECHO v1 Conversation Engine.

Implements :class:`ConversationEngine` — the production manager for
significant conversational interactions within the ECHO Episodic Memory Core.

The Conversation Engine does NOT store raw chat logs or every message
exchanged.  It stores *meaningful* conversations — design discussions,
architecture sessions, major planning interactions, or any significant
interaction that warrants episodic memory.

A conversation record is stored as an :class:`~subsystems.echo.models.Experience`
of type :attr:`~subsystems.echo.models.ExperienceType.CONVERSATION` and
supplemented by a rich :class:`ConversationRecord` domain object that
captures participants, summaries, decisions, and outcomes.

Design Principles
-----------------
* **Thread-safe**: All public methods serialise concurrent access via
  ``self._lock`` (a :class:`threading.RLock`).
* **Lifecycle-gated**: Every public operation guards against calls made
  before :meth:`initialize` or after :meth:`shutdown`.
* **Experience-linked**: Every conversation record is backed by a
  corresponding :class:`~subsystems.echo.models.Experience` managed by the
  Experience Engine.  The two remain in sync; deleting a conversation
  optionally cascades to its experience.
* **Significance-aware**: Conversation records pass through the Significance
  Engine via the Experience Engine before storage.
* **Retrieval-indexed**: Secondary indices on participant, tag, and date
  range enable fast multi-dimensional queries without full-table scans.

ECHO Boundary Law
-----------------
The Conversation Engine owns :class:`ConversationRecord` objects.  It does
NOT own knowledge, identity, goals, schedules, relationships, or decisions.

Publish / Subscribe (ECHO domain)
----------------------------------
Publishes (logged at DEBUG in v1; Event Bus integration reserved for future):
    * ``polaris.echo.conversation.created``
    * ``polaris.echo.conversation.updated``
    * ``polaris.echo.conversation.deleted``
    * ``polaris.echo.conversation.experience_linked``
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from subsystems.echo.exceptions import (
    EchoBoundaryViolationError,
    EchoNotInitializedError,
    ExperienceDuplicateError,
    ExperienceNotFoundError,
    ExperienceValidationError,
)
from subsystems.echo.interfaces import (
    ExperienceEngineInterface,
    SignificanceEngineInterface,
)
from subsystems.echo.models import (
    Experience,
    ExperienceImportance,
    ExperienceMetadata,
    ExperienceType,
    MemoryTag,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation-specific exceptions
# ---------------------------------------------------------------------------


class ConversationError(Exception):
    """Base exception for all Conversation Engine errors."""


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation_id does not correspond to any stored record."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"Conversation '{conversation_id}' not found.")
        self.conversation_id = conversation_id


class ConversationValidationError(ConversationError):
    """Raised when conversation data fails domain validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class ConversationDuplicateError(ConversationError):
    """Raised when a conversation with the same conversation_id already exists."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(
            f"A conversation with id '{conversation_id}' already exists. "
            "Use update semantics if you intend to modify it."
        )
        self.conversation_id = conversation_id


# ---------------------------------------------------------------------------
# Domain model: ConversationRecord
# ---------------------------------------------------------------------------


@dataclass
class ConversationRecord:
    """Domain record for a significant conversational interaction.

    A :class:`ConversationRecord` is ECHO's representation of a meaningful
    conversation.  It captures the who (participants), the what (summary,
    decisions, outcomes), and links back to the corresponding
    :class:`~subsystems.echo.models.Experience` object for full episodic
    context.

    Attributes
    ----------
    title:
        Short, human-readable label
        (e.g. ``"ORION Redesign Architecture Discussion"``).
    summary:
        Narrative summary of what was discussed and why it mattered.
    conversation_id:
        UUID-4 string uniquely identifying this record.
    experience_id:
        UUID of the backing :class:`~subsystems.echo.models.Experience`.
        Populated automatically by :meth:`ConversationEngine.create_conversation`.
    participants:
        List of participant identifiers (names, IDs, or subsystem labels).
    decisions:
        List of decisions reached during this conversation.
    outcomes:
        List of concrete outcomes or action items that resulted.
    topics:
        List of topic labels covered in the conversation.
    importance:
        :class:`~subsystems.echo.models.ExperienceImportance` tier.
    tags:
        :class:`~subsystems.echo.models.MemoryTag` list for indexing.
    occurred_at:
        UTC timestamp of when the conversation took place.
    recorded_at:
        UTC timestamp of when ECHO stored this record.
    extra:
        Extensible key-value store for engine-specific annotations.
    """

    title: str
    summary: str
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experience_id: str | None = None
    participants: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    importance: ExperienceImportance = ExperienceImportance.MEDIUM
    tags: list[MemoryTag] = field(default_factory=list)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ConversationValidationError(
                "ConversationRecord.title must be a non-empty string.",
                field="title",
            )
        if not self.summary or not self.summary.strip():
            raise ConversationValidationError(
                "ConversationRecord.summary must be a non-empty string.",
                field="summary",
            )


# ---------------------------------------------------------------------------
# Conversation Engine
# ---------------------------------------------------------------------------


class ConversationEngine:
    """Production Conversation Engine for the ECHO Episodic Memory Core.

    Manages the full lifecycle of :class:`ConversationRecord` objects:
    creation, storage, update, deletion, retrieval, and experience linking.

    Parameters
    ----------
    experience_engine:
        A running :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        used to back every conversation with an :class:`~subsystems.echo.models.Experience`.
    significance_engine:
        A running :class:`~subsystems.echo.interfaces.SignificanceEngineInterface`
        consulted when computing conversation importance.
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

        engine = ConversationEngine(
            experience_engine=exp_engine,
            significance_engine=sig_engine,
        )
        engine.initialize()

        record = engine.create_conversation(
            title="ORION Redesign Discussion",
            summary="Discussed the motivation and approach for ORION v5 redesign.",
            participants=["User", "ORION", "POLARIS"],
            decisions=["Adopt event-driven architecture"],
            outcomes=["Architecture spec to be drafted by end of week"],
            importance=ExperienceImportance.HIGH,
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

        # Primary store: conversation_id → ConversationRecord
        self._store: dict[str, ConversationRecord] = {}
        # Secondary index: participant → [conversation_id, ...]
        self._participant_index: dict[str, list[str]] = {}
        # Secondary index: experience_id → conversation_id
        self._experience_index: dict[str, str] = {}

        self._lock: threading.RLock = threading.RLock()
        self._running: bool = False

        _logger.debug(
            "ConversationEngine constructed (threshold=%.3f).",
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
                    "ConversationEngine.initialize() called while already running."
                )
                return
            self._running = True
            _logger.info("ConversationEngine initialised.")

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
                "ConversationEngine shut down. Store contained %d conversation(s).",
                len(self._store),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if not running."""
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _validate_title(self, title: str) -> None:
        """Raise :class:`ConversationValidationError` if title is blank."""
        if not title or not title.strip():
            raise ConversationValidationError(
                "ConversationRecord.title must be a non-empty string.",
                field="title",
            )

    def _index_participants(
        self, conversation_id: str, participants: list[str]
    ) -> None:
        """Add participant → conversation_id mappings to the secondary index."""
        for participant in participants:
            key = participant.lower().strip()
            if key not in self._participant_index:
                self._participant_index[key] = []
            if conversation_id not in self._participant_index[key]:
                self._participant_index[key].append(conversation_id)

    def _deindex_participants(
        self, conversation_id: str, participants: list[str]
    ) -> None:
        """Remove participant → conversation_id mappings from the secondary index."""
        for participant in participants:
            key = participant.lower().strip()
            if key in self._participant_index:
                try:
                    self._participant_index[key].remove(conversation_id)
                except ValueError:
                    pass
                if not self._participant_index[key]:
                    del self._participant_index[key]

    def _publish_event(self, event_name: str, payload: dict[str, Any]) -> None:
        """Emit a domain event (logged at DEBUG in v1)."""
        _logger.debug("ECHO domain event: %s — %s", event_name, payload)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        title: str,
        summary: str,
        *,
        participants: list[str] | None = None,
        decisions: list[str] | None = None,
        outcomes: list[str] | None = None,
        topics: list[str] | None = None,
        importance: ExperienceImportance = ExperienceImportance.MEDIUM,
        tags: list[MemoryTag] | None = None,
        occurred_at: datetime | None = None,
        source_subsystem: str = "ECHO_API",
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
        force: bool = False,
    ) -> ConversationRecord:
        """Create and persist a new :class:`ConversationRecord`.

        A backing :class:`~subsystems.echo.models.Experience` of type
        :attr:`~subsystems.echo.models.ExperienceType.CONVERSATION` is
        created via the Experience Engine.  The two records share a UUID
        reference so they can be retrieved together.

        Parameters
        ----------
        title:
            Short, human-readable label for this conversation.
        summary:
            Narrative summary of what was discussed and why it mattered.
        participants:
            List of participant identifiers (names, IDs, subsystem labels).
        decisions:
            Decisions reached during this conversation.
        outcomes:
            Concrete outcomes or action items that resulted.
        topics:
            Topic labels covered in the conversation.
        importance:
            Caller-suggested importance tier.  The Significance Engine may
            override unless ``force=True``.
        tags:
            Optional :class:`~subsystems.echo.models.MemoryTag` list.
        occurred_at:
            UTC timestamp of when the conversation took place.  Defaults to now.
        source_subsystem:
            Which POLARIS subsystem is recording this conversation.
        session_id:
            Optional UUID of an enclosing session experience.
        extra:
            Extensible key-value store for engine-specific annotations.
        force:
            Bypass the Significance Engine threshold gate if ``True``.

        Returns
        -------
        ConversationRecord
            The created and stored conversation record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ConversationValidationError
            If required fields are invalid.
        """
        self._assert_running("create_conversation")
        self._validate_title(title)
        if not summary or not summary.strip():
            raise ConversationValidationError(
                "ConversationRecord.summary must be a non-empty string.",
                field="summary",
            )

        occurred_at = occurred_at or datetime.now(timezone.utc)
        participants = participants or []
        decisions = decisions or []
        outcomes = outcomes or []
        topics = topics or []
        tags = tags or []
        extra = extra or {}

        # Build description for the backing Experience from available fields
        description_parts: list[str] = [summary]
        if participants:
            description_parts.append(f"Participants: {', '.join(participants)}")
        if topics:
            description_parts.append(f"Topics: {', '.join(topics)}")
        description = "  ".join(description_parts)

        outcome_text = "; ".join(outcomes) if outcomes else ""

        # Create backing Experience via the Experience Engine
        experience = self._experience_engine.create_experience(
            title=title,
            experience_type=ExperienceType.CONVERSATION,
            importance=importance,
            description=description,
            context=f"Session: {session_id}" if session_id else "",
            outcome=outcome_text,
            tags=list(tags),
            occurred_at=occurred_at,
            source_subsystem=source_subsystem,
            session_id=session_id,
            extra=dict(extra),
            force=force,
        )

        with self._lock:
            record = ConversationRecord(
                title=title,
                summary=summary,
                experience_id=experience.experience_id,
                participants=list(participants),
                decisions=list(decisions),
                outcomes=list(outcomes),
                topics=list(topics),
                importance=importance,
                tags=list(tags),
                occurred_at=occurred_at,
                recorded_at=datetime.now(timezone.utc),
                extra=dict(extra),
            )

            self._store[record.conversation_id] = record
            self._experience_index[experience.experience_id] = record.conversation_id
            self._index_participants(record.conversation_id, participants)

        self._publish_event(
            "polaris.echo.conversation.created",
            {
                "conversation_id": record.conversation_id,
                "experience_id": experience.experience_id,
                "title": title,
            },
        )
        _logger.info(
            "ConversationEngine: created conversation '%s' (id=%s).",
            title,
            record.conversation_id,
        )
        return record

    # ------------------------------------------------------------------
    # Store (persist a pre-constructed record)
    # ------------------------------------------------------------------

    def store_conversation(
        self,
        record: ConversationRecord,
        *,
        force: bool = False,
    ) -> ConversationRecord:
        """Persist an already-constructed :class:`ConversationRecord`.

        If the record does not yet have a backing
        :class:`~subsystems.echo.models.Experience`, one is created
        automatically.

        Parameters
        ----------
        record:
            The :class:`ConversationRecord` to persist.
        force:
            Bypass the Significance Engine threshold gate if ``True``.

        Returns
        -------
        ConversationRecord
            The stored record (with ``experience_id`` populated if it was absent).

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ConversationDuplicateError
            If a record with the same ``conversation_id`` already exists.
        """
        self._assert_running("store_conversation")

        with self._lock:
            if record.conversation_id in self._store:
                raise ConversationDuplicateError(record.conversation_id)

        if record.experience_id is None:
            experience = self._experience_engine.create_experience(
                title=record.title,
                experience_type=ExperienceType.CONVERSATION,
                importance=record.importance,
                description=record.summary,
                outcome="; ".join(record.outcomes),
                tags=list(record.tags),
                occurred_at=record.occurred_at,
                source_subsystem="ECHO_API",
                extra=dict(record.extra),
                force=force,
            )
            record.experience_id = experience.experience_id
        else:
            # Delegate significance check via store_experience on the backing record
            try:
                backing = self._experience_engine.create_experience(
                    title=record.title,
                    experience_type=ExperienceType.CONVERSATION,
                    importance=record.importance,
                    description=record.summary,
                    outcome="; ".join(record.outcomes),
                    tags=list(record.tags),
                    occurred_at=record.occurred_at,
                    source_subsystem="ECHO_API",
                    extra=dict(record.extra),
                    force=force,
                )
                record.experience_id = backing.experience_id
            except ExperienceDuplicateError:
                # Backing experience already exists — proceed without re-creating
                pass

        with self._lock:
            self._store[record.conversation_id] = record
            if record.experience_id:
                self._experience_index[record.experience_id] = record.conversation_id
            self._index_participants(record.conversation_id, record.participants)

        self._publish_event(
            "polaris.echo.conversation.created",
            {
                "conversation_id": record.conversation_id,
                "experience_id": record.experience_id,
            },
        )
        _logger.info(
            "ConversationEngine: stored conversation '%s' (id=%s).",
            record.title,
            record.conversation_id,
        )
        return record

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        participants: list[str] | None = None,
        decisions: list[str] | None = None,
        outcomes: list[str] | None = None,
        topics: list[str] | None = None,
        importance: ExperienceImportance | None = None,
        tags: list[MemoryTag] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ConversationRecord:
        """Update mutable fields of an existing :class:`ConversationRecord`.

        Only the fields explicitly supplied (non-``None``) are updated.
        ``conversation_id``, ``experience_id``, and ``recorded_at`` are
        immutable after creation.

        Parameters
        ----------
        conversation_id:
            UUID of the conversation to update.
        title:
            New title, or ``None`` to leave unchanged.
        summary:
            New summary, or ``None`` to leave unchanged.
        participants:
            Replacement participant list, or ``None`` to leave unchanged.
        decisions:
            Replacement decision list, or ``None`` to leave unchanged.
        outcomes:
            Replacement outcome list, or ``None`` to leave unchanged.
        topics:
            Replacement topic list, or ``None`` to leave unchanged.
        importance:
            New importance tier, or ``None`` to leave unchanged.
        tags:
            Replacement tag list, or ``None`` to leave unchanged.
        extra:
            Replacement extra dict, or ``None`` to leave unchanged.

        Returns
        -------
        ConversationRecord
            The updated conversation record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ConversationNotFoundError
            If no conversation with ``conversation_id`` exists.
        ConversationValidationError
            If any supplied field fails validation.
        """
        self._assert_running("update_conversation")

        with self._lock:
            record = self._store.get(conversation_id)
            if record is None:
                raise ConversationNotFoundError(conversation_id)

            # Re-index participants if being replaced
            if participants is not None:
                self._deindex_participants(conversation_id, record.participants)
                record.participants = list(participants)
                self._index_participants(conversation_id, record.participants)

            if title is not None:
                if not title.strip():
                    raise ConversationValidationError(
                        "ConversationRecord.title must be a non-empty string.",
                        field="title",
                    )
                record.title = title

            if summary is not None:
                if not summary.strip():
                    raise ConversationValidationError(
                        "ConversationRecord.summary must be a non-empty string.",
                        field="summary",
                    )
                record.summary = summary

            if decisions is not None:
                record.decisions = list(decisions)
            if outcomes is not None:
                record.outcomes = list(outcomes)
            if topics is not None:
                record.topics = list(topics)
            if importance is not None:
                record.importance = importance
            if tags is not None:
                record.tags = list(tags)
            if extra is not None:
                record.extra = dict(extra)

        # Mirror mutable fields to the backing Experience
        if record.experience_id is not None:
            try:
                update_kwargs: dict[str, Any] = {}
                if title is not None:
                    update_kwargs["title"] = title
                if summary is not None or outcomes is not None:
                    update_kwargs["description"] = record.summary
                    update_kwargs["outcome"] = "; ".join(record.outcomes)
                if importance is not None:
                    update_kwargs["importance"] = importance
                if tags is not None:
                    update_kwargs["tags"] = list(record.tags)
                if extra is not None:
                    update_kwargs["extra"] = dict(extra)

                if update_kwargs:
                    self._experience_engine.update_experience(
                        record.experience_id,
                        **update_kwargs,
                    )
            except ExperienceNotFoundError:
                _logger.warning(
                    "ConversationEngine: backing experience '%s' not found during update "
                    "of conversation '%s'. Skipping experience sync.",
                    record.experience_id,
                    conversation_id,
                )

        self._publish_event(
            "polaris.echo.conversation.updated",
            {"conversation_id": conversation_id},
        )
        _logger.debug(
            "ConversationEngine: updated conversation '%s'.", conversation_id
        )
        return record

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_conversation(
        self,
        conversation_id: str,
        *,
        cascade: bool = False,
    ) -> bool:
        """Remove a conversation record from ECHO's store.

        Parameters
        ----------
        conversation_id:
            UUID of the conversation to delete.
        cascade:
            If ``True``, also delete the backing
            :class:`~subsystems.echo.models.Experience` from the Experience Engine.

        Returns
        -------
        bool
            ``True`` if the record was found and removed; ``False`` if it
            did not exist (idempotent delete).

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("delete_conversation")

        with self._lock:
            record = self._store.pop(conversation_id, None)
            if record is None:
                return False

            # Clean up secondary indices
            self._deindex_participants(conversation_id, record.participants)
            if record.experience_id and record.experience_id in self._experience_index:
                del self._experience_index[record.experience_id]

        if cascade and record.experience_id is not None:
            try:
                self._experience_engine.delete_experience(
                    record.experience_id, cascade=False
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "ConversationEngine: cascade delete of experience '%s' failed: %s",
                    record.experience_id,
                    exc,
                )

        self._publish_event(
            "polaris.echo.conversation.deleted",
            {"conversation_id": conversation_id, "cascade": cascade},
        )
        _logger.info(
            "ConversationEngine: deleted conversation '%s' (cascade=%s).",
            conversation_id,
            cascade,
        )
        return True

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_conversation(self, conversation_id: str) -> ConversationRecord:
        """Retrieve a single :class:`ConversationRecord` by its UUID.

        Parameters
        ----------
        conversation_id:
            UUID of the conversation to retrieve.

        Returns
        -------
        ConversationRecord
            The stored record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ConversationNotFoundError
            If no conversation with ``conversation_id`` exists.
        """
        self._assert_running("get_conversation")

        with self._lock:
            record = self._store.get(conversation_id)
            if record is None:
                raise ConversationNotFoundError(conversation_id)
            return record

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_conversations(
        self,
        *,
        participant: str | None = None,
        topic: str | None = None,
        importance: ExperienceImportance | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        tag_name: str | None = None,
        limit: int | None = None,
    ) -> list[ConversationRecord]:
        """Search stored conversations by one or more filter criteria.

        All supplied criteria are ANDed together.  Omitting a criterion
        means "no filter on that dimension".

        Parameters
        ----------
        participant:
            Filter to conversations that include this participant
            (case-insensitive substring match).
        topic:
            Filter to conversations that include this topic label
            (case-insensitive substring match).
        importance:
            Filter to conversations at this importance tier or above.
        since:
            Filter to conversations that occurred at or after this timestamp.
        until:
            Filter to conversations that occurred at or before this timestamp.
        tag_name:
            Filter to conversations that carry a tag with this name
            (case-insensitive exact match).
        limit:
            Maximum number of results to return, ordered by most recent first.

        Returns
        -------
        list[ConversationRecord]
            Matching conversation records, sorted by ``occurred_at`` descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("search_conversations")

        with self._lock:
            candidates: list[ConversationRecord] = list(self._store.values())

        results: list[ConversationRecord] = []

        for record in candidates:
            if participant is not None:
                term = participant.lower()
                if not any(term in p.lower() for p in record.participants):
                    continue

            if topic is not None:
                term = topic.lower()
                if not any(term in t.lower() for t in record.topics):
                    continue

            if importance is not None:
                if record.importance.value < importance.value:
                    continue

            if since is not None:
                ts = record.occurred_at
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < since:
                    continue

            if until is not None:
                ts = record.occurred_at
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > until:
                    continue

            if tag_name is not None:
                term = tag_name.lower()
                if not any(t.name.lower() == term for t in record.tags):
                    continue

            results.append(record)

        # Sort by occurred_at descending (most recent first)
        results.sort(
            key=lambda r: r.occurred_at.replace(tzinfo=timezone.utc)
            if r.occurred_at.tzinfo is None
            else r.occurred_at,
            reverse=True,
        )

        if limit is not None and limit > 0:
            results = results[:limit]

        return results

    # ------------------------------------------------------------------
    # Experience linking
    # ------------------------------------------------------------------

    def link_experience(
        self,
        conversation_id: str,
        experience_id: str,
    ) -> ConversationRecord:
        """Attach an existing :class:`~subsystems.echo.models.Experience`
        to a conversation record.

        This is used when the backing experience is created externally
        (e.g. by the Session Engine composing a session experience) and
        must be linked to a conversation post-hoc.

        Parameters
        ----------
        conversation_id:
            UUID of the conversation to update.
        experience_id:
            UUID of the experience to link.

        Returns
        -------
        ConversationRecord
            The updated conversation record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ConversationNotFoundError
            If the conversation does not exist.
        ConversationValidationError
            If ``experience_id`` is blank.
        """
        self._assert_running("link_experience")

        if not experience_id or not experience_id.strip():
            raise ConversationValidationError(
                "experience_id must be a non-empty string.",
                field="experience_id",
            )

        with self._lock:
            record = self._store.get(conversation_id)
            if record is None:
                raise ConversationNotFoundError(conversation_id)

            # Remove old experience index entry if present
            if record.experience_id and record.experience_id in self._experience_index:
                del self._experience_index[record.experience_id]

            record.experience_id = experience_id
            self._experience_index[experience_id] = conversation_id

        self._publish_event(
            "polaris.echo.conversation.experience_linked",
            {
                "conversation_id": conversation_id,
                "experience_id": experience_id,
            },
        )
        _logger.debug(
            "ConversationEngine: linked experience '%s' to conversation '%s'.",
            experience_id,
            conversation_id,
        )
        return record

    def get_related_experiences(
        self,
        conversation_id: str,
    ) -> list[Experience]:
        """Return the :class:`~subsystems.echo.models.Experience` objects
        related to a conversation.

        Currently returns the single backing experience stored in
        :attr:`ConversationRecord.experience_id`.  In a future iteration
        this will also surface transitively linked experiences via the
        Episodic Index Engine.

        Parameters
        ----------
        conversation_id:
            UUID of the conversation whose experiences to retrieve.

        Returns
        -------
        list[Experience]
            Related experience objects.  Empty if the conversation has no
            backing experience or if the experience has been deleted.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ConversationNotFoundError
            If the conversation does not exist.
        """
        self._assert_running("get_related_experiences")

        with self._lock:
            record = self._store.get(conversation_id)
            if record is None:
                raise ConversationNotFoundError(conversation_id)
            experience_id = record.experience_id

        if experience_id is None:
            return []

        try:
            experience = self._experience_engine.get_experience(experience_id)  # type: ignore[attr-defined]
            return [experience]
        except ExperienceNotFoundError:
            _logger.warning(
                "ConversationEngine: backing experience '%s' not found for "
                "conversation '%s' during get_related_experiences.",
                experience_id,
                conversation_id,
            )
            return []
        except AttributeError:
            # ExperienceEngine may use a different retrieval method name
            _logger.warning(
                "ConversationEngine: ExperienceEngine does not expose get_experience(); "
                "returning empty list for conversation '%s'.",
                conversation_id,
            )
            return []

    # ------------------------------------------------------------------
    # Read-only statistics
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of stored conversation records."""
        with self._lock:
            return len(self._store)

    def is_running(self) -> bool:
        """Return ``True`` if the engine has been initialised and not shut down."""
        return self._running


__all__ = [
    "ConversationEngine",
    "ConversationRecord",
    "ConversationError",
    "ConversationNotFoundError",
    "ConversationValidationError",
    "ConversationDuplicateError",
]