"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/events.py

Concrete in-memory implementation of the LUNA Event Engine.

Owns and manages all LUNA subsystem events — the authoritative record of every
significant lifecycle transition, mutation, validation, synthesis, evolution,
indexing, and integrity operation performed across the LUNA knowledge store.

Event categories:
    KnowledgeCreated       — a new knowledge record was registered
    KnowledgeUpdated       — an existing record was mutated
    KnowledgeDeleted       — a record was soft-deleted / retracted / archived
    KnowledgeValidated     — a validation pass completed
    KnowledgeSynthesized   — a KnowledgeSynthesis or KnowledgePackage was produced
    KnowledgeEvolved       — a record transitioned status or confidence via evolution
    KnowledgeIndexed       — the index was updated (incremental or full rebuild)
    KnowledgeAudited       — an integrity or semantic audit was completed
    RelationshipCreated    — a ConceptRelationship edge was registered
    RelationshipRemoved    — a ConceptRelationship edge was deleted
    SkillProgressed        — a SkillProgressionModel stage was advanced
    DomainUpdated          — a KnowledgeDomain was mutated

Supported event areas:
    knowledge lifecycle · validation · synthesis · evolution ·
    indexing · semantic structure · integrity

Responsibilities:
    publish_event      — validate and store a new LunaEvent
    record_event       — alias for publish_event (lower-level programmatic API)
    retrieve_event     — fetch a single event by ID
    search_events      — full-text + field query over stored events
    filter_events      — structured filter by category / knowledge type / time range
    replay_events      — yield events in chronological order for a replay window
    clear_events       — purge all events (test / reset use-cases)
    export_events      — serialise all matching events to a list of dicts
    event_audit        — structural integrity check of the event store
    event_reporting    — human-readable statistics snapshot

Thread safety:    threading.RLock on every public operation.
Lifecycle-gated:  every public method raises LunaNotInitializedError before
                  initialize() or after shutdown().
In-memory v1 implementation.  No persistence layer.

Integrates with (notified by injection; the engine does NOT call into peers
directly — peers call publish_event / record_event when mutations occur):
    facts.py · concepts.py · skills.py · domains.py · procedures.py ·
    research.py · education.py · validation.py · synthesis.py ·
    evolution.py · knowledge_index.py · integrity.py · semantic_structure.py

Part of the POLARIS Cognitive Substrate:
    ASTRA        → Identity
    ECHO         → Experience
    LUNA         → Knowledge       ← this module
    CHRONOS      → Time
    CONSTELLATION → Relationships
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterator, Optional

from subsystems.luna.exceptions import (
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import LunaEngineLifecycle
from subsystems.luna.models import (
    KnowledgeStatus,
    KnowledgeType,
    _new_id,
    _utcnow,
)

logger = logging.getLogger(__name__)

_ENGINE_VERSION: str = "5.0.0"

# Maximum events kept in memory before the oldest are evicted (ring-buffer cap).
_DEFAULT_MAX_EVENTS: int = 50_000

# Maximum subscribers per event category.
_MAX_SUBSCRIBERS_PER_CATEGORY: int = 64


# ─────────────────────────────────────────────────────────────────────────────
# EVENT CATEGORY ENUM
# ─────────────────────────────────────────────────────────────────────────────

class LunaEventCategory(Enum):
    """Top-level classification of a LUNA subsystem event."""
    KNOWLEDGE_CREATED      = "knowledge_created"
    KNOWLEDGE_UPDATED      = "knowledge_updated"
    KNOWLEDGE_DELETED      = "knowledge_deleted"
    KNOWLEDGE_VALIDATED    = "knowledge_validated"
    KNOWLEDGE_SYNTHESIZED  = "knowledge_synthesized"
    KNOWLEDGE_EVOLVED      = "knowledge_evolved"
    KNOWLEDGE_INDEXED      = "knowledge_indexed"
    KNOWLEDGE_AUDITED      = "knowledge_audited"
    RELATIONSHIP_CREATED   = "relationship_created"
    RELATIONSHIP_REMOVED   = "relationship_removed"
    SKILL_PROGRESSED       = "skill_progressed"
    DOMAIN_UPDATED         = "domain_updated"


# ─────────────────────────────────────────────────────────────────────────────
# EVENT SEVERITY ENUM
# ─────────────────────────────────────────────────────────────────────────────

class LunaEventSeverity(Enum):
    """Severity band for a LUNA event."""
    DEBUG    = "debug"
    INFO     = "info"
    WARNING  = "warning"
    ERROR    = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {
            "debug": 0,
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4,
        }[self.value]

    def is_at_least(self, other: "LunaEventSeverity") -> bool:
        return self.rank >= other.rank


# ─────────────────────────────────────────────────────────────────────────────
# EVENT STATUS ENUM
# ─────────────────────────────────────────────────────────────────────────────

class LunaEventStatus(Enum):
    """Lifecycle status of a stored event."""
    PENDING    = "pending"    # recorded, awaiting subscriber delivery
    DELIVERED  = "delivered"  # all subscribers notified
    FAILED     = "failed"     # one or more subscriber callbacks raised
    REPLAYED   = "replayed"   # replayed from the event log
    ARCHIVED   = "archived"   # retained but excluded from active queries


# ─────────────────────────────────────────────────────────────────────────────
# LUNA EVENT MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LunaEvent:
    """
    A single LUNA subsystem event record.

    Every mutation, validation, synthesis, audit, or structural change that
    crosses an engine boundary in LUNA is captured as a LunaEvent.  Events
    are immutable after creation; status transitions are the only in-place
    mutation permitted.

    Fields:
        id              — UUID4 event identifier.
        category        — LunaEventCategory classifying the event type.
        severity        — LunaEventSeverity band.
        status          — Current delivery status.
        source_engine   — Name of the engine that emitted the event
                          (e.g. "ConceptEngine", "KnowledgeIntegrityEngine").
        knowledge_type  — KnowledgeType of the affected record, if applicable.
        knowledge_id    — ID of the affected knowledge record, if applicable.
        knowledge_name  — Human-readable name of the affected record.
        domain_ids      — Domain IDs associated with the event.
        payload         — Arbitrary structured data describing the change.
        message         — Human-readable event description.
        correlation_id  — Optional ID linking related events (e.g. a batch
                          operation that emits many events).
        occurred_at     — UTC datetime the event was emitted.
        recorded_at     — UTC datetime the event was stored in this engine.
        tags            — Arbitrary string tags for filtering.
        notes           — Free-text supplementary notes.
    """
    id: str
    category: LunaEventCategory
    severity: LunaEventSeverity
    status: LunaEventStatus
    source_engine: str
    knowledge_type: Optional[KnowledgeType]
    knowledge_id: Optional[str]
    knowledge_name: str
    domain_ids: list[str]
    payload: dict[str, Any]
    message: str
    correlation_id: Optional[str]
    occurred_at: datetime
    recorded_at: datetime
    tags: list[str]
    notes: str

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_error(self) -> bool:
        return self.severity.is_at_least(LunaEventSeverity.ERROR)

    @property
    def is_knowledge_lifecycle(self) -> bool:
        return self.category in {
            LunaEventCategory.KNOWLEDGE_CREATED,
            LunaEventCategory.KNOWLEDGE_UPDATED,
            LunaEventCategory.KNOWLEDGE_DELETED,
        }

    @property
    def is_structural(self) -> bool:
        return self.category in {
            LunaEventCategory.RELATIONSHIP_CREATED,
            LunaEventCategory.RELATIONSHIP_REMOVED,
            LunaEventCategory.DOMAIN_UPDATED,
        }

    @property
    def short_id(self) -> str:
        return self.id[:8]

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "source_engine": self.source_engine,
            "knowledge_type": self.knowledge_type.value if self.knowledge_type else None,
            "knowledge_id": self.knowledge_id,
            "knowledge_name": self.knowledge_name,
            "domain_ids": self.domain_ids,
            "payload": self.payload,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "occurred_at": self.occurred_at.isoformat(),
            "recorded_at": self.recorded_at.isoformat(),
            "tags": self.tags,
            "notes": self.notes,
        }

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        category: LunaEventCategory,
        source_engine: str,
        message: str,
        *,
        severity: LunaEventSeverity = LunaEventSeverity.INFO,
        knowledge_type: Optional[KnowledgeType] = None,
        knowledge_id: Optional[str] = None,
        knowledge_name: str = "",
        domain_ids: Optional[list[str]] = None,
        payload: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        tags: Optional[list[str]] = None,
        notes: str = "",
    ) -> "LunaEvent":
        now = _utcnow()
        return cls(
            id=_new_id(),
            category=category,
            severity=severity,
            status=LunaEventStatus.PENDING,
            source_engine=source_engine,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            knowledge_name=knowledge_name,
            domain_ids=list(domain_ids or []),
            payload=dict(payload or {}),
            message=message,
            correlation_id=correlation_id,
            occurred_at=occurred_at or now,
            recorded_at=now,
            tags=list(tags or []),
            notes=notes,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EVENT FILTER MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LunaEventFilter:
    """
    Declarative filter specification for event queries.

    All fields are optional; only non-None fields are applied.  Multiple
    fields are combined with AND logic.

    Fields:
        categories      — restrict to events in these categories.
        severities      — restrict to events at or above these severities.
        source_engines  — restrict to events from these engine names.
        knowledge_types — restrict to events affecting these knowledge types.
        knowledge_ids   — restrict to events for specific record IDs.
        domain_ids      — restrict to events whose domain_ids overlap.
        statuses        — restrict to events in these statuses.
        tags            — restrict to events carrying ALL of these tags.
        correlation_id  — restrict to a single correlation group.
        since           — only events with occurred_at >= since.
        until           — only events with occurred_at <= until.
        message_query   — case-insensitive substring match against message.
        limit           — maximum events to return (default 500).
        offset          — pagination offset (default 0).
    """
    categories: Optional[tuple[LunaEventCategory, ...]] = None
    severities: Optional[tuple[LunaEventSeverity, ...]] = None
    source_engines: Optional[tuple[str, ...]] = None
    knowledge_types: Optional[tuple[KnowledgeType, ...]] = None
    knowledge_ids: Optional[tuple[str, ...]] = None
    domain_ids: Optional[tuple[str, ...]] = None
    statuses: Optional[tuple[LunaEventStatus, ...]] = None
    tags: Optional[tuple[str, ...]] = None
    correlation_id: Optional[str] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    message_query: Optional[str] = None
    limit: int = 500
    offset: int = 0

    def matches(self, event: LunaEvent) -> bool:
        """Return True if the event satisfies all active filter criteria."""
        if self.categories and event.category not in self.categories:
            return False
        if self.severities and event.severity not in self.severities:
            return False
        if self.source_engines and event.source_engine not in self.source_engines:
            return False
        if self.knowledge_types:
            if event.knowledge_type is None or event.knowledge_type not in self.knowledge_types:
                return False
        if self.knowledge_ids and event.knowledge_id not in self.knowledge_ids:
            return False
        if self.domain_ids:
            if not set(self.domain_ids).intersection(event.domain_ids):
                return False
        if self.statuses and event.status not in self.statuses:
            return False
        if self.tags:
            if not all(t in event.tags for t in self.tags):
                return False
        if self.correlation_id and event.correlation_id != self.correlation_id:
            return False
        if self.since and event.occurred_at < self.since:
            return False
        if self.until and event.occurred_at > self.until:
            return False
        if self.message_query:
            if self.message_query.lower() not in event.message.lower():
                return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# EVENT REPLAY RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LunaEventReplayResult:
    """
    Result of a replay_events call.

    Fields:
        events          — events replayed in chronological order.
        total_replayed  — total events matching the replay window.
        since           — replay window start (inclusive).
        until           — replay window end (inclusive).
        replayed_at     — UTC datetime the replay was executed.
    """
    events: tuple["LunaEvent", ...]
    total_replayed: int
    since: Optional[datetime]
    until: Optional[datetime]
    replayed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_replayed": self.total_replayed,
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat() if self.until else None,
            "replayed_at": self.replayed_at.isoformat(),
            "events": [e.to_dict() for e in self.events],
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class LunaEventEngine(LunaEngineLifecycle):
    """
    In-memory, thread-safe implementation of the LUNA Event Engine (v1).

    The engine maintains a capped chronological ring-buffer of LunaEvent
    records and exposes publish, retrieve, search, filter, replay, export,
    audit, and reporting operations over that store.

    Internal state:
        _events         — OrderedDict-equivalent deque of LunaEvent objects,
                          oldest first, capped at _max_events.
        _event_index    — dict[event_id, LunaEvent] for O(1) lookup.
        _category_index — dict[category, list[event_id]] for category scans.
        _knowledge_index— dict[knowledge_id, list[event_id]] for record scans.
        _correlation_index — dict[correlation_id, list[event_id]].
        _subscribers    — dict[category, list[Callable]] for fan-out delivery.

    Subscriber callbacks:
        Callers may register a callback via subscribe(category, fn) to receive
        immediate notification when a matching event is published.  Callback
        exceptions are caught, logged, and recorded in the event's status
        without propagating to the publisher.

    Lifecycle::

        engine = LunaEventEngine(max_events=10_000)
        engine.initialize()

        event = engine.publish_event(
            category=LunaEventCategory.KNOWLEDGE_CREATED,
            source_engine="ConceptEngine",
            message="Concept 'PID Control' created.",
            knowledge_type=KnowledgeType.CONCEPT,
            knowledge_id="concept-pid-001",
            knowledge_name="PID Control",
        )

        results = engine.filter_events(
            LunaEventFilter(categories=(LunaEventCategory.KNOWLEDGE_CREATED,))
        )
        engine.shutdown()
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, *, max_events: int = _DEFAULT_MAX_EVENTS) -> None:
        self._max_events: int = max(1, max_events)

        # Lifecycle
        self._initialized: bool = False
        self._lock: threading.RLock = threading.RLock()
        self._started_at: Optional[datetime] = None

        # Primary store — chronological ring buffer (oldest first)
        self._event_ring: deque[LunaEvent] = deque()
        self._event_index: dict[str, LunaEvent] = {}

        # Secondary indexes
        self._category_index: dict[LunaEventCategory, list[str]] = defaultdict(list)
        self._knowledge_index: dict[str, list[str]] = defaultdict(list)
        self._engine_index: dict[str, list[str]] = defaultdict(list)
        self._correlation_index: dict[str, list[str]] = defaultdict(list)
        self._domain_index: dict[str, list[str]] = defaultdict(list)
        self._tag_index: dict[str, list[str]] = defaultdict(list)

        # Subscriber registry
        self._subscribers: dict[LunaEventCategory, list[Callable[[LunaEvent], None]]] = defaultdict(list)

        # Observability counters
        self._published_count: int = 0
        self._delivered_count: int = 0
        self._failed_deliveries: int = 0
        self._evicted_count: int = 0
        self._replay_count: int = 0
        self._export_count: int = 0
        self._audit_count: int = 0
        self._last_event_at: Optional[datetime] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Prepare the engine for operation.

        Idempotent — calling on an already-initialized engine is a no-op.

        Raises:
            LunaLifecycleError: Internal initialization failure.
        """
        with self._lock:
            if self._initialized:
                return
            try:
                self._event_ring.clear()
                self._event_index.clear()
                self._category_index.clear()
                self._knowledge_index.clear()
                self._engine_index.clear()
                self._correlation_index.clear()
                self._domain_index.clear()
                self._tag_index.clear()
                self._published_count = 0
                self._delivered_count = 0
                self._failed_deliveries = 0
                self._evicted_count = 0
                self._replay_count = 0
                self._export_count = 0
                self._audit_count = 0
                self._last_event_at = None
                self._started_at = _utcnow()
                self._initialized = True
                logger.info(
                    "LunaEventEngine initialized (version=%s, max_events=%d)",
                    _ENGINE_VERSION,
                    self._max_events,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="LunaEventEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        """
        Release resources and put the engine into a quiescent state.

        Idempotent — calling on an already-stopped engine is a no-op.

        Raises:
            LunaLifecycleError: Internal teardown failure.
        """
        with self._lock:
            if not self._initialized:
                return
            try:
                self._initialized = False
                logger.info(
                    "LunaEventEngine shut down (published=%d, evicted=%d)",
                    self._published_count,
                    self._evicted_count,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="LunaEventEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def is_initialized(self) -> bool:
        """Return True if the engine is fully started and ready."""
        return self._initialized

    # ── Internal guard ────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation=operation)

    # ── Internal index maintenance ────────────────────────────────────────────

    def _index_event(self, event: LunaEvent) -> None:
        """Register a new event in all secondary indexes."""
        eid = event.id
        self._category_index[event.category].append(eid)
        if event.knowledge_id:
            self._knowledge_index[event.knowledge_id].append(eid)
        self._engine_index[event.source_engine].append(eid)
        if event.correlation_id:
            self._correlation_index[event.correlation_id].append(eid)
        for domain_id in event.domain_ids:
            self._domain_index[domain_id].append(eid)
        for tag in event.tags:
            self._tag_index[tag].append(eid)

    def _deindex_event(self, event: LunaEvent) -> None:
        """Remove a evicted event from all secondary indexes."""
        eid = event.id
        cat_list = self._category_index.get(event.category)
        if cat_list and eid in cat_list:
            cat_list.remove(eid)
        if event.knowledge_id:
            kid_list = self._knowledge_index.get(event.knowledge_id)
            if kid_list and eid in kid_list:
                kid_list.remove(eid)
        eng_list = self._engine_index.get(event.source_engine)
        if eng_list and eid in eng_list:
            eng_list.remove(eid)
        if event.correlation_id:
            corr_list = self._correlation_index.get(event.correlation_id)
            if corr_list and eid in corr_list:
                corr_list.remove(eid)
        for domain_id in event.domain_ids:
            d_list = self._domain_index.get(domain_id)
            if d_list and eid in d_list:
                d_list.remove(eid)
        for tag in event.tags:
            t_list = self._tag_index.get(tag)
            if t_list and eid in t_list:
                t_list.remove(eid)

    def _evict_oldest(self) -> None:
        """Evict the oldest event when the ring buffer is at capacity."""
        if not self._event_ring:
            return
        oldest = self._event_ring.popleft()
        self._event_index.pop(oldest.id, None)
        self._deindex_event(oldest)
        self._evicted_count += 1
        logger.debug("LunaEvent evicted: id=%s category=%s", oldest.id, oldest.category.value)

    def _deliver_to_subscribers(self, event: LunaEvent) -> None:
        """
        Fan out a newly published event to all registered subscribers for its
        category.  Each callback failure is caught, logged, and counted
        individually without blocking delivery to subsequent subscribers.
        """
        callbacks = self._subscribers.get(event.category, [])
        delivery_failed = False
        for callback in callbacks:
            try:
                callback(event)
            except Exception as exc:
                self._failed_deliveries += 1
                delivery_failed = True
                logger.warning(
                    "LunaEvent subscriber callback raised for event %s: %s",
                    event.id,
                    exc,
                )
        event.status = (
            LunaEventStatus.FAILED if delivery_failed else LunaEventStatus.DELIVERED
        )
        if callbacks:
            self._delivered_count += 1

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLISH / RECORD
    # ─────────────────────────────────────────────────────────────────────────

    def publish_event(
        self,
        category: LunaEventCategory,
        source_engine: str,
        message: str,
        *,
        severity: LunaEventSeverity = LunaEventSeverity.INFO,
        knowledge_type: Optional[KnowledgeType] = None,
        knowledge_id: Optional[str] = None,
        knowledge_name: str = "",
        domain_ids: Optional[list[str]] = None,
        payload: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        tags: Optional[list[str]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Validate, store, and fan out a new LUNA event.

        This is the primary write entry point.  After storage the event is
        immediately delivered to all registered subscribers for its category.

        Args:
            category:       LunaEventCategory classification.
            source_engine:  Name of the engine emitting the event
                            (e.g. "ConceptEngine", "KnowledgeIntegrityEngine").
            message:        Human-readable description of what occurred.
            severity:       Severity band (default: INFO).
            knowledge_type: KnowledgeType of the affected record.
            knowledge_id:   ID of the affected record, if applicable.
            knowledge_name: Display name of the affected record.
            domain_ids:     Domain IDs associated with the event.
            payload:        Arbitrary structured context dict.
            correlation_id: Optional ID linking this event to related events.
            occurred_at:    UTC datetime the event logically occurred
                            (defaults to now).
            tags:           String tags for downstream filtering.
            notes:          Free-text notes.

        Returns:
            The stored and (if subscribers exist) delivered LunaEvent.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ValueError:              message or source_engine is empty.
        """
        self._require_initialized("publish_event")
        if not message:
            raise ValueError("LunaEvent message must not be empty.")
        if not source_engine:
            raise ValueError("LunaEvent source_engine must not be empty.")

        with self._lock:
            event = LunaEvent.create(
                category=category,
                source_engine=source_engine,
                message=message,
                severity=severity,
                knowledge_type=knowledge_type,
                knowledge_id=knowledge_id,
                knowledge_name=knowledge_name,
                domain_ids=domain_ids,
                payload=payload,
                correlation_id=correlation_id,
                occurred_at=occurred_at,
                tags=tags,
                notes=notes,
            )

            # Evict oldest if at capacity
            while len(self._event_ring) >= self._max_events:
                self._evict_oldest()

            # Store in ring and index
            self._event_ring.append(event)
            self._event_index[event.id] = event
            self._index_event(event)

            # Counters
            self._published_count += 1
            self._last_event_at = event.recorded_at

            # Fan out to subscribers (still inside lock for index consistency;
            # callbacks must not call back into the engine to avoid deadlock)
            self._deliver_to_subscribers(event)

            logger.debug(
                "LunaEvent published: id=%s category=%s source=%s",
                event.id,
                category.value,
                source_engine,
            )
            return event

    def record_event(
        self,
        category: LunaEventCategory,
        source_engine: str,
        message: str,
        *,
        severity: LunaEventSeverity = LunaEventSeverity.INFO,
        knowledge_type: Optional[KnowledgeType] = None,
        knowledge_id: Optional[str] = None,
        knowledge_name: str = "",
        domain_ids: Optional[list[str]] = None,
        payload: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        tags: Optional[list[str]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Lower-level programmatic alias for publish_event.

        Identical behaviour — provided so engine internals can call
        ``event_engine.record_event(...)`` without semantic ambiguity about
        whether the event is "public" or "internal".

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ValueError:              message or source_engine is empty.
        """
        return self.publish_event(
            category=category,
            source_engine=source_engine,
            message=message,
            severity=severity,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            knowledge_name=knowledge_name,
            domain_ids=domain_ids,
            payload=payload,
            correlation_id=correlation_id,
            occurred_at=occurred_at,
            tags=tags,
            notes=notes,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # RETRIEVE
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_event(self, event_id: str) -> LunaEvent:
        """
        Fetch a single stored event by ID.

        Args:
            event_id: The UUID of the event to retrieve.

        Returns:
            The matching LunaEvent.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KeyError:                No event with the given ID exists.
        """
        self._require_initialized("retrieve_event")
        with self._lock:
            event = self._event_index.get(event_id)
            if event is None:
                raise KeyError(f"LunaEvent '{event_id}' not found.")
            return event

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    def search_events(
        self,
        query: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LunaEvent]:
        """
        Full-text search over stored events.

        Matches against the event message, knowledge_name, source_engine,
        and tag values using case-insensitive substring matching.

        Results are returned in reverse-chronological order (most recent first).

        Args:
            query:   Search string (non-empty).
            limit:   Maximum events to return.
            offset:  Pagination offset.

        Returns:
            Matching LunaEvent records.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("search_events")
        with self._lock:
            if not query:
                events = list(reversed(self._event_ring))
                return events[offset: offset + limit]

            q = query.lower().strip()
            results: list[LunaEvent] = []
            for event in reversed(self._event_ring):
                if (
                    q in event.message.lower()
                    or q in event.knowledge_name.lower()
                    or q in event.source_engine.lower()
                    or any(q in tag.lower() for tag in event.tags)
                    or (event.knowledge_id and q in event.knowledge_id.lower())
                ):
                    results.append(event)

            return results[offset: offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # FILTER
    # ─────────────────────────────────────────────────────────────────────────

    def filter_events(self, event_filter: LunaEventFilter) -> list[LunaEvent]:
        """
        Apply a structured LunaEventFilter to the stored events.

        Results are returned in reverse-chronological order (most recent first).

        Args:
            event_filter: Declarative filter specification.

        Returns:
            Matching LunaEvent records respecting filter.limit and filter.offset.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("filter_events")
        with self._lock:
            matched: list[LunaEvent] = []
            for event in reversed(self._event_ring):
                if event_filter.matches(event):
                    matched.append(event)

            return matched[event_filter.offset: event_filter.offset + event_filter.limit]

    def filter_by_category(
        self,
        category: LunaEventCategory,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LunaEvent]:
        """
        Return all events matching a single category.

        Results are returned in reverse-chronological order.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("filter_by_category")
        with self._lock:
            event_ids = self._category_index.get(category, [])
            events = [
                self._event_index[eid]
                for eid in reversed(event_ids)
                if eid in self._event_index
            ]
            return events[offset: offset + limit]

    def filter_by_knowledge_id(
        self,
        knowledge_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LunaEvent]:
        """
        Return all events related to a specific knowledge record.

        Results are returned in reverse-chronological order.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("filter_by_knowledge_id")
        with self._lock:
            event_ids = self._knowledge_index.get(knowledge_id, [])
            events = [
                self._event_index[eid]
                for eid in reversed(event_ids)
                if eid in self._event_index
            ]
            return events[offset: offset + limit]

    def filter_by_source_engine(
        self,
        source_engine: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LunaEvent]:
        """
        Return all events emitted by a specific engine.

        Results are returned in reverse-chronological order.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("filter_by_source_engine")
        with self._lock:
            event_ids = self._engine_index.get(source_engine, [])
            events = [
                self._event_index[eid]
                for eid in reversed(event_ids)
                if eid in self._event_index
            ]
            return events[offset: offset + limit]

    def filter_by_severity(
        self,
        min_severity: LunaEventSeverity,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LunaEvent]:
        """
        Return all events at or above a minimum severity.

        Results are returned in reverse-chronological order.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("filter_by_severity")
        with self._lock:
            events = [
                e for e in reversed(self._event_ring)
                if e.severity.is_at_least(min_severity)
            ]
            return events[offset: offset + limit]

    def filter_by_time_range(
        self,
        since: Optional[datetime],
        until: Optional[datetime],
        *,
        limit: int = 500,
        offset: int = 0,
    ) -> list[LunaEvent]:
        """
        Return all events whose occurred_at falls within [since, until].

        Both bounds are inclusive.  None means unbounded on that side.

        Results are returned in chronological order.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("filter_by_time_range")
        with self._lock:
            events: list[LunaEvent] = []
            for event in self._event_ring:
                if since and event.occurred_at < since:
                    continue
                if until and event.occurred_at > until:
                    continue
                events.append(event)
            return events[offset: offset + limit]

    def filter_by_correlation_id(
        self,
        correlation_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[LunaEvent]:
        """
        Return all events sharing a correlation_id, in chronological order.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("filter_by_correlation_id")
        with self._lock:
            event_ids = self._correlation_index.get(correlation_id, [])
            events = [
                self._event_index[eid]
                for eid in event_ids
                if eid in self._event_index
            ]
            return events[offset: offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # REPLAY
    # ─────────────────────────────────────────────────────────────────────────

    def replay_events(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        categories: Optional[list[LunaEventCategory]] = None,
        limit: int = 5_000,
    ) -> LunaEventReplayResult:
        """
        Replay stored events in strict chronological order for a given window.

        Each replayed event has its status updated to REPLAYED.  The operation
        is non-destructive — events are not removed from the store.

        Args:
            since:      Inclusive lower bound on occurred_at.
            until:      Inclusive upper bound on occurred_at.
            categories: Restrict replay to specific event categories.
                        None means all categories.
            limit:      Maximum events to include in the replay result.

        Returns:
            LunaEventReplayResult containing the replayed events.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("replay_events")
        with self._lock:
            self._replay_count += 1
            replayed: list[LunaEvent] = []

            for event in self._event_ring:
                if since and event.occurred_at < since:
                    continue
                if until and event.occurred_at > until:
                    continue
                if categories and event.category not in categories:
                    continue
                event.status = LunaEventStatus.REPLAYED
                replayed.append(event)
                if len(replayed) >= limit:
                    break

            logger.info(
                "LunaEvent replay: window=[%s, %s] replayed=%d",
                since.isoformat() if since else "unbounded",
                until.isoformat() if until else "unbounded",
                len(replayed),
            )
            return LunaEventReplayResult(
                events=tuple(replayed),
                total_replayed=len(replayed),
                since=since,
                until=until,
                replayed_at=_utcnow(),
            )

    def iter_events(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        category: Optional[LunaEventCategory] = None,
    ) -> Iterator[LunaEvent]:
        """
        Yield events in chronological order without materialising the entire
        result set.

        Designed for consumers that process events in a streaming fashion.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("iter_events")
        with self._lock:
            snapshot = list(self._event_ring)

        for event in snapshot:
            if since and event.occurred_at < since:
                continue
            if until and event.occurred_at > until:
                continue
            if category and event.category != category:
                continue
            yield event

    # ─────────────────────────────────────────────────────────────────────────
    # CLEAR
    # ─────────────────────────────────────────────────────────────────────────

    def clear_events(
        self,
        *,
        category: Optional[LunaEventCategory] = None,
        before: Optional[datetime] = None,
    ) -> int:
        """
        Purge events from the store.

        When called without arguments, clears all events.  When category or
        before are provided, only matching events are removed.

        Args:
            category: Restrict removal to a specific category.
            before:   Remove only events whose occurred_at is strictly before
                      this datetime.

        Returns:
            The number of events removed.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("clear_events")
        with self._lock:
            if category is None and before is None:
                count = len(self._event_ring)
                self._event_ring.clear()
                self._event_index.clear()
                self._category_index.clear()
                self._knowledge_index.clear()
                self._engine_index.clear()
                self._correlation_index.clear()
                self._domain_index.clear()
                self._tag_index.clear()
                self._evicted_count += count
                logger.info("LunaEventEngine cleared: %d events removed.", count)
                return count

            # Selective removal
            to_remove = [
                event for event in self._event_ring
                if (category is None or event.category == category)
                and (before is None or event.occurred_at < before)
            ]
            for event in to_remove:
                self._event_ring.remove(event)
                self._event_index.pop(event.id, None)
                self._deindex_event(event)
                self._evicted_count += 1

            logger.info(
                "LunaEventEngine selective clear: %d events removed (category=%s, before=%s).",
                len(to_remove),
                category.value if category else "all",
                before.isoformat() if before else "none",
            )
            return len(to_remove)

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def export_events(
        self,
        event_filter: Optional[LunaEventFilter] = None,
        *,
        include_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Serialise all matching events to a list of plain dictionaries.

        Suitable for JSON serialisation, external logging systems, or audit
        pipeline ingestion.

        Args:
            event_filter:    Optional filter restricting which events are
                             exported.  None exports all stored events.
            include_payload: If False, the payload field is omitted from each
                             dict (useful for lightweight exports).

        Returns:
            List of event dicts, oldest first.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("export_events")
        with self._lock:
            self._export_count += 1
            if event_filter is None:
                events = list(self._event_ring)
            else:
                events = [e for e in self._event_ring if event_filter.matches(e)]

            result: list[dict[str, Any]] = []
            for event in events:
                d = event.to_dict()
                if not include_payload:
                    d.pop("payload", None)
                result.append(d)

            logger.debug("LunaEvent export: %d events exported.", len(result))
            return result

    # ─────────────────────────────────────────────────────────────────────────
    # SUBSCRIBER REGISTRY
    # ─────────────────────────────────────────────────────────────────────────

    def subscribe(
        self,
        category: LunaEventCategory,
        callback: Callable[[LunaEvent], None],
    ) -> None:
        """
        Register a callback to receive events of a specific category.

        The callback is invoked synchronously inside publish_event's lock.
        Callbacks MUST NOT call back into the event engine to avoid deadlock.
        Exceptions raised by the callback are caught and logged.

        Args:
            category: The event category to subscribe to.
            callback: A callable accepting a single LunaEvent argument.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            RuntimeError:            Subscriber limit exceeded for the category.
        """
        self._require_initialized("subscribe")
        with self._lock:
            existing = self._subscribers[category]
            if len(existing) >= _MAX_SUBSCRIBERS_PER_CATEGORY:
                raise RuntimeError(
                    f"Subscriber limit ({_MAX_SUBSCRIBERS_PER_CATEGORY}) "
                    f"reached for category '{category.value}'."
                )
            existing.append(callback)
            logger.debug(
                "Subscriber registered for category '%s' (total=%d).",
                category.value,
                len(existing),
            )

    def unsubscribe(
        self,
        category: LunaEventCategory,
        callback: Callable[[LunaEvent], None],
    ) -> bool:
        """
        Remove a previously registered subscriber callback.

        Args:
            category: The event category.
            callback: The exact callable that was previously registered.

        Returns:
            True if the callback was found and removed; False otherwise.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("unsubscribe")
        with self._lock:
            existing = self._subscribers.get(category, [])
            if callback in existing:
                existing.remove(callback)
                return True
            return False

    def subscriber_count(self, category: LunaEventCategory) -> int:
        """Return the number of subscribers registered for a category."""
        with self._lock:
            return len(self._subscribers.get(category, []))

    # ─────────────────────────────────────────────────────────────────────────
    # DOMAIN-SPECIFIC CONVENIENCE PUBLISHERS
    # ─────────────────────────────────────────────────────────────────────────

    def emit_knowledge_created(
        self,
        source_engine: str,
        knowledge_id: str,
        knowledge_name: str,
        knowledge_type: KnowledgeType,
        domain_ids: Optional[list[str]] = None,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a KnowledgeCreated event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.publish_event(
            category=LunaEventCategory.KNOWLEDGE_CREATED,
            source_engine=source_engine,
            message=f"Knowledge record '{knowledge_name}' created [{knowledge_type.value}].",
            severity=LunaEventSeverity.INFO,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            knowledge_name=knowledge_name,
            domain_ids=domain_ids,
            correlation_id=correlation_id,
            payload=payload or {
                "knowledge_id": knowledge_id,
                "knowledge_type": knowledge_type.value,
            },
            notes=notes,
        )

    def emit_knowledge_updated(
        self,
        source_engine: str,
        knowledge_id: str,
        knowledge_name: str,
        knowledge_type: KnowledgeType,
        domain_ids: Optional[list[str]] = None,
        *,
        changed_fields: Optional[list[str]] = None,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a KnowledgeUpdated event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.publish_event(
            category=LunaEventCategory.KNOWLEDGE_UPDATED,
            source_engine=source_engine,
            message=f"Knowledge record '{knowledge_name}' updated [{knowledge_type.value}].",
            severity=LunaEventSeverity.INFO,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            knowledge_name=knowledge_name,
            domain_ids=domain_ids,
            correlation_id=correlation_id,
            payload=payload or {
                "knowledge_id": knowledge_id,
                "knowledge_type": knowledge_type.value,
                "changed_fields": changed_fields or [],
            },
            notes=notes,
        )

    def emit_knowledge_deleted(
        self,
        source_engine: str,
        knowledge_id: str,
        knowledge_name: str,
        knowledge_type: KnowledgeType,
        new_status: KnowledgeStatus,
        domain_ids: Optional[list[str]] = None,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a KnowledgeDeleted event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.publish_event(
            category=LunaEventCategory.KNOWLEDGE_DELETED,
            source_engine=source_engine,
            message=(
                f"Knowledge record '{knowledge_name}' deleted → status '{new_status.value}' "
                f"[{knowledge_type.value}]."
            ),
            severity=LunaEventSeverity.WARNING,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            knowledge_name=knowledge_name,
            domain_ids=domain_ids,
            correlation_id=correlation_id,
            payload=payload or {
                "knowledge_id": knowledge_id,
                "knowledge_type": knowledge_type.value,
                "new_status": new_status.value,
            },
            notes=notes,
        )

    def emit_knowledge_validated(
        self,
        source_engine: str,
        knowledge_id: str,
        knowledge_name: str,
        knowledge_type: KnowledgeType,
        passed: bool,
        issue_count: int,
        domain_ids: Optional[list[str]] = None,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a KnowledgeValidated event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        severity = LunaEventSeverity.INFO if passed else LunaEventSeverity.WARNING
        return self.publish_event(
            category=LunaEventCategory.KNOWLEDGE_VALIDATED,
            source_engine=source_engine,
            message=(
                f"Validation {'passed' if passed else 'failed'} for '{knowledge_name}' "
                f"[{knowledge_type.value}] — {issue_count} issue(s)."
            ),
            severity=severity,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            knowledge_name=knowledge_name,
            domain_ids=domain_ids,
            correlation_id=correlation_id,
            payload=payload or {
                "knowledge_id": knowledge_id,
                "knowledge_type": knowledge_type.value,
                "passed": passed,
                "issue_count": issue_count,
            },
            notes=notes,
        )

    def emit_knowledge_synthesized(
        self,
        source_engine: str,
        synthesis_id: str,
        synthesis_name: str,
        source_domain_ids: list[str],
        source_count: int,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a KnowledgeSynthesized event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.publish_event(
            category=LunaEventCategory.KNOWLEDGE_SYNTHESIZED,
            source_engine=source_engine,
            message=(
                f"Knowledge synthesis '{synthesis_name}' completed from "
                f"{source_count} source records across {len(source_domain_ids)} domain(s)."
            ),
            severity=LunaEventSeverity.INFO,
            knowledge_type=None,
            knowledge_id=synthesis_id,
            knowledge_name=synthesis_name,
            domain_ids=source_domain_ids,
            correlation_id=correlation_id,
            payload=payload or {
                "synthesis_id": synthesis_id,
                "source_domain_ids": source_domain_ids,
                "source_count": source_count,
            },
            notes=notes,
        )

    def emit_knowledge_evolved(
        self,
        source_engine: str,
        knowledge_id: str,
        knowledge_name: str,
        knowledge_type: KnowledgeType,
        old_status: KnowledgeStatus,
        new_status: KnowledgeStatus,
        domain_ids: Optional[list[str]] = None,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a KnowledgeEvolved event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.publish_event(
            category=LunaEventCategory.KNOWLEDGE_EVOLVED,
            source_engine=source_engine,
            message=(
                f"Knowledge record '{knowledge_name}' evolved: "
                f"'{old_status.value}' → '{new_status.value}' [{knowledge_type.value}]."
            ),
            severity=LunaEventSeverity.INFO,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            knowledge_name=knowledge_name,
            domain_ids=domain_ids,
            correlation_id=correlation_id,
            payload=payload or {
                "knowledge_id": knowledge_id,
                "knowledge_type": knowledge_type.value,
                "old_status": old_status.value,
                "new_status": new_status.value,
            },
            notes=notes,
        )

    def emit_knowledge_indexed(
        self,
        source_engine: str,
        records_indexed: int,
        rebuild: bool = False,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a KnowledgeIndexed event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        op = "Full rebuild" if rebuild else "Incremental update"
        return self.publish_event(
            category=LunaEventCategory.KNOWLEDGE_INDEXED,
            source_engine=source_engine,
            message=f"{op}: {records_indexed} record(s) indexed.",
            severity=LunaEventSeverity.INFO,
            knowledge_type=None,
            knowledge_id=None,
            knowledge_name="",
            correlation_id=correlation_id,
            payload=payload or {
                "records_indexed": records_indexed,
                "rebuild": rebuild,
            },
            notes=notes,
        )

    def emit_knowledge_audited(
        self,
        source_engine: str,
        audit_id: str,
        records_scanned: int,
        issue_count: int,
        passed: bool,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a KnowledgeAudited event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        severity = LunaEventSeverity.INFO if passed else LunaEventSeverity.WARNING
        return self.publish_event(
            category=LunaEventCategory.KNOWLEDGE_AUDITED,
            source_engine=source_engine,
            message=(
                f"Integrity audit {'passed' if passed else 'failed'}: "
                f"{records_scanned} records scanned, {issue_count} issue(s) found."
            ),
            severity=severity,
            knowledge_type=None,
            knowledge_id=audit_id,
            knowledge_name=f"Audit {audit_id[:8]}",
            correlation_id=correlation_id,
            payload=payload or {
                "audit_id": audit_id,
                "records_scanned": records_scanned,
                "issue_count": issue_count,
                "passed": passed,
            },
            notes=notes,
        )

    def emit_relationship_created(
        self,
        source_engine: str,
        relationship_id: str,
        source_concept_id: str,
        target_concept_id: str,
        relationship_type: str,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a RelationshipCreated event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.publish_event(
            category=LunaEventCategory.RELATIONSHIP_CREATED,
            source_engine=source_engine,
            message=(
                f"ConceptRelationship '{relationship_type}' created: "
                f"'{source_concept_id}' → '{target_concept_id}'."
            ),
            severity=LunaEventSeverity.INFO,
            knowledge_type=KnowledgeType.CONCEPT,
            knowledge_id=relationship_id,
            knowledge_name=f"{source_concept_id} → {target_concept_id}",
            correlation_id=correlation_id,
            payload=payload or {
                "relationship_id": relationship_id,
                "source_concept_id": source_concept_id,
                "target_concept_id": target_concept_id,
                "relationship_type": relationship_type,
            },
            notes=notes,
        )

    def emit_relationship_removed(
        self,
        source_engine: str,
        relationship_id: str,
        source_concept_id: str,
        target_concept_id: str,
        relationship_type: str,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a RelationshipRemoved event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.publish_event(
            category=LunaEventCategory.RELATIONSHIP_REMOVED,
            source_engine=source_engine,
            message=(
                f"ConceptRelationship '{relationship_type}' removed: "
                f"'{source_concept_id}' → '{target_concept_id}'."
            ),
            severity=LunaEventSeverity.WARNING,
            knowledge_type=KnowledgeType.CONCEPT,
            knowledge_id=relationship_id,
            knowledge_name=f"{source_concept_id} → {target_concept_id}",
            correlation_id=correlation_id,
            payload=payload or {
                "relationship_id": relationship_id,
                "source_concept_id": source_concept_id,
                "target_concept_id": target_concept_id,
                "relationship_type": relationship_type,
            },
            notes=notes,
        )

    def emit_skill_progressed(
        self,
        source_engine: str,
        skill_id: str,
        skill_name: str,
        from_level: str,
        to_level: str,
        domain_ids: Optional[list[str]] = None,
        *,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a SkillProgressed event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.publish_event(
            category=LunaEventCategory.SKILL_PROGRESSED,
            source_engine=source_engine,
            message=(
                f"Skill '{skill_name}' progression model advanced: "
                f"'{from_level}' → '{to_level}'."
            ),
            severity=LunaEventSeverity.INFO,
            knowledge_type=KnowledgeType.SKILL,
            knowledge_id=skill_id,
            knowledge_name=skill_name,
            domain_ids=domain_ids,
            correlation_id=correlation_id,
            payload=payload or {
                "skill_id": skill_id,
                "from_level": from_level,
                "to_level": to_level,
            },
            notes=notes,
        )

    def emit_domain_updated(
        self,
        source_engine: str,
        domain_id: str,
        domain_name: str,
        *,
        changed_fields: Optional[list[str]] = None,
        correlation_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        notes: str = "",
    ) -> LunaEvent:
        """
        Publish a DomainUpdated event.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.publish_event(
            category=LunaEventCategory.DOMAIN_UPDATED,
            source_engine=source_engine,
            message=f"KnowledgeDomain '{domain_name}' updated.",
            severity=LunaEventSeverity.INFO,
            knowledge_type=KnowledgeType.DOMAIN,
            knowledge_id=domain_id,
            knowledge_name=domain_name,
            domain_ids=[domain_id],
            correlation_id=correlation_id,
            payload=payload or {
                "domain_id": domain_id,
                "changed_fields": changed_fields or [],
            },
            notes=notes,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT
    # ─────────────────────────────────────────────────────────────────────────

    def event_audit(self) -> dict[str, Any]:
        """
        Perform a structural integrity check of the event store.

        Checks:
            - All events in _event_index are also in _event_ring.
            - All events in _event_ring are also in _event_index.
            - All category_index entries resolve to known event IDs.
            - No events have empty id, category, source_engine, or message.
            - Ring buffer size does not exceed max_events.

        Returns::

            {
                "passed": bool,
                "issues": [{"severity": str, "message": str}],
                "stats": {...},
                "audited_at": str,
            }

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("event_audit")
        with self._lock:
            self._audit_count += 1
            issues: list[dict[str, Any]] = []

            ring_ids: set[str] = {e.id for e in self._event_ring}
            index_ids: set[str] = set(self._event_index.keys())

            # Ring vs index consistency
            for eid in ring_ids - index_ids:
                issues.append({
                    "severity": "ERROR",
                    "message": f"Event '{eid}' in ring buffer but missing from index.",
                })
            for eid in index_ids - ring_ids:
                issues.append({
                    "severity": "ERROR",
                    "message": f"Event '{eid}' in index but missing from ring buffer.",
                })

            # Category index consistency
            for category, event_ids in self._category_index.items():
                for eid in event_ids:
                    if eid not in self._event_index:
                        issues.append({
                            "severity": "WARNING",
                            "message": (
                                f"Category index '{category.value}' references "
                                f"unknown event '{eid}'."
                            ),
                        })

            # Event field integrity
            for event in self._event_ring:
                if not event.id:
                    issues.append({"severity": "ERROR", "message": "Event with empty id found."})
                if not event.source_engine:
                    issues.append({
                        "severity": "ERROR",
                        "message": f"Event '{event.id}' has empty source_engine.",
                    })
                if not event.message:
                    issues.append({
                        "severity": "ERROR",
                        "message": f"Event '{event.id}' has empty message.",
                    })

            # Ring buffer cap check
            if len(self._event_ring) > self._max_events:
                issues.append({
                    "severity": "ERROR",
                    "message": (
                        f"Ring buffer size {len(self._event_ring)} "
                        f"exceeds max_events {self._max_events}."
                    ),
                })

            return {
                "passed": all(i["severity"] != "ERROR" for i in issues),
                "issues": issues,
                "stats": {
                    "total_events": len(self._event_ring),
                    "index_entries": len(self._event_index),
                    "category_index_size": sum(len(v) for v in self._category_index.values()),
                    "knowledge_index_size": sum(len(v) for v in self._knowledge_index.values()),
                    "correlation_groups": len(self._correlation_index),
                },
                "audited_at": _utcnow().isoformat(),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────────────────────

    def event_reporting(self) -> dict[str, Any]:
        """
        Return a comprehensive human-readable statistics snapshot.

        Returns::

            {
                "engine": "LunaEventEngine",
                "version": str,
                "initialized": bool,
                "started_at": str | None,
                "max_events": int,
                "total_events": int,
                "published_count": int,
                "delivered_count": int,
                "failed_deliveries": int,
                "evicted_count": int,
                "replay_count": int,
                "export_count": int,
                "audit_count": int,
                "last_event_at": str | None,
                "category_distribution": {category_value: count},
                "severity_distribution": {severity_value: count},
                "source_engine_distribution": {engine: count},
                "status_distribution": {status_value: count},
                "subscriber_counts": {category_value: count},
                "oldest_event_at": str | None,
                "newest_event_at": str | None,
                "generated_at": str,
            }

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("event_reporting")
        with self._lock:
            category_dist: dict[str, int] = defaultdict(int)
            severity_dist: dict[str, int] = defaultdict(int)
            status_dist: dict[str, int] = defaultdict(int)

            for event in self._event_ring:
                category_dist[event.category.value] += 1
                severity_dist[event.severity.value] += 1
                status_dist[event.status.value] += 1

            engine_dist = {
                engine: len(ids)
                for engine, ids in self._engine_index.items()
                if ids
            }

            subscriber_counts = {
                cat.value: len(callbacks)
                for cat, callbacks in self._subscribers.items()
                if callbacks
            }

            oldest_at = self._event_ring[0].occurred_at.isoformat() if self._event_ring else None
            newest_at = self._event_ring[-1].occurred_at.isoformat() if self._event_ring else None

            return {
                "engine": "LunaEventEngine",
                "version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "max_events": self._max_events,
                "total_events": len(self._event_ring),
                "published_count": self._published_count,
                "delivered_count": self._delivered_count,
                "failed_deliveries": self._failed_deliveries,
                "evicted_count": self._evicted_count,
                "replay_count": self._replay_count,
                "export_count": self._export_count,
                "audit_count": self._audit_count,
                "last_event_at": (
                    self._last_event_at.isoformat() if self._last_event_at else None
                ),
                "category_distribution": dict(category_dist),
                "severity_distribution": dict(severity_dist),
                "source_engine_distribution": engine_dist,
                "status_distribution": dict(status_dist),
                "subscriber_counts": subscriber_counts,
                "oldest_event_at": oldest_at,
                "newest_event_at": newest_at,
                "generated_at": _utcnow().isoformat(),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY  (LunaEngineLifecycle contract)
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """
        Return a lightweight liveness/readiness summary.

        Required keys: engine, initialized, record_count, status.
        """
        with self._lock:
            return {
                "engine": "LunaEventEngine",
                "initialized": self._initialized,
                "record_count": len(self._event_ring),
                "status": "healthy" if self._initialized else "offline",
            }

    def diagnostics_report(self) -> dict[str, Any]:
        """
        Return a full introspection snapshot for operator debugging.

        Required keys: engine, initialized, record_count, status,
        index_size, duplicate_checks, mutation_count, last_mutation_at.
        """
        with self._lock:
            return {
                "engine": "LunaEventEngine",
                "version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "status": "healthy" if self._initialized else "offline",
                "record_count": len(self._event_ring),
                "index_size": len(self._event_index),
                "duplicate_checks": 0,
                "mutation_count": self._published_count,
                "last_mutation_at": (
                    self._last_event_at.isoformat() if self._last_event_at else None
                ),
                "max_events": self._max_events,
                "published_count": self._published_count,
                "delivered_count": self._delivered_count,
                "failed_deliveries": self._failed_deliveries,
                "evicted_count": self._evicted_count,
                "replay_count": self._replay_count,
                "export_count": self._export_count,
                "audit_count": self._audit_count,
                "category_index_entries": sum(
                    len(v) for v in self._category_index.values()
                ),
                "knowledge_index_entries": sum(
                    len(v) for v in self._knowledge_index.values()
                ),
                "correlation_groups": len(self._correlation_index),
                "registered_subscriber_categories": sum(
                    1 for v in self._subscribers.values() if v
                ),
                "started_at": self._started_at.isoformat() if self._started_at else None,
            }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "LunaEventCategory",
    "LunaEventSeverity",
    "LunaEventStatus",
    "LunaEvent",
    "LunaEventFilter",
    "LunaEventReplayResult",
    "LunaEventEngine",
]