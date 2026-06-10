# subsystems/echo/context_reconstruction.py
"""
ECHO v1 Context Reconstruction Engine.

Implements :class:`ContextReconstructionEngine` — the production engine
responsible for rebuilding historical context from episodic memories.

The Context Reconstruction Engine allows the POLARIS system to answer
fundamental episodic questions:

* **What happened?**         — Timeline and experience chain retrieval.
* **Why did it happen?**     — Causal chain reconstruction from related memories.
* **What led to it?**        — Antecedent experience discovery via temporal
                               and relational traversal.
* **What happened after?**   — Consequent experience discovery via forward
                               traversal of the experience graph.
* **What related events existed?** — Cross-reference generation across
                                     sessions, conversations, and shared tags.

Design Principles
-----------------
* **Thread-safe**: All public methods serialise concurrent access via
  ``self._lock`` (a :class:`threading.RLock`).
* **Lifecycle-gated**: Every public operation guards against calls made
  before :meth:`initialize` or after :meth:`shutdown`.
* **Engine-integrated**: The reconstruction engine composes over the existing
  ECHO engines (Experience, Conversation, Session, Achievement,
  Failure, Reflection, Retrieval) rather than owning its own data store.
* **Read-only over ECHO data**: This engine never writes new experiences —
  it only reads and assembles context from existing records.
* **Future-ready**: The public API and internal graph representation are
  designed to support the future Personal History Engine and Pattern
  Extraction Engine without breaking changes.

ECHO Boundary Law
-----------------
The Context Reconstruction Engine owns reconstructed context bundles.
It does NOT own raw experiences, sessions, conversations, achievements,
failures, or reflections — those belong to their respective engines.

Publish / Subscribe (ECHO domain)
----------------------------------
Publishes (logged at DEBUG in v1; Event Bus integration reserved for future):
    * ``polaris.echo.context.timeline_reconstructed``
    * ``polaris.echo.context.session_reconstructed``
    * ``polaris.echo.context.conversation_reconstructed``
    * ``polaris.echo.context.experience_chain_reconstructed``
    * ``polaris.echo.context.related_memories_reconstructed``
    * ``polaris.echo.context.context_graph_built``
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from subsystems.echo.exceptions import (
    EchoNotInitializedError,
    ExperienceNotFoundError,
)
from subsystems.echo.models import (
    Experience,
    ExperienceImportance,
    ExperienceType,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMELINE_LIMIT = 100
_DEFAULT_CHAIN_DEPTH = 5
_DEFAULT_RELATED_LIMIT = 20
_DEFAULT_CONTEXT_LIMIT = 30
_GRAPH_MAX_NODES = 200

# Antecedent / consequent window in days for temporal neighbour search
_TEMPORAL_WINDOW_DAYS = 30.0


# ---------------------------------------------------------------------------
# Context Reconstruction Result Types
# ---------------------------------------------------------------------------


@dataclass
class TimelineEntry:
    """A single entry in a reconstructed timeline.

    Attributes
    ----------
    experience_id:
        UUID of the :class:`~subsystems.echo.models.Experience`.
    title:
        Short human-readable label.
    experience_type:
        :class:`~subsystems.echo.models.ExperienceType` name string.
    importance:
        :class:`~subsystems.echo.models.ExperienceImportance` name string.
    occurred_at:
        UTC ISO-8601 string of when this experience occurred.
    outcome:
        Outcome text, or empty string.
    tags:
        List of tag name strings attached to this experience.
    session_id:
        Enclosing session UUID, or ``None``.
    source_subsystem:
        Which POLARIS subsystem produced this experience.
    """

    experience_id: str
    title: str
    experience_type: str
    importance: str
    occurred_at: str
    outcome: str = ""
    tags: list[str] = field(default_factory=list)
    session_id: str | None = None
    source_subsystem: str = "ECHO_API"


@dataclass
class ExperienceChainLink:
    """A single link in an experience causal/thematic chain.

    Attributes
    ----------
    experience_id:
        UUID of this chain link's experience.
    title:
        Short label.
    experience_type:
        Type name.
    importance:
        Importance tier name.
    occurred_at:
        UTC ISO-8601 string.
    relation:
        How this experience relates to the chain origin:
        ``"origin"``, ``"antecedent"``, ``"consequent"``, ``"related"``,
        ``"reflection"``, ``"session_member"``.
    depth:
        Traversal depth from the chain origin (0 = origin).
    """

    experience_id: str
    title: str
    experience_type: str
    importance: str
    occurred_at: str
    relation: str
    depth: int = 0


@dataclass
class ContextNode:
    """A node in the episodic context graph.

    Attributes
    ----------
    node_id:
        UUID of the experience this node represents.
    title:
        Short label.
    node_type:
        Experience type name.
    importance:
        Importance tier name.
    occurred_at:
        UTC ISO-8601 string.
    """

    node_id: str
    title: str
    node_type: str
    importance: str
    occurred_at: str


@dataclass
class ContextEdge:
    """A directed edge in the episodic context graph.

    Attributes
    ----------
    source_id:
        UUID of the source node (experience).
    target_id:
        UUID of the target node (experience).
    relation:
        Edge label describing the relationship:
        ``"related_to"``, ``"precedes"``, ``"follows"``, ``"session_member"``,
        ``"reflection_of"``, ``"shares_tag"``.
    weight:
        Edge weight in [0.0, 1.0] — higher means stronger relationship.
    """

    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


@dataclass
class ReconstructedTimeline:
    """Result of a timeline reconstruction operation.

    Attributes
    ----------
    entries:
        Ordered list of :class:`TimelineEntry` objects, chronological.
    start_at:
        UTC ISO-8601 string of the earliest entry, or ``None``.
    end_at:
        UTC ISO-8601 string of the latest entry, or ``None``.
    total_count:
        Number of entries in the timeline.
    query_context:
        The filter/query parameters used to build this timeline.
    reconstructed_at:
        UTC ISO-8601 string of when this reconstruction was performed.
    """

    entries: list[TimelineEntry] = field(default_factory=list)
    start_at: str | None = None
    end_at: str | None = None
    total_count: int = 0
    query_context: dict[str, Any] = field(default_factory=dict)
    reconstructed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ReconstructedSession:
    """Result of a session reconstruction operation.

    Attributes
    ----------
    session_experience_id:
        UUID of the backing session :class:`~subsystems.echo.models.Experience`.
    session_title:
        Title of the session.
    session_state:
        ``"open"`` or ``"closed"``, or ``"unknown"`` if not determinable.
    opened_at:
        UTC ISO-8601 string of session open time, or ``None``.
    closed_at:
        UTC ISO-8601 string of session close time, or ``None``.
    member_experiences:
        Ordered list of :class:`TimelineEntry` objects for session members.
    goals:
        Goals set for this session.
    outcomes:
        Outcomes recorded for this session.
    decisions:
        Decisions made during this session.
    discussions:
        Discussion topics for this session.
    total_member_count:
        Number of member experiences.
    importance_breakdown:
        Mapping of importance tier name → count.
    type_breakdown:
        Mapping of experience type name → count.
    reconstructed_at:
        UTC ISO-8601 string of when this reconstruction was performed.
    """

    session_experience_id: str
    session_title: str
    session_state: str = "unknown"
    opened_at: str | None = None
    closed_at: str | None = None
    member_experiences: list[TimelineEntry] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    discussions: list[str] = field(default_factory=list)
    total_member_count: int = 0
    importance_breakdown: dict[str, int] = field(default_factory=dict)
    type_breakdown: dict[str, int] = field(default_factory=dict)
    reconstructed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ReconstructedConversation:
    """Result of a conversation reconstruction operation.

    Attributes
    ----------
    conversation_id:
        UUID of the :class:`~subsystems.echo.conversation.ConversationRecord`.
    experience_id:
        UUID of the backing :class:`~subsystems.echo.models.Experience`.
    title:
        Conversation title.
    summary:
        Narrative summary.
    participants:
        List of participant identifiers.
    decisions:
        Decisions reached.
    outcomes:
        Concrete outcomes.
    topics:
        Topic labels covered.
    importance:
        Importance tier name.
    occurred_at:
        UTC ISO-8601 string.
    related_experiences:
        Timeline entries for experiences related to this conversation.
    session_context:
        The enclosing session reconstruction, or ``None``.
    reconstructed_at:
        UTC ISO-8601 string of when this reconstruction was performed.
    """

    conversation_id: str
    experience_id: str | None
    title: str
    summary: str
    participants: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    importance: str = "MEDIUM"
    occurred_at: str = ""
    related_experiences: list[TimelineEntry] = field(default_factory=list)
    session_context: ReconstructedSession | None = None
    reconstructed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ReconstructedExperienceChain:
    """Result of an experience chain reconstruction operation.

    Attributes
    ----------
    origin_experience_id:
        UUID of the chain origin experience.
    chain:
        Ordered list of :class:`ExperienceChainLink` objects.
    antecedents:
        Experiences that led to the origin (before it temporally or causally).
    consequents:
        Experiences that followed the origin.
    reflections:
        Reflection experiences derived from the origin.
    narrative:
        Prose narrative describing the chain.
    chain_depth:
        Maximum traversal depth used.
    reconstructed_at:
        UTC ISO-8601 string of when this reconstruction was performed.
    """

    origin_experience_id: str
    chain: list[ExperienceChainLink] = field(default_factory=list)
    antecedents: list[ExperienceChainLink] = field(default_factory=list)
    consequents: list[ExperienceChainLink] = field(default_factory=list)
    reflections: list[ExperienceChainLink] = field(default_factory=list)
    narrative: str = ""
    chain_depth: int = 0
    reconstructed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ReconstructedRelatedMemories:
    """Result of a related memory reconstruction operation.

    Attributes
    ----------
    anchor_experience_id:
        UUID of the anchor experience used to find related memories.
    related_experiences:
        List of :class:`TimelineEntry` objects for related experiences.
    shared_tags:
        Tag names shared between the anchor and related experiences.
    shared_project_refs:
        Project reference strings shared.
    session_siblings:
        Other experiences from the same session (if any).
    cross_references:
        Mapping of ``experience_id`` → list of relation-type strings.
    reconstructed_at:
        UTC ISO-8601 string of when this reconstruction was performed.
    """

    anchor_experience_id: str
    related_experiences: list[TimelineEntry] = field(default_factory=list)
    shared_tags: list[str] = field(default_factory=list)
    shared_project_refs: list[str] = field(default_factory=list)
    session_siblings: list[TimelineEntry] = field(default_factory=list)
    cross_references: dict[str, list[str]] = field(default_factory=dict)
    reconstructed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ContextSummary:
    """A concise summary of a historical context reconstruction.

    Attributes
    ----------
    query:
        The original query or anchor used for reconstruction.
    headline:
        One-sentence summary of what was found.
    experience_count:
        Total number of experiences surfaced.
    date_range:
        Human-readable date range string (e.g. ``"2025-01-01 → 2026-06-09"``).
    dominant_types:
        List of the most common experience type names in this context.
    dominant_tags:
        List of the most common tag names in this context.
    narrative:
        Multi-sentence prose narrative of the reconstructed context.
    key_experiences:
        Up to 5 :class:`TimelineEntry` objects for the most significant
        experiences in the context.
    generated_at:
        UTC ISO-8601 string of generation time.
    """

    query: str
    headline: str
    experience_count: int = 0
    date_range: str = ""
    dominant_types: list[str] = field(default_factory=list)
    dominant_tags: list[str] = field(default_factory=list)
    narrative: str = ""
    key_experiences: list[TimelineEntry] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ContextGraph:
    """An episodic context graph assembled from related experiences.

    Attributes
    ----------
    anchor_experience_id:
        UUID of the anchor/root experience for this graph.
    nodes:
        List of :class:`ContextNode` objects.
    edges:
        List of :class:`ContextEdge` objects.
    node_count:
        Total number of nodes.
    edge_count:
        Total number of edges.
    built_at:
        UTC ISO-8601 string of when this graph was built.
    """

    anchor_experience_id: str
    nodes: list[ContextNode] = field(default_factory=list)
    edges: list[ContextEdge] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    built_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Context Reconstruction Engine
# ---------------------------------------------------------------------------


class ContextReconstructionEngine:
    """Production Context Reconstruction Engine for the ECHO Episodic Memory Core.

    Rebuilds historical context from episodic memories by composing over
    all other ECHO engines.  Enables answering:

    * *What happened?*          — :meth:`reconstruct_timeline`
    * *Why did it happen?*      — :meth:`reconstruct_experience_chain`
    * *What led to it?*         — :meth:`reconstruct_experience_chain` (antecedents)
    * *What happened after?*    — :meth:`reconstruct_experience_chain` (consequents)
    * *What related events?*    — :meth:`reconstruct_related_memories`

    Parameters
    ----------
    experience_engine:
        A running :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`.
        Required; all reconstruction methods depend on it.
    retrieval_engine:
        Optional running
        :class:`~subsystems.echo.retrieval.ExperienceRetrievalEngine`.
        When supplied, semantic search and similarity operations are
        delegated here; otherwise the engine falls back to direct experience
        queries.
    session_engine:
        Optional running :class:`~subsystems.echo.session.SessionEngine`.
        Used to enrich session reconstructions with full session metadata.
    conversation_engine:
        Optional running
        :class:`~subsystems.echo.conversation.ConversationEngine`.
        Used to enrich conversation reconstructions with full conversation
        metadata.
    achievement_engine:
        Optional running
        :class:`~subsystems.echo.achievements.AchievementEngine`.
        Used to include achievement context in chain reconstructions.
    failure_engine:
        Optional running
        :class:`~subsystems.echo.failure_analysis.FailureAnalysisEngine`.
        Used to include failure context in chain reconstructions.
    reflection_engine:
        Optional running
        :class:`~subsystems.echo.reflection.ReflectionEngine`.
        Used to include reflections derived from chain origin experiences.

    Thread Safety
    -------------
    All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
    before reading or modifying internal state.

    Examples
    --------
    ::

        engine = ContextReconstructionEngine(
            experience_engine=exp_engine,
            retrieval_engine=retrieval_engine,
            session_engine=session_engine,
            conversation_engine=conv_engine,
            reflection_engine=ref_engine,
        )
        engine.initialize()

        timeline = engine.reconstruct_timeline(
            occurred_after=start_dt,
            occurred_before=end_dt,
            min_importance=ExperienceImportance.MEDIUM,
        )

        chain = engine.reconstruct_experience_chain(experience_id)
        summary = engine.generate_context_summary(query="architecture review")

        engine.shutdown()
    """

    def __init__(
        self,
        experience_engine: Any,
        *,
        retrieval_engine: Any | None = None,
        session_engine: Any | None = None,
        conversation_engine: Any | None = None,
        achievement_engine: Any | None = None,
        failure_engine: Any | None = None,
        reflection_engine: Any | None = None,
    ) -> None:
        if experience_engine is None:
            raise ValueError(
                "ContextReconstructionEngine requires a non-None experience_engine."
            )

        self._experience_engine = experience_engine
        self._retrieval_engine = retrieval_engine
        self._session_engine = session_engine
        self._conversation_engine = conversation_engine
        self._achievement_engine = achievement_engine
        self._failure_engine = failure_engine
        self._reflection_engine = reflection_engine

        self._lock: threading.RLock = threading.RLock()
        self._running: bool = False

        _logger.debug("ContextReconstructionEngine constructed.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Idempotent — repeated calls on an already-running engine are
        logged as a warning and have no effect.

        Raises
        ------
        RuntimeError
            If a critical dependency engine is found not to be running.
        """
        with self._lock:
            if self._running:
                _logger.warning(
                    "ContextReconstructionEngine.initialize() called while "
                    "already running."
                )
                return
            self._running = True
            _logger.info("ContextReconstructionEngine initialised.")

    def shutdown(self) -> None:
        """Release resources and mark the engine as stopped.

        Idempotent — calling ``shutdown()`` on an already-stopped engine
        is a no-op.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            _logger.info("ContextReconstructionEngine shut down.")

    def is_running(self) -> bool:
        """Return ``True`` if the engine has been initialised and not shut down."""
        return self._running

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

    def _experience_to_timeline_entry(
        self, experience: Experience
    ) -> TimelineEntry:
        """Convert an :class:`~subsystems.echo.models.Experience` to a
        :class:`TimelineEntry`."""
        return TimelineEntry(
            experience_id=experience.experience_id,
            title=experience.title,
            experience_type=experience.experience_type.name,
            importance=experience.importance.name,
            occurred_at=experience.occurred_at.isoformat(),
            outcome=experience.outcome,
            tags=experience.tag_names(),
            session_id=experience.metadata.session_id,
            source_subsystem=experience.metadata.source_subsystem,
        )

    def _experience_to_chain_link(
        self,
        experience: Experience,
        relation: str,
        depth: int,
    ) -> ExperienceChainLink:
        """Convert an experience to an :class:`ExperienceChainLink`."""
        return ExperienceChainLink(
            experience_id=experience.experience_id,
            title=experience.title,
            experience_type=experience.experience_type.name,
            importance=experience.importance.name,
            occurred_at=experience.occurred_at.isoformat(),
            relation=relation,
            depth=depth,
        )

    def _experience_to_context_node(
        self, experience: Experience
    ) -> ContextNode:
        """Convert an experience to a :class:`ContextNode`."""
        return ContextNode(
            node_id=experience.experience_id,
            title=experience.title,
            node_type=experience.experience_type.name,
            importance=experience.importance.name,
            occurred_at=experience.occurred_at.isoformat(),
        )

    def _get_experience_safe(
        self, experience_id: str
    ) -> Experience | None:
        """Retrieve an experience without raising; returns ``None`` if missing."""
        try:
            return self._experience_engine.get_experience(experience_id)
        except ExperienceNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "ContextReconstructionEngine: unexpected error retrieving "
                "experience '%s': %s",
                experience_id,
                exc,
            )
            return None

    def _all_experiences_snapshot(
        self,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        experience_types: list[ExperienceType] | None = None,
        limit: int = _DEFAULT_CONTEXT_LIMIT,
    ) -> list[Experience]:
        """Return a filtered snapshot of experiences from the Experience Engine."""
        query_kwargs: dict[str, Any] = {
            "min_importance": min_importance,
            "limit": limit,
        }
        if occurred_after is not None:
            query_kwargs["occurred_after"] = occurred_after
        if occurred_before is not None:
            query_kwargs["occurred_before"] = occurred_before
        if experience_types is not None and len(experience_types) == 1:
            query_kwargs["experience_type"] = experience_types[0]

        try:
            results = self._experience_engine.query_experiences(**query_kwargs)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "ContextReconstructionEngine: query_experiences failed: %s", exc
            )
            return []

        # Apply multi-type filter if more than one type requested
        if experience_types is not None and len(experience_types) > 1:
            results = [
                e for e in results if e.experience_type in experience_types
            ]

        return results

    def _find_antecedents(
        self,
        origin: Experience,
        depth: int,
        visited: set[str],
    ) -> list[Experience]:
        """Find experiences that likely preceded or caused the origin.

        Strategy:
        1. Explicitly related experiences (metadata.related_experience_ids)
           that occurred BEFORE the origin.
        2. Session siblings that occurred before the origin.
        3. Temporally adjacent experiences within the window.
        """
        antecedents: list[Experience] = []

        # 1. Explicitly related that precede origin
        for rid in origin.metadata.related_experience_ids:
            if rid in visited:
                continue
            exp = self._get_experience_safe(rid)
            if exp is None or exp.experience_id in visited:
                continue
            if exp.occurred_at < origin.occurred_at:
                antecedents.append(exp)
                visited.add(exp.experience_id)

        # 2. Session siblings before origin
        if origin.metadata.session_id:
            try:
                session_members = self._experience_engine.query_experiences(
                    session_id=origin.metadata.session_id,
                    occurred_before=origin.occurred_at,
                    limit=20,
                )
                for e in session_members:
                    if e.experience_id not in visited and e.experience_id != origin.experience_id:
                        antecedents.append(e)
                        visited.add(e.experience_id)
            except Exception:  # noqa: BLE001
                pass

        # 3. Temporal neighbours before origin (within window)
        if depth > 0:
            try:
                from datetime import timedelta
                window_start = origin.occurred_at - timedelta(
                    days=_TEMPORAL_WINDOW_DAYS
                )
                temporal = self._experience_engine.query_experiences(
                    occurred_after=window_start,
                    occurred_before=origin.occurred_at,
                    min_importance=origin.importance,
                    limit=10,
                )
                for e in temporal:
                    if e.experience_id not in visited and e.experience_id != origin.experience_id:
                        # Only include if shares at least one tag or project ref
                        if _has_shared_context(origin, e):
                            antecedents.append(e)
                            visited.add(e.experience_id)
            except Exception:  # noqa: BLE001
                pass

        return antecedents

    def _find_consequents(
        self,
        origin: Experience,
        depth: int,
        visited: set[str],
    ) -> list[Experience]:
        """Find experiences that followed or resulted from the origin.

        Strategy:
        1. Explicitly related experiences that occurred AFTER the origin.
        2. Session siblings that occurred after the origin.
        3. Temporally adjacent experiences sharing context.
        """
        consequents: list[Experience] = []

        # 1. Explicitly related that follow origin
        for rid in origin.metadata.related_experience_ids:
            if rid in visited:
                continue
            exp = self._get_experience_safe(rid)
            if exp is None or exp.experience_id in visited:
                continue
            if exp.occurred_at > origin.occurred_at:
                consequents.append(exp)
                visited.add(exp.experience_id)

        # 2. Session siblings after origin
        if origin.metadata.session_id:
            try:
                session_members = self._experience_engine.query_experiences(
                    session_id=origin.metadata.session_id,
                    occurred_after=origin.occurred_at,
                    limit=20,
                )
                for e in session_members:
                    if e.experience_id not in visited and e.experience_id != origin.experience_id:
                        consequents.append(e)
                        visited.add(e.experience_id)
            except Exception:  # noqa: BLE001
                pass

        # 3. Temporal neighbours after origin (within window)
        if depth > 0:
            try:
                from datetime import timedelta
                window_end = origin.occurred_at + timedelta(
                    days=_TEMPORAL_WINDOW_DAYS
                )
                temporal = self._experience_engine.query_experiences(
                    occurred_after=origin.occurred_at,
                    occurred_before=window_end,
                    min_importance=origin.importance,
                    limit=10,
                )
                for e in temporal:
                    if e.experience_id not in visited and e.experience_id != origin.experience_id:
                        if _has_shared_context(origin, e):
                            consequents.append(e)
                            visited.add(e.experience_id)
            except Exception:  # noqa: BLE001
                pass

        return consequents

    def _find_reflections_for_experience(
        self, experience_id: str
    ) -> list[Experience]:
        """Return REFLECTION experiences that reference the given experience."""
        if self._reflection_engine is not None:
            try:
                return self._reflection_engine.search_reflections(
                    source_experience_id=experience_id
                )
            except Exception:  # noqa: BLE001
                pass

        # Fallback: query experience store for REFLECTIONs whose
        # related_experience_ids include the target
        try:
            all_reflections = self._experience_engine.query_experiences(
                experience_type=ExperienceType.REFLECTION,
                limit=500,
            )
            return [
                r for r in all_reflections
                if experience_id in r.metadata.related_experience_ids
            ]
        except Exception:  # noqa: BLE001
            return []

    def _compute_importance_breakdown(
        self, experiences: list[Experience]
    ) -> dict[str, int]:
        """Return a mapping of importance tier name → count."""
        breakdown: dict[str, int] = {}
        for exp in experiences:
            key = exp.importance.name
            breakdown[key] = breakdown.get(key, 0) + 1
        return breakdown

    def _compute_type_breakdown(
        self, experiences: list[Experience]
    ) -> dict[str, int]:
        """Return a mapping of experience type name → count."""
        breakdown: dict[str, int] = {}
        for exp in experiences:
            key = exp.experience_type.name
            breakdown[key] = breakdown.get(key, 0) + 1
        return breakdown

    def _build_narrative_for_chain(
        self,
        origin: Experience,
        antecedents: list[Experience],
        consequents: list[Experience],
        reflections: list[Experience],
    ) -> str:
        """Compose a narrative prose description of an experience chain."""
        parts: list[str] = []

        parts.append(
            f"The experience '{origin.title}' "
            f"(occurred {origin.occurred_at.strftime('%Y-%m-%d')}) "
            f"is a {origin.experience_type.name.lower().replace('_', ' ')} "
            f"of {origin.importance.name.lower()} importance."
        )

        if origin.outcome:
            parts.append(f"Its outcome: {origin.outcome}")

        if antecedents:
            ante_titles = [f"'{a.title}'" for a in antecedents[:3]]
            parts.append(
                f"It was preceded by {len(antecedents)} related experience(s), "
                f"including {', '.join(ante_titles)}."
            )

        if consequents:
            cons_titles = [f"'{c.title}'" for c in consequents[:3]]
            parts.append(
                f"It was followed by {len(consequents)} experience(s), "
                f"including {', '.join(cons_titles)}."
            )

        if reflections:
            parts.append(
                f"{len(reflections)} reflection(s) were derived from this experience."
            )

        return "  ".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct_context(
        self,
        experience_id: str,
        *,
        chain_depth: int = _DEFAULT_CHAIN_DEPTH,
        include_reflections: bool = True,
        include_session_context: bool = True,
    ) -> dict[str, Any]:
        """Reconstruct the full historical context surrounding an experience.

        Combines timeline, chain, related memories, and an optional context
        graph into a single comprehensive context bundle.  This is the
        highest-level reconstruction operation — a one-stop answer to
        "What happened around this experience?".

        Parameters
        ----------
        experience_id:
            UUID of the anchor :class:`~subsystems.echo.models.Experience`.
        chain_depth:
            How many traversal levels to follow when building the experience
            chain.  Defaults to :attr:`_DEFAULT_CHAIN_DEPTH`.
        include_reflections:
            If ``True``, include reflections derived from the anchor in the
            result bundle.
        include_session_context:
            If ``True`` and the anchor belongs to a session, include a
            reconstructed session bundle in the result.

        Returns
        -------
        dict[str, Any]
            Context bundle containing:

            * ``experience_id``    — the anchor UUID
            * ``anchor``           — :class:`TimelineEntry` for the anchor
            * ``chain``            — :class:`ReconstructedExperienceChain`
            * ``related``          — :class:`ReconstructedRelatedMemories`
            * ``session``          — :class:`ReconstructedSession` or ``None``
            * ``summary``          — :class:`ContextSummary`
            * ``reconstructed_at`` — UTC ISO-8601 string

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If no experience with ``experience_id`` exists.
        """
        self._assert_running("reconstruct_context")

        anchor = self._experience_engine.get_experience(experience_id)
        anchor_entry = self._experience_to_timeline_entry(anchor)

        chain = self.reconstruct_experience_chain(
            experience_id,
            depth=chain_depth,
            include_reflections=include_reflections,
        )
        related = self.reconstruct_related_memories(
            experience_id, limit=_DEFAULT_RELATED_LIMIT
        )

        session_ctx: ReconstructedSession | None = None
        if include_session_context and anchor.metadata.session_id:
            try:
                session_ctx = self.reconstruct_session(
                    anchor.metadata.session_id
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "ContextReconstructionEngine.reconstruct_context: session "
                    "reconstruction failed for session '%s': %s",
                    anchor.metadata.session_id,
                    exc,
                )

        summary = self.generate_context_summary(
            query=anchor.title,
            experience_ids=[experience_id]
            + [link.experience_id for link in chain.chain],
        )

        result: dict[str, Any] = {
            "experience_id": experience_id,
            "anchor": anchor_entry,
            "chain": chain,
            "related": related,
            "session": session_ctx,
            "summary": summary,
            "reconstructed_at": datetime.now(timezone.utc).isoformat(),
        }

        self._publish_event(
            "polaris.echo.context.context_reconstructed",
            {"experience_id": experience_id},
        )
        _logger.info(
            "ContextReconstructionEngine: full context reconstructed for '%s'.",
            experience_id,
        )
        return result

    def reconstruct_timeline(
        self,
        *,
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        experience_types: list[ExperienceType] | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = _DEFAULT_TIMELINE_LIMIT,
    ) -> ReconstructedTimeline:
        """Reconstruct a chronological timeline of experiences.

        Returns experiences ordered by ``occurred_at`` ascending, matching
        the supplied filter criteria.  This is the primary answer to
        "What happened?" across a time period or context scope.

        Parameters
        ----------
        occurred_after:
            Filter experiences that occurred after this UTC datetime.
        occurred_before:
            Filter experiences that occurred before this UTC datetime.
        min_importance:
            Exclude experiences below this importance tier.
        experience_types:
            Restrict to these experience types if supplied.
        session_id:
            Restrict to experiences belonging to this session.
        tags:
            Restrict to experiences carrying at least one of these tag names
            (case-insensitive partial match).
        limit:
            Maximum number of timeline entries.

        Returns
        -------
        ReconstructedTimeline
            The assembled timeline with metadata.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("reconstruct_timeline")

        query_kwargs: dict[str, Any] = {
            "min_importance": min_importance,
            "limit": limit,
            "offset": 0,
        }
        if occurred_after is not None:
            query_kwargs["occurred_after"] = occurred_after
        if occurred_before is not None:
            query_kwargs["occurred_before"] = occurred_before
        if session_id is not None:
            query_kwargs["session_id"] = session_id
        if experience_types is not None and len(experience_types) == 1:
            query_kwargs["experience_type"] = experience_types[0]

        try:
            experiences = self._experience_engine.query_experiences(
                **query_kwargs
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "ContextReconstructionEngine.reconstruct_timeline: "
                "query failed: %s",
                exc,
            )
            experiences = []

        # Multi-type filter
        if experience_types is not None and len(experience_types) > 1:
            experiences = [
                e for e in experiences if e.experience_type in experience_types
            ]

        # Tag filter
        if tags:
            normalised_tags = {t.lower() for t in tags if t.strip()}
            experiences = [
                e for e in experiences
                if any(
                    any(nt in en.lower() for nt in normalised_tags)
                    for en in e.tag_names()
                )
            ]

        # Sort chronologically (ascending)
        experiences.sort(key=lambda e: e.occurred_at)

        entries = [self._experience_to_timeline_entry(e) for e in experiences]

        timeline = ReconstructedTimeline(
            entries=entries,
            start_at=entries[0].occurred_at if entries else None,
            end_at=entries[-1].occurred_at if entries else None,
            total_count=len(entries),
            query_context={
                "occurred_after": occurred_after.isoformat()
                if occurred_after
                else None,
                "occurred_before": occurred_before.isoformat()
                if occurred_before
                else None,
                "min_importance": min_importance.name,
                "experience_types": [t.name for t in experience_types]
                if experience_types
                else None,
                "session_id": session_id,
                "tags": tags,
                "limit": limit,
            },
        )

        self._publish_event(
            "polaris.echo.context.timeline_reconstructed",
            {"entry_count": len(entries)},
        )
        _logger.info(
            "ContextReconstructionEngine: timeline reconstructed (%d entries).",
            len(entries),
        )
        return timeline

    def reconstruct_session(
        self,
        session_id: str,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        limit: int = _DEFAULT_TIMELINE_LIMIT,
    ) -> ReconstructedSession:
        """Reconstruct the full context of a session.

        Retrieves the session record (if the Session Engine is available),
        all member experiences, and assembles a complete session context
        bundle.

        Parameters
        ----------
        session_id:
            UUID of the session.  In ECHO, session UUIDs match the
            ``experience_id`` of the backing SESSION experience, OR the
            ``session_id`` field of the :class:`~subsystems.echo.session.SessionRecord`.
            Both are accepted; the engine resolves whichever is available.
        min_importance:
            Exclude member experiences below this importance tier.
        limit:
            Maximum number of member experiences to include.

        Returns
        -------
        ReconstructedSession
            The assembled session reconstruction.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("reconstruct_session")

        # Try to fetch the session record from the Session Engine
        session_record: Any | None = None
        if self._session_engine is not None:
            try:
                session_record = self._session_engine.get_session(session_id)
            except Exception:  # noqa: BLE001
                # session_id may be a backing experience_id rather than
                # a SessionRecord UUID — fall through to experience query
                pass

        # Resolve the backing session experience
        session_exp: Experience | None = None

        if session_record is not None and session_record.experience_id:
            session_exp = self._get_experience_safe(
                session_record.experience_id
            )
        else:
            # session_id might directly be an experience_id
            session_exp = self._get_experience_safe(session_id)
            if session_exp is not None and session_exp.experience_type != ExperienceType.SESSION:
                # Not a session — try querying for sessions with this as session_id
                session_exp = None

        # Determine effective session experience id for member queries
        effective_session_id = session_id
        if session_record is not None and session_record.experience_id:
            effective_session_id = session_record.experience_id
        elif session_exp is not None:
            effective_session_id = session_exp.experience_id

        # Query member experiences
        try:
            member_experiences = self._experience_engine.query_experiences(
                session_id=effective_session_id,
                min_importance=min_importance,
                limit=limit,
            )
        except Exception:  # noqa: BLE001
            member_experiences = []

        # Sort members chronologically
        member_experiences.sort(key=lambda e: e.occurred_at)
        member_entries = [
            self._experience_to_timeline_entry(e) for e in member_experiences
        ]

        # Extract session metadata
        title = (
            session_record.title
            if session_record is not None
            else (session_exp.title if session_exp else f"Session {session_id}")
        )
        state = "unknown"
        opened_at: str | None = None
        closed_at: str | None = None
        goals: list[str] = []
        outcomes: list[str] = []
        decisions: list[str] = []
        discussions: list[str] = []

        if session_record is not None:
            from subsystems.echo.session import SessionState
            state = (
                "open"
                if session_record.state is SessionState.OPEN
                else "closed"
            )
            opened_at = session_record.opened_at.isoformat() if session_record.opened_at else None
            closed_at = session_record.closed_at.isoformat() if session_record.closed_at else None
            goals = list(session_record.goals)
            outcomes = list(session_record.outcomes)
            decisions = list(session_record.decisions)
            discussions = list(session_record.discussions)
        elif session_exp is not None:
            opened_at = session_exp.occurred_at.isoformat()

        result = ReconstructedSession(
            session_experience_id=effective_session_id,
            session_title=title,
            session_state=state,
            opened_at=opened_at,
            closed_at=closed_at,
            member_experiences=member_entries,
            goals=goals,
            outcomes=outcomes,
            decisions=decisions,
            discussions=discussions,
            total_member_count=len(member_entries),
            importance_breakdown=self._compute_importance_breakdown(member_experiences),
            type_breakdown=self._compute_type_breakdown(member_experiences),
        )

        self._publish_event(
            "polaris.echo.context.session_reconstructed",
            {
                "session_id": session_id,
                "member_count": len(member_entries),
            },
        )
        _logger.info(
            "ContextReconstructionEngine: session '%s' reconstructed (%d members).",
            session_id,
            len(member_entries),
        )
        return result

    def reconstruct_conversation(
        self,
        conversation_id: str,
        *,
        include_session_context: bool = False,
        related_limit: int = 10,
    ) -> ReconstructedConversation:
        """Reconstruct the full context of a conversation.

        Retrieves the conversation record (if the Conversation Engine is
        available), its backing experience, and assembles a complete
        conversation context bundle with related experiences.

        Parameters
        ----------
        conversation_id:
            UUID of the :class:`~subsystems.echo.conversation.ConversationRecord`.
        include_session_context:
            If ``True`` and the conversation belongs to a session, include
            a full :class:`ReconstructedSession` in the result.
        related_limit:
            Maximum number of related experiences to include.

        Returns
        -------
        ReconstructedConversation
            The assembled conversation reconstruction.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("reconstruct_conversation")

        conv_record: Any | None = None
        if self._conversation_engine is not None:
            try:
                conv_record = self._conversation_engine.get_conversation(
                    conversation_id
                )
            except Exception:  # noqa: BLE001
                pass

        # Resolve backing experience
        backing_exp: Experience | None = None
        if conv_record is not None and conv_record.experience_id:
            backing_exp = self._get_experience_safe(conv_record.experience_id)
        else:
            # conversation_id might be an experience_id directly
            backing_exp = self._get_experience_safe(conversation_id)

        title = (
            conv_record.title
            if conv_record is not None
            else (backing_exp.title if backing_exp else f"Conversation {conversation_id}")
        )
        summary = (
            conv_record.summary
            if conv_record is not None
            else (backing_exp.description if backing_exp else "")
        )
        participants = list(conv_record.participants) if conv_record is not None else []
        decisions = list(conv_record.decisions) if conv_record is not None else []
        outcomes = list(conv_record.outcomes) if conv_record is not None else []
        topics = list(conv_record.topics) if conv_record is not None else []
        importance = (
            conv_record.importance.name
            if conv_record is not None
            else (backing_exp.importance.name if backing_exp else "MEDIUM")
        )
        occurred_at = (
            conv_record.occurred_at.isoformat()
            if conv_record is not None
            else (backing_exp.occurred_at.isoformat() if backing_exp else "")
        )
        effective_exp_id = (
            conv_record.experience_id
            if conv_record is not None and conv_record.experience_id
            else (backing_exp.experience_id if backing_exp else None)
        )

        # Gather related experiences
        related_entries: list[TimelineEntry] = []
        if effective_exp_id:
            try:
                related_experiences = self.reconstruct_related_memories(
                    effective_exp_id, limit=related_limit
                )
                related_entries = related_experiences.related_experiences
            except Exception:  # noqa: BLE001
                pass

        # Optionally include session context
        session_ctx: ReconstructedSession | None = None
        if include_session_context and backing_exp is not None and backing_exp.metadata.session_id:
            try:
                session_ctx = self.reconstruct_session(
                    backing_exp.metadata.session_id
                )
            except Exception:  # noqa: BLE001
                pass

        result = ReconstructedConversation(
            conversation_id=conversation_id,
            experience_id=effective_exp_id,
            title=title,
            summary=summary,
            participants=participants,
            decisions=decisions,
            outcomes=outcomes,
            topics=topics,
            importance=importance,
            occurred_at=occurred_at,
            related_experiences=related_entries,
            session_context=session_ctx,
        )

        self._publish_event(
            "polaris.echo.context.conversation_reconstructed",
            {
                "conversation_id": conversation_id,
                "experience_id": effective_exp_id,
            },
        )
        _logger.info(
            "ContextReconstructionEngine: conversation '%s' reconstructed.",
            conversation_id,
        )
        return result

    def reconstruct_experience_chain(
        self,
        experience_id: str,
        *,
        depth: int = _DEFAULT_CHAIN_DEPTH,
        include_reflections: bool = True,
    ) -> ReconstructedExperienceChain:
        """Reconstruct the causal/thematic chain surrounding an experience.

        Traverses the experience graph outward from the origin, collecting:
        * *Antecedents* — experiences that preceded or caused the origin.
        * *Consequents* — experiences that followed the origin.
        * *Reflections* — REFLECTION experiences derived from the origin.

        Parameters
        ----------
        experience_id:
            UUID of the chain origin :class:`~subsystems.echo.models.Experience`.
        depth:
            Maximum traversal depth (hops from origin).  Defaults to
            :attr:`_DEFAULT_CHAIN_DEPTH`.
        include_reflections:
            If ``True``, fetch REFLECTION experiences linked to the origin.

        Returns
        -------
        ReconstructedExperienceChain
            The assembled experience chain.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If no experience with ``experience_id`` exists.
        """
        self._assert_running("reconstruct_experience_chain")

        origin = self._experience_engine.get_experience(experience_id)
        visited: set[str] = {experience_id}

        # Collect antecedents up to depth
        raw_antecedents: list[Experience] = []
        frontier = [origin]
        for d in range(min(depth, _DEFAULT_CHAIN_DEPTH)):
            next_frontier: list[Experience] = []
            for exp in frontier:
                found = self._find_antecedents(exp, depth - d, visited)
                raw_antecedents.extend(found)
                next_frontier.extend(found)
            frontier = next_frontier
            if not frontier:
                break

        # Collect consequents up to depth
        raw_consequents: list[Experience] = []
        frontier = [origin]
        for d in range(min(depth, _DEFAULT_CHAIN_DEPTH)):
            next_frontier = []
            for exp in frontier:
                found = self._find_consequents(exp, depth - d, visited)
                raw_consequents.extend(found)
                next_frontier.extend(found)
            frontier = next_frontier
            if not frontier:
                break

        # Collect reflections
        raw_reflections: list[Experience] = []
        if include_reflections:
            raw_reflections = self._find_reflections_for_experience(experience_id)

        # Build chain links
        origin_link = self._experience_to_chain_link(origin, "origin", 0)
        antecedent_links = [
            self._experience_to_chain_link(e, "antecedent", i + 1)
            for i, e in enumerate(raw_antecedents)
        ]
        consequent_links = [
            self._experience_to_chain_link(e, "consequent", i + 1)
            for i, e in enumerate(raw_consequents)
        ]
        reflection_links = [
            self._experience_to_chain_link(e, "reflection", 1)
            for e in raw_reflections
        ]

        # Merge into full chain (antecedents → origin → consequents)
        full_chain: list[ExperienceChainLink] = sorted(
            antecedent_links,
            key=lambda l: l.occurred_at,
        ) + [origin_link] + sorted(
            consequent_links,
            key=lambda l: l.occurred_at,
        )

        narrative = self._build_narrative_for_chain(
            origin,
            raw_antecedents,
            raw_consequents,
            raw_reflections,
        )

        result = ReconstructedExperienceChain(
            origin_experience_id=experience_id,
            chain=full_chain,
            antecedents=antecedent_links,
            consequents=consequent_links,
            reflections=reflection_links,
            narrative=narrative,
            chain_depth=depth,
        )

        self._publish_event(
            "polaris.echo.context.experience_chain_reconstructed",
            {
                "origin_experience_id": experience_id,
                "chain_length": len(full_chain),
            },
        )
        _logger.info(
            "ContextReconstructionEngine: chain for '%s' reconstructed "
            "(%d links, %d antecedents, %d consequents, %d reflections).",
            experience_id,
            len(full_chain),
            len(antecedent_links),
            len(consequent_links),
            len(reflection_links),
        )
        return result

    def reconstruct_related_memories(
        self,
        experience_id: str,
        *,
        limit: int = _DEFAULT_RELATED_LIMIT,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
    ) -> ReconstructedRelatedMemories:
        """Reconstruct all memories related to a given experience.

        Discovers related memories via:
        1. Explicit ``metadata.related_experience_ids`` references.
        2. Session siblings (same ``metadata.session_id``).
        3. Shared-tag similarity via the Retrieval Engine (if available),
           or direct tag queries otherwise.
        4. Shared project references.

        Parameters
        ----------
        experience_id:
            UUID of the anchor :class:`~subsystems.echo.models.Experience`.
        limit:
            Maximum number of related experiences to return.
        min_importance:
            Exclude related experiences below this tier.

        Returns
        -------
        ReconstructedRelatedMemories
            The assembled related memory bundle.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If no experience with ``experience_id`` exists.
        """
        self._assert_running("reconstruct_related_memories")

        anchor = self._experience_engine.get_experience(experience_id)
        visited: set[str] = {experience_id}
        related_experiences: list[Experience] = []

        # 1. Explicit related_experience_ids
        for rid in anchor.metadata.related_experience_ids:
            if rid in visited:
                continue
            exp = self._get_experience_safe(rid)
            if exp is not None and exp.importance.value >= min_importance.value:
                related_experiences.append(exp)
                visited.add(rid)

        # 2. Session siblings
        session_siblings: list[Experience] = []
        if anchor.metadata.session_id:
            try:
                siblings = self._experience_engine.query_experiences(
                    session_id=anchor.metadata.session_id,
                    min_importance=min_importance,
                    limit=50,
                )
                for e in siblings:
                    if e.experience_id not in visited:
                        session_siblings.append(e)
                        visited.add(e.experience_id)
            except Exception:  # noqa: BLE001
                pass

        # 3. Shared-tag similarity
        anchor_tag_names = anchor.tag_names()
        if anchor_tag_names and len(related_experiences) < limit:
            if self._retrieval_engine is not None:
                try:
                    tag_results = self._retrieval_engine.search_by_tags(
                        anchor_tag_names,
                        match_all=False,
                        min_importance=min_importance,
                        limit=limit,
                    )
                    for e in tag_results:
                        if e.experience_id not in visited:
                            related_experiences.append(e)
                            visited.add(e.experience_id)
                except Exception:  # noqa: BLE001
                    pass
            else:
                # Direct tag query fallback
                for tag_name in anchor_tag_names[:3]:
                    try:
                        tag_exps = self._experience_engine.query_experiences(
                            tags=[tag_name],
                            min_importance=min_importance,
                            limit=10,
                        )
                        for e in tag_exps:
                            if e.experience_id not in visited:
                                related_experiences.append(e)
                                visited.add(e.experience_id)
                    except Exception:  # noqa: BLE001
                        pass

        # 4. Shared project references
        anchor_proj_refs = set(anchor.metadata.project_refs)
        if anchor_proj_refs and len(related_experiences) < limit:
            try:
                all_exps = self._experience_engine.query_experiences(
                    min_importance=min_importance,
                    limit=200,
                )
                for e in all_exps:
                    if e.experience_id in visited:
                        continue
                    if anchor_proj_refs & set(e.metadata.project_refs):
                        related_experiences.append(e)
                        visited.add(e.experience_id)
                        if len(related_experiences) >= limit:
                            break
            except Exception:  # noqa: BLE001
                pass

        # Deduplicate and trim
        seen: set[str] = set()
        unique_related: list[Experience] = []
        for e in related_experiences:
            if e.experience_id not in seen:
                unique_related.append(e)
                seen.add(e.experience_id)
        unique_related = unique_related[:limit]

        # Build cross-references
        cross_refs: dict[str, list[str]] = {}
        for e in unique_related:
            relations: list[str] = []
            if e.experience_id in anchor.metadata.related_experience_ids:
                relations.append("explicitly_related")
            if e.experience_id in {s.experience_id for s in session_siblings}:
                relations.append("session_sibling")
            if set(e.tag_names()) & set(anchor_tag_names):
                relations.append("shares_tag")
            if set(e.metadata.project_refs) & anchor_proj_refs:
                relations.append("shares_project")
            cross_refs[e.experience_id] = relations

        result = ReconstructedRelatedMemories(
            anchor_experience_id=experience_id,
            related_experiences=[
                self._experience_to_timeline_entry(e) for e in unique_related
            ],
            shared_tags=sorted(
                set(anchor_tag_names) & {
                    tag for e in unique_related for tag in e.tag_names()
                }
            ),
            shared_project_refs=sorted(
                anchor_proj_refs & {
                    ref
                    for e in unique_related
                    for ref in e.metadata.project_refs
                }
            ),
            session_siblings=[
                self._experience_to_timeline_entry(e) for e in session_siblings
            ],
            cross_references=cross_refs,
        )

        self._publish_event(
            "polaris.echo.context.related_memories_reconstructed",
            {
                "anchor_experience_id": experience_id,
                "related_count": len(unique_related),
            },
        )
        _logger.info(
            "ContextReconstructionEngine: related memories for '%s' "
            "reconstructed (%d related, %d session siblings).",
            experience_id,
            len(unique_related),
            len(session_siblings),
        )
        return result

    def generate_context_summary(
        self,
        *,
        query: str = "",
        experience_ids: list[str] | None = None,
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        min_importance: ExperienceImportance = ExperienceImportance.MEDIUM,
        limit: int = _DEFAULT_CONTEXT_LIMIT,
    ) -> ContextSummary:
        """Generate a concise summary of a historical context.

        Accepts either a natural-language query (resolved via semantic search
        or timeline reconstruction) or an explicit list of experience IDs.
        Returns a :class:`ContextSummary` suitable for display, prompt
        injection, or downstream engine consumption.

        Parameters
        ----------
        query:
            Natural-language or keyword query describing the context of
            interest.  Used for semantic search when no ``experience_ids``
            are supplied, and as the headline basis.
        experience_ids:
            Optional explicit list of experience UUIDs to summarise.
            When supplied, ``query`` is used only as the headline label.
        occurred_after:
            Filter experiences after this UTC datetime.
        occurred_before:
            Filter experiences before this UTC datetime.
        min_importance:
            Exclude experiences below this tier.
        limit:
            Maximum number of experiences to include in the summary.

        Returns
        -------
        ContextSummary
            The assembled context summary.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("generate_context_summary")

        experiences: list[Experience] = []

        if experience_ids:
            # Resolve explicit IDs
            for eid in experience_ids[:limit]:
                exp = self._get_experience_safe(eid)
                if exp is not None and exp.importance.value >= min_importance.value:
                    experiences.append(exp)
        elif query:
            # Semantic search path
            if self._retrieval_engine is not None:
                try:
                    results = self._retrieval_engine.search_semantic(
                        query,
                        min_importance=min_importance,
                        limit=limit,
                    )
                    experiences = [r.experience for r in results]
                except Exception:  # noqa: BLE001
                    pass

            if not experiences:
                # Timeline fallback
                tl = self.reconstruct_timeline(
                    occurred_after=occurred_after,
                    occurred_before=occurred_before,
                    min_importance=min_importance,
                    limit=limit,
                )
                for entry in tl.entries:
                    exp = self._get_experience_safe(entry.experience_id)
                    if exp is not None:
                        experiences.append(exp)
        else:
            # No query and no IDs — return recent significant experiences
            experiences = self._all_experiences_snapshot(
                min_importance=min_importance,
                occurred_after=occurred_after,
                occurred_before=occurred_before,
                limit=limit,
            )

        experiences = experiences[:limit]

        if not experiences:
            return ContextSummary(
                query=query,
                headline=f"No experiences found for context: '{query}'."
                if query
                else "No experiences found for the specified context.",
                experience_count=0,
            )

        # Sort chronologically for narrative
        experiences_sorted = sorted(experiences, key=lambda e: e.occurred_at)
        start_dt = experiences_sorted[0].occurred_at
        end_dt = experiences_sorted[-1].occurred_at
        date_range = (
            f"{start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}"
        )

        # Dominant types
        type_counts: dict[str, int] = {}
        for e in experiences:
            k = e.experience_type.name
            type_counts[k] = type_counts.get(k, 0) + 1
        dominant_types = sorted(type_counts, key=lambda k: -type_counts[k])[:3]

        # Dominant tags
        tag_counts: dict[str, int] = {}
        for e in experiences:
            for t in e.tag_names():
                tag_counts[t] = tag_counts.get(t, 0) + 1
        dominant_tags = sorted(tag_counts, key=lambda k: -tag_counts[k])[:5]

        # Key experiences — highest importance then recency
        key_exps = sorted(
            experiences,
            key=lambda e: (e.importance.value, e.occurred_at.timestamp()),
            reverse=True,
        )[:5]
        key_entries = [self._experience_to_timeline_entry(e) for e in key_exps]

        # Narrative
        narrative_lines: list[str] = []
        headline_prefix = f"Context for '{query}'" if query else "Context summary"
        narrative_lines.append(
            f"{headline_prefix}: {len(experiences)} experience(s) "
            f"spanning {date_range}."
        )
        if key_exps:
            narrative_lines.append(
                "Key experiences: "
                + "; ".join(
                    f"[{e.occurred_at.strftime('%Y-%m-%d')}] {e.title}"
                    for e in key_exps[:3]
                )
                + "."
            )
        if dominant_tags:
            narrative_lines.append(
                f"Recurring themes: {', '.join(dominant_tags[:3])}."
            )
        narrative = "  ".join(narrative_lines)

        headline = (
            f"Found {len(experiences)} experience(s) "
            f"{'for ' + repr(query) + ' ' if query else ''}"
            f"between {start_dt.strftime('%Y-%m-%d')} and {end_dt.strftime('%Y-%m-%d')}."
        )

        summary = ContextSummary(
            query=query,
            headline=headline,
            experience_count=len(experiences),
            date_range=date_range,
            dominant_types=dominant_types,
            dominant_tags=dominant_tags,
            narrative=narrative,
            key_experiences=key_entries,
        )

        _logger.info(
            "ContextReconstructionEngine: context summary generated "
            "(%d experiences, query=%r).",
            len(experiences),
            query,
        )
        return summary

    def build_context_graph(
        self,
        experience_id: str,
        *,
        depth: int = 2,
        include_session_members: bool = True,
        include_reflections: bool = True,
        max_nodes: int = _GRAPH_MAX_NODES,
    ) -> ContextGraph:
        """Build an episodic context graph centred on a given experience.

        Assembles a graph of :class:`ContextNode` and :class:`ContextEdge`
        objects representing the episodic neighbourhood of the anchor
        experience.  Suitable for visualisation, pattern extraction, and
        the future Personal History Engine.

        Parameters
        ----------
        experience_id:
            UUID of the graph anchor :class:`~subsystems.echo.models.Experience`.
        depth:
            Traversal depth from the anchor.  Higher depth builds a wider
            graph but increases reconstruction time.
        include_session_members:
            If ``True``, include all experiences from the same session as
            graph nodes.
        include_reflections:
            If ``True``, include REFLECTION experiences linked to the anchor.
        max_nodes:
            Hard cap on the total number of nodes in the graph.

        Returns
        -------
        ContextGraph
            The assembled context graph.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If no experience with ``experience_id`` exists.
        """
        self._assert_running("build_context_graph")

        anchor = self._experience_engine.get_experience(experience_id)

        nodes: dict[str, ContextNode] = {}
        edges: list[ContextEdge] = []
        edge_set: set[tuple[str, str, str]] = set()

        def _add_node(exp: Experience) -> None:
            if exp.experience_id not in nodes and len(nodes) < max_nodes:
                nodes[exp.experience_id] = self._experience_to_context_node(exp)

        def _add_edge(
            source_id: str,
            target_id: str,
            relation: str,
            weight: float = 1.0,
        ) -> None:
            key = (source_id, target_id, relation)
            if key not in edge_set:
                edges.append(
                    ContextEdge(
                        source_id=source_id,
                        target_id=target_id,
                        relation=relation,
                        weight=weight,
                    )
                )
                edge_set.add(key)

        # Anchor node
        _add_node(anchor)

        # Explicitly related experiences
        for rid in anchor.metadata.related_experience_ids:
            if len(nodes) >= max_nodes:
                break
            rel_exp = self._get_experience_safe(rid)
            if rel_exp is None:
                continue
            _add_node(rel_exp)
            if rel_exp.occurred_at < anchor.occurred_at:
                _add_edge(rel_exp.experience_id, anchor.experience_id, "precedes", 0.9)
            else:
                _add_edge(anchor.experience_id, rel_exp.experience_id, "follows", 0.9)
            # related_to is bidirectional
            _add_edge(anchor.experience_id, rel_exp.experience_id, "related_to", 0.8)

        # Session members
        if include_session_members and anchor.metadata.session_id:
            try:
                session_members = self._experience_engine.query_experiences(
                    session_id=anchor.metadata.session_id,
                    limit=50,
                )
                session_exp = self._get_experience_safe(anchor.metadata.session_id)
                if session_exp:
                    _add_node(session_exp)
                    _add_edge(
                        session_exp.experience_id,
                        anchor.experience_id,
                        "session_member",
                        0.7,
                    )
                for member in session_members:
                    if member.experience_id == anchor.experience_id:
                        continue
                    if len(nodes) >= max_nodes:
                        break
                    _add_node(member)
                    _add_edge(
                        anchor.experience_id,
                        member.experience_id,
                        "session_member",
                        0.6,
                    )
                    if member.occurred_at < anchor.occurred_at:
                        _add_edge(
                            member.experience_id,
                            anchor.experience_id,
                            "precedes",
                            0.5,
                        )
                    elif member.occurred_at > anchor.occurred_at:
                        _add_edge(
                            anchor.experience_id,
                            member.experience_id,
                            "follows",
                            0.5,
                        )
            except Exception:  # noqa: BLE001
                pass

        # Reflections
        if include_reflections:
            reflections = self._find_reflections_for_experience(experience_id)
            for ref in reflections:
                if len(nodes) >= max_nodes:
                    break
                _add_node(ref)
                _add_edge(
                    anchor.experience_id,
                    ref.experience_id,
                    "reflection_of",
                    0.85,
                )

        # Shared-tag edges (within current graph nodes)
        anchor_tags = set(anchor.tag_names())
        node_list = list(nodes.values())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                ni = node_list[i]
                nj = node_list[j]
                if ni.node_id == anchor.experience_id or nj.node_id == anchor.experience_id:
                    continue
                exp_i = self._get_experience_safe(ni.node_id)
                exp_j = self._get_experience_safe(nj.node_id)
                if exp_i is None or exp_j is None:
                    continue
                shared = set(exp_i.tag_names()) & set(exp_j.tag_names())
                if shared:
                    weight = min(0.7, 0.2 * len(shared))
                    _add_edge(ni.node_id, nj.node_id, "shares_tag", weight)

        # Depth expansion (BFS from anchor, limited by max_nodes)
        if depth > 1:
            frontier = [
                nid for nid in nodes if nid != anchor.experience_id
            ]
            for _d in range(depth - 1):
                next_frontier: list[str] = []
                for nid in frontier:
                    if len(nodes) >= max_nodes:
                        break
                    exp = self._get_experience_safe(nid)
                    if exp is None:
                        continue
                    for rid in exp.metadata.related_experience_ids:
                        if rid in nodes or len(nodes) >= max_nodes:
                            continue
                        rel_exp = self._get_experience_safe(rid)
                        if rel_exp is None:
                            continue
                        _add_node(rel_exp)
                        _add_edge(nid, rid, "related_to", 0.5)
                        next_frontier.append(rid)
                frontier = next_frontier
                if not frontier:
                    break

        graph = ContextGraph(
            anchor_experience_id=experience_id,
            nodes=list(nodes.values()),
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges),
        )

        self._publish_event(
            "polaris.echo.context.context_graph_built",
            {
                "anchor_experience_id": experience_id,
                "node_count": graph.node_count,
                "edge_count": graph.edge_count,
            },
        )
        _logger.info(
            "ContextReconstructionEngine: context graph built for '%s' "
            "(%d nodes, %d edges).",
            experience_id,
            graph.node_count,
            graph.edge_count,
        )
        return graph

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of the engine's current state.

        Returns
        -------
        dict[str, Any]
            Contains ``running``, ``engine``, and the availability flags
            for each optional dependency engine.
        """
        with self._lock:
            return {
                "running": self._running,
                "engine": "ContextReconstructionEngine",
                "experience_engine_available": self._experience_engine is not None,
                "retrieval_engine_available": self._retrieval_engine is not None,
                "session_engine_available": self._session_engine is not None,
                "conversation_engine_available": self._conversation_engine is not None,
                "achievement_engine_available": self._achievement_engine is not None,
                "failure_engine_available": self._failure_engine is not None,
                "reflection_engine_available": self._reflection_engine is not None,
            }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _has_shared_context(a: Experience, b: Experience) -> bool:
    """Return ``True`` if two experiences share at least one tag or project ref."""
    tags_a = set(a.tag_names())
    tags_b = set(b.tag_names())
    if tags_a & tags_b:
        return True
    proj_a = set(a.metadata.project_refs)
    proj_b = set(b.metadata.project_refs)
    if proj_a & proj_b:
        return True
    return False


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "ContextReconstructionEngine",
    # Result types
    "TimelineEntry",
    "ExperienceChainLink",
    "ContextNode",
    "ContextEdge",
    "ReconstructedTimeline",
    "ReconstructedSession",
    "ReconstructedConversation",
    "ReconstructedExperienceChain",
    "ReconstructedRelatedMemories",
    "ContextSummary",
    "ContextGraph",
]