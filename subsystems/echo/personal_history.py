# subsystems/echo/personal_history.py
"""
ECHO v1 Personal History Engine.

Implements :class:`PersonalHistoryEngine` — the narrative layer of ECHO.

The Personal History Engine transforms episodic memories, reflections,
achievements, failures, patterns, and reconstructed contexts into a
coherent long-term life history.  It gives the POLARIS system a first-person,
time-ordered account of growth, struggle, and evolution — the subjective
experience of having lived through a sequence of meaningful events.

Core Responsibilities
---------------------
* **Life timeline generation**          — chronological assemblies of all
  significant experiences across the full recorded history.
* **Chapter creation**                  — segmentation of the timeline into
  named, bounded narrative arcs (e.g. "University", "First Job", "POLARIS Build").
* **Milestone tracking**                — identification and indexing of
  experiences that mark decisive turning points.
* **Growth trajectory reconstruction**  — longitudinal skill and capability
  arcs synthesised from achievements, reflections, and patterns.
* **Personal evolution analysis**       — cross-chapter comparison of
  importance distribution, failure rates, and achievement velocity.
* **Achievement history synthesis**     — ordered, domain-annotated achievement
  timelines with velocity and clustering metrics.
* **Failure history synthesis**         — domain-annotated failure timelines
  with lesson extraction status and recurrence detection.
* **Project history generation**        — per-project narrative bundles
  grouping all associated experiences under a single arc.
* **Historical narrative generation**   — prose-ready structured summaries
  (not free-form AI text) of any chapter, period, or milestone set.
* **Life phase detection**              — automatic segmentation of the
  recorded history into broad phases based on activity density and experience
  composition.
* **Continuity validation**             — detection of temporal gaps,
  orphaned chapters, and broken milestone references.
* **Narrative consistency checking**    — cross-chapter invariant checks:
  milestone ordering, chapter boundary overlap, orphaned experiences.
* **Historical summaries**              — concise digests of any time window.
* **Personal history reports**          — structured, auditable snapshots of
  the full history state.
* **Chapter indexing**                  — O(1) chapter lookup by ID, label,
  or tag.
* **Historical search**                 — experience-level full-text and
  metadata-predicate search within the history store.
* **History audit reports**             — structural consistency and coverage
  diagnostics for the entire history corpus.

Domain Objects (defined in this module)
----------------------------------------
* :class:`HistoryChapter`       — bounded narrative arc with ordered experiences.
* :class:`HistoricalPeriod`     — time-bounded window across one or many chapters.
* :class:`MilestoneRecord`      — annotated turning-point experience reference.
* :class:`PersonalNarrative`    — prose-ready structured summary of a history slice.
* :class:`GrowthTrajectory`     — longitudinal skill / domain growth arc.
* :class:`HistoricalSummary`    — concise digest of a time window or chapter.

Thread Safety
-------------
All public methods acquire ``self._lock`` (a :class:`threading.RLock`) before
reading or modifying internal state.  The engine is safe for concurrent use
across multiple POLARIS threads.

Lifecycle
---------
1. :meth:`initialize` — allocates internal structures and optionally performs
   an initial history build from the experience store.
2. :meth:`shutdown`   — releases all internal state and marks the engine stopped.

Any public method called outside the ``running`` state raises
:class:`~subsystems.echo.exceptions.EchoNotInitializedError`.

Architecture Notes
------------------
* v1 is a pure in-memory implementation.  All history state is derived on
  demand from the injected engine references and cached locally.
* The engine composes over ExperienceEngine, ReflectionEngine,
  PatternExtractionEngine, ContextReconstructionEngine, EpisodicIndexEngine,
  and MemoryIntegrityEngine — it owns none of their data.
* ECHO Boundary Law: the Personal History Engine reads from ECHO's experience
  store and emits structured narrative objects.  It does NOT own identity
  signals (ASTRA), goals (ODYSSEY), or knowledge (LUNA).
* Deterministic outputs: all scoring, ordering, and classification logic
  is deterministic.  Identical experience stores always produce identical
  history outputs.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from subsystems.echo.exceptions import (
    EchoNotInitializedError,
    ExperienceNotFoundError,
)
from subsystems.echo.interfaces import (
    ExperienceEngineInterface,
    ExperienceRetrievalEngineInterface,
)
from subsystems.echo.models import (
    Experience,
    ExperienceImportance,
    ExperienceType,
    MemoryTag,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum importance for an experience to appear in the life timeline
_TIMELINE_MIN_IMPORTANCE: ExperienceImportance = ExperienceImportance.MEDIUM

# Minimum importance for an experience to be considered a milestone candidate
_MILESTONE_MIN_IMPORTANCE: ExperienceImportance = ExperienceImportance.HIGH

# Minimum number of experiences required to form a chapter
_CHAPTER_MIN_EXPERIENCES: int = 1

# Number of days of inactivity that splits a life phase boundary
_PHASE_GAP_DAYS: float = 45.0

# Maximum depth for growth trajectory domain tracking
_MAX_GROWTH_DOMAINS: int = 50

# Velocity window: days used to calculate achievement and failure rates
_VELOCITY_WINDOW_DAYS: int = 30

# Narrative summary maximum experiences shown in prose
_NARRATIVE_MAX_ITEMS: int = 10

# Labels used for life phase classification
_PHASE_LABEL_ACTIVE: str = "active"
_PHASE_LABEL_TRANSITION: str = "transition"
_PHASE_LABEL_DORMANT: str = "dormant"


# ---------------------------------------------------------------------------
# Domain Objects
# ---------------------------------------------------------------------------


@dataclass
class HistoryChapter:
    """A bounded narrative arc within the personal history.

    A chapter groups a set of :class:`~subsystems.echo.models.Experience`
    objects under a human-readable label with an optional description.
    Chapters are the primary organisational unit of the Personal History
    Engine.  They may overlap in time but not in intent — each chapter
    captures a distinct thread of the personal story.

    Attributes
    ----------
    title:
        Human-readable label (e.g. ``"University Years"``,
        ``"POLARIS Build — Phase 1"``).
    chapter_id:
        UUID-4 unique identifier.
    description:
        Narrative context explaining what this chapter represents.
    experience_ids:
        Ordered list of experience UUIDs belonging to this chapter,
        sorted ascending by ``occurred_at``.
    milestone_ids:
        UUIDs of :class:`MilestoneRecord` objects associated with this
        chapter.
    tags:
        :class:`~subsystems.echo.models.MemoryTag` list for indexing.
    start_at:
        UTC timestamp of the earliest experience in the chapter.
        ``None`` if the chapter is empty.
    end_at:
        UTC timestamp of the latest experience in the chapter.
        ``None`` if the chapter has not yet closed.
    is_closed:
        ``True`` once the chapter has been explicitly closed and no new
        experiences will be added.
    created_at:
        UTC timestamp of chapter creation.
    updated_at:
        UTC timestamp of the most recent modification.
    metadata:
        Arbitrary key-value annotations for engine-specific use.
    """

    title: str
    chapter_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    experience_ids: list[str] = field(default_factory=list)
    milestone_ids: list[str] = field(default_factory=list)
    tags: list[MemoryTag] = field(default_factory=list)
    start_at: datetime | None = None
    end_at: datetime | None = None
    is_closed: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("HistoryChapter.title must be a non-empty string.")

    def tag_names(self) -> list[str]:
        """Return a flat list of tag name strings."""
        return [t.name for t in self.tags]

    def experience_count(self) -> int:
        """Return the number of experiences in this chapter."""
        return len(self.experience_ids)

    def duration_days(self) -> float | None:
        """Return the chapter duration in days, or ``None`` if undetermined."""
        if self.start_at is None:
            return None
        end = self.end_at or datetime.now(timezone.utc)
        return max(0.0, (end - self.start_at).total_seconds() / 86_400.0)


@dataclass
class HistoricalPeriod:
    """A time-bounded window that may span one or many chapters.

    :class:`HistoricalPeriod` objects provide a cross-cutting view of
    the history that is not constrained by chapter boundaries.  They are
    computed on demand by :meth:`PersonalHistoryEngine.get_period` and
    used as the input window for narrative generation and summary requests.

    Attributes
    ----------
    period_id:
        UUID-4 unique identifier.
    label:
        Human-readable label (e.g. ``"Q1 2024"``, ``"Year One"``).
    start_at:
        UTC start of the period (inclusive).
    end_at:
        UTC end of the period (inclusive).
    experience_ids:
        UUIDs of all experiences whose ``occurred_at`` falls within
        ``[start_at, end_at]``, sorted ascending.
    chapter_ids:
        UUIDs of chapters that intersect this period.
    milestone_ids:
        UUIDs of milestones that fall within this period.
    achievement_count:
        Number of ACHIEVEMENT-typed experiences in the period.
    failure_count:
        Number of FAILURE-typed experiences in the period.
    reflection_count:
        Number of REFLECTION-typed experiences in the period.
    dominant_domains:
        Top-three domains by experience count within this period.
    generated_at:
        UTC timestamp of when this period object was computed.
    """

    period_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    start_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    experience_ids: list[str] = field(default_factory=list)
    chapter_ids: list[str] = field(default_factory=list)
    milestone_ids: list[str] = field(default_factory=list)
    achievement_count: int = 0
    failure_count: int = 0
    reflection_count: int = 0
    dominant_domains: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def duration_days(self) -> float:
        """Return the period length in calendar days."""
        return max(0.0, (self.end_at - self.start_at).total_seconds() / 86_400.0)

    def experience_count(self) -> int:
        """Return the total number of experiences in this period."""
        return len(self.experience_ids)


@dataclass
class MilestoneRecord:
    """An annotated turning-point experience reference.

    Milestones are not separate experiences — they are a semantic overlay
    that annotates an existing :class:`~subsystems.echo.models.Experience`
    as a decisive turning point in the personal history.  They may be
    created automatically (by the engine's life-phase detection logic)
    or manually (by the caller via :meth:`PersonalHistoryEngine.add_milestone`).

    Attributes
    ----------
    milestone_id:
        UUID-4 unique identifier.
    experience_id:
        UUID of the :class:`~subsystems.echo.models.Experience` this
        milestone annotates.
    chapter_id:
        UUID of the :class:`HistoryChapter` this milestone belongs to,
        or ``None`` if not yet associated.
    label:
        Human-readable label for this milestone
        (e.g. ``"Shipped POLARIS v1"``).
    significance:
        Narrative significance description — what made this moment pivotal.
    milestone_type:
        Broad category: ``"achievement"``, ``"failure"``, ``"transition"``,
        ``"insight"``, ``"custom"``.
    occurred_at:
        UTC timestamp of the milestone moment (copied from the experience).
    is_automatic:
        ``True`` if this milestone was created by automatic phase detection;
        ``False`` if manually asserted by the caller.
    created_at:
        UTC timestamp of milestone record creation.
    metadata:
        Arbitrary key-value annotations.
    """

    experience_id: str
    label: str
    milestone_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chapter_id: str | None = None
    significance: str = ""
    milestone_type: str = "custom"
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_automatic: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    _VALID_TYPES = frozenset(
        {"achievement", "failure", "transition", "insight", "custom"}
    )

    def __post_init__(self) -> None:
        if not self.experience_id or not self.experience_id.strip():
            raise ValueError("MilestoneRecord.experience_id must be a non-empty string.")
        if not self.label or not self.label.strip():
            raise ValueError("MilestoneRecord.label must be a non-empty string.")
        if self.milestone_type not in self._VALID_TYPES:
            raise ValueError(
                f"MilestoneRecord.milestone_type must be one of "
                f"{sorted(self._VALID_TYPES)}; got {self.milestone_type!r}."
            )


@dataclass
class PersonalNarrative:
    """A prose-ready structured summary of a history slice.

    :class:`PersonalNarrative` objects are computed by
    :meth:`PersonalHistoryEngine.generate_narrative` and
    :meth:`PersonalHistoryEngine.generate_chapter_narrative`.  They
    provide all the structured data needed to render a human-readable
    account of a period, chapter, or the full history — without requiring
    free-form AI generation at retrieval time.

    Attributes
    ----------
    narrative_id:
        UUID-4 unique identifier.
    title:
        Short narrative title (e.g. ``"Year One: Building POLARIS"``).
    scope:
        ``"chapter"``, ``"period"``, ``"milestone_set"``, or ``"full"``.
    scope_ref:
        ID of the chapter, period, or milestone set being narrated.
    opening:
        One-sentence scene-setting statement for the period.
    key_events:
        Ordered list of (experience_id, title, occurred_at_iso, importance)
        tuples representing the most significant experiences.
    achievements_summary:
        Ordered list of achievement titles within the scope.
    failures_summary:
        Ordered list of failure titles within the scope.
    lessons:
        Extracted lesson strings from REFLECTION experiences in scope.
    dominant_themes:
        Top tag names or project references appearing in the scope.
    growth_indicators:
        Domain strings where ACHIEVEMENT density increased over the scope.
    closing:
        One-sentence closing statement summarising the outcome or legacy.
    experience_count:
        Total number of experiences included in this narrative.
    generated_at:
        UTC timestamp of narrative generation.
    metadata:
        Arbitrary key-value annotations.
    """

    title: str
    scope: str
    narrative_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope_ref: str = ""
    opening: str = ""
    key_events: list[dict[str, Any]] = field(default_factory=list)
    achievements_summary: list[str] = field(default_factory=list)
    failures_summary: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    dominant_themes: list[str] = field(default_factory=list)
    growth_indicators: list[str] = field(default_factory=list)
    closing: str = ""
    experience_count: int = 0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    _VALID_SCOPES = frozenset({"chapter", "period", "milestone_set", "full"})

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("PersonalNarrative.title must be a non-empty string.")
        if self.scope not in self._VALID_SCOPES:
            raise ValueError(
                f"PersonalNarrative.scope must be one of "
                f"{sorted(self._VALID_SCOPES)}; got {self.scope!r}."
            )


@dataclass
class GrowthTrajectory:
    """A longitudinal skill or domain growth arc.

    Growth trajectories are computed by
    :meth:`PersonalHistoryEngine.build_growth_trajectory`.  They aggregate
    ACHIEVEMENT-typed experiences within a domain to produce a time-ordered
    picture of capability accumulation and velocity.

    Attributes
    ----------
    trajectory_id:
        UUID-4 unique identifier.
    domain:
        The domain this trajectory tracks
        (e.g. ``"software"``, ``"academic"``, ``"leadership"``).
    achievement_ids:
        Ordered list of achievement experience UUIDs, ascending by
        ``occurred_at``.
    data_points:
        List of ``{"occurred_at": iso_str, "title": str,
        "importance": str, "cumulative_count": int}`` dicts, one per
        achievement, for graphing.
    total_achievements:
        Count of achievements in this domain.
    first_achievement_at:
        UTC timestamp of the earliest domain achievement, or ``None``.
    latest_achievement_at:
        UTC timestamp of the most recent domain achievement, or ``None``.
    velocity_per_30_days:
        Mean achievement count per 30-day window over the full trajectory
        duration.  ``0.0`` if fewer than two achievements exist.
    acceleration:
        Difference between the velocity in the second half of the
        trajectory and the first half.  Positive indicates acceleration.
    reflection_ids:
        UUIDs of REFLECTION experiences linked to this domain.
    generated_at:
        UTC timestamp of trajectory computation.
    """

    domain: str
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    achievement_ids: list[str] = field(default_factory=list)
    data_points: list[dict[str, Any]] = field(default_factory=list)
    total_achievements: int = 0
    first_achievement_at: datetime | None = None
    latest_achievement_at: datetime | None = None
    velocity_per_30_days: float = 0.0
    acceleration: float = 0.0
    reflection_ids: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.domain or not self.domain.strip():
            raise ValueError("GrowthTrajectory.domain must be a non-empty string.")

    def is_accelerating(self) -> bool:
        """Return ``True`` if the domain velocity is increasing over time."""
        return self.acceleration > 0.0


@dataclass
class HistoricalSummary:
    """A concise digest of a time window or chapter.

    :class:`HistoricalSummary` objects are lighter-weight than
    :class:`PersonalNarrative` — they provide counts, rates, and top items
    without the prose scaffolding.  They are produced by
    :meth:`PersonalHistoryEngine.summarise_window` and
    :meth:`PersonalHistoryEngine.summarise_chapter`.

    Attributes
    ----------
    summary_id:
        UUID-4 unique identifier.
    scope_label:
        Human-readable label for what was summarised.
    start_at:
        UTC start of the summarised window.
    end_at:
        UTC end of the summarised window (or ``datetime.now(utc)`` if open).
    total_experiences:
        Count of experiences in the window.
    achievement_count:
        Count of ACHIEVEMENT experiences.
    failure_count:
        Count of FAILURE experiences.
    reflection_count:
        Count of REFLECTION experiences.
    milestone_count:
        Count of milestones within the window.
    top_achievements:
        Titles of the top-three ACHIEVEMENT experiences by importance then
        recency.
    top_failures:
        Titles of the top-three FAILURE experiences by importance then
        recency.
    top_lessons:
        Outcome/description strings from the top-three REFLECTION
        experiences.
    dominant_tags:
        Up to five most-frequent tag names in the window.
    dominant_domains:
        Up to three most-frequent domain strings (from achievement/failure
        tags) in the window.
    integrity_issues:
        List of structural issues detected in the window (missing experiences,
        broken references, etc.).
    generated_at:
        UTC timestamp of summary generation.
    """

    scope_label: str
    start_at: datetime
    end_at: datetime
    summary_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_experiences: int = 0
    achievement_count: int = 0
    failure_count: int = 0
    reflection_count: int = 0
    milestone_count: int = 0
    top_achievements: list[str] = field(default_factory=list)
    top_failures: list[str] = field(default_factory=list)
    top_lessons: list[str] = field(default_factory=list)
    dominant_tags: list[str] = field(default_factory=list)
    dominant_domains: list[str] = field(default_factory=list)
    integrity_issues: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.scope_label or not self.scope_label.strip():
            raise ValueError("HistoricalSummary.scope_label must be a non-empty string.")


@dataclass
class HistoryAuditReport:
    """Structural consistency and coverage diagnostics for the history corpus.

    Produced by :meth:`PersonalHistoryEngine.generate_audit_report`.

    Attributes
    ----------
    report_id:
        UUID-4 unique identifier.
    total_chapters:
        Number of chapters registered in the engine.
    total_milestones:
        Number of milestones registered.
    total_indexed_experiences:
        Number of distinct experience IDs known to the history index.
    uncovered_experience_ids:
        Experience IDs present in the experience store at MEDIUM importance
        or above that belong to no chapter.
    orphaned_milestone_ids:
        Milestone IDs whose ``experience_id`` no longer exists in the store.
    broken_chapter_refs:
        Chapter IDs containing experience references that no longer exist.
    overlapping_chapter_pairs:
        Pairs of chapter IDs whose time windows overlap.
    temporal_gaps:
        List of ``{"from": iso_str, "to": iso_str, "gap_days": float}``
        dicts describing gaps in coverage of >= ``_PHASE_GAP_DAYS`` days.
    integrity_score:
        ``[0.0, 1.0]`` health metric. ``1.0`` = no issues detected.
    issues:
        Human-readable list of all detected issues.
    generated_at:
        UTC timestamp of report generation.
    """

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_chapters: int = 0
    total_milestones: int = 0
    total_indexed_experiences: int = 0
    uncovered_experience_ids: list[str] = field(default_factory=list)
    orphaned_milestone_ids: list[str] = field(default_factory=list)
    broken_chapter_refs: list[str] = field(default_factory=list)
    overlapping_chapter_pairs: list[tuple[str, str]] = field(default_factory=list)
    temporal_gaps: list[dict[str, Any]] = field(default_factory=list)
    integrity_score: float = 1.0
    issues: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Life Phase (internal)
# ---------------------------------------------------------------------------


@dataclass
class _LifePhase:
    """Internal representation of a detected life phase.

    Life phases are computed by :meth:`PersonalHistoryEngine._detect_life_phases`
    and exposed to callers via :meth:`PersonalHistoryEngine.get_life_phases`.
    They are not persisted across sessions.
    """

    phase_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = _PHASE_LABEL_ACTIVE
    start_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_at: datetime | None = None
    experience_ids: list[str] = field(default_factory=list)
    dominant_type: str = ""
    activity_density: float = 0.0  # experiences per 30 days


# ---------------------------------------------------------------------------
# PersonalHistoryEngine
# ---------------------------------------------------------------------------


class PersonalHistoryEngine:
    """Production implementation of the ECHO Personal History Engine.

    The Personal History Engine is the narrative layer of ECHO.  It transforms
    episodic memories, reflections, achievements, failures, patterns, and
    reconstructed contexts into a coherent long-term life history.

    Parameters
    ----------
    experience_engine:
        Running :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        providing full-store access and individual record fetches.
    retrieval_engine:
        Optional running
        :class:`~subsystems.echo.interfaces.ExperienceRetrievalEngineInterface`
        for semantically-scoped recall.  When ``None``, the engine falls back
        to direct experience store enumeration for all queries.
    reflection_engine:
        Optional running ``ReflectionEngine`` instance.  When supplied,
        REFLECTION-typed experiences are cross-referenced with generated
        insights during narrative construction.
    pattern_engine:
        Optional running ``PatternExtractionEngine`` instance.  When
        supplied, known patterns are woven into growth trajectories and
        chapter analysis.
    context_engine:
        Optional running ``ContextReconstructionEngine`` instance.  When
        supplied, causal chain data enriches milestone and period objects.
    episodic_index:
        Optional running ``EpisodicIndexEngine`` instance.  When supplied,
        tag and temporal lookups use the fast index rather than full scans.
    integrity_engine:
        Optional running ``MemoryIntegrityEngine`` instance.  When supplied,
        the audit report incorporates integrity scan results.

    Usage
    -----
    ::

        engine = PersonalHistoryEngine(experience_engine=exp_engine)
        engine.initialize()

        chapter = engine.create_chapter(
            title="POLARIS Build — Year One",
            description="The foundational year of system design and architecture.",
        )
        engine.assign_experience_to_chapter(chapter.chapter_id, experience_id)
        engine.add_milestone(
            experience_id=experience_id,
            label="Architecture Freeze",
            milestone_type="achievement",
        )

        narrative = engine.generate_chapter_narrative(chapter.chapter_id)
        engine.shutdown()
    """

    def __init__(
        self,
        experience_engine: ExperienceEngineInterface,
        *,
        retrieval_engine: Any | None = None,
        reflection_engine: Any | None = None,
        pattern_engine: Any | None = None,
        context_engine: Any | None = None,
        episodic_index: Any | None = None,
        integrity_engine: Any | None = None,
    ) -> None:
        if experience_engine is None:
            raise ValueError(
                "PersonalHistoryEngine requires a non-None experience_engine."
            )

        self._experience_engine = experience_engine
        self._retrieval_engine = retrieval_engine
        self._reflection_engine = reflection_engine
        self._pattern_engine = pattern_engine
        self._context_engine = context_engine
        self._episodic_index = episodic_index
        self._integrity_engine = integrity_engine

        # Primary stores
        self._chapters: dict[str, HistoryChapter] = {}
        self._milestones: dict[str, MilestoneRecord] = {}

        # Indices
        # experience_id → set[chapter_id]
        self._exp_to_chapters: dict[str, set[str]] = defaultdict(set)
        # chapter tag name → set[chapter_id]
        self._tag_to_chapters: dict[str, set[str]] = defaultdict(set)
        # experience_id → set[milestone_id]
        self._exp_to_milestones: dict[str, set[str]] = defaultdict(set)
        # chapter_id → set[milestone_id]
        self._chapter_to_milestones: dict[str, set[str]] = defaultdict(set)

        self._lock = threading.RLock()
        self._running = False

        _logger.debug("PersonalHistoryEngine constructed.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Must be called before any other method.  Calling ``initialize()``
        on an already-running engine is a no-op with a warning.

        Raises
        ------
        EchoError
            If initialisation fails for any reason.
        """
        with self._lock:
            if self._running:
                _logger.warning(
                    "PersonalHistoryEngine.initialize() called while already running."
                )
                return
            self._chapters.clear()
            self._milestones.clear()
            self._exp_to_chapters.clear()
            self._tag_to_chapters.clear()
            self._exp_to_milestones.clear()
            self._chapter_to_milestones.clear()
            self._running = True
            _logger.info("PersonalHistoryEngine initialised.")

    def shutdown(self) -> None:
        """Release all resources held by this engine.

        Clears all internal state and marks the engine as stopped.  Calling
        any public method after ``shutdown()`` will raise
        :class:`~subsystems.echo.exceptions.EchoNotInitializedError`.
        """
        with self._lock:
            if not self._running:
                return
            chapter_count = len(self._chapters)
            milestone_count = len(self._milestones)
            self._chapters.clear()
            self._milestones.clear()
            self._exp_to_chapters.clear()
            self._tag_to_chapters.clear()
            self._exp_to_milestones.clear()
            self._chapter_to_milestones.clear()
            self._running = False
            _logger.info(
                "PersonalHistoryEngine shut down. "
                "chapters=%d milestones=%d",
                chapter_count,
                milestone_count,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if the engine is not running."""
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _get_experience_or_raise(self, experience_id: str) -> Experience:
        """Fetch an experience from the store or raise ExperienceNotFoundError."""
        try:
            exp = self._experience_engine.get_experience(experience_id)
        except ExperienceNotFoundError:
            raise
        except Exception as exc:
            raise ExperienceNotFoundError(experience_id) from exc
        if exp is None:
            raise ExperienceNotFoundError(experience_id)
        return exp

    def _touch_chapter(self, chapter: HistoryChapter) -> None:
        """Update the chapter's ``updated_at`` timestamp in place."""
        chapter.updated_at = datetime.now(timezone.utc)

    def _recalculate_chapter_bounds(self, chapter: HistoryChapter) -> None:
        """Recompute ``start_at`` and ``end_at`` from the chapter's experiences."""
        if not chapter.experience_ids:
            chapter.start_at = None
            chapter.end_at = None
            return

        timestamps: list[datetime] = []
        for eid in chapter.experience_ids:
            try:
                exp = self._experience_engine.get_experience(eid)
                timestamps.append(exp.occurred_at)
            except Exception:
                pass  # broken references are reported by the audit, not here

        if not timestamps:
            chapter.start_at = None
            chapter.end_at = None
            return

        chapter.start_at = min(timestamps)
        if not chapter.is_closed:
            chapter.end_at = max(timestamps)
        else:
            chapter.end_at = chapter.end_at or max(timestamps)

    def _all_experiences_sorted(
        self,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
    ) -> list[Experience]:
        """Return all experiences at or above ``min_importance``, sorted ascending.

        Uses the experience engine's query interface.
        """
        try:
            exps = self._experience_engine.query_experiences(
                min_importance=min_importance,
                limit=100_000,
                offset=0,
            )
        except Exception:
            exps = []

        exps.sort(key=lambda e: e.occurred_at)
        return exps

    def _experiences_for_ids(self, ids: list[str]) -> list[Experience]:
        """Fetch experiences for a list of IDs, skipping missing ones."""
        result: list[Experience] = []
        for eid in ids:
            try:
                result.append(self._experience_engine.get_experience(eid))
            except Exception:
                pass
        return result

    @staticmethod
    def _importance_rank(exp: Experience) -> int:
        """Return a sortable integer rank from an experience's importance."""
        return exp.importance.value

    def _top_tag_names(
        self, experiences: list[Experience], top_n: int = 5
    ) -> list[str]:
        """Return the top-N most frequent tag names across a list of experiences."""
        counter: dict[str, int] = defaultdict(int)
        for exp in experiences:
            for name in exp.tag_names():
                counter[name] += 1
        return [
            name
            for name, _ in sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        ][:top_n]

    def _top_domains(
        self,
        experiences: list[Experience],
        top_n: int = 3,
    ) -> list[str]:
        """Return the top-N domain strings from tags of category ``'project'``
        or ``'topic'``.
        """
        counter: dict[str, int] = defaultdict(int)
        for exp in experiences:
            for tag in exp.tags:
                if tag.category in ("project", "topic"):
                    counter[tag.name] += 1
        return [
            name
            for name, _ in sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        ][:top_n]

    def _velocity(
        self, timestamps: list[datetime], window_days: int = _VELOCITY_WINDOW_DAYS
    ) -> float:
        """Compute mean count per ``window_days``-day window across ``timestamps``."""
        if len(timestamps) < 2:
            return 0.0
        span = (max(timestamps) - min(timestamps)).total_seconds() / 86_400.0
        if span < 1.0:
            return float(len(timestamps))
        return len(timestamps) / span * window_days

    def _half_velocity(self, timestamps: list[datetime]) -> tuple[float, float]:
        """Split timestamps at the median and return (first_half_v, second_half_v)."""
        if len(timestamps) < 2:
            return 0.0, 0.0
        mid = len(timestamps) // 2
        first = timestamps[:mid]
        second = timestamps[mid:]
        return self._velocity(first), self._velocity(second)

    # ------------------------------------------------------------------
    # Chapter Management
    # ------------------------------------------------------------------

    def create_chapter(
        self,
        title: str,
        *,
        description: str = "",
        tags: list[MemoryTag] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HistoryChapter:
        """Create and register a new :class:`HistoryChapter`.

        Parameters
        ----------
        title:
            Human-readable chapter label.  Must be non-empty.
        description:
            Narrative context for this chapter.
        tags:
            Optional :class:`~subsystems.echo.models.MemoryTag` list for
            indexing and search.
        metadata:
            Optional key-value annotations.

        Returns
        -------
        HistoryChapter
            The newly created and registered chapter.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        ValueError
            If ``title`` is empty or blank.
        """
        self._assert_running("create_chapter")

        if not title or not title.strip():
            raise ValueError("Chapter title must be a non-empty string.")

        chapter = HistoryChapter(
            title=title.strip(),
            description=description,
            tags=list(tags) if tags else [],
            metadata=dict(metadata) if metadata else {},
        )

        with self._lock:
            self._chapters[chapter.chapter_id] = chapter
            for tag in chapter.tags:
                self._tag_to_chapters[tag.name].add(chapter.chapter_id)

        _logger.info(
            "PersonalHistoryEngine: chapter created id=%s title=%r",
            chapter.chapter_id,
            chapter.title,
        )
        return chapter

    def get_chapter(self, chapter_id: str) -> HistoryChapter:
        """Return the :class:`HistoryChapter` with the given ID.

        Parameters
        ----------
        chapter_id:
            UUID of the chapter.

        Returns
        -------
        HistoryChapter
            The requested chapter.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no chapter with ``chapter_id`` exists.
        """
        self._assert_running("get_chapter")

        with self._lock:
            if chapter_id not in self._chapters:
                raise KeyError(f"No chapter with id '{chapter_id}' found.")
            return self._chapters[chapter_id]

    def list_chapters(
        self,
        *,
        include_closed: bool = True,
        tag_filter: str | None = None,
    ) -> list[HistoryChapter]:
        """Return all registered chapters, ordered by ``start_at`` ascending.

        Parameters
        ----------
        include_closed:
            If ``False``, only return chapters that are not yet closed.
        tag_filter:
            If supplied, return only chapters whose tag set includes this
            tag name.

        Returns
        -------
        list[HistoryChapter]
            Matching chapters sorted by ``start_at`` ascending, with
            chapters that have no ``start_at`` placed last.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("list_chapters")

        with self._lock:
            chapters = list(self._chapters.values())

        if not include_closed:
            chapters = [c for c in chapters if not c.is_closed]

        if tag_filter is not None:
            chapters = [c for c in chapters if tag_filter in c.tag_names()]

        chapters.sort(
            key=lambda c: (
                c.start_at is None,
                c.start_at.timestamp() if c.start_at else 0.0,
            )
        )
        return chapters

    def update_chapter(
        self,
        chapter_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        tags: list[MemoryTag] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HistoryChapter:
        """Update mutable fields of an existing :class:`HistoryChapter`.

        Only the fields explicitly supplied (non-``None``) are updated.
        ``chapter_id``, ``created_at``, ``experience_ids``, and
        ``milestone_ids`` are immutable through this method.

        Parameters
        ----------
        chapter_id:
            UUID of the chapter to update.
        title:
            New title, if provided.
        description:
            New description, if provided.
        tags:
            Replacement tag list, if provided.
        metadata:
            Replacement metadata dict, if provided.

        Returns
        -------
        HistoryChapter
            The updated chapter.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no chapter with ``chapter_id`` exists.
        ValueError
            If ``title`` is provided but empty.
        """
        self._assert_running("update_chapter")

        with self._lock:
            if chapter_id not in self._chapters:
                raise KeyError(f"No chapter with id '{chapter_id}' found.")

            chapter = self._chapters[chapter_id]

            if title is not None:
                if not title.strip():
                    raise ValueError("Chapter title must be a non-empty string.")
                chapter.title = title.strip()

            if description is not None:
                chapter.description = description

            if tags is not None:
                # Remove old tag index entries
                for old_tag in chapter.tags:
                    self._tag_to_chapters[old_tag.name].discard(chapter_id)
                chapter.tags = list(tags)
                # Add new tag index entries
                for new_tag in chapter.tags:
                    self._tag_to_chapters[new_tag.name].add(chapter_id)

            if metadata is not None:
                chapter.metadata = dict(metadata)

            self._touch_chapter(chapter)

        _logger.debug(
            "PersonalHistoryEngine: chapter updated id=%s", chapter_id
        )
        return chapter

    def close_chapter(
        self,
        chapter_id: str,
        *,
        closed_at: datetime | None = None,
    ) -> HistoryChapter:
        """Mark a chapter as closed, freezing its end boundary.

        A closed chapter will not have new experiences assigned to it via
        :meth:`assign_experience_to_chapter`.  The ``end_at`` field is set
        to ``closed_at`` (or ``datetime.now(utc)`` if not supplied).

        Parameters
        ----------
        chapter_id:
            UUID of the chapter to close.
        closed_at:
            UTC timestamp for the chapter's end.  Defaults to now.

        Returns
        -------
        HistoryChapter
            The closed chapter.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no chapter with ``chapter_id`` exists.
        """
        self._assert_running("close_chapter")

        with self._lock:
            if chapter_id not in self._chapters:
                raise KeyError(f"No chapter with id '{chapter_id}' found.")

            chapter = self._chapters[chapter_id]
            chapter.is_closed = True
            chapter.end_at = closed_at or datetime.now(timezone.utc)
            self._touch_chapter(chapter)

        _logger.info(
            "PersonalHistoryEngine: chapter closed id=%s end_at=%s",
            chapter_id,
            chapter.end_at.isoformat(),
        )
        return chapter

    def delete_chapter(self, chapter_id: str) -> None:
        """Remove a chapter from the engine.

        Experiences and milestones are NOT deleted — only the chapter record
        and its index entries are removed.

        Parameters
        ----------
        chapter_id:
            UUID of the chapter to delete.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no chapter with ``chapter_id`` exists.
        """
        self._assert_running("delete_chapter")

        with self._lock:
            if chapter_id not in self._chapters:
                raise KeyError(f"No chapter with id '{chapter_id}' found.")

            chapter = self._chapters.pop(chapter_id)

            # Remove tag index entries
            for tag in chapter.tags:
                self._tag_to_chapters[tag.name].discard(chapter_id)

            # Remove experience → chapter mappings
            for eid in chapter.experience_ids:
                self._exp_to_chapters[eid].discard(chapter_id)

            # Remove chapter → milestone reverse index
            self._chapter_to_milestones.pop(chapter_id, None)

            # Detach milestones from this chapter (do not delete them)
            for mid in list(chapter.milestone_ids):
                if mid in self._milestones:
                    self._milestones[mid].chapter_id = None

        _logger.info(
            "PersonalHistoryEngine: chapter deleted id=%s title=%r",
            chapter_id,
            chapter.title,
        )

    # ------------------------------------------------------------------
    # Experience Assignment
    # ------------------------------------------------------------------

    def assign_experience_to_chapter(
        self,
        chapter_id: str,
        experience_id: str,
    ) -> HistoryChapter:
        """Add an experience to a chapter's ordered list.

        The experience must exist in the experience store.  Assignment is
        idempotent — assigning the same experience to the same chapter twice
        is a no-op.

        Parameters
        ----------
        chapter_id:
            UUID of the target chapter.
        experience_id:
            UUID of the experience to assign.

        Returns
        -------
        HistoryChapter
            The updated chapter with recomputed time bounds.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no chapter with ``chapter_id`` exists.
        ExperienceNotFoundError
            If the experience does not exist in the store.
        ValueError
            If the chapter is closed.
        """
        self._assert_running("assign_experience_to_chapter")

        with self._lock:
            if chapter_id not in self._chapters:
                raise KeyError(f"No chapter with id '{chapter_id}' found.")

            chapter = self._chapters[chapter_id]

            if chapter.is_closed:
                raise ValueError(
                    f"Chapter '{chapter_id}' is closed. "
                    "Cannot assign new experiences to a closed chapter."
                )

            # Validate existence — will raise ExperienceNotFoundError if absent
            self._get_experience_or_raise(experience_id)

            if experience_id not in chapter.experience_ids:
                chapter.experience_ids.append(experience_id)
                self._exp_to_chapters[experience_id].add(chapter_id)
                self._recalculate_chapter_bounds(chapter)
                self._touch_chapter(chapter)
                _logger.debug(
                    "PersonalHistoryEngine: assigned exp=%s to chapter=%s",
                    experience_id,
                    chapter_id,
                )

        return chapter

    def unassign_experience_from_chapter(
        self,
        chapter_id: str,
        experience_id: str,
    ) -> HistoryChapter:
        """Remove an experience from a chapter.

        If the experience is not in the chapter, this is a no-op.

        Parameters
        ----------
        chapter_id:
            UUID of the target chapter.
        experience_id:
            UUID of the experience to remove.

        Returns
        -------
        HistoryChapter
            The updated chapter with recomputed time bounds.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no chapter with ``chapter_id`` exists.
        """
        self._assert_running("unassign_experience_from_chapter")

        with self._lock:
            if chapter_id not in self._chapters:
                raise KeyError(f"No chapter with id '{chapter_id}' found.")

            chapter = self._chapters[chapter_id]

            if experience_id in chapter.experience_ids:
                chapter.experience_ids.remove(experience_id)
                self._exp_to_chapters[experience_id].discard(chapter_id)
                self._recalculate_chapter_bounds(chapter)
                self._touch_chapter(chapter)

        return chapter

    def get_chapters_for_experience(self, experience_id: str) -> list[HistoryChapter]:
        """Return all chapters that contain the given experience.

        Parameters
        ----------
        experience_id:
            UUID of the experience.

        Returns
        -------
        list[HistoryChapter]
            Chapters containing the experience, sorted by ``start_at``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("get_chapters_for_experience")

        with self._lock:
            chapter_ids = list(self._exp_to_chapters.get(experience_id, set()))
            chapters = [
                self._chapters[cid]
                for cid in chapter_ids
                if cid in self._chapters
            ]

        chapters.sort(
            key=lambda c: (
                c.start_at is None,
                c.start_at.timestamp() if c.start_at else 0.0,
            )
        )
        return chapters

    # ------------------------------------------------------------------
    # Milestone Management
    # ------------------------------------------------------------------

    def add_milestone(
        self,
        experience_id: str,
        label: str,
        *,
        chapter_id: str | None = None,
        significance: str = "",
        milestone_type: str = "custom",
        metadata: dict[str, Any] | None = None,
    ) -> MilestoneRecord:
        """Annotate an experience as a milestone in the personal history.

        Parameters
        ----------
        experience_id:
            UUID of the experience to mark as a milestone.
        label:
            Human-readable label for this milestone.
        chapter_id:
            UUID of the chapter this milestone belongs to.  If supplied,
            the milestone is also registered in the chapter's
            ``milestone_ids`` list.
        significance:
            Narrative description of what made this moment pivotal.
        milestone_type:
            One of ``"achievement"``, ``"failure"``, ``"transition"``,
            ``"insight"``, ``"custom"``.
        metadata:
            Optional key-value annotations.

        Returns
        -------
        MilestoneRecord
            The created milestone.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        ExperienceNotFoundError
            If the experience does not exist in the store.
        KeyError
            If ``chapter_id`` is supplied but no such chapter exists.
        """
        self._assert_running("add_milestone")

        exp = self._get_experience_or_raise(experience_id)

        with self._lock:
            if chapter_id is not None and chapter_id not in self._chapters:
                raise KeyError(f"No chapter with id '{chapter_id}' found.")

            milestone = MilestoneRecord(
                experience_id=experience_id,
                label=label,
                chapter_id=chapter_id,
                significance=significance,
                milestone_type=milestone_type,
                occurred_at=exp.occurred_at,
                is_automatic=False,
                metadata=dict(metadata) if metadata else {},
            )

            self._milestones[milestone.milestone_id] = milestone
            self._exp_to_milestones[experience_id].add(milestone.milestone_id)

            if chapter_id is not None:
                chapter = self._chapters[chapter_id]
                if milestone.milestone_id not in chapter.milestone_ids:
                    chapter.milestone_ids.append(milestone.milestone_id)
                self._chapter_to_milestones[chapter_id].add(milestone.milestone_id)
                self._touch_chapter(chapter)

        _logger.info(
            "PersonalHistoryEngine: milestone added id=%s exp=%s label=%r",
            milestone.milestone_id,
            experience_id,
            label,
        )
        return milestone

    def get_milestone(self, milestone_id: str) -> MilestoneRecord:
        """Return the :class:`MilestoneRecord` with the given ID.

        Parameters
        ----------
        milestone_id:
            UUID of the milestone.

        Returns
        -------
        MilestoneRecord
            The requested milestone.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no milestone with ``milestone_id`` exists.
        """
        self._assert_running("get_milestone")

        with self._lock:
            if milestone_id not in self._milestones:
                raise KeyError(f"No milestone with id '{milestone_id}' found.")
            return self._milestones[milestone_id]

    def list_milestones(
        self,
        *,
        chapter_id: str | None = None,
        milestone_type: str | None = None,
    ) -> list[MilestoneRecord]:
        """Return all milestones, optionally filtered by chapter or type.

        Parameters
        ----------
        chapter_id:
            If supplied, return only milestones belonging to this chapter.
        milestone_type:
            If supplied, return only milestones of this type.

        Returns
        -------
        list[MilestoneRecord]
            Matching milestones sorted by ``occurred_at`` ascending.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("list_milestones")

        with self._lock:
            milestones = list(self._milestones.values())

        if chapter_id is not None:
            milestones = [m for m in milestones if m.chapter_id == chapter_id]

        if milestone_type is not None:
            milestones = [m for m in milestones if m.milestone_type == milestone_type]

        milestones.sort(key=lambda m: m.occurred_at)
        return milestones

    def delete_milestone(self, milestone_id: str) -> None:
        """Remove a milestone record.

        Parameters
        ----------
        milestone_id:
            UUID of the milestone to delete.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no milestone with ``milestone_id`` exists.
        """
        self._assert_running("delete_milestone")

        with self._lock:
            if milestone_id not in self._milestones:
                raise KeyError(f"No milestone with id '{milestone_id}' found.")

            milestone = self._milestones.pop(milestone_id)

            # Remove from exp → milestone index
            self._exp_to_milestones[milestone.experience_id].discard(milestone_id)

            # Remove from chapter
            if milestone.chapter_id and milestone.chapter_id in self._chapters:
                chapter = self._chapters[milestone.chapter_id]
                if milestone_id in chapter.milestone_ids:
                    chapter.milestone_ids.remove(milestone_id)
                self._chapter_to_milestones[milestone.chapter_id].discard(milestone_id)
                self._touch_chapter(chapter)

        _logger.debug(
            "PersonalHistoryEngine: milestone deleted id=%s", milestone_id
        )

    # ------------------------------------------------------------------
    # Life Timeline
    # ------------------------------------------------------------------

    def generate_life_timeline(
        self,
        *,
        min_importance: ExperienceImportance = _TIMELINE_MIN_IMPORTANCE,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return a chronological life timeline of significant experiences.

        Each entry in the returned list is a dict with the keys:
        ``experience_id``, ``title``, ``experience_type``, ``importance``,
        ``occurred_at`` (ISO-8601), ``outcome``, ``tags``, ``chapter_ids``,
        ``milestone_labels``, and ``significance_score``.

        Parameters
        ----------
        min_importance:
            Minimum :class:`~subsystems.echo.models.ExperienceImportance`
            tier.  Defaults to MEDIUM.
        start_at:
            If supplied, exclude experiences before this UTC timestamp.
        end_at:
            If supplied, exclude experiences after this UTC timestamp.
        limit:
            Maximum number of timeline entries to return.

        Returns
        -------
        list[dict[str, Any]]
            Timeline entries sorted ascending by ``occurred_at``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("generate_life_timeline")

        exps = self._all_experiences_sorted(min_importance=min_importance)

        if start_at is not None:
            exps = [e for e in exps if e.occurred_at >= start_at]
        if end_at is not None:
            exps = [e for e in exps if e.occurred_at <= end_at]

        exps = exps[:limit]

        with self._lock:
            exp_to_chapters_snapshot = {
                eid: list(cids) for eid, cids in self._exp_to_chapters.items()
            }
            exp_to_milestones_snapshot = {
                eid: list(mids) for eid, mids in self._exp_to_milestones.items()
            }
            milestones_snapshot = dict(self._milestones)

        entries: list[dict[str, Any]] = []
        for exp in exps:
            chapter_ids = exp_to_chapters_snapshot.get(exp.experience_id, [])
            milestone_ids = exp_to_milestones_snapshot.get(exp.experience_id, [])
            milestone_labels = [
                milestones_snapshot[mid].label
                for mid in milestone_ids
                if mid in milestones_snapshot
            ]
            entries.append(
                {
                    "experience_id": exp.experience_id,
                    "title": exp.title,
                    "experience_type": exp.experience_type.name,
                    "importance": exp.importance.name,
                    "occurred_at": exp.occurred_at.isoformat(),
                    "outcome": exp.outcome,
                    "tags": exp.tag_names(),
                    "chapter_ids": chapter_ids,
                    "milestone_labels": milestone_labels,
                    "significance_score": exp.metadata.significance_score,
                }
            )

        return entries

    # ------------------------------------------------------------------
    # Historical Period
    # ------------------------------------------------------------------

    def get_period(
        self,
        start_at: datetime,
        end_at: datetime,
        *,
        label: str = "",
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
    ) -> HistoricalPeriod:
        """Compute a :class:`HistoricalPeriod` for the given time window.

        Parameters
        ----------
        start_at:
            UTC start of the period (inclusive).
        end_at:
            UTC end of the period (inclusive).
        label:
            Optional human-readable label.
        min_importance:
            Minimum importance filter for included experiences.

        Returns
        -------
        HistoricalPeriod
            Populated period object.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        ValueError
            If ``end_at`` is before ``start_at``.
        """
        self._assert_running("get_period")

        if end_at < start_at:
            raise ValueError(
                f"end_at ({end_at.isoformat()}) must not precede "
                f"start_at ({start_at.isoformat()})."
            )

        exps = self._all_experiences_sorted(min_importance=min_importance)
        exps_in_window = [
            e for e in exps if start_at <= e.occurred_at <= end_at
        ]

        with self._lock:
            chapters_snapshot = list(self._chapters.values())
            milestones_snapshot = list(self._milestones.values())

        exp_ids = [e.experience_id for e in exps_in_window]
        exp_id_set = set(exp_ids)

        # Chapters that intersect the window
        intersecting_chapter_ids: list[str] = []
        for ch in chapters_snapshot:
            if ch.start_at is None:
                continue
            ch_end = ch.end_at or datetime.now(timezone.utc)
            if ch.start_at <= end_at and ch_end >= start_at:
                intersecting_chapter_ids.append(ch.chapter_id)

        # Milestones in the window
        milestone_ids = [
            m.milestone_id
            for m in milestones_snapshot
            if start_at <= m.occurred_at <= end_at
        ]

        # Type counts
        achievement_count = sum(
            1 for e in exps_in_window
            if e.experience_type == ExperienceType.ACHIEVEMENT
        )
        failure_count = sum(
            1 for e in exps_in_window
            if e.experience_type == ExperienceType.FAILURE
        )
        reflection_count = sum(
            1 for e in exps_in_window
            if e.experience_type == ExperienceType.REFLECTION
        )

        dominant_domains = self._top_domains(exps_in_window, top_n=3)

        return HistoricalPeriod(
            label=label or f"{start_at.date().isoformat()} – {end_at.date().isoformat()}",
            start_at=start_at,
            end_at=end_at,
            experience_ids=exp_ids,
            chapter_ids=intersecting_chapter_ids,
            milestone_ids=milestone_ids,
            achievement_count=achievement_count,
            failure_count=failure_count,
            reflection_count=reflection_count,
            dominant_domains=dominant_domains,
        )

    # ------------------------------------------------------------------
    # Narrative Generation
    # ------------------------------------------------------------------

    def generate_chapter_narrative(
        self, chapter_id: str
    ) -> PersonalNarrative:
        """Generate a :class:`PersonalNarrative` for a chapter.

        Parameters
        ----------
        chapter_id:
            UUID of the chapter to narrate.

        Returns
        -------
        PersonalNarrative
            Structured narrative for the chapter.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no chapter with ``chapter_id`` exists.
        """
        self._assert_running("generate_chapter_narrative")

        with self._lock:
            if chapter_id not in self._chapters:
                raise KeyError(f"No chapter with id '{chapter_id}' found.")
            chapter = self._chapters[chapter_id]
            exp_ids = list(chapter.experience_ids)

        return self._build_narrative(
            title=chapter.title,
            scope="chapter",
            scope_ref=chapter_id,
            experience_ids=exp_ids,
        )

    def generate_narrative(
        self,
        start_at: datetime,
        end_at: datetime,
        *,
        title: str = "",
        min_importance: ExperienceImportance = ExperienceImportance.MEDIUM,
    ) -> PersonalNarrative:
        """Generate a :class:`PersonalNarrative` for an arbitrary time window.

        Parameters
        ----------
        start_at:
            UTC start of the narrative window.
        end_at:
            UTC end of the narrative window.
        title:
            Optional title for the narrative.  Defaults to the date range.
        min_importance:
            Minimum importance for included experiences.

        Returns
        -------
        PersonalNarrative
            Structured narrative for the period.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        ValueError
            If ``end_at`` is before ``start_at``.
        """
        self._assert_running("generate_narrative")

        if end_at < start_at:
            raise ValueError(
                f"end_at ({end_at.isoformat()}) must not precede "
                f"start_at ({start_at.isoformat()})."
            )

        exps = self._all_experiences_sorted(min_importance=min_importance)
        exps_in_window = [e for e in exps if start_at <= e.occurred_at <= end_at]

        default_title = (
            title.strip()
            or f"{start_at.date().isoformat()} – {end_at.date().isoformat()}"
        )
        return self._build_narrative(
            title=default_title,
            scope="period",
            scope_ref="",
            experience_ids=[e.experience_id for e in exps_in_window],
        )

    def generate_full_narrative(self) -> PersonalNarrative:
        """Generate a :class:`PersonalNarrative` spanning the full recorded history.

        Returns
        -------
        PersonalNarrative
            Full-history structured narrative.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("generate_full_narrative")

        exps = self._all_experiences_sorted(
            min_importance=_TIMELINE_MIN_IMPORTANCE
        )
        return self._build_narrative(
            title="Full Personal History",
            scope="full",
            scope_ref="",
            experience_ids=[e.experience_id for e in exps],
        )

    def _build_narrative(
        self,
        title: str,
        scope: str,
        scope_ref: str,
        experience_ids: list[str],
    ) -> PersonalNarrative:
        """Internal helper: construct a :class:`PersonalNarrative` from IDs."""
        exps = self._experiences_for_ids(experience_ids)
        exps.sort(key=lambda e: e.occurred_at)

        achievements = [
            e for e in exps if e.experience_type == ExperienceType.ACHIEVEMENT
        ]
        failures = [
            e for e in exps if e.experience_type == ExperienceType.FAILURE
        ]
        reflections = [
            e for e in exps if e.experience_type == ExperienceType.REFLECTION
        ]

        # Key events: top NARRATIVE_MAX_ITEMS by importance then recency
        key_exp = sorted(
            exps,
            key=lambda e: (e.importance.value, e.occurred_at.timestamp()),
            reverse=True,
        )[:_NARRATIVE_MAX_ITEMS]
        key_events: list[dict[str, Any]] = [
            {
                "experience_id": e.experience_id,
                "title": e.title,
                "occurred_at": e.occurred_at.isoformat(),
                "importance": e.importance.name,
                "experience_type": e.experience_type.name,
            }
            for e in sorted(key_exp, key=lambda e: e.occurred_at)
        ]

        achievements_summary = [e.title for e in achievements]
        failures_summary = [e.title for e in failures]
        lessons = [
            e.outcome
            for e in reflections
            if e.outcome.strip()
        ] + [
            e.description
            for e in reflections
            if not e.outcome.strip() and e.description.strip()
        ]

        dominant_themes = self._top_tag_names(exps, top_n=5)
        growth_indicators = self._top_domains(achievements, top_n=3)

        # Opening: describe scope, time range, and dominant theme
        if exps:
            first_ts = exps[0].occurred_at.date().isoformat()
            last_ts = exps[-1].occurred_at.date().isoformat()
            opening = (
                f"This {scope} spans {len(exps)} experience(s) from "
                f"{first_ts} to {last_ts}, covering "
                f"{', '.join(dominant_themes[:2]) or 'various themes'}."
            )
        else:
            opening = f"This {scope} contains no recorded experiences."

        # Closing
        if achievements and failures:
            closing = (
                f"Overall: {len(achievements)} achievement(s) and "
                f"{len(failures)} failure(s) recorded, "
                f"with {len(reflections)} reflection(s) generated."
            )
        elif achievements:
            closing = (
                f"Overall: {len(achievements)} achievement(s) recorded "
                f"with {len(reflections)} reflection(s) generated."
            )
        elif failures:
            closing = (
                f"Overall: {len(failures)} failure(s) recorded "
                f"with {len(reflections)} reflection(s) generated."
            )
        else:
            closing = (
                f"Overall: {len(exps)} experience(s) recorded "
                f"with {len(reflections)} reflection(s) generated."
            )

        return PersonalNarrative(
            title=title,
            scope=scope,
            scope_ref=scope_ref,
            opening=opening,
            key_events=key_events,
            achievements_summary=achievements_summary,
            failures_summary=failures_summary,
            lessons=lessons[:_NARRATIVE_MAX_ITEMS],
            dominant_themes=dominant_themes,
            growth_indicators=growth_indicators,
            closing=closing,
            experience_count=len(exps),
        )

    # ------------------------------------------------------------------
    # Growth Trajectory
    # ------------------------------------------------------------------

    def build_growth_trajectory(self, domain: str) -> GrowthTrajectory:
        """Compute a :class:`GrowthTrajectory` for the given domain.

        Scans all ACHIEVEMENT-typed experiences whose tags include the
        domain (by tag name match or tag category ``"project"``/``"topic"``
        and name match) and assembles a time-ordered capability arc.

        Parameters
        ----------
        domain:
            The domain label to build a trajectory for
            (e.g. ``"software"``, ``"academic"``).  Must be non-empty.

        Returns
        -------
        GrowthTrajectory
            Populated trajectory object.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        ValueError
            If ``domain`` is empty or blank.
        """
        self._assert_running("build_growth_trajectory")

        if not domain or not domain.strip():
            raise ValueError("domain must be a non-empty string.")

        domain = domain.strip().lower()

        exps = self._all_experiences_sorted(min_importance=ExperienceImportance.LOW)

        # Filter to ACHIEVEMENT experiences matching the domain
        domain_achievements: list[Experience] = []
        for exp in exps:
            if exp.experience_type != ExperienceType.ACHIEVEMENT:
                continue
            tag_names_lower = [t.lower() for t in exp.tag_names()]
            if domain in tag_names_lower:
                domain_achievements.append(exp)
                continue
            for tag in exp.tags:
                if tag.category in ("project", "topic") and tag.name.lower() == domain:
                    domain_achievements.append(exp)
                    break

        # Reflections mentioning the domain (by tag)
        reflection_ids: list[str] = [
            exp.experience_id
            for exp in exps
            if exp.experience_type == ExperienceType.REFLECTION
            and domain in [t.lower() for t in exp.tag_names()]
        ]

        timestamps = [e.occurred_at for e in domain_achievements]
        velocity = self._velocity(timestamps, window_days=_VELOCITY_WINDOW_DAYS)
        first_v, second_v = self._half_velocity(timestamps)
        acceleration = second_v - first_v

        data_points: list[dict[str, Any]] = []
        for idx, exp in enumerate(domain_achievements, start=1):
            data_points.append(
                {
                    "occurred_at": exp.occurred_at.isoformat(),
                    "title": exp.title,
                    "importance": exp.importance.name,
                    "cumulative_count": idx,
                }
            )

        return GrowthTrajectory(
            domain=domain,
            achievement_ids=[e.experience_id for e in domain_achievements],
            data_points=data_points,
            total_achievements=len(domain_achievements),
            first_achievement_at=min(timestamps) if timestamps else None,
            latest_achievement_at=max(timestamps) if timestamps else None,
            velocity_per_30_days=velocity,
            acceleration=acceleration,
            reflection_ids=reflection_ids,
        )

    def build_all_growth_trajectories(self) -> list[GrowthTrajectory]:
        """Build :class:`GrowthTrajectory` objects for every known domain.

        Domains are discovered by enumerating the ``project`` and ``topic``
        tags across all ACHIEVEMENT-typed experiences.

        Returns
        -------
        list[GrowthTrajectory]
            One trajectory per discovered domain, sorted descending by
            ``total_achievements``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("build_all_growth_trajectories")

        exps = self._all_experiences_sorted(min_importance=ExperienceImportance.LOW)
        domains: set[str] = set()
        for exp in exps:
            if exp.experience_type != ExperienceType.ACHIEVEMENT:
                continue
            for tag in exp.tags:
                if tag.category in ("project", "topic"):
                    domains.add(tag.name.lower())
            for name in exp.tag_names():
                domains.add(name.lower())

        domains = set(list(domains)[:_MAX_GROWTH_DOMAINS])

        trajectories = [self.build_growth_trajectory(d) for d in domains]
        trajectories.sort(key=lambda t: t.total_achievements, reverse=True)
        return trajectories

    # ------------------------------------------------------------------
    # Achievement and Failure History
    # ------------------------------------------------------------------

    def get_achievement_history(
        self,
        *,
        domain: str | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return an ordered achievement history with velocity metadata.

        Each entry contains: ``experience_id``, ``title``, ``domain_tags``,
        ``importance``, ``occurred_at``, ``outcome``, ``tags``,
        ``significance_score``, and ``is_milestone``.

        Parameters
        ----------
        domain:
            If supplied, filter to achievements whose tags include this
            domain name.
        start_at:
            UTC lower bound (inclusive).
        end_at:
            UTC upper bound (inclusive).
        limit:
            Maximum entries to return.

        Returns
        -------
        list[dict[str, Any]]
            Achievement entries sorted ascending by ``occurred_at``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("get_achievement_history")

        exps = self._all_experiences_sorted(min_importance=ExperienceImportance.LOW)
        exps = [e for e in exps if e.experience_type == ExperienceType.ACHIEVEMENT]

        if domain is not None:
            domain_lower = domain.strip().lower()
            exps = [
                e for e in exps
                if domain_lower in [t.lower() for t in e.tag_names()]
            ]

        if start_at is not None:
            exps = [e for e in exps if e.occurred_at >= start_at]
        if end_at is not None:
            exps = [e for e in exps if e.occurred_at <= end_at]

        exps = exps[:limit]

        with self._lock:
            milestones_snapshot = dict(self._milestones)
            exp_to_milestones_snapshot = {
                eid: set(mids)
                for eid, mids in self._exp_to_milestones.items()
            }

        entries: list[dict[str, Any]] = []
        for exp in exps:
            is_milestone = bool(
                exp_to_milestones_snapshot.get(exp.experience_id)
            )
            domain_tags = [
                t.name
                for t in exp.tags
                if t.category in ("project", "topic")
            ]
            entries.append(
                {
                    "experience_id": exp.experience_id,
                    "title": exp.title,
                    "domain_tags": domain_tags,
                    "importance": exp.importance.name,
                    "occurred_at": exp.occurred_at.isoformat(),
                    "outcome": exp.outcome,
                    "tags": exp.tag_names(),
                    "significance_score": exp.metadata.significance_score,
                    "is_milestone": is_milestone,
                }
            )

        return entries

    def get_failure_history(
        self,
        *,
        domain: str | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        reflected_only: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return an ordered failure history with lesson metadata.

        Each entry contains: ``experience_id``, ``title``, ``domain_tags``,
        ``importance``, ``occurred_at``, ``outcome``, ``context``, ``tags``,
        ``significance_score``, and ``is_milestone``.

        Parameters
        ----------
        domain:
            If supplied, filter to failures whose tags include this domain.
        start_at:
            UTC lower bound (inclusive).
        end_at:
            UTC upper bound (inclusive).
        reflected_only:
            If ``True``, only return failures that have a related REFLECTION
            experience (linked via ``metadata.related_experience_ids``).
        limit:
            Maximum entries to return.

        Returns
        -------
        list[dict[str, Any]]
            Failure entries sorted ascending by ``occurred_at``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("get_failure_history")

        exps = self._all_experiences_sorted(min_importance=ExperienceImportance.LOW)
        exps = [e for e in exps if e.experience_type == ExperienceType.FAILURE]

        if domain is not None:
            domain_lower = domain.strip().lower()
            exps = [
                e for e in exps
                if domain_lower in [t.lower() for t in e.tag_names()]
            ]

        if start_at is not None:
            exps = [e for e in exps if e.occurred_at >= start_at]
        if end_at is not None:
            exps = [e for e in exps if e.occurred_at <= end_at]

        if reflected_only:
            # Keep only failures that are referenced by a REFLECTION experience
            all_exps = self._all_experiences_sorted(
                min_importance=ExperienceImportance.LOW
            )
            reflected_ids: set[str] = set()
            for e in all_exps:
                if e.experience_type == ExperienceType.REFLECTION:
                    for ref_id in e.metadata.related_experience_ids:
                        reflected_ids.add(ref_id)
            exps = [e for e in exps if e.experience_id in reflected_ids]

        exps = exps[:limit]

        with self._lock:
            exp_to_milestones_snapshot = {
                eid: set(mids)
                for eid, mids in self._exp_to_milestones.items()
            }

        entries: list[dict[str, Any]] = []
        for exp in exps:
            is_milestone = bool(
                exp_to_milestones_snapshot.get(exp.experience_id)
            )
            domain_tags = [
                t.name
                for t in exp.tags
                if t.category in ("project", "topic")
            ]
            entries.append(
                {
                    "experience_id": exp.experience_id,
                    "title": exp.title,
                    "domain_tags": domain_tags,
                    "importance": exp.importance.name,
                    "occurred_at": exp.occurred_at.isoformat(),
                    "outcome": exp.outcome,
                    "context": exp.context,
                    "tags": exp.tag_names(),
                    "significance_score": exp.metadata.significance_score,
                    "is_milestone": is_milestone,
                }
            )

        return entries

    # ------------------------------------------------------------------
    # Project History
    # ------------------------------------------------------------------

    def get_project_history(
        self,
        project_tag: str,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        limit: int = 300,
    ) -> dict[str, Any]:
        """Generate a per-project narrative bundle.

        Collects all experiences tagged with ``project_tag``, groups them
        by experience type, identifies milestones within the project arc,
        and computes basic trajectory metrics.

        Parameters
        ----------
        project_tag:
            The tag name identifying the project
            (e.g. ``"polaris"``, ``"semester-2"``).
        min_importance:
            Minimum importance tier to include.
        limit:
            Maximum total experiences in the bundle.

        Returns
        -------
        dict[str, Any]
            Project bundle with keys: ``project_tag``, ``experience_count``,
            ``timeline``, ``achievements``, ``failures``, ``reflections``,
            ``milestones``, ``first_at``, ``last_at``, ``dominant_tags``,
            ``narrative``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        ValueError
            If ``project_tag`` is empty.
        """
        self._assert_running("get_project_history")

        if not project_tag or not project_tag.strip():
            raise ValueError("project_tag must be a non-empty string.")

        project_tag = project_tag.strip()

        exps = self._all_experiences_sorted(min_importance=min_importance)
        project_exps = [
            e for e in exps if project_tag in e.tag_names()
        ][:limit]

        with self._lock:
            exp_to_milestones_snapshot = {
                eid: list(mids)
                for eid, mids in self._exp_to_milestones.items()
            }
            milestones_snapshot = dict(self._milestones)

        achievements = [
            e for e in project_exps
            if e.experience_type == ExperienceType.ACHIEVEMENT
        ]
        failures = [
            e for e in project_exps
            if e.experience_type == ExperienceType.FAILURE
        ]
        reflections = [
            e for e in project_exps
            if e.experience_type == ExperienceType.REFLECTION
        ]

        milestones_in_project: list[dict[str, Any]] = []
        for exp in project_exps:
            for mid in exp_to_milestones_snapshot.get(exp.experience_id, []):
                if mid in milestones_snapshot:
                    m = milestones_snapshot[mid]
                    milestones_in_project.append(
                        {
                            "milestone_id": m.milestone_id,
                            "label": m.label,
                            "milestone_type": m.milestone_type,
                            "occurred_at": m.occurred_at.isoformat(),
                            "significance": m.significance,
                        }
                    )

        timeline: list[dict[str, Any]] = [
            {
                "experience_id": e.experience_id,
                "title": e.title,
                "experience_type": e.experience_type.name,
                "importance": e.importance.name,
                "occurred_at": e.occurred_at.isoformat(),
            }
            for e in project_exps
        ]

        first_at = project_exps[0].occurred_at.isoformat() if project_exps else None
        last_at = project_exps[-1].occurred_at.isoformat() if project_exps else None

        dominant_tags = self._top_tag_names(project_exps, top_n=5)

        # Build a minimal prose narrative
        narrative = self._build_narrative(
            title=f"Project: {project_tag}",
            scope="period",
            scope_ref=project_tag,
            experience_ids=[e.experience_id for e in project_exps],
        )

        return {
            "project_tag": project_tag,
            "experience_count": len(project_exps),
            "timeline": timeline,
            "achievements": [
                {"experience_id": e.experience_id, "title": e.title}
                for e in achievements
            ],
            "failures": [
                {"experience_id": e.experience_id, "title": e.title}
                for e in failures
            ],
            "reflections": [
                {"experience_id": e.experience_id, "title": e.title}
                for e in reflections
            ],
            "milestones": milestones_in_project,
            "first_at": first_at,
            "last_at": last_at,
            "dominant_tags": dominant_tags,
            "narrative": {
                "opening": narrative.opening,
                "key_events": narrative.key_events,
                "closing": narrative.closing,
            },
        }

    # ------------------------------------------------------------------
    # Life Phase Detection
    # ------------------------------------------------------------------

    def get_life_phases(self) -> list[dict[str, Any]]:
        """Detect and return life phases from the experience history.

        A life phase is a continuous window of activity separated from
        adjacent windows by a gap of at least ``_PHASE_GAP_DAYS`` days.
        Each phase is classified as ``"active"``, ``"transition"``, or
        ``"dormant"`` based on activity density and experience type
        composition.

        Returns
        -------
        list[dict[str, Any]]
            Phase descriptors sorted ascending by ``start_at``, each with
            keys: ``phase_id``, ``label``, ``start_at``, ``end_at``,
            ``experience_count``, ``dominant_type``, ``activity_density``,
            and ``experience_ids``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("get_life_phases")

        phases = self._detect_life_phases()
        return [
            {
                "phase_id": p.phase_id,
                "label": p.label,
                "start_at": p.start_at.isoformat(),
                "end_at": p.end_at.isoformat() if p.end_at else None,
                "experience_count": len(p.experience_ids),
                "dominant_type": p.dominant_type,
                "activity_density": round(p.activity_density, 4),
                "experience_ids": p.experience_ids,
            }
            for p in phases
        ]

    def _detect_life_phases(self) -> list[_LifePhase]:
        """Internal: compute _LifePhase objects from the experience timeline."""
        exps = self._all_experiences_sorted(
            min_importance=ExperienceImportance.LOW
        )

        if not exps:
            return []

        gap_threshold = timedelta(days=_PHASE_GAP_DAYS)
        phases: list[_LifePhase] = []
        current_phase_exps: list[Experience] = [exps[0]]

        for exp in exps[1:]:
            gap = exp.occurred_at - current_phase_exps[-1].occurred_at
            if gap > gap_threshold:
                phases.append(
                    self._finalise_phase(current_phase_exps)
                )
                current_phase_exps = [exp]
            else:
                current_phase_exps.append(exp)

        if current_phase_exps:
            phases.append(self._finalise_phase(current_phase_exps))

        return phases

    def _finalise_phase(self, exps: list[Experience]) -> _LifePhase:
        """Build a :class:`_LifePhase` from a contiguous group of experiences."""
        start = exps[0].occurred_at
        end = exps[-1].occurred_at
        span_days = max(1.0, (end - start).total_seconds() / 86_400.0)
        density = len(exps) / span_days * 30.0  # per 30 days

        # Dominant type: most frequent ExperienceType
        type_counter: dict[str, int] = defaultdict(int)
        for e in exps:
            type_counter[e.experience_type.name] += 1
        dominant_type = max(type_counter, key=lambda k: type_counter[k])

        # Phase label
        if density >= 3.0:
            label = _PHASE_LABEL_ACTIVE
        elif density >= 0.5:
            label = _PHASE_LABEL_TRANSITION
        else:
            label = _PHASE_LABEL_DORMANT

        return _LifePhase(
            label=label,
            start_at=start,
            end_at=end,
            experience_ids=[e.experience_id for e in exps],
            dominant_type=dominant_type,
            activity_density=density,
        )

    # ------------------------------------------------------------------
    # Personal Evolution Analysis
    # ------------------------------------------------------------------

    def analyse_personal_evolution(self) -> dict[str, Any]:
        """Perform a cross-chapter personal evolution analysis.

        Compares achievement velocity, failure rate, reflection density,
        importance distribution, and dominant themes across all chapters
        in chronological order.

        Returns
        -------
        dict[str, Any]
            Analysis result with keys: ``chapter_count``, ``chapters``,
            ``achievement_velocity_trend``, ``failure_rate_trend``,
            ``reflection_density_trend``, ``overall_growth_score``,
            and ``dominant_evolution_themes``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("analyse_personal_evolution")

        with self._lock:
            chapters = sorted(
                self._chapters.values(),
                key=lambda c: (
                    c.start_at is None,
                    c.start_at.timestamp() if c.start_at else 0.0,
                ),
            )

        chapter_analyses: list[dict[str, Any]] = []

        for ch in chapters:
            exps = self._experiences_for_ids(ch.experience_ids)
            if not exps:
                chapter_analyses.append(
                    {
                        "chapter_id": ch.chapter_id,
                        "title": ch.title,
                        "experience_count": 0,
                        "achievement_count": 0,
                        "failure_count": 0,
                        "reflection_count": 0,
                        "duration_days": ch.duration_days(),
                        "achievement_velocity": 0.0,
                        "failure_rate": 0.0,
                        "importance_distribution": {},
                        "dominant_themes": [],
                    }
                )
                continue

            achievements = [
                e for e in exps if e.experience_type == ExperienceType.ACHIEVEMENT
            ]
            failures = [
                e for e in exps if e.experience_type == ExperienceType.FAILURE
            ]
            reflections = [
                e for e in exps if e.experience_type == ExperienceType.REFLECTION
            ]

            ach_velocity = self._velocity(
                [e.occurred_at for e in achievements]
            )
            fail_rate = self._velocity(
                [e.occurred_at for e in failures]
            )

            imp_dist: dict[str, int] = defaultdict(int)
            for e in exps:
                imp_dist[e.importance.name] += 1

            chapter_analyses.append(
                {
                    "chapter_id": ch.chapter_id,
                    "title": ch.title,
                    "experience_count": len(exps),
                    "achievement_count": len(achievements),
                    "failure_count": len(failures),
                    "reflection_count": len(reflections),
                    "duration_days": ch.duration_days(),
                    "achievement_velocity": round(ach_velocity, 4),
                    "failure_rate": round(fail_rate, 4),
                    "importance_distribution": dict(imp_dist),
                    "dominant_themes": self._top_tag_names(exps, top_n=5),
                }
            )

        # Trend lines (list of per-chapter values in chapter order)
        ach_trend = [a["achievement_velocity"] for a in chapter_analyses]
        fail_trend = [a["failure_rate"] for a in chapter_analyses]
        ref_trend = [a["reflection_count"] for a in chapter_analyses]

        # Overall growth score: ratio of achievements to (achievements + failures)
        total_ach = sum(a["achievement_count"] for a in chapter_analyses)
        total_fail = sum(a["failure_count"] for a in chapter_analyses)
        denom = total_ach + total_fail
        growth_score = (total_ach / denom) if denom > 0 else 0.0

        # Dominant themes across all chapters
        all_themes: dict[str, int] = defaultdict(int)
        for ca in chapter_analyses:
            for theme in ca["dominant_themes"]:
                all_themes[theme] += 1
        dominant_themes = [
            t
            for t, _ in sorted(
                all_themes.items(), key=lambda kv: kv[1], reverse=True
            )
        ][:5]

        return {
            "chapter_count": len(chapters),
            "chapters": chapter_analyses,
            "achievement_velocity_trend": ach_trend,
            "failure_rate_trend": fail_trend,
            "reflection_density_trend": ref_trend,
            "overall_growth_score": round(growth_score, 4),
            "dominant_evolution_themes": dominant_themes,
        }

    # ------------------------------------------------------------------
    # Historical Search
    # ------------------------------------------------------------------

    def search_history(
        self,
        query: str,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        experience_type: ExperienceType | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Full-text and metadata-predicate search within the history.

        The query is matched case-insensitively against experience ``title``,
        ``description``, ``outcome``, ``context``, and all tag names.
        Results are ranked by a simple relevance score (match count ×
        importance multiplier) then by recency.

        Parameters
        ----------
        query:
            Search string.  Must be non-empty.
        min_importance:
            Minimum importance tier.
        experience_type:
            If supplied, restrict to this type.
        start_at:
            UTC lower bound for ``occurred_at`` (inclusive).
        end_at:
            UTC upper bound for ``occurred_at`` (inclusive).
        limit:
            Maximum results to return.

        Returns
        -------
        list[dict[str, Any]]
            Matching entries sorted descending by relevance score then
            recency.  Each entry has: ``experience_id``, ``title``,
            ``experience_type``, ``importance``, ``occurred_at``,
            ``relevance_score``, ``matched_fields``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        ValueError
            If ``query`` is empty.
        """
        self._assert_running("search_history")

        if not query or not query.strip():
            raise ValueError("search_history: query must be a non-empty string.")

        query_lower = query.strip().lower()
        tokens = query_lower.split()

        exps = self._all_experiences_sorted(min_importance=min_importance)

        if experience_type is not None:
            exps = [e for e in exps if e.experience_type == experience_type]

        if start_at is not None:
            exps = [e for e in exps if e.occurred_at >= start_at]
        if end_at is not None:
            exps = [e for e in exps if e.occurred_at <= end_at]

        results: list[tuple[float, Experience, list[str]]] = []

        for exp in exps:
            hit_fields: list[str] = []
            score = 0.0
            imp_mult = exp.importance.value  # 1..4

            searchable = {
                "title": exp.title.lower(),
                "description": exp.description.lower(),
                "outcome": exp.outcome.lower(),
                "context": exp.context.lower(),
                "tags": " ".join(exp.tag_names()).lower(),
            }

            for token in tokens:
                for field_name, content in searchable.items():
                    if token in content:
                        score += imp_mult
                        if field_name not in hit_fields:
                            hit_fields.append(field_name)

            if score > 0.0:
                results.append((score, exp, hit_fields))

        # Sort by score desc, then occurred_at desc
        results.sort(key=lambda t: (t[0], t[1].occurred_at.timestamp()), reverse=True)

        return [
            {
                "experience_id": exp.experience_id,
                "title": exp.title,
                "experience_type": exp.experience_type.name,
                "importance": exp.importance.name,
                "occurred_at": exp.occurred_at.isoformat(),
                "relevance_score": round(score, 2),
                "matched_fields": hit_fields,
            }
            for score, exp, hit_fields in results[:limit]
        ]

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summarise_window(
        self,
        start_at: datetime,
        end_at: datetime,
        *,
        label: str = "",
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
    ) -> HistoricalSummary:
        """Generate a :class:`HistoricalSummary` for a time window.

        Parameters
        ----------
        start_at:
            UTC start of the window (inclusive).
        end_at:
            UTC end of the window (inclusive).
        label:
            Human-readable scope label.
        min_importance:
            Minimum importance filter.

        Returns
        -------
        HistoricalSummary
            Populated summary for the window.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        ValueError
            If ``end_at`` is before ``start_at``.
        """
        self._assert_running("summarise_window")

        if end_at < start_at:
            raise ValueError(
                f"end_at ({end_at.isoformat()}) must not precede "
                f"start_at ({start_at.isoformat()})."
            )

        exps = self._all_experiences_sorted(min_importance=min_importance)
        exps = [e for e in exps if start_at <= e.occurred_at <= end_at]

        return self._build_summary(
            scope_label=label or f"{start_at.date().isoformat()} – {end_at.date().isoformat()}",
            start_at=start_at,
            end_at=end_at,
            exps=exps,
        )

    def summarise_chapter(self, chapter_id: str) -> HistoricalSummary:
        """Generate a :class:`HistoricalSummary` for a chapter.

        Parameters
        ----------
        chapter_id:
            UUID of the chapter to summarise.

        Returns
        -------
        HistoricalSummary
            Populated summary for the chapter.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        KeyError
            If no chapter with ``chapter_id`` exists.
        """
        self._assert_running("summarise_chapter")

        with self._lock:
            if chapter_id not in self._chapters:
                raise KeyError(f"No chapter with id '{chapter_id}' found.")
            chapter = self._chapters[chapter_id]
            exp_ids = list(chapter.experience_ids)

        exps = self._experiences_for_ids(exp_ids)
        exps.sort(key=lambda e: e.occurred_at)

        now = datetime.now(timezone.utc)
        start_at = exps[0].occurred_at if exps else now
        end_at = (chapter.end_at or exps[-1].occurred_at) if exps else now

        with self._lock:
            milestone_count = len(self._chapter_to_milestones.get(chapter_id, set()))

        summary = self._build_summary(
            scope_label=chapter.title,
            start_at=start_at,
            end_at=end_at,
            exps=exps,
        )
        summary.milestone_count = milestone_count
        return summary

    def _build_summary(
        self,
        scope_label: str,
        start_at: datetime,
        end_at: datetime,
        exps: list[Experience],
    ) -> HistoricalSummary:
        """Internal helper: populate a :class:`HistoricalSummary`."""
        achievements = [
            e for e in exps if e.experience_type == ExperienceType.ACHIEVEMENT
        ]
        failures = [
            e for e in exps if e.experience_type == ExperienceType.FAILURE
        ]
        reflections = [
            e for e in exps if e.experience_type == ExperienceType.REFLECTION
        ]

        # Top items by importance desc, then recency
        top_ach = sorted(
            achievements,
            key=lambda e: (e.importance.value, e.occurred_at.timestamp()),
            reverse=True,
        )[:3]
        top_fail = sorted(
            failures,
            key=lambda e: (e.importance.value, e.occurred_at.timestamp()),
            reverse=True,
        )[:3]
        top_ref = sorted(
            reflections,
            key=lambda e: (e.importance.value, e.occurred_at.timestamp()),
            reverse=True,
        )[:3]

        top_lessons = [
            e.outcome if e.outcome.strip() else e.description
            for e in top_ref
        ]

        # Milestone count is patched after this helper if needed
        with self._lock:
            milestones = list(self._milestones.values())

        milestone_count = sum(
            1
            for m in milestones
            if start_at <= m.occurred_at <= end_at
        )

        return HistoricalSummary(
            scope_label=scope_label,
            start_at=start_at,
            end_at=end_at,
            total_experiences=len(exps),
            achievement_count=len(achievements),
            failure_count=len(failures),
            reflection_count=len(reflections),
            milestone_count=milestone_count,
            top_achievements=[e.title for e in top_ach],
            top_failures=[e.title for e in top_fail],
            top_lessons=top_lessons,
            dominant_tags=self._top_tag_names(exps, top_n=5),
            dominant_domains=self._top_domains(exps, top_n=3),
            integrity_issues=[],
        )

    # ------------------------------------------------------------------
    # Continuity Validation
    # ------------------------------------------------------------------

    def validate_continuity(self) -> list[str]:
        """Check for temporal gaps, orphaned chapters, and broken references.

        Returns a list of human-readable issue strings.  An empty list
        indicates full continuity across the registered history.

        Returns
        -------
        list[str]
            Issue descriptions.  Empty if no issues found.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("validate_continuity")

        issues: list[str] = []

        with self._lock:
            chapters = sorted(
                self._chapters.values(),
                key=lambda c: (
                    c.start_at is None,
                    c.start_at.timestamp() if c.start_at else 0.0,
                ),
            )
            milestones = list(self._milestones.values())
            milestone_ids_set = set(self._milestones.keys())

        # Check for temporal gaps between consecutive chapters
        previous_end: datetime | None = None
        for ch in chapters:
            if ch.start_at is None:
                issues.append(
                    f"Chapter '{ch.chapter_id}' ({ch.title!r}) has no "
                    "start timestamp — cannot validate temporal continuity."
                )
                continue
            if previous_end is not None:
                gap = (ch.start_at - previous_end).total_seconds() / 86_400.0
                if gap > _PHASE_GAP_DAYS:
                    issues.append(
                        f"Temporal gap of {gap:.1f} days between chapters "
                        f"ending at {previous_end.date().isoformat()} and "
                        f"starting at {ch.start_at.date().isoformat()}."
                    )
            ch_end = ch.end_at or datetime.now(timezone.utc)
            previous_end = max(previous_end, ch_end) if previous_end else ch_end

        # Check for chapter milestone references that don't exist
        for ch in chapters:
            for mid in ch.milestone_ids:
                if mid not in milestone_ids_set:
                    issues.append(
                        f"Chapter '{ch.chapter_id}' ({ch.title!r}) references "
                        f"non-existent milestone '{mid}'."
                    )

        # Check for milestone experience references that don't exist
        for m in milestones:
            if not self._experience_engine.experience_exists(m.experience_id):
                issues.append(
                    f"Milestone '{m.milestone_id}' ({m.label!r}) references "
                    f"non-existent experience '{m.experience_id}'."
                )

        return issues

    def check_narrative_consistency(self) -> list[str]:
        """Validate cross-chapter invariants.

        Checks:
        * Milestone chronological ordering within each chapter.
        * Chapter boundary overlaps.
        * Experiences assigned to no chapter (uncovered).
        * Experiences whose ``occurred_at`` falls outside their chapter's bounds.

        Returns
        -------
        list[str]
            Issue descriptions.  Empty if fully consistent.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("check_narrative_consistency")

        issues: list[str] = []

        with self._lock:
            chapters = list(self._chapters.values())
            milestones = dict(self._milestones)

        # Milestone ordering within chapters
        for ch in chapters:
            chapter_ms = [
                milestones[mid]
                for mid in ch.milestone_ids
                if mid in milestones
            ]
            chapter_ms.sort(key=lambda m: m.occurred_at)
            for i in range(1, len(chapter_ms)):
                if chapter_ms[i].occurred_at < chapter_ms[i - 1].occurred_at:
                    issues.append(
                        f"Milestone ordering inconsistency in chapter "
                        f"'{ch.chapter_id}': '{chapter_ms[i].label}' occurs "
                        f"before '{chapter_ms[i-1].label}' in the timeline."
                    )

        # Chapter boundary overlaps
        closed_chapters = [
            c for c in chapters if c.start_at is not None and c.end_at is not None
        ]
        for i, c1 in enumerate(closed_chapters):
            for c2 in closed_chapters[i + 1:]:
                if c2.start_at is None or c2.end_at is None:
                    continue
                # Overlap if one starts before the other ends
                if c1.start_at < c2.end_at and c2.start_at < c1.end_at:
                    issues.append(
                        f"Chapter overlap: '{c1.chapter_id}' ({c1.title!r}) "
                        f"and '{c2.chapter_id}' ({c2.title!r}) have overlapping "
                        f"time windows."
                    )

        # Experiences outside chapter bounds
        for ch in chapters:
            if ch.start_at is None:
                continue
            ch_end = ch.end_at or datetime.now(timezone.utc)
            for eid in ch.experience_ids:
                try:
                    exp = self._experience_engine.get_experience(eid)
                    if not (ch.start_at <= exp.occurred_at <= ch_end):
                        issues.append(
                            f"Experience '{eid}' ({exp.title!r}) in chapter "
                            f"'{ch.chapter_id}' has occurred_at "
                            f"{exp.occurred_at.isoformat()} outside chapter "
                            f"bounds [{ch.start_at.date().isoformat()}, "
                            f"{ch_end.date().isoformat()}]."
                        )
                except Exception:
                    pass  # broken references are reported in validate_continuity

        return issues

    # ------------------------------------------------------------------
    # Audit Report
    # ------------------------------------------------------------------

    def generate_audit_report(self) -> HistoryAuditReport:
        """Generate a :class:`HistoryAuditReport` for the full history corpus.

        Performs a comprehensive structural scan: uncovered experiences,
        orphaned milestones, broken chapter references, overlapping chapters,
        and temporal coverage gaps.  If an integrity engine is available,
        its scan results are incorporated.

        Returns
        -------
        HistoryAuditReport
            Populated audit report.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("generate_audit_report")

        issues: list[str] = []

        with self._lock:
            chapters = list(self._chapters.values())
            milestones = list(self._milestones.values())
            all_indexed_exp_ids: set[str] = set()
            for ch in chapters:
                all_indexed_exp_ids.update(ch.experience_ids)

        # ---- Uncovered experiences ----------------------------------------
        store_exps = self._all_experiences_sorted(
            min_importance=_TIMELINE_MIN_IMPORTANCE
        )
        store_exp_ids = {e.experience_id for e in store_exps}
        uncovered = [
            eid for eid in store_exp_ids if eid not in all_indexed_exp_ids
        ]
        if uncovered:
            issues.append(
                f"{len(uncovered)} experience(s) at MEDIUM+ importance are "
                "not assigned to any chapter."
            )

        # ---- Orphaned milestones ------------------------------------------
        orphaned_mids: list[str] = []
        for m in milestones:
            if not self._experience_engine.experience_exists(m.experience_id):
                orphaned_mids.append(m.milestone_id)
                issues.append(
                    f"Orphaned milestone '{m.milestone_id}' ({m.label!r}): "
                    f"experience '{m.experience_id}' no longer exists."
                )

        # ---- Broken chapter experience references -------------------------
        broken_chapter_ids: list[str] = []
        for ch in chapters:
            for eid in ch.experience_ids:
                if not self._experience_engine.experience_exists(eid):
                    if ch.chapter_id not in broken_chapter_ids:
                        broken_chapter_ids.append(ch.chapter_id)
                    issues.append(
                        f"Chapter '{ch.chapter_id}' ({ch.title!r}) contains "
                        f"broken reference to experience '{eid}'."
                    )

        # ---- Overlapping chapters -----------------------------------------
        overlapping_pairs: list[tuple[str, str]] = []
        closed = [
            c for c in chapters if c.start_at is not None and c.end_at is not None
        ]
        for i, c1 in enumerate(closed):
            for c2 in closed[i + 1:]:
                if c2.start_at is None or c2.end_at is None:
                    continue
                if c1.start_at < c2.end_at and c2.start_at < c1.end_at:
                    overlapping_pairs.append((c1.chapter_id, c2.chapter_id))
                    issues.append(
                        f"Overlapping chapters: '{c1.chapter_id}' and "
                        f"'{c2.chapter_id}'."
                    )

        # ---- Temporal gaps -----------------------------------------------
        sorted_chapters = sorted(
            [c for c in chapters if c.start_at is not None],
            key=lambda c: c.start_at,  # type: ignore[arg-type]
        )
        temporal_gaps: list[dict[str, Any]] = []
        prev_end: datetime | None = None
        for ch in sorted_chapters:
            if prev_end is not None and ch.start_at is not None:
                gap_days = (ch.start_at - prev_end).total_seconds() / 86_400.0
                if gap_days > _PHASE_GAP_DAYS:
                    temporal_gaps.append(
                        {
                            "from": prev_end.isoformat(),
                            "to": ch.start_at.isoformat(),
                            "gap_days": round(gap_days, 1),
                        }
                    )
                    issues.append(
                        f"Temporal gap of {gap_days:.1f} days between "
                        f"{prev_end.date().isoformat()} and "
                        f"{ch.start_at.date().isoformat()}."
                    )
            ch_end = ch.end_at or datetime.now(timezone.utc)
            prev_end = max(prev_end, ch_end) if prev_end else ch_end

        # ---- Integrity engine scan (optional) ----------------------------
        if self._integrity_engine is not None:
            try:
                report = self._integrity_engine.run_full_audit()
                if hasattr(report, "violations") and report.violations:
                    issues.append(
                        f"Memory integrity engine reported "
                        f"{len(report.violations)} violation(s). "
                        "Run MemoryIntegrityEngine.run_full_audit() for details."
                    )
            except Exception as exc:
                issues.append(
                    f"Integrity engine scan failed: {exc}"
                )

        # ---- Integrity score ---------------------------------------------
        total_checks = max(
            1,
            len(store_exps)
            + len(milestones)
            + len(chapters)
            + len(overlapping_pairs),
        )
        total_issues = (
            len(uncovered)
            + len(orphaned_mids)
            + len(broken_chapter_ids)
            + len(overlapping_pairs)
            + len(temporal_gaps)
        )
        integrity_score = max(
            0.0, 1.0 - total_issues / total_checks
        )

        return HistoryAuditReport(
            total_chapters=len(chapters),
            total_milestones=len(milestones),
            total_indexed_experiences=len(all_indexed_exp_ids),
            uncovered_experience_ids=uncovered,
            orphaned_milestone_ids=orphaned_mids,
            broken_chapter_refs=broken_chapter_ids,
            overlapping_chapter_pairs=overlapping_pairs,
            temporal_gaps=temporal_gaps,
            integrity_score=round(integrity_score, 4),
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Personal History Report
    # ------------------------------------------------------------------

    def generate_history_report(self) -> dict[str, Any]:
        """Generate a structured, auditable snapshot of the full history state.

        Combines the life timeline, chapter index, milestone index, phase
        detection, evolution analysis, and audit report into a single
        top-level dict suitable for export, health checks, and downstream
        consumption.

        Returns
        -------
        dict[str, Any]
            History report with keys: ``generated_at``, ``chapter_count``,
            ``milestone_count``, ``timeline_length``, ``life_phases``,
            ``evolution_analysis``, ``audit_report``, ``chapter_index``,
            and ``top_milestones``.

        Raises
        ------
        EchoNotInitializedError
            If the engine is not running.
        """
        self._assert_running("generate_history_report")

        timeline = self.generate_life_timeline(
            min_importance=_TIMELINE_MIN_IMPORTANCE
        )
        phases = self.get_life_phases()
        evolution = self.analyse_personal_evolution()
        audit = self.generate_audit_report()

        with self._lock:
            chapter_index = [
                {
                    "chapter_id": ch.chapter_id,
                    "title": ch.title,
                    "experience_count": ch.experience_count(),
                    "milestone_count": len(ch.milestone_ids),
                    "start_at": ch.start_at.isoformat() if ch.start_at else None,
                    "end_at": ch.end_at.isoformat() if ch.end_at else None,
                    "is_closed": ch.is_closed,
                    "tags": ch.tag_names(),
                }
                for ch in sorted(
                    self._chapters.values(),
                    key=lambda c: (
                        c.start_at is None,
                        c.start_at.timestamp() if c.start_at else 0.0,
                    ),
                )
            ]
            top_milestones = sorted(
                self._milestones.values(),
                key=lambda m: m.occurred_at,
            )

        top_milestone_dicts = [
            {
                "milestone_id": m.milestone_id,
                "label": m.label,
                "milestone_type": m.milestone_type,
                "occurred_at": m.occurred_at.isoformat(),
                "chapter_id": m.chapter_id,
                "significance": m.significance,
            }
            for m in top_milestones
        ]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "chapter_count": len(chapter_index),
            "milestone_count": len(top_milestone_dicts),
            "timeline_length": len(timeline),
            "life_phases": phases,
            "evolution_analysis": evolution,
            "audit_report": {
                "integrity_score": audit.integrity_score,
                "total_issues": len(audit.issues),
                "issues": audit.issues,
                "uncovered_count": len(audit.uncovered_experience_ids),
                "orphaned_milestone_count": len(audit.orphaned_milestone_ids),
                "temporal_gaps": audit.temporal_gaps,
            },
            "chapter_index": chapter_index,
            "top_milestones": top_milestone_dicts,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of the engine's current state.

        Useful for health checks, debugging, and test assertions.

        Returns
        -------
        dict[str, Any]
            Contains ``running``, ``chapter_count``, ``milestone_count``,
            ``indexed_experience_count``, ``closed_chapter_count``, and
            ``injected_engines``.
        """
        with self._lock:
            closed_count = sum(
                1 for c in self._chapters.values() if c.is_closed
            )
            indexed_exp_count = len(self._exp_to_chapters)

        return {
            "running": self._running,
            "chapter_count": len(self._chapters),
            "milestone_count": len(self._milestones),
            "indexed_experience_count": indexed_exp_count,
            "closed_chapter_count": closed_count,
            "injected_engines": {
                "experience_engine": self._experience_engine is not None,
                "retrieval_engine": self._retrieval_engine is not None,
                "reflection_engine": self._reflection_engine is not None,
                "pattern_engine": self._pattern_engine is not None,
                "context_engine": self._context_engine is not None,
                "episodic_index": self._episodic_index is not None,
                "integrity_engine": self._integrity_engine is not None,
            },
        }