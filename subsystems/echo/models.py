# subsystems/echo/models.py
"""
ECHO v1 Domain Models.

These dataclasses define the canonical data structures for the ECHO Episodic
Memory Core.  Every public API in ECHO produces or consumes these types.

ECHO owns:
    Experiences, Events, Conversations, Sessions, Achievements, Failures,
    Observations, Activity History, Personal History.

ECHO does NOT own:
    Knowledge, Identity, Goals, Schedules, Relationships, Decisions.
    Those belong to LUNA, ASTRA, CHRONOS, and ODYSSEY respectively.

Models
------
* :class:`ExperienceType`       — classification of experience kind
* :class:`ExperienceImportance` — four-tier importance classification
* :class:`MemoryTag`            — lightweight label for indexing
* :class:`ExperienceMetadata`   — auxiliary index and audit fields
* :class:`Experience`           — root episodic record (core unit of ECHO)
* :class:`EventRecord`          — discrete event building block
* :class:`AchievementRecord`    — completed accomplishment record
* :class:`FailureRecord`        — objective failure record for learning
* :class:`ObservationRecord`    — passive observation stored by ECHO
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, unique
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


@unique
class ExperienceType(Enum):
    """Classification of what kind of episodic experience this is.

    ECHO uses this type to route records to the correct internal engine and
    to drive retrieval queries from ORION and ODYSSEY.
    """

    EVENT = auto()
    """A discrete happening — project created, exam passed, feature shipped."""

    CONVERSATION = auto()
    """A significant interaction — design discussion, planning session, review."""

    SESSION = auto()
    """A bounded work or activity period grouping other experiences."""

    ACHIEVEMENT = auto()
    """A completed accomplishment — goal reached, milestone hit, skill unlocked."""

    FAILURE = auto()
    """An objective failure — deadline missed, project stalled, rework needed."""

    OBSERVATION = auto()
    """A passive observation of a pattern or behaviour, noted for later use."""

    REFLECTION = auto()
    """A synthesised lesson derived from one or more prior experiences."""

    MILESTONE = auto()
    """A significant marker in a personal history narrative."""


@unique
class ExperienceImportance(Enum):
    """Four-tier importance scale used by the Significance Engine.

    Only MEDIUM and above are candidates for long-term memory consolidation.
    LOW records may be pruned by the Memory Consolidation Engine.
    """

    LOW = 1
    """Routine noise.  May be pruned.  Not stored long-term by default."""

    MEDIUM = 2
    """Notable but not exceptional.  Stored; eligible for consolidation."""

    HIGH = 3
    """Significant experience.  Always stored and indexed in full."""

    CRITICAL = 4
    """Defining moment.  Permanent record; never pruned; always retrievable."""


# ---------------------------------------------------------------------------
# Memory Tag
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryTag:
    """Lightweight immutable label attached to an :class:`Experience`.

    Tags drive the :class:`EpisodicIndexEngine` (future) and power fast
    retrieval queries by project, person, topic, or arbitrary category.

    Attributes
    ----------
    name:
        The tag text (e.g. ``"polaris"``, ``"architecture"``, ``"refactor"``).
    category:
        Broad grouping: ``"project"``, ``"person"``, ``"topic"``,
        ``"goal"``, ``"custom"``.
    """

    name: str
    category: str = "custom"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("MemoryTag.name must be a non-empty string.")
        valid_categories = {"project", "person", "topic", "goal", "custom"}
        if self.category not in valid_categories:
            raise ValueError(
                f"MemoryTag.category must be one of {valid_categories}; "
                f"got '{self.category}'."
            )


# ---------------------------------------------------------------------------
# Experience Metadata
# ---------------------------------------------------------------------------


@dataclass
class ExperienceMetadata:
    """Auxiliary indexing and audit fields attached to every :class:`Experience`.

    This object is managed internally by ECHO engines and should not be
    constructed directly by callers outside the subsystem.

    Attributes
    ----------
    source_subsystem:
        Which POLARIS subsystem triggered the experience record
        (e.g. ``"ORION"``, ``"ODYSSEY"``, ``"ECHO_API"``).
    session_id:
        Optional reference to the enclosing :class:`Experience` whose type
        is ``SESSION``.  ``None`` for top-level experiences.
    related_experience_ids:
        UUIDs of causally or thematically related experiences.
    project_refs:
        Short project identifiers extracted from tags or supplied explicitly.
    significance_score:
        0.0-1.0 numeric score produced by the Significance Engine.  Higher
        values indicate greater long-term memory value.
    consolidated:
        True once the Memory Consolidation Engine has processed this record
        into long-term storage.
    consolidation_at:
        UTC timestamp of consolidation, or ``None`` if not yet consolidated.
    retrieval_count:
        How many times this experience has been fetched by other engines.
    last_retrieved_at:
        UTC timestamp of the most recent retrieval.
    """

    source_subsystem: str = "ECHO_API"
    session_id: str | None = None
    related_experience_ids: list[str] = field(default_factory=list)
    project_refs: list[str] = field(default_factory=list)
    significance_score: float = 0.0
    consolidated: bool = False
    consolidation_at: datetime | None = None
    retrieval_count: int = 0
    last_retrieved_at: datetime | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.significance_score <= 1.0):
            raise ValueError(
                "ExperienceMetadata.significance_score must be between 0.0 and 1.0."
            )
        if self.retrieval_count < 0:
            raise ValueError(
                "ExperienceMetadata.retrieval_count cannot be negative."
            )

    def record_retrieval(self) -> None:
        """Increment the retrieval counter and update the timestamp."""
        self.retrieval_count += 1
        self.last_retrieved_at = datetime.now(timezone.utc)

    def mark_consolidated(self) -> None:
        """Mark this experience as consolidated into long-term memory."""
        self.consolidated = True
        self.consolidation_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Core Experience
# ---------------------------------------------------------------------------


@dataclass
class Experience:
    """Root episodic record — the core unit of ECHO.

    An :class:`Experience` represents a meaningful thing that happened.
    It is NOT a fact (LUNA), NOT an identity signal (ASTRA), and NOT a
    plan (CHRONOS/ODYSSEY).  It is the raw episodic record of what occurred.

    The Experience Engine creates, stores, and manages the lifecycle of
    these objects.  The Significance Engine determines their
    :attr:`importance`.  The Memory Consolidation Engine decides whether
    they graduate to long-term memory.

    Attributes
    ----------
    title:
        Short, human-readable label (e.g. ``"Completed POLARIS architecture freeze"``).
    experience_type:
        :class:`ExperienceType` classification.
    importance:
        :class:`ExperienceImportance` tier, set or confirmed by the
        Significance Engine.
    experience_id:
        UUID-4 string uniquely identifying this experience.
    description:
        Narrative elaboration of what happened.
    context:
        Situational context at the time of the experience.
    outcome:
        What resulted from or followed this experience.
    tags:
        Frozenset of :class:`MemoryTag` objects for indexing.
    occurred_at:
        UTC timestamp of when this experience actually occurred.  Defaults
        to the creation time; callers may supply a historical timestamp.
    recorded_at:
        UTC timestamp of when ECHO stored this record.  Always set
        automatically; never supplied by callers.
    metadata:
        :class:`ExperienceMetadata` managed by ECHO engines.
    extra:
        Extensible key-value store for engine-specific annotations.
    """

    title: str
    experience_type: ExperienceType
    importance: ExperienceImportance
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    context: str = ""
    outcome: str = ""
    tags: list[MemoryTag] = field(default_factory=list)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: ExperienceMetadata = field(default_factory=ExperienceMetadata)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("Experience.title must be a non-empty string.")

    def is_significant(self) -> bool:
        """Return True if this experience is above the LOW importance threshold."""
        return self.importance != ExperienceImportance.LOW

    def is_permanent(self) -> bool:
        """Return True if this experience carries CRITICAL importance and must
        never be pruned by the Memory Consolidation Engine."""
        return self.importance == ExperienceImportance.CRITICAL

    def add_tag(self, tag: MemoryTag) -> None:
        """Attach a :class:`MemoryTag` to this experience (idempotent by name)."""
        existing_names = {t.name for t in self.tags}
        if tag.name not in existing_names:
            self.tags.append(tag)

    def tag_names(self) -> list[str]:
        """Return a flat list of tag name strings for quick lookups."""
        return [t.name for t in self.tags]


# ---------------------------------------------------------------------------
# Event Record
# ---------------------------------------------------------------------------


@dataclass
class EventRecord:
    """Discrete event — the atomic building block of ECHO's event history.

    Events are distinct from full :class:`Experience` objects in that they
    capture a single, concrete happening without narrative elaboration.
    Multiple events may compose a Session or be grouped under an Experience.

    Examples: *"Project Created"*, *"Goal Completed"*, *"Exam Passed"*,
    *"New Skill Learned"*.

    Attributes
    ----------
    event_name:
        Short label for the event (e.g. ``"ProjectCreated"``).
    event_id:
        UUID-4 unique identifier.
    experience_id:
        Reference to the parent :class:`Experience`, if this event is
        associated with a broader episode.  May be ``None`` for standalone events.
    payload:
        Structured key-value data describing the event.
    importance:
        Importance classification, defaults to MEDIUM.
    source_subsystem:
        Which subsystem emitted this event.
    occurred_at:
        UTC timestamp of the event.
    recorded_at:
        UTC timestamp of ECHO storage.
    """

    event_name: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experience_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    importance: ExperienceImportance = ExperienceImportance.MEDIUM
    source_subsystem: str = "ECHO_API"
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.event_name or not self.event_name.strip():
            raise ValueError("EventRecord.event_name must be a non-empty string.")


# ---------------------------------------------------------------------------
# Achievement Record
# ---------------------------------------------------------------------------


@dataclass
class AchievementRecord:
    """Record of a completed accomplishment.

    Achievements are a specialised experience type tracked by the Achievement
    Engine and surfaced to ASTRA's growth metrics.  They represent positive
    closures: goals reached, milestones hit, features shipped, skills earned.

    Examples: *"Completed Subsystem Design"*, *"Finished Semester"*,
    *"Released Version"*.

    Attributes
    ----------
    title:
        Short label (e.g. ``"Released POLARIS v1"``).
    achievement_id:
        UUID-4 unique identifier.
    experience_id:
        Optional reference to the parent :class:`Experience`.
    domain:
        Broad domain category (e.g. ``"software"``, ``"academic"``,
        ``"personal"``).
    description:
        Narrative of what was accomplished and why it matters.
    evidence:
        Observable proof items (e.g. commit hash, exam result, shipped URL).
    importance:
        Defaults to HIGH since achievements are inherently significant.
    tags:
        :class:`MemoryTag` list for indexing by project / person / topic.
    achieved_at:
        UTC timestamp of the achievement moment.
    recorded_at:
        UTC timestamp of ECHO storage.
    """

    title: str
    achievement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experience_id: str | None = None
    domain: str = "general"
    description: str = ""
    evidence: list[str] = field(default_factory=list)
    importance: ExperienceImportance = ExperienceImportance.HIGH
    tags: list[MemoryTag] = field(default_factory=list)
    achieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("AchievementRecord.title must be a non-empty string.")


# ---------------------------------------------------------------------------
# Failure Record
# ---------------------------------------------------------------------------


@dataclass
class FailureRecord:
    """Objective record of a failure, tracked for learning — not criticism.

    The Failure Analysis Engine stores these records to generate reflections
    (via the Reflection Engine) and feed pattern extraction (via the Pattern
    Extraction Engine).  Failures are first-class citizens of ECHO because
    learning requires honest accounting of what did not work.

    Examples: *"Project Stalled"*, *"Missed Deadline"*,
    *"Architecture Rework Needed"*.

    Attributes
    ----------
    title:
        Short label (e.g. ``"Architecture Rework Required"``).
    failure_id:
        UUID-4 unique identifier.
    experience_id:
        Optional reference to the parent :class:`Experience`.
    domain:
        Broad domain category.
    description:
        Objective account of what failed and the immediate circumstances.
    contributing_factors:
        List of identified root-cause or contributing factors.
    lesson:
        Initial lesson extracted at record time.  May be refined later by
        the Reflection Engine.
    importance:
        Defaults to MEDIUM; callers should escalate critical failures to HIGH.
    tags:
        :class:`MemoryTag` list for indexing.
    reflection_generated:
        True once the Reflection Engine has processed this failure into a
        formal :class:`Experience` of type ``REFLECTION``.
    failed_at:
        UTC timestamp of the failure moment.
    recorded_at:
        UTC timestamp of ECHO storage.
    """

    title: str
    failure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experience_id: str | None = None
    domain: str = "general"
    description: str = ""
    contributing_factors: list[str] = field(default_factory=list)
    lesson: str = ""
    importance: ExperienceImportance = ExperienceImportance.MEDIUM
    tags: list[MemoryTag] = field(default_factory=list)
    reflection_generated: bool = False
    failed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("FailureRecord.title must be a non-empty string.")

    def mark_reflected(self) -> None:
        """Flag that the Reflection Engine has processed this failure."""
        self.reflection_generated = True


# ---------------------------------------------------------------------------
# Observation Record
# ---------------------------------------------------------------------------


@dataclass
class ObservationRecord:
    """Passive observation stored by ECHO for future pattern extraction.

    Observations differ from Events in that they are not triggered by a
    discrete action — they are noticed patterns, ambient signals, or
    recurring behaviours that ECHO captures on behalf of other engines.

    The Pattern Extraction Engine consumes observations to discover recurring
    experience structures and feed ASTRA and NOVA.

    Attributes
    ----------
    summary:
        One-sentence observation (e.g. ``"User stays productive past midnight"``).
    observation_id:
        UUID-4 unique identifier.
    experience_id:
        Optional reference to the parent :class:`Experience`.
    domain:
        Broad domain category.
    detail:
        Expanded description of the observation.
    evidence_refs:
        UUIDs of :class:`Experience` or :class:`EventRecord` objects that
        support this observation.
    importance:
        Defaults to LOW since most observations require accumulation before
        they become significant.
    observed_at:
        UTC timestamp of the observation moment.
    recorded_at:
        UTC timestamp of ECHO storage.
    """

    summary: str
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experience_id: str | None = None
    domain: str = "general"
    detail: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    importance: ExperienceImportance = ExperienceImportance.LOW
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.summary or not self.summary.strip():
            raise ValueError("ObservationRecord.summary must be a non-empty string.")