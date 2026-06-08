# subsystems/astra/models.py
"""
ASTRA v5 Domain Models.

These dataclasses define the canonical data structures for the ASTRA Digital
Twin Core.  Every public API in ASTRA produces or consumes these types.

Models
------
* :class:`IdentityProfile`   — stable self-model for the user
* :class:`Goal`              — a single goal with full lifecycle metadata
* :class:`GoalState`         — enum for goal lifecycle states
* :class:`PreferenceProfile` — persistent user tendency models
* :class:`CapabilitySnapshot`— point-in-time capability assessment
* :class:`DigitalTwinState`  — the aggregated current-self model
* :class:`EvolutionRecord`   — an immutable audit log entry for identity changes
* :class:`ConsistencyReport` — output of the Consistency Engine
* :class:`FutureSelfReference` — lightweight pointer to a future-self model
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
class GoalState(Enum):
    """Lifecycle state of a :class:`Goal`.

    Ordering: ``ACTIVE`` > ``PAUSED`` > ``COMPLETED`` | ``ABANDONED``
    """

    ACTIVE = auto()
    """Goal is being actively pursued."""

    PAUSED = auto()
    """Goal is temporarily suspended; not abandoned."""

    COMPLETED = auto()
    """Goal has been successfully achieved."""

    ABANDONED = auto()
    """Goal was explicitly abandoned; will not be resumed."""

    DEFERRED = auto()
    """Goal is acknowledged but intentionally postponed."""


@unique
class GoalPriority(Enum):
    """Priority tier for a :class:`Goal`."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@unique
class GoalType(Enum):
    """Classification of a goal by time horizon and domain."""

    DAILY = auto()
    """Short-cycle operational goal (< 1 day)."""

    PROJECT = auto()
    """Bounded deliverable goal with a clear endpoint."""

    CAREER = auto()
    """Professional development and trajectory goal."""

    LIFE = auto()
    """Long-horizon existential or identity-level goal."""


@unique
class EvolutionTrigger(Enum):
    """What drove an identity evolution event."""

    MANUAL_UPDATE = auto()
    """Explicit API call to update identity."""

    EVIDENCE_THRESHOLD = auto()
    """Accumulated evidence crossed the confidence threshold."""

    PATTERN_CONVERGENCE = auto()
    """Multiple independent patterns converged on the same conclusion."""

    EXTERNAL_SIGNAL = auto()
    """An external subsystem (ECHO, LUNA, AURORA) emitted a significant signal."""


# ---------------------------------------------------------------------------
# Identity Profile
# ---------------------------------------------------------------------------


@dataclass
class IdentityProfile:
    """Stable self-model describing who the user is.

    This is NOT a user record or profile database entry.  It is a cognitive
    model of the user's core identity, maintained by the Identity Engine and
    updated only when sufficient evidence justifies an evolution event.

    Attributes
    ----------
    name:
        The user's preferred name or handle.
    background:
        Biographical and contextual background summary.
    education:
        Current or most recent educational context.
    career_direction:
        Where the user is heading professionally.
    interests:
        List of identified interest domains (ordered by strength).
    core_identity_tags:
        Short descriptive tags that characterize the user at an identity level
        (e.g. ``["builder", "systems-thinker", "entrepreneur"]``).
    created_at:
        UTC timestamp of profile creation.
    updated_at:
        UTC timestamp of the most recent evidence-backed update.
    version:
        Monotonically increasing version counter.
    metadata:
        Extensible key-value store for future identity dimensions.
    """

    name: str
    background: str = ""
    education: str = ""
    career_direction: str = ""
    interests: list[str] = field(default_factory=list)
    core_identity_tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("IdentityProfile.name must be a non-empty string.")
        if self.version < 1:
            raise ValueError("IdentityProfile.version must be >= 1.")

    def bump_version(self) -> None:
        """Increment the version counter and update the timestamp."""
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    """A single user goal with full lifecycle metadata.

    Goals are owned exclusively by ASTRA.  Plans, schedules, and execution
    details belong to CHRONOS and ODYSSEY respectively.

    Attributes
    ----------
    goal_id:
        UUID-4 string uniquely identifying this goal.
    title:
        Short, human-readable label (e.g. ``"Build POLARIS v5"``).
    goal_type:
        :class:`GoalType` classification.
    state:
        Current :class:`GoalState` in the lifecycle.
    priority:
        :class:`GoalPriority` tier.
    description:
        Longer elaboration of what the goal means and entails.
    motivation:
        *Why* this goal matters to the user.  Powers ODYSSEY's reasoning.
    success_criteria:
        Observable conditions that indicate completion.
    tags:
        Free-form labels for grouping and discovery.
    parent_goal_id:
        Optional reference to a parent goal (enables goal hierarchies).
    created_at:
        UTC creation timestamp.
    updated_at:
        UTC timestamp of the most recent state or metadata change.
    target_date:
        Optional soft target completion date.
    completed_at:
        Set when state transitions to ``COMPLETED``.
    progress_pct:
        0-100 integer representing subjective progress.
    metadata:
        Extensible key-value annotations.
    """

    title: str
    goal_type: GoalType
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: GoalState = GoalState.ACTIVE
    priority: GoalPriority = GoalPriority.MEDIUM
    description: str = ""
    motivation: str = ""
    success_criteria: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    parent_goal_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    target_date: datetime | None = None
    completed_at: datetime | None = None
    progress_pct: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("Goal.title must be a non-empty string.")
        if not (0 <= self.progress_pct <= 100):
            raise ValueError("Goal.progress_pct must be between 0 and 100.")

    def mark_updated(self) -> None:
        """Refresh the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def is_terminal(self) -> bool:
        """Return True if this goal is in a terminal state."""
        return self.state in (GoalState.COMPLETED, GoalState.ABANDONED)


# ---------------------------------------------------------------------------
# Preference Profile
# ---------------------------------------------------------------------------


@dataclass
class CommunicationPreferences:
    """How the user prefers to communicate and receive information."""

    preferred_detail_level: str = "detailed"
    """'brief' | 'normal' | 'detailed' | 'comprehensive'"""

    prefers_examples: bool = True
    """Whether the user finds concrete examples helpful."""

    prefers_analogies: bool = True
    """Whether the user connects well with analogical reasoning."""

    preferred_tone: str = "collaborative"
    """'formal' | 'casual' | 'collaborative' | 'direct'"""

    structured_output_preferred: bool = True
    """Whether structured (headers, bullets) output is preferred over prose."""


@dataclass
class LearningPreferences:
    """How the user absorbs and processes new knowledge."""

    preferred_approach: str = "systems_first"
    """'examples_first' | 'theory_first' | 'systems_first' | 'practice_first'"""

    prefers_big_picture_first: bool = True
    """True if the user wants overview before implementation details."""

    learns_by_building: bool = True
    """True if the user internalises knowledge through construction."""

    preferred_learning_pace: str = "fast"
    """'methodical' | 'normal' | 'fast' | 'intensive'"""

    depth_over_breadth: bool = False
    """True if deep mastery is preferred over broad coverage."""


@dataclass
class DevelopmentPreferences:
    """How the user prefers to develop software."""

    preferred_languages: list[str] = field(default_factory=list)
    """Ordered list of preferred programming languages."""

    architecture_first: bool = True
    """True if the user designs before implementing."""

    prefers_modularity: bool = True
    """True if highly modular, composable designs are preferred."""

    testing_discipline: str = "pragmatic"
    """'minimal' | 'pragmatic' | 'thorough' | 'tdd'"""

    documentation_style: str = "inline"
    """'none' | 'inline' | 'docstrings' | 'comprehensive'"""


@dataclass
class WorkflowPreferences:
    """How the user organises their work and time."""

    deep_work_sessions: bool = True
    """True if long uninterrupted focus blocks are preferred."""

    preferred_session_length_hours: float = 3.0
    """Typical preferred working session length."""

    async_communication_preferred: bool = True
    """True if the user prefers async over real-time communication."""

    batch_decisions: bool = False
    """True if the user prefers to batch decisions rather than decide immediately."""

    peak_hours: list[str] = field(default_factory=list)
    """Self-reported peak productivity hours (e.g. ['22:00', '02:00'])."""


@dataclass
class PreferenceProfile:
    """Aggregated persistent tendency model for the user.

    Preferences are NOT temporary choices — they represent stable tendencies
    that have been observed and confirmed over time.

    Attributes
    ----------
    communication:
        Communication style and delivery preferences.
    learning:
        Knowledge acquisition and processing preferences.
    development:
        Software development methodology preferences.
    workflow:
        Work organisation and time management preferences.
    updated_at:
        UTC timestamp of the most recent preference update.
    version:
        Monotonically increasing version counter.
    metadata:
        Extensible annotations for future preference dimensions.
    """

    communication: CommunicationPreferences = field(
        default_factory=CommunicationPreferences
    )
    learning: LearningPreferences = field(default_factory=LearningPreferences)
    development: DevelopmentPreferences = field(
        default_factory=DevelopmentPreferences
    )
    workflow: WorkflowPreferences = field(default_factory=WorkflowPreferences)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def bump_version(self) -> None:
        """Increment version and refresh timestamp."""
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Capability Snapshot
# ---------------------------------------------------------------------------


@dataclass
class CapabilityEntry:
    """A single assessed capability."""

    name: str
    """Capability label (e.g. ``"python_architecture"``)."""

    domain: str
    """Domain category (e.g. ``"programming"``, ``"design"``)."""

    confidence: float
    """0.0-1.0 confidence in the assessment."""

    evidence_count: int = 0
    """Number of observed evidence instances supporting this entry."""

    notes: str = ""
    """Human-readable rationale."""

    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("CapabilityEntry.confidence must be between 0.0 and 1.0.")
        if self.evidence_count < 0:
            raise ValueError("CapabilityEntry.evidence_count cannot be negative.")


@dataclass
class CapabilitySnapshot:
    """Point-in-time snapshot of assessed user capabilities.

    Capabilities are tracked by ASTRA but informed by observations from other
    subsystems.  This snapshot is embedded in the DigitalTwinState.

    Attributes
    ----------
    strengths:
        Demonstrated strengths with evidence backing.
    growth_areas:
        Areas where capability is developing but not yet strong.
    snapshot_at:
        UTC timestamp of this snapshot.
    """

    strengths: list[CapabilityEntry] = field(default_factory=list)
    growth_areas: list[CapabilityEntry] = field(default_factory=list)
    snapshot_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def all_entries(self) -> list[CapabilityEntry]:
        """Return all capability entries regardless of category."""
        return self.strengths + self.growth_areas


# ---------------------------------------------------------------------------
# Digital Twin State
# ---------------------------------------------------------------------------


@dataclass
class DigitalTwinState:
    """The aggregated current-self model — the heart of ASTRA.

    :class:`DigitalTwinState` is a single cognitive object that represents
    the user at a given point in time.  It is produced by the Digital Twin
    Engine by synthesising all ASTRA-owned dimensions.

    Consuming subsystems (ORION, VEGA, ODYSSEY, APOLLO, JANUS, PROMETHEUS,
    ZENITH) read this object to understand who they are serving.

    Attributes
    ----------
    identity:
        Current :class:`IdentityProfile`.
    goals:
        All non-abandoned goals, ordered by priority descending.
    preferences:
        Current :class:`PreferenceProfile`.
    capabilities:
        Latest :class:`CapabilitySnapshot`.
    growth_indicators:
        Key-value map of growth metrics (e.g. ``{"projects_completed": 3}``).
    future_self_refs:
        List of :class:`FutureSelfReference` pointers (1yr, 5yr, 10yr).
    consistency_score:
        0.0-1.0 score representing identity stability.
    generated_at:
        UTC timestamp of twin generation.
    twin_version:
        Monotonically increasing generation counter.
    metadata:
        Extensible annotations.
    """

    identity: IdentityProfile
    goals: list[Goal]
    preferences: PreferenceProfile
    capabilities: CapabilitySnapshot
    growth_indicators: dict[str, Any] = field(default_factory=dict)
    future_self_refs: list[FutureSelfReference] = field(default_factory=list)
    consistency_score: float = 1.0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    twin_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.consistency_score <= 1.0):
            raise ValueError(
                "DigitalTwinState.consistency_score must be between 0.0 and 1.0."
            )

    @property
    def active_goals(self) -> list[Goal]:
        """Return only goals in :attr:`GoalState.ACTIVE` state."""
        return [g for g in self.goals if g.state == GoalState.ACTIVE]

    @property
    def is_consistent(self) -> bool:
        """Return True if consistency score is above the stability threshold."""
        return self.consistency_score >= 0.6


# ---------------------------------------------------------------------------
# Evolution Record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolutionRecord:
    """Immutable audit-log entry capturing a single identity evolution event.

    Every time ASTRA updates the identity model based on accumulated evidence,
    an :class:`EvolutionRecord` is appended to the permanent evolution log.
    Records are never modified or deleted.

    Attributes
    ----------
    record_id:
        UUID-4 unique identifier.
    trigger:
        What drove this evolution (:class:`EvolutionTrigger`).
    changed_fields:
        List of field names that changed (e.g. ``["career_direction"]``).
    previous_snapshot:
        Serialised representation of the affected fields before the change.
    new_snapshot:
        Serialised representation of the affected fields after the change.
    confidence:
        0.0-1.0 confidence in the validity of this evolution.
    evidence_count:
        Number of evidence items that justified this change.
    notes:
        Human-readable rationale for the evolution.
    evolved_at:
        UTC timestamp of the evolution.
    """

    record_id: str
    trigger: EvolutionTrigger
    changed_fields: tuple[str, ...]
    previous_snapshot: dict[str, Any]
    new_snapshot: dict[str, Any]
    confidence: float
    evidence_count: int
    notes: str
    evolved_at: datetime

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("EvolutionRecord.confidence must be between 0.0 and 1.0.")
        if self.evidence_count < 0:
            raise ValueError("EvolutionRecord.evidence_count cannot be negative.")

    @classmethod
    def create(
        cls,
        *,
        trigger: EvolutionTrigger,
        changed_fields: list[str],
        previous_snapshot: dict[str, Any],
        new_snapshot: dict[str, Any],
        confidence: float,
        evidence_count: int,
        notes: str = "",
    ) -> "EvolutionRecord":
        """Factory that generates a UUID and timestamp automatically."""
        return cls(
            record_id=str(uuid.uuid4()),
            trigger=trigger,
            changed_fields=tuple(changed_fields),
            previous_snapshot=previous_snapshot,
            new_snapshot=new_snapshot,
            confidence=confidence,
            evidence_count=evidence_count,
            notes=notes,
            evolved_at=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Consistency Report
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """A single piece of evidence observed by the Consistency Engine."""

    source: str
    """Subsystem or component that provided this evidence."""

    signal_type: str
    """Classification of signal (e.g. ``"behavior_observation"``)."""

    weight: float
    """0.0-1.0 weight of this evidence in consistency calculations."""

    description: str
    """Human-readable description of the observation."""

    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("EvidenceItem.weight must be between 0.0 and 1.0.")


@dataclass
class ConsistencyReport:
    """Output of a Consistency Engine analysis run.

    Attributes
    ----------
    overall_score:
        0.0-1.0 stability score (1.0 = perfectly consistent).
    dimension_scores:
        Per-dimension breakdown (e.g. ``{"goals": 0.9, "identity": 0.85}``).
    drift_detected:
        True if the engine detected potential identity drift.
    drift_fields:
        Fields where drift was observed.
    evidence_items:
        Evidence items considered during this run.
    recommendation:
        Human-readable recommendation (e.g. ``"No action required"``).
    checked_at:
        UTC timestamp of this check.
    """

    overall_score: float
    dimension_scores: dict[str, float]
    drift_detected: bool
    drift_fields: list[str]
    evidence_items: list[EvidenceItem]
    recommendation: str
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not (0.0 <= self.overall_score <= 1.0):
            raise ValueError(
                "ConsistencyReport.overall_score must be between 0.0 and 1.0."
            )

    @property
    def is_stable(self) -> bool:
        """Return True if no drift was detected and score is above threshold."""
        return not self.drift_detected and self.overall_score >= 0.6


# ---------------------------------------------------------------------------
# Future Self Reference
# ---------------------------------------------------------------------------


@dataclass
class FutureSelfReference:
    """Lightweight pointer to a modelled future version of the user.

    Future Self *models* are owned by ASTRA; future *scenarios* are owned
    by JANUS.  This reference captures the headline attributes of the future
    model without duplicating the scenario reasoning.

    Attributes
    ----------
    horizon_label:
        Human-readable horizon label (e.g. ``"1-year"``, ``"5-year"``).
    horizon_years:
        Numeric horizon in years.
    headline_identity:
        One-sentence identity statement for this future version.
    projected_goals:
        Key goals expected to be active or completed at this horizon.
    growth_trajectory:
        Directional growth indicators at this horizon.
    confidence:
        0.0-1.0 confidence in this projection.
    last_updated:
        UTC timestamp of the most recent projection update.
    metadata:
        Extensible annotations.
    """

    horizon_label: str
    horizon_years: float
    headline_identity: str
    projected_goals: list[str] = field(default_factory=list)
    growth_trajectory: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.5
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon_years <= 0:
            raise ValueError("FutureSelfReference.horizon_years must be positive.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                "FutureSelfReference.confidence must be between 0.0 and 1.0."
            )