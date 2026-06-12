# subsystems/echo/patterns.py
"""
ECHO v1 Pattern Extraction Engine.

Implements :class:`PatternExtractionEngine` — the production engine
responsible for discovering recurring behavioral, operational, cognitive,
and experiential patterns across ECHO's episodic memory store.

The Pattern Extraction Engine observes patterns *in* ECHO's experience
store but does not own those patterns — discovered patterns are published
via domain events for ASTRA to consume and own.  ECHO stores events;
ASTRA stores patterns.  This engine sits at the boundary.

Design Principles
-----------------
* **Read-only over ECHO data**: This engine never writes new experiences.
  All analysis is performed over the existing store via the injected
  engine interfaces.
* **Deterministic scoring**: Confidence scores are computed from first
  principles (occurrence frequency, temporal regularity, tag coherence)
  with no randomness.  Identical inputs always produce identical scores.
* **Minimum occurrence gate**: A candidate sequence must appear at least
  ``min_occurrences`` times before it is elevated to a detected pattern.
* **ECHO Boundary Law**: Patterns are NOT stored in ECHO.  They are held
  in an in-process ``_pattern_store`` pending publication to ASTRA, or
  returned directly to the caller.
* **Thread-safe**: All public methods serialise concurrent access via
  ``self._lock`` (a :class:`threading.RLock`).
* **Lifecycle-gated**: Every public operation guards against calls made
  before :meth:`initialize` or after :meth:`shutdown`.

Pattern Categories
------------------
* :class:`AchievementPattern`  — recurring achievement clusters.
* :class:`FailurePattern`      — recurring failure modes.
* :class:`HabitPattern`        — consistently repeated behavioral sequences.
* :class:`BehavioralPattern`   — broader recurring behavioral tendencies.
* :class:`ProjectPattern`      — evolution and lifecycle patterns within projects.
* :class:`TemporalPattern`     — time-of-day / day-of-week / seasonal recurrence.
* :class:`GrowthPattern`       — skill and capability improvement trajectories.

Pattern Health
--------------
The engine exposes :meth:`get_pattern_health_metrics` which aggregates
per-category confidence, coverage, and recency into a single
``PatternHealthReport``.

Integration
-----------
Requires:
* :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
  for full-store access and individual record fetches.
* :class:`~subsystems.echo.interfaces.ExperienceRetrievalEngineInterface`
  for semantically-scoped recall, tag-based lookup, and importance filters.

Optionally accepts:
* :class:`~subsystems.echo.reflection.ReflectionEngine`
  for cross-referencing generated lessons with detected patterns.
* :class:`~subsystems.echo.context_reconstruction.ContextReconstructionEngine`
  for timeline-aware antecedent/consequent traversal during project and
  growth pattern analysis.
* :class:`~subsystems.echo.episodic_index.EpisodicIndexEngine`
  for O(1) tag-cardinality lookups and temporal bucket enumeration.
* :class:`~subsystems.echo.integrity.MemoryIntegrityEngine`
  for pre-scan integrity validation and broken-reference awareness.

Thread Safety
-------------
All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
before accessing shared mutable state.  The engine is safe for concurrent
access across multiple POLARIS threads.

Architecture Note
-----------------
v1 is a pure in-memory implementation.  All pattern state is derived on
demand from the experience store.  A ``_pattern_store`` caches the most
recently extracted set of patterns; callers wishing a fresh scan should
call :meth:`extract_patterns` rather than :meth:`get_known_patterns`.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto, unique
from typing import Any

from subsystems.echo.exceptions import EchoNotInitializedError
from subsystems.echo.interfaces import (
    ExperienceEngineInterface,
    ExperienceRetrievalEngineInterface,
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

_DEFAULT_MIN_OCCURRENCES: int = 3
_DEFAULT_LOOKBACK_DAYS: int = 90
_MAX_STORE_FETCH: int = 100_000

# Tag co-occurrence minimum joint frequency before the pair is considered
_TAG_COOCCURRENCE_MIN: int = 2

# Temporal pattern buckets
_HOUR_BUCKET_COUNT: int = 24
_DOW_BUCKET_COUNT: int = 7  # 0 = Monday … 6 = Sunday

# Growth trajectory minimum data-points (achievements) per domain
_GROWTH_MIN_DATAPOINTS: int = 3

# Confidence score component weights (must sum to 1.0 within each formula)
_CONF_FREQUENCY_WEIGHT: float = 0.45
_CONF_RECENCY_WEIGHT: float = 0.25
_CONF_CONSISTENCY_WEIGHT: float = 0.30

# Recency half-life for confidence decay (days)
_RECENCY_HALF_LIFE_DAYS: float = 60.0


# ---------------------------------------------------------------------------
# Pattern Category Enumeration
# ---------------------------------------------------------------------------


@unique
class PatternCategory(Enum):
    """Classification of a detected memory pattern.

    Each member corresponds to a distinct analysis pass in the
    :class:`PatternExtractionEngine`.
    """

    ACHIEVEMENT = auto()
    """Recurring achievement cluster — similar accomplishments over time."""

    FAILURE = auto()
    """Recurring failure mode — the same class of failure repeated."""

    HABIT = auto()
    """Consistent behavioral repetition — a sequence that fires regularly."""

    BEHAVIORAL = auto()
    """Broader recurring behavioral tendency not captured by habit."""

    PROJECT = auto()
    """Project lifecycle pattern — evolution within a project arc."""

    TEMPORAL = auto()
    """Time-based recurrence — hour, day-of-week, or seasonal regularity."""

    GROWTH = auto()
    """Skill or capability improvement trajectory across time."""


# ---------------------------------------------------------------------------
# Pattern Data Structures
# ---------------------------------------------------------------------------


@dataclass
class PatternEvidence:
    """A single piece of evidence supporting a detected pattern.

    Attributes
    ----------
    experience_id:
        UUID of the :class:`~subsystems.echo.models.Experience` that
        contributes this evidence.
    occurred_at:
        UTC ISO-8601 timestamp of the experience.
    relevance:
        Float in ``[0.0, 1.0]`` — how strongly this experience supports
        the pattern claim.
    note:
        Human-readable annotation explaining why this record is evidence.
    """

    experience_id: str
    occurred_at: str
    relevance: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError(
                f"PatternEvidence.relevance must be in [0.0, 1.0]; "
                f"got {self.relevance!r}."
            )


@dataclass
class DetectedPattern:
    """A recurring pattern discovered by the :class:`PatternExtractionEngine`.

    Attributes
    ----------
    pattern_id:
        UUID-4 unique identifier for this pattern instance.
    category:
        :class:`PatternCategory` classification.
    label:
        Short human-readable name for the pattern
        (e.g. ``"Late-stage scope creep failures"``).
    description:
        Detailed narrative of what the pattern describes and why it matters.
    confidence:
        Confidence score in ``[0.0, 1.0]``.  ``1.0`` = maximum certainty
        based on frequency, recency, and temporal consistency.
    occurrence_count:
        Number of times this pattern has been observed in the experience
        window.
    first_seen_at:
        UTC ISO-8601 string of the earliest evidence record.
    last_seen_at:
        UTC ISO-8601 string of the most recent evidence record.
    evidence:
        List of :class:`PatternEvidence` items that support this pattern.
    tags:
        Tag name strings that characterise this pattern's domain.
    related_pattern_ids:
        UUIDs of other :class:`DetectedPattern` objects that are
        thematically or temporally linked to this one.
    extra:
        Extensible key-value store for category-specific metadata.
    detected_at:
        UTC ISO-8601 string of when this pattern was extracted.
    """

    label: str
    category: PatternCategory
    confidence: float
    occurrence_count: int
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    first_seen_at: str = ""
    last_seen_at: str = ""
    evidence: list[PatternEvidence] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    related_pattern_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise ValueError("DetectedPattern.label must be a non-empty string.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"DetectedPattern.confidence must be in [0.0, 1.0]; "
                f"got {self.confidence!r}."
            )
        if self.occurrence_count < 0:
            raise ValueError(
                f"DetectedPattern.occurrence_count must be >= 0; "
                f"got {self.occurrence_count!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise this pattern to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            All pattern fields in a JSON-compatible structure, suitable for
            publication to ASTRA via the domain event bus.
        """
        return {
            "pattern_id": self.pattern_id,
            "category": self.category.name,
            "label": self.label,
            "description": self.description,
            "confidence": round(self.confidence, 6),
            "occurrence_count": self.occurrence_count,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "evidence": [
                {
                    "experience_id": e.experience_id,
                    "occurred_at": e.occurred_at,
                    "relevance": round(e.relevance, 6),
                    "note": e.note,
                }
                for e in self.evidence
            ],
            "tags": self.tags,
            "related_pattern_ids": self.related_pattern_ids,
            "extra": self.extra,
            "detected_at": self.detected_at,
        }


# ---------------------------------------------------------------------------
# Typed pattern aliases
# These are concrete DetectedPattern instances — the category field carries
# the sub-type identity.  The following dataclasses extend DetectedPattern
# with category-specific fields packed into the ``extra`` dict via their
# own constructors so that the uniform DetectedPattern wire format is
# always preserved.
# ---------------------------------------------------------------------------


@dataclass
class AchievementPattern(DetectedPattern):
    """A :class:`DetectedPattern` representing a recurring achievement cluster.

    Attributes (category-specific, stored in ``extra``)
    ---------------------------------------------------
    domain:
        Broad domain of the achievement cluster
        (e.g. ``"software"``, ``"academic"``).
    achievement_types:
        Most common ExperienceType names among the clustered records.
    milestone_ids:
        experience_ids of the most significant achievements in the cluster.
    """

    domain: str = "general"
    achievement_types: list[str] = field(default_factory=list)
    milestone_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.category = PatternCategory.ACHIEVEMENT
        self.extra.setdefault("domain", self.domain)
        self.extra.setdefault("achievement_types", self.achievement_types)
        self.extra.setdefault("milestone_ids", self.milestone_ids)


@dataclass
class FailurePattern(DetectedPattern):
    """A :class:`DetectedPattern` representing a recurring failure mode.

    Attributes (category-specific, stored in ``extra``)
    ---------------------------------------------------
    failure_mode:
        Short label for the failure class
        (e.g. ``"scope creep"``).
    average_severity:
        Mean :class:`~subsystems.echo.models.ExperienceImportance` ordinal
        across all failure evidence records.
    root_cause_tags:
        Tags most commonly co-occurring with these failures.
    """

    failure_mode: str = "unknown"
    average_severity: float = 0.0
    root_cause_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.category = PatternCategory.FAILURE
        self.extra.setdefault("failure_mode", self.failure_mode)
        self.extra.setdefault("average_severity", self.average_severity)
        self.extra.setdefault("root_cause_tags", self.root_cause_tags)


@dataclass
class HabitPattern(DetectedPattern):
    """A :class:`DetectedPattern` representing a consistently repeated habit.

    Attributes (category-specific, stored in ``extra``)
    ---------------------------------------------------
    habit_description:
        Human-readable description of the habitual behaviour.
    average_interval_days:
        Mean days between occurrences.
    interval_std_days:
        Standard deviation of the inter-occurrence interval in days.
        Low value = highly regular habit.
    dominant_hour_bucket:
        Hour-of-day (0–23) at which this habit most commonly occurs,
        or ``None`` when not time-concentrated.
    dominant_dow_bucket:
        Day-of-week (0=Monday … 6=Sunday) at which this habit most
        commonly occurs, or ``None`` when not day-concentrated.
    """

    habit_description: str = ""
    average_interval_days: float = 0.0
    interval_std_days: float = 0.0
    dominant_hour_bucket: int | None = None
    dominant_dow_bucket: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.category = PatternCategory.HABIT
        self.extra.setdefault("habit_description", self.habit_description)
        self.extra.setdefault("average_interval_days", self.average_interval_days)
        self.extra.setdefault("interval_std_days", self.interval_std_days)
        self.extra.setdefault("dominant_hour_bucket", self.dominant_hour_bucket)
        self.extra.setdefault("dominant_dow_bucket", self.dominant_dow_bucket)


@dataclass
class BehavioralPattern(DetectedPattern):
    """A :class:`DetectedPattern` capturing a recurring behavioral tendency.

    Attributes (category-specific, stored in ``extra``)
    ---------------------------------------------------
    behavior_summary:
        One-sentence description of the behavioral tendency.
    trigger_tags:
        Tags that most reliably precede this behavior in the experience
        timeline.
    outcome_quality:
        Aggregate outcome quality: ``"positive"``, ``"negative"``, or
        ``"mixed"``.
    """

    behavior_summary: str = ""
    trigger_tags: list[str] = field(default_factory=list)
    outcome_quality: str = "mixed"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.outcome_quality not in {"positive", "negative", "mixed"}:
            raise ValueError(
                "BehavioralPattern.outcome_quality must be "
                "'positive', 'negative', or 'mixed'."
            )
        self.category = PatternCategory.BEHAVIORAL
        self.extra.setdefault("behavior_summary", self.behavior_summary)
        self.extra.setdefault("trigger_tags", self.trigger_tags)
        self.extra.setdefault("outcome_quality", self.outcome_quality)


@dataclass
class ProjectPattern(DetectedPattern):
    """A :class:`DetectedPattern` representing evolution within a project arc.

    Attributes (category-specific, stored in ``extra``)
    ---------------------------------------------------
    project_ref:
        The project reference string (from ``Experience.metadata.project_refs``).
    phase_sequence:
        Ordered list of ExperienceType names observed through the project
        lifecycle.
    total_duration_days:
        Total span from first to last project experience in days.
    experience_density:
        Average number of experiences per week within the project window.
    """

    project_ref: str = ""
    phase_sequence: list[str] = field(default_factory=list)
    total_duration_days: float = 0.0
    experience_density: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.category = PatternCategory.PROJECT
        self.extra.setdefault("project_ref", self.project_ref)
        self.extra.setdefault("phase_sequence", self.phase_sequence)
        self.extra.setdefault("total_duration_days", self.total_duration_days)
        self.extra.setdefault("experience_density", self.experience_density)


@dataclass
class TemporalPattern(DetectedPattern):
    """A :class:`DetectedPattern` representing time-based recurrence.

    Attributes (category-specific, stored in ``extra``)
    ---------------------------------------------------
    recurrence_type:
        One of ``"hourly_cluster"``, ``"daily_cluster"``,
        ``"weekly_cluster"``, or ``"seasonal_cluster"``.
    peak_hour:
        Hour of day (0–23) with highest concentration, or ``None``.
    peak_dow:
        Day of week (0=Monday … 6=Sunday) with highest concentration,
        or ``None``.
    peak_month:
        Month number (1–12) with highest concentration, or ``None``.
    concentration_ratio:
        Fraction of total occurrences that fall within the peak bucket.
        Higher = more concentrated temporal pattern.
    """

    recurrence_type: str = "weekly_cluster"
    peak_hour: int | None = None
    peak_dow: int | None = None
    peak_month: int | None = None
    concentration_ratio: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        valid_types = {
            "hourly_cluster",
            "daily_cluster",
            "weekly_cluster",
            "seasonal_cluster",
        }
        if self.recurrence_type not in valid_types:
            raise ValueError(
                f"TemporalPattern.recurrence_type must be one of "
                f"{sorted(valid_types)!r}; got {self.recurrence_type!r}."
            )
        self.category = PatternCategory.TEMPORAL
        self.extra.setdefault("recurrence_type", self.recurrence_type)
        self.extra.setdefault("peak_hour", self.peak_hour)
        self.extra.setdefault("peak_dow", self.peak_dow)
        self.extra.setdefault("peak_month", self.peak_month)
        self.extra.setdefault("concentration_ratio", self.concentration_ratio)


@dataclass
class GrowthPattern(DetectedPattern):
    """A :class:`DetectedPattern` representing a skill improvement trajectory.

    Attributes (category-specific, stored in ``extra``)
    ---------------------------------------------------
    domain:
        Domain in which growth is being tracked
        (e.g. ``"software"``, ``"communication"``).
    growth_rate:
        Positive slope coefficient of the linear importance trend over the
        observation window.  In units of (importance_ordinal / day).
    milestone_count:
        Number of HIGH-or-CRITICAL importance experiences contributing to
        the trajectory.
    trajectory_direction:
        ``"improving"``, ``"plateauing"``, or ``"declining"`` based on the
        sign and magnitude of ``growth_rate``.
    """

    domain: str = "general"
    growth_rate: float = 0.0
    milestone_count: int = 0
    trajectory_direction: str = "improving"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.trajectory_direction not in {"improving", "plateauing", "declining"}:
            raise ValueError(
                "GrowthPattern.trajectory_direction must be 'improving', "
                "'plateauing', or 'declining'."
            )
        self.category = PatternCategory.GROWTH
        self.extra.setdefault("domain", self.domain)
        self.extra.setdefault("growth_rate", self.growth_rate)
        self.extra.setdefault("milestone_count", self.milestone_count)
        self.extra.setdefault(
            "trajectory_direction", self.trajectory_direction
        )


# ---------------------------------------------------------------------------
# Pattern Audit Report
# ---------------------------------------------------------------------------


@dataclass
class PatternAuditReport:
    """Structured summary produced by :meth:`PatternExtractionEngine.audit_patterns`.

    Attributes
    ----------
    total_patterns_detected:
        Total number of :class:`DetectedPattern` instances in the most
        recent extraction run.
    patterns_by_category:
        Mapping of :attr:`PatternCategory.name` → count.
    high_confidence_count:
        Number of patterns with confidence ``>= 0.75``.
    low_confidence_count:
        Number of patterns with confidence ``< 0.40``.
    average_confidence:
        Mean confidence score across all detected patterns.  ``0.0`` when
        no patterns have been extracted.
    most_recent_evidence_at:
        UTC ISO-8601 string of the most recent evidence timestamp across
        all detected patterns.  Empty string when no patterns exist.
    lookback_days_used:
        The ``lookback_days`` value that produced these patterns.
    min_occurrences_used:
        The ``min_occurrences`` threshold that produced these patterns.
    experiences_scanned:
        Number of experience records examined in the extraction pass.
    audited_at:
        UTC ISO-8601 string of when this audit was generated.
    elapsed_seconds:
        Wall-clock seconds the extraction and audit took to complete.
    errors:
        Non-fatal errors encountered during the extraction run.
    """

    total_patterns_detected: int = 0
    patterns_by_category: dict[str, int] = field(default_factory=dict)
    high_confidence_count: int = 0
    low_confidence_count: int = 0
    average_confidence: float = 0.0
    most_recent_evidence_at: str = ""
    lookback_days_used: int = _DEFAULT_LOOKBACK_DAYS
    min_occurrences_used: int = _DEFAULT_MIN_OCCURRENCES
    experiences_scanned: int = 0
    audited_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Return ``True`` when no errors were recorded during extraction."""
        return len(self.errors) == 0

    @property
    def total_patterns(self) -> int:
        """Alias for :attr:`total_patterns_detected`."""
        return self.total_patterns_detected

    def summary(self) -> str:
        """Return a single-line human-readable summary."""
        status = "HEALTHY" if self.is_healthy else "DEGRADED"
        return (
            f"PatternAuditReport [{status}] "
            f"patterns={self.total_patterns_detected} "
            f"avg_confidence={self.average_confidence:.3f} "
            f"high_conf={self.high_confidence_count} "
            f"scanned={self.experiences_scanned} "
            f"elapsed={self.elapsed_seconds:.3f}s"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Pattern Health Metrics
# ---------------------------------------------------------------------------


@dataclass
class PatternHealthMetrics:
    """Per-category health and coverage metrics for the pattern store.

    Attributes
    ----------
    total_patterns:
        Total number of cached :class:`DetectedPattern` objects.
    category_metrics:
        Mapping of :attr:`PatternCategory.name` → sub-metric dict
        containing ``count``, ``avg_confidence``, ``max_confidence``,
        ``min_confidence``, and ``avg_occurrence_count``.
    overall_confidence:
        Weighted average confidence across all patterns.
    store_freshness_hours:
        Age of the most recent extraction run in hours.  ``None`` when
        no extraction has ever been performed.
    generated_at:
        UTC ISO-8601 string of metric generation.
    """

    total_patterns: int = 0
    category_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall_confidence: float = 0.0
    store_freshness_hours: float | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        """Return a single-line human-readable summary."""
        freshness = (
            f"{self.store_freshness_hours:.1f}h"
            if self.store_freshness_hours is not None
            else "never"
        )
        return (
            f"PatternHealthMetrics "
            f"total={self.total_patterns} "
            f"overall_confidence={self.overall_confidence:.3f} "
            f"freshness={freshness}"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IMPORTANCE_ORDINAL: dict[ExperienceImportance, int] = {
    ExperienceImportance.LOW: 1,
    ExperienceImportance.MEDIUM: 2,
    ExperienceImportance.HIGH: 3,
    ExperienceImportance.CRITICAL: 4,
}


def _recency_factor(occurred_at: datetime) -> float:
    """Return a ``[0.0, 1.0]`` recency multiplier.

    Uses exponential decay with :data:`_RECENCY_HALF_LIFE_DAYS` as the
    half-life.  More recent timestamps produce values closer to ``1.0``.

    Parameters
    ----------
    occurred_at:
        UTC-aware datetime of the experience.

    Returns
    -------
    float
        Recency multiplier in ``[0.0, 1.0]``.
    """
    now = datetime.now(timezone.utc)
    delta_days = (now - occurred_at).total_seconds() / 86_400.0
    return math.pow(2.0, -delta_days / _RECENCY_HALF_LIFE_DAYS)


def _population_std(values: list[float]) -> float:
    """Return the population standard deviation of *values*.

    Returns ``0.0`` when fewer than two values are supplied.

    Parameters
    ----------
    values:
        Numeric sample.

    Returns
    -------
    float
        Population standard deviation.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def _linear_slope(x_values: list[float], y_values: list[float]) -> float:
    """Compute the least-squares linear slope of ``y = f(x)``.

    Returns ``0.0`` when the inputs have fewer than two points or when
    the variance of ``x_values`` is zero.

    Parameters
    ----------
    x_values:
        Independent variable (typically elapsed days).
    y_values:
        Dependent variable (typically importance ordinal).

    Returns
    -------
    float
        Slope coefficient.
    """
    n = len(x_values)
    if n < 2:
        return 0.0
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    var_x = sum((x - mean_x) ** 2 for x in x_values)
    if var_x == 0.0:
        return 0.0
    return cov / var_x


def _confidence_score(
    occurrence_count: int,
    min_occurrences: int,
    recency_values: list[float],
    interval_days: list[float],
) -> float:
    """Compute a deterministic pattern confidence score in ``[0.0, 1.0]``.

    The score combines three independent signals:

    * **Frequency signal** — how many more times than the minimum threshold
      this pattern has been observed.
    * **Recency signal**   — weighted mean recency of the evidence records.
    * **Consistency signal** — regularity of the inter-occurrence interval
      (low coefficient of variation = high consistency).

    Parameters
    ----------
    occurrence_count:
        Total number of times the pattern has been observed.
    min_occurrences:
        Minimum threshold — patterns at exactly this count score near zero.
    recency_values:
        Per-evidence recency multipliers from :func:`_recency_factor`.
    interval_days:
        List of inter-occurrence interval lengths in days.  May be empty
        for first-occurrence events.

    Returns
    -------
    float
        Confidence score clamped to ``[0.0, 1.0]``.
    """
    if occurrence_count < min_occurrences:
        return 0.0

    # Frequency: saturates at 10× the minimum threshold
    freq_norm = min(
        1.0,
        (occurrence_count - min_occurrences) / max(1, min_occurrences * 9),
    )
    frequency_signal = freq_norm

    # Recency: mean of per-evidence recency multipliers
    recency_signal = (
        sum(recency_values) / len(recency_values) if recency_values else 0.0
    )

    # Consistency: 1 − (coefficient of variation), floored at 0
    if len(interval_days) >= 2:
        mean_interval = sum(interval_days) / len(interval_days)
        std_interval = _population_std(interval_days)
        cv = std_interval / mean_interval if mean_interval > 0.0 else 1.0
        consistency_signal = max(0.0, 1.0 - cv)
    else:
        # Single gap or no gap — treat as uncertain
        consistency_signal = 0.5

    score = (
        _CONF_FREQUENCY_WEIGHT * frequency_signal
        + _CONF_RECENCY_WEIGHT * recency_signal
        + _CONF_CONSISTENCY_WEIGHT * consistency_signal
    )
    return max(0.0, min(1.0, score))


def _iso(dt: datetime) -> str:
    """Return a UTC ISO-8601 string for *dt*."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# PatternExtractionEngine
# ---------------------------------------------------------------------------


class PatternExtractionEngine:
    """Production implementation of the ECHO Pattern Extraction Engine.

    Discovers recurring behavioral, operational, cognitive, and experiential
    patterns across ECHO's episodic memory store.  Patterns are detected via
    seven independent analysis passes — one per :class:`PatternCategory` —
    each consuming experiences from the injected engine interfaces.

    ECHO Boundary Law: Detected patterns are **not** stored in ECHO.  They
    are held in an in-process ``_pattern_store`` for publication to ASTRA
    and are surfaced to callers via :meth:`extract_patterns` and
    :meth:`get_known_patterns`.

    Parameters
    ----------
    experience_engine:
        A running :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        that owns ECHO's experience store.
    retrieval_engine:
        A running :class:`~subsystems.echo.interfaces.ExperienceRetrievalEngineInterface`
        for semantically-scoped and tag-based recall.
    reflection_engine:
        Optional :class:`~subsystems.echo.reflection.ReflectionEngine`
        used to cross-reference generated lessons with detected patterns.
        When ``None``, reflection-backed evidence is omitted from pattern
        clustering.
    context_engine:
        Optional :class:`~subsystems.echo.context_reconstruction.ContextReconstructionEngine`
        used for timeline-aware traversal during project and growth analysis.
        When ``None``, temporal traversal falls back to raw experience
        ordering from the experience engine.
    episodic_index:
        Optional :class:`~subsystems.echo.episodic_index.EpisodicIndexEngine`
        for O(1) tag cardinality lookups.  When ``None``, tag statistics are
        computed inline from the raw experience set.
    integrity_engine:
        Optional :class:`~subsystems.echo.integrity.MemoryIntegrityEngine`
        consulted before each extraction run to skip broken-reference
        records.  When ``None``, all records are included.
    default_min_occurrences:
        Engine-level default for the minimum occurrence threshold applied
        when callers do not supply an explicit ``min_occurrences`` argument.
        Must be ``>= 2``.
    default_lookback_days:
        Engine-level default for the lookback window in days.  Must be
        ``>= 1``.

    Usage
    -----
    ::

        from subsystems.echo.patterns import PatternExtractionEngine

        engine = PatternExtractionEngine(
            experience_engine=exp_engine,
            retrieval_engine=ret_engine,
        )
        engine.initialize()

        patterns = engine.extract_patterns(
            min_occurrences=3,
            lookback_days=90,
        )
        for p in patterns:
            print(p.label, p.confidence)

        report = engine.audit_patterns()
        print(report.summary())

        engine.shutdown()
    """

    def __init__(
        self,
        experience_engine: ExperienceEngineInterface,
        retrieval_engine: ExperienceRetrievalEngineInterface,
        reflection_engine: Any | None = None,
        context_engine: Any | None = None,
        episodic_index: Any | None = None,
        integrity_engine: Any | None = None,
        *,
        default_min_occurrences: int = _DEFAULT_MIN_OCCURRENCES,
        default_lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        if default_min_occurrences < 2:
            raise ValueError(
                "default_min_occurrences must be >= 2; "
                f"got {default_min_occurrences!r}."
            )
        if default_lookback_days < 1:
            raise ValueError(
                "default_lookback_days must be >= 1; "
                f"got {default_lookback_days!r}."
            )

        self._experience_engine = experience_engine
        self._retrieval_engine = retrieval_engine
        self._reflection_engine = reflection_engine
        self._context_engine = context_engine
        self._episodic_index = episodic_index
        self._integrity_engine = integrity_engine
        self._default_min_occurrences = default_min_occurrences
        self._default_lookback_days = default_lookback_days

        # pattern_id → DetectedPattern (pending publication to ASTRA)
        self._pattern_store: dict[str, DetectedPattern] = {}

        # UTC timestamp of the most recent extract_patterns call; None until run.
        self._last_extraction_at: datetime | None = None

        # Parameters used in the most recent extraction run
        self._last_min_occurrences: int = default_min_occurrences
        self._last_lookback_days: int = default_lookback_days
        self._last_experiences_scanned: int = 0

        self._lock: threading.RLock = threading.RLock()
        self._running: bool = False

        _logger.debug(
            "PatternExtractionEngine constructed "
            "(min_occurrences=%d, lookback_days=%d).",
            default_min_occurrences,
            default_lookback_days,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Both :attr:`_experience_engine` and :attr:`_retrieval_engine` must
        already be initialised and running before this call.

        Raises
        ------
        EchoError
            If initialisation fails for any reason.
        """
        with self._lock:
            if self._running:
                _logger.warning(
                    "PatternExtractionEngine.initialize() called while "
                    "already running."
                )
                return
            self._running = True
            _logger.info("PatternExtractionEngine initialised.")

    def shutdown(self) -> None:
        """Release resources and mark the engine as stopped.

        Clears the in-process pattern store.  Any patterns that have not
        been published to ASTRA will be lost after this call.
        """
        with self._lock:
            if not self._running:
                return
            self._pattern_store.clear()
            self._running = False
            _logger.info("PatternExtractionEngine shut down.")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if the engine is not running."""
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _fetch_window(self, lookback_days: int) -> list[Experience]:
        """Return all experiences within the lookback window.

        Uses :meth:`~subsystems.echo.interfaces.ExperienceEngineInterface.query_experiences`
        with an ``occurred_after`` filter.  Returns an empty list and logs
        the error on failure so that partial extraction can still proceed.

        Parameters
        ----------
        lookback_days:
            How many calendar days back to include.

        Returns
        -------
        list[Experience]
            Experiences within the window, ordered by ``occurred_at``
            descending.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        try:
            return self._experience_engine.query_experiences(
                occurred_after=cutoff,
                limit=_MAX_STORE_FETCH,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "PatternExtractionEngine: failed to fetch window "
                "(lookback=%d days): %s",
                lookback_days,
                exc,
            )
            return []

    def _broken_ids(self) -> set[str]:
        """Return the set of experience IDs known to be broken references.

        When an :attr:`_integrity_engine` is available, runs a lightweight
        integrity score check; if degraded, retrieves the full broken
        reference map.  Returns an empty set when no integrity engine is
        injected or the check itself fails.

        Returns
        -------
        set[str]
            experience_ids that should be excluded from pattern analysis.
        """
        if self._integrity_engine is None:
            return set()
        try:
            report = self._integrity_engine.run_audit()
            broken: set[str] = set()
            for source_id, missing_refs in report.broken_references.items():
                broken.add(source_id)
                broken.update(missing_refs)
            return broken
        except Exception as exc:  # noqa: BLE001
            _logger.debug(
                "PatternExtractionEngine: integrity pre-scan failed "
                "(non-fatal): %s",
                exc,
            )
            return set()

    def _store_pattern(self, pattern: DetectedPattern) -> None:
        """Insert or replace a pattern in the in-process store.

        Patterns with the same ``pattern_id`` overwrite previous versions.
        All patterns in the store are awaiting publication to ASTRA.

        Parameters
        ----------
        pattern:
            :class:`DetectedPattern` to cache.
        """
        self._pattern_store[pattern.pattern_id] = pattern

    # ------------------------------------------------------------------
    # Primary extraction entry point
    # ------------------------------------------------------------------

    def extract_patterns(
        self,
        *,
        min_occurrences: int | None = None,
        lookback_days: int | None = None,
        categories: list[PatternCategory] | None = None,
    ) -> list[DetectedPattern]:
        """Scan recent experiences for recurring patterns across all categories.

        Runs seven independent analysis passes — one per
        :class:`PatternCategory` — over all experiences within the lookback
        window.  Results are cached in the internal pattern store and also
        returned directly to the caller.

        ECHO Boundary Law: the returned patterns are NOT stored in ECHO.
        They are intended for publication to ASTRA via the domain event bus.

        Parameters
        ----------
        min_occurrences:
            Minimum number of times a candidate must appear before it is
            elevated to a detected pattern.  Defaults to
            :attr:`_default_min_occurrences`.  Must be ``>= 2``.
        lookback_days:
            How many days of experience history to scan.  Defaults to
            :attr:`_default_lookback_days`.  Must be ``>= 1``.
        categories:
            If supplied, only the specified :class:`PatternCategory` passes
            are executed.  When ``None``, all seven passes run.

        Returns
        -------
        list[DetectedPattern]
            All detected :class:`DetectedPattern` instances from this run,
            sorted by confidence descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ValueError
            If ``min_occurrences < 2`` or ``lookback_days < 1``.
        """
        self._assert_running("extract_patterns")

        min_occ = min_occurrences if min_occurrences is not None else self._default_min_occurrences
        lb_days = lookback_days if lookback_days is not None else self._default_lookback_days

        if min_occ < 2:
            raise ValueError(
                f"min_occurrences must be >= 2; got {min_occ!r}."
            )
        if lb_days < 1:
            raise ValueError(
                f"lookback_days must be >= 1; got {lb_days!r}."
            )

        active_categories: set[PatternCategory] = (
            set(categories) if categories is not None else set(PatternCategory)
        )

        with self._lock:
            start_ts = time.monotonic()
            _logger.info(
                "PatternExtractionEngine: starting extraction "
                "(min_occurrences=%d, lookback_days=%d, categories=%s).",
                min_occ,
                lb_days,
                [c.name for c in sorted(active_categories, key=lambda c: c.name)],
            )

            # Fetch window and filter broken references
            window = self._fetch_window(lb_days)
            broken = self._broken_ids()
            if broken:
                before = len(window)
                window = [e for e in window if e.experience_id not in broken]
                _logger.debug(
                    "PatternExtractionEngine: excluded %d broken-reference "
                    "records from analysis window.",
                    before - len(window),
                )

            # Clear previous results — each call produces a fresh scan
            self._pattern_store.clear()

            errors: list[str] = []
            all_patterns: list[DetectedPattern] = []

            pass_map: dict[PatternCategory, Any] = {
                PatternCategory.ACHIEVEMENT: self._extract_achievement_patterns,
                PatternCategory.FAILURE: self._extract_failure_patterns,
                PatternCategory.HABIT: self._extract_habit_patterns,
                PatternCategory.BEHAVIORAL: self._extract_behavioral_patterns,
                PatternCategory.PROJECT: self._extract_project_patterns,
                PatternCategory.TEMPORAL: self._extract_temporal_patterns,
                PatternCategory.GROWTH: self._extract_growth_patterns,
            }

            for category, pass_fn in pass_map.items():
                if category not in active_categories:
                    continue
                try:
                    detected = pass_fn(window, min_occ)
                    for p in detected:
                        self._store_pattern(p)
                    all_patterns.extend(detected)
                    _logger.debug(
                        "PatternExtractionEngine: [%s] detected %d pattern(s).",
                        category.name,
                        len(detected),
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = (
                        f"PatternExtractionEngine: [{category.name}] pass "
                        f"failed: {exc}"
                    )
                    errors.append(msg)
                    _logger.error(msg, exc_info=True)

            # Cluster related patterns across categories
            try:
                self._cluster_related_patterns(all_patterns)
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "PatternExtractionEngine: cross-category clustering "
                    "failed (non-fatal): %s",
                    exc,
                )

            self._last_extraction_at = datetime.now(timezone.utc)
            self._last_min_occurrences = min_occ
            self._last_lookback_days = lb_days
            self._last_experiences_scanned = len(window)

            elapsed = time.monotonic() - start_ts
            _logger.info(
                "PatternExtractionEngine: extraction complete — "
                "%d pattern(s) from %d experience(s) in %.3fs.",
                len(all_patterns),
                len(window),
                elapsed,
            )

            all_patterns.sort(key=lambda p: p.confidence, reverse=True)
            return all_patterns

    # ------------------------------------------------------------------
    # get_known_patterns — return cached results
    # ------------------------------------------------------------------

    def get_known_patterns(
        self,
        *,
        category: PatternCategory | None = None,
        min_confidence: float = 0.0,
    ) -> list[DetectedPattern]:
        """Return the current set of detected patterns from the most recent run.

        Does not trigger a new extraction scan.  Callers that require
        up-to-date results should call :meth:`extract_patterns` first.

        Parameters
        ----------
        category:
            If supplied, only patterns of this :class:`PatternCategory` are
            returned.  When ``None``, all categories are included.
        min_confidence:
            Exclude patterns below this confidence threshold.  Must be in
            ``[0.0, 1.0]``.

        Returns
        -------
        list[DetectedPattern]
            Matching patterns sorted by confidence descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ValueError
            If ``min_confidence`` is outside ``[0.0, 1.0]``.
        """
        self._assert_running("get_known_patterns")

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0]; got {min_confidence!r}."
            )

        with self._lock:
            results = [
                p
                for p in self._pattern_store.values()
                if (category is None or p.category == category)
                and p.confidence >= min_confidence
            ]
            results.sort(key=lambda p: p.confidence, reverse=True)
            return results

    # ------------------------------------------------------------------
    # get_pattern — single pattern lookup
    # ------------------------------------------------------------------

    def get_pattern(self, pattern_id: str) -> DetectedPattern | None:
        """Return a specific cached pattern by its UUID.

        Parameters
        ----------
        pattern_id:
            UUID of the :class:`DetectedPattern` to retrieve.

        Returns
        -------
        DetectedPattern | None
            The matching pattern, or ``None`` if not found.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("get_pattern")

        with self._lock:
            return self._pattern_store.get(pattern_id)

    # ------------------------------------------------------------------
    # audit_patterns
    # ------------------------------------------------------------------

    def audit_patterns(
        self,
        *,
        min_occurrences: int | None = None,
        lookback_days: int | None = None,
    ) -> PatternAuditReport:
        """Run a fresh extraction scan and return a structured audit report.

        Combines a full :meth:`extract_patterns` call with aggregate
        statistics for monitoring and observability.

        Parameters
        ----------
        min_occurrences:
            Minimum occurrence threshold for this audit run.  Defaults to
            :attr:`_default_min_occurrences`.
        lookback_days:
            Lookback window in days.  Defaults to :attr:`_default_lookback_days`.

        Returns
        -------
        PatternAuditReport
            Structured summary of all detected patterns.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("audit_patterns")

        min_occ = min_occurrences if min_occurrences is not None else self._default_min_occurrences
        lb_days = lookback_days if lookback_days is not None else self._default_lookback_days

        start_ts = time.monotonic()
        errors: list[str] = []
        patterns: list[DetectedPattern] = []

        try:
            patterns = self.extract_patterns(
                min_occurrences=min_occ,
                lookback_days=lb_days,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Extraction failed: {exc}")
            _logger.error("PatternExtractionEngine.audit_patterns: %s", exc, exc_info=True)

        elapsed = time.monotonic() - start_ts

        # Aggregate statistics
        by_category: dict[str, int] = defaultdict(int)
        high_conf = 0
        low_conf = 0
        total_conf = 0.0
        most_recent = ""

        for p in patterns:
            by_category[p.category.name] += 1
            total_conf += p.confidence
            if p.confidence >= 0.75:
                high_conf += 1
            if p.confidence < 0.40:
                low_conf += 1
            if p.last_seen_at > most_recent:
                most_recent = p.last_seen_at

        avg_conf = total_conf / len(patterns) if patterns else 0.0

        with self._lock:
            scanned = self._last_experiences_scanned

        return PatternAuditReport(
            total_patterns_detected=len(patterns),
            patterns_by_category=dict(by_category),
            high_confidence_count=high_conf,
            low_confidence_count=low_conf,
            average_confidence=round(avg_conf, 6),
            most_recent_evidence_at=most_recent,
            lookback_days_used=lb_days,
            min_occurrences_used=min_occ,
            experiences_scanned=scanned,
            audited_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(elapsed, 6),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # get_pattern_health_metrics
    # ------------------------------------------------------------------

    def get_pattern_health_metrics(self) -> PatternHealthMetrics:
        """Return per-category health and coverage metrics for the pattern store.

        Operates over the cached pattern store — does not trigger a new
        extraction run.

        Returns
        -------
        PatternHealthMetrics
            Health metrics for the current pattern store state.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("get_pattern_health_metrics")

        with self._lock:
            patterns = list(self._pattern_store.values())

            by_cat: dict[str, list[DetectedPattern]] = defaultdict(list)
            for p in patterns:
                by_cat[p.category.name].append(p)

            category_metrics: dict[str, dict[str, Any]] = {}
            for cat_name, cat_patterns in by_cat.items():
                confs = [p.confidence for p in cat_patterns]
                occs = [p.occurrence_count for p in cat_patterns]
                category_metrics[cat_name] = {
                    "count": len(cat_patterns),
                    "avg_confidence": round(sum(confs) / len(confs), 6),
                    "max_confidence": round(max(confs), 6),
                    "min_confidence": round(min(confs), 6),
                    "avg_occurrence_count": round(
                        sum(occs) / len(occs), 3
                    ),
                }

            all_confs = [p.confidence for p in patterns]
            overall = sum(all_confs) / len(all_confs) if all_confs else 0.0

            freshness: float | None = None
            if self._last_extraction_at is not None:
                delta = datetime.now(timezone.utc) - self._last_extraction_at
                freshness = round(delta.total_seconds() / 3_600.0, 3)

            return PatternHealthMetrics(
                total_patterns=len(patterns),
                category_metrics=category_metrics,
                overall_confidence=round(overall, 6),
                store_freshness_hours=freshness,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )

    # ------------------------------------------------------------------
    # tag_cooccurrence_analysis
    # ------------------------------------------------------------------

    def analyze_tag_cooccurrence(
        self,
        *,
        lookback_days: int | None = None,
        min_joint_frequency: int | None = None,
    ) -> dict[str, Any]:
        """Analyse tag co-occurrence across the experience window.

        Identifies tag pairs that appear together more often than a
        minimum joint frequency threshold.  Useful for discovering
        latent thematic clusters without reference to experience content.

        Parameters
        ----------
        lookback_days:
            How many days of history to scan.  Defaults to
            :attr:`_default_lookback_days`.
        min_joint_frequency:
            Minimum number of experiences in which a pair must co-occur.
            Defaults to :data:`_TAG_COOCCURRENCE_MIN`.

        Returns
        -------
        dict[str, Any]
            Dictionary containing a ``pairs`` key with a list of
            co-occurrence records, each containing:
            ``tag_a``, ``tag_b``, ``joint_count``,
            ``tag_a_count``, ``tag_b_count``,
            ``jaccard_similarity``, sorted by ``joint_count`` descending.
            Also includes ``total_pairs`` and ``lookback_days_used``.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("analyze_tag_cooccurrence")

        lb_days = lookback_days if lookback_days is not None else self._default_lookback_days
        min_joint = min_joint_frequency if min_joint_frequency is not None else _TAG_COOCCURRENCE_MIN

        with self._lock:
            window = self._fetch_window(lb_days)

            # Build tag → experience_id set and co-occurrence counter
            tag_exp_sets: dict[str, set[str]] = defaultdict(set)
            pair_counter: Counter[tuple[str, str]] = Counter()

            for exp in window:
                tag_names = exp.tag_names()
                exp_id = exp.experience_id
                for t in tag_names:
                    tag_exp_sets[t].add(exp_id)

                sorted_tags = sorted(set(tag_names))
                for i, ta in enumerate(sorted_tags):
                    for tb in sorted_tags[i + 1 :]:
                        pair_counter[(ta, tb)] += 1

            pairs: list[dict[str, Any]] = []
            for (ta, tb), joint in pair_counter.items():
                if joint < min_joint:
                    continue
                a_count = len(tag_exp_sets[ta])
                b_count = len(tag_exp_sets[tb])
                union = a_count + b_count - joint
                jaccard = joint / union if union > 0 else 0.0
                pairs.append(
                    {
                        "tag_a": ta,
                        "tag_b": tb,
                        "joint_count": joint,
                        "tag_a_count": a_count,
                        "tag_b_count": b_count,
                        "jaccard_similarity": round(jaccard, 6),
                    }
                )

            pairs.sort(key=lambda r: r["joint_count"], reverse=True)
            return {
                "pairs": pairs,
                "total_pairs": len(pairs),
                "lookback_days_used": lb_days,
                "min_joint_frequency": min_joint,
                "experiences_scanned": len(window),
            }

    # ------------------------------------------------------------------
    # milestone_progression_tracking
    # ------------------------------------------------------------------

    def track_milestone_progression(
        self,
        *,
        lookback_days: int | None = None,
    ) -> list[dict[str, Any]]:
        """Track milestone progression across HIGH and CRITICAL experiences.

        Identifies sequences of HIGH/CRITICAL importance experiences that
        form natural milestone chains by shared tags or project references,
        measuring the elapsed time and importance trend between consecutive
        milestones.

        Parameters
        ----------
        lookback_days:
            How many days of history to scan.  Defaults to
            :attr:`_default_lookback_days`.

        Returns
        -------
        list[dict[str, Any]]
            List of milestone chain descriptors, each containing:
            ``chain_id``, ``label``, ``milestones`` (ordered list of
            experience_id, title, occurred_at, importance), ``total_days``,
            ``milestone_count``, ``avg_days_between_milestones``.
            Sorted by ``milestone_count`` descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("track_milestone_progression")

        lb_days = lookback_days if lookback_days is not None else self._default_lookback_days

        with self._lock:
            window = self._fetch_window(lb_days)
            milestone_exps = [
                e
                for e in window
                if e.importance in (
                    ExperienceImportance.HIGH,
                    ExperienceImportance.CRITICAL,
                )
            ]
            milestone_exps.sort(key=lambda e: e.occurred_at)

            # Group milestones by shared project reference first, then by
            # shared dominant tag when no project ref is present
            chain_map: dict[str, list[Experience]] = defaultdict(list)
            for exp in milestone_exps:
                if exp.metadata.project_refs:
                    for ref in exp.metadata.project_refs:
                        chain_map[f"project:{ref}"].append(exp)
                else:
                    tag_names = exp.tag_names()
                    if tag_names:
                        primary_tag = sorted(tag_names)[0]
                        chain_map[f"tag:{primary_tag}"].append(exp)
                    else:
                        chain_map["ungrouped"].append(exp)

            chains: list[dict[str, Any]] = []
            for chain_key, members in chain_map.items():
                if len(members) < 2:
                    continue
                sorted_members = sorted(members, key=lambda e: e.occurred_at)
                gaps = [
                    (
                        sorted_members[i + 1].occurred_at
                        - sorted_members[i].occurred_at
                    ).total_seconds()
                    / 86_400.0
                    for i in range(len(sorted_members) - 1)
                ]
                total_days = (
                    sorted_members[-1].occurred_at
                    - sorted_members[0].occurred_at
                ).total_seconds() / 86_400.0
                avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
                chains.append(
                    {
                        "chain_id": str(uuid.uuid4()),
                        "label": chain_key,
                        "milestones": [
                            {
                                "experience_id": e.experience_id,
                                "title": e.title,
                                "occurred_at": _iso(e.occurred_at),
                                "importance": e.importance.name,
                            }
                            for e in sorted_members
                        ],
                        "total_days": round(total_days, 3),
                        "milestone_count": len(sorted_members),
                        "avg_days_between_milestones": round(avg_gap, 3),
                    }
                )

            chains.sort(key=lambda c: c["milestone_count"], reverse=True)
            return chains

    # ------------------------------------------------------------------
    # behavioral_trend_analysis
    # ------------------------------------------------------------------

    def analyze_behavioral_trends(
        self,
        *,
        lookback_days: int | None = None,
        min_occurrences: int | None = None,
    ) -> list[dict[str, Any]]:
        """Analyse broad behavioral trends across the experience window.

        Groups experiences by ExperienceType and importance tier over time
        to detect directional shifts (e.g. increasing frequency of
        ACHIEVEMENT experiences or declining frequency of FAILURE experiences).

        Parameters
        ----------
        lookback_days:
            How many days of history to scan.
        min_occurrences:
            Minimum experiences per ExperienceType before a trend is reported.

        Returns
        -------
        list[dict[str, Any]]
            Trend descriptors per ExperienceType, each containing:
            ``experience_type``, ``total_count``, ``trend_direction``
            (``"rising"``, ``"falling"``, ``"stable"``), ``slope``,
            ``first_occurrence``, ``last_occurrence``, ``importance_distribution``.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("analyze_behavioral_trends")

        lb_days = lookback_days if lookback_days is not None else self._default_lookback_days
        min_occ = min_occurrences if min_occurrences is not None else self._default_min_occurrences

        with self._lock:
            window = self._fetch_window(lb_days)
            if not window:
                return []

            window_sorted = sorted(window, key=lambda e: e.occurred_at)
            epoch_start = window_sorted[0].occurred_at

            # Group by ExperienceType
            type_groups: dict[ExperienceType, list[Experience]] = defaultdict(list)
            for exp in window_sorted:
                type_groups[exp.experience_type].append(exp)

            trends: list[dict[str, Any]] = []
            for exp_type, members in type_groups.items():
                if len(members) < min_occ:
                    continue

                x_days = [
                    (e.occurred_at - epoch_start).total_seconds() / 86_400.0
                    for e in members
                ]
                y_ord = [float(_IMPORTANCE_ORDINAL[e.importance]) for e in members]
                slope = _linear_slope(x_days, y_ord)

                if slope > 0.005:
                    direction = "rising"
                elif slope < -0.005:
                    direction = "falling"
                else:
                    direction = "stable"

                imp_dist: dict[str, int] = defaultdict(int)
                for e in members:
                    imp_dist[e.importance.name] += 1

                trends.append(
                    {
                        "experience_type": exp_type.name,
                        "total_count": len(members),
                        "trend_direction": direction,
                        "slope": round(slope, 8),
                        "first_occurrence": _iso(members[0].occurred_at),
                        "last_occurrence": _iso(members[-1].occurred_at),
                        "importance_distribution": dict(imp_dist),
                    }
                )

            trends.sort(key=lambda t: t["total_count"], reverse=True)
            return trends

    # ------------------------------------------------------------------
    # snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of this engine's current state.

        Returns
        -------
        dict[str, Any]
            Contains ``running``, ``engine``, ``cached_pattern_count``,
            ``last_extraction_at``, ``last_min_occurrences``,
            ``last_lookback_days``, and ``last_experiences_scanned``.
        """
        with self._lock:
            return {
                "running": self._running,
                "engine": "PatternExtractionEngine",
                "cached_pattern_count": len(self._pattern_store),
                "last_extraction_at": (
                    _iso(self._last_extraction_at)
                    if self._last_extraction_at is not None
                    else None
                ),
                "last_min_occurrences": self._last_min_occurrences,
                "last_lookback_days": self._last_lookback_days,
                "last_experiences_scanned": self._last_experiences_scanned,
            }

    # ==================================================================
    # Private analysis passes
    # ==================================================================

    # ------------------------------------------------------------------
    # Pass 1: Achievement patterns
    # ------------------------------------------------------------------

    def _extract_achievement_patterns(
        self,
        window: list[Experience],
        min_occurrences: int,
    ) -> list[AchievementPattern]:
        """Detect recurring achievement clusters within *window*.

        Clusters ACHIEVEMENT-typed experiences by shared tag vocabulary
        using a single-link grouping strategy: experiences sharing at least
        one tag name form a cluster candidate.  Clusters meeting the
        minimum occurrence threshold are promoted to :class:`AchievementPattern`
        instances.

        Parameters
        ----------
        window:
            All experiences within the lookback window.
        min_occurrences:
            Minimum cluster size to promote a candidate to a pattern.

        Returns
        -------
        list[AchievementPattern]
            Detected achievement patterns, sorted by confidence descending.
        """
        achievement_exps = [
            e
            for e in window
            if e.experience_type == ExperienceType.ACHIEVEMENT
        ]

        if len(achievement_exps) < min_occurrences:
            return []

        # Group by dominant tag (the alphabetically first tag name)
        tag_groups: dict[str, list[Experience]] = defaultdict(list)
        untagged: list[Experience] = []

        for exp in achievement_exps:
            tag_names = exp.tag_names()
            if tag_names:
                dominant = sorted(tag_names)[0]
                tag_groups[dominant].append(exp)
            else:
                untagged.append(exp)

        # Also group untagged achievements as a catch-all cluster
        if len(untagged) >= min_occurrences:
            tag_groups["__untagged__"] = untagged

        patterns: list[AchievementPattern] = []
        for tag_label, members in tag_groups.items():
            if len(members) < min_occurrences:
                continue

            members_sorted = sorted(members, key=lambda e: e.occurred_at)
            recency_vals = [_recency_factor(e.occurred_at) for e in members]
            gaps = [
                (
                    members_sorted[i + 1].occurred_at
                    - members_sorted[i].occurred_at
                ).total_seconds()
                / 86_400.0
                for i in range(len(members_sorted) - 1)
            ]
            confidence = _confidence_score(
                occurrence_count=len(members),
                min_occurrences=min_occurrences,
                recency_values=recency_vals,
                interval_days=gaps,
            )

            # Collect tag vocabulary across the cluster
            all_cluster_tags: Counter[str] = Counter()
            for e in members:
                all_cluster_tags.update(e.tag_names())
            top_tags = [t for t, _ in all_cluster_tags.most_common(10)]

            # Milestone = highest-importance records
            milestone_ids = [
                e.experience_id
                for e in sorted(
                    members,
                    key=lambda e: _IMPORTANCE_ORDINAL[e.importance],
                    reverse=True,
                )[:5]
            ]

            # Domain heuristic: use the most frequent project_ref, else tag
            project_refs: Counter[str] = Counter()
            for e in members:
                project_refs.update(e.metadata.project_refs)
            domain = (
                project_refs.most_common(1)[0][0]
                if project_refs
                else (tag_label if tag_label != "__untagged__" else "general")
            )

            evidence = [
                PatternEvidence(
                    experience_id=e.experience_id,
                    occurred_at=_iso(e.occurred_at),
                    relevance=_recency_factor(e.occurred_at),
                    note=f"Achievement in cluster '{tag_label}'",
                )
                for e in members_sorted
            ]

            label = (
                f"Recurring achievements: {tag_label}"
                if tag_label != "__untagged__"
                else "Recurring untagged achievements"
            )

            patterns.append(
                AchievementPattern(
                    label=label,
                    category=PatternCategory.ACHIEVEMENT,
                    confidence=confidence,
                    occurrence_count=len(members),
                    description=(
                        f"Achievement cluster with {len(members)} occurrences "
                        f"spanning {tag_label!r} domain tags."
                    ),
                    first_seen_at=_iso(members_sorted[0].occurred_at),
                    last_seen_at=_iso(members_sorted[-1].occurred_at),
                    evidence=evidence,
                    tags=top_tags,
                    domain=domain,
                    achievement_types=[
                        e.experience_type.name for e in members_sorted
                    ],
                    milestone_ids=milestone_ids,
                )
            )

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    # ------------------------------------------------------------------
    # Pass 2: Failure patterns
    # ------------------------------------------------------------------

    def _extract_failure_patterns(
        self,
        window: list[Experience],
        min_occurrences: int,
    ) -> list[FailurePattern]:
        """Detect recurring failure modes within *window*.

        Groups FAILURE-typed experiences by their dominant tag, computing
        average severity and identifying the most common root-cause tags
        (tags that appear disproportionately in the failure cluster
        relative to the full window).

        Parameters
        ----------
        window:
            All experiences within the lookback window.
        min_occurrences:
            Minimum cluster size.

        Returns
        -------
        list[FailurePattern]
            Detected failure patterns, sorted by confidence descending.
        """
        failure_exps = [
            e
            for e in window
            if e.experience_type == ExperienceType.FAILURE
        ]

        if len(failure_exps) < min_occurrences:
            return []

        # Compute global tag frequency across the full window for IDF-like
        # root-cause significance
        global_tag_freq: Counter[str] = Counter()
        for exp in window:
            global_tag_freq.update(exp.tag_names())
        total_window = len(window)

        tag_groups: dict[str, list[Experience]] = defaultdict(list)
        untagged: list[Experience] = []
        for exp in failure_exps:
            tag_names = exp.tag_names()
            if tag_names:
                dominant = sorted(tag_names)[0]
                tag_groups[dominant].append(exp)
            else:
                untagged.append(exp)
        if len(untagged) >= min_occurrences:
            tag_groups["__untagged__"] = untagged

        patterns: list[FailurePattern] = []
        for tag_label, members in tag_groups.items():
            if len(members) < min_occurrences:
                continue

            members_sorted = sorted(members, key=lambda e: e.occurred_at)
            recency_vals = [_recency_factor(e.occurred_at) for e in members]
            gaps = [
                (
                    members_sorted[i + 1].occurred_at
                    - members_sorted[i].occurred_at
                ).total_seconds()
                / 86_400.0
                for i in range(len(members_sorted) - 1)
            ]
            confidence = _confidence_score(
                occurrence_count=len(members),
                min_occurrences=min_occurrences,
                recency_values=recency_vals,
                interval_days=gaps,
            )

            # Average severity
            avg_sev = sum(
                _IMPORTANCE_ORDINAL[e.importance] for e in members
            ) / len(members)

            # Root-cause tags: tags over-represented in this cluster
            cluster_tag_freq: Counter[str] = Counter()
            for e in members:
                cluster_tag_freq.update(e.tag_names())

            root_cause_tags: list[str] = []
            for tag_name, cluster_count in cluster_tag_freq.most_common(20):
                global_count = global_tag_freq.get(tag_name, 0)
                # Over-representation: cluster_rate > 2× global_rate
                cluster_rate = cluster_count / len(members)
                global_rate = global_count / total_window if total_window > 0 else 0.0
                if cluster_rate > 2.0 * global_rate:
                    root_cause_tags.append(tag_name)

            all_cluster_tags: Counter[str] = Counter()
            for e in members:
                all_cluster_tags.update(e.tag_names())
            top_tags = [t for t, _ in all_cluster_tags.most_common(10)]

            evidence = [
                PatternEvidence(
                    experience_id=e.experience_id,
                    occurred_at=_iso(e.occurred_at),
                    relevance=_recency_factor(e.occurred_at),
                    note=f"Failure in cluster '{tag_label}'",
                )
                for e in members_sorted
            ]

            failure_mode = (
                tag_label if tag_label != "__untagged__" else "unclassified"
            )

            patterns.append(
                FailurePattern(
                    label=f"Recurring failure mode: {failure_mode}",
                    category=PatternCategory.FAILURE,
                    confidence=confidence,
                    occurrence_count=len(members),
                    description=(
                        f"Failure cluster '{failure_mode}' repeated "
                        f"{len(members)} time(s) with average severity "
                        f"{avg_sev:.2f}/4."
                    ),
                    first_seen_at=_iso(members_sorted[0].occurred_at),
                    last_seen_at=_iso(members_sorted[-1].occurred_at),
                    evidence=evidence,
                    tags=top_tags,
                    failure_mode=failure_mode,
                    average_severity=round(avg_sev, 4),
                    root_cause_tags=root_cause_tags[:10],
                )
            )

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    # ------------------------------------------------------------------
    # Pass 3: Habit patterns
    # ------------------------------------------------------------------

    def _extract_habit_patterns(
        self,
        window: list[Experience],
        min_occurrences: int,
    ) -> list[HabitPattern]:
        """Detect consistently repeated behavioral sequences (habits).

        A habit is defined as an ExperienceType/tag combination that recurs
        with a low coefficient-of-variation in its inter-occurrence interval
        (i.e. the gaps between occurrences are regular).

        The pass groups experiences by (ExperienceType, dominant_tag) tuples
        and scores each group's regularity using the interval standard
        deviation.  Groups with CV < 0.6 and at least ``min_occurrences``
        members are elevated to :class:`HabitPattern` instances.

        Parameters
        ----------
        window:
            All experiences within the lookback window.
        min_occurrences:
            Minimum occurrence count.

        Returns
        -------
        list[HabitPattern]
            Detected habit patterns, sorted by confidence descending.
        """
        # Group by (type, dominant_tag)
        groups: dict[tuple[str, str], list[Experience]] = defaultdict(list)
        for exp in window:
            tag_names = exp.tag_names()
            dominant_tag = sorted(tag_names)[0] if tag_names else "__none__"
            key = (exp.experience_type.name, dominant_tag)
            groups[key].append(exp)

        patterns: list[HabitPattern] = []
        for (type_name, dominant_tag), members in groups.items():
            if len(members) < min_occurrences:
                continue

            members_sorted = sorted(members, key=lambda e: e.occurred_at)
            gaps = [
                (
                    members_sorted[i + 1].occurred_at
                    - members_sorted[i].occurred_at
                ).total_seconds()
                / 86_400.0
                for i in range(len(members_sorted) - 1)
            ]

            # Require at least two gaps to assess regularity
            if len(gaps) < 2:
                continue

            mean_gap = sum(gaps) / len(gaps)
            std_gap = _population_std(gaps)
            cv = std_gap / mean_gap if mean_gap > 0.0 else 1.0

            # Only promote if sufficiently regular (CV < 0.6)
            if cv >= 0.6:
                continue

            recency_vals = [_recency_factor(e.occurred_at) for e in members]
            confidence = _confidence_score(
                occurrence_count=len(members),
                min_occurrences=min_occurrences,
                recency_values=recency_vals,
                interval_days=gaps,
            )

            # Temporal concentration: dominant hour and day-of-week
            hour_counter: Counter[int] = Counter()
            dow_counter: Counter[int] = Counter()
            for e in members:
                hour_counter[e.occurred_at.hour] += 1
                dow_counter[e.occurred_at.weekday()] += 1

            peak_hour, peak_hour_count = hour_counter.most_common(1)[0]
            peak_dow, peak_dow_count = dow_counter.most_common(1)[0]

            dom_hour = peak_hour if peak_hour_count / len(members) >= 0.35 else None
            dom_dow = peak_dow if peak_dow_count / len(members) >= 0.35 else None

            all_tags: Counter[str] = Counter()
            for e in members:
                all_tags.update(e.tag_names())
            top_tags = [t for t, _ in all_tags.most_common(8)]

            evidence = [
                PatternEvidence(
                    experience_id=e.experience_id,
                    occurred_at=_iso(e.occurred_at),
                    relevance=_recency_factor(e.occurred_at),
                    note=(
                        f"Habit occurrence: type={type_name}, "
                        f"tag={dominant_tag}"
                    ),
                )
                for e in members_sorted
            ]

            habit_label = (
                f"{type_name}:{dominant_tag}"
                if dominant_tag != "__none__"
                else type_name
            )

            patterns.append(
                HabitPattern(
                    label=f"Habit pattern: {habit_label}",
                    category=PatternCategory.HABIT,
                    confidence=confidence,
                    occurrence_count=len(members),
                    description=(
                        f"Recurring {type_name} behaviour tagged '{dominant_tag}' "
                        f"with mean interval {mean_gap:.1f}d (CV={cv:.2f})."
                    ),
                    first_seen_at=_iso(members_sorted[0].occurred_at),
                    last_seen_at=_iso(members_sorted[-1].occurred_at),
                    evidence=evidence,
                    tags=top_tags,
                    habit_description=(
                        f"Regular {type_name.lower()} activity related to "
                        f"'{dominant_tag}' every ~{mean_gap:.1f} days."
                    ),
                    average_interval_days=round(mean_gap, 3),
                    interval_std_days=round(std_gap, 3),
                    dominant_hour_bucket=dom_hour,
                    dominant_dow_bucket=dom_dow,
                )
            )

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    # ------------------------------------------------------------------
    # Pass 4: Behavioral patterns
    # ------------------------------------------------------------------

    def _extract_behavioral_patterns(
        self,
        window: list[Experience],
        min_occurrences: int,
    ) -> list[BehavioralPattern]:
        """Detect broad recurring behavioral tendencies in *window*.

        Behavioral patterns are broader than habits — they capture
        tendencies identifiable by high tag co-occurrence across mixed
        ExperienceTypes.  A behavioral pattern emerges when a tag appears
        as the primary topic across at least ``min_occurrences`` experiences
        of different types, indicating a cross-domain tendency rather than a
        type-specific repetition.

        The outcome quality of each behavioral pattern is inferred from the
        proportion of positive-outcome types (ACHIEVEMENT, SESSION with
        positive outcomes) versus negative-outcome types (FAILURE) in the
        cluster.

        Parameters
        ----------
        window:
            All experiences within the lookback window.
        min_occurrences:
            Minimum cluster size.

        Returns
        -------
        list[BehavioralPattern]
            Detected behavioral patterns, sorted by confidence descending.
        """
        # Build tag → experiences mapping
        tag_to_exps: dict[str, list[Experience]] = defaultdict(list)
        for exp in window:
            for tag_name in exp.tag_names():
                tag_to_exps[tag_name].append(exp)

        patterns: list[BehavioralPattern] = []
        for tag_name, members in tag_to_exps.items():
            if len(members) < min_occurrences:
                continue

            # Require cross-type diversity (at least 2 distinct types)
            types_present = {e.experience_type for e in members}
            if len(types_present) < 2:
                continue

            members_sorted = sorted(members, key=lambda e: e.occurred_at)
            recency_vals = [_recency_factor(e.occurred_at) for e in members]
            gaps = [
                (
                    members_sorted[i + 1].occurred_at
                    - members_sorted[i].occurred_at
                ).total_seconds()
                / 86_400.0
                for i in range(len(members_sorted) - 1)
            ]

            confidence = _confidence_score(
                occurrence_count=len(members),
                min_occurrences=min_occurrences,
                recency_values=recency_vals,
                interval_days=gaps,
            )

            # Outcome quality heuristic
            positive_count = sum(
                1
                for e in members
                if e.experience_type == ExperienceType.ACHIEVEMENT
            )
            negative_count = sum(
                1
                for e in members
                if e.experience_type == ExperienceType.FAILURE
            )
            total = len(members)
            if positive_count / total >= 0.60:
                outcome_quality = "positive"
            elif negative_count / total >= 0.40:
                outcome_quality = "negative"
            else:
                outcome_quality = "mixed"

            # Trigger tags: other tags most commonly co-occurring
            co_tags: Counter[str] = Counter()
            for e in members:
                for t in e.tag_names():
                    if t != tag_name:
                        co_tags[t] += 1
            trigger_tags = [t for t, _ in co_tags.most_common(5)]

            all_tags_counter: Counter[str] = Counter()
            for e in members:
                all_tags_counter.update(e.tag_names())
            top_tags = [t for t, _ in all_tags_counter.most_common(10)]

            evidence = [
                PatternEvidence(
                    experience_id=e.experience_id,
                    occurred_at=_iso(e.occurred_at),
                    relevance=_recency_factor(e.occurred_at),
                    note=f"Behavioral tendency tagged '{tag_name}'",
                )
                for e in members_sorted
            ]

            type_names = sorted(t.name for t in types_present)
            behavior_summary = (
                f"Cross-domain tendency around '{tag_name}' observed across "
                f"{', '.join(type_names)} experience types."
            )

            patterns.append(
                BehavioralPattern(
                    label=f"Behavioral tendency: {tag_name}",
                    category=PatternCategory.BEHAVIORAL,
                    confidence=confidence,
                    occurrence_count=len(members),
                    description=(
                        f"Tag '{tag_name}' appears in {len(members)} experiences "
                        f"spanning {len(types_present)} distinct types with "
                        f"{outcome_quality} overall outcome quality."
                    ),
                    first_seen_at=_iso(members_sorted[0].occurred_at),
                    last_seen_at=_iso(members_sorted[-1].occurred_at),
                    evidence=evidence,
                    tags=top_tags,
                    behavior_summary=behavior_summary,
                    trigger_tags=trigger_tags,
                    outcome_quality=outcome_quality,
                )
            )

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    # ------------------------------------------------------------------
    # Pass 5: Project patterns
    # ------------------------------------------------------------------

    def _extract_project_patterns(
        self,
        window: list[Experience],
        min_occurrences: int,
    ) -> list[ProjectPattern]:
        """Detect project evolution and lifecycle patterns within *window*.

        Groups experiences by ``metadata.project_refs``.  For each project
        with at least ``min_occurrences`` experiences, a :class:`ProjectPattern`
        is produced capturing the observed phase sequence (ordered list of
        ExperienceType names), total project span, and experience density.

        Parameters
        ----------
        window:
            All experiences within the lookback window.
        min_occurrences:
            Minimum experience count per project.

        Returns
        -------
        list[ProjectPattern]
            Detected project patterns, sorted by confidence descending.
        """
        project_to_exps: dict[str, list[Experience]] = defaultdict(list)
        for exp in window:
            for ref in exp.metadata.project_refs:
                project_to_exps[ref].append(exp)

        patterns: list[ProjectPattern] = []
        for project_ref, members in project_to_exps.items():
            if len(members) < min_occurrences:
                continue

            members_sorted = sorted(members, key=lambda e: e.occurred_at)
            recency_vals = [_recency_factor(e.occurred_at) for e in members]
            gaps = [
                (
                    members_sorted[i + 1].occurred_at
                    - members_sorted[i].occurred_at
                ).total_seconds()
                / 86_400.0
                for i in range(len(members_sorted) - 1)
            ]

            confidence = _confidence_score(
                occurrence_count=len(members),
                min_occurrences=min_occurrences,
                recency_values=recency_vals,
                interval_days=gaps,
            )

            total_days = (
                members_sorted[-1].occurred_at
                - members_sorted[0].occurred_at
            ).total_seconds() / 86_400.0
            density = (
                (len(members) / (total_days / 7.0))
                if total_days > 0
                else float(len(members))
            )

            phase_seq = [e.experience_type.name for e in members_sorted]

            all_tags: Counter[str] = Counter()
            for e in members:
                all_tags.update(e.tag_names())
            top_tags = [t for t, _ in all_tags.most_common(10)]

            evidence = [
                PatternEvidence(
                    experience_id=e.experience_id,
                    occurred_at=_iso(e.occurred_at),
                    relevance=_recency_factor(e.occurred_at),
                    note=f"Project experience: {e.experience_type.name}",
                )
                for e in members_sorted
            ]

            patterns.append(
                ProjectPattern(
                    label=f"Project arc: {project_ref}",
                    category=PatternCategory.PROJECT,
                    confidence=confidence,
                    occurrence_count=len(members),
                    description=(
                        f"Project '{project_ref}' has {len(members)} experiences "
                        f"spanning {total_days:.1f} days at "
                        f"{density:.2f} experiences/week."
                    ),
                    first_seen_at=_iso(members_sorted[0].occurred_at),
                    last_seen_at=_iso(members_sorted[-1].occurred_at),
                    evidence=evidence,
                    tags=top_tags,
                    project_ref=project_ref,
                    phase_sequence=phase_seq,
                    total_duration_days=round(total_days, 3),
                    experience_density=round(density, 4),
                )
            )

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    # ------------------------------------------------------------------
    # Pass 6: Temporal patterns
    # ------------------------------------------------------------------

    def _extract_temporal_patterns(
        self,
        window: list[Experience],
        min_occurrences: int,
    ) -> list[TemporalPattern]:
        """Detect time-based recurrence patterns within *window*.

        Analyses four temporal dimensions independently:

        * **Hourly clustering** — experiences concentrated in specific
          hours of the day.
        * **Day-of-week clustering** — experiences concentrated on
          specific weekdays.
        * **Monthly / seasonal clustering** — experiences concentrated
          in specific calendar months.

        A dimension is reported when a single bucket accounts for at least
        35% of total experiences in the window AND the bucket contains at
        least ``min_occurrences`` experiences.

        Parameters
        ----------
        window:
            All experiences within the lookback window.
        min_occurrences:
            Minimum experiences per temporal bucket.

        Returns
        -------
        list[TemporalPattern]
            Detected temporal patterns, sorted by confidence descending.
        """
        if len(window) < min_occurrences:
            return []

        total = len(window)
        patterns: list[TemporalPattern] = []

        # 1. Hour-of-day clustering
        hour_counter: Counter[int] = Counter()
        for exp in window:
            hour_counter[exp.occurred_at.hour] += 1

        peak_hour, peak_hour_count = hour_counter.most_common(1)[0]
        if peak_hour_count >= min_occurrences:
            ratio = peak_hour_count / total
            if ratio >= 0.35:
                members = [
                    e for e in window if e.occurred_at.hour == peak_hour
                ]
                members_sorted = sorted(members, key=lambda e: e.occurred_at)
                recency_vals = [_recency_factor(e.occurred_at) for e in members]
                gaps = [
                    (
                        members_sorted[i + 1].occurred_at
                        - members_sorted[i].occurred_at
                    ).total_seconds()
                    / 86_400.0
                    for i in range(len(members_sorted) - 1)
                ]
                confidence = _confidence_score(
                    occurrence_count=len(members),
                    min_occurrences=min_occurrences,
                    recency_values=recency_vals,
                    interval_days=gaps,
                )
                evidence = [
                    PatternEvidence(
                        experience_id=e.experience_id,
                        occurred_at=_iso(e.occurred_at),
                        relevance=_recency_factor(e.occurred_at),
                        note=f"Occurs at hour {peak_hour:02d}:xx",
                    )
                    for e in members_sorted
                ]
                patterns.append(
                    TemporalPattern(
                        label=f"Hourly cluster at {peak_hour:02d}:xx",
                        category=PatternCategory.TEMPORAL,
                        confidence=confidence,
                        occurrence_count=len(members),
                        description=(
                            f"{len(members)} experiences ({ratio:.0%}) occur "
                            f"between {peak_hour:02d}:00 and {peak_hour:02d}:59."
                        ),
                        first_seen_at=_iso(members_sorted[0].occurred_at),
                        last_seen_at=_iso(members_sorted[-1].occurred_at),
                        evidence=evidence,
                        recurrence_type="hourly_cluster",
                        peak_hour=peak_hour,
                        concentration_ratio=round(ratio, 6),
                    )
                )

        # 2. Day-of-week clustering
        dow_counter: Counter[int] = Counter()
        _DOW_NAMES = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        for exp in window:
            dow_counter[exp.occurred_at.weekday()] += 1

        peak_dow, peak_dow_count = dow_counter.most_common(1)[0]
        if peak_dow_count >= min_occurrences:
            ratio = peak_dow_count / total
            if ratio >= 0.35:
                members = [
                    e
                    for e in window
                    if e.occurred_at.weekday() == peak_dow
                ]
                members_sorted = sorted(members, key=lambda e: e.occurred_at)
                recency_vals = [_recency_factor(e.occurred_at) for e in members]
                gaps = [
                    (
                        members_sorted[i + 1].occurred_at
                        - members_sorted[i].occurred_at
                    ).total_seconds()
                    / 86_400.0
                    for i in range(len(members_sorted) - 1)
                ]
                confidence = _confidence_score(
                    occurrence_count=len(members),
                    min_occurrences=min_occurrences,
                    recency_values=recency_vals,
                    interval_days=gaps,
                )
                evidence = [
                    PatternEvidence(
                        experience_id=e.experience_id,
                        occurred_at=_iso(e.occurred_at),
                        relevance=_recency_factor(e.occurred_at),
                        note=f"Occurs on {_DOW_NAMES[peak_dow]}",
                    )
                    for e in members_sorted
                ]
                patterns.append(
                    TemporalPattern(
                        label=f"Weekly cluster on {_DOW_NAMES[peak_dow]}s",
                        category=PatternCategory.TEMPORAL,
                        confidence=confidence,
                        occurrence_count=len(members),
                        description=(
                            f"{len(members)} experiences ({ratio:.0%}) occur "
                            f"on {_DOW_NAMES[peak_dow]}s."
                        ),
                        first_seen_at=_iso(members_sorted[0].occurred_at),
                        last_seen_at=_iso(members_sorted[-1].occurred_at),
                        evidence=evidence,
                        recurrence_type="weekly_cluster",
                        peak_dow=peak_dow,
                        concentration_ratio=round(ratio, 6),
                    )
                )

        # 3. Monthly / seasonal clustering
        month_counter: Counter[int] = Counter()
        _MONTH_NAMES = [
            "", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        for exp in window:
            month_counter[exp.occurred_at.month] += 1

        if month_counter:
            peak_month, peak_month_count = month_counter.most_common(1)[0]
            if peak_month_count >= min_occurrences:
                ratio = peak_month_count / total
                if ratio >= 0.35:
                    members = [
                        e for e in window if e.occurred_at.month == peak_month
                    ]
                    members_sorted = sorted(members, key=lambda e: e.occurred_at)
                    recency_vals = [
                        _recency_factor(e.occurred_at) for e in members
                    ]
                    gaps = [
                        (
                            members_sorted[i + 1].occurred_at
                            - members_sorted[i].occurred_at
                        ).total_seconds()
                        / 86_400.0
                        for i in range(len(members_sorted) - 1)
                    ]
                    confidence = _confidence_score(
                        occurrence_count=len(members),
                        min_occurrences=min_occurrences,
                        recency_values=recency_vals,
                        interval_days=gaps,
                    )
                    evidence = [
                        PatternEvidence(
                            experience_id=e.experience_id,
                            occurred_at=_iso(e.occurred_at),
                            relevance=_recency_factor(e.occurred_at),
                            note=f"Occurs in {_MONTH_NAMES[peak_month]}",
                        )
                        for e in members_sorted
                    ]
                    patterns.append(
                        TemporalPattern(
                            label=(
                                f"Seasonal cluster in "
                                f"{_MONTH_NAMES[peak_month]}"
                            ),
                            category=PatternCategory.TEMPORAL,
                            confidence=confidence,
                            occurrence_count=len(members),
                            description=(
                                f"{len(members)} experiences ({ratio:.0%}) "
                                f"occur in {_MONTH_NAMES[peak_month]}."
                            ),
                            first_seen_at=_iso(members_sorted[0].occurred_at),
                            last_seen_at=_iso(members_sorted[-1].occurred_at),
                            evidence=evidence,
                            recurrence_type="seasonal_cluster",
                            peak_month=peak_month,
                            concentration_ratio=round(ratio, 6),
                        )
                    )

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    # ------------------------------------------------------------------
    # Pass 7: Growth patterns
    # ------------------------------------------------------------------

    def _extract_growth_patterns(
        self,
        window: list[Experience],
        min_occurrences: int,
    ) -> list[GrowthPattern]:
        """Detect skill and capability improvement trajectories.

        Identifies domains (via project_refs or dominant tags) where the
        linear trend in experience importance ordinal over time is
        significantly positive (improving), flat (plateauing), or negative
        (declining).  Only domains with at least
        ``max(min_occurrences, _GROWTH_MIN_DATAPOINTS)`` data-points are
        analysed.

        Parameters
        ----------
        window:
            All experiences within the lookback window.
        min_occurrences:
            Minimum experience count per domain.

        Returns
        -------
        list[GrowthPattern]
            Detected growth patterns, sorted by confidence descending.
        """
        min_pts = max(min_occurrences, _GROWTH_MIN_DATAPOINTS)

        # Build domain → experiences mapping
        # Priority: project_ref → tag → fallback label
        domain_to_exps: dict[str, list[Experience]] = defaultdict(list)
        for exp in window:
            if exp.metadata.project_refs:
                for ref in exp.metadata.project_refs:
                    domain_to_exps[f"project:{ref}"].append(exp)
            else:
                tag_names = exp.tag_names()
                if tag_names:
                    domain_to_exps[f"tag:{sorted(tag_names)[0]}"].append(exp)
                else:
                    domain_to_exps["general"].append(exp)

        patterns: list[GrowthPattern] = []
        for domain_key, members in domain_to_exps.items():
            if len(members) < min_pts:
                continue

            members_sorted = sorted(members, key=lambda e: e.occurred_at)
            epoch = members_sorted[0].occurred_at

            x_days = [
                (e.occurred_at - epoch).total_seconds() / 86_400.0
                for e in members_sorted
            ]
            y_ord = [
                float(_IMPORTANCE_ORDINAL[e.importance])
                for e in members_sorted
            ]

            slope = _linear_slope(x_days, y_ord)

            if slope > 0.005:
                direction = "improving"
            elif slope < -0.005:
                direction = "declining"
            else:
                direction = "plateauing"

            # Milestone count = HIGH + CRITICAL experiences
            milestone_count = sum(
                1
                for e in members
                if e.importance
                in (ExperienceImportance.HIGH, ExperienceImportance.CRITICAL)
            )

            recency_vals = [_recency_factor(e.occurred_at) for e in members]
            gaps = [
                (
                    members_sorted[i + 1].occurred_at
                    - members_sorted[i].occurred_at
                ).total_seconds()
                / 86_400.0
                for i in range(len(members_sorted) - 1)
            ]

            confidence = _confidence_score(
                occurrence_count=len(members),
                min_occurrences=min_pts,
                recency_values=recency_vals,
                interval_days=gaps,
            )

            # Modulate confidence by direction: declining trajectories carry
            # their own informational value but are not boosted by the growth
            # quality signal
            if direction == "improving" and milestone_count >= 2:
                milestone_boost = min(0.15, 0.05 * milestone_count)
                confidence = min(1.0, confidence + milestone_boost)
            elif direction == "declining":
                # Slight confidence reduction — declining patterns are
                # important to surface but uncertain in interpretation
                confidence = max(0.0, confidence - 0.05)

            all_tags: Counter[str] = Counter()
            for e in members:
                all_tags.update(e.tag_names())
            top_tags = [t for t, _ in all_tags.most_common(8)]

            evidence = [
                PatternEvidence(
                    experience_id=e.experience_id,
                    occurred_at=_iso(e.occurred_at),
                    relevance=_recency_factor(e.occurred_at),
                    note=(
                        f"Growth data-point: importance={e.importance.name}"
                    ),
                )
                for e in members_sorted
            ]

            domain_label = domain_key.replace("project:", "").replace(
                "tag:", ""
            )

            patterns.append(
                GrowthPattern(
                    label=(
                        f"Growth trajectory [{direction}]: {domain_label}"
                    ),
                    category=PatternCategory.GROWTH,
                    confidence=confidence,
                    occurrence_count=len(members),
                    description=(
                        f"Domain '{domain_label}' shows a {direction} trend "
                        f"(slope={slope:.4f} importance/day) across "
                        f"{len(members)} experiences with "
                        f"{milestone_count} milestone(s)."
                    ),
                    first_seen_at=_iso(members_sorted[0].occurred_at),
                    last_seen_at=_iso(members_sorted[-1].occurred_at),
                    evidence=evidence,
                    tags=top_tags,
                    domain=domain_label,
                    growth_rate=round(slope, 8),
                    milestone_count=milestone_count,
                    trajectory_direction=direction,
                )
            )

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    # ------------------------------------------------------------------
    # Cross-category clustering
    # ------------------------------------------------------------------

    def _cluster_related_patterns(
        self,
        patterns: list[DetectedPattern],
    ) -> None:
        """Link related patterns across categories via shared tags.

        Populates ``pattern.related_pattern_ids`` for each pattern in
        *patterns* where another pattern shares at least one tag name.
        The relationship is bidirectional: if A links to B then B also
        links to A.

        Mutates the ``related_pattern_ids`` field of the supplied patterns
        in place.  The pattern store is NOT updated here — callers that
        need persistence should call :meth:`_store_pattern` afterwards.

        Parameters
        ----------
        patterns:
            The full list of detected patterns produced by the extraction
            run.  Must already be in the internal pattern store.
        """
        if len(patterns) < 2:
            return

        # Build tag → set[pattern_id] index
        tag_to_pids: dict[str, set[str]] = defaultdict(set)
        pid_to_pattern: dict[str, DetectedPattern] = {}
        for p in patterns:
            pid_to_pattern[p.pattern_id] = p
            for tag in p.tags:
                tag_to_pids[tag].add(p.pattern_id)

        # For each pattern, find peers sharing at least one tag
        for p in patterns:
            related: set[str] = set()
            for tag in p.tags:
                for peer_id in tag_to_pids.get(tag, set()):
                    if peer_id != p.pattern_id:
                        related.add(peer_id)
            p.related_pattern_ids = sorted(related)

            # Persist updated pattern to store
            self._pattern_store[p.pattern_id] = p