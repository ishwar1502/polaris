# subsystems/echo/interfaces.py
"""
ECHO v1 Engine Interfaces.

This module defines abstract base classes (engine contracts) for every engine
that belongs to the ECHO Episodic Memory Core.  All interfaces follow the
same pattern:

* They define the engine's public API as abstract methods.
* They document the contract that implementations must honour.
* They provide no fake intelligence or pass-through stubs.

Implemented now
---------------
* :class:`ExperienceEngineInterface`    — core experience lifecycle contract
* :class:`SignificanceEngineInterface`  — importance scoring and classification

Future engine contracts
-----------------------
* :class:`SessionEngineInterface`          — session grouping and lifecycle
* :class:`ReflectionEngineInterface`       — lesson generation from experiences
* :class:`ExperienceRetrievalEngineInterface` — retrieval and query surface
* :class:`MemoryConsolidationEngineInterface` — short-to-long-term promotion
* :class:`PatternExtractionEngineInterface`   — recurring experience patterns
* :class:`PersonalHistoryEngineInterface`     — life narrative continuity

ECHO Boundary Law
-----------------
All interfaces enforce ECHO's ownership boundary.  ECHO owns experiences,
events, conversations, sessions, achievements, failures, observations,
activity history, and personal history.  ECHO does NOT own knowledge,
identity, goals, schedules, relationships, or decisions.
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any

from subsystems.echo.models import (
    AchievementRecord,
    EventRecord,
    Experience,
    ExperienceImportance,
    ExperienceType,
    FailureRecord,
    MemoryTag,
    ObservationRecord,
)


# ---------------------------------------------------------------------------
# Experience Engine Interface
# ---------------------------------------------------------------------------


class ExperienceEngineInterface(abc.ABC):
    """Contract for the Experience Engine.

    The Experience Engine is ECHO's primary storage and lifecycle manager.
    It creates, stores, updates, deletes, and retrieves :class:`Experience`
    objects.  All write operations must pass through the Significance Engine
    before persistence unless explicitly overridden with ``force=True``.

    Implementation requirements
    ---------------------------
    * Must be thread-safe.  All public methods must serialise concurrent
      access via an appropriate lock mechanism.
    * Must reject storage of data types outside ECHO's ownership boundary.
    * Must delegate significance scoring to the SignificanceEngine on every
      create and update call (unless ``force=True`` is supplied).
    * Must publish domain events for each completed lifecycle operation.
    * Must update :attr:`ExperienceMetadata.retrieval_count` on every fetch.
    * Must never silently discard validation errors — raise on failure.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initialize(self) -> None:
        """Prepare the engine for use.

        Called once by the ECHO subsystem during startup.  Implementations
        should allocate internal state, connect to the persistence layer,
        and register any required event subscriptions.

        Raises
        ------
        EchoError
            If initialisation fails for any reason.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release all resources held by this engine.

        Called by the ECHO subsystem during graceful shutdown.  Must flush
        any pending writes and deregister event subscriptions.
        """

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def create_experience(
        self,
        title: str,
        experience_type: ExperienceType,
        importance: ExperienceImportance,
        *,
        description: str = "",
        context: str = "",
        outcome: str = "",
        tags: list[MemoryTag] | None = None,
        occurred_at: datetime | None = None,
        source_subsystem: str = "ECHO_API",
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Experience:
        """Create and persist a new :class:`Experience`.

        Parameters
        ----------
        title:
            Short, human-readable label for this experience.
        experience_type:
            :class:`ExperienceType` classification.
        importance:
            Caller-suggested :class:`ExperienceImportance` tier.  The
            Significance Engine may override this unless ``force=True``.
        description:
            Narrative elaboration of what happened.
        context:
            Situational context at the time of the experience.
        outcome:
            What resulted from this experience.
        tags:
            Optional list of :class:`MemoryTag` objects for indexing.
        occurred_at:
            UTC timestamp of when this experience occurred.  Defaults to
            now.  May be historical.
        source_subsystem:
            Which POLARIS subsystem is recording this experience.
        session_id:
            Optional UUID of the enclosing session experience.
        extra:
            Extensible key-value store for engine-specific annotations.
        force:
            If ``True``, bypass the Significance Engine threshold gate
            and store regardless of significance score.

        Returns
        -------
        Experience
            The created and stored experience with populated metadata.

        Raises
        ------
        ExperienceValidationError
            If required fields are invalid.
        BelowSignificanceThresholdError
            If the experience scores below the threshold and ``force=False``.
        ExperienceDuplicateError
            If the memory integrity check detects a duplicate.
        EchoBoundaryViolationError
            If the caller attempts to store data ECHO does not own.
        """

    # ------------------------------------------------------------------
    # Store (persist an already-constructed Experience)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def store_experience(
        self,
        experience: Experience,
        *,
        force: bool = False,
    ) -> Experience:
        """Persist an already-constructed :class:`Experience` object.

        Unlike :meth:`create_experience`, this method accepts a pre-built
        domain object.  The Significance Engine is still consulted unless
        ``force=True``.

        Parameters
        ----------
        experience:
            The :class:`Experience` to persist.
        force:
            Bypass the Significance Engine threshold if ``True``.

        Returns
        -------
        Experience
            The stored experience (with metadata updated in place).

        Raises
        ------
        ExperienceDuplicateError
            If an experience with the same ``experience_id`` already exists.
        BelowSignificanceThresholdError
            If below threshold and ``force=False``.
        """

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def update_experience(
        self,
        experience_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        context: str | None = None,
        outcome: str | None = None,
        importance: ExperienceImportance | None = None,
        tags: list[MemoryTag] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Experience:
        """Update mutable fields of an existing :class:`Experience`.

        Only the fields explicitly supplied (non-``None``) are updated.
        ``experience_id``, ``experience_type``, ``recorded_at``, and all
        metadata internal fields are immutable after creation.

        Parameters
        ----------
        experience_id:
            UUID of the experience to update.
        title:
            New title, or ``None`` to leave unchanged.
        description:
            New description, or ``None`` to leave unchanged.
        context:
            New context, or ``None`` to leave unchanged.
        outcome:
            New outcome, or ``None`` to leave unchanged.
        importance:
            New importance tier, or ``None`` to leave unchanged.
        tags:
            Replacement tag list, or ``None`` to leave unchanged.
        extra:
            Replacement extra dict, or ``None`` to leave unchanged.

        Returns
        -------
        Experience
            The updated experience.

        Raises
        ------
        ExperienceNotFoundError
            If no experience with ``experience_id`` exists.
        ExperienceValidationError
            If any supplied field fails validation.
        """

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def delete_experience(
        self,
        experience_id: str,
        *,
        cascade: bool = False,
    ) -> bool:
        """Remove an experience from ECHO's store.

        CRITICAL experiences are protected and cannot be deleted.  Attempts
        to delete a CRITICAL experience raise :class:`ExperienceError`.

        Parameters
        ----------
        experience_id:
            UUID of the experience to delete.
        cascade:
            If ``True``, also remove child records (events, achievements,
            failures, observations) that reference this experience.

        Returns
        -------
        bool
            ``True`` if the record was found and removed; ``False`` if it
            did not exist (idempotent delete).

        Raises
        ------
        ExperienceError
            If the experience is CRITICAL and deletion is therefore blocked.
        """

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_experience(self, experience_id: str) -> Experience:
        """Retrieve a single experience by its UUID.

        This call increments :attr:`ExperienceMetadata.retrieval_count`.

        Parameters
        ----------
        experience_id:
            UUID of the desired experience.

        Returns
        -------
        Experience
            The matching experience.

        Raises
        ------
        ExperienceNotFoundError
            If no experience with this UUID exists.
        """

    @abc.abstractmethod
    def query_experiences(
        self,
        *,
        experience_type: ExperienceType | None = None,
        importance: ExperienceImportance | None = None,
        min_importance: ExperienceImportance | None = None,
        tags: list[str] | None = None,
        source_subsystem: str | None = None,
        session_id: str | None = None,
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        consolidated: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Experience]:
        """Return experiences matching the supplied filter criteria.

        All filter parameters are optional and combined with logical AND.
        Results are ordered by ``occurred_at`` descending (most recent first).

        Parameters
        ----------
        experience_type:
            Filter by a specific :class:`ExperienceType`.
        importance:
            Filter by exact :class:`ExperienceImportance` tier.
        min_importance:
            Filter experiences at or above this importance tier.
        tags:
            Filter experiences that carry ALL of the supplied tag names.
        source_subsystem:
            Filter by the originating subsystem identifier.
        session_id:
            Filter experiences that belong to a specific session.
        occurred_after:
            Only return experiences that occurred after this UTC datetime.
        occurred_before:
            Only return experiences that occurred before this UTC datetime.
        consolidated:
            If ``True``, return only consolidated experiences.  If ``False``,
            return only unconsolidated experiences.  ``None`` returns both.
        limit:
            Maximum number of results to return.
        offset:
            Number of results to skip (for pagination).

        Returns
        -------
        list[Experience]
            Matching experiences, ordered by ``occurred_at`` descending.
        """

    @abc.abstractmethod
    def count_experiences(
        self,
        *,
        experience_type: ExperienceType | None = None,
        importance: ExperienceImportance | None = None,
        consolidated: bool | None = None,
    ) -> int:
        """Return the count of stored experiences matching the given filters.

        Parameters
        ----------
        experience_type:
            Count only this type of experience.
        importance:
            Count only this importance tier.
        consolidated:
            Count only consolidated (``True``) or unconsolidated (``False``)
            experiences.  ``None`` counts all.

        Returns
        -------
        int
            Number of matching experience records.
        """

    @abc.abstractmethod
    def experience_exists(self, experience_id: str) -> bool:
        """Return whether an experience with the given UUID exists.

        Parameters
        ----------
        experience_id:
            UUID to check.

        Returns
        -------
        bool
            ``True`` if the experience is in the store.
        """

    # ------------------------------------------------------------------
    # Bulk / Utility
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_recent_experiences(
        self,
        limit: int = 20,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
    ) -> list[Experience]:
        """Return the most recently recorded experiences.

        Parameters
        ----------
        limit:
            Maximum number of results.
        min_importance:
            Exclude experiences below this importance tier.

        Returns
        -------
        list[Experience]
            Most recently recorded experiences, newest first.
        """

    @abc.abstractmethod
    def get_significant_experiences(
        self,
        limit: int = 50,
    ) -> list[Experience]:
        """Return experiences above the LOW importance threshold.

        These are the experiences eligible for long-term memory
        consolidation by the Memory Consolidation Engine.

        Parameters
        ----------
        limit:
            Maximum number of results.

        Returns
        -------
        list[Experience]
            Significant experiences, ordered by significance score descending.
        """


# ---------------------------------------------------------------------------
# Significance Engine Interface
# ---------------------------------------------------------------------------


class SignificanceEngineInterface(abc.ABC):
    """Contract for the Significance Engine.

    The Significance Engine determines which experiences deserve memory.
    It assigns a numeric score (0.0–1.0) and an :class:`ExperienceImportance`
    tier to every candidate experience.  Only MEDIUM and above are eligible
    for long-term memory consolidation.

    Implementation requirements
    ---------------------------
    * Must be thread-safe.
    * Must produce deterministic scores for identical inputs.
    * Must never silently swallow scoring errors.
    * Must NOT modify :class:`Experience` objects passed to it — scoring is
      a read-only evaluation.  Callers update the experience after receiving
      the result.
    * Must be extensible: scoring rules must be registerable/configurable
      without changes to the engine's public API.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initialize(self) -> None:
        """Prepare the engine for use.

        Load scoring rules, configure thresholds, and ready internal state.

        Raises
        ------
        SignificanceError
            If initialisation fails.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release any resources held by this engine."""

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def score(self, experience: Experience) -> float:
        """Compute a significance score for a given experience.

        The score is a float in the range ``[0.0, 1.0]`` where higher values
        indicate greater long-term memory value.  The score reflects the
        interaction of experience type, importance tier, tag richness,
        description depth, source subsystem, and any registered custom rules.

        Parameters
        ----------
        experience:
            The candidate :class:`Experience` to evaluate.  Must not be
            ``None`` and must have a valid ``experience_id``.

        Returns
        -------
        float
            Significance score in [0.0, 1.0].

        Raises
        ------
        SignificanceScoringError
            If the score cannot be computed due to missing or invalid context.
        """

    @abc.abstractmethod
    def classify(self, experience: Experience) -> ExperienceImportance:
        """Derive the canonical :class:`ExperienceImportance` tier from an experience.

        Uses :meth:`score` internally and maps the numeric result to the
        four-tier importance scale:

        * 0.0 – <0.25 → ``LOW``
        * 0.25 – <0.55 → ``MEDIUM``
        * 0.55 – <0.80 → ``HIGH``
        * 0.80 – 1.0   → ``CRITICAL``

        Implementations MAY adjust these thresholds through configuration
        but must document the mapping they apply.

        Parameters
        ----------
        experience:
            The candidate :class:`Experience` to classify.

        Returns
        -------
        ExperienceImportance
            The significance tier recommended for this experience.

        Raises
        ------
        SignificanceScoringError
            If classification fails due to a scoring error.
        """

    @abc.abstractmethod
    def evaluate(self, experience: Experience) -> "SignificanceResult":
        """Produce a full :class:`SignificanceResult` for an experience.

        This is the preferred method for callers that need both the numeric
        score and the tier classification in one call, along with the
        breakdown of contributing factors.

        Parameters
        ----------
        experience:
            The candidate :class:`Experience` to evaluate.

        Returns
        -------
        SignificanceResult
            Score, tier, breakdown, and promotion eligibility.

        Raises
        ------
        SignificanceScoringError
            If evaluation fails.
        """

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_threshold(self) -> float:
        """Return the current minimum score for automatic storage.

        Experiences below this threshold are rejected by the Experience
        Engine unless ``force=True`` is passed.

        Returns
        -------
        float
            Threshold value in [0.0, 1.0].
        """

    @abc.abstractmethod
    def set_threshold(self, threshold: float) -> None:
        """Update the minimum significance threshold.

        Parameters
        ----------
        threshold:
            New threshold in [0.0, 1.0].

        Raises
        ------
        ValueError
            If ``threshold`` is outside [0.0, 1.0].
        """

    # ------------------------------------------------------------------
    # Eligibility checks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def is_eligible_for_storage(
        self,
        experience: Experience,
        *,
        force: bool = False,
    ) -> bool:
        """Return whether an experience should be stored in ECHO.

        An experience is eligible if its significance score meets or exceeds
        the configured threshold, or if ``force=True`` is supplied.

        Parameters
        ----------
        experience:
            The :class:`Experience` to check.
        force:
            If ``True``, always returns ``True`` regardless of score.

        Returns
        -------
        bool
            ``True`` if the experience should be stored.
        """

    @abc.abstractmethod
    def is_eligible_for_long_term_memory(self, experience: Experience) -> bool:
        """Return whether an experience qualifies for long-term memory consolidation.

        Only MEDIUM, HIGH, and CRITICAL experiences are eligible.  LOW
        experiences are candidates for pruning by the Memory Consolidation
        Engine and must not be promoted.

        Parameters
        ----------
        experience:
            The :class:`Experience` to check.

        Returns
        -------
        bool
            ``True`` if the experience may be consolidated to long-term memory.
        """

    @abc.abstractmethod
    def is_promotion_eligible(self, experience: Experience) -> bool:
        """Return whether a MEDIUM experience should be promoted to HIGH.

        Promotion is granted when accumulating evidence (retrieval count,
        related experience density, or time-based importance growth) pushes
        a MEDIUM experience across the HIGH threshold.

        Parameters
        ----------
        experience:
            A MEDIUM-tier :class:`Experience` to evaluate for promotion.

        Returns
        -------
        bool
            ``True`` if the experience warrants promotion to HIGH.
        """

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def score_batch(
        self, experiences: list[Experience]
    ) -> dict[str, float]:
        """Score multiple experiences in a single call.

        Parameters
        ----------
        experiences:
            List of :class:`Experience` objects to score.

        Returns
        -------
        dict[str, float]
            Mapping of ``experience_id`` → significance score.
        """

    @abc.abstractmethod
    def classify_batch(
        self, experiences: list[Experience]
    ) -> dict[str, ExperienceImportance]:
        """Classify multiple experiences in a single call.

        Parameters
        ----------
        experiences:
            List of :class:`Experience` objects to classify.

        Returns
        -------
        dict[str, ExperienceImportance]
            Mapping of ``experience_id`` → :class:`ExperienceImportance` tier.
        """


# ---------------------------------------------------------------------------
# Significance Result (value object produced by SignificanceEngine.evaluate)
# ---------------------------------------------------------------------------


class SignificanceResult:
    """Immutable result produced by :meth:`SignificanceEngineInterface.evaluate`.

    Attributes
    ----------
    experience_id:
        UUID of the evaluated experience.
    score:
        Numeric significance in [0.0, 1.0].
    importance:
        Recommended :class:`ExperienceImportance` tier.
    eligible_for_storage:
        Whether this experience meets the configured threshold.
    eligible_for_long_term_memory:
        Whether this experience qualifies for consolidation.
    promotion_eligible:
        Whether a MEDIUM experience should be promoted to HIGH.
    factor_breakdown:
        Mapping of scoring factor name → contribution score for
        explainability.  E.g. ``{"type_weight": 0.3, "tag_richness": 0.1}``.
    notes:
        Human-readable explanation of the scoring decision.
    """

    __slots__ = (
        "experience_id",
        "score",
        "importance",
        "eligible_for_storage",
        "eligible_for_long_term_memory",
        "promotion_eligible",
        "factor_breakdown",
        "notes",
    )

    def __init__(
        self,
        experience_id: str,
        score: float,
        importance: ExperienceImportance,
        eligible_for_storage: bool,
        eligible_for_long_term_memory: bool,
        promotion_eligible: bool,
        factor_breakdown: dict[str, float],
        notes: str = "",
    ) -> None:
        self.experience_id = experience_id
        self.score = score
        self.importance = importance
        self.eligible_for_storage = eligible_for_storage
        self.eligible_for_long_term_memory = eligible_for_long_term_memory
        self.promotion_eligible = promotion_eligible
        self.factor_breakdown = factor_breakdown
        self.notes = notes

    def __repr__(self) -> str:
        return (
            f"SignificanceResult(experience_id={self.experience_id!r}, "
            f"score={self.score:.3f}, importance={self.importance.name})"
        )


# ---------------------------------------------------------------------------
# Session Engine Interface (future)
# ---------------------------------------------------------------------------


class SessionEngineInterface(abc.ABC):
    """Contract for the Session Engine (future implementation).

    The Session Engine groups experiences into bounded activity periods.
    A session is itself an :class:`Experience` of type ``SESSION``, and
    child experiences reference it via ``metadata.session_id``.

    Implementation requirements
    ---------------------------
    * Must support concurrent sessions (e.g. work session + study session).
    * Must auto-close stale sessions after a configurable idle period.
    * Must aggregate session outcomes when closing.
    """

    @abc.abstractmethod
    def open_session(
        self,
        title: str,
        *,
        goals: list[str] | None = None,
        tags: list[MemoryTag] | None = None,
        source_subsystem: str = "ECHO_API",
    ) -> Experience:
        """Open a new session and return the session :class:`Experience`.

        Parameters
        ----------
        title:
            Short label for this session.
        goals:
            Optional list of goal descriptions intended for this session.
        tags:
            Optional indexing tags.
        source_subsystem:
            Originating subsystem.

        Returns
        -------
        Experience
            The created session experience (type ``SESSION``).
        """

    @abc.abstractmethod
    def close_session(
        self,
        session_id: str,
        *,
        outcome: str = "",
    ) -> Experience:
        """Close an open session and persist its final state.

        Parameters
        ----------
        session_id:
            UUID of the session experience to close.
        outcome:
            Summary of what was accomplished in this session.

        Returns
        -------
        Experience
            The updated, closed session experience.

        Raises
        ------
        ExperienceNotFoundError
            If ``session_id`` does not reference a known session.
        """

    @abc.abstractmethod
    def get_active_sessions(self) -> list[Experience]:
        """Return all currently open (not yet closed) sessions.

        Returns
        -------
        list[Experience]
            Open session experiences.
        """

    @abc.abstractmethod
    def get_session_experiences(self, session_id: str) -> list[Experience]:
        """Return all experiences that belong to a specific session.

        Parameters
        ----------
        session_id:
            UUID of the parent session.

        Returns
        -------
        list[Experience]
            Child experiences grouped under this session.
        """


# ---------------------------------------------------------------------------
# Reflection Engine Interface (future)
# ---------------------------------------------------------------------------


class ReflectionEngineInterface(abc.ABC):
    """Contract for the Reflection Engine (future implementation).

    The Reflection Engine generates lessons from experiences.  It consumes
    failure records, achievement records, and session summaries to produce
    :class:`Experience` objects of type ``REFLECTION``.  Reflections feed
    NOVA's recommendation engine.

    Implementation requirements
    ---------------------------
    * Must NOT store reflections without the Significance Engine's approval.
    * Must link reflections back to their source experiences via
      ``metadata.related_experience_ids``.
    * Must NOT generate reflections from LOW importance experiences without
      explicit configuration override.
    """

    @abc.abstractmethod
    def generate_reflection(
        self,
        source_experience_id: str,
        *,
        force: bool = False,
    ) -> Experience:
        """Generate a reflection from a single source experience.

        Parameters
        ----------
        source_experience_id:
            UUID of the experience to reflect on.
        force:
            Store the reflection even if the Significance Engine scores it
            below the threshold.

        Returns
        -------
        Experience
            The generated REFLECTION experience.
        """

    @abc.abstractmethod
    def generate_session_reflection(
        self,
        session_id: str,
    ) -> Experience:
        """Generate a reflection summarising an entire session.

        Parameters
        ----------
        session_id:
            UUID of the closed session to reflect on.

        Returns
        -------
        Experience
            The generated REFLECTION experience.
        """

    @abc.abstractmethod
    def get_reflections(
        self,
        *,
        source_experience_id: str | None = None,
        limit: int = 20,
    ) -> list[Experience]:
        """Return stored reflections, optionally filtered by source.

        Parameters
        ----------
        source_experience_id:
            If supplied, return only reflections linked to this source.
        limit:
            Maximum number of results.

        Returns
        -------
        list[Experience]
            Matching REFLECTION experiences.
        """


# ---------------------------------------------------------------------------
# Experience Retrieval Engine Interface (future)
# ---------------------------------------------------------------------------


class ExperienceRetrievalEngineInterface(abc.ABC):
    """Contract for the Experience Retrieval Engine (future implementation).

    Provides high-level retrieval queries used heavily by ORION and ODYSSEY.
    Abstracts away the underlying storage query mechanics behind semantic
    questions such as "Have we done this before?" and "What worked previously?"

    Implementation requirements
    ---------------------------
    * Must increment retrieval counts for every experience returned.
    * Must support fuzzy temporal queries ("around June 2025").
    * Must support multi-modal queries combining time, topic, and type.
    """

    @abc.abstractmethod
    def find_similar(
        self,
        reference_experience_id: str,
        *,
        limit: int = 10,
    ) -> list[Experience]:
        """Return experiences similar to a reference experience.

        Similarity is determined by shared tags, same experience type,
        overlapping time period, and shared project references.

        Parameters
        ----------
        reference_experience_id:
            UUID of the reference experience.
        limit:
            Maximum number of results.

        Returns
        -------
        list[Experience]
            Similar experiences, most similar first.
        """

    @abc.abstractmethod
    def find_by_topic(
        self,
        topic: str,
        *,
        limit: int = 20,
    ) -> list[Experience]:
        """Return experiences related to a given topic string.

        Parameters
        ----------
        topic:
            A keyword or phrase to match against titles, descriptions,
            and tag names.
        limit:
            Maximum number of results.

        Returns
        -------
        list[Experience]
            Matching experiences, ordered by relevance then recency.
        """

    @abc.abstractmethod
    def find_by_time_range(
        self,
        start: datetime,
        end: datetime,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        limit: int = 50,
    ) -> list[Experience]:
        """Return experiences that occurred within a time window.

        Parameters
        ----------
        start:
            UTC start of the time window (inclusive).
        end:
            UTC end of the time window (inclusive).
        min_importance:
            Exclude experiences below this tier.
        limit:
            Maximum number of results.

        Returns
        -------
        list[Experience]
            Experiences in the window, ordered by ``occurred_at``.
        """

    @abc.abstractmethod
    def recall_context(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Reconstruct historical context for a natural-language query.

        Powers the Context Reconstruction Engine's ability to answer
        questions like "Why did we redesign ORION?" by assembling the
        relevant experience chain.

        Parameters
        ----------
        query:
            Natural-language question about past experiences.

        Returns
        -------
        dict[str, Any]
            Context bundle containing matched experiences, timeline, and
            a narrative reconstruction.
        """


# ---------------------------------------------------------------------------
# Memory Consolidation Engine Interface (future)
# ---------------------------------------------------------------------------


class MemoryConsolidationEngineInterface(abc.ABC):
    """Contract for the Memory Consolidation Engine (future implementation).

    Converts short-term experience records into long-term memory.  Without
    consolidation, ECHO would suffer memory explosion.  The consolidation
    engine prunes LOW importance records, promotes eligible MEDIUM records,
    and permanently archives CRITICAL records.

    Implementation requirements
    ---------------------------
    * Must never prune CRITICAL experiences.
    * Must never promote below-threshold experiences without Significance
      Engine approval.
    * Must emit ``ExperienceConsolidated`` events on each promotion.
    * Should run as a background process on a configurable schedule.
    """

    @abc.abstractmethod
    def run_consolidation_cycle(self) -> "ConsolidationReport":
        """Execute one full consolidation cycle.

        Returns
        -------
        ConsolidationReport
            Summary of what was pruned, promoted, and archived.
        """

    @abc.abstractmethod
    def consolidate_experience(self, experience_id: str) -> Experience:
        """Force-consolidate a specific experience into long-term memory.

        Parameters
        ----------
        experience_id:
            UUID of the experience to consolidate.

        Returns
        -------
        Experience
            The consolidated experience with updated metadata.
        """

    @abc.abstractmethod
    def get_consolidation_candidates(
        self,
        limit: int = 100,
    ) -> list[Experience]:
        """Return experiences eligible for consolidation in this cycle.

        Parameters
        ----------
        limit:
            Maximum candidates to return.

        Returns
        -------
        list[Experience]
            Unconsolidated experiences at MEDIUM or above.
        """


# ---------------------------------------------------------------------------
# Pattern Extraction Engine Interface (future)
# ---------------------------------------------------------------------------


class PatternExtractionEngineInterface(abc.ABC):
    """Contract for the Pattern Extraction Engine (future implementation).

    Discovers recurring experience structures.  ECHO stores individual
    events; ASTRA stores patterns derived from those events.  This engine
    sits at the boundary, detecting patterns in ECHO's experience store
    and publishing results to ASTRA and NOVA.

    ECHO Law 3: ECHO stores events.  ASTRA stores patterns.  Never merged.

    Implementation requirements
    ---------------------------
    * Must NOT store patterns in ECHO — patterns belong to ASTRA.
    * Must publish ``PatternDiscovered`` events for ASTRA to consume.
    * Must require a minimum recurrence count before declaring a pattern.
    """

    @abc.abstractmethod
    def extract_patterns(
        self,
        *,
        min_occurrences: int = 3,
        lookback_days: int = 90,
    ) -> list[dict[str, Any]]:
        """Scan recent experiences for recurring patterns.

        Parameters
        ----------
        min_occurrences:
            Minimum number of occurrences before a sequence is a pattern.
        lookback_days:
            How many days of experience history to scan.

        Returns
        -------
        list[dict[str, Any]]
            Detected pattern descriptors.  These are published to ASTRA,
            not stored in ECHO.
        """

    @abc.abstractmethod
    def get_known_patterns(self) -> list[dict[str, Any]]:
        """Return the current set of detected (not yet published) patterns.

        Returns
        -------
        list[dict[str, Any]]
            Pattern descriptors awaiting publication to ASTRA.
        """


# ---------------------------------------------------------------------------
# Personal History Engine Interface (future)
# ---------------------------------------------------------------------------


class PersonalHistoryEngineInterface(abc.ABC):
    """Contract for the Personal History Engine (future implementation).

    Maintains the user's life narrative by aggregating significant
    experiences into coherent journey timelines.  Provides continuity
    across years, creating the subjective sense of a personal story.

    Examples of journeys: University Journey, POLARIS Journey,
    Skill Development Journey, Career Journey.

    Implementation requirements
    ---------------------------
    * Must source data exclusively from ECHO's experience store.
    * Must NOT generate identity claims — that is ASTRA's domain.
    * Must produce time-ordered narrative slices on demand.
    * Must support multiple concurrent journey timelines.
    """

    @abc.abstractmethod
    def get_journey(self, journey_id: str) -> dict[str, Any]:
        """Return the full narrative for a specific journey.

        Parameters
        ----------
        journey_id:
            Identifier for the journey (e.g. ``"university"``,
            ``"polaris"``, ``"career"``).

        Returns
        -------
        dict[str, Any]
            Journey descriptor including title, milestones, and timeline.
        """

    @abc.abstractmethod
    def list_journeys(self) -> list[dict[str, Any]]:
        """Return summaries of all active journeys.

        Returns
        -------
        list[dict[str, Any]]
            Journey summaries, most recently updated first.
        """

    @abc.abstractmethod
    def add_milestone(
        self,
        journey_id: str,
        experience_id: str,
        *,
        milestone_label: str = "",
    ) -> None:
        """Add an experience as a milestone in a journey.

        Parameters
        ----------
        journey_id:
            Target journey identifier.
        experience_id:
            UUID of the :class:`Experience` to mark as a milestone.
        milestone_label:
            Optional human-readable label to attach to this milestone.
        """

    @abc.abstractmethod
    def generate_narrative_slice(
        self,
        journey_id: str,
        start: datetime,
        end: datetime,
    ) -> str:
        """Generate a narrative summary of a journey segment.

        Parameters
        ----------
        journey_id:
            Journey to summarise.
        start:
            UTC start of the narrative window.
        end:
            UTC end of the narrative window.

        Returns
        -------
        str
            Human-readable narrative of the journey in that period.
        """


# ---------------------------------------------------------------------------
# Placeholder: ConsolidationReport (used by MemoryConsolidationEngineInterface)
# ---------------------------------------------------------------------------


class ConsolidationReport:
    """Summary produced by a :meth:`MemoryConsolidationEngineInterface.run_consolidation_cycle` call.

    Attributes
    ----------
    pruned_count:
        Number of LOW experiences removed in this cycle.
    promoted_count:
        Number of MEDIUM experiences promoted to HIGH.
    archived_count:
        Number of CRITICAL experiences archived to permanent storage.
    errors:
        List of error messages encountered during the cycle.
    ran_at:
        UTC timestamp of when this consolidation cycle ran.
    """

    __slots__ = ("pruned_count", "promoted_count", "archived_count", "errors", "ran_at")

    def __init__(
        self,
        pruned_count: int,
        promoted_count: int,
        archived_count: int,
        errors: list[str],
        ran_at: datetime,
    ) -> None:
        self.pruned_count = pruned_count
        self.promoted_count = promoted_count
        self.archived_count = archived_count
        self.errors = errors
        self.ran_at = ran_at

    def __repr__(self) -> str:
        return (
            f"ConsolidationReport(pruned={self.pruned_count}, "
            f"promoted={self.promoted_count}, archived={self.archived_count})"
        )