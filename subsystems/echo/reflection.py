# subsystems/echo/reflection.py
"""
ECHO v1 Reflection Engine.

Implements :class:`ReflectionEngine` — the production engine responsible for
generating lessons and insights from experiences, achievements, and failures.

Reflection is the bridge between episodic memory and future learning.  The
Reflection Engine consumes :class:`~subsystems.echo.models.Experience`,
:class:`~subsystems.echo.models.AchievementRecord`, and
:class:`~subsystems.echo.models.FailureRecord` objects to produce
:class:`~subsystems.echo.models.Experience` objects of type ``REFLECTION``.
These reflections feed NOVA's recommendation engine and support ASTRA's
growth tracking.

Design Principles
-----------------
* Reflections are synthesised lessons derived from one or more prior
  experiences — they are not raw copies of source data.
* The Significance Engine gates every generated reflection.  LOW importance
  sources do not trigger reflections without an explicit ``force=True`` flag.
* Every reflection links back to its source(s) via
  ``metadata.related_experience_ids``.
* Reflections are stored as :class:`~subsystems.echo.models.Experience`
  objects of type ``REFLECTION`` in the Experience Engine's store.
* The engine maintains its own index of reflection-to-source links for
  fast reverse lookups without re-scanning the full experience store.

Thread safety
-------------
All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
before reading or modifying internal state.  Safe for concurrent use across
multiple threads as required by the POLARIS runtime.

Architecture notes
------------------
* In v1, the supplementary index (``_reflection_index``) is an in-process
  ``dict[str, list[str]]`` mapping ``reflection_id → [source_experience_ids]``
  and ``_source_to_reflections`` mapping ``source_id → [reflection_ids]``.
* The ``_insights_store`` is a ``dict[str, LearningInsight]`` keyed by
  ``insight_id`` accumulating generated insights across reflection calls.
* Improvement suggestions are ephemeral and computed on demand; they are
  not persisted independently.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from subsystems.echo.exceptions import (
    AchievementNotFoundError,
    EchoNotInitializedError,
    ExperienceNotFoundError,
    ExperienceValidationError,
    FailureNotFoundError,
)
from subsystems.echo.models import (
    AchievementRecord,
    Experience,
    ExperienceImportance,
    ExperienceMetadata,
    ExperienceType,
    FailureRecord,
    MemoryTag,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reflection-specific data structures
# ---------------------------------------------------------------------------


@dataclass
class LearningInsight:
    """A synthesised learning insight derived from one or more reflections.

    Insights are higher-order generalisations that emerge when the Reflection
    Engine detects common themes across multiple reflections or experiences.
    They feed ASTRA's growth tracking and NOVA's recommendation surface.

    Attributes
    ----------
    insight_id:
        UUID-4 unique identifier.
    summary:
        One-sentence distilled insight
        (e.g. ``"Scope creep consistently precedes deadline failures"``).
    detail:
        Extended explanation of the insight and the evidence behind it.
    domain:
        Broad domain this insight applies to
        (e.g. ``"software"``, ``"academic"``, ``"general"``).
    source_reflection_ids:
        UUIDs of the :class:`~subsystems.echo.models.Experience` REFLECTION
        objects that contributed to this insight.
    source_experience_ids:
        UUIDs of the original (non-reflection) source experiences.
    tags:
        :class:`~subsystems.echo.models.MemoryTag` list for indexing.
    importance:
        Importance tier of this insight.
    generated_at:
        UTC timestamp of generation.
    """

    summary: str
    insight_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detail: str = ""
    domain: str = "general"
    source_reflection_ids: list[str] = field(default_factory=list)
    source_experience_ids: list[str] = field(default_factory=list)
    tags: list[MemoryTag] = field(default_factory=list)
    importance: ExperienceImportance = ExperienceImportance.MEDIUM
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not self.summary or not self.summary.strip():
            raise ValueError("LearningInsight.summary must be a non-empty string.")


@dataclass
class ImprovementSuggestion:
    """An actionable improvement suggestion derived from reflection analysis.

    Suggestions are ephemeral advisory outputs computed on demand by
    :meth:`ReflectionEngine.generate_insights`.  They are not persisted
    independently of the reflections that produced them.

    Attributes
    ----------
    suggestion_id:
        UUID-4 unique identifier (ephemeral; not persisted).
    suggestion:
        Concrete, actionable recommendation
        (e.g. ``"Define scope boundaries before project kickoff"``).
    rationale:
        Why this suggestion is warranted, referencing experience evidence.
    domain:
        Broad domain this suggestion applies to.
    priority:
        ``"high"``, ``"medium"``, or ``"low"``.
    source_ids:
        UUIDs of the experiences or reflections that motivated this suggestion.
    generated_at:
        UTC timestamp of generation.
    """

    suggestion: str
    suggestion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rationale: str = ""
    domain: str = "general"
    priority: str = "medium"
    source_ids: list[str] = field(default_factory=list)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not self.suggestion or not self.suggestion.strip():
            raise ValueError(
                "ImprovementSuggestion.suggestion must be a non-empty string."
            )
        if self.priority not in {"high", "medium", "low"}:
            raise ValueError(
                "ImprovementSuggestion.priority must be 'high', 'medium', or 'low'."
            )


# ---------------------------------------------------------------------------
# Reflection Engine
# ---------------------------------------------------------------------------


class ReflectionEngine:
    """Production implementation of the ECHO Reflection Engine.

    Generates lessons and insights from experiences, achievements, and
    failures.  Reflection records are stored as
    :class:`~subsystems.echo.models.Experience` objects of type
    ``REFLECTION`` via the injected Experience Engine.

    Parameters
    ----------
    experience_engine:
        A running ExperienceEngine instance.  The Reflection Engine writes
        all generated reflections through this engine so they participate in
        the full ECHO lifecycle (significance scoring, deduplication, event
        publishing).  If ``None``, the engine operates in detached mode —
        generated reflections are held in an internal buffer and are not
        pushed to the Experience Engine's store.
    significance_engine:
        Optional reference to the running SignificanceEngine.  When supplied,
        source experiences are significance-checked before reflection
        generation unless ``force=True`` is passed.  When ``None``, all
        MEDIUM-and-above sources are accepted without threshold checks.
    achievement_engine:
        Optional reference to the running AchievementEngine.  Required for
        :meth:`analyze_achievement` cross-validation.
    failure_engine:
        Optional reference to the running FailureAnalysisEngine.  Required
        for :meth:`analyze_failure` cross-validation and for flagging
        processed failure records via ``FailureRecord.mark_reflected()``.
    min_importance_for_reflection:
        Minimum :class:`~subsystems.echo.models.ExperienceImportance` tier
        that a source experience must carry before a reflection will be
        generated without ``force=True``.  Defaults to ``MEDIUM``.
    source_subsystem:
        Label written into generated reflection metadata.  Defaults to
        ``"ECHO_REFLECTION_ENGINE"``.

    Usage
    -----
    ::

        from subsystems.echo.experience import ExperienceEngine
        from subsystems.echo.significance import SignificanceEngine
        from subsystems.echo.achievements import AchievementEngine
        from subsystems.echo.failure_analysis import FailureAnalysisEngine
        from subsystems.echo.reflection import ReflectionEngine

        sig = SignificanceEngine()
        sig.initialize()

        exp = ExperienceEngine(significance_engine=sig)
        exp.initialize()

        ach = AchievementEngine(experience_engine=exp)
        ach.initialize()

        fae = FailureAnalysisEngine(experience_engine=exp)
        fae.initialize()

        ref = ReflectionEngine(
            experience_engine=exp,
            significance_engine=sig,
            achievement_engine=ach,
            failure_engine=fae,
        )
        ref.initialize()

        # Generate a reflection from an existing experience
        source = exp.create_experience(
            title="Project scope doubled mid-sprint",
            experience_type=ExperienceType.EVENT,
            importance=ExperienceImportance.HIGH,
            description="Requirements changed significantly after kickoff.",
            outcome="Deadline missed by two weeks.",
        )
        reflection = ref.generate_reflection(source.experience_id)

        ref.shutdown()
    """

    def __init__(
        self,
        experience_engine: Any | None = None,
        significance_engine: Any | None = None,
        achievement_engine: Any | None = None,
        failure_engine: Any | None = None,
        *,
        min_importance_for_reflection: ExperienceImportance = ExperienceImportance.MEDIUM,
        source_subsystem: str = "ECHO_REFLECTION_ENGINE",
    ) -> None:
        self._experience_engine = experience_engine
        self._significance_engine = significance_engine
        self._achievement_engine = achievement_engine
        self._failure_engine = failure_engine
        self._min_importance = min_importance_for_reflection
        self._source_subsystem = source_subsystem

        # reflection_id → list[source_experience_id]
        self._reflection_index: dict[str, list[str]] = {}
        # source_experience_id → list[reflection_id]
        self._source_to_reflections: dict[str, list[str]] = {}

        # Detached-mode buffer: reflection_id → Experience (not pushed to exp engine)
        self._detached_store: dict[str, Experience] = {}

        # insight_id → LearningInsight
        self._insights_store: dict[str, LearningInsight] = {}

        self._lock = threading.RLock()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Must be called before any other public method.  Idempotent: repeated
        calls on an already-running engine are no-ops.
        """
        with self._lock:
            if self._running:
                return
            self._reflection_index.clear()
            self._source_to_reflections.clear()
            self._detached_store.clear()
            self._insights_store.clear()
            self._running = True
            _logger.info("ReflectionEngine initialised.")

    def shutdown(self) -> None:
        """Release all resources held by this engine.

        Marks the engine as stopped.  Calling any public method after
        ``shutdown()`` will raise
        :class:`~subsystems.echo.exceptions.EchoNotInitializedError`.
        """
        with self._lock:
            if not self._running:
                return
            reflection_count = (
                len(self._reflection_index) + len(self._detached_store)
            )
            self._running = False
            _logger.info(
                "ReflectionEngine shut down.  %d reflection(s) tracked.",
                reflection_count,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _importance_value(self, importance: ExperienceImportance) -> int:
        return importance.value

    def _is_above_min_importance(
        self, importance: ExperienceImportance
    ) -> bool:
        return self._importance_value(importance) >= self._importance_value(
            self._min_importance
        )

    def _publish_event(self, event_name: str, reflection: Experience) -> None:
        _logger.debug(
            "EVENT %s: reflection_id=%s title=%r importance=%s sources=%s",
            event_name,
            reflection.experience_id,
            reflection.title,
            reflection.importance.name,
            reflection.metadata.related_experience_ids,
        )

    def _register_reflection_links(
        self,
        reflection_id: str,
        source_ids: list[str],
    ) -> None:
        """Update both sides of the reflection ↔ source index."""
        self._reflection_index[reflection_id] = list(source_ids)
        for sid in source_ids:
            self._source_to_reflections.setdefault(sid, [])
            if reflection_id not in self._source_to_reflections[sid]:
                self._source_to_reflections[sid].append(reflection_id)

    def _get_experience(self, experience_id: str) -> Experience:
        """Retrieve an experience from the Experience Engine.

        Raises
        ------
        ExperienceNotFoundError
            If the Experience Engine is absent or the record does not exist.
        """
        if self._experience_engine is None:
            raise ExperienceNotFoundError(experience_id)
        return self._experience_engine.get_experience(experience_id)

    def _store_reflection_experience(
        self, reflection: Experience, *, force: bool
    ) -> Experience:
        """Persist a reflection Experience through the appropriate path."""
        if self._experience_engine is not None:
            return self._experience_engine.store_experience(
                reflection, force=force
            )
        # Detached mode — hold in internal buffer
        self._detached_store[reflection.experience_id] = reflection
        return reflection

    def _build_reflection_experience(
        self,
        title: str,
        lesson: str,
        context: str,
        outcome: str,
        source_ids: list[str],
        tags: list[MemoryTag],
        importance: ExperienceImportance,
        occurred_at: datetime | None = None,
    ) -> Experience:
        """Construct an Experience of type REFLECTION."""
        metadata = ExperienceMetadata(
            source_subsystem=self._source_subsystem,
            related_experience_ids=list(source_ids),
        )
        return Experience(
            title=title,
            experience_type=ExperienceType.REFLECTION,
            importance=importance,
            description=lesson,
            context=context,
            outcome=outcome,
            tags=list(tags),
            occurred_at=occurred_at or datetime.now(timezone.utc),
            metadata=metadata,
            extra={
                "source_experience_ids": list(source_ids),
                "reflection_engine_version": "v1",
            },
        )

    def _derive_reflection_importance(
        self, source_importance: ExperienceImportance
    ) -> ExperienceImportance:
        """Derive appropriate reflection importance from source importance.

        Reflections are one tier below their source by default, but floor
        at MEDIUM so they are always eligible for consolidation.
        """
        tier_map = {
            ExperienceImportance.LOW: ExperienceImportance.LOW,
            ExperienceImportance.MEDIUM: ExperienceImportance.MEDIUM,
            ExperienceImportance.HIGH: ExperienceImportance.MEDIUM,
            ExperienceImportance.CRITICAL: ExperienceImportance.HIGH,
        }
        return tier_map.get(source_importance, ExperienceImportance.MEDIUM)

    def _synthesise_lesson_from_experience(
        self, experience: Experience
    ) -> tuple[str, str, str]:
        """Synthesise (lesson, context, outcome) strings from an Experience.

        Returns
        -------
        tuple[str, str, str]
            ``(lesson, context, outcome)`` derived from the source experience.
        """
        type_label = experience.experience_type.name.lower().replace("_", " ")
        lesson_parts: list[str] = []
        if experience.outcome:
            lesson_parts.append(
                f"From this {type_label}: {experience.outcome}"
            )
        elif experience.description:
            lesson_parts.append(
                f"Reflecting on this {type_label}: {experience.description}"
            )
        else:
            lesson_parts.append(
                f"Reflection generated from {type_label}: {experience.title}"
            )

        lesson = " ".join(lesson_parts)
        context = (
            experience.context
            if experience.context
            else f"Derived from experience: {experience.title}"
        )
        outcome = (
            experience.outcome
            if experience.outcome
            else "See source experience for outcome details."
        )
        return lesson, context, outcome

    def _synthesise_lesson_from_achievement(
        self, record: AchievementRecord
    ) -> tuple[str, str, str]:
        """Synthesise (lesson, context, outcome) from an AchievementRecord."""
        evidence_str = (
            "; ".join(record.evidence[:3]) if record.evidence else "no evidence listed"
        )
        lesson = (
            f"Achievement in {record.domain}: {record.title}. "
            f"{record.description}".strip()
            or f"Completed: {record.title}"
        )
        context = f"Domain: {record.domain}. Evidence: {evidence_str}."
        outcome = f"Successfully accomplished: {record.title}."
        return lesson, context, outcome

    def _synthesise_lesson_from_failure(
        self, record: FailureRecord
    ) -> tuple[str, str, str]:
        """Synthesise (lesson, context, outcome) from a FailureRecord."""
        factors = "; ".join(record.contributing_factors[:5]) or "unspecified"
        lesson = (
            record.lesson.strip()
            if record.lesson and record.lesson.strip()
            else (
                f"Failure in {record.domain}: {record.description}".strip()
                or f"Lesson needed from failure: {record.title}"
            )
        )
        context = (
            f"Failure: {record.title}. "
            f"Domain: {record.domain}. "
            f"Contributing factors: {factors}."
        )
        outcome = (
            f"Reflected on failure '{record.title}' "
            f"to prevent recurrence."
        )
        return lesson, context, outcome

    # ------------------------------------------------------------------
    # Generate Reflection
    # ------------------------------------------------------------------

    def generate_reflection(
        self,
        source_experience_id: str,
        *,
        force: bool = False,
    ) -> Experience:
        """Generate and store a reflection from a single source experience.

        The source experience is retrieved from the Experience Engine.  If
        the source carries an importance below the configured minimum
        (default: MEDIUM) and ``force`` is ``False``, the generation is
        skipped and a :class:`~subsystems.echo.exceptions.ExperienceValidationError`
        is raised.

        If a reflection has already been generated for this source in the
        current session, a new one is still created — the Reflection Engine
        does not deduplicate reflections across calls (the Experience Engine's
        integrity check handles that at the storage layer).

        Parameters
        ----------
        source_experience_id:
            UUID of the :class:`~subsystems.echo.models.Experience` to
            reflect on.
        force:
            If ``True``, bypass the minimum-importance gate and also pass
            ``force=True`` to the Experience Engine's store call so that
            the reflection is persisted regardless of significance score.

        Returns
        -------
        Experience
            The generated and stored REFLECTION experience.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If ``source_experience_id`` does not exist in the Experience Engine.
        ExperienceValidationError
            If the source experience is below the minimum importance tier
            and ``force=False``.
        """
        self._assert_running("generate_reflection")

        source = self._get_experience(source_experience_id)

        if not force and not self._is_above_min_importance(source.importance):
            raise ExperienceValidationError(
                f"Source experience '{source_experience_id}' has importance "
                f"{source.importance.name}, which is below the minimum "
                f"{self._min_importance.name} required for reflection. "
                "Pass force=True to override.",
                field="importance",
            )

        lesson, context, outcome = self._synthesise_lesson_from_experience(source)
        reflection_importance = self._derive_reflection_importance(source.importance)
        tags = list(source.tags)
        _add_reflection_tag(tags)

        reflection_title = f"Reflection: {source.title}"

        reflection = self._build_reflection_experience(
            title=reflection_title,
            lesson=lesson,
            context=context,
            outcome=outcome,
            source_ids=[source_experience_id],
            tags=tags,
            importance=reflection_importance,
            occurred_at=datetime.now(timezone.utc),
        )

        with self._lock:
            stored = self._store_reflection_experience(reflection, force=force)
            self._register_reflection_links(
                stored.experience_id, [source_experience_id]
            )

        self._publish_event("ReflectionGenerated", stored)
        _logger.info(
            "ReflectionEngine: generated reflection %s from source %s.",
            stored.experience_id,
            source_experience_id,
        )
        return stored

    # ------------------------------------------------------------------
    # Store Reflection
    # ------------------------------------------------------------------

    def store_reflection(
        self,
        reflection: Experience,
        *,
        source_ids: list[str] | None = None,
        force: bool = False,
    ) -> Experience:
        """Persist a pre-constructed REFLECTION experience.

        Use this when the caller has already built the reflection object and
        simply needs it stored and indexed.  The reflection's
        ``experience_type`` must be ``REFLECTION``; any other type raises
        :class:`~subsystems.echo.exceptions.ExperienceValidationError`.

        Parameters
        ----------
        reflection:
            The :class:`~subsystems.echo.models.Experience` to store.
            Must have ``experience_type == ExperienceType.REFLECTION``.
        source_ids:
            Optional list of source experience UUIDs to index as the
            reflection's origins.  If ``None``, the engine reads
            ``reflection.metadata.related_experience_ids`` as a fallback.
        force:
            Bypass the significance threshold when storing.

        Returns
        -------
        Experience
            The stored reflection (with metadata updated in place).

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceValidationError
            If ``reflection.experience_type`` is not ``REFLECTION``.
        """
        self._assert_running("store_reflection")

        if reflection.experience_type != ExperienceType.REFLECTION:
            raise ExperienceValidationError(
                f"store_reflection expects an Experience of type REFLECTION, "
                f"got {reflection.experience_type.name}.",
                field="experience_type",
            )

        resolved_sources = (
            list(source_ids)
            if source_ids is not None
            else list(reflection.metadata.related_experience_ids)
        )

        with self._lock:
            stored = self._store_reflection_experience(reflection, force=force)
            self._register_reflection_links(
                stored.experience_id, resolved_sources
            )

        self._publish_event("ReflectionStored", stored)
        _logger.info(
            "ReflectionEngine: stored reflection %s (sources: %s).",
            stored.experience_id,
            resolved_sources,
        )
        return stored

    # ------------------------------------------------------------------
    # Update Reflection
    # ------------------------------------------------------------------

    def update_reflection(
        self,
        reflection_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        context: str | None = None,
        outcome: str | None = None,
        importance: ExperienceImportance | None = None,
        tags: list[MemoryTag] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Experience:
        """Update mutable fields of an existing REFLECTION experience.

        Delegates to the Experience Engine's ``update_experience`` when
        connected; otherwise updates the detached-mode buffer directly.

        Parameters
        ----------
        reflection_id:
            UUID of the REFLECTION experience to update.
        title:
            New title, or ``None`` to leave unchanged.
        description:
            New description (lesson), or ``None`` to leave unchanged.
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
            The updated REFLECTION experience.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If no reflection with ``reflection_id`` exists.
        ExperienceValidationError
            If the retrieved experience is not of type REFLECTION.
        """
        self._assert_running("update_reflection")

        with self._lock:
            if self._experience_engine is not None:
                # Verify it is a REFLECTION before delegating update
                existing = self._experience_engine.get_experience(reflection_id)
                if existing.experience_type != ExperienceType.REFLECTION:
                    raise ExperienceValidationError(
                        f"Experience '{reflection_id}' is not a REFLECTION "
                        f"(type={existing.experience_type.name}).",
                        field="experience_type",
                    )
                updated = self._experience_engine.update_experience(
                    reflection_id,
                    title=title,
                    description=description,
                    context=context,
                    outcome=outcome,
                    importance=importance,
                    tags=tags,
                    extra=extra,
                )
                _logger.info(
                    "ReflectionEngine: updated reflection %s.", reflection_id
                )
                return updated

            # Detached mode
            reflection = self._detached_store.get(reflection_id)
            if reflection is None:
                raise ExperienceNotFoundError(reflection_id)
            if reflection.experience_type != ExperienceType.REFLECTION:
                raise ExperienceValidationError(
                    f"Experience '{reflection_id}' is not a REFLECTION.",
                    field="experience_type",
                )
            if title is not None:
                if not title.strip():
                    raise ExperienceValidationError(
                        "title must be a non-empty string.", field="title"
                    )
                reflection.title = title
            if description is not None:
                reflection.description = description
            if context is not None:
                reflection.context = context
            if outcome is not None:
                reflection.outcome = outcome
            if importance is not None:
                reflection.importance = importance
            if tags is not None:
                reflection.tags = list(tags)
            if extra is not None:
                reflection.extra = dict(extra)
            _logger.info(
                "ReflectionEngine: updated reflection %s (detached mode).",
                reflection_id,
            )
            return reflection

    # ------------------------------------------------------------------
    # Delete Reflection
    # ------------------------------------------------------------------

    def delete_reflection(self, reflection_id: str) -> bool:
        """Remove a reflection from ECHO's store and the internal index.

        If the Experience Engine is connected, the delete is delegated there.
        In detached mode, only the internal buffer is affected.

        Parameters
        ----------
        reflection_id:
            UUID of the REFLECTION experience to remove.

        Returns
        -------
        bool
            ``True`` if the record was found and removed; ``False`` if it
            did not exist (idempotent delete).

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceValidationError
            If the experience exists but is not a REFLECTION.
        """
        self._assert_running("delete_reflection")

        with self._lock:
            # Clean up index regardless of storage mode
            source_ids = self._reflection_index.pop(reflection_id, [])
            for sid in source_ids:
                refs = self._source_to_reflections.get(sid, [])
                if reflection_id in refs:
                    refs.remove(reflection_id)
                    if not refs:
                        self._source_to_reflections.pop(sid, None)

            if self._experience_engine is not None:
                # Verify type before delegating delete
                try:
                    existing = self._experience_engine.get_experience(
                        reflection_id
                    )
                    if existing.experience_type != ExperienceType.REFLECTION:
                        raise ExperienceValidationError(
                            f"Experience '{reflection_id}' is not a REFLECTION "
                            f"(type={existing.experience_type.name}).",
                            field="experience_type",
                        )
                    result = self._experience_engine.delete_experience(
                        reflection_id
                    )
                except ExperienceNotFoundError:
                    result = False
                _logger.info(
                    "ReflectionEngine: delete_reflection %s → %s.",
                    reflection_id,
                    result,
                )
                return result

            # Detached mode
            removed = self._detached_store.pop(reflection_id, None)
            result = removed is not None
            _logger.info(
                "ReflectionEngine: delete_reflection %s (detached) → %s.",
                reflection_id,
                result,
            )
            return result

    # ------------------------------------------------------------------
    # Get Reflection
    # ------------------------------------------------------------------

    def get_reflection(self, reflection_id: str) -> Experience:
        """Retrieve a single REFLECTION experience by ID.

        Parameters
        ----------
        reflection_id:
            UUID of the REFLECTION experience to retrieve.

        Returns
        -------
        Experience
            The matching REFLECTION experience.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If no reflection with ``reflection_id`` exists.
        ExperienceValidationError
            If the experience exists but is not of type REFLECTION.
        """
        self._assert_running("get_reflection")

        with self._lock:
            if self._experience_engine is not None:
                experience = self._experience_engine.get_experience(
                    reflection_id
                )
                if experience.experience_type != ExperienceType.REFLECTION:
                    raise ExperienceValidationError(
                        f"Experience '{reflection_id}' is not a REFLECTION "
                        f"(type={experience.experience_type.name}).",
                        field="experience_type",
                    )
                return experience

            reflection = self._detached_store.get(reflection_id)
            if reflection is None:
                raise ExperienceNotFoundError(reflection_id)
            return reflection

    # ------------------------------------------------------------------
    # Search Reflections
    # ------------------------------------------------------------------

    def search_reflections(
        self,
        *,
        source_experience_id: str | None = None,
        domain_keyword: str | None = None,
        tag_names: list[str] | None = None,
        min_importance: ExperienceImportance | None = None,
        limit: int = 20,
    ) -> list[Experience]:
        """Search stored REFLECTION experiences with optional filters.

        Filters are applied with AND semantics — all supplied filters must
        match for a reflection to be included.

        Parameters
        ----------
        source_experience_id:
            If supplied, return only reflections linked to this source.
        domain_keyword:
            Case-insensitive substring matched against ``description``,
            ``context``, and ``title``.
        tag_names:
            If supplied, only reflections carrying at least one of the
            named tags are returned.
        min_importance:
            Exclude reflections below this importance tier.
        limit:
            Maximum number of results.  Defaults to 20.

        Returns
        -------
        list[Experience]
            Matching REFLECTION experiences, ordered by ``occurred_at``
            descending (most recent first).

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("search_reflections")

        with self._lock:
            # Determine candidate reflection IDs
            if source_experience_id is not None:
                candidate_ids = list(
                    self._source_to_reflections.get(source_experience_id, [])
                )
            else:
                candidate_ids = list(self._reflection_index.keys())

            # Resolve candidates to Experience objects
            candidates: list[Experience] = []
            for rid in candidate_ids:
                try:
                    exp = self._resolve_reflection(rid)
                    if exp is not None:
                        candidates.append(exp)
                except (ExperienceNotFoundError, ExperienceValidationError):
                    continue

            # Apply filters
            results: list[Experience] = []
            tag_set = set(tag_names) if tag_names else None
            for reflection in candidates:
                if min_importance is not None:
                    if (
                        self._importance_value(reflection.importance)
                        < self._importance_value(min_importance)
                    ):
                        continue
                if domain_keyword is not None:
                    kw = domain_keyword.lower()
                    haystack = " ".join([
                        reflection.title,
                        reflection.description,
                        reflection.context,
                    ]).lower()
                    if kw not in haystack:
                        continue
                if tag_set is not None:
                    reflection_tags = {t.name for t in reflection.tags}
                    if not tag_set.intersection(reflection_tags):
                        continue
                results.append(reflection)

            # Sort by occurred_at descending
            results.sort(key=lambda r: r.occurred_at, reverse=True)
            return results[:limit]

    def _resolve_reflection(self, reflection_id: str) -> Experience | None:
        """Resolve a reflection ID to an Experience object.

        Checks detached store first, then delegates to Experience Engine.
        Returns ``None`` if not found in either location.
        """
        detached = self._detached_store.get(reflection_id)
        if detached is not None:
            return detached
        if self._experience_engine is not None:
            try:
                return self._experience_engine.get_experience(reflection_id)
            except ExperienceNotFoundError:
                return None
        return None

    # ------------------------------------------------------------------
    # Analyse Experience
    # ------------------------------------------------------------------

    def analyze_experience(
        self,
        experience_id: str,
    ) -> dict[str, Any]:
        """Produce a structured analysis bundle for a single experience.

        Does not generate or store a reflection — use :meth:`generate_reflection`
        for that.  This method is a read-only analysis suitable for
        pre-flight checks, dashboards, and ASTRA growth summaries.

        Parameters
        ----------
        experience_id:
            UUID of the experience to analyse.

        Returns
        -------
        dict[str, Any]
            Analysis bundle containing:

            * ``experience_id``         — record UUID
            * ``title``                 — experience title
            * ``experience_type``       — type name string
            * ``importance``            — importance tier name
            * ``significance_score``    — from metadata (0.0–1.0)
            * ``description``           — narrative description
            * ``context``               — situational context
            * ``outcome``               — outcome text
            * ``tags``                  — list of tag name strings
            * ``tag_count``             — number of tags
            * ``occurred_at``           — UTC ISO-8601 string
            * ``recorded_at``           — UTC ISO-8601 string
            * ``age_days``              — float days since occurrence
            * ``retrieval_count``       — metadata retrieval count
            * ``consolidated``          — whether consolidated to LTM
            * ``related_experience_ids`` — metadata related IDs
            * ``reflection_ids``        — reflections generated from this source
            * ``has_reflections``       — bool
            * ``eligible_for_reflection`` — True if above min importance
            * ``session_id``            — enclosing session ID or None

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If ``experience_id`` does not exist.
        """
        self._assert_running("analyze_experience")

        experience = self._get_experience(experience_id)

        with self._lock:
            reflection_ids = list(
                self._source_to_reflections.get(experience_id, [])
            )

        now = datetime.now(timezone.utc)
        age_days = (now - experience.occurred_at).total_seconds() / 86_400.0

        return {
            "experience_id": experience.experience_id,
            "title": experience.title,
            "experience_type": experience.experience_type.name,
            "importance": experience.importance.name,
            "significance_score": experience.metadata.significance_score,
            "description": experience.description,
            "context": experience.context,
            "outcome": experience.outcome,
            "tags": [t.name for t in experience.tags],
            "tag_count": len(experience.tags),
            "occurred_at": experience.occurred_at.isoformat(),
            "recorded_at": experience.recorded_at.isoformat(),
            "age_days": round(age_days, 2),
            "retrieval_count": experience.metadata.retrieval_count,
            "consolidated": experience.metadata.consolidated,
            "related_experience_ids": list(
                experience.metadata.related_experience_ids
            ),
            "reflection_ids": reflection_ids,
            "has_reflections": bool(reflection_ids),
            "eligible_for_reflection": self._is_above_min_importance(
                experience.importance
            ),
            "session_id": experience.metadata.session_id,
        }

    # ------------------------------------------------------------------
    # Analyse Achievement
    # ------------------------------------------------------------------

    def analyze_achievement(
        self,
        achievement_id: str,
    ) -> dict[str, Any]:
        """Produce a structured analysis bundle for an achievement record.

        Retrieves the achievement from the Achievement Engine (if connected)
        and analyses it for reflection potential.

        Parameters
        ----------
        achievement_id:
            UUID of the :class:`~subsystems.echo.models.AchievementRecord`
            to analyse.

        Returns
        -------
        dict[str, Any]
            Analysis bundle containing:

            * ``achievement_id``          — record UUID
            * ``title``                   — achievement title
            * ``domain``                  — domain category
            * ``description``             — narrative description
            * ``evidence``                — evidence list
            * ``evidence_count``          — number of evidence items
            * ``importance``              — importance tier name
            * ``tags``                    — tag name list
            * ``experience_id``           — linked experience UUID or None
            * ``achieved_at``             — UTC ISO-8601 string
            * ``recorded_at``             — UTC ISO-8601 string
            * ``age_days``                — float days since achievement
            * ``reflection_ids``          — reflections sourced from the
                                            linked experience (if any)
            * ``has_reflections``         — bool
            * ``eligible_for_reflection`` — True if importance >= MEDIUM

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        AchievementNotFoundError
            If the Achievement Engine is absent or the record does not exist.
        """
        self._assert_running("analyze_achievement")

        if self._achievement_engine is None:
            raise AchievementNotFoundError(achievement_id)

        record: AchievementRecord = self._achievement_engine.get_achievement(
            achievement_id
        )

        with self._lock:
            reflection_ids = list(
                self._source_to_reflections.get(
                    record.experience_id or "", []
                )
            )

        now = datetime.now(timezone.utc)
        age_days = (now - record.achieved_at).total_seconds() / 86_400.0

        return {
            "achievement_id": record.achievement_id,
            "title": record.title,
            "domain": record.domain,
            "description": record.description,
            "evidence": list(record.evidence),
            "evidence_count": len(record.evidence),
            "importance": record.importance.name,
            "tags": [t.name for t in record.tags],
            "experience_id": record.experience_id,
            "achieved_at": record.achieved_at.isoformat(),
            "recorded_at": record.recorded_at.isoformat(),
            "age_days": round(age_days, 2),
            "reflection_ids": reflection_ids,
            "has_reflections": bool(reflection_ids),
            "eligible_for_reflection": self._is_above_min_importance(
                record.importance
            ),
        }

    # ------------------------------------------------------------------
    # Analyse Failure
    # ------------------------------------------------------------------

    def analyze_failure(
        self,
        failure_id: str,
    ) -> dict[str, Any]:
        """Produce a structured analysis bundle for a failure record.

        Retrieves the failure from the Failure Analysis Engine (if connected)
        and analyses it for reflection potential and lesson extraction.

        Parameters
        ----------
        failure_id:
            UUID of the :class:`~subsystems.echo.models.FailureRecord`
            to analyse.

        Returns
        -------
        dict[str, Any]
            Analysis bundle containing:

            * ``failure_id``             — record UUID
            * ``title``                  — failure title
            * ``domain``                 — domain category
            * ``description``            — objective description
            * ``contributing_factors``   — list of causal factors
            * ``factor_count``           — number of identified factors
            * ``lesson``                 — recorded lesson (may be empty)
            * ``has_lesson``             — bool
            * ``reflection_generated``   — whether already reflected on
            * ``importance``             — importance tier name
            * ``tags``                   — tag name list
            * ``experience_id``          — linked experience UUID or None
            * ``failed_at``              — UTC ISO-8601 string
            * ``recorded_at``            — UTC ISO-8601 string
            * ``age_days``               — float days since failure
            * ``reflection_ids``         — reflections sourced from the
                                           linked experience (if any)
            * ``has_reflections``        — bool
            * ``eligible_for_reflection`` — True if importance >= configured min

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        FailureNotFoundError
            If the Failure Engine is absent or the record does not exist.
        """
        self._assert_running("analyze_failure")

        if self._failure_engine is None:
            raise FailureNotFoundError(failure_id)

        record: FailureRecord = self._failure_engine.get_failure(failure_id)

        with self._lock:
            reflection_ids = list(
                self._source_to_reflections.get(
                    record.experience_id or "", []
                )
            )

        now = datetime.now(timezone.utc)
        age_days = (now - record.failed_at).total_seconds() / 86_400.0

        return {
            "failure_id": record.failure_id,
            "title": record.title,
            "domain": record.domain,
            "description": record.description,
            "contributing_factors": list(record.contributing_factors),
            "factor_count": len(record.contributing_factors),
            "lesson": record.lesson,
            "has_lesson": bool(record.lesson and record.lesson.strip()),
            "reflection_generated": record.reflection_generated,
            "importance": record.importance.name,
            "tags": [t.name for t in record.tags],
            "experience_id": record.experience_id,
            "failed_at": record.failed_at.isoformat(),
            "recorded_at": record.recorded_at.isoformat(),
            "age_days": round(age_days, 2),
            "reflection_ids": reflection_ids,
            "has_reflections": bool(reflection_ids),
            "eligible_for_reflection": self._is_above_min_importance(
                record.importance
            ),
        }

    # ------------------------------------------------------------------
    # Generate Reflection from Achievement
    # ------------------------------------------------------------------

    def generate_achievement_reflection(
        self,
        achievement_id: str,
        *,
        force: bool = False,
    ) -> Experience:
        """Generate and store a REFLECTION derived from an achievement record.

        If the achievement is linked to a parent experience, the reflection
        is registered against that parent experience ID.

        Parameters
        ----------
        achievement_id:
            UUID of the :class:`~subsystems.echo.models.AchievementRecord`
            to reflect on.
        force:
            Bypass minimum importance and significance threshold gates.

        Returns
        -------
        Experience
            The generated and stored REFLECTION experience.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        AchievementNotFoundError
            If the Achievement Engine is absent or the record does not exist.
        ExperienceValidationError
            If the achievement is below minimum importance and ``force=False``.
        """
        self._assert_running("generate_achievement_reflection")

        if self._achievement_engine is None:
            raise AchievementNotFoundError(achievement_id)

        record: AchievementRecord = self._achievement_engine.get_achievement(
            achievement_id
        )

        if not force and not self._is_above_min_importance(record.importance):
            raise ExperienceValidationError(
                f"Achievement '{achievement_id}' has importance "
                f"{record.importance.name}, which is below the minimum "
                f"{self._min_importance.name} required for reflection. "
                "Pass force=True to override.",
                field="importance",
            )

        lesson, context, outcome = self._synthesise_lesson_from_achievement(
            record
        )
        reflection_importance = self._derive_reflection_importance(
            record.importance
        )
        tags = list(record.tags)
        _add_reflection_tag(tags)

        source_ids = (
            [record.experience_id] if record.experience_id else []
        )

        reflection = self._build_reflection_experience(
            title=f"Reflection: {record.title}",
            lesson=lesson,
            context=context,
            outcome=outcome,
            source_ids=source_ids,
            tags=tags,
            importance=reflection_importance,
        )
        reflection.extra["achievement_id"] = achievement_id

        with self._lock:
            stored = self._store_reflection_experience(reflection, force=force)
            self._register_reflection_links(stored.experience_id, source_ids)

        self._publish_event("ReflectionGeneratedFromAchievement", stored)
        _logger.info(
            "ReflectionEngine: generated achievement reflection %s from achievement %s.",
            stored.experience_id,
            achievement_id,
        )
        return stored

    # ------------------------------------------------------------------
    # Generate Reflection from Failure
    # ------------------------------------------------------------------

    def generate_failure_reflection(
        self,
        failure_id: str,
        *,
        force: bool = False,
    ) -> Experience:
        """Generate and store a REFLECTION derived from a failure record.

        After storing the reflection, the failure record is marked via
        ``FailureRecord.mark_reflected()`` to indicate the Reflection Engine
        has processed it.

        Parameters
        ----------
        failure_id:
            UUID of the :class:`~subsystems.echo.models.FailureRecord`
            to reflect on.
        force:
            Bypass minimum importance and significance threshold gates.

        Returns
        -------
        Experience
            The generated and stored REFLECTION experience.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        FailureNotFoundError
            If the Failure Engine is absent or the record does not exist.
        ExperienceValidationError
            If the failure is below minimum importance and ``force=False``.
        """
        self._assert_running("generate_failure_reflection")

        if self._failure_engine is None:
            raise FailureNotFoundError(failure_id)

        record: FailureRecord = self._failure_engine.get_failure(failure_id)

        if not force and not self._is_above_min_importance(record.importance):
            raise ExperienceValidationError(
                f"Failure '{failure_id}' has importance "
                f"{record.importance.name}, which is below the minimum "
                f"{self._min_importance.name} required for reflection. "
                "Pass force=True to override.",
                field="importance",
            )

        lesson, context, outcome = self._synthesise_lesson_from_failure(record)
        reflection_importance = self._derive_reflection_importance(
            record.importance
        )
        tags = list(record.tags)
        _add_reflection_tag(tags)

        source_ids = (
            [record.experience_id] if record.experience_id else []
        )

        reflection = self._build_reflection_experience(
            title=f"Reflection: {record.title}",
            lesson=lesson,
            context=context,
            outcome=outcome,
            source_ids=source_ids,
            tags=tags,
            importance=reflection_importance,
        )
        reflection.extra["failure_id"] = failure_id

        with self._lock:
            stored = self._store_reflection_experience(reflection, force=force)
            self._register_reflection_links(stored.experience_id, source_ids)

        # Mark the failure record as reflected on
        record.mark_reflected()

        self._publish_event("ReflectionGeneratedFromFailure", stored)
        _logger.info(
            "ReflectionEngine: generated failure reflection %s from failure %s.",
            stored.experience_id,
            failure_id,
        )
        return stored

    # ------------------------------------------------------------------
    # Generate Multi-Source Reflection
    # ------------------------------------------------------------------

    def generate_multi_source_reflection(
        self,
        source_experience_ids: list[str],
        *,
        title: str | None = None,
        force: bool = False,
    ) -> Experience:
        """Generate a single REFLECTION synthesising multiple source experiences.

        Useful for session-level or thematic reflections that aggregate
        lessons from several experiences at once.  All source experiences
        must be at or above the configured minimum importance unless
        ``force=True``.

        Parameters
        ----------
        source_experience_ids:
            Non-empty list of source experience UUIDs to reflect on.
        title:
            Optional custom title.  If ``None``, a title is derived from the
            count of sources and their shared tags.
        force:
            Bypass minimum importance and significance threshold gates.

        Returns
        -------
        Experience
            The generated and stored REFLECTION experience.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceValidationError
            If ``source_experience_ids`` is empty, or if any source is below
            minimum importance and ``force=False``.
        ExperienceNotFoundError
            If any source ID does not exist.
        """
        self._assert_running("generate_multi_source_reflection")

        if not source_experience_ids:
            raise ExperienceValidationError(
                "source_experience_ids must contain at least one ID.",
                field="source_experience_ids",
            )

        sources: list[Experience] = []
        for sid in source_experience_ids:
            src = self._get_experience(sid)
            if not force and not self._is_above_min_importance(src.importance):
                raise ExperienceValidationError(
                    f"Source experience '{sid}' has importance "
                    f"{src.importance.name}, which is below the minimum "
                    f"{self._min_importance.name}. Pass force=True to override.",
                    field="importance",
                )
            sources.append(src)

        # Aggregate lessons, context, outcome
        lessons = [
            s.outcome or s.description or s.title
            for s in sources
            if s.outcome or s.description
        ]
        lesson = (
            " | ".join(lessons[:5])
            if lessons
            else f"Aggregated reflection across {len(sources)} experiences."
        )
        context_parts = [s.title for s in sources[:5]]
        context = f"Sources: {'; '.join(context_parts)}."
        outcome = (
            f"Synthesised reflection from {len(sources)} experience(s)."
        )

        # Union of all tags
        seen_names: set[str] = set()
        combined_tags: list[MemoryTag] = []
        for src in sources:
            for t in src.tags:
                if t.name not in seen_names:
                    combined_tags.append(t)
                    seen_names.add(t.name)
        _add_reflection_tag(combined_tags)

        # Importance = max of sources
        max_importance = max(sources, key=lambda s: s.importance.value).importance
        reflection_importance = self._derive_reflection_importance(max_importance)

        derived_title = (
            title
            if title
            else f"Reflection across {len(sources)} experience(s)"
        )

        reflection = self._build_reflection_experience(
            title=derived_title,
            lesson=lesson,
            context=context,
            outcome=outcome,
            source_ids=list(source_experience_ids),
            tags=combined_tags,
            importance=reflection_importance,
        )

        with self._lock:
            stored = self._store_reflection_experience(reflection, force=force)
            self._register_reflection_links(
                stored.experience_id, list(source_experience_ids)
            )

        self._publish_event("ReflectionGeneratedMultiSource", stored)
        _logger.info(
            "ReflectionEngine: generated multi-source reflection %s from %d sources.",
            stored.experience_id,
            len(source_experience_ids),
        )
        return stored

    # ------------------------------------------------------------------
    # Generate Insights
    # ------------------------------------------------------------------

    def generate_insights(
        self,
        *,
        source_experience_ids: list[str] | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[LearningInsight]:
        """Generate and store learning insights from existing reflections.

        Insights are higher-order generalisations synthesised from the
        currently stored reflections.  Each call returns a fresh list of
        insights computed from the available reflection corpus filtered by
        the optional constraints.

        The generated insights are stored in ``_insights_store`` and can be
        retrieved later via :meth:`get_stored_insights`.

        Parameters
        ----------
        source_experience_ids:
            If supplied, only reflections linked to these sources are
            considered.
        domain:
            If supplied, only reflections whose title, description, or
            context contain this keyword (case-insensitive) are considered.
        limit:
            Maximum number of insights to generate and return.

        Returns
        -------
        list[LearningInsight]
            Generated learning insights, ordered by importance descending.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("generate_insights")

        # Collect candidate reflections
        candidate_filter_kwargs: dict[str, Any] = {
            "domain_keyword": domain,
            "limit": 500,  # broad sweep; we filter further below
        }
        if source_experience_ids:
            # Generate per-source and union
            all_reflections: list[Experience] = []
            seen_rids: set[str] = set()
            for sid in source_experience_ids:
                for r in self.search_reflections(
                    source_experience_id=sid,
                    domain_keyword=domain,
                    limit=500,
                ):
                    if r.experience_id not in seen_rids:
                        all_reflections.append(r)
                        seen_rids.add(r.experience_id)
        else:
            all_reflections = self.search_reflections(**candidate_filter_kwargs)

        if not all_reflections:
            _logger.debug("generate_insights: no reflections found for query.")
            return []

        insights: list[LearningInsight] = []
        used_reflection_ids: set[str] = set()
        used_source_ids: set[str] = set()

        # Group reflections by the most frequent tag theme
        tag_groups: dict[str, list[Experience]] = {}
        for reflection in all_reflections:
            for tag in reflection.tags:
                if tag.name == "reflection":
                    continue
                tag_groups.setdefault(tag.name, []).append(reflection)

        # Generate an insight per prominent tag group (≥2 reflections)
        generated = 0
        for tag_name, group in sorted(
            tag_groups.items(), key=lambda kv: len(kv[1]), reverse=True
        ):
            if generated >= limit:
                break
            if len(group) < 2:
                continue

            group_reflection_ids = [r.experience_id for r in group]
            group_source_ids: list[str] = []
            for r in group:
                group_source_ids.extend(
                    r.metadata.related_experience_ids
                )

            dominant_importance = max(
                group, key=lambda r: r.importance.value
            ).importance

            summary = (
                f"Recurring theme across {len(group)} reflection(s) "
                f"tagged '{tag_name}'."
            )
            detail_parts = [r.description[:120] for r in group[:3] if r.description]
            detail = (
                " | ".join(detail_parts)
                if detail_parts
                else f"Multiple reflections share the tag '{tag_name}'."
            )

            insight = LearningInsight(
                summary=summary,
                detail=detail,
                domain=domain or "general",
                source_reflection_ids=group_reflection_ids,
                source_experience_ids=list(set(group_source_ids)),
                tags=[MemoryTag(name=tag_name, category="topic")],
                importance=dominant_importance,
            )

            with self._lock:
                self._insights_store[insight.insight_id] = insight

            insights.append(insight)
            used_reflection_ids.update(group_reflection_ids)
            used_source_ids.update(group_source_ids)
            generated += 1

        # Generate a catch-all insight from any remaining reflections
        remaining = [
            r for r in all_reflections
            if r.experience_id not in used_reflection_ids
        ]
        if remaining and generated < limit:
            remaining_importance = max(
                remaining, key=lambda r: r.importance.value
            ).importance
            summary = (
                f"{len(remaining)} reflection(s) with no dominant recurring theme."
            )
            detail = "; ".join(
                [r.title for r in remaining[:5]]
            )
            remaining_source_ids: list[str] = []
            for r in remaining:
                remaining_source_ids.extend(r.metadata.related_experience_ids)
            catch_all = LearningInsight(
                summary=summary,
                detail=detail,
                domain=domain or "general",
                source_reflection_ids=[r.experience_id for r in remaining],
                source_experience_ids=list(set(remaining_source_ids)),
                importance=remaining_importance,
            )
            with self._lock:
                self._insights_store[catch_all.insight_id] = catch_all
            insights.append(catch_all)

        insights.sort(key=lambda i: i.importance.value, reverse=True)
        _logger.info(
            "ReflectionEngine: generated %d insight(s) from %d reflection(s).",
            len(insights),
            len(all_reflections),
        )
        return insights[:limit]

    # ------------------------------------------------------------------
    # Get Stored Insights
    # ------------------------------------------------------------------

    def get_stored_insights(
        self,
        *,
        min_importance: ExperienceImportance | None = None,
        limit: int = 50,
    ) -> list[LearningInsight]:
        """Return previously generated :class:`LearningInsight` objects.

        Parameters
        ----------
        min_importance:
            If supplied, exclude insights below this tier.
        limit:
            Maximum number of results.

        Returns
        -------
        list[LearningInsight]
            Stored insights, most recently generated first.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("get_stored_insights")

        with self._lock:
            insights = list(self._insights_store.values())

        if min_importance is not None:
            min_val = self._importance_value(min_importance)
            insights = [
                i for i in insights
                if self._importance_value(i.importance) >= min_val
            ]

        insights.sort(key=lambda i: i.generated_at, reverse=True)
        return insights[:limit]

    # ------------------------------------------------------------------
    # Get Improvement Suggestions
    # ------------------------------------------------------------------

    def get_improvement_suggestions(
        self,
        *,
        source_experience_ids: list[str] | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[ImprovementSuggestion]:
        """Derive actionable improvement suggestions from stored reflections.

        Suggestions are computed on demand from the reflection corpus and
        are not independently persisted.

        Parameters
        ----------
        source_experience_ids:
            Scope suggestions to reflections from these source experiences.
        domain:
            Filter by domain keyword (case-insensitive substring match).
        limit:
            Maximum number of suggestions to return.

        Returns
        -------
        list[ImprovementSuggestion]
            Actionable improvement suggestions, highest priority first.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("get_improvement_suggestions")

        # Collect relevant reflections
        if source_experience_ids:
            reflections: list[Experience] = []
            seen: set[str] = set()
            for sid in source_experience_ids:
                for r in self.search_reflections(
                    source_experience_id=sid,
                    domain_keyword=domain,
                    limit=200,
                ):
                    if r.experience_id not in seen:
                        reflections.append(r)
                        seen.add(r.experience_id)
        else:
            reflections = self.search_reflections(
                domain_keyword=domain, limit=200
            )

        if not reflections:
            return []

        suggestions: list[ImprovementSuggestion] = []

        # One suggestion per HIGH/CRITICAL reflection with outcome text
        for reflection in reflections:
            if len(suggestions) >= limit:
                break
            if reflection.importance not in (
                ExperienceImportance.HIGH,
                ExperienceImportance.CRITICAL,
            ):
                continue
            if not reflection.description:
                continue

            priority = (
                "high"
                if reflection.importance == ExperienceImportance.CRITICAL
                else "medium"
            )
            suggestion_text = (
                f"Review and act on: {reflection.title}."
            )
            rationale = reflection.description[:200]
            source_ids = list(reflection.metadata.related_experience_ids)
            source_ids.append(reflection.experience_id)

            suggestions.append(
                ImprovementSuggestion(
                    suggestion=suggestion_text,
                    rationale=rationale,
                    domain=domain or "general",
                    priority=priority,
                    source_ids=source_ids,
                )
            )

        # Fill remaining slots with MEDIUM reflection suggestions
        for reflection in reflections:
            if len(suggestions) >= limit:
                break
            if reflection.importance != ExperienceImportance.MEDIUM:
                continue
            if not reflection.description:
                continue

            source_ids = list(reflection.metadata.related_experience_ids)
            source_ids.append(reflection.experience_id)

            suggestions.append(
                ImprovementSuggestion(
                    suggestion=f"Consider reviewing: {reflection.title}.",
                    rationale=reflection.description[:200],
                    domain=domain or "general",
                    priority="low",
                    source_ids=source_ids,
                )
            )

        _logger.debug(
            "ReflectionEngine: derived %d improvement suggestion(s).",
            len(suggestions),
        )
        return suggestions[:limit]

    # ------------------------------------------------------------------
    # Link Reflection to Experience
    # ------------------------------------------------------------------

    def link_reflection_to_experience(
        self,
        reflection_id: str,
        source_experience_id: str,
    ) -> None:
        """Manually link an existing reflection to an additional source experience.

        Use this when an existing reflection is discovered to be relevant to
        a source that was not identified at generation time.

        Parameters
        ----------
        reflection_id:
            UUID of the REFLECTION experience to link.
        source_experience_id:
            UUID of the source experience to associate.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If the reflection or source experience does not exist.
        ExperienceValidationError
            If the reflection experience is not of type REFLECTION.
        """
        self._assert_running("link_reflection_to_experience")

        # Validate reflection exists and is of correct type
        self.get_reflection(reflection_id)

        # Validate source exists
        self._get_experience(source_experience_id)

        with self._lock:
            sources = self._reflection_index.get(reflection_id, [])
            if source_experience_id not in sources:
                sources.append(source_experience_id)
                self._reflection_index[reflection_id] = sources

            self._source_to_reflections.setdefault(
                source_experience_id, []
            )
            if reflection_id not in self._source_to_reflections[source_experience_id]:
                self._source_to_reflections[source_experience_id].append(
                    reflection_id
                )

        _logger.debug(
            "ReflectionEngine: linked reflection %s → source %s.",
            reflection_id,
            source_experience_id,
        )

    # ------------------------------------------------------------------
    # Snapshot / diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of internal engine state.

        Intended for health-check endpoints, monitoring dashboards, and
        the Runtime Kernel's health subsystem.

        Returns
        -------
        dict[str, Any]
            Snapshot containing:

            * ``running``               — bool
            * ``reflection_count``      — total indexed reflections
            * ``detached_count``        — reflections in detached buffer
            * ``source_count``          — unique source experience IDs indexed
            * ``insight_count``         — stored LearningInsight objects
            * ``min_importance``        — configured minimum importance name
            * ``source_subsystem``      — label for generated records
            * ``has_experience_engine`` — bool
            * ``has_significance_engine`` — bool
            * ``has_achievement_engine`` — bool
            * ``has_failure_engine``    — bool

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("snapshot")

        with self._lock:
            return {
                "running": self._running,
                "reflection_count": len(self._reflection_index),
                "detached_count": len(self._detached_store),
                "source_count": len(self._source_to_reflections),
                "insight_count": len(self._insights_store),
                "min_importance": self._min_importance.name,
                "source_subsystem": self._source_subsystem,
                "has_experience_engine": self._experience_engine is not None,
                "has_significance_engine": self._significance_engine is not None,
                "has_achievement_engine": self._achievement_engine is not None,
                "has_failure_engine": self._failure_engine is not None,
            }


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _add_reflection_tag(tags: list[MemoryTag]) -> None:
    """Append a ``reflection`` topic tag if not already present."""
    for t in tags:
        if t.name == "reflection":
            return
    tags.append(MemoryTag(name="reflection", category="topic"))