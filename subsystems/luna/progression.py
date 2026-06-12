"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/progression.py

Concrete in-memory implementation of the LUNA Skill Progression Engine.

Manages the structural progression of skills within LUNA: how a skill is
organized into mastery stages (Novice → Beginner → Intermediate → Advanced →
Expert → Master), what prerequisites must be satisfied before each stage is
reachable, and which learning paths are recommended based on current and target
proficiency.

This engine tracks skill STRUCTURE — the canonical model of what competency
looks like at each stage. It does NOT track individual user proficiency; that
is ASTRA's domain.

Responsibilities:
    track_skill_progression      — record a progression model against a skill
    advance_skill_stage          — evaluate and advance the active stage within a model
    evaluate_mastery             — determine mastery status for a given stage
    calculate_progress           — compute a normalized progress score across stages
    identify_skill_gaps          — find unmet stage requirements given current coverage
    generate_learning_recommendations — produce ordered learning recommendations
    determine_next_skills        — suggest skills to pursue after a given skill
    progression_history          — retrieve the ordered mutation history for a model
    progression_audit            — structured audit of a single model
    progression_reporting        — aggregate report across all models

Integrations:
    skills.py    — SkillEngine: resolves Skill records by ID, links progression models
    education.py — EducationalKnowledgeEngine: referenced for learning path alignment
    evolution.py — KnowledgeEvolutionEngine: mutation provenance (read-only)

Thread safety:  threading.RLock on all public operations.
Lifecycle-gated: every public method raises LunaNotInitializedError before
    initialize() or after shutdown().

In-memory v1 implementation. No persistence layer.

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
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from subsystems.luna.exceptions import (
    LunaLifecycleError,
    LunaNotInitializedError,
    SkillNotFoundError,
    SkillProgressionError,
)
from subsystems.luna.interfaces import AbstractSkillProgressionEngine
from subsystems.luna.models import (
    KnowledgeDifficulty,
    KnowledgeStatus,
    Skill,
    SkillLevel,
    SkillPrerequisite,
    SkillProgressionModel,
    SkillStage,
    _new_id,
    _utcnow,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE_VERSION: str = "5.0.0"
_MAX_STAGES: int = 10

# Ordered mastery levels used for stage validation and progress scoring
_ORDERED_LEVELS: list[SkillLevel] = [
    SkillLevel.NOVICE,
    SkillLevel.BEGINNER,
    SkillLevel.INTERMEDIATE,
    SkillLevel.ADVANCED,
    SkillLevel.EXPERT,
    SkillLevel.MASTER,
]

# Mastery threshold: stage is considered mastered when ≥ this fraction of
# its knowledge requirements are covered by the provided coverage set.
_MASTERY_THRESHOLD: float = 0.80


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

def _make_history_entry(
    *,
    entry_id: str,
    model_id: str,
    op: str,
    actor: str,
    notes: str,
    occurred_at: datetime,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a canonical, immutable progression history entry dict."""
    return {
        "id": entry_id,
        "model_id": model_id,
        "op": op,
        "actor": actor,
        "notes": notes,
        "occurred_at": occurred_at.isoformat(),
        "payload": payload or {},
    }


def _make_gap_entry(
    *,
    stage_level: SkillLevel,
    stage_label: str,
    missing_concept_ids: list[str],
    missing_fact_ids: list[str],
    missing_procedure_ids: list[str],
    missing_competencies: list[str],
) -> dict[str, Any]:
    """Return a canonical skill-gap dict for a single stage."""
    return {
        "stage_level": stage_level.value,
        "stage_label": stage_label,
        "missing_concept_ids": missing_concept_ids,
        "missing_fact_ids": missing_fact_ids,
        "missing_procedure_ids": missing_procedure_ids,
        "missing_competencies": missing_competencies,
        "total_gaps": (
            len(missing_concept_ids)
            + len(missing_fact_ids)
            + len(missing_procedure_ids)
            + len(missing_competencies)
        ),
    }


def _make_recommendation(
    *,
    rank: int,
    stage_level: SkillLevel,
    stage_label: str,
    concept_ids: list[str],
    fact_ids: list[str],
    procedure_ids: list[str],
    demonstration_tasks: list[str],
    rationale: str,
    estimated_hours: Optional[float],
) -> dict[str, Any]:
    """Return a canonical learning recommendation dict."""
    return {
        "rank": rank,
        "stage_level": stage_level.value,
        "stage_label": stage_label,
        "concept_ids": concept_ids,
        "fact_ids": fact_ids,
        "procedure_ids": procedure_ids,
        "demonstration_tasks": demonstration_tasks,
        "rationale": rationale,
        "estimated_hours": estimated_hours,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SkillProgressionEngine(AbstractSkillProgressionEngine):
    """
    In-memory, thread-safe implementation of the LUNA Skill Progression Engine
    (v1).

    Data stores:
        _model_store        — dict[model_id, SkillProgressionModel]
        _prerequisite_store — dict[prereq_id, SkillPrerequisite]

    Secondary indexes:
        _skill_to_model     — skill_id → model_id (one model per skill)
        _prereqs_by_model   — model_id → set[prereq_id]

    History & audit:
        _history            — model_id → list[history_entry dict]  (append-only)

    Optional integration handles (injected; may be None in v1):
        _skill_engine       — reference to SkillEngine for Skill resolution
        _education_engine   — reference to EducationalKnowledgeEngine
        _evolution_engine   — reference to KnowledgeEvolutionEngine

    Lifecycle::

        engine = SkillProgressionEngine(skill_engine=my_skill_engine)
        engine.initialize()
        model = engine.create_progression_model(skill_id="...", stages=[...])
        report = engine.progression_reporting()
        engine.shutdown()
    """

    def __init__(
        self,
        *,
        skill_engine: Optional[Any] = None,
        education_engine: Optional[Any] = None,
        evolution_engine: Optional[Any] = None,
    ) -> None:
        self._skill_engine = skill_engine
        self._education_engine = education_engine
        self._evolution_engine = evolution_engine

        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False
        self._started_at: Optional[datetime] = None

        # Primary stores
        self._model_store: dict[str, SkillProgressionModel] = {}
        self._prerequisite_store: dict[str, SkillPrerequisite] = {}

        # Secondary indexes
        self._skill_to_model: dict[str, str] = {}                    # skill_id → model_id
        self._prereqs_by_model: dict[str, set[str]] = defaultdict(set)  # model_id → set[prereq_id]

        # Append-only history: model_id → list[entry dict]
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Observability counters
        self._models_created: int = 0
        self._models_updated: int = 0
        self._models_deleted: int = 0
        self._prereqs_added: int = 0
        self._prereqs_removed: int = 0
        self._last_mutation_at: Optional[datetime] = None

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Prepare the Skill Progression Engine for operation.

        Idempotent: repeated calls after first initialization are no-ops.

        Raises:
            LunaLifecycleError: If internal setup fails.
        """
        with self._lock:
            if self._initialized:
                return
            try:
                self._model_store.clear()
                self._prerequisite_store.clear()
                self._skill_to_model.clear()
                self._prereqs_by_model.clear()
                self._history.clear()
                self._models_created = 0
                self._models_updated = 0
                self._models_deleted = 0
                self._prereqs_added = 0
                self._prereqs_removed = 0
                self._last_mutation_at = None
                self._started_at = _utcnow()
                self._initialized = True
                logger.info(
                    "SkillProgressionEngine initialized (version=%s)",
                    _ENGINE_VERSION,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="SkillProgressionEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        """
        Release resources and put the engine into a quiescent state.

        Idempotent: calling shutdown() on an already-stopped engine is a no-op.

        Raises:
            LunaLifecycleError: If teardown fails.
        """
        with self._lock:
            if not self._initialized:
                return
            try:
                self._initialized = False
                logger.info(
                    "SkillProgressionEngine shutdown "
                    "(models=%d, prereqs=%d, mutations=%d)",
                    len(self._model_store),
                    len(self._prerequisite_store),
                    self._models_created + self._models_updated + self._models_deleted,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="SkillProgressionEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def is_initialized(self) -> bool:
        return self._initialized

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL GUARDS & HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation=operation)

    def _resolve_model(
        self,
        model_id: str,
        operation: str,
    ) -> SkillProgressionModel:
        model = self._model_store.get(model_id)
        if model is None:
            raise SkillProgressionError(
                skill_id=model_id,
                message=f"Progression model not found: '{model_id}'",
                context={"operation": operation},
            )
        return model

    def _resolve_skill(self, skill_id: str, operation: str) -> Optional[Skill]:
        """
        Attempt to resolve a Skill from the injected SkillEngine.
        Returns None when no SkillEngine is wired (permissive mode).
        Raises SkillNotFoundError if the engine is present but the skill is absent.
        """
        if self._skill_engine is None:
            return None
        try:
            return self._skill_engine.retrieve_skill(skill_id)
        except Exception:
            raise SkillNotFoundError(
                skill_id=skill_id,
                context={"operation": operation},
            )

    def _validate_stages(
        self,
        stages: list[SkillStage],
        skill_id: str,
    ) -> None:
        """
        Validate that a stage list is structurally sound:
            - At least one stage.
            - At most _MAX_STAGES stages.
            - No duplicate levels.
            - Levels appear in non-decreasing rank order.
        """
        if not stages:
            raise SkillProgressionError(
                skill_id=skill_id,
                message="Stages list must not be empty.",
            )
        if len(stages) > _MAX_STAGES:
            raise SkillProgressionError(
                skill_id=skill_id,
                message=(
                    f"Too many stages: {len(stages)} exceeds limit of {_MAX_STAGES}."
                ),
            )
        seen_levels: set[SkillLevel] = set()
        prev_rank: int = -1
        for stage in stages:
            if stage.level in seen_levels:
                raise SkillProgressionError(
                    skill_id=skill_id,
                    stage=stage.level.value,
                    message=f"Duplicate stage level: '{stage.level.value}'.",
                )
            if stage.level.rank < prev_rank:
                raise SkillProgressionError(
                    skill_id=skill_id,
                    stage=stage.level.value,
                    message=(
                        f"Stage '{stage.level.value}' is out of order "
                        f"(rank {stage.level.rank} < previous rank {prev_rank})."
                    ),
                )
            seen_levels.add(stage.level)
            prev_rank = stage.level.rank

    def _record_history(
        self,
        model_id: str,
        op: str,
        actor: str = "system",
        notes: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        entry = _make_history_entry(
            entry_id=_new_id(),
            model_id=model_id,
            op=op,
            actor=actor,
            notes=notes,
            occurred_at=_utcnow(),
            payload=payload,
        )
        self._history[model_id].append(entry)
        self._last_mutation_at = _utcnow()

    def _touch_model(self, model: SkillProgressionModel) -> None:
        """Update the model's updated_at timestamp in-place."""
        model.updated_at = _utcnow()  # type: ignore[misc]

    # ─────────────────────────────────────────────────────────────────────────
    # CORE CRUD — PROGRESSION MODEL
    # ─────────────────────────────────────────────────────────────────────────

    def create_progression_model(
        self,
        skill_id: str,
        stages: list[SkillStage],
        *,
        skill_name: Optional[str] = None,
        description: str = "",
        notes: str = "",
    ) -> SkillProgressionModel:
        """
        Define the progression model for a skill.

        Only one progression model may exist per skill. Creating a second model
        for the same skill raises SkillProgressionError.

        Args:
            skill_id: The LUNA Skill this model describes.
            stages:   Ordered list of SkillStage records (Novice → Master).
            notes:    Optional free-text notes attached to the history entry.

        Returns:
            The newly created SkillProgressionModel.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      skill_id does not resolve (when SkillEngine
                                     is wired).
            SkillProgressionError:   Stages are invalid, out of order, or a
                                     model already exists for this skill.
        """
        self._require_initialized("create_progression_model")

        with self._lock:
            if skill_id in self._skill_to_model:
                existing_model_id = self._skill_to_model[skill_id]
                raise SkillProgressionError(
                    skill_id=skill_id,
                    message=(
                        f"A progression model already exists for skill '{skill_id}': "
                        f"model_id='{existing_model_id}'."
                    ),
                    context={"existing_model_id": existing_model_id},
                )

            self._validate_stages(stages, skill_id)

            skill = self._resolve_skill(skill_id, "create_progression_model")
            resolved_name = skill_name or (skill.name if skill is not None else skill_id)

            model = SkillProgressionModel.create(
                skill_id=skill_id,
                skill_name=resolved_name,
                description=description or notes or f"Progression model for skill '{resolved_name}'.",
                stages=stages,
            )

            self._model_store[model.id] = model
            self._skill_to_model[skill_id] = model.id

            self._record_history(
                model_id=model.id,
                op="create",
                notes=notes,
                payload={
                    "skill_id": skill_id,
                    "skill_name": resolved_name,
                    "stage_count": len(stages),
                },
            )
            self._models_created += 1
            logger.debug(
                "SkillProgressionEngine: created model '%s' for skill '%s'.",
                model.id,
                skill_id,
            )
            return model

    def update_progression_model(
        self,
        model_id: str,
        *,
        stages: Optional[list[SkillStage]] = None,
        notes: Optional[str] = None,
    ) -> SkillProgressionModel:
        """
        Update an existing progression model.

        Only the fields explicitly provided are mutated.

        Args:
            model_id: ID of the model to update.
            stages:   Replacement stage list. If provided, replaces all stages.
            notes:    Replacement description / notes.

        Returns:
            The mutated SkillProgressionModel.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found or stages are invalid.
        """
        self._require_initialized("update_progression_model")

        with self._lock:
            model = self._resolve_model(model_id, "update_progression_model")

            if stages is not None:
                self._validate_stages(stages, model.skill_id)
                model.stages = stages  # type: ignore[misc]

            if notes is not None:
                model.description = notes  # type: ignore[misc]

            self._touch_model(model)

            self._record_history(
                model_id=model.id,
                op="update",
                notes=notes or "",
                payload={
                    "stages_replaced": stages is not None,
                    "new_stage_count": len(model.stages) if stages is not None else None,
                },
            )
            self._models_updated += 1
            logger.debug(
                "SkillProgressionEngine: updated model '%s'.",
                model_id,
            )
            return model

    def delete_progression_model(self, model_id: str) -> SkillProgressionModel:
        """
        Remove a progression model from the store.

        Also removes all prerequisite records attached to the model.

        Args:
            model_id: ID of the model to remove.

        Returns:
            The removed SkillProgressionModel.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("delete_progression_model")

        with self._lock:
            model = self._resolve_model(model_id, "delete_progression_model")

            # Remove attached prerequisites
            prereq_ids = set(self._prereqs_by_model.get(model_id, set()))
            for pid in prereq_ids:
                self._prerequisite_store.pop(pid, None)
            self._prereqs_by_model.pop(model_id, None)

            # Remove skill → model mapping
            self._skill_to_model.pop(model.skill_id, None)

            # Remove from primary store
            del self._model_store[model_id]

            self._record_history(
                model_id=model_id,
                op="delete",
                payload={"skill_id": model.skill_id},
            )
            self._models_deleted += 1
            logger.debug(
                "SkillProgressionEngine: deleted model '%s' for skill '%s'.",
                model_id,
                model.skill_id,
            )
            return model

    def retrieve_progression_model(self, model_id: str) -> SkillProgressionModel:
        """
        Fetch a progression model by ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("retrieve_progression_model")
        with self._lock:
            return self._resolve_model(model_id, "retrieve_progression_model")

    def get_model_for_skill(self, skill_id: str) -> Optional[SkillProgressionModel]:
        """
        Return the progression model for a skill, or None if not defined.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_model_for_skill")
        with self._lock:
            model_id = self._skill_to_model.get(skill_id)
            if model_id is None:
                return None
            return self._model_store.get(model_id)

    def get_stage(
        self,
        model_id: str,
        level: SkillLevel,
    ) -> Optional[SkillStage]:
        """
        Return the SkillStage matching a specific SkillLevel within a model.

        Returns None if the model has no stage at that level.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("get_stage")
        with self._lock:
            model = self._resolve_model(model_id, "get_stage")
            return model.get_stage(level)

    def list_all_models(self) -> list[SkillProgressionModel]:
        """
        Return all progression models in insertion order.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("list_all_models")
        with self._lock:
            return list(self._model_store.values())

    # ─────────────────────────────────────────────────────────────────────────
    # PREREQUISITES
    # ─────────────────────────────────────────────────────────────────────────

    def add_prerequisite(
        self,
        model_id: str,
        prerequisite: SkillPrerequisite,
    ) -> SkillProgressionModel:
        """
        Add a prerequisite skill to a progression model.

        Duplicate prerequisite_skill_id entries are silently replaced.

        Args:
            model_id:     ID of the progression model.
            prerequisite: The SkillPrerequisite to attach.

        Returns:
            The updated SkillProgressionModel.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("add_prerequisite")

        with self._lock:
            model = self._resolve_model(model_id, "add_prerequisite")

            # Replace any existing prerequisite for the same prerequisite skill
            existing_prereq_id: Optional[str] = None
            for pid in list(self._prereqs_by_model.get(model_id, set())):
                pr = self._prerequisite_store.get(pid)
                if pr is not None and pr.prerequisite_skill_id == prerequisite.prerequisite_skill_id:
                    existing_prereq_id = pid
                    break

            if existing_prereq_id is not None:
                del self._prerequisite_store[existing_prereq_id]
                self._prereqs_by_model[model_id].discard(existing_prereq_id)

            self._prerequisite_store[prerequisite.id] = prerequisite
            self._prereqs_by_model[model_id].add(prerequisite.id)

            self._touch_model(model)
            self._record_history(
                model_id=model_id,
                op="add_prerequisite",
                payload={
                    "prerequisite_id": prerequisite.id,
                    "prerequisite_skill_id": prerequisite.prerequisite_skill_id,
                    "minimum_level": prerequisite.minimum_level.value,
                    "is_mandatory": prerequisite.is_mandatory,
                },
            )
            self._prereqs_added += 1
            return model

    def remove_prerequisite(
        self,
        model_id: str,
        prerequisite_skill_id: str,
    ) -> SkillProgressionModel:
        """
        Remove a prerequisite from a progression model by the prerequisite
        skill ID.

        No-op if no matching prerequisite is found.

        Args:
            model_id:               ID of the progression model.
            prerequisite_skill_id:  The skill ID to remove as a prerequisite.

        Returns:
            The updated SkillProgressionModel.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("remove_prerequisite")

        with self._lock:
            model = self._resolve_model(model_id, "remove_prerequisite")

            removed_id: Optional[str] = None
            for pid in list(self._prereqs_by_model.get(model_id, set())):
                pr = self._prerequisite_store.get(pid)
                if pr is not None and pr.prerequisite_skill_id == prerequisite_skill_id:
                    removed_id = pid
                    break

            if removed_id is not None:
                del self._prerequisite_store[removed_id]
                self._prereqs_by_model[model_id].discard(removed_id)
                self._prereqs_removed += 1

            self._touch_model(model)
            self._record_history(
                model_id=model_id,
                op="remove_prerequisite",
                payload={
                    "prerequisite_skill_id": prerequisite_skill_id,
                    "removed": removed_id is not None,
                },
            )
            return model

    def get_prerequisites(self, model_id: str) -> list[SkillPrerequisite]:
        """
        Return all prerequisites attached to a progression model.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("get_prerequisites")

        with self._lock:
            self._resolve_model(model_id, "get_prerequisites")
            return [
                self._prerequisite_store[pid]
                for pid in self._prereqs_by_model.get(model_id, set())
                if pid in self._prerequisite_store
            ]

    # ─────────────────────────────────────────────────────────────────────────
    # TRACK SKILL PROGRESSION
    # ─────────────────────────────────────────────────────────────────────────

    def track_skill_progression(
        self,
        skill_id: str,
        stages: list[SkillStage],
        *,
        notes: str = "",
    ) -> SkillProgressionModel:
        """
        Idempotent entry-point: create a new progression model for a skill, or
        update the stages on the existing model if one already exists.

        This method is the recommended way to register or refresh a skill's
        progression structure without having to check for prior existence.

        Args:
            skill_id: The LUNA Skill to track.
            stages:   The full stage list (Novice → Master, or any subset).
            notes:    Descriptive notes written to the history entry.

        Returns:
            The created or updated SkillProgressionModel.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Stage list is structurally invalid.
        """
        self._require_initialized("track_skill_progression")

        with self._lock:
            existing_model_id = self._skill_to_model.get(skill_id)
            if existing_model_id is not None:
                return self.update_progression_model(
                    existing_model_id,
                    stages=stages,
                    notes=notes,
                )
            return self.create_progression_model(skill_id, stages, notes=notes)

    # ─────────────────────────────────────────────────────────────────────────
    # ADVANCE SKILL STAGE
    # ─────────────────────────────────────────────────────────────────────────

    def advance_skill_stage(
        self,
        model_id: str,
        current_level: SkillLevel,
        *,
        covered_concept_ids: Optional[list[str]] = None,
        covered_fact_ids: Optional[list[str]] = None,
        covered_procedure_ids: Optional[list[str]] = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        """
        Evaluate whether the next stage is reachable from current_level and
        record the advance in the model's history if mastery is confirmed.

        The advance is a structural assertion: it records that the transition
        criteria for the current level are satisfied according to the provided
        coverage sets.  It does not mutate user proficiency (ASTRA's domain).

        Args:
            model_id:              The progression model to evaluate.
            current_level:         The SkillLevel the learner is currently at.
            covered_concept_ids:   Concept IDs the learner has demonstrated.
            covered_fact_ids:      Fact IDs the learner has demonstrated.
            covered_procedure_ids: Procedure IDs the learner has demonstrated.
            actor:                 Identifier of the agent requesting the advance.

        Returns:
            A dict with keys:
                advanced        (bool)    — whether the advance was confirmed
                from_level      (str)     — current_level.value
                to_level        (str|None)— next level value, or None at master
                mastery_score   (float)   — 0.0–1.0 score for current stage
                transition_criteria (str|None) — model-defined criteria string
                notes           (str)     — human-readable outcome description

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("advance_skill_stage")

        with self._lock:
            model = self._resolve_model(model_id, "advance_skill_stage")

            current_stage = model.get_stage(current_level)
            if current_stage is None:
                raise SkillProgressionError(
                    skill_id=model.skill_id,
                    stage=current_level.value,
                    message=(
                        f"Level '{current_level.value}' not defined in model '{model_id}'."
                    ),
                )

            next_level = current_level.next_level()
            mastery_result = self._compute_mastery(
                stage=current_stage,
                covered_concept_ids=covered_concept_ids or [],
                covered_fact_ids=covered_fact_ids or [],
                covered_procedure_ids=covered_procedure_ids or [],
            )
            mastery_score = mastery_result["score"]
            advanced = mastery_score >= _MASTERY_THRESHOLD

            transition_criteria: Optional[str] = None
            if next_level is not None:
                transition_criteria = model.get_transition_criteria(current_level, next_level)

            if advanced:
                notes_text = (
                    f"Stage '{current_level.value}' mastered "
                    f"(score={mastery_score:.2f}); "
                    + (
                        f"advancing to '{next_level.value}'."
                        if next_level else "already at terminal level."
                    )
                )
                self._record_history(
                    model_id=model_id,
                    op="advance_stage",
                    actor=actor,
                    notes=notes_text,
                    payload={
                        "from_level": current_level.value,
                        "to_level": next_level.value if next_level else None,
                        "mastery_score": mastery_score,
                    },
                )
            else:
                notes_text = (
                    f"Stage '{current_level.value}' not yet mastered "
                    f"(score={mastery_score:.2f} < threshold={_MASTERY_THRESHOLD})."
                )

            return {
                "advanced": advanced,
                "from_level": current_level.value,
                "to_level": next_level.value if next_level else None,
                "mastery_score": round(mastery_score, 4),
                "transition_criteria": transition_criteria,
                "notes": notes_text,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATE MASTERY
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_mastery(
        self,
        model_id: str,
        level: SkillLevel,
        *,
        covered_concept_ids: Optional[list[str]] = None,
        covered_fact_ids: Optional[list[str]] = None,
        covered_procedure_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Determine mastery status for a specific stage level given a coverage set.

        Mastery is achieved when ≥ 80% of the stage's knowledge requirements
        (concepts + facts + procedures) are present in the coverage sets.  Stages
        with zero requirements are considered fully mastered by definition.

        Args:
            model_id:              The progression model to evaluate.
            level:                 The SkillLevel of the stage to assess.
            covered_concept_ids:   Concept IDs the learner has demonstrated.
            covered_fact_ids:      Fact IDs the learner has demonstrated.
            covered_procedure_ids: Procedure IDs the learner has demonstrated.

        Returns:
            A dict with keys:
                level          (str)   — the level evaluated
                label          (str)   — the stage label
                mastered       (bool)  — True if score ≥ threshold
                score          (float) — normalized coverage score [0.0, 1.0]
                threshold      (float) — the mastery threshold used
                covered_count  (int)   — total requirements covered
                total_required (int)   — total requirements in stage
                gaps           (list)  — uncovered requirement IDs

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found, or level not in model.
        """
        self._require_initialized("evaluate_mastery")

        with self._lock:
            model = self._resolve_model(model_id, "evaluate_mastery")

            stage = model.get_stage(level)
            if stage is None:
                raise SkillProgressionError(
                    skill_id=model.skill_id,
                    stage=level.value,
                    message=(
                        f"Level '{level.value}' not defined in model '{model_id}'."
                    ),
                )

            result = self._compute_mastery(
                stage=stage,
                covered_concept_ids=covered_concept_ids or [],
                covered_fact_ids=covered_fact_ids or [],
                covered_procedure_ids=covered_procedure_ids or [],
            )
            return result

    def _compute_mastery(
        self,
        stage: SkillStage,
        covered_concept_ids: list[str],
        covered_fact_ids: list[str],
        covered_procedure_ids: list[str],
    ) -> dict[str, Any]:
        """
        Internal: compute a mastery assessment dict for a single stage.
        """
        covered_concepts = set(covered_concept_ids)
        covered_facts = set(covered_fact_ids)
        covered_procs = set(covered_procedure_ids)

        all_required_concepts = set(stage.required_concept_ids)
        all_required_facts = set(stage.required_fact_ids)
        all_required_procs = set(stage.required_procedure_ids)

        missing_concepts = sorted(all_required_concepts - covered_concepts)
        missing_facts = sorted(all_required_facts - covered_facts)
        missing_procs = sorted(all_required_procs - covered_procs)

        total_required = len(all_required_concepts) + len(all_required_facts) + len(all_required_procs)
        covered_count = (
            len(all_required_concepts & covered_concepts)
            + len(all_required_facts & covered_facts)
            + len(all_required_procs & covered_procs)
        )

        if total_required == 0:
            score = 1.0
        else:
            score = covered_count / total_required

        gaps = missing_concepts + missing_facts + missing_procs

        return {
            "level": stage.level.value,
            "label": stage.label,
            "mastered": score >= _MASTERY_THRESHOLD,
            "score": round(score, 4),
            "threshold": _MASTERY_THRESHOLD,
            "covered_count": covered_count,
            "total_required": total_required,
            "gaps": gaps,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CALCULATE PROGRESS
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_progress(
        self,
        model_id: str,
        *,
        covered_concept_ids: Optional[list[str]] = None,
        covered_fact_ids: Optional[list[str]] = None,
        covered_procedure_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Compute a normalized progress score across all stages in a model.

        The overall score is the average mastery score across all defined
        stages.  Per-stage breakdown is included for granular reporting.

        Args:
            model_id:              The progression model to score.
            covered_concept_ids:   Concept IDs the learner has demonstrated.
            covered_fact_ids:      Fact IDs the learner has demonstrated.
            covered_procedure_ids: Procedure IDs the learner has demonstrated.

        Returns:
            A dict with keys:
                model_id        (str)   — the model evaluated
                skill_id        (str)   — the associated skill
                skill_name      (str)   — the associated skill name
                overall_score   (float) — normalized score [0.0, 1.0]
                stages_mastered (int)   — count of fully mastered stages
                total_stages    (int)   — total stage count
                stage_scores    (list)  — per-stage mastery dicts

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("calculate_progress")

        with self._lock:
            model = self._resolve_model(model_id, "calculate_progress")

            cc = covered_concept_ids or []
            cf = covered_fact_ids or []
            cp = covered_procedure_ids or []

            stage_scores: list[dict[str, Any]] = []
            for stage in model.stages:
                result = self._compute_mastery(
                    stage=stage,
                    covered_concept_ids=cc,
                    covered_fact_ids=cf,
                    covered_procedure_ids=cp,
                )
                stage_scores.append(result)

            total = len(stage_scores)
            overall_score = (
                sum(s["score"] for s in stage_scores) / total
                if total > 0 else 0.0
            )
            stages_mastered = sum(1 for s in stage_scores if s["mastered"])

            return {
                "model_id": model_id,
                "skill_id": model.skill_id,
                "skill_name": model.skill_name,
                "overall_score": round(overall_score, 4),
                "stages_mastered": stages_mastered,
                "total_stages": total,
                "stage_scores": stage_scores,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # IDENTIFY SKILL GAPS
    # ─────────────────────────────────────────────────────────────────────────

    def identify_skill_gaps(
        self,
        model_id: str,
        target_level: SkillLevel,
        *,
        covered_concept_ids: Optional[list[str]] = None,
        covered_fact_ids: Optional[list[str]] = None,
        covered_procedure_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Find all unmet requirements across every stage up to and including
        target_level.

        Gaps are grouped by stage and categorized by requirement type
        (concept, fact, procedure).

        Args:
            model_id:              The progression model to analyze.
            target_level:          The highest SkillLevel to include in gap analysis.
            covered_concept_ids:   Concept IDs the learner has demonstrated.
            covered_fact_ids:      Fact IDs the learner has demonstrated.
            covered_procedure_ids: Procedure IDs the learner has demonstrated.

        Returns:
            A dict with keys:
                model_id        (str)  — the model analyzed
                skill_id        (str)  — the associated skill
                target_level    (str)  — the target level value
                stages_analyzed (int)  — stages within scope
                total_gaps      (int)  — aggregate gap count
                stage_gaps      (list) — per-stage gap dicts

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("identify_skill_gaps")

        with self._lock:
            model = self._resolve_model(model_id, "identify_skill_gaps")

            cc = set(covered_concept_ids or [])
            cf = set(covered_fact_ids or [])
            cp = set(covered_procedure_ids or [])

            stage_gaps: list[dict[str, Any]] = []
            total_gaps: int = 0

            for stage in model.stages:
                if stage.level.rank > target_level.rank:
                    break

                missing_concepts = sorted(set(stage.required_concept_ids) - cc)
                missing_facts = sorted(set(stage.required_fact_ids) - cf)
                missing_procs = sorted(set(stage.required_procedure_ids) - cp)
                # Competencies are described as plain strings; we report them
                # all as "not yet demonstrated" since we have no boolean coverage
                # for them — the caller provides concept/fact/procedure coverage.
                missing_competencies: list[str] = list(stage.competencies)

                gap_entry = _make_gap_entry(
                    stage_level=stage.level,
                    stage_label=stage.label,
                    missing_concept_ids=missing_concepts,
                    missing_fact_ids=missing_facts,
                    missing_procedure_ids=missing_procs,
                    missing_competencies=missing_competencies,
                )
                stage_gaps.append(gap_entry)
                total_gaps += gap_entry["total_gaps"]

            return {
                "model_id": model_id,
                "skill_id": model.skill_id,
                "target_level": target_level.value,
                "stages_analyzed": len(stage_gaps),
                "total_gaps": total_gaps,
                "stage_gaps": stage_gaps,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE LEARNING RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def generate_learning_recommendations(
        self,
        model_id: str,
        current_level: SkillLevel,
        *,
        covered_concept_ids: Optional[list[str]] = None,
        covered_fact_ids: Optional[list[str]] = None,
        covered_procedure_ids: Optional[list[str]] = None,
        max_recommendations: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Produce ordered learning recommendations for progressing beyond
        current_level.

        Recommendations are derived directly from stage requirements and
        demonstration tasks for stages the learner has not yet mastered,
        beginning from the stage at or just above current_level.

        Args:
            model_id:              The progression model to use.
            current_level:         The learner's current SkillLevel.
            covered_concept_ids:   Concepts already demonstrated.
            covered_fact_ids:      Facts already demonstrated.
            covered_procedure_ids: Procedures already demonstrated.
            max_recommendations:   Maximum number of recommendations to return.

        Returns:
            Ordered list of recommendation dicts, each with keys:
                rank              (int)        — 1-based priority rank
                stage_level       (str)        — the stage this addresses
                stage_label       (str)        — human-readable label
                concept_ids       (list[str])  — concepts to cover
                fact_ids          (list[str])  — facts to cover
                procedure_ids     (list[str])  — procedures to cover
                demonstration_tasks (list[str])— tasks to demonstrate mastery
                rationale         (str)        — why this is recommended
                estimated_hours   (float|None) — expected study time

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("generate_learning_recommendations")

        with self._lock:
            model = self._resolve_model(model_id, "generate_learning_recommendations")

            cc = set(covered_concept_ids or [])
            cf = set(covered_fact_ids or [])
            cp = set(covered_procedure_ids or [])

            recommendations: list[dict[str, Any]] = []
            rank: int = 1

            for stage in model.stages:
                if rank > max_recommendations:
                    break
                # Only recommend stages at or above the next level after current
                if stage.level.rank <= current_level.rank:
                    continue

                missing_concepts = sorted(set(stage.required_concept_ids) - cc)
                missing_facts = sorted(set(stage.required_fact_ids) - cf)
                missing_procs = sorted(set(stage.required_procedure_ids) - cp)

                # Skip fully satisfied stages
                if not missing_concepts and not missing_facts and not missing_procs:
                    continue

                prev_level = stage.level.previous_level()
                rationale = (
                    f"To reach '{stage.label}', cover the listed knowledge and "
                    "complete the demonstration tasks. "
                    + (
                        f"Transition criteria: "
                        + model.get_transition_criteria(prev_level, stage.level)
                        if prev_level
                        and model.get_transition_criteria(prev_level, stage.level)
                        else ""
                    )
                ).strip()

                rec = _make_recommendation(
                    rank=rank,
                    stage_level=stage.level,
                    stage_label=stage.label,
                    concept_ids=missing_concepts,
                    fact_ids=missing_facts,
                    procedure_ids=missing_procs,
                    demonstration_tasks=list(stage.demonstration_tasks),
                    rationale=rationale,
                    estimated_hours=stage.typical_duration_hours,
                )
                recommendations.append(rec)
                rank += 1

            return recommendations

    # ─────────────────────────────────────────────────────────────────────────
    # DETERMINE NEXT SKILLS
    # ─────────────────────────────────────────────────────────────────────────

    def determine_next_skills(
        self,
        skill_id: str,
        current_level: SkillLevel,
    ) -> list[dict[str, Any]]:
        """
        Suggest skills to pursue after reaching current_level in the given
        skill.

        Next-skill candidates are collected from two sources:
            1. SkillPrerequisite records in OTHER models that list skill_id as
               a prerequisite at or below current_level — meaning skill_id
               is now satisfied for those skills.
            2. Sub-skills of skill_id as declared in the Skill record (when
               a SkillEngine is wired).

        Args:
            skill_id:      The skill the learner has progressed in.
            current_level: The level the learner has reached.

        Returns:
            Ordered list of dicts, each with keys:
                skill_id        (str)  — the suggested next skill
                model_id        (str|None) — its progression model, if any
                reason          (str)  — why this is suggested
                minimum_level_met (bool) — True if current_level satisfies
                                           the prerequisite

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("determine_next_skills")

        with self._lock:
            candidates: list[dict[str, Any]] = []
            seen: set[str] = {skill_id}

            # Source 1: models that have a prerequisite on skill_id
            for model in self._model_store.values():
                if model.skill_id in seen:
                    continue
                for pid in self._prereqs_by_model.get(model.id, set()):
                    prereq = self._prerequisite_store.get(pid)
                    if prereq is None:
                        continue
                    if prereq.prerequisite_skill_id != skill_id:
                        continue
                    level_met = current_level.rank >= prereq.minimum_level.rank
                    candidates.append({
                        "skill_id": model.skill_id,
                        "model_id": model.id,
                        "reason": (
                            f"This skill requires '{skill_id}' at minimum level "
                            f"'{prereq.minimum_level.value}'"
                            + (" — requirement now satisfied." if level_met
                               else f" — current level '{current_level.value}' does not yet satisfy this.")
                        ),
                        "minimum_level_met": level_met,
                    })
                    seen.add(model.skill_id)

            # Source 2: sub-skills from the SkillEngine (if wired)
            if self._skill_engine is not None:
                try:
                    parent_skill: Optional[Skill] = self._skill_engine.retrieve_skill(skill_id)
                    if parent_skill is not None:
                        for sub_id in parent_skill.sub_skill_ids:
                            if sub_id in seen:
                                continue
                            sub_model_id = self._skill_to_model.get(sub_id)
                            candidates.append({
                                "skill_id": sub_id,
                                "model_id": sub_model_id,
                                "reason": (
                                    f"This is a sub-skill of '{skill_id}'; "
                                    f"progressing in the parent enables deeper specialization."
                                ),
                                "minimum_level_met": True,
                            })
                            seen.add(sub_id)
                except Exception:
                    pass

            return candidates

    # ─────────────────────────────────────────────────────────────────────────
    # PROGRESSION HISTORY
    # ─────────────────────────────────────────────────────────────────────────

    def progression_history(
        self,
        model_id: str,
        *,
        op_filter: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the ordered mutation history for a progression model.

        Args:
            model_id:   ID of the model whose history to retrieve.
            op_filter:  If provided, return only entries with this op value.
            limit:      Maximum number of entries to return (most recent first).

        Returns:
            List of history entry dicts in descending timestamp order.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("progression_history")

        with self._lock:
            self._resolve_model(model_id, "progression_history")

            entries = self._history.get(model_id, [])
            if op_filter is not None:
                entries = [e for e in entries if e["op"] == op_filter]

            # Return most recent first
            return list(reversed(entries))[:limit]

    # ─────────────────────────────────────────────────────────────────────────
    # PROGRESSION AUDIT
    # ─────────────────────────────────────────────────────────────────────────

    def progression_audit(self, model_id: str) -> dict[str, Any]:
        """
        Produce a structured audit of a single progression model.

        Includes model metadata, stage summary, prerequisite summary,
        history event counts, and structural health indicators.

        Args:
            model_id: ID of the model to audit.

        Returns:
            A dict with audit data keyed by section.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found.
        """
        self._require_initialized("progression_audit")

        with self._lock:
            model = self._resolve_model(model_id, "progression_audit")
            prereqs = [
                self._prerequisite_store[pid]
                for pid in self._prereqs_by_model.get(model_id, set())
                if pid in self._prerequisite_store
            ]
            history = self._history.get(model_id, [])
            op_counts: dict[str, int] = defaultdict(int)
            for entry in history:
                op_counts[entry["op"]] += 1

            stage_summary = [
                {
                    "level": s.level.value,
                    "label": s.label,
                    "knowledge_requirement_count": s.knowledge_requirement_count,
                    "competency_count": len(s.competencies),
                    "demonstration_task_count": len(s.demonstration_tasks),
                    "typical_duration_hours": s.typical_duration_hours,
                }
                for s in model.stages
            ]

            prereq_summary = [
                {
                    "prerequisite_id": pr.id,
                    "prerequisite_skill_id": pr.prerequisite_skill_id,
                    "minimum_level": pr.minimum_level.value,
                    "is_mandatory": pr.is_mandatory,
                    "rationale": pr.rationale,
                }
                for pr in prereqs
            ]

            total_knowledge_requirements = sum(
                s.knowledge_requirement_count for s in model.stages
            )
            min_level = model.min_level
            max_level = model.max_level

            return {
                "model_id": model_id,
                "skill_id": model.skill_id,
                "skill_name": model.skill_name,
                "description": model.description,
                "is_linear": model.is_linear,
                "estimated_total_hours": model.estimated_total_hours,
                "created_at": model.created_at.isoformat(),
                "updated_at": model.updated_at.isoformat(),
                "stage_count": model.stage_count,
                "min_level": min_level.value if min_level else None,
                "max_level": max_level.value if max_level else None,
                "total_knowledge_requirements": total_knowledge_requirements,
                "prerequisite_count": len(prereqs),
                "mandatory_prerequisite_count": sum(
                    1 for pr in prereqs if pr.is_mandatory
                ),
                "history_event_count": len(history),
                "history_op_counts": dict(op_counts),
                "stage_summary": stage_summary,
                "prerequisite_summary": prereq_summary,
                "health": {
                    "has_stages": model.stage_count > 0,
                    "has_novice_stage": model.get_stage(SkillLevel.NOVICE) is not None,
                    "has_master_stage": model.get_stage(SkillLevel.MASTER) is not None,
                    "all_stages_have_label": all(
                        bool(s.label) for s in model.stages
                    ),
                    "all_stages_have_description": all(
                        bool(s.description) for s in model.stages
                    ),
                },
                "generated_at": _utcnow().isoformat(),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # PROGRESSION REPORTING
    # ─────────────────────────────────────────────────────────────────────────

    def progression_reporting(self) -> dict[str, Any]:
        """
        Produce an aggregate report across all progression models in the store.

        Returns:
            A dict with keys:
                total_models            (int)
                total_prerequisites     (int)
                mandatory_prerequisites (int)
                models_with_no_stages   (int)
                models_without_novice   (int)
                models_without_master   (int)
                avg_stage_count         (float)
                avg_knowledge_requirements_per_stage (float)
                skill_coverage          (list[dict]) — per-model summary
                history_event_total     (int)
                generated_at            (str)

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("progression_reporting")

        with self._lock:
            models = list(self._model_store.values())
            total = len(models)

            total_stages: int = 0
            total_knowledge_reqs: int = 0
            models_no_stages: int = 0
            models_no_novice: int = 0
            models_no_master: int = 0
            history_event_total: int = sum(
                len(v) for v in self._history.values()
            )

            skill_coverage: list[dict[str, Any]] = []
            for model in models:
                stage_count = model.stage_count
                total_stages += stage_count
                req_count = sum(s.knowledge_requirement_count for s in model.stages)
                total_knowledge_reqs += req_count

                if stage_count == 0:
                    models_no_stages += 1
                if model.get_stage(SkillLevel.NOVICE) is None:
                    models_no_novice += 1
                if model.get_stage(SkillLevel.MASTER) is None:
                    models_no_master += 1

                skill_coverage.append({
                    "model_id": model.id,
                    "skill_id": model.skill_id,
                    "skill_name": model.skill_name,
                    "stage_count": stage_count,
                    "knowledge_requirement_count": req_count,
                    "prerequisite_count": len(
                        self._prereqs_by_model.get(model.id, set())
                    ),
                    "levels": [s.level.value for s in model.stages],
                })

            mandatory_prereqs = sum(
                1 for pr in self._prerequisite_store.values()
                if pr.is_mandatory
            )

            avg_stage_count = total_stages / total if total > 0 else 0.0
            stage_total_for_avg = total_stages or 1
            avg_req_per_stage = total_knowledge_reqs / stage_total_for_avg if total_stages > 0 else 0.0

            return {
                "total_models": total,
                "total_prerequisites": len(self._prerequisite_store),
                "mandatory_prerequisites": mandatory_prereqs,
                "models_with_no_stages": models_no_stages,
                "models_without_novice": models_no_novice,
                "models_without_master": models_no_master,
                "avg_stage_count": round(avg_stage_count, 2),
                "avg_knowledge_requirements_per_stage": round(avg_req_per_stage, 2),
                "skill_coverage": skill_coverage,
                "history_event_total": history_event_total,
                "generated_at": _utcnow().isoformat(),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT REPORT (interface contract)
    # ─────────────────────────────────────────────────────────────────────────

    def audit_report(self) -> dict[str, Any]:
        """
        Return skill progression store statistics.

        Delegates to progression_reporting() and enriches with engine-level
        counters for operational monitoring.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("audit_report")

        report = self.progression_reporting()
        report.update({
            "models_created": self._models_created,
            "models_updated": self._models_updated,
            "models_deleted": self._models_deleted,
            "prereqs_added": self._prereqs_added,
            "prereqs_removed": self._prereqs_removed,
            "last_mutation_at": (
                self._last_mutation_at.isoformat()
                if self._last_mutation_at else None
            ),
            "engine_version": _ENGINE_VERSION,
        })
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """
        Return a lightweight liveness/readiness summary.

        Required keys: engine, initialized, record_count, status.
        """
        return {
            "engine": "SkillProgressionEngine",
            "initialized": self._initialized,
            "record_count": len(self._model_store),
            "prerequisite_count": len(self._prerequisite_store),
            "status": "healthy" if self._initialized else "offline",
        }

    def diagnostics_report(self) -> dict[str, Any]:
        """
        Return a full introspection snapshot suitable for operator debugging.

        Required keys: engine, initialized, record_count, status, index_size,
        duplicate_checks, mutation_count, last_mutation_at.
        """
        report = self.health_report()
        mutation_count = (
            self._models_created + self._models_updated + self._models_deleted
            + self._prereqs_added + self._prereqs_removed
        )
        report.update({
            "index_size": len(self._skill_to_model),
            "duplicate_checks": 0,
            "mutation_count": mutation_count,
            "last_mutation_at": (
                self._last_mutation_at.isoformat()
                if self._last_mutation_at else None
            ),
            "models_created": self._models_created,
            "models_updated": self._models_updated,
            "models_deleted": self._models_deleted,
            "prereqs_added": self._prereqs_added,
            "prereqs_removed": self._prereqs_removed,
            "history_event_total": sum(
                len(v) for v in self._history.values()
            ),
            "engine_version": _ENGINE_VERSION,
            "started_at": (
                self._started_at.isoformat() if self._started_at else None
            ),
            "skill_engine_wired": self._skill_engine is not None,
            "education_engine_wired": self._education_engine is not None,
            "evolution_engine_wired": self._evolution_engine is not None,
        })
        return report


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "SkillProgressionEngine",
]


