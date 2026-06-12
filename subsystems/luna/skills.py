"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/skills.py

Concrete implementation of the LUNA Skill Engine.

The Skill Engine manages capability knowledge owned exclusively by LUNA:
    - Python Programming
    - PCB Design
    - CAD Modeling
    - ROS2 Development
    - Embedded Systems

Ownership law (LUNA Law 3):
    LUNA owns the skill definition — what the skill IS, how it progresses,
    what it requires, and how competency develops through stages.
    ASTRA owns the user's proficiency profile — how well the user performs it.

This engine never stores individual user progress.  It stores only the
canonical, reusable skill knowledge structure that all of POLARIS can consult.

Responsibilities:
    - Full CRUD lifecycle for Skill records
    - Skill prerequisite management (SkillPrerequisite graph)
    - Skill hierarchy management (sub-skills, composite skills)
    - Skill progression model management (SkillProgressionModel / SkillStage)
    - Structural validation per LUNA quality standards
    - SHA-256 fingerprint-based duplicate detection
    - Iterated-DFS cycle detection on the prerequisite graph
    - Domain indexing and free-text search
    - Immutable audit log and engine diagnostics

Implementation notes:
    - In-memory v1 store backed by dict[str, Skill]
    - Thread-safe via threading.RLock
    - Lifecycle-gated: every public method raises LunaNotInitializedError
      before initialize() is called or after shutdown()
    - Soft-delete only: deleted skills transition to RETRACTED
    - SkillProgressionModel and SkillPrerequisite are stored in co-located
      secondary stores, linked by IDs on the Skill record

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
from datetime import datetime, timezone
from typing import Any, Optional

from subsystems.luna.exceptions import (
    DuplicateSkillError,
    LunaLifecycleError,
    LunaNotInitializedError,
    SkillNotFoundError,
    SkillProgressionError,
    SkillValidationError,
)
from subsystems.luna.interfaces import AbstractSkillEngine
from subsystems.luna.models import (
    ConfidenceLevel,
    KnowledgeDifficulty,
    KnowledgeMetadata,
    KnowledgeSourceType,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeValidationResult,
    Skill,
    SkillLevel,
    SkillPrerequisite,
    SkillProgressionModel,
    SkillStage,
    SkillType,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
    ValidationStatus,
    _new_id,
    _stable_hash,
    _utcnow,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE_VERSION: str = "5.0.0"
_MAX_NAME_LENGTH: int = 256
_MAX_DESCRIPTION_LENGTH: int = 8_192
_MAX_STAGES: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION LOG ENTRY
# ─────────────────────────────────────────────────────────────────────────────

class _SkillOpEntry:
    """Immutable record of a single mutation applied to the skill store."""

    __slots__ = ("op", "skill_id", "timestamp", "notes")

    def __init__(self, op: str, skill_id: str, notes: str = "") -> None:
        self.op: str = op          # "create" | "update" | "delete" | "add_prereq" |
                                   # "remove_prereq" | "add_sub" | "remove_sub" |
                                   # "set_progression" | "remove_progression"
        self.skill_id: str = skill_id
        self.timestamp: datetime = _utcnow()
        self.notes: str = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "skill_id": self.skill_id,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SKILL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SkillEngine(AbstractSkillEngine):
    """
    In-memory, thread-safe implementation of the LUNA Skill Engine (v1).

    Data stores:
        _store              — dict[skill_id, Skill]          primary record store
        _progression_store  — dict[model_id, SkillProgressionModel]
        _prerequisite_store — dict[prereq_id, SkillPrerequisite]

    Secondary indexes:
        _fingerprint_index  — fingerprint → skill_id         deduplication
        _domain_index       — domain_id → set[skill_id]      domain filter
        _type_index         — SkillType.value → set[skill_id]
        _prereq_by_skill    — skill_id → set[prereq_record_id]  prerequisites FOR skill
        _prereq_graph       — skill_id → set[prerequisite_skill_id]  adjacency for cycle detection

    Cycle detection:
        Iterative DFS on _prereq_graph.  Adding edge A → B creates a cycle if
        B can already reach A through existing edges.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()

        # Primary stores
        self._store: dict[str, Skill] = {}
        self._progression_store: dict[str, SkillProgressionModel] = {}
        self._prerequisite_store: dict[str, SkillPrerequisite] = {}

        # Secondary indexes
        self._fingerprint_index: dict[str, str] = {}
        self._domain_index: dict[str, set[str]] = defaultdict(set)
        self._type_index: dict[str, set[str]] = defaultdict(set)

        # Prerequisite graph: skill_id → set[prerequisite_skill_id]
        # _prereq_graph[A] = {B, C} means A requires B and C
        self._prereq_graph: dict[str, set[str]] = defaultdict(set)

        # Reverse index: skill_id → set[SkillPrerequisite.id]
        self._prereq_by_skill: dict[str, set[str]] = defaultdict(set)

        # Sub-skill adjacency: parent_id → set[sub_skill_id]
        self._sub_skill_graph: dict[str, set[str]] = defaultdict(set)

        # Validation result cache: skill_id → KnowledgeValidationResult
        self._validation_cache: dict[str, KnowledgeValidationResult] = {}

        # Operation audit log
        self._op_log: list[_SkillOpEntry] = []

        # Counters
        self._mutation_count: int = 0
        self._duplicate_checks: int = 0
        self._last_mutation_at: Optional[datetime] = None

        # Lifecycle
        self._initialized: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Prepare the Skill Engine for operation.

        Idempotent: repeated calls after first initialization are no-ops.

        Raises:
            LunaLifecycleError: If internal setup fails.
        """
        with self._lock:
            if self._initialized:
                return
            try:
                self._store.clear()
                self._progression_store.clear()
                self._prerequisite_store.clear()
                self._fingerprint_index.clear()
                self._domain_index.clear()
                self._type_index.clear()
                self._prereq_graph.clear()
                self._prereq_by_skill.clear()
                self._sub_skill_graph.clear()
                self._validation_cache.clear()
                self._op_log.clear()
                self._mutation_count = 0
                self._duplicate_checks = 0
                self._last_mutation_at = None
                self._initialized = True
                logger.info("SkillEngine initialized (version=%s)", _ENGINE_VERSION)
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="SkillEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        """
        Release resources and mark the engine offline.

        Idempotent.

        Raises:
            LunaLifecycleError: If teardown fails.
        """
        with self._lock:
            if not self._initialized:
                return
            try:
                self._initialized = False
                logger.info(
                    "SkillEngine shutdown (skills=%d, progressions=%d, "
                    "prerequisites=%d, mutations=%d)",
                    len(self._store),
                    len(self._progression_store),
                    len(self._prerequisite_store),
                    self._mutation_count,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="SkillEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def is_initialized(self) -> bool:
        return self._initialized

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE GUARD
    # ─────────────────────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation=operation)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — INDEXING
    # ─────────────────────────────────────────────────────────────────────────

    def _index_skill(self, skill: Skill) -> None:
        """Register a skill in all secondary indexes."""
        self._fingerprint_index[skill.fingerprint] = skill.id
        for domain_id in skill.domain_ids:
            self._domain_index[domain_id].add(skill.id)
        self._type_index[skill.skill_type.value].add(skill.id)
        # Rebuild adjacency from stored data
        self._prereq_graph[skill.id] = set(skill.prerequisite_skill_ids)
        self._sub_skill_graph[skill.id] = set(skill.sub_skill_ids)

    def _deindex_skill(self, skill: Skill) -> None:
        """Remove a skill from all secondary indexes."""
        self._fingerprint_index.pop(skill.fingerprint, None)
        for domain_id in skill.domain_ids:
            self._domain_index[domain_id].discard(skill.id)
        self._type_index[skill.skill_type.value].discard(skill.id)
        self._prereq_graph.pop(skill.id, None)
        self._sub_skill_graph.pop(skill.id, None)

    def _reindex_skill(self, old: Skill, new: Skill) -> None:
        self._deindex_skill(old)
        self._index_skill(new)

    def _record_op(self, op: str, skill_id: str, notes: str = "") -> None:
        """Append an operation log entry and update mutation counters."""
        self._op_log.append(_SkillOpEntry(op=op, skill_id=skill_id, notes=notes))
        self._mutation_count += 1
        self._last_mutation_at = _utcnow()
        self._validation_cache.pop(skill_id, None)

    def _resolve_skill(self, skill_id: str, operation: str) -> Skill:
        """Return the skill or raise SkillNotFoundError."""
        skill = self._store.get(skill_id)
        if skill is None:
            raise SkillNotFoundError(
                skill_id=skill_id,
                context={"operation": operation},
            )
        return skill

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — DUPLICATE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _check_duplicate(self, skill: Skill) -> None:
        """Raise DuplicateSkillError if a live record with the same fingerprint exists."""
        self._duplicate_checks += 1
        existing_id = self._fingerprint_index.get(skill.fingerprint)
        if existing_id is None:
            return
        existing = self._store.get(existing_id)
        if existing is None:
            return
        if not existing.status.is_terminal:
            raise DuplicateSkillError(
                skill_id=skill.id,
                existing_id=existing_id,
                context={"fingerprint": skill.fingerprint},
            )

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — CYCLE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _would_create_prereq_cycle(self, from_id: str, to_id: str) -> bool:
        """
        Return True if adding the directed edge from_id → to_id in the
        prerequisite graph (meaning "from_id now requires to_id") would create
        a cycle.

        Uses iterative DFS from to_id.  If we can reach from_id via the
        existing prerequisites of to_id, a cycle would result.
        """
        visited: set[str] = set()
        stack: list[str] = [to_id]
        while stack:
            node = stack.pop()
            if node == from_id:
                return True
            if node in visited:
                continue
            visited.add(node)
            for prereq in self._prereq_graph.get(node, set()):
                if prereq not in visited:
                    stack.append(prereq)
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    def _run_validation(self, skill: Skill) -> KnowledgeValidationResult:
        """Execute all validation rules against a Skill instance."""
        issues: list[ValidationIssue] = []

        # Rule: name must be non-empty
        if not skill.name or not skill.name.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.CRITICAL,
                message="Skill name must not be empty.",
                field="name",
            ))

        # Rule: name length
        if len(skill.name) > _MAX_NAME_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"Skill name exceeds {_MAX_NAME_LENGTH} characters.",
                field="name",
            ))

        # Rule: description must be non-empty
        if not skill.description or not skill.description.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.CRITICAL,
                message="Skill description must not be empty.",
                field="description",
            ))

        # Rule: description length
        if len(skill.description) > _MAX_DESCRIPTION_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.WARNING,
                message=f"Skill description exceeds {_MAX_DESCRIPTION_LENGTH} characters.",
                field="description",
            ))

        # Rule: capability_description must be non-empty
        if not skill.capability_description or not skill.capability_description.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.ERROR,
                message="Skill capability_description must not be empty.",
                field="capability_description",
            ))

        # Rule: confidence_score must be in [0.0, 1.0]
        score = skill.metadata.confidence_score
        if not (0.0 <= score <= 1.0):
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"confidence_score {score!r} is outside [0.0, 1.0].",
                field="metadata.confidence_score",
            ))

        # Rule: domain_ids must not be empty
        if not skill.domain_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.MISSING_REFERENCES,
                severity=ValidationSeverity.ERROR,
                message="Skill must belong to at least one knowledge domain.",
                field="domain_ids",
            ))

        # Rule: knowledge_type must be SKILL
        if skill.knowledge_type != KnowledgeType.SKILL:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    f"knowledge_type must be KnowledgeType.SKILL, "
                    f"got {skill.knowledge_type!r}."
                ),
                field="knowledge_type",
            ))

        # Rule: no self-referential prerequisites
        if skill.id in skill.prerequisite_skill_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                severity=ValidationSeverity.ERROR,
                message="Skill references itself in prerequisite_skill_ids.",
                field="prerequisite_skill_ids",
            ))

        # Rule: no self-referential sub-skills
        if skill.id in skill.sub_skill_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                severity=ValidationSeverity.ERROR,
                message="Skill references itself in sub_skill_ids.",
                field="sub_skill_ids",
            ))

        # Rule: all prerequisite_skill_ids must exist
        for prereq_id in skill.prerequisite_skill_ids:
            if prereq_id not in self._store:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"prerequisite_skill_id '{prereq_id}' does not "
                        f"exist in the skill store."
                    ),
                    field="prerequisite_skill_ids",
                ))

        # Rule: all sub_skill_ids must exist
        for sub_id in skill.sub_skill_ids:
            if sub_id not in self._store:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"sub_skill_id '{sub_id}' does not "
                        f"exist in the skill store."
                    ),
                    field="sub_skill_ids",
                ))

        # Rule: low-confidence warning
        if skill.metadata.confidence_score < 0.40:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.LOW_CONFIDENCE,
                severity=ValidationSeverity.WARNING,
                message=(
                    f"Confidence score {skill.metadata.confidence_score:.2f} is low "
                    f"({ConfidenceLevel.from_score(skill.metadata.confidence_score).value})."
                ),
                field="metadata.confidence_score",
            ))

        # Rule: unknown source type warning
        if skill.metadata.source_type == KnowledgeSourceType.UNKNOWN:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.UNVERIFIED_CLAIM,
                severity=ValidationSeverity.WARNING,
                message="Skill source type is UNKNOWN; consider providing a reliable source.",
                field="metadata.source_type",
            ))

        return KnowledgeValidationResult.create(
            knowledge_id=skill.id,
            knowledge_name=skill.name,
            issues=issues,
            validator_version=_ENGINE_VERSION,
        )

    def _run_progression_validation(
        self,
        model: SkillProgressionModel,
    ) -> list[str]:
        """
        Validate a SkillProgressionModel structure.

        Returns a list of violation strings (empty = valid).
        """
        violations: list[str] = []

        if not model.skill_id:
            violations.append("SkillProgressionModel.skill_id must not be empty.")

        if not model.skill_name or not model.skill_name.strip():
            violations.append("SkillProgressionModel.skill_name must not be empty.")

        if not model.stages:
            violations.append("SkillProgressionModel must have at least one SkillStage.")

        if len(model.stages) > _MAX_STAGES:
            violations.append(
                f"SkillProgressionModel has {len(model.stages)} stages; "
                f"maximum is {_MAX_STAGES}."
            )

        # Levels must be distinct
        seen_levels: set[str] = set()
        for stage in model.stages:
            val = stage.level.value
            if val in seen_levels:
                violations.append(
                    f"SkillProgressionModel has duplicate stage level '{val}'."
                )
            seen_levels.add(val)

        # Levels must be in ascending rank order
        ranks = [s.level.rank for s in model.stages]
        for i in range(len(ranks) - 1):
            if ranks[i] >= ranks[i + 1]:
                violations.append(
                    f"Stage levels are not in strictly ascending order at index {i}."
                )

        # Each stage label must be non-empty
        for stage in model.stages:
            if not stage.label or not stage.label.strip():
                violations.append(
                    f"SkillStage at level '{stage.level.value}' has an empty label."
                )

        return violations

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD — SKILL
    # ─────────────────────────────────────────────────────────────────────────

    def create_skill(
        self,
        name: str,
        description: str,
        skill_type: SkillType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        applications: Optional[list[str]] = None,
        prerequisite_skill_ids: Optional[list[str]] = None,
        related_concept_ids: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> Skill:
        """
        Create and register a new skill definition owned by LUNA.

        Performs structural validation and duplicate detection before storing.

        Args:
            name:                   Short canonical name (e.g. "Python Programming").
            description:            Human-readable description of the skill.
            skill_type:             SkillType classification enum.
            difficulty:             KnowledgeDifficulty tier.
            domain_ids:             Owning knowledge domain IDs.
            metadata:               Provenance, confidence, and versioning metadata.
            applications:           Practical applications of this skill (stored as
                                    capability_description on the Skill record).
            prerequisite_skill_ids: IDs of skills that must be acquired first.
            related_concept_ids:    IDs of concepts underpinning this skill.
            aliases:                Alternate names or abbreviations.
            notes:                  Free-text notes.

        Returns:
            The newly created and stored Skill record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DuplicateSkillError:     An identical skill already exists.
            SkillValidationError:    Provided data fails validation.
            SkillNotFoundError:      A prerequisite skill ID does not exist.
        """
        self._require_initialized("create_skill")

        with self._lock:
            capability_description = (
                "; ".join(applications) if applications else description
            )

            candidate = Skill.create(
                name=name,
                description=description,
                skill_type=skill_type,
                capability_description=capability_description,
                difficulty=difficulty,
                domain_ids=list(domain_ids),
                metadata=metadata,
                required_concept_ids=related_concept_ids,
                prerequisite_skill_ids=prerequisite_skill_ids,
                aliases=aliases,
                notes=notes,
            )

            # Validate prerequisite IDs exist before full validation
            for prereq_id in (prerequisite_skill_ids or []):
                if prereq_id not in self._store:
                    raise SkillNotFoundError(
                        skill_id=prereq_id,
                        context={
                            "operation": "create_skill",
                            "reason": "prerequisite skill not found",
                        },
                    )

            # Structural validation
            result = self._run_validation(candidate)
            if result.has_blocking_issues:
                violations = [
                    i.message for i in result.issues if i.severity.is_blocking()
                ]
                raise SkillValidationError(
                    skill_id=candidate.id,
                    violations=violations,
                )

            # Duplicate detection
            self._check_duplicate(candidate)

            # Cycle detection for prerequisite graph
            for prereq_id in (prerequisite_skill_ids or []):
                if self._would_create_prereq_cycle(candidate.id, prereq_id):
                    raise SkillValidationError(
                        skill_id=candidate.id,
                        violations=[
                            f"Adding prerequisite '{prereq_id}' to skill "
                            f"'{candidate.id}' would create a circular dependency."
                        ],
                    )

            # Persist
            self._store[candidate.id] = candidate
            self._index_skill(candidate)

            self._record_op("create", candidate.id, notes=f"created: {name!r}")
            logger.debug(
                "SkillEngine.create_skill id=%s name=%r type=%s",
                candidate.short_id,
                name,
                skill_type.value,
            )
            return candidate

    def update_skill(
        self,
        skill_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        skill_type: Optional[SkillType] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        domain_ids: Optional[list[str]] = None,
        applications: Optional[list[str]] = None,
        prerequisite_skill_ids: Optional[list[str]] = None,
        related_concept_ids: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
    ) -> Skill:
        """
        Apply a partial update to an existing skill record.

        Only keyword-supplied fields are modified; omitted fields retain their
        current values.  Version is incremented automatically.

        Args:
            skill_id: ID of the skill to update.
            **fields: Any subset of Skill fields to overwrite.

        Returns:
            The updated Skill record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
            SkillValidationError:    Updated state fails validation or attempts to
                                     modify a terminal record.
            DuplicateSkillError:     Name/description change produces a fingerprint
                                     collision with another active skill.
        """
        self._require_initialized("update_skill")

        with self._lock:
            existing = self._resolve_skill(skill_id, "update_skill")

            if existing.status.is_terminal:
                raise SkillValidationError(
                    skill_id=skill_id,
                    violations=[
                        f"Cannot update a skill with terminal status "
                        f"'{existing.status.value}'."
                    ],
                )

            new_prereqs = (
                list(prerequisite_skill_ids)
                if prerequisite_skill_ids is not None
                else list(existing.prerequisite_skill_ids)
            )

            # Validate all new prerequisite IDs exist and avoid self-reference
            if prerequisite_skill_ids is not None:
                for prereq_id in new_prereqs:
                    if prereq_id == skill_id:
                        raise SkillValidationError(
                            skill_id=skill_id,
                            violations=["Skill cannot be its own prerequisite."],
                        )
                    if prereq_id not in self._store:
                        raise SkillNotFoundError(
                            skill_id=prereq_id,
                            context={
                                "operation": "update_skill",
                                "reason": "prerequisite not found",
                            },
                        )

            # Cycle detection for newly added prerequisites
            if prerequisite_skill_ids is not None:
                old_set = set(existing.prerequisite_skill_ids)
                for prereq_id in new_prereqs:
                    if prereq_id not in old_set:
                        if self._would_create_prereq_cycle(skill_id, prereq_id):
                            raise SkillValidationError(
                                skill_id=skill_id,
                                violations=[
                                    f"Adding prerequisite '{prereq_id}' would create "
                                    f"a circular dependency."
                                ],
                            )

            new_capability = (
                "; ".join(applications)
                if applications is not None
                else existing.capability_description
            )

            updated = Skill(
                id=existing.id,
                knowledge_type=existing.knowledge_type,
                name=name if name is not None else existing.name,
                description=description if description is not None else existing.description,
                status=status if status is not None else existing.status,
                difficulty=difficulty if difficulty is not None else existing.difficulty,
                domain_ids=list(domain_ids) if domain_ids is not None else list(existing.domain_ids),
                metadata=(
                    metadata.bump_version()
                    if metadata is not None
                    else existing.metadata.bump_version()
                ),
                aliases=list(aliases) if aliases is not None else list(existing.aliases),
                related_ids=list(existing.related_ids),
                notes=notes if notes is not None else existing.notes,
                skill_type=skill_type if skill_type is not None else existing.skill_type,
                capability_description=new_capability,
                required_tools=list(existing.required_tools),
                required_concept_ids=(
                    list(related_concept_ids)
                    if related_concept_ids is not None
                    else list(existing.required_concept_ids)
                ),
                required_fact_ids=list(existing.required_fact_ids),
                prerequisite_skill_ids=new_prereqs,
                sub_skill_ids=list(existing.sub_skill_ids),
                progression_model_id=existing.progression_model_id,
                practical_exercises=list(existing.practical_exercises),
                assessment_criteria=list(existing.assessment_criteria),
            )

            # Structural validation
            result = self._run_validation(updated)
            if result.has_blocking_issues:
                violations = [
                    i.message for i in result.issues if i.severity.is_blocking()
                ]
                raise SkillValidationError(skill_id=skill_id, violations=violations)

            # Duplicate fingerprint check (only if identity-affecting fields changed)
            if name is not None or description is not None:
                fp = updated.fingerprint
                existing_fp_id = self._fingerprint_index.get(fp)
                if existing_fp_id is not None and existing_fp_id != skill_id:
                    other = self._store.get(existing_fp_id)
                    if other and not other.status.is_terminal:
                        raise DuplicateSkillError(
                            skill_id=skill_id,
                            existing_id=existing_fp_id,
                            context={"fingerprint": fp},
                        )

            self._reindex_skill(existing, updated)
            self._store[skill_id] = updated
            self._record_op("update", skill_id, notes=f"updated: {updated.name!r}")

            logger.debug("SkillEngine.update_skill id=%s", skill_id[:8])
            return updated

    def delete_skill(self, skill_id: str, *, reason: str = "") -> Skill:
        """
        Soft-delete a skill by transitioning its status to RETRACTED.

        Skills are never physically removed.  Audit history and downstream
        references remain coherent through the status transition.

        Args:
            skill_id: ID of the skill to retract.
            reason:   Human-readable reason stored in notes.

        Returns:
            The retracted Skill record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
        """
        self._require_initialized("delete_skill")

        with self._lock:
            existing = self._resolve_skill(skill_id, "delete_skill")

            if existing.status == KnowledgeStatus.RETRACTED:
                return existing  # Idempotent

            deletion_note = (
                f"[RETRACTED {_utcnow().isoformat()}]"
                + (f" Reason: {reason}" if reason else "")
            )

            retracted = Skill(
                id=existing.id,
                knowledge_type=existing.knowledge_type,
                name=existing.name,
                description=existing.description,
                status=KnowledgeStatus.RETRACTED,
                difficulty=existing.difficulty,
                domain_ids=list(existing.domain_ids),
                metadata=existing.metadata.bump_version(),
                aliases=list(existing.aliases),
                related_ids=list(existing.related_ids),
                notes=f"{existing.notes}\n{deletion_note}".strip(),
                skill_type=existing.skill_type,
                capability_description=existing.capability_description,
                required_tools=list(existing.required_tools),
                required_concept_ids=list(existing.required_concept_ids),
                required_fact_ids=list(existing.required_fact_ids),
                prerequisite_skill_ids=list(existing.prerequisite_skill_ids),
                sub_skill_ids=list(existing.sub_skill_ids),
                progression_model_id=existing.progression_model_id,
                practical_exercises=list(existing.practical_exercises),
                assessment_criteria=list(existing.assessment_criteria),
            )

            self._deindex_skill(existing)
            self._store[skill_id] = retracted
            self._record_op("delete", skill_id, notes=reason)

            logger.debug(
                "SkillEngine.delete_skill id=%s reason=%r",
                skill_id[:8],
                reason,
            )
            return retracted

    def retrieve_skill(self, skill_id: str) -> Skill:
        """
        Fetch a single skill by its unique ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
        """
        self._require_initialized("retrieve_skill")
        with self._lock:
            return self._resolve_skill(skill_id, "retrieve_skill")

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    def search_skills(
        self,
        query: str,
        *,
        domain_ids: Optional[list[str]] = None,
        skill_types: Optional[list[SkillType]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Skill]:
        """
        Free-text search across skill name, description, capability_description,
        and aliases.

        Args:
            query:         Case-insensitive substring search string.
            domain_ids:    Restrict to skills belonging to these domains.
            skill_types:   Restrict to these SkillType values.
            difficulty:    Exact difficulty tier filter.
            status_filter: Restrict to these KnowledgeStatus values.
            limit:         Maximum number of results.
            offset:        Pagination offset.

        Returns:
            Ordered list of matching Skill records (most recently updated first).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("search_skills")

        with self._lock:
            q = query.lower().strip()
            results: list[Skill] = []
            domain_set = set(domain_ids) if domain_ids else None
            type_values = (
                {t.value for t in skill_types} if skill_types else None
            )

            for skill in self._store.values():
                # Status filter (default: draft, active, and validated)
                if status_filter:
                    if skill.status not in status_filter:
                        continue
                else:
                    if skill.status.is_terminal:
                        continue

                # Domain filter
                if domain_set:
                    if not domain_set.intersection(skill.domain_ids):
                        continue

                # SkillType filter
                if type_values:
                    if skill.skill_type.value not in type_values:
                        continue

                # Difficulty filter
                if difficulty is not None:
                    if skill.difficulty != difficulty:
                        continue

                # Text search
                if q:
                    searchable = " ".join([
                        skill.name,
                        skill.description,
                        skill.capability_description,
                        " ".join(skill.aliases),
                        " ".join(skill.required_tools),
                    ]).lower()
                    if q not in searchable:
                        continue

                results.append(skill)

            # Sort by updated_at descending (most recently modified first)
            results.sort(
                key=lambda s: s.metadata.updated_at,
                reverse=True,
            )
            return results[offset : offset + limit]

    def search_skills_by_domain(
        self,
        domain_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Skill]:
        """
        Return all active skills belonging to a given domain.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("search_skills_by_domain")

        with self._lock:
            skill_ids = self._domain_index.get(domain_id, set())
            results = [
                self._store[sid]
                for sid in skill_ids
                if sid in self._store and self._store[sid].status.is_usable
            ]
            results.sort(key=lambda s: s.name)
            return results[offset : offset + limit]

    def get_all_skills(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[Skill]:
        """
        Return a paginated slice of the skill store.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_all_skills")

        with self._lock:
            if status_filter:
                skills = [
                    s for s in self._store.values() if s.status in status_filter
                ]
            else:
                skills = list(self._store.values())
            skills.sort(key=lambda s: s.name)
            return skills[offset : offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # PREREQUISITE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def add_skill_prerequisite(
        self,
        skill_id: str,
        prerequisite_skill_id: str,
        minimum_level: SkillLevel,
        *,
        is_mandatory: bool = True,
        rationale: str = "",
    ) -> SkillPrerequisite:
        """
        Declare that skill_id requires prerequisite_skill_id at minimum_level.

        Creates a SkillPrerequisite record, updates the Skill record's
        prerequisite_skill_ids list, and updates the prerequisite adjacency graph.

        Args:
            skill_id:               The skill that has the prerequisite.
            prerequisite_skill_id:  The skill that must be acquired first.
            minimum_level:          Minimum SkillLevel required.
            is_mandatory:           Hard (True) vs. recommended (False) prerequisite.
            rationale:              Why this prerequisite is required.

        Returns:
            The newly created SkillPrerequisite record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Either skill ID does not exist.
            SkillValidationError:    Self-reference or duplicate prerequisite.
            SkillProgressionError:   Adding the edge would create a cycle.
        """
        self._require_initialized("add_skill_prerequisite")

        with self._lock:
            skill = self._resolve_skill(skill_id, "add_skill_prerequisite")
            self._resolve_skill(prerequisite_skill_id, "add_skill_prerequisite")

            if skill_id == prerequisite_skill_id:
                raise SkillValidationError(
                    skill_id=skill_id,
                    violations=["A skill cannot be its own prerequisite."],
                )

            if prerequisite_skill_id in skill.prerequisite_skill_ids:
                raise SkillValidationError(
                    skill_id=skill_id,
                    violations=[
                        f"Prerequisite '{prerequisite_skill_id}' already registered "
                        f"on skill '{skill_id}'."
                    ],
                )

            if self._would_create_prereq_cycle(skill_id, prerequisite_skill_id):
                raise SkillProgressionError(
                    skill_id=skill_id,
                    message=(
                        f"Adding prerequisite '{prerequisite_skill_id}' to skill "
                        f"'{skill_id}' would create a circular dependency."
                    ),
                )

            prereq_record = SkillPrerequisite.create(
                skill_id=skill_id,
                prerequisite_skill_id=prerequisite_skill_id,
                minimum_level=minimum_level,
                is_mandatory=is_mandatory,
                rationale=rationale,
            )
            self._prerequisite_store[prereq_record.id] = prereq_record
            self._prereq_by_skill[skill_id].add(prereq_record.id)

            # Update the Skill record
            updated_prereqs = list(skill.prerequisite_skill_ids) + [prerequisite_skill_id]
            updated_skill = Skill(
                id=skill.id,
                knowledge_type=skill.knowledge_type,
                name=skill.name,
                description=skill.description,
                status=skill.status,
                difficulty=skill.difficulty,
                domain_ids=list(skill.domain_ids),
                metadata=skill.metadata.bump_version(),
                aliases=list(skill.aliases),
                related_ids=list(skill.related_ids),
                notes=skill.notes,
                skill_type=skill.skill_type,
                capability_description=skill.capability_description,
                required_tools=list(skill.required_tools),
                required_concept_ids=list(skill.required_concept_ids),
                required_fact_ids=list(skill.required_fact_ids),
                prerequisite_skill_ids=updated_prereqs,
                sub_skill_ids=list(skill.sub_skill_ids),
                progression_model_id=skill.progression_model_id,
                practical_exercises=list(skill.practical_exercises),
                assessment_criteria=list(skill.assessment_criteria),
            )
            self._reindex_skill(skill, updated_skill)
            self._store[skill_id] = updated_skill

            self._record_op(
                "add_prereq",
                skill_id,
                notes=f"prereq={prerequisite_skill_id!r} level={minimum_level.value}",
            )
            logger.debug(
                "SkillEngine.add_skill_prerequisite skill=%s prereq=%s level=%s",
                skill_id[:8],
                prerequisite_skill_id[:8],
                minimum_level.value,
            )
            return prereq_record

    def remove_skill_prerequisite(
        self,
        skill_id: str,
        prerequisite_skill_id: str,
    ) -> Skill:
        """
        Remove a prerequisite relationship from a skill.

        Removes the SkillPrerequisite record and updates the Skill record's
        prerequisite_skill_ids list.

        Args:
            skill_id:               The skill to modify.
            prerequisite_skill_id:  The prerequisite to remove.

        Returns:
            The updated Skill record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
            SkillValidationError:    The prerequisite is not registered on this skill.
        """
        self._require_initialized("remove_skill_prerequisite")

        with self._lock:
            skill = self._resolve_skill(skill_id, "remove_skill_prerequisite")

            if prerequisite_skill_id not in skill.prerequisite_skill_ids:
                raise SkillValidationError(
                    skill_id=skill_id,
                    violations=[
                        f"Prerequisite '{prerequisite_skill_id}' is not registered "
                        f"on skill '{skill_id}'."
                    ],
                )

            # Remove matching SkillPrerequisite records
            to_remove = [
                pid
                for pid in self._prereq_by_skill.get(skill_id, set())
                if self._prerequisite_store.get(pid) is not None
                and self._prerequisite_store[pid].prerequisite_skill_id
                == prerequisite_skill_id
            ]
            for pid in to_remove:
                del self._prerequisite_store[pid]
                self._prereq_by_skill[skill_id].discard(pid)

            updated_prereqs = [
                p for p in skill.prerequisite_skill_ids if p != prerequisite_skill_id
            ]
            updated_skill = Skill(
                id=skill.id,
                knowledge_type=skill.knowledge_type,
                name=skill.name,
                description=skill.description,
                status=skill.status,
                difficulty=skill.difficulty,
                domain_ids=list(skill.domain_ids),
                metadata=skill.metadata.bump_version(),
                aliases=list(skill.aliases),
                related_ids=list(skill.related_ids),
                notes=skill.notes,
                skill_type=skill.skill_type,
                capability_description=skill.capability_description,
                required_tools=list(skill.required_tools),
                required_concept_ids=list(skill.required_concept_ids),
                required_fact_ids=list(skill.required_fact_ids),
                prerequisite_skill_ids=updated_prereqs,
                sub_skill_ids=list(skill.sub_skill_ids),
                progression_model_id=skill.progression_model_id,
                practical_exercises=list(skill.practical_exercises),
                assessment_criteria=list(skill.assessment_criteria),
            )
            self._reindex_skill(skill, updated_skill)
            self._store[skill_id] = updated_skill

            self._record_op(
                "remove_prereq",
                skill_id,
                notes=f"removed prereq={prerequisite_skill_id!r}",
            )
            logger.debug(
                "SkillEngine.remove_skill_prerequisite skill=%s prereq=%s",
                skill_id[:8],
                prerequisite_skill_id[:8],
            )
            return updated_skill

    def get_prerequisites(self, skill_id: str) -> list[SkillPrerequisite]:
        """
        Return all SkillPrerequisite records for the given skill.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
        """
        self._require_initialized("get_prerequisites")

        with self._lock:
            self._resolve_skill(skill_id, "get_prerequisites")
            prereq_ids = self._prereq_by_skill.get(skill_id, set())
            return [
                self._prerequisite_store[pid]
                for pid in prereq_ids
                if pid in self._prerequisite_store
            ]

    def get_transitive_prerequisites(self, skill_id: str) -> list[Skill]:
        """
        Return all skills that must be acquired before the given skill
        (recursive prerequisite closure), sorted from most foundational first.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
            SkillProgressionError:   A cycle is detected in the prerequisite graph.
        """
        self._require_initialized("get_transitive_prerequisites")

        with self._lock:
            self._resolve_skill(skill_id, "get_transitive_prerequisites")

            reachable: set[str] = set()
            frontier: deque[str] = deque([skill_id])
            while frontier:
                sid = frontier.popleft()
                for prereq in self._prereq_graph.get(sid, set()):
                    if prereq not in reachable:
                        reachable.add(prereq)
                        frontier.append(prereq)

            if skill_id in reachable:
                raise SkillProgressionError(
                    skill_id=skill_id,
                    message=(
                        f"Cycle detected in prerequisite graph for skill '{skill_id}'."
                    ),
                )

            # Topological sort (Kahn's algorithm on subgraph of reachable IDs)
            in_degree: dict[str, int] = {sid: 0 for sid in reachable}
            adj: dict[str, list[str]] = {sid: [] for sid in reachable}
            for sid in reachable:
                for prereq in self._prereq_graph.get(sid, set()):
                    if prereq in reachable:
                        adj[prereq].append(sid)
                        in_degree[sid] += 1

            queue: deque[str] = deque(
                sid for sid, deg in in_degree.items() if deg == 0
            )
            ordered: list[str] = []
            while queue:
                sid = queue.popleft()
                ordered.append(sid)
                for neighbor in adj.get(sid, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if len(ordered) != len(reachable):
                raise SkillProgressionError(
                    skill_id=skill_id,
                    message=(
                        f"Cycle detected in transitive prerequisites of '{skill_id}'."
                    ),
                )

            return [
                self._store[sid]
                for sid in ordered
                if sid in self._store
            ]

    # ─────────────────────────────────────────────────────────────────────────
    # SKILL HIERARCHY MANAGEMENT (SUB-SKILLS)
    # ─────────────────────────────────────────────────────────────────────────

    def add_sub_skill(self, parent_skill_id: str, sub_skill_id: str) -> Skill:
        """
        Register sub_skill_id as a component skill of parent_skill_id.

        Updates sub_skill_ids on the parent Skill record.

        Args:
            parent_skill_id: The composite skill.
            sub_skill_id:    The component skill to add.

        Returns:
            The updated parent Skill record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Either skill ID does not exist.
            SkillValidationError:    Self-reference, duplicate, or cycle.
        """
        self._require_initialized("add_sub_skill")

        with self._lock:
            parent = self._resolve_skill(parent_skill_id, "add_sub_skill")
            self._resolve_skill(sub_skill_id, "add_sub_skill")

            if parent_skill_id == sub_skill_id:
                raise SkillValidationError(
                    skill_id=parent_skill_id,
                    violations=["A skill cannot be its own sub-skill."],
                )

            if sub_skill_id in parent.sub_skill_ids:
                raise SkillValidationError(
                    skill_id=parent_skill_id,
                    violations=[
                        f"Sub-skill '{sub_skill_id}' is already registered "
                        f"on skill '{parent_skill_id}'."
                    ],
                )

            updated_skill = Skill(
                id=parent.id,
                knowledge_type=parent.knowledge_type,
                name=parent.name,
                description=parent.description,
                status=parent.status,
                difficulty=parent.difficulty,
                domain_ids=list(parent.domain_ids),
                metadata=parent.metadata.bump_version(),
                aliases=list(parent.aliases),
                related_ids=list(parent.related_ids),
                notes=parent.notes,
                skill_type=parent.skill_type,
                capability_description=parent.capability_description,
                required_tools=list(parent.required_tools),
                required_concept_ids=list(parent.required_concept_ids),
                required_fact_ids=list(parent.required_fact_ids),
                prerequisite_skill_ids=list(parent.prerequisite_skill_ids),
                sub_skill_ids=list(parent.sub_skill_ids) + [sub_skill_id],
                progression_model_id=parent.progression_model_id,
                practical_exercises=list(parent.practical_exercises),
                assessment_criteria=list(parent.assessment_criteria),
            )
            self._reindex_skill(parent, updated_skill)
            self._store[parent_skill_id] = updated_skill

            self._record_op(
                "add_sub",
                parent_skill_id,
                notes=f"sub_skill={sub_skill_id!r}",
            )
            logger.debug(
                "SkillEngine.add_sub_skill parent=%s sub=%s",
                parent_skill_id[:8],
                sub_skill_id[:8],
            )
            return updated_skill

    def remove_sub_skill(self, parent_skill_id: str, sub_skill_id: str) -> Skill:
        """
        Remove a sub-skill from a parent skill's sub_skill_ids list.

        Args:
            parent_skill_id: The composite skill.
            sub_skill_id:    The component skill to remove.

        Returns:
            The updated parent Skill record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Parent skill ID does not exist.
            SkillValidationError:    Sub-skill not registered on this skill.
        """
        self._require_initialized("remove_sub_skill")

        with self._lock:
            parent = self._resolve_skill(parent_skill_id, "remove_sub_skill")

            if sub_skill_id not in parent.sub_skill_ids:
                raise SkillValidationError(
                    skill_id=parent_skill_id,
                    violations=[
                        f"Sub-skill '{sub_skill_id}' is not registered on "
                        f"skill '{parent_skill_id}'."
                    ],
                )

            updated_skill = Skill(
                id=parent.id,
                knowledge_type=parent.knowledge_type,
                name=parent.name,
                description=parent.description,
                status=parent.status,
                difficulty=parent.difficulty,
                domain_ids=list(parent.domain_ids),
                metadata=parent.metadata.bump_version(),
                aliases=list(parent.aliases),
                related_ids=list(parent.related_ids),
                notes=parent.notes,
                skill_type=parent.skill_type,
                capability_description=parent.capability_description,
                required_tools=list(parent.required_tools),
                required_concept_ids=list(parent.required_concept_ids),
                required_fact_ids=list(parent.required_fact_ids),
                prerequisite_skill_ids=list(parent.prerequisite_skill_ids),
                sub_skill_ids=[
                    s for s in parent.sub_skill_ids if s != sub_skill_id
                ],
                progression_model_id=parent.progression_model_id,
                practical_exercises=list(parent.practical_exercises),
                assessment_criteria=list(parent.assessment_criteria),
            )
            self._reindex_skill(parent, updated_skill)
            self._store[parent_skill_id] = updated_skill

            self._record_op(
                "remove_sub",
                parent_skill_id,
                notes=f"removed sub_skill={sub_skill_id!r}",
            )
            return updated_skill

    def get_sub_skills(self, skill_id: str) -> list[Skill]:
        """
        Return direct sub-skills of the given skill.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
        """
        self._require_initialized("get_sub_skills")

        with self._lock:
            skill = self._resolve_skill(skill_id, "get_sub_skills")
            return [
                self._store[sid]
                for sid in skill.sub_skill_ids
                if sid in self._store
            ]

    # ─────────────────────────────────────────────────────────────────────────
    # PROGRESSION MODEL MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def create_progression_model(
        self,
        skill_id: str,
        description: str,
        stages: list[SkillStage],
        *,
        skill_name: Optional[str] = None,
        transition_criteria: Optional[dict[str, str]] = None,
        estimated_total_hours: Optional[float] = None,
        is_linear: bool = True,
    ) -> SkillProgressionModel:
        """
        Create and attach a SkillProgressionModel to an existing skill.

        LUNA owns this model (the *structure* of skill progression).
        ASTRA owns the user's position within this model.

        Args:
            skill_id:               The skill this model belongs to.
            description:            Purpose and scope of this progression path.
            stages:                 Ordered list of SkillStage definitions.
            transition_criteria:    Dict mapping "from_level→to_level" to criteria.
            estimated_total_hours:  Estimated hours to reach the top stage.
            is_linear:              True = no branching paths.

        Returns:
            The newly created SkillProgressionModel.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
            SkillProgressionError:   Stages fail structural validation.
        """
        self._require_initialized("create_progression_model")

        with self._lock:
            skill = self._resolve_skill(skill_id, "create_progression_model")

            model = SkillProgressionModel.create(
                skill_id=skill_id,
                skill_name=skill.name,
                description=description,
                stages=list(stages),
            )
            if transition_criteria:
                model.transition_criteria.update(transition_criteria)
            if estimated_total_hours is not None:
                model.estimated_total_hours = estimated_total_hours
            model.is_linear = is_linear

            violations = self._run_progression_validation(model)
            if violations:
                raise SkillProgressionError(
                    skill_id=skill_id,
                    message="; ".join(violations),
                )

            self._progression_store[model.id] = model

            # Attach to skill record
            updated_skill = Skill(
                id=skill.id,
                knowledge_type=skill.knowledge_type,
                name=skill.name,
                description=skill.description,
                status=skill.status,
                difficulty=skill.difficulty,
                domain_ids=list(skill.domain_ids),
                metadata=skill.metadata.bump_version(),
                aliases=list(skill.aliases),
                related_ids=list(skill.related_ids),
                notes=skill.notes,
                skill_type=skill.skill_type,
                capability_description=skill.capability_description,
                required_tools=list(skill.required_tools),
                required_concept_ids=list(skill.required_concept_ids),
                required_fact_ids=list(skill.required_fact_ids),
                prerequisite_skill_ids=list(skill.prerequisite_skill_ids),
                sub_skill_ids=list(skill.sub_skill_ids),
                progression_model_id=model.id,
                practical_exercises=list(skill.practical_exercises),
                assessment_criteria=list(skill.assessment_criteria),
            )
            self._reindex_skill(skill, updated_skill)
            self._store[skill_id] = updated_skill

            self._record_op(
                "set_progression",
                skill_id,
                notes=f"model_id={model.id!r} stages={model.stage_count}",
            )
            logger.debug(
                "SkillEngine.create_progression_model skill=%s model=%s stages=%d",
                skill_id[:8],
                model.id[:8],
                model.stage_count,
            )
            return model

    def update_progression_model(
        self,
        model_id: str,
        *,
        description: Optional[str] = None,
        stages: Optional[list[SkillStage]] = None,
        transition_criteria: Optional[dict[str, str]] = None,
        estimated_total_hours: Optional[float] = None,
        is_linear: Optional[bool] = None,
    ) -> SkillProgressionModel:
        """
        Apply a partial update to an existing SkillProgressionModel.

        Args:
            model_id: ID of the model to update.
            **fields: Any subset of model fields to overwrite.

        Returns:
            The updated SkillProgressionModel.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Model not found or updated stages fail validation.
        """
        self._require_initialized("update_progression_model")

        with self._lock:
            model = self._progression_store.get(model_id)
            if model is None:
                raise SkillProgressionError(
                    skill_id="",
                    message=f"SkillProgressionModel '{model_id}' not found.",
                )

            new_stages = stages if stages is not None else list(model.stages)
            new_description = description if description is not None else model.description
            new_criteria = (
                dict(transition_criteria)
                if transition_criteria is not None
                else dict(model.transition_criteria)
            )
            new_hours = (
                estimated_total_hours
                if estimated_total_hours is not None
                else model.estimated_total_hours
            )
            new_linear = is_linear if is_linear is not None else model.is_linear

            updated = SkillProgressionModel(
                id=model.id,
                skill_id=model.skill_id,
                skill_name=model.skill_name,
                description=new_description,
                stages=new_stages,
                transition_criteria=new_criteria,
                estimated_total_hours=new_hours,
                is_linear=new_linear,
                created_at=model.created_at,
                updated_at=_utcnow(),
            )

            violations = self._run_progression_validation(updated)
            if violations:
                raise SkillProgressionError(
                    skill_id=model.skill_id,
                    message="; ".join(violations),
                )

            self._progression_store[model_id] = updated
            self._record_op(
                "set_progression",
                model.skill_id,
                notes=f"updated model_id={model_id!r}",
            )
            return updated

    def retrieve_progression_model(self, model_id: str) -> Optional[SkillProgressionModel]:
        """
        Return a SkillProgressionModel by its ID, or None if not found.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("retrieve_progression_model")
        with self._lock:
            return self._progression_store.get(model_id)

    def get_progression_model_for_skill(
        self,
        skill_id: str,
    ) -> Optional[SkillProgressionModel]:
        """
        Return the SkillProgressionModel associated with a given skill,
        or None if no model has been assigned.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
        """
        self._require_initialized("get_progression_model_for_skill")

        with self._lock:
            skill = self._resolve_skill(skill_id, "get_progression_model_for_skill")
            if skill.progression_model_id is None:
                return None
            return self._progression_store.get(skill.progression_model_id)

    def remove_progression_model(self, skill_id: str) -> Skill:
        """
        Detach and delete the SkillProgressionModel from a skill.

        Args:
            skill_id: The skill whose model to remove.

        Returns:
            The updated Skill record (progression_model_id=None).

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
            SkillValidationError:    Skill has no progression model attached.
        """
        self._require_initialized("remove_progression_model")

        with self._lock:
            skill = self._resolve_skill(skill_id, "remove_progression_model")

            if skill.progression_model_id is None:
                raise SkillValidationError(
                    skill_id=skill_id,
                    violations=["Skill has no progression model attached."],
                )

            self._progression_store.pop(skill.progression_model_id, None)

            updated_skill = Skill(
                id=skill.id,
                knowledge_type=skill.knowledge_type,
                name=skill.name,
                description=skill.description,
                status=skill.status,
                difficulty=skill.difficulty,
                domain_ids=list(skill.domain_ids),
                metadata=skill.metadata.bump_version(),
                aliases=list(skill.aliases),
                related_ids=list(skill.related_ids),
                notes=skill.notes,
                skill_type=skill.skill_type,
                capability_description=skill.capability_description,
                required_tools=list(skill.required_tools),
                required_concept_ids=list(skill.required_concept_ids),
                required_fact_ids=list(skill.required_fact_ids),
                prerequisite_skill_ids=list(skill.prerequisite_skill_ids),
                sub_skill_ids=list(skill.sub_skill_ids),
                progression_model_id=None,
                practical_exercises=list(skill.practical_exercises),
                assessment_criteria=list(skill.assessment_criteria),
            )
            self._reindex_skill(skill, updated_skill)
            self._store[skill_id] = updated_skill

            self._record_op(
                "remove_progression",
                skill_id,
                notes="progression model removed",
            )
            return updated_skill

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    def validate_skill(self, skill_id: str) -> KnowledgeValidationResult:
        """
        Run structural and semantic validation on a skill.

        Results are cached until the next mutation on the same skill.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillNotFoundError:      Skill ID does not exist.
        """
        self._require_initialized("validate_skill")

        with self._lock:
            skill = self._resolve_skill(skill_id, "validate_skill")
            cached = self._validation_cache.get(skill_id)
            if cached is not None:
                return cached
            result = self._run_validation(skill)
            self._validation_cache[skill_id] = result
            return result

    # ─────────────────────────────────────────────────────────────────────────
    # INTEGRITY
    # ─────────────────────────────────────────────────────────────────────────

    def find_duplicate_skills(self) -> list[list[Skill]]:
        """
        Detect groups of active skills that share the same content fingerprint.

        Returns:
            A list of groups, each containing two or more skills that are
            semantically identical by SHA-256 fingerprint.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("find_duplicate_skills")

        with self._lock:
            fp_map: dict[str, list[Skill]] = defaultdict(list)
            for skill in self._store.values():
                if not skill.status.is_terminal:
                    fp_map[skill.fingerprint].append(skill)
            return [group for group in fp_map.values() if len(group) >= 2]

    def skill_exists(self, skill_id: str) -> bool:
        """
        Return True if a skill with the given ID exists (any status).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("skill_exists")
        with self._lock:
            return skill_id in self._store

    def get_skill_count(self, *, active_only: bool = False) -> int:
        """
        Return the number of skills in the store.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_skill_count")
        with self._lock:
            if active_only:
                return sum(
                    1 for s in self._store.values() if s.status.is_usable
                )
            return len(self._store)

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """
        Return a lightweight liveness/readiness summary.

        Required keys: engine, initialized, record_count, status.
        """
        total = len(self._store)
        return {
            "engine": "SkillEngine",
            "initialized": self._initialized,
            "record_count": total,
            "active_count": sum(
                1 for s in self._store.values() if s.status.is_usable
            ),
            "progression_model_count": len(self._progression_store),
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
        report.update({
            "index_size": len(self._fingerprint_index),
            "duplicate_checks": self._duplicate_checks,
            "mutation_count": self._mutation_count,
            "last_mutation_at": (
                self._last_mutation_at.isoformat()
                if self._last_mutation_at else None
            ),
            "domain_index_size": sum(len(v) for v in self._domain_index.values()),
            "type_index_size": sum(len(v) for v in self._type_index.values()),
            "prereq_graph_edges": sum(
                len(v) for v in self._prereq_graph.values()
            ),
            "op_log_length": len(self._op_log),
            "engine_version": _ENGINE_VERSION,
        })
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT
    # ─────────────────────────────────────────────────────────────────────────

    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate statistics for the skill store.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("audit_report")

        with self._lock:
            all_skills = list(self._store.values())
            total = len(all_skills)
            active = sum(1 for s in all_skills if s.status.is_usable)
            retracted = sum(
                1 for s in all_skills
                if s.status == KnowledgeStatus.RETRACTED
            )
            avg_confidence = (
                sum(s.metadata.confidence_score for s in all_skills) / total
                if total else 0.0
            )

            by_type: dict[str, int] = defaultdict(int)
            by_domain: dict[str, int] = defaultdict(int)
            by_difficulty: dict[str, int] = defaultdict(int)
            with_progression: int = 0
            composite_count: int = 0

            for skill in all_skills:
                by_type[skill.skill_type.value] += 1
                for did in skill.domain_ids:
                    by_domain[did] += 1
                by_difficulty[skill.difficulty.value] += 1
                if skill.progression_model_id is not None:
                    with_progression += 1
                if skill.is_composite:
                    composite_count += 1

            duplicate_groups = self.find_duplicate_skills()

            return {
                "total_skills": total,
                "active_skills": active,
                "retracted_skills": retracted,
                "avg_confidence": round(avg_confidence, 4),
                "skills_by_type": dict(by_type),
                "skills_by_domain": dict(by_domain),
                "skills_by_difficulty": dict(by_difficulty),
                "with_progression_model": with_progression,
                "composite_skills": composite_count,
                "progression_models": len(self._progression_store),
                "prerequisite_records": len(self._prerequisite_store),
                "duplicate_groups": len(duplicate_groups),
                "mutation_count": self._mutation_count,
                "generated_at": _utcnow().isoformat(),
            }