"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/procedures.py

Concrete implementation of the LUNA Procedural Knowledge Engine.

The Procedural Knowledge Engine manages how-to knowledge owned exclusively
by LUNA — ordered, goal-directed sequences of steps that encode executable
understanding of technical and operational processes:

    - How to tune a PID controller
    - How to deploy ROS2
    - How to train a machine learning model
    - How to build a drone
    - How to calibrate a robotic arm

Ownership law: LUNA owns all procedural knowledge (Law 1).  No external
module may write, delete, or modify procedure records without routing through
this engine.

Implementation notes:
    - In-memory v1 store backed by a dict[str, Procedure]
    - Thread-safe via threading.RLock (reentrant for nested engine calls)
    - Lifecycle-gated: every public method raises LunaNotInitializedError
      before initialize() has been called or after shutdown()
    - Duplicate detection by SHA-256 content fingerprint (KnowledgeType,
      name, description)
    - Soft-delete only: deleted records transition to RETRACTED, never removed
    - Full dependency graph maintained with cycle detection
    - Step ordering and structural integrity validated on every write
    - Full audit trail via an immutable operation log

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
    ProcedureError,
    ProcedureNotFoundError,
    ProcedureValidationError,
)
from subsystems.luna.interfaces import AbstractProceduralKnowledgeEngine
from subsystems.luna.models import (
    ConfidenceLevel,
    KnowledgeDependency,
    KnowledgeDifficulty,
    KnowledgeMetadata,
    KnowledgeReference,
    KnowledgeSourceType,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeValidationResult,
    Procedure,
    ProcedureStep,
    ProcedureType,
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
_VALIDATOR_VERSION: str = "5.0.0"

# Structural limits
_MAX_NAME_LENGTH: int = 256
_MAX_DESCRIPTION_LENGTH: int = 8_192
_MAX_GOAL_LENGTH: int = 2_048
_MAX_STEPS: int = 500
_MAX_STEP_INSTRUCTION_LENGTH: int = 8_192

# Confidence threshold below which a LOW_CONFIDENCE warning is emitted
_LOW_CONFIDENCE_THRESHOLD: float = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION LOG ENTRY
# ─────────────────────────────────────────────────────────────────────────────

class _ProcedureOpEntry:
    """Immutable record of a single mutation applied to the procedure store."""

    __slots__ = ("op", "procedure_id", "timestamp", "notes")

    def __init__(self, op: str, procedure_id: str, notes: str = "") -> None:
        self.op: str = op                        # "create" | "update" | "delete"
        self.procedure_id: str = procedure_id
        self.timestamp: datetime = _utcnow()
        self.notes: str = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "procedure_id": self.procedure_id,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PROCEDURAL KNOWLEDGE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ProceduralKnowledgeEngine(AbstractProceduralKnowledgeEngine):
    """
    In-memory, thread-safe implementation of the LUNA Procedural Knowledge
    Engine (v1).

    All public operations are guarded by a reentrant lock so that callers
    using the engine from multiple threads see a consistent store.

    Responsibilities:
        - CRUD for Procedure records and their ProcedureStep sequences
        - Prerequisite and dependency graph management
        - Step validation: ordering, completeness, instruction integrity
        - Circular dependency detection across the dependency graph
        - Reference integrity checking (KnowledgeReference, KnowledgeDependency)
        - Confidence score tracking and low-confidence flagging
        - Indexes by ProcedureType, domain, difficulty, and status
        - Duplicate detection by content fingerprint
        - Immutable audit log for all mutations

    Lifecycle:
        engine = ProceduralKnowledgeEngine()
        engine.initialize()
        # ... use engine ...
        engine.shutdown()
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()

        # ── Primary store ────────────────────────────────────────────────────
        # procedure_id → Procedure
        self._store: dict[str, Procedure] = {}

        # ── Deduplication index ──────────────────────────────────────────────
        # fingerprint → procedure_id
        self._fingerprint_index: dict[str, str] = {}

        # ── Secondary indexes ────────────────────────────────────────────────
        # domain_id → set[procedure_id]
        self._domain_index: dict[str, set[str]] = defaultdict(set)

        # ProcedureType.value → set[procedure_id]
        self._type_index: dict[str, set[str]] = defaultdict(set)

        # KnowledgeDifficulty.value → set[procedure_id]
        self._difficulty_index: dict[str, set[str]] = defaultdict(set)

        # KnowledgeStatus.value → set[procedure_id]
        self._status_index: dict[str, set[str]] = defaultdict(set)

        # ── Dependency graph ─────────────────────────────────────────────────
        # dependency_id → KnowledgeDependency
        self._dependencies: dict[str, KnowledgeDependency] = {}

        # procedure_id → set[dependency_id] (outgoing: this procedure depends on)
        self._dep_outgoing: dict[str, set[str]] = defaultdict(set)

        # procedure_id → set[dependency_id] (incoming: other procedures depend on this)
        self._dep_incoming: dict[str, set[str]] = defaultdict(set)

        # ── Reference store ──────────────────────────────────────────────────
        # reference_id → KnowledgeReference
        self._references: dict[str, KnowledgeReference] = {}

        # procedure_id → set[reference_id]
        self._ref_index: dict[str, set[str]] = defaultdict(set)

        # ── Validation cache ─────────────────────────────────────────────────
        # procedure_id → KnowledgeValidationResult
        self._validation_cache: dict[str, KnowledgeValidationResult] = {}

        # ── Audit log ────────────────────────────────────────────────────────
        self._op_log: list[_ProcedureOpEntry] = []

        # ── Counters ─────────────────────────────────────────────────────────
        self._mutation_count: int = 0
        self._last_mutation_at: Optional[datetime] = None
        self._duplicate_check_count: int = 0

        # ── Lifecycle ────────────────────────────────────────────────────────
        self._initialized: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Prepare the Procedural Knowledge Engine for operation.

        Idempotent: repeated calls after first initialization are no-ops.

        Raises:
            LunaLifecycleError: If initialization encounters an internal error.
        """
        with self._lock:
            if self._initialized:
                return
            try:
                self._store.clear()
                self._fingerprint_index.clear()
                self._domain_index.clear()
                self._type_index.clear()
                self._difficulty_index.clear()
                self._status_index.clear()
                self._dependencies.clear()
                self._dep_outgoing.clear()
                self._dep_incoming.clear()
                self._references.clear()
                self._ref_index.clear()
                self._validation_cache.clear()
                self._op_log.clear()
                self._mutation_count = 0
                self._last_mutation_at = None
                self._duplicate_check_count = 0
                self._initialized = True
                logger.info(
                    "ProceduralKnowledgeEngine initialized (version=%s)",
                    _ENGINE_VERSION,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="ProceduralKnowledgeEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        """
        Release all resources and mark the engine offline.

        Idempotent: repeated calls after first shutdown are no-ops.

        Raises:
            LunaLifecycleError: If teardown encounters an internal error.
        """
        with self._lock:
            if not self._initialized:
                return
            try:
                self._initialized = False
                logger.info(
                    "ProceduralKnowledgeEngine shutdown "
                    "(procedures=%d, dependencies=%d, mutations=%d)",
                    len(self._store),
                    len(self._dependencies),
                    self._mutation_count,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="ProceduralKnowledgeEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def is_initialized(self) -> bool:
        return self._initialized

    # ─────────────────────────────────────────────────────────────────────────
    # GUARDS
    # ─────────────────────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation=operation)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _index_procedure(self, procedure: Procedure) -> None:
        """Register a procedure in all secondary indexes."""
        self._fingerprint_index[procedure.fingerprint] = procedure.id
        for domain_id in procedure.domain_ids:
            self._domain_index[domain_id].add(procedure.id)
        self._type_index[procedure.procedure_type.value].add(procedure.id)
        self._difficulty_index[procedure.difficulty.value].add(procedure.id)
        self._status_index[procedure.status.value].add(procedure.id)

    def _deindex_procedure(self, procedure: Procedure) -> None:
        """Remove a procedure from all secondary indexes."""
        self._fingerprint_index.pop(procedure.fingerprint, None)
        for domain_id in procedure.domain_ids:
            self._domain_index[domain_id].discard(procedure.id)
        self._type_index[procedure.procedure_type.value].discard(procedure.id)
        self._difficulty_index[procedure.difficulty.value].discard(procedure.id)
        self._status_index[procedure.status.value].discard(procedure.id)

    def _reindex_procedure(self, old: Procedure, new: Procedure) -> None:
        """Swap index entries when a procedure is updated."""
        self._deindex_procedure(old)
        self._index_procedure(new)

    def _record_op(self, op: str, procedure_id: str, notes: str = "") -> None:
        """Append an operation log entry and bump mutation counters."""
        self._op_log.append(
            _ProcedureOpEntry(op=op, procedure_id=procedure_id, notes=notes)
        )
        self._mutation_count += 1
        self._last_mutation_at = _utcnow()
        # Invalidate cached validation result on any mutation
        self._validation_cache.pop(procedure_id, None)

    def _check_duplicate(self, procedure: Procedure) -> None:
        """
        Raise ProcedureError if the procedure's fingerprint is already present
        in a non-terminal record.
        """
        self._duplicate_check_count += 1
        existing_id = self._fingerprint_index.get(procedure.fingerprint)
        if existing_id is None:
            return
        existing = self._store.get(existing_id)
        if existing is None:
            return
        if existing.status not in (KnowledgeStatus.RETRACTED, KnowledgeStatus.ARCHIVED):
            raise ProcedureError(
                message=(
                    f"Duplicate procedure detected: incoming procedure conflicts "
                    f"with existing record '{existing_id}' "
                    f"(fingerprint={procedure.fingerprint[:16]}…)"
                ),
                context={
                    "incoming_id": procedure.id,
                    "existing_id": existing_id,
                    "fingerprint": procedure.fingerprint,
                },
            )

    def _resolve_procedure(self, procedure_id: str, operation: str) -> Procedure:
        """Return the procedure or raise ProcedureNotFoundError."""
        procedure = self._store.get(procedure_id)
        if procedure is None:
            raise ProcedureNotFoundError(
                procedure_id=procedure_id,
                context={"operation": operation},
            )
        return procedure

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION (INTERNAL)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_validation(self, procedure: Procedure) -> KnowledgeValidationResult:
        """
        Execute all structural and semantic validation rules against a
        Procedure instance.  Returns a KnowledgeValidationResult capturing
        every issue found.
        """
        issues: list[ValidationIssue] = []

        # ── Name ──────────────────────────────────────────────────────────────

        if not procedure.name or not procedure.name.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.CRITICAL,
                message="Procedure name must not be empty.",
                field="name",
            ))
        elif len(procedure.name) > _MAX_NAME_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"Procedure name exceeds {_MAX_NAME_LENGTH} characters.",
                field="name",
            ))

        # ── Description ───────────────────────────────────────────────────────

        if not procedure.description or not procedure.description.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.ERROR,
                message="Procedure description must not be empty.",
                field="description",
            ))
        elif len(procedure.description) > _MAX_DESCRIPTION_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.WARNING,
                message=(
                    f"Procedure description exceeds {_MAX_DESCRIPTION_LENGTH} characters."
                ),
                field="description",
            ))

        # ── Goal ──────────────────────────────────────────────────────────────

        if not procedure.goal or not procedure.goal.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.ERROR,
                message="Procedure goal must not be empty.",
                field="goal",
            ))
        elif len(procedure.goal) > _MAX_GOAL_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.WARNING,
                message=f"Procedure goal exceeds {_MAX_GOAL_LENGTH} characters.",
                field="goal",
            ))

        # ── KnowledgeType consistency ─────────────────────────────────────────

        if procedure.knowledge_type != KnowledgeType.PROCEDURE:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    f"knowledge_type must be KnowledgeType.PROCEDURE, "
                    f"got {procedure.knowledge_type!r}."
                ),
                field="knowledge_type",
            ))

        # ── Domain IDs ────────────────────────────────────────────────────────

        if not procedure.domain_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.ERROR,
                message="Procedure must belong to at least one knowledge domain.",
                field="domain_ids",
            ))

        # ── Step ordering and structure ───────────────────────────────────────

        if procedure.steps:
            step_issues = self._validate_steps(procedure)
            issues.extend(step_issues)

        # ── Step count limit ──────────────────────────────────────────────────

        if len(procedure.steps) > _MAX_STEPS:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.WARNING,
                message=(
                    f"Procedure has {len(procedure.steps)} steps; "
                    f"consider decomposing procedures exceeding {_MAX_STEPS} steps."
                ),
                field="steps",
            ))

        # ── Prerequisite self-reference ───────────────────────────────────────

        if procedure.id in procedure.required_skill_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.ERROR,
                message="Procedure references itself in required_skill_ids.",
                field="required_skill_ids",
            ))

        if procedure.id in procedure.required_concept_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.ERROR,
                message="Procedure references itself in required_concept_ids.",
                field="required_concept_ids",
            ))

        # ── Confidence score range ────────────────────────────────────────────

        score = procedure.metadata.confidence_score
        if not (0.0 <= score <= 1.0):
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"confidence_score {score!r} is outside [0.0, 1.0].",
                field="metadata.confidence_score",
            ))
        elif score < _LOW_CONFIDENCE_THRESHOLD:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.LOW_CONFIDENCE,
                severity=ValidationSeverity.WARNING,
                message=(
                    f"Confidence score {score:.2f} is low "
                    f"({ConfidenceLevel.from_score(score).value}). "
                    "Consider providing additional supporting references."
                ),
                field="metadata.confidence_score",
            ))

        # ── Metadata integrity ────────────────────────────────────────────────

        if not procedure.metadata.source or not procedure.metadata.source.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.MISSING_REFERENCES,
                severity=ValidationSeverity.WARNING,
                message="Procedure metadata source is empty.",
                field="metadata.source",
            ))

        # ── Circular dependency check (within this engine's store) ────────────

        cycle = self._detect_dependency_cycle(procedure.id)
        if cycle:
            cycle_str = " → ".join(cycle)
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                severity=ValidationSeverity.CRITICAL,
                message=f"Circular dependency detected: {cycle_str}.",
                field="dependencies",
            ))

        # ── Broken dependency references ──────────────────────────────────────

        for dep_id in self._dep_outgoing.get(procedure.id, set()):
            dep = self._dependencies.get(dep_id)
            if dep is not None and dep.dependency_id not in self._store:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Dependency '{dep_id}' references procedure "
                        f"'{dep.dependency_id}' which does not exist in the store."
                    ),
                    field="dependencies",
                ))

        return KnowledgeValidationResult.create(
            knowledge_id=procedure.id,
            knowledge_name=procedure.name,
            issues=issues,
            validator_version=_VALIDATOR_VERSION,
        )

    def _validate_steps(self, procedure: Procedure) -> list[ValidationIssue]:
        """
        Validate the step sequence of a Procedure.

        Checks performed:
            - No duplicate step numbers
            - Steps are consecutively ordered starting at 1 (no gaps)
            - Each step has a non-empty title and instruction
            - Instruction length within bounds
            - Critical steps have non-empty expected_outcome
        """
        issues: list[ValidationIssue] = []
        steps = procedure.steps

        if not steps:
            return issues

        step_numbers = [s.step_number for s in steps]
        seen_numbers: set[int] = set()
        duplicates: list[int] = []

        for n in step_numbers:
            if n in seen_numbers:
                duplicates.append(n)
            seen_numbers.add(n)

        if duplicates:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Duplicate step number(s) found: "
                    f"{sorted(set(duplicates))}. Each step must have a unique number."
                ),
                field="steps",
            ))

        # Check for ordered contiguous numbering starting at 1
        sorted_numbers = sorted(seen_numbers)
        expected = list(range(1, len(steps) + 1))
        if sorted_numbers != expected:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Step numbers are not contiguous starting from 1. "
                    f"Expected {expected}, got {sorted_numbers}."
                ),
                field="steps",
            ))

        for step in steps:
            # Title
            if not step.title or not step.title.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message=f"Step {step.step_number} has an empty title.",
                    field=f"steps[{step.step_number}].title",
                ))

            # Instruction
            if not step.instruction or not step.instruction.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Step {step.step_number} has an empty instruction.",
                    field=f"steps[{step.step_number}].instruction",
                ))
            elif len(step.instruction) > _MAX_STEP_INSTRUCTION_LENGTH:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.FORMAT_ERROR,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Step {step.step_number} instruction exceeds "
                        f"{_MAX_STEP_INSTRUCTION_LENGTH} characters."
                    ),
                    field=f"steps[{step.step_number}].instruction",
                ))

            # Critical steps must have expected outcomes
            if step.is_critical and not step.expected_outcome.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Step {step.step_number} is marked critical but has no "
                        "expected_outcome defined."
                    ),
                    field=f"steps[{step.step_number}].expected_outcome",
                ))

            # Negative step number guard
            if step.step_number < 1:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.FORMAT_ERROR,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Step number {step.step_number} is invalid; "
                        "step numbers must be ≥ 1."
                    ),
                    field=f"steps[{step.step_number}].step_number",
                ))

        return issues

    # ─────────────────────────────────────────────────────────────────────────
    # DEPENDENCY GRAPH HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_dependency_cycle(
        self,
        start_id: str,
        visited: Optional[set[str]] = None,
        path: Optional[list[str]] = None,
    ) -> list[str]:
        """
        DFS-based cycle detection in the dependency graph starting from start_id.

        Returns the cycle path as a list of procedure IDs if a cycle is found,
        or an empty list if the graph is acyclic from start_id.
        """
        if visited is None:
            visited = set()
        if path is None:
            path = []

        if start_id in path:
            # Cycle detected — return the cycle portion of the path
            cycle_start = path.index(start_id)
            return path[cycle_start:] + [start_id]

        if start_id in visited:
            return []

        path = path + [start_id]
        visited.add(start_id)

        for dep_id in self._dep_outgoing.get(start_id, set()):
            dep = self._dependencies.get(dep_id)
            if dep is None:
                continue
            cycle = self._detect_dependency_cycle(dep.dependency_id, visited, path)
            if cycle:
                return cycle

        return []

    def _resolve_dependency_chain(
        self, procedure_id: str, depth: int = 0, seen: Optional[set[str]] = None
    ) -> list[str]:
        """
        Recursively resolve the transitive dependency chain for a procedure.

        Returns a topologically ordered list of dependency procedure IDs,
        from deepest prerequisite to immediate prerequisite.

        Args:
            procedure_id: Root procedure to resolve from.
            depth:        Current recursion depth (cycle guard at 256).
            seen:         Set of already-visited procedure IDs.

        Returns:
            Ordered list of transitive dependency procedure IDs (no duplicates).
        """
        if seen is None:
            seen = set()
        if depth > 256 or procedure_id in seen:
            return []

        seen.add(procedure_id)
        result: list[str] = []

        for dep_id in self._dep_outgoing.get(procedure_id, set()):
            dep = self._dependencies.get(dep_id)
            if dep is None:
                continue
            transitive = self._resolve_dependency_chain(
                dep.dependency_id, depth + 1, seen
            )
            result.extend(transitive)
            if dep.dependency_id not in result:
                result.append(dep.dependency_id)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD — CREATE
    # ─────────────────────────────────────────────────────────────────────────

    def create_procedure(
        self,
        name: str,
        description: str,
        procedure_type: ProcedureType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        steps: Optional[list[Any]] = None,
        prerequisites: Optional[list[str]] = None,
        estimated_duration_minutes: Optional[int] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> Procedure:
        """
        Create and store a new procedure.

        Performs duplicate detection by content fingerprint before inserting.
        All provided steps are validated for ordering and structural integrity.
        Any prerequisite IDs are registered as hard KnowledgeDependency entries
        in the dependency graph.  Circular dependencies introduced by the new
        record are detected and blocked.

        Args:
            name:                         Short canonical name.
            description:                  Human-readable explanation.
            procedure_type:               ProcedureType classification.
            difficulty:                   KnowledgeDifficulty tier.
            domain_ids:                   Owning KnowledgeDomain IDs.
            metadata:                     Provenance, confidence, and versioning.
            steps:                        Ordered list of ProcedureStep objects.
            prerequisites:                IDs of procedures that must be completed
                                          first (registered as hard dependencies).
            estimated_duration_minutes:   Approximate wall-clock completion time.
            aliases:                      Alternate names.
            notes:                        Free-text notes.

        Returns:
            The newly created and stored Procedure record.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureError:           Duplicate procedure detected by fingerprint.
            ProcedureValidationError: Provided data fails structural validation.
        """
        self._require_initialized("create_procedure")

        with self._lock:
            # Normalize steps
            coerced_steps: list[ProcedureStep] = []
            if steps:
                for item in steps:
                    if isinstance(item, ProcedureStep):
                        coerced_steps.append(item)
                    elif isinstance(item, dict):
                        coerced_steps.append(ProcedureStep(
                            step_number=item.get("step_number", 0),
                            title=item.get("title", ""),
                            instruction=item.get("instruction", ""),
                            sub_steps=tuple(item.get("sub_steps", [])),
                            warnings=tuple(item.get("warnings", [])),
                            expected_outcome=item.get("expected_outcome", ""),
                            is_critical=item.get("is_critical", False),
                            estimated_minutes=item.get("estimated_minutes"),
                        ))
                    else:
                        raise ProcedureValidationError(
                            procedure_id="<new>",
                            violations=[
                                f"Step item has unsupported type "
                                f"{type(item).__name__}; expected ProcedureStep or dict."
                            ],
                        )

            procedure = Procedure(
                id=_new_id(),
                knowledge_type=KnowledgeType.PROCEDURE,
                name=name,
                description=description,
                status=KnowledgeStatus.DRAFT,
                difficulty=difficulty,
                domain_ids=list(domain_ids),
                metadata=metadata,
                aliases=list(aliases or []),
                related_ids=[],
                notes=notes,
                procedure_type=procedure_type,
                goal=description,
                steps=coerced_steps,
                required_skill_ids=[],
                required_concept_ids=[],
                required_tools=[],
                required_materials=[],
                preconditions=[],
                postconditions=[],
                common_pitfalls=[],
                expected_duration_minutes=estimated_duration_minutes,
                is_reversible=True,
            )

            # Duplicate detection
            self._check_duplicate(procedure)

            # Structural validation
            result = self._run_validation(procedure)
            if result.has_blocking_issues:
                violations = [i.message for i in result.issues if i.severity.is_blocking()]
                raise ProcedureValidationError(
                    procedure_id=procedure.id, violations=violations
                )

            # Register in primary store and all indexes
            self._store[procedure.id] = procedure
            self._index_procedure(procedure)

            # Register prerequisite dependencies
            if prerequisites:
                for prereq_id in prerequisites:
                    # Validate that the prerequisite refers to a known
                    # procedure in this store before wiring the edge.
                    if prereq_id not in self._store:
                        # Remove the procedure we just registered and re-raise
                        self._deindex_procedure(procedure)
                        del self._store[procedure.id]
                        raise ProcedureNotFoundError(
                            procedure_id=prereq_id,
                            context={"operation": "create_procedure"},
                        )

                    dep = KnowledgeDependency.create(
                        dependent_id=procedure.id,
                        dependency_id=prereq_id,
                        dependency_type="requires",
                        is_hard=True,
                        description=f"Prerequisite for procedure '{name}'",
                    )
                    self._dependencies[dep.id] = dep
                    self._dep_outgoing[procedure.id].add(dep.id)
                    self._dep_incoming[prereq_id].add(dep.id)

                    # Check for cycles introduced by this prerequisite
                    cycle = self._detect_dependency_cycle(procedure.id)
                    if cycle:
                        # Roll back the dependency we just added
                        self._dep_outgoing[procedure.id].discard(dep.id)
                        self._dep_incoming[prereq_id].discard(dep.id)
                        del self._dependencies[dep.id]
                        # Remove the procedure and re-raise
                        self._deindex_procedure(procedure)
                        del self._store[procedure.id]
                        cycle_str = " → ".join(cycle)
                        raise ProcedureValidationError(
                            procedure_id=procedure.id,
                            violations=[
                                f"Prerequisite '{prereq_id}' introduces a circular "
                                f"dependency: {cycle_str}."
                            ],
                        )

            self._record_op("create", procedure.id, notes=f"created: {name!r}")
            logger.debug(
                "ProceduralKnowledgeEngine.create_procedure id=%s name=%r",
                procedure.id[:8],
                name,
            )
            return procedure

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD — UPDATE
    # ─────────────────────────────────────────────────────────────────────────

    def update_procedure(
        self,
        procedure_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        procedure_type: Optional[ProcedureType] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        domain_ids: Optional[list[str]] = None,
        steps: Optional[list[Any]] = None,
        prerequisites: Optional[list[str]] = None,
        estimated_duration_minutes: Optional[int] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
        goal: Optional[str] = None,
        required_skill_ids: Optional[list[str]] = None,
        required_concept_ids: Optional[list[str]] = None,
        required_tools: Optional[list[str]] = None,
        required_materials: Optional[list[str]] = None,
        preconditions: Optional[list[str]] = None,
        postconditions: Optional[list[str]] = None,
        common_pitfalls: Optional[list[str]] = None,
        is_reversible: Optional[bool] = None,
        aliases: Optional[list[str]] = None,
        related_ids: Optional[list[str]] = None,
    ) -> Procedure:
        """
        Apply a partial update to an existing procedure record.

        Only keyword-supplied fields are modified; omitted fields retain their
        current values.  Version is incremented automatically on the metadata.
        If steps are replaced, full step validation is re-run.  If prerequisites
        are replaced, the dependency graph is rebuilt and cycle detection is run.

        Args:
            procedure_id:  ID of the procedure to update.
            **fields:      Any subset of Procedure fields to overwrite.

        Returns:
            The updated Procedure record.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   No procedure with the given ID exists.
            ProcedureValidationError: Updated state fails structural validation.
        """
        self._require_initialized("update_procedure")

        with self._lock:
            existing = self._resolve_procedure(procedure_id, "update_procedure")

            if existing.status.is_terminal:
                raise ProcedureValidationError(
                    procedure_id=procedure_id,
                    violations=[
                        f"Cannot update a procedure with terminal status "
                        f"'{existing.status.value}'."
                    ],
                )

            # Coerce steps if provided
            coerced_steps: list[ProcedureStep]
            if steps is not None:
                coerced_steps = []
                for item in steps:
                    if isinstance(item, ProcedureStep):
                        coerced_steps.append(item)
                    elif isinstance(item, dict):
                        coerced_steps.append(ProcedureStep(
                            step_number=item.get("step_number", 0),
                            title=item.get("title", ""),
                            instruction=item.get("instruction", ""),
                            sub_steps=tuple(item.get("sub_steps", [])),
                            warnings=tuple(item.get("warnings", [])),
                            expected_outcome=item.get("expected_outcome", ""),
                            is_critical=item.get("is_critical", False),
                            estimated_minutes=item.get("estimated_minutes"),
                        ))
                    else:
                        raise ProcedureValidationError(
                            procedure_id=procedure_id,
                            violations=[
                                f"Step item has unsupported type "
                                f"{type(item).__name__}; expected ProcedureStep or dict."
                            ],
                        )
            else:
                coerced_steps = list(existing.steps)

            updated = Procedure(
                id=existing.id,
                knowledge_type=existing.knowledge_type,
                name=name if name is not None else existing.name,
                description=(
                    description if description is not None else existing.description
                ),
                status=status if status is not None else existing.status,
                difficulty=difficulty if difficulty is not None else existing.difficulty,
                domain_ids=(
                    list(domain_ids) if domain_ids is not None else list(existing.domain_ids)
                ),
                metadata=(
                    metadata.bump_version()
                    if metadata is not None
                    else existing.metadata.bump_version()
                ),
                aliases=(
                    list(aliases) if aliases is not None else list(existing.aliases)
                ),
                related_ids=(
                    list(related_ids) if related_ids is not None else list(existing.related_ids)
                ),
                notes=notes if notes is not None else existing.notes,
                procedure_type=(
                    procedure_type if procedure_type is not None else existing.procedure_type
                ),
                goal=goal if goal is not None else existing.goal,
                steps=coerced_steps,
                required_skill_ids=(
                    list(required_skill_ids)
                    if required_skill_ids is not None
                    else list(existing.required_skill_ids)
                ),
                required_concept_ids=(
                    list(required_concept_ids)
                    if required_concept_ids is not None
                    else list(existing.required_concept_ids)
                ),
                required_tools=(
                    list(required_tools)
                    if required_tools is not None
                    else list(existing.required_tools)
                ),
                required_materials=(
                    list(required_materials)
                    if required_materials is not None
                    else list(existing.required_materials)
                ),
                preconditions=(
                    list(preconditions)
                    if preconditions is not None
                    else list(existing.preconditions)
                ),
                postconditions=(
                    list(postconditions)
                    if postconditions is not None
                    else list(existing.postconditions)
                ),
                common_pitfalls=(
                    list(common_pitfalls)
                    if common_pitfalls is not None
                    else list(existing.common_pitfalls)
                ),
                expected_duration_minutes=(
                    estimated_duration_minutes
                    if estimated_duration_minutes is not None
                    else existing.expected_duration_minutes
                ),
                is_reversible=(
                    is_reversible if is_reversible is not None else existing.is_reversible
                ),
            )

            # Structural validation of updated record
            result = self._run_validation(updated)
            if result.has_blocking_issues:
                violations = [
                    i.message for i in result.issues if i.severity.is_blocking()
                ]
                raise ProcedureValidationError(
                    procedure_id=procedure_id, violations=violations
                )

            # Fingerprint duplicate check only if identity fields changed
            if name is not None or description is not None:
                fp = updated.fingerprint
                existing_fp_id = self._fingerprint_index.get(fp)
                if existing_fp_id is not None and existing_fp_id != procedure_id:
                    conflicting = self._store.get(existing_fp_id)
                    if conflicting and not conflicting.status.is_terminal:
                        raise ProcedureError(
                            message=(
                                f"Update would produce a duplicate procedure: "
                                f"fingerprint matches existing record '{existing_fp_id}'."
                            ),
                            context={
                                "procedure_id": procedure_id,
                                "existing_id": existing_fp_id,
                                "fingerprint": fp,
                            },
                        )

            # Rebuild dependency graph if prerequisites supplied
            if prerequisites is not None:
                # Remove old outgoing dependencies for this procedure
                for dep_id in list(self._dep_outgoing.get(procedure_id, set())):
                    dep = self._dependencies.get(dep_id)
                    if dep is not None:
                        self._dep_incoming[dep.dependency_id].discard(dep_id)
                    self._dependencies.pop(dep_id, None)
                self._dep_outgoing[procedure_id].clear()

                # Add new prerequisites
                new_deps: list[tuple[str, KnowledgeDependency]] = []
                for prereq_id in prerequisites:
                    dep = KnowledgeDependency.create(
                        dependent_id=procedure_id,
                        dependency_id=prereq_id,
                        dependency_type="requires",
                        is_hard=True,
                        description=f"Prerequisite for procedure '{updated.name}'",
                    )
                    self._dependencies[dep.id] = dep
                    self._dep_outgoing[procedure_id].add(dep.id)
                    self._dep_incoming[prereq_id].add(dep.id)
                    new_deps.append((prereq_id, dep))

                    cycle = self._detect_dependency_cycle(procedure_id)
                    if cycle:
                        # Roll back all new deps
                        for _, d in new_deps:
                            self._dep_outgoing[procedure_id].discard(d.id)
                            self._dep_incoming[d.dependency_id].discard(d.id)
                            self._dependencies.pop(d.id, None)
                        cycle_str = " → ".join(cycle)
                        raise ProcedureValidationError(
                            procedure_id=procedure_id,
                            violations=[
                                f"Updated prerequisites introduce a circular "
                                f"dependency: {cycle_str}."
                            ],
                        )

            self._reindex_procedure(existing, updated)
            self._store[procedure_id] = updated
            self._record_op("update", procedure_id, notes=f"updated: {updated.name!r}")

            logger.debug(
                "ProceduralKnowledgeEngine.update_procedure id=%s", procedure_id[:8]
            )
            return updated

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD — DELETE
    # ─────────────────────────────────────────────────────────────────────────

    def delete_procedure(self, procedure_id: str, *, reason: str = "") -> Procedure:
        """
        Soft-delete a procedure by transitioning its status to RETRACTED.

        Procedures are never physically removed; deletion is a status transition
        so that audit history and downstream references remain coherent.
        Outgoing dependency edges are preserved so that dependent procedures can
        detect broken references via validate_procedure().

        Args:
            procedure_id:  ID of the procedure to retract.
            reason:        Human-readable reason for deletion (stored in notes).

        Returns:
            The retracted Procedure record.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   No procedure with the given ID exists.
        """
        self._require_initialized("delete_procedure")

        with self._lock:
            existing = self._resolve_procedure(procedure_id, "delete_procedure")

            if existing.status == KnowledgeStatus.RETRACTED:
                return existing  # Idempotent

            deletion_note = (
                f"[RETRACTED {_utcnow().isoformat()}]"
                + (f" Reason: {reason}" if reason else "")
            )

            retracted = Procedure(
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
                procedure_type=existing.procedure_type,
                goal=existing.goal,
                steps=list(existing.steps),
                required_skill_ids=list(existing.required_skill_ids),
                required_concept_ids=list(existing.required_concept_ids),
                required_tools=list(existing.required_tools),
                required_materials=list(existing.required_materials),
                preconditions=list(existing.preconditions),
                postconditions=list(existing.postconditions),
                common_pitfalls=list(existing.common_pitfalls),
                expected_duration_minutes=existing.expected_duration_minutes,
                is_reversible=existing.is_reversible,
            )

            # Retracted procedures are not retrievable via search indexes
            self._deindex_procedure(existing)
            self._store[procedure_id] = retracted
            self._record_op("delete", procedure_id, notes=reason)

            logger.debug(
                "ProceduralKnowledgeEngine.delete_procedure id=%s reason=%r",
                procedure_id[:8],
                reason,
            )
            return retracted

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD — RETRIEVE
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_procedure(self, procedure_id: str) -> Procedure:
        """
        Fetch a single procedure by its unique ID.

        Args:
            procedure_id: The procedure's UUID string.

        Returns:
            The matching Procedure record (any lifecycle status).

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   No procedure with the given ID exists.
        """
        self._require_initialized("retrieve_procedure")
        with self._lock:
            return self._resolve_procedure(procedure_id, "retrieve_procedure")

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    def search_procedures(
        self,
        query: str,
        *,
        domain_ids: Optional[list[str]] = None,
        procedure_types: Optional[list[ProcedureType]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Procedure]:
        """
        Search procedures by name, description, goal, step content, and aliases.

        Args:
            query:           Free-text search string (case-insensitive substring).
            domain_ids:      Restrict to procedures belonging to these domains.
            procedure_types: Restrict to these ProcedureType values.
            difficulty:      Exact difficulty tier filter.
            status_filter:   Restrict to these KnowledgeStatus values.
            min_confidence:  Minimum metadata.confidence_score threshold.
            limit:           Maximum number of results to return.
            offset:          Pagination offset.

        Returns:
            Ordered list of matching Procedure records (highest confidence first,
            then alphabetically by name).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("search_procedures")

        with self._lock:
            q = query.strip().lower()

            # Narrow candidate set using domain index if filter provided
            if domain_ids:
                candidate_ids: set[str] = set()
                for did in domain_ids:
                    candidate_ids.update(self._domain_index.get(did, set()))
                candidates = [
                    self._store[pid] for pid in candidate_ids if pid in self._store
                ]
            else:
                candidates = list(self._store.values())

            active_statuses = set(status_filter) if status_filter else None

            type_set = (
                {pt.value for pt in procedure_types} if procedure_types else None
            )

            results: list[Procedure] = []
            for proc in candidates:
                if active_statuses is not None:
                    if proc.status not in active_statuses:
                        continue
                else:
                    if proc.status.is_terminal:
                        continue
                if proc.metadata.confidence_score < min_confidence:
                    continue
                if type_set and proc.procedure_type.value not in type_set:
                    continue
                if difficulty and proc.difficulty != difficulty:
                    continue

                if q:
                    # Build full searchable corpus from key text fields
                    step_text = " ".join(
                        f"{s.title} {s.instruction}" for s in proc.steps
                    )
                    searchable = " ".join([
                        proc.name,
                        proc.description,
                        proc.goal,
                        " ".join(proc.aliases),
                        " ".join(proc.preconditions),
                        " ".join(proc.postconditions),
                        step_text,
                    ]).lower()
                    if q not in searchable:
                        continue

                results.append(proc)

            # Sort: highest confidence first, then name ascending
            results.sort(
                key=lambda p: (-p.metadata.confidence_score, p.name.lower())
            )
            return results[offset: offset + limit]

    def search_procedures_by_type(
        self,
        procedure_type: ProcedureType,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Procedure]:
        """
        Return all procedures of the given ProcedureType.

        Args:
            procedure_type:  The ProcedureType to filter by.
            status_filter:   Restrict to these KnowledgeStatus values.
            limit:           Maximum number of results.
            offset:          Pagination offset.

        Returns:
            List of matching Procedure records sorted by name.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("search_procedures_by_type")

        with self._lock:
            allowed = (
                set(status_filter)
                if status_filter
                else {KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE}
            )
            ids = self._type_index.get(procedure_type.value, set())
            procedures = [
                self._store[pid]
                for pid in ids
                if pid in self._store and self._store[pid].status in allowed
            ]
            procedures.sort(key=lambda p: p.name.lower())
            return procedures[offset: offset + limit]

    def search_procedures_by_domain(
        self,
        domain_id: str,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Procedure]:
        """
        Return all procedures associated with a given knowledge domain.

        Args:
            domain_id:     Domain to query.
            status_filter: Restrict to these KnowledgeStatus values.
            limit:         Maximum number of results.
            offset:        Pagination offset.

        Returns:
            List of matching Procedure records sorted by name.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("search_procedures_by_domain")

        with self._lock:
            allowed = (
                set(status_filter)
                if status_filter
                else {KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE}
            )
            ids = self._domain_index.get(domain_id, set())
            procedures = [
                self._store[pid]
                for pid in ids
                if pid in self._store and self._store[pid].status in allowed
            ]
            procedures.sort(key=lambda p: p.name.lower())
            return procedures[offset: offset + limit]

    def search_procedures_by_difficulty(
        self,
        difficulty: KnowledgeDifficulty,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Procedure]:
        """
        Return all procedures at the given difficulty tier.

        Args:
            difficulty:    Exact KnowledgeDifficulty tier to filter by.
            status_filter: Restrict to these KnowledgeStatus values.
            limit:         Maximum number of results.
            offset:        Pagination offset.

        Returns:
            List of matching Procedure records sorted by name.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("search_procedures_by_difficulty")

        with self._lock:
            allowed = (
                set(status_filter)
                if status_filter
                else {KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE}
            )
            ids = self._difficulty_index.get(difficulty.value, set())
            procedures = [
                self._store[pid]
                for pid in ids
                if pid in self._store and self._store[pid].status in allowed
            ]
            procedures.sort(key=lambda p: p.name.lower())
            return procedures[offset: offset + limit]

    def search_procedures_by_status(
        self,
        status: KnowledgeStatus,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Procedure]:
        """
        Return all procedures with the given lifecycle status.

        Args:
            status:  KnowledgeStatus to filter by.
            limit:   Maximum number of results.
            offset:  Pagination offset.

        Returns:
            List of matching Procedure records sorted by name.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("search_procedures_by_status")

        with self._lock:
            ids = self._status_index.get(status.value, set())
            procedures = [
                self._store[pid]
                for pid in ids
                if pid in self._store
            ]
            procedures.sort(key=lambda p: p.name.lower())
            return procedures[offset: offset + limit]

    def get_all_procedures(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[Procedure]:
        """
        Return a paginated slice of the entire procedure store.

        Args:
            status_filter: If provided, restrict to these statuses.
            limit:         Maximum number of results.
            offset:        Pagination offset.

        Returns:
            All matching Procedure records sorted by name.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_all_procedures")

        with self._lock:
            allowed = set(status_filter) if status_filter else None
            procedures = [
                p
                for p in self._store.values()
                if allowed is None or p.status in allowed
            ]
            procedures.sort(key=lambda p: p.name.lower())
            return procedures[offset: offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # PREREQUISITE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def add_prerequisite(
        self,
        procedure_id: str,
        prerequisite_id: str,
        *,
        is_hard: bool = True,
        description: str = "",
    ) -> KnowledgeDependency:
        """
        Add a prerequisite edge: prerequisite_id must be completed before
        procedure_id.

        Args:
            procedure_id:     The dependent procedure.
            prerequisite_id:  The procedure that must precede it.
            is_hard:          True = blocking prerequisite; False = recommended.
            description:      Human-readable rationale for the dependency.

        Returns:
            The newly created KnowledgeDependency record.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   Either procedure ID does not exist.
            ProcedureValidationError: Adding the prerequisite would create a cycle.
        """
        self._require_initialized("add_prerequisite")

        with self._lock:
            self._resolve_procedure(procedure_id, "add_prerequisite")
            # prerequisite may be outside this store (cross-engine reference)
            # — we only perform cycle detection within our store

            dep = KnowledgeDependency.create(
                dependent_id=procedure_id,
                dependency_id=prerequisite_id,
                dependency_type="requires",
                is_hard=is_hard,
                description=description,
            )
            self._dependencies[dep.id] = dep
            self._dep_outgoing[procedure_id].add(dep.id)
            self._dep_incoming[prerequisite_id].add(dep.id)

            cycle = self._detect_dependency_cycle(procedure_id)
            if cycle:
                # Roll back
                self._dep_outgoing[procedure_id].discard(dep.id)
                self._dep_incoming[prerequisite_id].discard(dep.id)
                del self._dependencies[dep.id]
                cycle_str = " → ".join(cycle)
                raise ProcedureValidationError(
                    procedure_id=procedure_id,
                    violations=[
                        f"Adding prerequisite '{prerequisite_id}' introduces a "
                        f"circular dependency: {cycle_str}."
                    ],
                )

            # Invalidate validation cache for this procedure
            self._validation_cache.pop(procedure_id, None)
            logger.debug(
                "ProceduralKnowledgeEngine.add_prerequisite "
                "procedure=%s prerequisite=%s",
                procedure_id[:8],
                prerequisite_id[:8],
            )
            return dep

    def remove_prerequisite(
        self, procedure_id: str, prerequisite_id: str
    ) -> None:
        """
        Remove the dependency edge where procedure_id depends on prerequisite_id.

        If no such edge exists, the call is a no-op.

        Args:
            procedure_id:     The dependent procedure.
            prerequisite_id:  The procedure to disconnect.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ProcedureNotFoundError:  procedure_id does not exist in the store.
        """
        self._require_initialized("remove_prerequisite")

        with self._lock:
            self._resolve_procedure(procedure_id, "remove_prerequisite")

            to_remove: list[str] = []
            for dep_id in self._dep_outgoing.get(procedure_id, set()):
                dep = self._dependencies.get(dep_id)
                if dep is not None and dep.dependency_id == prerequisite_id:
                    to_remove.append(dep_id)

            for dep_id in to_remove:
                self._dep_outgoing[procedure_id].discard(dep_id)
                self._dep_incoming[prerequisite_id].discard(dep_id)
                del self._dependencies[dep_id]

            if to_remove:
                self._validation_cache.pop(procedure_id, None)

    def get_prerequisites(
        self, procedure_id: str, *, hard_only: bool = False
    ) -> list[KnowledgeDependency]:
        """
        Return direct prerequisite dependency records for the given procedure.

        Args:
            procedure_id:  Target procedure.
            hard_only:     If True, return only blocking (hard) dependencies.

        Returns:
            List of KnowledgeDependency records where dependent_id = procedure_id.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ProcedureNotFoundError:  procedure_id does not exist.
        """
        self._require_initialized("get_prerequisites")

        with self._lock:
            self._resolve_procedure(procedure_id, "get_prerequisites")
            deps = [
                self._dependencies[dep_id]
                for dep_id in self._dep_outgoing.get(procedure_id, set())
                if dep_id in self._dependencies
            ]
            if hard_only:
                deps = [d for d in deps if d.is_hard]
            return deps

    def get_dependents(self, procedure_id: str) -> list[KnowledgeDependency]:
        """
        Return dependency records where other procedures depend on the given one.

        Args:
            procedure_id:  The procedure whose dependents are sought.

        Returns:
            List of KnowledgeDependency records where dependency_id = procedure_id.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ProcedureNotFoundError:  procedure_id does not exist.
        """
        self._require_initialized("get_dependents")

        with self._lock:
            self._resolve_procedure(procedure_id, "get_dependents")
            return [
                self._dependencies[dep_id]
                for dep_id in self._dep_incoming.get(procedure_id, set())
                if dep_id in self._dependencies
            ]

    def get_transitive_prerequisites(
        self, procedure_id: str
    ) -> list[str]:
        """
        Return all procedure IDs that must be completed before the given
        procedure, including transitive dependencies (recursive closure).

        The returned list is topologically ordered from deepest prerequisite
        to immediate prerequisite.

        Args:
            procedure_id:  Target procedure.

        Returns:
            Ordered list of procedure IDs (no duplicates, no self-reference).

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   procedure_id does not exist.
            ProcedureValidationError: A cycle is detected in the dependency graph.
        """
        self._require_initialized("get_transitive_prerequisites")

        with self._lock:
            self._resolve_procedure(procedure_id, "get_transitive_prerequisites")

            cycle = self._detect_dependency_cycle(procedure_id)
            if cycle:
                cycle_str = " → ".join(cycle)
                raise ProcedureValidationError(
                    procedure_id=procedure_id,
                    violations=[
                        f"Circular dependency detected during transitive resolution: "
                        f"{cycle_str}."
                    ],
                )

            return self._resolve_dependency_chain(procedure_id)

    # ─────────────────────────────────────────────────────────────────────────
    # DEPENDENCY MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def add_dependency(
        self,
        dependent_id: str,
        dependency_id: str,
        *,
        dependency_type: str = "requires",
        is_hard: bool = True,
        description: str = "",
    ) -> KnowledgeDependency:
        """
        Explicitly add a typed dependency between two procedures.

        More general than add_prerequisite — supports all dependency types
        (requires, recommends, enhances) and is not restricted to procedures
        within this store (cross-engine dependencies are permitted).

        Args:
            dependent_id:     Procedure that carries the dependency.
            dependency_id:    Procedure (or external ID) being depended upon.
            dependency_type:  "requires" | "recommends" | "enhances".
            is_hard:          Whether the dependency is blocking.
            description:      Human-readable rationale.

        Returns:
            The created KnowledgeDependency record.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   dependent_id does not exist in the store.
            ProcedureValidationError: Adding this dependency would create a cycle.
        """
        self._require_initialized("add_dependency")

        with self._lock:
            self._resolve_procedure(dependent_id, "add_dependency")

            dep = KnowledgeDependency.create(
                dependent_id=dependent_id,
                dependency_id=dependency_id,
                dependency_type=dependency_type,
                is_hard=is_hard,
                description=description,
            )
            self._dependencies[dep.id] = dep
            self._dep_outgoing[dependent_id].add(dep.id)
            self._dep_incoming[dependency_id].add(dep.id)

            # Cycle detection only applies to procedures in our store
            if dependency_id in self._store:
                cycle = self._detect_dependency_cycle(dependent_id)
                if cycle:
                    self._dep_outgoing[dependent_id].discard(dep.id)
                    self._dep_incoming[dependency_id].discard(dep.id)
                    del self._dependencies[dep.id]
                    cycle_str = " → ".join(cycle)
                    raise ProcedureValidationError(
                        procedure_id=dependent_id,
                        violations=[
                            f"Adding dependency '{dependency_id}' introduces a "
                            f"circular dependency: {cycle_str}."
                        ],
                    )

            self._validation_cache.pop(dependent_id, None)
            return dep

    def remove_dependency(self, dependency_id: str) -> None:
        """
        Remove a KnowledgeDependency by its ID.

        If the dependency does not exist, the call is a no-op.

        Args:
            dependency_id: ID of the KnowledgeDependency record to remove.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("remove_dependency")

        with self._lock:
            dep = self._dependencies.pop(dependency_id, None)
            if dep is None:
                return
            self._dep_outgoing[dep.dependent_id].discard(dependency_id)
            self._dep_incoming[dep.dependency_id].discard(dependency_id)
            self._validation_cache.pop(dep.dependent_id, None)

    def get_dependency(self, dependency_id: str) -> Optional[KnowledgeDependency]:
        """
        Fetch a single KnowledgeDependency by its ID.

        Returns None if not found.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_dependency")
        with self._lock:
            return self._dependencies.get(dependency_id)

    def list_dependencies(
        self,
        *,
        dependent_id: Optional[str] = None,
        dependency_id: Optional[str] = None,
    ) -> list[KnowledgeDependency]:
        """
        List dependency records, optionally filtered by dependent or dependency side.

        Args:
            dependent_id:   If provided, return deps where dependent_id matches.
            dependency_id:  If provided, return deps where dependency_id matches.

        Returns:
            List of KnowledgeDependency records.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("list_dependencies")

        with self._lock:
            if dependent_id is not None:
                dep_ids = self._dep_outgoing.get(dependent_id, set())
                return [
                    self._dependencies[d]
                    for d in dep_ids
                    if d in self._dependencies
                ]
            if dependency_id is not None:
                dep_ids = self._dep_incoming.get(dependency_id, set())
                return [
                    self._dependencies[d]
                    for d in dep_ids
                    if d in self._dependencies
                ]
            return list(self._dependencies.values())

    # ─────────────────────────────────────────────────────────────────────────
    # REFERENCE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def add_reference(
        self,
        procedure_id: str,
        target: str,
        reference_type: str,
        *,
        description: str = "",
    ) -> KnowledgeReference:
        """
        Attach a typed citation reference to a procedure.

        Args:
            procedure_id:    The procedure that is doing the referencing.
            target:          Target knowledge record ID or external URI.
            reference_type:  "supports" | "contradicts" | "elaborates" | "cites".
            description:     Optional human-readable explanation.

        Returns:
            The created KnowledgeReference record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ProcedureNotFoundError:  procedure_id does not exist.
        """
        self._require_initialized("add_reference")

        with self._lock:
            self._resolve_procedure(procedure_id, "add_reference")

            ref = KnowledgeReference.create(
                source_id=procedure_id,
                target=target,
                reference_type=reference_type,
                description=description,
            )
            self._references[ref.id] = ref
            self._ref_index[procedure_id].add(ref.id)
            self._validation_cache.pop(procedure_id, None)
            return ref

    def remove_reference(self, reference_id: str) -> None:
        """
        Remove a KnowledgeReference by its ID.

        No-op if the reference does not exist.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("remove_reference")

        with self._lock:
            ref = self._references.pop(reference_id, None)
            if ref is None:
                return
            self._ref_index[ref.source_id].discard(reference_id)
            self._validation_cache.pop(ref.source_id, None)

    def get_references(self, procedure_id: str) -> list[KnowledgeReference]:
        """
        Return all KnowledgeReference records attached to the given procedure.

        Args:
            procedure_id:  Target procedure.

        Returns:
            List of KnowledgeReference records.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ProcedureNotFoundError:  procedure_id does not exist.
        """
        self._require_initialized("get_references")

        with self._lock:
            self._resolve_procedure(procedure_id, "get_references")
            ref_ids = self._ref_index.get(procedure_id, set())
            return [
                self._references[rid]
                for rid in ref_ids
                if rid in self._references
            ]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def add_step(self, procedure_id: str, step: ProcedureStep) -> Procedure:
        """
        Append a new step to an existing procedure's step sequence.

        The step is appended to the end of the current sequence.  If the
        provided step_number conflicts with an existing step, a new step_number
        is assigned automatically (len(existing_steps) + 1).  Full step
        validation is re-run after insertion.

        Args:
            procedure_id:  The procedure to modify.
            step:          The ProcedureStep to append.

        Returns:
            The updated Procedure record.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   procedure_id does not exist.
            ProcedureValidationError: Updated step sequence fails validation.
        """
        self._require_initialized("add_step")

        with self._lock:
            existing = self._resolve_procedure(procedure_id, "add_step")

            if existing.status.is_terminal:
                raise ProcedureValidationError(
                    procedure_id=procedure_id,
                    violations=[
                        f"Cannot add steps to a procedure with terminal "
                        f"status '{existing.status.value}'."
                    ],
                )

            current_steps = list(existing.steps)
            existing_numbers = {s.step_number for s in current_steps}

            # Auto-assign step number if conflict
            if step.step_number in existing_numbers or step.step_number < 1:
                new_number = max(existing_numbers, default=0) + 1
                # Rebuild step with corrected number
                step = ProcedureStep(
                    step_number=new_number,
                    title=step.title,
                    instruction=step.instruction,
                    sub_steps=step.sub_steps,
                    warnings=step.warnings,
                    expected_outcome=step.expected_outcome,
                    is_critical=step.is_critical,
                    estimated_minutes=step.estimated_minutes,
                )

            current_steps.append(step)
            # Re-sort by step number to maintain ordering invariant
            current_steps.sort(key=lambda s: s.step_number)

            updated = Procedure(
                id=existing.id,
                knowledge_type=existing.knowledge_type,
                name=existing.name,
                description=existing.description,
                status=existing.status,
                difficulty=existing.difficulty,
                domain_ids=list(existing.domain_ids),
                metadata=existing.metadata.bump_version(),
                aliases=list(existing.aliases),
                related_ids=list(existing.related_ids),
                notes=existing.notes,
                procedure_type=existing.procedure_type,
                goal=existing.goal,
                steps=current_steps,
                required_skill_ids=list(existing.required_skill_ids),
                required_concept_ids=list(existing.required_concept_ids),
                required_tools=list(existing.required_tools),
                required_materials=list(existing.required_materials),
                preconditions=list(existing.preconditions),
                postconditions=list(existing.postconditions),
                common_pitfalls=list(existing.common_pitfalls),
                expected_duration_minutes=existing.expected_duration_minutes,
                is_reversible=existing.is_reversible,
            )

            step_issues = self._validate_steps(updated)
            blocking = [i for i in step_issues if i.severity.is_blocking()]
            if blocking:
                raise ProcedureValidationError(
                    procedure_id=procedure_id,
                    violations=[i.message for i in blocking],
                )

            self._reindex_procedure(existing, updated)
            self._store[procedure_id] = updated
            self._record_op(
                "update",
                procedure_id,
                notes=f"add_step: step {step.step_number}",
            )
            return updated

    def remove_step(self, procedure_id: str, step_number: int) -> Procedure:
        """
        Remove the step with the given step_number from a procedure.

        Remaining steps are renumbered consecutively from 1 to maintain the
        ordering invariant.

        Args:
            procedure_id:  The procedure to modify.
            step_number:   The step number to remove.

        Returns:
            The updated Procedure record.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   procedure_id does not exist.
            ProcedureValidationError: step_number does not exist in the procedure,
                                      or the procedure has terminal status.
        """
        self._require_initialized("remove_step")

        with self._lock:
            existing = self._resolve_procedure(procedure_id, "remove_step")

            if existing.status.is_terminal:
                raise ProcedureValidationError(
                    procedure_id=procedure_id,
                    violations=[
                        f"Cannot remove steps from a procedure with terminal "
                        f"status '{existing.status.value}'."
                    ],
                )

            current_steps = list(existing.steps)
            original_len = len(current_steps)
            current_steps = [s for s in current_steps if s.step_number != step_number]

            if len(current_steps) == original_len:
                raise ProcedureValidationError(
                    procedure_id=procedure_id,
                    violations=[
                        f"Step {step_number} does not exist in procedure "
                        f"'{procedure_id}'."
                    ],
                )

            # Renumber remaining steps consecutively
            renumbered: list[ProcedureStep] = []
            for i, s in enumerate(
                sorted(current_steps, key=lambda x: x.step_number), start=1
            ):
                renumbered.append(ProcedureStep(
                    step_number=i,
                    title=s.title,
                    instruction=s.instruction,
                    sub_steps=s.sub_steps,
                    warnings=s.warnings,
                    expected_outcome=s.expected_outcome,
                    is_critical=s.is_critical,
                    estimated_minutes=s.estimated_minutes,
                ))

            updated = Procedure(
                id=existing.id,
                knowledge_type=existing.knowledge_type,
                name=existing.name,
                description=existing.description,
                status=existing.status,
                difficulty=existing.difficulty,
                domain_ids=list(existing.domain_ids),
                metadata=existing.metadata.bump_version(),
                aliases=list(existing.aliases),
                related_ids=list(existing.related_ids),
                notes=existing.notes,
                procedure_type=existing.procedure_type,
                goal=existing.goal,
                steps=renumbered,
                required_skill_ids=list(existing.required_skill_ids),
                required_concept_ids=list(existing.required_concept_ids),
                required_tools=list(existing.required_tools),
                required_materials=list(existing.required_materials),
                preconditions=list(existing.preconditions),
                postconditions=list(existing.postconditions),
                common_pitfalls=list(existing.common_pitfalls),
                expected_duration_minutes=existing.expected_duration_minutes,
                is_reversible=existing.is_reversible,
            )

            self._reindex_procedure(existing, updated)
            self._store[procedure_id] = updated
            self._record_op(
                "update",
                procedure_id,
                notes=f"remove_step: step {step_number}",
            )
            return updated

    def replace_steps(
        self, procedure_id: str, steps: list[ProcedureStep]
    ) -> Procedure:
        """
        Replace the entire step sequence of a procedure.

        All provided steps are validated for ordering, uniqueness, and
        structural completeness before the replacement is committed.

        Args:
            procedure_id:  The procedure to modify.
            steps:         Complete replacement step list.

        Returns:
            The updated Procedure record.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   procedure_id does not exist.
            ProcedureValidationError: The new step sequence fails validation.
        """
        self._require_initialized("replace_steps")

        with self._lock:
            existing = self._resolve_procedure(procedure_id, "replace_steps")

            if existing.status.is_terminal:
                raise ProcedureValidationError(
                    procedure_id=procedure_id,
                    violations=[
                        f"Cannot replace steps on a procedure with terminal "
                        f"status '{existing.status.value}'."
                    ],
                )

            updated = Procedure(
                id=existing.id,
                knowledge_type=existing.knowledge_type,
                name=existing.name,
                description=existing.description,
                status=existing.status,
                difficulty=existing.difficulty,
                domain_ids=list(existing.domain_ids),
                metadata=existing.metadata.bump_version(),
                aliases=list(existing.aliases),
                related_ids=list(existing.related_ids),
                notes=existing.notes,
                procedure_type=existing.procedure_type,
                goal=existing.goal,
                steps=list(steps),
                required_skill_ids=list(existing.required_skill_ids),
                required_concept_ids=list(existing.required_concept_ids),
                required_tools=list(existing.required_tools),
                required_materials=list(existing.required_materials),
                preconditions=list(existing.preconditions),
                postconditions=list(existing.postconditions),
                common_pitfalls=list(existing.common_pitfalls),
                expected_duration_minutes=existing.expected_duration_minutes,
                is_reversible=existing.is_reversible,
            )

            step_issues = self._validate_steps(updated)
            blocking = [i for i in step_issues if i.severity.is_blocking()]
            if blocking:
                raise ProcedureValidationError(
                    procedure_id=procedure_id,
                    violations=[i.message for i in blocking],
                )

            self._reindex_procedure(existing, updated)
            self._store[procedure_id] = updated
            self._record_op(
                "update",
                procedure_id,
                notes=f"replace_steps: {len(steps)} steps",
            )
            return updated

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION (PUBLIC)
    # ─────────────────────────────────────────────────────────────────────────

    def validate_procedure(self, procedure_id: str) -> KnowledgeValidationResult:
        """
        Run structural and semantic validation on a single procedure.

        Checks performed:
            - Non-empty name, description, goal
            - confidence_score in [0.0, 1.0]
            - knowledge_type == PROCEDURE
            - domain_ids non-empty
            - Step ordering: no duplicate step numbers, contiguous from 1
            - Step structural integrity: title, instruction, outcome on critical steps
            - Circular dependency detection in dependency graph
            - Broken dependency references (dependencies pointing to unknown records)
            - Self-referential required_skill_ids / required_concept_ids
            - Metadata source non-empty

        Results are cached; the cache is invalidated on any mutation to the
        procedure or its dependency graph.

        Args:
            procedure_id:  ID of the procedure to validate.

        Returns:
            A KnowledgeValidationResult capturing pass/fail and all issues.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ProcedureNotFoundError:  No procedure with the given ID exists.
        """
        self._require_initialized("validate_procedure")

        with self._lock:
            procedure = self._resolve_procedure(procedure_id, "validate_procedure")

            if procedure_id in self._validation_cache:
                return self._validation_cache[procedure_id]

            result = self._run_validation(procedure)
            self._validation_cache[procedure_id] = result
            return result

    def validate_all_procedures(self) -> list[KnowledgeValidationResult]:
        """
        Run validation on every procedure in the store.

        Useful for bulk health checks and audit pipelines.

        Returns:
            List of KnowledgeValidationResult, one per stored procedure.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("validate_all_procedures")

        with self._lock:
            results: list[KnowledgeValidationResult] = []
            for proc in self._store.values():
                if proc.id in self._validation_cache:
                    results.append(self._validation_cache[proc.id])
                else:
                    result = self._run_validation(proc)
                    self._validation_cache[proc.id] = result
                    results.append(result)
            return results

    # ─────────────────────────────────────────────────────────────────────────
    # INTEGRITY
    # ─────────────────────────────────────────────────────────────────────────

    def find_duplicate_procedures(self) -> list[list[Procedure]]:
        """
        Detect groups of procedures that share the same content fingerprint.

        Returns:
            A list of groups, where each group contains two or more procedures
            that are semantically identical by fingerprint.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("find_duplicate_procedures")

        with self._lock:
            fp_groups: dict[str, list[Procedure]] = defaultdict(list)
            for proc in self._store.values():
                if not proc.status.is_terminal:
                    fp_groups[proc.fingerprint].append(proc)
            return [g for g in fp_groups.values() if len(g) > 1]

    def find_broken_dependencies(self) -> list[KnowledgeDependency]:
        """
        Return all KnowledgeDependency records that reference procedures not
        present in this engine's store.

        Note: Cross-engine dependencies (e.g. to skills or concepts) will
        always appear as broken from this engine's perspective — callers should
        filter appropriately based on their cross-engine lookup capabilities.

        Returns:
            List of KnowledgeDependency records with unresolvable dependency_ids.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("find_broken_dependencies")

        with self._lock:
            broken: list[KnowledgeDependency] = []
            for dep in self._dependencies.values():
                if dep.dependency_id not in self._store:
                    broken.append(dep)
            return broken

    def find_circular_dependencies(self) -> list[list[str]]:
        """
        Scan the entire dependency graph for cyclic edges.

        Returns:
            A list of cycles, where each cycle is an ordered list of
            procedure IDs forming the cycle.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("find_circular_dependencies")

        with self._lock:
            all_cycles: list[list[str]] = []
            checked: set[str] = set()

            for procedure_id in self._store:
                if procedure_id in checked:
                    continue
                cycle = self._detect_dependency_cycle(procedure_id)
                if cycle:
                    # Normalize cycle representation to avoid duplicates
                    min_node = min(cycle[:-1])
                    min_idx = cycle.index(min_node)
                    normalized = cycle[min_idx:-1] + cycle[:min_idx] + [min_node]
                    if normalized not in all_cycles:
                        all_cycles.append(cycle)
                checked.add(procedure_id)

            return all_cycles

    def procedure_exists(self, procedure_id: str) -> bool:
        """
        Return True if a procedure with the given ID exists in the store
        (any lifecycle status, including retracted).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("procedure_exists")
        with self._lock:
            return procedure_id in self._store

    def get_procedure_count(self, *, active_only: bool = False) -> int:
        """
        Return the number of procedures in the store.

        Args:
            active_only: If True, count only VALIDATED and ACTIVE procedures.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_procedure_count")
        with self._lock:
            if not active_only:
                return len(self._store)
            return sum(
                1
                for p in self._store.values()
                if p.status in (KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE)
            )

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIDENCE TRACKING
    # ─────────────────────────────────────────────────────────────────────────

    def get_low_confidence_procedures(
        self, threshold: float = _LOW_CONFIDENCE_THRESHOLD
    ) -> list[Procedure]:
        """
        Return all active procedures with confidence_score below threshold.

        Args:
            threshold: Confidence score ceiling (exclusive).  Defaults to 0.40.

        Returns:
            List of Procedure records sorted by ascending confidence score.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_low_confidence_procedures")

        with self._lock:
            low = [
                p
                for p in self._store.values()
                if (
                    p.status in (KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE)
                    and p.metadata.confidence_score < threshold
                )
            ]
            low.sort(key=lambda p: p.metadata.confidence_score)
            return low

    def update_confidence(
        self, procedure_id: str, confidence_score: float
    ) -> Procedure:
        """
        Update the confidence score of a procedure record.

        Creates a bumped metadata version with the new score.

        Args:
            procedure_id:     Target procedure.
            confidence_score: New score in [0.0, 1.0].

        Returns:
            The updated Procedure record.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            ProcedureNotFoundError:   procedure_id does not exist.
            ProcedureValidationError: Score is outside [0.0, 1.0].
        """
        self._require_initialized("update_confidence")

        if not (0.0 <= confidence_score <= 1.0):
            raise ProcedureValidationError(
                procedure_id=procedure_id,
                violations=[
                    f"confidence_score {confidence_score!r} is outside [0.0, 1.0]."
                ],
            )

        with self._lock:
            existing = self._resolve_procedure(procedure_id, "update_confidence")

            new_metadata = existing.metadata.bump_version(
                confidence_score=confidence_score
            )
            updated = Procedure(
                id=existing.id,
                knowledge_type=existing.knowledge_type,
                name=existing.name,
                description=existing.description,
                status=existing.status,
                difficulty=existing.difficulty,
                domain_ids=list(existing.domain_ids),
                metadata=new_metadata,
                aliases=list(existing.aliases),
                related_ids=list(existing.related_ids),
                notes=existing.notes,
                procedure_type=existing.procedure_type,
                goal=existing.goal,
                steps=list(existing.steps),
                required_skill_ids=list(existing.required_skill_ids),
                required_concept_ids=list(existing.required_concept_ids),
                required_tools=list(existing.required_tools),
                required_materials=list(existing.required_materials),
                preconditions=list(existing.preconditions),
                postconditions=list(existing.postconditions),
                common_pitfalls=list(existing.common_pitfalls),
                expected_duration_minutes=existing.expected_duration_minutes,
                is_reversible=existing.is_reversible,
            )

            self._reindex_procedure(existing, updated)
            self._store[procedure_id] = updated
            self._record_op(
                "update",
                procedure_id,
                notes=f"confidence updated: {confidence_score:.4f}",
            )
            return updated

    # ─────────────────────────────────────────────────────────────────────────
    # INDEXING
    # ─────────────────────────────────────────────────────────────────────────

    def get_index_summary(self) -> dict[str, Any]:
        """
        Return a summary of all secondary index sizes and compositions.

        Returns a dict with keys:
            total_procedures         (int)
            by_type                  (dict[str, int])
            by_domain                (dict[str, int])
            by_difficulty            (dict[str, int])
            by_status                (dict[str, int])
            fingerprint_entries      (int)
            dependency_edges         (int)
            reference_entries        (int)

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_index_summary")

        with self._lock:
            return {
                "total_procedures": len(self._store),
                "by_type": {k: len(v) for k, v in self._type_index.items()},
                "by_domain": {k: len(v) for k, v in self._domain_index.items()},
                "by_difficulty": {k: len(v) for k, v in self._difficulty_index.items()},
                "by_status": {k: len(v) for k, v in self._status_index.items()},
                "fingerprint_entries": len(self._fingerprint_index),
                "dependency_edges": len(self._dependencies),
                "reference_entries": len(self._references),
            }

    def rebuild_indexes(self) -> int:
        """
        Rebuild all secondary indexes from scratch using the current store.

        Useful as a recovery operation if index corruption is suspected.

        Returns:
            The number of procedures re-indexed.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("rebuild_indexes")

        with self._lock:
            self._fingerprint_index.clear()
            self._domain_index.clear()
            self._type_index.clear()
            self._difficulty_index.clear()
            self._status_index.clear()

            count = 0
            for proc in self._store.values():
                self._index_procedure(proc)
                count += 1

            logger.info(
                "ProceduralKnowledgeEngine.rebuild_indexes: %d procedures re-indexed",
                count,
            )
            return count

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT
    # ─────────────────────────────────────────────────────────────────────────

    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate statistics for the entire procedure store.

        Required keys:
            engine                  (str)
            version                 (str)
            total_procedures        (int)
            active_procedures       (int)
            draft_procedures        (int)
            deprecated_procedures   (int)
            retracted_procedures    (int)
            avg_confidence          (float)
            procedures_by_type      (dict[str, int])
            procedures_by_domain    (dict[str, int])
            procedures_by_difficulty(dict[str, int])
            duplicate_groups        (int)
            low_confidence_ids      (list[str])
            stale_ids               (list[str])
            broken_dependency_count (int)
            circular_dependency_count(int)
            total_dependencies      (int)
            total_references        (int)
            mutation_count          (int)
            op_log_entries          (int)
            generated_at            (str)

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("audit_report")

        with self._lock:
            all_procs = list(self._store.values())
            total = len(all_procs)

            status_counts: dict[str, int] = defaultdict(int)
            type_counts: dict[str, int] = defaultdict(int)
            domain_counts: dict[str, int] = defaultdict(int)
            difficulty_counts: dict[str, int] = defaultdict(int)
            confidence_sum = 0.0
            low_confidence_ids: list[str] = []
            stale_ids: list[str] = []

            for proc in all_procs:
                status_counts[proc.status.value] += 1
                type_counts[proc.procedure_type.value] += 1
                difficulty_counts[proc.difficulty.value] += 1
                for did in proc.domain_ids:
                    domain_counts[did] += 1
                confidence_sum += proc.metadata.confidence_score
                if proc.metadata.confidence_score < _LOW_CONFIDENCE_THRESHOLD:
                    low_confidence_ids.append(proc.id)
                if proc.metadata.is_stale:
                    stale_ids.append(proc.id)

            avg_confidence = confidence_sum / total if total > 0 else 0.0
            duplicate_groups = self.find_duplicate_procedures()
            broken_deps = self.find_broken_dependencies()
            circular_deps = self.find_circular_dependencies()

            return {
                "engine": "ProceduralKnowledgeEngine",
                "version": _ENGINE_VERSION,
                "total_procedures": total,
                "active_procedures": (
                    status_counts.get(KnowledgeStatus.ACTIVE.value, 0)
                    + status_counts.get(KnowledgeStatus.VALIDATED.value, 0)
                ),
                "draft_procedures": status_counts.get(KnowledgeStatus.DRAFT.value, 0),
                "deprecated_procedures": status_counts.get(
                    KnowledgeStatus.DEPRECATED.value, 0
                ),
                "retracted_procedures": status_counts.get(
                    KnowledgeStatus.RETRACTED.value, 0
                ),
                "avg_confidence": round(avg_confidence, 4),
                "procedures_by_type": dict(type_counts),
                "procedures_by_domain": dict(domain_counts),
                "procedures_by_difficulty": dict(difficulty_counts),
                "duplicate_groups": len(duplicate_groups),
                "low_confidence_ids": low_confidence_ids,
                "stale_ids": stale_ids,
                "broken_dependency_count": len(broken_deps),
                "circular_dependency_count": len(circular_deps),
                "total_dependencies": len(self._dependencies),
                "total_references": len(self._references),
                "duplicate_check_count": self._duplicate_check_count,
                "mutation_count": self._mutation_count,
                "op_log_entries": len(self._op_log),
                "generated_at": _utcnow().isoformat(),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        with self._lock:
            active = (
                sum(
                    1
                    for p in self._store.values()
                    if p.status
                    in (KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE)
                )
                if self._initialized
                else 0
            )
            return {
                "engine": "ProceduralKnowledgeEngine",
                "initialized": self._initialized,
                "record_count": len(self._store) if self._initialized else 0,
                "active_count": active,
                "status": "healthy" if self._initialized else "offline",
            }

    def diagnostics_report(self) -> dict[str, Any]:
        with self._lock:
            report = self.health_report()
            report.update({
                "index_size": (
                    len(self._fingerprint_index)
                    + len(self._domain_index)
                    + len(self._type_index)
                    + len(self._difficulty_index)
                    + len(self._status_index)
                ),
                "duplicate_checks": self._duplicate_check_count,
                "mutation_count": self._mutation_count,
                "last_mutation_at": (
                    self._last_mutation_at.isoformat()
                    if self._last_mutation_at
                    else None
                ),
                "fingerprint_index_size": len(self._fingerprint_index),
                "domain_index_domains": len(self._domain_index),
                "type_index_types": len(self._type_index),
                "difficulty_index_tiers": len(self._difficulty_index),
                "status_index_statuses": len(self._status_index),
                "dependency_edges": len(self._dependencies),
                "dependency_graph_nodes_outgoing": len(self._dep_outgoing),
                "dependency_graph_nodes_incoming": len(self._dep_incoming),
                "reference_entries": len(self._references),
                "validation_cache_size": len(self._validation_cache),
                "op_log_entries": len(self._op_log),
                "engine_version": _ENGINE_VERSION,
                "validator_version": _VALIDATOR_VERSION,
            })
            return report


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ProceduralKnowledgeEngine",
]