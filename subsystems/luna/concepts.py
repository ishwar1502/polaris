"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/concepts.py

Concrete implementation of the LUNA Concept Engine.

The Concept Engine manages abstract conceptual knowledge owned by LUNA:
    - Machine Learning
    - Robotics
    - PID Control
    - State Space Representation
    - Systems Engineering
    - Mechatronics

Concepts form the semantic scaffold on which facts, skills, and procedures
hang.  They are organized into parent-child hierarchies, linked to their
prerequisite concepts, and associated with the facts that instantiate them.

Ownership laws observed by this module:
    Law 1: LUNA owns knowledge.  Nobody else.
    Law 2: LUNA owns concepts.  CONSTELLATION only links them.

Implementation notes:
    - In-memory v1 store backed by a dict[str, Concept]
    - Thread-safe via threading.RLock
    - Lifecycle-gated: every public method raises LunaNotInitializedError
      before initialize() is called or after shutdown()
    - Duplicate detection by SHA-256 content fingerprint
    - Cycle detection via DFS on the prerequisite/child graph
    - Integrates with FactEngine for cross-engine fact validation
    - Soft-delete only: deleted concepts transition to RETRACTED

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
    ConceptNotFoundError,
    ConceptRelationshipError,
    ConceptValidationError,
    DuplicateConceptError,
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractConceptEngine
from subsystems.luna.models import (
    ConfidenceLevel,
    Concept,
    ConceptType,
    KnowledgeDifficulty,
    KnowledgeMetadata,
    KnowledgeSourceType,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeValidationResult,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
    _new_id,
    _stable_hash,
    _utcnow,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE VERSION
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE_VERSION: str = "5.0.0"

_MAX_NAME_LENGTH: int = 256
_MAX_DESCRIPTION_LENGTH: int = 8_192


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION LOG ENTRY
# ─────────────────────────────────────────────────────────────────────────────

class _ConceptOpEntry:
    """Immutable record of a single mutation applied to the concept store."""

    __slots__ = ("op", "concept_id", "timestamp", "notes")

    def __init__(self, op: str, concept_id: str, notes: str = "") -> None:
        self.op: str = op              # "create" | "update" | "delete" | "link" | "unlink"
        self.concept_id: str = concept_id
        self.timestamp: datetime = _utcnow()
        self.notes: str = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "concept_id": self.concept_id,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ConceptEngine(AbstractConceptEngine):
    """
    In-memory, thread-safe implementation of the LUNA Concept Engine (v1).

    Hierarchy management:
        prerequisite_concept_ids  — concepts the user must understand BEFORE this one
        child_concept_ids         — sub-concepts that build on this one

    The engine maintains two adjacency dictionaries for fast graph traversal:
        _prereq_graph[A]   = {B, C}  means A requires B and C
        _children_graph[A] = {D, E}  means D and E are children of A

    Cycle detection uses iterative DFS to avoid recursion-depth limits.

    Optional FactEngine integration:
        If a FactEngine instance is supplied at construction time, the
        validate_concept() method additionally checks that all fact_ids
        registered on a concept resolve to real facts.

    Lifecycle:
        engine = ConceptEngine(fact_engine=my_fact_engine)
        engine.initialize()
        # ... use engine ...
        engine.shutdown()
    """

    def __init__(self, fact_engine: Optional[Any] = None) -> None:
        """
        Args:
            fact_engine: Optional AbstractFactEngine instance.  When provided,
                         validate_concept() will verify fact_ids against it.
        """
        self._fact_engine = fact_engine
        self._lock: threading.RLock = threading.RLock()

        # Primary store: concept_id → Concept
        self._store: dict[str, Concept] = {}

        # Deduplication index: fingerprint → concept_id
        self._fingerprint_index: dict[str, str] = {}

        # Domain index: domain_id → set[concept_id]
        self._domain_index: dict[str, set[str]] = defaultdict(set)

        # ConceptType index: ConceptType.value → set[concept_id]
        self._type_index: dict[str, set[str]] = defaultdict(set)

        # Prerequisite adjacency: concept_id → set[prerequisite_ids]
        # Meaning: _prereq_graph[A] = {B} ⟺ A requires B
        self._prereq_graph: dict[str, set[str]] = defaultdict(set)

        # Children adjacency: concept_id → set[child_concept_ids]
        # Meaning: _children_graph[A] = {B} ⟺ B is a child of A
        self._children_graph: dict[str, set[str]] = defaultdict(set)

        # Validation result cache: concept_id → KnowledgeValidationResult
        self._validation_cache: dict[str, KnowledgeValidationResult] = {}

        # Immutable operation log
        self._op_log: list[_ConceptOpEntry] = []

        # Counters
        self._mutation_count: int = 0
        self._last_mutation_at: Optional[datetime] = None

        # Lifecycle flag
        self._initialized: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Prepare the Concept Engine for operation.

        Idempotent: repeated calls after the first initialization are no-ops.

        Raises:
            LunaLifecycleError: If internal setup fails.
        """
        with self._lock:
            if self._initialized:
                return
            try:
                self._store.clear()
                self._fingerprint_index.clear()
                self._domain_index.clear()
                self._type_index.clear()
                self._prereq_graph.clear()
                self._children_graph.clear()
                self._validation_cache.clear()
                self._op_log.clear()
                self._mutation_count = 0
                self._last_mutation_at = None
                self._initialized = True
                logger.info("ConceptEngine initialized (version=%s)", _ENGINE_VERSION)
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="ConceptEngine",
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
                    "ConceptEngine shutdown (concepts=%d, mutations=%d)",
                    len(self._store),
                    self._mutation_count,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="ConceptEngine",
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
    # INTERNAL HELPERS — INDEXING
    # ─────────────────────────────────────────────────────────────────────────

    def _index_concept(self, concept: Concept) -> None:
        """Register a concept in all secondary indexes."""
        self._fingerprint_index[concept.fingerprint] = concept.id
        for domain_id in concept.domain_ids:
            self._domain_index[domain_id].add(concept.id)
        self._type_index[concept.concept_type.value].add(concept.id)
        # Rebuild adjacency entries from the concept's stored relationships
        self._prereq_graph[concept.id] = set(concept.prerequisite_concept_ids)
        for prereq_id in concept.prerequisite_concept_ids:
            self._children_graph[prereq_id].add(concept.id)
        self._children_graph[concept.id] = set(concept.child_concept_ids)
        for child_id in concept.child_concept_ids:
            self._prereq_graph[child_id].add(concept.id)

    def _deindex_concept(self, concept: Concept) -> None:
        """Remove a concept from all secondary indexes."""
        self._fingerprint_index.pop(concept.fingerprint, None)
        for domain_id in concept.domain_ids:
            self._domain_index[domain_id].discard(concept.id)
        self._type_index[concept.concept_type.value].discard(concept.id)
        # Remove adjacency entries
        for prereq_id in self._prereq_graph.pop(concept.id, set()):
            self._children_graph[prereq_id].discard(concept.id)
        for child_id in self._children_graph.pop(concept.id, set()):
            self._prereq_graph[child_id].discard(concept.id)

    def _reindex_concept(self, old: Concept, new: Concept) -> None:
        self._deindex_concept(old)
        self._index_concept(new)

    def _record_op(self, op: str, concept_id: str, notes: str = "") -> None:
        """Append an operation log entry and update mutation counters."""
        self._op_log.append(_ConceptOpEntry(op=op, concept_id=concept_id, notes=notes))
        self._mutation_count += 1
        self._last_mutation_at = _utcnow()
        self._validation_cache.pop(concept_id, None)

    def _resolve_concept(self, concept_id: str, operation: str) -> Concept:
        """Return the concept or raise ConceptNotFoundError."""
        concept = self._store.get(concept_id)
        if concept is None:
            raise ConceptNotFoundError(
                concept_id=concept_id,
                context={"operation": operation},
            )
        return concept

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — DUPLICATE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _check_duplicate(self, concept: Concept) -> None:
        """Raise DuplicateConceptError if the fingerprint already exists in a live record."""
        existing_id = self._fingerprint_index.get(concept.fingerprint)
        if existing_id is None:
            return
        existing = self._store.get(existing_id)
        if existing is None:
            return
        if not existing.status.is_terminal:
            raise DuplicateConceptError(
                concept_id=concept.id,
                existing_id=existing_id,
                context={"fingerprint": concept.fingerprint},
            )

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — CYCLE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _would_create_cycle(self, from_id: str, to_id: str) -> bool:
        """
        Return True if adding the directed edge from_id → to_id (meaning
        "from_id now has to_id as a prerequisite") would create a cycle in
        the prerequisite graph.

        Uses iterative DFS from to_id; if we can reach from_id, a cycle
        would result.
        """
        # If to_id already transitively requires from_id, adding from_id → to_id
        # creates a cycle (from_id → to_id → ... → from_id).
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

    def _topological_prerequisites(self, concept_id: str) -> list[str]:
        """
        Return all transitive prerequisite concept IDs in topological order
        (shallowest/most-foundational first).

        Raises:
            ConceptRelationshipError: If a cycle is detected.
        """
        # Kahn's algorithm on the prerequisite subgraph rooted at concept_id
        # Build in-degree for BFS-based topological sort
        reachable: set[str] = set()
        frontier: deque[str] = deque([concept_id])
        while frontier:
            cid = frontier.popleft()
            for prereq in self._prereq_graph.get(cid, set()):
                if prereq not in reachable:
                    reachable.add(prereq)
                    frontier.append(prereq)

        if concept_id in reachable:
            raise ConceptRelationshipError(
                source_id=concept_id,
                target_id=concept_id,
                relationship="prerequisite",
                message=f"Cycle detected in prerequisite graph for concept '{concept_id}'.",
            )

        # Topological ordering of reachable set
        in_degree: dict[str, int] = {cid: 0 for cid in reachable}
        adj: dict[str, list[str]] = {cid: [] for cid in reachable}
        for cid in reachable:
            for prereq in self._prereq_graph.get(cid, set()):
                if prereq in reachable:
                    adj[prereq].append(cid)
                    in_degree[cid] += 1

        queue: deque[str] = deque(
            cid for cid, deg in in_degree.items() if deg == 0
        )
        ordered: list[str] = []
        while queue:
            cid = queue.popleft()
            ordered.append(cid)
            for neighbor in adj.get(cid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(reachable):
            raise ConceptRelationshipError(
                source_id=concept_id,
                target_id="<multiple>",
                relationship="prerequisite",
                message=f"Cycle detected in transitive prerequisites of '{concept_id}'.",
            )

        return ordered

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    def _run_validation(self, concept: Concept) -> KnowledgeValidationResult:
        """Execute all validation rules against a Concept instance."""
        issues: list[ValidationIssue] = []

        # Rule: name must be non-empty
        if not concept.name or not concept.name.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.CRITICAL,
                message="Concept name must not be empty.",
                field="name",
            ))

        # Rule: name length
        if len(concept.name) > _MAX_NAME_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"Concept name exceeds {_MAX_NAME_LENGTH} characters.",
                field="name",
            ))

        # Rule: description must be non-empty
        if not concept.description or not concept.description.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.CRITICAL,
                message="Concept description must not be empty.",
                field="description",
            ))

        # Rule: description length
        if len(concept.description) > _MAX_DESCRIPTION_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.WARNING,
                message=f"Concept description exceeds {_MAX_DESCRIPTION_LENGTH} characters.",
                field="description",
            ))

        # Rule: confidence_score in [0.0, 1.0]
        score = concept.metadata.confidence_score
        if not (0.0 <= score <= 1.0):
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"confidence_score {score!r} is outside [0.0, 1.0].",
                field="metadata.confidence_score",
            ))

        # Rule: domain_ids must not be empty
        if not concept.domain_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.ERROR,
                message="Concept must belong to at least one knowledge domain.",
                field="domain_ids",
            ))

        # Rule: knowledge_type must be CONCEPT
        if concept.knowledge_type != KnowledgeType.CONCEPT:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    f"knowledge_type must be KnowledgeType.CONCEPT, "
                    f"got {concept.knowledge_type!r}."
                ),
                field="knowledge_type",
            ))

        # Rule: no self-referential prerequisites
        if concept.id in concept.prerequisite_concept_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message="Concept references itself in prerequisite_concept_ids.",
                field="prerequisite_concept_ids",
            ))

        # Rule: no self-referential children
        if concept.id in concept.child_concept_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message="Concept references itself in child_concept_ids.",
                field="child_concept_ids",
            ))

        # Rule: all prerequisite_concept_ids must exist in the store
        for prereq_id in concept.prerequisite_concept_ids:
            if prereq_id not in self._store:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"prerequisite_concept_id '{prereq_id}' does not "
                        f"exist in the concept store."
                    ),
                    field="prerequisite_concept_ids",
                ))

        # Rule: all child_concept_ids must exist in the store
        for child_id in concept.child_concept_ids:
            if child_id not in self._store:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"child_concept_id '{child_id}' does not "
                        f"exist in the concept store."
                    ),
                    field="child_concept_ids",
                ))

        # Rule: all fact_ids must exist in the fact store (if integrated)
        if self._fact_engine is not None and self._fact_engine.is_initialized():
            for fact_id in concept.fact_ids:
                if not self._fact_engine.fact_exists(fact_id):
                    issues.append(ValidationIssue.create(
                        issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"fact_id '{fact_id}' does not exist in the fact store."
                        ),
                        field="fact_ids",
                    ))

        # Rule: low-confidence warning
        if concept.metadata.confidence_score < 0.40:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.LOW_CONFIDENCE,
                severity=ValidationSeverity.WARNING,
                message=(
                    f"Confidence score {concept.metadata.confidence_score:.2f} is low "
                    f"({ConfidenceLevel.from_score(concept.metadata.confidence_score).value})."
                ),
                field="metadata.confidence_score",
            ))

        # Rule: unknown source type warning
        if concept.metadata.source_type == KnowledgeSourceType.UNKNOWN:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.UNRELIABLE_SOURCE,
                severity=ValidationSeverity.WARNING,
                message="Concept source type is UNKNOWN; consider providing a reliable source.",
                field="metadata.source_type",
            ))

        return KnowledgeValidationResult.create(
            knowledge_id=concept.id,
            knowledge_name=concept.name,
            issues=issues,
            validator_version=_ENGINE_VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD
    # ─────────────────────────────────────────────────────────────────────────

    def create_concept(
        self,
        name: str,
        description: str,
        concept_type: ConceptType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        core_ideas: Optional[list[str]] = None,
        applications: Optional[list[str]] = None,
        prerequisite_concept_ids: Optional[list[str]] = None,
        is_foundational: bool = False,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> Concept:
        self._require_initialized("create_concept")

        with self._lock:
            candidate = Concept.create(
                name=name,
                description=description,
                concept_type=concept_type,
                difficulty=difficulty,
                domain_ids=list(domain_ids),
                metadata=metadata,
                core_ideas=core_ideas,
                applications=applications,
                prerequisite_concept_ids=prerequisite_concept_ids,
                is_foundational=is_foundational,
                aliases=aliases,
                notes=notes,
            )

            # Validate prerequisite IDs exist before running full validation
            for prereq_id in (prerequisite_concept_ids or []):
                if prereq_id not in self._store:
                    raise ConceptNotFoundError(
                        concept_id=prereq_id,
                        context={
                            "operation": "create_concept",
                            "reason": "prerequisite concept not found",
                        },
                    )

            # Structural validation
            result = self._run_validation(candidate)
            if result.has_blocking_issues:
                violations = [i.message for i in result.issues if i.severity.is_blocking()]
                raise ConceptValidationError(
                    concept_id=candidate.id,
                    violations=violations,
                )

            # Duplicate detection
            self._check_duplicate(candidate)

            # Cycle detection: candidate → each prerequisite
            for prereq_id in (prerequisite_concept_ids or []):
                if self._would_create_cycle(candidate.id, prereq_id):
                    raise ConceptRelationshipError(
                        source_id=candidate.id,
                        target_id=prereq_id,
                        relationship="prerequisite",
                        message=(
                            f"Adding prerequisite '{prereq_id}' to concept "
                            f"'{candidate.id}' would create a cycle."
                        ),
                    )

            # Persist
            self._store[candidate.id] = candidate
            self._index_concept(candidate)

            # Update parent concepts' child_concept_ids lists
            for prereq_id in (prerequisite_concept_ids or []):
                prereq = self._store[prereq_id]
                if candidate.id not in prereq.child_concept_ids:
                    updated_prereq = Concept(
                        id=prereq.id,
                        knowledge_type=prereq.knowledge_type,
                        name=prereq.name,
                        description=prereq.description,
                        status=prereq.status,
                        difficulty=prereq.difficulty,
                        domain_ids=list(prereq.domain_ids),
                        metadata=prereq.metadata.bump_version(),
                        aliases=list(prereq.aliases),
                        related_ids=list(prereq.related_ids),
                        notes=prereq.notes,
                        concept_type=prereq.concept_type,
                        core_ideas=list(prereq.core_ideas),
                        applications=list(prereq.applications),
                        prerequisite_concept_ids=list(prereq.prerequisite_concept_ids),
                        child_concept_ids=list(prereq.child_concept_ids) + [candidate.id],
                        fact_ids=list(prereq.fact_ids),
                        is_foundational=prereq.is_foundational,
                    )
                    self._reindex_concept(prereq, updated_prereq)
                    self._store[prereq_id] = updated_prereq

            self._record_op("create", candidate.id, notes=f"created: {name!r}")
            logger.debug(
                "ConceptEngine.create_concept id=%s name=%r type=%s",
                candidate.short_id,
                name,
                concept_type.value,
            )
            return candidate

    def update_concept(
        self,
        concept_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        concept_type: Optional[ConceptType] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        domain_ids: Optional[list[str]] = None,
        core_ideas: Optional[list[str]] = None,
        applications: Optional[list[str]] = None,
        prerequisite_concept_ids: Optional[list[str]] = None,
        child_concept_ids: Optional[list[str]] = None,
        fact_ids: Optional[list[str]] = None,
        is_foundational: Optional[bool] = None,
        aliases: Optional[list[str]] = None,
        related_ids: Optional[list[str]] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
    ) -> Concept:
        self._require_initialized("update_concept")

        with self._lock:
            existing = self._resolve_concept(concept_id, "update_concept")

            if existing.status.is_terminal:
                raise ConceptValidationError(
                    concept_id=concept_id,
                    violations=[
                        f"Cannot update a concept with terminal status "
                        f"'{existing.status.value}'."
                    ],
                )

            new_prereqs = (
                list(prerequisite_concept_ids)
                if prerequisite_concept_ids is not None
                else list(existing.prerequisite_concept_ids)
            )
            new_children = (
                list(child_concept_ids)
                if child_concept_ids is not None
                else list(existing.child_concept_ids)
            )

            # Validate all new prerequisite IDs exist
            if prerequisite_concept_ids is not None:
                for prereq_id in new_prereqs:
                    if prereq_id == concept_id:
                        raise ConceptValidationError(
                            concept_id=concept_id,
                            violations=["Concept cannot be its own prerequisite."],
                        )
                    if prereq_id not in self._store:
                        raise ConceptNotFoundError(
                            concept_id=prereq_id,
                            context={"operation": "update_concept", "reason": "prerequisite not found"},
                        )

            # Cycle detection for new prerequisites
            if prerequisite_concept_ids is not None:
                for prereq_id in new_prereqs:
                    if prereq_id not in existing.prerequisite_concept_ids:
                        if self._would_create_cycle(concept_id, prereq_id):
                            raise ConceptRelationshipError(
                                source_id=concept_id,
                                target_id=prereq_id,
                                relationship="prerequisite",
                                message=(
                                    f"Adding prerequisite '{prereq_id}' to concept "
                                    f"'{concept_id}' would create a cycle."
                                ),
                            )

            # Build updated concept
            updated = Concept(
                id=existing.id,
                knowledge_type=existing.knowledge_type,
                name=name if name is not None else existing.name,
                description=description if description is not None else existing.description,
                status=status if status is not None else existing.status,
                difficulty=difficulty if difficulty is not None else existing.difficulty,
                domain_ids=list(domain_ids) if domain_ids is not None else list(existing.domain_ids),
                metadata=metadata.bump_version() if metadata is not None else existing.metadata.bump_version(),
                aliases=list(aliases) if aliases is not None else list(existing.aliases),
                related_ids=list(related_ids) if related_ids is not None else list(existing.related_ids),
                notes=notes if notes is not None else existing.notes,
                concept_type=concept_type if concept_type is not None else existing.concept_type,
                core_ideas=list(core_ideas) if core_ideas is not None else list(existing.core_ideas),
                applications=list(applications) if applications is not None else list(existing.applications),
                prerequisite_concept_ids=new_prereqs,
                child_concept_ids=new_children,
                fact_ids=list(fact_ids) if fact_ids is not None else list(existing.fact_ids),
                is_foundational=is_foundational if is_foundational is not None else existing.is_foundational,
            )

            # Structural validation
            result = self._run_validation(updated)
            if result.has_blocking_issues:
                violations = [i.message for i in result.issues if i.severity.is_blocking()]
                raise ConceptValidationError(concept_id=concept_id, violations=violations)

            # Duplicate fingerprint check (only if identity-affecting fields changed)
            if name is not None or description is not None:
                fp = updated.fingerprint
                existing_fp_id = self._fingerprint_index.get(fp)
                if existing_fp_id is not None and existing_fp_id != concept_id:
                    other = self._store.get(existing_fp_id)
                    if other and not other.status.is_terminal:
                        raise DuplicateConceptError(
                            concept_id=concept_id,
                            existing_id=existing_fp_id,
                            context={"fingerprint": fp},
                        )

            # Sync parent concepts: handle added/removed prerequisites
            if prerequisite_concept_ids is not None:
                old_set = set(existing.prerequisite_concept_ids)
                new_set = set(new_prereqs)
                added = new_set - old_set
                removed = old_set - new_set
                for prereq_id in added:
                    if prereq_id in self._store:
                        self._add_child_to_concept(prereq_id, concept_id)
                for prereq_id in removed:
                    if prereq_id in self._store:
                        self._remove_child_from_concept(prereq_id, concept_id)

            self._reindex_concept(existing, updated)
            self._store[concept_id] = updated
            self._record_op("update", concept_id, notes=f"updated: {updated.name!r}")

            logger.debug("ConceptEngine.update_concept id=%s", concept_id[:8])
            return updated

    def delete_concept(self, concept_id: str, *, reason: str = "") -> Concept:
        self._require_initialized("delete_concept")

        with self._lock:
            existing = self._resolve_concept(concept_id, "delete_concept")

            if existing.status == KnowledgeStatus.RETRACTED:
                return existing  # Idempotent

            deletion_note = (
                f"[RETRACTED {_utcnow().isoformat()}]"
                + (f" Reason: {reason}" if reason else "")
            )

            retracted = Concept(
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
                concept_type=existing.concept_type,
                core_ideas=list(existing.core_ideas),
                applications=list(existing.applications),
                prerequisite_concept_ids=list(existing.prerequisite_concept_ids),
                child_concept_ids=list(existing.child_concept_ids),
                fact_ids=list(existing.fact_ids),
                is_foundational=existing.is_foundational,
            )

            self._deindex_concept(existing)
            self._store[concept_id] = retracted
            self._record_op("delete", concept_id, notes=reason)

            logger.debug("ConceptEngine.delete_concept id=%s reason=%r", concept_id[:8], reason)
            return retracted

    def retrieve_concept(self, concept_id: str) -> Concept:
        self._require_initialized("retrieve_concept")
        with self._lock:
            return self._resolve_concept(concept_id, "retrieve_concept")

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    def search_concepts(
        self,
        query: str,
        *,
        domain_ids: Optional[list[str]] = None,
        concept_types: Optional[list[ConceptType]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        foundational_only: bool = False,
        min_confidence: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Concept]:
        self._require_initialized("search_concepts")

        with self._lock:
            q = query.strip().lower()

            if domain_ids:
                candidate_ids: set[str] = set()
                for did in domain_ids:
                    candidate_ids.update(self._domain_index.get(did, set()))
                candidates = [self._store[cid] for cid in candidate_ids if cid in self._store]
            else:
                candidates = list(self._store.values())

            active_statuses = set(status_filter) if status_filter else {
                KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE
            }
            type_set = {ct.value for ct in concept_types} if concept_types else None

            results: list[Concept] = []
            for concept in candidates:
                if concept.status not in active_statuses:
                    continue
                if concept.metadata.confidence_score < min_confidence:
                    continue
                if type_set and concept.concept_type.value not in type_set:
                    continue
                if difficulty and concept.difficulty != difficulty:
                    continue
                if foundational_only and not concept.is_foundational:
                    continue
                if q:
                    searchable = " ".join([
                        concept.name,
                        concept.description,
                        " ".join(concept.aliases),
                        " ".join(concept.core_ideas),
                        " ".join(concept.applications),
                    ]).lower()
                    if q not in searchable:
                        continue
                results.append(concept)

            results.sort(key=lambda c: (-c.metadata.confidence_score, c.name.lower()))
            return results[offset: offset + limit]

    def search_concepts_by_domain(
        self,
        domain_id: str,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Concept]:
        self._require_initialized("search_concepts_by_domain")

        with self._lock:
            allowed = set(status_filter) if status_filter else {
                KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE
            }
            ids = self._domain_index.get(domain_id, set())
            concepts = [
                self._store[cid]
                for cid in ids
                if cid in self._store and self._store[cid].status in allowed
            ]
            concepts.sort(key=lambda c: c.name.lower())
            return concepts[offset: offset + limit]

    def get_all_concepts(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[Concept]:
        self._require_initialized("get_all_concepts")

        with self._lock:
            allowed = set(status_filter) if status_filter else None
            concepts = [
                c for c in self._store.values()
                if allowed is None or c.status in allowed
            ]
            concepts.sort(key=lambda c: c.name.lower())
            return concepts[offset: offset + limit]

    def get_foundational_concepts(self) -> list[Concept]:
        self._require_initialized("get_foundational_concepts")

        with self._lock:
            return [
                c for c in self._store.values()
                if c.is_foundational
                and not c.status.is_terminal
            ]

    # ─────────────────────────────────────────────────────────────────────────
    # HIERARCHY
    # ─────────────────────────────────────────────────────────────────────────

    def get_children(self, concept_id: str) -> list[Concept]:
        self._require_initialized("get_children")

        with self._lock:
            self._resolve_concept(concept_id, "get_children")
            child_ids = self._children_graph.get(concept_id, set())
            return [
                self._store[cid]
                for cid in child_ids
                if cid in self._store
            ]

    def get_prerequisites(self, concept_id: str) -> list[Concept]:
        self._require_initialized("get_prerequisites")

        with self._lock:
            self._resolve_concept(concept_id, "get_prerequisites")
            prereq_ids = self._prereq_graph.get(concept_id, set())
            return [
                self._store[pid]
                for pid in prereq_ids
                if pid in self._store
            ]

    def get_transitive_prerequisites(self, concept_id: str) -> list[Concept]:
        self._require_initialized("get_transitive_prerequisites")

        with self._lock:
            self._resolve_concept(concept_id, "get_transitive_prerequisites")
            ordered_ids = self._topological_prerequisites(concept_id)
            return [
                self._store[pid]
                for pid in ordered_ids
                if pid in self._store
            ]

    def add_prerequisite(self, concept_id: str, prerequisite_id: str) -> Concept:
        self._require_initialized("add_prerequisite")

        with self._lock:
            concept = self._resolve_concept(concept_id, "add_prerequisite")
            self._resolve_concept(prerequisite_id, "add_prerequisite")

            if concept_id == prerequisite_id:
                raise ConceptRelationshipError(
                    source_id=concept_id,
                    target_id=prerequisite_id,
                    relationship="prerequisite",
                    message="A concept cannot be its own prerequisite.",
                )

            if prerequisite_id in concept.prerequisite_concept_ids:
                return concept  # Idempotent

            if self._would_create_cycle(concept_id, prerequisite_id):
                raise ConceptRelationshipError(
                    source_id=concept_id,
                    target_id=prerequisite_id,
                    relationship="prerequisite",
                    message=(
                        f"Adding prerequisite '{prerequisite_id}' to concept "
                        f"'{concept_id}' would create a cycle."
                    ),
                )

            updated = Concept(
                id=concept.id,
                knowledge_type=concept.knowledge_type,
                name=concept.name,
                description=concept.description,
                status=concept.status,
                difficulty=concept.difficulty,
                domain_ids=list(concept.domain_ids),
                metadata=concept.metadata.bump_version(),
                aliases=list(concept.aliases),
                related_ids=list(concept.related_ids),
                notes=concept.notes,
                concept_type=concept.concept_type,
                core_ideas=list(concept.core_ideas),
                applications=list(concept.applications),
                prerequisite_concept_ids=list(concept.prerequisite_concept_ids) + [prerequisite_id],
                child_concept_ids=list(concept.child_concept_ids),
                fact_ids=list(concept.fact_ids),
                is_foundational=concept.is_foundational,
            )
            self._reindex_concept(concept, updated)
            self._store[concept_id] = updated

            # Register reverse edge
            self._add_child_to_concept(prerequisite_id, concept_id)

            self._record_op(
                "link",
                concept_id,
                notes=f"added prerequisite: {prerequisite_id}",
            )
            return updated

    def remove_prerequisite(self, concept_id: str, prerequisite_id: str) -> Concept:
        self._require_initialized("remove_prerequisite")

        with self._lock:
            concept = self._resolve_concept(concept_id, "remove_prerequisite")
            self._resolve_concept(prerequisite_id, "remove_prerequisite")

            if prerequisite_id not in concept.prerequisite_concept_ids:
                return concept  # Idempotent

            updated = Concept(
                id=concept.id,
                knowledge_type=concept.knowledge_type,
                name=concept.name,
                description=concept.description,
                status=concept.status,
                difficulty=concept.difficulty,
                domain_ids=list(concept.domain_ids),
                metadata=concept.metadata.bump_version(),
                aliases=list(concept.aliases),
                related_ids=list(concept.related_ids),
                notes=concept.notes,
                concept_type=concept.concept_type,
                core_ideas=list(concept.core_ideas),
                applications=list(concept.applications),
                prerequisite_concept_ids=[
                    p for p in concept.prerequisite_concept_ids
                    if p != prerequisite_id
                ],
                child_concept_ids=list(concept.child_concept_ids),
                fact_ids=list(concept.fact_ids),
                is_foundational=concept.is_foundational,
            )
            self._reindex_concept(concept, updated)
            self._store[concept_id] = updated

            self._remove_child_from_concept(prerequisite_id, concept_id)
            self._record_op(
                "unlink",
                concept_id,
                notes=f"removed prerequisite: {prerequisite_id}",
            )
            return updated

    def add_child_concept(self, parent_id: str, child_id: str) -> Concept:
        self._require_initialized("add_child_concept")

        with self._lock:
            parent = self._resolve_concept(parent_id, "add_child_concept")
            self._resolve_concept(child_id, "add_child_concept")

            if parent_id == child_id:
                raise ConceptRelationshipError(
                    source_id=parent_id,
                    target_id=child_id,
                    relationship="child",
                    message="A concept cannot be its own child.",
                )

            if child_id in parent.child_concept_ids:
                return parent  # Idempotent

            # Adding child_id as child of parent_id is equivalent to
            # making parent_id a prerequisite of child_id.
            if self._would_create_cycle(child_id, parent_id):
                raise ConceptRelationshipError(
                    source_id=parent_id,
                    target_id=child_id,
                    relationship="child",
                    message=(
                        f"Adding '{child_id}' as child of '{parent_id}' "
                        f"would create a cycle."
                    ),
                )

            # Update parent's child_concept_ids
            updated_parent = Concept(
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
                concept_type=parent.concept_type,
                core_ideas=list(parent.core_ideas),
                applications=list(parent.applications),
                prerequisite_concept_ids=list(parent.prerequisite_concept_ids),
                child_concept_ids=list(parent.child_concept_ids) + [child_id],
                fact_ids=list(parent.fact_ids),
                is_foundational=parent.is_foundational,
            )
            self._reindex_concept(parent, updated_parent)
            self._store[parent_id] = updated_parent

            # Update child's prerequisite_concept_ids
            child = self._store[child_id]
            if parent_id not in child.prerequisite_concept_ids:
                updated_child = Concept(
                    id=child.id,
                    knowledge_type=child.knowledge_type,
                    name=child.name,
                    description=child.description,
                    status=child.status,
                    difficulty=child.difficulty,
                    domain_ids=list(child.domain_ids),
                    metadata=child.metadata.bump_version(),
                    aliases=list(child.aliases),
                    related_ids=list(child.related_ids),
                    notes=child.notes,
                    concept_type=child.concept_type,
                    core_ideas=list(child.core_ideas),
                    applications=list(child.applications),
                    prerequisite_concept_ids=list(child.prerequisite_concept_ids) + [parent_id],
                    child_concept_ids=list(child.child_concept_ids),
                    fact_ids=list(child.fact_ids),
                    is_foundational=child.is_foundational,
                )
                self._reindex_concept(child, updated_child)
                self._store[child_id] = updated_child

            self._record_op("link", parent_id, notes=f"added child: {child_id}")
            return updated_parent

    def remove_child_concept(self, parent_id: str, child_id: str) -> Concept:
        self._require_initialized("remove_child_concept")

        with self._lock:
            parent = self._resolve_concept(parent_id, "remove_child_concept")
            self._resolve_concept(child_id, "remove_child_concept")

            if child_id not in parent.child_concept_ids:
                return parent  # Idempotent

            updated_parent = Concept(
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
                concept_type=parent.concept_type,
                core_ideas=list(parent.core_ideas),
                applications=list(parent.applications),
                prerequisite_concept_ids=list(parent.prerequisite_concept_ids),
                child_concept_ids=[c for c in parent.child_concept_ids if c != child_id],
                fact_ids=list(parent.fact_ids),
                is_foundational=parent.is_foundational,
            )
            self._reindex_concept(parent, updated_parent)
            self._store[parent_id] = updated_parent

            # Remove parent from child's prerequisites
            child = self._store.get(child_id)
            if child and parent_id in child.prerequisite_concept_ids:
                updated_child = Concept(
                    id=child.id,
                    knowledge_type=child.knowledge_type,
                    name=child.name,
                    description=child.description,
                    status=child.status,
                    difficulty=child.difficulty,
                    domain_ids=list(child.domain_ids),
                    metadata=child.metadata.bump_version(),
                    aliases=list(child.aliases),
                    related_ids=list(child.related_ids),
                    notes=child.notes,
                    concept_type=child.concept_type,
                    core_ideas=list(child.core_ideas),
                    applications=list(child.applications),
                    prerequisite_concept_ids=[
                        p for p in child.prerequisite_concept_ids if p != parent_id
                    ],
                    child_concept_ids=list(child.child_concept_ids),
                    fact_ids=list(child.fact_ids),
                    is_foundational=child.is_foundational,
                )
                self._reindex_concept(child, updated_child)
                self._store[child_id] = updated_child

            self._record_op("unlink", parent_id, notes=f"removed child: {child_id}")
            return updated_parent

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HIERARCHY SYNC HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _add_child_to_concept(self, concept_id: str, new_child_id: str) -> None:
        """
        Append new_child_id to concept_id.child_concept_ids without going
        through the public add_child_concept (avoids double cycle-check).
        """
        concept = self._store.get(concept_id)
        if concept is None or new_child_id in concept.child_concept_ids:
            return
        updated = Concept(
            id=concept.id,
            knowledge_type=concept.knowledge_type,
            name=concept.name,
            description=concept.description,
            status=concept.status,
            difficulty=concept.difficulty,
            domain_ids=list(concept.domain_ids),
            metadata=concept.metadata.bump_version(),
            aliases=list(concept.aliases),
            related_ids=list(concept.related_ids),
            notes=concept.notes,
            concept_type=concept.concept_type,
            core_ideas=list(concept.core_ideas),
            applications=list(concept.applications),
            prerequisite_concept_ids=list(concept.prerequisite_concept_ids),
            child_concept_ids=list(concept.child_concept_ids) + [new_child_id],
            fact_ids=list(concept.fact_ids),
            is_foundational=concept.is_foundational,
        )
        self._reindex_concept(concept, updated)
        self._store[concept_id] = updated

    def _remove_child_from_concept(self, concept_id: str, child_id: str) -> None:
        """
        Remove child_id from concept_id.child_concept_ids without going
        through the public remove_child_concept.
        """
        concept = self._store.get(concept_id)
        if concept is None or child_id not in concept.child_concept_ids:
            return
        updated = Concept(
            id=concept.id,
            knowledge_type=concept.knowledge_type,
            name=concept.name,
            description=concept.description,
            status=concept.status,
            difficulty=concept.difficulty,
            domain_ids=list(concept.domain_ids),
            metadata=concept.metadata.bump_version(),
            aliases=list(concept.aliases),
            related_ids=list(concept.related_ids),
            notes=concept.notes,
            concept_type=concept.concept_type,
            core_ideas=list(concept.core_ideas),
            applications=list(concept.applications),
            prerequisite_concept_ids=list(concept.prerequisite_concept_ids),
            child_concept_ids=[c for c in concept.child_concept_ids if c != child_id],
            fact_ids=list(concept.fact_ids),
            is_foundational=concept.is_foundational,
        )
        self._reindex_concept(concept, updated)
        self._store[concept_id] = updated

    # ─────────────────────────────────────────────────────────────────────────
    # FACT LINKAGE
    # ─────────────────────────────────────────────────────────────────────────

    def attach_fact(self, concept_id: str, fact_id: str) -> Concept:
        self._require_initialized("attach_fact")

        with self._lock:
            concept = self._resolve_concept(concept_id, "attach_fact")

            if fact_id in concept.fact_ids:
                return concept  # Idempotent

            updated = Concept(
                id=concept.id,
                knowledge_type=concept.knowledge_type,
                name=concept.name,
                description=concept.description,
                status=concept.status,
                difficulty=concept.difficulty,
                domain_ids=list(concept.domain_ids),
                metadata=concept.metadata.bump_version(),
                aliases=list(concept.aliases),
                related_ids=list(concept.related_ids),
                notes=concept.notes,
                concept_type=concept.concept_type,
                core_ideas=list(concept.core_ideas),
                applications=list(concept.applications),
                prerequisite_concept_ids=list(concept.prerequisite_concept_ids),
                child_concept_ids=list(concept.child_concept_ids),
                fact_ids=list(concept.fact_ids) + [fact_id],
                is_foundational=concept.is_foundational,
            )
            self._reindex_concept(concept, updated)
            self._store[concept_id] = updated
            self._record_op("link", concept_id, notes=f"attached fact: {fact_id}")
            return updated

    def detach_fact(self, concept_id: str, fact_id: str) -> Concept:
        self._require_initialized("detach_fact")

        with self._lock:
            concept = self._resolve_concept(concept_id, "detach_fact")

            if fact_id not in concept.fact_ids:
                return concept  # Idempotent

            updated = Concept(
                id=concept.id,
                knowledge_type=concept.knowledge_type,
                name=concept.name,
                description=concept.description,
                status=concept.status,
                difficulty=concept.difficulty,
                domain_ids=list(concept.domain_ids),
                metadata=concept.metadata.bump_version(),
                aliases=list(concept.aliases),
                related_ids=list(concept.related_ids),
                notes=concept.notes,
                concept_type=concept.concept_type,
                core_ideas=list(concept.core_ideas),
                applications=list(concept.applications),
                prerequisite_concept_ids=list(concept.prerequisite_concept_ids),
                child_concept_ids=list(concept.child_concept_ids),
                fact_ids=[f for f in concept.fact_ids if f != fact_id],
                is_foundational=concept.is_foundational,
            )
            self._reindex_concept(concept, updated)
            self._store[concept_id] = updated
            self._record_op("unlink", concept_id, notes=f"detached fact: {fact_id}")
            return updated

    def get_facts_for_concept(self, concept_id: str) -> list[str]:
        self._require_initialized("get_facts_for_concept")
        with self._lock:
            concept = self._resolve_concept(concept_id, "get_facts_for_concept")
            return list(concept.fact_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION (PUBLIC)
    # ─────────────────────────────────────────────────────────────────────────

    def validate_concept(self, concept_id: str) -> KnowledgeValidationResult:
        self._require_initialized("validate_concept")

        with self._lock:
            concept = self._resolve_concept(concept_id, "validate_concept")

            if concept_id in self._validation_cache:
                return self._validation_cache[concept_id]

            result = self._run_validation(concept)
            self._validation_cache[concept_id] = result
            return result

    # ─────────────────────────────────────────────────────────────────────────
    # INTEGRITY
    # ─────────────────────────────────────────────────────────────────────────

    def find_duplicate_concepts(self) -> list[list[Concept]]:
        self._require_initialized("find_duplicate_concepts")

        with self._lock:
            fp_groups: dict[str, list[Concept]] = defaultdict(list)
            for concept in self._store.values():
                if not concept.status.is_terminal:
                    fp_groups[concept.fingerprint].append(concept)
            return [group for group in fp_groups.values() if len(group) > 1]

    def find_orphaned_concepts(self) -> list[Concept]:
        self._require_initialized("find_orphaned_concepts")

        with self._lock:
            return [
                c for c in self._store.values()
                if (
                    not c.status.is_terminal
                    and not c.domain_ids
                    and not c.prerequisite_concept_ids
                    and not self._children_graph.get(c.id)  # no parent points to this
                )
            ]

    def concept_exists(self, concept_id: str) -> bool:
        self._require_initialized("concept_exists")
        with self._lock:
            return concept_id in self._store

    def get_concept_count(self, *, active_only: bool = False) -> int:
        self._require_initialized("get_concept_count")
        with self._lock:
            if not active_only:
                return len(self._store)
            return sum(
                1 for c in self._store.values()
                if c.status in (KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE)
            )

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT
    # ─────────────────────────────────────────────────────────────────────────

    def audit_report(self) -> dict[str, Any]:
        self._require_initialized("audit_report")

        with self._lock:
            all_concepts = list(self._store.values())
            total = len(all_concepts)

            status_counts: dict[str, int] = defaultdict(int)
            type_counts: dict[str, int] = defaultdict(int)
            domain_counts: dict[str, int] = defaultdict(int)
            confidence_sum = 0.0
            foundational_count = 0

            for concept in all_concepts:
                status_counts[concept.status.value] += 1
                type_counts[concept.concept_type.value] += 1
                for did in concept.domain_ids:
                    domain_counts[did] += 1
                confidence_sum += concept.metadata.confidence_score
                if concept.is_foundational:
                    foundational_count += 1

            avg_confidence = confidence_sum / total if total > 0 else 0.0
            duplicate_groups = self.find_duplicate_concepts()
            orphans = self.find_orphaned_concepts()

            # Compute max hierarchy depth via BFS from root concepts
            # (concepts with no prerequisites)
            roots = [
                c.id for c in all_concepts
                if not c.status.is_terminal
                and not self._prereq_graph.get(c.id)
            ]
            max_depth = 0
            if roots:
                depth_map: dict[str, int] = {r: 0 for r in roots}
                queue: deque[str] = deque(roots)
                while queue:
                    cid = queue.popleft()
                    current_depth = depth_map.get(cid, 0)
                    max_depth = max(max_depth, current_depth)
                    for child_id in self._children_graph.get(cid, set()):
                        if child_id not in depth_map:
                            depth_map[child_id] = current_depth + 1
                            queue.append(child_id)

            return {
                "engine": "ConceptEngine",
                "version": _ENGINE_VERSION,
                "total_concepts": total,
                "active_concepts": (
                    status_counts.get(KnowledgeStatus.ACTIVE.value, 0)
                    + status_counts.get(KnowledgeStatus.VALIDATED.value, 0)
                ),
                "draft_concepts": status_counts.get(KnowledgeStatus.DRAFT.value, 0),
                "deprecated_concepts": status_counts.get(KnowledgeStatus.DEPRECATED.value, 0),
                "retracted_concepts": status_counts.get(KnowledgeStatus.RETRACTED.value, 0),
                "foundational_count": foundational_count,
                "avg_confidence": round(avg_confidence, 4),
                "concepts_by_type": dict(type_counts),
                "concepts_by_domain": dict(domain_counts),
                "duplicate_groups": len(duplicate_groups),
                "orphaned_count": len(orphans),
                "max_hierarchy_depth": max_depth,
                "mutation_count": self._mutation_count,
                "op_log_entries": len(self._op_log),
                "generated_at": _utcnow().isoformat(),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        with self._lock:
            active = sum(
                1 for c in self._store.values()
                if c.status in (KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE)
            ) if self._initialized else 0

            return {
                "engine": "ConceptEngine",
                "initialized": self._initialized,
                "record_count": len(self._store) if self._initialized else 0,
                "active_count": active,
                "status": "healthy" if self._initialized else "offline",
            }

    def diagnostics_report(self) -> dict[str, Any]:
        with self._lock:
            report = self.health_report()
            report.update({
                "index_size": len(self._fingerprint_index),
                "duplicate_checks": len(self._fingerprint_index),
                "mutation_count": self._mutation_count,
                "last_mutation_at": (
                    self._last_mutation_at.isoformat()
                    if self._last_mutation_at
                    else None
                ),
                "domain_index_domains": len(self._domain_index),
                "type_index_types": len(self._type_index),
                "prereq_graph_nodes": len(self._prereq_graph),
                "children_graph_nodes": len(self._children_graph),
                "validation_cache_size": len(self._validation_cache),
                "op_log_entries": len(self._op_log),
                "fact_engine_integrated": self._fact_engine is not None,
                "engine_version": _ENGINE_VERSION,
            })
            return report


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ConceptEngine",
]