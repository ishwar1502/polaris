"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/facts.py

Concrete implementation of the LUNA Fact Engine.

The Fact Engine manages atomic knowledge owned exclusively by LUNA:
    - Ohm's Law (V = IR)
    - Newton's Second Law (F = ma)
    - ROS2 concepts (nodes, topics, services)
    - Python syntax rules
    - Control theory constants and principles

Ownership law: LUNA owns all facts (Law 1).  No external module may write,
delete, or modify fact records without routing through this engine.

Implementation notes:
    - In-memory v1 store backed by a dict[str, Fact]
    - Thread-safe via threading.RLock (reentrant for nested engine calls)
    - Lifecycle-gated: every public method raises LunaNotInitializedError
      before initialize() has been called or after shutdown()
    - Duplicate detection by SHA-256 content fingerprint (KnowledgeType,
      name, description)
    - Soft-delete only: deleted records transition to RETRACTED, never removed
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
    DuplicateFactError,
    FactNotFoundError,
    FactValidationError,
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractFactEngine
from subsystems.luna.models import (
    ConfidenceLevel,
    Fact,
    FactType,
    KnowledgeDifficulty,
    KnowledgeMetadata,
    KnowledgeSourceType,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeValidationResult,
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
# ENGINE VERSION
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE_VERSION: str = "5.0.0"

# Maximum statement length tolerated during validation (characters)
_MAX_STATEMENT_LENGTH: int = 4_096

# Maximum name length
_MAX_NAME_LENGTH: int = 256


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION LOG ENTRY
# ─────────────────────────────────────────────────────────────────────────────

class _FactOpEntry:
    """Immutable record of a single mutation applied to the fact store."""

    __slots__ = ("op", "fact_id", "timestamp", "notes")

    def __init__(self, op: str, fact_id: str, notes: str = "") -> None:
        self.op: str = op                    # "create" | "update" | "delete"
        self.fact_id: str = fact_id
        self.timestamp: datetime = _utcnow()
        self.notes: str = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "fact_id": self.fact_id,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FACT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class FactEngine(AbstractFactEngine):
    """
    In-memory, thread-safe implementation of the LUNA Fact Engine (v1).

    All public operations are guarded by a reentrant lock so that
    callers using the engine from multiple threads see a consistent store.

    Lifecycle:
        engine = FactEngine()
        engine.initialize()
        # ... use engine ...
        engine.shutdown()
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()

        # Primary store: fact_id → Fact
        self._store: dict[str, Fact] = {}

        # Deduplication index: fingerprint → fact_id
        self._fingerprint_index: dict[str, str] = {}

        # Domain index: domain_id → set[fact_id]
        self._domain_index: dict[str, set[str]] = defaultdict(set)

        # FactType index: FactType.value → set[fact_id]
        self._type_index: dict[str, set[str]] = defaultdict(set)

        # Validation result cache: fact_id → KnowledgeValidationResult
        self._validation_cache: dict[str, KnowledgeValidationResult] = {}

        # Immutable operation log
        self._op_log: list[_FactOpEntry] = []

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
        Prepare the Fact Engine for operation.

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
                self._validation_cache.clear()
                self._op_log.clear()
                self._mutation_count = 0
                self._last_mutation_at = None
                self._initialized = True
                logger.info("FactEngine initialized (version=%s)", _ENGINE_VERSION)
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="FactEngine",
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
                    "FactEngine shutdown (facts=%d, mutations=%d)",
                    len(self._store),
                    self._mutation_count,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="FactEngine",
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

    def _index_fact(self, fact: Fact) -> None:
        """Register a fact in all secondary indexes."""
        self._fingerprint_index[fact.fingerprint] = fact.id
        for domain_id in fact.domain_ids:
            self._domain_index[domain_id].add(fact.id)
        self._type_index[fact.fact_type.value].add(fact.id)

    def _deindex_fact(self, fact: Fact) -> None:
        """Remove a fact from all secondary indexes."""
        self._fingerprint_index.pop(fact.fingerprint, None)
        for domain_id in fact.domain_ids:
            self._domain_index[domain_id].discard(fact.id)
        self._type_index[fact.fact_type.value].discard(fact.id)

    def _reindex_fact(self, old_fact: Fact, new_fact: Fact) -> None:
        """Swap index entries when a fact is updated."""
        self._deindex_fact(old_fact)
        self._index_fact(new_fact)

    def _record_op(self, op: str, fact_id: str, notes: str = "") -> None:
        """Append an operation log entry and bump mutation counters."""
        self._op_log.append(_FactOpEntry(op=op, fact_id=fact_id, notes=notes))
        self._mutation_count += 1
        self._last_mutation_at = _utcnow()
        # Invalidate cached validation result on mutation
        self._validation_cache.pop(fact_id, None)

    def _check_duplicate(self, fact: Fact) -> None:
        """
        Raise DuplicateFactError if the fact's fingerprint is already present.
        Only checks against non-retracted, non-archived records.
        """
        existing_id = self._fingerprint_index.get(fact.fingerprint)
        if existing_id is None:
            return
        existing = self._store.get(existing_id)
        if existing is None:
            return
        if existing.status not in (KnowledgeStatus.RETRACTED, KnowledgeStatus.ARCHIVED):
            raise DuplicateFactError(
                fact_id=fact.id,
                existing_id=existing_id,
                context={"fingerprint": fact.fingerprint},
            )

    def _resolve_fact(self, fact_id: str, operation: str) -> Fact:
        """Return the fact or raise FactNotFoundError."""
        fact = self._store.get(fact_id)
        if fact is None:
            raise FactNotFoundError(
                fact_id=fact_id,
                context={"operation": operation},
            )
        return fact

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION (INTERNAL)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_validation(self, fact: Fact) -> KnowledgeValidationResult:
        """Execute all validation rules against a Fact instance."""
        issues: list[ValidationIssue] = []

        # Rule: name must be non-empty
        if not fact.name or not fact.name.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.CRITICAL,
                message="Fact name must not be empty.",
                field="name",
            ))

        # Rule: name length
        if len(fact.name) > _MAX_NAME_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"Fact name exceeds {_MAX_NAME_LENGTH} characters.",
                field="name",
            ))

        # Rule: statement must be non-empty
        if not fact.statement or not fact.statement.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.CRITICAL,
                message="Fact statement must not be empty.",
                field="statement",
            ))

        # Rule: statement length
        if len(fact.statement) > _MAX_STATEMENT_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.WARNING,
                message=f"Fact statement exceeds {_MAX_STATEMENT_LENGTH} characters.",
                field="statement",
            ))

        # Rule: description must be non-empty
        if not fact.description or not fact.description.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.ERROR,
                message="Fact description must not be empty.",
                field="description",
            ))

        # Rule: confidence_score in [0.0, 1.0]
        score = fact.metadata.confidence_score
        if not (0.0 <= score <= 1.0):
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"confidence_score {score!r} is outside [0.0, 1.0].",
                field="metadata.confidence_score",
            ))

        # Rule: domain_ids must not be empty
        if not fact.domain_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.ERROR,
                message="Fact must belong to at least one knowledge domain.",
                field="domain_ids",
            ))

        # Rule: no self-referential supporting_fact_ids
        if fact.id in fact.supporting_fact_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message="Fact references itself in supporting_fact_ids.",
                field="supporting_fact_ids",
            ))

        # Rule: knowledge_type must be FACT
        if fact.knowledge_type != KnowledgeType.FACT:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    f"knowledge_type must be KnowledgeType.FACT, "
                    f"got {fact.knowledge_type!r}."
                ),
                field="knowledge_type",
            ))

        # Rule: low-confidence warning
        if fact.metadata.confidence_score < 0.40:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.LOW_CONFIDENCE,
                severity=ValidationSeverity.WARNING,
                message=(
                    f"Confidence score {fact.metadata.confidence_score:.2f} is low "
                    f"({ConfidenceLevel.from_score(fact.metadata.confidence_score).value})."
                ),
                field="metadata.confidence_score",
            ))

        # Rule: unknown source type warning
        if fact.metadata.source_type == KnowledgeSourceType.UNKNOWN:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.UNRELIABLE_SOURCE,
                severity=ValidationSeverity.WARNING,
                message="Fact source type is UNKNOWN; consider providing a reliable source.",
                field="metadata.source_type",
            ))

        return KnowledgeValidationResult.create(
            knowledge_id=fact.id,
            knowledge_name=fact.name,
            issues=issues,
            validator_version=_ENGINE_VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD
    # ─────────────────────────────────────────────────────────────────────────

    def create_fact(
        self,
        name: str,
        description: str,
        statement: str,
        fact_type: FactType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        formal_notation: Optional[str] = None,
        units: Optional[str] = None,
        conditions: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> Fact:
        self._require_initialized("create_fact")

        with self._lock:
            # Build a candidate fact (gets a fresh UUID at construction)
            candidate = Fact.create(
                name=name,
                description=description,
                statement=statement,
                fact_type=fact_type,
                difficulty=difficulty,
                domain_ids=list(domain_ids),
                metadata=metadata,
                formal_notation=formal_notation,
                units=units,
                conditions=conditions,
                aliases=aliases,
                notes=notes,
            )

            # Structural validation before persistence
            result = self._run_validation(candidate)
            if result.has_blocking_issues:
                violations = [i.message for i in result.issues if i.severity.is_blocking()]
                raise FactValidationError(
                    fact_id=candidate.id,
                    violations=violations,
                )

            # Duplicate detection
            self._check_duplicate(candidate)

            # Persist
            self._store[candidate.id] = candidate
            self._index_fact(candidate)
            self._record_op("create", candidate.id, notes=f"created: {name!r}")

            logger.debug(
                "FactEngine.create_fact id=%s name=%r type=%s",
                candidate.short_id,
                name,
                fact_type.value,
            )
            return candidate

    def update_fact(
        self,
        fact_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        statement: Optional[str] = None,
        fact_type: Optional[FactType] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        domain_ids: Optional[list[str]] = None,
        formal_notation: Optional[str] = None,
        units: Optional[str] = None,
        conditions: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        related_ids: Optional[list[str]] = None,
        supporting_fact_ids: Optional[list[str]] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
    ) -> Fact:
        self._require_initialized("update_fact")

        with self._lock:
            existing = self._resolve_fact(fact_id, "update_fact")

            if existing.status.is_terminal:
                raise FactValidationError(
                    fact_id=fact_id,
                    violations=[
                        f"Cannot update a fact with terminal status "
                        f"'{existing.status.value}'."
                    ],
                )

            # Build updated Fact by copying existing and applying changes
            updated = Fact(
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
                fact_type=fact_type if fact_type is not None else existing.fact_type,
                statement=statement if statement is not None else existing.statement,
                formal_notation=formal_notation if formal_notation is not None else existing.formal_notation,
                units=units if units is not None else existing.units,
                conditions=list(conditions) if conditions is not None else list(existing.conditions),
                counterexamples=list(existing.counterexamples),
                supporting_fact_ids=(
                    list(supporting_fact_ids)
                    if supporting_fact_ids is not None
                    else list(existing.supporting_fact_ids)
                ),
            )

            # Validate updated state
            result = self._run_validation(updated)
            if result.has_blocking_issues:
                violations = [i.message for i in result.issues if i.severity.is_blocking()]
                raise FactValidationError(fact_id=fact_id, violations=violations)

            # Check duplicate only if name/description/statement changed
            if (
                name is not None
                or description is not None
                or statement is not None
            ):
                fp = updated.fingerprint
                existing_fp_id = self._fingerprint_index.get(fp)
                if existing_fp_id is not None and existing_fp_id != fact_id:
                    existing_stored = self._store.get(existing_fp_id)
                    if existing_stored and not existing_stored.status.is_terminal:
                        raise DuplicateFactError(
                            fact_id=fact_id,
                            existing_id=existing_fp_id,
                            context={"fingerprint": fp},
                        )

            self._reindex_fact(existing, updated)
            self._store[fact_id] = updated
            self._record_op("update", fact_id, notes=f"updated: {updated.name!r}")

            logger.debug("FactEngine.update_fact id=%s", fact_id[:8])
            return updated

    def delete_fact(self, fact_id: str, *, reason: str = "") -> Fact:
        self._require_initialized("delete_fact")

        with self._lock:
            existing = self._resolve_fact(fact_id, "delete_fact")

            if existing.status == KnowledgeStatus.RETRACTED:
                return existing  # Idempotent

            deletion_note = (
                f"[RETRACTED {_utcnow().isoformat()}]"
                + (f" Reason: {reason}" if reason else "")
            )

            retracted = Fact(
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
                fact_type=existing.fact_type,
                statement=existing.statement,
                formal_notation=existing.formal_notation,
                units=existing.units,
                conditions=list(existing.conditions),
                counterexamples=list(existing.counterexamples),
                supporting_fact_ids=list(existing.supporting_fact_ids),
            )

            # Deindex (retracted facts are not retrievable via search)
            self._deindex_fact(existing)
            self._store[fact_id] = retracted
            self._record_op("delete", fact_id, notes=reason)

            logger.debug("FactEngine.delete_fact id=%s reason=%r", fact_id[:8], reason)
            return retracted

    def retrieve_fact(self, fact_id: str) -> Fact:
        self._require_initialized("retrieve_fact")

        with self._lock:
            return self._resolve_fact(fact_id, "retrieve_fact")

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    def search_facts(
        self,
        query: str,
        *,
        domain_ids: Optional[list[str]] = None,
        fact_types: Optional[list[FactType]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Fact]:
        self._require_initialized("search_facts")

        with self._lock:
            q = query.strip().lower()

            # Determine candidate set from domain index if domain filter provided
            if domain_ids:
                candidate_ids: set[str] = set()
                for did in domain_ids:
                    candidate_ids.update(self._domain_index.get(did, set()))
                candidates = [self._store[fid] for fid in candidate_ids if fid in self._store]
            else:
                candidates = list(self._store.values())

            # Apply filters
            active_statuses = status_filter or [KnowledgeStatus.DRAFT, KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE]
            type_set = {ft.value for ft in fact_types} if fact_types else None

            results: list[Fact] = []
            for fact in candidates:
                if fact.status not in active_statuses:
                    continue
                if fact.metadata.confidence_score < min_confidence:
                    continue
                if type_set and fact.fact_type.value not in type_set:
                    continue
                if difficulty and fact.difficulty != difficulty:
                    continue

                # Text matching across name, statement, description, aliases
                if q:
                    searchable = " ".join([
                        fact.name,
                        fact.statement,
                        fact.description,
                        " ".join(fact.aliases),
                        fact.formal_notation or "",
                    ]).lower()
                    if q not in searchable:
                        continue

                results.append(fact)

            # Sort by confidence descending, then name ascending
            results.sort(
                key=lambda f: (-f.metadata.confidence_score, f.name.lower())
            )

            return results[offset: offset + limit]

    def search_facts_by_type(
        self,
        fact_type: FactType,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Fact]:
        self._require_initialized("search_facts_by_type")

        with self._lock:
            ids = self._type_index.get(fact_type.value, set())
            facts = [
                self._store[fid]
                for fid in ids
                if fid in self._store
                and self._store[fid].status in (
                    KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE
                )
            ]
            facts.sort(key=lambda f: f.name.lower())
            return facts[offset: offset + limit]

    def search_facts_by_domain(
        self,
        domain_id: str,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Fact]:
        self._require_initialized("search_facts_by_domain")

        with self._lock:
            allowed = set(status_filter) if status_filter else {
                KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE
            }
            ids = self._domain_index.get(domain_id, set())
            facts = [
                self._store[fid]
                for fid in ids
                if fid in self._store and self._store[fid].status in allowed
            ]
            facts.sort(key=lambda f: f.name.lower())
            return facts[offset: offset + limit]

    def get_all_facts(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[Fact]:
        self._require_initialized("get_all_facts")

        with self._lock:
            allowed = set(status_filter) if status_filter else None
            facts = [
                f for f in self._store.values()
                if allowed is None or f.status in allowed
            ]
            facts.sort(key=lambda f: f.name.lower())
            return facts[offset: offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION (PUBLIC)
    # ─────────────────────────────────────────────────────────────────────────

    def validate_fact(self, fact_id: str) -> KnowledgeValidationResult:
        self._require_initialized("validate_fact")

        with self._lock:
            fact = self._resolve_fact(fact_id, "validate_fact")

            # Use cached result if the fact hasn't been mutated since last check
            if fact_id in self._validation_cache:
                return self._validation_cache[fact_id]

            result = self._run_validation(fact)
            self._validation_cache[fact_id] = result
            return result

    # ─────────────────────────────────────────────────────────────────────────
    # INTEGRITY
    # ─────────────────────────────────────────────────────────────────────────

    def find_duplicate_facts(self) -> list[list[Fact]]:
        self._require_initialized("find_duplicate_facts")

        with self._lock:
            # Group by fingerprint
            fp_groups: dict[str, list[Fact]] = defaultdict(list)
            for fact in self._store.values():
                if not fact.status.is_terminal:
                    fp_groups[fact.fingerprint].append(fact)

            return [
                group for group in fp_groups.values()
                if len(group) > 1
            ]

    def fact_exists(self, fact_id: str) -> bool:
        self._require_initialized("fact_exists")
        with self._lock:
            return fact_id in self._store

    def get_fact_count(self, *, active_only: bool = False) -> int:
        self._require_initialized("get_fact_count")
        with self._lock:
            if not active_only:
                return len(self._store)
            return sum(
                1 for f in self._store.values()
                if f.status in (KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE)
            )

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT
    # ─────────────────────────────────────────────────────────────────────────

    def audit_report(self) -> dict[str, Any]:
        self._require_initialized("audit_report")

        with self._lock:
            all_facts = list(self._store.values())
            total = len(all_facts)

            status_counts: dict[str, int] = defaultdict(int)
            type_counts: dict[str, int] = defaultdict(int)
            domain_counts: dict[str, int] = defaultdict(int)
            confidence_sum = 0.0
            low_confidence_ids: list[str] = []
            stale_ids: list[str] = []

            for fact in all_facts:
                status_counts[fact.status.value] += 1
                type_counts[fact.fact_type.value] += 1
                for did in fact.domain_ids:
                    domain_counts[did] += 1
                confidence_sum += fact.metadata.confidence_score
                if fact.metadata.confidence_score < 0.40:
                    low_confidence_ids.append(fact.id)
                if fact.metadata.is_stale:
                    stale_ids.append(fact.id)

            avg_confidence = confidence_sum / total if total > 0 else 0.0
            duplicate_groups = self.find_duplicate_facts()

            return {
                "engine": "FactEngine",
                "version": _ENGINE_VERSION,
                "total_facts": total,
                "active_facts": status_counts.get(KnowledgeStatus.ACTIVE.value, 0)
                    + status_counts.get(KnowledgeStatus.VALIDATED.value, 0),
                "draft_facts": status_counts.get(KnowledgeStatus.DRAFT.value, 0),
                "deprecated_facts": status_counts.get(KnowledgeStatus.DEPRECATED.value, 0),
                "retracted_facts": status_counts.get(KnowledgeStatus.RETRACTED.value, 0),
                "avg_confidence": round(avg_confidence, 4),
                "facts_by_type": dict(type_counts),
                "facts_by_domain": dict(domain_counts),
                "duplicate_groups": len(duplicate_groups),
                "low_confidence_ids": low_confidence_ids,
                "stale_ids": stale_ids,
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
                1 for f in self._store.values()
                if f.status in (KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE)
            ) if self._initialized else 0

            return {
                "engine": "FactEngine",
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
                "validation_cache_size": len(self._validation_cache),
                "op_log_entries": len(self._op_log),
                "engine_version": _ENGINE_VERSION,
            })
            return report


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "FactEngine",
]