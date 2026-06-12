"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/domains.py

Concrete implementation of the LUNA Knowledge Domain Engine.

The Knowledge Domain Engine manages the top-level organizational structure
for all knowledge within LUNA.  Domains are the primary indexing namespace
for every other LUNA knowledge type (facts, concepts, skills, procedures, etc.)

Examples of domains owned by LUNA:
    - Artificial Intelligence
    - Robotics
    - Electronics
    - Mathematics
    - Control Systems
    - Business

Responsibilities:
    - Full CRUD lifecycle for KnowledgeDomain records
    - Parent/child domain hierarchy management (tree structure, cycle detection)
    - Concept, skill, and procedure assignment to domains
    - Structural validation per LUNA quality standards
    - SHA-256 fingerprint-based duplicate detection
    - DomainStructure assembly for downstream engine consumers
    - Free-text search and indexing
    - Immutable audit log and engine diagnostics

Implementation notes:
    - In-memory v1 store backed by dict[str, KnowledgeDomain]
    - Thread-safe via threading.RLock
    - Lifecycle-gated: every public method raises LunaNotInitializedError
      before initialize() is called or after shutdown()
    - Soft-delete only: deleted domains transition to RETRACTED
    - Parent-child hierarchy tracks parent_domain_id on child records
      and sub_domain_ids on parent records for bidirectional traversal

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
    DomainNotFoundError,
    DomainValidationError,
    DuplicateDomainError,
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractKnowledgeDomainEngine
from subsystems.luna.models import (
    ConfidenceLevel,
    DomainStructure,
    KnowledgeDifficulty,
    KnowledgeDomain,
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
# ENGINE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE_VERSION: str = "5.0.0"
_MAX_NAME_LENGTH: int = 256
_MAX_DESCRIPTION_LENGTH: int = 8_192
_MAX_HIERARCHY_DEPTH: int = 16


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION LOG ENTRY
# ─────────────────────────────────────────────────────────────────────────────

class _DomainOpEntry:
    """Immutable record of a single mutation applied to the domain store."""

    __slots__ = ("op", "domain_id", "timestamp", "notes")

    def __init__(self, op: str, domain_id: str, notes: str = "") -> None:
        self.op: str = op          # "create" | "update" | "delete" |
                                   # "assign_concept" | "unassign_concept" |
                                   # "assign_skill" | "unassign_skill" |
                                   # "assign_procedure" | "unassign_procedure" |
                                   # "add_subdomain" | "remove_subdomain"
        self.domain_id: str = domain_id
        self.timestamp: datetime = _utcnow()
        self.notes: str = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "domain_id": self.domain_id,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE DOMAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeDomainEngine(AbstractKnowledgeDomainEngine):
    """
    In-memory, thread-safe implementation of the LUNA Knowledge Domain Engine (v1).

    Data stores:
        _store               — dict[domain_id, KnowledgeDomain]
        _structure_store     — dict[domain_id, DomainStructure]

    Secondary indexes:
        _fingerprint_index   — fingerprint → domain_id
        _parent_index        — parent_domain_id → set[child_domain_id]
        _name_index          — normalised_name → domain_id (fast name lookup)

    Hierarchy semantics:
        Each KnowledgeDomain optionally has a parent_domain_id.
        The engine maintains parent→child adjacency in _parent_index, and
        enforces acyclicity via iterative DFS before committing parent changes.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()

        # Primary stores
        self._store: dict[str, KnowledgeDomain] = {}
        self._structure_store: dict[str, DomainStructure] = {}

        # Secondary indexes
        self._fingerprint_index: dict[str, str] = {}
        self._name_index: dict[str, str] = {}  # normalised_name → domain_id
        self._parent_index: dict[str, set[str]] = defaultdict(set)
        # _parent_index[parent_id] = {child_id, ...}

        # Validation result cache
        self._validation_cache: dict[str, KnowledgeValidationResult] = {}

        # Operation audit log
        self._op_log: list[_DomainOpEntry] = []

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
        Prepare the Knowledge Domain Engine for operation.

        Idempotent: repeated calls after first initialization are no-ops.

        Raises:
            LunaLifecycleError: If internal setup fails.
        """
        with self._lock:
            if self._initialized:
                return
            try:
                self._store.clear()
                self._structure_store.clear()
                self._fingerprint_index.clear()
                self._name_index.clear()
                self._parent_index.clear()
                self._validation_cache.clear()
                self._op_log.clear()
                self._mutation_count = 0
                self._duplicate_checks = 0
                self._last_mutation_at = None
                self._initialized = True
                logger.info(
                    "KnowledgeDomainEngine initialized (version=%s)", _ENGINE_VERSION
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="KnowledgeDomainEngine",
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
                    "KnowledgeDomainEngine shutdown (domains=%d, mutations=%d)",
                    len(self._store),
                    self._mutation_count,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="KnowledgeDomainEngine",
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

    def _index_domain(self, domain: KnowledgeDomain) -> None:
        """Register a domain in all secondary indexes."""
        self._fingerprint_index[domain.fingerprint] = domain.id
        self._name_index[domain.name.lower().strip()] = domain.id
        if domain.parent_domain_id:
            self._parent_index[domain.parent_domain_id].add(domain.id)

    def _deindex_domain(self, domain: KnowledgeDomain) -> None:
        """Remove a domain from all secondary indexes."""
        self._fingerprint_index.pop(domain.fingerprint, None)
        self._name_index.pop(domain.name.lower().strip(), None)
        if domain.parent_domain_id:
            self._parent_index[domain.parent_domain_id].discard(domain.id)

    def _reindex_domain(self, old: KnowledgeDomain, new: KnowledgeDomain) -> None:
        self._deindex_domain(old)
        self._index_domain(new)

    def _record_op(self, op: str, domain_id: str, notes: str = "") -> None:
        """Append an operation log entry and update mutation counters."""
        self._op_log.append(
            _DomainOpEntry(op=op, domain_id=domain_id, notes=notes)
        )
        self._mutation_count += 1
        self._last_mutation_at = _utcnow()
        self._validation_cache.pop(domain_id, None)

    def _resolve_domain(self, domain_id: str, operation: str) -> KnowledgeDomain:
        """Return the domain or raise DomainNotFoundError."""
        domain = self._store.get(domain_id)
        if domain is None:
            raise DomainNotFoundError(
                domain_id=domain_id,
                context={"operation": operation},
            )
        return domain

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — DUPLICATE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _check_duplicate(self, domain: KnowledgeDomain) -> None:
        """Raise DuplicateDomainError if a live record with same fingerprint exists."""
        self._duplicate_checks += 1
        existing_id = self._fingerprint_index.get(domain.fingerprint)
        if existing_id is None:
            return
        existing = self._store.get(existing_id)
        if existing is None:
            return
        if not existing.status.is_terminal:
            raise DuplicateDomainError(
                domain_id=domain.id,
                existing_id=existing_id,
                context={"fingerprint": domain.fingerprint},
            )

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — CYCLE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _would_create_hierarchy_cycle(
        self, domain_id: str, proposed_parent_id: str
    ) -> bool:
        """
        Return True if making proposed_parent_id the parent of domain_id would
        create a cycle in the domain hierarchy.

        A cycle exists if domain_id is already an ancestor of proposed_parent_id.
        Uses iterative DFS on the parent chain of proposed_parent_id.
        """
        visited: set[str] = set()
        current_id: Optional[str] = proposed_parent_id
        while current_id is not None:
            if current_id == domain_id:
                return True
            if current_id in visited:
                break
            visited.add(current_id)
            current = self._store.get(current_id)
            if current is None:
                break
            current_id = current.parent_domain_id
        return False

    def _hierarchy_depth(self, domain_id: str) -> int:
        """Return the depth of domain_id in the hierarchy (root = 0)."""
        depth = 0
        current_id: Optional[str] = domain_id
        visited: set[str] = set()
        while current_id is not None:
            if current_id in visited:
                break
            visited.add(current_id)
            domain = self._store.get(current_id)
            if domain is None or domain.parent_domain_id is None:
                break
            depth += 1
            current_id = domain.parent_domain_id
        return depth

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    def _run_validation(
        self, domain: KnowledgeDomain
    ) -> KnowledgeValidationResult:
        """Execute all validation rules against a KnowledgeDomain instance."""
        issues: list[ValidationIssue] = []

        # Rule: name must be non-empty
        if not domain.name or not domain.name.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.CRITICAL,
                message="Domain name must not be empty.",
                field="name",
            ))

        # Rule: name length
        if len(domain.name) > _MAX_NAME_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"Domain name exceeds {_MAX_NAME_LENGTH} characters.",
                field="name",
            ))

        # Rule: description must be non-empty
        if not domain.description or not domain.description.strip():
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.CRITICAL,
                message="Domain description must not be empty.",
                field="description",
            ))

        # Rule: description length
        if len(domain.description) > _MAX_DESCRIPTION_LENGTH:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.WARNING,
                message=(
                    f"Domain description exceeds {_MAX_DESCRIPTION_LENGTH} characters."
                ),
                field="description",
            ))

        # Rule: confidence_score must be in [0.0, 1.0]
        score = domain.metadata.confidence_score
        if not (0.0 <= score <= 1.0):
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"confidence_score {score!r} is outside [0.0, 1.0].",
                field="metadata.confidence_score",
            ))

        # Rule: knowledge_type must be DOMAIN
        if domain.knowledge_type != KnowledgeType.DOMAIN:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.FORMAT_ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    f"knowledge_type must be KnowledgeType.DOMAIN, "
                    f"got {domain.knowledge_type!r}."
                ),
                field="knowledge_type",
            ))

        # Rule: no self-referential parent
        if domain.parent_domain_id == domain.id:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                severity=ValidationSeverity.CRITICAL,
                message="Domain cannot be its own parent.",
                field="parent_domain_id",
            ))

        # Rule: no self-referential sub-domains
        if domain.id in domain.sub_domain_ids:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                severity=ValidationSeverity.ERROR,
                message="Domain references itself in sub_domain_ids.",
                field="sub_domain_ids",
            ))

        # Rule: parent must exist if specified
        if (
            domain.parent_domain_id is not None
            and domain.parent_domain_id not in self._store
        ):
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                severity=ValidationSeverity.ERROR,
                message=(
                    f"parent_domain_id '{domain.parent_domain_id}' does not "
                    f"exist in the domain store."
                ),
                field="parent_domain_id",
            ))

        # Rule: hierarchy depth limit
        if domain.parent_domain_id is not None:
            depth = self._hierarchy_depth(domain.id)
            if depth > _MAX_HIERARCHY_DEPTH:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.FORMAT_ERROR,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Domain hierarchy depth {depth} exceeds recommended "
                        f"maximum of {_MAX_HIERARCHY_DEPTH}."
                    ),
                    field="parent_domain_id",
                ))

        # Rule: low-confidence warning
        if domain.metadata.confidence_score < 0.40:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.LOW_CONFIDENCE,
                severity=ValidationSeverity.WARNING,
                message=(
                    f"Confidence score {domain.metadata.confidence_score:.2f} is low "
                    f"({ConfidenceLevel.from_score(domain.metadata.confidence_score).value})."
                ),
                field="metadata.confidence_score",
            ))

        # Rule: unknown source type warning
        if domain.metadata.source_type == KnowledgeSourceType.UNKNOWN:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.UNVERIFIED_CLAIM,
                severity=ValidationSeverity.WARNING,
                message=(
                    "Domain source type is UNKNOWN; "
                    "consider providing a reliable source."
                ),
                field="metadata.source_type",
            ))

        return KnowledgeValidationResult.create(
            knowledge_id=domain.id,
            knowledge_name=domain.name,
            issues=issues,
            validator_version=_ENGINE_VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — DOMAIN STRUCTURE
    # ─────────────────────────────────────────────────────────────────────────

    def _get_or_create_structure(self, domain_id: str) -> DomainStructure:
        """Return the DomainStructure for domain_id, creating it if absent."""
        if domain_id not in self._structure_store:
            domain = self._store.get(domain_id)
            name = domain.name if domain else domain_id
            self._structure_store[domain_id] = DomainStructure.create(
                domain_id=domain_id,
                domain_name=name,
            )
        return self._structure_store[domain_id]

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD — DOMAIN
    # ─────────────────────────────────────────────────────────────────────────

    def create_domain(
        self,
        name: str,
        description: str,
        difficulty: KnowledgeDifficulty,
        metadata: KnowledgeMetadata,
        parent_domain_ids: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> KnowledgeDomain:
        """
        Create and register a new knowledge domain owned by LUNA.

        If parent_domain_ids is provided, the first entry is used as the
        canonical parent (KnowledgeDomain.parent_domain_id) and the domain is
        added to the parent's sub_domain_ids list.

        Args:
            name:              Short canonical name (e.g. "Robotics").
            description:       Human-readable description.
            difficulty:        KnowledgeDifficulty tier.
            metadata:          Provenance, confidence, and versioning metadata.
            parent_domain_ids: Optional list of parent domain IDs. First entry is
                               treated as the direct parent in the hierarchy.
            aliases:           Alternate names or abbreviations.
            notes:             Free-text notes.

        Returns:
            The newly created and stored KnowledgeDomain record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DuplicateDomainError:    An identical domain already exists.
            DomainValidationError:   Provided data fails validation.
            DomainNotFoundError:     A specified parent domain ID does not exist.
        """
        self._require_initialized("create_domain")

        with self._lock:
            # Determine parent: use first entry of parent_domain_ids
            parent_id: Optional[str] = None
            if parent_domain_ids:
                parent_id = parent_domain_ids[0]
                if parent_id not in self._store:
                    raise DomainNotFoundError(
                        domain_id=parent_id,
                        context={
                            "operation": "create_domain",
                            "reason": "parent domain not found",
                        },
                    )

            is_root = parent_id is None

            candidate = KnowledgeDomain.create(
                name=name,
                description=description,
                difficulty=difficulty,
                metadata=metadata,
                parent_domain_id=parent_id,
                is_root_domain=is_root,
                aliases=aliases,
                notes=notes,
            )

            # Cycle detection: if parent specified, confirm no cycle
            if parent_id is not None:
                if self._would_create_hierarchy_cycle(candidate.id, parent_id):
                    raise DomainValidationError(
                        domain_id=candidate.id,
                        violations=[
                            f"Setting parent '{parent_id}' would create a "
                            f"cycle in the domain hierarchy."
                        ],
                    )

            # Structural validation
            result = self._run_validation(candidate)
            if result.has_blocking_issues:
                violations = [
                    i.message for i in result.issues if i.severity.is_blocking()
                ]
                raise DomainValidationError(
                    domain_id=candidate.id,
                    violations=violations,
                )

            # Duplicate detection
            self._check_duplicate(candidate)

            # Persist
            self._store[candidate.id] = candidate
            self._index_domain(candidate)

            # Register as sub-domain on parent
            if parent_id is not None:
                parent = self._store[parent_id]
                if candidate.id not in parent.sub_domain_ids:
                    updated_parent = KnowledgeDomain(
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
                        parent_domain_id=parent.parent_domain_id,
                        sub_domain_ids=list(parent.sub_domain_ids) + [candidate.id],
                        core_concept_ids=list(parent.core_concept_ids),
                        core_skill_ids=list(parent.core_skill_ids),
                        core_fact_ids=list(parent.core_fact_ids),
                        standard_references=list(parent.standard_references),
                        is_root_domain=parent.is_root_domain,
                    )
                    self._reindex_domain(parent, updated_parent)
                    self._store[parent_id] = updated_parent
                    # Update parent index
                    self._parent_index[parent_id].add(candidate.id)

                    # Reflect in parent's structure
                    ps = self._get_or_create_structure(parent_id)
                    if candidate.id not in ps.sub_domain_ids:
                        ps.sub_domain_ids.append(candidate.id)
                        ps.recalculate_total()

            # Create structure for new domain
            self._get_or_create_structure(candidate.id)

            self._record_op("create", candidate.id, notes=f"created: {name!r}")
            logger.debug(
                "KnowledgeDomainEngine.create_domain id=%s name=%r parent=%s",
                candidate.short_id,
                name,
                parent_id,
            )
            return candidate

    def update_domain(
        self,
        domain_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        parent_domain_ids: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
    ) -> KnowledgeDomain:
        """
        Apply a partial update to an existing domain record.

        Only keyword-supplied fields are modified; omitted fields retain their
        current values.  Version is incremented automatically.

        When parent_domain_ids is supplied, the first entry becomes the new
        parent.  The old parent's sub_domain_ids list is cleaned up and the new
        parent's list is updated.

        Args:
            domain_id: ID of the domain to update.
            **fields:  Any subset of KnowledgeDomain fields to overwrite.

        Returns:
            The updated KnowledgeDomain record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
            DomainValidationError:   Updated state fails validation or attempts
                                     to modify a terminal record.
            DuplicateDomainError:    Name change produces a fingerprint collision.
        """
        self._require_initialized("update_domain")

        with self._lock:
            existing = self._resolve_domain(domain_id, "update_domain")

            if existing.status.is_terminal:
                raise DomainValidationError(
                    domain_id=domain_id,
                    violations=[
                        f"Cannot update a domain with terminal status "
                        f"'{existing.status.value}'."
                    ],
                )

            # Determine new parent
            new_parent_id: Optional[str]
            if parent_domain_ids is not None:
                new_parent_id = parent_domain_ids[0] if parent_domain_ids else None
            else:
                new_parent_id = existing.parent_domain_id

            # Validate new parent exists
            if new_parent_id is not None and new_parent_id not in self._store:
                raise DomainNotFoundError(
                    domain_id=new_parent_id,
                    context={
                        "operation": "update_domain",
                        "reason": "parent domain not found",
                    },
                )

            # Cycle detection for parent change
            if new_parent_id is not None and new_parent_id != existing.parent_domain_id:
                if self._would_create_hierarchy_cycle(domain_id, new_parent_id):
                    raise DomainValidationError(
                        domain_id=domain_id,
                        violations=[
                            f"Setting parent '{new_parent_id}' would create a "
                            f"cycle in the domain hierarchy."
                        ],
                    )

            new_is_root = new_parent_id is None

            updated = KnowledgeDomain(
                id=existing.id,
                knowledge_type=existing.knowledge_type,
                name=name if name is not None else existing.name,
                description=(
                    description if description is not None else existing.description
                ),
                status=status if status is not None else existing.status,
                difficulty=(
                    difficulty if difficulty is not None else existing.difficulty
                ),
                domain_ids=list(existing.domain_ids),
                metadata=(
                    metadata.bump_version()
                    if metadata is not None
                    else existing.metadata.bump_version()
                ),
                aliases=(
                    list(aliases) if aliases is not None else list(existing.aliases)
                ),
                related_ids=list(existing.related_ids),
                notes=notes if notes is not None else existing.notes,
                parent_domain_id=new_parent_id,
                sub_domain_ids=list(existing.sub_domain_ids),
                core_concept_ids=list(existing.core_concept_ids),
                core_skill_ids=list(existing.core_skill_ids),
                core_fact_ids=list(existing.core_fact_ids),
                standard_references=list(existing.standard_references),
                is_root_domain=new_is_root,
            )

            # Structural validation
            result = self._run_validation(updated)
            if result.has_blocking_issues:
                violations = [
                    i.message for i in result.issues if i.severity.is_blocking()
                ]
                raise DomainValidationError(domain_id=domain_id, violations=violations)

            # Duplicate fingerprint check (only if identity-affecting fields changed)
            if name is not None or description is not None:
                fp = updated.fingerprint
                existing_fp_id = self._fingerprint_index.get(fp)
                if existing_fp_id is not None and existing_fp_id != domain_id:
                    other = self._store.get(existing_fp_id)
                    if other and not other.status.is_terminal:
                        raise DuplicateDomainError(
                            domain_id=domain_id,
                            existing_id=existing_fp_id,
                            context={"fingerprint": fp},
                        )

            # Handle parent change: update old and new parent sub_domain_ids
            old_parent_id = existing.parent_domain_id
            if new_parent_id != old_parent_id:
                # Remove from old parent
                if old_parent_id is not None and old_parent_id in self._store:
                    old_parent = self._store[old_parent_id]
                    if domain_id in old_parent.sub_domain_ids:
                        new_subs = [
                            s for s in old_parent.sub_domain_ids if s != domain_id
                        ]
                        refreshed_old_parent = KnowledgeDomain(
                            id=old_parent.id,
                            knowledge_type=old_parent.knowledge_type,
                            name=old_parent.name,
                            description=old_parent.description,
                            status=old_parent.status,
                            difficulty=old_parent.difficulty,
                            domain_ids=list(old_parent.domain_ids),
                            metadata=old_parent.metadata.bump_version(),
                            aliases=list(old_parent.aliases),
                            related_ids=list(old_parent.related_ids),
                            notes=old_parent.notes,
                            parent_domain_id=old_parent.parent_domain_id,
                            sub_domain_ids=new_subs,
                            core_concept_ids=list(old_parent.core_concept_ids),
                            core_skill_ids=list(old_parent.core_skill_ids),
                            core_fact_ids=list(old_parent.core_fact_ids),
                            standard_references=list(old_parent.standard_references),
                            is_root_domain=old_parent.is_root_domain,
                        )
                        self._reindex_domain(old_parent, refreshed_old_parent)
                        self._store[old_parent_id] = refreshed_old_parent
                    self._parent_index[old_parent_id].discard(domain_id)

                # Add to new parent
                if new_parent_id is not None and new_parent_id in self._store:
                    new_parent = self._store[new_parent_id]
                    if domain_id not in new_parent.sub_domain_ids:
                        refreshed_new_parent = KnowledgeDomain(
                            id=new_parent.id,
                            knowledge_type=new_parent.knowledge_type,
                            name=new_parent.name,
                            description=new_parent.description,
                            status=new_parent.status,
                            difficulty=new_parent.difficulty,
                            domain_ids=list(new_parent.domain_ids),
                            metadata=new_parent.metadata.bump_version(),
                            aliases=list(new_parent.aliases),
                            related_ids=list(new_parent.related_ids),
                            notes=new_parent.notes,
                            parent_domain_id=new_parent.parent_domain_id,
                            sub_domain_ids=list(new_parent.sub_domain_ids)
                            + [domain_id],
                            core_concept_ids=list(new_parent.core_concept_ids),
                            core_skill_ids=list(new_parent.core_skill_ids),
                            core_fact_ids=list(new_parent.core_fact_ids),
                            standard_references=list(
                                new_parent.standard_references
                            ),
                            is_root_domain=new_parent.is_root_domain,
                        )
                        self._reindex_domain(new_parent, refreshed_new_parent)
                        self._store[new_parent_id] = refreshed_new_parent
                    self._parent_index[new_parent_id].add(domain_id)

            self._reindex_domain(existing, updated)
            self._store[domain_id] = updated
            self._record_op("update", domain_id, notes=f"updated: {updated.name!r}")

            logger.debug(
                "KnowledgeDomainEngine.update_domain id=%s", domain_id[:8]
            )
            return updated

    def delete_domain(
        self, domain_id: str, *, reason: str = ""
    ) -> KnowledgeDomain:
        """
        Soft-delete a domain by transitioning its status to RETRACTED.

        Domains are never physically removed; deletion is a status transition
        so that audit history and downstream references remain coherent.

        Args:
            domain_id: ID of the domain to retract.
            reason:    Human-readable reason stored in notes.

        Returns:
            The retracted KnowledgeDomain record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
        """
        self._require_initialized("delete_domain")

        with self._lock:
            existing = self._resolve_domain(domain_id, "delete_domain")

            if existing.status == KnowledgeStatus.RETRACTED:
                return existing  # Idempotent

            deletion_note = (
                f"[RETRACTED {_utcnow().isoformat()}]"
                + (f" Reason: {reason}" if reason else "")
            )

            retracted = KnowledgeDomain(
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
                parent_domain_id=existing.parent_domain_id,
                sub_domain_ids=list(existing.sub_domain_ids),
                core_concept_ids=list(existing.core_concept_ids),
                core_skill_ids=list(existing.core_skill_ids),
                core_fact_ids=list(existing.core_fact_ids),
                standard_references=list(existing.standard_references),
                is_root_domain=existing.is_root_domain,
            )

            self._deindex_domain(existing)
            self._store[domain_id] = retracted
            self._record_op("delete", domain_id, notes=reason)

            logger.debug(
                "KnowledgeDomainEngine.delete_domain id=%s reason=%r",
                domain_id[:8],
                reason,
            )
            return retracted

    def retrieve_domain(self, domain_id: str) -> KnowledgeDomain:
        """
        Fetch a single domain by its unique ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
        """
        self._require_initialized("retrieve_domain")
        with self._lock:
            return self._resolve_domain(domain_id, "retrieve_domain")

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    def search_domains(
        self,
        query: str,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 50,
    ) -> list[KnowledgeDomain]:
        """
        Free-text search across domain names, descriptions, and aliases.

        Args:
            query:         Case-insensitive substring search string.
            status_filter: Restrict to these KnowledgeStatus values.
            limit:         Maximum number of results.

        Returns:
            Matching domain records sorted by name ascending.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("search_domains")

        with self._lock:
            q = query.lower().strip()
            results: list[KnowledgeDomain] = []

            for domain in self._store.values():
                # Status filter
                if status_filter:
                    if domain.status not in status_filter:
                        continue
                else:
                    if not domain.status.is_usable:
                        continue

                # Text search
                if q:
                    searchable = " ".join([
                        domain.name,
                        domain.description,
                        " ".join(domain.aliases),
                    ]).lower()
                    if q not in searchable:
                        continue

                results.append(domain)

            results.sort(key=lambda d: d.name)
            return results[:limit]

    def get_all_domains(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
    ) -> list[KnowledgeDomain]:
        """
        Return every domain record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_all_domains")

        with self._lock:
            if status_filter:
                domains = [
                    d for d in self._store.values() if d.status in status_filter
                ]
            else:
                domains = list(self._store.values())
            domains.sort(key=lambda d: d.name)
            return domains

    def get_root_domains(self) -> list[KnowledgeDomain]:
        """
        Return top-level domains (no parent_domain_id).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_root_domains")

        with self._lock:
            roots = [
                d
                for d in self._store.values()
                if d.parent_domain_id is None and not d.status.is_terminal
            ]
            roots.sort(key=lambda d: d.name)
            return roots

    def get_child_domains(self, domain_id: str) -> list[KnowledgeDomain]:
        """
        Return direct child domains of the given domain.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
        """
        self._require_initialized("get_child_domains")

        with self._lock:
            self._resolve_domain(domain_id, "get_child_domains")
            child_ids = self._parent_index.get(domain_id, set())
            children = [
                self._store[cid]
                for cid in child_ids
                if cid in self._store and self._store[cid].status.is_usable
            ]
            children.sort(key=lambda d: d.name)
            return children

    def get_ancestor_domains(self, domain_id: str) -> list[KnowledgeDomain]:
        """
        Return all ancestor domains from parent to root (exclusive of self).

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
        """
        self._require_initialized("get_ancestor_domains")

        with self._lock:
            self._resolve_domain(domain_id, "get_ancestor_domains")
            ancestors: list[KnowledgeDomain] = []
            visited: set[str] = set()
            current_id: Optional[str] = self._store[domain_id].parent_domain_id
            while current_id is not None:
                if current_id in visited:
                    break
                visited.add(current_id)
                ancestor = self._store.get(current_id)
                if ancestor is None:
                    break
                ancestors.append(ancestor)
                current_id = ancestor.parent_domain_id
            return ancestors

    def get_descendant_domains(self, domain_id: str) -> list[KnowledgeDomain]:
        """
        Return all descendant domains (recursive sub-domains) of a given domain.

        Result is ordered breadth-first.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
        """
        self._require_initialized("get_descendant_domains")

        with self._lock:
            self._resolve_domain(domain_id, "get_descendant_domains")
            result: list[KnowledgeDomain] = []
            visited: set[str] = set()
            queue: deque[str] = deque(self._parent_index.get(domain_id, set()))
            while queue:
                cid = queue.popleft()
                if cid in visited:
                    continue
                visited.add(cid)
                domain = self._store.get(cid)
                if domain is not None:
                    result.append(domain)
                    for grandchild_id in self._parent_index.get(cid, set()):
                        if grandchild_id not in visited:
                            queue.append(grandchild_id)
            return result

    # ─────────────────────────────────────────────────────────────────────────
    # CONCEPT ASSIGNMENT
    # ─────────────────────────────────────────────────────────────────────────

    def assign_concept(self, domain_id: str, concept_id: str) -> KnowledgeDomain:
        """
        Add concept_id to domain's core_concept_ids list.

        Args:
            domain_id:  Target domain.
            concept_id: Concept to assign.

        Returns:
            The updated KnowledgeDomain record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
            DomainValidationError:   Concept already assigned.
        """
        self._require_initialized("assign_concept")

        with self._lock:
            domain = self._resolve_domain(domain_id, "assign_concept")

            if concept_id in domain.core_concept_ids:
                raise DomainValidationError(
                    domain_id=domain_id,
                    violations=[
                        f"Concept '{concept_id}' is already assigned to "
                        f"domain '{domain_id}'."
                    ],
                )

            updated = KnowledgeDomain(
                id=domain.id,
                knowledge_type=domain.knowledge_type,
                name=domain.name,
                description=domain.description,
                status=domain.status,
                difficulty=domain.difficulty,
                domain_ids=list(domain.domain_ids),
                metadata=domain.metadata.bump_version(),
                aliases=list(domain.aliases),
                related_ids=list(domain.related_ids),
                notes=domain.notes,
                parent_domain_id=domain.parent_domain_id,
                sub_domain_ids=list(domain.sub_domain_ids),
                core_concept_ids=list(domain.core_concept_ids) + [concept_id],
                core_skill_ids=list(domain.core_skill_ids),
                core_fact_ids=list(domain.core_fact_ids),
                standard_references=list(domain.standard_references),
                is_root_domain=domain.is_root_domain,
            )
            self._reindex_domain(domain, updated)
            self._store[domain_id] = updated

            structure = self._get_or_create_structure(domain_id)
            if concept_id not in structure.concept_ids:
                structure.concept_ids.append(concept_id)
                structure.recalculate_total()

            self._record_op(
                "assign_concept",
                domain_id,
                notes=f"concept={concept_id!r}",
            )
            return updated

    def unassign_concept(self, domain_id: str, concept_id: str) -> KnowledgeDomain:
        """
        Remove concept_id from domain's core_concept_ids list.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
            DomainValidationError:   Concept not assigned to this domain.
        """
        self._require_initialized("unassign_concept")

        with self._lock:
            domain = self._resolve_domain(domain_id, "unassign_concept")

            if concept_id not in domain.core_concept_ids:
                raise DomainValidationError(
                    domain_id=domain_id,
                    violations=[
                        f"Concept '{concept_id}' is not assigned to "
                        f"domain '{domain_id}'."
                    ],
                )

            updated = KnowledgeDomain(
                id=domain.id,
                knowledge_type=domain.knowledge_type,
                name=domain.name,
                description=domain.description,
                status=domain.status,
                difficulty=domain.difficulty,
                domain_ids=list(domain.domain_ids),
                metadata=domain.metadata.bump_version(),
                aliases=list(domain.aliases),
                related_ids=list(domain.related_ids),
                notes=domain.notes,
                parent_domain_id=domain.parent_domain_id,
                sub_domain_ids=list(domain.sub_domain_ids),
                core_concept_ids=[
                    c for c in domain.core_concept_ids if c != concept_id
                ],
                core_skill_ids=list(domain.core_skill_ids),
                core_fact_ids=list(domain.core_fact_ids),
                standard_references=list(domain.standard_references),
                is_root_domain=domain.is_root_domain,
            )
            self._reindex_domain(domain, updated)
            self._store[domain_id] = updated

            structure = self._get_or_create_structure(domain_id)
            if concept_id in structure.concept_ids:
                structure.concept_ids.remove(concept_id)
                structure.recalculate_total()

            self._record_op(
                "unassign_concept",
                domain_id,
                notes=f"concept={concept_id!r}",
            )
            return updated

    # ─────────────────────────────────────────────────────────────────────────
    # SKILL ASSIGNMENT
    # ─────────────────────────────────────────────────────────────────────────

    def assign_skill(self, domain_id: str, skill_id: str) -> KnowledgeDomain:
        """
        Add skill_id to domain's core_skill_ids list.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
            DomainValidationError:   Skill already assigned.
        """
        self._require_initialized("assign_skill")

        with self._lock:
            domain = self._resolve_domain(domain_id, "assign_skill")

            if skill_id in domain.core_skill_ids:
                raise DomainValidationError(
                    domain_id=domain_id,
                    violations=[
                        f"Skill '{skill_id}' is already assigned to "
                        f"domain '{domain_id}'."
                    ],
                )

            updated = KnowledgeDomain(
                id=domain.id,
                knowledge_type=domain.knowledge_type,
                name=domain.name,
                description=domain.description,
                status=domain.status,
                difficulty=domain.difficulty,
                domain_ids=list(domain.domain_ids),
                metadata=domain.metadata.bump_version(),
                aliases=list(domain.aliases),
                related_ids=list(domain.related_ids),
                notes=domain.notes,
                parent_domain_id=domain.parent_domain_id,
                sub_domain_ids=list(domain.sub_domain_ids),
                core_concept_ids=list(domain.core_concept_ids),
                core_skill_ids=list(domain.core_skill_ids) + [skill_id],
                core_fact_ids=list(domain.core_fact_ids),
                standard_references=list(domain.standard_references),
                is_root_domain=domain.is_root_domain,
            )
            self._reindex_domain(domain, updated)
            self._store[domain_id] = updated

            structure = self._get_or_create_structure(domain_id)
            if skill_id not in structure.skill_ids:
                structure.skill_ids.append(skill_id)
                structure.recalculate_total()

            self._record_op("assign_skill", domain_id, notes=f"skill={skill_id!r}")
            return updated

    def unassign_skill(self, domain_id: str, skill_id: str) -> KnowledgeDomain:
        """
        Remove skill_id from domain's core_skill_ids list.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
            DomainValidationError:   Skill not assigned to this domain.
        """
        self._require_initialized("unassign_skill")

        with self._lock:
            domain = self._resolve_domain(domain_id, "unassign_skill")

            if skill_id not in domain.core_skill_ids:
                raise DomainValidationError(
                    domain_id=domain_id,
                    violations=[
                        f"Skill '{skill_id}' is not assigned to "
                        f"domain '{domain_id}'."
                    ],
                )

            updated = KnowledgeDomain(
                id=domain.id,
                knowledge_type=domain.knowledge_type,
                name=domain.name,
                description=domain.description,
                status=domain.status,
                difficulty=domain.difficulty,
                domain_ids=list(domain.domain_ids),
                metadata=domain.metadata.bump_version(),
                aliases=list(domain.aliases),
                related_ids=list(domain.related_ids),
                notes=domain.notes,
                parent_domain_id=domain.parent_domain_id,
                sub_domain_ids=list(domain.sub_domain_ids),
                core_concept_ids=list(domain.core_concept_ids),
                core_skill_ids=[
                    s for s in domain.core_skill_ids if s != skill_id
                ],
                core_fact_ids=list(domain.core_fact_ids),
                standard_references=list(domain.standard_references),
                is_root_domain=domain.is_root_domain,
            )
            self._reindex_domain(domain, updated)
            self._store[domain_id] = updated

            structure = self._get_or_create_structure(domain_id)
            if skill_id in structure.skill_ids:
                structure.skill_ids.remove(skill_id)
                structure.recalculate_total()

            self._record_op(
                "unassign_skill", domain_id, notes=f"skill={skill_id!r}"
            )
            return updated

    # ─────────────────────────────────────────────────────────────────────────
    # PROCEDURE ASSIGNMENT
    # ─────────────────────────────────────────────────────────────────────────

    def assign_procedure(
        self, domain_id: str, procedure_id: str
    ) -> KnowledgeDomain:
        """
        Register a procedure under this domain's DomainStructure.

        Procedures are not stored on KnowledgeDomain.core_* lists directly;
        they are tracked in the DomainStructure.procedure_ids.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
            DomainValidationError:   Procedure already assigned.
        """
        self._require_initialized("assign_procedure")

        with self._lock:
            domain = self._resolve_domain(domain_id, "assign_procedure")

            structure = self._get_or_create_structure(domain_id)
            if procedure_id in structure.procedure_ids:
                raise DomainValidationError(
                    domain_id=domain_id,
                    violations=[
                        f"Procedure '{procedure_id}' is already assigned to "
                        f"domain '{domain_id}'."
                    ],
                )

            structure.procedure_ids.append(procedure_id)
            structure.recalculate_total()

            self._record_op(
                "assign_procedure",
                domain_id,
                notes=f"procedure={procedure_id!r}",
            )
            return domain

    def unassign_procedure(
        self, domain_id: str, procedure_id: str
    ) -> KnowledgeDomain:
        """
        Remove a procedure from this domain's DomainStructure.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
            DomainValidationError:   Procedure not assigned to this domain.
        """
        self._require_initialized("unassign_procedure")

        with self._lock:
            domain = self._resolve_domain(domain_id, "unassign_procedure")

            structure = self._get_or_create_structure(domain_id)
            if procedure_id not in structure.procedure_ids:
                raise DomainValidationError(
                    domain_id=domain_id,
                    violations=[
                        f"Procedure '{procedure_id}' is not assigned to "
                        f"domain '{domain_id}'."
                    ],
                )

            structure.procedure_ids.remove(procedure_id)
            structure.recalculate_total()

            self._record_op(
                "unassign_procedure",
                domain_id,
                notes=f"procedure={procedure_id!r}",
            )
            return domain

    # ─────────────────────────────────────────────────────────────────────────
    # DOMAIN STRUCTURE
    # ─────────────────────────────────────────────────────────────────────────

    def get_domain_structure(self, domain_id: str) -> DomainStructure:
        """
        Return the aggregated DomainStructure for a knowledge domain.

        The structure is created lazily on first access and updated
        incrementally by assign/unassign operations.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
        """
        self._require_initialized("get_domain_structure")

        with self._lock:
            self._resolve_domain(domain_id, "get_domain_structure")
            return self._get_or_create_structure(domain_id)

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    def validate_domain(self, domain_id: str) -> KnowledgeValidationResult:
        """
        Run structural and semantic validation on a domain.

        Results are cached until the next mutation on the same domain.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DomainNotFoundError:     Domain ID does not exist.
        """
        self._require_initialized("validate_domain")

        with self._lock:
            domain = self._resolve_domain(domain_id, "validate_domain")
            cached = self._validation_cache.get(domain_id)
            if cached is not None:
                return cached
            result = self._run_validation(domain)
            self._validation_cache[domain_id] = result
            return result

    # ─────────────────────────────────────────────────────────────────────────
    # INTEGRITY
    # ─────────────────────────────────────────────────────────────────────────

    def domain_exists(self, domain_id: str) -> bool:
        """
        Return True if a domain with the given ID exists (any status).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("domain_exists")
        with self._lock:
            return domain_id in self._store

    def get_domain_count(self, *, active_only: bool = False) -> int:
        """
        Return the number of domains in the store.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("get_domain_count")
        with self._lock:
            if active_only:
                return sum(
                    1 for d in self._store.values() if d.status.is_usable
                )
            return len(self._store)

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """Return a lightweight liveness/readiness summary."""
        total = len(self._store)
        return {
            "engine": "KnowledgeDomainEngine",
            "initialized": self._initialized,
            "record_count": total,
            "active_count": sum(
                1 for d in self._store.values() if d.status.is_usable
            ),
            "root_domain_count": sum(
                1
                for d in self._store.values()
                if d.parent_domain_id is None and d.status.is_usable
            ),
            "status": "healthy" if self._initialized else "offline",
        }

    def diagnostics_report(self) -> dict[str, Any]:
        """Return a full introspection snapshot suitable for operator debugging."""
        report = self.health_report()
        report.update({
            "index_size": len(self._fingerprint_index),
            "duplicate_checks": self._duplicate_checks,
            "mutation_count": self._mutation_count,
            "last_mutation_at": (
                self._last_mutation_at.isoformat()
                if self._last_mutation_at else None
            ),
            "parent_index_size": sum(len(v) for v in self._parent_index.values()),
            "structure_store_size": len(self._structure_store),
            "op_log_length": len(self._op_log),
            "engine_version": _ENGINE_VERSION,
        })
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT
    # ─────────────────────────────────────────────────────────────────────────

    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate domain store statistics.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("audit_report")

        with self._lock:
            all_domains = list(self._store.values())
            total = len(all_domains)
            active = sum(1 for d in all_domains if d.status.is_usable)
            retracted = sum(
                1 for d in all_domains if d.status == KnowledgeStatus.RETRACTED
            )
            root_count = sum(
                1 for d in all_domains if d.parent_domain_id is None
            )
            avg_confidence = (
                sum(d.metadata.confidence_score for d in all_domains) / total
                if total else 0.0
            )

            by_difficulty: dict[str, int] = defaultdict(int)
            concept_coverage: dict[str, int] = {}
            skill_coverage: dict[str, int] = {}
            for domain in all_domains:
                by_difficulty[domain.difficulty.value] += 1
                concept_coverage[domain.name] = len(domain.core_concept_ids)
                skill_coverage[domain.name] = len(domain.core_skill_ids)

            # Duplicate detection
            fp_map: dict[str, list[str]] = defaultdict(list)
            for domain in all_domains:
                if not domain.status.is_terminal:
                    fp_map[domain.fingerprint].append(domain.id)
            duplicate_groups = sum(
                1 for group in fp_map.values() if len(group) >= 2
            )

            return {
                "total_domains": total,
                "active_domains": active,
                "retracted_domains": retracted,
                "root_domains": root_count,
                "avg_confidence": round(avg_confidence, 4),
                "domains_by_difficulty": dict(by_difficulty),
                "concept_coverage": concept_coverage,
                "skill_coverage": skill_coverage,
                "duplicate_groups": duplicate_groups,
                "mutation_count": self._mutation_count,
                "generated_at": _utcnow().isoformat(),
            }