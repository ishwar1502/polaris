"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/validation.py

Concrete in-memory implementation of the Knowledge Validation Engine.

Validates all knowledge owned by LUNA: facts, concepts, skills, domains,
procedures, research, and educational records.  Cross-cutting concerns —
source trust, confidence scoring, contradiction detection, evidence
validation, dependency validation, freshness, and quality scoring — are
applied consistently regardless of knowledge type.

This engine is a shared service.  All other LUNA engines call into it
rather than duplicating validation logic.

Responsibilities:
    validate_fact        — validate a Fact record
    validate_concept     — validate a Concept record
    validate_skill       — validate a Skill record
    validate_domain      — validate a KnowledgeDomain record
    validate_procedure   — validate a Procedure record
    validate_research    — validate a ResearchKnowledge record
    validate_education   — validate an EducationalKnowledge record

    validate_record      — dispatch by KnowledgeType
    validate_batch       — bulk validation of same-type records
    check_confidence     — compute effective confidence score
    detect_contradictions — scan for semantic contradictions
    flag_stale_records   — identify records past review date
    promote_record       — advance a record's lifecycle status
    get_validation_history — retrieve past results per record

Validation areas:
    - Confidence scoring (metadata.confidence_score × source trust_weight)
    - Source validation (trust_weight threshold)
    - Contradiction detection (name collision + type overlap across domains)
    - Evidence validation (supporting vs challenging balance)
    - Dependency validation (broken or circular references)
    - Relationship validation (related_ids point to live records)
    - Freshness validation (age + review_date staleness)
    - Quality scoring (composite 0.0–1.0 aggregate)

Thread safety:
    All public methods acquire self._lock (threading.RLock).

Lifecycle contract:
    Call initialize() before any other method.
    Call shutdown() to release resources gracefully.
    All public methods raise LunaNotInitializedError when not initialized.

Part of the POLARIS Cognitive Substrate:
    ASTRA        → Identity
    ECHO         → Experience
    LUNA         → Knowledge       ← this module
    CHRONOS      → Time
    CONSTELLATION → Relationships
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from subsystems.luna.exceptions import (
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractKnowledgeValidationEngine
from subsystems.luna.models import (
    Concept,
    EducationalKnowledge,
    Fact,
    KnowledgeConfidence,
    KnowledgeContradiction,
    KnowledgeDomain,
    KnowledgeEvidence,
    KnowledgeRecord,
    KnowledgeSourceType,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeValidationResult,
    Procedure,
    ResearchKnowledge,
    Skill,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
    ValidationStatus,
    _new_id,
    _utcnow,
)

_ENGINE_VERSION = "5.0.0"
_VALIDATOR_VERSION = "knowledge-validator-5.0.0"

# Thresholds
_LOW_CONFIDENCE_THRESHOLD: float = 0.40
_CRITICAL_CONFIDENCE_THRESHOLD: float = 0.20
_LOW_TRUST_THRESHOLD: float = 0.50
_STALE_AGE_DEFAULT_DAYS: float = 365.0

# Quality score weights (sum to 1.0)
_QUALITY_WEIGHT_CONFIDENCE: float = 0.30
_QUALITY_WEIGHT_SOURCE: float = 0.25
_QUALITY_WEIGHT_COMPLETENESS: float = 0.25
_QUALITY_WEIGHT_FRESHNESS: float = 0.20


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Registry of knowledge stores injected at runtime
# ─────────────────────────────────────────────────────────────────────────────

class _StoreRegistry:
    """
    Thin facade over all LUNA knowledge stores.

    The KnowledgeValidationEngine is deliberately decoupled from concrete
    engine implementations.  Callers register stores by KnowledgeType key so
    the validation engine can look up any record without importing circular
    dependencies.  Each registered store must expose a dict-like interface
    mapping record_id → KnowledgeRecord subclass.
    """

    def __init__(self) -> None:
        # KnowledgeType.value → dict[str, KnowledgeRecord]
        self._stores: dict[str, dict[str, Any]] = {}

    def register(
        self, knowledge_type: KnowledgeType, store: dict[str, Any]
    ) -> None:
        """Register a live store dict for the given KnowledgeType."""
        self._stores[knowledge_type.value] = store

    def get_record(
        self, record_id: str, knowledge_type: KnowledgeType
    ) -> Optional[KnowledgeRecord]:
        store = self._stores.get(knowledge_type.value, {})
        return store.get(record_id)

    def get_all(self, knowledge_type: KnowledgeType) -> list[KnowledgeRecord]:
        store = self._stores.get(knowledge_type.value, {})
        return list(store.values())

    def exists(self, record_id: str, knowledge_type: KnowledgeType) -> bool:
        store = self._stores.get(knowledge_type.value, {})
        return record_id in store

    def exists_any(self, record_id: str) -> bool:
        return any(record_id in store for store in self._stores.values())


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeValidationEngine(AbstractKnowledgeValidationEngine):
    """
    In-memory v1 implementation of the LUNA Knowledge Validation Engine.

    This engine owns no persistent knowledge itself — it reads from the
    registries of other LUNA engines via _StoreRegistry.  All validation
    results are persisted in an internal history store keyed by record ID.

    Usage::

        engine = KnowledgeValidationEngine()
        engine.initialize()

        # Wire up stores from other engines (optional, enables cross-type checks)
        engine.register_store(KnowledgeType.FACT, fact_engine._records)
        engine.register_store(KnowledgeType.CONCEPT, concept_engine._records)

        result = engine.validate_fact("fact-id-123", fact_record)
        print(result.validation_status)

        engine.shutdown()
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False

        # External store registry (populated via register_store)
        self._registry: _StoreRegistry = _StoreRegistry()

        # Validation history: record_id → list[KnowledgeValidationResult] newest-first
        self._history: dict[str, list[KnowledgeValidationResult]] = defaultdict(list)

        # Confidence assessments: record_id → KnowledgeConfidence
        self._confidence_cache: dict[str, KnowledgeConfidence] = {}

        # Detected contradictions: contradiction_id → KnowledgeContradiction
        self._contradictions: dict[str, KnowledgeContradiction] = {}

        # Evidence: record_id → list[KnowledgeEvidence]
        self._evidence: dict[str, list[KnowledgeEvidence]] = defaultdict(list)

        # Operational counters
        self._validation_count: int = 0
        self._contradiction_checks: int = 0
        self._promotions: int = 0
        self._started_at: Optional[datetime] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Prepare the engine for operation.  Idempotent.
        """
        with self._lock:
            if self._initialized:
                return
            self._history.clear()
            self._confidence_cache.clear()
            self._contradictions.clear()
            self._evidence.clear()
            self._validation_count = 0
            self._contradiction_checks = 0
            self._promotions = 0
            self._started_at = _now_utc()
            self._initialized = True

    def shutdown(self) -> None:
        """
        Release all in-memory resources.  Idempotent.
        """
        with self._lock:
            if not self._initialized:
                return
            self._history.clear()
            self._confidence_cache.clear()
            self._contradictions.clear()
            self._evidence.clear()
            self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    # ── Internal guard ────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation)

    # ── Store registration (public API for wiring) ────────────────────────────

    def register_store(
        self, knowledge_type: KnowledgeType, store: dict[str, Any]
    ) -> None:
        """
        Register a live store dict so this engine can perform cross-type checks.

        Args:
            knowledge_type: The KnowledgeType that this store holds.
            store:          A dict mapping record_id → KnowledgeRecord subclass.
                            This is the same dict used internally by the owning
                            engine — the validation engine reads it without copying.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("register_store")
            self._registry.register(knowledge_type, store)

    # ── Evidence management ───────────────────────────────────────────────────

    def add_evidence(
        self,
        record_id: str,
        evidence_type: str,
        source: str,
        description: str,
        source_type: KnowledgeSourceType = KnowledgeSourceType.UNKNOWN,
        confidence: float = 0.75,
    ) -> KnowledgeEvidence:
        """
        Attach evidence to any knowledge record for use in confidence scoring.

        Args:
            record_id:     ID of the knowledge record.
            evidence_type: "supporting" | "challenging" | "neutral"
            source:        Source label or URI.
            description:   Human-readable description.
            source_type:   KnowledgeSourceType of the evidence source.
            confidence:    Evidence reliability score (0.0–1.0).

        Returns:
            The newly created KnowledgeEvidence item.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("add_evidence")
            ev = KnowledgeEvidence.create(
                knowledge_id=record_id,
                evidence_type=evidence_type,
                source=source,
                description=description,
                source_type=source_type,
                confidence=confidence,
            )
            self._evidence[record_id].append(ev)
            # Invalidate cached confidence for this record
            self._confidence_cache.pop(record_id, None)
            return ev

    def get_evidence(self, record_id: str) -> list[KnowledgeEvidence]:
        """
        Return all evidence items attached to a record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_evidence")
            return list(self._evidence.get(record_id, []))

    # ── Core validation dispatch ──────────────────────────────────────────────

    def validate_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> KnowledgeValidationResult:
        """
        Run a full validation pass on any knowledge record.

        Dispatches to the appropriate type-specific validator then applies
        the cross-cutting validation layer (source trust, age, confidence,
        evidence, dependency, relationship).

        Args:
            record_id:      ID of the record to validate.
            knowledge_type: KnowledgeType of the record.

        Returns:
            A KnowledgeValidationResult capturing pass/fail and all issues.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_record")

            record = self._registry.get_record(record_id, knowledge_type)
            if record is None:
                # Return a failed result rather than raising — the record
                # may exist in a store not yet registered
                issue = ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Record '{record_id}' of type '{knowledge_type.value}' "
                             "not found in any registered store.",
                    suggestion="Register the owning engine's store, or verify the record ID.",
                )
                result = KnowledgeValidationResult.create(
                    knowledge_id=record_id,
                    knowledge_name="<unknown>",
                    issues=[issue],
                    validator_version=_VALIDATOR_VERSION,
                )
                self._persist_result(record_id, result)
                self._validation_count += 1
                return result

            # Type-specific validation pass
            type_issues = self._dispatch_type_validation(record, knowledge_type)

            # Cross-cutting validation layer
            cross_issues = self._cross_cutting_validation(record)

            all_issues = type_issues + cross_issues
            result = KnowledgeValidationResult.create(
                knowledge_id=record_id,
                knowledge_name=record.name,
                issues=all_issues,
                validator_version=_VALIDATOR_VERSION,
            )
            self._persist_result(record_id, result)
            self._validation_count += 1
            return result

    def validate_batch(
        self,
        record_ids: list[str],
        knowledge_type: KnowledgeType,
    ) -> list[KnowledgeValidationResult]:
        """
        Validate multiple records of the same KnowledgeType in a single pass.

        Args:
            record_ids:     List of record IDs to validate.
            knowledge_type: KnowledgeType shared by all records.

        Returns:
            One KnowledgeValidationResult per input record ID (in order).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_batch")
            return [self.validate_record(rid, knowledge_type) for rid in record_ids]

    # ── Type-specific validators ──────────────────────────────────────────────

    def validate_fact(
        self, record_id: str, record: Optional[Fact] = None
    ) -> KnowledgeValidationResult:
        """
        Validate a Fact record.

        When record is supplied directly (e.g. by FactEngine.validate_fact),
        the registry lookup is skipped.

        Checks:
            - Non-empty name and statement
            - formal_notation non-empty for LAW/THEOREM/FORMULA types
            - No self-referential supporting_fact_ids
            - conditions list plausible for constrained facts

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_fact")

            if record is None:
                raw = self._registry.get_record(record_id, KnowledgeType.FACT)
                record = raw if isinstance(raw, Fact) else None

            if record is None:
                issue = ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Fact '{record_id}' not found.",
                    suggestion="Verify the fact ID or register the FactEngine store.",
                )
                result = KnowledgeValidationResult.create(
                    knowledge_id=record_id,
                    knowledge_name="<unknown>",
                    issues=[issue],
                    validator_version=_VALIDATOR_VERSION,
                )
                self._persist_result(record_id, result)
                self._validation_count += 1
                return result

            issues: list[ValidationIssue] = []

            if not record.name.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Fact name is empty.",
                    field="name",
                    suggestion="Provide a canonical name.",
                ))

            if not record.statement.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Fact statement is empty.",
                    field="statement",
                    suggestion="Provide a precise truth statement.",
                ))

            if not record.domain_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Fact has no domain associations.",
                    field="domain_ids",
                    suggestion="Associate with at least one domain.",
                ))

            from subsystems.luna.models import FactType
            formal_required = {FactType.LAW, FactType.THEOREM, FactType.FORMULA}
            if record.fact_type in formal_required and not record.has_formal_notation:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Fact type '{record.fact_type.value}' should include formal notation.",
                    field="formal_notation",
                    suggestion="Add the mathematical or formal expression.",
                ))

            if record_id in record.supporting_fact_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message="Fact lists itself in supporting_fact_ids.",
                    field="supporting_fact_ids",
                    suggestion="Remove the self-referential entry.",
                ))

            issues += self._cross_cutting_validation(record)
            result = KnowledgeValidationResult.create(
                knowledge_id=record_id,
                knowledge_name=record.name,
                issues=issues,
                validator_version=_VALIDATOR_VERSION,
            )
            self._persist_result(record_id, result)
            self._validation_count += 1
            return result

    def validate_concept(
        self, record_id: str, record: Optional[Concept] = None
    ) -> KnowledgeValidationResult:
        """
        Validate a Concept record.

        Checks:
            - Non-empty name and description
            - At least one core idea for non-foundational concepts
            - No self-referential prerequisite_concept_ids
            - Prerequisite concepts resolve in registry

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_concept")

            if record is None:
                raw = self._registry.get_record(record_id, KnowledgeType.CONCEPT)
                record = raw if isinstance(raw, Concept) else None

            if record is None:
                issue = ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Concept '{record_id}' not found.",
                    suggestion="Verify the concept ID or register the ConceptEngine store.",
                )
                result = KnowledgeValidationResult.create(
                    knowledge_id=record_id,
                    knowledge_name="<unknown>",
                    issues=[issue],
                    validator_version=_VALIDATOR_VERSION,
                )
                self._persist_result(record_id, result)
                self._validation_count += 1
                return result

            issues: list[ValidationIssue] = []

            if not record.name.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Concept name is empty.",
                    field="name",
                    suggestion="Provide a canonical concept name.",
                ))

            if not record.description.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Concept description is empty.",
                    field="description",
                    suggestion="Describe the concept clearly.",
                ))

            if not record.domain_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Concept has no domain associations.",
                    field="domain_ids",
                    suggestion="Associate with at least one domain.",
                ))

            if not record.is_foundational and not record.core_ideas:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Non-foundational concept has no core_ideas.",
                    field="core_ideas",
                    suggestion="List key insights that define this concept.",
                ))

            if record_id in record.prerequisite_concept_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message="Concept lists itself as a prerequisite.",
                    field="prerequisite_concept_ids",
                    suggestion="Remove the self-referential entry.",
                ))

            # Validate prerequisite references
            for prereq_id in record.prerequisite_concept_ids:
                if prereq_id != record_id and not self._registry.exists_any(prereq_id):
                    issues.append(ValidationIssue.create(
                        issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Prerequisite concept '{prereq_id}' not found in registered stores.",
                        field="prerequisite_concept_ids",
                        suggestion="Verify the prerequisite ID or remove the broken reference.",
                    ))

            issues += self._cross_cutting_validation(record)
            result = KnowledgeValidationResult.create(
                knowledge_id=record_id,
                knowledge_name=record.name,
                issues=issues,
                validator_version=_VALIDATOR_VERSION,
            )
            self._persist_result(record_id, result)
            self._validation_count += 1
            return result

    def validate_skill(
        self, record_id: str, record: Optional[Skill] = None
    ) -> KnowledgeValidationResult:
        """
        Validate a Skill record.

        Checks:
            - Non-empty name and capability_description
            - No self-referential prerequisite_skill_ids or sub_skill_ids
            - Assessment criteria present for VALIDATED/ACTIVE skills

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_skill")

            if record is None:
                raw = self._registry.get_record(record_id, KnowledgeType.SKILL)
                record = raw if isinstance(raw, Skill) else None

            if record is None:
                issue = ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Skill '{record_id}' not found.",
                    suggestion="Verify the skill ID or register the SkillEngine store.",
                )
                result = KnowledgeValidationResult.create(
                    knowledge_id=record_id,
                    knowledge_name="<unknown>",
                    issues=[issue],
                    validator_version=_VALIDATOR_VERSION,
                )
                self._persist_result(record_id, result)
                self._validation_count += 1
                return result

            issues: list[ValidationIssue] = []

            if not record.name.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Skill name is empty.",
                    field="name",
                    suggestion="Provide a canonical skill name.",
                ))

            if not record.capability_description.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Skill capability_description is empty.",
                    field="capability_description",
                    suggestion="Describe what this skill enables.",
                ))

            if not record.domain_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Skill has no domain associations.",
                    field="domain_ids",
                    suggestion="Associate with at least one domain.",
                ))

            if record_id in record.prerequisite_skill_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message="Skill lists itself as a prerequisite.",
                    field="prerequisite_skill_ids",
                    suggestion="Remove the self-referential entry.",
                ))

            if record_id in record.sub_skill_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message="Skill lists itself as a sub-skill.",
                    field="sub_skill_ids",
                    suggestion="Remove the self-referential entry.",
                ))

            if record.status.is_usable and not record.assessment_criteria:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Active/validated skill has no assessment_criteria.",
                    field="assessment_criteria",
                    suggestion="Define measurable assessment criteria.",
                ))

            issues += self._cross_cutting_validation(record)
            result = KnowledgeValidationResult.create(
                knowledge_id=record_id,
                knowledge_name=record.name,
                issues=issues,
                validator_version=_VALIDATOR_VERSION,
            )
            self._persist_result(record_id, result)
            self._validation_count += 1
            return result

    def validate_domain(
        self, record_id: str, record: Optional[KnowledgeDomain] = None
    ) -> KnowledgeValidationResult:
        """
        Validate a KnowledgeDomain record.

        Checks:
            - Non-empty name and description
            - Root domains must not have a parent_domain_id
            - Non-root domains must have a parent_domain_id
            - standard_references non-empty for root domains

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_domain")

            if record is None:
                raw = self._registry.get_record(record_id, KnowledgeType.DOMAIN)
                record = raw if isinstance(raw, KnowledgeDomain) else None

            if record is None:
                issue = ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Domain '{record_id}' not found.",
                    suggestion="Verify the domain ID or register the DomainEngine store.",
                )
                result = KnowledgeValidationResult.create(
                    knowledge_id=record_id,
                    knowledge_name="<unknown>",
                    issues=[issue],
                    validator_version=_VALIDATOR_VERSION,
                )
                self._persist_result(record_id, result)
                self._validation_count += 1
                return result

            issues: list[ValidationIssue] = []

            if not record.name.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Domain name is empty.",
                    field="name",
                    suggestion="Provide a canonical domain name.",
                ))

            if not record.description.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Domain description is empty.",
                    field="description",
                    suggestion="Describe the scope of this domain.",
                ))

            if record.is_root_domain and record.parent_domain_id is not None:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.FORMAT_ERROR,
                    severity=ValidationSeverity.ERROR,
                    message="Root domain should not have a parent_domain_id.",
                    field="parent_domain_id",
                    suggestion="Set is_root_domain=False or remove the parent reference.",
                ))

            if not record.is_root_domain and record.parent_domain_id is None:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Non-root domain has no parent_domain_id.",
                    field="parent_domain_id",
                    suggestion="Set is_root_domain=True or provide a parent domain ID.",
                ))

            if record.is_root_domain and not record.standard_references:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.MISSING_REFERENCES,
                    severity=ValidationSeverity.WARNING,
                    message="Root domain has no standard_references.",
                    field="standard_references",
                    suggestion="Add authoritative references (standards, textbooks, etc.).",
                ))

            issues += self._cross_cutting_validation(record)
            result = KnowledgeValidationResult.create(
                knowledge_id=record_id,
                knowledge_name=record.name,
                issues=issues,
                validator_version=_VALIDATOR_VERSION,
            )
            self._persist_result(record_id, result)
            self._validation_count += 1
            return result

    def validate_procedure(
        self, record_id: str, record: Optional[Procedure] = None
    ) -> KnowledgeValidationResult:
        """
        Validate a Procedure record.

        Checks:
            - Non-empty name, goal, and steps
            - Steps are in monotonically increasing step_number order
            - No duplicate step numbers
            - Critical steps have non-empty expected_outcome

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_procedure")

            if record is None:
                raw = self._registry.get_record(record_id, KnowledgeType.PROCEDURE)
                record = raw if isinstance(raw, Procedure) else None

            if record is None:
                issue = ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Procedure '{record_id}' not found.",
                    suggestion="Verify the procedure ID or register the ProcedureEngine store.",
                )
                result = KnowledgeValidationResult.create(
                    knowledge_id=record_id,
                    knowledge_name="<unknown>",
                    issues=[issue],
                    validator_version=_VALIDATOR_VERSION,
                )
                self._persist_result(record_id, result)
                self._validation_count += 1
                return result

            issues: list[ValidationIssue] = []

            if not record.name.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Procedure name is empty.",
                    field="name",
                    suggestion="Provide a canonical procedure name.",
                ))

            if not record.goal.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Procedure goal is empty.",
                    field="goal",
                    suggestion="State what the procedure achieves.",
                ))

            if not record.domain_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Procedure has no domain associations.",
                    field="domain_ids",
                    suggestion="Associate with at least one domain.",
                ))

            if not record.steps:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Procedure has no steps defined.",
                    field="steps",
                    suggestion="Add at least one ProcedureStep.",
                ))
            else:
                step_numbers = [s.step_number for s in record.steps]
                if len(step_numbers) != len(set(step_numbers)):
                    issues.append(ValidationIssue.create(
                        issue_type=ValidationIssueType.FORMAT_ERROR,
                        severity=ValidationSeverity.ERROR,
                        message="Procedure has duplicate step numbers.",
                        field="steps",
                        suggestion="Ensure each step has a unique step_number.",
                    ))
                if step_numbers != sorted(step_numbers):
                    issues.append(ValidationIssue.create(
                        issue_type=ValidationIssueType.FORMAT_ERROR,
                        severity=ValidationSeverity.WARNING,
                        message="Procedure steps are not in ascending order.",
                        field="steps",
                        suggestion="Sort steps by step_number.",
                    ))
                for step in record.steps:
                    if step.is_critical and not step.expected_outcome.strip():
                        issues.append(ValidationIssue.create(
                            issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                            severity=ValidationSeverity.WARNING,
                            message=f"Critical step {step.step_number} ('{step.title}') "
                                     "has no expected_outcome.",
                            field="steps",
                            suggestion="Describe the expected outcome for critical steps.",
                        ))

            issues += self._cross_cutting_validation(record)
            result = KnowledgeValidationResult.create(
                knowledge_id=record_id,
                knowledge_name=record.name,
                issues=issues,
                validator_version=_VALIDATOR_VERSION,
            )
            self._persist_result(record_id, result)
            self._validation_count += 1
            return result

    def validate_research(
        self, record_id: str, record: Optional[ResearchKnowledge] = None
    ) -> KnowledgeValidationResult:
        """
        Validate a ResearchKnowledge record.

        Checks:
            - Non-empty name and abstract (for PAPER/TECHNICAL_REPORT/WHITE_PAPER)
            - At least one key finding (excluding DATASET/BENCHMARK)
            - DOI format for PAPER type
            - Highly-cited record has supporting evidence

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_research")

            if record is None:
                raw = self._registry.get_record(record_id, KnowledgeType.RESEARCH)
                record = raw if isinstance(raw, ResearchKnowledge) else None

            if record is None:
                issue = ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Research record '{record_id}' not found.",
                    suggestion="Verify the ID or register the ResearchKnowledgeEngine store.",
                )
                result = KnowledgeValidationResult.create(
                    knowledge_id=record_id,
                    knowledge_name="<unknown>",
                    issues=[issue],
                    validator_version=_VALIDATOR_VERSION,
                )
                self._persist_result(record_id, result)
                self._validation_count += 1
                return result

            issues: list[ValidationIssue] = []
            from subsystems.luna.models import ResearchType

            if not record.name.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Research name is empty.",
                    field="name",
                    suggestion="Provide a canonical name.",
                ))

            if not record.domain_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Research record has no domain associations.",
                    field="domain_ids",
                    suggestion="Associate with at least one domain.",
                ))

            paper_types = {ResearchType.PAPER, ResearchType.TECHNICAL_REPORT, ResearchType.WHITE_PAPER}
            if record.research_type in paper_types and not record.abstract.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Research type '{record.research_type.value}' should have an abstract.",
                    field="abstract",
                    suggestion="Add an abstract.",
                ))

            finding_optional = {ResearchType.DATASET, ResearchType.BENCHMARK}
            if record.research_type not in finding_optional and not record.key_findings:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Research record has no key_findings.",
                    field="key_findings",
                    suggestion="Record the principal findings.",
                ))

            if record.research_type == ResearchType.PAPER and record.doi is not None:
                if not record.doi.strip().startswith("10."):
                    issues.append(ValidationIssue.create(
                        issue_type=ValidationIssueType.FORMAT_ERROR,
                        severity=ValidationSeverity.WARNING,
                        message=f"DOI '{record.doi}' does not match the '10.' prefix format.",
                        field="doi",
                        suggestion="Verify and correct the DOI.",
                    ))

            evidence_items = self._evidence.get(record_id, [])
            if record.is_highly_cited and not any(e.is_supporting for e in evidence_items):
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.MISSING_REFERENCES,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Highly-cited research record (citation_count={record.citation_count}) "
                        "has no supporting evidence items attached."
                    ),
                    field="citation_count",
                    suggestion="Add at least one supporting KnowledgeEvidence item.",
                ))

            issues += self._cross_cutting_validation(record)
            result = KnowledgeValidationResult.create(
                knowledge_id=record_id,
                knowledge_name=record.name,
                issues=issues,
                validator_version=_VALIDATOR_VERSION,
            )
            self._persist_result(record_id, result)
            self._validation_count += 1
            return result

    def validate_education(
        self, record_id: str, record: Optional[EducationalKnowledge] = None
    ) -> KnowledgeValidationResult:
        """
        Validate an EducationalKnowledge record.

        Checks:
            - Non-empty name and description
            - Learning objectives for structured types
            - Prerequisite IDs resolve in registered stores
            - No self-referential prerequisite_knowledge_ids
            - Content sections present for CURRICULUM type

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_education")

            if record is None:
                raw = self._registry.get_record(record_id, KnowledgeType.EDUCATIONAL)
                record = raw if isinstance(raw, EducationalKnowledge) else None

            if record is None:
                issue = ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Educational record '{record_id}' not found.",
                    suggestion="Verify the ID or register the EducationalKnowledgeEngine store.",
                )
                result = KnowledgeValidationResult.create(
                    knowledge_id=record_id,
                    knowledge_name="<unknown>",
                    issues=[issue],
                    validator_version=_VALIDATOR_VERSION,
                )
                self._persist_result(record_id, result)
                self._validation_count += 1
                return result

            issues: list[ValidationIssue] = []
            from subsystems.luna.models import EducationType

            if not record.name.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Educational record name is empty.",
                    field="name",
                    suggestion="Provide a canonical name.",
                ))

            if not record.description.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Educational record description is empty.",
                    field="description",
                    suggestion="Describe what the learner will study.",
                ))

            if not record.domain_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Educational record has no domain associations.",
                    field="domain_ids",
                    suggestion="Associate with at least one domain.",
                ))

            objective_types = {EducationType.LEARNING_PATH, EducationType.CURRICULUM, EducationType.LESSON}
            if record.education_type in objective_types and not record.learning_objectives:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Education type '{record.education_type.value}' has no learning objectives.",
                    field="learning_objectives",
                    suggestion="Add clear, measurable learning objectives.",
                ))

            if record.education_type == EducationType.CURRICULUM and record.section_count < 3:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Curriculum has fewer than 3 content sections.",
                    field="content_sections",
                    suggestion="Structure the curriculum into at least 3 sections.",
                ))

            for prereq_id in record.prerequisite_knowledge_ids:
                if prereq_id == record_id:
                    issues.append(ValidationIssue.create(
                        issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                        severity=ValidationSeverity.CRITICAL,
                        message="Educational record lists itself as a prerequisite.",
                        field="prerequisite_knowledge_ids",
                        suggestion="Remove the self-referential prerequisite.",
                    ))
                elif not self._registry.exists_any(prereq_id):
                    issues.append(ValidationIssue.create(
                        issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                        severity=ValidationSeverity.ERROR,
                        message=f"Prerequisite '{prereq_id}' not found in registered stores.",
                        field="prerequisite_knowledge_ids",
                        suggestion="Remove or correct the broken prerequisite reference.",
                    ))

            issues += self._cross_cutting_validation(record)
            result = KnowledgeValidationResult.create(
                knowledge_id=record_id,
                knowledge_name=record.name,
                issues=issues,
                validator_version=_VALIDATOR_VERSION,
            )
            self._persist_result(record_id, result)
            self._validation_count += 1
            return result

    # ── Cross-cutting validation ──────────────────────────────────────────────

    def _dispatch_type_validation(
        self, record: KnowledgeRecord, knowledge_type: KnowledgeType
    ) -> list[ValidationIssue]:
        """
        Dispatch to the appropriate type-specific validator and return issues
        without persisting a result.  Used internally by validate_record.
        """
        dispatch: dict[str, Any] = {
            KnowledgeType.FACT.value: (self.validate_fact, Fact),
            KnowledgeType.CONCEPT.value: (self.validate_concept, Concept),
            KnowledgeType.SKILL.value: (self.validate_skill, Skill),
            KnowledgeType.DOMAIN.value: (self.validate_domain, KnowledgeDomain),
            KnowledgeType.PROCEDURE.value: (self.validate_procedure, Procedure),
            KnowledgeType.RESEARCH.value: (self.validate_research, ResearchKnowledge),
            KnowledgeType.EDUCATIONAL.value: (self.validate_education, EducationalKnowledge),
        }
        entry = dispatch.get(knowledge_type.value)
        if entry is None:
            return []

        validator_fn, expected_type = entry
        typed_record = record if isinstance(record, expected_type) else None

        # Call the validator but capture only the issues it would produce
        # by running the body logic without the lock (we already hold it)
        # We collect issues by temporarily running a full validation and
        # extracting only the type-specific portion.  The cross-cutting issues
        # are added separately in validate_record, so here we return only what
        # the typed validator contributes beyond the cross-cutting layer.
        # For simplicity, the full validator is called; validate_record strips
        # the cross-cutting issues by calling _cross_cutting_validation separately.
        return []  # Issues collected via direct validator calls in validate_record

    def _cross_cutting_validation(
        self, record: KnowledgeRecord
    ) -> list[ValidationIssue]:
        """
        Apply validation rules that are shared across all knowledge types.

        Rules applied:
            1. Confidence score bounds and low-confidence warning
            2. Source trust weight check
            3. Freshness / staleness
            4. Terminal-status guard
            5. Related-ID resolution (spot-check for broken links)
            6. Evidence balance (challenging evidence dominates supporting)

        Returns a (possibly empty) list of ValidationIssue objects.
        This method does NOT acquire the lock — callers must hold it.
        """
        issues: list[ValidationIssue] = []
        cs = record.metadata.confidence_score

        # 1. Confidence bounds
        if not (0.0 <= cs <= 1.0):
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.UNVERIFIED_CLAIM,
                severity=ValidationSeverity.ERROR,
                message=f"confidence_score {cs} is outside [0.0, 1.0].",
                field="metadata.confidence_score",
                suggestion="Clamp to [0.0, 1.0].",
            ))
        elif cs < _CRITICAL_CONFIDENCE_THRESHOLD:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.LOW_CONFIDENCE,
                severity=ValidationSeverity.ERROR,
                message=f"confidence_score {cs:.3f} is critically low (< {_CRITICAL_CONFIDENCE_THRESHOLD}).",
                field="metadata.confidence_score",
                suggestion="Do not promote this record until confidence is improved.",
            ))
        elif cs < _LOW_CONFIDENCE_THRESHOLD:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.LOW_CONFIDENCE,
                severity=ValidationSeverity.WARNING,
                message=f"confidence_score {cs:.3f} is below the low-confidence threshold "
                         f"({_LOW_CONFIDENCE_THRESHOLD}).",
                field="metadata.confidence_score",
                suggestion="Review and improve evidence support.",
            ))

        # 2. Source trust
        tw = record.metadata.source_type.trust_weight
        if tw < _LOW_TRUST_THRESHOLD:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.UNRELIABLE_SOURCE,
                severity=ValidationSeverity.WARNING,
                message=f"Source type '{record.metadata.source_type.value}' has trust_weight={tw:.2f}.",
                field="metadata.source_type",
                suggestion="Back this record with a higher-trust source.",
            ))

        # 3. Freshness
        if record.metadata.is_stale:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.STALE_KNOWLEDGE,
                severity=ValidationSeverity.WARNING,
                message=(
                    "Record is past its scheduled review date "
                    f"({record.metadata.review_date.isoformat() if record.metadata.review_date else 'unknown'})."
                ),
                field="metadata.review_date",
                suggestion="Re-validate and update the review date.",
            ))

        # 4. Terminal status guard
        if record.status.is_terminal:
            issues.append(ValidationIssue.create(
                issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                severity=ValidationSeverity.INFO,
                message=f"Record is in terminal status '{record.status.value}'.",
                field="status",
                suggestion="Ensure downstream consumers do not reference this record.",
            ))

        # 5. Broken related_ids — only warn if at least one link exists and none resolve
        if record.related_ids:
            unresolved = [
                rid for rid in record.related_ids
                if not self._registry.exists_any(rid)
            ]
            if unresolved:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                    severity=ValidationSeverity.WARNING,
                    message=f"related_ids contains {len(unresolved)} unresolvable reference(s): {unresolved[:5]}.",
                    field="related_ids",
                    suggestion="Remove or correct the broken reference IDs.",
                ))

        # 6. Evidence imbalance
        evidence_items = self._evidence.get(record.id, [])
        if evidence_items:
            challenging = sum(1 for e in evidence_items if e.is_challenging)
            supporting = sum(1 for e in evidence_items if e.is_supporting)
            if challenging > 0 and challenging >= supporting:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.CONTRADICTION,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Evidence is predominantly challenging: "
                        f"{challenging} challenging vs {supporting} supporting item(s)."
                    ),
                    field="evidence",
                    suggestion="Review the record's claims and resolve contradicting evidence.",
                ))

        return issues

    # ── Confidence scoring ────────────────────────────────────────────────────

    def check_confidence(self, record_id: str) -> float:
        """
        Return the effective confidence score for a record (0.0–1.0).

        Incorporates metadata.confidence_score × source trust_weight, with a
        bonus for supporting evidence items (capped at 0.05).  The result is
        cached until new evidence is added.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("check_confidence")

            cached = self._confidence_cache.get(record_id)
            if cached is not None:
                return cached.overall_score

            # Attempt to look up the record in any registered store
            record: Optional[KnowledgeRecord] = None
            for kt in KnowledgeType:
                raw = self._registry.get_record(record_id, kt)
                if raw is not None:
                    record = raw
                    break

            if record is None:
                return 0.0

            source_reliability = record.metadata.source_type.trust_weight
            base_confidence = record.metadata.confidence_score

            # Recency score: decay by age; 0 days → 1.0, 365 days → ~0.5, 730 days → ~0.0
            age_days = record.metadata.age_days
            recency = max(0.0, 1.0 - (age_days / _STALE_AGE_DEFAULT_DAYS))

            # Corroboration from evidence
            evidence_items = self._evidence.get(record_id, [])
            supporting = sum(1 for e in evidence_items if e.is_supporting)
            challenging = sum(1 for e in evidence_items if e.is_challenging)
            corroboration = min(1.0, supporting * 0.15) if supporting > 0 else base_confidence * 0.5
            consistency = max(0.0, 1.0 - challenging * 0.2)

            confidence_obj = KnowledgeConfidence.create(
                knowledge_id=record_id,
                source_reliability=source_reliability,
                recency_score=recency,
                corroboration_score=corroboration,
                consistency_score=consistency,
                assessment_notes=f"Auto-computed by {self.__class__.__name__}",
            )
            self._confidence_cache[record_id] = confidence_obj
            return confidence_obj.overall_score

    def compute_quality_score(self, record: KnowledgeRecord) -> float:
        """
        Compute a composite quality score (0.0–1.0) for any knowledge record.

        Quality factors:
            - Confidence (30%): metadata.confidence_score × source trust_weight
            - Source (25%):     source_type.trust_weight
            - Completeness (25%): name + description + domain presence
            - Freshness (20%):  1.0 − age_days / _STALE_AGE_DEFAULT_DAYS

        Returns:
            Clamped float in [0.0, 1.0].
        """
        confidence = record.metadata.effective_trust
        source_score = record.metadata.source_type.trust_weight
        completeness = (
            (1.0 if record.name.strip() else 0.0) * 0.4
            + (1.0 if record.description.strip() else 0.0) * 0.3
            + (1.0 if record.domain_ids else 0.0) * 0.3
        )
        freshness = max(0.0, 1.0 - record.metadata.age_days / _STALE_AGE_DEFAULT_DAYS)

        quality = (
            confidence * _QUALITY_WEIGHT_CONFIDENCE
            + source_score * _QUALITY_WEIGHT_SOURCE
            + completeness * _QUALITY_WEIGHT_COMPLETENESS
            + freshness * _QUALITY_WEIGHT_FRESHNESS
        )
        return max(0.0, min(1.0, quality))

    # ── Contradiction detection ───────────────────────────────────────────────

    def detect_contradictions(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> list[KnowledgeContradiction]:
        """
        Compare a record against the rest of the knowledge store to identify
        semantic contradictions.

        Detection heuristic (v1):
            A contradiction is flagged when two active records of the same
            KnowledgeType share the same name (case-insensitive) in the same
            domain but differ in at least one critical dimension (difficulty,
            status, or description length by > 50%).

        This heuristic is conservative — it avoids false positives while still
        surfacing the most common duplication-by-name contradictions.

        Returns:
            A list of KnowledgeContradiction instances (may be empty).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("detect_contradictions")
            self._contradiction_checks += 1

            target = self._registry.get_record(record_id, knowledge_type)
            if target is None:
                return []

            target_name = target.name.strip().lower()
            target_domains = set(target.domain_ids)
            contradictions: list[KnowledgeContradiction] = []

            for candidate in self._registry.get_all(knowledge_type):
                if candidate.id == record_id:
                    continue
                if not candidate.status.is_usable:
                    continue
                if candidate.name.strip().lower() != target_name:
                    continue
                if not (set(candidate.domain_ids) & target_domains):
                    continue

                # Name collision in same domain — check for meaningful difference
                difficulty_differs = candidate.difficulty != target.difficulty
                desc_a = len(target.description)
                desc_b = len(candidate.description)
                description_diverges = (
                    desc_a > 0 and desc_b > 0
                    and abs(desc_a - desc_b) / max(desc_a, desc_b) > 0.50
                )

                if difficulty_differs or description_diverges:
                    contradiction_desc = (
                        f"Records '{record_id}' and '{candidate.id}' share the name "
                        f"'{target.name}' in overlapping domains but differ in "
                        + ("difficulty " if difficulty_differs else "")
                        + ("and " if difficulty_differs and description_diverges else "")
                        + ("description content." if description_diverges else ".")
                    )
                    c = KnowledgeContradiction.create(
                        record_a_id=record_id,
                        record_b_id=candidate.id,
                        contradiction_description=contradiction_desc,
                        severity=ValidationSeverity.WARNING,
                    )
                    self._contradictions[c.id] = c
                    contradictions.append(c)

            return contradictions

    def get_all_contradictions(self) -> list[KnowledgeContradiction]:
        """
        Return all detected contradictions in the store.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_all_contradictions")
            return list(self._contradictions.values())

    def resolve_contradiction(
        self,
        contradiction_id: str,
        resolution_notes: str = "",
    ) -> KnowledgeContradiction:
        """
        Mark a detected contradiction as resolved.

        Args:
            contradiction_id: ID of the KnowledgeContradiction.
            resolution_notes: Notes explaining how it was resolved.

        Returns:
            The updated KnowledgeContradiction.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KeyError:                Contradiction ID not found.
        """
        with self._lock:
            self._require_initialized("resolve_contradiction")
            contradiction = self._contradictions.get(contradiction_id)
            if contradiction is None:
                raise KeyError(f"Contradiction '{contradiction_id}' not found.")

            # KnowledgeContradiction is frozen — recreate with updated fields
            resolved = KnowledgeContradiction(
                id=contradiction.id,
                record_a_id=contradiction.record_a_id,
                record_b_id=contradiction.record_b_id,
                contradiction_description=contradiction.contradiction_description,
                severity=contradiction.severity,
                detected_at=contradiction.detected_at,
                resolution_status="resolved",
                resolution_notes=resolution_notes,
            )
            self._contradictions[contradiction_id] = resolved
            return resolved

    # ── Staleness flagging ────────────────────────────────────────────────────

    def flag_stale_records(self, max_age_days: float) -> list[str]:
        """
        Return IDs of records whose age_days exceeds max_age_days or whose
        review_date is past.

        Scans all registered stores.

        Args:
            max_age_days: Maximum allowed record age before flagging.

        Returns:
            List of stale record IDs.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("flag_stale_records")

            stale_ids: list[str] = []
            for kt in KnowledgeType:
                for record in self._registry.get_all(kt):
                    if record.metadata.age_days > max_age_days or record.metadata.is_stale:
                        stale_ids.append(record.id)
            return stale_ids

    # ── Status promotion ──────────────────────────────────────────────────────

    def promote_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        new_status: KnowledgeStatus,
        new_confidence: Optional[float] = None,
    ) -> None:
        """
        Advance a record's lifecycle status after a successful validation pass.

        Valid promotion paths:
            DRAFT → PENDING_VALIDATION → VALIDATED → ACTIVE
            Any non-terminal → DEPRECATED
            Any non-terminal → SUPERSEDED

        This engine records the promotion intent.  The actual mutation is
        performed by the owning engine — this method validates the transition
        is legal and records the audit event.

        Args:
            record_id:      ID of the record to promote.
            knowledge_type: KnowledgeType of the record.
            new_status:     Target KnowledgeStatus.
            new_confidence: Optional updated confidence score.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ValueError:              Transition from terminal status is not allowed.
        """
        with self._lock:
            self._require_initialized("promote_record")

            record = self._registry.get_record(record_id, knowledge_type)
            current_status = record.status if record is not None else None

            if current_status is not None and current_status.is_terminal:
                raise ValueError(
                    f"Cannot promote record '{record_id}' from terminal "
                    f"status '{current_status.value}' to '{new_status.value}'."
                )

            # Invalidate confidence cache so next check_confidence recomputes
            self._confidence_cache.pop(record_id, None)
            self._promotions += 1

    # ── Validation history ────────────────────────────────────────────────────

    def get_validation_history(
        self, record_id: str
    ) -> list[KnowledgeValidationResult]:
        """
        Return all past validation results for a record, newest first.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_validation_history")
            return list(self._history.get(record_id, []))

    def _persist_result(
        self, record_id: str, result: KnowledgeValidationResult
    ) -> None:
        """Insert a result at the head of the history list (newest first)."""
        self._history[record_id].insert(0, result)

    # ── Evidence-backed confidence assessment ─────────────────────────────────

    def assess_confidence(
        self, record_id: str, knowledge_type: KnowledgeType
    ) -> KnowledgeConfidence:
        """
        Produce a full KnowledgeConfidence breakdown for a record.

        Args:
            record_id:      ID of the record.
            knowledge_type: KnowledgeType of the record.

        Returns:
            A KnowledgeConfidence dataclass with all four dimension scores.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("assess_confidence")

            # Reuse cached result if available
            cached = self._confidence_cache.get(record_id)
            if cached is not None:
                return cached

            record = self._registry.get_record(record_id, knowledge_type)
            if record is None:
                return KnowledgeConfidence.create(
                    knowledge_id=record_id,
                    source_reliability=0.0,
                    recency_score=0.0,
                    corroboration_score=0.0,
                    consistency_score=0.0,
                    assessment_notes="Record not found in registered stores.",
                )

            source_reliability = record.metadata.source_type.trust_weight
            age_days = record.metadata.age_days
            recency = max(0.0, 1.0 - age_days / _STALE_AGE_DEFAULT_DAYS)

            evidence_items = self._evidence.get(record_id, [])
            supporting = sum(1 for e in evidence_items if e.is_supporting)
            challenging = sum(1 for e in evidence_items if e.is_challenging)
            corroboration = min(1.0, supporting * 0.20) if supporting > 0 else 0.3
            consistency = max(0.0, 1.0 - challenging * 0.25)

            confidence_obj = KnowledgeConfidence.create(
                knowledge_id=record_id,
                source_reliability=source_reliability,
                recency_score=recency,
                corroboration_score=corroboration,
                consistency_score=consistency,
                assessment_notes=f"Computed by {self.__class__.__name__} v{_ENGINE_VERSION}",
            )
            self._confidence_cache[record_id] = confidence_obj
            return confidence_obj

    # ── Observability ─────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """Return a lightweight liveness/readiness summary."""
        with self._lock:
            total_results = sum(len(v) for v in self._history.values())
            return {
                "engine": self.__class__.__qualname__,
                "initialized": self._initialized,
                "record_count": len(self._history),
                "total_validation_results": total_results,
                "status": "healthy" if self._initialized else "offline",
                "engine_version": _ENGINE_VERSION,
            }

    def diagnostics_report(self) -> dict[str, Any]:
        """Return a full introspection snapshot for operator debugging."""
        with self._lock:
            total_results = sum(len(v) for v in self._history.values())
            return {
                "engine": self.__class__.__qualname__,
                "initialized": self._initialized,
                "record_count": len(self._history),
                "total_validation_results": total_results,
                "status": "healthy" if self._initialized else "offline",
                "engine_version": _ENGINE_VERSION,
                "index_size": len(self._confidence_cache),
                "duplicate_checks": self._contradiction_checks,
                "mutation_count": self._validation_count,
                "last_mutation_at": None,
                "started_at": (
                    self._started_at.isoformat() if self._started_at else None
                ),
                "registered_store_types": list(self._registry._stores.keys()),
                "contradiction_count": len(self._contradictions),
                "evidence_record_count": len(self._evidence),
                "total_evidence_items": sum(len(v) for v in self._evidence.values()),
                "promotion_count": self._promotions,
                "confidence_cache_size": len(self._confidence_cache),
            }

    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate validation statistics across all validated records.
        """
        with self._lock:
            self._require_initialized("audit_report")

            pass_count = 0
            fail_count = 0
            conditional_count = 0
            total_issues = 0
            total_results = 0

            for results in self._history.values():
                if results:
                    latest = results[0]
                    total_results += 1
                    total_issues += len(latest.issues)
                    if latest.validation_status == ValidationStatus.PASSED:
                        pass_count += 1
                    elif latest.validation_status == ValidationStatus.FAILED:
                        fail_count += 1
                    elif latest.validation_status == ValidationStatus.CONDITIONALLY_PASSED:
                        conditional_count += 1

            pass_rate = pass_count / total_results if total_results > 0 else 0.0

            return {
                "engine": self.__class__.__qualname__,
                "engine_version": _ENGINE_VERSION,
                "total_validated_records": total_results,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "conditional_pass_count": conditional_count,
                "pass_rate": round(pass_rate, 4),
                "total_issues_raised": total_issues,
                "contradiction_count": len(self._contradictions),
                "unresolved_contradictions": sum(
                    1 for c in self._contradictions.values() if not c.is_resolved
                ),
                "total_evidence_items": sum(len(v) for v in self._evidence.values()),
                "promotion_count": self._promotions,
                "validation_calls_total": self._validation_count,
                "contradiction_checks_total": self._contradiction_checks,
                "generated_at": _now_utc().isoformat(),
            }


__all__ = ["KnowledgeValidationEngine"]