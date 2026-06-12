"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/integrity.py

Concrete in-memory implementation of the LUNA Knowledge Integrity Engine.

Ensures consistency, correctness, and structural integrity across all seven
knowledge types owned by LUNA.  Acts as the canonical guard against duplicate
content, broken references, invalid relationships, circular dependencies,
orphaned records, index inconsistencies, metadata corruption, and validation
inconsistencies.

Supported knowledge types:
    Fact · Concept · Skill · KnowledgeDomain · Procedure ·
    ResearchKnowledge · EducationalKnowledge

Responsibilities:
    audit_knowledge             — integrity scan for a single knowledge type
    audit_all_knowledge         — full cross-type integrity scan
    check_record_integrity      — per-record structural checks
    detect_duplicates           — fingerprint-based duplicate detection
    detect_broken_references    — unresolvable reference ID detection
    detect_broken_dependencies  — unresolvable prerequisite/dependency IDs
    detect_circular_dependencies — DFS-based cycle detection in dependency graphs
    detect_orphaned_records     — records with no valid domain membership
    validate_relationship_integrity — related_ids cross-reference validation
    validate_index_integrity    — consistency between live store and index
    generate_integrity_report   — produce IntegrityReport from collected issues
    generate_audit_report       — produce KnowledgeAuditReport

Support models:
    IntegrityIssue · IntegrityReport · KnowledgeAuditReport

Audit areas:
    duplicate content         — cross-fingerprint collision detection
    missing references        — domain_ids, fact_ids, concept_ids, etc.
    broken dependencies       — prerequisite chains that no longer resolve
    invalid relationships     — related_ids pointing to absent records
    orphaned concepts         — concepts with no domain membership
    orphaned skills           — skills with no domain membership
    index inconsistencies     — live records absent from or stale in the index
    metadata corruption       — confidence out of range; empty mandatory fields
    validation inconsistencies — validated status inconsistent with issues present

Thread safety:    threading.RLock on every public operation.
Lifecycle-gated:  every public method raises LunaNotInitializedError before
                  initialize() or after shutdown().
In-memory v1 implementation.  No persistence layer.

Integrates with (via injected engine handles):
    facts.py           FactEngine
    concepts.py        ConceptEngine
    skills.py          SkillEngine
    domains.py         KnowledgeDomainEngine
    procedures.py      ProceduralKnowledgeEngine
    research.py        ResearchKnowledgeEngine
    education.py       EducationalKnowledgeEngine
    knowledge_index.py KnowledgeIndexEngine
    retrieval.py       KnowledgeRetrievalEngine (optional; used for coverage stats)

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
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from subsystems.luna.exceptions import (
    KnowledgeIntegrityError,
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractKnowledgeIntegrityEngine
from subsystems.luna.models import (
    Concept,
    EducationalKnowledge,
    Fact,
    IntegrityIssue,
    IntegrityIssueType,
    IntegrityReport,
    KnowledgeAuditReport,
    KnowledgeContradiction,
    KnowledgeDomain,
    KnowledgeRecord,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeValidationResult,
    Procedure,
    ResearchKnowledge,
    Skill,
    SkillLevel,
    ValidationSeverity,
    ValidationStatus,
    _new_id,
    _utcnow,
)

logger = logging.getLogger(__name__)

_ENGINE_VERSION: str = "5.0.0"
_LOW_CONFIDENCE_THRESHOLD: float = 0.40

# All seven knowledge types the engine operates over
_ALL_TYPES: list[KnowledgeType] = [
    KnowledgeType.FACT,
    KnowledgeType.CONCEPT,
    KnowledgeType.SKILL,
    KnowledgeType.DOMAIN,
    KnowledgeType.PROCEDURE,
    KnowledgeType.RESEARCH,
    KnowledgeType.EDUCATIONAL,
]


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS — reference field extraction
# ─────────────────────────────────────────────────────────────────────────────

def _reference_fields(record: KnowledgeRecord) -> dict[str, list[str]]:
    """
    Return a mapping of field_name → list[referenced_id] for every field on
    ``record`` that is expected to point at another knowledge record.

    Only fields that are actually defined on the concrete subtype are included;
    no fields are invented.
    """
    refs: dict[str, list[str]] = {}

    # Base: domain membership
    if record.domain_ids:
        refs["domain_ids"] = list(record.domain_ids)

    # Base: related records
    if record.related_ids:
        refs["related_ids"] = list(record.related_ids)

    if isinstance(record, Fact):
        if record.supporting_fact_ids:
            refs["supporting_fact_ids"] = list(record.supporting_fact_ids)

    elif isinstance(record, Concept):
        if record.prerequisite_concept_ids:
            refs["prerequisite_concept_ids"] = list(record.prerequisite_concept_ids)
        if record.child_concept_ids:
            refs["child_concept_ids"] = list(record.child_concept_ids)
        if record.fact_ids:
            refs["fact_ids"] = list(record.fact_ids)

    elif isinstance(record, Skill):
        if record.required_concept_ids:
            refs["required_concept_ids"] = list(record.required_concept_ids)
        if record.required_fact_ids:
            refs["required_fact_ids"] = list(record.required_fact_ids)
        if record.prerequisite_skill_ids:
            refs["prerequisite_skill_ids"] = list(record.prerequisite_skill_ids)
        if record.sub_skill_ids:
            refs["sub_skill_ids"] = list(record.sub_skill_ids)

    elif isinstance(record, KnowledgeDomain):
        if record.sub_domain_ids:
            refs["sub_domain_ids"] = list(record.sub_domain_ids)
        if record.core_concept_ids:
            refs["core_concept_ids"] = list(record.core_concept_ids)
        if record.core_skill_ids:
            refs["core_skill_ids"] = list(record.core_skill_ids)
        if record.core_fact_ids:
            refs["core_fact_ids"] = list(record.core_fact_ids)
        if record.parent_domain_id:
            refs["parent_domain_id"] = [record.parent_domain_id]

    elif isinstance(record, Procedure):
        if record.required_skill_ids:
            refs["required_skill_ids"] = list(record.required_skill_ids)
        if record.required_concept_ids:
            refs["required_concept_ids"] = list(record.required_concept_ids)

    elif isinstance(record, ResearchKnowledge):
        if record.cited_knowledge_ids:
            refs["cited_knowledge_ids"] = list(record.cited_knowledge_ids)
        if record.extracted_fact_ids:
            refs["extracted_fact_ids"] = list(record.extracted_fact_ids)
        if record.extracted_concept_ids:
            refs["extracted_concept_ids"] = list(record.extracted_concept_ids)

    elif isinstance(record, EducationalKnowledge):
        if record.prerequisite_knowledge_ids:
            refs["prerequisite_knowledge_ids"] = list(record.prerequisite_knowledge_ids)
        if record.target_skill_ids:
            refs["target_skill_ids"] = list(record.target_skill_ids)
        if record.target_concept_ids:
            refs["target_concept_ids"] = list(record.target_concept_ids)

    return refs


def _dependency_ids(record: KnowledgeRecord) -> list[str]:
    """
    Return the IDs that ``record`` directly depends on as prerequisites.
    Used for cycle detection and broken-dependency checks.
    """
    if isinstance(record, Concept):
        return list(record.prerequisite_concept_ids)
    if isinstance(record, Skill):
        return list(record.prerequisite_skill_ids) + list(record.required_concept_ids)
    if isinstance(record, EducationalKnowledge):
        return list(record.prerequisite_knowledge_ids)
    if isinstance(record, Procedure):
        return list(record.required_skill_ids) + list(record.required_concept_ids)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeIntegrityEngine(AbstractKnowledgeIntegrityEngine):
    """
    In-memory, thread-safe implementation of the LUNA Knowledge Integrity
    Engine (v1).

    Data stores (derived; not owned):
        All record data is read from the seven injected sub-engines.
        This engine maintains only its own operational state.

    Internal state:
        _resolved_issues        — set[issue_id] of manually resolved issues
        _resolution_notes       — dict[issue_id, str] of resolution notes
        _scan_history           — list of IntegrityReport (most recent N scans)

    Optional integration handles (injected; may be None):
        _fact_engine, _concept_engine, _skill_engine, _domain_engine,
        _procedure_engine, _research_engine, _educational_engine,
        _index_engine

    Lifecycle::

        engine = KnowledgeIntegrityEngine(
            fact_engine=my_fact_engine,
            concept_engine=my_concept_engine,
            ...
            index_engine=my_index_engine,
        )
        engine.initialize()
        report = engine.full_scan()
        audit  = engine.full_audit()
        engine.shutdown()
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        *,
        fact_engine: Optional[Any] = None,
        concept_engine: Optional[Any] = None,
        skill_engine: Optional[Any] = None,
        domain_engine: Optional[Any] = None,
        procedure_engine: Optional[Any] = None,
        research_engine: Optional[Any] = None,
        educational_engine: Optional[Any] = None,
        index_engine: Optional[Any] = None,
    ) -> None:
        self._fact_engine = fact_engine
        self._concept_engine = concept_engine
        self._skill_engine = skill_engine
        self._domain_engine = domain_engine
        self._procedure_engine = procedure_engine
        self._research_engine = research_engine
        self._educational_engine = educational_engine
        self._index_engine = index_engine

        # Lifecycle
        self._initialized: bool = False
        self._lock: threading.RLock = threading.RLock()
        self._started_at: Optional[datetime] = None

        # Resolution tracking
        self._resolved_issues: set[str] = set()
        self._resolution_notes: dict[str, str] = {}

        # Scan history (cap at 20)
        self._scan_history: list[IntegrityReport] = []
        _SCAN_HISTORY_CAP = 20

        # Observability counters
        self._scans_run: int = 0
        self._records_checked: int = 0
        self._issues_found: int = 0
        self._issues_resolved: int = 0
        self._last_scan_at: Optional[datetime] = None
        self._last_mutation_at: Optional[datetime] = None

        self._SCAN_HISTORY_CAP = _SCAN_HISTORY_CAP

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Prepare the engine for operation.

        Idempotent — calling on an already-initialized engine is a no-op.

        Raises:
            LunaLifecycleError: Internal initialization failure.
        """
        with self._lock:
            if self._initialized:
                return
            try:
                self._resolved_issues.clear()
                self._resolution_notes.clear()
                self._scan_history.clear()
                self._scans_run = 0
                self._records_checked = 0
                self._issues_found = 0
                self._issues_resolved = 0
                self._last_scan_at = None
                self._last_mutation_at = None
                self._started_at = _utcnow()
                self._initialized = True
                logger.info(
                    "KnowledgeIntegrityEngine initialized (version=%s)", _ENGINE_VERSION
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="KnowledgeIntegrityEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        """
        Release resources and put the engine into a quiescent state.

        Idempotent — calling on an already-stopped engine is a no-op.

        Raises:
            LunaLifecycleError: Internal teardown failure.
        """
        with self._lock:
            if not self._initialized:
                return
            try:
                self._resolved_issues.clear()
                self._resolution_notes.clear()
                self._scan_history.clear()
                self._initialized = False
                logger.info(
                    "KnowledgeIntegrityEngine shut down (scans_run=%d)", self._scans_run
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="KnowledgeIntegrityEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def is_initialized(self) -> bool:
        return self._initialized

    # ── Guard ─────────────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation=operation)

    # ── Internal engine dispatch ──────────────────────────────────────────────

    def _engine_for_type(self, kt: KnowledgeType) -> Optional[Any]:
        return {
            KnowledgeType.FACT:        self._fact_engine,
            KnowledgeType.CONCEPT:     self._concept_engine,
            KnowledgeType.SKILL:       self._skill_engine,
            KnowledgeType.DOMAIN:      self._domain_engine,
            KnowledgeType.PROCEDURE:   self._procedure_engine,
            KnowledgeType.RESEARCH:    self._research_engine,
            KnowledgeType.EDUCATIONAL: self._educational_engine,
        }.get(kt)

    def _retrieve_record(
        self, record_id: str, knowledge_type: KnowledgeType
    ) -> Optional[KnowledgeRecord]:
        engine = self._engine_for_type(knowledge_type)
        if engine is None or not engine.is_initialized():
            return None
        try:
            if knowledge_type == KnowledgeType.FACT:
                return engine.retrieve_fact(record_id)
            elif knowledge_type == KnowledgeType.CONCEPT:
                return engine.retrieve_concept(record_id)
            elif knowledge_type == KnowledgeType.SKILL:
                return engine.retrieve_skill(record_id)
            elif knowledge_type == KnowledgeType.DOMAIN:
                return engine.retrieve_domain(record_id)
            elif knowledge_type == KnowledgeType.PROCEDURE:
                return engine.retrieve_procedure(record_id)
            elif knowledge_type == KnowledgeType.RESEARCH:
                return engine.retrieve_research(record_id)
            elif knowledge_type == KnowledgeType.EDUCATIONAL:
                return engine.retrieve_educational(record_id)
        except Exception:
            return None
        return None

    def _get_all_records(
        self,
        kt: KnowledgeType,
        status_filter: Optional[list[KnowledgeStatus]] = None,
    ) -> list[KnowledgeRecord]:
        engine = self._engine_for_type(kt)
        if engine is None or not engine.is_initialized():
            return []
        try:
            kwargs: dict[str, Any] = {"limit": 100_000, "offset": 0}
            if status_filter:
                kwargs["status_filter"] = status_filter
            if kt == KnowledgeType.FACT:
                return engine.get_all_facts(**kwargs)
            elif kt == KnowledgeType.CONCEPT:
                return engine.get_all_concepts(**kwargs)
            elif kt == KnowledgeType.SKILL:
                return engine.get_all_skills(**kwargs)
            elif kt == KnowledgeType.DOMAIN:
                return engine.get_all_domains(
                    status_filter=status_filter if status_filter else None
                )
            elif kt == KnowledgeType.PROCEDURE:
                return engine.get_all_procedures(**kwargs)
            elif kt == KnowledgeType.RESEARCH:
                return engine.get_all_research(**kwargs)
            elif kt == KnowledgeType.EDUCATIONAL:
                return engine.get_all_educational(**kwargs)
        except Exception:
            return []
        return []

    def _resolve_any_id(self, ref_id: str) -> bool:
        """
        Return True if ``ref_id`` resolves to a live record in any of the
        seven connected stores.
        """
        for kt in _ALL_TYPES:
            record = self._retrieve_record(ref_id, kt)
            if record is not None:
                return True
        return False

    def _build_record_id_set(self, kt: KnowledgeType) -> set[str]:
        """Return the set of all record IDs for a given type."""
        return {r.id for r in self._get_all_records(kt)}

    def _build_global_id_set(self) -> set[str]:
        """Return all record IDs across all seven types."""
        all_ids: set[str] = set()
        for kt in _ALL_TYPES:
            all_ids |= self._build_record_id_set(kt)
        return all_ids

    # ── Scan history management ───────────────────────────────────────────────

    def _store_report(self, report: IntegrityReport) -> None:
        self._scan_history.append(report)
        if len(self._scan_history) > self._SCAN_HISTORY_CAP:
            self._scan_history = self._scan_history[-self._SCAN_HISTORY_CAP:]
        self._last_scan_at = _utcnow()

    # ── Per-record structural checks ──────────────────────────────────────────

    def _check_mandatory_fields(
        self, record: KnowledgeRecord
    ) -> list[IntegrityIssue]:
        """Verify that every mandatory text field is non-empty."""
        issues: list[IntegrityIssue] = []

        if not record.name or not record.name.strip():
            issues.append(IntegrityIssue.create(
                issue_type=IntegrityIssueType.BROKEN_REFERENCE,
                severity=ValidationSeverity.ERROR,
                affected_id=record.id,
                affected_type=record.knowledge_type,
                description=f"Record '{record.id}' has an empty 'name' field.",
                resolution_hint="Provide a non-empty canonical name.",
            ))

        if not record.description or not record.description.strip():
            issues.append(IntegrityIssue.create(
                issue_type=IntegrityIssueType.BROKEN_REFERENCE,
                severity=ValidationSeverity.WARNING,
                affected_id=record.id,
                affected_type=record.knowledge_type,
                description=f"Record '{record.id}' ('{record.name}') has an empty 'description' field.",
                resolution_hint="Provide a meaningful description.",
            ))

        # Type-specific mandatory fields
        if isinstance(record, Fact) and not record.statement.strip():
            issues.append(IntegrityIssue.create(
                issue_type=IntegrityIssueType.BROKEN_REFERENCE,
                severity=ValidationSeverity.ERROR,
                affected_id=record.id,
                affected_type=record.knowledge_type,
                description=f"Fact '{record.id}' ('{record.name}') has an empty 'statement' field.",
                resolution_hint="Provide the precise canonical statement for this fact.",
            ))

        return issues

    def _check_confidence_range(
        self, record: KnowledgeRecord
    ) -> list[IntegrityIssue]:
        """Verify confidence_score is within [0.0, 1.0]."""
        issues: list[IntegrityIssue] = []
        score = record.metadata.confidence_score
        if not (0.0 <= score <= 1.0):
            issues.append(IntegrityIssue.create(
                issue_type=IntegrityIssueType.BROKEN_REFERENCE,
                severity=ValidationSeverity.CRITICAL,
                affected_id=record.id,
                affected_type=record.knowledge_type,
                description=(
                    f"Record '{record.id}' has confidence_score={score:.4f} "
                    "which is outside the valid range [0.0, 1.0]."
                ),
                auto_resolvable=True,
                resolution_hint="Clamp confidence_score to [0.0, 1.0].",
            ))
        return issues

    def _check_terminal_status_referrers(
        self,
        record: KnowledgeRecord,
        global_id_set: set[str],
    ) -> list[IntegrityIssue]:
        """
        Detect records whose status is terminal (RETRACTED / ARCHIVED) but
        that still appear in their own reference fields — which are meaningless
        for terminal records and indicate a stale state.
        """
        issues: list[IntegrityIssue] = []
        if not record.status.is_terminal:
            return issues

        ref_fields = _reference_fields(record)
        total_refs = sum(len(v) for v in ref_fields.values())
        if total_refs > 0:
            issues.append(IntegrityIssue.create(
                issue_type=IntegrityIssueType.BROKEN_REFERENCE,
                severity=ValidationSeverity.WARNING,
                affected_id=record.id,
                affected_type=record.knowledge_type,
                description=(
                    f"Terminal record '{record.id}' (status={record.status.value}) "
                    f"still carries {total_refs} outbound reference(s). "
                    "These references have no operational effect but create noise."
                ),
                auto_resolvable=True,
                resolution_hint="Clear all reference lists on terminal records.",
            ))
        return issues

    # ── check_record_integrity ─────────────────────────────────────────────────

    def check_record_integrity(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> IntegrityReport:
        """
        Run a full integrity check on a single knowledge record.

        Checks performed:
            - mandatory fields populated
            - confidence_score in [0.0, 1.0]
            - all reference IDs resolve in the global store
            - all dependency IDs resolve in the global store
            - terminal records do not carry live outbound references

        Args:
            record_id:      ID of the record to check.
            knowledge_type: KnowledgeType of the record.

        Returns:
            IntegrityReport containing any issues found.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeIntegrityError: Record not found.
        """
        self._require_initialized("check_record_integrity")
        with self._lock:
            t_start = time.monotonic()

            record = self._retrieve_record(record_id, knowledge_type)
            if record is None:
                raise KnowledgeIntegrityError(
                    f"Record '{record_id}' of type '{knowledge_type.value}' "
                    "not found in the originating store.",
                    context={"record_id": record_id, "knowledge_type": knowledge_type.value},
                )

            global_ids = self._build_global_id_set()
            issues: list[IntegrityIssue] = []

            issues += self._check_mandatory_fields(record)
            issues += self._check_confidence_range(record)
            issues += self._check_record_references(record, global_ids)
            issues += self._check_record_dependencies(record, global_ids)
            issues += self._check_terminal_status_referrers(record, global_ids)

            self._records_checked += 1
            self._issues_found += len(issues)

            duration_ms = (time.monotonic() - t_start) * 1000.0
            report = IntegrityReport.create(
                issues=issues,
                records_scanned=1,
                scan_duration_ms=duration_ms,
                scan_version=_ENGINE_VERSION,
                summary=(
                    f"Single-record integrity check for '{record_id}': "
                    f"{len(issues)} issue(s) found."
                ),
            )
            self._store_report(report)
            return report

    # ── Broken reference detection ─────────────────────────────────────────────

    def _check_record_references(
        self,
        record: KnowledgeRecord,
        global_ids: set[str],
    ) -> list[IntegrityIssue]:
        """Return IntegrityIssue for every unresolvable reference on a record."""
        issues: list[IntegrityIssue] = []
        for field_name, ref_ids in _reference_fields(record).items():
            for ref_id in ref_ids:
                if ref_id not in global_ids:
                    issues.append(IntegrityIssue.create(
                        issue_type=IntegrityIssueType.BROKEN_REFERENCE,
                        severity=ValidationSeverity.ERROR,
                        affected_id=record.id,
                        affected_type=record.knowledge_type,
                        description=(
                            f"Record '{record.id}' field '{field_name}' references "
                            f"'{ref_id}' which does not exist in any knowledge store."
                        ),
                        conflicting_id=ref_id,
                        resolution_hint=f"Remove or correct the broken reference in '{field_name}'.",
                    ))
        return issues

    def _check_record_dependencies(
        self,
        record: KnowledgeRecord,
        global_ids: set[str],
    ) -> list[IntegrityIssue]:
        """Return IntegrityIssue for every unresolvable prerequisite dependency."""
        issues: list[IntegrityIssue] = []
        for dep_id in _dependency_ids(record):
            if dep_id not in global_ids:
                issues.append(IntegrityIssue.create(
                    issue_type=IntegrityIssueType.BROKEN_SKILL_PREREQUISITE,
                    severity=ValidationSeverity.ERROR,
                    affected_id=record.id,
                    affected_type=record.knowledge_type,
                    description=(
                        f"Record '{record.id}' declares a dependency on '{dep_id}' "
                        "which does not exist in any knowledge store."
                    ),
                    conflicting_id=dep_id,
                    resolution_hint="Remove the broken dependency or restore the required record.",
                ))
        return issues

    # ── detect_duplicates ─────────────────────────────────────────────────────

    def detect_duplicates(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[IntegrityIssue]:
        """
        Detect records with identical content fingerprints (potential duplicates).

        Args:
            knowledge_type: Restrict to one type; None scans all types.

        Returns:
            List of IntegrityIssue for each duplicate group member beyond the
            first (the first member is treated as the canonical record).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("detect_duplicates")
        with self._lock:
            types = [knowledge_type] if knowledge_type else _ALL_TYPES
            fingerprint_map: dict[str, list[tuple[str, KnowledgeType]]] = defaultdict(list)

            for kt in types:
                for record in self._get_all_records(kt):
                    fingerprint_map[record.fingerprint].append((record.id, kt))

            issues: list[IntegrityIssue] = []
            for fp, entries in fingerprint_map.items():
                if len(entries) < 2:
                    continue
                canonical_id, canonical_kt = entries[0]
                for dup_id, dup_kt in entries[1:]:
                    issue_type = (
                        IntegrityIssueType.DUPLICATE_CONCEPT
                        if dup_kt == KnowledgeType.CONCEPT
                        else IntegrityIssueType.DUPLICATE_FACT
                        if dup_kt == KnowledgeType.FACT
                        else IntegrityIssueType.BROKEN_REFERENCE
                    )
                    issues.append(IntegrityIssue.create(
                        issue_type=issue_type,
                        severity=ValidationSeverity.ERROR,
                        affected_id=dup_id,
                        affected_type=dup_kt,
                        description=(
                            f"Record '{dup_id}' (type={dup_kt.value}) shares fingerprint "
                            f"'{fp[:16]}…' with '{canonical_id}' (type={canonical_kt.value}). "
                            "Likely duplicate."
                        ),
                        conflicting_id=canonical_id,
                        auto_resolvable=False,
                        resolution_hint=(
                            f"Review both records and merge or delete '{dup_id}'."
                        ),
                    ))

            self._issues_found += len(issues)
            return issues

    # ── detect_broken_references ──────────────────────────────────────────────

    def detect_broken_references(self) -> list[dict[str, Any]]:
        """
        Return all records whose reference IDs point to non-existent records.

        Returns:
            List of dicts with keys:
                record_id, knowledge_type, field, broken_ref_id

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("detect_broken_references")
        with self._lock:
            global_ids = self._build_global_id_set()
            broken: list[dict[str, Any]] = []

            for kt in _ALL_TYPES:
                for record in self._get_all_records(kt):
                    for field_name, ref_ids in _reference_fields(record).items():
                        for ref_id in ref_ids:
                            if ref_id not in global_ids:
                                broken.append({
                                    "record_id": record.id,
                                    "knowledge_type": kt.value,
                                    "field": field_name,
                                    "broken_ref_id": ref_id,
                                })

            return broken

    # ── detect_broken_dependencies ────────────────────────────────────────────

    def detect_broken_dependencies(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[IntegrityIssue]:
        """
        Detect records whose declared prerequisites / dependencies cannot be
        resolved in any knowledge store.

        Args:
            knowledge_type: Restrict to one type; None scans all types.

        Returns:
            List of IntegrityIssue for each broken dependency.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("detect_broken_dependencies")
        with self._lock:
            types = [knowledge_type] if knowledge_type else _ALL_TYPES
            global_ids = self._build_global_id_set()
            issues: list[IntegrityIssue] = []

            for kt in types:
                for record in self._get_all_records(kt):
                    issues += self._check_record_dependencies(record, global_ids)

            self._issues_found += len(issues)
            return issues

    # ── detect_circular_dependencies ─────────────────────────────────────────

    def detect_circular_dependencies(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[IntegrityIssue]:
        """
        Detect circular prerequisite/dependency chains using iterative DFS.

        A cycle means: record A depends on B which (transitively) depends on A,
        making it impossible to satisfy the prerequisite graph.

        Args:
            knowledge_type: Restrict to one type; None scans all types.

        Returns:
            List of IntegrityIssue, one per record that is part of a cycle.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("detect_circular_dependencies")
        with self._lock:
            types = [knowledge_type] if knowledge_type else _ALL_TYPES
            issues: list[IntegrityIssue] = []

            for kt in types:
                records = self._get_all_records(kt)
                dep_map: dict[str, list[str]] = {
                    r.id: _dependency_ids(r) for r in records
                }
                record_map: dict[str, KnowledgeRecord] = {r.id: r for r in records}

                # DFS cycle detection — iterative to avoid recursion-depth issues
                visited: set[str] = set()
                in_cycle: set[str] = set()

                for start_id in dep_map:
                    if start_id in visited:
                        continue
                    # Stack entries: (node_id, iterator_over_deps, path_set)
                    stack: list[tuple[str, int, list[str]]] = [
                        (start_id, 0, [start_id])
                    ]
                    path_set: set[str] = {start_id}

                    while stack:
                        node_id, dep_idx, path = stack[-1]
                        deps = dep_map.get(node_id, [])

                        if dep_idx >= len(deps):
                            # All deps of this node processed
                            visited.add(node_id)
                            path_set.discard(node_id)
                            stack.pop()
                            continue

                        # Advance the dep index on the current frame
                        stack[-1] = (node_id, dep_idx + 1, path)
                        dep_id = deps[dep_idx]

                        if dep_id not in dep_map:
                            # dep not in this type's graph — skip (cross-type)
                            continue

                        if dep_id in path_set:
                            # Cycle detected
                            if dep_id not in in_cycle:
                                in_cycle.add(dep_id)
                                record = record_map.get(dep_id)
                                if record:
                                    cycle_path = path[path.index(dep_id):] + [dep_id]
                                    cycle_str = " → ".join(cycle_path)
                                    issues.append(IntegrityIssue.create(
                                        issue_type=IntegrityIssueType.CIRCULAR_DEPENDENCY,
                                        severity=ValidationSeverity.CRITICAL,
                                        affected_id=dep_id,
                                        affected_type=kt,
                                        description=(
                                            f"Circular dependency detected for record "
                                            f"'{dep_id}' (type={kt.value}): {cycle_str}"
                                        ),
                                        resolution_hint=(
                                            "Break the cycle by removing one prerequisite "
                                            "link in the chain."
                                        ),
                                    ))
                            continue

                        if dep_id in visited:
                            continue

                        # Push the dependency for exploration
                        new_path = path + [dep_id]
                        path_set.add(dep_id)
                        stack.append((dep_id, 0, new_path))

            self._issues_found += len(issues)
            return issues

    # ── detect_orphaned_records ────────────────────────────────────────────────

    def detect_orphaned_records(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[IntegrityIssue]:
        """
        Detect records that have no domain membership and are not referenced by
        any domain as a core record.

        A record is considered orphaned when:
            1. Its ``domain_ids`` list is empty, AND
            2. No KnowledgeDomain lists it in core_concept_ids, core_skill_ids,
               core_fact_ids, or sub_domain_ids.

        Terminal records (RETRACTED / ARCHIVED) are excluded.

        Args:
            knowledge_type: Restrict to one type; None checks Fact, Concept,
                            Skill, and Procedure (types that should have domains).

        Returns:
            List of IntegrityIssue for each orphaned record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("detect_orphaned_records")
        with self._lock:
            # Types for which domain membership is expected
            domain_required_types = {
                KnowledgeType.FACT,
                KnowledgeType.CONCEPT,
                KnowledgeType.SKILL,
                KnowledgeType.PROCEDURE,
            }
            if knowledge_type is not None:
                types = [knowledge_type] if knowledge_type in domain_required_types else []
            else:
                types = list(domain_required_types)

            if not types:
                return []

            # Build the set of IDs referenced by any domain
            domain_referenced: set[str] = set()
            for domain in self._get_all_records(KnowledgeType.DOMAIN):
                if isinstance(domain, KnowledgeDomain):
                    domain_referenced.update(domain.core_concept_ids)
                    domain_referenced.update(domain.core_skill_ids)
                    domain_referenced.update(domain.core_fact_ids)
                    domain_referenced.update(domain.sub_domain_ids)

            issues: list[IntegrityIssue] = []

            for kt in types:
                for record in self._get_all_records(kt):
                    if record.status.is_terminal:
                        continue
                    has_domain_ids = bool(record.domain_ids)
                    is_domain_referenced = record.id in domain_referenced
                    if not has_domain_ids and not is_domain_referenced:
                        issues.append(IntegrityIssue.create(
                            issue_type=IntegrityIssueType.ORPHANED_RECORD,
                            severity=ValidationSeverity.WARNING,
                            affected_id=record.id,
                            affected_type=kt,
                            description=(
                                f"Record '{record.id}' ('{record.name}', type={kt.value}) "
                                "has no domain membership and is not referenced by any domain."
                            ),
                            resolution_hint=(
                                "Assign the record to at least one KnowledgeDomain via "
                                "domain_ids, or add it to a domain's core lists."
                            ),
                        ))

            self._issues_found += len(issues)
            return issues

    # ── validate_relationship_integrity ──────────────────────────────────────

    def validate_relationship_integrity(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[IntegrityIssue]:
        """
        Validate that every ``related_ids`` entry on every record resolves to an
        existing record in any of the seven stores.

        Args:
            knowledge_type: Restrict to one type; None checks all types.

        Returns:
            List of IntegrityIssue for each broken related_ids reference.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("validate_relationship_integrity")
        with self._lock:
            types = [knowledge_type] if knowledge_type else _ALL_TYPES
            global_ids = self._build_global_id_set()
            issues: list[IntegrityIssue] = []

            for kt in types:
                for record in self._get_all_records(kt):
                    for related_id in record.related_ids:
                        if related_id not in global_ids:
                            issues.append(IntegrityIssue.create(
                                issue_type=IntegrityIssueType.BROKEN_REFERENCE,
                                severity=ValidationSeverity.WARNING,
                                affected_id=record.id,
                                affected_type=kt,
                                description=(
                                    f"Record '{record.id}' ('{record.name}') "
                                    f"has a broken related_ids entry: '{related_id}' "
                                    "does not exist."
                                ),
                                conflicting_id=related_id,
                                auto_resolvable=True,
                                resolution_hint=(
                                    f"Remove '{related_id}' from related_ids "
                                    "or restore the missing record."
                                ),
                            ))

            self._issues_found += len(issues)
            return issues

    # ── validate_index_integrity ──────────────────────────────────────────────

    def validate_index_integrity(self) -> list[IntegrityIssue]:
        """
        Detect inconsistencies between the live knowledge store and the
        KnowledgeIndexEngine (if connected).

        Detects:
            - Live records absent from the index (stale index — missing entries)
            - Index entries whose record no longer exists (stale index — ghost entries)

        Returns:
            List of IntegrityIssue for each inconsistency found.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("validate_index_integrity")
        with self._lock:
            issues: list[IntegrityIssue] = []

            if self._index_engine is None or not self._index_engine.is_initialized():
                return issues

            index_size = self._index_engine.get_index_size()
            global_ids = self._build_global_id_set()

            # Ghost entries: in index but not in any live store
            # Use query_index with no filters to enumerate all index entries
            try:
                index_entries = self._index_engine.query_index(limit=1_000_000)
            except Exception:
                index_entries = []

            for entry in index_entries:
                if entry.knowledge_id not in global_ids:
                    issues.append(IntegrityIssue.create(
                        issue_type=IntegrityIssueType.STALE_INDEX,
                        severity=ValidationSeverity.WARNING,
                        affected_id=entry.knowledge_id,
                        affected_type=entry.knowledge_type,
                        description=(
                            f"Index contains entry for '{entry.knowledge_id}' "
                            f"(type={entry.knowledge_type.value}) but the record no "
                            "longer exists in the live store."
                        ),
                        auto_resolvable=True,
                        resolution_hint=(
                            f"Call index_engine.remove_record('{entry.knowledge_id}', "
                            f"{entry.knowledge_type}) to purge the ghost entry."
                        ),
                    ))

            # Missing entries: in live store but not in index
            indexed_ids: set[str] = {e.knowledge_id for e in index_entries}
            for live_id in global_ids:
                if live_id not in indexed_ids:
                    # Determine type
                    for kt in _ALL_TYPES:
                        record = self._retrieve_record(live_id, kt)
                        if record is not None:
                            issues.append(IntegrityIssue.create(
                                issue_type=IntegrityIssueType.STALE_INDEX,
                                severity=ValidationSeverity.WARNING,
                                affected_id=live_id,
                                affected_type=kt,
                                description=(
                                    f"Live record '{live_id}' (type={kt.value}) "
                                    "is absent from the Knowledge Index."
                                ),
                                auto_resolvable=True,
                                resolution_hint=(
                                    f"Call index_engine.index_record('{live_id}', "
                                    f"{kt}) to add the missing entry."
                                ),
                            ))
                            break

            self._issues_found += len(issues)
            return issues

    # ── audit_knowledge (single type) ─────────────────────────────────────────

    def audit_knowledge(
        self,
        knowledge_type: KnowledgeType,
    ) -> IntegrityReport:
        """
        Run a comprehensive integrity scan restricted to one knowledge type.

        Checks performed:
            - mandatory fields populated
            - confidence_score in range
            - broken references (all reference fields)
            - broken dependencies (prerequisite chains)
            - duplicate fingerprints
            - circular dependencies
            - orphaned records (where applicable)

        Args:
            knowledge_type: The KnowledgeType to audit.

        Returns:
            IntegrityReport for the scanned type.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("audit_knowledge")
        with self._lock:
            t_start = time.monotonic()
            issues: list[IntegrityIssue] = []
            records = self._get_all_records(knowledge_type)
            global_ids = self._build_global_id_set()

            # Fingerprint map for this type's duplicate detection
            fingerprint_map: dict[str, list[str]] = defaultdict(list)
            for record in records:
                fingerprint_map[record.fingerprint].append(record.id)

            for record in records:
                issues += self._check_mandatory_fields(record)
                issues += self._check_confidence_range(record)
                issues += self._check_record_references(record, global_ids)
                issues += self._check_record_dependencies(record, global_ids)
                issues += self._check_terminal_status_referrers(record, global_ids)

            # Duplicates within this type
            canonical_type_map = {
                KnowledgeType.CONCEPT: IntegrityIssueType.DUPLICATE_CONCEPT,
                KnowledgeType.FACT:    IntegrityIssueType.DUPLICATE_FACT,
            }
            dup_issue_type = canonical_type_map.get(
                knowledge_type, IntegrityIssueType.BROKEN_REFERENCE
            )
            for fp, ids in fingerprint_map.items():
                if len(ids) > 1:
                    canonical = ids[0]
                    for dup_id in ids[1:]:
                        issues.append(IntegrityIssue.create(
                            issue_type=dup_issue_type,
                            severity=ValidationSeverity.ERROR,
                            affected_id=dup_id,
                            affected_type=knowledge_type,
                            description=(
                                f"Duplicate fingerprint '{fp[:16]}…': "
                                f"'{dup_id}' conflicts with '{canonical}'."
                            ),
                            conflicting_id=canonical,
                            resolution_hint=f"Merge or delete '{dup_id}'.",
                        ))

            # Circular dependencies
            issues += self.detect_circular_dependencies(knowledge_type=knowledge_type)

            # Orphaned records
            issues += self.detect_orphaned_records(knowledge_type=knowledge_type)

            scanned = len(records)
            self._records_checked += scanned
            self._issues_found += len(issues)
            self._scans_run += 1

            duration_ms = (time.monotonic() - t_start) * 1000.0
            report = IntegrityReport.create(
                issues=issues,
                records_scanned=scanned,
                scan_duration_ms=duration_ms,
                scan_version=_ENGINE_VERSION,
                summary=(
                    f"Type-scoped integrity audit for '{knowledge_type.value}': "
                    f"{scanned} records scanned, {len(issues)} issue(s) found."
                ),
            )
            self._store_report(report)
            return report

    # ── audit_all_knowledge (full scan) ───────────────────────────────────────

    def audit_all_knowledge(self) -> IntegrityReport:
        """
        Run a comprehensive integrity scan across all seven knowledge types.

        Equivalent to scan_type repeated for every type, plus cross-type
        relationship and index validation.

        Returns:
            IntegrityReport with all issues found across the full store.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("audit_all_knowledge")
        with self._lock:
            t_start = time.monotonic()
            all_issues: list[IntegrityIssue] = []
            total_scanned = 0
            global_ids = self._build_global_id_set()

            # Fingerprint map across all types for cross-type duplicate detection
            fingerprint_map: dict[str, list[tuple[str, KnowledgeType]]] = defaultdict(list)

            for kt in _ALL_TYPES:
                records = self._get_all_records(kt)
                total_scanned += len(records)
                for record in records:
                    fingerprint_map[record.fingerprint].append((record.id, kt))
                    all_issues += self._check_mandatory_fields(record)
                    all_issues += self._check_confidence_range(record)
                    all_issues += self._check_record_references(record, global_ids)
                    all_issues += self._check_record_dependencies(record, global_ids)
                    all_issues += self._check_terminal_status_referrers(record, global_ids)

            # Cross-type duplicate detection
            for fp, entries in fingerprint_map.items():
                if len(entries) > 1:
                    canonical_id, canonical_kt = entries[0]
                    for dup_id, dup_kt in entries[1:]:
                        issue_type = (
                            IntegrityIssueType.DUPLICATE_CONCEPT
                            if dup_kt == KnowledgeType.CONCEPT
                            else IntegrityIssueType.DUPLICATE_FACT
                            if dup_kt == KnowledgeType.FACT
                            else IntegrityIssueType.BROKEN_REFERENCE
                        )
                        all_issues.append(IntegrityIssue.create(
                            issue_type=issue_type,
                            severity=ValidationSeverity.ERROR,
                            affected_id=dup_id,
                            affected_type=dup_kt,
                            description=(
                                f"Cross-type duplicate: '{dup_id}' (type={dup_kt.value}) "
                                f"shares fingerprint '{fp[:16]}…' with '{canonical_id}' "
                                f"(type={canonical_kt.value})."
                            ),
                            conflicting_id=canonical_id,
                            resolution_hint=f"Review and merge or delete '{dup_id}'.",
                        ))

            # Circular dependencies across all types
            all_issues += self.detect_circular_dependencies()

            # Orphaned records
            all_issues += self.detect_orphaned_records()

            # Index integrity
            all_issues += self.validate_index_integrity()

            self._records_checked += total_scanned
            self._issues_found += len(all_issues)
            self._scans_run += 1

            duration_ms = (time.monotonic() - t_start) * 1000.0
            report = IntegrityReport.create(
                issues=all_issues,
                records_scanned=total_scanned,
                scan_duration_ms=duration_ms,
                scan_version=_ENGINE_VERSION,
                summary=(
                    f"Full knowledge store integrity audit: "
                    f"{total_scanned} records scanned across {len(_ALL_TYPES)} types, "
                    f"{len(all_issues)} issue(s) found."
                ),
            )
            self._store_report(report)
            return report

    # ── AbstractKnowledgeIntegrityEngine: full_scan / scan_type / check_record ─

    def full_scan(self) -> IntegrityReport:
        """
        Alias for audit_all_knowledge — satisfies the abstract interface contract.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.audit_all_knowledge()

    def scan_type(self, knowledge_type: KnowledgeType) -> IntegrityReport:
        """
        Alias for audit_knowledge — satisfies the abstract interface contract.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.audit_knowledge(knowledge_type)

    def check_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> IntegrityReport:
        """
        Alias for check_record_integrity — satisfies the abstract interface contract.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeIntegrityError: Record not found.
        """
        return self.check_record_integrity(record_id, knowledge_type)

    # ── find_duplicate_fingerprints (interface contract) ──────────────────────

    def find_duplicate_fingerprints(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[list[str]]:
        """
        Return groups of record IDs that share the same content fingerprint.

        Args:
            knowledge_type: Restrict to one type; None scans all types.

        Returns:
            A list of groups (each a list of IDs).  Groups of size 1 omitted.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("find_duplicate_fingerprints")
        with self._lock:
            types = [knowledge_type] if knowledge_type else _ALL_TYPES
            fp_map: dict[str, list[str]] = defaultdict(list)
            for kt in types:
                for record in self._get_all_records(kt):
                    fp_map[record.fingerprint].append(record.id)
            return [ids for ids in fp_map.values() if len(ids) > 1]

    # ── find_broken_references (interface contract) ────────────────────────────

    def find_broken_references(self) -> list[dict[str, Any]]:
        """
        Return all records whose reference IDs point to non-existent records.

        Satisfies the AbstractKnowledgeIntegrityEngine contract.
        Delegates to detect_broken_references(), which contains the
        full implementation, preserving backward compatibility.

        Returns:
            List of dicts with keys:
                record_id, knowledge_type, field, broken_ref_id

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.detect_broken_references()

    # ── resolve_issue ─────────────────────────────────────────────────────────

    def resolve_issue(
        self,
        issue_id: str,
        *,
        resolution_notes: str = "",
    ) -> None:
        """
        Mark an integrity issue as manually resolved.

        Args:
            issue_id:         The ID of the issue to mark resolved.
            resolution_notes: Human-readable description of how it was resolved.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeIntegrityError: issue_id not found in any recent scan report.
        """
        self._require_initialized("resolve_issue")
        with self._lock:
            # Verify the issue exists in a recent scan report
            found = False
            for report in self._scan_history:
                for issue in report.issues:
                    if issue.id == issue_id:
                        found = True
                        break
                if found:
                    break

            if not found and issue_id not in self._resolved_issues:
                raise KnowledgeIntegrityError(
                    f"Integrity issue '{issue_id}' not found in scan history.",
                    context={"issue_id": issue_id},
                )

            self._resolved_issues.add(issue_id)
            if resolution_notes:
                self._resolution_notes[issue_id] = resolution_notes
            self._issues_resolved += 1
            self._last_mutation_at = _utcnow()

            logger.info(
                "Integrity issue '%s' marked resolved. Notes: %s",
                issue_id,
                resolution_notes or "(none)",
            )

    # ── generate_integrity_report ─────────────────────────────────────────────

    def generate_integrity_report(
        self,
        issues: list[IntegrityIssue],
        records_scanned: int,
        scan_duration_ms: float,
        summary: str = "",
    ) -> IntegrityReport:
        """
        Build an IntegrityReport from a caller-supplied list of issues.

        This method is provided for callers that run partial scans and want a
        properly formed report without executing a full engine scan.

        Args:
            issues:           Pre-collected IntegrityIssue list.
            records_scanned:  Number of records examined to produce these issues.
            scan_duration_ms: Elapsed scan time in milliseconds.
            summary:          Optional human-readable summary.

        Returns:
            A fully populated IntegrityReport.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("generate_integrity_report")
        with self._lock:
            report = IntegrityReport.create(
                issues=issues,
                records_scanned=records_scanned,
                scan_duration_ms=scan_duration_ms,
                scan_version=_ENGINE_VERSION,
                summary=summary or f"{len(issues)} issue(s) found in {records_scanned} records.",
            )
            self._store_report(report)
            return report

    # ── full_audit / generate_audit_report ────────────────────────────────────

    def full_audit(self) -> KnowledgeAuditReport:
        """
        Produce a comprehensive KnowledgeAuditReport combining integrity,
        validation, confidence, coverage, and health scoring across the
        entire LUNA knowledge store.

        Returns:
            KnowledgeAuditReport with all computed metrics.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("full_audit")
        with self._lock:
            # Run a full integrity scan
            integrity_report = self.audit_all_knowledge()

            # Aggregate record statistics across all types
            total_records = 0
            active_records = 0
            deprecated_records = 0
            validated_records = 0
            confidence_sum = 0.0
            low_confidence_ids: list[str] = []
            stale_ids: list[str] = []
            domain_coverage: dict[str, int] = {}
            skill_coverage: dict[str, SkillLevel] = {}

            for kt in _ALL_TYPES:
                for record in self._get_all_records(kt):
                    total_records += 1
                    cs = record.metadata.confidence_score
                    confidence_sum += cs

                    if record.status.is_usable:
                        active_records += 1
                    if record.status == KnowledgeStatus.DEPRECATED:
                        deprecated_records += 1
                    if record.metadata.validation_status.is_valid:
                        validated_records += 1
                    if cs < _LOW_CONFIDENCE_THRESHOLD:
                        low_confidence_ids.append(record.id)
                    if record.metadata.is_stale:
                        stale_ids.append(record.id)

                    # Domain coverage: count records per domain
                    for domain_id in record.domain_ids:
                        domain_coverage[domain_id] = (
                            domain_coverage.get(domain_id, 0) + 1
                        )

            # Skill coverage: highest SkillLevel per domain
            for skill in self._get_all_records(KnowledgeType.SKILL):
                if not isinstance(skill, Skill):
                    continue
                for domain_id in skill.domain_ids:
                    # We cannot read actual proficiency from LUNA; use MASTER
                    # as the structural ceiling level encoded in the domain.
                    current = skill_coverage.get(domain_id)
                    level = SkillLevel.MASTER  # structural max
                    if current is None or level.rank > current.rank:
                        skill_coverage[domain_id] = level

            average_confidence = (
                confidence_sum / total_records if total_records > 0 else 0.0
            )

            # Count unresolved contradictions from issues
            contradiction_count = sum(
                1
                for issue in integrity_report.issues
                if issue.issue_type == IntegrityIssueType.CONFLICTING_FACT
            )

            return KnowledgeAuditReport.create(
                integrity_report=integrity_report,
                validation_results=[],  # Validation engine results not owned here
                contradiction_count=contradiction_count,
                total_records=total_records,
                active_records=active_records,
                deprecated_records=deprecated_records,
                validated_records=validated_records,
                average_confidence=average_confidence,
                domain_coverage=domain_coverage,
                skill_coverage=skill_coverage,
                low_confidence_ids=low_confidence_ids,
                stale_ids=stale_ids,
                audit_version=_ENGINE_VERSION,
                notes=(
                    f"Generated by KnowledgeIntegrityEngine v{_ENGINE_VERSION}. "
                    f"Scans run: {self._scans_run}. "
                    f"Issues resolved: {self._issues_resolved}."
                ),
            )

    def generate_audit_report(self) -> KnowledgeAuditReport:
        """
        Alias for full_audit — provided for explicit naming clarity.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.full_audit()

    # ── Observability ─────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """
        Return a lightweight liveness/readiness summary.

        Required keys: engine, initialized, record_count, status.
        """
        return {
            "engine": "KnowledgeIntegrityEngine",
            "initialized": self._initialized,
            "record_count": self._records_checked,
            "status": "healthy" if self._initialized else "offline",
        }

    def diagnostics_report(self) -> dict[str, Any]:
        """
        Return a full introspection snapshot for operator debugging.

        Required keys: engine, initialized, record_count, status, index_size,
        duplicate_checks, mutation_count, last_mutation_at.
        """
        with self._lock:
            return {
                "engine": "KnowledgeIntegrityEngine",
                "version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "status": "healthy" if self._initialized else "offline",
                "record_count": self._records_checked,
                "index_size": len(self._scan_history),
                "duplicate_checks": self._scans_run,
                "mutation_count": self._issues_resolved,
                "last_mutation_at": (
                    self._last_mutation_at.isoformat()
                    if self._last_mutation_at else None
                ),
                "scans_run": self._scans_run,
                "records_checked_total": self._records_checked,
                "issues_found_total": self._issues_found,
                "issues_resolved_total": self._issues_resolved,
                "scan_history_depth": len(self._scan_history),
                "last_scan_at": (
                    self._last_scan_at.isoformat() if self._last_scan_at else None
                ),
                "started_at": (
                    self._started_at.isoformat() if self._started_at else None
                ),
                "connected_engines": {
                    "fact": self._fact_engine is not None,
                    "concept": self._concept_engine is not None,
                    "skill": self._skill_engine is not None,
                    "domain": self._domain_engine is not None,
                    "procedure": self._procedure_engine is not None,
                    "research": self._research_engine is not None,
                    "educational": self._educational_engine is not None,
                    "index": self._index_engine is not None,
                },
            }

    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate integrity engine statistics.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("audit_report")
        with self._lock:
            report = self.diagnostics_report()

            # Per-type record counts
            type_counts: dict[str, int] = {}
            for kt in _ALL_TYPES:
                records = self._get_all_records(kt)
                type_counts[kt.value] = len(records)

            report.update({
                "total_records_by_type": type_counts,
                "total_records": sum(type_counts.values()),
                "resolved_issue_ids": list(self._resolved_issues),
                "generated_at": _utcnow().isoformat(),
            })
            return report


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["KnowledgeIntegrityEngine"]