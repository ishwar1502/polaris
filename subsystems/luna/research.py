"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/research.py

Concrete in-memory implementation of the Research Knowledge Engine.

Manages research-oriented knowledge records: scientific papers, technical
reports, experimental results, literature reviews, datasets, benchmarks, and
white papers.  Primary consumers of this engine are PROMETHEUS and VULCAN.

Responsibilities:
    - Full CRUD lifecycle for ResearchKnowledge records
    - Citation and source tracking via KnowledgeReference
    - Evidence management via KnowledgeEvidence
    - Per-record and engine-wide confidence scoring
    - Content-fingerprint-based duplicate detection
    - Paginated free-text and type-based search
    - Structural validation producing KnowledgeValidationResult
    - Comprehensive audit reporting

Thread safety:
    All public methods acquire self._lock (threading.RLock) before touching
    any internal store.  The lock is re-entrant so that public helpers that
    delegate to other public methods do not deadlock.

Lifecycle contract:
    Call initialize() before any other method.
    Call shutdown() to release resources gracefully.
    All public methods raise LunaNotInitializedError when the engine is not
    in the initialized state.

Part of the POLARIS Cognitive Substrate:
    ASTRA        → Identity
    ECHO         → Experience
    LUNA         → Knowledge       ← this module
    CHRONOS      → Time
    CONSTELLATION → Relationships
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from subsystems.luna.exceptions import (
    LunaNotInitializedError,
    ResearchKnowledgeNotFoundError,
    ResearchValidationError,
)
from subsystems.luna.interfaces import AbstractResearchKnowledgeEngine
from subsystems.luna.models import (
    ConfidenceLevel,
    KnowledgeDifficulty,
    KnowledgeEvidence,
    KnowledgeMetadata,
    KnowledgeReference,
    KnowledgeSourceType,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeValidationResult,
    ResearchKnowledge,
    ResearchType,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
    ValidationStatus,
    _new_id,
    _utcnow,
)

_ENGINE_VERSION = "5.0.0"
_VALIDATOR_VERSION = "research-validator-5.0.0"

# Minimum acceptable confidence score before WARNING is issued
_LOW_CONFIDENCE_THRESHOLD = 0.40
# Minimum acceptable source trust weight before WARNING is issued
_LOW_TRUST_THRESHOLD = 0.50


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


class ResearchKnowledgeEngine(AbstractResearchKnowledgeEngine):
    """
    In-memory v1 implementation of the LUNA Research Knowledge Engine.

    Stores ResearchKnowledge records in a dict keyed by record ID, with
    secondary indexes for research type and domain membership.  All
    mutations are protected by a reentrant lock for full thread safety.

    Supporting structures maintained per record:
        - KnowledgeEvidence  : evidence items attached to the record
        - KnowledgeReference : typed citation links to other records / URIs

    Usage::

        engine = ResearchKnowledgeEngine()
        engine.initialize()

        meta = KnowledgeMetadata.create(
            source="arXiv:2301.00001",
            source_type=KnowledgeSourceType.ACADEMIC_PAPER,
            confidence_score=0.90,
        )
        record = engine.create_research(
            name="Attention is All You Need",
            description="Transformer architecture paper",
            research_type=ResearchType.PAPER,
            difficulty=KnowledgeDifficulty.ADVANCED,
            domain_ids=["domain-ai"],
            metadata=meta,
            abstract="We propose a new architecture based solely on attention.",
            key_findings=["Self-attention replaces recurrence", "BLEU score SOTA"],
        )

        engine.shutdown()
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False

        # Primary store: record_id → ResearchKnowledge
        self._records: dict[str, ResearchKnowledge] = {}

        # Secondary index: research_type.value → set of record IDs
        self._type_index: dict[str, set[str]] = defaultdict(set)

        # Secondary index: domain_id → set of record IDs
        self._domain_index: dict[str, set[str]] = defaultdict(set)

        # Fingerprint index: fingerprint_hash → record_id (first seen)
        self._fingerprint_index: dict[str, str] = {}

        # Supporting structures: record_id → list of KnowledgeEvidence
        self._evidence: dict[str, list[KnowledgeEvidence]] = defaultdict(list)

        # Supporting structures: record_id → list of KnowledgeReference
        self._references: dict[str, list[KnowledgeReference]] = defaultdict(list)

        # Operational counters for diagnostics
        self._mutation_count: int = 0
        self._duplicate_checks: int = 0
        self._last_mutation_at: Optional[datetime] = None
        self._started_at: Optional[datetime] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Prepare the engine for operation.  Idempotent — safe to call multiple
        times on the same instance.
        """
        with self._lock:
            if self._initialized:
                return
            self._records.clear()
            self._type_index.clear()
            self._domain_index.clear()
            self._fingerprint_index.clear()
            self._evidence.clear()
            self._references.clear()
            self._mutation_count = 0
            self._duplicate_checks = 0
            self._last_mutation_at = None
            self._started_at = _now_utc()
            self._initialized = True

    def shutdown(self) -> None:
        """
        Release all in-memory resources.  Idempotent — safe to call on an
        already-stopped engine.
        """
        with self._lock:
            if not self._initialized:
                return
            self._records.clear()
            self._type_index.clear()
            self._domain_index.clear()
            self._fingerprint_index.clear()
            self._evidence.clear()
            self._references.clear()
            self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    # ── Internal guard ────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation)

    # ── Internal index helpers ────────────────────────────────────────────────

    def _index_record(self, record: ResearchKnowledge) -> None:
        """Add a record to all secondary indexes."""
        self._type_index[record.research_type.value].add(record.id)
        for domain_id in record.domain_ids:
            self._domain_index[domain_id].add(record.id)
        self._fingerprint_index.setdefault(record.fingerprint, record.id)

    def _deindex_record(self, record: ResearchKnowledge) -> None:
        """Remove a record from all secondary indexes."""
        self._type_index[record.research_type.value].discard(record.id)
        for domain_id in record.domain_ids:
            self._domain_index[domain_id].discard(record.id)
        # Only remove from fingerprint index if this record is the registered owner
        if self._fingerprint_index.get(record.fingerprint) == record.id:
            del self._fingerprint_index[record.fingerprint]

    def _reindex_record(
        self, old: ResearchKnowledge, new: ResearchKnowledge
    ) -> None:
        """Atomically replace old index entries with those of the updated record."""
        self._deindex_record(old)
        self._index_record(new)

    def _record_mutation(self) -> None:
        self._mutation_count += 1
        self._last_mutation_at = _now_utc()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create_research(
        self,
        name: str,
        description: str,
        research_type: ResearchType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        abstract: str = "",
        key_findings: Optional[list[str]] = None,
        methodology: str = "",
        citation: str = "",
        doi: Optional[str] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> ResearchKnowledge:
        """
        Create and persist a new ResearchKnowledge record.

        Performs content-fingerprint duplicate detection before inserting.
        The record is created in DRAFT status.

        Args:
            name:            Canonical short name (e.g. "Attention is All You Need").
            description:     Human-readable description of what this record contains.
            research_type:   ResearchType classification enum.
            difficulty:      KnowledgeDifficulty tier.
            domain_ids:      Non-empty list of owning domain IDs.
            metadata:        Provenance, confidence, and versioning metadata.
            abstract:        Full abstract text.
            key_findings:    Principal results or conclusions.
            methodology:     Description of the research methodology used.
            citation:        Human-readable citation string (APA, BibTeX, etc.).
            doi:             Optional DOI string (e.g. "10.1234/example").
            aliases:         Alternate names or abbreviations.
            notes:           Free-text notes.

        Returns:
            The newly created ResearchKnowledge record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ResearchValidationError: Provided data fails basic validation.
        """
        with self._lock:
            self._require_initialized("create_research")

            # Pre-insert validation
            violations: list[str] = []
            if not name or not name.strip():
                violations.append("name must not be empty")
            if not domain_ids:
                violations.append("domain_ids must contain at least one entry")
            if not (0.0 <= metadata.confidence_score <= 1.0):
                violations.append(
                    f"confidence_score {metadata.confidence_score} is out of range [0.0, 1.0]"
                )
            if violations:
                raise ResearchValidationError(
                    research_id="<new>",
                    violations=violations,
                )

            record = ResearchKnowledge.create(
                name=name.strip(),
                description=description,
                title=name.strip(),
                research_type=research_type,
                difficulty=difficulty,
                domain_ids=list(domain_ids),
                metadata=metadata,
                abstract=abstract,
                authors=None,
                key_findings=key_findings,
                aliases=aliases,
                notes=notes,
            )
            # Attach additional fields not covered by the factory
            record.methodology = methodology
            record.doi = doi
            # citation stored as the first element of metadata.references if provided
            if citation:
                object.__setattr__(
                    record,
                    "notes",
                    (record.notes + f"\nCitation: {citation}").strip()
                    if record.notes
                    else f"Citation: {citation}",
                )

            # Duplicate detection
            self._duplicate_checks += 1
            existing_id = self._fingerprint_index.get(record.fingerprint)
            if existing_id is not None:
                raise ResearchValidationError(
                    research_id=record.id,
                    violations=[
                        f"A research record with an identical fingerprint already exists: '{existing_id}'"
                    ],
                )

            self._records[record.id] = record
            self._index_record(record)
            self._record_mutation()
            return record

    def update_research(
        self,
        research_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        research_type: Optional[ResearchType] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        domain_ids: Optional[list[str]] = None,
        abstract: Optional[str] = None,
        key_findings: Optional[list[str]] = None,
        methodology: Optional[str] = None,
        citation: Optional[str] = None,
        doi: Optional[str] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
    ) -> ResearchKnowledge:
        """
        Apply a partial update to an existing ResearchKnowledge record.

        Only keyword-supplied fields are modified.  The record's metadata
        version is incremented automatically via bump_version().

        Args:
            research_id: ID of the record to update.
            **fields:    Any subset of ResearchKnowledge fields to overwrite.

        Returns:
            The updated ResearchKnowledge record.

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
            ResearchValidationError:        Updated state fails validation.
        """
        with self._lock:
            self._require_initialized("update_research")

            old = self._records.get(research_id)
            if old is None:
                raise ResearchKnowledgeNotFoundError(research_id)
            if old.status.is_terminal:
                raise ResearchValidationError(
                    research_id=research_id,
                    violations=[
                        f"Cannot update a record in terminal status '{old.status.value}'"
                    ],
                )

            # Build updated record by cloning fields
            new_name = name.strip() if name is not None else old.name
            new_description = description if description is not None else old.description
            new_research_type = research_type if research_type is not None else old.research_type
            new_difficulty = difficulty if difficulty is not None else old.difficulty
            new_domain_ids = list(domain_ids) if domain_ids is not None else list(old.domain_ids)
            new_abstract = abstract if abstract is not None else old.abstract
            new_key_findings = list(key_findings) if key_findings is not None else list(old.key_findings)
            new_methodology = methodology if methodology is not None else old.methodology
            new_doi = doi if doi is not None else old.doi
            new_status = status if status is not None else old.status
            new_notes = notes if notes is not None else old.notes
            new_metadata = (
                metadata
                if metadata is not None
                else old.metadata.bump_version()
            )

            if not new_name:
                raise ResearchValidationError(
                    research_id=research_id,
                    violations=["name must not be empty"],
                )
            if not new_domain_ids:
                raise ResearchValidationError(
                    research_id=research_id,
                    violations=["domain_ids must contain at least one entry"],
                )

            updated = ResearchKnowledge(
                id=old.id,
                knowledge_type=old.knowledge_type,
                name=new_name,
                description=new_description,
                status=new_status,
                difficulty=new_difficulty,
                domain_ids=new_domain_ids,
                metadata=new_metadata,
                aliases=list(old.aliases),
                related_ids=list(old.related_ids),
                notes=new_notes,
                research_type=new_research_type,
                title=new_name,
                abstract=new_abstract,
                authors=list(old.authors),
                publication_venue=old.publication_venue,
                publication_date=old.publication_date,
                doi=new_doi,
                url=old.url,
                key_findings=new_key_findings,
                methodology=new_methodology,
                limitations=list(old.limitations),
                cited_knowledge_ids=list(old.cited_knowledge_ids),
                extracted_fact_ids=list(old.extracted_fact_ids),
                extracted_concept_ids=list(old.extracted_concept_ids),
                citation_count=old.citation_count,
            )

            if citation is not None:
                updated.notes = (
                    (updated.notes + f"\nCitation: {citation}").strip()
                    if updated.notes
                    else f"Citation: {citation}"
                )

            self._reindex_record(old, updated)
            self._records[research_id] = updated
            self._record_mutation()
            return updated

    def delete_research(
        self, research_id: str, *, reason: str = ""
    ) -> ResearchKnowledge:
        """
        Soft-delete a research record by transitioning its status to RETRACTED.

        The record is never physically removed; all history and downstream
        references remain coherent.  The deletion reason is appended to notes.

        Args:
            research_id: ID of the record to retract.
            reason:      Human-readable reason (stored in notes).

        Returns:
            The retracted ResearchKnowledge record.

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("delete_research")

            record = self._records.get(research_id)
            if record is None:
                raise ResearchKnowledgeNotFoundError(research_id)
            if record.status == KnowledgeStatus.RETRACTED:
                return record

            appended_notes = record.notes
            if reason:
                appended_notes = (
                    f"{appended_notes}\n[RETRACTED] {reason}".strip()
                    if appended_notes
                    else f"[RETRACTED] {reason}"
                )

            retracted = ResearchKnowledge(
                id=record.id,
                knowledge_type=record.knowledge_type,
                name=record.name,
                description=record.description,
                status=KnowledgeStatus.RETRACTED,
                difficulty=record.difficulty,
                domain_ids=list(record.domain_ids),
                metadata=record.metadata.bump_version(),
                aliases=list(record.aliases),
                related_ids=list(record.related_ids),
                notes=appended_notes,
                research_type=record.research_type,
                title=record.title,
                abstract=record.abstract,
                authors=list(record.authors),
                publication_venue=record.publication_venue,
                publication_date=record.publication_date,
                doi=record.doi,
                url=record.url,
                key_findings=list(record.key_findings),
                methodology=record.methodology,
                limitations=list(record.limitations),
                cited_knowledge_ids=list(record.cited_knowledge_ids),
                extracted_fact_ids=list(record.extracted_fact_ids),
                extracted_concept_ids=list(record.extracted_concept_ids),
                citation_count=record.citation_count,
            )

            self._reindex_record(record, retracted)
            self._records[research_id] = retracted
            self._record_mutation()
            return retracted

    def retrieve_research(self, research_id: str) -> ResearchKnowledge:
        """
        Fetch a single ResearchKnowledge record by its unique ID.

        Args:
            research_id: The record's UUID string.

        Returns:
            The matching ResearchKnowledge record.

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("retrieve_research")
            record = self._records.get(research_id)
            if record is None:
                raise ResearchKnowledgeNotFoundError(research_id)
            return record

    # ── Search ────────────────────────────────────────────────────────────────

    def search_research(
        self,
        query: str,
        *,
        domain_ids: Optional[list[str]] = None,
        research_types: Optional[list[ResearchType]] = None,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ResearchKnowledge]:
        """
        Full-text search across name, abstract, key_findings, and description.

        Args:
            query:          Case-insensitive substring matched against name,
                            abstract, description, and key_findings.
            domain_ids:     Restrict to records belonging to these domains.
            research_types: Restrict to these ResearchType values.
            status_filter:  Restrict to these KnowledgeStatus values.
            min_confidence: Minimum metadata.confidence_score threshold.
            limit:          Maximum number of results to return.
            offset:         Pagination offset.

        Returns:
            Matching records sorted by confidence score descending.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("search_research")

            needle = query.lower().strip()
            results: list[ResearchKnowledge] = []

            # Start from domain-restricted candidate set if domain filter supplied
            if domain_ids:
                candidate_ids: set[str] = set()
                for did in domain_ids:
                    candidate_ids |= self._domain_index.get(did, set())
                candidates = [self._records[rid] for rid in candidate_ids if rid in self._records]
            else:
                candidates = list(self._records.values())

            type_set = {rt.value for rt in research_types} if research_types else None
            status_set = {s.value for s in status_filter} if status_filter else None

            for record in candidates:
                if type_set and record.research_type.value not in type_set:
                    continue
                if status_set and record.status.value not in status_set:
                    continue
                if record.metadata.confidence_score < min_confidence:
                    continue
                if needle:
                    haystack = " ".join([
                        record.name,
                        record.description,
                        record.abstract,
                        record.title,
                        " ".join(record.key_findings),
                        " ".join(record.aliases),
                    ]).lower()
                    if needle not in haystack:
                        continue
                results.append(record)

            results.sort(key=lambda r: r.metadata.confidence_score, reverse=True)
            return results[offset: offset + limit]

    def get_all_research(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[ResearchKnowledge]:
        """
        Return a paginated, optionally status-filtered slice of all records.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_all_research")

            status_set = {s.value for s in status_filter} if status_filter else None
            records = [
                r for r in self._records.values()
                if status_set is None or r.status.value in status_set
            ]
            records.sort(key=lambda r: r.metadata.created_at, reverse=True)
            return records[offset: offset + limit]

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_research(self, research_id: str) -> KnowledgeValidationResult:
        """
        Run a full structural and semantic validation pass on a single record.

        Checks performed:
            - Non-empty name and description
            - At least one domain_id present
            - confidence_score within [0.0, 1.0]
            - Abstract non-empty for PAPER and TECHNICAL_REPORT types
            - At least one key_finding for non-DATASET types
            - Source trust weight above minimum threshold
            - Staleness / scheduled review

        Returns:
            A KnowledgeValidationResult capturing pass/fail and all issues.

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("validate_research")

            record = self._records.get(research_id)
            if record is None:
                raise ResearchKnowledgeNotFoundError(research_id)

            issues: list[ValidationIssue] = []

            # 1. Mandatory field checks
            if not record.name or not record.name.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Research record has an empty name.",
                    field="name",
                    suggestion="Provide a canonical name for the research record.",
                ))

            if not record.description or not record.description.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Research record has an empty description.",
                    field="description",
                    suggestion="Provide a human-readable description.",
                ))

            if not record.domain_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Research record has no domain associations.",
                    field="domain_ids",
                    suggestion="Associate the record with at least one knowledge domain.",
                ))

            # 2. Confidence score bounds
            cs = record.metadata.confidence_score
            if not (0.0 <= cs <= 1.0):
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.UNVERIFIED_CLAIM,
                    severity=ValidationSeverity.ERROR,
                    message=f"confidence_score {cs} is outside [0.0, 1.0].",
                    field="metadata.confidence_score",
                    suggestion="Clamp the confidence score to the valid range.",
                ))
            elif cs < _LOW_CONFIDENCE_THRESHOLD:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.LOW_CONFIDENCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"confidence_score {cs:.3f} is below the low-confidence threshold "
                             f"({_LOW_CONFIDENCE_THRESHOLD}).",
                    field="metadata.confidence_score",
                    suggestion="Improve evidence or re-assess confidence.",
                ))

            # 3. Source trust
            tw = record.metadata.source_type.trust_weight
            if tw < _LOW_TRUST_THRESHOLD:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.UNRELIABLE_SOURCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Source type '{record.metadata.source_type.value}' has a low "
                             f"trust weight ({tw:.2f}).",
                    field="metadata.source_type",
                    suggestion="Reference a higher-trust source where possible.",
                ))

            # 4. Abstract completeness for formal research types
            paper_types = {ResearchType.PAPER, ResearchType.TECHNICAL_REPORT, ResearchType.WHITE_PAPER}
            if record.research_type in paper_types and not record.abstract.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Research type '{record.research_type.value}' should include an abstract.",
                    field="abstract",
                    suggestion="Add an abstract summarising the research.",
                ))

            # 5. Key findings (skip DATASET and BENCHMARK — they use metrics instead)
            finding_optional_types = {ResearchType.DATASET, ResearchType.BENCHMARK}
            if record.research_type not in finding_optional_types and not record.key_findings:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Research record has no key_findings recorded.",
                    field="key_findings",
                    suggestion="List the principal findings or conclusions.",
                ))

            # 6. DOI validity for PAPER type
            if record.research_type == ResearchType.PAPER and record.doi is not None:
                doi_val = record.doi.strip()
                if not doi_val.startswith("10."):
                    issues.append(ValidationIssue.create(
                        issue_type=ValidationIssueType.FORMAT_ERROR,
                        severity=ValidationSeverity.WARNING,
                        message=f"DOI '{doi_val}' does not follow the standard '10.' prefix format.",
                        field="doi",
                        suggestion="Verify and correct the DOI.",
                    ))

            # 7. Staleness
            if record.metadata.is_stale:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.STALE_KNOWLEDGE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Research record is past its scheduled review date "
                             f"({record.metadata.review_date.isoformat() if record.metadata.review_date else 'unknown'}).",
                    field="metadata.review_date",
                    suggestion="Re-validate and update the review date.",
                ))

            # 8. Terminal status guard
            if record.status.is_terminal:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.INFO,
                    message=f"Record is in terminal status '{record.status.value}' and should not be referenced.",
                    field="status",
                    suggestion="Ensure downstream consumers do not reference retracted records.",
                ))

            return KnowledgeValidationResult.create(
                knowledge_id=research_id,
                knowledge_name=record.name,
                issues=issues,
                validator_version=_VALIDATOR_VERSION,
            )

    # ── Evidence management ───────────────────────────────────────────────────

    def add_evidence(
        self,
        research_id: str,
        evidence_type: str,
        source: str,
        description: str,
        source_type: KnowledgeSourceType = KnowledgeSourceType.UNKNOWN,
        confidence: float = 0.75,
    ) -> KnowledgeEvidence:
        """
        Attach a KnowledgeEvidence item to a research record.

        Args:
            research_id:  ID of the target research record.
            evidence_type: "supporting" | "challenging" | "neutral"
            source:       Source label or URI.
            description:  Human-readable description of the evidence.
            source_type:  KnowledgeSourceType classification.
            confidence:   Evidence confidence (0.0–1.0).

        Returns:
            The newly created KnowledgeEvidence item.

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("add_evidence")
            if research_id not in self._records:
                raise ResearchKnowledgeNotFoundError(research_id)

            evidence = KnowledgeEvidence.create(
                knowledge_id=research_id,
                evidence_type=evidence_type,
                source=source,
                description=description,
                source_type=source_type,
                confidence=confidence,
            )
            self._evidence[research_id].append(evidence)
            self._record_mutation()
            return evidence

    def get_evidence(self, research_id: str) -> list[KnowledgeEvidence]:
        """
        Return all evidence items attached to a research record.

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("get_evidence")
            if research_id not in self._records:
                raise ResearchKnowledgeNotFoundError(research_id)
            return list(self._evidence.get(research_id, []))

    # ── Citation / Reference management ──────────────────────────────────────

    def add_citation(
        self,
        research_id: str,
        target: str,
        reference_type: str,
        description: str = "",
    ) -> KnowledgeReference:
        """
        Attach a KnowledgeReference (citation link) to a research record.

        Args:
            research_id:    ID of the source research record.
            target:         Target record ID or external URI.
            reference_type: "supports" | "contradicts" | "elaborates" | "cites"
            description:    Optional human-readable annotation.

        Returns:
            The newly created KnowledgeReference.

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("add_citation")
            if research_id not in self._records:
                raise ResearchKnowledgeNotFoundError(research_id)

            ref = KnowledgeReference.create(
                source_id=research_id,
                target=target,
                reference_type=reference_type,
                description=description,
            )
            self._references[research_id].append(ref)

            # Keep cited_knowledge_ids in sync for internal links
            record = self._records[research_id]
            if target in self._records and target not in record.cited_knowledge_ids:
                record.cited_knowledge_ids.append(target)

            self._record_mutation()
            return ref

    def get_citations(self, research_id: str) -> list[KnowledgeReference]:
        """
        Return all citation references attached to a research record.

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("get_citations")
            if research_id not in self._records:
                raise ResearchKnowledgeNotFoundError(research_id)
            return list(self._references.get(research_id, []))

    def get_citation_network(self, research_id: str) -> dict[str, Any]:
        """
        Return a citation graph summary for the given record.

        The returned dict contains:
            source_id:    The root research record ID.
            outbound:     List of records this record cites.
            inbound:      List of records that cite this record.

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("get_citation_network")
            if research_id not in self._records:
                raise ResearchKnowledgeNotFoundError(research_id)

            outbound = [ref.target for ref in self._references.get(research_id, [])]
            inbound: list[str] = [
                rid
                for rid, refs in self._references.items()
                if rid != research_id and any(r.target == research_id for r in refs)
            ]
            return {
                "source_id": research_id,
                "outbound": outbound,
                "inbound": inbound,
            }

    # ── Confidence scoring ────────────────────────────────────────────────────

    def compute_confidence_score(self, research_id: str) -> float:
        """
        Compute an enriched effective confidence score for a research record.

        Incorporates:
            - metadata.confidence_score (base)
            - source trust_weight
            - evidence corroboration (bonus for ≥2 supporting evidence items)
            - citation count bonus (capped at 0.05)

        Returns:
            A float in [0.0, 1.0].

        Raises:
            LunaNotInitializedError:        Engine not initialized.
            ResearchKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("compute_confidence_score")
            record = self._records.get(research_id)
            if record is None:
                raise ResearchKnowledgeNotFoundError(research_id)

            base = record.metadata.effective_trust
            evidence_items = self._evidence.get(research_id, [])
            supporting = sum(1 for e in evidence_items if e.is_supporting)
            corroboration_bonus = min(0.05, supporting * 0.01)
            citation_bonus = min(0.05, record.citation_count * 0.001)
            score = base + corroboration_bonus + citation_bonus
            return max(0.0, min(1.0, score))

    # ── Indexing helpers ──────────────────────────────────────────────────────

    def search_by_research_type(
        self,
        research_type: ResearchType,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[ResearchKnowledge]:
        """
        Return all records of a specific ResearchType.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("search_by_research_type")

            status_set = {s.value for s in status_filter} if status_filter else None
            ids = self._type_index.get(research_type.value, set())
            records = [
                self._records[rid]
                for rid in ids
                if rid in self._records
                and (status_set is None or self._records[rid].status.value in status_set)
            ]
            records.sort(key=lambda r: r.metadata.confidence_score, reverse=True)
            return records[offset: offset + limit]

    def search_by_domain(
        self,
        domain_id: str,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[ResearchKnowledge]:
        """
        Return all records associated with a given knowledge domain.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("search_by_domain")

            status_set = {s.value for s in status_filter} if status_filter else None
            ids = self._domain_index.get(domain_id, set())
            records = [
                self._records[rid]
                for rid in ids
                if rid in self._records
                and (status_set is None or self._records[rid].status.value in status_set)
            ]
            records.sort(key=lambda r: r.metadata.confidence_score, reverse=True)
            return records[offset: offset + limit]

    # ── Duplicate detection ───────────────────────────────────────────────────

    def find_duplicate_research(self) -> list[list[ResearchKnowledge]]:
        """
        Return groups of research records that share the same content fingerprint.

        Each group contains two or more records that are semantically identical
        by fingerprint.  The caller should decide which record to keep and
        retract the rest.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("find_duplicate_research")

            fingerprint_map: dict[str, list[ResearchKnowledge]] = defaultdict(list)
            for record in self._records.values():
                fingerprint_map[record.fingerprint].append(record)

            return [
                group
                for group in fingerprint_map.values()
                if len(group) >= 2
            ]

    # ── Existence / count helpers ─────────────────────────────────────────────

    def research_exists(self, research_id: str) -> bool:
        """
        Return True if a research record with the given ID exists (any status).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("research_exists")
            return research_id in self._records

    def get_research_count(self, *, active_only: bool = False) -> int:
        """
        Return the total number of research records in the store.

        Args:
            active_only: When True, count only VALIDATED and ACTIVE records.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_research_count")
            if not active_only:
                return len(self._records)
            return sum(1 for r in self._records.values() if r.status.is_usable)

    # ── Observability ─────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """
        Return a lightweight liveness/readiness summary.

        Required keys: engine, initialized, record_count, status.
        """
        with self._lock:
            total = len(self._records)
            active = sum(1 for r in self._records.values() if r.status.is_usable)
            return {
                "engine": self.__class__.__qualname__,
                "initialized": self._initialized,
                "record_count": total,
                "active_record_count": active,
                "status": "healthy" if self._initialized else "offline",
                "engine_version": _ENGINE_VERSION,
            }

    def diagnostics_report(self) -> dict[str, Any]:
        """
        Return a full introspection snapshot for operator debugging.
        """
        with self._lock:
            total = len(self._records)
            active = sum(1 for r in self._records.values() if r.status.is_usable)
            type_breakdown = {
                rt: len(ids) for rt, ids in self._type_index.items() if ids
            }
            return {
                "engine": self.__class__.__qualname__,
                "initialized": self._initialized,
                "record_count": total,
                "active_record_count": active,
                "status": "healthy" if self._initialized else "offline",
                "engine_version": _ENGINE_VERSION,
                "index_size": len(self._fingerprint_index),
                "duplicate_checks": self._duplicate_checks,
                "mutation_count": self._mutation_count,
                "last_mutation_at": (
                    self._last_mutation_at.isoformat()
                    if self._last_mutation_at
                    else None
                ),
                "started_at": (
                    self._started_at.isoformat() if self._started_at else None
                ),
                "type_breakdown": type_breakdown,
                "domain_count": len(
                    [d for d, ids in self._domain_index.items() if ids]
                ),
                "evidence_item_count": sum(
                    len(v) for v in self._evidence.values()
                ),
                "reference_count": sum(
                    len(v) for v in self._references.values()
                ),
            }

    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate statistics for the research knowledge store.

        Includes status breakdown, confidence distribution, and type coverage.
        """
        with self._lock:
            self._require_initialized("audit_report")

            total = len(self._records)
            status_breakdown: dict[str, int] = defaultdict(int)
            type_breakdown: dict[str, int] = defaultdict(int)
            confidence_sum = 0.0
            low_confidence_ids: list[str] = []
            stale_ids: list[str] = []

            for record in self._records.values():
                status_breakdown[record.status.value] += 1
                type_breakdown[record.research_type.value] += 1
                confidence_sum += record.metadata.confidence_score
                if record.metadata.confidence_score < _LOW_CONFIDENCE_THRESHOLD:
                    low_confidence_ids.append(record.id)
                if record.metadata.is_stale:
                    stale_ids.append(record.id)

            avg_confidence = confidence_sum / total if total > 0 else 0.0

            return {
                "engine": self.__class__.__qualname__,
                "engine_version": _ENGINE_VERSION,
                "total_records": total,
                "status_breakdown": dict(status_breakdown),
                "type_breakdown": dict(type_breakdown),
                "average_confidence": round(avg_confidence, 4),
                "low_confidence_count": len(low_confidence_ids),
                "low_confidence_ids": low_confidence_ids,
                "stale_count": len(stale_ids),
                "stale_ids": stale_ids,
                "duplicate_checks_performed": self._duplicate_checks,
                "mutation_count": self._mutation_count,
                "evidence_items_total": sum(len(v) for v in self._evidence.values()),
                "references_total": sum(len(v) for v in self._references.values()),
                "generated_at": _now_utc().isoformat(),
            }


__all__ = ["ResearchKnowledgeEngine"]