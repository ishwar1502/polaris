"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/education.py

Concrete in-memory implementation of the Educational Knowledge Engine.

Manages educational and instructional knowledge records: courses, learning
paths, tutorials, study guides, training materials, assessments, and
curricula.  The primary consumer of this engine is APOLLO.

Responsibilities:
    - Full CRUD lifecycle for EducationalKnowledge records
    - Curriculum management: group related educational records under a domain
    - Prerequisite management: track learning dependencies between records
    - Learning path management: ordered lists of educational records per domain
    - Difficulty progression: ensure prerequisite difficulty is ≤ dependent
    - Structural validation producing KnowledgeValidationResult
    - Paginated free-text, type-based, and domain-based search
    - Comprehensive audit reporting

Thread safety:
    All public methods acquire self._lock (threading.RLock) before touching
    any internal store.  The lock is re-entrant so helper methods that call
    other public methods do not deadlock.

Lifecycle contract:
    Call initialize() before any other method.
    Call shutdown() to release resources gracefully.
    All public methods raise LunaNotInitializedError when the engine is not in
    the initialized state.

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
    EducationalKnowledgeNotFoundError,
    EducationalKnowledgeValidationError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractEducationalKnowledgeEngine
from subsystems.luna.models import (
    ContentSection,
    EducationType,
    KnowledgeDifficulty,
    KnowledgeMetadata,
    KnowledgeSourceType,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeValidationResult,
    SkillPrerequisite,
    SkillProgressionModel,
    EducationalKnowledge,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
    ValidationStatus,
    _new_id,
    _utcnow,
)

_ENGINE_VERSION = "5.0.0"
_VALIDATOR_VERSION = "educational-validator-5.0.0"

_LOW_CONFIDENCE_THRESHOLD = 0.40
_LOW_TRUST_THRESHOLD = 0.50

# Minimum number of learning objectives expected for comprehensive content types
_COMPREHENSIVE_MIN_OBJECTIVES = 2
_COMPREHENSIVE_MIN_SECTIONS = 3


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


class EducationalKnowledgeEngine(AbstractEducationalKnowledgeEngine):
    """
    In-memory v1 implementation of the LUNA Educational Knowledge Engine.

    Stores EducationalKnowledge records keyed by record ID with secondary
    indexes for education type and domain membership.  Supporting structures
    (SkillPrerequisite, SkillProgressionModel) are stored per record.

    Usage::

        engine = EducationalKnowledgeEngine()
        engine.initialize()

        meta = KnowledgeMetadata.create(
            source="Internal Curriculum v3",
            source_type=KnowledgeSourceType.TECHNICAL_DOCUMENTATION,
            confidence_score=0.88,
        )
        record = engine.create_educational(
            name="Introduction to Python",
            description="Beginner Python programming course",
            education_type=EducationType.LEARNING_PATH,
            difficulty=KnowledgeDifficulty.BEGINNER,
            domain_ids=["domain-programming"],
            metadata=meta,
            learning_objectives=[
                "Understand basic Python syntax",
                "Write simple scripts",
            ],
            estimated_duration_hours=20.0,
        )

        engine.shutdown()
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False

        # Primary store: record_id → EducationalKnowledge
        self._records: dict[str, EducationalKnowledge] = {}

        # Secondary index: education_type.value → set of record IDs
        self._type_index: dict[str, set[str]] = defaultdict(set)

        # Secondary index: domain_id → set of record IDs
        self._domain_index: dict[str, set[str]] = defaultdict(set)

        # Fingerprint index: fingerprint_hash → record_id (first seen)
        self._fingerprint_index: dict[str, str] = {}

        # Prerequisite graph: record_id → list[SkillPrerequisite]
        self._prerequisites: dict[str, list[SkillPrerequisite]] = defaultdict(list)

        # Skill progression models: record_id → SkillProgressionModel
        # One optional progression model can be attached per educational record
        self._progression_models: dict[str, SkillProgressionModel] = {}

        # Curriculum grouping: curriculum_name → ordered list of record IDs
        # This allows callers to associate educational records with a named curriculum
        self._curricula: dict[str, list[str]] = {}

        # Operational counters
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
            self._prerequisites.clear()
            self._progression_models.clear()
            self._curricula.clear()
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
            self._prerequisites.clear()
            self._progression_models.clear()
            self._curricula.clear()
            self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    # ── Internal guard ────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation)

    # ── Internal index helpers ────────────────────────────────────────────────

    def _index_record(self, record: EducationalKnowledge) -> None:
        self._type_index[record.education_type.value].add(record.id)
        for domain_id in record.domain_ids:
            self._domain_index[domain_id].add(record.id)
        self._fingerprint_index.setdefault(record.fingerprint, record.id)

    def _deindex_record(self, record: EducationalKnowledge) -> None:
        self._type_index[record.education_type.value].discard(record.id)
        for domain_id in record.domain_ids:
            self._domain_index[domain_id].discard(record.id)
        if self._fingerprint_index.get(record.fingerprint) == record.id:
            del self._fingerprint_index[record.fingerprint]

    def _reindex_record(
        self, old: EducationalKnowledge, new: EducationalKnowledge
    ) -> None:
        self._deindex_record(old)
        self._index_record(new)

    def _record_mutation(self) -> None:
        self._mutation_count += 1
        self._last_mutation_at = _now_utc()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create_educational(
        self,
        name: str,
        description: str,
        education_type: EducationType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        learning_objectives: Optional[list[str]] = None,
        prerequisite_ids: Optional[list[str]] = None,
        estimated_duration_hours: Optional[float] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> EducationalKnowledge:
        """
        Create and persist a new EducationalKnowledge record.

        Performs content-fingerprint duplicate detection before inserting.
        The record is created in DRAFT status.

        Args:
            name:                     Canonical short name.
            description:              Human-readable description.
            education_type:           EducationType classification enum.
            difficulty:               KnowledgeDifficulty tier.
            domain_ids:               Non-empty list of owning domain IDs.
            metadata:                 Provenance, confidence, and versioning metadata.
            learning_objectives:      Ordered list of objectives the learner achieves.
            prerequisite_ids:         IDs of records that must be completed first.
            estimated_duration_hours: Total estimated learning time in hours.
            aliases:                  Alternate names or abbreviations.
            notes:                    Free-text notes.

        Returns:
            The newly created EducationalKnowledge record.

        Raises:
            LunaNotInitializedError:            Engine not initialized.
            EducationalKnowledgeValidationError: Provided data fails validation.
        """
        with self._lock:
            self._require_initialized("create_educational")

            violations: list[str] = []
            if not name or not name.strip():
                violations.append("name must not be empty")
            if not domain_ids:
                violations.append("domain_ids must contain at least one entry")
            if not (0.0 <= metadata.confidence_score <= 1.0):
                violations.append(
                    f"confidence_score {metadata.confidence_score} is out of range [0.0, 1.0]"
                )
            if estimated_duration_hours is not None and estimated_duration_hours < 0:
                violations.append("estimated_duration_hours must not be negative")
            if violations:
                raise EducationalKnowledgeValidationError(
                    content_id="<new>",
                    violations=violations,
                )

            record = EducationalKnowledge.create(
                name=name.strip(),
                description=description,
                education_type=education_type,
                difficulty=difficulty,
                domain_ids=list(domain_ids),
                metadata=metadata,
                learning_objectives=learning_objectives,
                target_skill_ids=None,
                target_concept_ids=None,
                aliases=aliases,
                notes=notes,
            )
            if prerequisite_ids:
                record.prerequisite_knowledge_ids = list(prerequisite_ids)
            if estimated_duration_hours is not None:
                record.estimated_duration_hours = estimated_duration_hours

            # Duplicate detection
            self._duplicate_checks += 1
            existing_id = self._fingerprint_index.get(record.fingerprint)
            if existing_id is not None:
                raise EducationalKnowledgeValidationError(
                    content_id=record.id,
                    violations=[
                        f"An educational record with an identical fingerprint already exists: '{existing_id}'"
                    ],
                )

            self._records[record.id] = record
            self._index_record(record)
            self._record_mutation()
            return record

    def update_educational(
        self,
        educational_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        education_type: Optional[EducationType] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        domain_ids: Optional[list[str]] = None,
        learning_objectives: Optional[list[str]] = None,
        prerequisite_ids: Optional[list[str]] = None,
        estimated_duration_hours: Optional[float] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
    ) -> EducationalKnowledge:
        """
        Apply a partial update to an existing EducationalKnowledge record.

        Only keyword-supplied fields are modified.  The record's metadata
        version is incremented automatically via bump_version().

        Args:
            educational_id: ID of the record to update.
            **fields:       Any subset of EducationalKnowledge fields to overwrite.

        Returns:
            The updated EducationalKnowledge record.

        Raises:
            LunaNotInitializedError:             Engine not initialized.
            EducationalKnowledgeNotFoundError:   Record not found.
            EducationalKnowledgeValidationError: Updated state fails validation.
        """
        with self._lock:
            self._require_initialized("update_educational")

            old = self._records.get(educational_id)
            if old is None:
                raise EducationalKnowledgeNotFoundError(educational_id)
            if old.status.is_terminal:
                raise EducationalKnowledgeValidationError(
                    content_id=educational_id,
                    violations=[
                        f"Cannot update a record in terminal status '{old.status.value}'"
                    ],
                )

            new_name = name.strip() if name is not None else old.name
            new_description = description if description is not None else old.description
            new_education_type = education_type if education_type is not None else old.education_type
            new_difficulty = difficulty if difficulty is not None else old.difficulty
            new_domain_ids = list(domain_ids) if domain_ids is not None else list(old.domain_ids)
            new_objectives = list(learning_objectives) if learning_objectives is not None else list(old.learning_objectives)
            new_prereq_ids = list(prerequisite_ids) if prerequisite_ids is not None else list(old.prerequisite_knowledge_ids)
            new_duration = estimated_duration_hours if estimated_duration_hours is not None else old.estimated_duration_hours
            new_status = status if status is not None else old.status
            new_notes = notes if notes is not None else old.notes
            new_metadata = (
                metadata if metadata is not None else old.metadata.bump_version()
            )

            if not new_name:
                raise EducationalKnowledgeValidationError(
                    content_id=educational_id,
                    violations=["name must not be empty"],
                )
            if not new_domain_ids:
                raise EducationalKnowledgeValidationError(
                    content_id=educational_id,
                    violations=["domain_ids must contain at least one entry"],
                )
            if new_duration is not None and new_duration < 0:
                raise EducationalKnowledgeValidationError(
                    content_id=educational_id,
                    violations=["estimated_duration_hours must not be negative"],
                )

            updated = EducationalKnowledge(
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
                education_type=new_education_type,
                learning_objectives=new_objectives,
                prerequisite_knowledge_ids=new_prereq_ids,
                target_skill_ids=list(old.target_skill_ids),
                target_concept_ids=list(old.target_concept_ids),
                estimated_duration_hours=new_duration,
                assessment_type=old.assessment_type,
                learning_outcomes=list(old.learning_outcomes),
                content_sections=list(old.content_sections),
                is_self_contained=old.is_self_contained,
            )

            self._reindex_record(old, updated)
            self._records[educational_id] = updated
            self._record_mutation()
            return updated

    def delete_educational(
        self, educational_id: str, *, reason: str = ""
    ) -> EducationalKnowledge:
        """
        Soft-delete an educational record by transitioning its status to RETRACTED.

        The record is never physically removed.  The deletion reason is appended
        to notes.

        Args:
            educational_id: ID of the record to retract.
            reason:         Human-readable reason stored in notes.

        Returns:
            The retracted EducationalKnowledge record.

        Raises:
            LunaNotInitializedError:           Engine not initialized.
            EducationalKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("delete_educational")

            record = self._records.get(educational_id)
            if record is None:
                raise EducationalKnowledgeNotFoundError(educational_id)
            if record.status == KnowledgeStatus.RETRACTED:
                return record

            appended_notes = record.notes
            if reason:
                appended_notes = (
                    f"{appended_notes}\n[RETRACTED] {reason}".strip()
                    if appended_notes
                    else f"[RETRACTED] {reason}"
                )

            retracted = EducationalKnowledge(
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
                education_type=record.education_type,
                learning_objectives=list(record.learning_objectives),
                prerequisite_knowledge_ids=list(record.prerequisite_knowledge_ids),
                target_skill_ids=list(record.target_skill_ids),
                target_concept_ids=list(record.target_concept_ids),
                estimated_duration_hours=record.estimated_duration_hours,
                assessment_type=record.assessment_type,
                learning_outcomes=list(record.learning_outcomes),
                content_sections=list(record.content_sections),
                is_self_contained=record.is_self_contained,
            )

            self._reindex_record(record, retracted)
            self._records[educational_id] = retracted
            self._record_mutation()
            return retracted

    def retrieve_educational(self, educational_id: str) -> EducationalKnowledge:
        """
        Fetch a single EducationalKnowledge record by its unique ID.

        Raises:
            LunaNotInitializedError:           Engine not initialized.
            EducationalKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("retrieve_educational")
            record = self._records.get(educational_id)
            if record is None:
                raise EducationalKnowledgeNotFoundError(educational_id)
            return record

    # ── Search ────────────────────────────────────────────────────────────────

    def search_educational(
        self,
        query: str,
        *,
        domain_ids: Optional[list[str]] = None,
        education_types: Optional[list[EducationType]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EducationalKnowledge]:
        """
        Full-text search across name, description, learning_objectives, and aliases.

        Args:
            query:           Case-insensitive substring.
            domain_ids:      Restrict to records belonging to these domains.
            education_types: Restrict to these EducationType values.
            difficulty:      Exact difficulty tier filter.
            status_filter:   Restrict to these KnowledgeStatus values.
            limit:           Maximum number of results.
            offset:          Pagination offset.

        Returns:
            Matching records sorted by confidence descending.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("search_educational")

            needle = query.lower().strip()

            if domain_ids:
                candidate_ids: set[str] = set()
                for did in domain_ids:
                    candidate_ids |= self._domain_index.get(did, set())
                candidates = [
                    self._records[rid]
                    for rid in candidate_ids
                    if rid in self._records
                ]
            else:
                candidates = list(self._records.values())

            type_set = {et.value for et in education_types} if education_types else None
            status_set = {s.value for s in status_filter} if status_filter else None

            results: list[EducationalKnowledge] = []
            for record in candidates:
                if type_set and record.education_type.value not in type_set:
                    continue
                if difficulty and record.difficulty != difficulty:
                    continue
                if status_set and record.status.value not in status_set:
                    continue
                if needle:
                    haystack = " ".join([
                        record.name,
                        record.description,
                        " ".join(record.learning_objectives),
                        " ".join(record.learning_outcomes),
                        " ".join(record.aliases),
                    ]).lower()
                    if needle not in haystack:
                        continue
                results.append(record)

            results.sort(key=lambda r: r.metadata.confidence_score, reverse=True)
            return results[offset: offset + limit]

    def get_all_educational(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[EducationalKnowledge]:
        """
        Return a paginated, optionally status-filtered slice of all records.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_all_educational")

            status_set = {s.value for s in status_filter} if status_filter else None
            records = [
                r for r in self._records.values()
                if status_set is None or r.status.value in status_set
            ]
            records.sort(key=lambda r: r.metadata.created_at, reverse=True)
            return records[offset: offset + limit]

    # ── Learning path management ──────────────────────────────────────────────

    def get_learning_path(
        self,
        domain_id: str,
        target_difficulty: KnowledgeDifficulty,
    ) -> list[EducationalKnowledge]:
        """
        Return an ordered learning path for a domain up to a target difficulty.

        Records are filtered to the given domain and to difficulty levels ≤
        target_difficulty, then sorted by difficulty rank (ascending) so the
        learner progresses from foundational to the target level.  Within the
        same difficulty, records are sorted by estimated_duration_hours.

        Only VALIDATED and ACTIVE records are included.

        Args:
            domain_id:          The domain to build a path for.
            target_difficulty:  The highest difficulty level to include.

        Returns:
            Ordered list of EducationalKnowledge records.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_learning_path")

            ids = self._domain_index.get(domain_id, set())
            path: list[EducationalKnowledge] = []
            for rid in ids:
                record = self._records.get(rid)
                if record is None:
                    continue
                if not record.status.is_usable:
                    continue
                if record.difficulty.rank > target_difficulty.rank:
                    continue
                path.append(record)

            path.sort(
                key=lambda r: (
                    r.difficulty.rank,
                    r.estimated_duration_hours or 0.0,
                )
            )
            return path

    # ── Prerequisite management ───────────────────────────────────────────────

    def add_prerequisite(
        self,
        educational_id: str,
        prerequisite_skill_id: str,
        minimum_level: "SkillLevel",  # type: ignore[name-defined]  # imported via models
        is_mandatory: bool = True,
        rationale: str = "",
    ) -> SkillPrerequisite:
        """
        Attach a SkillPrerequisite to an educational record.

        The prerequisite expresses: "To engage with this educational content,
        the learner must have achieved <minimum_level> in <prerequisite_skill_id>."

        Args:
            educational_id:          ID of the target educational record.
            prerequisite_skill_id:   LUNA skill ID required as prerequisite.
            minimum_level:           The required SkillLevel.
            is_mandatory:            True = blocking requirement, False = recommended.
            rationale:               Human-readable justification.

        Returns:
            The newly created SkillPrerequisite.

        Raises:
            LunaNotInitializedError:           Engine not initialized.
            EducationalKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("add_prerequisite")
            if educational_id not in self._records:
                raise EducationalKnowledgeNotFoundError(educational_id)

            prereq = SkillPrerequisite.create(
                skill_id=educational_id,
                prerequisite_skill_id=prerequisite_skill_id,
                minimum_level=minimum_level,
                is_mandatory=is_mandatory,
                rationale=rationale,
            )
            self._prerequisites[educational_id].append(prereq)
            self._record_mutation()
            return prereq

    def get_prerequisites(
        self, educational_id: str
    ) -> list[SkillPrerequisite]:
        """
        Return all SkillPrerequisite items attached to an educational record.

        Raises:
            LunaNotInitializedError:           Engine not initialized.
            EducationalKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("get_prerequisites")
            if educational_id not in self._records:
                raise EducationalKnowledgeNotFoundError(educational_id)
            return list(self._prerequisites.get(educational_id, []))

    def remove_prerequisite(
        self, educational_id: str, prerequisite_id: str
    ) -> bool:
        """
        Remove a specific SkillPrerequisite from an educational record.

        Args:
            educational_id:   ID of the educational record.
            prerequisite_id:  ID of the SkillPrerequisite to remove.

        Returns:
            True if the prerequisite was found and removed, False otherwise.

        Raises:
            LunaNotInitializedError:           Engine not initialized.
            EducationalKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("remove_prerequisite")
            if educational_id not in self._records:
                raise EducationalKnowledgeNotFoundError(educational_id)

            existing = self._prerequisites.get(educational_id, [])
            updated = [p for p in existing if p.id != prerequisite_id]
            if len(updated) == len(existing):
                return False
            self._prerequisites[educational_id] = updated
            self._record_mutation()
            return True

    # ── Skill progression model management ───────────────────────────────────

    def attach_progression_model(
        self,
        educational_id: str,
        model: SkillProgressionModel,
    ) -> SkillProgressionModel:
        """
        Attach a SkillProgressionModel to an educational record.

        One model per record — attaching a new model replaces any existing one.

        Args:
            educational_id: ID of the educational record.
            model:          The SkillProgressionModel to attach.

        Returns:
            The attached SkillProgressionModel.

        Raises:
            LunaNotInitializedError:           Engine not initialized.
            EducationalKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("attach_progression_model")
            if educational_id not in self._records:
                raise EducationalKnowledgeNotFoundError(educational_id)

            self._progression_models[educational_id] = model
            self._record_mutation()
            return model

    def get_progression_model(
        self, educational_id: str
    ) -> Optional[SkillProgressionModel]:
        """
        Return the SkillProgressionModel attached to an educational record,
        or None if not set.

        Raises:
            LunaNotInitializedError:           Engine not initialized.
            EducationalKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("get_progression_model")
            if educational_id not in self._records:
                raise EducationalKnowledgeNotFoundError(educational_id)
            return self._progression_models.get(educational_id)

    # ── Curriculum management ─────────────────────────────────────────────────

    def create_curriculum(
        self, curriculum_name: str, record_ids: list[str]
    ) -> list[str]:
        """
        Define a named curriculum as an ordered list of educational record IDs.

        The curriculum acts as a course playlist.  All referenced records must
        already exist in the engine.

        Args:
            curriculum_name: Unique curriculum name.
            record_ids:      Ordered list of educational record IDs.

        Returns:
            The validated list of record IDs stored in the curriculum.

        Raises:
            LunaNotInitializedError:             Engine not initialized.
            EducationalKnowledgeValidationError: One or more record IDs not found.
        """
        with self._lock:
            self._require_initialized("create_curriculum")

            missing = [rid for rid in record_ids if rid not in self._records]
            if missing:
                raise EducationalKnowledgeValidationError(
                    content_id=curriculum_name,
                    violations=[
                        f"The following record IDs do not exist in the engine: {missing}"
                    ],
                )

            self._curricula[curriculum_name] = list(record_ids)
            self._record_mutation()
            return list(record_ids)

    def get_curriculum(
        self, curriculum_name: str
    ) -> list[EducationalKnowledge]:
        """
        Return the ordered list of EducationalKnowledge records in a curriculum.

        Raises:
            LunaNotInitializedError:             Engine not initialized.
            EducationalKnowledgeValidationError: Curriculum not found.
        """
        with self._lock:
            self._require_initialized("get_curriculum")

            ids = self._curricula.get(curriculum_name)
            if ids is None:
                raise EducationalKnowledgeValidationError(
                    content_id=curriculum_name,
                    violations=[f"Curriculum '{curriculum_name}' does not exist."],
                )
            return [
                self._records[rid]
                for rid in ids
                if rid in self._records
            ]

    def list_curricula(self) -> dict[str, int]:
        """
        Return a mapping of curriculum_name → record count for all registered curricula.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("list_curricula")
            return {name: len(ids) for name, ids in self._curricula.items()}

    # ── Difficulty progression validation ─────────────────────────────────────

    def validate_difficulty_progression(
        self, domain_id: str
    ) -> list[dict[str, Any]]:
        """
        Validate that prerequisite records within a domain have difficulty
        ranks that are less than or equal to their dependent records.

        Returns a list of violations, each a dict with keys:
            record_id, prerequisite_id, record_difficulty, prerequisite_difficulty, issue

        An empty list means the domain's progression graph is valid.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("validate_difficulty_progression")

            violations: list[dict[str, Any]] = []
            ids = self._domain_index.get(domain_id, set())

            for rid in ids:
                record = self._records.get(rid)
                if record is None:
                    continue
                for prereq_id in record.prerequisite_knowledge_ids:
                    prereq = self._records.get(prereq_id)
                    if prereq is None:
                        violations.append({
                            "record_id": rid,
                            "prerequisite_id": prereq_id,
                            "record_difficulty": record.difficulty.value,
                            "prerequisite_difficulty": None,
                            "issue": f"Prerequisite record '{prereq_id}' does not exist.",
                        })
                        continue
                    if prereq.difficulty.rank > record.difficulty.rank:
                        violations.append({
                            "record_id": rid,
                            "prerequisite_id": prereq_id,
                            "record_difficulty": record.difficulty.value,
                            "prerequisite_difficulty": prereq.difficulty.value,
                            "issue": (
                                f"Prerequisite '{prereq_id}' ({prereq.difficulty.value}) "
                                f"is harder than dependent '{rid}' ({record.difficulty.value})."
                            ),
                        })
            return violations

    # ── Content section management ────────────────────────────────────────────

    def add_content_section(
        self,
        educational_id: str,
        section_number: int,
        title: str,
        summary: str,
        knowledge_ids: Optional[list[str]] = None,
        duration_minutes: Optional[int] = None,
    ) -> ContentSection:
        """
        Append a ContentSection to an educational record.

        Args:
            educational_id: ID of the target educational record.
            section_number: Ordering number for the section.
            title:          Section title.
            summary:        Brief summary of the section's content.
            knowledge_ids:  Optional list of knowledge record IDs covered.
            duration_minutes: Optional estimated reading/viewing time.

        Returns:
            The newly created ContentSection.

        Raises:
            LunaNotInitializedError:           Engine not initialized.
            EducationalKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("add_content_section")
            record = self._records.get(educational_id)
            if record is None:
                raise EducationalKnowledgeNotFoundError(educational_id)

            section = ContentSection(
                section_number=section_number,
                title=title,
                summary=summary,
                knowledge_ids=tuple(knowledge_ids or []),
                duration_minutes=duration_minutes,
            )
            record.content_sections.append(section)
            record.content_sections.sort(key=lambda s: s.section_number)
            self._record_mutation()
            return section

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_educational(
        self, educational_id: str
    ) -> KnowledgeValidationResult:
        """
        Run a full structural and semantic validation pass on a single record.

        Checks performed:
            - Non-empty name and description
            - At least one domain_id
            - confidence_score within [0.0, 1.0]
            - Learning objectives present for LEARNING_PATH and CURRICULUM types
            - Sections meet comprehensiveness threshold for CURRICULUM type
            - Prerequisite knowledge IDs resolve to existing records
            - Difficulty progression consistency among prerequisites
            - Source trust weight above minimum threshold
            - Staleness / scheduled review

        Returns:
            A KnowledgeValidationResult capturing pass/fail and all issues.

        Raises:
            LunaNotInitializedError:           Engine not initialized.
            EducationalKnowledgeNotFoundError: Record not found.
        """
        with self._lock:
            self._require_initialized("validate_educational")

            record = self._records.get(educational_id)
            if record is None:
                raise EducationalKnowledgeNotFoundError(educational_id)

            issues: list[ValidationIssue] = []

            # 1. Mandatory field checks
            if not record.name or not record.name.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Educational record has an empty name.",
                    field="name",
                    suggestion="Provide a canonical name for the record.",
                ))

            if not record.description or not record.description.strip():
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message="Educational record has an empty description.",
                    field="description",
                    suggestion="Provide a human-readable description.",
                ))

            if not record.domain_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.ERROR,
                    message="Educational record has no domain associations.",
                    field="domain_ids",
                    suggestion="Associate the record with at least one domain.",
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

            # 4. Learning objectives for structured types
            objective_required_types = {
                EducationType.LEARNING_PATH,
                EducationType.CURRICULUM,
                EducationType.LESSON,
                EducationType.TUTORIAL,
            }
            if (
                record.education_type in objective_required_types
                and len(record.learning_objectives) < _COMPREHENSIVE_MIN_OBJECTIVES
            ):
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Education type '{record.education_type.value}' should have at least "
                        f"{_COMPREHENSIVE_MIN_OBJECTIVES} learning objectives "
                        f"(found {len(record.learning_objectives)})."
                    ),
                    field="learning_objectives",
                    suggestion="Add clear, measurable learning objectives.",
                ))

            # 5. Curriculum / learning-path section completeness
            if (
                record.education_type in {EducationType.CURRICULUM, EducationType.LEARNING_PATH}
                and record.section_count < _COMPREHENSIVE_MIN_SECTIONS
            ):
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Education type '{record.education_type.value}' should have at least "
                        f"{_COMPREHENSIVE_MIN_SECTIONS} content sections "
                        f"(found {record.section_count})."
                    ),
                    field="content_sections",
                    suggestion="Structure the curriculum into clearly defined sections.",
                ))

            # 6. Prerequisite resolution — every listed ID must exist
            for prereq_id in record.prerequisite_knowledge_ids:
                if prereq_id not in self._records:
                    issues.append(ValidationIssue.create(
                        issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                        severity=ValidationSeverity.ERROR,
                        message=f"Prerequisite record '{prereq_id}' does not exist in the engine.",
                        field="prerequisite_knowledge_ids",
                        suggestion="Remove the broken reference or create the missing record.",
                    ))
                else:
                    prereq = self._records[prereq_id]
                    if prereq.difficulty.rank > record.difficulty.rank:
                        issues.append(ValidationIssue.create(
                            issue_type=ValidationIssueType.BROKEN_DEPENDENCY,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Prerequisite '{prereq_id}' ({prereq.difficulty.value}) "
                                f"has a higher difficulty than this record "
                                f"({record.difficulty.value})."
                            ),
                            field="prerequisite_knowledge_ids",
                            suggestion="Review prerequisite difficulty progression.",
                        ))

            # 7. Self-referential prerequisite check
            if educational_id in record.prerequisite_knowledge_ids:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.CIRCULAR_DEPENDENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message="Record lists itself as a prerequisite.",
                    field="prerequisite_knowledge_ids",
                    suggestion="Remove the self-referential prerequisite.",
                ))

            # 8. Staleness
            if record.metadata.is_stale:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.STALE_KNOWLEDGE,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Educational record is past its scheduled review date "
                        f"({record.metadata.review_date.isoformat() if record.metadata.review_date else 'unknown'})."
                    ),
                    field="metadata.review_date",
                    suggestion="Re-validate and update the review date.",
                ))

            # 9. Terminal status guard
            if record.status.is_terminal:
                issues.append(ValidationIssue.create(
                    issue_type=ValidationIssueType.INCOMPLETE_DEFINITION,
                    severity=ValidationSeverity.INFO,
                    message=f"Record is in terminal status '{record.status.value}'.",
                    field="status",
                    suggestion="Ensure downstream consumers do not reference retracted records.",
                ))

            return KnowledgeValidationResult.create(
                knowledge_id=educational_id,
                knowledge_name=record.name,
                issues=issues,
                validator_version=_VALIDATOR_VERSION,
            )

    # ── Existence / count helpers ─────────────────────────────────────────────

    def educational_exists(self, educational_id: str) -> bool:
        """
        Return True if a record with the given ID exists (any status).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("educational_exists")
            return educational_id in self._records

    def get_educational_count(self, *, active_only: bool = False) -> int:
        """
        Return the total number of educational records in the store.

        Args:
            active_only: When True, count only VALIDATED and ACTIVE records.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        with self._lock:
            self._require_initialized("get_educational_count")
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
                et: len(ids) for et, ids in self._type_index.items() if ids
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
                "curriculum_count": len(self._curricula),
                "prerequisite_count": sum(
                    len(v) for v in self._prerequisites.values()
                ),
                "progression_model_count": len(self._progression_models),
            }

    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate statistics for the educational knowledge store.

        Includes status breakdown, confidence distribution, type coverage,
        curriculum summary, and prerequisite stats.
        """
        with self._lock:
            self._require_initialized("audit_report")

            total = len(self._records)
            status_breakdown: dict[str, int] = defaultdict(int)
            type_breakdown: dict[str, int] = defaultdict(int)
            difficulty_breakdown: dict[str, int] = defaultdict(int)
            confidence_sum = 0.0
            low_confidence_ids: list[str] = []
            stale_ids: list[str] = []
            total_objectives = 0
            has_assessment_count = 0

            for record in self._records.values():
                status_breakdown[record.status.value] += 1
                type_breakdown[record.education_type.value] += 1
                difficulty_breakdown[record.difficulty.value] += 1
                confidence_sum += record.metadata.confidence_score
                total_objectives += len(record.learning_objectives)
                if record.has_assessment:
                    has_assessment_count += 1
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
                "difficulty_breakdown": dict(difficulty_breakdown),
                "average_confidence": round(avg_confidence, 4),
                "low_confidence_count": len(low_confidence_ids),
                "low_confidence_ids": low_confidence_ids,
                "stale_count": len(stale_ids),
                "stale_ids": stale_ids,
                "total_learning_objectives": total_objectives,
                "records_with_assessment": has_assessment_count,
                "curriculum_count": len(self._curricula),
                "prerequisite_count": sum(len(v) for v in self._prerequisites.values()),
                "progression_model_count": len(self._progression_models),
                "duplicate_checks_performed": self._duplicate_checks,
                "mutation_count": self._mutation_count,
                "generated_at": _now_utc().isoformat(),
            }


__all__ = ["EducationalKnowledgeEngine"]