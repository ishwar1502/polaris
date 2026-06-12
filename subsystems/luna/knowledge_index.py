"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/knowledge_index.py

Concrete in-memory implementation of the LUNA Knowledge Index Engine.

Provides centralized, fast-access indexing for all seven knowledge types owned
by LUNA.  Acts as the primary lookup accelerator for the Retrieval Engine and
any POLARIS consumer that needs to scan, filter, or navigate the knowledge store
without holding direct references to the seven individual type engines.

Supported knowledge types:
    Fact · Concept · Skill · KnowledgeDomain · Procedure ·
    ResearchKnowledge · EducationalKnowledge

Index operations:
    index_record          — add or refresh a single record's entry
    remove_record         — purge a record from all index structures
    rebuild_index         — full rebuild from all connected engines
    clear_index           — reset every index structure to empty

Lookup operations:
    lookup_by_id          — primary-key lookup → KnowledgeIndexEntry | None
    lookup_by_type        — all entries for a KnowledgeType
    lookup_by_domain      — entries belonging to a domain
    lookup_by_tag         — entries carrying a specific tag
    lookup_by_status      — entries in a given KnowledgeStatus
    lookup_by_confidence  — entries meeting a minimum confidence threshold
    lookup_by_difficulty  — entries at a KnowledgeDifficulty level
    lookup_by_relationship — entries referencing a particular record (related_ids)
    lookup_by_dependency  — entries that declare a prerequisite dependency

Specialised index lookups:
    lookup_concept        — ConceptIndexEntry by concept_id
    lookup_domain         — DomainIndexEntry  by domain_id
    lookup_skill          — SkillIndexEntry   by skill_id

Combined query:
    query_index           — multi-filter search with paging

Index management:
    incremental indexing  — index_record / remove_record operate on one record
    full rebuild          — rebuild_index / reindex_all replace the whole index
    consistency validation — validate_index: surface stale / broken entries
    duplicate detection   — detect_duplicate_fingerprints
    index statistics      — index_statistics
    index health report   — health_report / diagnostics_report / audit_report

Thread safety:    threading.RLock on every public operation.
Lifecycle-gated:  every public method raises LunaNotInitializedError before
                  initialize() or after shutdown().
In-memory v1 implementation.  No persistence layer.

Integrates with (via injected engine handles):
    facts.py       FactEngine
    concepts.py    ConceptEngine
    skills.py      SkillEngine
    domains.py     KnowledgeDomainEngine
    procedures.py  ProceduralKnowledgeEngine
    research.py    ResearchKnowledgeEngine
    education.py   EducationalKnowledgeEngine
    retrieval.py   KnowledgeRetrievalEngine  (consumer; no hard dependency)
    progression.py SkillProgressionEngine    (consumer; no hard dependency)

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
    KnowledgeIndexError,
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractKnowledgeIndexEngine
from subsystems.luna.models import (
    Concept,
    ConceptIndexEntry,
    DomainIndexEntry,
    EducationalKnowledge,
    Fact,
    KnowledgeDependency,
    KnowledgeDifficulty,
    KnowledgeDomain,
    KnowledgeIndexEntry,
    KnowledgeRecord,
    KnowledgeStatus,
    KnowledgeType,
    Procedure,
    ResearchKnowledge,
    Skill,
    SkillIndexEntry,
    _new_id,
    _utcnow,
)

logger = logging.getLogger(__name__)

_ENGINE_VERSION: str = "5.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _prerequisite_ids_for(record: KnowledgeRecord) -> list[str]:
    """
    Return the IDs of records that the given record declares as prerequisites /
    hard dependencies.  Used to build the dependency lookup index.
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


def _related_ids_for(record: KnowledgeRecord) -> list[str]:
    """Return related_ids, which back the relationship lookup index."""
    return list(record.related_ids)


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeIndexEngine(AbstractKnowledgeIndexEngine):
    """
    In-memory, thread-safe implementation of the LUNA Knowledge Index Engine
    (v1).

    Primary index:
        _index          — dict[knowledge_id, KnowledgeIndexEntry]

    Specialised sub-indexes:
        _concept_index  — dict[concept_id,  ConceptIndexEntry]
        _domain_index   — dict[domain_id,   DomainIndexEntry]
        _skill_index    — dict[skill_id,    SkillIndexEntry]

    Secondary scan indexes (value = set[knowledge_id]):
        _by_type        — KnowledgeType → set[id]
        _by_domain      — domain_id     → set[id]
        _by_tag         — tag           → set[id]
        _by_status      — KnowledgeStatus → set[id]
        _by_difficulty  — KnowledgeDifficulty → set[id]
        _by_relationship — referenced_id  → set[id]  (records citing that id)
        _by_dependency   — dependency_id  → set[id]  (records that require it)
        _by_fingerprint — fingerprint    → list[id]  (dedup detection)

    Injected engine handles (all optional; missing engines skip their type):
        _fact_engine, _concept_engine, _skill_engine, _domain_engine,
        _procedure_engine, _research_engine, _educational_engine

    Lifecycle::

        engine = KnowledgeIndexEngine(
            fact_engine=my_fact_engine,
            concept_engine=my_concept_engine,
            ...
        )
        engine.initialize()
        entry = engine.index_record(record_id, KnowledgeType.FACT)
        results = engine.lookup_by_domain("domain-xyz")
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
    ) -> None:
        self._fact_engine = fact_engine
        self._concept_engine = concept_engine
        self._skill_engine = skill_engine
        self._domain_engine = domain_engine
        self._procedure_engine = procedure_engine
        self._research_engine = research_engine
        self._educational_engine = educational_engine

        # Lifecycle
        self._initialized: bool = False
        self._lock: threading.RLock = threading.RLock()
        self._started_at: Optional[datetime] = None

        # Primary indexes
        self._index: dict[str, KnowledgeIndexEntry] = {}
        self._concept_index: dict[str, ConceptIndexEntry] = {}
        self._domain_index: dict[str, DomainIndexEntry] = {}
        self._skill_index: dict[str, SkillIndexEntry] = {}

        # Secondary scan indexes
        self._by_type: dict[KnowledgeType, set[str]] = defaultdict(set)
        self._by_domain: dict[str, set[str]] = defaultdict(set)
        self._by_tag: dict[str, set[str]] = defaultdict(set)
        self._by_status: dict[KnowledgeStatus, set[str]] = defaultdict(set)
        self._by_difficulty: dict[KnowledgeDifficulty, set[str]] = defaultdict(set)
        self._by_relationship: dict[str, set[str]] = defaultdict(set)
        self._by_dependency: dict[str, set[str]] = defaultdict(set)
        self._by_fingerprint: dict[str, list[str]] = defaultdict(list)

        # Observability counters
        self._records_indexed: int = 0
        self._records_removed: int = 0
        self._rebuild_count: int = 0
        self._lookup_count: int = 0
        self._last_mutation_at: Optional[datetime] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Prepare the engine for operation.

        Idempotent: calling initialize() on an already-initialized engine is a
        no-op.

        Raises:
            LunaLifecycleError: Internal initialization failure.
        """
        with self._lock:
            if self._initialized:
                return
            try:
                self._clear_all_structures()
                self._initialized = True
                self._started_at = _utcnow()
                logger.info(
                    "KnowledgeIndexEngine initialized (version=%s)", _ENGINE_VERSION
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="KnowledgeIndexEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        """
        Release all resources and mark the engine as offline.

        Idempotent: calling shutdown() on an already-stopped engine is a no-op.

        Raises:
            LunaLifecycleError: Internal teardown failure.
        """
        with self._lock:
            if not self._initialized:
                return
            try:
                self._clear_all_structures()
                self._initialized = False
                logger.info("KnowledgeIndexEngine shut down.")
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="KnowledgeIndexEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def is_initialized(self) -> bool:
        return self._initialized

    # ── Lifecycle guard ───────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation=operation)

    # ── Internal structure management ─────────────────────────────────────────

    def _clear_all_structures(self) -> None:
        """Reset every in-memory index to empty."""
        self._index.clear()
        self._concept_index.clear()
        self._domain_index.clear()
        self._skill_index.clear()
        self._by_type.clear()
        self._by_domain.clear()
        self._by_tag.clear()
        self._by_status.clear()
        self._by_difficulty.clear()
        self._by_relationship.clear()
        self._by_dependency.clear()
        self._by_fingerprint.clear()

    def _engine_for_type(self, kt: KnowledgeType) -> Optional[Any]:
        """Return the injected sub-engine for a given KnowledgeType, or None."""
        return {
            KnowledgeType.FACT:        self._fact_engine,
            KnowledgeType.CONCEPT:     self._concept_engine,
            KnowledgeType.SKILL:       self._skill_engine,
            KnowledgeType.DOMAIN:      self._domain_engine,
            KnowledgeType.PROCEDURE:   self._procedure_engine,
            KnowledgeType.RESEARCH:    self._research_engine,
            KnowledgeType.EDUCATIONAL: self._educational_engine,
        }.get(kt)

    def _fetch_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> Optional[KnowledgeRecord]:
        """
        Resolve a record ID to a KnowledgeRecord via the appropriate sub-engine.

        Returns None if the engine is absent or the record cannot be found.
        """
        engine = self._engine_for_type(knowledge_type)
        if engine is None or not engine.is_initialized():
            return None
        try:
            if knowledge_type == KnowledgeType.FACT:
                return engine.retrieve_fact(record_id)
            if knowledge_type == KnowledgeType.CONCEPT:
                return engine.retrieve_concept(record_id)
            if knowledge_type == KnowledgeType.SKILL:
                return engine.retrieve_skill(record_id)
            if knowledge_type == KnowledgeType.DOMAIN:
                return engine.retrieve_domain(record_id)
            if knowledge_type == KnowledgeType.PROCEDURE:
                return engine.retrieve_procedure(record_id)
            if knowledge_type == KnowledgeType.RESEARCH:
                return engine.retrieve_research(record_id)
            if knowledge_type == KnowledgeType.EDUCATIONAL:
                return engine.retrieve_educational(record_id)
        except Exception:
            return None
        return None

    def _iter_all_records(
        self,
        knowledge_type: KnowledgeType,
    ) -> list[KnowledgeRecord]:
        """
        Return every record of the given type from the connected sub-engine.
        Returns an empty list if the engine is absent or uninitialized.
        """
        engine = self._engine_for_type(knowledge_type)
        if engine is None or not engine.is_initialized():
            return []
        try:
            if knowledge_type == KnowledgeType.FACT:
                return engine.list_facts()
            if knowledge_type == KnowledgeType.CONCEPT:
                return engine.list_concepts()
            if knowledge_type == KnowledgeType.SKILL:
                return engine.list_skills()
            if knowledge_type == KnowledgeType.DOMAIN:
                return engine.list_domains()
            if knowledge_type == KnowledgeType.PROCEDURE:
                return engine.list_procedures()
            if knowledge_type == KnowledgeType.RESEARCH:
                return engine.list_research()
            if knowledge_type == KnowledgeType.EDUCATIONAL:
                return engine.list_educational()
        except Exception:
            return []
        return []

    # ── Secondary-index bookkeeping ───────────────────────────────────────────

    def _add_to_secondary_indexes(
        self,
        entry: KnowledgeIndexEntry,
        record: KnowledgeRecord,
    ) -> None:
        """Register a new entry in every secondary index."""
        kid = entry.knowledge_id

        self._by_type[entry.knowledge_type].add(kid)
        self._by_status[entry.status].add(kid)
        self._by_difficulty[entry.difficulty].add(kid)

        for domain_id in entry.domain_ids:
            self._by_domain[domain_id].add(kid)

        for tag in entry.tags:
            self._by_tag[tag].add(kid)

        # Fingerprint dedup index
        fp = entry.fingerprint
        if kid not in self._by_fingerprint[fp]:
            self._by_fingerprint[fp].append(kid)

        # Relationship index: records that reference a particular ID
        for related_id in _related_ids_for(record):
            self._by_relationship[related_id].add(kid)

        # Dependency index: records that depend on a particular ID
        for dep_id in _prerequisite_ids_for(record):
            self._by_dependency[dep_id].add(kid)

    def _remove_from_secondary_indexes(
        self,
        entry: KnowledgeIndexEntry,
        record: Optional[KnowledgeRecord],
    ) -> None:
        """Remove an entry from every secondary index."""
        kid = entry.knowledge_id

        self._by_type[entry.knowledge_type].discard(kid)
        self._by_status[entry.status].discard(kid)
        self._by_difficulty[entry.difficulty].discard(kid)

        for domain_id in entry.domain_ids:
            self._by_domain[domain_id].discard(kid)

        for tag in entry.tags:
            self._by_tag[tag].discard(kid)

        # Fingerprint dedup index
        fp = entry.fingerprint
        if fp in self._by_fingerprint:
            try:
                self._by_fingerprint[fp].remove(kid)
            except ValueError:
                pass
            if not self._by_fingerprint[fp]:
                del self._by_fingerprint[fp]

        # Relationship and dependency indexes
        if record is not None:
            for related_id in _related_ids_for(record):
                self._by_relationship[related_id].discard(kid)
            for dep_id in _prerequisite_ids_for(record):
                self._by_dependency[dep_id].discard(kid)
        else:
            # If we no longer have the record, scan and purge by value
            for refs in self._by_relationship.values():
                refs.discard(kid)
            for deps in self._by_dependency.values():
                deps.discard(kid)

    # ── Core index mutations ──────────────────────────────────────────────────

    def index_record(
        self,
        record_or_id,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> KnowledgeIndexEntry:
        """
        Add or refresh the index entry for a single knowledge record.

        Accepts either:
          - A KnowledgeRecord object directly: ``index_record(record)``
          - An ID + KnowledgeType pair: ``index_record(record_id, knowledge_type)``

        If the record is already indexed, the existing entry is removed and a
        fresh entry is built from the current record state (incremental update).

        Returns:
            The newly created KnowledgeIndexEntry.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeIndexError:     Record not found in the originating store.
        """
        self._require_initialized("index_record")
        with self._lock:
            # Accept a KnowledgeRecord object directly
            if hasattr(record_or_id, "id") and hasattr(record_or_id, "knowledge_type"):
                record = record_or_id
                record_id = record.id
                kt = record.knowledge_type
            else:
                record_id = record_or_id
                kt = knowledge_type
                record = self._fetch_record(record_id, kt)

            if record is None:
                raise KnowledgeIndexError(
                    index_key=record_id,
                    reason=(
                        f"Record '{record_id}' of type '{kt.value}' "
                        "not found in originating store"
                    ),
                )

            # Remove stale entry if present
            if record_id in self._index:
                old_entry = self._index[record_id]
                old_record = self._fetch_record(record_id, kt)
                self._remove_from_secondary_indexes(old_entry, old_record)
                del self._index[record_id]

            # Build fresh base entry
            entry = KnowledgeIndexEntry.from_record(record)
            self._index[record_id] = entry
            self._add_to_secondary_indexes(entry, record)

            # Build specialised entries
            if isinstance(record, Concept):
                self._concept_index[record_id] = ConceptIndexEntry.from_concept(record)
            elif isinstance(record, KnowledgeDomain):
                self._domain_index[record_id] = DomainIndexEntry.from_domain(record)
            elif isinstance(record, Skill):
                self._skill_index[record_id] = SkillIndexEntry.from_skill(record)

            self._records_indexed += 1
            self._last_mutation_at = _utcnow()

            logger.debug(
                "Indexed record id=%s type=%s", record_id, kt.value
            )
            return entry

    def remove_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> None:
        """
        Remove a record's entry from the index.

        A no-op if the record is not currently indexed.

        Args:
            record_id:      ID of the record to deindex.
            knowledge_type: KnowledgeType of the record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("remove_record")
        with self._lock:
            entry = self._index.get(record_id)
            if entry is None:
                return

            # Attempt to retrieve record for precise secondary-index cleanup
            record = self._fetch_record(record_id, knowledge_type)
            self._remove_from_secondary_indexes(entry, record)
            del self._index[record_id]

            # Remove specialised entries
            self._concept_index.pop(record_id, None)
            self._domain_index.pop(record_id, None)
            self._skill_index.pop(record_id, None)

            self._records_removed += 1
            self._last_mutation_at = _utcnow()

            logger.debug(
                "Removed index entry id=%s type=%s", record_id, knowledge_type.value
            )

    # AbstractKnowledgeIndexEngine.deindex_record alias
    def deindex_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> None:
        """Alias for remove_record (satisfies AbstractKnowledgeIndexEngine contract)."""
        self.remove_record(record_id, knowledge_type)

    def rebuild_index(self) -> int:
        """
        Full index rebuild.

        Clears the entire index and re-ingests every record from every
        connected sub-engine.

        Returns:
            Total number of records indexed.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("rebuild_index")
        with self._lock:
            self._clear_all_structures()
            total = 0
            all_types = [
                KnowledgeType.FACT,
                KnowledgeType.CONCEPT,
                KnowledgeType.SKILL,
                KnowledgeType.DOMAIN,
                KnowledgeType.PROCEDURE,
                KnowledgeType.RESEARCH,
                KnowledgeType.EDUCATIONAL,
            ]
            for kt in all_types:
                records = self._iter_all_records(kt)
                for record in records:
                    try:
                        entry = KnowledgeIndexEntry.from_record(record)
                        self._index[record.id] = entry
                        self._add_to_secondary_indexes(entry, record)

                        if isinstance(record, Concept):
                            self._concept_index[record.id] = (
                                ConceptIndexEntry.from_concept(record)
                            )
                        elif isinstance(record, KnowledgeDomain):
                            self._domain_index[record.id] = (
                                DomainIndexEntry.from_domain(record)
                            )
                        elif isinstance(record, Skill):
                            self._skill_index[record.id] = (
                                SkillIndexEntry.from_skill(record)
                            )

                        total += 1
                    except Exception as exc:
                        logger.warning(
                            "Skipped record id=%s during rebuild: %s",
                            record.id,
                            exc,
                        )

            self._rebuild_count += 1
            self._records_indexed += total
            self._last_mutation_at = _utcnow()
            logger.info(
                "KnowledgeIndexEngine rebuild complete: %d records indexed", total
            )
            return total

    # AbstractKnowledgeIndexEngine.reindex_all alias
    def reindex_all(self) -> int:
        """Alias for rebuild_index (satisfies AbstractKnowledgeIndexEngine contract)."""
        return self.rebuild_index()

    def clear_index(self) -> None:
        """
        Reset every index structure to empty without a rebuild.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("clear_index")
        with self._lock:
            self._clear_all_structures()
            self._last_mutation_at = _utcnow()
            logger.info("KnowledgeIndexEngine index cleared.")

    # ── Primary lookup ────────────────────────────────────────────────────────

    def lookup_by_id(
        self,
        record_id: str,
    ) -> Optional[KnowledgeIndexEntry]:
        """
        Return the index entry for a record ID, or None if absent.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_by_id")
        with self._lock:
            self._lookup_count += 1
            return self._index.get(record_id)

    # ── Secondary lookups ─────────────────────────────────────────────────────

    def lookup_by_type(
        self,
        knowledge_type: KnowledgeType,
        *,
        active_only: bool = False,
    ) -> list[KnowledgeIndexEntry]:
        """
        Return all index entries for a knowledge type.

        Args:
            knowledge_type: Filter to this type.
            active_only:    If True, restrict to VALIDATED and ACTIVE statuses.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_by_type")
        with self._lock:
            self._lookup_count += 1
            ids = self._by_type.get(knowledge_type, set())
            results = [
                self._index[kid]
                for kid in ids
                if kid in self._index
            ]
            if active_only:
                results = [e for e in results if e.status.is_usable]
            results.sort(key=lambda e: -e.confidence_score)
            return results

    def lookup_by_domain(
        self,
        domain_id: str,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
        active_only: bool = False,
    ) -> list[KnowledgeIndexEntry]:
        """
        Return all index entries whose domain_ids include the given domain.

        Args:
            domain_id:      Domain to filter by.
            knowledge_type: Optional additional type filter.
            active_only:    If True, restrict to usable statuses.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_by_domain")
        with self._lock:
            self._lookup_count += 1
            ids = self._by_domain.get(domain_id, set())
            results = [
                self._index[kid]
                for kid in ids
                if kid in self._index
            ]
            if knowledge_type is not None:
                results = [e for e in results if e.knowledge_type == knowledge_type]
            if active_only:
                results = [e for e in results if e.status.is_usable]
            results.sort(key=lambda e: -e.confidence_score)
            return results

    def lookup_by_tag(
        self,
        tag: str,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
        active_only: bool = False,
    ) -> list[KnowledgeIndexEntry]:
        """
        Return all index entries that carry the given tag.

        Args:
            tag:            Tag string to match (exact).
            knowledge_type: Optional additional type filter.
            active_only:    If True, restrict to usable statuses.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_by_tag")
        with self._lock:
            self._lookup_count += 1
            ids = self._by_tag.get(tag, set())
            results = [
                self._index[kid]
                for kid in ids
                if kid in self._index
            ]
            if knowledge_type is not None:
                results = [e for e in results if e.knowledge_type == knowledge_type]
            if active_only:
                results = [e for e in results if e.status.is_usable]
            results.sort(key=lambda e: -e.confidence_score)
            return results

    def lookup_by_status(
        self,
        status: KnowledgeStatus,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[KnowledgeIndexEntry]:
        """
        Return all index entries in the given KnowledgeStatus.

        Args:
            status:         The status to filter by.
            knowledge_type: Optional additional type filter.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_by_status")
        with self._lock:
            self._lookup_count += 1
            ids = self._by_status.get(status, set())
            results = [
                self._index[kid]
                for kid in ids
                if kid in self._index
            ]
            if knowledge_type is not None:
                results = [e for e in results if e.knowledge_type == knowledge_type]
            results.sort(key=lambda e: -e.confidence_score)
            return results

    def lookup_by_confidence(
        self,
        min_confidence: float,
        *,
        max_confidence: float = 1.0,
        knowledge_type: Optional[KnowledgeType] = None,
        active_only: bool = False,
    ) -> list[KnowledgeIndexEntry]:
        """
        Return all index entries whose confidence_score is within [min, max].

        Args:
            min_confidence: Lower bound (inclusive).
            max_confidence: Upper bound (inclusive, default 1.0).
            knowledge_type: Optional type filter.
            active_only:    If True, restrict to usable statuses.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_by_confidence")
        with self._lock:
            self._lookup_count += 1
            results = [
                e
                for e in self._index.values()
                if min_confidence <= e.confidence_score <= max_confidence
            ]
            if knowledge_type is not None:
                results = [e for e in results if e.knowledge_type == knowledge_type]
            if active_only:
                results = [e for e in results if e.status.is_usable]
            results.sort(key=lambda e: -e.confidence_score)
            return results

    def lookup_by_difficulty(
        self,
        difficulty: KnowledgeDifficulty,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
        active_only: bool = False,
    ) -> list[KnowledgeIndexEntry]:
        """
        Return all index entries at the given difficulty level.

        Args:
            difficulty:     The KnowledgeDifficulty level to filter by.
            knowledge_type: Optional type filter.
            active_only:    If True, restrict to usable statuses.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_by_difficulty")
        with self._lock:
            self._lookup_count += 1
            ids = self._by_difficulty.get(difficulty, set())
            results = [
                self._index[kid]
                for kid in ids
                if kid in self._index
            ]
            if knowledge_type is not None:
                results = [e for e in results if e.knowledge_type == knowledge_type]
            if active_only:
                results = [e for e in results if e.status.is_usable]
            results.sort(key=lambda e: -e.confidence_score)
            return results

    def lookup_by_relationship(
        self,
        referenced_id: str,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[KnowledgeIndexEntry]:
        """
        Return index entries for records that cite ``referenced_id`` in their
        ``related_ids`` field.

        Args:
            referenced_id:  The ID being referenced.
            knowledge_type: Optional type filter.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_by_relationship")
        with self._lock:
            self._lookup_count += 1
            ids = self._by_relationship.get(referenced_id, set())
            results = [
                self._index[kid]
                for kid in ids
                if kid in self._index
            ]
            if knowledge_type is not None:
                results = [e for e in results if e.knowledge_type == knowledge_type]
            results.sort(key=lambda e: -e.confidence_score)
            return results

    def lookup_by_dependency(
        self,
        dependency_id: str,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
        hard_only: bool = False,
    ) -> list[KnowledgeIndexEntry]:
        """
        Return index entries for records that declare a prerequisite/dependency
        on ``dependency_id``.

        Args:
            dependency_id:  The ID of the required record.
            knowledge_type: Optional type filter.
            hard_only:      Reserved for future typed-dependency support;
                            currently all indexed dependencies are treated as
                            hard (blocking) prerequisites.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_by_dependency")
        with self._lock:
            self._lookup_count += 1
            ids = self._by_dependency.get(dependency_id, set())
            results = [
                self._index[kid]
                for kid in ids
                if kid in self._index
            ]
            if knowledge_type is not None:
                results = [e for e in results if e.knowledge_type == knowledge_type]
            results.sort(key=lambda e: -e.confidence_score)
            return results

    # ── Specialised sub-index lookups ─────────────────────────────────────────

    def lookup_concept(
        self,
        concept_id: str,
    ) -> Optional[ConceptIndexEntry]:
        """
        Return the ConceptIndexEntry for a concept, or None if not indexed.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_concept")
        with self._lock:
            self._lookup_count += 1
            return self._concept_index.get(concept_id)

    def lookup_domain(
        self,
        domain_id: str,
    ) -> Optional[DomainIndexEntry]:
        """
        Return the DomainIndexEntry for a domain, or None if not indexed.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_domain")
        with self._lock:
            self._lookup_count += 1
            return self._domain_index.get(domain_id)

    def lookup_skill(
        self,
        skill_id: str,
    ) -> Optional[SkillIndexEntry]:
        """
        Return the SkillIndexEntry for a skill, or None if not indexed.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("lookup_skill")
        with self._lock:
            self._lookup_count += 1
            return self._skill_index.get(skill_id)

    # ── Combined query ────────────────────────────────────────────────────────

    def query_index(
        self,
        *,
        query: str = "",
        knowledge_types: Optional[list[KnowledgeType]] = None,
        domain_ids: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KnowledgeIndexEntry]:
        """
        Query the index using a combination of filters.

        All supplied filters are applied as AND conditions.  Results are sorted
        by confidence_score descending, then by name ascending as a tiebreaker.

        Args:
            query:           Free-text substring match against name / aliases.
            knowledge_types: Restrict to these types (OR among supplied types).
            domain_ids:      Restrict to entries that belong to any of these domains.
            tags:            Restrict to entries carrying all of these tags.
            difficulty:      Exact difficulty match.
            min_confidence:  Minimum confidence_score (inclusive).
            limit:           Maximum results to return (default 100).
            offset:          Skip this many results for pagination (default 0).

        Returns:
            Sorted, paged list of KnowledgeIndexEntry.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("query_index")
        with self._lock:
            self._lookup_count += 1

            # Start from full candidate pool or narrow by type
            if knowledge_types:
                candidate_ids: set[str] = set()
                for kt in knowledge_types:
                    candidate_ids |= self._by_type.get(kt, set())
                candidates = [
                    self._index[kid]
                    for kid in candidate_ids
                    if kid in self._index
                ]
            else:
                candidates = list(self._index.values())

            # Domain filter (record must belong to at least one supplied domain)
            if domain_ids:
                domain_set = set(domain_ids)
                candidates = [
                    e for e in candidates
                    if domain_set.intersection(set(e.domain_ids))
                ]

            # Tag filter (record must carry ALL supplied tags)
            if tags:
                tag_set = set(tags)
                candidates = [
                    e for e in candidates
                    if tag_set.issubset(set(e.tags))
                ]

            # Difficulty filter
            if difficulty is not None:
                candidates = [
                    e for e in candidates
                    if e.difficulty == difficulty
                ]

            # Confidence filter
            if min_confidence > 0.0:
                candidates = [
                    e for e in candidates
                    if e.confidence_score >= min_confidence
                ]

            # Text filter
            if query:
                q = query.lower().strip()
                candidates = [
                    e for e in candidates
                    if e.matches_query(q)
                ]

            # Sort: confidence descending, name ascending as tiebreaker
            candidates.sort(key=lambda e: (-e.confidence_score, e.name_lower))

            # Paging
            return candidates[offset: offset + limit]

    # ── Index management ──────────────────────────────────────────────────────

    def validate_index(self) -> list[dict[str, Any]]:
        """
        Perform a consistency check on the index.

        Detects:
            - Entries whose record no longer exists in the originating store
              (stale entries).
            - Entries that have a different fingerprint from what is currently
              stored in the record (stale content).
            - Entries present in the base index but missing from the
              specialised sub-indexes (Concept / Domain / Skill).

        Returns:
            List of issue dicts, each with keys:
                record_id, issue_type, details

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("validate_index")
        with self._lock:
            issues: list[dict[str, Any]] = []

            for kid, entry in list(self._index.items()):
                kt = entry.knowledge_type
                record = self._fetch_record(kid, kt)

                if record is None:
                    issues.append({
                        "record_id": kid,
                        "issue_type": "stale_entry",
                        "details": (
                            f"Record '{kid}' of type '{kt.value}' no longer "
                            "exists in the originating store."
                        ),
                    })
                    continue

                # Fingerprint drift check
                if record.fingerprint != entry.fingerprint:
                    issues.append({
                        "record_id": kid,
                        "issue_type": "stale_content",
                        "details": (
                            f"Record '{kid}' fingerprint has changed: "
                            f"index='{entry.fingerprint[:12]}…' "
                            f"current='{record.fingerprint[:12]}…'"
                        ),
                    })

                # Sub-index completeness checks
                if kt == KnowledgeType.CONCEPT and kid not in self._concept_index:
                    issues.append({
                        "record_id": kid,
                        "issue_type": "missing_concept_subindex",
                        "details": (
                            f"Concept '{kid}' present in base index but "
                            "absent from concept sub-index."
                        ),
                    })
                elif kt == KnowledgeType.DOMAIN and kid not in self._domain_index:
                    issues.append({
                        "record_id": kid,
                        "issue_type": "missing_domain_subindex",
                        "details": (
                            f"Domain '{kid}' present in base index but "
                            "absent from domain sub-index."
                        ),
                    })
                elif kt == KnowledgeType.SKILL and kid not in self._skill_index:
                    issues.append({
                        "record_id": kid,
                        "issue_type": "missing_skill_subindex",
                        "details": (
                            f"Skill '{kid}' present in base index but "
                            "absent from skill sub-index."
                        ),
                    })

            return issues

    def detect_duplicate_fingerprints(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[list[str]]:
        """
        Return groups of record IDs that share the same content fingerprint.

        Args:
            knowledge_type: If provided, restrict to one type.

        Returns:
            A list of groups, where each group is a list of record IDs with the
            same fingerprint.  Groups with only one member are excluded.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("detect_duplicate_fingerprints")
        with self._lock:
            groups: list[list[str]] = []
            for fp, ids in self._by_fingerprint.items():
                filtered = ids
                if knowledge_type is not None:
                    filtered = [
                        kid for kid in ids
                        if kid in self._index
                        and self._index[kid].knowledge_type == knowledge_type
                    ]
                if len(filtered) > 1:
                    groups.append(list(filtered))
            return groups

    def index_statistics(self) -> dict[str, Any]:
        """
        Return a comprehensive statistical snapshot of the index state.

        Returns:
            A dict containing:
                total_entries         (int)
                entries_by_type       (dict[str, int])
                entries_by_status     (dict[str, int])
                entries_by_difficulty (dict[str, int])
                unique_domains        (int)
                unique_tags           (int)
                concept_subindex_size (int)
                domain_subindex_size  (int)
                skill_subindex_size   (int)
                relationship_index_keys (int)
                dependency_index_keys   (int)
                duplicate_fingerprint_groups (int)
                generated_at          (str)

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("index_statistics")
        with self._lock:
            entries_by_type: dict[str, int] = {
                kt.value: len(ids)
                for kt, ids in self._by_type.items()
            }
            entries_by_status: dict[str, int] = {
                st.value: len(ids)
                for st, ids in self._by_status.items()
            }
            entries_by_difficulty: dict[str, int] = {
                d.value: len(ids)
                for d, ids in self._by_difficulty.items()
            }
            dup_groups = sum(
                1
                for ids in self._by_fingerprint.values()
                if len(ids) > 1
            )
            return {
                "total_entries": len(self._index),
                "entries_by_type": entries_by_type,
                "entries_by_status": entries_by_status,
                "entries_by_difficulty": entries_by_difficulty,
                "unique_domains": len(self._by_domain),
                "unique_tags": len(self._by_tag),
                "concept_subindex_size": len(self._concept_index),
                "domain_subindex_size": len(self._domain_index),
                "skill_subindex_size": len(self._skill_index),
                "relationship_index_keys": len(self._by_relationship),
                "dependency_index_keys": len(self._by_dependency),
                "duplicate_fingerprint_groups": dup_groups,
                "generated_at": _utcnow().isoformat(),
            }

    # ── AbstractKnowledgeIndexEngine surface ──────────────────────────────────

    def get_index_size(self) -> int:
        """Return the total number of base index entries."""
        self._require_initialized("get_index_size")
        with self._lock:
            return len(self._index)

    def get_domains_in_index(self) -> list[str]:
        """Return a deduplicated, sorted list of all domain IDs in the index."""
        self._require_initialized("get_domains_in_index")
        with self._lock:
            return sorted(self._by_domain.keys())

    def get_tags_in_index(self) -> list[str]:
        """Return a deduplicated, sorted list of all tags in the index."""
        self._require_initialized("get_tags_in_index")
        with self._lock:
            return sorted(self._by_tag.keys())

    # ── Observability ─────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """
        Return a lightweight liveness/readiness summary.

        Required keys: engine, initialized, record_count, status.
        """
        return {
            "engine": "KnowledgeIndexEngine",
            "initialized": self._initialized,
            "record_count": len(self._index),
            "status": "healthy" if self._initialized else "offline",
        }

    def diagnostics_report(self) -> dict[str, Any]:
        """
        Return a full introspection snapshot for operator debugging.

        Required keys: engine, initialized, record_count, status, index_size,
        duplicate_checks, mutation_count, last_mutation_at.
        """
        with self._lock:
            mutation_count = self._records_indexed + self._records_removed
            dup_groups = sum(
                1 for ids in self._by_fingerprint.values() if len(ids) > 1
            )
            return {
                "engine": "KnowledgeIndexEngine",
                "version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "status": "healthy" if self._initialized else "offline",
                "record_count": len(self._index),
                "index_size": len(self._index),
                "duplicate_checks": dup_groups,
                "mutation_count": mutation_count,
                "last_mutation_at": (
                    self._last_mutation_at.isoformat()
                    if self._last_mutation_at else None
                ),
                "records_indexed": self._records_indexed,
                "records_removed": self._records_removed,
                "rebuild_count": self._rebuild_count,
                "lookup_count": self._lookup_count,
                "concept_subindex_size": len(self._concept_index),
                "domain_subindex_size": len(self._domain_index),
                "skill_subindex_size": len(self._skill_index),
                "unique_domains": len(self._by_domain),
                "unique_tags": len(self._by_tag),
                "relationship_index_keys": len(self._by_relationship),
                "dependency_index_keys": len(self._by_dependency),
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
                },
            }

    def audit_report(self) -> dict[str, Any]:
        """
        Return index statistics for audit / health monitoring.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("audit_report")
        with self._lock:
            stats = self.index_statistics()
            stats.update({
                "engine": "KnowledgeIndexEngine",
                "version": _ENGINE_VERSION,
                "records_indexed_total": self._records_indexed,
                "records_removed_total": self._records_removed,
                "rebuild_count": self._rebuild_count,
                "lookup_count": self._lookup_count,
                "last_mutation_at": (
                    self._last_mutation_at.isoformat()
                    if self._last_mutation_at else None
                ),
                "started_at": (
                    self._started_at.isoformat() if self._started_at else None
                ),
            })
            return stats


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["KnowledgeIndexEngine"]
