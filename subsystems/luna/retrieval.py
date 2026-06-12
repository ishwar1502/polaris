"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/retrieval.py

Concrete in-memory implementation of the LUNA Knowledge Retrieval Engine.

Responsibilities:
    - retrieve_by_id / retrieve_many             — direct ID lookup by type
    - search_all                                 — unified cross-type full-text search
    - search_by_domain                           — domain-scoped retrieval
    - search_by_tags                             — metadata tag-based retrieval
    - search_by_type / search_by_confidence      — filtered retrieval
    - search_by_difficulty                       — difficulty-scoped retrieval
    - semantic_search                            — name/description similarity
    - relationship_search                        — traverse related_ids graph
    - dependency_search                          — prerequisite / dependency traversal
    - ranked retrieval with relevance scoring
    - related knowledge discovery via related_ids
    - get_related / get_total_count

Works across all seven LUNA knowledge types:
    Facts, Concepts, Skills, Domains, Procedures, ResearchKnowledge,
    EducationalKnowledge

Thread safety:  threading.RLock on all public operations.
Lifecycle-gated: every public method raises LunaNotInitializedError before
    initialize() or after shutdown().

Part of the POLARIS Cognitive Substrate:
    LUNA → Knowledge  ← this module
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from subsystems.luna.exceptions import (
    KnowledgeRetrievalError,
    KnowledgeSearchError,
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractKnowledgeRetrievalEngine
from subsystems.luna.models import (
    EducationalKnowledge,
    Fact,
    Concept,
    KnowledgeDifficulty,
    KnowledgeDomain,
    KnowledgeRecord,
    KnowledgeStatus,
    KnowledgeType,
    Procedure,
    ResearchKnowledge,
    Skill,
    _utcnow,
)

logger = logging.getLogger(__name__)

_ENGINE_VERSION: str = "5.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _searchable_text(record: KnowledgeRecord) -> str:
    """Return all searchable text for a record concatenated as lowercase."""
    parts: list[str] = [
        record.name,
        record.description,
        " ".join(record.aliases),
        record.notes,
    ]
    # Type-specific extensions
    if isinstance(record, Fact):
        parts += [record.statement, record.formal_notation or ""]
    elif isinstance(record, Concept):
        parts += list(record.core_ideas) + list(record.applications)
    elif isinstance(record, Skill):
        parts += list(record.learning_outcomes) + list(record.practical_applications)
    elif isinstance(record, Procedure):
        parts += list(record.preconditions) + list(record.expected_outcomes)
    elif isinstance(record, ResearchKnowledge):
        parts += [record.abstract or "", " ".join(record.keywords)]
    elif isinstance(record, EducationalKnowledge):
        parts += (
            list(record.learning_objectives)
            + list(record.learning_outcomes)
        )
    return " ".join(p for p in parts if p).lower()


def _relevance_score(record: KnowledgeRecord, query_lower: str) -> float:
    """
    Compute a relevance score in [0.0, 1.0].

    Scoring:
        0.30  — exact name match
        0.20  — name starts-with query
        0.10  — name contains query
        0.05  — description / body contains query
      + 0.40  — confidence_score weight
    """
    if not query_lower:
        return record.metadata.confidence_score

    name_lower = record.name.lower()
    score = record.metadata.confidence_score * 0.40

    if name_lower == query_lower:
        score += 0.30
    elif name_lower.startswith(query_lower):
        score += 0.20
    elif query_lower in name_lower:
        score += 0.10

    if query_lower in record.description.lower():
        score += 0.05

    return min(score, 1.0)


def _get_tags(record: KnowledgeRecord) -> set[str]:
    """Extract tags from metadata (stored as frozenset or set)."""
    tags = getattr(record.metadata, "tags", None)
    if tags is None:
        return set()
    return set(tags)


def _get_domain_ids(record: KnowledgeRecord) -> list[str]:
    """Return domain_ids regardless of whether the record uses domain_ids or similar."""
    return list(getattr(record, "domain_ids", []))


def _get_prerequisite_ids(record: KnowledgeRecord) -> list[str]:
    """Return prerequisite-style IDs for dependency traversal."""
    if isinstance(record, Concept):
        return list(record.prerequisite_concept_ids)
    if isinstance(record, EducationalKnowledge):
        return list(record.prerequisite_knowledge_ids)
    if isinstance(record, Skill):
        return [p.prerequisite_skill_id for p in getattr(record, "prerequisites", [])]
    if isinstance(record, Procedure):
        return list(getattr(record, "prerequisite_procedure_ids", []))
    return []


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeRetrievalEngine(AbstractKnowledgeRetrievalEngine):
    """
    In-memory, thread-safe implementation of the LUNA Knowledge Retrieval
    Engine (v1).

    The engine does **not** own any knowledge stores.  It holds references to
    the seven individual LUNA engine instances and delegates all reads to them.
    It therefore stays thin and cache-friendly.

    Injected engines (all optional; absent engines simply return no results
    for the corresponding knowledge type):
        fact_engine, concept_engine, skill_engine, domain_engine,
        procedure_engine, research_engine, educational_engine

    Lifecycle::

        engine = KnowledgeRetrievalEngine(
            fact_engine=my_fact_engine,
            concept_engine=my_concept_engine,
            ...
        )
        engine.initialize()
        results = engine.search_all("PID controller")
        engine.shutdown()
    """

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

        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False

        # Observability counters
        self._query_count: int = 0
        self._cache_hit_count: int = 0
        self._started_at: Optional[datetime] = None

        # Simple single-record lookup cache (record_id → record)
        self._id_cache: dict[str, KnowledgeRecord] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            try:
                self._id_cache.clear()
                self._query_count = 0
                self._cache_hit_count = 0
                self._started_at = _utcnow()
                self._initialized = True
                logger.info("KnowledgeRetrievalEngine initialized (version=%s)", _ENGINE_VERSION)
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="KnowledgeRetrievalEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                return
            try:
                self._id_cache.clear()
                self._initialized = False
                logger.info(
                    "KnowledgeRetrievalEngine shutdown (queries=%d)", self._query_count
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="KnowledgeRetrievalEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def is_initialized(self) -> bool:
        return self._initialized

    # ─────────────────────────────────────────────────────────────────────────
    # GUARD
    # ─────────────────────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation=operation)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: ENGINE DISPATCH
    # ─────────────────────────────────────────────────────────────────────────

    def _engine_for_type(self, knowledge_type: KnowledgeType) -> Optional[Any]:
        return {
            KnowledgeType.FACT: self._fact_engine,
            KnowledgeType.CONCEPT: self._concept_engine,
            KnowledgeType.SKILL: self._skill_engine,
            KnowledgeType.DOMAIN: self._domain_engine,
            KnowledgeType.PROCEDURE: self._procedure_engine,
            KnowledgeType.RESEARCH: self._research_engine,
            KnowledgeType.EDUCATIONAL: self._educational_engine,
        }.get(knowledge_type)

    def _retrieve_record_direct(
        self, record_id: str, knowledge_type: KnowledgeType
    ) -> Optional[KnowledgeRecord]:
        """Delegate to the appropriate engine's retrieve method."""
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

    def _get_all_from_engine(
        self,
        knowledge_type: KnowledgeType,
        status_filter: Optional[list[KnowledgeStatus]] = None,
    ) -> list[KnowledgeRecord]:
        """Fetch all records of one type from the appropriate engine."""
        engine = self._engine_for_type(knowledge_type)
        if engine is None or not engine.is_initialized():
            return []
        try:
            kwargs: dict[str, Any] = {"limit": 10_000, "offset": 0}
            if status_filter:
                kwargs["status_filter"] = status_filter
            if knowledge_type == KnowledgeType.FACT:
                return engine.get_all_facts(**kwargs)
            elif knowledge_type == KnowledgeType.CONCEPT:
                return engine.get_all_concepts(**kwargs)
            elif knowledge_type == KnowledgeType.SKILL:
                return engine.get_all_skills(**kwargs)
            elif knowledge_type == KnowledgeType.DOMAIN:
                return engine.get_all_domains(
                    status_filter=status_filter if status_filter else None
                )
            elif knowledge_type == KnowledgeType.PROCEDURE:
                return engine.get_all_procedures(**kwargs)
            elif knowledge_type == KnowledgeType.RESEARCH:
                return engine.get_all_research(**kwargs)
            elif knowledge_type == KnowledgeType.EDUCATIONAL:
                return engine.get_all_educational(**kwargs)
        except Exception:
            return []
        return []

    def _active_types(
        self, knowledge_types: Optional[list[KnowledgeType]] = None
    ) -> list[KnowledgeType]:
        if knowledge_types:
            return knowledge_types
        return [
            KnowledgeType.FACT,
            KnowledgeType.CONCEPT,
            KnowledgeType.SKILL,
            KnowledgeType.DOMAIN,
            KnowledgeType.PROCEDURE,
            KnowledgeType.RESEARCH,
            KnowledgeType.EDUCATIONAL,
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # RETRIEVE BY ID
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_by_id(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> KnowledgeRecord:
        """
        Retrieve any knowledge record by ID and type.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeRetrievalError: Record not found.
        """
        self._require_initialized("retrieve_by_id")
        with self._lock:
            self._query_count += 1

            # Cache check
            cache_key = f"{knowledge_type.value}:{record_id}"
            if cache_key in self._id_cache:
                self._cache_hit_count += 1
                return self._id_cache[cache_key]

            record = self._retrieve_record_direct(record_id, knowledge_type)
            if record is None:
                raise KnowledgeRetrievalError(
                    f"Record not found: id='{record_id}' type='{knowledge_type.value}'",
                    context={"record_id": record_id, "knowledge_type": knowledge_type.value},
                )
            self._id_cache[cache_key] = record
            return record

    def retrieve_many(
        self,
        record_ids: list[str],
        knowledge_type: KnowledgeType,
    ) -> list[KnowledgeRecord]:
        """Retrieve multiple records in a single call. Missing IDs are silently skipped."""
        self._require_initialized("retrieve_many")
        with self._lock:
            self._query_count += 1
            results: list[KnowledgeRecord] = []
            for rid in record_ids:
                record = self._retrieve_record_direct(rid, knowledge_type)
                if record is not None:
                    results.append(record)
            return results

    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED FULL-TEXT SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    def search_all(
        self,
        query: str,
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        domain_ids: Optional[list[str]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        min_confidence: float = 0.0,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeRecord]:
        """
        Unified cross-type full-text search across all LUNA knowledge stores.

        Returns results sorted by relevance score descending.
        """
        self._require_initialized("search_all")
        with self._lock:
            self._query_count += 1

            active_statuses: set[KnowledgeStatus] = (
                set(status_filter)
                if status_filter
                else {KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE}
            )
            q = query.strip().lower()
            domain_set = set(domain_ids) if domain_ids else None
            types = self._active_types(knowledge_types)

            scored: list[tuple[float, KnowledgeRecord]] = []

            for kt in types:
                for record in self._get_all_from_engine(kt):
                    if record.status not in active_statuses:
                        continue
                    if record.metadata.confidence_score < min_confidence:
                        continue
                    if difficulty and record.difficulty != difficulty:
                        continue
                    if domain_set:
                        rec_domains = set(_get_domain_ids(record))
                        if not rec_domains.intersection(domain_set):
                            continue
                    if q and q not in _searchable_text(record):
                        continue
                    score = _relevance_score(record, q)
                    scored.append((score, record))

            scored.sort(key=lambda t: (-t[0], t[1].name.lower()))
            return [r for _, r in scored[offset: offset + limit]]

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH BY DOMAIN
    # ─────────────────────────────────────────────────────────────────────────

    def search_by_domain(
        self,
        domain_id: str,
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[KnowledgeRecord]:
        """Return all active records belonging to a domain, optionally filtered by type."""
        self._require_initialized("search_by_domain")
        with self._lock:
            self._query_count += 1
            types = self._active_types(knowledge_types)
            results: list[KnowledgeRecord] = []

            for kt in types:
                for record in self._get_all_from_engine(kt):
                    if not record.status.is_usable:
                        continue
                    if domain_id in _get_domain_ids(record):
                        results.append(record)

            results.sort(key=lambda r: (-r.metadata.confidence_score, r.name.lower()))
            return results[offset: offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH BY TAGS
    # ─────────────────────────────────────────────────────────────────────────

    def search_by_tags(
        self,
        tags: list[str],
        *,
        match_all: bool = False,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        limit: int = 100,
    ) -> list[KnowledgeRecord]:
        """
        Return records whose metadata.tags overlap with the given tag list.

        Args:
            tags:       Tags to match against.
            match_all:  When True, records must have ALL provided tags.
        """
        self._require_initialized("search_by_tags")
        with self._lock:
            self._query_count += 1
            tag_set = set(t.lower() for t in tags)
            types = self._active_types(knowledge_types)
            results: list[KnowledgeRecord] = []

            for kt in types:
                for record in self._get_all_from_engine(kt):
                    if not record.status.is_usable:
                        continue
                    record_tags = {t.lower() for t in _get_tags(record)}
                    if not record_tags:
                        continue
                    if match_all:
                        if not tag_set.issubset(record_tags):
                            continue
                    else:
                        if not tag_set.intersection(record_tags):
                            continue
                    results.append(record)

            results.sort(key=lambda r: (-r.metadata.confidence_score, r.name.lower()))
            return results[:limit]

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH BY TYPE
    # ─────────────────────────────────────────────────────────────────────────

    def search_by_type(
        self,
        query: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = None,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[KnowledgeRecord]:
        """Return all records of a specific knowledge type, optionally filtered by query."""
        self._require_initialized("search_by_type")
        with self._lock:
            self._query_count += 1
            # Resolve the type to query: callers may pass (query, type) positionally
            # or just (knowledge_type=...) as a keyword.
            kt = knowledge_type
            # If query was passed as a KnowledgeType (legacy positional call), treat it as the type
            if isinstance(query, KnowledgeType):
                kt = query
                query = None

            types = [kt] if kt is not None else list(self._active_types())
            active_statuses: set[KnowledgeStatus] = (
                set(status_filter)
                if status_filter
                else {KnowledgeStatus.DRAFT, KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE}
            )
            records: list[KnowledgeRecord] = []
            for t in types:
                for r in self._get_all_from_engine(t, status_filter):
                    if r.status not in active_statuses:
                        continue
                    if query:
                        q = query.strip().lower()
                        searchable = f"{r.name} {r.description}".lower()
                        if q not in searchable:
                            continue
                    records.append(r)
            records.sort(key=lambda r: (-r.metadata.confidence_score, r.name.lower()))
            return records[offset: offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH BY CONFIDENCE
    # ─────────────────────────────────────────────────────────────────────────

    def search_by_confidence(
        self,
        min_confidence: float,
        *,
        max_confidence: float = 1.0,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[KnowledgeRecord]:
        """Return records with confidence scores within [min_confidence, max_confidence]."""
        self._require_initialized("search_by_confidence")
        with self._lock:
            self._query_count += 1
            types = self._active_types(knowledge_types)
            results: list[KnowledgeRecord] = []
            for kt in types:
                for record in self._get_all_from_engine(kt):
                    if not record.status.is_usable:
                        continue
                    if min_confidence <= record.metadata.confidence_score <= max_confidence:
                        results.append(record)
            results.sort(key=lambda r: -r.metadata.confidence_score)
            return results[offset: offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH BY DIFFICULTY
    # ─────────────────────────────────────────────────────────────────────────

    def search_by_difficulty(
        self,
        difficulty: KnowledgeDifficulty,
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[KnowledgeRecord]:
        """Return all active records at the specified difficulty level."""
        self._require_initialized("search_by_difficulty")
        with self._lock:
            self._query_count += 1
            types = self._active_types(knowledge_types)
            results: list[KnowledgeRecord] = []
            for kt in types:
                for record in self._get_all_from_engine(kt):
                    if not record.status.is_usable:
                        continue
                    if record.difficulty == difficulty:
                        results.append(record)
            results.sort(key=lambda r: (-r.metadata.confidence_score, r.name.lower()))
            return results[offset: offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # SEMANTIC SEARCH  (name/description similarity ranking)
    # ─────────────────────────────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        top_k: int = 20,
    ) -> list[tuple[KnowledgeRecord, float]]:
        """
        Return the top-k most semantically similar records for a query,
        each paired with its relevance score.

        In-memory v1 implementation uses token-level overlap scoring over the
        full searchable text, weighted by confidence.

        Returns:
            List of (record, score) tuples ordered by score descending.
        """
        self._require_initialized("semantic_search")
        with self._lock:
            self._query_count += 1
            q = query.strip().lower()
            query_tokens = set(q.split())
            types = self._active_types(knowledge_types)
            scored: list[tuple[float, KnowledgeRecord]] = []

            for kt in types:
                for record in self._get_all_from_engine(kt):
                    if not record.status.is_usable:
                        continue
                    text = _searchable_text(record)
                    text_tokens = set(text.split())
                    if not text_tokens:
                        continue
                    overlap = len(query_tokens & text_tokens)
                    if overlap == 0:
                        continue
                    jaccard = overlap / len(query_tokens | text_tokens)
                    confidence_weight = record.metadata.confidence_score
                    score = jaccard * 0.60 + confidence_weight * 0.40
                    scored.append((score, record))

            scored.sort(key=lambda t: (-t[0], t[1].name.lower()))
            return [(r, s) for s, r in scored[:top_k]]

    # ─────────────────────────────────────────────────────────────────────────
    # RELATIONSHIP SEARCH  (traverse related_ids)
    # ─────────────────────────────────────────────────────────────────────────

    def relationship_search(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        depth: int = 1,
        limit: int = 50,
    ) -> list[KnowledgeRecord]:
        """
        Return records reachable from record_id via related_ids up to `depth` hops.

        Args:
            record_id:      Starting record.
            knowledge_type: Type of the starting record.
            depth:          Number of relationship hops to traverse (1 = direct only).
            limit:          Maximum results to return.

        Returns:
            List of related records (excluding the seed record), sorted by confidence.
        """
        self._require_initialized("relationship_search")
        with self._lock:
            self._query_count += 1

            visited: set[str] = {record_id}
            current_frontier: list[str] = [record_id]
            found: list[KnowledgeRecord] = []

            for _ in range(max(1, depth)):
                next_frontier: list[str] = []
                for rid in current_frontier:
                    # Try each type until we find the record
                    for kt in self._active_types():
                        record = self._retrieve_record_direct(rid, kt)
                        if record is not None:
                            for related_id in record.related_ids:
                                if related_id not in visited:
                                    visited.add(related_id)
                                    next_frontier.append(related_id)
                                    related = self._retrieve_record_direct(related_id, kt)
                                    if related is not None:
                                        found.append(related)
                            break
                current_frontier = next_frontier
                if not current_frontier:
                    break

            found.sort(key=lambda r: -r.metadata.confidence_score)
            return found[:limit]

    # ─────────────────────────────────────────────────────────────────────────
    # DEPENDENCY SEARCH  (traverse prerequisites)
    # ─────────────────────────────────────────────────────────────────────────

    def dependency_search(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        include_transitive: bool = False,
        limit: int = 100,
    ) -> list[KnowledgeRecord]:
        """
        Return the prerequisite / dependency records for a given record.

        Args:
            record_id:           Target record.
            knowledge_type:      Knowledge type of the target record.
            include_transitive:  When True, traverse the full dependency graph.
            limit:               Maximum results to return.

        Returns:
            Ordered list of dependency records (most foundational first).
        """
        self._require_initialized("dependency_search")
        with self._lock:
            self._query_count += 1

            root = self._retrieve_record_direct(record_id, knowledge_type)
            if root is None:
                return []

            visited: set[str] = {record_id}
            frontier: list[str] = _get_prerequisite_ids(root)
            results: list[KnowledgeRecord] = []

            while frontier:
                next_frontier: list[str] = []
                for pid in frontier:
                    if pid in visited:
                        continue
                    visited.add(pid)
                    dep_record = self._retrieve_record_direct(pid, knowledge_type)
                    if dep_record is not None:
                        results.append(dep_record)
                        if include_transitive:
                            for grandparent in _get_prerequisite_ids(dep_record):
                                if grandparent not in visited:
                                    next_frontier.append(grandparent)
                frontier = next_frontier if include_transitive else []

            # Most foundational (highest confidence, shallowest) first
            results.sort(key=lambda r: (-r.metadata.confidence_score, r.name.lower()))
            return results[:limit]

    # ─────────────────────────────────────────────────────────────────────────
    # RANKED RETRIEVAL  (top-N across all types)
    # ─────────────────────────────────────────────────────────────────────────

    def ranked_retrieval(
        self,
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        domain_ids: Optional[list[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeRecord]:
        """
        Return records ranked by confidence score descending.

        Useful for surfacing the most reliable knowledge in a domain or type.
        """
        self._require_initialized("ranked_retrieval")
        with self._lock:
            self._query_count += 1
            types = self._active_types(knowledge_types)
            domain_set = set(domain_ids) if domain_ids else None
            results: list[KnowledgeRecord] = []

            for kt in types:
                for record in self._get_all_from_engine(kt):
                    if not record.status.is_usable:
                        continue
                    if domain_set:
                        if not set(_get_domain_ids(record)).intersection(domain_set):
                            continue
                    results.append(record)

            results.sort(key=lambda r: (-r.metadata.confidence_score, r.name.lower()))
            return results[offset: offset + limit]

    # ─────────────────────────────────────────────────────────────────────────
    # GET RELATED  (AbstractKnowledgeRetrievalEngine contract)
    # ─────────────────────────────────────────────────────────────────────────

    def get_related(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        limit: int = 20,
    ) -> list[KnowledgeRecord]:
        """Return records referenced in the given record's related_ids."""
        self._require_initialized("get_related")
        with self._lock:
            self._query_count += 1
            record = self._retrieve_record_direct(record_id, knowledge_type)
            if record is None:
                return []

            results: list[KnowledgeRecord] = []
            for related_id in record.related_ids:
                # Try all types to resolve each related_id
                for kt in self._active_types():
                    related = self._retrieve_record_direct(related_id, kt)
                    if related is not None:
                        results.append(related)
                        break

            results.sort(key=lambda r: -r.metadata.confidence_score)
            return results[:limit]

    # ─────────────────────────────────────────────────────────────────────────
    # RELATED KNOWLEDGE DISCOVERY
    # ─────────────────────────────────────────────────────────────────────────

    def discover_related(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        same_domain_only: bool = False,
        limit: int = 20,
    ) -> list[KnowledgeRecord]:
        """
        Discover knowledge records that are conceptually related to a given
        record based on shared domain membership and similar names/descriptions.

        Args:
            record_id:         The seed record.
            knowledge_type:    Knowledge type of the seed.
            same_domain_only:  Restrict results to the seed's domains.
            limit:             Maximum results to return.

        Returns:
            List of related records sorted by relevance score.
        """
        self._require_initialized("discover_related")
        with self._lock:
            self._query_count += 1
            seed = self._retrieve_record_direct(record_id, knowledge_type)
            if seed is None:
                return []

            seed_domains = set(_get_domain_ids(seed))
            seed_name_tokens = set(seed.name.lower().split())
            types = self._active_types()
            scored: list[tuple[float, KnowledgeRecord]] = []

            for kt in types:
                for record in self._get_all_from_engine(kt):
                    if record.id == record_id:
                        continue
                    if not record.status.is_usable:
                        continue
                    if same_domain_only:
                        if not seed_domains.intersection(_get_domain_ids(record)):
                            continue
                    # Score by name token overlap
                    rec_name_tokens = set(record.name.lower().split())
                    overlap = len(seed_name_tokens & rec_name_tokens)
                    if overlap == 0:
                        continue
                    score = overlap / len(seed_name_tokens | rec_name_tokens)
                    # Domain bonus
                    if seed_domains.intersection(_get_domain_ids(record)):
                        score *= 1.20
                    score = min(score * 0.60 + record.metadata.confidence_score * 0.40, 1.0)
                    scored.append((score, record))

            scored.sort(key=lambda t: (-t[0], t[1].name.lower()))
            return [r for _, r in scored[:limit]]

    # ─────────────────────────────────────────────────────────────────────────
    # GET TOTAL COUNT
    # ─────────────────────────────────────────────────────────────────────────

    def get_total_count(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
        active_only: bool = False,
    ) -> int:
        """Return total record count, optionally scoped to one knowledge type."""
        self._require_initialized("get_total_count")
        with self._lock:
            types = [knowledge_type] if knowledge_type else self._active_types()
            total = 0
            for kt in types:
                engine = self._engine_for_type(kt)
                if engine is None or not engine.is_initialized():
                    continue
                try:
                    if kt == KnowledgeType.FACT:
                        total += engine.get_fact_count(active_only=active_only)
                    elif kt == KnowledgeType.CONCEPT:
                        total += engine.get_concept_count(active_only=active_only)
                    elif kt == KnowledgeType.SKILL:
                        total += engine.get_skill_count(active_only=active_only)
                    elif kt == KnowledgeType.DOMAIN:
                        total += engine.get_domain_count(active_only=active_only)
                    elif kt == KnowledgeType.PROCEDURE:
                        total += engine.get_procedure_count(active_only=active_only)
                    elif kt == KnowledgeType.RESEARCH:
                        total += engine.get_research_count(active_only=active_only)
                    elif kt == KnowledgeType.EDUCATIONAL:
                        total += engine.get_educational_count(active_only=active_only)
                except Exception:
                    pass
            return total

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "engine": "KnowledgeRetrievalEngine",
                "initialized": self._initialized,
                "record_count": self.get_total_count() if self._initialized else 0,
                "status": "healthy" if self._initialized else "offline",
            }

    def diagnostics_report(self) -> dict[str, Any]:
        with self._lock:
            total = self.get_total_count() if self._initialized else 0
            active = self.get_total_count(active_only=True) if self._initialized else 0
            cache_hit_rate = (
                self._cache_hit_count / self._query_count
                if self._query_count > 0
                else 0.0
            )
            return {
                "engine": "KnowledgeRetrievalEngine",
                "version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "status": "healthy" if self._initialized else "offline",
                "total_records": total,
                "active_records": active,
                "query_count": self._query_count,
                "cache_hit_count": self._cache_hit_count,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "id_cache_size": len(self._id_cache),
                "started_at": self._started_at.isoformat() if self._started_at else None,
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
        self._require_initialized("audit_report")
        with self._lock:
            type_counts: dict[str, int] = {}
            for kt in self._active_types():
                type_counts[kt.value] = self.get_total_count(knowledge_type=kt)
            cache_hit_rate = (
                self._cache_hit_count / self._query_count
                if self._query_count > 0
                else 0.0
            )
            return {
                "engine": "KnowledgeRetrievalEngine",
                "version": _ENGINE_VERSION,
                "query_count": self._query_count,
                "cache_hit_count": self._cache_hit_count,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "id_cache_size": len(self._id_cache),
                "total_records_by_type": type_counts,
                "total_records": sum(type_counts.values()),
                "generated_at": _utcnow().isoformat(),
            }


__all__ = ["KnowledgeRetrievalEngine"]