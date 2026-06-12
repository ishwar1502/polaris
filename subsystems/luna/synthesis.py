"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/synthesis.py

Concrete in-memory implementation of the LUNA Knowledge Synthesis Engine.

Responsibilities:
    - synthesize_knowledge         — gather and unify records from multiple domains
    - merge_knowledge              — merge two existing KnowledgeSynthesis records
    - infer_relationships          — derive ConceptRelationships from co-occurrence
    - build_concept_map            — construct a labelled concept adjacency map
    - build_skill_map              — construct a skill prerequisite map
    - build_domain_summary         — aggregate domain-level statistics
    - build_learning_path          — order concepts/skills by difficulty + prerequisites
    - create_knowledge_package     — bundle a synthesis into a portable KnowledgePackage
    - create_knowledge_composition — merge packages into a KnowledgeComposition
    - generate_synthesis_report    — produce a human-readable audit summary

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
    KnowledgeCompositionError,
    KnowledgePackageError,
    KnowledgeRetrievalError,
    KnowledgeSynthesisError,
    LunaLifecycleError,
    LunaNotInitializedError,
)
from subsystems.luna.interfaces import AbstractKnowledgeSynthesisEngine
from subsystems.luna.models import (
    Concept,
    ConceptRelationship,
    ConceptRelationshipType,
    EducationalKnowledge,
    Fact,
    KnowledgeComposition,
    KnowledgeDifficulty,
    KnowledgeDomain,
    KnowledgeMetadata,
    KnowledgePackage,
    KnowledgeRecord,
    KnowledgeSourceType,
    KnowledgeStatus,
    KnowledgeSynthesis,
    KnowledgeType,
    Procedure,
    ResearchKnowledge,
    Skill,
    _new_id,
    _utcnow,
)

logger = logging.getLogger(__name__)

_ENGINE_VERSION: str = "5.0.0"

# Minimum confidence score for a record to participate in synthesis.
_DEFAULT_MIN_CONFIDENCE: float = 0.50

# Strategy label written into every KnowledgeComposition built here.
_DEFAULT_COMPOSITION_STRATEGY: str = "curated"


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _difficulty_rank(record: KnowledgeRecord) -> int:
    """Return a numeric rank for sorting by difficulty ascending."""
    return record.difficulty.rank


def _usable(record: KnowledgeRecord, min_confidence: float) -> bool:
    """Return True if the record is active and meets the confidence floor."""
    return (
        record.status.is_usable
        and record.metadata.confidence_score >= min_confidence
    )


def _all_from_engine(engine: Any, knowledge_type: KnowledgeType) -> list[KnowledgeRecord]:
    """
    Pull every record from a single sub-engine using the canonical list method.
    Returns an empty list if the engine is absent or faults.
    """
    if engine is None or not engine.is_initialized():
        return []
    try:
        dispatch: dict[KnowledgeType, str] = {
            KnowledgeType.FACT: "get_all_facts",
            KnowledgeType.CONCEPT: "get_all_concepts",
            KnowledgeType.SKILL: "get_all_skills",
            KnowledgeType.DOMAIN: "get_all_domains",
            KnowledgeType.PROCEDURE: "get_all_procedures",
            KnowledgeType.RESEARCH: "get_all_research",
            KnowledgeType.EDUCATIONAL: "get_all_educational",
        }
        method_name = dispatch.get(knowledge_type)
        if method_name is None:
            return []
        method = getattr(engine, method_name, None)
        if method is None:
            return []
        # Domain engines may not accept keyword args in the same way.
        if knowledge_type == KnowledgeType.DOMAIN:
            return method()
        return method(limit=10_000, offset=0)
    except Exception:
        return []


def _prerequisite_ids(record: KnowledgeRecord) -> list[str]:
    """Return prerequisite IDs for any record type that carries them."""
    if isinstance(record, Concept):
        return list(record.prerequisite_concept_ids)
    if isinstance(record, Skill):
        return list(record.prerequisite_skill_ids)
    if isinstance(record, EducationalKnowledge):
        return list(record.prerequisite_knowledge_ids)
    if isinstance(record, Procedure):
        return list(record.required_concept_ids) + list(record.required_skill_ids)
    return []


def _domain_ids(record: KnowledgeRecord) -> list[str]:
    """Return the domain_ids of any knowledge record."""
    return list(getattr(record, "domain_ids", []))


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeSynthesisEngine(AbstractKnowledgeSynthesisEngine):
    """
    In-memory, thread-safe implementation of the LUNA Knowledge Synthesis
    Engine (v1).

    The engine does **not** own any individual knowledge stores. It holds
    references to the seven LUNA sub-engine instances and reads from them
    on demand. All higher-order structures (KnowledgeSynthesis,
    KnowledgePackage, KnowledgeComposition) are owned and stored here.

    Injected engines (all optional; absent engines return empty collections
    for the corresponding knowledge type):
        fact_engine, concept_engine, skill_engine, domain_engine,
        procedure_engine, research_engine, educational_engine

    Lifecycle::

        engine = KnowledgeSynthesisEngine(
            fact_engine=my_fact_engine,
            concept_engine=my_concept_engine,
            ...
        )
        engine.initialize()
        synthesis = engine.synthesize_knowledge(
            domain_ids=["d-ai", "d-robotics"],
            label="Autonomous Robotics",
        )
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

        # In-memory stores
        self._syntheses: dict[str, KnowledgeSynthesis] = {}
        self._packages: dict[str, KnowledgePackage] = {}
        self._compositions: dict[str, KnowledgeComposition] = {}

        # Observability
        self._synthesis_count: int = 0
        self._package_count: int = 0
        self._composition_count: int = 0
        self._started_at: Optional[datetime] = None

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            try:
                self._syntheses.clear()
                self._packages.clear()
                self._compositions.clear()
                self._synthesis_count = 0
                self._package_count = 0
                self._composition_count = 0
                self._started_at = _utcnow()
                self._initialized = True
                logger.info(
                    "KnowledgeSynthesisEngine initialized (version=%s)", _ENGINE_VERSION
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="KnowledgeSynthesisEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                return
            try:
                self._syntheses.clear()
                self._packages.clear()
                self._compositions.clear()
                self._initialized = False
                logger.info(
                    "KnowledgeSynthesisEngine shutdown "
                    "(syntheses=%d packages=%d compositions=%d)",
                    self._synthesis_count,
                    self._package_count,
                    self._composition_count,
                )
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="KnowledgeSynthesisEngine",
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

    def _active_types(
        self,
        knowledge_types: Optional[list[KnowledgeType]] = None,
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

    def _collect_records(
        self,
        domain_ids: Optional[list[str]],
        knowledge_types: Optional[list[KnowledgeType]],
        min_confidence: float,
    ) -> list[KnowledgeRecord]:
        """
        Collect all usable records from the specified domain(s) and type(s).
        If domain_ids is None or empty, collects across all domains.
        """
        domain_set: Optional[set[str]] = set(domain_ids) if domain_ids else None
        results: list[KnowledgeRecord] = []

        for kt in self._active_types(knowledge_types):
            engine = self._engine_for_type(kt)
            records = _all_from_engine(engine, kt)
            for record in records:
                if not _usable(record, min_confidence):
                    continue
                if domain_set is not None:
                    rec_domains = set(_domain_ids(record))
                    if not rec_domains.intersection(domain_set):
                        continue
                results.append(record)

        return results

    def _retrieve_synthesis_internal(
        self, synthesis_id: str
    ) -> KnowledgeSynthesis:
        s = self._syntheses.get(synthesis_id)
        if s is None:
            raise KnowledgeRetrievalError(
                f"Synthesis not found: id='{synthesis_id}'",
                context={"synthesis_id": synthesis_id},
            )
        return s

    def _retrieve_package_internal(self, package_id: str) -> KnowledgePackage:
        p = self._packages.get(package_id)
        if p is None:
            raise KnowledgePackageError(
                package_id=package_id,
                reason="Package not found in synthesis store",
            )
        return p

    # ─────────────────────────────────────────────────────────────────────────
    # SYNTHESIZE KNOWLEDGE
    # ─────────────────────────────────────────────────────────────────────────

    def synthesize_knowledge(
        self,
        domain_ids: list[str],
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        label: str = "",
    ) -> KnowledgeSynthesis:
        """
        Gather all usable, high-confidence knowledge from the given domains and
        synthesize it into a KnowledgeSynthesis record.

        Args:
            domain_ids:       Source domain IDs to gather knowledge from.
            knowledge_types:  Restrict to these knowledge types; None = all.
            min_confidence:   Minimum confidence threshold for inclusion.
            label:            Optional human-readable label for this synthesis.

        Returns:
            A new KnowledgeSynthesis record stored in the engine.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeSynthesisError:  If synthesis cannot be produced.
        """
        self._require_initialized("synthesize_knowledge")
        with self._lock:
            try:
                records = self._collect_records(
                    domain_ids, knowledge_types, min_confidence
                )

                concept_ids: list[str] = [
                    r.id for r in records if r.knowledge_type == KnowledgeType.CONCEPT
                ]
                fact_ids: list[str] = [
                    r.id for r in records if r.knowledge_type == KnowledgeType.FACT
                ]
                source_knowledge_ids: list[str] = [r.id for r in records]

                # Derive integration points: domains with ≥2 knowledge types present
                domain_type_sets: dict[str, set[str]] = defaultdict(set)
                for record in records:
                    for did in _domain_ids(record):
                        domain_type_sets[did].add(record.knowledge_type.value)
                integration_points: list[str] = [
                    did
                    for did, types in domain_type_sets.items()
                    if len(types) >= 2
                ]

                # Emergent insight: concepts that are referenced by ≥2 other types
                concept_ref_count: dict[str, int] = defaultdict(int)
                for record in records:
                    for prereq_id in _prerequisite_ids(record):
                        concept_ref_count[prereq_id] += 1
                emergent_insights: list[str] = [
                    cid for cid, cnt in concept_ref_count.items() if cnt >= 2
                ]

                avg_confidence = (
                    sum(r.metadata.confidence_score for r in records) / len(records)
                    if records
                    else 0.0
                )

                name = label or (
                    f"Synthesis of {', '.join(domain_ids[:3])}"
                    + (" ..." if len(domain_ids) > 3 else "")
                )
                description = (
                    f"Synthesized {len(records)} knowledge records "
                    f"from {len(domain_ids)} domain(s) "
                    f"with confidence ≥ {min_confidence:.2f}."
                )

                synthesis = KnowledgeSynthesis.create(
                    name=name,
                    description=description,
                    source_domain_ids=list(domain_ids),
                    source_knowledge_ids=source_knowledge_ids,
                    synthesis_rationale=(
                        f"Collected {len(records)} usable records across "
                        f"{len(domain_ids)} domain(s). "
                        f"Average confidence: {avg_confidence:.3f}."
                    ),
                )
                synthesis.synthesized_concept_ids.extend(concept_ids)
                synthesis.synthesized_fact_ids.extend(fact_ids)
                synthesis.integration_points.extend(integration_points)
                synthesis.emergent_insights.extend(emergent_insights)
                synthesis.confidence_score = round(
                    max(0.0, min(1.0, avg_confidence)), 4
                )

                self._syntheses[synthesis.id] = synthesis
                self._synthesis_count += 1
                logger.debug(
                    "synthesize_knowledge: id=%s records=%d domains=%d",
                    synthesis.id,
                    len(records),
                    len(domain_ids),
                )
                return synthesis

            except LunaNotInitializedError:
                raise
            except KnowledgeSynthesisError:
                raise
            except Exception as exc:
                raise KnowledgeSynthesisError(
                    f"synthesize_knowledge failed: {exc}",
                    context={"domain_ids": domain_ids},
                    cause=exc,
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # MERGE KNOWLEDGE
    # ─────────────────────────────────────────────────────────────────────────

    def merge_knowledge(
        self,
        synthesis_id_a: str,
        synthesis_id_b: str,
        *,
        label: str = "",
    ) -> KnowledgeSynthesis:
        """
        Merge two existing KnowledgeSynthesis records into a new one.

        The result contains the union of source domains, source knowledge IDs,
        synthesized concept/fact IDs, integration points, and emergent insights.
        Confidence score is the weighted average of the two inputs.

        Args:
            synthesis_id_a: First synthesis to merge.
            synthesis_id_b: Second synthesis to merge.
            label:          Optional label for the resulting synthesis.

        Returns:
            A new KnowledgeSynthesis stored in the engine.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeRetrievalError:  Either synthesis ID not found.
            KnowledgeSynthesisError:  Merge cannot be completed.
        """
        self._require_initialized("merge_knowledge")
        with self._lock:
            try:
                a = self._retrieve_synthesis_internal(synthesis_id_a)
                b = self._retrieve_synthesis_internal(synthesis_id_b)

                merged_domain_ids: list[str] = list(
                    dict.fromkeys(a.source_domain_ids + b.source_domain_ids)
                )
                merged_source_ids: list[str] = list(
                    dict.fromkeys(a.source_knowledge_ids + b.source_knowledge_ids)
                )
                merged_concept_ids: list[str] = list(
                    dict.fromkeys(
                        a.synthesized_concept_ids + b.synthesized_concept_ids
                    )
                )
                merged_fact_ids: list[str] = list(
                    dict.fromkeys(a.synthesized_fact_ids + b.synthesized_fact_ids)
                )
                merged_integration: list[str] = list(
                    dict.fromkeys(a.integration_points + b.integration_points)
                )
                merged_emergent: list[str] = list(
                    dict.fromkeys(a.emergent_insights + b.emergent_insights)
                )

                n_a = len(a.source_knowledge_ids) or 1
                n_b = len(b.source_knowledge_ids) or 1
                total = n_a + n_b
                merged_confidence = (
                    a.confidence_score * n_a + b.confidence_score * n_b
                ) / total

                name = label or f"Merge of [{a.name}] + [{b.name}]"
                description = (
                    f"Merged synthesis of {len(merged_source_ids)} unique records "
                    f"across {len(merged_domain_ids)} domain(s)."
                )
                rationale = (
                    f"Merged '{a.name}' ({n_a} records, conf={a.confidence_score:.3f}) "
                    f"with '{b.name}' ({n_b} records, conf={b.confidence_score:.3f})."
                )

                merged = KnowledgeSynthesis.create(
                    name=name,
                    description=description,
                    source_domain_ids=merged_domain_ids,
                    source_knowledge_ids=merged_source_ids,
                    synthesis_rationale=rationale,
                )
                merged.synthesized_concept_ids.extend(merged_concept_ids)
                merged.synthesized_fact_ids.extend(merged_fact_ids)
                merged.integration_points.extend(merged_integration)
                merged.emergent_insights.extend(merged_emergent)
                merged.confidence_score = round(
                    max(0.0, min(1.0, merged_confidence)), 4
                )

                self._syntheses[merged.id] = merged
                self._synthesis_count += 1
                logger.debug(
                    "merge_knowledge: merged=%s a=%s b=%s records=%d",
                    merged.id,
                    synthesis_id_a,
                    synthesis_id_b,
                    len(merged_source_ids),
                )
                return merged

            except (LunaNotInitializedError, KnowledgeRetrievalError):
                raise
            except KnowledgeSynthesisError:
                raise
            except Exception as exc:
                raise KnowledgeSynthesisError(
                    f"merge_knowledge failed: {exc}",
                    context={
                        "synthesis_id_a": synthesis_id_a,
                        "synthesis_id_b": synthesis_id_b,
                    },
                    cause=exc,
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # INFER RELATIONSHIPS
    # ─────────────────────────────────────────────────────────────────────────

    def infer_relationships(
        self,
        synthesis_id: str,
        *,
        min_co_occurrence: int = 2,
    ) -> list[ConceptRelationship]:
        """
        Infer ConceptRelationship instances from concept co-occurrence within a
        synthesis.

        Two concepts are considered related if they appear together in the
        same domain at least ``min_co_occurrence`` times.  The relationship
        type assigned is PART_OF; weight is derived from co-occurrence
        frequency normalized to [0.5, 1.0].

        Args:
            synthesis_id:       The synthesis to analyze.
            min_co_occurrence:  Minimum shared-domain overlap to emit a relationship.

        Returns:
            List of inferred ConceptRelationship instances (not persisted to
            any concept engine — callers should store them separately).

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeRetrievalError:  Synthesis ID not found.
            KnowledgeSynthesisError:  Inference cannot be completed.
        """
        self._require_initialized("infer_relationships")
        with self._lock:
            try:
                synthesis = self._retrieve_synthesis_internal(synthesis_id)

                # Collect concepts participating in this synthesis
                concept_ids = synthesis.synthesized_concept_ids
                if not concept_ids:
                    return []

                concept_engine = self._concept_engine
                if concept_engine is None or not concept_engine.is_initialized():
                    return []

                concepts: list[Concept] = []
                for cid in concept_ids:
                    try:
                        c = concept_engine.retrieve_concept(cid)
                        concepts.append(c)
                    except Exception:
                        continue

                if len(concepts) < 2:
                    return []

                # Build domain → concept_ids index
                domain_to_concepts: dict[str, list[str]] = defaultdict(list)
                for concept in concepts:
                    for did in concept.domain_ids:
                        domain_to_concepts[did].append(concept.id)

                # Co-occurrence counting: (concept_a, concept_b) → count
                co_counts: dict[tuple[str, str], int] = defaultdict(int)
                for ids in domain_to_concepts.values():
                    unique = list(dict.fromkeys(ids))
                    for i in range(len(unique)):
                        for j in range(i + 1, len(unique)):
                            key = (unique[i], unique[j])
                            co_counts[key] += 1

                if not co_counts:
                    return []

                max_co = max(co_counts.values()) or 1
                relationships: list[ConceptRelationship] = []

                for (source_id, target_id), count in co_counts.items():
                    if count < min_co_occurrence:
                        continue
                    weight = 0.5 + 0.5 * (count / max_co)
                    rel = ConceptRelationship.create(
                        source_concept_id=source_id,
                        target_concept_id=target_id,
                        relationship_type=ConceptRelationshipType.PART_OF,
                        description=(
                            f"Inferred from co-occurrence in synthesis '{synthesis.name}' "
                            f"(count={count})."
                        ),
                        weight=round(weight, 4),
                        is_bidirectional=True,
                    )
                    relationships.append(rel)

                logger.debug(
                    "infer_relationships: synthesis=%s inferred=%d",
                    synthesis_id,
                    len(relationships),
                )
                return relationships

            except (LunaNotInitializedError, KnowledgeRetrievalError):
                raise
            except KnowledgeSynthesisError:
                raise
            except Exception as exc:
                raise KnowledgeSynthesisError(
                    f"infer_relationships failed: {exc}",
                    context={"synthesis_id": synthesis_id},
                    cause=exc,
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD CONCEPT MAP
    # ─────────────────────────────────────────────────────────────────────────

    def build_concept_map(
        self,
        domain_ids: list[str],
        *,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        include_facts: bool = True,
    ) -> dict[str, Any]:
        """
        Build a labelled concept adjacency map for the given domains.

        The map captures every Concept, its prerequisite edges, its child
        edges, and (optionally) the Fact IDs attached to each concept.

        Returns:
            A dict with the schema::

                {
                    "generated_at": str,            # ISO-8601
                    "domain_ids": list[str],
                    "concept_count": int,
                    "nodes": {
                        "<concept_id>": {
                            "id": str,
                            "name": str,
                            "concept_type": str,
                            "difficulty": str,
                            "is_foundational": bool,
                            "domain_ids": list[str],
                            "prerequisite_ids": list[str],
                            "child_ids": list[str],
                            "fact_ids": list[str],          # empty when include_facts=False
                            "confidence": float,
                        }
                    },
                    "edges": [
                        {
                            "source": str,
                            "target": str,
                            "type": str,    # "prerequisite" | "child"
                        }
                    ]
                }

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeSynthesisError:  Map cannot be built.
        """
        self._require_initialized("build_concept_map")
        with self._lock:
            try:
                domain_set = set(domain_ids)
                concept_engine = self._concept_engine
                concepts: list[Concept] = []

                if concept_engine is not None and concept_engine.is_initialized():
                    all_concepts: list[Concept] = _all_from_engine(  # type: ignore[assignment]
                        concept_engine, KnowledgeType.CONCEPT
                    )
                    for c in all_concepts:
                        if not _usable(c, min_confidence):
                            continue
                        if domain_set and not set(c.domain_ids).intersection(domain_set):
                            continue
                        concepts.append(c)

                nodes: dict[str, Any] = {}
                edges: list[dict[str, str]] = []
                concept_id_set = {c.id for c in concepts}

                for c in concepts:
                    nodes[c.id] = {
                        "id": c.id,
                        "name": c.name,
                        "concept_type": c.concept_type.value,
                        "difficulty": c.difficulty.value,
                        "is_foundational": c.is_foundational,
                        "domain_ids": list(c.domain_ids),
                        "prerequisite_ids": list(c.prerequisite_concept_ids),
                        "child_ids": list(c.child_concept_ids),
                        "fact_ids": list(c.fact_ids) if include_facts else [],
                        "confidence": c.metadata.confidence_score,
                    }
                    for prereq_id in c.prerequisite_concept_ids:
                        if prereq_id in concept_id_set:
                            edges.append(
                                {
                                    "source": prereq_id,
                                    "target": c.id,
                                    "type": "prerequisite",
                                }
                            )
                    for child_id in c.child_concept_ids:
                        if child_id in concept_id_set:
                            edges.append(
                                {
                                    "source": c.id,
                                    "target": child_id,
                                    "type": "child",
                                }
                            )

                return {
                    "generated_at": _utcnow().isoformat(),
                    "domain_ids": list(domain_ids),
                    "concept_count": len(nodes),
                    "nodes": nodes,
                    "edges": edges,
                }

            except LunaNotInitializedError:
                raise
            except KnowledgeSynthesisError:
                raise
            except Exception as exc:
                raise KnowledgeSynthesisError(
                    f"build_concept_map failed: {exc}",
                    context={"domain_ids": domain_ids},
                    cause=exc,
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD SKILL MAP
    # ─────────────────────────────────────────────────────────────────────────

    def build_skill_map(
        self,
        domain_ids: list[str],
        *,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> dict[str, Any]:
        """
        Build a skill prerequisite map for the given domains.

        Returns:
            A dict with the schema::

                {
                    "generated_at": str,
                    "domain_ids": list[str],
                    "skill_count": int,
                    "nodes": {
                        "<skill_id>": {
                            "id": str,
                            "name": str,
                            "skill_type": str,
                            "difficulty": str,
                            "domain_ids": list[str],
                            "prerequisite_skill_ids": list[str],
                            "sub_skill_ids": list[str],
                            "required_concept_ids": list[str],
                            "required_tools": list[str],
                            "confidence": float,
                        }
                    },
                    "edges": [
                        {
                            "source": str,
                            "target": str,
                            "type": str,    # "prerequisite" | "sub_skill"
                        }
                    ]
                }

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeSynthesisError:  Map cannot be built.
        """
        self._require_initialized("build_skill_map")
        with self._lock:
            try:
                domain_set = set(domain_ids)
                skill_engine = self._skill_engine
                skills: list[Skill] = []

                if skill_engine is not None and skill_engine.is_initialized():
                    all_skills: list[Skill] = _all_from_engine(  # type: ignore[assignment]
                        skill_engine, KnowledgeType.SKILL
                    )
                    for s in all_skills:
                        if not _usable(s, min_confidence):
                            continue
                        if domain_set and not set(s.domain_ids).intersection(domain_set):
                            continue
                        skills.append(s)

                nodes: dict[str, Any] = {}
                edges: list[dict[str, str]] = []
                skill_id_set = {s.id for s in skills}

                for s in skills:
                    nodes[s.id] = {
                        "id": s.id,
                        "name": s.name,
                        "skill_type": s.skill_type.value,
                        "difficulty": s.difficulty.value,
                        "domain_ids": list(s.domain_ids),
                        "prerequisite_skill_ids": list(s.prerequisite_skill_ids),
                        "sub_skill_ids": list(s.sub_skill_ids),
                        "required_concept_ids": list(s.required_concept_ids),
                        "required_tools": list(s.required_tools),
                        "confidence": s.metadata.confidence_score,
                    }
                    for prereq_id in s.prerequisite_skill_ids:
                        if prereq_id in skill_id_set:
                            edges.append(
                                {
                                    "source": prereq_id,
                                    "target": s.id,
                                    "type": "prerequisite",
                                }
                            )
                    for sub_id in s.sub_skill_ids:
                        if sub_id in skill_id_set:
                            edges.append(
                                {
                                    "source": s.id,
                                    "target": sub_id,
                                    "type": "sub_skill",
                                }
                            )

                return {
                    "generated_at": _utcnow().isoformat(),
                    "domain_ids": list(domain_ids),
                    "skill_count": len(nodes),
                    "nodes": nodes,
                    "edges": edges,
                }

            except LunaNotInitializedError:
                raise
            except KnowledgeSynthesisError:
                raise
            except Exception as exc:
                raise KnowledgeSynthesisError(
                    f"build_skill_map failed: {exc}",
                    context={"domain_ids": domain_ids},
                    cause=exc,
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD DOMAIN SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def build_domain_summary(
        self,
        domain_id: str,
        *,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> dict[str, Any]:
        """
        Aggregate statistics and metadata for a single knowledge domain.

        Returns:
            A dict with the schema::

                {
                    "domain_id": str,
                    "domain_name": str,
                    "generated_at": str,
                    "record_counts": {
                        "facts": int, "concepts": int, "skills": int,
                        "procedures": int, "research": int, "educational": int,
                        "total": int,
                    },
                    "average_confidence": float,
                    "difficulty_distribution": { "<level>": int, ... },
                    "foundational_concept_ids": list[str],
                    "root_skill_ids": list[str],
                    "sub_domain_ids": list[str],
                    "tags": list[str],
                }

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeSynthesisError:  Summary cannot be built.
        """
        self._require_initialized("build_domain_summary")
        with self._lock:
            try:
                records = self._collect_records([domain_id], None, min_confidence)

                counts: dict[str, int] = {
                    "facts": 0,
                    "concepts": 0,
                    "skills": 0,
                    "procedures": 0,
                    "research": 0,
                    "educational": 0,
                    "total": len(records),
                }
                type_to_key: dict[KnowledgeType, str] = {
                    KnowledgeType.FACT: "facts",
                    KnowledgeType.CONCEPT: "concepts",
                    KnowledgeType.SKILL: "skills",
                    KnowledgeType.PROCEDURE: "procedures",
                    KnowledgeType.RESEARCH: "research",
                    KnowledgeType.EDUCATIONAL: "educational",
                }
                difficulty_dist: dict[str, int] = defaultdict(int)
                all_tags: set[str] = set()
                total_confidence = 0.0

                for r in records:
                    key = type_to_key.get(r.knowledge_type)
                    if key:
                        counts[key] += 1
                    difficulty_dist[r.difficulty.value] += 1
                    all_tags.update(r.metadata.tags)
                    total_confidence += r.metadata.confidence_score

                avg_confidence = (
                    total_confidence / len(records) if records else 0.0
                )

                foundational_concept_ids: list[str] = [
                    r.id
                    for r in records
                    if isinstance(r, Concept) and r.is_foundational
                ]

                # Root skills = skills with no prerequisites within this domain
                domain_skill_ids = {
                    r.id for r in records if r.knowledge_type == KnowledgeType.SKILL
                }
                root_skill_ids: list[str] = [
                    r.id
                    for r in records
                    if r.knowledge_type == KnowledgeType.SKILL
                    and isinstance(r, Skill)
                    and not any(
                        p in domain_skill_ids for p in r.prerequisite_skill_ids
                    )
                ]

                # Sub-domains
                sub_domain_ids: list[str] = []
                domain_engine = self._domain_engine
                if domain_engine is not None and domain_engine.is_initialized():
                    try:
                        domain_obj: KnowledgeDomain = domain_engine.retrieve_domain(
                            domain_id
                        )
                        domain_name = domain_obj.name
                        sub_domain_ids = list(domain_obj.sub_domain_ids)
                    except Exception:
                        domain_name = domain_id
                else:
                    domain_name = domain_id

                return {
                    "domain_id": domain_id,
                    "domain_name": domain_name,
                    "generated_at": _utcnow().isoformat(),
                    "record_counts": counts,
                    "average_confidence": round(avg_confidence, 4),
                    "difficulty_distribution": dict(difficulty_dist),
                    "foundational_concept_ids": foundational_concept_ids,
                    "root_skill_ids": root_skill_ids,
                    "sub_domain_ids": sub_domain_ids,
                    "tags": sorted(all_tags),
                }

            except LunaNotInitializedError:
                raise
            except KnowledgeSynthesisError:
                raise
            except Exception as exc:
                raise KnowledgeSynthesisError(
                    f"build_domain_summary failed: {exc}",
                    context={"domain_id": domain_id},
                    cause=exc,
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD LEARNING PATH
    # ─────────────────────────────────────────────────────────────────────────

    def build_learning_path(
        self,
        domain_ids: list[str],
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        target_difficulty: Optional[KnowledgeDifficulty] = None,
    ) -> list[dict[str, Any]]:
        """
        Produce a topologically ordered learning sequence for the given domains.

        Records are ordered so that prerequisites always precede dependants.
        Within the same prerequisite tier, records are sorted by difficulty
        ascending, then by name alphabetically.

        Args:
            domain_ids:         Domains to include.
            knowledge_types:    Optional type restriction (default: Concepts + Skills).
            min_confidence:     Minimum confidence threshold.
            target_difficulty:  When provided, exclude records harder than this level.

        Returns:
            Ordered list of lightweight record descriptors::

                [
                    {
                        "position": int,        # 1-based order
                        "id": str,
                        "name": str,
                        "knowledge_type": str,
                        "difficulty": str,
                        "domain_ids": list[str],
                        "prerequisite_ids": list[str],
                        "confidence": float,
                    },
                    ...
                ]

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeSynthesisError:  Path cannot be built.
        """
        self._require_initialized("build_learning_path")
        with self._lock:
            try:
                types = knowledge_types or [
                    KnowledgeType.CONCEPT,
                    KnowledgeType.SKILL,
                ]
                records = self._collect_records(domain_ids, types, min_confidence)

                if target_difficulty is not None:
                    records = [
                        r
                        for r in records
                        if not r.difficulty.is_harder_than(target_difficulty)
                    ]

                # Build prerequisite-based topological sort (Kahn's algorithm)
                record_map: dict[str, KnowledgeRecord] = {r.id: r for r in records}
                in_degree: dict[str, int] = {r.id: 0 for r in records}
                dependants: dict[str, list[str]] = defaultdict(list)

                for record in records:
                    for prereq_id in _prerequisite_ids(record):
                        if prereq_id in record_map:
                            in_degree[record.id] += 1
                            dependants[prereq_id].append(record.id)

                # Queue starts with all records that have no prerequisites
                queue: list[KnowledgeRecord] = sorted(
                    [r for r in records if in_degree[r.id] == 0],
                    key=lambda r: (_difficulty_rank(r), r.name.lower()),
                )
                ordered: list[KnowledgeRecord] = []
                visited: set[str] = set()

                while queue:
                    # Pop first (lowest difficulty, alphabetical)
                    current = queue.pop(0)
                    if current.id in visited:
                        continue
                    visited.add(current.id)
                    ordered.append(current)

                    # Reduce in-degree of dependants; enqueue newly ready ones
                    newly_ready: list[KnowledgeRecord] = []
                    for dep_id in dependants.get(current.id, []):
                        if dep_id in visited:
                            continue
                        in_degree[dep_id] -= 1
                        if in_degree[dep_id] == 0:
                            dep = record_map.get(dep_id)
                            if dep is not None:
                                newly_ready.append(dep)
                    newly_ready.sort(
                        key=lambda r: (_difficulty_rank(r), r.name.lower())
                    )
                    queue = newly_ready + queue

                # Any remaining unvisited records (due to cycles or orphans)
                for r in sorted(
                    [r for r in records if r.id not in visited],
                    key=lambda r: (_difficulty_rank(r), r.name.lower()),
                ):
                    ordered.append(r)

                path: list[dict[str, Any]] = []
                for position, record in enumerate(ordered, start=1):
                    path.append(
                        {
                            "position": position,
                            "id": record.id,
                            "name": record.name,
                            "knowledge_type": record.knowledge_type.value,
                            "difficulty": record.difficulty.value,
                            "domain_ids": _domain_ids(record),
                            "prerequisite_ids": _prerequisite_ids(record),
                            "confidence": record.metadata.confidence_score,
                        }
                    )

                logger.debug(
                    "build_learning_path: domain_ids=%s path_len=%d",
                    domain_ids,
                    len(path),
                )
                return path

            except LunaNotInitializedError:
                raise
            except KnowledgeSynthesisError:
                raise
            except Exception as exc:
                raise KnowledgeSynthesisError(
                    f"build_learning_path failed: {exc}",
                    context={"domain_ids": domain_ids},
                    cause=exc,
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # CREATE KNOWLEDGE PACKAGE
    # ─────────────────────────────────────────────────────────────────────────

    def create_knowledge_package(
        self,
        synthesis_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        purpose: str = "",
        target_consumer: str = "ORION",
        domain_ids: Optional[list[str]] = None,
        concept_ids: Optional[list[str]] = None,
        fact_ids: Optional[list[str]] = None,
        skill_ids: Optional[list[str]] = None,
        label: Optional[str] = None,
        notes: str = "",
    ) -> KnowledgePackage:
        """
        Bundle a KnowledgeSynthesis into a portable KnowledgePackage.

        Partitions the synthesis source IDs into per-type buckets and populates
        all relevant KnowledgePackage fields.  Marks the package ``is_complete``
        once assembled.

        Args:
            synthesis_id:     The synthesis to package.
            purpose:          Intended use of this package.
            target_consumer:  Consumer module name (e.g. "ORION").
            label:            Optional name override; defaults to synthesis name.
            notes:            Free-text notes stored in metadata.

        Returns:
            A fully populated KnowledgePackage stored in the engine.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeRetrievalError:  Synthesis not found.
            KnowledgePackageError:    Package cannot be assembled.
        """
        self._require_initialized("create_knowledge_package")
        with self._lock:
            # Direct-creation path: caller provides name/description explicitly
            # (no synthesis_id required)
            if synthesis_id is None:
                try:
                    package = KnowledgePackage.create(
                        name=name or label or "Knowledge Package",
                        description=description or "",
                        purpose=purpose or "Direct package",
                        target_consumer=target_consumer,
                    )
                    if domain_ids:
                        package.domain_ids.extend(domain_ids)
                    if concept_ids:
                        package.concept_ids.extend(concept_ids)
                    if fact_ids:
                        package.fact_ids.extend(fact_ids)
                    if skill_ids:
                        package.skill_ids.extend(skill_ids)
                    package.is_complete = True
                    self._packages[package.id] = package
                    self._package_count += 1
                    return package
                except Exception as exc:
                    raise KnowledgePackageError(
                        package_id="<new>",
                        reason=str(exc),
                    ) from exc

            try:
                synthesis = self._retrieve_synthesis_internal(synthesis_id)

                # Resolve all source IDs to their knowledge types
                type_buckets: dict[KnowledgeType, list[str]] = defaultdict(list)

                for kid in synthesis.source_knowledge_ids:
                    resolved = False
                    for kt in self._active_types():
                        engine = self._engine_for_type(kt)
                        if engine is None or not engine.is_initialized():
                            continue
                        try:
                            dispatch: dict[KnowledgeType, str] = {
                                KnowledgeType.FACT: "retrieve_fact",
                                KnowledgeType.CONCEPT: "retrieve_concept",
                                KnowledgeType.SKILL: "retrieve_skill",
                                KnowledgeType.DOMAIN: "retrieve_domain",
                                KnowledgeType.PROCEDURE: "retrieve_procedure",
                                KnowledgeType.RESEARCH: "retrieve_research",
                                KnowledgeType.EDUCATIONAL: "retrieve_educational",
                            }
                            method_name = dispatch.get(kt)
                            if method_name is None:
                                continue
                            method = getattr(engine, method_name, None)
                            if method is None:
                                continue
                            method(kid)  # will raise if not found
                            type_buckets[kt].append(kid)
                            resolved = True
                            break
                        except Exception:
                            continue
                    # If the ID could not be resolved, silently skip it;
                    # synthesis may reference records no longer present.
                    _ = resolved

                # Determine representative difficulty (modal difficulty)
                all_records = self._collect_records(
                    synthesis.source_domain_ids, None, 0.0
                )
                diff_counts: dict[KnowledgeDifficulty, int] = defaultdict(int)
                for r in all_records:
                    diff_counts[r.difficulty] += 1
                representative_difficulty: KnowledgeDifficulty = (
                    max(diff_counts, key=lambda d: diff_counts[d])
                    if diff_counts
                    else KnowledgeDifficulty.INTERMEDIATE
                )

                package = KnowledgePackage.create(
                    name=label or synthesis.name,
                    description=synthesis.description,
                    purpose=purpose or f"Packaged from synthesis '{synthesis.name}'",
                    target_consumer=target_consumer,
                    difficulty=representative_difficulty,
                )
                package.domain_ids.extend(synthesis.source_domain_ids)
                package.concept_ids.extend(
                    type_buckets.get(KnowledgeType.CONCEPT, [])
                )
                package.fact_ids.extend(
                    type_buckets.get(KnowledgeType.FACT, [])
                )
                package.skill_ids.extend(
                    type_buckets.get(KnowledgeType.SKILL, [])
                )
                package.procedure_ids.extend(
                    type_buckets.get(KnowledgeType.PROCEDURE, [])
                )
                package.research_ids.extend(
                    type_buckets.get(KnowledgeType.RESEARCH, [])
                )
                package.educational_ids.extend(
                    type_buckets.get(KnowledgeType.EDUCATIONAL, [])
                )
                package.synthesis_id = synthesis_id
                package.is_complete = True

                self._packages[package.id] = package
                self._package_count += 1
                logger.debug(
                    "create_knowledge_package: id=%s synthesis=%s items=%d",
                    package.id,
                    synthesis_id,
                    package.total_items,
                )
                return package

            except (LunaNotInitializedError, KnowledgeRetrievalError):
                raise
            except KnowledgePackageError:
                raise
            except Exception as exc:
                raise KnowledgePackageError(
                    package_id="<new>",
                    reason=str(exc),
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # CREATE KNOWLEDGE COMPOSITION
    # ─────────────────────────────────────────────────────────────────────────

    def create_knowledge_composition(
        self,
        package_ids: list[str],
        *,
        label: str = "",
        confidence_threshold: float = _DEFAULT_MIN_CONFIDENCE,
        composition_strategy: str = _DEFAULT_COMPOSITION_STRATEGY,
    ) -> KnowledgeComposition:
        """
        Compose multiple KnowledgePackages into a single KnowledgeComposition.

        Included IDs are the union of all record IDs across the input packages
        whose source confidence meets the threshold. Excluded IDs are those
        that were present but fell below the threshold.

        Args:
            package_ids:            Packages to compose.
            label:                  Optional name for this composition.
            confidence_threshold:   Confidence floor for inclusion.
            composition_strategy:   Strategy label (e.g. "curated", "exhaustive").

        Returns:
            A KnowledgeComposition stored in the engine.

        Raises:
            LunaNotInitializedError:   Engine not initialized.
            KnowledgePackageError:     A package ID is not found.
            KnowledgeCompositionError: Composition cannot be completed.
        """
        self._require_initialized("create_knowledge_composition")
        with self._lock:
            try:
                if not package_ids:
                    raise KnowledgeCompositionError(
                        component_ids=package_ids,
                        reason="No package IDs provided for composition.",
                    )

                all_packages = [
                    self._retrieve_package_internal(pid) for pid in package_ids
                ]

                # Collect all IDs across all packages
                candidate_ids: set[str] = set()
                for pkg in all_packages:
                    candidate_ids.update(pkg.concept_ids)
                    candidate_ids.update(pkg.fact_ids)
                    candidate_ids.update(pkg.skill_ids)
                    candidate_ids.update(pkg.procedure_ids)
                    candidate_ids.update(pkg.research_ids)
                    candidate_ids.update(pkg.educational_ids)

                # Resolve confidence for each candidate
                included: list[str] = []
                excluded: list[str] = []

                for kid in sorted(candidate_ids):
                    conf = self._resolve_record_confidence(kid)
                    if conf >= confidence_threshold:
                        included.append(kid)
                    else:
                        excluded.append(kid)

                composition = KnowledgeComposition.create(
                    package_id=package_ids[0] if len(package_ids) == 1 else _new_id(),
                    included_ids=included,
                    excluded_ids=excluded,
                    inclusion_rationale=(
                        f"Records with confidence ≥ {confidence_threshold:.2f} "
                        f"from {len(all_packages)} package(s): "
                        + (label or ", ".join(pkg.name for pkg in all_packages[:3]))
                    ),
                    exclusion_rationale=(
                        f"Records with confidence < {confidence_threshold:.2f} "
                        f"were excluded from the composition."
                    ),
                    composition_strategy=composition_strategy,
                    confidence_threshold=confidence_threshold,
                )

                self._compositions[composition.id] = composition
                self._composition_count += 1
                logger.debug(
                    "create_knowledge_composition: id=%s packages=%d "
                    "included=%d excluded=%d",
                    composition.id,
                    len(package_ids),
                    len(included),
                    len(excluded),
                )
                return composition

            except (LunaNotInitializedError, KnowledgePackageError):
                raise
            except KnowledgeCompositionError:
                raise
            except Exception as exc:
                raise KnowledgeCompositionError(
                    component_ids=package_ids,
                    reason=str(exc),
                ) from exc

    def _resolve_record_confidence(self, record_id: str) -> float:
        """
        Attempt to resolve the confidence score of a record by probing all
        sub-engines.  Returns 0.0 if the record cannot be found in any engine.
        """
        dispatch: dict[KnowledgeType, str] = {
            KnowledgeType.FACT: "retrieve_fact",
            KnowledgeType.CONCEPT: "retrieve_concept",
            KnowledgeType.SKILL: "retrieve_skill",
            KnowledgeType.DOMAIN: "retrieve_domain",
            KnowledgeType.PROCEDURE: "retrieve_procedure",
            KnowledgeType.RESEARCH: "retrieve_research",
            KnowledgeType.EDUCATIONAL: "retrieve_educational",
        }
        for kt, method_name in dispatch.items():
            engine = self._engine_for_type(kt)
            if engine is None or not engine.is_initialized():
                continue
            method = getattr(engine, method_name, None)
            if method is None:
                continue
            try:
                record: KnowledgeRecord = method(record_id)
                return record.metadata.confidence_score
            except Exception:
                continue
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE SYNTHESIS REPORT
    # ─────────────────────────────────────────────────────────────────────────

    def generate_synthesis_report(
        self,
        synthesis_id: str,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive human-readable report for a KnowledgeSynthesis.

        The report includes:
            - Synthesis metadata and provenance
            - Record counts by knowledge type
            - Difficulty distribution across all source records
            - Integration coverage (domains with multi-type knowledge)
            - Emergent insights (cross-referenced concepts)
            - Confidence statistics
            - Associated packages

        Returns:
            A structured dict containing the full report.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            KnowledgeRetrievalError:  Synthesis not found.
            KnowledgeSynthesisError:  Report cannot be generated.
        """
        self._require_initialized("generate_synthesis_report")
        with self._lock:
            try:
                synthesis = self._retrieve_synthesis_internal(synthesis_id)

                # Resolve all source records for statistics
                records = self._collect_records(
                    synthesis.source_domain_ids, None, 0.0
                )

                type_counts: dict[str, int] = defaultdict(int)
                diff_dist: dict[str, int] = defaultdict(int)
                conf_sum = 0.0
                conf_min = 1.0
                conf_max = 0.0
                active_count = 0
                validated_count = 0

                for r in records:
                    type_counts[r.knowledge_type.value] += 1
                    diff_dist[r.difficulty.value] += 1
                    c = r.metadata.confidence_score
                    conf_sum += c
                    conf_min = min(conf_min, c)
                    conf_max = max(conf_max, c)
                    if r.status.is_usable:
                        active_count += 1
                    if r.metadata.validation_status.is_valid:
                        validated_count += 1

                total = len(records)
                avg_confidence = conf_sum / total if total else 0.0

                # Packages backed by this synthesis
                linked_package_ids: list[str] = [
                    pid
                    for pid, pkg in self._packages.items()
                    if pkg.synthesis_id == synthesis_id
                ]

                report: dict[str, Any] = {
                    "synthesis_id": synthesis.id,
                    "name": synthesis.name,
                    "description": synthesis.description,
                    "generated_at": _utcnow().isoformat(),
                    "synthesis_created_at": synthesis.created_at.isoformat(),
                    "synthesis_updated_at": synthesis.updated_at.isoformat(),
                    "synthesis_confidence_score": synthesis.confidence_score,
                    "is_cross_domain": synthesis.is_cross_domain,
                    "is_rich": synthesis.is_rich,
                    "source_domains": synthesis.source_domain_ids,
                    "domain_count": synthesis.domain_count,
                    "source_record_count": len(synthesis.source_knowledge_ids),
                    "resolved_record_count": total,
                    "active_record_count": active_count,
                    "validated_record_count": validated_count,
                    "record_counts_by_type": dict(type_counts),
                    "difficulty_distribution": dict(diff_dist),
                    "confidence_statistics": {
                        "average": round(avg_confidence, 4),
                        "minimum": round(conf_min, 4) if total else 0.0,
                        "maximum": round(conf_max, 4) if total else 0.0,
                    },
                    "synthesized_concept_count": len(synthesis.synthesized_concept_ids),
                    "synthesized_fact_count": len(synthesis.synthesized_fact_ids),
                    "integration_point_count": len(synthesis.integration_points),
                    "integration_points": synthesis.integration_points,
                    "emergent_insight_count": len(synthesis.emergent_insights),
                    "emergent_insights": synthesis.emergent_insights,
                    "synthesis_rationale": synthesis.synthesis_rationale,
                    "linked_package_ids": linked_package_ids,
                    "linked_package_count": len(linked_package_ids),
                    "engine_version": _ENGINE_VERSION,
                }

                logger.debug(
                    "generate_synthesis_report: synthesis=%s records=%d packages=%d",
                    synthesis_id,
                    total,
                    len(linked_package_ids),
                )
                return report

            except (LunaNotInitializedError, KnowledgeRetrievalError):
                raise
            except KnowledgeSynthesisError:
                raise
            except Exception as exc:
                raise KnowledgeSynthesisError(
                    f"generate_synthesis_report failed: {exc}",
                    context={"synthesis_id": synthesis_id},
                    cause=exc,
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # AbstractKnowledgeSynthesisEngine CONTRACT
    # ─────────────────────────────────────────────────────────────────────────

    def synthesize(
        self,
        domain_ids: list[str],
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        label: str = "",
    ) -> KnowledgeSynthesis:
        """Alias for synthesize_knowledge (satisfies AbstractKnowledgeSynthesisEngine)."""
        return self.synthesize_knowledge(
            domain_ids,
            knowledge_types=knowledge_types,
            min_confidence=min_confidence,
            label=label,
        )

    def build_package(
        self,
        synthesis_id: str,
        *,
        label: Optional[str] = None,
        notes: str = "",
    ) -> KnowledgePackage:
        """Alias for create_knowledge_package (satisfies AbstractKnowledgeSynthesisEngine)."""
        return self.create_knowledge_package(
            synthesis_id,
            label=label,
            notes=notes,
        )

    def compose(
        self,
        package_ids: list[str],
        *,
        label: str = "",
    ) -> KnowledgeComposition:
        """Alias for create_knowledge_composition (satisfies AbstractKnowledgeSynthesisEngine)."""
        return self.create_knowledge_composition(package_ids, label=label)

    def retrieve_synthesis(self, synthesis_id: str) -> KnowledgeSynthesis:
        """Fetch a synthesis record by ID."""
        self._require_initialized("retrieve_synthesis")
        with self._lock:
            return self._retrieve_synthesis_internal(synthesis_id)

    def retrieve_package(self, package_id: str) -> KnowledgePackage:
        """Fetch a package by ID."""
        self._require_initialized("retrieve_package")
        with self._lock:
            return self._retrieve_package_internal(package_id)

    def list_syntheses(self, *, limit: int = 50) -> list[KnowledgeSynthesis]:
        """Return recent synthesis records, newest first."""
        self._require_initialized("list_syntheses")
        with self._lock:
            items = sorted(
                self._syntheses.values(),
                key=lambda s: s.created_at,
                reverse=True,
            )
            return items[:limit]

    def list_packages(self, *, limit: int = 50) -> list[KnowledgePackage]:
        """Return recent package records, newest first."""
        self._require_initialized("list_packages")
        with self._lock:
            items = sorted(
                self._packages.values(),
                key=lambda p: p.assembled_at,
                reverse=True,
            )
            return items[:limit]

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "engine": "KnowledgeSynthesisEngine",
                "initialized": self._initialized,
                "record_count": (
                    len(self._syntheses)
                    + len(self._packages)
                    + len(self._compositions)
                ),
                "status": "healthy" if self._initialized else "offline",
            }

    def diagnostics_report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "engine": "KnowledgeSynthesisEngine",
                "version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "status": "healthy" if self._initialized else "offline",
                "record_count": (
                    len(self._syntheses)
                    + len(self._packages)
                    + len(self._compositions)
                ),
                "index_size": len(self._syntheses),
                "duplicate_checks": 0,
                "mutation_count": (
                    self._synthesis_count
                    + self._package_count
                    + self._composition_count
                ),
                "last_mutation_at": None,  # not tracked at this granularity
                "synthesis_count": len(self._syntheses),
                "package_count": len(self._packages),
                "composition_count": len(self._compositions),
                "total_syntheses_created": self._synthesis_count,
                "total_packages_created": self._package_count,
                "total_compositions_created": self._composition_count,
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
        """Return synthesis statistics."""
        self._require_initialized("audit_report")
        with self._lock:
            cross_domain_count = sum(
                1 for s in self._syntheses.values() if s.is_cross_domain
            )
            rich_count = sum(
                1 for s in self._syntheses.values() if s.is_rich
            )
            complete_package_count = sum(
                1 for p in self._packages.values() if p.is_complete
            )
            avg_synthesis_confidence = (
                sum(s.confidence_score for s in self._syntheses.values())
                / len(self._syntheses)
                if self._syntheses
                else 0.0
            )
            return {
                "engine": "KnowledgeSynthesisEngine",
                "version": _ENGINE_VERSION,
                "synthesis_count": len(self._syntheses),
                "package_count": len(self._packages),
                "composition_count": len(self._compositions),
                "cross_domain_synthesis_count": cross_domain_count,
                "rich_synthesis_count": rich_count,
                "complete_package_count": complete_package_count,
                "average_synthesis_confidence": round(avg_synthesis_confidence, 4),
                "generated_at": _utcnow().isoformat(),
            }


__all__ = ["KnowledgeSynthesisEngine"]