"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/semantic_structure.py

Concrete in-memory implementation of the LUNA Semantic Structure Engine.

Manages the semantic organization of all knowledge owned by LUNA, providing
hierarchical trees, concept relationship graphs, dependency maps, and domain
structural views that form the navigational backbone of POLARIS reasoning.

Supported knowledge types:
    Fact · Concept · Skill · KnowledgeDomain · Procedure ·
    ResearchKnowledge · EducationalKnowledge

Responsibilities:
    Node management:
        create_semantic_node     — register a knowledge record as a graph node
        update_semantic_node     — mutate node label, weight, annotations
        remove_semantic_node     — detach node and re-parent its children

    Relationship management:
        create_relationship      — add a directed ConceptRelationship edge
        update_relationship      — change weight, description, bidirectionality
        remove_relationship      — delete an edge by ID

    Graph management:
        build_semantic_graph     — construct adjacency graph for a domain
        rebuild_semantic_graph   — full re-derive from live engines
        validate_semantic_graph  — structural consistency check

    Hierarchy management:
        create_hierarchy         — create a SemanticHierarchy rooted at a domain
        update_hierarchy         — partial-update name, description, notes
        remove_hierarchy         — delete hierarchy and orphan its nodes

    Discovery:
        discover_related_knowledge — cross-type related record discovery
        discover_related_concepts  — concept-only discovery via relationship graph
        discover_related_skills    — skill-only discovery via dependency edges

    Analysis:
        dependency_analysis      — dependency depth map for a record
        hierarchy_analysis       — structural stats for a hierarchy
        graph_analysis           — graph-wide metrics (degree, centrality, etc.)

    Audit / Reporting:
        semantic_audit           — structural integrity check across all data
        semantic_reporting       — human-readable snapshot dict

Support models:
    SemanticNode · SemanticHierarchy · ConceptRelationship ·
    KnowledgeDependency · DomainStructure

Thread safety:    threading.RLock on every public operation.
Lifecycle-gated:  every public method raises LunaNotInitializedError before
                  initialize() or after shutdown().
In-memory v1 implementation.  No persistence layer.

Integrates with (via injected engine handles):
    concepts.py        ConceptEngine
    domains.py         KnowledgeDomainEngine
    knowledge_index.py KnowledgeIndexEngine
    integrity.py       KnowledgeIntegrityEngine
    retrieval.py       KnowledgeRetrievalEngine

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
from datetime import datetime
from typing import Any, Optional

from subsystems.luna.exceptions import (
    LunaLifecycleError,
    LunaNotInitializedError,
    RelationshipCycleError,
    SemanticHierarchyError,
    SemanticNodeError,
    SemanticStructureError,
)
from subsystems.luna.interfaces import AbstractSemanticStructureEngine
from subsystems.luna.models import (
    Concept,
    ConceptRelationship,
    ConceptRelationshipType,
    DomainStructure,
    EducationalKnowledge,
    Fact,
    KnowledgeDependency,
    KnowledgeDomain,
    KnowledgeMetadata,
    KnowledgeRecord,
    KnowledgeStatus,
    KnowledgeType,
    Procedure,
    ResearchKnowledge,
    SemanticHierarchy,
    SemanticNode,
    Skill,
    _new_id,
    _utcnow,
)

logger = logging.getLogger(__name__)

_ENGINE_VERSION: str = "5.0.0"

# Knowledge types managed by the semantic structure engine
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
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_domain_ids(record: KnowledgeRecord) -> list[str]:
    """Return domain_ids for any KnowledgeRecord subtype."""
    return list(getattr(record, "domain_ids", []))


def _get_prerequisite_ids(record: KnowledgeRecord) -> list[str]:
    """Return prerequisite IDs relevant to dependency edges."""
    if isinstance(record, Concept):
        return list(record.prerequisite_concept_ids)
    if isinstance(record, Skill):
        return list(record.prerequisite_skill_ids) + list(record.required_concept_ids)
    if isinstance(record, EducationalKnowledge):
        return list(record.prerequisite_knowledge_ids)
    if isinstance(record, Procedure):
        return list(getattr(record, "required_skill_ids", [])) + list(
            getattr(record, "required_concept_ids", [])
        )
    return []


def _searchable_tokens(record: KnowledgeRecord) -> set[str]:
    """Return a set of lowercase tokens for semantic matching."""
    parts: list[str] = [record.name, record.description]
    parts += list(record.aliases)
    if isinstance(record, Fact):
        parts.append(record.statement)
    elif isinstance(record, Concept):
        parts += list(record.core_ideas)
    elif isinstance(record, Skill):
        parts.append(record.capability_description)
    return {t for p in parts if p for t in p.lower().split()}


def _node_label(record: KnowledgeRecord) -> str:
    """Return a display label for a node derived from a knowledge record."""
    return record.name


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL OPERATION LOG
# ─────────────────────────────────────────────────────────────────────────────

class _SemanticOpEntry:
    """Immutable log entry for a single mutation inside the engine."""

    __slots__ = ("op", "target_id", "timestamp", "notes")

    def __init__(self, op: str, target_id: str, notes: str = "") -> None:
        self.op: str = op
        self.target_id: str = target_id
        self.timestamp: datetime = _utcnow()
        self.notes: str = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "target_id": self.target_id,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SemanticStructureEngine(AbstractSemanticStructureEngine):
    """
    In-memory, thread-safe implementation of the LUNA Semantic Structure
    Engine (v1).

    The engine maintains four primary in-memory stores:

        _nodes          dict[node_id, SemanticNode]
                        All nodes ever registered, keyed by their own ID.

        _hierarchies    dict[hierarchy_id, SemanticHierarchy]
                        All SemanticHierarchy instances.

        _relationships  dict[relationship_id, ConceptRelationship]
                        All ConceptRelationship edges.

        _dependencies   dict[dependency_id, KnowledgeDependency]
                        All KnowledgeDependency records.

    Secondary indexes (all derived; rebuilt on mutation):

        _hierarchy_nodes    dict[hierarchy_id, set[node_id]]
        _record_nodes       dict[knowledge_id, set[node_id]]
        _domain_hierarchies dict[domain_id, set[hierarchy_id]]
        _source_rels        dict[source_concept_id, set[relationship_id]]
        _target_rels        dict[target_concept_id, set[relationship_id]]
        _dependent_deps     dict[dependent_id, set[dependency_id]]
        _dependency_deps    dict[dependency_id_target, set[dependency_id]]

    Injected engines (all optional):
        _concept_engine, _domain_engine, _index_engine,
        _integrity_engine, _retrieval_engine

    Lifecycle::

        engine = SemanticStructureEngine(
            concept_engine=my_concept_engine,
            domain_engine=my_domain_engine,
            index_engine=my_index_engine,
        )
        engine.initialize()
        hierarchy = engine.create_hierarchy(
            name="Control Systems",
            description="...",
            root_domain_id="domain-control",
            metadata=meta,
        )
        engine.shutdown()
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        *,
        concept_engine: Optional[Any] = None,
        domain_engine: Optional[Any] = None,
        index_engine: Optional[Any] = None,
        integrity_engine: Optional[Any] = None,
        retrieval_engine: Optional[Any] = None,
    ) -> None:
        self._concept_engine = concept_engine
        self._domain_engine = domain_engine
        self._index_engine = index_engine
        self._integrity_engine = integrity_engine
        self._retrieval_engine = retrieval_engine

        # Lifecycle
        self._initialized: bool = False
        self._lock: threading.RLock = threading.RLock()
        self._started_at: Optional[datetime] = None

        # Primary stores
        self._nodes: dict[str, SemanticNode] = {}
        self._hierarchies: dict[str, SemanticHierarchy] = {}
        self._relationships: dict[str, ConceptRelationship] = {}
        self._dependencies: dict[str, KnowledgeDependency] = {}

        # Secondary indexes
        self._hierarchy_nodes: dict[str, set[str]] = defaultdict(set)
        self._record_nodes: dict[str, set[str]] = defaultdict(set)
        self._domain_hierarchies: dict[str, set[str]] = defaultdict(set)
        self._source_rels: dict[str, set[str]] = defaultdict(set)
        self._target_rels: dict[str, set[str]] = defaultdict(set)
        self._dependent_deps: dict[str, set[str]] = defaultdict(set)
        self._dependency_deps: dict[str, set[str]] = defaultdict(set)

        # Operation log (capped)
        self._op_log: deque[_SemanticOpEntry] = deque(maxlen=500)

        # Observability counters
        self._mutation_count: int = 0
        self._last_mutation_at: Optional[datetime] = None
        self._graph_builds: int = 0
        self._graph_validations: int = 0
        self._audit_count: int = 0

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
                self._nodes.clear()
                self._hierarchies.clear()
                self._relationships.clear()
                self._dependencies.clear()
                self._hierarchy_nodes.clear()
                self._record_nodes.clear()
                self._domain_hierarchies.clear()
                self._source_rels.clear()
                self._target_rels.clear()
                self._dependent_deps.clear()
                self._dependency_deps.clear()
                self._op_log.clear()
                self._mutation_count = 0
                self._last_mutation_at = None
                self._graph_builds = 0
                self._graph_validations = 0
                self._audit_count = 0
                self._started_at = _utcnow()
                self._initialized = True
                logger.info("SemanticStructureEngine initialized at %s", self._started_at.isoformat())
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="startup",
                    engine="SemanticStructureEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        """
        Release all resources and mark the engine offline.

        Idempotent — calling on an already-stopped engine is a no-op.

        Raises:
            LunaLifecycleError: Internal shutdown failure.
        """
        with self._lock:
            if not self._initialized:
                return
            try:
                self._initialized = False
                logger.info("SemanticStructureEngine shut down.")
            except Exception as exc:
                raise LunaLifecycleError(
                    phase="shutdown",
                    engine="SemanticStructureEngine",
                    reason=str(exc),
                    cause=exc,
                ) from exc

    def is_initialized(self) -> bool:
        """Return True if the engine is fully started and ready."""
        return self._initialized

    # ── Internal guard ────────────────────────────────────────────────────────

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise LunaNotInitializedError(operation=operation)

    def _record_mutation(self, op: str, target_id: str, notes: str = "") -> None:
        self._mutation_count += 1
        self._last_mutation_at = _utcnow()
        self._op_log.append(_SemanticOpEntry(op=op, target_id=target_id, notes=notes))

    # ── Internal helpers — record resolution ──────────────────────────────────

    def _resolve_record(self, record_id: str, knowledge_type: KnowledgeType) -> Optional[KnowledgeRecord]:
        """
        Attempt to resolve a knowledge record from the appropriate injected engine.
        Returns None if the engine is unavailable or the record is not found.
        """
        try:
            if knowledge_type == KnowledgeType.CONCEPT and self._concept_engine:
                return self._concept_engine.retrieve_concept(record_id)
            if knowledge_type == KnowledgeType.DOMAIN and self._domain_engine:
                return self._domain_engine.retrieve_domain(record_id)
            if self._retrieval_engine and self._retrieval_engine.is_initialized():
                return self._retrieval_engine.retrieve_by_id(record_id, knowledge_type)
        except Exception:
            pass
        return None

    def _resolve_domain(self, domain_id: str) -> Optional[KnowledgeDomain]:
        """Resolve a KnowledgeDomain from the domain engine."""
        if not self._domain_engine:
            return None
        try:
            return self._domain_engine.retrieve_domain(domain_id)
        except Exception:
            return None

    def _all_records_for_domain(self, domain_id: str) -> list[KnowledgeRecord]:
        """
        Collect all active knowledge records whose domain_ids include domain_id.
        Uses the retrieval engine when available; falls back to per-engine scanning.
        """
        results: list[KnowledgeRecord] = []
        if self._retrieval_engine and self._retrieval_engine.is_initialized():
            try:
                results = self._retrieval_engine.search_by_domain(domain_id, limit=10_000)
                return [r for r in results if r.status.is_usable]
            except Exception:
                pass

        # Per-engine fallback
        engines = [
            (self._concept_engine, KnowledgeType.CONCEPT),
            (self._domain_engine, KnowledgeType.DOMAIN),
        ]
        for engine, _ in engines:
            if engine and engine.is_initialized():
                try:
                    raw = engine.list_all()  # best-effort; engines may not expose this
                    for r in raw:
                        if domain_id in _get_domain_ids(r) and r.status.is_usable:
                            results.append(r)
                except Exception:
                    pass
        return results

    def _all_concepts(self) -> list[Concept]:
        """Return all active concepts from the concept engine."""
        if not self._concept_engine or not self._concept_engine.is_initialized():
            return []
        try:
            return self._concept_engine.list_concepts(active_only=True)
        except Exception:
            return []

    def _all_domains(self) -> list[KnowledgeDomain]:
        """Return all domains from the domain engine."""
        if not self._domain_engine or not self._domain_engine.is_initialized():
            return []
        try:
            return self._domain_engine.list_domains()
        except Exception:
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # NODE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def create_semantic_node(
        self,
        knowledge_id: str,
        knowledge_type: KnowledgeType,
        label: str,
        *,
        parent_node_id: Optional[str] = None,
        weight: float = 1.0,
        annotations: Optional[dict[str, str]] = None,
    ) -> SemanticNode:
        """
        Register a knowledge record as a free-standing semantic node.

        The node exists independently of any hierarchy and can later be
        attached to one or more hierarchies via add_node.

        Args:
            knowledge_id:    FK to a KnowledgeRecord.
            knowledge_type:  The type of that record.
            label:           Display label; defaults to the record name if empty.
            parent_node_id:  Optional parent within a flat graph context.
            weight:          Semantic prominence weight [0.0 – 1.0].
            annotations:     Arbitrary string key-value metadata.

        Returns:
            The created SemanticNode.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Duplicate node for the same knowledge_id
                                     already exists.
        """
        self._require_initialized("create_semantic_node")
        with self._lock:
            # Prevent duplicate nodes for the same knowledge_id
            if knowledge_id in self._record_nodes and self._record_nodes[knowledge_id]:
                raise SemanticStructureError(
                    f"A semantic node for knowledge record '{knowledge_id}' already exists.",
                    context={"knowledge_id": knowledge_id, "knowledge_type": knowledge_type.value},
                )
            node = SemanticNode(
                id=_new_id(),
                knowledge_id=knowledge_id,
                knowledge_type=knowledge_type,
                label=label or knowledge_id,
                parent_node_id=parent_node_id,
                child_node_ids=[],
                depth=0,
                weight=max(0.0, min(1.0, weight)),
                annotations=dict(annotations or {}),
            )
            if parent_node_id and parent_node_id in self._nodes:
                parent = self._nodes[parent_node_id]
                node = SemanticNode(
                    id=node.id,
                    knowledge_id=node.knowledge_id,
                    knowledge_type=node.knowledge_type,
                    label=node.label,
                    parent_node_id=parent_node_id,
                    child_node_ids=node.child_node_ids,
                    depth=parent.depth + 1,
                    weight=node.weight,
                    annotations=node.annotations,
                )
                parent.child_node_ids.append(node.id)

            self._nodes[node.id] = node
            self._record_nodes[knowledge_id].add(node.id)
            self._record_mutation("create_semantic_node", node.id)
            logger.debug("SemanticNode created: id=%s knowledge_id=%s", node.id, knowledge_id)
            return node

    def update_semantic_node(
        self,
        node_id: str,
        *,
        label: Optional[str] = None,
        weight: Optional[float] = None,
        annotations: Optional[dict[str, str]] = None,
    ) -> SemanticNode:
        """
        Apply a partial update to an existing semantic node.

        Args:
            node_id:     The node to update.
            label:       New display label.
            weight:      New semantic weight [0.0 – 1.0].
            annotations: Full replacement for the annotations dict.

        Returns:
            The updated SemanticNode.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticNodeError:       Node not found.
        """
        self._require_initialized("update_semantic_node")
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise SemanticNodeError(node_id=node_id, reason="Node not found.")
            if label is not None:
                node.label = label
            if weight is not None:
                node.weight = max(0.0, min(1.0, weight))
            if annotations is not None:
                node.annotations = dict(annotations)
            self._record_mutation("update_semantic_node", node_id)
            logger.debug("SemanticNode updated: id=%s", node_id)
            return node

    def remove_semantic_node(self, node_id: str) -> SemanticNode:
        """
        Remove a semantic node.  Child nodes are re-parented to the removed
        node's parent (or become root nodes if the parent was None).

        Args:
            node_id: The node to remove.

        Returns:
            The removed SemanticNode.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticNodeError:       Node not found.
        """
        self._require_initialized("remove_semantic_node")
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise SemanticNodeError(node_id=node_id, reason="Node not found.")

            # Re-parent children
            new_parent_id = node.parent_node_id
            for child_id in list(node.child_node_ids):
                child = self._nodes.get(child_id)
                if child is None:
                    continue
                child.parent_node_id = new_parent_id
                if new_parent_id is not None:
                    new_parent = self._nodes.get(new_parent_id)
                    if new_parent is not None and child_id not in new_parent.child_node_ids:
                        new_parent.child_node_ids.append(child_id)

            # Detach from parent
            if node.parent_node_id:
                parent = self._nodes.get(node.parent_node_id)
                if parent and node_id in parent.child_node_ids:
                    parent.child_node_ids.remove(node_id)

            # Remove from stores
            del self._nodes[node_id]
            self._record_nodes[node.knowledge_id].discard(node_id)

            # Remove from any hierarchy that references this node
            for h_id, node_set in self._hierarchy_nodes.items():
                node_set.discard(node_id)
                hierarchy = self._hierarchies.get(h_id)
                if hierarchy and node_id in hierarchy.nodes:
                    del hierarchy.nodes[node_id]

            self._record_mutation("remove_semantic_node", node_id)
            logger.debug("SemanticNode removed: id=%s", node_id)
            return node

    def get_semantic_node(self, node_id: str) -> SemanticNode:
        """
        Fetch a semantic node by ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticNodeError:       Node not found.
        """
        self._require_initialized("get_semantic_node")
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise SemanticNodeError(node_id=node_id, reason="Node not found.")
            return node

    def list_semantic_nodes(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
        knowledge_id: Optional[str] = None,
    ) -> list[SemanticNode]:
        """
        List all semantic nodes, optionally filtered by knowledge type or
        the underlying knowledge record ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("list_semantic_nodes")
        with self._lock:
            nodes = list(self._nodes.values())
            if knowledge_type is not None:
                nodes = [n for n in nodes if n.knowledge_type == knowledge_type]
            if knowledge_id is not None:
                nodes = [n for n in nodes if n.knowledge_id == knowledge_id]
            return nodes

    # ─────────────────────────────────────────────────────────────────────────
    # RELATIONSHIP MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def create_relationship(
        self,
        source_concept_id: str,
        target_concept_id: str,
        relationship_type: ConceptRelationshipType,
        description: str,
        *,
        weight: float = 1.0,
        is_bidirectional: bool = False,
    ) -> ConceptRelationship:
        """
        Create a directed ConceptRelationship edge between two concept records.

        Args:
            source_concept_id:   Origin concept ID.
            target_concept_id:   Destination concept ID.
            relationship_type:   Semantic relationship type.
            description:         Human-readable description of the link.
            weight:              Semantic strength [0.0 – 1.0].
            is_bidirectional:    If True, the link is traversable in both directions.

        Returns:
            The created ConceptRelationship.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Source or target concept not found.
        """
        self._require_initialized("create_relationship")
        with self._lock:
            relationship = ConceptRelationship.create(
                source_concept_id=source_concept_id,
                target_concept_id=target_concept_id,
                relationship_type=relationship_type,
                description=description,
                weight=weight,
                is_bidirectional=is_bidirectional,
            )
            self._relationships[relationship.id] = relationship
            self._source_rels[source_concept_id].add(relationship.id)
            self._target_rels[target_concept_id].add(relationship.id)
            if is_bidirectional:
                self._source_rels[target_concept_id].add(relationship.id)
                self._target_rels[source_concept_id].add(relationship.id)

            self._record_mutation("create_relationship", relationship.id)
            logger.debug(
                "ConceptRelationship created: id=%s %s -[%s]-> %s",
                relationship.id,
                source_concept_id,
                relationship_type.value,
                target_concept_id,
            )
            return relationship

    def update_relationship(
        self,
        relationship_id: str,
        *,
        description: Optional[str] = None,
        weight: Optional[float] = None,
        is_bidirectional: Optional[bool] = None,
    ) -> ConceptRelationship:
        """
        Replace the mutable fields of an existing ConceptRelationship.

        Because ConceptRelationship is frozen, this method removes the old
        entry and inserts a replacement with the same ID and updated fields.

        Args:
            relationship_id:  The relationship to update.
            description:      New description.
            weight:           New semantic weight.
            is_bidirectional: New bidirectionality flag.

        Returns:
            The new ConceptRelationship replacing the old one.

        Raises:
            LunaNotInitializedError:  Engine not initialized.
            SemanticStructureError:   Relationship not found.
        """
        self._require_initialized("update_relationship")
        with self._lock:
            old = self._relationships.get(relationship_id)
            if old is None:
                raise SemanticStructureError(
                    f"ConceptRelationship '{relationship_id}' not found.",
                    context={"relationship_id": relationship_id},
                )
            new_weight = max(0.0, min(1.0, weight)) if weight is not None else old.weight
            new_desc = description if description is not None else old.description
            new_bidir = is_bidirectional if is_bidirectional is not None else old.is_bidirectional

            # Remove old bidirectionality index entries
            self._source_rels[old.source_concept_id].discard(relationship_id)
            self._target_rels[old.target_concept_id].discard(relationship_id)
            if old.is_bidirectional:
                self._source_rels[old.target_concept_id].discard(relationship_id)
                self._target_rels[old.source_concept_id].discard(relationship_id)

            updated = ConceptRelationship(
                id=old.id,
                source_concept_id=old.source_concept_id,
                target_concept_id=old.target_concept_id,
                relationship_type=old.relationship_type,
                description=new_desc,
                weight=new_weight,
                is_bidirectional=new_bidir,
                created_at=old.created_at,
            )
            self._relationships[relationship_id] = updated
            self._source_rels[updated.source_concept_id].add(relationship_id)
            self._target_rels[updated.target_concept_id].add(relationship_id)
            if new_bidir:
                self._source_rels[updated.target_concept_id].add(relationship_id)
                self._target_rels[updated.source_concept_id].add(relationship_id)

            self._record_mutation("update_relationship", relationship_id)
            return updated

    def remove_relationship(self, relationship_id: str) -> ConceptRelationship:
        """
        Delete a ConceptRelationship edge.

        Args:
            relationship_id: The relationship to remove.

        Returns:
            The removed ConceptRelationship.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Relationship not found.
        """
        self._require_initialized("remove_relationship")
        with self._lock:
            rel = self._relationships.get(relationship_id)
            if rel is None:
                raise SemanticStructureError(
                    f"ConceptRelationship '{relationship_id}' not found.",
                    context={"relationship_id": relationship_id},
                )
            del self._relationships[relationship_id]
            self._source_rels[rel.source_concept_id].discard(relationship_id)
            self._target_rels[rel.target_concept_id].discard(relationship_id)
            if rel.is_bidirectional:
                self._source_rels[rel.target_concept_id].discard(relationship_id)
                self._target_rels[rel.source_concept_id].discard(relationship_id)

            self._record_mutation("remove_relationship", relationship_id)
            logger.debug("ConceptRelationship removed: id=%s", relationship_id)
            return rel

    def get_relationship(self, relationship_id: str) -> ConceptRelationship:
        """
        Fetch a ConceptRelationship by ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Relationship not found.
        """
        self._require_initialized("get_relationship")
        with self._lock:
            rel = self._relationships.get(relationship_id)
            if rel is None:
                raise SemanticStructureError(
                    f"ConceptRelationship '{relationship_id}' not found.",
                    context={"relationship_id": relationship_id},
                )
            return rel

    def list_relationships(
        self,
        *,
        source_concept_id: Optional[str] = None,
        target_concept_id: Optional[str] = None,
        relationship_type: Optional[ConceptRelationshipType] = None,
    ) -> list[ConceptRelationship]:
        """
        Return all ConceptRelationship records, with optional filters.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("list_relationships")
        with self._lock:
            rels = list(self._relationships.values())
            if source_concept_id is not None:
                rel_ids = self._source_rels.get(source_concept_id, set())
                rels = [r for r in rels if r.id in rel_ids]
            if target_concept_id is not None:
                rel_ids = self._target_rels.get(target_concept_id, set())
                rels = [r for r in rels if r.id in rel_ids]
            if relationship_type is not None:
                rels = [r for r in rels if r.relationship_type == relationship_type]
            return rels

    # ─────────────────────────────────────────────────────────────────────────
    # DEPENDENCY MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def create_dependency(
        self,
        dependent_id: str,
        dependency_id: str,
        dependency_type: str = "requires",
        *,
        is_hard: bool = True,
        description: str = "",
    ) -> KnowledgeDependency:
        """
        Register a KnowledgeDependency between two knowledge records.

        Args:
            dependent_id:    The record that depends on the other.
            dependency_id:   The record that is required.
            dependency_type: Semantic type ("requires", "recommends", "enhances").
            is_hard:         If True, dependency is a blocking prerequisite.
            description:     Optional explanation.

        Returns:
            The created KnowledgeDependency.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("create_dependency")
        with self._lock:
            dep = KnowledgeDependency.create(
                dependent_id=dependent_id,
                dependency_id=dependency_id,
                dependency_type=dependency_type,
                is_hard=is_hard,
                description=description,
            )
            self._dependencies[dep.id] = dep
            self._dependent_deps[dependent_id].add(dep.id)
            self._dependency_deps[dependency_id].add(dep.id)
            self._record_mutation("create_dependency", dep.id)
            logger.debug(
                "KnowledgeDependency created: id=%s %s -[%s]-> %s",
                dep.id, dependent_id, dependency_type, dependency_id,
            )
            return dep

    def remove_dependency(self, dependency_id: str) -> KnowledgeDependency:
        """
        Remove a KnowledgeDependency by its own ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Dependency record not found.
        """
        self._require_initialized("remove_dependency")
        with self._lock:
            dep = self._dependencies.get(dependency_id)
            if dep is None:
                raise SemanticStructureError(
                    f"KnowledgeDependency '{dependency_id}' not found.",
                    context={"dependency_id": dependency_id},
                )
            del self._dependencies[dependency_id]
            self._dependent_deps[dep.dependent_id].discard(dependency_id)
            self._dependency_deps[dep.dependency_id].discard(dependency_id)
            self._record_mutation("remove_dependency", dependency_id)
            return dep

    def list_dependencies(
        self,
        *,
        dependent_id: Optional[str] = None,
        dependency_id: Optional[str] = None,
        is_hard: Optional[bool] = None,
    ) -> list[KnowledgeDependency]:
        """
        Return all KnowledgeDependency records with optional filters.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("list_dependencies")
        with self._lock:
            deps = list(self._dependencies.values())
            if dependent_id is not None:
                dep_ids = self._dependent_deps.get(dependent_id, set())
                deps = [d for d in deps if d.id in dep_ids]
            if dependency_id is not None:
                dep_ids = self._dependency_deps.get(dependency_id, set())
                deps = [d for d in deps if d.id in dep_ids]
            if is_hard is not None:
                deps = [d for d in deps if d.is_hard == is_hard]
            return deps

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def build_semantic_graph(
        self,
        domain_id: Optional[str] = None,
        *,
        include_cross_domain: bool = True,
    ) -> dict[str, Any]:
        """
        Construct an adjacency-list graph of concept relationships and
        dependency edges.

        When ``domain_id`` is provided the graph is scoped to that domain.
        When omitted (the default), the full graph across all nodes is returned.

        Returns a plain dict — not persisted.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Domain not found (only when domain_id given).
        """
        self._require_initialized("build_semantic_graph")
        with self._lock:
            self._graph_builds += 1

            # ── Global path: no domain filter ────────────────────────────────
            if domain_id is None:
                nodes_out: list[dict[str, Any]] = []
                edges_out: list[dict[str, Any]] = []

                for node in self._nodes.values():
                    nodes_out.append({
                        "id": node.id,
                        "label": node.label,
                        "knowledge_type": node.knowledge_type.value,
                    })

                for rel in self._relationships.values():
                    edges_out.append({
                        "source": rel.source_concept_id,
                        "target": rel.target_concept_id,
                        "edge_type": "relationship",
                        "label": rel.relationship_type.value,
                        "weight": rel.weight,
                        "is_bidirectional": rel.is_bidirectional,
                    })

                for dep in self._dependencies.values():
                    edges_out.append({
                        "source": dep.dependent_id,
                        "target": dep.dependency_id,
                        "edge_type": "dependency",
                        "label": dep.dependency_type,
                        "weight": 1.0 if dep.is_hard else 0.5,
                        "is_bidirectional": False,
                    })

                return {
                    "domain_id": None,
                    "nodes": nodes_out,
                    "edges": edges_out,
                    "node_count": len(nodes_out),
                    "edge_count": len(edges_out),
                    "built_at": _utcnow().isoformat(),
                }

            # ── Domain-scoped path ────────────────────────────────────────────
            records = self._all_records_for_domain(domain_id)
            record_ids: set[str] = {r.id for r in records}

            nodes_out = []
            edges_out = []
            seen_node_ids: set[str] = set()

            for record in records:
                if record.id not in seen_node_ids:
                    nodes_out.append({
                        "id": record.id,
                        "label": record.name,
                        "knowledge_type": record.knowledge_type.value,
                    })
                    seen_node_ids.add(record.id)

            # Concept relationship edges
            for rel in self._relationships.values():
                in_domain = (
                    rel.source_concept_id in record_ids
                    and (include_cross_domain or rel.target_concept_id in record_ids)
                )
                if in_domain:
                    edges_out.append({
                        "source": rel.source_concept_id,
                        "target": rel.target_concept_id,
                        "edge_type": "relationship",
                        "label": rel.relationship_type.value,
                        "weight": rel.weight,
                        "is_bidirectional": rel.is_bidirectional,
                    })

            # Dependency edges
            for dep in self._dependencies.values():
                in_domain = (
                    dep.dependent_id in record_ids
                    and (include_cross_domain or dep.dependency_id in record_ids)
                )
                if in_domain:
                    edges_out.append({
                        "source": dep.dependent_id,
                        "target": dep.dependency_id,
                        "edge_type": "dependency",
                        "label": dep.dependency_type,
                        "weight": 1.0 if dep.is_hard else 0.5,
                        "is_bidirectional": False,
                    })

            return {
                "domain_id": domain_id,
                "nodes": nodes_out,
                "edges": edges_out,
                "node_count": len(nodes_out),
                "edge_count": len(edges_out),
                "built_at": _utcnow().isoformat(),
            }

    def rebuild_semantic_graph(self) -> dict[str, Any]:
        """
        Fully re-derive the semantic graph for all domains from the live engines.

        Clears and re-populates internal relationship and dependency indexes
        by scanning the concept engine for declared relationships and each record
        type for declared prerequisite chains.

        Returns:
            A summary dict with totals for nodes, relationships, and dependencies
            that were rebuilt.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("rebuild_semantic_graph")
        with self._lock:
            self._graph_builds += 1
            relationships_added = 0
            dependencies_added = 0
            nodes_added = 0

            # Rebuild relationships from concepts
            concepts = self._all_concepts()
            for concept in concepts:
                # Ensure each concept has a node entry
                if concept.id not in self._record_nodes or not self._record_nodes[concept.id]:
                    node = SemanticNode.create(
                        knowledge_id=concept.id,
                        knowledge_type=KnowledgeType.CONCEPT,
                        label=concept.name,
                    )
                    self._nodes[node.id] = node
                    self._record_nodes[concept.id].add(node.id)
                    nodes_added += 1

                # Derive IS_A relationships from child_concept_ids
                for child_id in concept.child_concept_ids:
                    key = f"IS_A:{child_id}:{concept.id}"
                    already_exists = any(
                        r.source_concept_id == child_id
                        and r.target_concept_id == concept.id
                        and r.relationship_type == ConceptRelationshipType.IS_A
                        for r in self._relationships.values()
                    )
                    if not already_exists:
                        rel = ConceptRelationship.create(
                            source_concept_id=child_id,
                            target_concept_id=concept.id,
                            relationship_type=ConceptRelationshipType.IS_A,
                            description=f"'{child_id}' is a sub-concept of '{concept.id}'.",
                        )
                        self._relationships[rel.id] = rel
                        self._source_rels[child_id].add(rel.id)
                        self._target_rels[concept.id].add(rel.id)
                        relationships_added += 1

                # Derive PRECEDES relationships from prerequisite_concept_ids
                for prereq_id in concept.prerequisite_concept_ids:
                    already_exists = any(
                        r.source_concept_id == prereq_id
                        and r.target_concept_id == concept.id
                        and r.relationship_type == ConceptRelationshipType.PRECEDES
                        for r in self._relationships.values()
                    )
                    if not already_exists:
                        rel = ConceptRelationship.create(
                            source_concept_id=prereq_id,
                            target_concept_id=concept.id,
                            relationship_type=ConceptRelationshipType.PRECEDES,
                            description=f"'{prereq_id}' precedes '{concept.id}'.",
                        )
                        self._relationships[rel.id] = rel
                        self._source_rels[prereq_id].add(rel.id)
                        self._target_rels[concept.id].add(rel.id)
                        relationships_added += 1

                    # Also record as a KnowledgeDependency
                    already_dep = any(
                        d.dependent_id == concept.id and d.dependency_id == prereq_id
                        for d in self._dependencies.values()
                    )
                    if not already_dep:
                        dep = KnowledgeDependency.create(
                            dependent_id=concept.id,
                            dependency_id=prereq_id,
                            dependency_type="requires",
                            is_hard=True,
                        )
                        self._dependencies[dep.id] = dep
                        self._dependent_deps[concept.id].add(dep.id)
                        self._dependency_deps[prereq_id].add(dep.id)
                        dependencies_added += 1

            self._record_mutation("rebuild_semantic_graph", "global")
            logger.info(
                "SemanticGraph rebuilt: nodes_added=%d relationships_added=%d dependencies_added=%d",
                nodes_added, relationships_added, dependencies_added,
            )
            return {
                "nodes_added": nodes_added,
                "relationships_added": relationships_added,
                "dependencies_added": dependencies_added,
                "total_nodes": len(self._nodes),
                "total_relationships": len(self._relationships),
                "total_dependencies": len(self._dependencies),
                "rebuilt_at": _utcnow().isoformat(),
            }

    def validate_semantic_graph(self) -> dict[str, Any]:
        """
        Perform structural consistency checks across the entire semantic graph.

        Checks:
            - All relationship source/target IDs resolve to known nodes or live records.
            - All dependency dependent_id/dependency_id references are non-empty.
            - No cycles exist in the hard-dependency graph.
            - All node knowledge_ids are non-empty strings.

        Returns:
            A validation summary dict::

                {
                    "valid": bool,
                    "issues": [{"severity": str, "message": str}],
                    "stats": {...},
                    "validated_at": str,
                }

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("validate_semantic_graph")
        with self._lock:
            self._graph_validations += 1
            issues: list[dict[str, Any]] = []

            # Node integrity
            for node_id, node in self._nodes.items():
                if not node.knowledge_id:
                    issues.append({
                        "severity": "ERROR",
                        "message": f"Node '{node_id}' has empty knowledge_id.",
                    })
                if not node.label:
                    issues.append({
                        "severity": "WARNING",
                        "message": f"Node '{node_id}' has empty label.",
                    })
                if node.parent_node_id and node.parent_node_id not in self._nodes:
                    issues.append({
                        "severity": "ERROR",
                        "message": (
                            f"Node '{node_id}' references non-existent "
                            f"parent '{node.parent_node_id}'."
                        ),
                    })
                for child_id in node.child_node_ids:
                    if child_id not in self._nodes:
                        issues.append({
                            "severity": "ERROR",
                            "message": (
                                f"Node '{node_id}' references non-existent "
                                f"child '{child_id}'."
                            ),
                        })

            # Relationship integrity
            for rel_id, rel in self._relationships.items():
                if not rel.source_concept_id:
                    issues.append({
                        "severity": "ERROR",
                        "message": f"Relationship '{rel_id}' has empty source_concept_id.",
                    })
                if not rel.target_concept_id:
                    issues.append({
                        "severity": "ERROR",
                        "message": f"Relationship '{rel_id}' has empty target_concept_id.",
                    })
                if rel.source_concept_id == rel.target_concept_id:
                    issues.append({
                        "severity": "WARNING",
                        "message": f"Relationship '{rel_id}' is a self-loop.",
                    })

            # Dependency integrity + cycle detection
            for dep_id, dep in self._dependencies.items():
                if not dep.dependent_id or not dep.dependency_id:
                    issues.append({
                        "severity": "ERROR",
                        "message": f"Dependency '{dep_id}' has empty dependent/dependency ID.",
                    })
                if dep.dependent_id == dep.dependency_id:
                    issues.append({
                        "severity": "ERROR",
                        "message": f"Dependency '{dep_id}' is a self-loop on '{dep.dependent_id}'.",
                    })

            # Hard-dependency cycle detection (DFS)
            hard_cycles = self._detect_dependency_cycles(hard_only=True)
            for cycle in hard_cycles:
                issues.append({
                    "severity": "ERROR",
                    "message": f"Hard-dependency cycle detected: {' → '.join(cycle)}",
                })

            return {
                "valid": all(i["severity"] != "ERROR" for i in issues),
                "issues": issues,
                "stats": {
                    "total_nodes": len(self._nodes),
                    "total_relationships": len(self._relationships),
                    "total_dependencies": len(self._dependencies),
                    "hard_cycles_found": len(hard_cycles),
                },
                "validated_at": _utcnow().isoformat(),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # HIERARCHY MANAGEMENT  (AbstractSemanticStructureEngine contract)
    # ─────────────────────────────────────────────────────────────────────────

    def create_hierarchy(
        self,
        name: str,
        description: str,
        root_domain_id: str,
        metadata: KnowledgeMetadata,
        notes: str = "",
    ) -> SemanticHierarchy:
        """
        Create a new SemanticHierarchy rooted at a KnowledgeDomain node.

        The domain is automatically registered as the root SemanticNode.

        Args:
            name:           Human-readable name for the hierarchy.
            description:    Description of what this hierarchy organizes.
            root_domain_id: The KnowledgeDomain ID that forms the root.
            metadata:       KnowledgeMetadata for provenance tracking.
            notes:          Optional free-text notes.

        Returns:
            The created SemanticHierarchy.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Root domain does not exist in the engine.
        """
        self._require_initialized("create_hierarchy")
        with self._lock:
            # Build root node for the domain
            root_node = SemanticNode.create(
                knowledge_id=root_domain_id,
                knowledge_type=KnowledgeType.DOMAIN,
                label=name,
                depth=0,
            )
            self._nodes[root_node.id] = root_node
            self._record_nodes[root_domain_id].add(root_node.id)

            hierarchy = SemanticHierarchy.create(
                name=name,
                description=description,
                root_node=root_node,
                domain_id=root_domain_id,
            )
            self._hierarchies[hierarchy.id] = hierarchy
            self._hierarchy_nodes[hierarchy.id].add(root_node.id)
            self._domain_hierarchies[root_domain_id].add(hierarchy.id)

            self._record_mutation("create_hierarchy", hierarchy.id, notes=notes)
            logger.info(
                "SemanticHierarchy created: id=%s name='%s' root_domain_id=%s",
                hierarchy.id, name, root_domain_id,
            )
            return hierarchy

    def update_hierarchy(
        self,
        hierarchy_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> SemanticHierarchy:
        """
        Apply a partial update to a SemanticHierarchy's mutable fields.

        Args:
            hierarchy_id: The hierarchy to update.
            name:         New hierarchy name.
            description:  New description.
            notes:        Notes string (stored as a mutation log entry).

        Returns:
            The updated SemanticHierarchy.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticHierarchyError:  Hierarchy not found.
        """
        self._require_initialized("update_hierarchy")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticHierarchyError(
                    hierarchy_id=hierarchy_id,
                    reason="Hierarchy not found.",
                )
            if name is not None:
                hierarchy.name = name
            if description is not None:
                hierarchy.description = description
            hierarchy.updated_at = _utcnow()
            self._record_mutation("update_hierarchy", hierarchy_id, notes=notes or "")
            return hierarchy

    def remove_hierarchy(self, hierarchy_id: str) -> SemanticHierarchy:
        """
        Remove a SemanticHierarchy and all SemanticNode entries it owns.

        Nodes that belong exclusively to this hierarchy are removed from the
        global node store.  Nodes shared with other hierarchies remain.

        Args:
            hierarchy_id: The hierarchy to remove.

        Returns:
            The removed SemanticHierarchy.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticHierarchyError:  Hierarchy not found.
        """
        self._require_initialized("remove_hierarchy")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticHierarchyError(
                    hierarchy_id=hierarchy_id,
                    reason="Hierarchy not found.",
                )
            # Remove nodes that belong only to this hierarchy
            node_ids = self._hierarchy_nodes.get(hierarchy_id, set())
            for node_id in list(node_ids):
                # Check if the node belongs to other hierarchies
                in_other = any(
                    node_id in nset
                    for hid, nset in self._hierarchy_nodes.items()
                    if hid != hierarchy_id
                )
                if not in_other:
                    node = self._nodes.pop(node_id, None)
                    if node:
                        self._record_nodes[node.knowledge_id].discard(node_id)

            del self._hierarchies[hierarchy_id]
            self._hierarchy_nodes.pop(hierarchy_id, None)
            if hierarchy.domain_id:
                self._domain_hierarchies[hierarchy.domain_id].discard(hierarchy_id)

            self._record_mutation("remove_hierarchy", hierarchy_id)
            logger.info("SemanticHierarchy removed: id=%s", hierarchy_id)
            return hierarchy

    def retrieve_hierarchy(self, hierarchy_id: str) -> SemanticHierarchy:
        """
        Fetch a SemanticHierarchy by ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticHierarchyError:  Hierarchy not found.
        """
        self._require_initialized("retrieve_hierarchy")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticHierarchyError(
                    hierarchy_id=hierarchy_id,
                    reason="Hierarchy not found.",
                )
            return hierarchy

    def delete_hierarchy(self, hierarchy_id: str) -> SemanticHierarchy:
        """Alias for remove_hierarchy — satisfies AbstractSemanticStructureEngine."""
        return self.remove_hierarchy(hierarchy_id)

    # ── Node operations within a hierarchy ───────────────────────────────────

    def add_node(
        self,
        hierarchy_id: str,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        parent_node_id: Optional[str] = None,
        label: str = "",
        position: int = 0,
    ) -> SemanticNode:
        """
        Add a knowledge record as a child node within an existing hierarchy.

        The parent defaults to the hierarchy's root node when parent_node_id
        is not specified.

        Args:
            hierarchy_id:   Target hierarchy.
            record_id:      The knowledge record to attach.
            knowledge_type: The type of that record.
            parent_node_id: Parent node in the tree; None → attach to root.
            label:          Display label override.
            position:       Hint for sort order among siblings (not enforced
                            in in-memory v1; reserved for persistence layers).

        Returns:
            The created SemanticNode.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Hierarchy or parent node not found.
            RelationshipCycleError:  Adding this node would create a cycle.
        """
        self._require_initialized("add_node")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticStructureError(
                    f"Hierarchy '{hierarchy_id}' not found.",
                    context={"hierarchy_id": hierarchy_id},
                )
            # Resolve parent
            effective_parent_id = parent_node_id or hierarchy.root_node_id
            parent = hierarchy.nodes.get(effective_parent_id)
            if parent is None:
                raise SemanticStructureError(
                    f"Parent node '{effective_parent_id}' not found in hierarchy '{hierarchy_id}'.",
                    context={"hierarchy_id": hierarchy_id, "parent_node_id": effective_parent_id},
                )

            # Cycle guard: the new record_id must not already be an ancestor
            if self._would_create_cycle_in_hierarchy(hierarchy, effective_parent_id, record_id):
                raise RelationshipCycleError(
                    node_ids=[record_id, effective_parent_id],
                    hierarchy_id=hierarchy_id,
                )

            node = SemanticNode.create(
                knowledge_id=record_id,
                knowledge_type=knowledge_type,
                label=label or record_id,
                parent_node_id=effective_parent_id,
                depth=parent.depth + 1,
            )
            # Link into parent
            parent.child_node_ids.append(node.id)

            # Register globally and in hierarchy
            self._nodes[node.id] = node
            hierarchy.nodes[node.id] = node
            hierarchy.updated_at = _utcnow()

            self._hierarchy_nodes[hierarchy_id].add(node.id)
            self._record_nodes[record_id].add(node.id)

            self._record_mutation("add_node", node.id)
            logger.debug(
                "Node added to hierarchy '%s': node_id=%s record_id=%s",
                hierarchy_id, node.id, record_id,
            )
            return node

    def remove_node(self, hierarchy_id: str, node_id: str) -> None:
        """
        Remove a node from a hierarchy.  Child nodes are re-parented to the
        removed node's parent within the hierarchy.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Hierarchy or node not found.
        """
        self._require_initialized("remove_node")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticStructureError(
                    f"Hierarchy '{hierarchy_id}' not found.",
                    context={"hierarchy_id": hierarchy_id},
                )
            node = hierarchy.nodes.get(node_id)
            if node is None:
                raise SemanticNodeError(
                    node_id=node_id,
                    hierarchy_id=hierarchy_id,
                    reason="Node not found in hierarchy.",
                )
            if node_id == hierarchy.root_node_id:
                raise SemanticHierarchyError(
                    hierarchy_id=hierarchy_id,
                    reason="Cannot remove the root node from a hierarchy.",
                )

            new_parent_id = node.parent_node_id
            # Re-parent children
            for child_id in list(node.child_node_ids):
                child = hierarchy.nodes.get(child_id)
                if child is None:
                    continue
                child.parent_node_id = new_parent_id
                if new_parent_id:
                    new_parent = hierarchy.nodes.get(new_parent_id)
                    if new_parent and child_id not in new_parent.child_node_ids:
                        new_parent.child_node_ids.append(child_id)

            # Detach from parent
            if node.parent_node_id:
                parent = hierarchy.nodes.get(node.parent_node_id)
                if parent and node_id in parent.child_node_ids:
                    parent.child_node_ids.remove(node_id)

            del hierarchy.nodes[node_id]
            hierarchy.updated_at = _utcnow()
            self._hierarchy_nodes[hierarchy_id].discard(node_id)

            # Only remove from global store if not in other hierarchies
            in_other = any(
                node_id in nset
                for hid, nset in self._hierarchy_nodes.items()
                if hid != hierarchy_id
            )
            if not in_other:
                self._nodes.pop(node_id, None)
                self._record_nodes[node.knowledge_id].discard(node_id)

            self._record_mutation("remove_node", node_id)

    def get_node(self, hierarchy_id: str, node_id: str) -> SemanticNode:
        """
        Fetch a node from within a hierarchy.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticNodeError:       Node not found.
        """
        self._require_initialized("get_node")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticStructureError(
                    f"Hierarchy '{hierarchy_id}' not found.",
                    context={"hierarchy_id": hierarchy_id},
                )
            node = hierarchy.nodes.get(node_id)
            if node is None:
                raise SemanticNodeError(
                    node_id=node_id,
                    hierarchy_id=hierarchy_id,
                    reason="Node not found in hierarchy.",
                )
            return node

    def get_children_nodes(self, hierarchy_id: str, node_id: str) -> list[SemanticNode]:
        """
        Return direct child nodes of the given node within a hierarchy.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticNodeError:       Node not found in hierarchy.
        """
        self._require_initialized("get_children_nodes")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticStructureError(
                    f"Hierarchy '{hierarchy_id}' not found.",
                    context={"hierarchy_id": hierarchy_id},
                )
            node = hierarchy.nodes.get(node_id)
            if node is None:
                raise SemanticNodeError(
                    node_id=node_id,
                    hierarchy_id=hierarchy_id,
                    reason="Node not found in hierarchy.",
                )
            return hierarchy.get_children(node_id)

    def get_root_nodes(self, hierarchy_id: str) -> list[SemanticNode]:
        """
        Return root-level nodes (no parent) of the given hierarchy.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Hierarchy not found.
        """
        self._require_initialized("get_root_nodes")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticStructureError(
                    f"Hierarchy '{hierarchy_id}' not found.",
                    context={"hierarchy_id": hierarchy_id},
                )
            return [n for n in hierarchy.nodes.values() if n.is_root]

    def get_full_tree(self, hierarchy_id: str) -> dict[str, Any]:
        """
        Return a nested dict representation of the entire hierarchy tree.

        Schema::

            {
                "hierarchy_id": str,
                "name": str,
                "nodes": [
                    {
                        "node_id": str,
                        "record_id": str,
                        "label": str,
                        "children": [ ... ]
                    }
                ]
            }

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Hierarchy not found.
        """
        self._require_initialized("get_full_tree")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticStructureError(
                    f"Hierarchy '{hierarchy_id}' not found.",
                    context={"hierarchy_id": hierarchy_id},
                )

            def _build_subtree(node: SemanticNode) -> dict[str, Any]:
                return {
                    "node_id": node.id,
                    "record_id": node.knowledge_id,
                    "label": node.label,
                    "knowledge_type": node.knowledge_type.value,
                    "depth": node.depth,
                    "weight": node.weight,
                    "children": [
                        _build_subtree(hierarchy.nodes[cid])
                        for cid in node.child_node_ids
                        if cid in hierarchy.nodes
                    ],
                }

            root_nodes = [n for n in hierarchy.nodes.values() if n.is_root]
            return {
                "hierarchy_id": hierarchy_id,
                "name": hierarchy.name,
                "description": hierarchy.description,
                "nodes": [_build_subtree(r) for r in root_nodes],
                "node_count": hierarchy.node_count,
                "max_depth": hierarchy.max_depth,
            }

    def build_domain_structure(self, domain_id: str) -> DomainStructure:
        """
        Compute and return the full DomainStructure for a knowledge domain.

        Aggregates all concepts, facts, skills, procedures, sub-domains, and
        hierarchy IDs registered under this domain.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Domain engine unavailable and domain
                                     cannot be resolved.
        """
        self._require_initialized("build_domain_structure")
        with self._lock:
            domain = self._resolve_domain(domain_id)
            domain_name = domain.name if domain else domain_id

            ds = DomainStructure.create(domain_id=domain_id, domain_name=domain_name)

            if domain:
                ds.concept_ids = list(getattr(domain, "core_concept_ids", []))
                ds.fact_ids = list(getattr(domain, "core_fact_ids", []))
                ds.skill_ids = list(getattr(domain, "core_skill_ids", []))
                ds.sub_domain_ids = list(getattr(domain, "sub_domain_ids", []))

            # Collect procedure IDs from live records
            records = self._all_records_for_domain(domain_id)
            for r in records:
                if r.knowledge_type == KnowledgeType.PROCEDURE and r.id not in ds.procedure_ids:
                    ds.procedure_ids.append(r.id)
                if r.knowledge_type == KnowledgeType.CONCEPT and r.id not in ds.concept_ids:
                    ds.concept_ids.append(r.id)
                if r.knowledge_type == KnowledgeType.SKILL and r.id not in ds.skill_ids:
                    ds.skill_ids.append(r.id)
                if r.knowledge_type == KnowledgeType.FACT and r.id not in ds.fact_ids:
                    ds.fact_ids.append(r.id)

            # Hierarchy IDs
            ds.hierarchy_ids = list(self._domain_hierarchies.get(domain_id, set()))

            # Dependency IDs owned by this domain's records
            domain_record_ids = {
                r.id for r in records
            }
            dep_ids = [
                dep.id
                for dep in self._dependencies.values()
                if dep.dependent_id in domain_record_ids
            ]
            ds.dependency_ids = dep_ids

            ds.recalculate_total()
            return ds

    def list_hierarchies(
        self,
        *,
        domain_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[SemanticHierarchy]:
        """
        Return all hierarchies, optionally filtered by root domain.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("list_hierarchies")
        with self._lock:
            hierarchies = list(self._hierarchies.values())
            if domain_id is not None:
                h_ids = self._domain_hierarchies.get(domain_id, set())
                hierarchies = [h for h in hierarchies if h.id in h_ids]
            return hierarchies[:limit]

    def get_hierarchy_depth(self, hierarchy_id: str) -> int:
        """
        Return the maximum depth of the hierarchy tree.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Hierarchy not found.
        """
        self._require_initialized("get_hierarchy_depth")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticStructureError(
                    f"Hierarchy '{hierarchy_id}' not found.",
                    context={"hierarchy_id": hierarchy_id},
                )
            return hierarchy.max_depth

    def detect_cycles(self, hierarchy_id: str) -> list[list[str]]:
        """
        Scan a hierarchy for cyclic parent-child relationships.

        Returns:
            A list of cycles, where each cycle is an ordered list of node IDs.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Hierarchy not found.
        """
        self._require_initialized("detect_cycles")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticStructureError(
                    f"Hierarchy '{hierarchy_id}' not found.",
                    context={"hierarchy_id": hierarchy_id},
                )
            return self._dfs_detect_cycles_in_hierarchy(hierarchy)

    # ─────────────────────────────────────────────────────────────────────────
    # DISCOVERY
    # ─────────────────────────────────────────────────────────────────────────

    def discover_related_knowledge(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        same_domain_only: bool = False,
        limit: int = 20,
    ) -> list[KnowledgeRecord]:
        """
        Discover knowledge records that are semantically related to a given
        record.

        Discovery strategy (layered):
            1. Explicit related_ids from the seed record.
            2. Shared domain membership.
            3. Name-token overlap scored by Jaccard similarity.
            4. ConceptRelationship edges if the seed is a concept.

        Args:
            record_id:        The seed record to find relatives of.
            knowledge_type:   Knowledge type of the seed record.
            same_domain_only: Restrict results to the seed's domains.
            limit:            Maximum number of records to return.

        Returns:
            List of related KnowledgeRecord objects sorted by relevance
            (descending confidence score × overlap score).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("discover_related_knowledge")
        with self._lock:
            seed = self._resolve_record(record_id, knowledge_type)
            if seed is None:
                return []

            seed_domains = set(_get_domain_ids(seed))
            seed_tokens = _searchable_tokens(seed)
            related_ids: set[str] = set(seed.related_ids)

            # Add concept relationship targets
            for rel_id in self._source_rels.get(record_id, set()):
                rel = self._relationships.get(rel_id)
                if rel:
                    related_ids.add(rel.target_concept_id)
                    if rel.is_bidirectional:
                        related_ids.add(rel.source_concept_id)
            for rel_id in self._target_rels.get(record_id, set()):
                rel = self._relationships.get(rel_id)
                if rel and rel.is_bidirectional:
                    related_ids.add(rel.source_concept_id)

            scored: list[tuple[float, KnowledgeRecord]] = []
            seen: set[str] = {record_id}

            # Resolve explicit related_ids first
            for rid in related_ids:
                if rid in seen:
                    continue
                seen.add(rid)
                for kt in _ALL_TYPES:
                    rec = self._resolve_record(rid, kt)
                    if rec and rec.status.is_usable:
                        if same_domain_only and not seed_domains.intersection(_get_domain_ids(rec)):
                            continue
                        scored.append((rec.metadata.confidence_score, rec))
                        break

            # Domain-scoped discovery via retrieval engine
            if self._retrieval_engine and self._retrieval_engine.is_initialized():
                try:
                    candidates = self._retrieval_engine.discover_related(
                        record_id,
                        knowledge_type,
                        same_domain_only=same_domain_only,
                        limit=limit * 2,
                    )
                    for rec in candidates:
                        if rec.id not in seen and rec.status.is_usable:
                            rec_tokens = _searchable_tokens(rec)
                            if seed_tokens and rec_tokens:
                                overlap = len(seed_tokens & rec_tokens)
                                jaccard = overlap / len(seed_tokens | rec_tokens)
                                score = min(jaccard * 0.6 + rec.metadata.confidence_score * 0.4, 1.0)
                            else:
                                score = rec.metadata.confidence_score * 0.4
                            if score > 0.0:
                                scored.append((score, rec))
                                seen.add(rec.id)
                except Exception:
                    pass

            scored.sort(key=lambda t: (-t[0], t[1].name.lower()))
            return [r for _, r in scored[:limit]]

    def discover_related_concepts(
        self,
        concept_id: str,
        *,
        relationship_types: Optional[list[ConceptRelationshipType]] = None,
        min_weight: float = 0.0,
        limit: int = 20,
    ) -> list[Concept]:
        """
        Discover concepts related to a given concept via ConceptRelationship
        edges.

        Args:
            concept_id:         The seed concept.
            relationship_types: Restrict to specific relationship types.
                                None means all types.
            min_weight:         Minimum relationship weight to include.
            limit:              Maximum number of concepts to return.

        Returns:
            List of Concept records sorted by relationship weight descending.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("discover_related_concepts")
        with self._lock:
            all_rel_ids = (
                self._source_rels.get(concept_id, set())
                | self._target_rels.get(concept_id, set())
            )
            scored: list[tuple[float, Concept]] = []
            seen: set[str] = {concept_id}

            for rel_id in all_rel_ids:
                rel = self._relationships.get(rel_id)
                if rel is None:
                    continue
                if relationship_types and rel.relationship_type not in relationship_types:
                    continue
                if rel.weight < min_weight:
                    continue
                # Determine the other end
                other_id = (
                    rel.target_concept_id
                    if rel.source_concept_id == concept_id
                    else rel.source_concept_id
                )
                if other_id in seen:
                    continue
                seen.add(other_id)
                concept = self._resolve_record(other_id, KnowledgeType.CONCEPT)
                if isinstance(concept, Concept) and concept.status.is_usable:
                    scored.append((rel.weight, concept))

            scored.sort(key=lambda t: (-t[0], t[1].name.lower()))
            return [c for _, c in scored[:limit]]

    def discover_related_skills(
        self,
        skill_id: str,
        *,
        include_prerequisites: bool = True,
        include_dependents: bool = True,
        limit: int = 20,
    ) -> list[Skill]:
        """
        Discover skills related to a given skill via dependency edges.

        Args:
            skill_id:              The seed skill.
            include_prerequisites: Include skills that this skill depends on.
            include_dependents:    Include skills that depend on this skill.
            limit:                 Maximum number of skills to return.

        Returns:
            List of Skill records sorted by dependency type (hard first)
            then name.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("discover_related_skills")
        with self._lock:
            scored: list[tuple[float, Skill]] = []
            seen: set[str] = {skill_id}

            # Prerequisites: dependencies where skill_id is the dependent
            if include_prerequisites:
                for dep_id in self._dependent_deps.get(skill_id, set()):
                    dep = self._dependencies.get(dep_id)
                    if dep is None:
                        continue
                    if dep.dependency_id in seen:
                        continue
                    seen.add(dep.dependency_id)
                    rec = self._resolve_record(dep.dependency_id, KnowledgeType.SKILL)
                    if isinstance(rec, Skill) and rec.status.is_usable:
                        weight = 1.0 if dep.is_hard else 0.5
                        scored.append((weight, rec))

            # Dependents: dependencies where skill_id is the required record
            if include_dependents:
                for dep_id in self._dependency_deps.get(skill_id, set()):
                    dep = self._dependencies.get(dep_id)
                    if dep is None:
                        continue
                    if dep.dependent_id in seen:
                        continue
                    seen.add(dep.dependent_id)
                    rec = self._resolve_record(dep.dependent_id, KnowledgeType.SKILL)
                    if isinstance(rec, Skill) and rec.status.is_usable:
                        weight = 0.8 if dep.is_hard else 0.4
                        scored.append((weight, rec))

            scored.sort(key=lambda t: (-t[0], t[1].name.lower()))
            return [s for _, s in scored[:limit]]

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def dependency_analysis(
        self,
        record_id: str,
        *,
        max_depth: int = 10,
    ) -> dict[str, Any]:
        """
        Produce a full dependency depth map for a knowledge record.

        Traverses the KnowledgeDependency graph from the seed record,
        resolving all transitive hard and soft dependencies up to max_depth
        levels.

        Returns::

            {
                "record_id": str,
                "max_depth": int,
                "depth_map": {
                    "0": [record_id],
                    "1": [direct_dep_id, ...],
                    "2": [...],
                },
                "all_dependency_ids": [str],
                "hard_dependency_count": int,
                "soft_dependency_count": int,
                "cycle_detected": bool,
                "analysed_at": str,
            }

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("dependency_analysis")
        with self._lock:
            depth_map: dict[str, list[str]] = {"0": [record_id]}
            all_deps: set[str] = set()
            hard_count = 0
            soft_count = 0
            cycle_detected = False
            frontier: set[str] = {record_id}
            visited: set[str] = {record_id}

            for depth in range(1, max_depth + 1):
                next_frontier: set[str] = set()
                for current_id in frontier:
                    for dep_id in self._dependent_deps.get(current_id, set()):
                        dep = self._dependencies.get(dep_id)
                        if dep is None:
                            continue
                        if dep.dependency_id in visited:
                            if dep.dependency_id != record_id:
                                # Not a true cycle back to root, skip
                                pass
                            else:
                                cycle_detected = True
                            continue
                        visited.add(dep.dependency_id)
                        next_frontier.add(dep.dependency_id)
                        all_deps.add(dep.dependency_id)
                        if dep.is_hard:
                            hard_count += 1
                        else:
                            soft_count += 1

                if not next_frontier:
                    break
                depth_map[str(depth)] = sorted(next_frontier)
                frontier = next_frontier

            return {
                "record_id": record_id,
                "max_depth": max_depth,
                "depth_map": depth_map,
                "all_dependency_ids": sorted(all_deps),
                "hard_dependency_count": hard_count,
                "soft_dependency_count": soft_count,
                "cycle_detected": cycle_detected,
                "analysed_at": _utcnow().isoformat(),
            }

    def hierarchy_analysis(self, hierarchy_id: str) -> dict[str, Any]:
        """
        Compute structural statistics for a SemanticHierarchy.

        Returns::

            {
                "hierarchy_id": str,
                "name": str,
                "node_count": int,
                "max_depth": int,
                "avg_depth": float,
                "leaf_count": int,
                "branch_count": int,
                "root_node_id": str,
                "root_children": int,
                "knowledge_type_distribution": {type_value: count},
                "analysed_at": str,
            }

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SemanticStructureError:  Hierarchy not found.
        """
        self._require_initialized("hierarchy_analysis")
        with self._lock:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if hierarchy is None:
                raise SemanticStructureError(
                    f"Hierarchy '{hierarchy_id}' not found.",
                    context={"hierarchy_id": hierarchy_id},
                )
            nodes = list(hierarchy.nodes.values())
            depths = [n.depth for n in nodes]
            leaf_count = sum(1 for n in nodes if n.is_leaf)
            branch_count = sum(1 for n in nodes if not n.is_leaf)
            avg_depth = sum(depths) / len(depths) if depths else 0.0

            type_dist: dict[str, int] = defaultdict(int)
            for n in nodes:
                type_dist[n.knowledge_type.value] += 1

            root = hierarchy.root_node
            root_children = len(root.child_node_ids) if root else 0

            return {
                "hierarchy_id": hierarchy_id,
                "name": hierarchy.name,
                "node_count": len(nodes),
                "max_depth": max(depths) if depths else 0,
                "avg_depth": round(avg_depth, 3),
                "leaf_count": leaf_count,
                "branch_count": branch_count,
                "root_node_id": hierarchy.root_node_id,
                "root_children": root_children,
                "knowledge_type_distribution": dict(type_dist),
                "analysed_at": _utcnow().isoformat(),
            }

    def graph_analysis(self) -> dict[str, Any]:
        """
        Compute graph-wide metrics across all registered semantic relationships
        and dependencies.

        Metrics:
            - Total nodes, relationships, dependencies.
            - Average out-degree per node in the relationship graph.
            - Top-5 most connected concept IDs (by total in+out degree).
            - Relationship type distribution.
            - Dependency type distribution.
            - Isolated nodes (no edges).

        Returns:
            A dict of graph metrics.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("graph_analysis")
        with self._lock:
            degree_map: dict[str, int] = defaultdict(int)
            rel_type_dist: dict[str, int] = defaultdict(int)
            dep_type_dist: dict[str, int] = defaultdict(int)

            for rel in self._relationships.values():
                degree_map[rel.source_concept_id] += 1
                degree_map[rel.target_concept_id] += 1
                rel_type_dist[rel.relationship_type.value] += 1

            for dep in self._dependencies.values():
                degree_map[dep.dependent_id] += 1
                degree_map[dep.dependency_id] += 1
                dep_type_dist[dep.dependency_type] += 1

            total_concepts = len(set(
                list(self._source_rels.keys()) + list(self._target_rels.keys())
            ))
            total_edges = len(self._relationships)
            avg_out_degree = (
                total_edges / total_concepts if total_concepts > 0 else 0.0
            )

            top_connected = sorted(
                degree_map.items(), key=lambda t: -t[1]
            )[:5]

            all_node_ids = set(self._nodes.keys())
            node_knowledge_ids: set[str] = {n.knowledge_id for n in self._nodes.values()}
            edge_ids = set(degree_map.keys())
            isolated = node_knowledge_ids - edge_ids

            return {
                "total_nodes": len(self._nodes),
                "total_relationships": len(self._relationships),
                "total_dependencies": len(self._dependencies),
                "total_hierarchies": len(self._hierarchies),
                "avg_out_degree": round(avg_out_degree, 4),
                "top_connected_nodes": [
                    {"knowledge_id": kid, "degree": deg}
                    for kid, deg in top_connected
                ],
                "relationship_type_distribution": dict(rel_type_dist),
                "dependency_type_distribution": dict(dep_type_dist),
                "isolated_knowledge_ids": sorted(isolated),
                "isolated_count": len(isolated),
                "analysed_at": _utcnow().isoformat(),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT / REPORTING
    # ─────────────────────────────────────────────────────────────────────────

    def semantic_audit(self) -> dict[str, Any]:
        """
        Perform a comprehensive structural integrity audit of all semantic
        data managed by this engine.

        Checks:
            - All hierarchy root_node_ids resolve to existing nodes.
            - All node parent/child references are internally consistent.
            - No dangling relationship endpoints in the relationship store.
            - No self-loop relationships or dependencies.
            - No hard-dependency cycles.
            - Index consistency between _hierarchy_nodes and hierarchy.nodes.

        Returns::

            {
                "passed": bool,
                "issues": [{"severity": str, "area": str, "message": str}],
                "stats": {...},
                "audited_at": str,
            }

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("semantic_audit")
        with self._lock:
            self._audit_count += 1
            issues: list[dict[str, Any]] = []

            # ── Hierarchy integrity ───────────────────────────────────────────
            for h_id, hierarchy in self._hierarchies.items():
                if hierarchy.root_node_id not in hierarchy.nodes:
                    issues.append({
                        "severity": "ERROR",
                        "area": "hierarchy",
                        "message": (
                            f"Hierarchy '{h_id}' root_node_id "
                            f"'{hierarchy.root_node_id}' not in nodes."
                        ),
                    })
                # Index consistency
                index_node_ids = self._hierarchy_nodes.get(h_id, set())
                store_node_ids = set(hierarchy.nodes.keys())
                extra_in_index = index_node_ids - store_node_ids
                missing_from_index = store_node_ids - index_node_ids
                for nid in extra_in_index:
                    issues.append({
                        "severity": "WARNING",
                        "area": "hierarchy_index",
                        "message": f"Node '{nid}' in hierarchy_nodes index but not in hierarchy '{h_id}'.",
                    })
                for nid in missing_from_index:
                    issues.append({
                        "severity": "WARNING",
                        "area": "hierarchy_index",
                        "message": f"Node '{nid}' in hierarchy '{h_id}' but missing from hierarchy_nodes index.",
                    })
                # Cycle detection
                cycles = self._dfs_detect_cycles_in_hierarchy(hierarchy)
                for cycle in cycles:
                    issues.append({
                        "severity": "ERROR",
                        "area": "hierarchy_cycle",
                        "message": f"Cycle in hierarchy '{h_id}': {' → '.join(cycle)}",
                    })

            # ── Node integrity ────────────────────────────────────────────────
            for node_id, node in self._nodes.items():
                if not node.knowledge_id:
                    issues.append({
                        "severity": "ERROR",
                        "area": "node",
                        "message": f"Node '{node_id}' has empty knowledge_id.",
                    })
                if node.parent_node_id and node.parent_node_id not in self._nodes:
                    issues.append({
                        "severity": "ERROR",
                        "area": "node",
                        "message": (
                            f"Node '{node_id}' parent '{node.parent_node_id}' "
                            f"not in global node store."
                        ),
                    })
                for child_id in node.child_node_ids:
                    if child_id not in self._nodes:
                        issues.append({
                            "severity": "ERROR",
                            "area": "node",
                            "message": (
                                f"Node '{node_id}' child '{child_id}' "
                                f"not in global node store."
                            ),
                        })

            # ── Relationship integrity ────────────────────────────────────────
            for rel_id, rel in self._relationships.items():
                if rel.source_concept_id == rel.target_concept_id:
                    issues.append({
                        "severity": "WARNING",
                        "area": "relationship",
                        "message": f"Relationship '{rel_id}' is a self-loop.",
                    })
                if rel_id not in self._source_rels.get(rel.source_concept_id, set()):
                    issues.append({
                        "severity": "WARNING",
                        "area": "relationship_index",
                        "message": (
                            f"Relationship '{rel_id}' not in source_rels "
                            f"for '{rel.source_concept_id}'."
                        ),
                    })

            # ── Dependency integrity ──────────────────────────────────────────
            hard_cycles = self._detect_dependency_cycles(hard_only=True)
            for cycle in hard_cycles:
                issues.append({
                    "severity": "ERROR",
                    "area": "dependency_cycle",
                    "message": f"Hard-dependency cycle: {' → '.join(cycle)}",
                })

            for dep_id, dep in self._dependencies.items():
                if dep.dependent_id == dep.dependency_id:
                    issues.append({
                        "severity": "ERROR",
                        "area": "dependency",
                        "message": f"Dependency '{dep_id}' is a self-loop on '{dep.dependent_id}'.",
                    })

            passed = all(i["severity"] != "ERROR" for i in issues)
            logger.info(
                "semantic_audit complete: passed=%s issues=%d", passed, len(issues)
            )
            return {
                "passed": passed,
                "issues": issues,
                "stats": {
                    "hierarchies": len(self._hierarchies),
                    "nodes": len(self._nodes),
                    "relationships": len(self._relationships),
                    "dependencies": len(self._dependencies),
                    "hard_cycles": len(hard_cycles),
                },
                "audited_at": _utcnow().isoformat(),
            }

    def semantic_reporting(self) -> dict[str, Any]:
        """
        Return a comprehensive human-readable snapshot of the engine's current
        state, suitable for dashboards and health summaries.

        Returns::

            {
                "engine": "SemanticStructureEngine",
                "version": str,
                "initialized": bool,
                "started_at": str | None,
                "hierarchy_count": int,
                "node_count": int,
                "relationship_count": int,
                "dependency_count": int,
                "graph_builds": int,
                "graph_validations": int,
                "audit_count": int,
                "mutation_count": int,
                "last_mutation_at": str | None,
                "hierarchy_summaries": [...],
                "relationship_type_distribution": {...},
                "dependency_type_distribution": {...},
                "generated_at": str,
            }

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        self._require_initialized("semantic_reporting")
        with self._lock:
            rel_type_dist: dict[str, int] = defaultdict(int)
            for rel in self._relationships.values():
                rel_type_dist[rel.relationship_type.value] += 1

            dep_type_dist: dict[str, int] = defaultdict(int)
            for dep in self._dependencies.values():
                dep_type_dist[dep.dependency_type] += 1

            hierarchy_summaries = [
                {
                    "id": h.id,
                    "name": h.name,
                    "domain_id": h.domain_id,
                    "node_count": h.node_count,
                    "max_depth": h.max_depth,
                    "created_at": h.created_at.isoformat(),
                    "updated_at": h.updated_at.isoformat(),
                }
                for h in self._hierarchies.values()
            ]

            return {
                "engine": "SemanticStructureEngine",
                "version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "hierarchy_count": len(self._hierarchies),
                "node_count": len(self._nodes),
                "relationship_count": len(self._relationships),
                "dependency_count": len(self._dependencies),
                "graph_builds": self._graph_builds,
                "graph_validations": self._graph_validations,
                "audit_count": self._audit_count,
                "mutation_count": self._mutation_count,
                "last_mutation_at": (
                    self._last_mutation_at.isoformat() if self._last_mutation_at else None
                ),
                "hierarchy_summaries": hierarchy_summaries,
                "relationship_type_distribution": dict(rel_type_dist),
                "dependency_type_distribution": dict(dep_type_dist),
                "generated_at": _utcnow().isoformat(),
            }

    def audit_report(self) -> dict[str, Any]:
        """
        Return semantic structure statistics (AbstractSemanticStructureEngine
        contract).  Delegates to semantic_reporting().

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """
        return self.semantic_reporting()

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """
        Return a lightweight liveness/readiness summary.

        Required keys: engine, initialized, record_count, status.
        """
        with self._lock:
            return {
                "engine": "SemanticStructureEngine",
                "initialized": self._initialized,
                "record_count": len(self._nodes),
                "status": "healthy" if self._initialized else "offline",
            }

    def diagnostics_report(self) -> dict[str, Any]:
        """
        Return a full introspection snapshot for operator debugging.

        Required keys: engine, initialized, record_count, status,
        index_size, duplicate_checks, mutation_count, last_mutation_at.
        """
        with self._lock:
            return {
                "engine": "SemanticStructureEngine",
                "version": _ENGINE_VERSION,
                "initialized": self._initialized,
                "status": "healthy" if self._initialized else "offline",
                "record_count": len(self._nodes),
                "index_size": (
                    len(self._hierarchy_nodes)
                    + len(self._record_nodes)
                    + len(self._source_rels)
                    + len(self._target_rels)
                ),
                "duplicate_checks": 0,   # Not applicable to this engine
                "mutation_count": self._mutation_count,
                "last_mutation_at": (
                    self._last_mutation_at.isoformat() if self._last_mutation_at else None
                ),
                "hierarchy_count": len(self._hierarchies),
                "relationship_count": len(self._relationships),
                "dependency_count": len(self._dependencies),
                "graph_builds": self._graph_builds,
                "graph_validations": self._graph_validations,
                "audit_count": self._audit_count,
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "connected_engines": {
                    "concept": self._concept_engine is not None,
                    "domain": self._domain_engine is not None,
                    "index": self._index_engine is not None,
                    "integrity": self._integrity_engine is not None,
                    "retrieval": self._retrieval_engine is not None,
                },
            }

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL CYCLE DETECTION HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _would_create_cycle_in_hierarchy(
        self,
        hierarchy: SemanticHierarchy,
        parent_node_id: str,
        new_knowledge_id: str,
    ) -> bool:
        """
        Return True if attaching new_knowledge_id as a child of parent_node_id
        would create a cycle in the hierarchy.

        Cycles arise when new_knowledge_id is already an ancestor of
        parent_node_id (i.e. adding it as a child would close the loop).
        """
        # Walk ancestors of parent_node_id; if any have knowledge_id == new_knowledge_id
        # or node_id == the hypothetical new node, we have a cycle.
        visited: set[str] = set()
        queue: deque[str] = deque([parent_node_id])
        while queue:
            current_node_id = queue.popleft()
            if current_node_id in visited:
                continue
            visited.add(current_node_id)
            current = hierarchy.nodes.get(current_node_id)
            if current is None:
                continue
            if current.knowledge_id == new_knowledge_id:
                return True
            if current.parent_node_id:
                queue.append(current.parent_node_id)
        return False

    def _dfs_detect_cycles_in_hierarchy(
        self, hierarchy: SemanticHierarchy
    ) -> list[list[str]]:
        """
        DFS-based cycle detection within a single hierarchy's parent-child tree.

        Returns a list of cycles, each represented as an ordered list of node IDs.
        For a well-formed tree there should be no cycles.
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        path: list[str] = []
        path_set: set[str] = set()

        def dfs(node_id: str) -> None:
            if node_id in path_set:
                cycle_start = path.index(node_id)
                cycles.append(path[cycle_start:] + [node_id])
                return
            if node_id in visited:
                return
            visited.add(node_id)
            path.append(node_id)
            path_set.add(node_id)
            node = hierarchy.nodes.get(node_id)
            if node:
                for child_id in node.child_node_ids:
                    dfs(child_id)
            path.pop()
            path_set.discard(node_id)

        for node_id in hierarchy.nodes:
            dfs(node_id)

        return cycles

    def _detect_dependency_cycles(self, *, hard_only: bool = False) -> list[list[str]]:
        """
        DFS-based cycle detection across all KnowledgeDependency edges.

        Args:
            hard_only: If True, only traverse hard (blocking) dependencies.

        Returns:
            A list of cycles, each as an ordered list of knowledge record IDs.
        """
        # Build adjacency list: dependent_id → [dependency_id, ...]
        adj: dict[str, list[str]] = defaultdict(list)
        for dep in self._dependencies.values():
            if hard_only and not dep.is_hard:
                continue
            adj[dep.dependent_id].append(dep.dependency_id)

        cycles: list[list[str]] = []
        visited: set[str] = set()
        path: list[str] = []
        path_set: set[str] = set()

        def dfs(node: str) -> None:
            if node in path_set:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            path_set.add(node)
            for neighbor in adj.get(node, []):
                dfs(neighbor)
            path.pop()
            path_set.discard(node)

        for node in list(adj.keys()):
            dfs(node)

        return cycles


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["SemanticStructureEngine"]
