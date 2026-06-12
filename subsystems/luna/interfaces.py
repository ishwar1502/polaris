"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/interfaces.py

Abstract contracts (ABCs) for every LUNA engine.

Every concrete engine implementation in LUNA MUST inherit from one of the
abstract classes defined here and provide a complete, non-stub implementation
of every abstract method.  This module is the single authoritative definition
of the LUNA engine surface; no engine may expose methods that are absent from
its corresponding ABC.

LUNA engine inventory (§4 of spec):
    FactEngine                 →  AbstractFactEngine
    ConceptEngine              →  AbstractConceptEngine
    SkillEngine                →  AbstractSkillEngine
    KnowledgeDomainEngine      →  AbstractKnowledgeDomainEngine
    ProceduralKnowledgeEngine  →  AbstractProceduralKnowledgeEngine
    ResearchKnowledgeEngine    →  AbstractResearchKnowledgeEngine
    EducationalKnowledgeEngine →  AbstractEducationalKnowledgeEngine
    KnowledgeValidationEngine  →  AbstractKnowledgeValidationEngine
    KnowledgeRetrievalEngine   →  AbstractKnowledgeRetrievalEngine
    KnowledgeSynthesisEngine   →  AbstractKnowledgeSynthesisEngine
    KnowledgeEvolutionEngine   →  AbstractKnowledgeEvolutionEngine
    SkillProgressionEngine     →  AbstractSkillProgressionEngine
    KnowledgeIntegrityEngine   →  AbstractKnowledgeIntegrityEngine
    KnowledgeIndexEngine       →  AbstractKnowledgeIndexEngine
    SemanticStructureEngine    →  AbstractSemanticStructureEngine

Lifecycle contract (all engines):
    initialize()      — allocate resources, warm caches, open handles
    shutdown()        — release resources gracefully
    is_initialized()  — guard predicate used at every call-site
    health_report()   — lightweight liveness summary (dict)
    diagnostics_report() — full introspection snapshot (dict)

Part of the POLARIS Cognitive Substrate:
    ASTRA        → Identity
    ECHO         → Experience
    LUNA         → Knowledge       ← this module
    CHRONOS      → Time
    CONSTELLATION → Relationships
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from subsystems.luna.models import (
    # Enums
    ConceptType,
    EducationType,
    FactType,
    KnowledgeDifficulty,
    KnowledgeSourceType,
    KnowledgeStatus,
    KnowledgeType,
    ProcedureType,
    ResearchType,
    SkillLevel,
    SkillType,
    ValidationStatus,
    # Core models
    Concept,
    ConceptRelationship,
    ConceptRelationshipType,
    DomainStructure,
    EducationalKnowledge,
    Fact,
    IntegrityReport,
    KnowledgeAuditReport,
    KnowledgeComposition,
    KnowledgeDomain,
    KnowledgeDependency,
    KnowledgeIndexEntry,
    ConceptIndexEntry,
    DomainIndexEntry,
    SkillIndexEntry,
    KnowledgeMetadata,
    KnowledgePackage,
    KnowledgeRecord,
    KnowledgeSynthesis,
    KnowledgeValidationResult,
    Procedure,
    ResearchKnowledge,
    SemanticHierarchy,
    SemanticNode,
    Skill,
    SkillPrerequisite,
    SkillProgressionModel,
    SkillStage,
)


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE MIXIN
# ─────────────────────────────────────────────────────────────────────────────

class LunaEngineLifecycle(ABC):
    """
    Shared lifecycle contract inherited by every LUNA engine.

    Implementing classes must manage an internal ``_initialized`` flag and
    raise ``LunaNotInitializedError`` in every public method before the engine
    has been started, and after it has been shut down.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> None:
        """
        Allocate resources, warm caches, and prepare the engine for operation.

        Must be idempotent: calling initialize() on an already-initialized
        engine must be a no-op (not raise an error).

        Raises:
            LunaLifecycleError: If initialization fails for any internal reason.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """
        Release resources and put the engine into a quiescent state.

        After shutdown(), is_initialized() must return False and all mutating
        methods must raise LunaNotInitializedError.

        Must be idempotent: calling shutdown() on an already-stopped engine
        must be a no-op.

        Raises:
            LunaLifecycleError: If teardown fails for any internal reason.
        """

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Return True if the engine is fully started and ready to serve requests.
        """

    # ── Observability ─────────────────────────────────────────────────────────

    @abstractmethod
    def health_report(self) -> dict[str, Any]:
        """
        Return a lightweight liveness/readiness summary.

        Required keys:
            engine        (str)   — fully-qualified engine class name
            initialized   (bool)  — mirrors is_initialized()
            record_count  (int)   — number of records currently held
            status        (str)   — "healthy" | "degraded" | "offline"

        Additional engine-specific keys are permitted.
        """

    @abstractmethod
    def diagnostics_report(self) -> dict[str, Any]:
        """
        Return a full introspection snapshot suitable for operator debugging.

        Required keys (superset of health_report):
            engine            (str)
            initialized       (bool)
            record_count      (int)
            status            (str)
            index_size        (int)   — number of index entries
            duplicate_checks  (int)   — fingerprint-based dedup checks performed
            mutation_count    (int)   — total create/update/delete operations
            last_mutation_at  (str | None) — ISO-8601 datetime of last write

        Additional engine-specific keys are permitted.
        """


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT FACT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractFactEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Fact Engine.

    The Fact Engine is responsible for the full lifecycle of atomic facts
    (truth claims) owned by LUNA: Ohm's Law, Newton's Laws, Python syntax
    rules, control theory constants, etc.

    Ownership law: LUNA owns all facts.  No external module may write,
    delete, or modify facts without going through this engine.

    CRUD surface:
        create_fact      — add a new atomic fact
        update_fact      — mutate a non-terminal fact record
        delete_fact      — soft-delete (status → DEPRECATED or RETRACTED)
        retrieve_fact    — fetch a single fact by ID

    Search surface:
        search_facts        — free-text / field query
        search_facts_by_type — filter by FactType
        search_facts_by_domain — filter by domain ID

    Validation surface:
        validate_fact       — run structural + confidence checks

    Integrity surface:
        find_duplicate_facts — return groups of likely-duplicate facts

    Audit surface:
        audit_report        — aggregate statistics for the fact store
    """

    # ── CRUD ──────────────────────────────────────────────────────────────────

    @abstractmethod
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
        """
        Create and store a new atomic fact.

        Performs duplicate detection by content fingerprint before inserting.

        Args:
            name:             Short canonical name (e.g. "Ohm's Law").
            description:      Human-readable explanation.
            statement:        Precise, canonical truth statement.
            fact_type:        Classification enum (LAW, FORMULA, DEFINITION …).
            difficulty:       KnowledgeDifficulty tier.
            domain_ids:       List of owning KnowledgeDomain IDs.
            metadata:         Provenance, confidence, and versioning metadata.
            formal_notation:  Optional mathematical/formal expression.
            units:            Optional SI unit string.
            conditions:       Conditions under which the fact holds.
            aliases:          Alternate names or abbreviations.
            notes:            Free-text notes.

        Returns:
            The newly created and stored Fact record.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
            DuplicateFactError:      If an identical fact already exists.
            FactValidationError:     If the provided data fails validation.
        """

    @abstractmethod
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
        """
        Apply a partial update to an existing fact record.

        Only keyword-supplied fields are modified; omitted fields retain their
        current values.  Version is incremented automatically.

        Args:
            fact_id:  ID of the fact to update.
            **fields: Any subset of Fact fields to overwrite.

        Returns:
            The updated Fact record.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
            FactNotFoundError:       If no fact with the given ID exists.
            FactValidationError:     If the updated state fails validation.
        """

    @abstractmethod
    def delete_fact(self, fact_id: str, *, reason: str = "") -> Fact:
        """
        Soft-delete a fact by transitioning its status to RETRACTED.

        Facts are never physically removed; deletion is a status transition
        so that audit history and downstream references remain coherent.

        Args:
            fact_id: ID of the fact to retract.
            reason:  Human-readable reason for deletion (stored in notes).

        Returns:
            The retracted Fact record.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
            FactNotFoundError:       If no fact with the given ID exists.
        """

    @abstractmethod
    def retrieve_fact(self, fact_id: str) -> Fact:
        """
        Fetch a single fact by its unique ID.

        Args:
            fact_id: The fact's UUID string.

        Returns:
            The matching Fact record.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
            FactNotFoundError:       If no fact with the given ID exists.
        """

    # ── Search ────────────────────────────────────────────────────────────────

    @abstractmethod
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
        """
        Search the fact store against name, statement, description, and aliases.

        Args:
            query:          Free-text search string (case-insensitive substring).
            domain_ids:     Restrict to facts belonging to these domains.
            fact_types:     Restrict to these FactType values.
            difficulty:     Exact difficulty tier filter.
            status_filter:  Restrict to these KnowledgeStatus values.
            min_confidence: Minimum metadata.confidence_score threshold.
            limit:          Maximum number of results to return.
            offset:         Pagination offset.

        Returns:
            Ordered list of matching Fact records (highest relevance first).

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
        """

    @abstractmethod
    def search_facts_by_type(
        self,
        fact_type: FactType,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Fact]:
        """
        Return all active facts of a given FactType.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
        """

    @abstractmethod
    def search_facts_by_domain(
        self,
        domain_id: str,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Fact]:
        """
        Return all facts associated with a given knowledge domain.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
        """

    @abstractmethod
    def get_all_facts(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[Fact]:
        """
        Return a paginated slice of the fact store.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
        """

    # ── Validation ────────────────────────────────────────────────────────────

    @abstractmethod
    def validate_fact(self, fact_id: str) -> KnowledgeValidationResult:
        """
        Run structural and semantic validation on a single fact.

        Checks performed:
            - Non-empty name and statement
            - confidence_score in [0.0, 1.0]
            - fact_type consistency
            - domain_ids non-empty
            - No self-referential supporting_fact_ids

        Returns:
            A KnowledgeValidationResult capturing pass/fail and all issues.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
            FactNotFoundError:       If no fact with the given ID exists.
        """

    # ── Integrity ─────────────────────────────────────────────────────────────

    @abstractmethod
    def find_duplicate_facts(self) -> list[list[Fact]]:
        """
        Detect groups of facts that share the same content fingerprint.

        Returns:
            A list of groups, where each group contains two or more facts
            that are semantically identical by fingerprint.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
        """

    @abstractmethod
    def fact_exists(self, fact_id: str) -> bool:
        """
        Return True if a fact with the given ID exists (any status).

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
        """

    @abstractmethod
    def get_fact_count(self, *, active_only: bool = False) -> int:
        """
        Return the number of facts in the store.

        Args:
            active_only: When True, count only VALIDATED and ACTIVE facts.

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
        """

    # ── Audit ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate statistics for the entire fact store.

        Required keys:
            total_facts        (int)
            active_facts       (int)
            draft_facts        (int)
            deprecated_facts   (int)
            retracted_facts    (int)
            avg_confidence     (float)
            facts_by_type      (dict[str, int])
            facts_by_domain    (dict[str, int])
            duplicate_groups   (int)
            low_confidence_ids (list[str])
            stale_ids          (list[str])
            generated_at       (str)

        Raises:
            LunaNotInitializedError: If the engine has not been initialized.
        """


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT CONCEPT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractConceptEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Concept Engine.

    Manages conceptual knowledge — ideas and abstractions that form knowledge
    domains: Machine Learning, Robotics, PID Control, Mechatronics, etc.

    Ownership law (Law 2): LUNA owns concepts.  CONSTELLATION may only link
    them.  External modules must never mutate concept records directly.

    Hierarchy responsibilities:
        - Maintain parent/child (prerequisite / child_concept) edges
        - Detect and refuse relationship cycles
        - Link facts into their parent concepts (fact_ids)
        - Resolve transitive prerequisite chains
    """

    # ── CRUD ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def create_concept(
        self,
        name: str,
        description: str,
        concept_type: ConceptType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        core_ideas: Optional[list[str]] = None,
        applications: Optional[list[str]] = None,
        prerequisite_concept_ids: Optional[list[str]] = None,
        is_foundational: bool = False,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> Concept:
        """
        Create and store a new concept.

        Performs duplicate detection by name fingerprint.  All prerequisite
        IDs are validated to exist before the concept is stored.

        Returns:
            The newly created and stored Concept record.

        Raises:
            LunaNotInitializedError:  Engine not yet initialized.
            DuplicateConceptError:    An identical concept already exists.
            ConceptValidationError:   Provided data fails validation.
            ConceptNotFoundError:     A prerequisite ID does not exist.
        """

    @abstractmethod
    def update_concept(
        self,
        concept_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        concept_type: Optional[ConceptType] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        domain_ids: Optional[list[str]] = None,
        core_ideas: Optional[list[str]] = None,
        applications: Optional[list[str]] = None,
        prerequisite_concept_ids: Optional[list[str]] = None,
        child_concept_ids: Optional[list[str]] = None,
        fact_ids: Optional[list[str]] = None,
        is_foundational: Optional[bool] = None,
        aliases: Optional[list[str]] = None,
        related_ids: Optional[list[str]] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
    ) -> Concept:
        """
        Apply a partial update to an existing concept.

        Version is incremented automatically.  Cycle detection is re-run
        after any modification to prerequisite_concept_ids or child_concept_ids.

        Raises:
            LunaNotInitializedError:   Engine not initialized.
            ConceptNotFoundError:      Concept ID does not exist.
            ConceptValidationError:    Updated state fails validation.
            ConceptRelationshipError:  Update would introduce a hierarchy cycle.
        """

    @abstractmethod
    def delete_concept(self, concept_id: str, *, reason: str = "") -> Concept:
        """
        Soft-delete a concept (status → RETRACTED).

        Does not cascade: child concepts remain in the store with a stale
        parent reference; callers are responsible for re-parenting.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Concept ID does not exist.
        """

    @abstractmethod
    def retrieve_concept(self, concept_id: str) -> Concept:
        """
        Fetch a single concept by its unique ID.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Concept ID does not exist.
        """

    # ── Search ────────────────────────────────────────────────────────────────

    @abstractmethod
    def search_concepts(
        self,
        query: str,
        *,
        domain_ids: Optional[list[str]] = None,
        concept_types: Optional[list[ConceptType]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        foundational_only: bool = False,
        min_confidence: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Concept]:
        """
        Search concepts by name, description, core_ideas, and aliases.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """

    @abstractmethod
    def search_concepts_by_domain(
        self,
        domain_id: str,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Concept]:
        """Return all concepts belonging to a given domain."""

    @abstractmethod
    def get_all_concepts(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[Concept]:
        """Return a paginated slice of the concept store."""

    @abstractmethod
    def get_foundational_concepts(self) -> list[Concept]:
        """Return all active concepts marked is_foundational = True."""

    # ── Hierarchy ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_children(self, concept_id: str) -> list[Concept]:
        """
        Return direct child concepts of the given concept.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Concept ID does not exist.
        """

    @abstractmethod
    def get_prerequisites(self, concept_id: str) -> list[Concept]:
        """
        Return direct prerequisite concepts for the given concept.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Concept ID does not exist.
        """

    @abstractmethod
    def get_transitive_prerequisites(self, concept_id: str) -> list[Concept]:
        """
        Return all concepts that must be understood before the given concept
        (recursive prerequisite closure), sorted from root to leaf.

        Raises:
            LunaNotInitializedError:   Engine not initialized.
            ConceptNotFoundError:      Concept ID does not exist.
            ConceptRelationshipError:  A cycle is detected in the prerequisite graph.
        """

    @abstractmethod
    def add_prerequisite(self, concept_id: str, prerequisite_id: str) -> Concept:
        """
        Add a prerequisite edge: prerequisite_id → concept_id.

        Performs cycle detection before committing.

        Raises:
            LunaNotInitializedError:   Engine not initialized.
            ConceptNotFoundError:      Either concept ID does not exist.
            ConceptRelationshipError:  Edge would create a cycle.
        """

    @abstractmethod
    def remove_prerequisite(self, concept_id: str, prerequisite_id: str) -> Concept:
        """
        Remove a prerequisite edge.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Either concept ID does not exist.
        """

    @abstractmethod
    def add_child_concept(self, parent_id: str, child_id: str) -> Concept:
        """
        Register child_id as a child of parent_id.

        Also updates child_id's prerequisite_concept_ids to include parent_id.
        Performs cycle detection before committing.

        Raises:
            LunaNotInitializedError:   Engine not initialized.
            ConceptNotFoundError:      Either concept ID does not exist.
            ConceptRelationshipError:  Relationship would create a cycle.
        """

    @abstractmethod
    def remove_child_concept(self, parent_id: str, child_id: str) -> Concept:
        """
        Remove a child relationship from parent.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Either concept ID does not exist.
        """

    # ── Fact linkage ──────────────────────────────────────────────────────────

    @abstractmethod
    def attach_fact(self, concept_id: str, fact_id: str) -> Concept:
        """
        Link a fact to a concept (append to concept.fact_ids).

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Concept ID does not exist.
        """

    @abstractmethod
    def detach_fact(self, concept_id: str, fact_id: str) -> Concept:
        """
        Remove a fact link from a concept.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Concept ID does not exist.
        """

    @abstractmethod
    def get_facts_for_concept(self, concept_id: str) -> list[str]:
        """
        Return the list of fact IDs attached to the given concept.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Concept ID does not exist.
        """

    # ── Validation ────────────────────────────────────────────────────────────

    @abstractmethod
    def validate_concept(self, concept_id: str) -> KnowledgeValidationResult:
        """
        Run structural, semantic, and referential validation on a concept.

        Checks performed:
            - Non-empty name and description
            - confidence_score in [0.0, 1.0]
            - All prerequisite_concept_ids resolve to existing concepts
            - All fact_ids resolve to existing facts (cross-engine lookup)
            - No cycles in prerequisite/child edges
            - domain_ids non-empty

        Raises:
            LunaNotInitializedError: Engine not initialized.
            ConceptNotFoundError:    Concept ID does not exist.
        """

    # ── Integrity ─────────────────────────────────────────────────────────────

    @abstractmethod
    def find_duplicate_concepts(self) -> list[list[Concept]]:
        """
        Detect groups of concepts sharing the same content fingerprint.

        Returns:
            A list of duplicate groups (each with ≥ 2 members).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """

    @abstractmethod
    def find_orphaned_concepts(self) -> list[Concept]:
        """
        Return concepts with no domain_ids, no parent, and no prerequisites.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """

    @abstractmethod
    def concept_exists(self, concept_id: str) -> bool:
        """Return True if a concept with the given ID exists (any status)."""

    @abstractmethod
    def get_concept_count(self, *, active_only: bool = False) -> int:
        """Return the number of concepts in the store."""

    # ── Audit ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """
        Return aggregate statistics for the concept store.

        Required keys:
            total_concepts       (int)
            active_concepts      (int)
            foundational_count   (int)
            avg_confidence       (float)
            concepts_by_type     (dict[str, int])
            concepts_by_domain   (dict[str, int])
            duplicate_groups     (int)
            orphaned_count       (int)
            max_hierarchy_depth  (int)
            generated_at         (str)

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT SKILL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractSkillEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Skill Engine.

    LUNA owns the canonical skill definition (what a skill IS).
    ASTRA owns the user's proficiency profile (how well the user performs it).

    Ownership law (Law 3): LUNA owns skills; ASTRA owns skill tendencies.
    """

    @abstractmethod
    def create_skill(
        self,
        name: str,
        description: str,
        skill_type: SkillType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        applications: Optional[list[str]] = None,
        prerequisite_skill_ids: Optional[list[str]] = None,
        related_concept_ids: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> Skill:
        """
        Create and register a new skill definition.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            DuplicateSkillError:     Skill already exists by fingerprint.
            SkillValidationError:    Provided data fails validation.
        """

    @abstractmethod
    def update_skill(
        self,
        skill_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        skill_type: Optional[SkillType] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        domain_ids: Optional[list[str]] = None,
        applications: Optional[list[str]] = None,
        prerequisite_skill_ids: Optional[list[str]] = None,
        related_concept_ids: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
    ) -> Skill:
        """Update fields on an existing skill. Version auto-incremented."""

    @abstractmethod
    def delete_skill(self, skill_id: str, *, reason: str = "") -> Skill:
        """Soft-delete a skill (status → RETRACTED)."""

    @abstractmethod
    def retrieve_skill(self, skill_id: str) -> Skill:
        """Fetch a single skill by ID."""

    @abstractmethod
    def search_skills(
        self,
        query: str,
        *,
        domain_ids: Optional[list[str]] = None,
        skill_types: Optional[list[SkillType]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Skill]:
        """Free-text search across skill name, description, and aliases."""

    @abstractmethod
    def search_skills_by_domain(
        self,
        domain_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Skill]:
        """Return all skills belonging to a given domain."""

    @abstractmethod
    def get_all_skills(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[Skill]:
        """Return a paginated slice of the skill store."""

    @abstractmethod
    def validate_skill(self, skill_id: str) -> KnowledgeValidationResult:
        """Run structural and semantic validation on a skill."""

    @abstractmethod
    def find_duplicate_skills(self) -> list[list[Skill]]:
        """Return groups of skills sharing the same fingerprint."""

    @abstractmethod
    def skill_exists(self, skill_id: str) -> bool:
        """Return True if a skill with the given ID exists."""

    @abstractmethod
    def get_skill_count(self, *, active_only: bool = False) -> int:
        """Return the number of skills in the store."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return aggregate statistics for the skill store."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT KNOWLEDGE DOMAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractKnowledgeDomainEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Knowledge Domain Engine.

    Organises LUNA's knowledge into major disciplines: AI, Robotics,
    Electronics, Mathematics, Business, etc.  Domains act as the primary
    indexing namespace for all other LUNA knowledge types.
    """

    @abstractmethod
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
        """Create a new knowledge domain."""

    @abstractmethod
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
        """Apply a partial update to an existing domain."""

    @abstractmethod
    def delete_domain(self, domain_id: str, *, reason: str = "") -> KnowledgeDomain:
        """Soft-delete a domain."""

    @abstractmethod
    def retrieve_domain(self, domain_id: str) -> KnowledgeDomain:
        """Fetch a domain by ID."""

    @abstractmethod
    def search_domains(
        self,
        query: str,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 50,
    ) -> list[KnowledgeDomain]:
        """Free-text search across domain names and descriptions."""

    @abstractmethod
    def get_all_domains(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
    ) -> list[KnowledgeDomain]:
        """Return every domain record."""

    @abstractmethod
    def get_root_domains(self) -> list[KnowledgeDomain]:
        """Return top-level domains (no parent_domain_ids)."""

    @abstractmethod
    def get_child_domains(self, domain_id: str) -> list[KnowledgeDomain]:
        """Return direct child domains of the given domain."""

    @abstractmethod
    def validate_domain(self, domain_id: str) -> KnowledgeValidationResult:
        """Run structural validation on a domain record."""

    @abstractmethod
    def domain_exists(self, domain_id: str) -> bool:
        """Return True if a domain with the given ID exists."""

    @abstractmethod
    def get_domain_count(self, *, active_only: bool = False) -> int:
        """Return the number of domains in the store."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return aggregate domain store statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT PROCEDURAL KNOWLEDGE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractProceduralKnowledgeEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Procedural Knowledge Engine.

    Stores how-to knowledge: tuning a PID controller, deploying ROS2,
    training a model, building a drone.  Critical for execution planning
    by downstream POLARIS subsystems (ORION, VULCAN).
    """

    @abstractmethod
    def create_procedure(
        self,
        name: str,
        description: str,
        procedure_type: ProcedureType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        steps: Optional[list[Any]] = None,
        prerequisites: Optional[list[str]] = None,
        estimated_duration_minutes: Optional[int] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> Procedure:
        """Create and store a new procedure."""

    @abstractmethod
    def update_procedure(
        self,
        procedure_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        procedure_type: Optional[ProcedureType] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        domain_ids: Optional[list[str]] = None,
        steps: Optional[list[Any]] = None,
        prerequisites: Optional[list[str]] = None,
        estimated_duration_minutes: Optional[int] = None,
        status: Optional[KnowledgeStatus] = None,
        notes: Optional[str] = None,
        metadata: Optional[KnowledgeMetadata] = None,
    ) -> Procedure:
        """Apply a partial update to an existing procedure."""

    @abstractmethod
    def delete_procedure(self, procedure_id: str, *, reason: str = "") -> Procedure:
        """Soft-delete a procedure."""

    @abstractmethod
    def retrieve_procedure(self, procedure_id: str) -> Procedure:
        """Fetch a procedure by ID."""

    @abstractmethod
    def search_procedures(
        self,
        query: str,
        *,
        domain_ids: Optional[list[str]] = None,
        procedure_types: Optional[list[ProcedureType]] = None,
        difficulty: Optional[KnowledgeDifficulty] = None,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Procedure]:
        """Search procedures by name, description, and step content."""

    @abstractmethod
    def get_all_procedures(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[Procedure]:
        """Return a paginated slice of the procedure store."""

    @abstractmethod
    def validate_procedure(self, procedure_id: str) -> KnowledgeValidationResult:
        """Run structural validation on a procedure."""

    @abstractmethod
    def procedure_exists(self, procedure_id: str) -> bool:
        """Return True if a procedure with the given ID exists."""

    @abstractmethod
    def get_procedure_count(self, *, active_only: bool = False) -> int:
        """Return the number of procedures in the store."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return aggregate procedure store statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT RESEARCH KNOWLEDGE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractResearchKnowledgeEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Research Knowledge Engine.

    Stores research-derived information sourced from papers, technical
    reports, documentation, and experiments.  Primary consumers are
    PROMETHEUS and VULCAN.
    """

    @abstractmethod
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
        """Create and store a new research record."""

    @abstractmethod
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
        """Apply a partial update to an existing research record."""

    @abstractmethod
    def delete_research(self, research_id: str, *, reason: str = "") -> ResearchKnowledge:
        """Soft-delete a research record."""

    @abstractmethod
    def retrieve_research(self, research_id: str) -> ResearchKnowledge:
        """Fetch a research record by ID."""

    @abstractmethod
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
        """Search research records by name, abstract, and findings."""

    @abstractmethod
    def get_all_research(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[ResearchKnowledge]:
        """Return a paginated slice of all research records."""

    @abstractmethod
    def validate_research(self, research_id: str) -> KnowledgeValidationResult:
        """Run structural validation on a research record."""

    @abstractmethod
    def research_exists(self, research_id: str) -> bool:
        """Return True if a research record with the given ID exists."""

    @abstractmethod
    def get_research_count(self, *, active_only: bool = False) -> int:
        """Return the number of research records in the store."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return aggregate research store statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT EDUCATIONAL KNOWLEDGE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractEducationalKnowledgeEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Educational Knowledge Engine.

    Stores learning paths, curriculum structures, tutorials, difficulty
    levels, and prerequisites.  Primary consumer is APOLLO.
    """

    @abstractmethod
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
        """Create and store new educational content."""

    @abstractmethod
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
        """Apply a partial update to an existing educational record."""

    @abstractmethod
    def delete_educational(
        self, educational_id: str, *, reason: str = ""
    ) -> EducationalKnowledge:
        """Soft-delete an educational record."""

    @abstractmethod
    def retrieve_educational(self, educational_id: str) -> EducationalKnowledge:
        """Fetch an educational record by ID."""

    @abstractmethod
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
        """Search educational records."""

    @abstractmethod
    def get_all_educational(
        self,
        *,
        status_filter: Optional[list[KnowledgeStatus]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[EducationalKnowledge]:
        """Return a paginated slice of all educational records."""

    @abstractmethod
    def get_learning_path(
        self, domain_id: str, target_difficulty: KnowledgeDifficulty
    ) -> list[EducationalKnowledge]:
        """
        Return an ordered list of educational records forming a learning path
        for the given domain up to the target difficulty.
        """

    @abstractmethod
    def validate_educational(self, educational_id: str) -> KnowledgeValidationResult:
        """Run structural validation on an educational record."""

    @abstractmethod
    def educational_exists(self, educational_id: str) -> bool:
        """Return True if an educational record with the given ID exists."""

    @abstractmethod
    def get_educational_count(self, *, active_only: bool = False) -> int:
        """Return the number of educational records in the store."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return aggregate educational store statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT KNOWLEDGE VALIDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractKnowledgeValidationEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Knowledge Validation Engine.

    Verifies knowledge quality by checking source reliability, confidence
    scores, knowledge age, and contradictions.  Critical for reducing
    hallucinations in POLARIS reasoning.
    """

    @abstractmethod
    def validate_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> KnowledgeValidationResult:
        """
        Run a full validation pass on any knowledge record.

        Dispatches to the appropriate engine for type-specific checks then
        applies cross-cutting validation rules (source trust, age, confidence).

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """

    @abstractmethod
    def validate_batch(
        self,
        record_ids: list[str],
        knowledge_type: KnowledgeType,
    ) -> list[KnowledgeValidationResult]:
        """Validate multiple records of the same type in a single pass."""

    @abstractmethod
    def check_confidence(self, record_id: str) -> float:
        """
        Return the effective confidence score for a record (0.0 – 1.0).

        Combines metadata.confidence_score with source trust_weight.
        """

    @abstractmethod
    def detect_contradictions(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> list[Any]:
        """
        Compare a record against the rest of the knowledge store to identify
        semantic contradictions.

        Returns:
            A list of KnowledgeContradiction instances (may be empty).
        """

    @abstractmethod
    def flag_stale_records(self, max_age_days: float) -> list[str]:
        """
        Return IDs of records whose age_days exceeds max_age_days or whose
        review_date is past.
        """

    @abstractmethod
    def promote_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        new_status: KnowledgeStatus,
        new_confidence: Optional[float] = None,
    ) -> None:
        """
        Advance a record's status (e.g. DRAFT → PENDING_VALIDATION → VALIDATED
        → ACTIVE) after a successful validation pass.
        """

    @abstractmethod
    def get_validation_history(self, record_id: str) -> list[KnowledgeValidationResult]:
        """Return all past validation results for a record, newest first."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return aggregate validation statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT KNOWLEDGE RETRIEVAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractKnowledgeRetrievalEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Knowledge Retrieval Engine.

    Optimised for fast, multi-type knowledge lookup.  Provides a unified
    query surface across all knowledge types stored by LUNA.
    """

    @abstractmethod
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

    @abstractmethod
    def retrieve_many(
        self,
        record_ids: list[str],
        knowledge_type: KnowledgeType,
    ) -> list[KnowledgeRecord]:
        """Retrieve multiple records in a single call. Missing IDs are skipped."""

    @abstractmethod
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

    @abstractmethod
    def search_by_domain(
        self,
        domain_id: str,
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[KnowledgeRecord]:
        """Return all records belonging to a domain, optionally filtered by type."""

    @abstractmethod
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
            tags:      Tags to match against.
            match_all: When True, records must have ALL provided tags.
        """

    @abstractmethod
    def get_related(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        limit: int = 20,
    ) -> list[KnowledgeRecord]:
        """Return records referenced in the given record's related_ids."""

    @abstractmethod
    def get_total_count(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
        active_only: bool = False,
    ) -> int:
        """Return total record count, optionally scoped to one knowledge type."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return retrieval statistics (query counts, cache hit rate, etc.)."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT KNOWLEDGE SYNTHESIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractKnowledgeSynthesisEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Knowledge Synthesis Engine.

    Combines knowledge from multiple domains into coherent packages.
    Primary consumer: ORION (reasoning engine).

    Example synthesis:
        Robotics + AI + Control Systems + Embedded Systems
        → Autonomous Robotics Knowledge Package
    """

    @abstractmethod
    def synthesize(
        self,
        domain_ids: list[str],
        *,
        knowledge_types: Optional[list[KnowledgeType]] = None,
        min_confidence: float = 0.5,
        label: str = "",
    ) -> KnowledgeSynthesis:
        """
        Produce a synthesis record from the specified domains.

        Collects all active, high-confidence knowledge records belonging to
        the given domains, merges their semantic structures, and returns a
        KnowledgeSynthesis artefact.
        """

    @abstractmethod
    def build_package(
        self,
        synthesis_id: str,
        *,
        label: Optional[str] = None,
        notes: str = "",
    ) -> KnowledgePackage:
        """Convert a synthesis into a portable KnowledgePackage."""

    @abstractmethod
    def compose(
        self,
        package_ids: list[str],
        *,
        label: str = "",
    ) -> KnowledgeComposition:
        """Merge multiple packages into a KnowledgeComposition."""

    @abstractmethod
    def retrieve_synthesis(self, synthesis_id: str) -> KnowledgeSynthesis:
        """Fetch a synthesis record by ID."""

    @abstractmethod
    def retrieve_package(self, package_id: str) -> KnowledgePackage:
        """Fetch a package by ID."""

    @abstractmethod
    def list_syntheses(self, *, limit: int = 50) -> list[KnowledgeSynthesis]:
        """Return recent synthesis records."""

    @abstractmethod
    def list_packages(self, *, limit: int = 50) -> list[KnowledgePackage]:
        """Return recent packages."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return synthesis statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT KNOWLEDGE EVOLUTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractKnowledgeEvolutionEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Knowledge Evolution Engine.

    Manages safe, audited updates to existing knowledge.

    Can:  add facts, refine concepts, improve explanations.
    Cannot: modify identity (ASTRA domain), modify memories (ECHO domain),
            modify LUNA's own architecture.
    """

    @abstractmethod
    def propose_update(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        changes: dict[str, Any],
        *,
        rationale: str = "",
        source: str = "",
        confidence_delta: float = 0.0,
    ) -> str:
        """
        Propose an update to an existing knowledge record.

        The proposal is queued for review rather than applied immediately.

        Returns:
            A proposal ID (string).
        """

    @abstractmethod
    def apply_update(self, proposal_id: str) -> KnowledgeRecord:
        """
        Apply an approved proposal to the target record.

        Increments the record's version and bumps its metadata.updated_at.

        Raises:
            KnowledgeRetrievalError: Proposal ID not found.
            KnowledgeValidationError: Post-update state fails validation.
        """

    @abstractmethod
    def reject_update(self, proposal_id: str, *, reason: str = "") -> None:
        """Reject a pending proposal without applying it."""

    @abstractmethod
    def get_proposal(self, proposal_id: str) -> dict[str, Any]:
        """Return a proposal dict by ID."""

    @abstractmethod
    def list_pending_proposals(self) -> list[dict[str, Any]]:
        """Return all proposals in PENDING state."""

    @abstractmethod
    def get_evolution_history(self, record_id: str) -> list[dict[str, Any]]:
        """Return the ordered list of applied proposals for a record."""

    @abstractmethod
    def deprecate_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
        *,
        superseded_by: Optional[str] = None,
        reason: str = "",
    ) -> KnowledgeRecord:
        """
        Mark a record as DEPRECATED (or SUPERSEDED if superseded_by is given).
        """

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return evolution statistics (proposals, applied, rejected, etc.)."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT SKILL PROGRESSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractSkillProgressionEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Skill Progression Engine.

    Tracks how skills develop structurally: Novice → Beginner →
    Intermediate → Advanced → Expert → Master.

    Important: This engine tracks skill STRUCTURE (what mastery looks like at
    each stage), NOT individual user proficiency — that is ASTRA's domain.
    """

    @abstractmethod
    def create_progression_model(
        self,
        skill_id: str,
        stages: list[SkillStage],
        *,
        notes: str = "",
    ) -> SkillProgressionModel:
        """
        Define the progression model for a skill.

        Args:
            skill_id: The LUNA skill this model describes.
            stages:   Ordered list of SkillStage records (Novice → Master).

        Raises:
            LunaNotInitializedError: Engine not initialized.
            SkillProgressionError:   Stages are invalid or out of order.
        """

    @abstractmethod
    def update_progression_model(
        self,
        model_id: str,
        *,
        stages: Optional[list[SkillStage]] = None,
        notes: Optional[str] = None,
    ) -> SkillProgressionModel:
        """Update an existing progression model."""

    @abstractmethod
    def delete_progression_model(self, model_id: str) -> SkillProgressionModel:
        """Remove a progression model."""

    @abstractmethod
    def retrieve_progression_model(self, model_id: str) -> SkillProgressionModel:
        """Fetch a progression model by ID."""

    @abstractmethod
    def get_model_for_skill(self, skill_id: str) -> Optional[SkillProgressionModel]:
        """Return the progression model for a skill, or None if not defined."""

    @abstractmethod
    def get_stage(
        self,
        model_id: str,
        level: SkillLevel,
    ) -> Optional[SkillStage]:
        """Return the SkillStage matching a specific SkillLevel within a model."""

    @abstractmethod
    def add_prerequisite(
        self,
        model_id: str,
        prerequisite: SkillPrerequisite,
    ) -> SkillProgressionModel:
        """Add a prerequisite skill to a progression model."""

    @abstractmethod
    def remove_prerequisite(
        self,
        model_id: str,
        prerequisite_skill_id: str,
    ) -> SkillProgressionModel:
        """Remove a prerequisite from a progression model."""

    @abstractmethod
    def list_all_models(self) -> list[SkillProgressionModel]:
        """Return all progression models."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return skill progression store statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT KNOWLEDGE INTEGRITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractKnowledgeIntegrityEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Knowledge Integrity Engine.

    Critical subsystem: protects knowledge consistency across the entire
    LUNA store.

    Prevents:
        - Duplicate concepts, facts, and skills
        - Conflicting facts within the same domain
        - Corrupted or incomplete knowledge records
        - Broken cross-references between records
    """

    @abstractmethod
    def full_scan(self) -> IntegrityReport:
        """
        Run a comprehensive integrity scan across all LUNA knowledge stores.

        Checks:
            - Cross-type duplicate detection (fingerprint collision)
            - Broken reference IDs (domain_ids, related_ids, fact_ids, etc.)
            - Confidence scores out of range [0.0, 1.0]
            - Terminal-status records with live referrers
            - Empty mandatory fields (name, statement/description)

        Returns:
            An IntegrityReport documenting all discovered issues.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """

    @abstractmethod
    def scan_type(
        self,
        knowledge_type: KnowledgeType,
    ) -> IntegrityReport:
        """
        Run an integrity scan restricted to one knowledge type.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """

    @abstractmethod
    def check_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> IntegrityReport:
        """
        Check a single record for integrity issues.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """

    @abstractmethod
    def find_broken_references(self) -> list[dict[str, Any]]:
        """
        Return all records whose reference IDs point to non-existent records.

        Each entry in the returned list is a dict with keys:
            record_id, knowledge_type, field, broken_ref_id
        """

    @abstractmethod
    def find_duplicate_fingerprints(
        self,
        *,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> list[list[str]]:
        """
        Return groups of record IDs that share the same fingerprint hash.

        Args:
            knowledge_type: If provided, restrict scan to one type.
        """

    @abstractmethod
    def resolve_issue(
        self,
        issue_id: str,
        *,
        resolution_notes: str = "",
    ) -> None:
        """
        Mark an integrity issue as manually resolved.

        Raises:
            KnowledgeIntegrityError: Issue ID not found.
        """

    @abstractmethod
    def full_audit(self) -> KnowledgeAuditReport:
        """
        Produce a comprehensive KnowledgeAuditReport combining integrity,
        validation, confidence, coverage, and health scoring.
        """

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return aggregate integrity engine statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT KNOWLEDGE INDEX ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractKnowledgeIndexEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Knowledge Index Engine.

    Maintains fast-access indexes for topics, domains, skills, concepts,
    research areas, and difficulty levels.  Acts as the primary lookup
    accelerator for the Retrieval Engine.
    """

    @abstractmethod
    def index_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> KnowledgeIndexEntry:
        """
        Add or update the index entry for a knowledge record.

        Raises:
            LunaNotInitializedError: Engine not initialized.
            KnowledgeIndexError:     Record not found in the originating store.
        """

    @abstractmethod
    def deindex_record(
        self,
        record_id: str,
        knowledge_type: KnowledgeType,
    ) -> None:
        """Remove the index entry for a record."""

    @abstractmethod
    def lookup_by_id(
        self,
        record_id: str,
    ) -> Optional[KnowledgeIndexEntry]:
        """Return the index entry for a record ID, or None if absent."""

    @abstractmethod
    def lookup_concept(self, concept_id: str) -> Optional[ConceptIndexEntry]:
        """Return a concept-specific index entry."""

    @abstractmethod
    def lookup_domain(self, domain_id: str) -> Optional[DomainIndexEntry]:
        """Return a domain-specific index entry."""

    @abstractmethod
    def lookup_skill(self, skill_id: str) -> Optional[SkillIndexEntry]:
        """Return a skill-specific index entry."""

    @abstractmethod
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
        Query the index with combined filters.

        Returns index entries sorted by relevance / confidence descending.
        """

    @abstractmethod
    def reindex_all(self) -> int:
        """
        Rebuild the entire index from all LUNA knowledge stores.

        Returns:
            The number of records indexed.

        Raises:
            LunaNotInitializedError: Engine not initialized.
        """

    @abstractmethod
    def get_index_size(self) -> int:
        """Return the total number of index entries."""

    @abstractmethod
    def get_domains_in_index(self) -> list[str]:
        """Return a deduplicated list of all domain IDs present in the index."""

    @abstractmethod
    def get_tags_in_index(self) -> list[str]:
        """Return a deduplicated list of all tags present in the index."""

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return index statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT SEMANTIC STRUCTURE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AbstractSemanticStructureEngine(LunaEngineLifecycle):
    """
    Contract for the LUNA Semantic Structure Engine.

    Builds and manages semantic organisation — hierarchical trees of
    knowledge that provide logical structure for POLARIS reasoning.

    Example structure:
        Control Systems
        ├── PID
        ├── State Space
        ├── Feedback
        └── Stability
    """

    @abstractmethod
    def create_hierarchy(
        self,
        name: str,
        description: str,
        root_domain_id: str,
        metadata: KnowledgeMetadata,
        notes: str = "",
    ) -> SemanticHierarchy:
        """
        Create a new semantic hierarchy rooted at a knowledge domain.

        Raises:
            LunaNotInitializedError:   Engine not initialized.
            SemanticStructureError:    Root domain ID does not exist.
        """

    @abstractmethod
    def update_hierarchy(
        self,
        hierarchy_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> SemanticHierarchy:
        """Apply a partial update to an existing hierarchy."""

    @abstractmethod
    def delete_hierarchy(self, hierarchy_id: str) -> SemanticHierarchy:
        """Remove a semantic hierarchy and all its nodes."""

    @abstractmethod
    def retrieve_hierarchy(self, hierarchy_id: str) -> SemanticHierarchy:
        """Fetch a hierarchy by ID."""

    @abstractmethod
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
        Add a knowledge record as a node in the hierarchy.

        Args:
            hierarchy_id:  Target hierarchy.
            record_id:     The knowledge record to attach.
            knowledge_type: The type of the record.
            parent_node_id: Parent node in the tree; None for root nodes.
            label:         Display label override (defaults to record name).
            position:      Sort order among siblings.

        Raises:
            LunaNotInitializedError:   Engine not initialized.
            SemanticStructureError:    Hierarchy or parent node not found.
            RelationshipCycleError:    Adding this node would create a cycle.
        """

    @abstractmethod
    def remove_node(self, hierarchy_id: str, node_id: str) -> None:
        """Remove a node from the hierarchy. Child nodes are re-parented to the node's parent."""

    @abstractmethod
    def get_node(self, hierarchy_id: str, node_id: str) -> SemanticNode:
        """Fetch a specific node within a hierarchy."""

    @abstractmethod
    def get_children_nodes(
        self, hierarchy_id: str, node_id: str
    ) -> list[SemanticNode]:
        """Return direct child nodes of the given node."""

    @abstractmethod
    def get_root_nodes(self, hierarchy_id: str) -> list[SemanticNode]:
        """Return root-level nodes (no parent) of the given hierarchy."""

    @abstractmethod
    def get_full_tree(self, hierarchy_id: str) -> dict[str, Any]:
        """
        Return a nested dict representation of the entire hierarchy tree.

        Schema:
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
        """

    @abstractmethod
    def build_domain_structure(self, domain_id: str) -> DomainStructure:
        """
        Compute and return the full DomainStructure for a knowledge domain.

        Aggregates all hierarchies whose root is the given domain plus any
        concept sub-trees registered under it.
        """

    @abstractmethod
    def list_hierarchies(
        self,
        *,
        domain_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[SemanticHierarchy]:
        """Return all hierarchies, optionally filtered by root domain."""

    @abstractmethod
    def get_hierarchy_depth(self, hierarchy_id: str) -> int:
        """Return the maximum depth of the hierarchy tree."""

    @abstractmethod
    def detect_cycles(self, hierarchy_id: str) -> list[list[str]]:
        """
        Scan the hierarchy for cyclic parent-child relationships.

        Returns:
            A list of cycles, where each cycle is an ordered list of node IDs.
        """

    @abstractmethod
    def audit_report(self) -> dict[str, Any]:
        """Return semantic structure statistics."""


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Lifecycle mixin
    "LunaEngineLifecycle",
    # Engine contracts
    "AbstractFactEngine",
    "AbstractConceptEngine",
    "AbstractSkillEngine",
    "AbstractKnowledgeDomainEngine",
    "AbstractProceduralKnowledgeEngine",
    "AbstractResearchKnowledgeEngine",
    "AbstractEducationalKnowledgeEngine",
    "AbstractKnowledgeValidationEngine",
    "AbstractKnowledgeRetrievalEngine",
    "AbstractKnowledgeSynthesisEngine",
    "AbstractKnowledgeEvolutionEngine",
    "AbstractSkillProgressionEngine",
    "AbstractKnowledgeIntegrityEngine",
    "AbstractKnowledgeIndexEngine",
    "AbstractSemanticStructureEngine",
]