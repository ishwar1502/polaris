"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/schemas.py

Lossless serialization and deserialization support for all LUNA domain models.

Provides:
    - to_dict()  / from_dict()  for every LUNA model class
    - Schema utility helpers: serialize_model, deserialize_model,
      serialize_collection, deserialize_collection, validate_schema_version

Serialization rules:
    - Enum      → string (enum.value)
    - datetime  → ISO-8601 string (timezone-aware; UTC assumed on deserialize)
    - UUID      → string
    - tuple     → list  (round-trips back to tuple on frozenfields)
    - Optional  → None preserved
    - Nested models recursively serialized / deserialized
    - All information preserved; output is deterministic

Schema versioning:
    SCHEMA_VERSION is embedded in every top-level serialized payload under
    the key "_schema_version".  validate_schema_version() checks compatibility.

Part of the POLARIS Cognitive Substrate:
    ASTRA        → Identity
    ECHO         → Experience
    LUNA         → Knowledge       ← this module
    CHRONOS      → Time
    CONSTELLATION → Relationships
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, TypeVar, Type

from subsystems.luna.models import (
    # Enums
    KnowledgeType,
    FactType,
    ConceptType,
    SkillType,
    ProcedureType,
    ResearchType,
    EducationType,
    KnowledgeDifficulty,
    KnowledgeStatus,
    KnowledgeSourceType,
    ValidationStatus,
    ConfidenceLevel,
    SkillLevel,
    ConceptRelationshipType,
    ValidationIssueType,
    ValidationSeverity,
    IntegrityIssueType,
    # Metadata
    KnowledgeMetadata,
    # Core domain models
    KnowledgeRecord,
    Fact,
    Concept,
    Skill,
    KnowledgeDomain,
    Procedure,
    ProcedureStep,
    ResearchKnowledge,
    EducationalKnowledge,
    ContentSection,
    # Structural models
    ConceptRelationship,
    KnowledgeReference,
    KnowledgeDependency,
    SemanticNode,
    SemanticHierarchy,
    DomainStructure,
    # Skill models
    SkillStage,
    SkillProgressionModel,
    SkillPrerequisite,
    # Validation models
    ValidationIssue,
    KnowledgeValidationResult,
    KnowledgeConfidence,
    KnowledgeContradiction,
    KnowledgeEvidence,
    # Synthesis models
    KnowledgeSynthesis,
    KnowledgePackage,
    KnowledgeComposition,
    # Indexing models
    KnowledgeIndexEntry,
    ConceptIndexEntry,
    DomainIndexEntry,
    SkillIndexEntry,
    # Integrity models
    IntegrityIssue,
    IntegrityReport,
    KnowledgeAuditReport,
)
from subsystems.luna.exceptions import LunaError

# ─────────────────────────────────────────────────────────────────────────────
# VERSIONING
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION: str = "5.0.0"
"""
Current schema version.  Increment the MAJOR component on breaking changes,
MINOR on backward-compatible additions, PATCH on purely cosmetic changes.
"""

_COMPATIBLE_SCHEMA_PREFIXES: tuple[str, ...] = ("5.",)
"""
Any serialized payload whose ``_schema_version`` starts with one of these
prefixes is considered compatible with this module.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA ERRORS
# ─────────────────────────────────────────────────────────────────────────────

class SchemaError(LunaError):
    """Raised when serialization or deserialization fails."""


class SchemaVersionError(SchemaError):
    """Raised when a payload's schema version is incompatible."""


class SchemaMissingFieldError(SchemaError):
    """Raised when a required field is absent from a serialized payload."""

    def __init__(self, model: str, field: str, data: dict[str, Any]) -> None:
        super().__init__(
            f"Required field '{field}' missing from {model} payload",
            context={"model": model, "field": field, "keys_present": list(data.keys())},
        )


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _dt_to_iso(dt: datetime) -> str:
    """Serialize a datetime to an ISO-8601 string.  Always UTC."""
    return dt.isoformat()


def _iso_to_dt(value: str) -> datetime:
    """
    Deserialize an ISO-8601 string to a timezone-aware datetime.
    If no timezone info is present the value is assumed to be UTC.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _opt_dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    return _dt_to_iso(dt) if dt is not None else None


def _opt_iso_to_dt(value: Optional[str]) -> Optional[datetime]:
    return _iso_to_dt(value) if value is not None else None


def _require(data: dict[str, Any], field: str, model: str) -> Any:
    """Return data[field] or raise SchemaMissingFieldError."""
    if field not in data:
        raise SchemaMissingFieldError(model=model, field=field, data=data)
    return data[field]


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA UTILITIES  (public API)
# ─────────────────────────────────────────────────────────────────────────────

_T = TypeVar("_T")

# Registry maps a model class → its from_dict function.
# Populated at the bottom of this module after all schemas are defined.
_DESERIALIZER_REGISTRY: dict[type, Any] = {}


def serialize_model(model: Any, *, include_schema_version: bool = True) -> dict[str, Any]:
    """
    Serialize any registered LUNA model to a plain dictionary.

    Delegates to the model's own ``to_dict()`` method and optionally injects
    the ``_schema_version`` key for storage/transport payloads.

    Args:
        model:                  A LUNA domain model instance.
        include_schema_version: When True (default), the key
                                ``_schema_version`` is added to the output.

    Returns:
        A serializable plain dictionary.

    Raises:
        SchemaError: If the model has no ``to_dict()`` method.
    """
    if not hasattr(model, "to_dict"):
        raise SchemaError(
            f"Model '{type(model).__name__}' does not implement to_dict()",
            context={"model_type": type(model).__name__},
        )
    data = model.to_dict()
    if include_schema_version:
        data["_schema_version"] = SCHEMA_VERSION
    return data


def deserialize_model(model_class: Type[_T], data: dict[str, Any]) -> _T:
    """
    Deserialize a plain dictionary back into the specified LUNA model class.

    Looks up the registered ``from_dict`` function for *model_class* and
    delegates to it.

    Args:
        model_class:    The target LUNA model class.
        data:           The serialized dictionary, as produced by
                        ``serialize_model`` or a model's own ``to_dict()``.

    Returns:
        A fully reconstructed instance of *model_class*.

    Raises:
        SchemaError:         If no deserializer is registered for *model_class*.
        SchemaMissingFieldError: If a required field is absent from *data*.
    """
    deserializer = _DESERIALIZER_REGISTRY.get(model_class)
    if deserializer is None:
        raise SchemaError(
            f"No deserializer registered for '{model_class.__name__}'",
            context={"model_class": model_class.__name__},
        )
    return deserializer(data)


def serialize_collection(
    models: list[Any],
    *,
    include_schema_version: bool = True,
) -> list[dict[str, Any]]:
    """
    Serialize a list of LUNA model instances to a list of plain dictionaries.

    Args:
        models:                 List of LUNA domain model instances.
        include_schema_version: Injected into each item when True (default).

    Returns:
        List of serialized dictionaries, one per model.
    """
    return [serialize_model(m, include_schema_version=include_schema_version) for m in models]


def deserialize_collection(
    model_class: Type[_T],
    data: list[dict[str, Any]],
) -> list[_T]:
    """
    Deserialize a list of plain dictionaries into the specified LUNA model class.

    Args:
        model_class:    The target LUNA model class for every item.
        data:           List of serialized dictionaries.

    Returns:
        List of fully reconstructed *model_class* instances.
    """
    return [deserialize_model(model_class, item) for item in data]


def validate_schema_version(data: dict[str, Any]) -> str:
    """
    Check that *data* carries a ``_schema_version`` key compatible with this
    module's ``SCHEMA_VERSION``.

    Compatibility is defined by ``_COMPATIBLE_SCHEMA_PREFIXES``: a stored
    version is accepted if it starts with any entry in that tuple.

    Args:
        data:   A serialized dictionary that may contain ``_schema_version``.

    Returns:
        The stored schema version string.

    Raises:
        SchemaVersionError: If the key is absent or the version is incompatible.
    """
    stored = data.get("_schema_version")
    if stored is None:
        raise SchemaVersionError(
            "Payload is missing '_schema_version' key; cannot verify compatibility",
            context={"current_version": SCHEMA_VERSION},
        )
    if not any(str(stored).startswith(prefix) for prefix in _COMPATIBLE_SCHEMA_PREFIXES):
        raise SchemaVersionError(
            f"Schema version '{stored}' is incompatible with current version '{SCHEMA_VERSION}'",
            context={
                "stored_version": stored,
                "current_version": SCHEMA_VERSION,
                "compatible_prefixes": list(_COMPATIBLE_SCHEMA_PREFIXES),
            },
        )
    return str(stored)


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeMetadata
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_metadata_to_dict(meta: KnowledgeMetadata) -> dict[str, Any]:
    """Serialize a KnowledgeMetadata instance to a plain dictionary."""
    return meta.to_dict()


def knowledge_metadata_from_dict(data: dict[str, Any]) -> KnowledgeMetadata:
    """
    Deserialize a plain dictionary into a KnowledgeMetadata instance.

    All fields are restored verbatim; derived properties recompute on access.
    """
    m = "KnowledgeMetadata"
    return KnowledgeMetadata(
        source=_require(data, "source", m),
        source_type=KnowledgeSourceType(_require(data, "source_type", m)),
        confidence_score=_require(data, "confidence_score", m),
        validation_status=ValidationStatus(_require(data, "validation_status", m)),
        created_at=_iso_to_dt(_require(data, "created_at", m)),
        updated_at=_iso_to_dt(_require(data, "updated_at", m)),
        version=_require(data, "version", m),
        tags=tuple(_require(data, "tags", m)),
        references=tuple(_require(data, "references", m)),
        author=data.get("author"),
        language=data.get("language", "en"),
        review_date=_opt_iso_to_dt(data.get("review_date")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeRecord  (base)
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_record_to_dict(record: KnowledgeRecord) -> dict[str, Any]:
    """Serialize a KnowledgeRecord base instance."""
    return record.to_dict()


def knowledge_record_from_dict(data: dict[str, Any]) -> KnowledgeRecord:
    """
    Deserialize a plain dictionary into a KnowledgeRecord base instance.

    For polymorphic payloads (Fact, Concept, Skill, …), prefer the
    dedicated ``from_dict`` functions or ``knowledge_record_from_dict_polymorphic``.
    """
    m = "KnowledgeRecord"
    return KnowledgeRecord(
        id=_require(data, "id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        status=KnowledgeStatus(_require(data, "status", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        domain_ids=list(_require(data, "domain_ids", m)),
        metadata=knowledge_metadata_from_dict(_require(data, "metadata", m)),
        aliases=list(data.get("aliases", [])),
        related_ids=list(data.get("related_ids", [])),
        notes=data.get("notes", ""),
    )


def knowledge_record_from_dict_polymorphic(data: dict[str, Any]) -> KnowledgeRecord:
    """
    Inspect ``knowledge_type`` in *data* and dispatch to the correct subclass
    ``from_dict`` function.

    Raises:
        SchemaError: If the ``knowledge_type`` value is unrecognised.
    """
    m = "KnowledgeRecord (polymorphic)"
    raw_type = _require(data, "knowledge_type", m)
    try:
        kt = KnowledgeType(raw_type)
    except ValueError as exc:
        raise SchemaError(
            f"Unknown knowledge_type '{raw_type}'",
            context={"knowledge_type": raw_type},
            cause=exc,
        ) from exc

    _dispatch: dict[KnowledgeType, Any] = {
        KnowledgeType.FACT: fact_from_dict,
        KnowledgeType.CONCEPT: concept_from_dict,
        KnowledgeType.SKILL: skill_from_dict,
        KnowledgeType.DOMAIN: knowledge_domain_from_dict,
        KnowledgeType.PROCEDURE: procedure_from_dict,
        KnowledgeType.RESEARCH: research_knowledge_from_dict,
        KnowledgeType.EDUCATIONAL: educational_knowledge_from_dict,
    }
    handler = _dispatch.get(kt)
    if handler is None:
        # Fallback to base record for COMPOSITE / SEMANTIC_STRUCTURE / unknown
        return knowledge_record_from_dict(data)
    return handler(data)


# ─────────────────────────────────────────────────────────────────────────────
# Fact
# ─────────────────────────────────────────────────────────────────────────────

def fact_to_dict(fact: Fact) -> dict[str, Any]:
    """Serialize a Fact instance."""
    return fact.to_dict()


def fact_from_dict(data: dict[str, Any]) -> Fact:
    """Deserialize a plain dictionary into a Fact instance."""
    m = "Fact"
    return Fact(
        id=_require(data, "id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        status=KnowledgeStatus(_require(data, "status", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        domain_ids=list(_require(data, "domain_ids", m)),
        metadata=knowledge_metadata_from_dict(_require(data, "metadata", m)),
        aliases=list(data.get("aliases", [])),
        related_ids=list(data.get("related_ids", [])),
        notes=data.get("notes", ""),
        fact_type=FactType(_require(data, "fact_type", m)),
        statement=data.get("statement", ""),
        formal_notation=data.get("formal_notation"),
        units=data.get("units"),
        conditions=list(data.get("conditions", [])),
        counterexamples=list(data.get("counterexamples", [])),
        supporting_fact_ids=list(data.get("supporting_fact_ids", [])),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Concept
# ─────────────────────────────────────────────────────────────────────────────

def concept_to_dict(concept: Concept) -> dict[str, Any]:
    """Serialize a Concept instance."""
    return concept.to_dict()


def concept_from_dict(data: dict[str, Any]) -> Concept:
    """Deserialize a plain dictionary into a Concept instance."""
    m = "Concept"
    return Concept(
        id=_require(data, "id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        status=KnowledgeStatus(_require(data, "status", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        domain_ids=list(_require(data, "domain_ids", m)),
        metadata=knowledge_metadata_from_dict(_require(data, "metadata", m)),
        aliases=list(data.get("aliases", [])),
        related_ids=list(data.get("related_ids", [])),
        notes=data.get("notes", ""),
        concept_type=ConceptType(_require(data, "concept_type", m)),
        core_ideas=list(data.get("core_ideas", [])),
        applications=list(data.get("applications", [])),
        prerequisite_concept_ids=list(data.get("prerequisite_concept_ids", [])),
        child_concept_ids=list(data.get("child_concept_ids", [])),
        fact_ids=list(data.get("fact_ids", [])),
        is_foundational=bool(data.get("is_foundational", False)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Skill
# ─────────────────────────────────────────────────────────────────────────────

def skill_to_dict(skill: Skill) -> dict[str, Any]:
    """Serialize a Skill instance."""
    return skill.to_dict()


def skill_from_dict(data: dict[str, Any]) -> Skill:
    """Deserialize a plain dictionary into a Skill instance."""
    m = "Skill"
    return Skill(
        id=_require(data, "id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        status=KnowledgeStatus(_require(data, "status", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        domain_ids=list(_require(data, "domain_ids", m)),
        metadata=knowledge_metadata_from_dict(_require(data, "metadata", m)),
        aliases=list(data.get("aliases", [])),
        related_ids=list(data.get("related_ids", [])),
        notes=data.get("notes", ""),
        skill_type=SkillType(_require(data, "skill_type", m)),
        capability_description=data.get("capability_description", ""),
        required_tools=list(data.get("required_tools", [])),
        required_concept_ids=list(data.get("required_concept_ids", [])),
        required_fact_ids=list(data.get("required_fact_ids", [])),
        prerequisite_skill_ids=list(data.get("prerequisite_skill_ids", [])),
        sub_skill_ids=list(data.get("sub_skill_ids", [])),
        progression_model_id=data.get("progression_model_id"),
        practical_exercises=list(data.get("practical_exercises", [])),
        assessment_criteria=list(data.get("assessment_criteria", [])),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeDomain
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_domain_to_dict(domain: KnowledgeDomain) -> dict[str, Any]:
    """Serialize a KnowledgeDomain instance."""
    return domain.to_dict()


def knowledge_domain_from_dict(data: dict[str, Any]) -> KnowledgeDomain:
    """Deserialize a plain dictionary into a KnowledgeDomain instance."""
    m = "KnowledgeDomain"
    return KnowledgeDomain(
        id=_require(data, "id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        status=KnowledgeStatus(_require(data, "status", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        domain_ids=list(_require(data, "domain_ids", m)),
        metadata=knowledge_metadata_from_dict(_require(data, "metadata", m)),
        aliases=list(data.get("aliases", [])),
        related_ids=list(data.get("related_ids", [])),
        notes=data.get("notes", ""),
        parent_domain_id=data.get("parent_domain_id"),
        sub_domain_ids=list(data.get("sub_domain_ids", [])),
        core_concept_ids=list(data.get("core_concept_ids", [])),
        core_skill_ids=list(data.get("core_skill_ids", [])),
        core_fact_ids=list(data.get("core_fact_ids", [])),
        standard_references=list(data.get("standard_references", [])),
        is_root_domain=bool(data.get("is_root_domain", False)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ProcedureStep
# ─────────────────────────────────────────────────────────────────────────────

def procedure_step_to_dict(step: ProcedureStep) -> dict[str, Any]:
    """Serialize a ProcedureStep instance."""
    return step.to_dict()


def procedure_step_from_dict(data: dict[str, Any]) -> ProcedureStep:
    """Deserialize a plain dictionary into a ProcedureStep instance."""
    m = "ProcedureStep"
    return ProcedureStep(
        step_number=_require(data, "step_number", m),
        title=_require(data, "title", m),
        instruction=_require(data, "instruction", m),
        sub_steps=tuple(data.get("sub_steps", [])),
        warnings=tuple(data.get("warnings", [])),
        expected_outcome=data.get("expected_outcome", ""),
        is_critical=bool(data.get("is_critical", False)),
        estimated_minutes=data.get("estimated_minutes"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Procedure
# ─────────────────────────────────────────────────────────────────────────────

def procedure_to_dict(procedure: Procedure) -> dict[str, Any]:
    """Serialize a Procedure instance."""
    return procedure.to_dict()


def procedure_from_dict(data: dict[str, Any]) -> Procedure:
    """Deserialize a plain dictionary into a Procedure instance."""
    m = "Procedure"
    return Procedure(
        id=_require(data, "id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        status=KnowledgeStatus(_require(data, "status", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        domain_ids=list(_require(data, "domain_ids", m)),
        metadata=knowledge_metadata_from_dict(_require(data, "metadata", m)),
        aliases=list(data.get("aliases", [])),
        related_ids=list(data.get("related_ids", [])),
        notes=data.get("notes", ""),
        procedure_type=ProcedureType(_require(data, "procedure_type", m)),
        goal=data.get("goal", ""),
        steps=[procedure_step_from_dict(s) for s in data.get("steps", [])],
        required_skill_ids=list(data.get("required_skill_ids", [])),
        required_concept_ids=list(data.get("required_concept_ids", [])),
        required_tools=list(data.get("required_tools", [])),
        required_materials=list(data.get("required_materials", [])),
        preconditions=list(data.get("preconditions", [])),
        postconditions=list(data.get("postconditions", [])),
        common_pitfalls=list(data.get("common_pitfalls", [])),
        expected_duration_minutes=data.get("expected_duration_minutes"),
        is_reversible=bool(data.get("is_reversible", True)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ResearchKnowledge
# ─────────────────────────────────────────────────────────────────────────────

def research_knowledge_to_dict(research: ResearchKnowledge) -> dict[str, Any]:
    """Serialize a ResearchKnowledge instance."""
    return research.to_dict()


def research_knowledge_from_dict(data: dict[str, Any]) -> ResearchKnowledge:
    """Deserialize a plain dictionary into a ResearchKnowledge instance."""
    m = "ResearchKnowledge"
    return ResearchKnowledge(
        id=_require(data, "id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        status=KnowledgeStatus(_require(data, "status", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        domain_ids=list(_require(data, "domain_ids", m)),
        metadata=knowledge_metadata_from_dict(_require(data, "metadata", m)),
        aliases=list(data.get("aliases", [])),
        related_ids=list(data.get("related_ids", [])),
        notes=data.get("notes", ""),
        research_type=ResearchType(_require(data, "research_type", m)),
        title=data.get("title", ""),
        abstract=data.get("abstract", ""),
        authors=list(data.get("authors", [])),
        publication_venue=data.get("publication_venue"),
        publication_date=_opt_iso_to_dt(data.get("publication_date")),
        doi=data.get("doi"),
        url=data.get("url"),
        key_findings=list(data.get("key_findings", [])),
        methodology=data.get("methodology", ""),
        limitations=list(data.get("limitations", [])),
        cited_knowledge_ids=list(data.get("cited_knowledge_ids", [])),
        extracted_fact_ids=list(data.get("extracted_fact_ids", [])),
        extracted_concept_ids=list(data.get("extracted_concept_ids", [])),
        citation_count=int(data.get("citation_count", 0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ContentSection
# ─────────────────────────────────────────────────────────────────────────────

def content_section_to_dict(section: ContentSection) -> dict[str, Any]:
    """Serialize a ContentSection instance."""
    return section.to_dict()


def content_section_from_dict(data: dict[str, Any]) -> ContentSection:
    """Deserialize a plain dictionary into a ContentSection instance."""
    m = "ContentSection"
    return ContentSection(
        section_number=_require(data, "section_number", m),
        title=_require(data, "title", m),
        summary=_require(data, "summary", m),
        knowledge_ids=tuple(data.get("knowledge_ids", [])),
        duration_minutes=data.get("duration_minutes"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EducationalKnowledge
# ─────────────────────────────────────────────────────────────────────────────

def educational_knowledge_to_dict(edu: EducationalKnowledge) -> dict[str, Any]:
    """Serialize an EducationalKnowledge instance."""
    return edu.to_dict()


def educational_knowledge_from_dict(data: dict[str, Any]) -> EducationalKnowledge:
    """Deserialize a plain dictionary into an EducationalKnowledge instance."""
    m = "EducationalKnowledge"
    return EducationalKnowledge(
        id=_require(data, "id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        status=KnowledgeStatus(_require(data, "status", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        domain_ids=list(_require(data, "domain_ids", m)),
        metadata=knowledge_metadata_from_dict(_require(data, "metadata", m)),
        aliases=list(data.get("aliases", [])),
        related_ids=list(data.get("related_ids", [])),
        notes=data.get("notes", ""),
        education_type=EducationType(_require(data, "education_type", m)),
        learning_objectives=list(data.get("learning_objectives", [])),
        prerequisite_knowledge_ids=list(data.get("prerequisite_knowledge_ids", [])),
        target_skill_ids=list(data.get("target_skill_ids", [])),
        target_concept_ids=list(data.get("target_concept_ids", [])),
        estimated_duration_hours=data.get("estimated_duration_hours"),
        assessment_type=data.get("assessment_type"),
        learning_outcomes=list(data.get("learning_outcomes", [])),
        content_sections=[content_section_from_dict(s) for s in data.get("content_sections", [])],
        is_self_contained=bool(data.get("is_self_contained", False)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ConceptRelationship
# ─────────────────────────────────────────────────────────────────────────────

def concept_relationship_to_dict(rel: ConceptRelationship) -> dict[str, Any]:
    """Serialize a ConceptRelationship instance."""
    return rel.to_dict()


def concept_relationship_from_dict(data: dict[str, Any]) -> ConceptRelationship:
    """Deserialize a plain dictionary into a ConceptRelationship instance."""
    m = "ConceptRelationship"
    return ConceptRelationship(
        id=_require(data, "id", m),
        source_concept_id=_require(data, "source_concept_id", m),
        target_concept_id=_require(data, "target_concept_id", m),
        relationship_type=ConceptRelationshipType(_require(data, "relationship_type", m)),
        description=_require(data, "description", m),
        weight=float(data.get("weight", 1.0)),
        is_bidirectional=bool(data.get("is_bidirectional", False)),
        created_at=_iso_to_dt(_require(data, "created_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeReference
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_reference_to_dict(ref: KnowledgeReference) -> dict[str, Any]:
    """Serialize a KnowledgeReference instance."""
    return ref.to_dict()


def knowledge_reference_from_dict(data: dict[str, Any]) -> KnowledgeReference:
    """Deserialize a plain dictionary into a KnowledgeReference instance."""
    m = "KnowledgeReference"
    return KnowledgeReference(
        id=_require(data, "id", m),
        source_id=_require(data, "source_id", m),
        target=_require(data, "target", m),
        reference_type=_require(data, "reference_type", m),
        description=data.get("description", ""),
        created_at=_iso_to_dt(_require(data, "created_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeDependency
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_dependency_to_dict(dep: KnowledgeDependency) -> dict[str, Any]:
    """Serialize a KnowledgeDependency instance."""
    return dep.to_dict()


def knowledge_dependency_from_dict(data: dict[str, Any]) -> KnowledgeDependency:
    """Deserialize a plain dictionary into a KnowledgeDependency instance."""
    m = "KnowledgeDependency"
    return KnowledgeDependency(
        id=_require(data, "id", m),
        dependent_id=_require(data, "dependent_id", m),
        dependency_id=_require(data, "dependency_id", m),
        dependency_type=data.get("dependency_type", "requires"),
        is_hard=bool(data.get("is_hard", True)),
        description=data.get("description", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SemanticNode
# ─────────────────────────────────────────────────────────────────────────────

def semantic_node_to_dict(node: SemanticNode) -> dict[str, Any]:
    """Serialize a SemanticNode instance."""
    return node.to_dict()


def semantic_node_from_dict(data: dict[str, Any]) -> SemanticNode:
    """Deserialize a plain dictionary into a SemanticNode instance."""
    m = "SemanticNode"
    return SemanticNode(
        id=_require(data, "id", m),
        knowledge_id=_require(data, "knowledge_id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        label=_require(data, "label", m),
        parent_node_id=data.get("parent_node_id"),
        child_node_ids=list(data.get("child_node_ids", [])),
        depth=int(data.get("depth", 0)),
        weight=float(data.get("weight", 1.0)),
        annotations=dict(data.get("annotations", {})),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SemanticHierarchy
# ─────────────────────────────────────────────────────────────────────────────

def semantic_hierarchy_to_dict(hierarchy: SemanticHierarchy) -> dict[str, Any]:
    """Serialize a SemanticHierarchy instance (includes all nodes)."""
    return hierarchy.to_dict()


def semantic_hierarchy_from_dict(data: dict[str, Any]) -> SemanticHierarchy:
    """Deserialize a plain dictionary into a SemanticHierarchy instance."""
    m = "SemanticHierarchy"
    raw_nodes: dict[str, Any] = data.get("nodes", {})
    nodes: dict[str, SemanticNode] = {
        k: semantic_node_from_dict(v) for k, v in raw_nodes.items()
    }
    return SemanticHierarchy(
        id=_require(data, "id", m),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        root_node_id=_require(data, "root_node_id", m),
        nodes=nodes,
        domain_id=data.get("domain_id"),
        created_at=_iso_to_dt(_require(data, "created_at", m)),
        updated_at=_iso_to_dt(_require(data, "updated_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DomainStructure
# ─────────────────────────────────────────────────────────────────────────────

def domain_structure_to_dict(structure: DomainStructure) -> dict[str, Any]:
    """Serialize a DomainStructure instance."""
    return structure.to_dict()


def domain_structure_from_dict(data: dict[str, Any]) -> DomainStructure:
    """Deserialize a plain dictionary into a DomainStructure instance."""
    m = "DomainStructure"
    return DomainStructure(
        id=_require(data, "id", m),
        domain_id=_require(data, "domain_id", m),
        domain_name=_require(data, "domain_name", m),
        concept_ids=list(data.get("concept_ids", [])),
        fact_ids=list(data.get("fact_ids", [])),
        skill_ids=list(data.get("skill_ids", [])),
        procedure_ids=list(data.get("procedure_ids", [])),
        sub_domain_ids=list(data.get("sub_domain_ids", [])),
        hierarchy_ids=list(data.get("hierarchy_ids", [])),
        dependency_ids=list(data.get("dependency_ids", [])),
        total_records=int(data.get("total_records", 0)),
        last_updated=_iso_to_dt(_require(data, "last_updated", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SkillStage
# ─────────────────────────────────────────────────────────────────────────────

def skill_stage_to_dict(stage: SkillStage) -> dict[str, Any]:
    """Serialize a SkillStage instance."""
    return stage.to_dict()


def skill_stage_from_dict(data: dict[str, Any]) -> SkillStage:
    """Deserialize a plain dictionary into a SkillStage instance."""
    m = "SkillStage"
    return SkillStage(
        level=SkillLevel(_require(data, "level", m)),
        label=_require(data, "label", m),
        description=_require(data, "description", m),
        required_concept_ids=tuple(data.get("required_concept_ids", [])),
        required_fact_ids=tuple(data.get("required_fact_ids", [])),
        required_procedure_ids=tuple(data.get("required_procedure_ids", [])),
        competencies=tuple(data.get("competencies", [])),
        demonstration_tasks=tuple(data.get("demonstration_tasks", [])),
        typical_duration_hours=data.get("typical_duration_hours"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SkillProgressionModel
# ─────────────────────────────────────────────────────────────────────────────

def skill_progression_model_to_dict(model: SkillProgressionModel) -> dict[str, Any]:
    """Serialize a SkillProgressionModel instance."""
    return model.to_dict()


def skill_progression_model_from_dict(data: dict[str, Any]) -> SkillProgressionModel:
    """Deserialize a plain dictionary into a SkillProgressionModel instance."""
    m = "SkillProgressionModel"
    return SkillProgressionModel(
        id=_require(data, "id", m),
        skill_id=_require(data, "skill_id", m),
        skill_name=_require(data, "skill_name", m),
        description=_require(data, "description", m),
        stages=[skill_stage_from_dict(s) for s in data.get("stages", [])],
        transition_criteria=dict(data.get("transition_criteria", {})),
        estimated_total_hours=data.get("estimated_total_hours"),
        is_linear=bool(data.get("is_linear", True)),
        created_at=_iso_to_dt(_require(data, "created_at", m)),
        updated_at=_iso_to_dt(_require(data, "updated_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SkillPrerequisite
# ─────────────────────────────────────────────────────────────────────────────

def skill_prerequisite_to_dict(prereq: SkillPrerequisite) -> dict[str, Any]:
    """Serialize a SkillPrerequisite instance."""
    return prereq.to_dict()


def skill_prerequisite_from_dict(data: dict[str, Any]) -> SkillPrerequisite:
    """Deserialize a plain dictionary into a SkillPrerequisite instance."""
    m = "SkillPrerequisite"
    return SkillPrerequisite(
        id=_require(data, "id", m),
        skill_id=_require(data, "skill_id", m),
        prerequisite_skill_id=_require(data, "prerequisite_skill_id", m),
        minimum_level=SkillLevel(_require(data, "minimum_level", m)),
        is_mandatory=bool(data.get("is_mandatory", True)),
        rationale=data.get("rationale", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ValidationIssue
# ─────────────────────────────────────────────────────────────────────────────

def validation_issue_to_dict(issue: ValidationIssue) -> dict[str, Any]:
    """Serialize a ValidationIssue instance."""
    return issue.to_dict()


def validation_issue_from_dict(data: dict[str, Any]) -> ValidationIssue:
    """Deserialize a plain dictionary into a ValidationIssue instance."""
    m = "ValidationIssue"
    return ValidationIssue(
        id=_require(data, "id", m),
        issue_type=ValidationIssueType(_require(data, "issue_type", m)),
        severity=ValidationSeverity(_require(data, "severity", m)),
        message=_require(data, "message", m),
        field=data.get("field"),
        suggestion=data.get("suggestion", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeValidationResult
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_validation_result_to_dict(result: KnowledgeValidationResult) -> dict[str, Any]:
    """Serialize a KnowledgeValidationResult instance."""
    return result.to_dict()


def knowledge_validation_result_from_dict(data: dict[str, Any]) -> KnowledgeValidationResult:
    """Deserialize a plain dictionary into a KnowledgeValidationResult instance."""
    m = "KnowledgeValidationResult"
    return KnowledgeValidationResult(
        id=_require(data, "id", m),
        knowledge_id=_require(data, "knowledge_id", m),
        knowledge_name=_require(data, "knowledge_name", m),
        validation_status=ValidationStatus(_require(data, "validation_status", m)),
        issues=tuple(validation_issue_from_dict(i) for i in data.get("issues", [])),
        validated_at=_iso_to_dt(_require(data, "validated_at", m)),
        validator_version=_require(data, "validator_version", m),
        notes=data.get("notes", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeConfidence
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_confidence_to_dict(conf: KnowledgeConfidence) -> dict[str, Any]:
    """Serialize a KnowledgeConfidence instance."""
    return conf.to_dict()


def knowledge_confidence_from_dict(data: dict[str, Any]) -> KnowledgeConfidence:
    """Deserialize a plain dictionary into a KnowledgeConfidence instance."""
    m = "KnowledgeConfidence"
    return KnowledgeConfidence(
        id=_require(data, "id", m),
        knowledge_id=_require(data, "knowledge_id", m),
        overall_score=float(_require(data, "overall_score", m)),
        source_reliability=float(_require(data, "source_reliability", m)),
        recency_score=float(_require(data, "recency_score", m)),
        corroboration_score=float(_require(data, "corroboration_score", m)),
        consistency_score=float(_require(data, "consistency_score", m)),
        assessed_at=_iso_to_dt(_require(data, "assessed_at", m)),
        assessment_notes=data.get("assessment_notes", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeContradiction
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_contradiction_to_dict(contradiction: KnowledgeContradiction) -> dict[str, Any]:
    """Serialize a KnowledgeContradiction instance."""
    return contradiction.to_dict()


def knowledge_contradiction_from_dict(data: dict[str, Any]) -> KnowledgeContradiction:
    """Deserialize a plain dictionary into a KnowledgeContradiction instance."""
    m = "KnowledgeContradiction"
    return KnowledgeContradiction(
        id=_require(data, "id", m),
        record_a_id=_require(data, "record_a_id", m),
        record_b_id=_require(data, "record_b_id", m),
        contradiction_description=_require(data, "contradiction_description", m),
        severity=ValidationSeverity(_require(data, "severity", m)),
        detected_at=_iso_to_dt(_require(data, "detected_at", m)),
        resolution_status=data.get("resolution_status", "unresolved"),
        resolution_notes=data.get("resolution_notes", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeEvidence
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_evidence_to_dict(evidence: KnowledgeEvidence) -> dict[str, Any]:
    """Serialize a KnowledgeEvidence instance."""
    return evidence.to_dict()


def knowledge_evidence_from_dict(data: dict[str, Any]) -> KnowledgeEvidence:
    """Deserialize a plain dictionary into a KnowledgeEvidence instance."""
    m = "KnowledgeEvidence"
    return KnowledgeEvidence(
        id=_require(data, "id", m),
        knowledge_id=_require(data, "knowledge_id", m),
        evidence_type=_require(data, "evidence_type", m),
        source=_require(data, "source", m),
        source_type=KnowledgeSourceType(_require(data, "source_type", m)),
        description=_require(data, "description", m),
        confidence=float(_require(data, "confidence", m)),
        collected_at=_iso_to_dt(_require(data, "collected_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeSynthesis
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_synthesis_to_dict(synthesis: KnowledgeSynthesis) -> dict[str, Any]:
    """Serialize a KnowledgeSynthesis instance."""
    return synthesis.to_dict()


def knowledge_synthesis_from_dict(data: dict[str, Any]) -> KnowledgeSynthesis:
    """Deserialize a plain dictionary into a KnowledgeSynthesis instance."""
    m = "KnowledgeSynthesis"
    return KnowledgeSynthesis(
        id=_require(data, "id", m),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        source_domain_ids=list(_require(data, "source_domain_ids", m)),
        source_knowledge_ids=list(_require(data, "source_knowledge_ids", m)),
        synthesized_concept_ids=list(data.get("synthesized_concept_ids", [])),
        synthesized_fact_ids=list(data.get("synthesized_fact_ids", [])),
        synthesis_rationale=data.get("synthesis_rationale", ""),
        integration_points=list(data.get("integration_points", [])),
        emergent_insights=list(data.get("emergent_insights", [])),
        confidence_score=float(data.get("confidence_score", 0.70)),
        created_at=_iso_to_dt(_require(data, "created_at", m)),
        updated_at=_iso_to_dt(_require(data, "updated_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgePackage
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_package_to_dict(package: KnowledgePackage) -> dict[str, Any]:
    """Serialize a KnowledgePackage instance."""
    return package.to_dict()


def knowledge_package_from_dict(data: dict[str, Any]) -> KnowledgePackage:
    """Deserialize a plain dictionary into a KnowledgePackage instance."""
    m = "KnowledgePackage"
    return KnowledgePackage(
        id=_require(data, "id", m),
        name=_require(data, "name", m),
        description=_require(data, "description", m),
        purpose=_require(data, "purpose", m),
        target_consumer=_require(data, "target_consumer", m),
        domain_ids=list(data.get("domain_ids", [])),
        concept_ids=list(data.get("concept_ids", [])),
        fact_ids=list(data.get("fact_ids", [])),
        skill_ids=list(data.get("skill_ids", [])),
        procedure_ids=list(data.get("procedure_ids", [])),
        research_ids=list(data.get("research_ids", [])),
        educational_ids=list(data.get("educational_ids", [])),
        synthesis_id=data.get("synthesis_id"),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        estimated_size_kb=data.get("estimated_size_kb"),
        assembled_at=_iso_to_dt(_require(data, "assembled_at", m)),
        is_complete=bool(data.get("is_complete", False)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeComposition
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_composition_to_dict(composition: KnowledgeComposition) -> dict[str, Any]:
    """Serialize a KnowledgeComposition instance."""
    return composition.to_dict()


def knowledge_composition_from_dict(data: dict[str, Any]) -> KnowledgeComposition:
    """Deserialize a plain dictionary into a KnowledgeComposition instance."""
    m = "KnowledgeComposition"
    return KnowledgeComposition(
        id=_require(data, "id", m),
        package_id=_require(data, "package_id", m),
        included_ids=tuple(_require(data, "included_ids", m)),
        excluded_ids=tuple(_require(data, "excluded_ids", m)),
        inclusion_rationale=_require(data, "inclusion_rationale", m),
        exclusion_rationale=_require(data, "exclusion_rationale", m),
        composition_strategy=data.get("composition_strategy", "curated"),
        confidence_threshold=float(_require(data, "confidence_threshold", m)),
        composed_at=_iso_to_dt(_require(data, "composed_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeIndexEntry
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_index_entry_to_dict(entry: KnowledgeIndexEntry) -> dict[str, Any]:
    """Serialize a KnowledgeIndexEntry instance."""
    return entry.to_dict()


def knowledge_index_entry_from_dict(data: dict[str, Any]) -> KnowledgeIndexEntry:
    """
    Deserialize a plain dictionary into a KnowledgeIndexEntry instance.

    Note: ``name_lower`` and ``aliases_lower`` are re-derived from
    ``name`` and any aliases present in the payload to guarantee
    consistency; if they are stored explicitly they are used directly.
    """
    m = "KnowledgeIndexEntry"
    name: str = _require(data, "name", m)
    # name_lower is a derived search-optimisation field; re-derive if absent.
    name_lower: str = data.get("name_lower", name.lower())
    # aliases_lower is not emitted by to_dict() — re-derive from name_lower
    # or fall back to an empty tuple.
    aliases_lower: tuple[str, ...] = tuple(data.get("aliases_lower", []))
    return KnowledgeIndexEntry(
        id=_require(data, "id", m),
        knowledge_id=_require(data, "knowledge_id", m),
        knowledge_type=KnowledgeType(_require(data, "knowledge_type", m)),
        name=name,
        name_lower=name_lower,
        aliases_lower=aliases_lower,
        domain_ids=tuple(_require(data, "domain_ids", m)),
        tags=tuple(_require(data, "tags", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        status=KnowledgeStatus(_require(data, "status", m)),
        confidence_score=float(_require(data, "confidence_score", m)),
        fingerprint=_require(data, "fingerprint", m),
        indexed_at=_iso_to_dt(_require(data, "indexed_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ConceptIndexEntry
# ─────────────────────────────────────────────────────────────────────────────

def concept_index_entry_to_dict(entry: ConceptIndexEntry) -> dict[str, Any]:
    """Serialize a ConceptIndexEntry instance."""
    return entry.to_dict()


def concept_index_entry_from_dict(data: dict[str, Any]) -> ConceptIndexEntry:
    """Deserialize a plain dictionary into a ConceptIndexEntry instance."""
    m = "ConceptIndexEntry"
    name: str = _require(data, "name", m)
    return ConceptIndexEntry(
        id=_require(data, "id", m),
        concept_id=_require(data, "concept_id", m),
        name=name,
        name_lower=data.get("name_lower", name.lower()),
        concept_type=ConceptType(_require(data, "concept_type", m)),
        domain_ids=tuple(_require(data, "domain_ids", m)),
        prerequisite_concept_ids=tuple(_require(data, "prerequisite_concept_ids", m)),
        child_concept_ids=tuple(_require(data, "child_concept_ids", m)),
        is_foundational=bool(_require(data, "is_foundational", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        indexed_at=_iso_to_dt(_require(data, "indexed_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DomainIndexEntry
# ─────────────────────────────────────────────────────────────────────────────

def domain_index_entry_to_dict(entry: DomainIndexEntry) -> dict[str, Any]:
    """Serialize a DomainIndexEntry instance."""
    return entry.to_dict()


def domain_index_entry_from_dict(data: dict[str, Any]) -> DomainIndexEntry:
    """Deserialize a plain dictionary into a DomainIndexEntry instance."""
    m = "DomainIndexEntry"
    name: str = _require(data, "name", m)
    return DomainIndexEntry(
        id=_require(data, "id", m),
        domain_id=_require(data, "domain_id", m),
        name=name,
        name_lower=data.get("name_lower", name.lower()),
        parent_domain_id=data.get("parent_domain_id"),
        sub_domain_count=int(_require(data, "sub_domain_count", m)),
        concept_count=int(_require(data, "concept_count", m)),
        skill_count=int(_require(data, "skill_count", m)),
        fact_count=int(_require(data, "fact_count", m)),
        is_root_domain=bool(_require(data, "is_root_domain", m)),
        indexed_at=_iso_to_dt(_require(data, "indexed_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SkillIndexEntry
# ─────────────────────────────────────────────────────────────────────────────

def skill_index_entry_to_dict(entry: SkillIndexEntry) -> dict[str, Any]:
    """Serialize a SkillIndexEntry instance."""
    return entry.to_dict()


def skill_index_entry_from_dict(data: dict[str, Any]) -> SkillIndexEntry:
    """Deserialize a plain dictionary into a SkillIndexEntry instance."""
    m = "SkillIndexEntry"
    name: str = _require(data, "name", m)
    return SkillIndexEntry(
        id=_require(data, "id", m),
        skill_id=_require(data, "skill_id", m),
        name=name,
        name_lower=data.get("name_lower", name.lower()),
        skill_type=SkillType(_require(data, "skill_type", m)),
        domain_ids=tuple(_require(data, "domain_ids", m)),
        prerequisite_skill_ids=tuple(_require(data, "prerequisite_skill_ids", m)),
        has_progression_model=bool(_require(data, "has_progression_model", m)),
        difficulty=KnowledgeDifficulty(_require(data, "difficulty", m)),
        indexed_at=_iso_to_dt(_require(data, "indexed_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# IntegrityIssue
# ─────────────────────────────────────────────────────────────────────────────

def integrity_issue_to_dict(issue: IntegrityIssue) -> dict[str, Any]:
    """Serialize an IntegrityIssue instance."""
    return issue.to_dict()


def integrity_issue_from_dict(data: dict[str, Any]) -> IntegrityIssue:
    """Deserialize a plain dictionary into an IntegrityIssue instance."""
    m = "IntegrityIssue"
    return IntegrityIssue(
        id=_require(data, "id", m),
        issue_type=IntegrityIssueType(_require(data, "issue_type", m)),
        severity=ValidationSeverity(_require(data, "severity", m)),
        affected_id=_require(data, "affected_id", m),
        affected_type=KnowledgeType(_require(data, "affected_type", m)),
        description=_require(data, "description", m),
        conflicting_id=data.get("conflicting_id"),
        auto_resolvable=bool(data.get("auto_resolvable", False)),
        resolution_hint=data.get("resolution_hint", ""),
        detected_at=_iso_to_dt(_require(data, "detected_at", m)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# IntegrityReport
# ─────────────────────────────────────────────────────────────────────────────

def integrity_report_to_dict(report: IntegrityReport) -> dict[str, Any]:
    """Serialize an IntegrityReport instance."""
    return report.to_dict()


def integrity_report_from_dict(data: dict[str, Any]) -> IntegrityReport:
    """Deserialize a plain dictionary into an IntegrityReport instance."""
    m = "IntegrityReport"
    return IntegrityReport(
        id=_require(data, "id", m),
        issues=[integrity_issue_from_dict(i) for i in data.get("issues", [])],
        records_scanned=int(_require(data, "records_scanned", m)),
        scan_duration_ms=float(_require(data, "scan_duration_ms", m)),
        scanned_at=_iso_to_dt(_require(data, "scanned_at", m)),
        scan_version=_require(data, "scan_version", m),
        summary=data.get("summary", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeAuditReport
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_audit_report_to_dict(report: KnowledgeAuditReport) -> dict[str, Any]:
    """Serialize a KnowledgeAuditReport instance."""
    return report.to_dict()


def knowledge_audit_report_from_dict(data: dict[str, Any]) -> KnowledgeAuditReport:
    """Deserialize a plain dictionary into a KnowledgeAuditReport instance."""
    m = "KnowledgeAuditReport"
    return KnowledgeAuditReport(
        id=_require(data, "id", m),
        integrity_report=integrity_report_from_dict(_require(data, "integrity_report", m)),
        validation_results=[
            knowledge_validation_result_from_dict(v)
            for v in data.get("validation_results", [])
        ],
        contradiction_count=int(_require(data, "contradiction_count", m)),
        total_records=int(_require(data, "total_records", m)),
        active_records=int(_require(data, "active_records", m)),
        deprecated_records=int(_require(data, "deprecated_records", m)),
        validated_records=int(_require(data, "validated_records", m)),
        average_confidence=float(_require(data, "average_confidence", m)),
        domain_coverage=dict(_require(data, "domain_coverage", m)),
        skill_coverage={
            k: SkillLevel(v)
            for k, v in data.get("skill_coverage", {}).items()
        },
        low_confidence_ids=list(data.get("low_confidence_ids", [])),
        stale_ids=list(data.get("stale_ids", [])),
        generated_at=_iso_to_dt(_require(data, "generated_at", m)),
        audit_version=_require(data, "audit_version", m),
        notes=data.get("notes", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DESERIALIZER REGISTRY  (populated after all schemas are defined)
# ─────────────────────────────────────────────────────────────────────────────

_DESERIALIZER_REGISTRY.update(
    {
        # Metadata
        KnowledgeMetadata: knowledge_metadata_from_dict,
        # Core domain models
        KnowledgeRecord: knowledge_record_from_dict,
        Fact: fact_from_dict,
        Concept: concept_from_dict,
        Skill: skill_from_dict,
        KnowledgeDomain: knowledge_domain_from_dict,
        ProcedureStep: procedure_step_from_dict,
        Procedure: procedure_from_dict,
        ResearchKnowledge: research_knowledge_from_dict,
        ContentSection: content_section_from_dict,
        EducationalKnowledge: educational_knowledge_from_dict,
        # Structural models
        ConceptRelationship: concept_relationship_from_dict,
        KnowledgeReference: knowledge_reference_from_dict,
        KnowledgeDependency: knowledge_dependency_from_dict,
        SemanticNode: semantic_node_from_dict,
        SemanticHierarchy: semantic_hierarchy_from_dict,
        DomainStructure: domain_structure_from_dict,
        # Skill models
        SkillStage: skill_stage_from_dict,
        SkillProgressionModel: skill_progression_model_from_dict,
        SkillPrerequisite: skill_prerequisite_from_dict,
        # Validation models
        ValidationIssue: validation_issue_from_dict,
        KnowledgeValidationResult: knowledge_validation_result_from_dict,
        KnowledgeConfidence: knowledge_confidence_from_dict,
        KnowledgeContradiction: knowledge_contradiction_from_dict,
        KnowledgeEvidence: knowledge_evidence_from_dict,
        # Synthesis models
        KnowledgeSynthesis: knowledge_synthesis_from_dict,
        KnowledgePackage: knowledge_package_from_dict,
        KnowledgeComposition: knowledge_composition_from_dict,
        # Indexing models
        KnowledgeIndexEntry: knowledge_index_entry_from_dict,
        ConceptIndexEntry: concept_index_entry_from_dict,
        DomainIndexEntry: domain_index_entry_from_dict,
        SkillIndexEntry: skill_index_entry_from_dict,
        # Integrity models
        IntegrityIssue: integrity_issue_from_dict,
        IntegrityReport: integrity_report_from_dict,
        KnowledgeAuditReport: knowledge_audit_report_from_dict,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Versioning
    "SCHEMA_VERSION",
    # Schema errors
    "SchemaError",
    "SchemaVersionError",
    "SchemaMissingFieldError",
    # Schema utilities
    "serialize_model",
    "deserialize_model",
    "serialize_collection",
    "deserialize_collection",
    "validate_schema_version",
    # KnowledgeMetadata
    "knowledge_metadata_to_dict",
    "knowledge_metadata_from_dict",
    # KnowledgeRecord
    "knowledge_record_to_dict",
    "knowledge_record_from_dict",
    "knowledge_record_from_dict_polymorphic",
    # Fact
    "fact_to_dict",
    "fact_from_dict",
    # Concept
    "concept_to_dict",
    "concept_from_dict",
    # Skill
    "skill_to_dict",
    "skill_from_dict",
    # KnowledgeDomain
    "knowledge_domain_to_dict",
    "knowledge_domain_from_dict",
    # ProcedureStep
    "procedure_step_to_dict",
    "procedure_step_from_dict",
    # Procedure
    "procedure_to_dict",
    "procedure_from_dict",
    # ResearchKnowledge
    "research_knowledge_to_dict",
    "research_knowledge_from_dict",
    # ContentSection
    "content_section_to_dict",
    "content_section_from_dict",
    # EducationalKnowledge
    "educational_knowledge_to_dict",
    "educational_knowledge_from_dict",
    # ConceptRelationship
    "concept_relationship_to_dict",
    "concept_relationship_from_dict",
    # KnowledgeReference
    "knowledge_reference_to_dict",
    "knowledge_reference_from_dict",
    # KnowledgeDependency
    "knowledge_dependency_to_dict",
    "knowledge_dependency_from_dict",
    # SemanticNode
    "semantic_node_to_dict",
    "semantic_node_from_dict",
    # SemanticHierarchy
    "semantic_hierarchy_to_dict",
    "semantic_hierarchy_from_dict",
    # DomainStructure
    "domain_structure_to_dict",
    "domain_structure_from_dict",
    # SkillStage
    "skill_stage_to_dict",
    "skill_stage_from_dict",
    # SkillProgressionModel
    "skill_progression_model_to_dict",
    "skill_progression_model_from_dict",
    # SkillPrerequisite
    "skill_prerequisite_to_dict",
    "skill_prerequisite_from_dict",
    # ValidationIssue
    "validation_issue_to_dict",
    "validation_issue_from_dict",
    # KnowledgeValidationResult
    "knowledge_validation_result_to_dict",
    "knowledge_validation_result_from_dict",
    # KnowledgeConfidence
    "knowledge_confidence_to_dict",
    "knowledge_confidence_from_dict",
    # KnowledgeContradiction
    "knowledge_contradiction_to_dict",
    "knowledge_contradiction_from_dict",
    # KnowledgeEvidence
    "knowledge_evidence_to_dict",
    "knowledge_evidence_from_dict",
    # KnowledgeSynthesis
    "knowledge_synthesis_to_dict",
    "knowledge_synthesis_from_dict",
    # KnowledgePackage
    "knowledge_package_to_dict",
    "knowledge_package_from_dict",
    # KnowledgeComposition
    "knowledge_composition_to_dict",
    "knowledge_composition_from_dict",
    # KnowledgeIndexEntry
    "knowledge_index_entry_to_dict",
    "knowledge_index_entry_from_dict",
    # ConceptIndexEntry
    "concept_index_entry_to_dict",
    "concept_index_entry_from_dict",
    # DomainIndexEntry
    "domain_index_entry_to_dict",
    "domain_index_entry_from_dict",
    # SkillIndexEntry
    "skill_index_entry_to_dict",
    "skill_index_entry_from_dict",
    # IntegrityIssue
    "integrity_issue_to_dict",
    "integrity_issue_from_dict",
    # IntegrityReport
    "integrity_report_to_dict",
    "integrity_report_from_dict",
    # KnowledgeAuditReport
    "knowledge_audit_report_to_dict",
    "knowledge_audit_report_from_dict",
]