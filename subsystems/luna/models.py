"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/models.py

LUNA is the Semantic Knowledge Core of POLARIS.
It owns all facts, concepts, skills, knowledge domains,
procedural knowledge, research knowledge, educational knowledge,
skill models, and semantic structures.

LUNA is NOT memory, experience, identity, or history.
LUNA stores what is known. ECHO stores what happened.

Part of the POLARIS Cognitive Substrate:
    ASTRA  → Identity
    ECHO   → Experience
    LUNA   → Knowledge
    CHRONOS → Time
    CONSTELLATION → Relationships
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _stable_hash(*parts: str) -> str:
    """Generate a stable SHA-256 hex digest from string parts."""
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeType(Enum):
    """Top-level classification of a knowledge record."""
    FACT = "fact"
    CONCEPT = "concept"
    SKILL = "skill"
    DOMAIN = "domain"
    PROCEDURE = "procedure"
    RESEARCH = "research"
    EDUCATIONAL = "educational"
    SEMANTIC_STRUCTURE = "semantic_structure"
    COMPOSITE = "composite"


class FactType(Enum):
    """Classification of an atomic fact."""
    DEFINITION = "definition"
    LAW = "law"
    THEOREM = "theorem"
    FORMULA = "formula"
    CONSTANT = "constant"
    STANDARD = "standard"
    SPECIFICATION = "specification"
    MEASUREMENT = "measurement"
    RELATIONSHIP = "relationship"
    CONSTRAINT = "constraint"
    PROPERTY = "property"
    AXIOM = "axiom"


class ConceptType(Enum):
    """Classification of an abstract concept."""
    PRINCIPLE = "principle"
    THEORY = "theory"
    PARADIGM = "paradigm"
    METHODOLOGY = "methodology"
    FRAMEWORK = "framework"
    PATTERN = "pattern"
    ARCHITECTURE = "architecture"
    ABSTRACTION = "abstraction"
    MODEL = "model"
    SYSTEM = "system"
    PROCESS = "process"
    MECHANISM = "mechanism"


class SkillType(Enum):
    """Classification of a capability."""
    TECHNICAL = "technical"
    ANALYTICAL = "analytical"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    INTEGRATION = "integration"
    TESTING = "testing"
    OPERATIONAL = "operational"
    COMMUNICATION = "communication"
    RESEARCH = "research"
    MATHEMATICAL = "mathematical"


class ProcedureType(Enum):
    """Classification of procedural knowledge."""
    SETUP = "setup"
    CONFIGURATION = "configuration"
    CALIBRATION = "calibration"
    DEPLOYMENT = "deployment"
    TROUBLESHOOTING = "troubleshooting"
    DESIGN_PROCESS = "design_process"
    DEVELOPMENT_WORKFLOW = "development_workflow"
    TESTING_PROTOCOL = "testing_protocol"
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"
    INTEGRATION = "integration"
    ANALYSIS = "analysis"


class ResearchType(Enum):
    """Classification of research knowledge."""
    PAPER = "paper"
    TECHNICAL_REPORT = "technical_report"
    DOCUMENTATION = "documentation"
    EXPERIMENT = "experiment"
    SURVEY = "survey"
    CASE_STUDY = "case_study"
    WHITE_PAPER = "white_paper"
    PATENT = "patent"
    DATASET = "dataset"
    BENCHMARK = "benchmark"


class EducationType(Enum):
    """Classification of educational content."""
    CURRICULUM = "curriculum"
    LEARNING_PATH = "learning_path"
    LESSON = "lesson"
    EXERCISE = "exercise"
    PROJECT = "project"
    ASSESSMENT = "assessment"
    REFERENCE = "reference"
    TUTORIAL = "tutorial"
    GLOSSARY = "glossary"
    CASE_STUDY = "case_study"


class KnowledgeDifficulty(Enum):
    """Difficulty tier of a piece of knowledge."""
    FOUNDATIONAL = "foundational"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    RESEARCH_FRONTIER = "research_frontier"

    @property
    def rank(self) -> int:
        _order = {
            "foundational": 0,
            "beginner": 1,
            "intermediate": 2,
            "advanced": 3,
            "expert": 4,
            "research_frontier": 5,
        }
        return _order[self.value]

    def is_harder_than(self, other: "KnowledgeDifficulty") -> bool:
        return self.rank > other.rank

    def is_easier_than(self, other: "KnowledgeDifficulty") -> bool:
        return self.rank < other.rank


class KnowledgeStatus(Enum):
    """Lifecycle status of a knowledge record."""
    DRAFT = "draft"
    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    ARCHIVED = "archived"

    @property
    def is_usable(self) -> bool:
        return self in {KnowledgeStatus.VALIDATED, KnowledgeStatus.ACTIVE}

    @property
    def is_terminal(self) -> bool:
        return self in {
            KnowledgeStatus.RETRACTED,
            KnowledgeStatus.ARCHIVED,
        }


class KnowledgeSourceType(Enum):
    """Origin classification of knowledge."""
    ACADEMIC_PAPER = "academic_paper"
    TEXTBOOK = "textbook"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    OFFICIAL_STANDARD = "official_standard"
    VERIFIED_EXPERIMENT = "verified_experiment"
    EXPERT_CONSENSUS = "expert_consensus"
    PEER_REVIEWED = "peer_reviewed"
    OPEN_SOURCE_CODEBASE = "open_source_codebase"
    DATASET = "dataset"
    USER_INPUT = "user_input"
    DERIVED = "derived"
    SYNTHESIZED = "synthesized"
    UNKNOWN = "unknown"

    @property
    def trust_weight(self) -> float:
        """Base trust weight for this source type."""
        _weights = {
            "official_standard": 1.0,
            "peer_reviewed": 0.95,
            "academic_paper": 0.90,
            "textbook": 0.88,
            "expert_consensus": 0.85,
            "technical_documentation": 0.82,
            "verified_experiment": 0.80,
            "open_source_codebase": 0.70,
            "dataset": 0.68,
            "derived": 0.65,
            "synthesized": 0.60,
            "user_input": 0.50,
            "unknown": 0.30,
        }
        return _weights.get(self.value, 0.30)


class ValidationStatus(Enum):
    """Result of a knowledge validation pass."""
    NOT_VALIDATED = "not_validated"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    CONDITIONALLY_PASSED = "conditionally_passed"

    @property
    def is_valid(self) -> bool:
        return self in {ValidationStatus.PASSED, ValidationStatus.CONDITIONALLY_PASSED}


class ConfidenceLevel(Enum):
    """Semantic confidence band for a knowledge claim."""
    CERTAIN = "certain"           # ≥ 0.95
    HIGH = "high"                 # ≥ 0.80
    MODERATE = "moderate"         # ≥ 0.60
    LOW = "low"                   # ≥ 0.40
    SPECULATIVE = "speculative"   # ≥ 0.20
    UNKNOWN = "unknown"           # < 0.20

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        if score >= 0.95:
            return cls.CERTAIN
        if score >= 0.80:
            return cls.HIGH
        if score >= 0.60:
            return cls.MODERATE
        if score >= 0.40:
            return cls.LOW
        if score >= 0.20:
            return cls.SPECULATIVE
        return cls.UNKNOWN


class SkillLevel(Enum):
    """Proficiency level within a skill progression model."""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"

    @property
    def rank(self) -> int:
        _order = {
            "novice": 0,
            "beginner": 1,
            "intermediate": 2,
            "advanced": 3,
            "expert": 4,
            "master": 5,
        }
        return _order[self.value]

    def next_level(self) -> Optional["SkillLevel"]:
        levels = list(SkillLevel)
        current_idx = levels.index(self)
        if current_idx < len(levels) - 1:
            return levels[current_idx + 1]
        return None

    def previous_level(self) -> Optional["SkillLevel"]:
        levels = list(SkillLevel)
        current_idx = levels.index(self)
        if current_idx > 0:
            return levels[current_idx - 1]
        return None

    def is_above(self, other: "SkillLevel") -> bool:
        return self.rank > other.rank

    def is_below(self, other: "SkillLevel") -> bool:
        return self.rank < other.rank


# ─────────────────────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KnowledgeMetadata:
    """
    Immutable metadata attached to every LUNA knowledge record.

    Carries provenance, confidence, temporal, and versioning information.
    """
    source: str
    source_type: KnowledgeSourceType
    confidence_score: float                        # 0.0 – 1.0
    validation_status: ValidationStatus
    created_at: datetime
    updated_at: datetime
    version: int                                   # monotonically increasing
    tags: tuple[str, ...]                          # immutable tag set
    references: tuple[str, ...]                    # external reference URIs / IDs
    author: Optional[str] = None
    language: str = "en"
    review_date: Optional[datetime] = None         # scheduled re-validation date

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return ConfidenceLevel.from_score(self.confidence_score)

    @property
    def is_stale(self) -> bool:
        """True if the record is past its scheduled review date."""
        if self.review_date is None:
            return False
        return _utcnow() > self.review_date

    @property
    def age_days(self) -> float:
        delta = _utcnow() - self.created_at
        return delta.total_seconds() / 86_400

    @property
    def is_validated(self) -> bool:
        return self.validation_status.is_valid

    @property
    def effective_trust(self) -> float:
        """
        Effective trust = confidence_score × source trust_weight.
        Clamped to [0.0, 1.0].
        """
        raw = self.confidence_score * self.source_type.trust_weight
        return max(0.0, min(1.0, raw))

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        source: str,
        source_type: KnowledgeSourceType = KnowledgeSourceType.UNKNOWN,
        confidence_score: float = 0.5,
        tags: Optional[list[str]] = None,
        references: Optional[list[str]] = None,
        author: Optional[str] = None,
        language: str = "en",
        review_date: Optional[datetime] = None,
    ) -> "KnowledgeMetadata":
        now = _utcnow()
        return cls(
            source=source,
            source_type=source_type,
            confidence_score=max(0.0, min(1.0, confidence_score)),
            validation_status=ValidationStatus.NOT_VALIDATED,
            created_at=now,
            updated_at=now,
            version=1,
            tags=tuple(tags or []),
            references=tuple(references or []),
            author=author,
            language=language,
            review_date=review_date,
        )

    def bump_version(
        self,
        confidence_score: Optional[float] = None,
        validation_status: Optional[ValidationStatus] = None,
        tags: Optional[list[str]] = None,
        references: Optional[list[str]] = None,
        review_date: Optional[datetime] = None,
    ) -> "KnowledgeMetadata":
        """Return a new KnowledgeMetadata with incremented version and updated_at."""
        return KnowledgeMetadata(
            source=self.source,
            source_type=self.source_type,
            confidence_score=max(0.0, min(1.0, confidence_score)) if confidence_score is not None else self.confidence_score,
            validation_status=validation_status if validation_status is not None else self.validation_status,
            created_at=self.created_at,
            updated_at=_utcnow(),
            version=self.version + 1,
            tags=tuple(tags) if tags is not None else self.tags,
            references=tuple(references) if references is not None else self.references,
            author=self.author,
            language=self.language,
            review_date=review_date if review_date is not None else self.review_date,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "confidence_score": self.confidence_score,
            "validation_status": self.validation_status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "tags": list(self.tags),
            "references": list(self.references),
            "author": self.author,
            "language": self.language,
            "review_date": self.review_date.isoformat() if self.review_date else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CORE DOMAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeRecord:
    """
    Base class for all LUNA knowledge records.

    Every knowledge artefact in LUNA descends from this class.
    Provides identity, lifecycle, and metadata scaffolding.
    """
    id: str
    knowledge_type: KnowledgeType
    name: str
    description: str
    status: KnowledgeStatus
    difficulty: KnowledgeDifficulty
    domain_ids: list[str]                          # parent domain IDs
    metadata: KnowledgeMetadata
    aliases: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    notes: str = ""

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.status.is_usable

    @property
    def is_high_confidence(self) -> bool:
        return self.metadata.confidence_score >= 0.80

    @property
    def fingerprint(self) -> str:
        """Stable content fingerprint for deduplication."""
        return _stable_hash(self.knowledge_type.value, self.name, self.description)

    @property
    def short_id(self) -> str:
        return self.id[:8]

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "knowledge_type": self.knowledge_type.value,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "difficulty": self.difficulty.value,
            "domain_ids": self.domain_ids,
            "metadata": self.metadata.to_dict(),
            "aliases": self.aliases,
            "related_ids": self.related_ids,
            "notes": self.notes,
        }

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        knowledge_type: KnowledgeType,
        name: str,
        description: str,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        aliases: Optional[list[str]] = None,
        related_ids: Optional[list[str]] = None,
        notes: str = "",
    ) -> "KnowledgeRecord":
        return cls(
            id=_new_id(),
            knowledge_type=knowledge_type,
            name=name,
            description=description,
            status=KnowledgeStatus.DRAFT,
            difficulty=difficulty,
            domain_ids=domain_ids,
            metadata=metadata,
            aliases=aliases or [],
            related_ids=related_ids or [],
            notes=notes,
        )


@dataclass
class Fact(KnowledgeRecord):
    """
    Atomic fact owned by LUNA.

    A Fact is the smallest unit of knowledge — a single truth claim
    about the world (law, formula, definition, specification, etc.).

    Examples:
        - Ohm's Law: V = IR
        - Newton's Second Law: F = ma
        - Python list indexing starts at 0
    """
    fact_type: FactType = FactType.DEFINITION
    statement: str = ""                            # Precise, canonical statement
    formal_notation: Optional[str] = None         # Mathematical / formal expression
    units: Optional[str] = None                   # SI units, if applicable
    conditions: list[str] = field(default_factory=list)   # Conditions under which true
    counterexamples: list[str] = field(default_factory=list)
    supporting_fact_ids: list[str] = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_universal(self) -> bool:
        """True if this fact holds without conditions."""
        return len(self.conditions) == 0

    @property
    def has_formal_notation(self) -> bool:
        return self.formal_notation is not None and len(self.formal_notation.strip()) > 0

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "fact_type": self.fact_type.value,
            "statement": self.statement,
            "formal_notation": self.formal_notation,
            "units": self.units,
            "conditions": self.conditions,
            "counterexamples": self.counterexamples,
            "supporting_fact_ids": self.supporting_fact_ids,
        })
        return base

    @classmethod
    def create(  # type: ignore[override]
        cls,
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
    ) -> "Fact":
        return cls(
            id=_new_id(),
            knowledge_type=KnowledgeType.FACT,
            name=name,
            description=description,
            status=KnowledgeStatus.DRAFT,
            difficulty=difficulty,
            domain_ids=domain_ids,
            metadata=metadata,
            aliases=aliases or [],
            notes=notes,
            fact_type=fact_type,
            statement=statement,
            formal_notation=formal_notation,
            units=units,
            conditions=conditions or [],
        )


@dataclass
class Concept(KnowledgeRecord):
    """
    Abstract concept owned by LUNA.

    Concepts form knowledge domains and provide the semantic scaffold
    on which facts, skills, and procedures hang.

    Examples:
        - Machine Learning
        - PID Control
        - State Space Representation
        - Mechatronics
    """
    concept_type: ConceptType = ConceptType.PRINCIPLE
    core_ideas: list[str] = field(default_factory=list)     # Key insight bullets
    applications: list[str] = field(default_factory=list)   # Where this concept applies
    prerequisite_concept_ids: list[str] = field(default_factory=list)
    child_concept_ids: list[str] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)        # Facts that instantiate this concept
    is_foundational: bool = False

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def has_prerequisites(self) -> bool:
        return len(self.prerequisite_concept_ids) > 0

    @property
    def has_children(self) -> bool:
        return len(self.child_concept_ids) > 0

    @property
    def is_leaf_concept(self) -> bool:
        return not self.has_children

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "concept_type": self.concept_type.value,
            "core_ideas": self.core_ideas,
            "applications": self.applications,
            "prerequisite_concept_ids": self.prerequisite_concept_ids,
            "child_concept_ids": self.child_concept_ids,
            "fact_ids": self.fact_ids,
            "is_foundational": self.is_foundational,
        })
        return base

    @classmethod
    def create(  # type: ignore[override]
        cls,
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
    ) -> "Concept":
        return cls(
            id=_new_id(),
            knowledge_type=KnowledgeType.CONCEPT,
            name=name,
            description=description,
            status=KnowledgeStatus.DRAFT,
            difficulty=difficulty,
            domain_ids=domain_ids,
            metadata=metadata,
            aliases=aliases or [],
            notes=notes,
            concept_type=concept_type,
            core_ideas=core_ideas or [],
            applications=applications or [],
            prerequisite_concept_ids=prerequisite_concept_ids or [],
            is_foundational=is_foundational,
        )


@dataclass
class Skill(KnowledgeRecord):
    """
    Capability owned by LUNA.

    LUNA owns the skill definition and structure.
    ASTRA owns the user's proficiency profile against that skill.

    Examples:
        - Python Programming
        - PCB Design
        - CAD Modeling
        - ROS2 Development
        - Embedded Systems
    """
    skill_type: SkillType = SkillType.TECHNICAL
    capability_description: str = ""              # What this skill enables
    required_tools: list[str] = field(default_factory=list)
    required_concept_ids: list[str] = field(default_factory=list)
    required_fact_ids: list[str] = field(default_factory=list)
    prerequisite_skill_ids: list[str] = field(default_factory=list)
    sub_skill_ids: list[str] = field(default_factory=list)
    progression_model_id: Optional[str] = None    # FK → SkillProgressionModel
    practical_exercises: list[str] = field(default_factory=list)
    assessment_criteria: list[str] = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_composite(self) -> bool:
        return len(self.sub_skill_ids) > 0

    @property
    def has_progression_model(self) -> bool:
        return self.progression_model_id is not None

    @property
    def requires_prerequisites(self) -> bool:
        return (
            len(self.prerequisite_skill_ids) > 0
            or len(self.required_concept_ids) > 0
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "skill_type": self.skill_type.value,
            "capability_description": self.capability_description,
            "required_tools": self.required_tools,
            "required_concept_ids": self.required_concept_ids,
            "required_fact_ids": self.required_fact_ids,
            "prerequisite_skill_ids": self.prerequisite_skill_ids,
            "sub_skill_ids": self.sub_skill_ids,
            "progression_model_id": self.progression_model_id,
            "practical_exercises": self.practical_exercises,
            "assessment_criteria": self.assessment_criteria,
        })
        return base

    @classmethod
    def create(  # type: ignore[override]
        cls,
        name: str,
        description: str,
        skill_type: SkillType,
        capability_description: str,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        required_concept_ids: Optional[list[str]] = None,
        prerequisite_skill_ids: Optional[list[str]] = None,
        required_tools: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> "Skill":
        return cls(
            id=_new_id(),
            knowledge_type=KnowledgeType.SKILL,
            name=name,
            description=description,
            status=KnowledgeStatus.DRAFT,
            difficulty=difficulty,
            domain_ids=domain_ids,
            metadata=metadata,
            aliases=aliases or [],
            notes=notes,
            skill_type=skill_type,
            capability_description=capability_description,
            required_concept_ids=required_concept_ids or [],
            prerequisite_skill_ids=prerequisite_skill_ids or [],
            required_tools=required_tools or [],
        )


@dataclass
class KnowledgeDomain(KnowledgeRecord):
    """
    Major discipline or knowledge area owned by LUNA.

    Domains provide the top-level organizational structure for
    all knowledge within LUNA.

    Examples:
        - Artificial Intelligence
        - Robotics
        - Electronics
        - Mathematics
        - Control Systems
        - Business
    """
    parent_domain_id: Optional[str] = None
    sub_domain_ids: list[str] = field(default_factory=list)
    core_concept_ids: list[str] = field(default_factory=list)
    core_skill_ids: list[str] = field(default_factory=list)
    core_fact_ids: list[str] = field(default_factory=list)
    standard_references: list[str] = field(default_factory=list)
    is_root_domain: bool = False

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_subdomain(self) -> bool:
        return self.parent_domain_id is not None

    @property
    def has_subdomains(self) -> bool:
        return len(self.sub_domain_ids) > 0

    @property
    def concept_count(self) -> int:
        return len(self.core_concept_ids)

    @property
    def skill_count(self) -> int:
        return len(self.core_skill_ids)

    @property
    def fact_count(self) -> int:
        return len(self.core_fact_ids)

    @property
    def total_knowledge_count(self) -> int:
        return self.concept_count + self.skill_count + self.fact_count

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "parent_domain_id": self.parent_domain_id,
            "sub_domain_ids": self.sub_domain_ids,
            "core_concept_ids": self.core_concept_ids,
            "core_skill_ids": self.core_skill_ids,
            "core_fact_ids": self.core_fact_ids,
            "standard_references": self.standard_references,
            "is_root_domain": self.is_root_domain,
        })
        return base

    @classmethod
    def create(  # type: ignore[override]
        cls,
        name: str,
        description: str,
        difficulty: KnowledgeDifficulty,
        metadata: KnowledgeMetadata,
        parent_domain_id: Optional[str] = None,
        is_root_domain: bool = False,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> "KnowledgeDomain":
        return cls(
            id=_new_id(),
            knowledge_type=KnowledgeType.DOMAIN,
            name=name,
            description=description,
            status=KnowledgeStatus.DRAFT,
            difficulty=difficulty,
            domain_ids=[],
            metadata=metadata,
            aliases=aliases or [],
            notes=notes,
            parent_domain_id=parent_domain_id,
            is_root_domain=is_root_domain,
        )


@dataclass
class Procedure(KnowledgeRecord):
    """
    Procedural / how-to knowledge owned by LUNA.

    Procedures encode executable knowledge — the ordered steps
    required to accomplish a technical or operational goal.

    Examples:
        - How to tune a PID controller
        - How to deploy ROS2
        - How to train a neural network
        - How to build a drone
    """
    procedure_type: ProcedureType = ProcedureType.DEVELOPMENT_WORKFLOW
    goal: str = ""                                           # What the procedure achieves
    steps: list["ProcedureStep"] = field(default_factory=list)
    required_skill_ids: list[str] = field(default_factory=list)
    required_concept_ids: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    required_materials: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    common_pitfalls: list[str] = field(default_factory=list)
    expected_duration_minutes: Optional[int] = None
    is_reversible: bool = True

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def has_preconditions(self) -> bool:
        return len(self.preconditions) > 0

    @property
    def is_complex(self) -> bool:
        return self.step_count > 10

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "procedure_type": self.procedure_type.value,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "required_skill_ids": self.required_skill_ids,
            "required_concept_ids": self.required_concept_ids,
            "required_tools": self.required_tools,
            "required_materials": self.required_materials,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "common_pitfalls": self.common_pitfalls,
            "expected_duration_minutes": self.expected_duration_minutes,
            "is_reversible": self.is_reversible,
        })
        return base

    @classmethod
    def create(  # type: ignore[override]
        cls,
        name: str,
        description: str,
        goal: str,
        procedure_type: ProcedureType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        required_skill_ids: Optional[list[str]] = None,
        required_concept_ids: Optional[list[str]] = None,
        required_tools: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> "Procedure":
        return cls(
            id=_new_id(),
            knowledge_type=KnowledgeType.PROCEDURE,
            name=name,
            description=description,
            status=KnowledgeStatus.DRAFT,
            difficulty=difficulty,
            domain_ids=domain_ids,
            metadata=metadata,
            aliases=aliases or [],
            notes=notes,
            procedure_type=procedure_type,
            goal=goal,
            required_skill_ids=required_skill_ids or [],
            required_concept_ids=required_concept_ids or [],
            required_tools=required_tools or [],
        )


@dataclass(frozen=True)
class ProcedureStep:
    """A single ordered step within a Procedure."""
    step_number: int
    title: str
    instruction: str
    sub_steps: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    expected_outcome: str = ""
    is_critical: bool = False
    estimated_minutes: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "title": self.title,
            "instruction": self.instruction,
            "sub_steps": list(self.sub_steps),
            "warnings": list(self.warnings),
            "expected_outcome": self.expected_outcome,
            "is_critical": self.is_critical,
            "estimated_minutes": self.estimated_minutes,
        }


@dataclass
class ResearchKnowledge(KnowledgeRecord):
    """
    Research-derived knowledge owned by LUNA.

    Stores knowledge extracted from papers, technical reports,
    documentation, and experiments. Used heavily by PROMETHEUS and VULCAN.
    """
    research_type: ResearchType = ResearchType.PAPER
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    publication_venue: Optional[str] = None
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    key_findings: list[str] = field(default_factory=list)
    methodology: str = ""
    limitations: list[str] = field(default_factory=list)
    cited_knowledge_ids: list[str] = field(default_factory=list)
    extracted_fact_ids: list[str] = field(default_factory=list)
    extracted_concept_ids: list[str] = field(default_factory=list)
    citation_count: int = 0

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_highly_cited(self) -> bool:
        return self.citation_count >= 100

    @property
    def has_doi(self) -> bool:
        return self.doi is not None

    @property
    def knowledge_density(self) -> float:
        """Ratio of extracted knowledge items to findings."""
        total_extracted = len(self.extracted_fact_ids) + len(self.extracted_concept_ids)
        if len(self.key_findings) == 0:
            return 0.0
        return total_extracted / len(self.key_findings)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "research_type": self.research_type.value,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "publication_venue": self.publication_venue,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "doi": self.doi,
            "url": self.url,
            "key_findings": self.key_findings,
            "methodology": self.methodology,
            "limitations": self.limitations,
            "cited_knowledge_ids": self.cited_knowledge_ids,
            "extracted_fact_ids": self.extracted_fact_ids,
            "extracted_concept_ids": self.extracted_concept_ids,
            "citation_count": self.citation_count,
        })
        return base

    @classmethod
    def create(  # type: ignore[override]
        cls,
        name: str,
        description: str,
        title: str,
        research_type: ResearchType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        abstract: str = "",
        authors: Optional[list[str]] = None,
        key_findings: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> "ResearchKnowledge":
        return cls(
            id=_new_id(),
            knowledge_type=KnowledgeType.RESEARCH,
            name=name,
            description=description,
            status=KnowledgeStatus.DRAFT,
            difficulty=difficulty,
            domain_ids=domain_ids,
            metadata=metadata,
            aliases=aliases or [],
            notes=notes,
            research_type=research_type,
            title=title,
            abstract=abstract,
            authors=authors or [],
            key_findings=key_findings or [],
        )


@dataclass
class EducationalKnowledge(KnowledgeRecord):
    """
    Educational content owned by LUNA. Primary consumer is APOLLO.

    Stores learning paths, curriculum structures, difficulty levels,
    and prerequisite graphs to support systematic teaching.
    """
    education_type: EducationType = EducationType.LEARNING_PATH
    learning_objectives: list[str] = field(default_factory=list)
    prerequisite_knowledge_ids: list[str] = field(default_factory=list)
    target_skill_ids: list[str] = field(default_factory=list)
    target_concept_ids: list[str] = field(default_factory=list)
    estimated_duration_hours: Optional[float] = None
    assessment_type: Optional[str] = None
    learning_outcomes: list[str] = field(default_factory=list)
    content_sections: list["ContentSection"] = field(default_factory=list)
    is_self_contained: bool = False

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def section_count(self) -> int:
        return len(self.content_sections)

    @property
    def has_assessment(self) -> bool:
        return self.assessment_type is not None

    @property
    def objective_count(self) -> int:
        return len(self.learning_objectives)

    @property
    def is_comprehensive(self) -> bool:
        return (
            self.section_count >= 3
            and self.has_assessment
            and len(self.learning_objectives) >= 2
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "education_type": self.education_type.value,
            "learning_objectives": self.learning_objectives,
            "prerequisite_knowledge_ids": self.prerequisite_knowledge_ids,
            "target_skill_ids": self.target_skill_ids,
            "target_concept_ids": self.target_concept_ids,
            "estimated_duration_hours": self.estimated_duration_hours,
            "assessment_type": self.assessment_type,
            "learning_outcomes": self.learning_outcomes,
            "content_sections": [s.to_dict() for s in self.content_sections],
            "is_self_contained": self.is_self_contained,
        })
        return base

    @classmethod
    def create(  # type: ignore[override]
        cls,
        name: str,
        description: str,
        education_type: EducationType,
        difficulty: KnowledgeDifficulty,
        domain_ids: list[str],
        metadata: KnowledgeMetadata,
        learning_objectives: Optional[list[str]] = None,
        target_skill_ids: Optional[list[str]] = None,
        target_concept_ids: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        notes: str = "",
    ) -> "EducationalKnowledge":
        return cls(
            id=_new_id(),
            knowledge_type=KnowledgeType.EDUCATIONAL,
            name=name,
            description=description,
            status=KnowledgeStatus.DRAFT,
            difficulty=difficulty,
            domain_ids=domain_ids,
            metadata=metadata,
            aliases=aliases or [],
            notes=notes,
            education_type=education_type,
            learning_objectives=learning_objectives or [],
            target_skill_ids=target_skill_ids or [],
            target_concept_ids=target_concept_ids or [],
        )


@dataclass(frozen=True)
class ContentSection:
    """A titled section within EducationalKnowledge content."""
    section_number: int
    title: str
    summary: str
    knowledge_ids: tuple[str, ...] = field(default_factory=tuple)
    duration_minutes: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_number": self.section_number,
            "title": self.title,
            "summary": self.summary,
            "knowledge_ids": list(self.knowledge_ids),
            "duration_minutes": self.duration_minutes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ConceptRelationshipType(Enum):
    """Semantic relationship type between two Concepts."""
    IS_A = "is_a"                      # Inheritance / specialization
    HAS_A = "has_a"                    # Composition
    USES = "uses"                      # Dependency
    EXTENDS = "extends"                # Extension
    CONTRASTS_WITH = "contrasts_with"  # Semantic contrast
    PRECEDES = "precedes"              # Temporal / logical ordering
    ENABLES = "enables"                # Causal enablement
    CONTRADICTS = "contradicts"        # Semantic conflict
    EQUIVALENT = "equivalent"          # Same concept, different name
    PART_OF = "part_of"                # Meronymy
    DERIVED_FROM = "derived_from"      # Derivation


@dataclass(frozen=True)
class ConceptRelationship:
    """
    Directed semantic relationship between two Concepts.

    LUNA owns concepts. CONSTELLATION links concepts.
    ConceptRelationship is the edge description owned by LUNA;
    CONSTELLATION manages the graph traversal layer.
    """
    id: str
    source_concept_id: str
    target_concept_id: str
    relationship_type: ConceptRelationshipType
    description: str
    weight: float = 1.0                # Semantic strength 0.0–1.0
    is_bidirectional: bool = False
    created_at: datetime = field(default_factory=_utcnow)

    @property
    def is_strong(self) -> bool:
        return self.weight >= 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_concept_id": self.source_concept_id,
            "target_concept_id": self.target_concept_id,
            "relationship_type": self.relationship_type.value,
            "description": self.description,
            "weight": self.weight,
            "is_bidirectional": self.is_bidirectional,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def create(
        cls,
        source_concept_id: str,
        target_concept_id: str,
        relationship_type: ConceptRelationshipType,
        description: str,
        weight: float = 1.0,
        is_bidirectional: bool = False,
    ) -> "ConceptRelationship":
        return cls(
            id=_new_id(),
            source_concept_id=source_concept_id,
            target_concept_id=target_concept_id,
            relationship_type=relationship_type,
            description=description,
            weight=max(0.0, min(1.0, weight)),
            is_bidirectional=is_bidirectional,
        )


@dataclass(frozen=True)
class KnowledgeReference:
    """
    A typed citation link from one knowledge record to another
    or to an external source.
    """
    id: str
    source_id: str                     # ID of the referencing record
    target: str                        # Target ID or URI
    reference_type: str                # "supports", "contradicts", "elaborates", "cites"
    description: str = ""
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target": self.target,
            "reference_type": self.reference_type,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def create(
        cls,
        source_id: str,
        target: str,
        reference_type: str,
        description: str = "",
    ) -> "KnowledgeReference":
        return cls(
            id=_new_id(),
            source_id=source_id,
            target=target,
            reference_type=reference_type,
            description=description,
        )


@dataclass(frozen=True)
class KnowledgeDependency:
    """
    Directed dependency between two knowledge records.

    Encodes: record A requires record B to be understood/applied.
    """
    id: str
    dependent_id: str                  # The record that depends
    dependency_id: str                 # The record that is required
    dependency_type: str               # "requires", "recommends", "enhances"
    is_hard: bool = True               # Hard = blocking prerequisite
    description: str = ""

    @property
    def is_soft(self) -> bool:
        return not self.is_hard

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dependent_id": self.dependent_id,
            "dependency_id": self.dependency_id,
            "dependency_type": self.dependency_type,
            "is_hard": self.is_hard,
            "description": self.description,
        }

    @classmethod
    def create(
        cls,
        dependent_id: str,
        dependency_id: str,
        dependency_type: str = "requires",
        is_hard: bool = True,
        description: str = "",
    ) -> "KnowledgeDependency":
        return cls(
            id=_new_id(),
            dependent_id=dependent_id,
            dependency_id=dependency_id,
            dependency_type=dependency_type,
            is_hard=is_hard,
            description=description,
        )


@dataclass
class SemanticNode:
    """
    A node in a semantic hierarchy tree.

    Represents a concept or domain as a positioned element
    within the SemanticHierarchy structure.
    """
    id: str
    knowledge_id: str                  # FK → KnowledgeRecord
    knowledge_type: KnowledgeType
    label: str
    parent_node_id: Optional[str] = None
    child_node_ids: list[str] = field(default_factory=list)
    depth: int = 0                     # Distance from root
    weight: float = 1.0               # Semantic prominence
    annotations: dict[str, str] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.parent_node_id is None

    @property
    def is_leaf(self) -> bool:
        return len(self.child_node_ids) == 0

    @property
    def child_count(self) -> int:
        return len(self.child_node_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "knowledge_id": self.knowledge_id,
            "knowledge_type": self.knowledge_type.value,
            "label": self.label,
            "parent_node_id": self.parent_node_id,
            "child_node_ids": self.child_node_ids,
            "depth": self.depth,
            "weight": self.weight,
            "annotations": self.annotations,
        }

    @classmethod
    def create(
        cls,
        knowledge_id: str,
        knowledge_type: KnowledgeType,
        label: str,
        parent_node_id: Optional[str] = None,
        depth: int = 0,
    ) -> "SemanticNode":
        return cls(
            id=_new_id(),
            knowledge_id=knowledge_id,
            knowledge_type=knowledge_type,
            label=label,
            parent_node_id=parent_node_id,
            depth=depth,
        )


@dataclass
class SemanticHierarchy:
    """
    A semantic tree structure representing the logical organization
    of knowledge within a domain.

    Example:
        Control Systems
        ├── PID
        ├── State Space
        ├── Feedback
        └── Stability
    """
    id: str
    name: str
    description: str
    root_node_id: str
    nodes: dict[str, SemanticNode] = field(default_factory=dict)  # node_id → SemanticNode
    domain_id: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def root_node(self) -> Optional[SemanticNode]:
        return self.nodes.get(self.root_node_id)

    @property
    def max_depth(self) -> int:
        if not self.nodes:
            return 0
        return max(n.depth for n in self.nodes.values())

    def get_children(self, node_id: str) -> list[SemanticNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[cid] for cid in node.child_node_ids if cid in self.nodes]

    def get_ancestors(self, node_id: str) -> list[SemanticNode]:
        ancestors = []
        current = self.nodes.get(node_id)
        while current and current.parent_node_id:
            parent = self.nodes.get(current.parent_node_id)
            if parent:
                ancestors.append(parent)
            current = parent
        return ancestors

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "root_node_id": self.root_node_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "domain_id": self.domain_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        root_node: SemanticNode,
        domain_id: Optional[str] = None,
    ) -> "SemanticHierarchy":
        return cls(
            id=_new_id(),
            name=name,
            description=description,
            root_node_id=root_node.id,
            nodes={root_node.id: root_node},
            domain_id=domain_id,
        )


@dataclass
class DomainStructure:
    """
    The complete organizational structure of a KnowledgeDomain.

    Aggregates the domain's concepts, facts, skills, procedures,
    sub-domains, and semantic hierarchy into a single navigable model.
    """
    id: str
    domain_id: str
    domain_name: str
    concept_ids: list[str] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    skill_ids: list[str] = field(default_factory=list)
    procedure_ids: list[str] = field(default_factory=list)
    sub_domain_ids: list[str] = field(default_factory=list)
    hierarchy_ids: list[str] = field(default_factory=list)
    dependency_ids: list[str] = field(default_factory=list)
    total_records: int = 0
    last_updated: datetime = field(default_factory=_utcnow)

    @property
    def is_populated(self) -> bool:
        return self.total_records > 0

    @property
    def is_rich(self) -> bool:
        return (
            len(self.concept_ids) >= 3
            and len(self.skill_ids) >= 1
            and len(self.fact_ids) >= 3
        )

    def recalculate_total(self) -> None:
        self.total_records = (
            len(self.concept_ids)
            + len(self.fact_ids)
            + len(self.skill_ids)
            + len(self.procedure_ids)
        )
        self.last_updated = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain_id": self.domain_id,
            "domain_name": self.domain_name,
            "concept_ids": self.concept_ids,
            "fact_ids": self.fact_ids,
            "skill_ids": self.skill_ids,
            "procedure_ids": self.procedure_ids,
            "sub_domain_ids": self.sub_domain_ids,
            "hierarchy_ids": self.hierarchy_ids,
            "dependency_ids": self.dependency_ids,
            "total_records": self.total_records,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def create(cls, domain_id: str, domain_name: str) -> "DomainStructure":
        return cls(id=_new_id(), domain_id=domain_id, domain_name=domain_name)


# ─────────────────────────────────────────────────────────────────────────────
# SKILL MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillStage:
    """
    A discrete stage in a skill progression model.

    Defines what competencies, knowledge, and outputs characterize
    a learner at a given SkillLevel.
    """
    level: SkillLevel
    label: str                                     # Human-readable label (e.g. "Beginner")
    description: str
    required_concept_ids: tuple[str, ...] = field(default_factory=tuple)
    required_fact_ids: tuple[str, ...] = field(default_factory=tuple)
    required_procedure_ids: tuple[str, ...] = field(default_factory=tuple)
    competencies: tuple[str, ...] = field(default_factory=tuple)
    demonstration_tasks: tuple[str, ...] = field(default_factory=tuple)
    typical_duration_hours: Optional[float] = None

    @property
    def knowledge_requirement_count(self) -> int:
        return (
            len(self.required_concept_ids)
            + len(self.required_fact_ids)
            + len(self.required_procedure_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "label": self.label,
            "description": self.description,
            "required_concept_ids": list(self.required_concept_ids),
            "required_fact_ids": list(self.required_fact_ids),
            "required_procedure_ids": list(self.required_procedure_ids),
            "competencies": list(self.competencies),
            "demonstration_tasks": list(self.demonstration_tasks),
            "typical_duration_hours": self.typical_duration_hours,
        }


@dataclass
class SkillProgressionModel:
    """
    Structured skill development trajectory owned by LUNA.

    Encodes the full path from novice to master for a given skill,
    including stage definitions, transition criteria, and milestones.

    IMPORTANT:
        LUNA owns this model (the structure of progression).
        ASTRA owns the user's position within this model.
    """
    id: str
    skill_id: str                      # FK → Skill
    skill_name: str
    description: str
    stages: list[SkillStage] = field(default_factory=list)
    transition_criteria: dict[str, str] = field(default_factory=dict)  # "novice→beginner" → criteria
    estimated_total_hours: Optional[float] = None
    is_linear: bool = True             # False = branching paths allowed
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def levels(self) -> list[SkillLevel]:
        return [s.level for s in self.stages]

    @property
    def min_level(self) -> Optional[SkillLevel]:
        return self.stages[0].level if self.stages else None

    @property
    def max_level(self) -> Optional[SkillLevel]:
        return self.stages[-1].level if self.stages else None

    def get_stage(self, level: SkillLevel) -> Optional[SkillStage]:
        for stage in self.stages:
            if stage.level == level:
                return stage
        return None

    def get_transition_criteria(self, from_level: SkillLevel, to_level: SkillLevel) -> Optional[str]:
        key = f"{from_level.value}→{to_level.value}"
        return self.transition_criteria.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "description": self.description,
            "stages": [s.to_dict() for s in self.stages],
            "transition_criteria": self.transition_criteria,
            "estimated_total_hours": self.estimated_total_hours,
            "is_linear": self.is_linear,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def create(
        cls,
        skill_id: str,
        skill_name: str,
        description: str,
        stages: Optional[list[SkillStage]] = None,
    ) -> "SkillProgressionModel":
        return cls(
            id=_new_id(),
            skill_id=skill_id,
            skill_name=skill_name,
            description=description,
            stages=stages or [],
        )


@dataclass(frozen=True)
class SkillPrerequisite:
    """
    A prerequisite relationship between two skills.

    Encodes: to learn skill A, skill B must be at a minimum level first.
    LUNA owns this structural knowledge. ASTRA evaluates readiness.
    """
    id: str
    skill_id: str                      # The skill requiring the prerequisite
    prerequisite_skill_id: str         # The required skill
    minimum_level: SkillLevel          # Minimum level needed
    is_mandatory: bool = True
    rationale: str = ""

    @property
    def is_recommended(self) -> bool:
        return not self.is_mandatory

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill_id": self.skill_id,
            "prerequisite_skill_id": self.prerequisite_skill_id,
            "minimum_level": self.minimum_level.value,
            "is_mandatory": self.is_mandatory,
            "rationale": self.rationale,
        }

    @classmethod
    def create(
        cls,
        skill_id: str,
        prerequisite_skill_id: str,
        minimum_level: SkillLevel,
        is_mandatory: bool = True,
        rationale: str = "",
    ) -> "SkillPrerequisite":
        return cls(
            id=_new_id(),
            skill_id=skill_id,
            prerequisite_skill_id=prerequisite_skill_id,
            minimum_level=minimum_level,
            is_mandatory=is_mandatory,
            rationale=rationale,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ValidationIssueType(Enum):
    """Classification of a validation issue found in a knowledge record."""
    UNRELIABLE_SOURCE = "unreliable_source"
    LOW_CONFIDENCE = "low_confidence"
    STALE_KNOWLEDGE = "stale_knowledge"
    CONTRADICTION = "contradiction"
    MISSING_REFERENCES = "missing_references"
    BROKEN_DEPENDENCY = "broken_dependency"
    DUPLICATE_CONTENT = "duplicate_content"
    INCOMPLETE_DEFINITION = "incomplete_definition"
    UNVERIFIED_CLAIM = "unverified_claim"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    FORMAT_ERROR = "format_error"


class ValidationSeverity(Enum):
    """Severity of a validation issue."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "error": 2, "critical": 3}[self.value]

    def is_blocking(self) -> bool:
        return self in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}


@dataclass(frozen=True)
class KnowledgeValidationResult:
    """
    Result of a single validation pass over a knowledge record.
    Produced by the Knowledge Validation Engine.
    """
    id: str
    knowledge_id: str
    knowledge_name: str
    validation_status: ValidationStatus
    issues: tuple["ValidationIssue", ...]
    validated_at: datetime
    validator_version: str
    notes: str = ""

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0

    @property
    def has_blocking_issues(self) -> bool:
        return any(i.severity.is_blocking() for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL)

    @property
    def highest_severity(self) -> Optional[ValidationSeverity]:
        if not self.issues:
            return None
        return max(self.issues, key=lambda i: i.severity.rank).severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "knowledge_id": self.knowledge_id,
            "knowledge_name": self.knowledge_name,
            "validation_status": self.validation_status.value,
            "issues": [i.to_dict() for i in self.issues],
            "validated_at": self.validated_at.isoformat(),
            "validator_version": self.validator_version,
            "notes": self.notes,
        }

    @classmethod
    def create(
        cls,
        knowledge_id: str,
        knowledge_name: str,
        issues: list["ValidationIssue"],
        validator_version: str,
        notes: str = "",
    ) -> "KnowledgeValidationResult":
        has_blocking = any(i.severity.is_blocking() for i in issues)
        if len(issues) == 0:
            status = ValidationStatus.PASSED
        elif has_blocking:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.CONDITIONALLY_PASSED
        return cls(
            id=_new_id(),
            knowledge_id=knowledge_id,
            knowledge_name=knowledge_name,
            validation_status=status,
            issues=tuple(issues),
            validated_at=_utcnow(),
            validator_version=validator_version,
            notes=notes,
        )


@dataclass(frozen=True)
class ValidationIssue:
    """A single issue identified during knowledge validation."""
    id: str
    issue_type: ValidationIssueType
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None        # Which field triggered the issue
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "suggestion": self.suggestion,
        }

    @classmethod
    def create(
        cls,
        issue_type: ValidationIssueType,
        severity: ValidationSeverity,
        message: str,
        field: Optional[str] = None,
        suggestion: str = "",
    ) -> "ValidationIssue":
        return cls(
            id=_new_id(),
            issue_type=issue_type,
            severity=severity,
            message=message,
            field=field,
            suggestion=suggestion,
        )


@dataclass(frozen=True)
class KnowledgeConfidence:
    """
    Structured confidence assessment for a knowledge record.

    Decomposes overall confidence into contributing factors.
    """
    id: str
    knowledge_id: str
    overall_score: float               # 0.0 – 1.0
    source_reliability: float          # 0.0 – 1.0
    recency_score: float               # 0.0 – 1.0 (1.0 = very recent)
    corroboration_score: float         # 0.0 – 1.0 (multiple sources agree)
    consistency_score: float           # 0.0 – 1.0 (no internal contradictions)
    assessed_at: datetime
    assessment_notes: str = ""

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return ConfidenceLevel.from_score(self.overall_score)

    @property
    def weakest_dimension(self) -> str:
        scores = {
            "source_reliability": self.source_reliability,
            "recency_score": self.recency_score,
            "corroboration_score": self.corroboration_score,
            "consistency_score": self.consistency_score,
        }
        return min(scores, key=scores.get)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "knowledge_id": self.knowledge_id,
            "overall_score": self.overall_score,
            "source_reliability": self.source_reliability,
            "recency_score": self.recency_score,
            "corroboration_score": self.corroboration_score,
            "consistency_score": self.consistency_score,
            "assessed_at": self.assessed_at.isoformat(),
            "assessment_notes": self.assessment_notes,
        }

    @classmethod
    def create(
        cls,
        knowledge_id: str,
        source_reliability: float,
        recency_score: float,
        corroboration_score: float,
        consistency_score: float,
        assessment_notes: str = "",
    ) -> "KnowledgeConfidence":
        overall = (
            source_reliability * 0.35
            + recency_score * 0.20
            + corroboration_score * 0.25
            + consistency_score * 0.20
        )
        return cls(
            id=_new_id(),
            knowledge_id=knowledge_id,
            overall_score=max(0.0, min(1.0, overall)),
            source_reliability=max(0.0, min(1.0, source_reliability)),
            recency_score=max(0.0, min(1.0, recency_score)),
            corroboration_score=max(0.0, min(1.0, corroboration_score)),
            consistency_score=max(0.0, min(1.0, consistency_score)),
            assessed_at=_utcnow(),
            assessment_notes=assessment_notes,
        )


@dataclass(frozen=True)
class KnowledgeContradiction:
    """
    A detected contradiction between two knowledge records.
    Managed by the Knowledge Integrity Engine.
    """
    id: str
    record_a_id: str
    record_b_id: str
    contradiction_description: str
    severity: ValidationSeverity
    detected_at: datetime
    resolution_status: str = "unresolved"   # "unresolved" | "resolved" | "acknowledged"
    resolution_notes: str = ""

    @property
    def is_resolved(self) -> bool:
        return self.resolution_status == "resolved"

    @property
    def is_blocking(self) -> bool:
        return self.severity.is_blocking() and not self.is_resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "record_a_id": self.record_a_id,
            "record_b_id": self.record_b_id,
            "contradiction_description": self.contradiction_description,
            "severity": self.severity.value,
            "detected_at": self.detected_at.isoformat(),
            "resolution_status": self.resolution_status,
            "resolution_notes": self.resolution_notes,
        }

    @classmethod
    def create(
        cls,
        record_a_id: str,
        record_b_id: str,
        contradiction_description: str,
        severity: ValidationSeverity = ValidationSeverity.WARNING,
    ) -> "KnowledgeContradiction":
        return cls(
            id=_new_id(),
            record_a_id=record_a_id,
            record_b_id=record_b_id,
            contradiction_description=contradiction_description,
            severity=severity,
            detected_at=_utcnow(),
        )


@dataclass(frozen=True)
class KnowledgeEvidence:
    """
    A piece of evidence supporting or challenging a knowledge claim.
    """
    id: str
    knowledge_id: str
    evidence_type: str                 # "supporting" | "challenging" | "neutral"
    source: str
    source_type: KnowledgeSourceType
    description: str
    confidence: float                  # 0.0 – 1.0
    collected_at: datetime

    @property
    def is_supporting(self) -> bool:
        return self.evidence_type == "supporting"

    @property
    def is_challenging(self) -> bool:
        return self.evidence_type == "challenging"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "knowledge_id": self.knowledge_id,
            "evidence_type": self.evidence_type,
            "source": self.source,
            "source_type": self.source_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "collected_at": self.collected_at.isoformat(),
        }

    @classmethod
    def create(
        cls,
        knowledge_id: str,
        evidence_type: str,
        source: str,
        description: str,
        source_type: KnowledgeSourceType = KnowledgeSourceType.UNKNOWN,
        confidence: float = 0.75,
    ) -> "KnowledgeEvidence":
        return cls(
            id=_new_id(),
            knowledge_id=knowledge_id,
            evidence_type=evidence_type,
            source=source,
            source_type=source_type,
            description=description,
            confidence=max(0.0, min(1.0, confidence)),
            collected_at=_utcnow(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeSynthesis:
    """
    A synthesis operation that combines knowledge from multiple sources
    or domains into a coherent, integrated understanding.

    Used heavily by ORION.

    Example:
        Robotics + AI + Control Systems + Embedded Systems
        → Autonomous Robotics Knowledge Package
    """
    id: str
    name: str
    description: str
    source_domain_ids: list[str]
    source_knowledge_ids: list[str]
    synthesized_concept_ids: list[str] = field(default_factory=list)
    synthesized_fact_ids: list[str] = field(default_factory=list)
    synthesis_rationale: str = ""
    integration_points: list[str] = field(default_factory=list)
    emergent_insights: list[str] = field(default_factory=list)
    confidence_score: float = 0.70
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    @property
    def domain_count(self) -> int:
        return len(self.source_domain_ids)

    @property
    def is_cross_domain(self) -> bool:
        return self.domain_count > 1

    @property
    def is_rich(self) -> bool:
        return len(self.emergent_insights) > 0 and len(self.integration_points) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_domain_ids": self.source_domain_ids,
            "source_knowledge_ids": self.source_knowledge_ids,
            "synthesized_concept_ids": self.synthesized_concept_ids,
            "synthesized_fact_ids": self.synthesized_fact_ids,
            "synthesis_rationale": self.synthesis_rationale,
            "integration_points": self.integration_points,
            "emergent_insights": self.emergent_insights,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        source_domain_ids: list[str],
        source_knowledge_ids: list[str],
        synthesis_rationale: str = "",
    ) -> "KnowledgeSynthesis":
        return cls(
            id=_new_id(),
            name=name,
            description=description,
            source_domain_ids=source_domain_ids,
            source_knowledge_ids=source_knowledge_ids,
            synthesis_rationale=synthesis_rationale,
        )


@dataclass
class KnowledgePackage:
    """
    A curated, self-contained bundle of knowledge assembled for a
    specific purpose or consumer module.

    Produced by the Knowledge Synthesis Engine for ORION, APOLLO, etc.

    Example: "Autonomous Robotics Knowledge Package" assembled
    for reasoning about a drone-building goal.
    """
    id: str
    name: str
    description: str
    purpose: str
    target_consumer: str               # e.g. "ORION", "APOLLO", "VULCAN"
    domain_ids: list[str] = field(default_factory=list)
    concept_ids: list[str] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    skill_ids: list[str] = field(default_factory=list)
    procedure_ids: list[str] = field(default_factory=list)
    research_ids: list[str] = field(default_factory=list)
    educational_ids: list[str] = field(default_factory=list)
    synthesis_id: Optional[str] = None
    difficulty: KnowledgeDifficulty = KnowledgeDifficulty.INTERMEDIATE
    estimated_size_kb: Optional[float] = None
    assembled_at: datetime = field(default_factory=_utcnow)
    is_complete: bool = False

    @property
    def total_items(self) -> int:
        return (
            len(self.concept_ids)
            + len(self.fact_ids)
            + len(self.skill_ids)
            + len(self.procedure_ids)
            + len(self.research_ids)
            + len(self.educational_ids)
        )

    @property
    def is_multi_domain(self) -> bool:
        return len(self.domain_ids) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "purpose": self.purpose,
            "target_consumer": self.target_consumer,
            "domain_ids": self.domain_ids,
            "concept_ids": self.concept_ids,
            "fact_ids": self.fact_ids,
            "skill_ids": self.skill_ids,
            "procedure_ids": self.procedure_ids,
            "research_ids": self.research_ids,
            "educational_ids": self.educational_ids,
            "synthesis_id": self.synthesis_id,
            "difficulty": self.difficulty.value,
            "estimated_size_kb": self.estimated_size_kb,
            "assembled_at": self.assembled_at.isoformat(),
            "is_complete": self.is_complete,
        }

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        purpose: str,
        target_consumer: str,
        difficulty: KnowledgeDifficulty = KnowledgeDifficulty.INTERMEDIATE,
    ) -> "KnowledgePackage":
        return cls(
            id=_new_id(),
            name=name,
            description=description,
            purpose=purpose,
            target_consumer=target_consumer,
            difficulty=difficulty,
        )


@dataclass(frozen=True)
class KnowledgeComposition:
    """
    Metadata describing how a KnowledgePackage was assembled —
    which source records were included, excluded, and why.
    """
    id: str
    package_id: str
    included_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    inclusion_rationale: str
    exclusion_rationale: str
    composition_strategy: str           # "exhaustive" | "curated" | "threshold" | "requested"
    confidence_threshold: float
    composed_at: datetime

    @property
    def inclusion_count(self) -> int:
        return len(self.included_ids)

    @property
    def exclusion_count(self) -> int:
        return len(self.excluded_ids)

    @property
    def inclusion_rate(self) -> float:
        total = self.inclusion_count + self.exclusion_count
        if total == 0:
            return 0.0
        return self.inclusion_count / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "package_id": self.package_id,
            "included_ids": list(self.included_ids),
            "excluded_ids": list(self.excluded_ids),
            "inclusion_rationale": self.inclusion_rationale,
            "exclusion_rationale": self.exclusion_rationale,
            "composition_strategy": self.composition_strategy,
            "confidence_threshold": self.confidence_threshold,
            "composed_at": self.composed_at.isoformat(),
        }

    @classmethod
    def create(
        cls,
        package_id: str,
        included_ids: list[str],
        excluded_ids: list[str],
        inclusion_rationale: str,
        exclusion_rationale: str,
        composition_strategy: str = "curated",
        confidence_threshold: float = 0.60,
    ) -> "KnowledgeComposition":
        return cls(
            id=_new_id(),
            package_id=package_id,
            included_ids=tuple(included_ids),
            excluded_ids=tuple(excluded_ids),
            inclusion_rationale=inclusion_rationale,
            exclusion_rationale=exclusion_rationale,
            composition_strategy=composition_strategy,
            confidence_threshold=confidence_threshold,
            composed_at=_utcnow(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# INDEXING MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KnowledgeIndexEntry:
    """
    A lightweight index entry providing fast lookup for any knowledge record.
    Maintained by the Knowledge Index Engine.
    """
    id: str
    knowledge_id: str
    knowledge_type: KnowledgeType
    name: str
    name_lower: str                    # Pre-lowercased for fast search
    aliases_lower: tuple[str, ...]
    domain_ids: tuple[str, ...]
    tags: tuple[str, ...]
    difficulty: KnowledgeDifficulty
    status: KnowledgeStatus
    confidence_score: float
    fingerprint: str
    indexed_at: datetime

    @property
    def is_searchable(self) -> bool:
        return self.status.is_usable

    def matches_query(self, query: str) -> bool:
        q = query.lower().strip()
        if q in self.name_lower:
            return True
        return any(q in alias for alias in self.aliases_lower)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "knowledge_id": self.knowledge_id,
            "knowledge_type": self.knowledge_type.value,
            "name": self.name,
            "domain_ids": list(self.domain_ids),
            "tags": list(self.tags),
            "difficulty": self.difficulty.value,
            "status": self.status.value,
            "confidence_score": self.confidence_score,
            "fingerprint": self.fingerprint,
            "indexed_at": self.indexed_at.isoformat(),
        }

    @classmethod
    def from_record(cls, record: KnowledgeRecord) -> "KnowledgeIndexEntry":
        return cls(
            id=_new_id(),
            knowledge_id=record.id,
            knowledge_type=record.knowledge_type,
            name=record.name,
            name_lower=record.name.lower(),
            aliases_lower=tuple(a.lower() for a in record.aliases),
            domain_ids=tuple(record.domain_ids),
            tags=record.metadata.tags,
            difficulty=record.difficulty,
            status=record.status,
            confidence_score=record.metadata.confidence_score,
            fingerprint=record.fingerprint,
            indexed_at=_utcnow(),
        )


@dataclass(frozen=True)
class ConceptIndexEntry:
    """Index entry specialized for Concepts — includes concept type and relationships."""
    id: str
    concept_id: str
    name: str
    name_lower: str
    concept_type: ConceptType
    domain_ids: tuple[str, ...]
    prerequisite_concept_ids: tuple[str, ...]
    child_concept_ids: tuple[str, ...]
    is_foundational: bool
    difficulty: KnowledgeDifficulty
    indexed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "concept_id": self.concept_id,
            "name": self.name,
            "concept_type": self.concept_type.value,
            "domain_ids": list(self.domain_ids),
            "prerequisite_concept_ids": list(self.prerequisite_concept_ids),
            "child_concept_ids": list(self.child_concept_ids),
            "is_foundational": self.is_foundational,
            "difficulty": self.difficulty.value,
            "indexed_at": self.indexed_at.isoformat(),
        }

    @classmethod
    def from_concept(cls, concept: Concept) -> "ConceptIndexEntry":
        return cls(
            id=_new_id(),
            concept_id=concept.id,
            name=concept.name,
            name_lower=concept.name.lower(),
            concept_type=concept.concept_type,
            domain_ids=tuple(concept.domain_ids),
            prerequisite_concept_ids=tuple(concept.prerequisite_concept_ids),
            child_concept_ids=tuple(concept.child_concept_ids),
            is_foundational=concept.is_foundational,
            difficulty=concept.difficulty,
            indexed_at=_utcnow(),
        )


@dataclass(frozen=True)
class DomainIndexEntry:
    """Index entry specialized for KnowledgeDomains."""
    id: str
    domain_id: str
    name: str
    name_lower: str
    parent_domain_id: Optional[str]
    sub_domain_count: int
    concept_count: int
    skill_count: int
    fact_count: int
    is_root_domain: bool
    indexed_at: datetime

    @property
    def total_items(self) -> int:
        return self.concept_count + self.skill_count + self.fact_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain_id": self.domain_id,
            "name": self.name,
            "parent_domain_id": self.parent_domain_id,
            "sub_domain_count": self.sub_domain_count,
            "concept_count": self.concept_count,
            "skill_count": self.skill_count,
            "fact_count": self.fact_count,
            "is_root_domain": self.is_root_domain,
            "indexed_at": self.indexed_at.isoformat(),
        }

    @classmethod
    def from_domain(cls, domain: KnowledgeDomain) -> "DomainIndexEntry":
        return cls(
            id=_new_id(),
            domain_id=domain.id,
            name=domain.name,
            name_lower=domain.name.lower(),
            parent_domain_id=domain.parent_domain_id,
            sub_domain_count=len(domain.sub_domain_ids),
            concept_count=len(domain.core_concept_ids),
            skill_count=len(domain.core_skill_ids),
            fact_count=len(domain.core_fact_ids),
            is_root_domain=domain.is_root_domain,
            indexed_at=_utcnow(),
        )


@dataclass(frozen=True)
class SkillIndexEntry:
    """Index entry specialized for Skills."""
    id: str
    skill_id: str
    name: str
    name_lower: str
    skill_type: SkillType
    domain_ids: tuple[str, ...]
    prerequisite_skill_ids: tuple[str, ...]
    has_progression_model: bool
    difficulty: KnowledgeDifficulty
    indexed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill_id": self.skill_id,
            "name": self.name,
            "skill_type": self.skill_type.value,
            "domain_ids": list(self.domain_ids),
            "prerequisite_skill_ids": list(self.prerequisite_skill_ids),
            "has_progression_model": self.has_progression_model,
            "difficulty": self.difficulty.value,
            "indexed_at": self.indexed_at.isoformat(),
        }

    @classmethod
    def from_skill(cls, skill: Skill) -> "SkillIndexEntry":
        return cls(
            id=_new_id(),
            skill_id=skill.id,
            name=skill.name,
            name_lower=skill.name.lower(),
            skill_type=skill.skill_type,
            domain_ids=tuple(skill.domain_ids),
            prerequisite_skill_ids=tuple(skill.prerequisite_skill_ids),
            has_progression_model=skill.has_progression_model,
            difficulty=skill.difficulty,
            indexed_at=_utcnow(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRITY MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IntegrityIssueType(Enum):
    """Classification of a knowledge integrity issue."""
    DUPLICATE_CONCEPT = "duplicate_concept"
    DUPLICATE_FACT = "duplicate_fact"
    CONFLICTING_FACT = "conflicting_fact"
    BROKEN_REFERENCE = "broken_reference"
    ORPHANED_RECORD = "orphaned_record"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    MISSING_DOMAIN = "missing_domain"
    STALE_INDEX = "stale_index"
    CORRUPTED_HIERARCHY = "corrupted_hierarchy"
    INVALID_PROGRESSION_MODEL = "invalid_progression_model"
    BROKEN_SKILL_PREREQUISITE = "broken_skill_prerequisite"


@dataclass(frozen=True)
class IntegrityIssue:
    """
    A single integrity problem found in the LUNA knowledge store.
    Produced by the Knowledge Integrity Engine.
    """
    id: str
    issue_type: IntegrityIssueType
    severity: ValidationSeverity
    affected_id: str                   # ID of the affected record
    affected_type: KnowledgeType
    description: str
    conflicting_id: Optional[str] = None   # ID of the conflicting record, if any
    auto_resolvable: bool = False
    resolution_hint: str = ""
    detected_at: datetime = field(default_factory=_utcnow)

    @property
    def requires_human_review(self) -> bool:
        return not self.auto_resolvable and self.severity.is_blocking()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "affected_id": self.affected_id,
            "affected_type": self.affected_type.value,
            "description": self.description,
            "conflicting_id": self.conflicting_id,
            "auto_resolvable": self.auto_resolvable,
            "resolution_hint": self.resolution_hint,
            "detected_at": self.detected_at.isoformat(),
        }

    @classmethod
    def create(
        cls,
        issue_type: IntegrityIssueType,
        severity: ValidationSeverity,
        affected_id: str,
        affected_type: KnowledgeType,
        description: str,
        conflicting_id: Optional[str] = None,
        auto_resolvable: bool = False,
        resolution_hint: str = "",
    ) -> "IntegrityIssue":
        return cls(
            id=_new_id(),
            issue_type=issue_type,
            severity=severity,
            affected_id=affected_id,
            affected_type=affected_type,
            description=description,
            conflicting_id=conflicting_id,
            auto_resolvable=auto_resolvable,
            resolution_hint=resolution_hint,
        )


@dataclass
class IntegrityReport:
    """
    A full integrity scan report for the LUNA knowledge store.
    Produced by the Knowledge Integrity Engine.
    """
    id: str
    issues: list[IntegrityIssue]
    records_scanned: int
    scan_duration_ms: float
    scanned_at: datetime
    scan_version: str
    summary: str = ""

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0

    @property
    def has_blocking_issues(self) -> bool:
        return any(i.severity.is_blocking() for i in self.issues)

    @property
    def auto_resolvable_count(self) -> int:
        return sum(1 for i in self.issues if i.auto_resolvable)

    @property
    def requires_human_review_count(self) -> int:
        return sum(1 for i in self.issues if i.requires_human_review)

    def issues_by_type(self, issue_type: IntegrityIssueType) -> list[IntegrityIssue]:
        return [i for i in self.issues if i.issue_type == issue_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "issues": [i.to_dict() for i in self.issues],
            "records_scanned": self.records_scanned,
            "scan_duration_ms": self.scan_duration_ms,
            "scanned_at": self.scanned_at.isoformat(),
            "scan_version": self.scan_version,
            "summary": self.summary,
            "issue_count": self.issue_count,
            "critical_count": self.critical_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }

    @classmethod
    def create(
        cls,
        issues: list[IntegrityIssue],
        records_scanned: int,
        scan_duration_ms: float,
        scan_version: str,
        summary: str = "",
    ) -> "IntegrityReport":
        return cls(
            id=_new_id(),
            issues=issues,
            records_scanned=records_scanned,
            scan_duration_ms=scan_duration_ms,
            scanned_at=_utcnow(),
            scan_version=scan_version,
            summary=summary,
        )


@dataclass
class KnowledgeAuditReport:
    """
    A comprehensive audit report covering validation, confidence,
    integrity, and coverage across the entire LUNA knowledge store.

    Combines outputs from the Validation, Integrity, and Index engines
    into a single executive summary for POLARIS health monitoring.
    """
    id: str
    integrity_report: IntegrityReport
    validation_results: list[KnowledgeValidationResult]
    contradiction_count: int
    total_records: int
    active_records: int
    deprecated_records: int
    validated_records: int
    average_confidence: float
    domain_coverage: dict[str, int]    # domain_name → record count
    skill_coverage: dict[str, SkillLevel]  # domain_name → max skill level
    low_confidence_ids: list[str]
    stale_ids: list[str]
    generated_at: datetime
    audit_version: str
    notes: str = ""

    @property
    def validation_pass_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.validated_records / self.total_records

    @property
    def active_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.active_records / self.total_records

    @property
    def health_score(self) -> float:
        """
        Composite knowledge store health score (0.0 – 1.0).

        Weighted combination of:
            - Validation pass rate (30%)
            - Average confidence (30%)
            - Active record rate (20%)
            - Integrity cleanliness (20%)
        """
        integrity_score = 1.0 if self.integrity_report.is_clean else max(
            0.0, 1.0 - (self.integrity_report.issue_count / max(1, self.total_records))
        )
        return (
            self.validation_pass_rate * 0.30
            + self.average_confidence * 0.30
            + self.active_rate * 0.20
            + integrity_score * 0.20
        )

    @property
    def health_label(self) -> str:
        score = self.health_score
        if score >= 0.90:
            return "Excellent"
        if score >= 0.75:
            return "Good"
        if score >= 0.55:
            return "Fair"
        if score >= 0.35:
            return "Poor"
        return "Critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "integrity_report": self.integrity_report.to_dict(),
            "validation_results_count": len(self.validation_results),
            "contradiction_count": self.contradiction_count,
            "total_records": self.total_records,
            "active_records": self.active_records,
            "deprecated_records": self.deprecated_records,
            "validated_records": self.validated_records,
            "average_confidence": self.average_confidence,
            "domain_coverage": self.domain_coverage,
            "low_confidence_count": len(self.low_confidence_ids),
            "stale_count": len(self.stale_ids),
            "generated_at": self.generated_at.isoformat(),
            "audit_version": self.audit_version,
            "health_score": round(self.health_score, 4),
            "health_label": self.health_label,
            "notes": self.notes,
        }

    @classmethod
    def create(
        cls,
        integrity_report: IntegrityReport,
        validation_results: list[KnowledgeValidationResult],
        contradiction_count: int,
        total_records: int,
        active_records: int,
        deprecated_records: int,
        validated_records: int,
        average_confidence: float,
        domain_coverage: dict[str, int],
        skill_coverage: dict[str, SkillLevel],
        low_confidence_ids: list[str],
        stale_ids: list[str],
        audit_version: str,
        notes: str = "",
    ) -> "KnowledgeAuditReport":
        return cls(
            id=_new_id(),
            integrity_report=integrity_report,
            validation_results=validation_results,
            contradiction_count=contradiction_count,
            total_records=total_records,
            active_records=active_records,
            deprecated_records=deprecated_records,
            validated_records=validated_records,
            average_confidence=max(0.0, min(1.0, average_confidence)),
            domain_coverage=domain_coverage,
            skill_coverage=skill_coverage,
            low_confidence_ids=low_confidence_ids,
            stale_ids=stale_ids,
            generated_at=_utcnow(),
            audit_version=audit_version,
            notes=notes,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Utilities
    "_utcnow",
    "_new_id",
    "_stable_hash",
    # Core Enums
    "KnowledgeType",
    "FactType",
    "ConceptType",
    "SkillType",
    "ProcedureType",
    "ResearchType",
    "EducationType",
    "KnowledgeDifficulty",
    "KnowledgeStatus",
    "KnowledgeSourceType",
    "ValidationStatus",
    "ConfidenceLevel",
    "SkillLevel",
    # Metadata
    "KnowledgeMetadata",
    # Core Domain Models
    "KnowledgeRecord",
    "Fact",
    "Concept",
    "Skill",
    "KnowledgeDomain",
    "Procedure",
    "ProcedureStep",
    "ResearchKnowledge",
    "EducationalKnowledge",
    "ContentSection",
    # Structural Models
    "ConceptRelationshipType",
    "ConceptRelationship",
    "KnowledgeReference",
    "KnowledgeDependency",
    "SemanticNode",
    "SemanticHierarchy",
    "DomainStructure",
    # Skill Models
    "SkillStage",
    "SkillProgressionModel",
    "SkillPrerequisite",
    # Validation Models
    "ValidationIssueType",
    "ValidationSeverity",
    "KnowledgeValidationResult",
    "ValidationIssue",
    "KnowledgeConfidence",
    "KnowledgeContradiction",
    "KnowledgeEvidence",
    # Synthesis Models
    "KnowledgeSynthesis",
    "KnowledgePackage",
    "KnowledgeComposition",
    # Indexing Models
    "KnowledgeIndexEntry",
    "ConceptIndexEntry",
    "DomainIndexEntry",
    "SkillIndexEntry",
    # Integrity Models
    "IntegrityIssueType",
    "IntegrityIssue",
    "IntegrityReport",
    "KnowledgeAuditReport",
]