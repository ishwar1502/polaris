"""
LUNA v5 — Semantic Knowledge Core
Module: subsystems/luna/exceptions.py

Structured, production-ready exception hierarchy for all LUNA engines.

LUNA owns facts, concepts, skills, knowledge domains, procedural knowledge,
research knowledge, educational knowledge, skill models, and semantic
structures.  It does NOT own identity, experience, relationships, or time.

Exception taxonomy mirrors the LUNA internal architecture:

    LunaError
    └── KnowledgeError
        ├── FactError
        ├── ConceptError
        ├── SkillError
        ├── KnowledgeDomainError
        ├── ProcedureError
        ├── ResearchKnowledgeError
        ├── EducationalKnowledgeError
        ├── KnowledgeValidationError
        ├── KnowledgeSynthesisError
        ├── KnowledgeRetrievalError
        ├── KnowledgeIntegrityError
        └── SemanticStructureError

    LunaError (subsystem branch)
        ├── LunaNotInitializedError
        ├── LunaLifecycleError
        └── LunaBoundaryViolationError

Part of the POLARIS Cognitive Substrate:
    ASTRA        → Identity
    ECHO         → Experience
    LUNA         → Knowledge        ← this module
    CHRONOS      → Time
    CONSTELLATION → Relationships
"""

from __future__ import annotations

from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# ROOT EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class LunaError(Exception):
    """
    Root exception for all LUNA subsystem errors.

    Every exception raised by LUNA inherits from this class, enabling
    callers to catch the entire LUNA exception surface with a single
    ``except LunaError`` clause.

    Attributes:
        message:  Human-readable description of the failure.
        context:  Arbitrary key-value metadata that helps diagnose the error
                  (e.g. record IDs, engine name, confidence scores).
        cause:    The underlying exception that triggered this one, if any.
                  Mirrors ``__cause__`` but is also accessible as a plain
                  attribute for structured logging.
    """

    def __init__(
        self,
        message: str,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = context or {}
        self.cause: Optional[BaseException] = cause
        if cause is not None:
            self.__cause__ = cause

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def with_context(self, **kwargs: Any) -> "LunaError":
        """Return *self* after merging extra key-value pairs into context."""
        self.context.update(kwargs)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialise the exception to a plain dictionary for structured logs."""
        return {
            "exception": type(self).__name__,
            "message": self.message,
            "context": self.context,
            "cause": str(self.cause) if self.cause is not None else None,
        }

    def __repr__(self) -> str:
        ctx = f", context={self.context!r}" if self.context else ""
        return f"{type(self).__name__}({self.message!r}{ctx})"

    def __str__(self) -> str:
        if self.context:
            pairs = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{pairs}]"
        return self.message


class KnowledgeError(LunaError):
    """
    Base class for all errors that concern a knowledge record or the
    knowledge store itself.

    Separates data-layer failures from subsystem lifecycle failures so that
    callers can distinguish between "something went wrong with the knowledge"
    and "LUNA itself is not operational".
    """


# ─────────────────────────────────────────────────────────────────────────────
# FACT EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class FactError(KnowledgeError):
    """
    Base class for all errors raised by the Fact Engine.

    Facts are atomic truth claims owned exclusively by LUNA
    (e.g. Ohm's Law, Newton's Laws, Python syntax rules).
    """


class FactNotFoundError(FactError):
    """
    Raised when a requested fact cannot be located in the knowledge store.

    Args:
        fact_id:    The ID or lookup key that produced no result.
        message:    Optional override for the default message.
        context:    Supplementary metadata (query type, domain, etc.).
        cause:      Underlying exception, if any.

    Example::

        raise FactNotFoundError(
            fact_id="fact-ohm-law-001",
            context={"domain": "electronics", "query_type": "direct_id"},
        )
    """

    def __init__(
        self,
        fact_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = {"fact_id": fact_id, **(context or {})}
        super().__init__(
            message or f"Fact not found: '{fact_id}'",
            context=ctx,
            cause=cause,
        )
        self.fact_id: str = fact_id


class FactValidationError(FactError):
    """
    Raised when a fact fails structural or semantic validation.

    Args:
        fact_id:        The ID of the offending fact record.
        violations:     List of human-readable violation descriptions.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.

    Example::

        raise FactValidationError(
            fact_id="fact-draft-xyz",
            violations=["statement is empty", "confidence score out of range"],
        )
    """

    def __init__(
        self,
        fact_id: str,
        violations: Optional[list[str]] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.fact_id: str = fact_id
        self.violations: list[str] = violations or []
        ctx = {
            "fact_id": fact_id,
            "violations": self.violations,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Fact validation failed for '{fact_id}': "
                + ("; ".join(self.violations) if self.violations else "unspecified violations")
            ),
            context=ctx,
            cause=cause,
        )


class DuplicateFactError(FactError):
    """
    Raised when an attempt is made to add a fact that already exists in the
    knowledge store, either by ID or by semantic fingerprint.

    Args:
        fact_id:        The ID of the incoming (duplicate) fact.
        existing_id:    The ID of the already-stored conflicting fact.
        message:        Optional override for the default message.
        context:        Supplementary metadata (fingerprint hash, domain, etc.).
        cause:          Underlying exception, if any.

    Example::

        raise DuplicateFactError(
            fact_id="fact-new-abc",
            existing_id="fact-ohm-law-001",
            context={"fingerprint": "a3f9..."},
        )
    """

    def __init__(
        self,
        fact_id: str,
        existing_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.fact_id: str = fact_id
        self.existing_id: str = existing_id
        ctx = {
            "fact_id": fact_id,
            "existing_id": existing_id,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Duplicate fact detected: incoming '{fact_id}' "
                f"conflicts with existing '{existing_id}'"
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class ConceptError(KnowledgeError):
    """
    Base class for all errors raised by the Concept Engine.

    Concepts are ideas and abstractions (e.g. Machine Learning, Robotics,
    Control Systems).  They form knowledge domains and are owned exclusively
    by LUNA; CONSTELLATION may *link* concepts but never owns them.
    """


class ConceptNotFoundError(ConceptError):
    """
    Raised when a requested concept cannot be located in the knowledge store.

    Args:
        concept_id:     The ID or name that produced no result.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        concept_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = {"concept_id": concept_id, **(context or {})}
        super().__init__(
            message or f"Concept not found: '{concept_id}'",
            context=ctx,
            cause=cause,
        )
        self.concept_id: str = concept_id


class ConceptValidationError(ConceptError):
    """
    Raised when a concept record fails structural or semantic validation.

    Args:
        concept_id:     The ID of the offending concept.
        violations:     List of human-readable violation descriptions.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        concept_id: str,
        violations: Optional[list[str]] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.concept_id: str = concept_id
        self.violations: list[str] = violations or []
        ctx = {
            "concept_id": concept_id,
            "violations": self.violations,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Concept validation failed for '{concept_id}': "
                + ("; ".join(self.violations) if self.violations else "unspecified violations")
            ),
            context=ctx,
            cause=cause,
        )


class DuplicateConceptError(ConceptError):
    """
    Raised when an attempt is made to add a concept that already exists.

    Args:
        concept_id:     The ID of the incoming concept.
        existing_id:    The ID of the already-stored conflicting concept.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        concept_id: str,
        existing_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.concept_id: str = concept_id
        self.existing_id: str = existing_id
        ctx = {
            "concept_id": concept_id,
            "existing_id": existing_id,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Duplicate concept detected: incoming '{concept_id}' "
                f"conflicts with existing '{existing_id}'"
            ),
            context=ctx,
            cause=cause,
        )


class ConceptRelationshipError(ConceptError):
    """
    Raised when a relationship between two concepts is invalid, missing, or
    would violate the semantic structure of the knowledge graph.

    Note: LUNA owns concepts; CONSTELLATION links them.  This exception
    concerns the *ownership* layer — malformed relationship definitions stored
    inside LUNA — not the CONSTELLATION link layer.

    Args:
        source_id:      ID of the source concept.
        target_id:      ID of the target concept.
        relationship:   Name or type of the attempted relationship.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.source_id: str = source_id
        self.target_id: str = target_id
        self.relationship: str = relationship
        ctx = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship": relationship,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Invalid concept relationship '{relationship}' "
                f"from '{source_id}' to '{target_id}'"
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SKILL EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class SkillError(KnowledgeError):
    """
    Base class for all errors raised by the Skill Engine.

    LUNA owns the skill definition (what the skill *is*).
    ASTRA owns the user's proficiency profile (how well the user performs it).
    """


class SkillNotFoundError(SkillError):
    """
    Raised when a requested skill cannot be located in the knowledge store.

    Args:
        skill_id:   The ID or name that produced no result.
        message:    Optional override for the default message.
        context:    Supplementary metadata.
        cause:      Underlying exception, if any.
    """

    def __init__(
        self,
        skill_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = {"skill_id": skill_id, **(context or {})}
        super().__init__(
            message or f"Skill not found: '{skill_id}'",
            context=ctx,
            cause=cause,
        )
        self.skill_id: str = skill_id


class SkillValidationError(SkillError):
    """
    Raised when a skill record fails structural or semantic validation.

    Args:
        skill_id:       The ID of the offending skill.
        violations:     List of human-readable violation descriptions.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        skill_id: str,
        violations: Optional[list[str]] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.skill_id: str = skill_id
        self.violations: list[str] = violations or []
        ctx = {
            "skill_id": skill_id,
            "violations": self.violations,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Skill validation failed for '{skill_id}': "
                + ("; ".join(self.violations) if self.violations else "unspecified violations")
            ),
            context=ctx,
            cause=cause,
        )


class DuplicateSkillError(SkillError):
    """
    Raised when an attempt is made to add a skill that already exists.

    Args:
        skill_id:       The ID of the incoming skill.
        existing_id:    The ID of the already-stored conflicting skill.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        skill_id: str,
        existing_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.skill_id: str = skill_id
        self.existing_id: str = existing_id
        ctx = {
            "skill_id": skill_id,
            "existing_id": existing_id,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Duplicate skill detected: incoming '{skill_id}' "
                f"conflicts with existing '{existing_id}'"
            ),
            context=ctx,
            cause=cause,
        )


class SkillProgressionError(SkillError):
    """
    Raised when the Skill Progression Engine encounters an invalid or
    inconsistent progression model (e.g. missing stage, broken prerequisite
    chain, non-monotonic difficulty ordering).

    Note: This tracks skill *structure* — not user proficiency, which is
    owned by ASTRA.

    Args:
        skill_id:       The ID of the skill whose progression model is broken.
        stage:          The stage name or label where the problem was detected.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        skill_id: str,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.skill_id: str = skill_id
        self.stage: Optional[str] = stage
        ctx = {
            "skill_id": skill_id,
            **({"stage": stage} if stage is not None else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Skill progression error for '{skill_id}'"
                + (f" at stage '{stage}'" if stage else "")
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeDomainError(KnowledgeError):
    """
    Base class for all errors raised by the Knowledge Domain Engine.

    Domains organise major disciplines (AI, Robotics, Electronics,
    Mathematics, Business, …) and provide structural hierarchy for
    the entire knowledge store.
    """


class DomainNotFoundError(KnowledgeDomainError):
    """
    Raised when a requested knowledge domain cannot be located.

    Args:
        domain_id:  The ID or name that produced no result.
        message:    Optional override for the default message.
        context:    Supplementary metadata.
        cause:      Underlying exception, if any.
    """

    def __init__(
        self,
        domain_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = {"domain_id": domain_id, **(context or {})}
        super().__init__(
            message or f"Knowledge domain not found: '{domain_id}'",
            context=ctx,
            cause=cause,
        )
        self.domain_id: str = domain_id


class DomainValidationError(KnowledgeDomainError):
    """
    Raised when a domain record fails structural or semantic validation.

    Args:
        domain_id:      The ID of the offending domain.
        violations:     List of human-readable violation descriptions.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        domain_id: str,
        violations: Optional[list[str]] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.domain_id: str = domain_id
        self.violations: list[str] = violations or []
        ctx = {
            "domain_id": domain_id,
            "violations": self.violations,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Domain validation failed for '{domain_id}': "
                + ("; ".join(self.violations) if self.violations else "unspecified violations")
            ),
            context=ctx,
            cause=cause,
        )


class DuplicateDomainError(KnowledgeDomainError):
    """
    Raised when an attempt is made to register a domain that already exists.

    Args:
        domain_id:      The ID of the incoming domain.
        existing_id:    The ID of the already-stored conflicting domain.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        domain_id: str,
        existing_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.domain_id: str = domain_id
        self.existing_id: str = existing_id
        ctx = {
            "domain_id": domain_id,
            "existing_id": existing_id,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Duplicate domain detected: incoming '{domain_id}' "
                f"conflicts with existing '{existing_id}'"
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PROCEDURAL KNOWLEDGE EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class ProcedureError(KnowledgeError):
    """
    Base class for all errors raised by the Procedural Knowledge Engine.

    Procedures encode *how-to* knowledge (e.g. how to tune a PID controller,
    how to deploy ROS2) and are critical for execution planning by ORION.
    """


class ProcedureNotFoundError(ProcedureError):
    """
    Raised when a requested procedure cannot be located in the knowledge store.

    Args:
        procedure_id:   The ID or name that produced no result.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        procedure_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = {"procedure_id": procedure_id, **(context or {})}
        super().__init__(
            message or f"Procedure not found: '{procedure_id}'",
            context=ctx,
            cause=cause,
        )
        self.procedure_id: str = procedure_id


class ProcedureValidationError(ProcedureError):
    """
    Raised when a procedure record fails structural or semantic validation
    (e.g. missing steps, broken step ordering, invalid prerequisite reference).

    Args:
        procedure_id:   The ID of the offending procedure.
        violations:     List of human-readable violation descriptions.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        procedure_id: str,
        violations: Optional[list[str]] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.procedure_id: str = procedure_id
        self.violations: list[str] = violations or []
        ctx = {
            "procedure_id": procedure_id,
            "violations": self.violations,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Procedure validation failed for '{procedure_id}': "
                + ("; ".join(self.violations) if self.violations else "unspecified violations")
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH KNOWLEDGE EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class ResearchKnowledgeError(KnowledgeError):
    """
    Base class for all errors raised by the Research Knowledge Engine.

    Research knowledge is sourced from papers, technical reports,
    documentation, and experiments.  It is consumed heavily by PROMETHEUS
    and VULCAN.
    """


class ResearchKnowledgeNotFoundError(ResearchKnowledgeError):
    """
    Raised when a requested research knowledge record cannot be located.

    Args:
        research_id:    The ID or reference key that produced no result.
        message:        Optional override for the default message.
        context:        Supplementary metadata (source type, DOI, etc.).
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        research_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = {"research_id": research_id, **(context or {})}
        super().__init__(
            message or f"Research knowledge record not found: '{research_id}'",
            context=ctx,
            cause=cause,
        )
        self.research_id: str = research_id


class ResearchValidationError(ResearchKnowledgeError):
    """
    Raised when a research knowledge record fails validation.

    Args:
        research_id:    The ID of the offending record.
        violations:     List of human-readable violation descriptions.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        research_id: str,
        violations: Optional[list[str]] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.research_id: str = research_id
        self.violations: list[str] = violations or []
        ctx = {
            "research_id": research_id,
            "violations": self.violations,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Research knowledge validation failed for '{research_id}': "
                + ("; ".join(self.violations) if self.violations else "unspecified violations")
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATIONAL KNOWLEDGE EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class EducationalKnowledgeError(KnowledgeError):
    """
    Base class for all errors raised by the Educational Knowledge Engine.

    Educational content (learning paths, curricula, difficulty levels,
    prerequisites) is stored by LUNA and consumed primarily by APOLLO.
    """


class EducationalKnowledgeNotFoundError(EducationalKnowledgeError):
    """
    Raised when a requested educational knowledge record cannot be located.

    Args:
        content_id:     The ID or name that produced no result.
        message:        Optional override for the default message.
        context:        Supplementary metadata (content type, target audience, etc.).
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        content_id: str,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = {"content_id": content_id, **(context or {})}
        super().__init__(
            message or f"Educational knowledge record not found: '{content_id}'",
            context=ctx,
            cause=cause,
        )
        self.content_id: str = content_id


class EducationalKnowledgeValidationError(EducationalKnowledgeError):
    """
    Raised when an educational knowledge record fails validation
    (e.g. broken prerequisite reference, missing difficulty level, empty
    learning path).

    Args:
        content_id:     The ID of the offending record.
        violations:     List of human-readable violation descriptions.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        content_id: str,
        violations: Optional[list[str]] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.content_id: str = content_id
        self.violations: list[str] = violations or []
        ctx = {
            "content_id": content_id,
            "violations": self.violations,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Educational knowledge validation failed for '{content_id}': "
                + ("; ".join(self.violations) if self.violations else "unspecified violations")
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeValidationError(KnowledgeError):
    """
    Base class for all errors raised by the Knowledge Validation Engine.

    The Validation Engine verifies source reliability, confidence scores,
    knowledge age, and contradiction detection.  It is critical for reducing
    hallucinations in POLARIS.
    """


class KnowledgeConfidenceError(KnowledgeValidationError):
    """
    Raised when a knowledge record's confidence score is absent, out of range,
    or falls below the minimum threshold required for activation.

    Args:
        record_id:          The ID of the knowledge record.
        confidence:         The actual confidence value encountered (may be None
                            if entirely absent).
        minimum_required:   The threshold that was not met, if applicable.
        message:            Optional override for the default message.
        context:            Supplementary metadata.
        cause:              Underlying exception, if any.
    """

    def __init__(
        self,
        record_id: str,
        confidence: Optional[float] = None,
        minimum_required: Optional[float] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.record_id: str = record_id
        self.confidence: Optional[float] = confidence
        self.minimum_required: Optional[float] = minimum_required
        ctx: dict[str, Any] = {"record_id": record_id}
        if confidence is not None:
            ctx["confidence"] = confidence
        if minimum_required is not None:
            ctx["minimum_required"] = minimum_required
        ctx.update(context or {})
        if message is None:
            if confidence is not None and minimum_required is not None:
                message = (
                    f"Confidence {confidence:.3f} for record '{record_id}' "
                    f"is below the required minimum of {minimum_required:.3f}"
                )
            elif confidence is not None:
                message = (
                    f"Invalid confidence score {confidence!r} "
                    f"for record '{record_id}'"
                )
            else:
                message = f"Confidence score missing for record '{record_id}'"
        super().__init__(message, context=ctx, cause=cause)


class KnowledgeContradictionError(KnowledgeValidationError):
    """
    Raised when two knowledge records make mutually incompatible claims and
    the contradiction cannot be automatically resolved.

    Args:
        record_id_a:    ID of the first conflicting record.
        record_id_b:    ID of the second conflicting record.
        description:    Human-readable description of the contradiction.
        message:        Optional override for the default message.
        context:        Supplementary metadata (domain, conflict type, etc.).
        cause:          Underlying exception, if any.

    Example::

        raise KnowledgeContradictionError(
            record_id_a="fact-voltage-001",
            record_id_b="fact-voltage-002",
            description="Both records define Ohm's Law with contradictory units",
        )
    """

    def __init__(
        self,
        record_id_a: str,
        record_id_b: str,
        description: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.record_id_a: str = record_id_a
        self.record_id_b: str = record_id_b
        self.description: str = description
        ctx = {
            "record_id_a": record_id_a,
            "record_id_b": record_id_b,
            **({"description": description} if description else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Knowledge contradiction between '{record_id_a}' and '{record_id_b}'"
                + (f": {description}" if description else "")
            ),
            context=ctx,
            cause=cause,
        )


class KnowledgeEvidenceError(KnowledgeValidationError):
    """
    Raised when a knowledge record lacks sufficient evidence to be accepted
    into the validated knowledge store, or when its cited evidence is
    unreachable, malformed, or below the minimum trust threshold.

    Args:
        record_id:          The ID of the knowledge record.
        evidence_ids:       IDs or references of the problematic evidence items.
        message:            Optional override for the default message.
        context:            Supplementary metadata (source type, trust weight, etc.).
        cause:              Underlying exception, if any.
    """

    def __init__(
        self,
        record_id: str,
        evidence_ids: Optional[list[str]] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.record_id: str = record_id
        self.evidence_ids: list[str] = evidence_ids or []
        ctx = {
            "record_id": record_id,
            **({"evidence_ids": self.evidence_ids} if self.evidence_ids else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Insufficient or invalid evidence for record '{record_id}'"
                + (
                    f" (problematic evidence: {self.evidence_ids})"
                    if self.evidence_ids else ""
                )
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeSynthesisError(KnowledgeError):
    """
    Base class for all errors raised by the Knowledge Synthesis Engine.

    The Synthesis Engine combines knowledge from multiple domains to produce
    unified knowledge packages (e.g. Robotics + AI + Control Systems →
    Autonomous Robotics Knowledge Package).  It is consumed heavily by ORION.
    """


class KnowledgeCompositionError(KnowledgeSynthesisError):
    """
    Raised when individual knowledge units cannot be composed into a coherent
    synthesis artefact (e.g. incompatible domains, missing bridge concept,
    circular dependency between components).

    Args:
        component_ids:  IDs of the knowledge records involved in the failed
                        composition.
        reason:         Human-readable description of the composition failure.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        component_ids: Optional[list[str]] = None,
        reason: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.component_ids: list[str] = component_ids or []
        self.reason: str = reason
        ctx = {
            **({"component_ids": self.component_ids} if self.component_ids else {}),
            **({"reason": reason} if reason else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                "Knowledge composition failed"
                + (f" for components {self.component_ids}" if self.component_ids else "")
                + (f": {reason}" if reason else "")
            ),
            context=ctx,
            cause=cause,
        )


class KnowledgePackageError(KnowledgeSynthesisError):
    """
    Raised when a knowledge package is malformed, incomplete, or cannot be
    assembled from its constituent synthesis units.

    Args:
        package_id:     The ID of the offending package.
        reason:         Human-readable description of the problem.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        package_id: str,
        reason: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.package_id: str = package_id
        self.reason: str = reason
        ctx = {
            "package_id": package_id,
            **({"reason": reason} if reason else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Knowledge package error for '{package_id}'"
                + (f": {reason}" if reason else "")
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeRetrievalError(KnowledgeError):
    """
    Base class for all errors raised by the Knowledge Retrieval Engine.

    The Retrieval Engine answers semantic queries ("What is a PID controller?",
    "Explain ROS2 nodes.") and must return results quickly.
    """


class KnowledgeSearchError(KnowledgeRetrievalError):
    """
    Raised when a knowledge search query fails — due to a malformed query,
    an internal engine error, or an inability to produce any meaningful result.

    Args:
        query:          The query string or descriptor that caused the failure.
        reason:         Human-readable description of why the search failed.
        message:        Optional override for the default message.
        context:        Supplementary metadata (engine, filters, etc.).
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        query: str,
        reason: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.query: str = query
        self.reason: str = reason
        ctx = {
            "query": query,
            **({"reason": reason} if reason else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Knowledge search failed for query '{query}'"
                + (f": {reason}" if reason else "")
            ),
            context=ctx,
            cause=cause,
        )


class KnowledgeIndexError(KnowledgeRetrievalError):
    """
    Raised when the Knowledge Index Engine encounters a corruption, miss, or
    inconsistency that prevents fast-access index operations.

    Args:
        index_key:      The topic, domain, skill, or concept key that triggered
                        the error.
        reason:         Human-readable description of the index problem.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        index_key: str,
        reason: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.index_key: str = index_key
        self.reason: str = reason
        ctx = {
            "index_key": index_key,
            **({"reason": reason} if reason else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Knowledge index error for key '{index_key}'"
                + (f": {reason}" if reason else "")
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRITY EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeIntegrityError(KnowledgeError):
    """
    Base class for all errors raised by the Knowledge Integrity Engine.

    The Integrity Engine is a critical subsystem that protects knowledge
    consistency by preventing duplicate concepts, conflicting facts,
    corrupted knowledge, and broken references.
    """


class DuplicateKnowledgeError(KnowledgeIntegrityError):
    """
    Raised when a general-purpose duplicate knowledge record is detected that
    does not belong to a more specific category (fact, concept, skill, domain).

    Prefer the specific sub-exceptions (``DuplicateFactError``,
    ``DuplicateConceptError``, etc.) where possible.

    Args:
        record_id:      The ID of the incoming (duplicate) record.
        existing_id:    The ID of the already-stored conflicting record.
        record_type:    The knowledge type label (e.g. "procedure", "research").
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        record_id: str,
        existing_id: str,
        record_type: str = "knowledge",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.record_id: str = record_id
        self.existing_id: str = existing_id
        self.record_type: str = record_type
        ctx = {
            "record_id": record_id,
            "existing_id": existing_id,
            "record_type": record_type,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Duplicate {record_type} detected: incoming '{record_id}' "
                f"conflicts with existing '{existing_id}'"
            ),
            context=ctx,
            cause=cause,
        )


class BrokenKnowledgeReferenceError(KnowledgeIntegrityError):
    """
    Raised when a knowledge record contains a reference (to a fact, concept,
    skill, domain, procedure, etc.) that no longer resolves — either because
    the target was deleted, archived, or never existed.

    Args:
        source_id:      The ID of the record that contains the broken reference.
        reference_id:   The ID that cannot be resolved.
        reference_type: Optional label describing what the reference points to
                        (e.g. "concept", "skill", "domain").
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        source_id: str,
        reference_id: str,
        reference_type: str = "record",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.source_id: str = source_id
        self.reference_id: str = reference_id
        self.reference_type: str = reference_type
        ctx = {
            "source_id": source_id,
            "reference_id": reference_id,
            "reference_type": reference_type,
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Broken {reference_type} reference in '{source_id}': "
                f"target '{reference_id}' cannot be resolved"
            ),
            context=ctx,
            cause=cause,
        )


class KnowledgeCorruptionError(KnowledgeIntegrityError):
    """
    Raised when a knowledge record or the knowledge store itself is found to
    be in a corrupted state — meaning its internal data is structurally
    inconsistent, partially written, or has failed a checksum/hash
    verification.

    Args:
        record_id:      The ID of the corrupted record (or a store-level
                        identifier if the corruption is store-wide).
        details:        Human-readable description of the corruption.
        message:        Optional override for the default message.
        context:        Supplementary metadata (checksum expected/actual, etc.).
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        record_id: str,
        details: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.record_id: str = record_id
        self.details: str = details
        ctx = {
            "record_id": record_id,
            **({"details": details} if details else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Knowledge corruption detected in record '{record_id}'"
                + (f": {details}" if details else "")
            ),
            context=ctx,
            cause=cause,
        )


class KnowledgeAuditError(KnowledgeIntegrityError):
    """
    Raised when a knowledge audit run fails to complete, encounters an
    unrecoverable error mid-scan, or produces a report that itself is invalid.

    Args:
        audit_id:       The ID of the audit run that failed.
        phase:          The audit phase during which the failure occurred
                        (e.g. "scanning", "reporting", "validation").
        reason:         Human-readable description of the failure.
        message:        Optional override for the default message.
        context:        Supplementary metadata (records_scanned, engine, etc.).
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        audit_id: str,
        phase: str = "",
        reason: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.audit_id: str = audit_id
        self.phase: str = phase
        self.reason: str = reason
        ctx = {
            "audit_id": audit_id,
            **({"phase": phase} if phase else {}),
            **({"reason": reason} if reason else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Knowledge audit '{audit_id}' failed"
                + (f" during '{phase}'" if phase else "")
                + (f": {reason}" if reason else "")
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC STRUCTURE EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class SemanticStructureError(KnowledgeError):
    """
    Base class for all errors raised by the Semantic Structure Engine.

    The Semantic Structure Engine builds and maintains the logical hierarchy
    of knowledge (e.g. Control Systems → PID, State Space, Feedback, Stability).
    """


class SemanticHierarchyError(SemanticStructureError):
    """
    Raised when the semantic hierarchy is invalid — for example, a parent
    node is missing, the hierarchy depth exceeds the permitted maximum, or
    a rearrangement would destroy existing child relationships.

    Args:
        hierarchy_id:   The ID of the affected hierarchy.
        reason:         Human-readable description of the problem.
        message:        Optional override for the default message.
        context:        Supplementary metadata (depth, parent_id, etc.).
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        hierarchy_id: str,
        reason: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.hierarchy_id: str = hierarchy_id
        self.reason: str = reason
        ctx = {
            "hierarchy_id": hierarchy_id,
            **({"reason": reason} if reason else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                f"Semantic hierarchy error in '{hierarchy_id}'"
                + (f": {reason}" if reason else "")
            ),
            context=ctx,
            cause=cause,
        )


class SemanticNodeError(SemanticStructureError):
    """
    Raised when a semantic node within a hierarchy is missing, malformed,
    or cannot be linked to its parent or children.

    Args:
        node_id:        The ID of the problematic semantic node.
        hierarchy_id:   The ID of the containing hierarchy, if known.
        reason:         Human-readable description of the problem.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.
    """

    def __init__(
        self,
        node_id: str,
        hierarchy_id: Optional[str] = None,
        reason: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.node_id: str = node_id
        self.hierarchy_id: Optional[str] = hierarchy_id
        self.reason: str = reason
        ctx: dict[str, Any] = {"node_id": node_id}
        if hierarchy_id is not None:
            ctx["hierarchy_id"] = hierarchy_id
        if reason:
            ctx["reason"] = reason
        ctx.update(context or {})
        super().__init__(
            message or (
                f"Semantic node error for node '{node_id}'"
                + (f" in hierarchy '{hierarchy_id}'" if hierarchy_id else "")
                + (f": {reason}" if reason else "")
            ),
            context=ctx,
            cause=cause,
        )


class RelationshipCycleError(SemanticStructureError):
    """
    Raised when adding or modifying a relationship would introduce a cycle
    into what must remain a directed acyclic graph (DAG) — for instance,
    making Concept A a child of Concept B when B is already a descendant of A.

    Args:
        node_ids:       Ordered list of node IDs that form the cycle, starting
                        and ending at the same node (if the full cycle path is
                        known).
        hierarchy_id:   The ID of the containing hierarchy, if known.
        message:        Optional override for the default message.
        context:        Supplementary metadata.
        cause:          Underlying exception, if any.

    Example::

        raise RelationshipCycleError(
            node_ids=["concept-A", "concept-B", "concept-C", "concept-A"],
            hierarchy_id="hierarchy-control-systems",
        )
    """

    def __init__(
        self,
        node_ids: Optional[list[str]] = None,
        hierarchy_id: Optional[str] = None,
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.node_ids: list[str] = node_ids or []
        self.hierarchy_id: Optional[str] = hierarchy_id
        ctx: dict[str, Any] = {}
        if self.node_ids:
            ctx["node_ids"] = self.node_ids
        if hierarchy_id is not None:
            ctx["hierarchy_id"] = hierarchy_id
        ctx.update(context or {})
        cycle_str = " → ".join(self.node_ids) if self.node_ids else "unknown cycle"
        super().__init__(
            message or (
                f"Relationship cycle detected: {cycle_str}"
                + (f" in hierarchy '{hierarchy_id}'" if hierarchy_id else "")
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SUBSYSTEM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class LunaNotInitializedError(LunaError):
    """
    Raised when an operation is attempted on the LUNA subsystem before it has
    been fully initialised.

    This is a lifecycle guard: callers must ensure that LUNA has completed its
    startup sequence before issuing knowledge queries or writes.

    Args:
        operation:  The name of the operation that was attempted.
        message:    Optional override for the default message.
        context:    Supplementary metadata (engine name, init phase, etc.).
        cause:      Underlying exception, if any.

    Example::

        raise LunaNotInitializedError(operation="retrieve_fact")
    """

    def __init__(
        self,
        operation: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.operation: str = operation
        ctx = {
            **({"operation": operation} if operation else {}),
            **(context or {}),
        }
        super().__init__(
            message or (
                "LUNA subsystem is not initialised"
                + (f": cannot execute '{operation}'" if operation else "")
            ),
            context=ctx,
            cause=cause,
        )


class LunaLifecycleError(LunaError):
    """
    Raised when a LUNA lifecycle transition fails or is attempted out of order
    (e.g. trying to shut down an engine that has not started, or re-initialising
    an already-running subsystem without a proper teardown).

    Args:
        phase:      The lifecycle phase that failed (e.g. "startup", "shutdown",
                    "restart", "teardown").
        engine:     The name of the specific engine involved, if applicable.
        reason:     Human-readable description of why the transition failed.
        message:    Optional override for the default message.
        context:    Supplementary metadata.
        cause:      Underlying exception, if any.
    """

    def __init__(
        self,
        phase: str,
        engine: str = "",
        reason: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.phase: str = phase
        self.engine: str = engine
        self.reason: str = reason
        ctx: dict[str, Any] = {"phase": phase}
        if engine:
            ctx["engine"] = engine
        if reason:
            ctx["reason"] = reason
        ctx.update(context or {})
        super().__init__(
            message or (
                f"LUNA lifecycle error during '{phase}'"
                + (f" in engine '{engine}'" if engine else "")
                + (f": {reason}" if reason else "")
            ),
            context=ctx,
            cause=cause,
        )


class LunaBoundaryViolationError(LunaError):
    """
    Raised when an external module attempts to write to, delete from, or
    assume ownership of knowledge artefacts that belong exclusively to LUNA.

    LUNA's ownership laws are absolute:

    - Law 1: LUNA owns knowledge.  Nobody else.
    - Law 2: LUNA owns concepts.  CONSTELLATION only links them.
    - Law 3: LUNA owns skills.    ASTRA owns skill tendencies.
    - Law 4: LUNA stores truth claims.  ECHO stores experiences.
    - Law 5: LUNA stores what is known.  ECHO stores what happened.

    Args:
        violating_module:   Name of the module that attempted the violation.
        attempted_action:   Description of the action that was blocked
                            (e.g. "write concept", "delete fact", "claim skill ownership").
        target_id:          The ID of the knowledge record the violation concerned.
        law:                The LUNA ownership law that was violated (e.g. "Law 1").
        message:            Optional override for the default message.
        context:            Supplementary metadata.
        cause:              Underlying exception, if any.

    Example::

        raise LunaBoundaryViolationError(
            violating_module="CONSTELLATION",
            attempted_action="write concept",
            target_id="concept-pid-control",
            law="Law 2",
        )
    """

    def __init__(
        self,
        violating_module: str,
        attempted_action: str,
        target_id: str = "",
        law: str = "",
        message: Optional[str] = None,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.violating_module: str = violating_module
        self.attempted_action: str = attempted_action
        self.target_id: str = target_id
        self.law: str = law
        ctx: dict[str, Any] = {
            "violating_module": violating_module,
            "attempted_action": attempted_action,
        }
        if target_id:
            ctx["target_id"] = target_id
        if law:
            ctx["law"] = law
        ctx.update(context or {})
        super().__init__(
            message or (
                f"LUNA boundary violation: '{violating_module}' attempted to "
                f"'{attempted_action}'"
                + (f" on record '{target_id}'" if target_id else "")
                + (f" — violates LUNA {law}" if law else "")
            ),
            context=ctx,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Root
    "LunaError",
    "KnowledgeError",
    # Fact Engine
    "FactError",
    "FactNotFoundError",
    "FactValidationError",
    "DuplicateFactError",
    # Concept Engine
    "ConceptError",
    "ConceptNotFoundError",
    "ConceptValidationError",
    "DuplicateConceptError",
    "ConceptRelationshipError",
    # Skill Engine
    "SkillError",
    "SkillNotFoundError",
    "SkillValidationError",
    "DuplicateSkillError",
    "SkillProgressionError",
    # Domain Engine
    "KnowledgeDomainError",
    "DomainNotFoundError",
    "DomainValidationError",
    "DuplicateDomainError",
    # Procedural Knowledge Engine
    "ProcedureError",
    "ProcedureNotFoundError",
    "ProcedureValidationError",
    # Research Knowledge Engine
    "ResearchKnowledgeError",
    "ResearchKnowledgeNotFoundError",
    "ResearchValidationError",
    # Educational Knowledge Engine
    "EducationalKnowledgeError",
    "EducationalKnowledgeNotFoundError",
    "EducationalKnowledgeValidationError",
    # Knowledge Validation Engine
    "KnowledgeValidationError",
    "KnowledgeConfidenceError",
    "KnowledgeContradictionError",
    "KnowledgeEvidenceError",
    # Knowledge Synthesis Engine
    "KnowledgeSynthesisError",
    "KnowledgeCompositionError",
    "KnowledgePackageError",
    # Knowledge Retrieval Engine
    "KnowledgeRetrievalError",
    "KnowledgeSearchError",
    "KnowledgeIndexError",
    # Knowledge Integrity Engine
    "KnowledgeIntegrityError",
    "DuplicateKnowledgeError",
    "BrokenKnowledgeReferenceError",
    "KnowledgeCorruptionError",
    "KnowledgeAuditError",
    # Semantic Structure Engine
    "SemanticStructureError",
    "SemanticHierarchyError",
    "SemanticNodeError",
    "RelationshipCycleError",
    # Subsystem Lifecycle
    "LunaNotInitializedError",
    "LunaLifecycleError",
    "LunaBoundaryViolationError",
]
