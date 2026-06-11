# subsystems/echo/__init__.py
"""
ECHO Episodic Memory Core — Public API.

All public types, engines, models, exceptions, and serialisation helpers
exported by the ECHO subsystem are re-exported from this module.  External
packages must import exclusively from ``subsystems.echo``; internal module
paths are private implementation details.

Quick-start
-----------
::

    from subsystems.echo import EchoSubsystem, ExperienceType, ExperienceImportance

    echo = EchoSubsystem()
    echo.initialize()

    exp = echo.experience_engine.create_experience(
        title="Completed POLARIS milestone",
        experience_type=ExperienceType.ACHIEVEMENT,
        importance=ExperienceImportance.HIGH,
    )

    echo.shutdown()
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Root subsystem
# ---------------------------------------------------------------------------

from subsystems.echo.echo import (
    EchoSubsystem,
    EchoSubsystemStatus,
)

# ---------------------------------------------------------------------------
# Core engines — implemented (Tier 0–5 per dependency order)
# ---------------------------------------------------------------------------

# Tier 0
from subsystems.echo.significance import SignificanceEngine

# Tier 1
from subsystems.echo.experience import ExperienceEngine

# Tier 2
from subsystems.echo.retrieval import ExperienceRetrievalEngine
from subsystems.echo.consolidation import MemoryConsolidationEngine

# Tier 3
from subsystems.echo.integrity import MemoryIntegrityEngine
from subsystems.echo.episodic_index import EpisodicIndexEngine
from subsystems.echo.reflection import ReflectionEngine

# Tier 4
from subsystems.echo.context_reconstruction import ContextReconstructionEngine
from subsystems.echo.patterns import PatternExtractionEngine

# Tier 5
from subsystems.echo.personal_history import PersonalHistoryEngine

# Domain engines (concrete implementations now available)
from subsystems.echo.events import EventEngine
from subsystems.echo.conversation import ConversationEngine
from subsystems.echo.session import SessionEngine
from subsystems.echo.achievements import AchievementEngine
from subsystems.echo.failure_analysis import FailureAnalysisEngine

# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

from subsystems.echo.models import (
    # Enumerations
    ExperienceType,
    ExperienceImportance,
    # Value types
    MemoryTag,
    ExperienceMetadata,
    # Domain records
    Experience,
    EventRecord,
    AchievementRecord,
    FailureRecord,
    ObservationRecord,
)

# Conversation and session domain records (defined in their engine modules)
from subsystems.echo.conversation import ConversationRecord
from subsystems.echo.session import (
    SessionRecord,
    SessionState,
    SessionHistoryEntry,
)

# ---------------------------------------------------------------------------
# Serialisation helpers (schemas.py — canonical to/from dict layer)
# ---------------------------------------------------------------------------

from subsystems.echo.schemas import (
    # MemoryTag
    memory_tag_to_dict,
    memory_tag_from_dict,
    # ExperienceMetadata
    experience_metadata_to_dict,
    experience_metadata_from_dict,
    # Experience
    experience_to_dict,
    experience_from_dict,
    # EventRecord
    event_record_to_dict,
    event_record_from_dict,
    # AchievementRecord
    achievement_record_to_dict,
    achievement_record_from_dict,
    # FailureRecord
    failure_record_to_dict,
    failure_record_from_dict,
    # ObservationRecord
    observation_record_to_dict,
    observation_record_from_dict,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

from subsystems.echo.exceptions import (
    # Root
    EchoError,
    EchoNotInitializedError,
    EchoBoundaryViolationError,
    # Experience
    ExperienceError,
    ExperienceNotFoundError,
    ExperienceValidationError,
    ExperienceDuplicateError,
    ExperienceStorageError,
    # Significance
    SignificanceError,
    SignificanceScoringError,
    BelowSignificanceThresholdError,
    # Event
    EventError,
    EventNotFoundError,
    EventValidationError,
    # Achievement
    AchievementError,
    AchievementNotFoundError,
    AchievementValidationError,
    # Failure
    FailureError,
    FailureNotFoundError,
    FailureValidationError,
    # Observation
    ObservationError,
    ObservationNotFoundError,
    ObservationValidationError,
    # Memory integrity
    MemoryIntegrityError,
    DuplicateExperienceError,
    BrokenReferenceError,
    MemoryCorruptionError,
)

# Conversation and session exceptions (defined in their engine modules)
from subsystems.echo.conversation import (
    ConversationError,
    ConversationNotFoundError,
    ConversationValidationError,
    ConversationDuplicateError,
)
from subsystems.echo.session import (
    SessionError,
    SessionNotFoundError,
    SessionValidationError,
    SessionDuplicateError,
    SessionStateError,
)

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -----------------------------------------------------------------------
    # Subsystem
    # -----------------------------------------------------------------------
    "EchoSubsystem",
    "EchoSubsystemStatus",

    # -----------------------------------------------------------------------
    # Engines
    # -----------------------------------------------------------------------
    # Tier 0
    "SignificanceEngine",
    # Tier 1
    "ExperienceEngine",
    # Tier 2
    "ExperienceRetrievalEngine",
    "MemoryConsolidationEngine",
    # Tier 3
    "MemoryIntegrityEngine",
    "EpisodicIndexEngine",
    "ReflectionEngine",
    # Tier 4
    "ContextReconstructionEngine",
    "PatternExtractionEngine",
    # Tier 5
    "PersonalHistoryEngine",
    # Domain engines
    "EventEngine",
    "ConversationEngine",
    "SessionEngine",
    "AchievementEngine",
    "FailureAnalysisEngine",

    # -----------------------------------------------------------------------
    # Models — enumerations
    # -----------------------------------------------------------------------
    "ExperienceType",
    "ExperienceImportance",

    # Models — value types
    "MemoryTag",
    "ExperienceMetadata",

    # Models — domain records
    "Experience",
    "EventRecord",
    "AchievementRecord",
    "FailureRecord",
    "ObservationRecord",
    "ConversationRecord",
    "SessionRecord",
    "SessionState",
    "SessionHistoryEntry",

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------
    "memory_tag_to_dict",
    "memory_tag_from_dict",
    "experience_metadata_to_dict",
    "experience_metadata_from_dict",
    "experience_to_dict",
    "experience_from_dict",
    "event_record_to_dict",
    "event_record_from_dict",
    "achievement_record_to_dict",
    "achievement_record_from_dict",
    "failure_record_to_dict",
    "failure_record_from_dict",
    "observation_record_to_dict",
    "observation_record_from_dict",

    # -----------------------------------------------------------------------
    # Exceptions — root
    # -----------------------------------------------------------------------
    "EchoError",
    "EchoNotInitializedError",
    "EchoBoundaryViolationError",

    # Exceptions — experience
    "ExperienceError",
    "ExperienceNotFoundError",
    "ExperienceValidationError",
    "ExperienceDuplicateError",
    "ExperienceStorageError",

    # Exceptions — significance
    "SignificanceError",
    "SignificanceScoringError",
    "BelowSignificanceThresholdError",

    # Exceptions — event
    "EventError",
    "EventNotFoundError",
    "EventValidationError",

    # Exceptions — achievement
    "AchievementError",
    "AchievementNotFoundError",
    "AchievementValidationError",

    # Exceptions — failure
    "FailureError",
    "FailureNotFoundError",
    "FailureValidationError",

    # Exceptions — observation
    "ObservationError",
    "ObservationNotFoundError",
    "ObservationValidationError",

    # Exceptions — memory integrity
    "MemoryIntegrityError",
    "DuplicateExperienceError",
    "BrokenReferenceError",
    "MemoryCorruptionError",

    # Exceptions — conversation
    "ConversationError",
    "ConversationNotFoundError",
    "ConversationValidationError",
    "ConversationDuplicateError",

    # Exceptions — session
    "SessionError",
    "SessionNotFoundError",
    "SessionValidationError",
    "SessionDuplicateError",
    "SessionStateError",
]
