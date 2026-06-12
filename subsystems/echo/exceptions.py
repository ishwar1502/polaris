# subsystems/echo/exceptions.py
"""
ECHO v1 Exception Hierarchy.

All exceptions raised by ECHO subsystem code inherit from
:class:`EchoError` so callers can catch the entire ECHO error taxonomy
with a single clause while still distinguishing specific failure modes.

Hierarchy
---------
EchoError
├── EchoNotInitializedError
├── ExperienceError
│   ├── ExperienceNotFoundError
│   ├── ExperienceValidationError
│   ├── ExperienceDuplicateError
│   └── ExperienceStorageError
├── SignificanceError
│   ├── SignificanceScoringError
│   └── BelowSignificanceThresholdError
├── EventError
│   ├── EventNotFoundError
│   └── EventValidationError
├── AchievementError
│   ├── AchievementNotFoundError
│   └── AchievementValidationError
├── FailureError
│   ├── FailureNotFoundError
│   └── FailureValidationError
├── ObservationError
│   ├── ObservationNotFoundError
│   └── ObservationValidationError
├── MemoryIntegrityError
│   ├── DuplicateExperienceError
│   ├── BrokenReferenceError
│   └── MemoryCorruptionError
└── EchoBoundaryViolationError
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class EchoError(Exception):
    """Root exception for all ECHO subsystem errors.

    Catch this to handle any ECHO failure without needing to enumerate
    every specific error type.
    """


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class EchoNotInitializedError(EchoError):
    """Raised when an ECHO operation is called before the subsystem is running."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"ECHO is not running. Cannot execute '{operation}'. "
            "Call initialize() and start() first."
        )
        self.operation = operation


# ---------------------------------------------------------------------------
# Experience Errors
# ---------------------------------------------------------------------------


class ExperienceError(EchoError):
    """Base class for all Experience Engine errors."""


class ExperienceNotFoundError(ExperienceError):
    """Raised when an experience_id does not correspond to any stored experience."""

    def __init__(self, experience_id: str) -> None:
        super().__init__(f"Experience '{experience_id}' not found.")
        self.experience_id = experience_id


class ExperienceValidationError(ExperienceError):
    """Raised when experience data fails domain validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class ExperienceDuplicateError(ExperienceError):
    """Raised when the Memory Integrity Engine detects a duplicate experience.

    A duplicate is defined as an experience with a matching ``experience_id``
    or an identical (title, experience_type, occurred_at) triple.
    """

    def __init__(self, experience_id: str) -> None:
        super().__init__(
            f"An experience with id '{experience_id}' already exists. "
            "Use update semantics if you intend to modify it."
        )
        self.experience_id = experience_id


class ExperienceStorageError(ExperienceError):
    """Raised when a persistence operation for an experience fails."""

    def __init__(self, experience_id: str, reason: str) -> None:
        super().__init__(
            f"Failed to store experience '{experience_id}': {reason}"
        )
        self.experience_id = experience_id
        self.reason = reason


# ---------------------------------------------------------------------------
# Significance Errors
# ---------------------------------------------------------------------------


class SignificanceError(EchoError):
    """Base class for all Significance Engine errors."""


class SignificanceScoringError(SignificanceError):
    """Raised when the Significance Engine cannot produce a valid score.

    This may occur when required context fields are absent or when
    scoring inputs fall outside expected ranges.
    """

    def __init__(self, message: str, *, experience_id: str | None = None) -> None:
        super().__init__(message)
        self.experience_id = experience_id


class BelowSignificanceThresholdError(SignificanceError):
    """Raised when a caller attempts to force-store an experience that the
    Significance Engine has classified as below the minimum threshold.

    This is a soft gate — callers may override with ``force=True`` on the
    Experience Engine's store method.

    Attributes
    ----------
    experience_id:
        The experience that failed the threshold check.
    score:
        The computed significance score (0.0-1.0).
    threshold:
        The minimum score required for automatic storage.
    """

    def __init__(
        self,
        experience_id: str,
        score: float,
        threshold: float,
    ) -> None:
        super().__init__(
            f"Experience '{experience_id}' scored {score:.3f}, "
            f"below the significance threshold of {threshold:.3f}. "
            "Pass force=True to store anyway."
        )
        self.experience_id = experience_id
        self.score = score
        self.threshold = threshold


# ---------------------------------------------------------------------------
# Event Errors
# ---------------------------------------------------------------------------


class EventError(EchoError):
    """Base class for all Event Engine errors."""


class EventNotFoundError(EventError):
    """Raised when an event_id does not correspond to any stored event."""

    def __init__(self, event_id: str) -> None:
        super().__init__(f"Event '{event_id}' not found.")
        self.event_id = event_id


class EventValidationError(EventError):
    """Raised when event data fails domain validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


# ---------------------------------------------------------------------------
# Achievement Errors
# ---------------------------------------------------------------------------


class AchievementError(EchoError):
    """Base class for all Achievement Engine errors."""


class AchievementNotFoundError(AchievementError):
    """Raised when an achievement_id does not correspond to any stored record."""

    def __init__(self, achievement_id: str) -> None:
        super().__init__(f"Achievement '{achievement_id}' not found.")
        self.achievement_id = achievement_id


class AchievementValidationError(AchievementError):
    """Raised when achievement data fails domain validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


# ---------------------------------------------------------------------------
# Failure Errors
# ---------------------------------------------------------------------------


class FailureError(EchoError):
    """Base class for all Failure Analysis Engine errors."""


class FailureNotFoundError(FailureError):
    """Raised when a failure_id does not correspond to any stored record."""

    def __init__(self, failure_id: str) -> None:
        super().__init__(f"Failure record '{failure_id}' not found.")
        self.failure_id = failure_id


class FailureValidationError(FailureError):
    """Raised when failure data fails domain validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


# ---------------------------------------------------------------------------
# Observation Errors
# ---------------------------------------------------------------------------


class ObservationError(EchoError):
    """Base class for all Observation errors."""


class ObservationNotFoundError(ObservationError):
    """Raised when an observation_id does not correspond to any stored record."""

    def __init__(self, observation_id: str) -> None:
        super().__init__(f"Observation '{observation_id}' not found.")
        self.observation_id = observation_id


class ObservationValidationError(ObservationError):
    """Raised when observation data fails domain validation constraints."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


# ---------------------------------------------------------------------------
# Memory Integrity Errors
# ---------------------------------------------------------------------------


class MemoryIntegrityError(EchoError):
    """Base class for Memory Integrity Engine errors.

    These errors indicate structural inconsistencies in ECHO's memory store
    that must be resolved before continued operation.
    """


class DuplicateExperienceError(MemoryIntegrityError):
    """Raised when the integrity check detects duplicate experience records.

    Distinct from :class:`ExperienceDuplicateError` — this is raised
    *post-hoc* by the Memory Integrity Engine during a consistency scan,
    rather than at insertion time.
    """

    def __init__(self, experience_id: str, duplicate_count: int) -> None:
        super().__init__(
            f"Memory integrity violation: experience '{experience_id}' "
            f"exists {duplicate_count} times. Deduplication required."
        )
        self.experience_id = experience_id
        self.duplicate_count = duplicate_count


class BrokenReferenceError(MemoryIntegrityError):
    """Raised when a record contains a reference to a non-existent experience."""

    def __init__(self, source_id: str, missing_ref: str) -> None:
        super().__init__(
            f"Broken reference in '{source_id}': referenced experience "
            f"'{missing_ref}' does not exist."
        )
        self.source_id = source_id
        self.missing_ref = missing_ref


class MemoryCorruptionError(MemoryIntegrityError):
    """Raised when the Memory Integrity Engine detects corrupted data that
    cannot be automatically repaired."""

    def __init__(self, message: str, *, affected_id: str | None = None) -> None:
        super().__init__(message)
        self.affected_id = affected_id


# ---------------------------------------------------------------------------
# Boundary Violation
# ---------------------------------------------------------------------------


class EchoBoundaryViolationError(EchoError):
    """Raised when a caller attempts to store data that ECHO does not own.

    ECHO owns: Experiences, Events, Conversations, Sessions, Achievements,
    Failures, Observations, Activity History, Personal History.

    ECHO does NOT own: Knowledge, Identity, Goals, Schedules,
    Relationships, Decisions.  Any attempt to store these will raise this
    exception, directing the caller to the correct subsystem.

    Attributes
    ----------
    attempted_type:
        The data type or label the caller tried to store.
    correct_subsystem:
        The subsystem that should own this data type.
    """

    def __init__(self, attempted_type: str, correct_subsystem: str) -> None:
        super().__init__(
            f"ECHO boundary violation: '{attempted_type}' does not belong to ECHO. "
            f"This data is owned by {correct_subsystem}. "
            "Store it through the correct subsystem interface."
        )
        self.attempted_type = attempted_type
        self.correct_subsystem = correct_subsystem