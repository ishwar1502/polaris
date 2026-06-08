# subsystems/astra/__init__.py
"""
ASTRA v5 — Digital Twin Core of POLARIS.

Public API for the ASTRA subsystem.  Import everything you need from here;
never reach into the internal modules directly.

Quick-start
-----------
>>> from subsystems.astra import AstraSubsystem
>>> astra = AstraSubsystem()
>>> astra.initialize()
>>> astra.start()
>>> astra.update_identity({"name": "Heisenberg", "core_identity_tags": ["builder"]})
>>> twin = astra.generate_twin()
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Subsystem entry point
# ---------------------------------------------------------------------------

from subsystems.astra.astra import AstraSubsystem

# ---------------------------------------------------------------------------
# Engine classes
# ---------------------------------------------------------------------------

from subsystems.astra.digital_twin import DigitalTwinEngine
from subsystems.astra.capability import CapabilityManager, AstraCapabilityRegistry as CapabilityRegistry
from subsystems.astra.identity import IdentityEngine
from subsystems.astra.goals import GoalEngine
from subsystems.astra.preferences import PreferenceEngine
from subsystems.astra.consistency import ConsistencyEngine
from subsystems.astra.evolution import IdentityEvolutionEngine

# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------

from subsystems.astra.models import (
    # Enumerations
    GoalState,
    GoalPriority,
    GoalType,
    EvolutionTrigger,
    # Core models
    IdentityProfile,
    Goal,
    PreferenceProfile,
    CommunicationPreferences,
    LearningPreferences,
    DevelopmentPreferences,
    WorkflowPreferences,
    CapabilityEntry,
    CapabilitySnapshot,
    DigitalTwinState,
    EvolutionRecord,
    EvidenceItem,
    ConsistencyReport,
    FutureSelfReference,
)

# ---------------------------------------------------------------------------
# Public events
# ---------------------------------------------------------------------------

from subsystems.astra.events import (
    # Type constants
    ET_IDENTITY_UPDATED,
    ET_GOAL_CREATED,
    ET_GOAL_UPDATED,
    ET_GOAL_REMOVED,
    ET_PREFERENCE_CHANGED,
    ET_DIGITAL_TWIN_UPDATED,
    ET_CONSISTENCY_CHECK_COMPLETED,
    ET_IDENTITY_EVOLVED,
    ASTRA_SOURCE,
    # Factory functions
    identity_updated,
    goal_created,
    goal_updated,
    goal_removed,
    preference_changed,
    digital_twin_updated,
    consistency_check_completed,
    identity_evolved,
)

# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

from subsystems.astra.exceptions import (
    AstraError,
    AstraNotInitializedError,
    IdentityNotFoundError,
    IdentityValidationError,
    GoalNotFoundError,
    GoalValidationError,
    GoalStateError,
    PreferenceValidationError,
    DigitalTwinError,
    ConsistencyError,
    EvolutionError,
    InsufficientEvidenceError,
)

# ---------------------------------------------------------------------------
# Capability exceptions (from capability module)
# ---------------------------------------------------------------------------

from subsystems.astra.capability import (
    CapabilityNotFoundError,
    CapabilityAlreadyExistsError,
    CapabilityValidationError,
)

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

from subsystems.astra.schemas import (
    identity_profile_to_dict,
    identity_profile_from_dict,
    goal_to_dict,
    goal_from_dict,
    preference_profile_to_dict,
    preference_profile_from_dict,
    capability_snapshot_to_dict,
    capability_snapshot_from_dict,
    future_self_ref_to_dict,
    future_self_ref_from_dict,
    digital_twin_state_to_dict,
    evolution_record_to_dict,
    consistency_report_to_dict,
)

# ---------------------------------------------------------------------------
# Interfaces (abstract engine contracts)
# ---------------------------------------------------------------------------

from subsystems.astra.interfaces import (
    MotivationEngineInterface,
    StrengthEngineInterface,
    WeaknessEngineInterface,
    HabitEngineInterface,
    BehaviorPatternEngineInterface,
    DecisionPatternEngineInterface,
    LearningStyleEngineInterface,
    RelationshipEngineInterface,
    GrowthEngineInterface,
    FutureSelfEngineInterface,
)

__all__ = [
    # --- Subsystem ---
    "AstraSubsystem",
    # --- Engines ---
    "DigitalTwinEngine",
    "CapabilityManager",
    "CapabilityRegistry",
    "IdentityEngine",
    "GoalEngine",
    "PreferenceEngine",
    "ConsistencyEngine",
    "IdentityEvolutionEngine",
    # --- Models ---
    "GoalState",
    "GoalPriority",
    "GoalType",
    "EvolutionTrigger",
    "IdentityProfile",
    "Goal",
    "PreferenceProfile",
    "CommunicationPreferences",
    "LearningPreferences",
    "DevelopmentPreferences",
    "WorkflowPreferences",
    "CapabilityEntry",
    "CapabilitySnapshot",
    "DigitalTwinState",
    "EvolutionRecord",
    "EvidenceItem",
    "ConsistencyReport",
    "FutureSelfReference",
    # --- Events ---
    "ET_IDENTITY_UPDATED",
    "ET_GOAL_CREATED",
    "ET_GOAL_UPDATED",
    "ET_GOAL_REMOVED",
    "ET_PREFERENCE_CHANGED",
    "ET_DIGITAL_TWIN_UPDATED",
    "ET_CONSISTENCY_CHECK_COMPLETED",
    "ET_IDENTITY_EVOLVED",
    "ASTRA_SOURCE",
    "identity_updated",
    "goal_created",
    "goal_updated",
    "goal_removed",
    "preference_changed",
    "digital_twin_updated",
    "consistency_check_completed",
    "identity_evolved",
    # --- Exceptions ---
    "AstraError",
    "AstraNotInitializedError",
    "IdentityNotFoundError",
    "IdentityValidationError",
    "GoalNotFoundError",
    "GoalValidationError",
    "GoalStateError",
    "PreferenceValidationError",
    "DigitalTwinError",
    "ConsistencyError",
    "EvolutionError",
    "InsufficientEvidenceError",
    "CapabilityNotFoundError",
    "CapabilityAlreadyExistsError",
    "CapabilityValidationError",
    # --- Schemas ---
    "identity_profile_to_dict",
    "identity_profile_from_dict",
    "goal_to_dict",
    "goal_from_dict",
    "preference_profile_to_dict",
    "preference_profile_from_dict",
    "capability_snapshot_to_dict",
    "capability_snapshot_from_dict",
    "future_self_ref_to_dict",
    "future_self_ref_from_dict",
    "digital_twin_state_to_dict",
    "evolution_record_to_dict",
    "consistency_report_to_dict",
    # --- Interfaces ---
    "MotivationEngineInterface",
    "StrengthEngineInterface",
    "WeaknessEngineInterface",
    "HabitEngineInterface",
    "BehaviorPatternEngineInterface",
    "DecisionPatternEngineInterface",
    "LearningStyleEngineInterface",
    "RelationshipEngineInterface",
    "GrowthEngineInterface",
    "FutureSelfEngineInterface",
]
