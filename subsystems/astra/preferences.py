# subsystems/astra/preferences.py
"""
ASTRA v5 Preference Engine.

Manages the user's persistent tendency models across four dimensions:
communication, learning, development, and workflow.

Preferences represent stable patterns, NOT temporary choices.  A single
session preference does not update this model; only repeated, confirmed
tendencies are reflected here.

All mutations are thread-safe via an internal :class:`threading.RLock`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from subsystems.astra.exceptions import PreferenceValidationError
from subsystems.astra.models import (
    CommunicationPreferences,
    DevelopmentPreferences,
    LearningPreferences,
    PreferenceProfile,
    WorkflowPreferences,
)

_logger = logging.getLogger(__name__)


class PreferenceEngine:
    """Thread-safe manager for the user's persistent preference model.

    The engine starts with default preference values and evolves them as
    the user's tendencies are confirmed.  Direct attribute access is not
    permitted; all updates flow through :meth:`update_preferences`.
    """

    def __init__(self, initial_profile: PreferenceProfile | None = None) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._profile: PreferenceProfile = initial_profile or PreferenceProfile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_preferences(self) -> PreferenceProfile:
        """Return the current preference profile.

        Returns
        -------
        PreferenceProfile
            The current profile (not a copy; do not mutate directly).
        """
        with self._lock:
            return self._profile

    def update_preferences(self, updates: dict[str, Any]) -> PreferenceProfile:
        """Apply validated updates to the preference profile.

        *updates* is a nested dictionary.  Top-level keys correspond to
        preference dimensions.  Nested keys correspond to fields within that
        dimension.

        Example::

            engine.update_preferences({
                "communication": {"preferred_tone": "direct"},
                "learning": {"prefers_big_picture_first": True},
            })

        Parameters
        ----------
        updates:
            Nested update dictionary.

        Returns
        -------
        PreferenceProfile
            The updated profile.

        Raises
        ------
        PreferenceValidationError
            If unknown dimensions or fields are specified.
        """
        allowed_dimensions = {
            "communication",
            "learning",
            "development",
            "workflow",
            "metadata",
        }
        unknown_dims = set(updates) - allowed_dimensions
        if unknown_dims:
            raise PreferenceValidationError(
                f"Unknown preference dimensions: {sorted(unknown_dims)}. "
                f"Allowed: {sorted(allowed_dimensions)}.",
                field=next(iter(unknown_dims)),
            )

        changed_dimensions = []

        with self._lock:
            if "communication" in updates:
                self._apply_communication(updates["communication"])
                changed_dimensions.append("communication")

            if "learning" in updates:
                self._apply_learning(updates["learning"])
                changed_dimensions.append("learning")

            if "development" in updates:
                self._apply_development(updates["development"])
                changed_dimensions.append("development")

            if "workflow" in updates:
                self._apply_workflow(updates["workflow"])
                changed_dimensions.append("workflow")

            if "metadata" in updates:
                val = updates["metadata"]
                if not isinstance(val, dict):
                    raise PreferenceValidationError(
                        "'metadata' must be a dictionary.", field="metadata"
                    )
                self._profile.metadata.update(val)
                changed_dimensions.append("metadata")

            if changed_dimensions:
                self._profile.bump_version()

            _logger.debug(
                "Preferences updated: dimensions=%s version=%d",
                changed_dimensions,
                self._profile.version,
            )
            return self._profile

    def get_changed_dimensions(self, updates: dict[str, Any]) -> list[str]:
        """Return dimensions that would change given *updates*.

        Parameters
        ----------
        updates:
            Proposed update dictionary.

        Returns
        -------
        list[str]
            Dimension names that would be modified.
        """
        return [dim for dim in updates if dim in {
            "communication", "learning", "development", "workflow", "metadata"
        }]

    # ------------------------------------------------------------------
    # Internal dimension applicators
    # ------------------------------------------------------------------

    def _apply_communication(self, updates: dict[str, Any]) -> None:
        """Apply updates to the communication preferences sub-object."""
        allowed = {
            "preferred_detail_level",
            "prefers_examples",
            "prefers_analogies",
            "preferred_tone",
            "structured_output_preferred",
        }
        self._check_unknown(updates, allowed, "communication")
        c = self._profile.communication

        if "preferred_detail_level" in updates:
            val = str(updates["preferred_detail_level"])
            valid = {"brief", "normal", "detailed", "comprehensive"}
            if val not in valid:
                raise PreferenceValidationError(
                    f"preferred_detail_level must be one of {valid}.",
                    field="preferred_detail_level",
                )
            c.preferred_detail_level = val

        if "prefers_examples" in updates:
            c.prefers_examples = bool(updates["prefers_examples"])

        if "prefers_analogies" in updates:
            c.prefers_analogies = bool(updates["prefers_analogies"])

        if "preferred_tone" in updates:
            val = str(updates["preferred_tone"])
            valid = {"formal", "casual", "collaborative", "direct"}
            if val not in valid:
                raise PreferenceValidationError(
                    f"preferred_tone must be one of {valid}.", field="preferred_tone"
                )
            c.preferred_tone = val

        if "structured_output_preferred" in updates:
            c.structured_output_preferred = bool(
                updates["structured_output_preferred"]
            )

    def _apply_learning(self, updates: dict[str, Any]) -> None:
        """Apply updates to the learning preferences sub-object."""
        allowed = {
            "preferred_approach",
            "prefers_big_picture_first",
            "learns_by_building",
            "preferred_learning_pace",
            "depth_over_breadth",
        }
        self._check_unknown(updates, allowed, "learning")
        l = self._profile.learning

        if "preferred_approach" in updates:
            val = str(updates["preferred_approach"])
            valid = {
                "examples_first",
                "theory_first",
                "systems_first",
                "practice_first",
            }
            if val not in valid:
                raise PreferenceValidationError(
                    f"preferred_approach must be one of {valid}.",
                    field="preferred_approach",
                )
            l.preferred_approach = val

        if "prefers_big_picture_first" in updates:
            l.prefers_big_picture_first = bool(updates["prefers_big_picture_first"])

        if "learns_by_building" in updates:
            l.learns_by_building = bool(updates["learns_by_building"])

        if "preferred_learning_pace" in updates:
            val = str(updates["preferred_learning_pace"])
            valid = {"methodical", "normal", "fast", "intensive"}
            if val not in valid:
                raise PreferenceValidationError(
                    f"preferred_learning_pace must be one of {valid}.",
                    field="preferred_learning_pace",
                )
            l.preferred_learning_pace = val

        if "depth_over_breadth" in updates:
            l.depth_over_breadth = bool(updates["depth_over_breadth"])

    def _apply_development(self, updates: dict[str, Any]) -> None:
        """Apply updates to the development preferences sub-object."""
        allowed = {
            "preferred_languages",
            "architecture_first",
            "prefers_modularity",
            "testing_discipline",
            "documentation_style",
        }
        self._check_unknown(updates, allowed, "development")
        d = self._profile.development

        if "preferred_languages" in updates:
            val = updates["preferred_languages"]
            if not isinstance(val, list):
                raise PreferenceValidationError(
                    "'preferred_languages' must be a list.", field="preferred_languages"
                )
            d.preferred_languages = [str(lang) for lang in val]

        if "architecture_first" in updates:
            d.architecture_first = bool(updates["architecture_first"])

        if "prefers_modularity" in updates:
            d.prefers_modularity = bool(updates["prefers_modularity"])

        if "testing_discipline" in updates:
            val = str(updates["testing_discipline"])
            valid = {"minimal", "pragmatic", "thorough", "tdd"}
            if val not in valid:
                raise PreferenceValidationError(
                    f"testing_discipline must be one of {valid}.",
                    field="testing_discipline",
                )
            d.testing_discipline = val

        if "documentation_style" in updates:
            val = str(updates["documentation_style"])
            valid = {"none", "inline", "docstrings", "comprehensive"}
            if val not in valid:
                raise PreferenceValidationError(
                    f"documentation_style must be one of {valid}.",
                    field="documentation_style",
                )
            d.documentation_style = val

    def _apply_workflow(self, updates: dict[str, Any]) -> None:
        """Apply updates to the workflow preferences sub-object."""
        allowed = {
            "deep_work_sessions",
            "preferred_session_length_hours",
            "async_communication_preferred",
            "batch_decisions",
            "peak_hours",
        }
        self._check_unknown(updates, allowed, "workflow")
        w = self._profile.workflow

        if "deep_work_sessions" in updates:
            w.deep_work_sessions = bool(updates["deep_work_sessions"])

        if "preferred_session_length_hours" in updates:
            val = float(updates["preferred_session_length_hours"])
            if val <= 0:
                raise PreferenceValidationError(
                    "preferred_session_length_hours must be positive.",
                    field="preferred_session_length_hours",
                )
            w.preferred_session_length_hours = val

        if "async_communication_preferred" in updates:
            w.async_communication_preferred = bool(
                updates["async_communication_preferred"]
            )

        if "batch_decisions" in updates:
            w.batch_decisions = bool(updates["batch_decisions"])

        if "peak_hours" in updates:
            val = updates["peak_hours"]
            if not isinstance(val, list):
                raise PreferenceValidationError(
                    "'peak_hours' must be a list.", field="peak_hours"
                )
            w.peak_hours = [str(h) for h in val]

    @staticmethod
    def _check_unknown(
        updates: dict[str, Any], allowed: set[str], dimension: str
    ) -> None:
        """Raise if *updates* contains keys not in *allowed*."""
        unknown = set(updates) - allowed
        if unknown:
            raise PreferenceValidationError(
                f"Unknown fields in preference dimension '{dimension}': "
                f"{sorted(unknown)}. Allowed: {sorted(allowed)}.",
                field=next(iter(unknown)),
            )