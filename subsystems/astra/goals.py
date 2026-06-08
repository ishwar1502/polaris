# subsystems/astra/goals.py
"""
ASTRA v5 Goal Engine.

The Goal Engine tracks all user goals across the full lifecycle from
creation through completion or abandonment.  It enforces valid state
transitions and maintains a thread-safe registry.

ASTRA owns goal *definitions* — what the goals are, why they matter, and
their current state.  Plans, schedules, and execution details belong to
CHRONOS and ODYSSEY respectively.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from subsystems.astra.exceptions import (
    GoalNotFoundError,
    GoalStateError,
    GoalValidationError,
)
from subsystems.astra.models import Goal, GoalPriority, GoalState, GoalType

_logger = logging.getLogger(__name__)

# Valid goal state transitions
_VALID_TRANSITIONS: dict[GoalState, frozenset[GoalState]] = {
    GoalState.ACTIVE: frozenset({
        GoalState.PAUSED,
        GoalState.COMPLETED,
        GoalState.ABANDONED,
        GoalState.DEFERRED,
    }),
    GoalState.PAUSED: frozenset({
        GoalState.ACTIVE,
        GoalState.ABANDONED,
        GoalState.DEFERRED,
    }),
    GoalState.DEFERRED: frozenset({
        GoalState.ACTIVE,
        GoalState.ABANDONED,
    }),
    GoalState.COMPLETED: frozenset(),   # terminal
    GoalState.ABANDONED: frozenset(),   # terminal
}


class GoalEngine:
    """Thread-safe registry and lifecycle manager for user goals.

    All mutations (create, update, remove) are protected by an internal
    :class:`threading.RLock`.  Reads return copies or snapshots to avoid
    holding the lock during consumer processing.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._goals: dict[str, Goal] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_goals(
        self,
        *,
        state: GoalState | None = None,
        goal_type: GoalType | None = None,
        priority: GoalPriority | None = None,
    ) -> list[Goal]:
        """Return goals, optionally filtered.

        Parameters
        ----------
        state:
            Filter by :class:`~subsystems.astra.models.GoalState`.
        goal_type:
            Filter by :class:`~subsystems.astra.models.GoalType`.
        priority:
            Filter by :class:`~subsystems.astra.models.GoalPriority`.

        Returns
        -------
        list[Goal]
            Matching goals ordered by priority descending, then created_at ascending.
        """
        with self._lock:
            goals = list(self._goals.values())

        if state is not None:
            goals = [g for g in goals if g.state == state]
        if goal_type is not None:
            goals = [g for g in goals if g.goal_type == goal_type]
        if priority is not None:
            goals = [g for g in goals if g.priority == priority]

        goals.sort(key=lambda g: (-g.priority.value, g.created_at))
        return goals

    def get_goal(self, goal_id: str) -> Goal:
        """Return a single goal by ID.

        Parameters
        ----------
        goal_id:
            UUID of the goal.

        Returns
        -------
        Goal
            The goal.

        Raises
        ------
        GoalNotFoundError
            If *goal_id* does not exist.
        """
        with self._lock:
            if goal_id not in self._goals:
                raise GoalNotFoundError(goal_id)
            return self._goals[goal_id]

    def create_goal(
        self,
        title: str,
        goal_type: GoalType,
        *,
        priority: GoalPriority = GoalPriority.MEDIUM,
        description: str = "",
        motivation: str = "",
        success_criteria: list[str] | None = None,
        tags: list[str] | None = None,
        parent_goal_id: str | None = None,
        target_date: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Goal:
        """Create and register a new goal.

        Parameters
        ----------
        title:
            Short, human-readable label.
        goal_type:
            :class:`~subsystems.astra.models.GoalType` classification.
        priority:
            :class:`~subsystems.astra.models.GoalPriority` tier.
        description:
            Longer elaboration.
        motivation:
            Why this goal matters.
        success_criteria:
            Observable completion conditions.
        tags:
            Free-form labels.
        parent_goal_id:
            Optional parent goal UUID.
        target_date:
            Optional soft completion target.
        metadata:
            Optional key-value annotations.

        Returns
        -------
        Goal
            The newly created and registered goal.

        Raises
        ------
        GoalValidationError
            If *title* is empty or *parent_goal_id* is not found.
        """
        if not title or not title.strip():
            raise GoalValidationError("Goal title cannot be empty.", field="title")

        if parent_goal_id is not None:
            with self._lock:
                if parent_goal_id not in self._goals:
                    raise GoalValidationError(
                        f"Parent goal '{parent_goal_id}' does not exist.",
                        field="parent_goal_id",
                    )

        goal = Goal(
            title=title.strip(),
            goal_type=goal_type,
            goal_id=str(uuid.uuid4()),
            priority=priority,
            description=description,
            motivation=motivation,
            success_criteria=list(success_criteria or []),
            tags=list(tags or []),
            parent_goal_id=parent_goal_id,
            target_date=target_date,
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._goals[goal.goal_id] = goal

        _logger.debug(
            "Goal created: id=%s title=%r type=%s priority=%s",
            goal.goal_id,
            goal.title,
            goal.goal_type.name,
            goal.priority.name,
        )
        return goal

    def update_goal(self, goal_id: str, updates: dict[str, Any]) -> Goal:
        """Apply validated updates to an existing goal.

        Accepted update fields: ``title``, ``description``, ``motivation``,
        ``priority``, ``state``, ``progress_pct``, ``success_criteria``,
        ``tags``, ``target_date``, ``metadata``.

        State transitions are validated against :data:`_VALID_TRANSITIONS`.

        Parameters
        ----------
        goal_id:
            UUID of the goal to update.
        updates:
            Dictionary of field-name → new-value pairs.

        Returns
        -------
        Goal
            The updated goal.

        Raises
        ------
        GoalNotFoundError
            If *goal_id* does not exist.
        GoalValidationError
            If *updates* contains unknown or invalid fields.
        GoalStateError
            If a state transition is not permitted.
        """
        allowed_fields = {
            "title",
            "description",
            "motivation",
            "priority",
            "state",
            "progress_pct",
            "success_criteria",
            "tags",
            "target_date",
            "metadata",
        }
        unknown = set(updates) - allowed_fields
        if unknown:
            raise GoalValidationError(
                f"Unknown goal update fields: {sorted(unknown)}.",
                field=next(iter(unknown)),
            )

        with self._lock:
            if goal_id not in self._goals:
                raise GoalNotFoundError(goal_id)
            goal = self._goals[goal_id]

            if goal.is_terminal():
                raise GoalStateError(
                    goal_id,
                    goal.state.name,
                    updates.get("state", goal.state.name),
                )

            self._apply_goal_updates(goal, updates)
            _logger.debug(
                "Goal updated: id=%s fields=%s state=%s",
                goal_id,
                list(updates.keys()),
                goal.state.name,
            )
            return goal

    def remove_goal(self, goal_id: str) -> Goal:
        """Permanently remove a goal from the registry.

        Unlike abandoning, removal is a hard delete.  Use with care;
        prefer transitioning to :attr:`GoalState.ABANDONED` for audit trails.

        Parameters
        ----------
        goal_id:
            UUID of the goal to remove.

        Returns
        -------
        Goal
            The removed goal.

        Raises
        ------
        GoalNotFoundError
            If *goal_id* does not exist.
        """
        with self._lock:
            if goal_id not in self._goals:
                raise GoalNotFoundError(goal_id)
            removed = self._goals.pop(goal_id)

        _logger.debug("Goal removed: id=%s title=%r", goal_id, removed.title)
        return removed

    def get_active_goals(self) -> list[Goal]:
        """Convenience method — return all ACTIVE goals."""
        return self.get_goals(state=GoalState.ACTIVE)

    def get_goal_count(self) -> int:
        """Return the total number of registered goals."""
        with self._lock:
            return len(self._goals)

    def get_changed_fields(self, goal_id: str, updates: dict[str, Any]) -> list[str]:
        """Return which fields would actually change given *updates*.

        Parameters
        ----------
        goal_id:
            UUID of the goal.
        updates:
            Proposed updates dictionary.

        Returns
        -------
        list[str]
            Fields whose values would differ.
        """
        with self._lock:
            if goal_id not in self._goals:
                return list(updates.keys())
            goal = self._goals[goal_id]
            changed = []
            for field, new_val in updates.items():
                current = getattr(goal, field, None)
                if isinstance(current, (GoalState, GoalPriority, GoalType)):
                    if current.name != (new_val.name if hasattr(new_val, "name") else str(new_val)):
                        changed.append(field)
                elif current != new_val:
                    changed.append(field)
            return changed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_goal_updates(self, goal: Goal, updates: dict[str, Any]) -> None:
        """Apply *updates* to *goal* (lock must be held)."""
        if "title" in updates:
            val = str(updates["title"]).strip()
            if not val:
                raise GoalValidationError("Goal title cannot be empty.", field="title")
            goal.title = val

        if "description" in updates:
            goal.description = str(updates["description"])

        if "motivation" in updates:
            goal.motivation = str(updates["motivation"])

        if "priority" in updates:
            val = updates["priority"]
            if not isinstance(val, GoalPriority):
                try:
                    val = GoalPriority[str(val).upper()]
                except KeyError:
                    raise GoalValidationError(
                        f"Invalid priority: {updates['priority']!r}.",
                        field="priority",
                    )
            goal.priority = val

        if "state" in updates:
            new_state = updates["state"]
            if not isinstance(new_state, GoalState):
                try:
                    new_state = GoalState[str(new_state).upper()]
                except KeyError:
                    raise GoalValidationError(
                        f"Invalid state: {updates['state']!r}.", field="state"
                    )
            allowed = _VALID_TRANSITIONS.get(goal.state, frozenset())
            if new_state not in allowed:
                raise GoalStateError(goal.goal_id, goal.state.name, new_state.name)
            goal.state = new_state
            if new_state == GoalState.COMPLETED:
                goal.completed_at = datetime.now(timezone.utc)

        if "progress_pct" in updates:
            val = int(updates["progress_pct"])
            if not (0 <= val <= 100):
                raise GoalValidationError(
                    "progress_pct must be between 0 and 100.", field="progress_pct"
                )
            goal.progress_pct = val

        if "success_criteria" in updates:
            val = updates["success_criteria"]
            if not isinstance(val, list):
                raise GoalValidationError(
                    "'success_criteria' must be a list.", field="success_criteria"
                )
            goal.success_criteria = [str(c) for c in val]

        if "tags" in updates:
            val = updates["tags"]
            if not isinstance(val, list):
                raise GoalValidationError("'tags' must be a list.", field="tags")
            goal.tags = [str(t) for t in val]

        if "target_date" in updates:
            goal.target_date = updates["target_date"]

        if "metadata" in updates:
            val = updates["metadata"]
            if not isinstance(val, dict):
                raise GoalValidationError(
                    "'metadata' must be a dictionary.", field="metadata"
                )
            goal.metadata.update(val)

        goal.mark_updated()