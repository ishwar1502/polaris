# subsystems/astra/interfaces.py
"""
ASTRA v5 Engine Interfaces.

This module defines abstract base classes (engine contracts) for every engine
that will be fully implemented in future ASTRA iterations.  All interfaces
follow the same pattern:

* They define the engine's public API as abstract methods.
* They document the contract that implementations must honour.
* They provide no fake intelligence or pass-through stubs.

Future engines covered
----------------------
* :class:`MotivationEngineInterface`   — why goals matter
* :class:`StrengthEngineInterface`     — evidence-based strength tracking
* :class:`WeaknessEngineInterface`     — operational limitation tracking
* :class:`HabitEngineInterface`        — recurring behaviour modelling
* :class:`BehaviorPatternEngineInterface` — long-term pattern recognition
* :class:`DecisionPatternEngineInterface` — decision-making style analysis
* :class:`LearningStyleEngineInterface`   — knowledge acquisition modelling
* :class:`RelationshipEngineInterface`    — relationship state modelling
* :class:`GrowthEngineInterface`          — development trajectory tracking
* :class:`FutureSelfEngineInterface`      — future version modelling
"""

from __future__ import annotations

import abc
from typing import Any


# ---------------------------------------------------------------------------
# Motivation Engine
# ---------------------------------------------------------------------------


class MotivationEngineInterface(abc.ABC):
    """Contract for the Motivation Engine.

    The Motivation Engine models *why* goals matter to the user.  It is
    distinct from the Goal Engine (which tracks *what* the goals are).
    Understanding motivation is critical for ODYSSEY's planning and
    prioritisation reasoning.

    Implementation requirements
    ---------------------------
    * Must parse motivation signals from goal descriptions and user behaviour.
    * Must NOT store experiences or events — only motivation patterns.
    * Must differentiate between intrinsic and extrinsic motivators.
    * Must be updated when goals are created, updated, or completed.
    """

    @abc.abstractmethod
    def get_motivation(self, goal_id: str) -> dict[str, Any]:
        """Return the motivation model for a specific goal.

        Parameters
        ----------
        goal_id:
            UUID of the goal to analyse.

        Returns
        -------
        dict[str, Any]
            Motivation analysis including primary motivators, intensity,
            and rationale.
        """

    @abc.abstractmethod
    def get_core_motivators(self) -> list[dict[str, Any]]:
        """Return the user's top-level motivational drivers.

        These are cross-goal motivators that appear consistently across
        multiple goals and time periods.

        Returns
        -------
        list[dict[str, Any]]
            Ordered list of motivator models, most significant first.
        """

    @abc.abstractmethod
    def analyse_goal_motivation(
        self, goal_id: str, goal_description: str, motivation_text: str
    ) -> dict[str, Any]:
        """Analyse and register motivation for a goal.

        Parameters
        ----------
        goal_id:
            UUID of the goal.
        goal_description:
            The goal's description text.
        motivation_text:
            The user's stated motivation.

        Returns
        -------
        dict[str, Any]
            Parsed motivation model.
        """


# ---------------------------------------------------------------------------
# Strength Engine
# ---------------------------------------------------------------------------


class StrengthEngineInterface(abc.ABC):
    """Contract for the Strength Engine.

    The Strength Engine tracks demonstrated strengths with evidence backing.
    Strengths are NOT assumed — they must be observed and accumulated from
    evidence before being registered.

    Implementation requirements
    ---------------------------
    * Must require minimum evidence count before registering a strength.
    * Must support confidence decay over time without fresh evidence.
    * Must expose evidence audit trail for each strength.
    * Must differentiate between domain-specific and cross-domain strengths.
    """

    @abc.abstractmethod
    def get_strengths(self) -> list[dict[str, Any]]:
        """Return all registered strengths with evidence metadata.

        Returns
        -------
        list[dict[str, Any]]
            Evidence-backed strength entries.
        """

    @abc.abstractmethod
    def record_strength_evidence(
        self, strength_name: str, domain: str, evidence_description: str, weight: float
    ) -> None:
        """Record an observation that supports or reinforces a strength.

        Parameters
        ----------
        strength_name:
            Canonical label for the strength.
        domain:
            Domain in which the strength was observed.
        evidence_description:
            Human-readable description of the observation.
        weight:
            0.0-1.0 weight of this evidence.
        """

    @abc.abstractmethod
    def get_top_strengths(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the top *limit* strengths by evidence weight.

        Parameters
        ----------
        limit:
            Maximum number of strengths to return.

        Returns
        -------
        list[dict[str, Any]]
            Top strengths ordered by aggregate evidence weight.
        """


# ---------------------------------------------------------------------------
# Weakness Engine
# ---------------------------------------------------------------------------


class WeaknessEngineInterface(abc.ABC):
    """Contract for the Weakness Engine.

    Tracks recurring operational limitations.  The engine is strictly
    non-judgmental — it identifies patterns that affect performance so other
    engines and subsystems can compensate or route around them.

    Implementation requirements
    ---------------------------
    * Must NEVER be used to judge, criticise, or rank the user.
    * Must track operational impact, not moral or personal value.
    * Must support mitigation strategies registered per limitation.
    * Must flag when a limitation has been actively compensated for.
    """

    @abc.abstractmethod
    def get_limitations(self) -> list[dict[str, Any]]:
        """Return all tracked operational limitations.

        Returns
        -------
        list[dict[str, Any]]
            Limitation entries with operational context and evidence.
        """

    @abc.abstractmethod
    def record_limitation_evidence(
        self,
        limitation_name: str,
        context: str,
        impact_description: str,
        weight: float,
    ) -> None:
        """Record an observation of an operational limitation.

        Parameters
        ----------
        limitation_name:
            Canonical label for the limitation.
        context:
            Context in which the limitation was observed.
        impact_description:
            What impact this limitation had.
        weight:
            0.0-1.0 weight of this evidence.
        """

    @abc.abstractmethod
    def register_mitigation(self, limitation_name: str, strategy: str) -> None:
        """Register a mitigation strategy for a limitation.

        Parameters
        ----------
        limitation_name:
            Limitation to mitigate.
        strategy:
            Description of the compensating strategy.
        """


# ---------------------------------------------------------------------------
# Habit Engine
# ---------------------------------------------------------------------------


class HabitEngineInterface(abc.ABC):
    """Contract for the Habit Engine.

    Models recurring behaviours that occur consistently across contexts.
    Habits are distinct from one-off events (ECHO) and conscious choices
    (goals).  They represent automatic or semi-automatic patterns.

    Implementation requirements
    ---------------------------
    * Must require recurrence across multiple time periods to qualify.
    * Must track habit strength as a function of frequency and consistency.
    * Must differentiate productive and counter-productive habits.
    * Must support habit lifecycle: emerging → established → dormant.
    """

    @abc.abstractmethod
    def get_habits(self) -> list[dict[str, Any]]:
        """Return all tracked habits with strength and recurrence metadata.

        Returns
        -------
        list[dict[str, Any]]
            Habit entries with strength, recurrence, and classification.
        """

    @abc.abstractmethod
    def record_habit_occurrence(
        self, habit_name: str, context: str, timestamp_iso: str
    ) -> None:
        """Record a single occurrence of a habit.

        Parameters
        ----------
        habit_name:
            Canonical label for the habit.
        context:
            Context in which the habit occurred.
        timestamp_iso:
            ISO-8601 UTC timestamp of the occurrence.
        """

    @abc.abstractmethod
    def get_habit_strength(self, habit_name: str) -> float:
        """Return the current strength score (0.0-1.0) for a habit.

        Parameters
        ----------
        habit_name:
            Canonical label for the habit.

        Returns
        -------
        float
            Strength score, 0.0 (dormant) to 1.0 (strong established habit).
        """


# ---------------------------------------------------------------------------
# Behavior Pattern Engine
# ---------------------------------------------------------------------------


class BehaviorPatternEngineInterface(abc.ABC):
    """Contract for the Behavioral Pattern Engine.

    Identifies long-term patterns in the user's behaviour across multiple
    domains.  Unlike habits (regular recurrence), patterns are higher-order
    cycles (e.g. ambition → rapid progress → complexity explosion → refactor).

    Implementation requirements
    ---------------------------
    * Must operate on aggregated data, not individual events.
    * Must identify cycle lengths and trigger conditions.
    * Must support named pattern templates for common patterns.
    * Must produce pattern confidence scores based on observation history.
    """

    @abc.abstractmethod
    def get_patterns(self) -> list[dict[str, Any]]:
        """Return all identified behavioural patterns.

        Returns
        -------
        list[dict[str, Any]]
            Pattern entries with cycle description, confidence, and history.
        """

    @abc.abstractmethod
    def record_pattern_signal(
        self, pattern_name: str, phase: str, signal_description: str
    ) -> None:
        """Record a signal that advances or confirms a pattern phase.

        Parameters
        ----------
        pattern_name:
            Canonical pattern label.
        phase:
            Current phase of the pattern cycle.
        signal_description:
            Human-readable description of the observed signal.
        """

    @abc.abstractmethod
    def get_active_pattern_phases(self) -> list[dict[str, Any]]:
        """Return patterns that are currently in an active phase.

        Returns
        -------
        list[dict[str, Any]]
            Active pattern phases with estimated progression.
        """


# ---------------------------------------------------------------------------
# Decision Pattern Engine
# ---------------------------------------------------------------------------


class DecisionPatternEngineInterface(abc.ABC):
    """Contract for the Decision Pattern Engine.

    Understands *how* the user makes decisions — risk tolerance, time
    horizons, and decision style.  Used heavily by VEGA for strategic
    analysis and recommendation calibration.

    Implementation requirements
    ---------------------------
    * Must model risk tolerance as a dynamic, context-sensitive property.
    * Must differentiate between fast/intuitive and slow/deliberate modes.
    * Must track decision regret signals from AURORA (emotional subsystem).
    * Must produce a decision profile usable by VEGA.
    """

    @abc.abstractmethod
    def get_decision_profile(self) -> dict[str, Any]:
        """Return the current decision-making style profile.

        Returns
        -------
        dict[str, Any]
            Decision profile including risk tolerance, time preference,
            style, and observed decision quality metrics.
        """

    @abc.abstractmethod
    def record_decision(
        self,
        decision_description: str,
        risk_level: str,
        time_horizon: str,
        outcome: str | None = None,
    ) -> None:
        """Record an observed decision for pattern analysis.

        Parameters
        ----------
        decision_description:
            Brief description of the decision made.
        risk_level:
            'low' | 'medium' | 'high' | 'extreme'.
        time_horizon:
            'immediate' | 'short_term' | 'medium_term' | 'long_term'.
        outcome:
            Optional outcome classification once known: 'positive' | 'negative' | 'neutral'.
        """

    @abc.abstractmethod
    def get_risk_tolerance(self, domain: str | None = None) -> float:
        """Return the current risk tolerance score (0.0 = risk-averse, 1.0 = risk-seeking).

        Parameters
        ----------
        domain:
            Optional domain for context-specific tolerance (e.g. 'technical').
            Returns overall tolerance if None.

        Returns
        -------
        float
            Risk tolerance score.
        """


# ---------------------------------------------------------------------------
# Learning Style Engine
# ---------------------------------------------------------------------------


class LearningStyleEngineInterface(abc.ABC):
    """Contract for the Learning Style Engine.

    Models how the user absorbs, processes, and retains new knowledge.
    Used extensively by APOLLO (the knowledge and learning subsystem) to
    optimise information presentation and curriculum design.

    Implementation requirements
    ---------------------------
    * Must identify primary and secondary learning modalities.
    * Must track learning velocity by domain.
    * Must identify knowledge integration patterns (systems vs details).
    * Must support APOLLO's curriculum optimisation queries.
    """

    @abc.abstractmethod
    def get_learning_style_profile(self) -> dict[str, Any]:
        """Return the comprehensive learning style profile.

        Returns
        -------
        dict[str, Any]
            Learning style analysis including modalities, velocity,
            integration patterns, and domain-specific observations.
        """

    @abc.abstractmethod
    def record_learning_observation(
        self,
        topic: str,
        approach_used: str,
        comprehension_signal: str,
        notes: str = "",
    ) -> None:
        """Record an observation about how the user learned something.

        Parameters
        ----------
        topic:
            What was being learned.
        approach_used:
            Learning approach observed (e.g. 'build_it', 'read_theory').
        comprehension_signal:
            Signal of comprehension level: 'fast', 'normal', 'struggled'.
        notes:
            Optional context.
        """

    @abc.abstractmethod
    def get_optimal_approach(self, topic_type: str) -> str:
        """Return the optimal learning approach for a given topic type.

        Parameters
        ----------
        topic_type:
            Classification of the topic (e.g. 'systems', 'algorithms', 'concepts').

        Returns
        -------
        str
            Recommended learning approach label.
        """


# ---------------------------------------------------------------------------
# Relationship Engine
# ---------------------------------------------------------------------------


class RelationshipEngineInterface(abc.ABC):
    """Contract for the Relationship Engine.

    Models the user's human relationships — trust levels, importance,
    relationship type, and interaction patterns.  The engine tracks
    relationship *state*, NOT conversations or messages (those belong to
    ECHO and CONSTELLATION respectively).

    Implementation requirements
    ---------------------------
    * Must NEVER store message content or conversation summaries.
    * Must track: trust, importance, relationship_type, interaction_frequency.
    * Must support relationship lifecycle: new → developing → established → dormant.
    * Must produce relationship models usable by JANUS and VEGA.
    """

    @abc.abstractmethod
    def get_relationship(self, person_id: str) -> dict[str, Any] | None:
        """Return the relationship model for a specific person.

        Parameters
        ----------
        person_id:
            Unique identifier for the person.

        Returns
        -------
        dict[str, Any] | None
            Relationship state model, or None if not tracked.
        """

    @abc.abstractmethod
    def get_all_relationships(self) -> list[dict[str, Any]]:
        """Return all tracked relationships.

        Returns
        -------
        list[dict[str, Any]]
            All relationship models ordered by importance descending.
        """

    @abc.abstractmethod
    def update_relationship(
        self,
        person_id: str,
        trust_delta: float,
        importance: str | None = None,
        interaction_note: str = "",
    ) -> None:
        """Update a relationship model based on an interaction signal.

        Parameters
        ----------
        person_id:
            Unique identifier for the person.
        trust_delta:
            Change to apply to trust score (-1.0 to +1.0).
        importance:
            Optional new importance tier: 'low' | 'medium' | 'high' | 'critical'.
        interaction_note:
            Brief note about the interaction (no content, only metadata).
        """


# ---------------------------------------------------------------------------
# Growth Engine
# ---------------------------------------------------------------------------


class GrowthEngineInterface(abc.ABC):
    """Contract for the Growth Engine.

    Tracks the user's development over time across skills, knowledge,
    project maturity, and decision quality.  Produces growth metrics
    consumed by ODYSSEY, JANUS, and PROMETHEUS.

    Implementation requirements
    ---------------------------
    * Must track growth velocity, not just absolute level.
    * Must identify plateaus and breakthrough moments.
    * Must support domain-specific growth curves.
    * Must produce the growth_indicators dict embedded in DigitalTwinState.
    """

    @abc.abstractmethod
    def get_growth_metrics(self) -> dict[str, Any]:
        """Return the current growth metrics map.

        Returns
        -------
        dict[str, Any]
            Growth metrics including velocity, trajectory, milestones,
            and domain-specific indicators.
        """

    @abc.abstractmethod
    def record_growth_signal(
        self, domain: str, signal_type: str, magnitude: float, notes: str = ""
    ) -> None:
        """Record a growth signal in a given domain.

        Parameters
        ----------
        domain:
            Domain of growth (e.g. 'python', 'architecture', 'leadership').
        signal_type:
            'skill_gain' | 'knowledge_integration' | 'project_completion' | 'decision_quality'.
        magnitude:
            0.0-1.0 magnitude of the growth signal.
        notes:
            Optional context.
        """

    @abc.abstractmethod
    def get_growth_trajectory(self, domain: str | None = None) -> dict[str, Any]:
        """Return the growth trajectory for a domain or overall.

        Parameters
        ----------
        domain:
            Optional domain filter. Returns overall trajectory if None.

        Returns
        -------
        dict[str, Any]
            Trajectory data including direction, velocity, and trend.
        """


# ---------------------------------------------------------------------------
# Future Self Engine
# ---------------------------------------------------------------------------


class FutureSelfEngineInterface(abc.ABC):
    """Contract for the Future Self Engine.

    Models future versions of the user at defined time horizons (1yr, 5yr,
    10yr).  Future *models* are owned by ASTRA; future *scenarios* are owned
    by JANUS.  This engine produces :class:`~subsystems.astra.models.FutureSelfReference`
    objects embedded in the DigitalTwinState.

    Implementation requirements
    ---------------------------
    * Must base projections on current identity, goals, capabilities, and growth.
    * Must NOT produce speculative scenarios (JANUS owns scenarios).
    * Must produce confidence-weighted projections, not deterministic forecasts.
    * Must support multiple horizon models simultaneously.
    * Must update projections when significant identity or goal changes occur.
    """

    @abc.abstractmethod
    def get_future_self(self, horizon_years: float) -> dict[str, Any] | None:
        """Return the future self model for a given horizon.

        Parameters
        ----------
        horizon_years:
            Numeric time horizon in years (e.g. 1.0, 5.0, 10.0).

        Returns
        -------
        dict[str, Any] | None
            Future self model, or None if not yet computed.
        """

    @abc.abstractmethod
    def generate_future_self(
        self,
        horizon_years: float,
        *,
        identity_profile: dict[str, Any],
        active_goals: list[dict[str, Any]],
        growth_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate or refresh the future self model for a given horizon.

        Parameters
        ----------
        horizon_years:
            Time horizon in years.
        identity_profile:
            Serialised current identity profile.
        active_goals:
            List of serialised active goals.
        growth_metrics:
            Current growth metrics.

        Returns
        -------
        dict[str, Any]
            Generated future self model.
        """

    @abc.abstractmethod
    def get_all_horizons(self) -> list[dict[str, Any]]:
        """Return all currently computed future self models.

        Returns
        -------
        list[dict[str, Any]]
            All future self models ordered by horizon ascending.
        """