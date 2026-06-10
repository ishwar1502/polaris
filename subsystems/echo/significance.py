# subsystems/echo/significance.py
"""
ECHO v1 Significance Engine.

Implements :class:`SignificanceEngine` — the production implementation of
:class:`~subsystems.echo.interfaces.SignificanceEngineInterface`.

The Significance Engine is the memory gatekeeper of ECHO.  It determines
which experiences deserve memory by assigning a numeric significance score
(0.0–1.0) and mapping that score to an :class:`ExperienceImportance` tier.
Only MEDIUM and above are eligible for long-term memory consolidation.

Design Principles
-----------------
* **Read-only evaluation**: The engine never modifies :class:`Experience`
  objects passed to it.  Callers are responsible for applying the result.
* **Deterministic**: Identical inputs always produce identical scores.
* **Configurable**: All thresholds and factor weights are injectable at
  construction time or updatable at runtime.
* **Thread-safe**: All public methods serialise concurrent access via
  ``self._lock`` (a :class:`threading.RLock`).
* **Extensible**: Custom scoring rules may be registered without touching
  the engine's public API.

Scoring Model
-------------
The significance score is a weighted sum of the following factors:

1. **type_weight**      — base weight derived from :class:`ExperienceType`.
2. **importance_hint**  — the caller-supplied :class:`ExperienceImportance`
                           tier contributes a normalised bonus.
3. **tag_richness**     — tag count and category diversity.
4. **description_depth**— non-empty description, context, and outcome fields
                           each contribute a small increment.
5. **retrieval_bonus**  — experiences already retrieved frequently are
                           proven valuable and gain a bonus.
6. **session_bonus**    — experiences nested in a session receive a modest
                           bonus, as they are part of deliberate work.
7. **custom_rules**     — registered :class:`ScoringRule` callables that
                           may add or subtract up to their declared weight.

The raw weighted sum is clamped to [0.0, 1.0] before being returned.

Tier Boundaries (configurable via :class:`SignificanceThresholds`)
------------------------------------------------------------------
* [0.00, 0.25) → LOW
* [0.25, 0.55) → MEDIUM
* [0.55, 0.80) → HIGH
* [0.80, 1.00] → CRITICAL

Promotion Eligibility
---------------------
A MEDIUM experience is eligible for promotion to HIGH when its retrieval
count meets the configured promotion threshold OR when the density of
related experiences surpasses the configured density threshold.

Integration Points
------------------
* **Memory Consolidation Engine**: consult :meth:`eligible_for_consolidation`
  before promoting experiences to long-term storage.
* **Reflection Engine**: consult :meth:`is_significant` to gate reflection
  generation — only MEDIUM and above should trigger reflections by default.
* **Experience Engine**: calls :meth:`score_experience` on every
  ``create_experience`` / ``store_experience`` invocation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Final

from subsystems.echo.exceptions import (
    EchoNotInitializedError,
    SignificanceError,
    SignificanceScoringError,
)
from subsystems.echo.interfaces import SignificanceEngineInterface, SignificanceResult
from subsystems.echo.models import (
    Experience,
    ExperienceImportance,
    ExperienceType,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

#: Weights assigned per :class:`ExperienceType`.  These reflect the inherent
#: significance of each type within the ECHO domain.
_TYPE_WEIGHTS: Final[dict[ExperienceType, float]] = {
    ExperienceType.MILESTONE:     0.85,
    ExperienceType.ACHIEVEMENT:   0.75,
    ExperienceType.REFLECTION:    0.70,
    ExperienceType.FAILURE:       0.60,
    ExperienceType.SESSION:       0.50,
    ExperienceType.CONVERSATION:  0.45,
    ExperienceType.EVENT:         0.40,
    ExperienceType.OBSERVATION:   0.20,
}

#: Normalised bonus per :class:`ExperienceImportance` tier supplied by caller.
_IMPORTANCE_BONUSES: Final[dict[ExperienceImportance, float]] = {
    ExperienceImportance.LOW:      0.00,
    ExperienceImportance.MEDIUM:   0.08,
    ExperienceImportance.HIGH:     0.16,
    ExperienceImportance.CRITICAL: 0.25,
}

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SignificanceThresholds:
    """Configurable tier boundary thresholds for the Significance Engine.

    All values must satisfy:  ``0.0 <= low < medium < high < critical <= 1.0``

    Attributes
    ----------
    low:
        Scores *below* this value receive :attr:`ExperienceImportance.LOW`.
        The lower bound is 0.0 (inclusive).
    medium:
        Scores from ``low`` (inclusive) up to ``medium`` (exclusive) receive
        :attr:`ExperienceImportance.MEDIUM`.
    high:
        Scores from ``medium`` (inclusive) up to ``high`` (exclusive) receive
        :attr:`ExperienceImportance.HIGH`.
    critical:
        Scores at or above ``high`` receive :attr:`ExperienceImportance.HIGH`
        or :attr:`ExperienceImportance.CRITICAL`.  Scores at or above
        ``critical`` receive :attr:`ExperienceImportance.CRITICAL`.
    storage_threshold:
        Minimum score for an experience to be accepted by the Experience
        Engine without ``force=True``.
    promotion_retrieval_count:
        Minimum :attr:`ExperienceMetadata.retrieval_count` for a MEDIUM
        experience to be considered for promotion to HIGH.
    promotion_related_density:
        Minimum number of ``related_experience_ids`` for a MEDIUM
        experience to be considered for promotion to HIGH.
    """

    low: float = 0.25
    medium: float = 0.25
    high: float = 0.55
    critical: float = 0.80
    storage_threshold: float = 0.15
    promotion_retrieval_count: int = 5
    promotion_related_density: int = 3

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Raise :class:`ValueError` if the threshold relationships are violated."""
        if not (0.0 <= self.storage_threshold <= 1.0):
            raise ValueError(
                f"storage_threshold must be in [0.0, 1.0]; got {self.storage_threshold}."
            )
        if not (0.0 <= self.low <= 1.0):
            raise ValueError(f"low must be in [0.0, 1.0]; got {self.low}.")
        if not (self.low <= self.medium):
            raise ValueError(
                f"medium ({self.medium}) must be >= low ({self.low})."
            )
        if not (self.medium < self.high):
            raise ValueError(
                f"high ({self.high}) must be > medium ({self.medium})."
            )
        if not (self.high < self.critical <= 1.0):
            raise ValueError(
                f"critical ({self.critical}) must be > high ({self.high}) "
                f"and <= 1.0."
            )
        if self.promotion_retrieval_count < 0:
            raise ValueError("promotion_retrieval_count cannot be negative.")
        if self.promotion_related_density < 0:
            raise ValueError("promotion_related_density cannot be negative.")


@dataclass
class FactorWeights:
    """Configurable contribution weights for each scoring factor.

    The engine multiplies each raw factor score by its corresponding weight
    before summing.  Weights do not need to sum to 1.0 — the final score is
    clamped to [0.0, 1.0].

    Attributes
    ----------
    type_weight:
        Contribution of the experience-type base weight.
    importance_hint:
        Contribution of the caller-supplied importance bonus.
    tag_richness:
        Maximum contribution from tag count and category diversity.
    description_depth:
        Maximum contribution from populated text fields.
    retrieval_bonus:
        Maximum contribution from historical retrieval frequency.
    session_bonus:
        Fixed bonus applied when the experience belongs to a session.
    """

    type_weight: float = 0.45
    importance_hint: float = 0.20
    tag_richness: float = 0.12
    description_depth: float = 0.10
    retrieval_bonus: float = 0.08
    session_bonus: float = 0.05

    def __post_init__(self) -> None:
        for fname, fval in self.__dict__.items():
            if fval < 0.0:
                raise ValueError(
                    f"FactorWeights.{fname} must be >= 0.0; got {fval}."
                )


# ---------------------------------------------------------------------------
# Scoring Rule protocol
# ---------------------------------------------------------------------------

#: A scoring rule callable signature.  Receives the experience and returns a
#: float delta in [-rule_max_weight, +rule_max_weight].  Rules registered with
#: :meth:`SignificanceEngine.register_rule` are called in registration order.
ScoringRule = Callable[[Experience], float]


@dataclass
class _RegisteredRule:
    """Internal record for a registered custom scoring rule."""

    name: str
    rule: ScoringRule
    max_weight: float = 0.10

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Rule name must be a non-empty string.")
        if self.max_weight < 0.0:
            raise ValueError("max_weight must be >= 0.0.")


# ---------------------------------------------------------------------------
# SignificanceScore value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignificanceScore:
    """Immutable scoring record returned by :meth:`SignificanceEngine.score_experience`.

    Attributes
    ----------
    experience_id:
        UUID of the evaluated experience.
    raw_score:
        Unclamped weighted sum before final clamping.
    score:
        Final significance score clamped to [0.0, 1.0].
    importance:
        :class:`ExperienceImportance` tier derived from ``score``.
    factor_breakdown:
        Per-factor contribution mapping for explainability.
    notes:
        Human-readable scoring rationale.
    """

    experience_id: str
    raw_score: float
    score: float
    importance: ExperienceImportance
    factor_breakdown: dict[str, float]
    notes: str = ""


# ---------------------------------------------------------------------------
# SignificanceEngine
# ---------------------------------------------------------------------------


class SignificanceEngine(SignificanceEngineInterface):
    """Production implementation of the ECHO Significance Engine.

    Determines which experiences deserve memory by computing a significance
    score and mapping it to an :class:`ExperienceImportance` tier.

    Parameters
    ----------
    thresholds:
        Tier boundary configuration.  If ``None``, defaults are used.
    factor_weights:
        Scoring factor weight configuration.  If ``None``, defaults are used.

    Thread Safety
    -------------
    All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
    before reading or modifying engine state.

    Examples
    --------
    ::

        engine = SignificanceEngine()
        engine.initialize()

        score = engine.score_experience(experience)
        result = engine.evaluate(experience)

        if engine.is_significant(experience):
            # safe to proceed with storage
            ...

        engine.shutdown()
    """

    def __init__(
        self,
        thresholds: SignificanceThresholds | None = None,
        factor_weights: FactorWeights | None = None,
    ) -> None:
        self._thresholds: SignificanceThresholds = (
            thresholds if thresholds is not None else SignificanceThresholds()
        )
        self._weights: FactorWeights = (
            factor_weights if factor_weights is not None else FactorWeights()
        )
        self._rules: list[_RegisteredRule] = []
        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Must be called before any scoring operation.  Idempotent: calling
        ``initialize()`` on an already-running engine is a no-op.

        Raises
        ------
        SignificanceError
            If initialisation fails.
        """
        with self._lock:
            if self._initialized:
                return
            try:
                # Validate current configuration at startup.
                self._thresholds._validate()
                self._initialized = True
                _logger.info(
                    "SignificanceEngine initialised. "
                    "storage_threshold=%.3f, tiers=(low<%.2f, medium<%.2f, "
                    "high<%.2f, critical>=%.2f)",
                    self._thresholds.storage_threshold,
                    self._thresholds.low,
                    self._thresholds.medium,
                    self._thresholds.high,
                    self._thresholds.critical,
                )
            except Exception as exc:
                raise SignificanceError(
                    f"SignificanceEngine initialisation failed: {exc}"
                ) from exc

    def shutdown(self) -> None:
        """Release all resources held by this engine.

        After shutdown, any call to a scoring method raises
        :class:`EchoNotInitializedError`.
        """
        with self._lock:
            self._initialized = False
            _logger.info("SignificanceEngine shut down.")

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _require_initialized(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if the engine is not running."""
        if not self._initialized:
            raise EchoNotInitializedError(f"SignificanceEngine.{operation}")

    # ------------------------------------------------------------------
    # Rule Registration
    # ------------------------------------------------------------------

    def register_rule(
        self,
        name: str,
        rule: ScoringRule,
        *,
        max_weight: float = 0.10,
    ) -> None:
        """Register a custom scoring rule.

        Custom rules extend the significance model without modifying the
        engine's core logic.  Rules are applied after the built-in factors
        and their contributions are clamped to ``[-max_weight, +max_weight]``.

        Parameters
        ----------
        name:
            Unique label for this rule (used in factor breakdown output).
        rule:
            Callable ``(Experience) -> float``.  The returned delta is
            clamped to ``[-max_weight, +max_weight]``.
        max_weight:
            Maximum absolute contribution from this rule.  Defaults to 0.10.

        Raises
        ------
        ValueError
            If ``name`` is empty or a rule with the same name already exists.
        SignificanceError
            If the engine has not been initialised.
        """
        with self._lock:
            self._require_initialized("register_rule")
            existing_names = {r.name for r in self._rules}
            if name in existing_names:
                raise ValueError(
                    f"A scoring rule named '{name}' is already registered."
                )
            self._rules.append(
                _RegisteredRule(name=name, rule=rule, max_weight=max_weight)
            )
            _logger.debug("Registered scoring rule '%s' (max_weight=%.3f).", name, max_weight)

    def unregister_rule(self, name: str) -> bool:
        """Remove a previously registered scoring rule by name.

        Parameters
        ----------
        name:
            The name supplied at :meth:`register_rule` time.

        Returns
        -------
        bool
            ``True`` if the rule was found and removed; ``False`` otherwise.
        """
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.name != name]
            removed = len(self._rules) < before
            if removed:
                _logger.debug("Unregistered scoring rule '%s'.", name)
            return removed

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _compute_type_factor(self, experience: Experience) -> float:
        """Return the normalised type weight for the experience type."""
        return _TYPE_WEIGHTS.get(experience.experience_type, 0.30)

    def _compute_importance_factor(self, experience: Experience) -> float:
        """Return the normalised importance hint bonus."""
        return _IMPORTANCE_BONUSES.get(experience.importance, 0.0)

    def _compute_tag_richness(self, experience: Experience) -> float:
        """Return a [0.0, 1.0] tag richness score.

        Scoring model:
        * Each tag contributes a base increment (up to a cap of 8 tags).
        * Category diversity (unique categories) adds a small multiplier.
        """
        tags = experience.tags
        if not tags:
            return 0.0
        tag_count_score = min(len(tags) / 8.0, 1.0)
        unique_categories = len({t.category for t in tags})
        diversity_bonus = min(unique_categories / 4.0, 1.0) * 0.25
        return min(tag_count_score + diversity_bonus, 1.0)

    def _compute_description_depth(self, experience: Experience) -> float:
        """Return a [0.0, 1.0] score based on populated narrative fields.

        Each of {description, context, outcome} contributes equally.
        A field is considered 'populated' when it has more than 10 characters.
        """
        populated = sum(
            1
            for text in (experience.description, experience.context, experience.outcome)
            if len(text.strip()) > 10
        )
        return populated / 3.0

    def _compute_retrieval_bonus(self, experience: Experience) -> float:
        """Return a [0.0, 1.0] bonus proportional to retrieval frequency.

        Retrieval count is capped at 20 for normalisation purposes.
        """
        count = experience.metadata.retrieval_count
        return min(count / 20.0, 1.0)

    def _compute_session_bonus(self, experience: Experience) -> float:
        """Return 1.0 if this experience belongs to a session, else 0.0."""
        return 1.0 if experience.metadata.session_id is not None else 0.0

    def _apply_custom_rules(
        self,
        experience: Experience,
    ) -> dict[str, float]:
        """Execute all registered custom rules and return per-rule deltas.

        Each rule delta is clamped to ``[-rule.max_weight, +rule.max_weight]``
        before being included in the result.

        Parameters
        ----------
        experience:
            The :class:`Experience` to evaluate.

        Returns
        -------
        dict[str, float]
            Mapping of rule name → clamped delta contribution.
        """
        rule_contributions: dict[str, float] = {}
        for registered in self._rules:
            try:
                raw_delta = registered.rule(experience)
                clamped = max(
                    -registered.max_weight,
                    min(registered.max_weight, float(raw_delta)),
                )
                rule_contributions[f"rule:{registered.name}"] = clamped
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "Custom scoring rule '%s' raised an exception for "
                    "experience '%s': %s.  Contribution set to 0.0.",
                    registered.name,
                    experience.experience_id,
                    exc,
                )
                rule_contributions[f"rule:{registered.name}"] = 0.0
        return rule_contributions

    def _compute_raw_score(
        self,
        experience: Experience,
    ) -> tuple[float, dict[str, float]]:
        """Compute the raw (unclamped) significance score and factor breakdown.

        Parameters
        ----------
        experience:
            The :class:`Experience` to score.

        Returns
        -------
        tuple[float, dict[str, float]]
            ``(raw_score, factor_breakdown)`` where ``factor_breakdown`` maps
            factor names to their weighted contributions.
        """
        w = self._weights

        type_raw = self._compute_type_factor(experience)
        importance_raw = self._compute_importance_factor(experience)
        tag_raw = self._compute_tag_richness(experience)
        depth_raw = self._compute_description_depth(experience)
        retrieval_raw = self._compute_retrieval_bonus(experience)
        session_raw = self._compute_session_bonus(experience)

        breakdown: dict[str, float] = {
            "type_weight":       type_raw * w.type_weight,
            "importance_hint":   importance_raw * w.importance_hint,
            "tag_richness":      tag_raw * w.tag_richness,
            "description_depth": depth_raw * w.description_depth,
            "retrieval_bonus":   retrieval_raw * w.retrieval_bonus,
            "session_bonus":     session_raw * w.session_bonus,
        }

        custom = self._apply_custom_rules(experience)
        breakdown.update(custom)

        raw_score = sum(breakdown.values())
        return raw_score, breakdown

    def _clamp(self, value: float) -> float:
        """Clamp a float to [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

    def _tier_from_score(self, score: float) -> ExperienceImportance:
        """Map a clamped significance score to an :class:`ExperienceImportance` tier.

        Uses the thresholds configured on this engine instance.
        """
        t = self._thresholds
        if score >= t.critical:
            return ExperienceImportance.CRITICAL
        if score >= t.high:
            return ExperienceImportance.HIGH
        if score >= t.medium:
            return ExperienceImportance.MEDIUM
        return ExperienceImportance.LOW

    def _build_notes(
        self,
        experience: Experience,
        score: float,
        importance: ExperienceImportance,
        breakdown: dict[str, float],
    ) -> str:
        """Compose a human-readable explanation of the scoring decision."""
        top_factors = sorted(breakdown.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        top_str = ", ".join(f"{k}={v:.3f}" for k, v in top_factors)
        return (
            f"Experience '{experience.title}' scored {score:.3f} "
            f"→ {importance.name}.  "
            f"Top contributing factors: [{top_str}]."
        )

    # ------------------------------------------------------------------
    # Public API — SignificanceEngineInterface
    # ------------------------------------------------------------------

    def score(self, experience: Experience) -> float:
        """Compute a significance score for the given experience.

        Implements :meth:`SignificanceEngineInterface.score`.

        Parameters
        ----------
        experience:
            The :class:`Experience` to evaluate.

        Returns
        -------
        float
            Significance score in [0.0, 1.0].

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SignificanceScoringError
            If the score cannot be computed.
        """
        with self._lock:
            self._require_initialized("score")
            return self.score_experience(experience).score

    def score_experience(self, experience: Experience) -> SignificanceScore:
        """Compute a full :class:`SignificanceScore` for the given experience.

        This is the primary method used by the Experience Engine before
        deciding whether to persist a candidate experience.

        Parameters
        ----------
        experience:
            The :class:`Experience` to evaluate.  Must not be ``None``.

        Returns
        -------
        SignificanceScore
            Fully populated scoring record.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SignificanceScoringError
            If the score cannot be computed.
        """
        with self._lock:
            self._require_initialized("score_experience")
            if experience is None:
                raise SignificanceScoringError(
                    "Cannot score a None experience.",
                    experience_id=None,
                )
            try:
                raw_score, breakdown = self._compute_raw_score(experience)
                clamped = self._clamp(raw_score)
                importance = self._tier_from_score(clamped)
                notes = self._build_notes(experience, clamped, importance, breakdown)
                return SignificanceScore(
                    experience_id=experience.experience_id,
                    raw_score=raw_score,
                    score=clamped,
                    importance=importance,
                    factor_breakdown=breakdown,
                    notes=notes,
                )
            except SignificanceScoringError:
                raise
            except Exception as exc:
                raise SignificanceScoringError(
                    f"Failed to compute significance score for experience "
                    f"'{experience.experience_id}': {exc}",
                    experience_id=experience.experience_id,
                ) from exc

    def classify(self, experience: Experience) -> ExperienceImportance:
        """Derive the canonical :class:`ExperienceImportance` tier.

        Implements :meth:`SignificanceEngineInterface.classify`.

        Parameters
        ----------
        experience:
            The :class:`Experience` to classify.

        Returns
        -------
        ExperienceImportance
            Recommended importance tier for this experience.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SignificanceScoringError
            If classification fails due to a scoring error.
        """
        with self._lock:
            self._require_initialized("classify")
            return self.score_experience(experience).importance

    def is_significant(self, experience: Experience) -> bool:
        """Return ``True`` if the experience is above the LOW tier.

        A convenience method for the Reflection Engine and other consumers
        that only need a boolean significance gate without the full scoring
        detail.

        Parameters
        ----------
        experience:
            The :class:`Experience` to evaluate.

        Returns
        -------
        bool
            ``True`` when the computed importance is MEDIUM, HIGH, or CRITICAL.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SignificanceScoringError
            If scoring fails.
        """
        with self._lock:
            self._require_initialized("is_significant")
            return self.classify(experience) != ExperienceImportance.LOW

    def evaluate(self, experience: Experience) -> SignificanceResult:
        """Produce a full :class:`SignificanceResult` for an experience.

        Implements :meth:`SignificanceEngineInterface.evaluate`.

        Parameters
        ----------
        experience:
            The :class:`Experience` to evaluate.

        Returns
        -------
        SignificanceResult
            Score, tier, breakdown, and all eligibility flags.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SignificanceScoringError
            If evaluation fails.
        """
        with self._lock:
            self._require_initialized("evaluate")
            sig_score = self.score_experience(experience)
            eligible_storage = self.is_eligible_for_storage(experience)
            eligible_ltm = self.is_eligible_for_long_term_memory(experience)
            promotion = self.is_promotion_eligible(experience)
            return SignificanceResult(
                experience_id=sig_score.experience_id,
                score=sig_score.score,
                importance=sig_score.importance,
                eligible_for_storage=eligible_storage,
                eligible_for_long_term_memory=eligible_ltm,
                promotion_eligible=promotion,
                factor_breakdown=sig_score.factor_breakdown,
                notes=sig_score.notes,
            )

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def get_threshold(self) -> float:
        """Return the current minimum score for automatic storage.

        Implements :meth:`SignificanceEngineInterface.get_threshold`.

        Returns
        -------
        float
            Current storage threshold in [0.0, 1.0].
        """
        with self._lock:
            return self._thresholds.storage_threshold

    def set_threshold(self, threshold: float) -> None:
        """Update the minimum significance threshold.

        Implements :meth:`SignificanceEngineInterface.set_threshold`.

        Parameters
        ----------
        threshold:
            New threshold in [0.0, 1.0].

        Raises
        ------
        ValueError
            If ``threshold`` is outside [0.0, 1.0].
        """
        with self._lock:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(
                    f"threshold must be in [0.0, 1.0]; got {threshold}."
                )
            self._thresholds.storage_threshold = threshold
            _logger.info("SignificanceEngine storage threshold updated to %.3f.", threshold)

    def get_thresholds(self) -> SignificanceThresholds:
        """Return a copy of the current :class:`SignificanceThresholds`.

        Returns a shallow copy so callers cannot mutate engine state directly.

        Returns
        -------
        SignificanceThresholds
            Current threshold configuration.
        """
        with self._lock:
            import copy
            return copy.copy(self._thresholds)

    def update_thresholds(self, thresholds: SignificanceThresholds) -> None:
        """Replace the current threshold configuration atomically.

        Parameters
        ----------
        thresholds:
            New :class:`SignificanceThresholds` instance.  The instance is
            validated before being applied.

        Raises
        ------
        ValueError
            If ``thresholds`` fails validation.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        with self._lock:
            self._require_initialized("update_thresholds")
            # Validate before replacing — may raise ValueError.
            thresholds._validate()
            self._thresholds = thresholds
            _logger.info(
                "SignificanceEngine thresholds updated: "
                "storage=%.3f, low<%.2f, medium<%.2f, high<%.2f, critical>=%.2f.",
                thresholds.storage_threshold,
                thresholds.low,
                thresholds.medium,
                thresholds.high,
                thresholds.critical,
            )

    # ------------------------------------------------------------------
    # Eligibility checks
    # ------------------------------------------------------------------

    def is_eligible_for_storage(
        self,
        experience: Experience,
        *,
        force: bool = False,
    ) -> bool:
        """Return whether an experience should be stored in ECHO.

        Implements :meth:`SignificanceEngineInterface.is_eligible_for_storage`.

        Parameters
        ----------
        experience:
            The :class:`Experience` to check.
        force:
            If ``True``, always returns ``True`` regardless of score.

        Returns
        -------
        bool
            ``True`` if the experience meets the storage threshold or
            ``force=True`` is set.
        """
        if force:
            return True
        with self._lock:
            self._require_initialized("is_eligible_for_storage")
            sig = self.score_experience(experience)
            return sig.score >= self._thresholds.storage_threshold

    def is_eligible_for_long_term_memory(self, experience: Experience) -> bool:
        """Return whether an experience qualifies for long-term memory consolidation.

        Implements :meth:`SignificanceEngineInterface.is_eligible_for_long_term_memory`.

        Only MEDIUM, HIGH, and CRITICAL experiences are eligible.  LOW
        experiences must not be promoted to long-term memory.

        Parameters
        ----------
        experience:
            The :class:`Experience` to check.

        Returns
        -------
        bool
            ``True`` if the experience may be consolidated to long-term memory.
        """
        with self._lock:
            self._require_initialized("is_eligible_for_long_term_memory")
            importance = self.classify(experience)
            return importance != ExperienceImportance.LOW

    def eligible_for_consolidation(self, experience: Experience) -> bool:
        """Alias for :meth:`is_eligible_for_long_term_memory`.

        Provided for expressive integration with the Memory Consolidation
        Engine which uses the phrase "consolidation eligibility".

        Parameters
        ----------
        experience:
            The :class:`Experience` to check.

        Returns
        -------
        bool
            ``True`` if the experience is a consolidation candidate.
        """
        return self.is_eligible_for_long_term_memory(experience)

    def is_promotion_eligible(self, experience: Experience) -> bool:
        """Return whether a MEDIUM experience should be promoted to HIGH.

        Implements :meth:`SignificanceEngineInterface.is_promotion_eligible`.

        Promotion is granted when any of the following conditions hold:

        * :attr:`ExperienceMetadata.retrieval_count` meets or exceeds
          :attr:`SignificanceThresholds.promotion_retrieval_count`.
        * The number of ``related_experience_ids`` meets or exceeds
          :attr:`SignificanceThresholds.promotion_related_density`.

        Only MEDIUM experiences are considered for promotion.  Experiences
        already at HIGH or CRITICAL are not candidates.

        Parameters
        ----------
        experience:
            The :class:`Experience` to evaluate.

        Returns
        -------
        bool
            ``True`` if the experience warrants promotion to HIGH.
        """
        with self._lock:
            self._require_initialized("is_promotion_eligible")
            importance = self.classify(experience)
            if importance != ExperienceImportance.MEDIUM:
                return False
            t = self._thresholds
            retrieval_ok = (
                experience.metadata.retrieval_count >= t.promotion_retrieval_count
            )
            density_ok = (
                len(experience.metadata.related_experience_ids)
                >= t.promotion_related_density
            )
            return retrieval_ok or density_ok

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def score_batch(
        self, experiences: list[Experience]
    ) -> dict[str, float]:
        """Score multiple experiences in a single call.

        Implements :meth:`SignificanceEngineInterface.score_batch`.

        Parameters
        ----------
        experiences:
            List of :class:`Experience` objects to score.

        Returns
        -------
        dict[str, float]
            Mapping of ``experience_id`` → significance score.
        """
        with self._lock:
            self._require_initialized("score_batch")
            return {
                exp.experience_id: self.score_experience(exp).score
                for exp in experiences
            }

    def classify_batch(
        self, experiences: list[Experience]
    ) -> dict[str, ExperienceImportance]:
        """Classify multiple experiences in a single call.

        Implements :meth:`SignificanceEngineInterface.classify_batch`.

        Parameters
        ----------
        experiences:
            List of :class:`Experience` objects to classify.

        Returns
        -------
        dict[str, ExperienceImportance]
            Mapping of ``experience_id`` → :class:`ExperienceImportance` tier.
        """
        with self._lock:
            self._require_initialized("classify_batch")
            return {
                exp.experience_id: self.score_experience(exp).importance
                for exp in experiences
            }

    # ------------------------------------------------------------------
    # Ranking and comparison
    # ------------------------------------------------------------------

    def rank_experiences(
        self,
        experiences: list[Experience],
        *,
        descending: bool = True,
    ) -> list[Experience]:
        """Return experiences sorted by significance score.

        Parameters
        ----------
        experiences:
            List of :class:`Experience` objects to rank.
        descending:
            If ``True`` (default), most significant first.  Pass ``False``
            for ascending order (least significant first).

        Returns
        -------
        list[Experience]
            Sorted list — original list is not modified.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        with self._lock:
            self._require_initialized("rank_experiences")
            scores = {
                exp.experience_id: self.score_experience(exp).score
                for exp in experiences
            }
            return sorted(
                experiences,
                key=lambda e: scores[e.experience_id],
                reverse=descending,
            )

    def compare_significance(
        self,
        experience_a: Experience,
        experience_b: Experience,
    ) -> int:
        """Compare two experiences by significance score.

        Parameters
        ----------
        experience_a:
            First :class:`Experience`.
        experience_b:
            Second :class:`Experience`.

        Returns
        -------
        int
            * ``-1`` if ``experience_a`` is less significant than
              ``experience_b``.
            * ``0`` if they are equally significant.
            * ``1`` if ``experience_a`` is more significant than
              ``experience_b``.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        SignificanceScoringError
            If either experience cannot be scored.
        """
        with self._lock:
            self._require_initialized("compare_significance")
            score_a = self.score_experience(experience_a).score
            score_b = self.score_experience(experience_b).score
            if score_a < score_b:
                return -1
            if score_a > score_b:
                return 1
            return 0