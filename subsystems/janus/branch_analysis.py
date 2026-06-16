"""
JANUS v5.1 — Future Modeling & Scenario Intelligence Core
Module: subsystems/janus/branch_analysis.py

Branch Analysis Engine — IBranchAnalysisEngine implementation.

Responsibility:
    Explore alternative decision branches, constructing ScenarioBranch objects
    that map triggering choices to FutureStates. Supports strategic exploration
    through branch generation, lineage tracking, ancestry tracking, comparison,
    ranking, pruning, and statistics.

Constitutional rule:
    Branch analysis supports strategic exploration.
    It never selects a branch; VEGA selects.

Bounded Exploration Law:
    Unlimited branch generation is forbidden.
    The engine enforces maximum branch depth, maximum active branches,
    probability pruning, confidence pruning, and resource limits.
    Quality over quantity.

JANUS owns branch analysis exclusively.
VEGA owns branch selection.
ZENITH owns planning.
ASTRA owns identity modeling.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .exceptions import (
    JanusAlreadyInitializedError,
    JanusBranchAnalysisError,
    JanusBranchConstructionError,
    JanusBranchNotFoundError,
    JanusBranchProbabilityConflictError,
    JanusConstitutionalViolationError,
    JanusNotInitializedError,
    JanusShutdownError,
    JanusValidationError,
)
from .interfaces import IBranchAnalysisEngine
from .models import (
    ConfidenceProfile,
    FutureState,
    OpportunityAssessment,
    ProbabilityLevel,
    RiskAssessment,
    ScenarioBranch,
)

_LOG = logging.getLogger(__name__)

_ENGINE_NAME: str = "BranchAnalysisEngine"
_ENGINE_VERSION: str = "5.1.0"

# ---------------------------------------------------------------------------
# Bounded Exploration Constants (overridable via BranchAnalysisConfig)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_BRANCHES_PER_SCENARIO: int = 32
_DEFAULT_MAX_BRANCH_DEPTH: int = 8
_DEFAULT_MIN_PROBABILITY_THRESHOLD: float = 0.05
_DEFAULT_MIN_CONFIDENCE_THRESHOLD: float = 0.10
_DEFAULT_MAX_TOTAL_ACTIVE_BRANCHES: int = 512


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchAnalysisConfig:
    """
    Immutable configuration for the Branch Analysis Engine.

    All bounded-exploration limits are centralized here so that callers
    can tune them without modifying engine internals.
    """

    max_branches_per_scenario: int = _DEFAULT_MAX_BRANCHES_PER_SCENARIO
    """Maximum number of ScenarioBranches permitted per scenario_id."""

    max_branch_depth: int = _DEFAULT_MAX_BRANCH_DEPTH
    """Maximum lineage depth before further child branches are rejected."""

    min_probability_threshold: float = _DEFAULT_MIN_PROBABILITY_THRESHOLD
    """Branches with probability below this value are pruned on ingestion."""

    min_confidence_threshold: float = _DEFAULT_MIN_CONFIDENCE_THRESHOLD
    """Branches whose ConfidenceProfile.overall is below this value are pruned."""

    max_total_active_branches: int = _DEFAULT_MAX_TOTAL_ACTIVE_BRANCHES
    """Hard ceiling on the total number of live ScenarioBranch objects in store."""

    def __post_init__(self) -> None:
        if self.max_branches_per_scenario < 1:
            raise ValueError("max_branches_per_scenario must be >= 1")
        if self.max_branch_depth < 1:
            raise ValueError("max_branch_depth must be >= 1")
        if not 0.0 <= self.min_probability_threshold <= 1.0:
            raise ValueError("min_probability_threshold must be in [0, 1]")
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError("min_confidence_threshold must be in [0, 1]")
        if self.max_total_active_branches < 1:
            raise ValueError("max_total_active_branches must be >= 1")


# ---------------------------------------------------------------------------
# Internal Records
# ---------------------------------------------------------------------------


@dataclass
class _BranchRecord:
    """Internal bookkeeping record stored alongside every ScenarioBranch."""

    branch: ScenarioBranch
    scenario_id: str
    parent_branch_id: Optional[str]
    depth: int
    registered_at: datetime
    pruned: bool = False
    prune_reason: Optional[str] = None

    @property
    def branch_id(self) -> str:
        return self.branch.branch_id


# ---------------------------------------------------------------------------
# Statistics & Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchStatistics:
    """Point-in-time statistics snapshot for the Branch Analysis Engine."""

    total_branches: int
    active_branches: int
    pruned_branches: int
    scenarios_tracked: int
    average_branches_per_scenario: float
    average_branch_probability: float
    average_branch_confidence: float
    deepest_lineage_depth: int
    branch_count_by_scenario: dict[str, int]
    generated_at: datetime


@dataclass(frozen=True)
class BranchHealthReport:
    """Health report for the Branch Analysis Engine lifecycle."""

    engine_name: str
    engine_version: str
    is_initialized: bool
    is_shut_down: bool
    total_branches: int
    active_branches: int
    pruned_branches: int
    scenarios_tracked: int
    capacity_utilization: float  # active / max_total_active_branches
    config: BranchAnalysisConfig
    generated_at: datetime


@dataclass(frozen=True)
class BranchDiagnosticsReport:
    """Detailed diagnostics snapshot for engineering inspection."""

    engine_name: str
    engine_version: str
    is_initialized: bool
    config: BranchAnalysisConfig
    total_records: int
    active_records: int
    pruned_records: int
    scenario_branch_counts: dict[str, int]
    scenario_max_depths: dict[str, int]
    branch_ids_by_scenario: dict[str, list[str]]
    pruned_branch_ids: list[str]
    generated_at: datetime


@dataclass(frozen=True)
class BranchLineage:
    """Full lineage chain from a branch back to the root of its scenario tree."""

    branch_id: str
    scenario_id: str
    ancestors: tuple[str, ...]   # ordered root → direct parent (branch_ids)
    depth: int


@dataclass(frozen=True)
class BranchComparisonEntry:
    """Comparative metrics for a single branch within a multi-branch comparison."""

    branch_id: str
    label: str
    triggering_choice: str
    probability: float
    probability_level: ProbabilityLevel
    confidence_overall: float
    risk_composite_score: float
    opportunity_composite_score: float
    future_state_horizon: str
    rank_by_probability: int    # 1 = highest probability
    rank_by_confidence: int
    rank_by_risk: int           # 1 = lowest risk (best)
    rank_by_opportunity: int    # 1 = highest opportunity (best)


# ---------------------------------------------------------------------------
# Branch Analysis Engine Implementation
# ---------------------------------------------------------------------------


class BranchAnalysisEngine(IBranchAnalysisEngine):
    """
    Production implementation of IBranchAnalysisEngine.

    Thread-safe. All public methods acquire the internal RLock.

    Bounded Exploration Enforcement:
        - construct_branch: validates probability and confidence thresholds,
          enforces per-scenario branch count, enforces total capacity, enforces
          depth limits.
        - prune_branches_by_probability / prune_branches_by_confidence: explicit
          pruning passes for resource management.
        - _evict_if_at_capacity: protects the global ceiling.

    Constitutional Invariants:
        - This engine never selects a branch. It only explores and compares.
        - Any operation that implies selection raises
          JanusConstitutionalViolationError.
    """

    def __init__(
        self,
        config: Optional[BranchAnalysisConfig] = None,
    ) -> None:
        self._config: BranchAnalysisConfig = config or BranchAnalysisConfig()
        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False
        self._shut_down: bool = False

        # Primary store: branch_id → _BranchRecord
        self._records: dict[str, _BranchRecord] = {}

        # Scenario index: scenario_id → set[branch_id]
        self._scenario_index: dict[str, set[str]] = {}

        # Lineage index: branch_id → parent_branch_id (None = root)
        self._parent_index: dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                raise JanusAlreadyInitializedError(_ENGINE_NAME)
            self._records.clear()
            self._scenario_index.clear()
            self._parent_index.clear()
            self._initialized = True
            self._shut_down = False
            _LOG.info(
                "[%s] initialized (version=%s, config=%r)",
                _ENGINE_NAME,
                _ENGINE_VERSION,
                self._config,
            )

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                raise JanusNotInitializedError(_ENGINE_NAME)
            if self._shut_down:
                return
            branch_count = len(self._records)
            self._records.clear()
            self._scenario_index.clear()
            self._parent_index.clear()
            self._shut_down = True
            _LOG.info(
                "[%s] shut down (released %d branch records)",
                _ENGINE_NAME,
                branch_count,
            )

    @property
    def is_initialized(self) -> bool:
        return self._initialized and not self._shut_down

    @property
    def engine_name(self) -> str:
        return _ENGINE_NAME

    @property
    def engine_version(self) -> str:
        return _ENGINE_VERSION

    # ------------------------------------------------------------------
    # IBranchAnalysisEngine — Core Operations
    # ------------------------------------------------------------------

    def construct_branch(
        self,
        label: str,
        description: str,
        triggering_choice: str,
        future_state: FutureState,
        probability: float,
        risk_assessment: RiskAssessment,
        opportunity_assessment: OpportunityAssessment,
        confidence: ConfidenceProfile,
        *,
        scenario_id: str,
        parent_branch_id: Optional[str] = None,
    ) -> ScenarioBranch:
        """
        Construct and register a ScenarioBranch.

        Extended signature adds `scenario_id` and optional `parent_branch_id`
        to support lineage tracking.  The base interface contract is honoured;
        these keyword-only extensions are additive.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchConstructionError: invalid inputs.
            JanusBranchAnalysisError: bounded-exploration limits exceeded.
        """
        self._assert_operational()
        self._validate_construct_inputs(
            label=label,
            description=description,
            triggering_choice=triggering_choice,
            probability=probability,
            scenario_id=scenario_id,
            parent_branch_id=parent_branch_id,
        )

        with self._lock:
            self._assert_operational()

            # Bounded-exploration: probability floor
            if probability < self._config.min_probability_threshold:
                raise JanusBranchConstructionError(
                    f"Branch probability {probability:.4f} is below the minimum threshold "
                    f"{self._config.min_probability_threshold:.4f}. "
                    "Bounded exploration: low-probability branches are pruned.",
                    context={
                        "probability": probability,
                        "threshold": self._config.min_probability_threshold,
                        "scenario_id": scenario_id,
                    },
                )

            # Bounded-exploration: confidence floor
            if confidence.overall < self._config.min_confidence_threshold:
                raise JanusBranchConstructionError(
                    f"Branch confidence {confidence.overall:.4f} is below the minimum "
                    f"threshold {self._config.min_confidence_threshold:.4f}. "
                    "Bounded exploration: low-confidence branches are pruned.",
                    context={
                        "confidence": confidence.overall,
                        "threshold": self._config.min_confidence_threshold,
                        "scenario_id": scenario_id,
                    },
                )

            # Bounded-exploration: per-scenario branch count
            existing_ids = self._scenario_index.get(scenario_id, set())
            active_count = sum(
                1
                for bid in existing_ids
                if not self._records[bid].pruned
            )
            if active_count >= self._config.max_branches_per_scenario:
                raise JanusBranchAnalysisError(
                    f"Scenario '{scenario_id}' already has {active_count} active branches "
                    f"(limit={self._config.max_branches_per_scenario}). "
                    "Bounded exploration: per-scenario branch limit reached.",
                    engine=_ENGINE_NAME,
                    context={
                        "scenario_id": scenario_id,
                        "active_count": active_count,
                        "limit": self._config.max_branches_per_scenario,
                    },
                )

            # Bounded-exploration: depth
            depth = self._compute_depth(parent_branch_id)
            if depth > self._config.max_branch_depth:
                raise JanusBranchAnalysisError(
                    f"Branch depth {depth} exceeds maximum allowed depth "
                    f"{self._config.max_branch_depth}. "
                    "Bounded exploration: branch tree is too deep.",
                    engine=_ENGINE_NAME,
                    context={
                        "depth": depth,
                        "max_depth": self._config.max_branch_depth,
                        "parent_branch_id": parent_branch_id,
                    },
                )

            # Bounded-exploration: total capacity
            total_active = sum(1 for r in self._records.values() if not r.pruned)
            if total_active >= self._config.max_total_active_branches:
                raise JanusBranchAnalysisError(
                    f"Total active branch count {total_active} has reached the "
                    f"global ceiling {self._config.max_total_active_branches}. "
                    "Bounded exploration: global capacity exhausted.",
                    engine=_ENGINE_NAME,
                    context={
                        "total_active": total_active,
                        "limit": self._config.max_total_active_branches,
                    },
                )

            try:
                branch = ScenarioBranch.create(
                    label=label,
                    description=description,
                    triggering_choice=triggering_choice,
                    future_state=future_state,
                    probability=probability,
                    risk_assessment=risk_assessment,
                    opportunity_assessment=opportunity_assessment,
                    confidence=confidence,
                )
            except Exception as exc:
                raise JanusBranchConstructionError(
                    f"ScenarioBranch.create failed: {exc}",
                    context={"scenario_id": scenario_id},
                ) from exc

            record = _BranchRecord(
                branch=branch,
                scenario_id=scenario_id,
                parent_branch_id=parent_branch_id,
                depth=depth,
                registered_at=datetime.utcnow(),
            )
            self._records[branch.branch_id] = record
            self._scenario_index.setdefault(scenario_id, set()).add(branch.branch_id)
            self._parent_index[branch.branch_id] = parent_branch_id

            _LOG.debug(
                "[%s] constructed branch '%s' for scenario '%s' "
                "(depth=%d, probability=%.4f, parent=%s)",
                _ENGINE_NAME,
                branch.branch_id,
                scenario_id,
                depth,
                probability,
                parent_branch_id,
            )
            return branch

    def get_branch(self, branch_id: str) -> ScenarioBranch:
        """
        Retrieve a ScenarioBranch by its branch_id.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: branch_id does not exist.
        """
        self._assert_operational()
        if not branch_id or not branch_id.strip():
            raise JanusValidationError(
                "branch_id must be a non-empty string.",
                field="branch_id",
                value=branch_id,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            record = self._records.get(branch_id)
            if record is None:
                raise JanusBranchNotFoundError(branch_id)
            return record.branch

    def analyze_branches(self, scenario_id: str) -> tuple[ScenarioBranch, ...]:
        """
        Return all active (non-pruned) ScenarioBranches for the given scenario,
        ordered by probability descending.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: scenario_id is blank.
        """
        self._assert_operational()
        if not scenario_id or not scenario_id.strip():
            raise JanusValidationError(
                "scenario_id must be a non-empty string.",
                field="scenario_id",
                value=scenario_id,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            branch_ids = self._scenario_index.get(scenario_id, set())
            active_branches = [
                self._records[bid].branch
                for bid in branch_ids
                if not self._records[bid].pruned
            ]
            active_branches.sort(key=lambda b: b.probability, reverse=True)
            return tuple(active_branches)

    def validate_branch_probabilities(self, scenario_id: str) -> bool:
        """
        Validate that the active branch probabilities within a scenario do not
        collectively exceed 1.0 (mutual exclusivity invariant).

        Note: branches within a scenario may be overlapping (non-exclusive), so
        we validate that no single branch has probability > 1.0 and that the
        sum does not exceed 1.0 + tolerance (exclusive branch model).

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchProbabilityConflictError: probabilities are inconsistent.
        """
        self._assert_operational()
        if not scenario_id or not scenario_id.strip():
            raise JanusValidationError(
                "scenario_id must be a non-empty string.",
                field="scenario_id",
                value=scenario_id,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            branch_ids = self._scenario_index.get(scenario_id, set())
            probabilities = [
                self._records[bid].branch.probability
                for bid in branch_ids
                if not self._records[bid].pruned
            ]
            if not probabilities:
                return True
            total = sum(probabilities)
            _TOLERANCE = 1e-6
            if total > 1.0 + _TOLERANCE:
                raise JanusBranchProbabilityConflictError(
                    scenario_id=scenario_id,
                    total=total,
                )
            return True

    def dominant_branch(self, scenario_id: str) -> ScenarioBranch:
        """
        Return the active ScenarioBranch with the highest probability for
        the given scenario.

        Constitutional note: returning the dominant branch for inspection is not
        selection. VEGA selects; JANUS reports.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: no active branches exist for the scenario.
        """
        self._assert_operational()
        branches = self.analyze_branches(scenario_id)
        if not branches:
            raise JanusBranchNotFoundError(
                f"No active branches found for scenario '{scenario_id}'."
            )
        return branches[0]

    def compare_branches(
        self, branch_ids: tuple[str, ...]
    ) -> dict[str, dict[str, Any]]:
        """
        Produce a structured comparison of the given ScenarioBranches.

        Returns a dict keyed by branch_id containing comparative metrics:
            - label
            - triggering_choice
            - probability
            - probability_level
            - confidence_overall
            - risk_composite_score
            - opportunity_composite_score
            - future_state_horizon
            - rank_by_probability  (1 = highest)
            - rank_by_confidence
            - rank_by_risk         (1 = lowest risk = best)
            - rank_by_opportunity  (1 = highest opportunity = best)

        Constitutional note: comparison ranks branches. It never selects one.
        VEGA selects.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: any branch_id does not exist.
            JanusValidationError: branch_ids is empty.
        """
        self._assert_operational()
        if not branch_ids:
            raise JanusValidationError(
                "branch_ids must contain at least one branch_id.",
                field="branch_ids",
                value=branch_ids,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            entries: list[BranchComparisonEntry] = []
            for bid in branch_ids:
                record = self._records.get(bid)
                if record is None:
                    raise JanusBranchNotFoundError(bid)
                b = record.branch
                entries.append(
                    BranchComparisonEntry(
                        branch_id=bid,
                        label=b.label,
                        triggering_choice=b.triggering_choice,
                        probability=b.probability,
                        probability_level=ProbabilityLevel.from_float(b.probability),
                        confidence_overall=b.confidence.overall,
                        risk_composite_score=b.risk_assessment.composite_risk_score,
                        opportunity_composite_score=(
                            b.opportunity_assessment.composite_opportunity_score
                        ),
                        future_state_horizon=b.future_state.horizon.label,
                        rank_by_probability=0,
                        rank_by_confidence=0,
                        rank_by_risk=0,
                        rank_by_opportunity=0,
                    )
                )

            # Compute ranks
            entries_by_prob = sorted(entries, key=lambda e: e.probability, reverse=True)
            entries_by_conf = sorted(
                entries, key=lambda e: e.confidence_overall, reverse=True
            )
            entries_by_risk = sorted(
                entries, key=lambda e: e.risk_composite_score
            )  # lower risk = better
            entries_by_opp = sorted(
                entries, key=lambda e: e.opportunity_composite_score, reverse=True
            )

            rank_prob = {e.branch_id: i + 1 for i, e in enumerate(entries_by_prob)}
            rank_conf = {e.branch_id: i + 1 for i, e in enumerate(entries_by_conf)}
            rank_risk = {e.branch_id: i + 1 for i, e in enumerate(entries_by_risk)}
            rank_opp = {e.branch_id: i + 1 for i, e in enumerate(entries_by_opp)}

            result: dict[str, dict[str, Any]] = {}
            for e in entries:
                result[e.branch_id] = {
                    "label": e.label,
                    "triggering_choice": e.triggering_choice,
                    "probability": e.probability,
                    "probability_level": e.probability_level.value,
                    "confidence_overall": e.confidence_overall,
                    "risk_composite_score": e.risk_composite_score,
                    "opportunity_composite_score": e.opportunity_composite_score,
                    "future_state_horizon": e.future_state_horizon,
                    "rank_by_probability": rank_prob[e.branch_id],
                    "rank_by_confidence": rank_conf[e.branch_id],
                    "rank_by_risk": rank_risk[e.branch_id],
                    "rank_by_opportunity": rank_opp[e.branch_id],
                }
            return result

    # ------------------------------------------------------------------
    # Lineage & Ancestry
    # ------------------------------------------------------------------

    def get_branch_lineage(self, branch_id: str) -> BranchLineage:
        """
        Return the full lineage chain from a branch back to its scenario root.

        The `ancestors` tuple is ordered root → direct parent (branch_ids).
        Depth 0 = root branch (no parent).

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: branch_id does not exist.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            record = self._records.get(branch_id)
            if record is None:
                raise JanusBranchNotFoundError(branch_id)
            ancestors = self._collect_ancestors(branch_id)
            return BranchLineage(
                branch_id=branch_id,
                scenario_id=record.scenario_id,
                ancestors=tuple(ancestors),
                depth=record.depth,
            )

    def get_branch_ancestry(self, branch_id: str) -> tuple[ScenarioBranch, ...]:
        """
        Return the ordered list of ancestor ScenarioBranch objects from the
        scenario root down to the direct parent of the requested branch.

        Returns an empty tuple for a root branch.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: branch_id or any ancestor does not exist.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            record = self._records.get(branch_id)
            if record is None:
                raise JanusBranchNotFoundError(branch_id)
            ancestor_ids = self._collect_ancestors(branch_id)
            ancestors: list[ScenarioBranch] = []
            for aid in ancestor_ids:
                ar = self._records.get(aid)
                if ar is None:
                    raise JanusBranchNotFoundError(
                        f"Ancestor branch '{aid}' referenced by '{branch_id}' "
                        "no longer exists in the engine store."
                    )
                ancestors.append(ar.branch)
            return tuple(ancestors)

    def get_branch_children(
        self, parent_branch_id: str
    ) -> tuple[ScenarioBranch, ...]:
        """
        Return all active (non-pruned) direct children of the given branch.

        Ordered by probability descending.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: parent_branch_id does not exist.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            if parent_branch_id not in self._records:
                raise JanusBranchNotFoundError(parent_branch_id)
            children = [
                r.branch
                for r in self._records.values()
                if r.parent_branch_id == parent_branch_id and not r.pruned
            ]
            children.sort(key=lambda b: b.probability, reverse=True)
            return tuple(children)

    def get_branch_depth(self, branch_id: str) -> int:
        """
        Return the depth of a branch in its lineage tree.

        Depth 0 = root branch.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: branch_id does not exist.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            record = self._records.get(branch_id)
            if record is None:
                raise JanusBranchNotFoundError(branch_id)
            return record.depth

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_branches(
        self,
        scenario_id: str,
        *,
        by: str = "probability",
    ) -> tuple[ScenarioBranch, ...]:
        """
        Return all active ScenarioBranches for a scenario ranked by the
        specified criterion.

        `by` accepts: 'probability', 'confidence', 'risk' (ascending,
        lower = better), 'opportunity' (descending, higher = better).

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: unknown ranking criterion or blank scenario_id.
        """
        self._assert_operational()
        _VALID_CRITERIA = {"probability", "confidence", "risk", "opportunity"}
        if by not in _VALID_CRITERIA:
            raise JanusValidationError(
                f"Unknown ranking criterion '{by}'. "
                f"Valid values: {sorted(_VALID_CRITERIA)}.",
                field="by",
                value=by,
                engine=_ENGINE_NAME,
            )
        branches = list(self.analyze_branches(scenario_id))  # already sorted by prob
        if by == "probability":
            pass  # already sorted
        elif by == "confidence":
            branches.sort(key=lambda b: b.confidence.overall, reverse=True)
        elif by == "risk":
            branches.sort(key=lambda b: b.risk_assessment.composite_risk_score)
        elif by == "opportunity":
            branches.sort(
                key=lambda b: b.opportunity_assessment.composite_opportunity_score,
                reverse=True,
            )
        return tuple(branches)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_branch(self, branch_id: str, reason: str) -> None:
        """
        Manually prune a specific branch by its branch_id.

        Pruned branches are retained in the store (for diagnostics) but
        excluded from all operational queries.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: branch_id does not exist.
        """
        self._assert_operational()
        if not branch_id or not branch_id.strip():
            raise JanusValidationError(
                "branch_id must be a non-empty string.",
                field="branch_id",
                value=branch_id,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            record = self._records.get(branch_id)
            if record is None:
                raise JanusBranchNotFoundError(branch_id)
            if not record.pruned:
                record.pruned = True
                record.prune_reason = reason
                _LOG.debug(
                    "[%s] pruned branch '%s': %s",
                    _ENGINE_NAME,
                    branch_id,
                    reason,
                )

    def prune_branches_by_probability(
        self, scenario_id: str, threshold: Optional[float] = None
    ) -> int:
        """
        Prune all active branches in a scenario whose probability falls below
        `threshold`.  If threshold is None, the engine's configured
        min_probability_threshold is used.

        Returns the count of newly pruned branches.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: invalid threshold.
        """
        self._assert_operational()
        effective_threshold = (
            threshold
            if threshold is not None
            else self._config.min_probability_threshold
        )
        if not 0.0 <= effective_threshold <= 1.0:
            raise JanusValidationError(
                f"Probability threshold must be in [0, 1], got {effective_threshold}.",
                field="threshold",
                value=effective_threshold,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            branch_ids = self._scenario_index.get(scenario_id, set())
            pruned_count = 0
            for bid in branch_ids:
                record = self._records[bid]
                if (
                    not record.pruned
                    and record.branch.probability < effective_threshold
                ):
                    record.pruned = True
                    record.prune_reason = (
                        f"probability {record.branch.probability:.4f} "
                        f"< threshold {effective_threshold:.4f}"
                    )
                    pruned_count += 1
            if pruned_count:
                _LOG.debug(
                    "[%s] pruned %d branch(es) in scenario '%s' "
                    "(probability < %.4f)",
                    _ENGINE_NAME,
                    pruned_count,
                    scenario_id,
                    effective_threshold,
                )
            return pruned_count

    def prune_branches_by_confidence(
        self, scenario_id: str, threshold: Optional[float] = None
    ) -> int:
        """
        Prune all active branches in a scenario whose ConfidenceProfile.overall
        falls below `threshold`. If threshold is None, the engine's configured
        min_confidence_threshold is used.

        Returns the count of newly pruned branches.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusValidationError: invalid threshold.
        """
        self._assert_operational()
        effective_threshold = (
            threshold
            if threshold is not None
            else self._config.min_confidence_threshold
        )
        if not 0.0 <= effective_threshold <= 1.0:
            raise JanusValidationError(
                f"Confidence threshold must be in [0, 1], got {effective_threshold}.",
                field="threshold",
                value=effective_threshold,
                engine=_ENGINE_NAME,
            )
        with self._lock:
            self._assert_operational()
            branch_ids = self._scenario_index.get(scenario_id, set())
            pruned_count = 0
            for bid in branch_ids:
                record = self._records[bid]
                if (
                    not record.pruned
                    and record.branch.confidence.overall < effective_threshold
                ):
                    record.pruned = True
                    record.prune_reason = (
                        f"confidence {record.branch.confidence.overall:.4f} "
                        f"< threshold {effective_threshold:.4f}"
                    )
                    pruned_count += 1
            if pruned_count:
                _LOG.debug(
                    "[%s] pruned %d branch(es) in scenario '%s' "
                    "(confidence < %.4f)",
                    _ENGINE_NAME,
                    pruned_count,
                    scenario_id,
                    effective_threshold,
                )
            return pruned_count

    def prune_all_branches_for_scenario(
        self, scenario_id: str, reason: str
    ) -> int:
        """
        Prune all active branches registered under a scenario.

        Used when a scenario is invalidated or archived.

        Returns the count of newly pruned branches.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            branch_ids = self._scenario_index.get(scenario_id, set())
            pruned_count = 0
            for bid in branch_ids:
                record = self._records[bid]
                if not record.pruned:
                    record.pruned = True
                    record.prune_reason = reason
                    pruned_count += 1
            if pruned_count:
                _LOG.debug(
                    "[%s] pruned all %d branch(es) for scenario '%s': %s",
                    _ENGINE_NAME,
                    pruned_count,
                    scenario_id,
                    reason,
                )
            return pruned_count

    # ------------------------------------------------------------------
    # Statistics & Observability
    # ------------------------------------------------------------------

    def get_statistics(self) -> BranchStatistics:
        """
        Return a point-in-time statistics snapshot.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            all_records = list(self._records.values())
            total = len(all_records)
            active_records = [r for r in all_records if not r.pruned]
            pruned = total - len(active_records)
            scenarios = set(self._scenario_index.keys())

            branch_count_by_scenario: dict[str, int] = {}
            for sid, bids in self._scenario_index.items():
                branch_count_by_scenario[sid] = sum(
                    1 for bid in bids if not self._records[bid].pruned
                )

            avg_per_scenario = (
                (len(active_records) / len(scenarios)) if scenarios else 0.0
            )
            avg_prob = (
                sum(r.branch.probability for r in active_records) / len(active_records)
                if active_records
                else 0.0
            )
            avg_conf = (
                sum(r.branch.confidence.overall for r in active_records)
                / len(active_records)
                if active_records
                else 0.0
            )
            max_depth = max((r.depth for r in all_records), default=0)

            return BranchStatistics(
                total_branches=total,
                active_branches=len(active_records),
                pruned_branches=pruned,
                scenarios_tracked=len(scenarios),
                average_branches_per_scenario=avg_per_scenario,
                average_branch_probability=avg_prob,
                average_branch_confidence=avg_conf,
                deepest_lineage_depth=max_depth,
                branch_count_by_scenario=branch_count_by_scenario,
                generated_at=datetime.utcnow(),
            )

    def get_health_report(self) -> BranchHealthReport:
        """
        Return a lifecycle and capacity health report.

        Safe to call even before initialization or after shutdown.
        """
        with self._lock:
            total = len(self._records)
            active = sum(1 for r in self._records.values() if not r.pruned)
            pruned = total - active
            scenarios = len(self._scenario_index)
            capacity = (
                active / self._config.max_total_active_branches
                if self._config.max_total_active_branches > 0
                else 0.0
            )
            return BranchHealthReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                is_initialized=self._initialized,
                is_shut_down=self._shut_down,
                total_branches=total,
                active_branches=active,
                pruned_branches=pruned,
                scenarios_tracked=scenarios,
                capacity_utilization=capacity,
                config=self._config,
                generated_at=datetime.utcnow(),
            )

    def get_diagnostics_report(self) -> BranchDiagnosticsReport:
        """
        Return a detailed diagnostics snapshot for engineering inspection.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            self._assert_operational()
            all_records = list(self._records.values())
            total = len(all_records)
            active = sum(1 for r in all_records if not r.pruned)
            pruned_ids = [r.branch_id for r in all_records if r.pruned]

            scenario_branch_counts: dict[str, int] = {}
            scenario_max_depths: dict[str, int] = {}
            branch_ids_by_scenario: dict[str, list[str]] = {}

            for sid, bids in self._scenario_index.items():
                active_bids = [
                    bid for bid in bids if not self._records[bid].pruned
                ]
                scenario_branch_counts[sid] = len(active_bids)
                depths = [self._records[bid].depth for bid in bids]
                scenario_max_depths[sid] = max(depths) if depths else 0
                branch_ids_by_scenario[sid] = list(bids)

            return BranchDiagnosticsReport(
                engine_name=_ENGINE_NAME,
                engine_version=_ENGINE_VERSION,
                is_initialized=self._initialized,
                config=self._config,
                total_records=total,
                active_records=active,
                pruned_records=total - active,
                scenario_branch_counts=scenario_branch_counts,
                scenario_max_depths=scenario_max_depths,
                branch_ids_by_scenario=branch_ids_by_scenario,
                pruned_branch_ids=pruned_ids,
                generated_at=datetime.utcnow(),
            )

    # ------------------------------------------------------------------
    # Branch Analysis Utilities
    # ------------------------------------------------------------------

    def branch_exists(self, branch_id: str) -> bool:
        """
        Return True if a branch with the given branch_id exists (active or pruned).

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            return branch_id in self._records

    def is_branch_pruned(self, branch_id: str) -> bool:
        """
        Return True if the branch has been pruned.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: branch_id does not exist.
        """
        self._assert_operational()
        with self._lock:
            record = self._records.get(branch_id)
            if record is None:
                raise JanusBranchNotFoundError(branch_id)
            return record.pruned

    def get_scenario_for_branch(self, branch_id: str) -> str:
        """
        Return the scenario_id under which a branch is registered.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
            JanusBranchNotFoundError: branch_id does not exist.
        """
        self._assert_operational()
        with self._lock:
            record = self._records.get(branch_id)
            if record is None:
                raise JanusBranchNotFoundError(branch_id)
            return record.scenario_id

    def list_scenario_ids(self) -> tuple[str, ...]:
        """
        Return all scenario_ids currently tracked by the engine.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            return tuple(self._scenario_index.keys())

    def count_active_branches(self, scenario_id: str) -> int:
        """
        Return the count of non-pruned branches for a scenario.

        Returns 0 if the scenario is not tracked.

        Raises:
            JanusNotInitializedError: engine not yet initialized.
            JanusShutdownError: engine has been shut down.
        """
        self._assert_operational()
        with self._lock:
            bids = self._scenario_index.get(scenario_id, set())
            return sum(1 for bid in bids if not self._records[bid].pruned)

    def assert_no_selection(self, operation: str) -> None:
        """
        Constitutional guard: raise JanusConstitutionalViolationError if
        `operation` implies branch selection.

        This method is available for callers (e.g., orchestrators) that want
        to document constitutional compliance explicitly.

        Selection vocabulary that triggers violation: any operation name
        containing 'select', 'choose', 'decide', 'approve', 'commit'.
        """
        _SELECTION_TERMS = {"select", "choose", "decide", "approve", "commit"}
        lower_op = operation.lower()
        for term in _SELECTION_TERMS:
            if term in lower_op:
                raise JanusConstitutionalViolationError(
                    engine=_ENGINE_NAME,
                    operation=operation,
                    rightful_owner="VEGA",
                )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _assert_operational(self) -> None:
        """Raise the appropriate lifecycle exception if the engine is not usable."""
        if not self._initialized:
            raise JanusNotInitializedError(_ENGINE_NAME)
        if self._shut_down:
            raise JanusShutdownError(_ENGINE_NAME)

    def _validate_construct_inputs(
        self,
        *,
        label: str,
        description: str,
        triggering_choice: str,
        probability: float,
        scenario_id: str,
        parent_branch_id: Optional[str],
    ) -> None:
        """Validate raw inputs before acquiring the lock."""
        if not label or not label.strip():
            raise JanusValidationError(
                "label must be a non-empty string.",
                field="label",
                value=label,
                engine=_ENGINE_NAME,
            )
        if not description or not description.strip():
            raise JanusValidationError(
                "description must be a non-empty string.",
                field="description",
                value=description,
                engine=_ENGINE_NAME,
            )
        if not triggering_choice or not triggering_choice.strip():
            raise JanusValidationError(
                "triggering_choice must be a non-empty string.",
                field="triggering_choice",
                value=triggering_choice,
                engine=_ENGINE_NAME,
            )
        if not 0.0 <= probability <= 1.0:
            raise JanusValidationError(
                f"probability must be in [0, 1], got {probability}.",
                field="probability",
                value=probability,
                engine=_ENGINE_NAME,
            )
        if not scenario_id or not scenario_id.strip():
            raise JanusValidationError(
                "scenario_id must be a non-empty string.",
                field="scenario_id",
                value=scenario_id,
                engine=_ENGINE_NAME,
            )
        if parent_branch_id is not None and (
            not isinstance(parent_branch_id, str) or not parent_branch_id.strip()
        ):
            raise JanusValidationError(
                "parent_branch_id must be None or a non-empty string.",
                field="parent_branch_id",
                value=parent_branch_id,
                engine=_ENGINE_NAME,
            )

    def _compute_depth(self, parent_branch_id: Optional[str]) -> int:
        """
        Compute the depth of a new branch given its parent.

        Depth 0 = root (no parent).
        Depth n = parent.depth + 1.

        Raises JanusBranchNotFoundError if parent_branch_id is provided but
        does not exist in the store.
        """
        if parent_branch_id is None:
            return 0
        parent_record = self._records.get(parent_branch_id)
        if parent_record is None:
            raise JanusBranchNotFoundError(
                f"Parent branch '{parent_branch_id}' does not exist. "
                "Cannot compute lineage depth."
            )
        return parent_record.depth + 1

    def _collect_ancestors(self, branch_id: str) -> list[str]:
        """
        Walk the parent chain from branch_id back to the root, returning
        ancestor branch_ids ordered root → direct parent.

        Does not include branch_id itself.
        """
        ancestors: list[str] = []
        current = self._parent_index.get(branch_id)
        while current is not None:
            ancestors.append(current)
            current = self._parent_index.get(current)
        ancestors.reverse()  # root first
        return ancestors