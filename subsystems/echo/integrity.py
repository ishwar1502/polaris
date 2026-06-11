# subsystems/echo/integrity.py
"""
ECHO v1 Memory Integrity Engine.

Implements :class:`MemoryIntegrityEngine` — the production engine responsible
for auditing the structural consistency of ECHO's in-process memory store.

Responsibilities
----------------
* **Duplicate detection**           — experiences sharing the same
  (title, experience_type, occurred_at) triple.
* **Broken reference detection**    — ``metadata.session_id`` and entries in
  ``metadata.related_experience_ids`` that point to non-existent records.
* **Orphan record detection**       — experiences whose ``session_id``
  references an experience that is not of type ``SESSION``.
* **Memory consistency validation** — metadata field invariants:
  significance_score ∈ [0.0, 1.0], non-negative retrieval_count,
  consolidated/consolidation_at consistency, and self-referential IDs.
* **Integrity scoring**             — a [0.0, 1.0] health metric where 1.0
  is perfectly consistent and 0.0 indicates maximum violation density.
* **Audit report generation**       — a structured :class:`IntegrityReport`
  enumerating every violation with remediation recommendations.

Integration
-----------
Requires:
* :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
  for existence queries, full-store enumeration, and individual record
  fetches.
* :class:`~subsystems.echo.interfaces.ExperienceRetrievalEngineInterface`
  (held for future deep-scan extensions; injected for DI completeness).

Thread Safety
-------------
All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
before touching the experience store.  Check passes are isolated so that
a failure in one pass does not abort the audit — errors are collected into
:attr:`IntegrityReport.errors` instead.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, unique
from typing import Any

from subsystems.echo.exceptions import (
    BrokenReferenceError,
    DuplicateExperienceError,
    EchoNotInitializedError,
)
from subsystems.echo.interfaces import (
    ExperienceEngineInterface,
    ExperienceRetrievalEngineInterface,
)
from subsystems.echo.models import (
    Experience,
    ExperienceImportance,
    ExperienceType,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Violation Type Enumeration
# ---------------------------------------------------------------------------


@unique
class ViolationType(Enum):
    """Classification of a single integrity violation.

    Each member maps to a specific structural invariant that the
    :class:`MemoryIntegrityEngine` enforces across ECHO's memory store.
    """

    DUPLICATE_EXPERIENCE = auto()
    """Two or more experiences share a (title, type, occurred_at) triple."""

    BROKEN_SESSION_REFERENCE = auto()
    """``metadata.session_id`` references an experience_id that does not exist."""

    BROKEN_RELATED_REFERENCE = auto()
    """An entry in ``metadata.related_experience_ids`` does not exist in the store."""

    ORPHAN_SESSION_MEMBER = auto()
    """``metadata.session_id`` points to an experience that is not of type ``SESSION``."""

    SELF_REFERENTIAL_ID = auto()
    """An experience's own ``experience_id`` appears in its ``related_experience_ids``."""

    INVALID_SIGNIFICANCE_SCORE = auto()
    """``metadata.significance_score`` is outside the valid range ``[0.0, 1.0]``."""

    INVALID_RETRIEVAL_COUNT = auto()
    """``metadata.retrieval_count`` is negative."""

    CONSOLIDATION_TIMESTAMP_MISSING = auto()
    """``metadata.consolidated`` is ``True`` but ``metadata.consolidation_at`` is ``None``."""

    CONSOLIDATION_TIMESTAMP_SPURIOUS = auto()
    """``metadata.consolidation_at`` is set but ``metadata.consolidated`` is ``False``."""


# Penalty weight per violation type used by the integrity score formula.
# Higher weight = greater score reduction per occurrence.
_VIOLATION_WEIGHTS: dict[ViolationType, float] = {
    ViolationType.DUPLICATE_EXPERIENCE: 0.15,
    ViolationType.BROKEN_SESSION_REFERENCE: 0.20,
    ViolationType.BROKEN_RELATED_REFERENCE: 0.10,
    ViolationType.ORPHAN_SESSION_MEMBER: 0.08,
    ViolationType.SELF_REFERENTIAL_ID: 0.05,
    ViolationType.INVALID_SIGNIFICANCE_SCORE: 0.12,
    ViolationType.INVALID_RETRIEVAL_COUNT: 0.10,
    ViolationType.CONSOLIDATION_TIMESTAMP_MISSING: 0.06,
    ViolationType.CONSOLIDATION_TIMESTAMP_SPURIOUS: 0.04,
}


# ---------------------------------------------------------------------------
# Violation Record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntegrityViolation:
    """A single structural inconsistency found during an audit scan.

    Attributes
    ----------
    violation_type:
        Category as a :class:`ViolationType` member.
    affected_experience_id:
        UUID of the record containing the violation.
    description:
        Human-readable explanation of the invariant that was broken.
    detail:
        Optional machine-readable supplementary information (e.g. the
        missing reference ID or the out-of-range score value).
    """

    violation_type: ViolationType
    affected_experience_id: str
    description: str
    detail: str = ""

    def __str__(self) -> str:
        base = (
            f"[{self.violation_type.name}] "
            f"{self.affected_experience_id}: {self.description}"
        )
        return base + (f" (detail: {self.detail})" if self.detail else "")


# ---------------------------------------------------------------------------
# Integrity Report
# ---------------------------------------------------------------------------


@dataclass
class IntegrityReport:
    """Structured audit report produced by :meth:`MemoryIntegrityEngine.run_audit`.

    Attributes
    ----------
    total_experiences_scanned:
        Number of experience records examined in this audit run.
    integrity_score:
        Health metric in ``[0.0, 1.0]``.  ``1.0`` = no violations;
        ``0.0`` = maximum violation density across the store.
    violations:
        All :class:`IntegrityViolation` instances discovered, in check-pass
        order (duplicates → references → metadata consistency).
    violations_by_type:
        Violation count keyed by :attr:`ViolationType.name` string.
    duplicate_groups:
        Mapping of duplicate-key string → list of ``experience_id`` values
        that share that key.  Empty when no duplicates are found.
    broken_references:
        Mapping of ``experience_id`` → list of missing reference IDs
        (from both session and related-experience references).
    orphan_session_members:
        ``experience_id`` values whose ``session_id`` points to a non-SESSION
        experience.
    recommendations:
        Ordered list of human-readable remediation suggestions derived from
        the violation set.
    audited_at:
        UTC timestamp of this audit run.
    elapsed_seconds:
        Wall-clock seconds the audit took to complete.
    errors:
        Non-fatal errors encountered during the scan.  A non-empty list
        indicates that at least one check pass was partially skipped.
    """

    total_experiences_scanned: int = 0
    integrity_score: float = 1.0
    violations: list[IntegrityViolation] = field(default_factory=list)
    violations_by_type: dict[str, int] = field(default_factory=dict)
    duplicate_groups: dict[str, list[str]] = field(default_factory=dict)
    broken_references: dict[str, list[str]] = field(default_factory=dict)
    orphan_session_members: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    audited_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def violation_count(self) -> int:
        """Total number of violations across all categories."""
        return len(self.violations)

    @property
    def is_healthy(self) -> bool:
        """Return ``True`` when the integrity score is at or above ``0.90``."""
        return self.integrity_score >= 0.90

    @property
    def has_critical_violations(self) -> bool:
        """Return ``True`` if any broken-reference or duplicate violations exist.

        These categories indicate that references within the store are
        structurally unsound and must be resolved before retrieval results
        can be trusted.
        """
        critical_types = {
            ViolationType.BROKEN_SESSION_REFERENCE,
            ViolationType.BROKEN_RELATED_REFERENCE,
            ViolationType.DUPLICATE_EXPERIENCE,
        }
        return any(v.violation_type in critical_types for v in self.violations)

    def summary(self) -> str:
        """Return a one-line human-readable summary of the audit result."""
        status = "HEALTHY" if self.is_healthy else "DEGRADED"
        return (
            f"IntegrityReport [{status}] "
            f"score={self.integrity_score:.3f} "
            f"violations={self.violation_count} "
            f"scanned={self.total_experiences_scanned} "
            f"elapsed={self.elapsed_seconds:.3f}s"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# MemoryIntegrityEngine
# ---------------------------------------------------------------------------


class MemoryIntegrityEngine:
    """Production implementation of ECHO's Memory Integrity Engine.

    Performs a multi-pass structural audit of ECHO's experience store and
    produces an :class:`IntegrityReport` enumerating every inconsistency.

    Parameters
    ----------
    experience_engine:
        A running :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        that owns the experience store.
    retrieval_engine:
        A running :class:`~subsystems.echo.interfaces.ExperienceRetrievalEngineInterface`
        injected for DI completeness and future deep-scan extensions.
    duplicate_time_tolerance_seconds:
        Maximum absolute timestamp difference in seconds within which two
        experiences sharing the same (title, type) are considered temporal
        duplicates.  Defaults to ``0`` — exact timestamp matching only.

    Usage
    -----
    ::

        engine = MemoryIntegrityEngine(
            experience_engine=exp_engine,
            retrieval_engine=ret_engine,
        )
        engine.initialize()

        report = engine.run_audit()
        print(report.summary())

        if report.has_critical_violations:
            for v in report.violations:
                print(v)

        engine.shutdown()
    """

    def __init__(
        self,
        experience_engine: ExperienceEngineInterface,
        retrieval_engine: ExperienceRetrievalEngineInterface,
        *,
        duplicate_time_tolerance_seconds: int = 0,
    ) -> None:
        if duplicate_time_tolerance_seconds < 0:
            raise ValueError(
                "duplicate_time_tolerance_seconds must be >= 0; "
                f"got {duplicate_time_tolerance_seconds!r}."
            )

        self._experience_engine = experience_engine
        self._retrieval_engine = retrieval_engine
        self._duplicate_time_tolerance = duplicate_time_tolerance_seconds

        self._lock: threading.RLock = threading.RLock()
        self._running: bool = False

        _logger.debug(
            "MemoryIntegrityEngine constructed (dup_tolerance=%ds).",
            duplicate_time_tolerance_seconds,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use.

        Both dependent engines must already be initialised and running
        before this method is called.

        Raises
        ------
        EchoError
            If initialisation fails for any reason.
        """
        with self._lock:
            if self._running:
                _logger.warning(
                    "MemoryIntegrityEngine.initialize() called while already running."
                )
                return
            self._running = True
            _logger.info("MemoryIntegrityEngine initialised.")

    def shutdown(self) -> None:
        """Release resources and mark the engine as stopped."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            _logger.info("MemoryIntegrityEngine shut down.")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if the engine is not running."""
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _fetch_all_experiences(self) -> list[Experience]:
        """Return a full snapshot of all experiences from the backing store.

        Uses :meth:`~subsystems.echo.interfaces.ExperienceEngineInterface.query_experiences`
        with a ceiling limit large enough to capture every record in the
        current in-process store.  Returns an empty list on failure,
        logging the error, so that partial audits can still complete.
        """
        try:
            return self._experience_engine.query_experiences(limit=100_000)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "MemoryIntegrityEngine: could not fetch experiences for audit: %s", exc
            )
            return []

    def _duplicate_key(self, exp: Experience) -> str:
        """Produce the canonical deduplication key for *exp*.

        Key form: ``"{title_lower}|{type_name}|{occurred_at_isoformat}"``.

        When ``duplicate_time_tolerance_seconds > 0`` the timestamp is
        truncated to the nearest tolerance bucket (integer division) so
        that two records within the tolerance window hash to the same key.
        """
        ts = exp.occurred_at.isoformat()
        if self._duplicate_time_tolerance > 0:
            epoch = int(exp.occurred_at.timestamp())
            bucket = (epoch // self._duplicate_time_tolerance) * self._duplicate_time_tolerance
            ts = str(bucket)
        return f"{exp.title.strip().lower()}|{exp.experience_type.name}|{ts}"

    # ------------------------------------------------------------------
    # Check pass 1: Duplicate detection
    # ------------------------------------------------------------------

    def _check_duplicates(
        self,
        experiences: list[Experience],
    ) -> tuple[list[IntegrityViolation], dict[str, list[str]]]:
        """Detect experiences sharing the same canonical deduplication key.

        Two experiences are duplicates when they have the same normalised
        title, experience type, and (optionally bucketed) ``occurred_at``
        timestamp.  The ``experience_id`` is intentionally excluded from
        the key so that records with distinct IDs but identical content
        are still flagged.

        Returns
        -------
        tuple[list[IntegrityViolation], dict[str, list[str]]]
            A list of violations and a mapping of duplicate-key →
            [experience_id, ...] for every group with more than one member.
        """
        key_map: dict[str, list[str]] = {}
        for exp in experiences:
            key_map.setdefault(self._duplicate_key(exp), []).append(exp.experience_id)

        violations: list[IntegrityViolation] = []
        duplicate_groups: dict[str, list[str]] = {}

        for key, ids in key_map.items():
            if len(ids) <= 1:
                continue
            duplicate_groups[key] = ids
            peer_count = len(ids) - 1
            for exp_id in ids:
                violations.append(
                    IntegrityViolation(
                        violation_type=ViolationType.DUPLICATE_EXPERIENCE,
                        affected_experience_id=exp_id,
                        description=(
                            f"Experience shares the (title, type, occurred_at) key "
                            f"with {peer_count} other record(s)."
                        ),
                        detail=(
                            f"duplicate_key={key!r} "
                            f"group_size={len(ids)} "
                            f"group_ids={ids!r}"
                        ),
                    )
                )

        return violations, duplicate_groups

    # ------------------------------------------------------------------
    # Check pass 2: Reference validation
    # ------------------------------------------------------------------

    def _check_references(
        self,
        experiences: list[Experience],
        known_ids: set[str],
    ) -> tuple[list[IntegrityViolation], dict[str, list[str]], list[str]]:
        """Validate all cross-experience references for each record.

        Checks applied per experience:

        1. If ``metadata.session_id`` is set:
           a. The referenced ID must exist in *known_ids*.
           b. The referenced experience must have type ``SESSION``.
        2. For each entry in ``metadata.related_experience_ids``:
           a. The entry must not equal the experience's own ID.
           b. The entry must exist in *known_ids*.

        Parameters
        ----------
        experiences:
            Full snapshot of the experience store.
        known_ids:
            Set of all ``experience_id`` values currently in the store.

        Returns
        -------
        tuple
            * ``violations`` — all reference violations found.
            * ``broken_references`` — mapping of experience_id → list of
              missing reference IDs (union of session and related refs).
            * ``orphan_session_members`` — experience_ids whose session_id
              points to a non-SESSION-typed experience.
        """
        id_to_type: dict[str, ExperienceType] = {
            exp.experience_id: exp.experience_type for exp in experiences
        }

        violations: list[IntegrityViolation] = []
        broken_references: dict[str, list[str]] = {}
        orphan_session_members: list[str] = []

        for exp in experiences:
            exp_id = exp.experience_id
            meta = exp.metadata
            exp_broken: list[str] = []

            # --- session_id ---
            if meta.session_id is not None:
                if meta.session_id not in known_ids:
                    exp_broken.append(meta.session_id)
                    violations.append(
                        IntegrityViolation(
                            violation_type=ViolationType.BROKEN_SESSION_REFERENCE,
                            affected_experience_id=exp_id,
                            description=(
                                "metadata.session_id references an experience "
                                "that does not exist in the store."
                            ),
                            detail=f"missing_session_id={meta.session_id!r}",
                        )
                    )
                else:
                    ref_type = id_to_type.get(meta.session_id)
                    if ref_type is not None and ref_type != ExperienceType.SESSION:
                        orphan_session_members.append(exp_id)
                        violations.append(
                            IntegrityViolation(
                                violation_type=ViolationType.ORPHAN_SESSION_MEMBER,
                                affected_experience_id=exp_id,
                                description=(
                                    f"metadata.session_id references experience "
                                    f"'{meta.session_id}' which has type "
                                    f"'{ref_type.name}', not SESSION."
                                ),
                                detail=(
                                    f"referenced_id={meta.session_id!r} "
                                    f"referenced_type={ref_type.name}"
                                ),
                            )
                        )

            # --- related_experience_ids ---
            for ref_id in meta.related_experience_ids:
                if ref_id == exp_id:
                    violations.append(
                        IntegrityViolation(
                            violation_type=ViolationType.SELF_REFERENTIAL_ID,
                            affected_experience_id=exp_id,
                            description=(
                                "metadata.related_experience_ids contains the "
                                "experience's own experience_id."
                            ),
                            detail=f"self_ref_id={ref_id!r}",
                        )
                    )
                elif ref_id not in known_ids:
                    exp_broken.append(ref_id)
                    violations.append(
                        IntegrityViolation(
                            violation_type=ViolationType.BROKEN_RELATED_REFERENCE,
                            affected_experience_id=exp_id,
                            description=(
                                "metadata.related_experience_ids contains a "
                                "reference to an experience that does not exist."
                            ),
                            detail=f"missing_ref_id={ref_id!r}",
                        )
                    )

            if exp_broken:
                broken_references[exp_id] = exp_broken

        return violations, broken_references, orphan_session_members

    # ------------------------------------------------------------------
    # Check pass 3: Metadata consistency
    # ------------------------------------------------------------------

    def _check_metadata_consistency(
        self,
        experiences: list[Experience],
    ) -> list[IntegrityViolation]:
        """Validate numeric and boolean metadata field invariants.

        Invariants checked:

        * ``significance_score`` ∈ ``[0.0, 1.0]``.
        * ``retrieval_count`` ≥ ``0``.
        * ``consolidated is True`` → ``consolidation_at is not None``.
        * ``consolidated is False`` → ``consolidation_at is None``.

        Returns
        -------
        list[IntegrityViolation]
            All metadata consistency violations found.
        """
        violations: list[IntegrityViolation] = []

        for exp in experiences:
            exp_id = exp.experience_id
            meta = exp.metadata

            if not (0.0 <= meta.significance_score <= 1.0):
                violations.append(
                    IntegrityViolation(
                        violation_type=ViolationType.INVALID_SIGNIFICANCE_SCORE,
                        affected_experience_id=exp_id,
                        description=(
                            f"metadata.significance_score={meta.significance_score!r} "
                            "is outside the valid range [0.0, 1.0]."
                        ),
                        detail=f"score={meta.significance_score!r}",
                    )
                )

            if meta.retrieval_count < 0:
                violations.append(
                    IntegrityViolation(
                        violation_type=ViolationType.INVALID_RETRIEVAL_COUNT,
                        affected_experience_id=exp_id,
                        description=(
                            f"metadata.retrieval_count={meta.retrieval_count!r} "
                            "is negative."
                        ),
                        detail=f"retrieval_count={meta.retrieval_count!r}",
                    )
                )

            if meta.consolidated and meta.consolidation_at is None:
                violations.append(
                    IntegrityViolation(
                        violation_type=ViolationType.CONSOLIDATION_TIMESTAMP_MISSING,
                        affected_experience_id=exp_id,
                        description=(
                            "metadata.consolidated is True but "
                            "metadata.consolidation_at is None."
                        ),
                    )
                )
            elif not meta.consolidated and meta.consolidation_at is not None:
                violations.append(
                    IntegrityViolation(
                        violation_type=ViolationType.CONSOLIDATION_TIMESTAMP_SPURIOUS,
                        affected_experience_id=exp_id,
                        description=(
                            "metadata.consolidation_at is set but "
                            "metadata.consolidated is False."
                        ),
                        detail=(
                            f"consolidation_at={meta.consolidation_at.isoformat()!r}"
                        ),
                    )
                )

        return violations

    # ------------------------------------------------------------------
    # Score and report helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_integrity_score(
        violations: list[IntegrityViolation],
        total: int,
    ) -> float:
        """Compute the ``[0.0, 1.0]`` integrity score from violation weights.

        Formula: ``score = 1.0 - total_penalty / max(total, 1)``

        The penalty for each violation is its type weight (from
        :data:`_VIOLATION_WEIGHTS`).  Dividing by the store size ensures
        that a fixed number of violations in a large store penalises less
        than in a small store.  The result is clamped to ``[0.0, 1.0]``.

        An empty store returns ``1.0`` (vacuously consistent).
        """
        if total == 0 or not violations:
            return 1.0
        total_penalty = sum(
            _VIOLATION_WEIGHTS.get(v.violation_type, 0.05)
            for v in violations
        )
        raw = 1.0 - (total_penalty / total)
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _build_violations_by_type(
        violations: list[IntegrityViolation],
    ) -> dict[str, int]:
        """Build a violation-count index keyed by :attr:`ViolationType.name`."""
        counts: dict[str, int] = {}
        for v in violations:
            key = v.violation_type.name
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _build_recommendations(
        violations: list[IntegrityViolation],
        duplicate_groups: dict[str, list[str]],
        broken_references: dict[str, list[str]],
        orphan_session_members: list[str],
    ) -> list[str]:
        """Derive ordered remediation recommendations from the violation set.

        One recommendation is generated per non-zero violation category.
        The list ends with a clean-bill-of-health message when no violations
        were found.
        """
        recommendations: list[str] = []
        type_counts: dict[ViolationType, int] = {}
        for v in violations:
            type_counts[v.violation_type] = type_counts.get(v.violation_type, 0) + 1

        def _count(vt: ViolationType) -> int:
            return type_counts.get(vt, 0)

        n_dup = _count(ViolationType.DUPLICATE_EXPERIENCE)
        if n_dup:
            recommendations.append(
                f"Deduplicate {len(duplicate_groups)} experience group(s) "
                f"({n_dup} affected record(s)).  Retain the most recently "
                "recorded instance and delete or merge older copies."
            )

        n_broken_session = _count(ViolationType.BROKEN_SESSION_REFERENCE)
        if n_broken_session:
            recommendations.append(
                f"Clear or repair metadata.session_id on "
                f"{n_broken_session} experience(s) referencing non-existent "
                "session records.  Re-link to a valid SESSION experience or "
                "set session_id to None."
            )

        n_broken_related = _count(ViolationType.BROKEN_RELATED_REFERENCE)
        if n_broken_related:
            recommendations.append(
                f"Remove {n_broken_related} stale entry/entries from "
                f"metadata.related_experience_ids across "
                f"{len(broken_references)} experience(s)."
            )

        if orphan_session_members:
            recommendations.append(
                f"Fix {len(orphan_session_members)} experience(s) whose "
                "metadata.session_id points to a non-SESSION-typed experience.  "
                "Update the reference to a SESSION experience or clear the field."
            )

        n_self_ref = _count(ViolationType.SELF_REFERENTIAL_ID)
        if n_self_ref:
            recommendations.append(
                f"Remove {n_self_ref} self-referential entry/entries from "
                "metadata.related_experience_ids — an experience cannot "
                "reference itself."
            )

        n_sig = _count(ViolationType.INVALID_SIGNIFICANCE_SCORE)
        if n_sig:
            recommendations.append(
                f"Re-score {n_sig} experience(s) whose significance_score "
                "is outside [0.0, 1.0] by routing them through the "
                "Significance Engine."
            )

        n_ret = _count(ViolationType.INVALID_RETRIEVAL_COUNT)
        if n_ret:
            recommendations.append(
                f"Reset metadata.retrieval_count to 0 on {n_ret} "
                "experience(s) with a negative value."
            )

        n_ts_missing = _count(ViolationType.CONSOLIDATION_TIMESTAMP_MISSING)
        if n_ts_missing:
            recommendations.append(
                f"Set metadata.consolidation_at on {n_ts_missing} "
                "experience(s) that are marked consolidated but lack a "
                "consolidation timestamp."
            )

        n_ts_spurious = _count(ViolationType.CONSOLIDATION_TIMESTAMP_SPURIOUS)
        if n_ts_spurious:
            recommendations.append(
                f"Clear metadata.consolidation_at on {n_ts_spurious} "
                "unconsolidated experience(s) that carry a spurious "
                "consolidation timestamp."
            )

        if not recommendations:
            recommendations.append(
                "No violations detected.  "
                "ECHO memory store is structurally consistent."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_audit(self) -> IntegrityReport:
        """Execute a full integrity audit of ECHO's memory store.

        Runs three check passes in sequence:

        1. **Duplicate detection** — experiences sharing a
           (title, type, occurred_at) key.
        2. **Reference validation** — broken session/related references,
           orphan session members, and self-referential IDs.
        3. **Metadata consistency** — field range and flag invariants.

        Each pass is failure-isolated: an unexpected exception in one pass
        is recorded in :attr:`IntegrityReport.errors` rather than aborting
        the audit.

        Returns
        -------
        IntegrityReport
            Fully populated audit report.

        Raises
        ------
        EchoNotInitializedError
            If this engine has not been initialised.
        """
        self._assert_running("run_audit")

        started = time.monotonic()
        audited_at = datetime.now(timezone.utc)
        errors: list[str] = []

        _logger.info("MemoryIntegrityEngine: starting full audit.")

        with self._lock:
            experiences = self._fetch_all_experiences()
            total = len(experiences)
            known_ids: set[str] = {exp.experience_id for exp in experiences}

            all_violations: list[IntegrityViolation] = []
            duplicate_groups: dict[str, list[str]] = {}
            broken_references: dict[str, list[str]] = {}
            orphan_session_members: list[str] = []

            # Pass 1: Duplicate detection
            try:
                dup_violations, duplicate_groups = self._check_duplicates(experiences)
                all_violations.extend(dup_violations)
                _logger.debug(
                    "MemoryIntegrityEngine pass 1 (duplicates): "
                    "%d violation(s), %d group(s).",
                    len(dup_violations),
                    len(duplicate_groups),
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"Duplicate check failed: {exc}"
                errors.append(msg)
                _logger.error("MemoryIntegrityEngine: %s", msg)

            # Pass 2: Reference validation
            try:
                ref_violations, broken_references, orphan_session_members = (
                    self._check_references(experiences, known_ids)
                )
                all_violations.extend(ref_violations)
                _logger.debug(
                    "MemoryIntegrityEngine pass 2 (references): "
                    "%d violation(s), %d broken ref source(s), %d orphan(s).",
                    len(ref_violations),
                    len(broken_references),
                    len(orphan_session_members),
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"Reference check failed: {exc}"
                errors.append(msg)
                _logger.error("MemoryIntegrityEngine: %s", msg)

            # Pass 3: Metadata consistency
            try:
                meta_violations = self._check_metadata_consistency(experiences)
                all_violations.extend(meta_violations)
                _logger.debug(
                    "MemoryIntegrityEngine pass 3 (metadata): %d violation(s).",
                    len(meta_violations),
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"Metadata consistency check failed: {exc}"
                errors.append(msg)
                _logger.error("MemoryIntegrityEngine: %s", msg)

            integrity_score = self._compute_integrity_score(all_violations, total)
            violations_by_type = self._build_violations_by_type(all_violations)
            recommendations = self._build_recommendations(
                all_violations,
                duplicate_groups,
                broken_references,
                orphan_session_members,
            )
            elapsed = time.monotonic() - started

            report = IntegrityReport(
                total_experiences_scanned=total,
                integrity_score=integrity_score,
                violations=all_violations,
                violations_by_type=violations_by_type,
                duplicate_groups=duplicate_groups,
                broken_references=broken_references,
                orphan_session_members=orphan_session_members,
                recommendations=recommendations,
                audited_at=audited_at,
                elapsed_seconds=elapsed,
                errors=errors,
            )

        _logger.info("MemoryIntegrityEngine: %s", report.summary())
        return report

    def check_experience_integrity(
        self,
        experience_id: str | Experience,
    ) -> list[IntegrityViolation]:
        """Run all integrity checks on a single experience record.

        Provides per-record validation without scanning the full store.
        Fetches the target experience and the complete known-ID set from
        the store to resolve reference checks.

        Parameters
        ----------
        experience_id:
            UUID of the experience to audit, or an
            :class:`~subsystems.echo.models.Experience` instance whose
            ``experience_id`` will be used automatically.

        Returns
        -------
        list[IntegrityViolation]
            All violations found on this record.  An empty list indicates
            the record is structurally clean.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        ExperienceNotFoundError
            If no experience with *experience_id* exists in the store.
        """
        self._assert_running("check_experience_integrity")

        # Accept either a raw ID string or an Experience instance.
        if isinstance(experience_id, Experience):
            experience_id = experience_id.experience_id

        with self._lock:
            # Fetch target; raises ExperienceNotFoundError if absent.
            exp = self._experience_engine.get_experience(experience_id)
            all_experiences = self._fetch_all_experiences()
            known_ids: set[str] = {e.experience_id for e in all_experiences}

            violations: list[IntegrityViolation] = []

            # Metadata consistency (single record)
            violations.extend(self._check_metadata_consistency([exp]))

            # Reference checks (single record, full known-ID set)
            ref_violations, _, _ = self._check_references([exp], known_ids)
            violations.extend(ref_violations)

            # Duplicate check: count how many store records share this key
            candidate_key = self._duplicate_key(exp)
            peer_ids = [
                e.experience_id
                for e in all_experiences
                if e.experience_id != experience_id
                and self._duplicate_key(e) == candidate_key
            ]
            if peer_ids:
                group_ids = [experience_id] + peer_ids
                violations.append(
                    IntegrityViolation(
                        violation_type=ViolationType.DUPLICATE_EXPERIENCE,
                        affected_experience_id=experience_id,
                        description=(
                            f"Experience shares (title, type, occurred_at) key "
                            f"with {len(peer_ids)} other record(s)."
                        ),
                        detail=(
                            f"duplicate_key={candidate_key!r} "
                            f"group_size={len(group_ids)} "
                            f"group_ids={group_ids!r}"
                        ),
                    )
                )

        return violations

    def assert_reference_integrity(
        self,
        experience_id: str,
        ref_id: str,
    ) -> None:
        """Assert that *ref_id* resolves to an existing experience.

        Called by other ECHO engines before writing a cross-reference to
        guarantee that no broken references are introduced into the store.

        Parameters
        ----------
        experience_id:
            The experience that will hold the reference (used in the error
            message if the check fails).
        ref_id:
            The ``experience_id`` being referenced.

        Raises
        ------
        BrokenReferenceError
            If *ref_id* does not exist in the experience store.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("assert_reference_integrity")

        with self._lock:
            if not self._experience_engine.experience_exists(ref_id):
                raise BrokenReferenceError(
                    source_id=experience_id,
                    missing_ref=ref_id,
                )

    def assert_no_duplicate(self, experience: Experience) -> None:
        """Assert that *experience* is not a content-duplicate of an existing record.

        Compares the candidate's (title, type, occurred_at) key against
        every record currently in the store, ignoring the candidate's own
        ``experience_id`` so that update-before-insert patterns do not
        produce false positives.

        Parameters
        ----------
        experience:
            The candidate :class:`~subsystems.echo.models.Experience` to check.

        Raises
        ------
        DuplicateExperienceError
            If a record with an identical deduplication key exists.
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("assert_no_duplicate")

        with self._lock:
            candidate_key = self._duplicate_key(experience)
            all_experiences = self._fetch_all_experiences()
            peers = [
                e.experience_id
                for e in all_experiences
                if e.experience_id != experience.experience_id
                and self._duplicate_key(e) == candidate_key
            ]
            if peers:
                raise DuplicateExperienceError(
                    experience_id=experience.experience_id,
                    duplicate_count=len(peers) + 1,
                )

    def get_integrity_score(self) -> float:
        """Return the current integrity score without generating a full report.

        Runs the same three check passes as :meth:`run_audit` but discards
        all report artefacts and returns only the computed score.  Cheaper
        than a full audit when only the health metric is needed (e.g.
        periodic health-check polling).

        Returns
        -------
        float
            Integrity score in ``[0.0, 1.0]``.

        Raises
        ------
        EchoNotInitializedError
            If the engine has not been initialised.
        """
        self._assert_running("get_integrity_score")

        with self._lock:
            experiences = self._fetch_all_experiences()
            total = len(experiences)
            known_ids: set[str] = {exp.experience_id for exp in experiences}

            all_violations: list[IntegrityViolation] = []

            try:
                dup_v, _ = self._check_duplicates(experiences)
                all_violations.extend(dup_v)
            except Exception:  # noqa: BLE001
                pass

            try:
                ref_v, _, _ = self._check_references(experiences, known_ids)
                all_violations.extend(ref_v)
            except Exception:  # noqa: BLE001
                pass

            try:
                meta_v = self._check_metadata_consistency(experiences)
                all_violations.extend(meta_v)
            except Exception:  # noqa: BLE001
                pass

            return self._compute_integrity_score(all_violations, total)

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of this engine's current state.

        Returns
        -------
        dict[str, Any]
            Contains ``running``, ``engine``, and
            ``duplicate_time_tolerance_seconds``.
        """
        with self._lock:
            return {
                "running": self._running,
                "engine": "MemoryIntegrityEngine",
                "duplicate_time_tolerance_seconds": self._duplicate_time_tolerance,
            }