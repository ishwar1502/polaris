# subsystems/echo/retrieval.py
"""
ECHO v1 Experience Retrieval Engine.

Implements :class:`ExperienceRetrievalEngine` — the production implementation
of :class:`~subsystems.echo.interfaces.ExperienceRetrievalEngineInterface`.

The Retrieval Engine provides high-level query surfaces used by ORION and
ODYSSEY to interrogate ECHO's experience store.  Every fetch operation
increments the :attr:`~subsystems.echo.models.ExperienceMetadata.retrieval_count`
on all returned experiences and updates the ``last_retrieved_at`` timestamp.

Retrieval Modes
---------------
* **Semantic search**   — full-text match against title, description, context,
                          outcome, and tag names.  Scored by term frequency.
* **Tag search**        — exact and partial tag-name matching with optional
                          category filtering.
* **Time-range search** — window query on ``occurred_at`` with importance floor.
* **Similarity search** — find experiences sharing tags, type, or project refs
                          with a reference experience.
* **Session-aware**     — restrict any query to a specific session scope.
* **Context recall**    — natural-language query that assembles a structured
                          context bundle (experiences + timeline + narrative).

Thread Safety
-------------
All public methods acquire ``self._lock`` (a :class:`threading.RLock`)
before accessing shared mutable state.  The engine is safe for concurrent
reads and writes across multiple threads.

Architecture Note
-----------------
v1 stores all data in-process (dict-backed).  The retrieval engine holds a
**reference** to the :class:`~subsystems.echo.experience.ExperienceEngine`'s
store via the interface and never owns its own copy.  Future versions will
delegate to the Memory Gateway persistence layer.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from subsystems.echo.exceptions import (
    EchoNotInitializedError,
    ExperienceNotFoundError,
)
from subsystems.echo.interfaces import ExperienceEngineInterface, ExperienceRetrievalEngineInterface
from subsystems.echo.models import (
    Experience,
    ExperienceImportance,
    ExperienceType,
    MemoryTag,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMPORTANCE_ORDER: dict[ExperienceImportance, int] = {
    ExperienceImportance.LOW: 1,
    ExperienceImportance.MEDIUM: 2,
    ExperienceImportance.HIGH: 3,
    ExperienceImportance.CRITICAL: 4,
}

_DEFAULT_SEMANTIC_LIMIT = 20
_DEFAULT_SIMILAR_LIMIT = 10
_DEFAULT_TIME_RANGE_LIMIT = 50
_DEFAULT_TAG_LIMIT = 30
_DEFAULT_CONTEXT_BUNDLE_LIMIT = 15

# Weights for relevance scoring components
_TITLE_WEIGHT = 3.0
_DESCRIPTION_WEIGHT = 1.5
_CONTEXT_WEIGHT = 1.0
_OUTCOME_WEIGHT = 1.0
_TAG_WEIGHT = 2.0

# Recency decay half-life in days — more recent experiences rank higher
_RECENCY_HALF_LIFE_DAYS = 180.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split *text* into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _term_frequency_score(tokens: list[str], query_terms: set[str]) -> float:
    """Return the fraction of *query_terms* that appear in *tokens*, weighted
    by occurrence count.  Returns 0.0 if either sequence is empty."""
    if not tokens or not query_terms:
        return 0.0
    match_count = sum(1 for t in tokens if t in query_terms)
    coverage = len(query_terms & set(tokens)) / len(query_terms)
    frequency = match_count / len(tokens)
    return 0.7 * coverage + 0.3 * frequency


def _recency_factor(occurred_at: datetime) -> float:
    """Return a [0, 1] recency multiplier.  More recent → closer to 1."""
    now = datetime.now(timezone.utc)
    delta_days = (now - occurred_at).total_seconds() / 86400.0
    # Exponential decay: factor = 2^(-delta / half_life)
    return math.pow(2.0, -delta_days / _RECENCY_HALF_LIFE_DAYS)


def _importance_factor(importance: ExperienceImportance) -> float:
    """Map an importance tier to a [0.25, 1.0] multiplier."""
    return _IMPORTANCE_ORDER[importance] / 4.0


def _score_experience_for_query(
    experience: Experience,
    query_terms: set[str],
) -> float:
    """Compute a composite relevance score for *experience* against a set of
    query tokens.  Returns 0.0 for no match."""
    title_score = _term_frequency_score(
        _tokenize(experience.title), query_terms
    ) * _TITLE_WEIGHT

    desc_score = _term_frequency_score(
        _tokenize(experience.description), query_terms
    ) * _DESCRIPTION_WEIGHT

    ctx_score = _term_frequency_score(
        _tokenize(experience.context), query_terms
    ) * _CONTEXT_WEIGHT

    outcome_score = _term_frequency_score(
        _tokenize(experience.outcome), query_terms
    ) * _OUTCOME_WEIGHT

    tag_tokens = _tokenize(" ".join(experience.tag_names()))
    tag_score = _term_frequency_score(tag_tokens, query_terms) * _TAG_WEIGHT

    raw = title_score + desc_score + ctx_score + outcome_score + tag_score
    if raw == 0.0:
        return 0.0

    # Weight by recency and importance
    recency = _recency_factor(experience.occurred_at)
    importance = _importance_factor(experience.importance)

    return raw * (0.6 + 0.25 * recency + 0.15 * importance)


def _similarity_score(reference: Experience, candidate: Experience) -> float:
    """Compute a 0–1 similarity score between *reference* and *candidate*.

    Factors:
    * Shared tag names (weighted highest)
    * Shared project refs
    * Same experience type
    * Temporal proximity (within 30 days)
    """
    if reference.experience_id == candidate.experience_id:
        return 0.0

    ref_tags = set(reference.tag_names())
    cand_tags = set(candidate.tag_names())
    tag_union = ref_tags | cand_tags
    tag_intersection = ref_tags & cand_tags
    tag_sim = len(tag_intersection) / len(tag_union) if tag_union else 0.0

    ref_proj = set(reference.metadata.project_refs)
    cand_proj = set(candidate.metadata.project_refs)
    proj_union = ref_proj | cand_proj
    proj_intersection = ref_proj & cand_proj
    proj_sim = len(proj_intersection) / len(proj_union) if proj_union else 0.0

    type_sim = 1.0 if reference.experience_type == candidate.experience_type else 0.0

    delta_days = abs(
        (reference.occurred_at - candidate.occurred_at).total_seconds()
    ) / 86400.0
    temporal_sim = max(0.0, 1.0 - delta_days / 30.0)

    return (
        0.40 * tag_sim
        + 0.30 * proj_sim
        + 0.20 * type_sim
        + 0.10 * temporal_sim
    )


# ---------------------------------------------------------------------------
# Retrieval Result Types
# ---------------------------------------------------------------------------


class RetrievalResult:
    """Container returned by semantic and tag search operations.

    Attributes
    ----------
    experience:
        The matched :class:`~subsystems.echo.models.Experience`.
    score:
        Composite relevance score.  Higher is better.  Not normalised
        across different query types.
    match_reason:
        Human-readable explanation of why this experience matched.
    """

    __slots__ = ("experience", "score", "match_reason")

    def __init__(
        self,
        experience: Experience,
        score: float,
        match_reason: str = "",
    ) -> None:
        self.experience = experience
        self.score = score
        self.match_reason = match_reason

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(experience_id={self.experience.experience_id!r}, "
            f"score={self.score:.4f}, reason={self.match_reason!r})"
        )


# ---------------------------------------------------------------------------
# ExperienceRetrievalEngine
# ---------------------------------------------------------------------------


class ExperienceRetrievalEngine(ExperienceRetrievalEngineInterface):
    """Production implementation of the ECHO Experience Retrieval Engine.

    Parameters
    ----------
    experience_engine:
        A running :class:`~subsystems.echo.interfaces.ExperienceEngineInterface`
        whose store is queried by all retrieval operations.

    Usage
    -----
    ::

        retrieval = ExperienceRetrievalEngine(experience_engine=eng)
        retrieval.initialize()

        results = retrieval.search_semantic("architecture review", limit=10)
        for r in results:
            print(r.experience.title, r.score)

        retrieval.shutdown()
    """

    def __init__(self, experience_engine: ExperienceEngineInterface) -> None:
        self._experience_engine = experience_engine
        self._lock: threading.RLock = threading.RLock()
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the engine for use."""
        with self._lock:
            if self._running:
                _logger.warning(
                    "ExperienceRetrievalEngine.initialize() called while already running."
                )
                return
            self._running = True
            _logger.info("ExperienceRetrievalEngine initialised.")

    def shutdown(self) -> None:
        """Release resources and stop the engine."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            _logger.info("ExperienceRetrievalEngine shut down.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_running(self, operation: str) -> None:
        """Raise :class:`EchoNotInitializedError` if not running."""
        if not self._running:
            raise EchoNotInitializedError(operation)

    def _all_experiences(self) -> list[Experience]:
        """Return all experiences from the backing engine (snapshot)."""
        return self._experience_engine.query_experiences()

    def _get_experience(self, experience_id: str) -> Experience:
        """Fetch one experience by ID, raising if missing."""
        return self._experience_engine.get_experience(experience_id)

    def _record_retrieval(self, experiences: list[Experience]) -> None:
        """Increment retrieval counters for all returned experiences."""
        for exp in experiences:
            exp.metadata.record_retrieval()

    def _importance_at_least(
        self,
        experience: Experience,
        min_importance: ExperienceImportance,
    ) -> bool:
        """Return True if *experience* meets or exceeds *min_importance*."""
        return (
            _IMPORTANCE_ORDER[experience.importance]
            >= _IMPORTANCE_ORDER[min_importance]
        )

    # ------------------------------------------------------------------
    # Semantic Search
    # ------------------------------------------------------------------

    def search_semantic(
        self,
        query: str,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        experience_types: list[ExperienceType] | None = None,
        session_id: str | None = None,
        limit: int = _DEFAULT_SEMANTIC_LIMIT,
    ) -> list[RetrievalResult]:
        """Full-text relevance search across all experience fields.

        Parameters
        ----------
        query:
            Natural-language or keyword query string.
        min_importance:
            Exclude experiences below this tier.
        experience_types:
            If supplied, restrict results to these types.
        session_id:
            If supplied, restrict results to experiences whose
            ``metadata.session_id`` matches.
        limit:
            Maximum number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Matched results ordered by descending relevance score.
        """
        with self._lock:
            self._assert_running("search_semantic")

            query_terms = set(_tokenize(query))
            if not query_terms:
                return []

            candidates = self._all_experiences()
            results: list[RetrievalResult] = []

            for exp in candidates:
                if not self._importance_at_least(exp, min_importance):
                    continue
                if experience_types and exp.experience_type not in experience_types:
                    continue
                if session_id and exp.metadata.session_id != session_id:
                    continue

                score = _score_experience_for_query(exp, query_terms)
                if score > 0.0:
                    matched_tags = [
                        t for t in exp.tag_names()
                        if any(qt in t.lower() for qt in query_terms)
                    ]
                    reason_parts = []
                    if any(qt in _tokenize(exp.title) for qt in query_terms):
                        reason_parts.append("title match")
                    if any(qt in _tokenize(exp.description) for qt in query_terms):
                        reason_parts.append("description match")
                    if matched_tags:
                        reason_parts.append(f"tags: {matched_tags}")
                    reason = "; ".join(reason_parts) if reason_parts else "content match"

                    results.append(RetrievalResult(exp, score, reason))

            results.sort(key=lambda r: r.score, reverse=True)
            top = results[:limit]
            self._record_retrieval([r.experience for r in top])

            _logger.debug(
                "search_semantic(%r) → %d/%d results.",
                query,
                len(top),
                len(results),
            )
            return top

    # ------------------------------------------------------------------
    # Tag Search
    # ------------------------------------------------------------------

    def search_by_tags(
        self,
        tag_names: list[str],
        *,
        match_all: bool = False,
        tag_category: str | None = None,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        session_id: str | None = None,
        limit: int = _DEFAULT_TAG_LIMIT,
    ) -> list[RetrievalResult]:
        """Retrieve experiences that carry specified tags.

        Parameters
        ----------
        tag_names:
            List of tag name strings to search for (case-insensitive).
        match_all:
            If ``True``, only return experiences that have ALL supplied
            tags.  If ``False`` (default), return experiences with ANY.
        tag_category:
            If supplied, only count tags belonging to this category.
        min_importance:
            Exclude experiences below this tier.
        session_id:
            Restrict to a session scope if supplied.
        limit:
            Maximum number of results.

        Returns
        -------
        list[RetrievalResult]
            Matching experiences wrapped in :class:`RetrievalResult`, ordered
            by descending importance then recency.
        """
        with self._lock:
            self._assert_running("search_by_tags")

            if not tag_names:
                return []

            normalised = {n.lower().strip() for n in tag_names if n.strip()}
            if not normalised:
                return []

            matches: list[tuple[Experience, int]] = []

            for exp in self._all_experiences():
                if not self._importance_at_least(exp, min_importance):
                    continue
                if session_id and exp.metadata.session_id != session_id:
                    continue

                exp_tags: list[MemoryTag] = exp.tags
                if tag_category:
                    exp_tags = [t for t in exp_tags if t.category == tag_category]

                exp_tag_names = {t.name.lower() for t in exp_tags}
                matched = exp_tag_names & normalised

                if match_all:
                    if len(matched) == len(normalised):
                        matches.append((exp, len(matched)))
                else:
                    if matched:
                        matches.append((exp, len(matched)))

            matches.sort(
                key=lambda item: (
                    -_IMPORTANCE_ORDER[item[0].importance],
                    -item[1],
                    -item[0].occurred_at.timestamp(),
                )
            )

            top_exps = [exp for exp, _ in matches[:limit]]
            self._record_retrieval(top_exps)

            _logger.debug(
                "search_by_tags(%r, match_all=%s) → %d results.",
                tag_names,
                match_all,
                len(top_exps),
            )
            return [
                RetrievalResult(
                    exp,
                    float(_IMPORTANCE_ORDER[exp.importance]),
                    f"tag match: {sorted(set(t.name.lower() for t in exp.tags) & normalised)}",
                )
                for exp in top_exps
            ]

    # ------------------------------------------------------------------
    # Time-Range Search
    # ------------------------------------------------------------------

    def find_by_time_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        experience_types: list[ExperienceType] | None = None,
        session_id: str | None = None,
        limit: int = _DEFAULT_TIME_RANGE_LIMIT,
    ) -> list[RetrievalResult]:
        """Return experiences that occurred within a UTC time window.

        Parameters
        ----------
        start:
            Window start (inclusive).  Timezone-aware required.
            Alias: ``occurred_after``.
        end:
            Window end (inclusive).  Timezone-aware required.
            Alias: ``occurred_before``.
        occurred_after:
            Keyword alias for *start*.  Takes precedence over *start* when
            both are supplied.
        occurred_before:
            Keyword alias for *end*.  Takes precedence over *end* when both
            are supplied.
        min_importance:
            Exclude experiences below this tier.
        experience_types:
            Restrict to these types if supplied.
        session_id:
            Restrict to session scope if supplied.
        limit:
            Maximum number of results.

        Returns
        -------
        list[RetrievalResult]
            Experiences wrapped in :class:`RetrievalResult`, ordered
            chronologically by ``occurred_at``.

        Raises
        ------
        ValueError
            If the resolved start is after the resolved end, or either is naive.
        """
        # Keyword aliases take precedence
        resolved_start = occurred_after if occurred_after is not None else start
        resolved_end = occurred_before if occurred_before is not None else end

        if resolved_start is None or resolved_end is None:
            raise ValueError(
                "find_by_time_range requires both start/occurred_after and "
                "end/occurred_before."
            )

        with self._lock:
            self._assert_running("find_by_time_range")

            if resolved_start.tzinfo is None or resolved_end.tzinfo is None:
                raise ValueError(
                    "find_by_time_range requires timezone-aware datetimes."
                )
            if resolved_start > resolved_end:
                raise ValueError(
                    f"start ({resolved_start.isoformat()}) must be <= end "
                    f"({resolved_end.isoformat()})."
                )

            results: list[Experience] = []

            for exp in self._all_experiences():
                if not self._importance_at_least(exp, min_importance):
                    continue
                if experience_types and exp.experience_type not in experience_types:
                    continue
                if session_id and exp.metadata.session_id != session_id:
                    continue

                oc = exp.occurred_at
                if oc.tzinfo is None:
                    oc = oc.replace(tzinfo=timezone.utc)

                if resolved_start <= oc <= resolved_end:
                    results.append(exp)

            results.sort(key=lambda e: e.occurred_at)
            top = results[:limit]
            self._record_retrieval(top)

            _logger.debug(
                "find_by_time_range([%s, %s]) → %d results.",
                resolved_start.isoformat(),
                resolved_end.isoformat(),
                len(top),
            )
            return [RetrievalResult(exp, 1.0, "time range match") for exp in top]

    # ------------------------------------------------------------------
    # Similarity Search  (interface contract)
    # ------------------------------------------------------------------

    def find_similar(
        self,
        reference_experience_id: str,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        limit: int = _DEFAULT_SIMILAR_LIMIT,
    ) -> list[Experience]:
        """Return experiences similar to a reference experience.

        Similarity is computed from shared tags, project refs, experience
        type, and temporal proximity.

        Parameters
        ----------
        reference_experience_id:
            UUID of the reference experience.
        min_importance:
            Exclude candidates below this tier.
        limit:
            Maximum results.

        Returns
        -------
        list[Experience]
            Most similar experiences, highest similarity first.

        Raises
        ------
        ExperienceNotFoundError
            If the reference experience does not exist.
        """
        with self._lock:
            self._assert_running("find_similar")

            reference = self._get_experience(reference_experience_id)

            scored: list[tuple[Experience, float]] = []
            for exp in self._all_experiences():
                if exp.experience_id == reference_experience_id:
                    continue
                if not self._importance_at_least(exp, min_importance):
                    continue
                score = _similarity_score(reference, exp)
                if score > 0.0:
                    scored.append((exp, score))

            scored.sort(key=lambda item: item[1], reverse=True)
            top = [exp for exp, _ in scored[:limit]]
            self._record_retrieval(top)

            _logger.debug(
                "find_similar(%r) → %d results.",
                reference_experience_id,
                len(top),
            )
            return top

    # ------------------------------------------------------------------
    # Topic Search  (interface contract)
    # ------------------------------------------------------------------

    def find_by_topic(
        self,
        topic: str,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        limit: int = _DEFAULT_SEMANTIC_LIMIT,
    ) -> list[Experience]:
        """Return experiences related to a topic string.

        Delegates to :meth:`search_semantic` and strips the
        :class:`RetrievalResult` wrapper.

        Parameters
        ----------
        topic:
            Keyword or phrase to match against all experience fields.
        min_importance:
            Exclude experiences below this tier.
        limit:
            Maximum results.

        Returns
        -------
        list[Experience]
            Matching experiences ordered by relevance then recency.
        """
        # NOTE: retrieval count already updated inside search_semantic;
        # no double-increment needed here.
        results = self.search_semantic(
            topic,
            min_importance=min_importance,
            limit=limit,
        )
        return [r.experience for r in results]

    # ------------------------------------------------------------------
    # Session-Aware Retrieval
    # ------------------------------------------------------------------

    def get_session_experiences(
        self,
        session_id: str,
        *,
        min_importance: ExperienceImportance = ExperienceImportance.LOW,
        experience_types: list[ExperienceType] | None = None,
        limit: int = 100,
    ) -> list[Experience]:
        """Return all experiences belonging to a session, ordered by time.

        Parameters
        ----------
        session_id:
            UUID of the target session experience.
        min_importance:
            Exclude experiences below this tier.
        experience_types:
            Restrict to these types if supplied.
        limit:
            Maximum results.

        Returns
        -------
        list[Experience]
            Session members ordered by ``occurred_at`` ascending.
        """
        with self._lock:
            self._assert_running("get_session_experiences")

            results: list[Experience] = []
            for exp in self._all_experiences():
                if exp.metadata.session_id != session_id:
                    continue
                if not self._importance_at_least(exp, min_importance):
                    continue
                if experience_types and exp.experience_type not in experience_types:
                    continue
                results.append(exp)

            results.sort(key=lambda e: e.occurred_at)
            top = results[:limit]
            self._record_retrieval(top)

            _logger.debug(
                "get_session_experiences(%r) → %d results.", session_id, len(top)
            )
            return top

    # ------------------------------------------------------------------
    # Relevance Ranking
    # ------------------------------------------------------------------

    def rank_by_relevance(
        self,
        experiences: list[Experience],
        query: str,
    ) -> list[RetrievalResult]:
        """Re-rank an externally supplied list of experiences by query relevance.

        Useful when caller has pre-filtered a set and needs relevance ordering.

        Parameters
        ----------
        experiences:
            Candidate :class:`~subsystems.echo.models.Experience` objects.
        query:
            Query string to rank against.

        Returns
        -------
        list[RetrievalResult]
            Input experiences scored and sorted by descending relevance.
        """
        with self._lock:
            self._assert_running("rank_by_relevance")

            query_terms = set(_tokenize(query))
            if not query_terms:
                return [RetrievalResult(e, 0.0) for e in experiences]

            results = [
                RetrievalResult(e, _score_experience_for_query(e, query_terms))
                for e in experiences
            ]
            results.sort(key=lambda r: r.score, reverse=True)
            return results

    # ------------------------------------------------------------------
    # Context Recall  (interface contract)
    # ------------------------------------------------------------------

    def recall_context(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Reconstruct historical context for a natural-language query.

        Assembles a structured context bundle containing:
        * ``experiences`` — top matched :class:`Experience` objects
        * ``timeline``    — chronologically ordered experience titles + dates
        * ``narrative``   — concise textual summary for prompt injection
        * ``query``       — the original query string
        * ``retrieved_at`` — ISO-8601 UTC timestamp of this call

        Parameters
        ----------
        query:
            Natural-language question about past experiences.

        Returns
        -------
        dict[str, Any]
            Context bundle ready for ORION / ODYSSEY consumption.
        """
        with self._lock:
            self._assert_running("recall_context")

        # Fetch top matches (lock released; search_semantic re-acquires)
        semantic_results = self.search_semantic(
            query,
            min_importance=ExperienceImportance.LOW,
            limit=_DEFAULT_CONTEXT_BUNDLE_LIMIT,
        )
        experiences = [r.experience for r in semantic_results]

        # Build timeline
        sorted_exps = sorted(experiences, key=lambda e: e.occurred_at)
        timeline = [
            {
                "experience_id": e.experience_id,
                "title": e.title,
                "type": e.experience_type.name,
                "occurred_at": e.occurred_at.isoformat(),
                "importance": e.importance.name,
            }
            for e in sorted_exps
        ]

        # Build narrative
        if not experiences:
            narrative = f"No experiences found related to '{query}'."
        else:
            entries = [
                f"[{e.occurred_at.strftime('%Y-%m-%d')}] {e.title}"
                + (f": {e.outcome}" if e.outcome else "")
                for e in sorted_exps[:8]
            ]
            narrative = (
                f"Context for '{query}' ({len(experiences)} experience(s) found):\n"
                + "\n".join(entries)
            )

        return {
            "query": query,
            "experiences": experiences,
            "timeline": timeline,
            "narrative": narrative,
            "experience_count": len(experiences),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_recent_by_importance(
        self,
        min_importance: ExperienceImportance,
        *,
        limit: int = 20,
        experience_types: list[ExperienceType] | None = None,
    ) -> list[Experience]:
        """Return the most recent experiences at or above an importance floor.

        Parameters
        ----------
        min_importance:
            Minimum importance tier.
        limit:
            Maximum results.
        experience_types:
            Restrict to these types if supplied.

        Returns
        -------
        list[Experience]
            Ordered by ``occurred_at`` descending.
        """
        with self._lock:
            self._assert_running("get_recent_by_importance")

            results: list[Experience] = []
            for exp in self._all_experiences():
                if not self._importance_at_least(exp, min_importance):
                    continue
                if experience_types and exp.experience_type not in experience_types:
                    continue
                results.append(exp)

            results.sort(key=lambda e: e.occurred_at, reverse=True)
            top = results[:limit]
            self._record_retrieval(top)
            return top

    def snapshot(self) -> dict[str, Any]:
        """Return diagnostic state for monitoring / debugging."""
        with self._lock:
            return {
                "running": self._running,
                "engine": "ExperienceRetrievalEngine",
            }