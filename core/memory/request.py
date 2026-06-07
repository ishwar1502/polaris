# core/memory/request.py
"""
Immutable :class:`MemoryRequest` primitive for the POLARIS v5 Memory Gateway.

Every operation submitted to the gateway is wrapped in a
:class:`MemoryRequest`.  Requests are **immutable** once created; use the
:meth:`MemoryRequest.create` factory to obtain a validated instance.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final

from core.memory.exceptions import MemoryValidationError
from core.memory.models import (
    MemoryType,
    _KEY_MAX_LEN,
    _METADATA_MAX_KEYS,
)


# ---------------------------------------------------------------------------
# Valid operation names
# ---------------------------------------------------------------------------

VALID_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"store", "retrieve", "delete", "exists", "search"}
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_request_id(value: str) -> None:
    if not value or not value.strip():
        raise MemoryValidationError(
            "request_id must be a non-empty string.",
            field="request_id",
            invalid_value=repr(value),
            operation="validation",
        )


def _validate_operation(value: str) -> None:
    if value not in VALID_OPERATIONS:
        raise MemoryValidationError(
            f"operation {value!r} is not valid. "
            f"Must be one of: {sorted(VALID_OPERATIONS)}.",
            field="operation",
            invalid_value=value,
            operation="validation",
        )


def _validate_key(value: str | None, operation: str) -> None:
    """Key is required for store/retrieve/delete/exists; optional for search."""
    if operation in {"store", "retrieve", "delete", "exists"}:
        if not value or not value.strip():
            raise MemoryValidationError(
                f"key must be a non-empty string for operation {operation!r}.",
                field="key",
                invalid_value=repr(value),
                operation=operation,
            )
    if value is not None and len(value) > _KEY_MAX_LEN:
        raise MemoryValidationError(
            f"key exceeds maximum length of {_KEY_MAX_LEN} characters.",
            field="key",
            invalid_value=value[:64] + "...",
            operation=operation,
        )


def _validate_metadata(value: dict[str, Any]) -> None:
    if len(value) > _METADATA_MAX_KEYS:
        raise MemoryValidationError(
            f"metadata exceeds maximum of {_METADATA_MAX_KEYS} keys.",
            field="metadata",
            invalid_value=str(len(value)),
            operation="validation",
        )


# ---------------------------------------------------------------------------
# MemoryRequest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryRequest:
    """Immutable descriptor for a single memory operation.

    Attributes
    ----------
    memory_type:
        Target :class:`~core.memory.models.MemoryType` for this request.
    operation:
        One of ``"store"``, ``"retrieve"``, ``"delete"``, ``"exists"``,
        ``"search"``.
    request_id:
        UUID-4 string uniquely identifying this request; auto-generated if
        not supplied.
    key:
        Record key for keyed operations (``store``, ``retrieve``, ``delete``,
        ``exists``).  May be ``None`` for ``search`` requests.
    value:
        Data to persist; used only by ``store`` operations.  The gateway
        imposes no type constraint — serialisability is the provider's concern.
    metadata:
        Optional key-value annotations (e.g. TTL hints, source tags).
        Stored as a frozen tuple-of-pairs to preserve immutability; access
        via :meth:`get_metadata`.
    timestamp:
        UTC :class:`~datetime.datetime` of creation; auto-set if not supplied.
    """

    memory_type: MemoryType
    operation: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key: str | None = None
    value: Any = None
    # Stored as tuple-of-pairs to remain hashable inside a frozen dataclass.
    _metadata_raw: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        _validate_operation(self.operation)
        _validate_key(self.key, self.operation)
        _validate_metadata(dict(self._metadata_raw))
        if self.timestamp.tzinfo is None:
            raise MemoryValidationError(
                "MemoryRequest.timestamp must be timezone-aware.",
                field="timestamp",
                invalid_value=str(self.timestamp),
                operation=self.operation,
            )

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the request metadata as a plain dictionary (copy)."""
        return dict(self._metadata_raw)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve a single metadata value by *key*."""
        return dict(self._metadata_raw).get(key, default)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        memory_type: MemoryType,
        operation: str,
        key: str | None = None,
        value: Any = None,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> "MemoryRequest":
        """Validated factory for :class:`MemoryRequest`.

        Parameters
        ----------
        memory_type:
            Target memory system.
        operation:
            Operation name (``"store"``, ``"retrieve"``, ``"delete"``,
            ``"exists"``, ``"search"``).
        key:
            Record key; required for all operations except ``search``.
        value:
            Data payload; used by ``store`` only.
        metadata:
            Optional key-value annotations.
        request_id:
            Optional explicit request id; auto-generated if ``None``.

        Returns
        -------
        MemoryRequest
            A fully validated, immutable request instance.
        """
        raw_meta: tuple[tuple[str, Any], ...] = tuple(
            (k, v) for k, v in (metadata or {}).items()
        )
        return cls(
            memory_type=memory_type,
            operation=operation,
            request_id=request_id or str(uuid.uuid4()),
            key=key,
            value=value,
            _metadata_raw=raw_meta,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise this request to a plain dictionary."""
        return {
            "request_id": self.request_id,
            "memory_type": self.memory_type.value,
            "operation": self.operation,
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MemoryRequest("
            f"id={self.request_id[:8]}…, "
            f"type={self.memory_type.value!r}, "
            f"op={self.operation!r}, "
            f"key={self.key!r})"
        )