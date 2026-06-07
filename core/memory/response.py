# core/memory/response.py
"""
Immutable :class:`MemoryResponse` primitive for the POLARIS v5 Memory Gateway.

Every operation dispatched through the gateway returns a
:class:`MemoryResponse`.  Responses are **immutable** once created; use the
:meth:`MemoryResponse.success_response` and
:meth:`MemoryResponse.failure_response` factories for convenient construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.memory.models import MemoryType


# ---------------------------------------------------------------------------
# MemoryResponse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryResponse:
    """Immutable result of a memory operation.

    Attributes
    ----------
    success:
        ``True`` if the operation completed without error; ``False`` otherwise.
    memory_type:
        The :class:`~core.memory.models.MemoryType` that served the request.
    data:
        Operation result payload.  Semantics are operation-dependent:

        * ``store``    — ``None`` on success.
        * ``retrieve`` — the stored value, or ``None`` if not found.
        * ``delete``   — ``True`` if a record was deleted, ``False`` otherwise.
        * ``exists``   — ``True`` if the key exists, ``False`` otherwise.
        * ``search``   — list of matching records / keys.
    message:
        Human-readable status message.  Contains error details on failure.
    timestamp:
        UTC :class:`~datetime.datetime` at which the response was produced;
        auto-set if not supplied.
    """

    success: bool
    memory_type: MemoryType
    data: Any = None
    message: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def success_response(
        cls,
        memory_type: MemoryType,
        data: Any = None,
        message: str = "OK",
    ) -> "MemoryResponse":
        """Build a successful :class:`MemoryResponse`.

        Parameters
        ----------
        memory_type:
            Memory system that produced this response.
        data:
            Result payload (operation-specific).
        message:
            Optional human-readable status message.

        Returns
        -------
        MemoryResponse
            Immutable success response.
        """
        return cls(success=True, memory_type=memory_type, data=data, message=message)

    @classmethod
    def failure_response(
        cls,
        memory_type: MemoryType,
        message: str,
        data: Any = None,
    ) -> "MemoryResponse":
        """Build a failure :class:`MemoryResponse`.

        Parameters
        ----------
        memory_type:
            Memory system that was targeted.
        message:
            Human-readable description of the failure.
        data:
            Optional supplementary data (e.g. partial results).

        Returns
        -------
        MemoryResponse
            Immutable failure response.
        """
        return cls(success=False, memory_type=memory_type, data=data, message=message)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise this response to a plain dictionary."""
        return {
            "success": self.success,
            "memory_type": self.memory_type.value,
            "data": self.data,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        status = "OK" if self.success else "FAIL"
        return (
            f"MemoryResponse("
            f"status={status}, "
            f"type={self.memory_type.value!r}, "
            f"message={self.message!r})"
        )