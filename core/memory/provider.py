# core/memory/provider.py
"""
Abstract :class:`MemoryProvider` base class for the POLARIS v5 Memory Gateway.

Concrete providers implement this interface and are registered with the
:class:`~core.memory.registry.MemoryRegistry`.  The
:class:`~core.memory.router.MemoryRouter` selects and invokes providers
transparently; subsystems never interact with providers directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.memory.models import MemoryType
from core.memory.request import MemoryRequest
from core.memory.response import MemoryResponse


# ---------------------------------------------------------------------------
# MemoryProvider
# ---------------------------------------------------------------------------


class MemoryProvider(ABC):
    """Abstract base class for all POLARIS memory backend providers.

    Subclasses must implement all five abstract methods.  Each method receives
    a fully validated :class:`~core.memory.request.MemoryRequest` and must
    return an appropriate :class:`~core.memory.response.MemoryResponse`.

    Provider identity
    -----------------
    Every provider has a stable string :attr:`provider_id` (e.g. ``"echo"``)
    and declares which :attr:`memory_type` it serves.  The registry uses both
    to prevent duplicate or conflicting registrations.

    Availability
    ------------
    Providers may become temporarily unavailable (backend down, circuit
    breaker open, etc.).  Callers should check :meth:`is_available` before
    dispatching if pre-flight validation is required; the router does this
    automatically.
    """

    def __init__(self, provider_id: str, memory_type: MemoryType) -> None:
        """Initialise the provider.

        Parameters
        ----------
        provider_id:
            Stable, unique string identifier for this provider
            (e.g. ``"echo"``, ``"luna"``).
        memory_type:
            The :class:`~core.memory.models.MemoryType` this provider serves.
        """
        if not provider_id or not provider_id.strip():
            raise ValueError("MemoryProvider.provider_id must be a non-empty string.")
        self._provider_id: str = provider_id
        self._memory_type: MemoryType = memory_type

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        """Stable string identifier for this provider instance."""
        return self._provider_id

    @property
    def memory_type(self) -> MemoryType:
        """The :class:`~core.memory.models.MemoryType` served by this provider."""
        return self._memory_type

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` if this provider is ready to accept requests.

        The default implementation always returns ``True``.  Override to
        implement health checks, circuit-breaker logic, etc.
        """
        return True

    # ------------------------------------------------------------------
    # Abstract operations
    # ------------------------------------------------------------------

    @abstractmethod
    def store(self, request: MemoryRequest) -> MemoryResponse:
        """Persist *value* under *key* in this provider's backing store.

        Parameters
        ----------
        request:
            A :class:`~core.memory.request.MemoryRequest` with
            ``operation == "store"``.

        Returns
        -------
        MemoryResponse
            ``success=True`` on success; ``success=False`` with an error
            message on failure.
        """

    @abstractmethod
    def retrieve(self, request: MemoryRequest) -> MemoryResponse:
        """Retrieve the value stored under *key*.

        Parameters
        ----------
        request:
            A :class:`~core.memory.request.MemoryRequest` with
            ``operation == "retrieve"``.

        Returns
        -------
        MemoryResponse
            ``data`` contains the stored value, or ``None`` if the key does
            not exist.
        """

    @abstractmethod
    def delete(self, request: MemoryRequest) -> MemoryResponse:
        """Delete the record identified by *key*.

        Parameters
        ----------
        request:
            A :class:`~core.memory.request.MemoryRequest` with
            ``operation == "delete"``.

        Returns
        -------
        MemoryResponse
            ``data=True`` if a record was deleted; ``data=False`` if the key
            did not exist.
        """

    @abstractmethod
    def exists(self, request: MemoryRequest) -> MemoryResponse:
        """Check whether *key* exists in the backing store.

        Parameters
        ----------
        request:
            A :class:`~core.memory.request.MemoryRequest` with
            ``operation == "exists"``.

        Returns
        -------
        MemoryResponse
            ``data=True`` if the key exists; ``data=False`` otherwise.
        """

    @abstractmethod
    def search(self, request: MemoryRequest) -> MemoryResponse:
        """Search the backing store using the query in *request.value*.

        Parameters
        ----------
        request:
            A :class:`~core.memory.request.MemoryRequest` with
            ``operation == "search"``.  The query is conveyed in
            ``request.value``.

        Returns
        -------
        MemoryResponse
            ``data`` contains a list of matching records or keys.
        """

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"id={self.provider_id!r}, "
            f"type={self.memory_type.value!r}, "
            f"available={self.is_available()})"
        )