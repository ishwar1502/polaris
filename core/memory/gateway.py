# core/memory/gateway.py
"""
:class:`MemoryGateway` — the sole public API for all memory operations in
POLARIS v5.

Architecture contract
---------------------
No cognitive subsystem may call a
:class:`~core.memory.provider.MemoryProvider` directly.  All memory access
must flow through::

    MemoryGateway
        └── MemoryRouter
                └── MemoryProvider (ECHO / LUNA / CHRONOS / CONSTELLATION)

The gateway hides routing, provider selection, request construction, and
error normalisation.  Callers receive a clean
:class:`~core.memory.response.MemoryResponse` for every operation.

Thread Safety
-------------
The gateway delegates all mutable state to the
:class:`~core.memory.registry.MemoryRegistry` (which is internally
thread-safe) and the :class:`~core.memory.router.MemoryRouter` (which is
stateless beyond its registry reference).  The gateway itself acquires no
additional locks; it is safe for concurrent use.
"""

from __future__ import annotations

import threading
from typing import Any

from core.memory.exceptions import MemoryError, MemoryValidationError
from core.memory.models import MemoryType
from core.memory.registry import MemoryRegistry
from core.memory.request import MemoryRequest
from core.memory.response import MemoryResponse
from core.memory.router import MemoryRouter


# ---------------------------------------------------------------------------
# MemoryGateway
# ---------------------------------------------------------------------------


class MemoryGateway:
    """Public façade for all memory operations.

    Instantiate once and share across subsystems.  The gateway accepts a
    :class:`~core.memory.registry.MemoryRegistry` at construction time; the
    same registry should be pre-populated with providers before any operations
    are invoked.

    Parameters
    ----------
    registry:
        Pre-populated (or initially empty) provider registry.

    Examples
    --------
    .. code-block:: python

        registry = MemoryRegistry()
        registry.register(EchoProvider())
        registry.register(LunaProvider())

        gateway = MemoryGateway(registry)

        resp = gateway.store(
            memory_type=MemoryType.EXPERIENCE,
            key="event:42",
            value={"summary": "first walk"},
        )
        assert resp.success
    """

    def __init__(self, registry: MemoryRegistry) -> None:
        if not isinstance(registry, MemoryRegistry):
            raise TypeError(
                f"Expected a MemoryRegistry instance, got {type(registry).__name__!r}."
            )
        self._registry: MemoryRegistry = registry
        self._router: MemoryRouter = MemoryRouter(registry)

    # ------------------------------------------------------------------
    # Core CRUD operations
    # ------------------------------------------------------------------

    def store(
        self,
        *,
        memory_type: MemoryType,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryResponse:
        """Persist *value* under *key* in the memory system for *memory_type*.

        Parameters
        ----------
        memory_type:
            Target :class:`~core.memory.models.MemoryType`.
        key:
            Record identifier.
        value:
            Data to persist.
        metadata:
            Optional key-value annotations forwarded to the provider.

        Returns
        -------
        MemoryResponse
            ``success=True`` on successful persistence.
        """
        request = MemoryRequest.create(
            memory_type=memory_type,
            operation="store",
            key=key,
            value=value,
            metadata=metadata,
        )
        return self._router.route(request)

    def retrieve(
        self,
        *,
        memory_type: MemoryType,
        key: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryResponse:
        """Retrieve the value stored under *key*.

        Parameters
        ----------
        memory_type:
            Target :class:`~core.memory.models.MemoryType`.
        key:
            Record identifier to look up.
        metadata:
            Optional key-value annotations forwarded to the provider.

        Returns
        -------
        MemoryResponse
            ``data`` contains the stored value, or ``None`` if absent.
        """
        request = MemoryRequest.create(
            memory_type=memory_type,
            operation="retrieve",
            key=key,
            metadata=metadata,
        )
        return self._router.route(request)

    def delete(
        self,
        *,
        memory_type: MemoryType,
        key: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryResponse:
        """Delete the record identified by *key*.

        Parameters
        ----------
        memory_type:
            Target :class:`~core.memory.models.MemoryType`.
        key:
            Record identifier to delete.
        metadata:
            Optional key-value annotations forwarded to the provider.

        Returns
        -------
        MemoryResponse
            ``data=True`` if a record was removed; ``data=False`` if the key
            did not exist.
        """
        request = MemoryRequest.create(
            memory_type=memory_type,
            operation="delete",
            key=key,
            metadata=metadata,
        )
        return self._router.route(request)

    def exists(
        self,
        *,
        memory_type: MemoryType,
        key: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryResponse:
        """Check whether *key* exists in the memory system for *memory_type*.

        Parameters
        ----------
        memory_type:
            Target :class:`~core.memory.models.MemoryType`.
        key:
            Record identifier to probe.
        metadata:
            Optional key-value annotations forwarded to the provider.

        Returns
        -------
        MemoryResponse
            ``data=True`` if the key exists; ``data=False`` otherwise.
        """
        request = MemoryRequest.create(
            memory_type=memory_type,
            operation="exists",
            key=key,
            metadata=metadata,
        )
        return self._router.route(request)

    def search(
        self,
        *,
        memory_type: MemoryType,
        query: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryResponse:
        """Search the memory system for *memory_type* using *query*.

        Parameters
        ----------
        memory_type:
            Target :class:`~core.memory.models.MemoryType`.
        query:
            Search query; the provider defines its own query semantics
            (string, dict, vector, etc.).
        metadata:
            Optional key-value annotations forwarded to the provider.

        Returns
        -------
        MemoryResponse
            ``data`` contains a list of matching records or keys.
        """
        request = MemoryRequest.create(
            memory_type=memory_type,
            operation="search",
            key=None,
            value=query,
            metadata=metadata,
        )
        return self._router.route(request)

    def query(
        self,
        *,
        memory_type: MemoryType,
        key: str | None = None,
        query: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryResponse:
        """Unified query method that dispatches to :meth:`retrieve` or
        :meth:`search` depending on whether *key* or *query* is provided.

        This is a convenience method for callers that receive either a direct
        key lookup or an open-ended search at runtime.

        Parameters
        ----------
        memory_type:
            Target :class:`~core.memory.models.MemoryType`.
        key:
            If provided, performs a :meth:`retrieve` operation.
        query:
            If provided (and *key* is ``None``), performs a :meth:`search`
            operation.
        metadata:
            Optional key-value annotations forwarded to the provider.

        Returns
        -------
        MemoryResponse
            Result from the delegated :meth:`retrieve` or :meth:`search`.

        Raises
        ------
        MemoryValidationError
            If neither *key* nor *query* is supplied.
        """
        if key is not None:
            return self.retrieve(memory_type=memory_type, key=key, metadata=metadata)
        if query is not None:
            return self.search(memory_type=memory_type, query=query, metadata=metadata)
        raise MemoryValidationError(
            "MemoryGateway.query() requires either 'key' or 'query' to be provided.",
            field="key/query",
            invalid_value=None,
            memory_type=memory_type.value,
            operation="query",
        )

    # ------------------------------------------------------------------
    # Registry access (read-only surface)
    # ------------------------------------------------------------------

    @property
    def registry(self) -> MemoryRegistry:
        """Read-only reference to the underlying provider registry.

        Intended for introspection and testing; avoid mutating the registry
        after the gateway has begun serving requests in a production context.
        """
        return self._registry

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MemoryGateway("
            f"providers={self._registry.registered_types()!r})"
        )