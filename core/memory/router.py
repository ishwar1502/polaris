# core/memory/router.py
"""
:class:`MemoryRouter` for the POLARIS v5 Memory Gateway.

The router is the single decision-making component that selects the correct
:class:`~core.memory.provider.MemoryProvider` for an incoming
:class:`~core.memory.request.MemoryRequest` and dispatches the call.

Routing rules
-------------
====================  =====================
``MemoryType``        Provider
====================  =====================
``EXPERIENCE``        ECHO
``KNOWLEDGE``         LUNA
``TEMPORAL``          CHRONOS
``RELATIONSHIP``      CONSTELLATION
====================  =====================

The router does **not** own providers; it queries the
:class:`~core.memory.registry.MemoryRegistry` on each dispatch.  This allows
providers to be hot-swapped (e.g. during testing) without restarting the
gateway.
"""

from __future__ import annotations

from core.memory.exceptions import MemoryProviderError, MemoryRoutingError
from core.memory.models import MemoryType
from core.memory.registry import MemoryRegistry
from core.memory.request import MemoryRequest
from core.memory.response import MemoryResponse


# ---------------------------------------------------------------------------
# MemoryRouter
# ---------------------------------------------------------------------------


class MemoryRouter:
    """Routes :class:`~core.memory.request.MemoryRequest` objects to the
    appropriate :class:`~core.memory.provider.MemoryProvider` via the
    :class:`~core.memory.registry.MemoryRegistry`.

    Parameters
    ----------
    registry:
        The :class:`~core.memory.registry.MemoryRegistry` used to resolve
        providers.  The router holds a reference but does not own the
        registry; the same registry instance should be shared with the
        :class:`~core.memory.gateway.MemoryGateway`.
    """

    def __init__(self, registry: MemoryRegistry) -> None:
        if not isinstance(registry, MemoryRegistry):
            raise TypeError(
                f"Expected a MemoryRegistry instance, got {type(registry).__name__!r}."
            )
        self._registry = registry

    # ------------------------------------------------------------------
    # Public routing API
    # ------------------------------------------------------------------

    def route(self, request: MemoryRequest) -> MemoryResponse:
        """Route *request* to the correct provider and return its response.

        Parameters
        ----------
        request:
            A fully validated :class:`~core.memory.request.MemoryRequest`.

        Returns
        -------
        MemoryResponse
            The response produced by the resolved provider.

        Raises
        ------
        MemoryRoutingError
            If no provider is registered for the requested
            :class:`~core.memory.models.MemoryType`, or the registered
            provider is currently unavailable.
        MemoryProviderError
            If the provider raises an unexpected exception during dispatch.
        """
        self._validate_routing(request)
        provider = self._registry.get_provider(request.memory_type)
        if not provider.is_available():
            raise MemoryRoutingError(
                f"Provider {provider.provider_id!r} for "
                f"MemoryType.{request.memory_type.name} is not available.",
                memory_type=request.memory_type.value,
                operation=request.operation,
            )
        return self._dispatch(provider, request)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_routing(self, request: MemoryRequest) -> None:
        """Ensure the registry has a registration for the requested type."""
        if not self._registry.is_registered(request.memory_type):
            raise MemoryRoutingError(
                f"No provider registered for MemoryType.{request.memory_type.name}. "
                "Cannot route request.",
                memory_type=request.memory_type.value,
                operation=request.operation,
            )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        provider,  # MemoryProvider — forward ref avoided for circular imports
        request: MemoryRequest,
    ) -> MemoryResponse:
        """Dispatch *request* to the appropriate method on *provider*.

        Parameters
        ----------
        provider:
            Resolved :class:`~core.memory.provider.MemoryProvider`.
        request:
            Request to dispatch.

        Raises
        ------
        MemoryProviderError
            Wraps any unexpected exception raised by the provider.
        """
        op = request.operation
        try:
            if op == "store":
                return provider.store(request)
            elif op == "retrieve":
                return provider.retrieve(request)
            elif op == "delete":
                return provider.delete(request)
            elif op == "exists":
                return provider.exists(request)
            elif op == "search":
                return provider.search(request)
            else:
                # Guarded by MemoryRequest validation; should never reach here.
                raise MemoryRoutingError(
                    f"Unknown operation {op!r}.",
                    memory_type=request.memory_type.value,
                    operation=op,
                )
        except (MemoryRoutingError, MemoryProviderError):
            raise
        except Exception as exc:
            raise MemoryProviderError(
                f"Provider {provider.provider_id!r} raised an unexpected error "
                f"during {op!r}: {exc}",
                provider_id=provider.provider_id,
                memory_type=request.memory_type.value,
                operation=op,
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MemoryRouter(registry={self._registry!r})"
        )