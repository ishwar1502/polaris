# core/memory/registry.py
"""
Thread-safe :class:`MemoryRegistry` for the POLARIS v5 Memory Gateway.

The registry is the authoritative source of truth for which
:class:`~core.memory.provider.MemoryProvider` serves each
:class:`~core.memory.models.MemoryType`.  The
:class:`~core.memory.router.MemoryRouter` queries the registry on every
routing decision; subsystems never interact with the registry directly.

Thread Safety
-------------
All public methods acquire the internal :class:`threading.RLock` before
mutating or inspecting shared state, making the registry safe for concurrent
use from multiple threads.
"""

from __future__ import annotations

import threading
from typing import Iterator

from core.memory.exceptions import MemoryProviderError, MemoryRoutingError
from core.memory.models import MemoryType
from core.memory.provider import MemoryProvider


# ---------------------------------------------------------------------------
# MemoryRegistry
# ---------------------------------------------------------------------------


class MemoryRegistry:
    """Thread-safe registry that maps :class:`MemoryType` to
    :class:`MemoryProvider` instances.

    Usage
    -----
    .. code-block:: python

        registry = MemoryRegistry()
        registry.register(EchoProvider())
        provider = registry.get_provider(MemoryType.EXPERIENCE)

    At most **one** provider may be registered per :class:`MemoryType` at any
    time.  Attempting to register a second provider for the same type raises
    :class:`~core.memory.exceptions.MemoryProviderError` unless the existing
    provider is first removed via :meth:`unregister`.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        # memory_type.value → MemoryProvider
        self._providers: dict[str, MemoryProvider] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, provider: MemoryProvider) -> None:
        """Register a :class:`~core.memory.provider.MemoryProvider`.

        Parameters
        ----------
        provider:
            Fully initialised provider instance to register.

        Raises
        ------
        MemoryProviderError
            If a provider is already registered for the same
            :class:`MemoryType`.
        TypeError
            If *provider* is not a :class:`MemoryProvider` instance.
        """
        if not isinstance(provider, MemoryProvider):
            raise TypeError(
                f"Expected a MemoryProvider instance, got {type(provider).__name__!r}."
            )
        key = provider.memory_type.value
        with self._lock:
            if key in self._providers:
                existing = self._providers[key]
                raise MemoryProviderError(
                    f"A provider for MemoryType.{provider.memory_type.name} is already "
                    f"registered (provider_id={existing.provider_id!r}). "
                    f"Call unregister() first.",
                    provider_id=existing.provider_id,
                    memory_type=provider.memory_type.value,
                    operation="register",
                )
            self._providers[key] = provider

    def unregister(self, memory_type: MemoryType) -> None:
        """Remove the provider registered for *memory_type*.

        Parameters
        ----------
        memory_type:
            The :class:`MemoryType` whose provider should be removed.

        Raises
        ------
        MemoryRoutingError
            If no provider is registered for *memory_type*.
        """
        key = memory_type.value
        with self._lock:
            if key not in self._providers:
                raise MemoryRoutingError(
                    f"No provider registered for MemoryType.{memory_type.name}.",
                    memory_type=memory_type.value,
                    operation="unregister",
                )
            del self._providers[key]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_provider(self, memory_type: MemoryType) -> MemoryProvider:
        """Return the provider registered for *memory_type*.

        Parameters
        ----------
        memory_type:
            The :class:`MemoryType` to look up.

        Returns
        -------
        MemoryProvider
            The registered provider.

        Raises
        ------
        MemoryRoutingError
            If no provider is registered for *memory_type*.
        """
        key = memory_type.value
        with self._lock:
            provider = self._providers.get(key)
        if provider is None:
            raise MemoryRoutingError(
                f"No provider registered for MemoryType.{memory_type.name}. "
                "Register a provider before issuing requests.",
                memory_type=memory_type.value,
                operation="get_provider",
            )
        return provider

    def is_registered(self, memory_type: MemoryType) -> bool:
        """Return ``True`` if a provider is registered for *memory_type*.

        Parameters
        ----------
        memory_type:
            The :class:`MemoryType` to query.
        """
        with self._lock:
            return memory_type.value in self._providers

    def is_available(self, memory_type: MemoryType) -> bool:
        """Return ``True`` if a provider is registered *and* available.

        A provider is considered available when both it is registered and
        its :meth:`~core.memory.provider.MemoryProvider.is_available` method
        returns ``True``.

        Parameters
        ----------
        memory_type:
            The :class:`MemoryType` to check.
        """
        with self._lock:
            provider = self._providers.get(memory_type.value)
        if provider is None:
            return False
        return provider.is_available()

    # ------------------------------------------------------------------
    # Iteration / introspection
    # ------------------------------------------------------------------

    def registered_types(self) -> list[MemoryType]:
        """Return a snapshot list of all currently registered
        :class:`MemoryType` values."""
        with self._lock:
            return [MemoryType(k) for k in self._providers]

    def all_providers(self) -> list[MemoryProvider]:
        """Return a snapshot list of all registered provider instances."""
        with self._lock:
            return list(self._providers.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._providers)

    def __iter__(self) -> Iterator[MemoryProvider]:
        with self._lock:
            snapshot = list(self._providers.values())
        return iter(snapshot)

    def __repr__(self) -> str:  # pragma: no cover
        with self._lock:
            types = [k for k in self._providers]
        return f"MemoryRegistry(registered={types!r})"