# core/loader/manifest.py
"""
Manifest and descriptor dataclasses for the POLARIS v5 Module Loader.

Two primary types are defined here:

* :class:`ModuleManifest` — immutable, validated description of a module read
  from a ``manifest.json`` (or equivalent) file.  This is the canonical
  contract a module author publishes.

* :class:`ModuleDescriptor` — mutable runtime envelope that wraps a manifest
  with loader-side bookkeeping: import path, class reference, instance
  reference, and current :class:`~core.loader.models.ModuleState`.

Design
------
Both classes are :func:`dataclasses.dataclass` instances.
:class:`ModuleManifest` is *frozen* (immutable); :class:`ModuleDescriptor`
is *not* frozen because the loader updates its fields as the module progresses
through lifecycle stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from core.loader.models import ModuleState


# ---------------------------------------------------------------------------
# Semver validation helper (local copy avoids circular import with types pkg)
# ---------------------------------------------------------------------------

_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[a-zA-Z0-9._-]+))?(?:\+(?P<build>[a-zA-Z0-9._-]+))?$"
)


def _validate_semver(version: str) -> None:
    """Raise :class:`ValueError` if *version* is not valid SemVer 2.0.0."""
    if not isinstance(version, str) or not _SEMVER_RE.match(version):
        raise ValueError(
            f"version must be a SemVer 2.0.0 string (e.g. '1.0.0'); "
            f"got {version!r}."
        )


# ---------------------------------------------------------------------------
# ModuleManifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleManifest:
    """Immutable, validated descriptor for a single POLARIS v5 module.

    A *manifest* captures all static metadata a module declares about itself.
    It is read once during discovery and never mutated thereafter.

    Attributes
    ----------
    id:
        Globally unique module identifier. Convention:
        ``"polaris.<domain>.<name>"``, e.g. ``"polaris.memory.echo"``.
    name:
        Short, human-readable display name (e.g. ``"Echo Memory"``).
    version:
        SemVer 2.0.0 string (e.g. ``"1.0.0"``).
    description:
        One-paragraph description of the module's responsibilities.
    dependencies:
        Tuple of module IDs that must be in the ``RUNNING`` state before
        this module may start.  Order is not significant; the dependency
        graph determines load order.
    capabilities:
        Tuple of capability names declared by this module
        (e.g. ``"vector_search"``).  These are simple strings; the loader
        does not resolve them against a capability registry.
    """

    id: str
    name: str
    version: str
    description: str
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    capabilities: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # --- id ---
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("ModuleManifest.id must be a non-empty string.")

        # --- name ---
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ModuleManifest.name must be a non-empty string.")

        # --- version ---
        _validate_semver(self.version)

        # --- description ---
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                "ModuleManifest.description must be a non-empty string."
            )

        # --- dependencies ---
        if not isinstance(self.dependencies, tuple):
            # Allow lists / iterables at construction time by coercing to tuple
            # via object.__setattr__ (frozen dataclass).
            object.__setattr__(self, "dependencies", tuple(self.dependencies))
        for dep in self.dependencies:
            if not isinstance(dep, str) or not dep.strip():
                raise ValueError(
                    f"Each dependency must be a non-empty string; got {dep!r}."
                )
        if self.id in self.dependencies:
            raise ValueError(
                f"Module {self.id!r} cannot declare itself as a dependency."
            )
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(
                f"Module {self.id!r} has duplicate entries in dependencies."
            )

        # --- capabilities ---
        if not isinstance(self.capabilities, tuple):
            object.__setattr__(self, "capabilities", tuple(self.capabilities))
        for cap in self.capabilities:
            if not isinstance(cap, str) or not cap.strip():
                raise ValueError(
                    f"Each capability must be a non-empty string; got {cap!r}."
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleManifest":
        """Construct a :class:`ModuleManifest` from a plain dictionary.

        Parameters
        ----------
        data:
            Dictionary with keys ``"id"``, ``"name"``, ``"version"``,
            ``"description"``, and optionally ``"dependencies"`` and
            ``"capabilities"``.

        Returns
        -------
        ModuleManifest
            Validated manifest instance.

        Raises
        ------
        ValueError
            If required fields are missing or any field fails validation.
        KeyError
            If a required key is absent from *data*.
        """
        try:
            return cls(
                id=data["id"],
                name=data["name"],
                version=data["version"],
                description=data["description"],
                dependencies=tuple(data.get("dependencies", ())),
                capabilities=tuple(data.get("capabilities", ())),
            )
        except KeyError as exc:
            raise ValueError(
                f"ModuleManifest.from_dict: missing required field {exc}."
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialise this manifest to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of this manifest.
        """
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "capabilities": list(self.capabilities),
        }


# ---------------------------------------------------------------------------
# ModuleDescriptor
# ---------------------------------------------------------------------------


@dataclass
class ModuleDescriptor:
    """Mutable runtime envelope wrapping a :class:`ModuleManifest`.

    The descriptor is created by :class:`~core.loader.discovery.ModuleDiscovery`
    and subsequently updated by the :class:`~core.loader.loader.ModuleLoader`
    as the module progresses through its lifecycle.

    Attributes
    ----------
    manifest:
        The immutable :class:`ModuleManifest` read from disk.
    module_path:
        Fully-qualified Python import path to the module class
        (e.g. ``"polaris.subsystems.echo.EchoSubsystem"``).
    state:
        Current :class:`~core.loader.models.ModuleState`.
    module_class:
        The imported Python class object, or ``None`` before the module has
        been loaded.
    instance:
        The live module instance, or ``None`` before the module has been
        initialized.
    error:
        The last exception recorded for this module, if any.
    """

    manifest: ModuleManifest
    module_path: str
    state: ModuleState = field(default=ModuleState.DISCOVERED)
    module_class: Optional[Type[Any]] = field(default=None, compare=False)
    instance: Optional[Any] = field(default=None, compare=False)
    error: Optional[BaseException] = field(default=None, compare=False)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Shortcut to :attr:`ModuleManifest.id`."""
        return self.manifest.id

    @property
    def name(self) -> str:
        """Shortcut to :attr:`ModuleManifest.name`."""
        return self.manifest.name

    @property
    def version(self) -> str:
        """Shortcut to :attr:`ModuleManifest.version`."""
        return self.manifest.version

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Shortcut to :attr:`ModuleManifest.dependencies`."""
        return self.manifest.dependencies

    @property
    def is_loaded(self) -> bool:
        """``True`` if the module class has been imported."""
        return self.module_class is not None

    @property
    def is_initialized(self) -> bool:
        """``True`` if the module instance exists."""
        return self.instance is not None

    @property
    def is_running(self) -> bool:
        """``True`` if the module is in the ``RUNNING`` state."""
        return self.state is ModuleState.RUNNING

    @property
    def has_failed(self) -> bool:
        """``True`` if the module is in the ``FAILED`` state."""
        return self.state is ModuleState.FAILED

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ModuleDescriptor("
            f"id={self.manifest.id!r}, "
            f"state={self.state.name}, "
            f"version={self.manifest.version!r})"
        )


__all__ = [
    "ModuleManifest",
    "ModuleDescriptor",
]