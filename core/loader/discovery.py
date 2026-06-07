# core/loader/discovery.py
"""
Module discovery for the POLARIS v5 Module Loader.

:class:`ModuleDiscovery` scans a configurable set of discovery paths,
locates ``manifest.json`` files (or in-process registrations), validates and
parses them into :class:`~core.loader.manifest.ModuleManifest` objects, and
wraps each in a :class:`~core.loader.manifest.ModuleDescriptor`.

Two discovery modes are supported:

1. **Filesystem** — scan directories for ``manifest.json`` files.
2. **In-process** — call :meth:`register_manifest` directly with a fully
   constructed :class:`~core.loader.manifest.ModuleManifest`.  This is the
   primary mode used in tests and for programmatic module registration.

Thread safety
-------------
All public methods are protected by a :class:`threading.Lock`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Iterator

from core.loader.exceptions import ModuleDiscoveryError, ModuleValidationError
from core.loader.manifest import ModuleDescriptor, ModuleManifest
from core.loader.models import ModuleState

_logger = logging.getLogger(__name__)

# Canonical filename for filesystem manifests.
MANIFEST_FILENAME: str = "manifest.json"


class ModuleDiscovery:
    """Discovers and registers module manifests.

    Parameters
    ----------
    search_paths:
        Optional list of directory paths to scan for ``manifest.json`` files.
        Additional paths can be added later via :meth:`add_search_path`.
    """

    def __init__(
        self,
        search_paths: list[str | Path] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._descriptors: dict[str, ModuleDescriptor] = {}
        self._search_paths: list[Path] = []

        if search_paths:
            for path in search_paths:
                self.add_search_path(path)

    # ------------------------------------------------------------------
    # Search path management
    # ------------------------------------------------------------------

    def add_search_path(self, path: str | Path) -> None:
        """Add a directory to the discovery search path.

        Parameters
        ----------
        path:
            Filesystem directory to scan for ``manifest.json`` files.

        Raises
        ------
        ModuleDiscoveryError
            If *path* does not exist or is not a directory.
        """
        p = Path(path)
        if not p.exists():
            raise ModuleDiscoveryError(
                f"Search path does not exist: {p}",
                module_path=str(p),
            )
        if not p.is_dir():
            raise ModuleDiscoveryError(
                f"Search path is not a directory: {p}",
                module_path=str(p),
            )
        with self._lock:
            if p not in self._search_paths:
                self._search_paths.append(p)
                _logger.debug("Added discovery search path: %s", p)

    # ------------------------------------------------------------------
    # In-process registration
    # ------------------------------------------------------------------

    def register_manifest(
        self,
        manifest: ModuleManifest,
        module_path: str,
    ) -> ModuleDescriptor:
        """Register a manifest directly (in-process mode).

        This is the primary API for registering modules that are already
        available in the Python runtime without requiring a filesystem scan.

        Parameters
        ----------
        manifest:
            Fully constructed and validated :class:`ModuleManifest`.
        module_path:
            Fully-qualified Python import path to the module class,
            e.g. ``"polaris.subsystems.echo.EchoSubsystem"``.

        Returns
        -------
        ModuleDescriptor
            The newly created descriptor in :attr:`~ModuleState.DISCOVERED`
            state.

        Raises
        ------
        ModuleDiscoveryError
            If a module with the same ``id`` is already registered.
        """
        with self._lock:
            if manifest.id in self._descriptors:
                raise ModuleDiscoveryError(
                    f"Module {manifest.id!r} is already registered. "
                    "Unregister it before re-registering.",
                    module_id=manifest.id,
                )
            descriptor = ModuleDescriptor(
                manifest=manifest,
                module_path=module_path,
                state=ModuleState.DISCOVERED,
            )
            self._descriptors[manifest.id] = descriptor
            _logger.info(
                "Discovered module %r v%s via in-process registration.",
                manifest.id,
                manifest.version,
            )
            return descriptor

    def unregister(self, module_id: str) -> None:
        """Remove a previously registered module from the discovery registry.

        Parameters
        ----------
        module_id:
            ID of the module to remove.

        Raises
        ------
        ModuleDiscoveryError
            If no module with *module_id* is registered.
        """
        with self._lock:
            if module_id not in self._descriptors:
                raise ModuleDiscoveryError(
                    f"Cannot unregister module {module_id!r}: not found.",
                    module_id=module_id,
                )
            del self._descriptors[module_id]
            _logger.info("Unregistered module %r.", module_id)

    # ------------------------------------------------------------------
    # Filesystem discovery
    # ------------------------------------------------------------------

    def scan(self) -> list[ModuleDescriptor]:
        """Scan all registered search paths and discover manifests.

        For each ``manifest.json`` found, a :class:`ModuleDescriptor` is
        created and stored.  Modules already registered (via a previous
        :meth:`scan` or :meth:`register_manifest` call) are skipped.

        Returns
        -------
        list[ModuleDescriptor]
            All descriptors discovered during this scan (including previously
            registered ones that were not re-discovered).

        Notes
        -----
        Scan errors for individual directories are logged as warnings and
        do not abort the overall scan.
        """
        with self._lock:
            paths = list(self._search_paths)

        newly_found: list[ModuleDescriptor] = []
        for search_path in paths:
            found = self._scan_directory(search_path)
            newly_found.extend(found)

        _logger.info(
            "Discovery scan complete; %d new modules found across %d path(s).",
            len(newly_found),
            len(paths),
        )
        return self.all_descriptors()

    def _scan_directory(self, directory: Path) -> list[ModuleDescriptor]:
        """Recursively scan *directory* for manifest files."""
        discovered: list[ModuleDescriptor] = []
        try:
            for root, _, files in os.walk(directory):
                if MANIFEST_FILENAME in files:
                    manifest_path = Path(root) / MANIFEST_FILENAME
                    descriptor = self._load_manifest_file(manifest_path)
                    if descriptor is not None:
                        discovered.append(descriptor)
        except OSError as exc:
            _logger.warning(
                "Error scanning directory %s: %s", directory, exc
            )
        return discovered

    def _load_manifest_file(self, manifest_path: Path) -> ModuleDescriptor | None:
        """Parse a ``manifest.json`` file and register the resulting manifest.

        Returns ``None`` if the module is already registered or if parsing
        fails (errors are logged).
        """
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning(
                "Failed to read manifest file %s: %s", manifest_path, exc
            )
            return None

        try:
            manifest = ModuleManifest.from_dict(data)
        except (ValueError, KeyError) as exc:
            _logger.warning(
                "Invalid manifest at %s: %s", manifest_path, exc
            )
            return None

        # module_path defaults to the directory name if not in the manifest.
        module_path: str = data.get(
            "module_path",
            str(manifest_path.parent).replace(os.sep, "."),
        )

        with self._lock:
            if manifest.id in self._descriptors:
                _logger.debug(
                    "Module %r already registered; skipping %s.",
                    manifest.id,
                    manifest_path,
                )
                return None
            descriptor = ModuleDescriptor(
                manifest=manifest,
                module_path=module_path,
                state=ModuleState.DISCOVERED,
            )
            self._descriptors[manifest.id] = descriptor
            _logger.info(
                "Discovered module %r v%s from %s.",
                manifest.id,
                manifest.version,
                manifest_path,
            )
            return descriptor

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_descriptor(self, module_id: str) -> ModuleDescriptor:
        """Return the descriptor for *module_id*.

        Parameters
        ----------
        module_id:
            The module's unique identifier.

        Returns
        -------
        ModuleDescriptor
            The descriptor registered for *module_id*.

        Raises
        ------
        ModuleDiscoveryError
            If no module with *module_id* is registered.
        """
        with self._lock:
            if module_id not in self._descriptors:
                raise ModuleDiscoveryError(
                    f"Module {module_id!r} not found in the discovery registry.",
                    module_id=module_id,
                )
            return self._descriptors[module_id]

    def all_descriptors(self) -> list[ModuleDescriptor]:
        """Return a snapshot of all registered descriptors.

        Returns
        -------
        list[ModuleDescriptor]
            All currently registered :class:`ModuleDescriptor` instances,
            ordered by module ID for determinism.
        """
        with self._lock:
            return sorted(self._descriptors.values(), key=lambda d: d.id)

    def all_manifests(self) -> list[ModuleManifest]:
        """Return all registered manifests (sorted by module ID).

        Returns
        -------
        list[ModuleManifest]
            All manifests currently in the discovery registry.
        """
        return [d.manifest for d in self.all_descriptors()]

    def is_registered(self, module_id: str) -> bool:
        """Return ``True`` if a module with *module_id* is registered."""
        with self._lock:
            return module_id in self._descriptors

    def __len__(self) -> int:
        """Return the number of registered modules."""
        with self._lock:
            return len(self._descriptors)

    def __contains__(self, module_id: object) -> bool:
        """Support ``in`` operator."""
        if not isinstance(module_id, str):
            return False
        with self._lock:
            return module_id in self._descriptors

    def __iter__(self) -> Iterator[ModuleDescriptor]:
        """Iterate over a snapshot of all descriptors."""
        return iter(self.all_descriptors())


__all__ = ["ModuleDiscovery", "MANIFEST_FILENAME"]