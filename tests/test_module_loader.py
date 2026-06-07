# tests/test_module_loader.py
"""
Comprehensive pytest suite for the POLARIS v5 Module Loader (Phase 4).

Coverage targets
----------------
* ModuleManifest validation (creation, from_dict, to_dict, edge cases)
* ModuleDescriptor properties and mutation
* ModuleState enumeration behaviour
* DependencyGraph (build, validate, topological sort, cycle detection)
* ModuleDiscovery (in-process registration, filesystem scan, queries)
* ModuleLoader lifecycle (discover → validate → load → init → start → stop → unload)
* load_all / shutdown_all
* Thread safety
* Failure / error handling
* Integration with event bus and subsystem registry stubs

Test count target: 75+
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import json

import pytest

from core.loader import (
    CircularDependencyError,
    DependencyGraph,
    DependencyResolutionError,
    ModuleDescriptor,
    ModuleDiscovery,
    ModuleDiscoveryError,
    ModuleLoader,
    ModuleLoaderError,
    ModuleManifest,
    ModuleState,
    ModuleValidationError,
    OPERATIONAL_STATES,
    STOPPABLE_STATES,
    TERMINAL_STATES,
)


# ===========================================================================
# Helpers / fixtures
# ===========================================================================


def make_manifest(
    module_id: str = "polaris.test.alpha",
    *,
    name: str = "Alpha",
    version: str = "1.0.0",
    description: str = "Test module alpha.",
    dependencies: tuple[str, ...] = (),
    capabilities: tuple[str, ...] = (),
) -> ModuleManifest:
    """Factory for creating test manifests with sensible defaults."""
    return ModuleManifest(
        id=module_id,
        name=name,
        version=version,
        description=description,
        dependencies=dependencies,
        capabilities=capabilities,
    )


class _StubInstance:
    """Minimal stub that mimics a loaded module instance."""

    def __init__(self) -> None:
        self.initialized = False
        self.started = False
        self.stopped = False

    def initialize(self) -> None:
        self.initialized = True

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _FailingInstance:
    """Stub whose lifecycle methods raise on demand."""

    def __init__(self, *, fail_on: str = "start") -> None:
        self._fail_on = fail_on

    def initialize(self) -> None:
        if self._fail_on == "initialize":
            raise RuntimeError("initialize() deliberately failed")

    def start(self) -> None:
        if self._fail_on == "start":
            raise RuntimeError("start() deliberately failed")

    def stop(self) -> None:
        if self._fail_on == "stop":
            raise RuntimeError("stop() deliberately failed")


class _StubClass:
    """A class that can be 'imported' via module_path tricks."""

    def __init__(self) -> None:
        self._stub = _StubInstance()

    def initialize(self) -> None:
        self._stub.initialize()

    def start(self) -> None:
        self._stub.start()

    def stop(self) -> None:
        self._stub.stop()


@pytest.fixture
def discovery() -> ModuleDiscovery:
    return ModuleDiscovery()


@pytest.fixture
def loader() -> ModuleLoader:
    return ModuleLoader()


@pytest.fixture
def alpha_manifest() -> ModuleManifest:
    return make_manifest("polaris.test.alpha")


@pytest.fixture
def beta_manifest() -> ModuleManifest:
    return make_manifest(
        "polaris.test.beta",
        name="Beta",
        dependencies=("polaris.test.alpha",),
    )


@pytest.fixture
def gamma_manifest() -> ModuleManifest:
    return make_manifest(
        "polaris.test.gamma",
        name="Gamma",
        dependencies=("polaris.test.beta",),
    )


# ===========================================================================
# 1. ModuleManifest tests
# ===========================================================================


class TestModuleManifest:

    def test_basic_creation(self):
        m = make_manifest()
        assert m.id == "polaris.test.alpha"
        assert m.name == "Alpha"
        assert m.version == "1.0.0"
        assert m.description == "Test module alpha."
        assert m.dependencies == ()
        assert m.capabilities == ()

    def test_with_dependencies_and_capabilities(self):
        m = make_manifest(
            dependencies=("dep.one", "dep.two"),
            capabilities=("cap_a", "cap_b"),
        )
        assert "dep.one" in m.dependencies
        assert "cap_a" in m.capabilities

    def test_immutable(self):
        m = make_manifest()
        with pytest.raises((AttributeError, TypeError)):
            m.id = "new.id"  # type: ignore[misc]

    def test_invalid_id_empty(self):
        with pytest.raises(ValueError, match="id"):
            ModuleManifest(id="", name="X", version="1.0.0", description="d")

    def test_invalid_id_whitespace(self):
        with pytest.raises(ValueError, match="id"):
            ModuleManifest(id="   ", name="X", version="1.0.0", description="d")

    def test_invalid_name_empty(self):
        with pytest.raises(ValueError, match="name"):
            ModuleManifest(id="a.b.c", name="", version="1.0.0", description="d")

    def test_invalid_version_not_semver(self):
        with pytest.raises(ValueError, match="SemVer"):
            ModuleManifest(id="a.b", name="X", version="not-semver", description="d")

    def test_invalid_version_partial(self):
        with pytest.raises(ValueError, match="SemVer"):
            ModuleManifest(id="a.b", name="X", version="1.0", description="d")

    def test_valid_semver_with_prerelease(self):
        m = ModuleManifest(
            id="a.b",
            name="X",
            version="2.0.0-beta.1",
            description="desc",
        )
        assert m.version == "2.0.0-beta.1"

    def test_invalid_description_empty(self):
        with pytest.raises(ValueError, match="description"):
            ModuleManifest(id="a.b", name="X", version="1.0.0", description="")

    def test_self_dependency_rejected(self):
        with pytest.raises(ValueError, match="itself"):
            ModuleManifest(
                id="a.b",
                name="X",
                version="1.0.0",
                description="d",
                dependencies=("a.b",),
            )

    def test_duplicate_dependency_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            ModuleManifest(
                id="a.b",
                name="X",
                version="1.0.0",
                description="d",
                dependencies=("c.d", "c.d"),
            )

    def test_invalid_capability_empty(self):
        with pytest.raises(ValueError, match="capability"):
            ModuleManifest(
                id="a.b",
                name="X",
                version="1.0.0",
                description="d",
                capabilities=("",),
            )

    def test_from_dict_success(self):
        data = {
            "id": "x.y.z",
            "name": "XYZ",
            "version": "3.2.1",
            "description": "A module.",
            "dependencies": ["a.b"],
            "capabilities": ["do_something"],
        }
        m = ModuleManifest.from_dict(data)
        assert m.id == "x.y.z"
        assert m.dependencies == ("a.b",)
        assert m.capabilities == ("do_something",)

    def test_from_dict_missing_required_key(self):
        with pytest.raises(ValueError, match="missing required field"):
            ModuleManifest.from_dict({"id": "x", "name": "X"})

    def test_from_dict_defaults_no_deps(self):
        m = ModuleManifest.from_dict(
            {"id": "x.y", "name": "Y", "version": "1.0.0", "description": "d"}
        )
        assert m.dependencies == ()
        assert m.capabilities == ()

    def test_to_dict_roundtrip(self):
        m = make_manifest(dependencies=("a.b",), capabilities=("c",))
        d = m.to_dict()
        m2 = ModuleManifest.from_dict(d)
        assert m == m2

    def test_list_coerced_to_tuple_for_dependencies(self):
        # Pass a list; __post_init__ converts to tuple.
        m = ModuleManifest(
            id="a.b",
            name="AB",
            version="1.0.0",
            description="d",
            dependencies=["c.d"],  # type: ignore[arg-type]
        )
        assert isinstance(m.dependencies, tuple)


# ===========================================================================
# 2. ModuleDescriptor tests
# ===========================================================================


class TestModuleDescriptor:

    def test_initial_state(self):
        m = make_manifest()
        d = ModuleDescriptor(manifest=m, module_path="some.path")
        assert d.state is ModuleState.DISCOVERED
        assert d.module_class is None
        assert d.instance is None
        assert d.error is None

    def test_id_name_version_proxied_to_manifest(self):
        m = make_manifest("p.t.x", name="X-mod", version="2.1.0")
        d = ModuleDescriptor(manifest=m, module_path="p")
        assert d.id == "p.t.x"
        assert d.name == "X-mod"
        assert d.version == "2.1.0"

    def test_dependencies_proxied(self):
        m = make_manifest(dependencies=("a.b",))
        d = ModuleDescriptor(manifest=m, module_path="p")
        assert d.dependencies == ("a.b",)

    def test_is_loaded_false_initially(self):
        d = ModuleDescriptor(manifest=make_manifest(), module_path="p")
        assert not d.is_loaded

    def test_is_loaded_true_when_class_set(self):
        d = ModuleDescriptor(manifest=make_manifest(), module_path="p")
        d.module_class = _StubClass
        assert d.is_loaded

    def test_is_initialized_false_initially(self):
        d = ModuleDescriptor(manifest=make_manifest(), module_path="p")
        assert not d.is_initialized

    def test_is_initialized_true_when_instance_set(self):
        d = ModuleDescriptor(manifest=make_manifest(), module_path="p")
        d.instance = _StubInstance()
        assert d.is_initialized

    def test_is_running(self):
        d = ModuleDescriptor(manifest=make_manifest(), module_path="p")
        assert not d.is_running
        d.state = ModuleState.RUNNING
        assert d.is_running

    def test_has_failed(self):
        d = ModuleDescriptor(manifest=make_manifest(), module_path="p")
        assert not d.has_failed
        d.state = ModuleState.FAILED
        assert d.has_failed

    def test_state_is_mutable(self):
        d = ModuleDescriptor(manifest=make_manifest(), module_path="p")
        d.state = ModuleState.VALIDATED
        assert d.state is ModuleState.VALIDATED


# ===========================================================================
# 3. ModuleState tests
# ===========================================================================


class TestModuleState:

    def test_all_states_defined(self):
        states = set(ModuleState)
        expected = {
            "DISCOVERED", "VALIDATED", "LOADED",
            "INITIALIZED", "RUNNING", "PAUSED",
            "STOPPED", "FAILED",
        }
        assert {s.name for s in states} == expected

    def test_operational_states(self):
        assert ModuleState.RUNNING.is_operational()
        assert ModuleState.PAUSED.is_operational()
        assert not ModuleState.STOPPED.is_operational()
        assert not ModuleState.FAILED.is_operational()

    def test_terminal_states(self):
        assert ModuleState.STOPPED.is_terminal()
        assert ModuleState.FAILED.is_terminal()
        assert not ModuleState.RUNNING.is_terminal()

    def test_can_start(self):
        assert ModuleState.INITIALIZED.can_start()
        assert not ModuleState.RUNNING.can_start()
        assert not ModuleState.LOADED.can_start()

    def test_can_stop(self):
        for state in (
            ModuleState.RUNNING,
            ModuleState.PAUSED,
            ModuleState.INITIALIZED,
            ModuleState.LOADED,
        ):
            assert state.can_stop(), f"{state.name} should be stoppable"
        assert not ModuleState.STOPPED.can_stop()
        assert not ModuleState.FAILED.can_stop()

    def test_operational_states_constant(self):
        assert ModuleState.RUNNING in OPERATIONAL_STATES
        assert ModuleState.PAUSED in OPERATIONAL_STATES
        assert ModuleState.STOPPED not in OPERATIONAL_STATES

    def test_terminal_states_constant(self):
        assert ModuleState.STOPPED in TERMINAL_STATES
        assert ModuleState.FAILED in TERMINAL_STATES
        assert ModuleState.RUNNING not in TERMINAL_STATES

    def test_stoppable_states_constant(self):
        assert ModuleState.RUNNING in STOPPABLE_STATES
        assert ModuleState.STOPPED not in STOPPABLE_STATES


# ===========================================================================
# 4. DependencyGraph tests
# ===========================================================================


class TestDependencyGraph:

    def _manifests(self, *ids: str, deps: dict[str, tuple[str, ...]] | None = None) -> list[ModuleManifest]:
        deps = deps or {}
        return [
            make_manifest(mid, dependencies=deps.get(mid, ()))
            for mid in ids
        ]

    def test_empty_graph(self):
        g = DependencyGraph([])
        assert len(g) == 0

    def test_single_node(self):
        g = DependencyGraph(self._manifests("a.b"))
        assert "a.b" in g
        assert len(g) == 1

    def test_topological_order_no_deps(self):
        g = DependencyGraph(self._manifests("a", "b", "c"))
        order = g.topological_order()
        assert set(order) == {"a", "b", "c"}
        assert len(order) == 3

    def test_topological_order_respects_dependencies(self):
        manifests = self._manifests(
            "c", "a", "b",
            deps={"b": ("a",), "c": ("b",)},
        )
        g = DependencyGraph(manifests)
        g.validate()
        order = g.topological_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_diamond_dependency(self):
        # a → b, a → c, b → d, c → d
        manifests = self._manifests(
            "a", "b", "c", "d",
            deps={"b": ("a",), "c": ("a",), "d": ("b", "c")},
        )
        g = DependencyGraph(manifests)
        g.validate()
        order = g.topological_order()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_duplicate_module_id_raises(self):
        m1 = make_manifest("dup")
        m2 = make_manifest("dup")
        with pytest.raises(ModuleValidationError, match="Duplicate"):
            DependencyGraph([m1, m2])

    def test_missing_dependency_raises(self):
        m = make_manifest("a", dependencies=("missing.dep",))
        g = DependencyGraph([m])
        with pytest.raises(DependencyResolutionError, match="missing"):
            g.validate()

    def test_circular_two_nodes(self):
        manifests = self._manifests(
            "x", "y",
            deps={"x": ("y",), "y": ("x",)},
        )
        g = DependencyGraph(manifests)
        with pytest.raises(CircularDependencyError):
            g.validate()

    def test_circular_three_nodes(self):
        manifests = self._manifests(
            "p", "q", "r",
            deps={"p": ("r",), "q": ("p",), "r": ("q",)},
        )
        g = DependencyGraph(manifests)
        with pytest.raises(CircularDependencyError):
            g.validate()

    def test_dependencies_of(self):
        manifests = self._manifests("a", "b", deps={"b": ("a",)})
        g = DependencyGraph(manifests)
        assert g.dependencies_of("b") == {"a"}
        assert g.dependencies_of("a") == set()

    def test_dependents_of(self):
        manifests = self._manifests("a", "b", deps={"b": ("a",)})
        g = DependencyGraph(manifests)
        assert g.dependents_of("a") == {"b"}
        assert g.dependents_of("b") == set()

    def test_transitive_dependencies(self):
        manifests = self._manifests(
            "a", "b", "c",
            deps={"b": ("a",), "c": ("b",)},
        )
        g = DependencyGraph(manifests)
        g.validate()
        trans = g.transitive_dependencies("c")
        assert trans == {"a", "b"}

    def test_transitive_dependencies_empty_for_root(self):
        manifests = self._manifests("a", "b", deps={"b": ("a",)})
        g = DependencyGraph(manifests)
        assert g.transitive_dependencies("a") == frozenset()

    def test_contains(self):
        g = DependencyGraph(self._manifests("x.y"))
        assert "x.y" in g
        assert "z.z" not in g

    def test_module_ids_property(self):
        g = DependencyGraph(self._manifests("a", "b", "c"))
        assert g.module_ids == {"a", "b", "c"}

    def test_manifest_access(self):
        m = make_manifest("a.b.c")
        g = DependencyGraph([m])
        assert g.manifest("a.b.c") is m

    def test_validate_passes_for_valid_graph(self):
        manifests = self._manifests("a", "b", deps={"b": ("a",)})
        g = DependencyGraph(manifests)
        g.validate()  # must not raise

    def test_missing_dep_error_carries_module_id(self):
        m = make_manifest("a", dependencies=("gone",))
        g = DependencyGraph([m])
        with pytest.raises(DependencyResolutionError) as exc_info:
            g.validate()
        assert exc_info.value.module_id == "a"
        assert "gone" in exc_info.value.missing


# ===========================================================================
# 5. ModuleDiscovery tests
# ===========================================================================


class TestModuleDiscovery:

    def test_register_manifest(self, discovery):
        m = make_manifest()
        d = discovery.register_manifest(m, module_path="some.path")
        assert d.id == m.id
        assert d.state is ModuleState.DISCOVERED
        assert d.module_path == "some.path"

    def test_register_duplicate_raises(self, discovery):
        m = make_manifest()
        discovery.register_manifest(m, module_path="p")
        with pytest.raises(ModuleDiscoveryError, match="already registered"):
            discovery.register_manifest(m, module_path="p")

    def test_get_descriptor_success(self, discovery):
        m = make_manifest()
        discovery.register_manifest(m, module_path="p")
        d = discovery.get_descriptor(m.id)
        assert d.id == m.id

    def test_get_descriptor_not_found(self, discovery):
        with pytest.raises(ModuleDiscoveryError):
            discovery.get_descriptor("nonexistent.module")

    def test_is_registered(self, discovery):
        m = make_manifest()
        assert not discovery.is_registered(m.id)
        discovery.register_manifest(m, module_path="p")
        assert discovery.is_registered(m.id)

    def test_unregister(self, discovery):
        m = make_manifest()
        discovery.register_manifest(m, module_path="p")
        discovery.unregister(m.id)
        assert not discovery.is_registered(m.id)

    def test_unregister_nonexistent_raises(self, discovery):
        with pytest.raises(ModuleDiscoveryError, match="not found"):
            discovery.unregister("ghost.module")

    def test_all_descriptors_sorted(self, discovery):
        for mid in ("z.z", "a.a", "m.m"):
            discovery.register_manifest(make_manifest(mid), module_path="p")
        ids = [d.id for d in discovery.all_descriptors()]
        assert ids == sorted(ids)

    def test_all_manifests(self, discovery):
        m1 = make_manifest("a.a")
        m2 = make_manifest("b.b")
        discovery.register_manifest(m1, module_path="p")
        discovery.register_manifest(m2, module_path="p")
        manifests = discovery.all_manifests()
        assert m1 in manifests
        assert m2 in manifests

    def test_len(self, discovery):
        assert len(discovery) == 0
        discovery.register_manifest(make_manifest(), module_path="p")
        assert len(discovery) == 1

    def test_contains(self, discovery):
        m = make_manifest()
        discovery.register_manifest(m, module_path="p")
        assert m.id in discovery
        assert "not.here" not in discovery

    def test_iter(self, discovery):
        for mid in ("a", "b", "c"):
            discovery.register_manifest(make_manifest(mid), module_path="p")
        ids = [d.id for d in discovery]
        assert set(ids) == {"a", "b", "c"}

    def test_add_search_path_nonexistent_raises(self):
        d = ModuleDiscovery()
        with pytest.raises(ModuleDiscoveryError, match="does not exist"):
            d.add_search_path("/this/path/does/not/exist")

    def test_add_search_path_file_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        d = ModuleDiscovery()
        with pytest.raises(ModuleDiscoveryError, match="not a directory"):
            d.add_search_path(f)

    def test_filesystem_scan(self, tmp_path):
        """Discovery can find manifest.json files on disk."""
        mod_dir = tmp_path / "mymod"
        mod_dir.mkdir()
        manifest_data = {
            "id": "polaris.test.fs",
            "name": "FS Module",
            "version": "1.0.0",
            "description": "Loaded from filesystem.",
        }
        (mod_dir / "manifest.json").write_text(json.dumps(manifest_data))

        d = ModuleDiscovery(search_paths=[tmp_path])
        descriptors = d.scan()
        ids = [desc.id for desc in descriptors]
        assert "polaris.test.fs" in ids

    def test_filesystem_scan_invalid_json_skipped(self, tmp_path):
        """Manifests with bad JSON are skipped with a warning."""
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "manifest.json").write_text("NOT JSON {{{{")

        d = ModuleDiscovery(search_paths=[tmp_path])
        descriptors = d.scan()  # must not raise
        assert len(descriptors) == 0

    def test_filesystem_scan_missing_field_skipped(self, tmp_path):
        """Manifests missing required fields are skipped."""
        bad_dir = tmp_path / "incomplete"
        bad_dir.mkdir()
        (bad_dir / "manifest.json").write_text(json.dumps({"id": "x"}))

        d = ModuleDiscovery(search_paths=[tmp_path])
        descriptors = d.scan()
        assert len(descriptors) == 0

    def test_scan_does_not_duplicate_already_registered(self, tmp_path):
        """Modules already in-process registered are not duplicated by scan."""
        mod_dir = tmp_path / "mod"
        mod_dir.mkdir()
        manifest_data = {
            "id": "polaris.test.nodup",
            "name": "NoDup",
            "version": "1.0.0",
            "description": "d",
        }
        (mod_dir / "manifest.json").write_text(json.dumps(manifest_data))

        d = ModuleDiscovery(search_paths=[tmp_path])
        m = ModuleManifest.from_dict(manifest_data)
        d.register_manifest(m, module_path="p")  # pre-register
        descriptors = d.scan()
        # Should still be exactly 1 entry for this id
        ids = [desc.id for desc in descriptors if desc.id == "polaris.test.nodup"]
        assert len(ids) == 1


# ===========================================================================
# 6. ModuleLoader lifecycle tests
# ===========================================================================


def _make_loader_with_stub_class(
    manifest: ModuleManifest,
    *,
    instance_factory: type | None = None,
) -> tuple[ModuleLoader, type]:
    """Register a fake importable class with the loader and return both."""
    loader = ModuleLoader()
    # Create a stub class and shim it into sys.modules
    StubClass = instance_factory or _StubClass
    # Build a fake module
    fake_module_name = f"_fake_module_{manifest.id.replace('.', '_')}"
    fake_mod = type(sys)("fake_module")
    fake_mod.StubModule = StubClass  # type: ignore[attr-defined]
    sys.modules[fake_module_name] = fake_mod
    loader.discover(manifest, module_path=f"{fake_module_name}.StubModule")
    return loader, StubClass


class TestModuleLoaderLifecycle:

    def test_discover_returns_descriptor(self, loader, alpha_manifest):
        d = loader.discover(alpha_manifest, module_path="p")
        assert d.id == alpha_manifest.id
        assert d.state is ModuleState.DISCOVERED

    def test_discover_duplicate_raises(self, loader, alpha_manifest):
        loader.discover(alpha_manifest, module_path="p")
        with pytest.raises(ModuleLoaderError):
            loader.discover(alpha_manifest, module_path="p")

    def test_validate_transitions_to_validated(self, loader, alpha_manifest):
        loader.discover(alpha_manifest, module_path="p")
        loader.validate()
        d = loader.get_descriptor(alpha_manifest.id)
        assert d.state is ModuleState.VALIDATED

    def test_validate_with_no_modules_is_noop(self, loader):
        loader.validate()  # must not raise

    def test_validate_missing_dependency_raises(self, loader):
        m = make_manifest("a", dependencies=("missing.dep",))
        loader.discover(m, module_path="p")
        with pytest.raises(DependencyResolutionError):
            loader.validate()

    def test_validate_circular_raises(self, loader):
        m1 = make_manifest("x", dependencies=("y",))
        m2 = make_manifest("y", dependencies=("x",))
        loader.discover(m1, module_path="p")
        loader.discover(m2, module_path="p")
        with pytest.raises(CircularDependencyError):
            loader.validate()

    def test_load_transitions_to_loaded(self):
        m = make_manifest("test.load")
        loader, _ = _make_loader_with_stub_class(m)
        loader.validate()
        loader.load(m.id)
        d = loader.get_descriptor(m.id)
        assert d.state is ModuleState.LOADED
        assert d.module_class is not None

    def test_load_not_validated_raises(self, loader, alpha_manifest):
        loader.discover(alpha_manifest, module_path="p")
        with pytest.raises(ModuleLoaderError, match="VALIDATED"):
            loader.load(alpha_manifest.id)

    def test_load_unregistered_raises(self, loader):
        with pytest.raises(ModuleLoaderError):
            loader.load("does.not.exist")

    def test_load_bad_import_path_raises(self, loader, alpha_manifest):
        loader.discover(alpha_manifest, module_path="no_such_module_xyz.Cls")
        loader.validate()
        with pytest.raises(ModuleLoaderError, match="Failed to import"):
            loader.load(alpha_manifest.id)

    def test_load_bad_class_name_raises(self, loader, alpha_manifest):
        # Put a real module but bad class name
        loader.discover(alpha_manifest, module_path="os.NoSuchClass")
        loader.validate()
        with pytest.raises(ModuleLoaderError, match="not found"):
            loader.load(alpha_manifest.id)

    def test_initialize_transitions_to_initialized(self):
        m = make_manifest("test.init")
        loader, _ = _make_loader_with_stub_class(m)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        d = loader.get_descriptor(m.id)
        assert d.state is ModuleState.INITIALIZED
        assert d.instance is not None

    def test_initialize_not_loaded_raises(self, loader, alpha_manifest):
        loader.discover(alpha_manifest, module_path="p")
        loader.validate()
        with pytest.raises(ModuleLoaderError, match="LOADED"):
            loader.initialize(alpha_manifest.id)

    def test_start_transitions_to_running(self):
        m = make_manifest("test.start")
        loader, _ = _make_loader_with_stub_class(m)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        loader.start(m.id)
        d = loader.get_descriptor(m.id)
        assert d.state is ModuleState.RUNNING

    def test_start_not_initialized_raises(self):
        m = make_manifest("test.start.fail")
        loader, _ = _make_loader_with_stub_class(m)
        loader.validate()
        loader.load(m.id)
        with pytest.raises(ModuleLoaderError, match="INITIALIZED"):
            loader.start(m.id)

    def test_start_dependency_not_running_raises(self):
        m_dep = make_manifest("dep.mod")
        m = make_manifest("main.mod", dependencies=("dep.mod",))
        loader = ModuleLoader()
        for man in (m_dep, m):
            fake_name = f"_fake_{man.id.replace('.', '_')}"
            fake_mod = type(sys)("f")
            fake_mod.Stub = _StubClass  # type: ignore[attr-defined]
            sys.modules[fake_name] = fake_mod
            loader.discover(man, module_path=f"{fake_name}.Stub")
        loader.validate()
        loader.load(m_dep.id)
        loader.initialize(m_dep.id)
        # dep is INITIALIZED, not RUNNING
        loader.load(m.id)
        loader.initialize(m.id)
        with pytest.raises(DependencyResolutionError, match="not RUNNING"):
            loader.start(m.id)

    def test_stop_transitions_to_stopped(self):
        m = make_manifest("test.stop")
        loader, _ = _make_loader_with_stub_class(m)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        loader.start(m.id)
        loader.stop(m.id)
        d = loader.get_descriptor(m.id)
        assert d.state is ModuleState.STOPPED

    def test_stop_already_stopped_is_idempotent(self):
        m = make_manifest("test.stop2")
        loader, _ = _make_loader_with_stub_class(m)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        loader.start(m.id)
        loader.stop(m.id)
        loader.stop(m.id)  # second call must not raise
        assert loader.get_descriptor(m.id).state is ModuleState.STOPPED

    def test_stop_not_stoppable_raises(self, loader, alpha_manifest):
        loader.discover(alpha_manifest, module_path="p")
        loader.validate()
        with pytest.raises(ModuleLoaderError):
            loader.stop(alpha_manifest.id)

    def test_unload_resets_to_discovered(self):
        m = make_manifest("test.unload")
        loader, _ = _make_loader_with_stub_class(m)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        loader.start(m.id)
        loader.stop(m.id)
        loader.unload(m.id)
        d = loader.get_descriptor(m.id)
        assert d.state is ModuleState.DISCOVERED
        assert d.module_class is None
        assert d.instance is None

    def test_unload_running_module_raises(self):
        m = make_manifest("test.unload.running")
        loader, _ = _make_loader_with_stub_class(m)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        loader.start(m.id)
        with pytest.raises(ModuleLoaderError, match="RUNNING"):
            loader.unload(m.id)

    def test_unload_with_running_dependents_raises(self):
        m_dep = make_manifest("base.mod")
        m_main = make_manifest("top.mod", dependencies=("base.mod",))
        loader = ModuleLoader()
        for man in (m_dep, m_main):
            fn = f"_fk_{man.id.replace('.', '_')}"
            fmod = type(sys)("f")
            fmod.Stub = _StubClass  # type: ignore[attr-defined]
            sys.modules[fn] = fmod
            loader.discover(man, module_path=f"{fn}.Stub")
        loader.validate()
        for mid in ("base.mod", "top.mod"):
            loader.load(mid)
            loader.initialize(mid)
        loader.start("base.mod")
        loader.start("top.mod")
        with pytest.raises(ModuleLoaderError, match="dependent"):
            loader.unload("base.mod")


# ===========================================================================
# 7. Bulk operations (load_all / shutdown_all)
# ===========================================================================


def _register_chain(loader: ModuleLoader, ids: list[str]) -> None:
    """Register a linear chain: ids[0] ← ids[1] ← ids[2] … in loader."""
    for i, mid in enumerate(ids):
        deps = (ids[i - 1],) if i > 0 else ()
        m = make_manifest(mid, dependencies=deps)
        fn = f"_chain_{mid.replace('.', '_')}"
        fmod = type(sys)("f")
        fmod.Stub = _StubClass  # type: ignore[attr-defined]
        sys.modules[fn] = fmod
        loader.discover(m, module_path=f"{fn}.Stub")


class TestBulkOperations:

    def test_load_all_starts_all_modules(self):
        loader = ModuleLoader()
        _register_chain(loader, ["m.a", "m.b", "m.c"])
        loader.load_all()
        for mid in ("m.a", "m.b", "m.c"):
            assert loader.get_descriptor(mid).state is ModuleState.RUNNING

    def test_load_all_respects_dependency_order(self):
        loader = ModuleLoader()
        _register_chain(loader, ["o.a", "o.b", "o.c"])
        loader.load_all()
        # All running means order was satisfied
        for mid in ("o.a", "o.b", "o.c"):
            assert loader.is_running(mid)

    def test_shutdown_all_stops_all_running(self):
        loader = ModuleLoader()
        _register_chain(loader, ["s.a", "s.b"])
        loader.load_all()
        loader.shutdown_all()
        for mid in ("s.a", "s.b"):
            assert loader.get_descriptor(mid).state is ModuleState.STOPPED

    def test_shutdown_all_with_no_modules_is_noop(self):
        loader = ModuleLoader()
        loader.shutdown_all()  # must not raise

    def test_load_all_is_idempotent_for_already_running(self):
        loader = ModuleLoader()
        _register_chain(loader, ["i.a"])
        loader.load_all()
        loader.load_all()  # must not raise or break state
        assert loader.is_running("i.a")


# ===========================================================================
# 8. State query helpers
# ===========================================================================


class TestQueryHelpers:

    def test_descriptors_in_state(self):
        loader = ModuleLoader()
        _register_chain(loader, ["q.a", "q.b"])
        loader.validate()
        loader.load("q.a")
        loader.initialize("q.a")
        loader.start("q.a")
        loader.load("q.b")
        loader.initialize("q.b")
        loader.start("q.b")
        running = loader.descriptors_in_state(ModuleState.RUNNING)
        assert len(running) == 2

    def test_is_running_true(self):
        loader = ModuleLoader()
        _register_chain(loader, ["r.a"])
        loader.load_all()
        assert loader.is_running("r.a")

    def test_is_running_false_for_unknown(self):
        loader = ModuleLoader()
        assert not loader.is_running("not.there")

    def test_all_descriptors_returns_all(self):
        loader = ModuleLoader()
        _register_chain(loader, ["all.a", "all.b", "all.c"])
        all_d = loader.all_descriptors()
        assert {d.id for d in all_d} == {"all.a", "all.b", "all.c"}

    def test_get_descriptor_unknown_raises(self, loader):
        with pytest.raises(ModuleLoaderError):
            loader.get_descriptor("no.such.module")


# ===========================================================================
# 9. Failure handling
# ===========================================================================


class _FailInitClass:
    def initialize(self):
        raise RuntimeError("init failed")

    def start(self):
        pass

    def stop(self):
        pass


class _FailStartClass:
    def initialize(self):
        pass

    def start(self):
        raise RuntimeError("start failed")

    def stop(self):
        pass


class _FailStopClass:
    def initialize(self):
        pass

    def start(self):
        pass

    def stop(self):
        raise RuntimeError("stop failed")


def _register_with_class(loader: ModuleLoader, manifest: ModuleManifest, cls: type) -> None:
    fn = f"_fail_{manifest.id.replace('.', '_')}"
    fmod = type(sys)("f")
    fmod.Cls = cls  # type: ignore[attr-defined]
    sys.modules[fn] = fmod
    loader.discover(manifest, module_path=f"{fn}.Cls")


class TestFailureHandling:

    def test_initialize_failure_marks_failed(self):
        loader = ModuleLoader()
        m = make_manifest("fail.init")
        _register_with_class(loader, m, _FailInitClass)
        loader.validate()
        loader.load(m.id)
        with pytest.raises(ModuleLoaderError):
            loader.initialize(m.id)
        assert loader.get_descriptor(m.id).state is ModuleState.FAILED

    def test_start_failure_marks_failed(self):
        loader = ModuleLoader()
        m = make_manifest("fail.start")
        _register_with_class(loader, m, _FailStartClass)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        with pytest.raises(ModuleLoaderError):
            loader.start(m.id)
        assert loader.get_descriptor(m.id).state is ModuleState.FAILED

    def test_stop_failure_marks_failed(self):
        loader = ModuleLoader()
        m = make_manifest("fail.stop")
        _register_with_class(loader, m, _FailStopClass)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        loader.start(m.id)
        with pytest.raises(ModuleLoaderError):
            loader.stop(m.id)
        assert loader.get_descriptor(m.id).state is ModuleState.FAILED

    def test_failed_module_error_attribute_set(self):
        loader = ModuleLoader()
        m = make_manifest("fail.attr")
        _register_with_class(loader, m, _FailInitClass)
        loader.validate()
        loader.load(m.id)
        with pytest.raises(ModuleLoaderError):
            loader.initialize(m.id)
        d = loader.get_descriptor(m.id)
        assert d.error is not None

    def test_stop_failure_during_shutdown_all_continues(self):
        """shutdown_all must continue past failing modules."""
        loader = ModuleLoader()
        m_good = make_manifest("g.good")
        m_bad = make_manifest("g.bad")
        _register_with_class(loader, m_good, _StubClass)
        _register_with_class(loader, m_bad, _FailStopClass)
        loader.validate()
        loader.load(m_good.id)
        loader.initialize(m_good.id)
        loader.start(m_good.id)
        loader.load(m_bad.id)
        loader.initialize(m_bad.id)
        loader.start(m_bad.id)
        loader.shutdown_all()  # must not raise
        # good module should be STOPPED
        assert loader.get_descriptor(m_good.id).state is ModuleState.STOPPED


# ===========================================================================
# 10. Thread safety tests
# ===========================================================================


class TestThreadSafety:

    def test_concurrent_discover(self):
        """Many threads can call discover() concurrently without corruption."""
        loader = ModuleLoader()
        errors: list[Exception] = []

        def _register(idx: int) -> None:
            try:
                m = make_manifest(f"thread.mod.{idx}")
                loader.discover(m, module_path=f"p{idx}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_register, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent discover errors: {errors}"
        assert len(loader.all_descriptors()) == 20

    def test_concurrent_query(self):
        """all_descriptors() is safe to call from multiple threads."""
        loader = ModuleLoader()
        for i in range(10):
            loader.discover(make_manifest(f"cq.mod.{i}"), module_path=f"p{i}")

        results: list[int] = []
        errors: list[Exception] = []

        def _query() -> None:
            try:
                results.append(len(loader.all_descriptors()))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_query) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r == 10 for r in results)

    def test_concurrent_load_all(self):
        """load_all is serialised; concurrent calls must not corrupt state."""
        loader = ModuleLoader()
        _register_chain(loader, ["ts.a", "ts.b"])

        ready = threading.Barrier(3)
        errors: list[Exception] = []

        def _call() -> None:
            ready.wait()
            try:
                loader.load_all()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_call) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for mid in ("ts.a", "ts.b"):
            assert loader.is_running(mid)

    def test_concurrent_shutdown(self):
        """shutdown_all is safe when called from multiple threads."""
        loader = ModuleLoader()
        _register_chain(loader, ["sd.a", "sd.b"])
        loader.load_all()

        errors: list[Exception] = []

        def _shutdown() -> None:
            try:
                loader.shutdown_all()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_shutdown) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ===========================================================================
# 11. Event bus integration
# ===========================================================================


class TestEventBusIntegration:

    def test_start_publishes_event(self):
        mock_bus = MagicMock()
        loader = ModuleLoader(event_bus=mock_bus)
        m = make_manifest("ev.mod")
        _register_with_class(loader, m, _StubClass)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        loader.start(m.id)
        mock_bus.publish.assert_called()

    def test_stop_publishes_event(self):
        mock_bus = MagicMock()
        loader = ModuleLoader(event_bus=mock_bus)
        m = make_manifest("ev.stop")
        _register_with_class(loader, m, _StubClass)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        loader.start(m.id)
        loader.stop(m.id)
        # publish called at least twice (start + stop)
        assert mock_bus.publish.call_count >= 2

    def test_event_bus_failure_does_not_abort_start(self):
        """If event publishing fails, start() must still succeed."""
        mock_bus = MagicMock()
        mock_bus.publish.side_effect = RuntimeError("bus down")
        loader = ModuleLoader(event_bus=mock_bus)
        m = make_manifest("ev.robust")
        _register_with_class(loader, m, _StubClass)
        loader.validate()
        loader.load(m.id)
        loader.initialize(m.id)
        loader.start(m.id)  # must not raise
        assert loader.is_running(m.id)


# ===========================================================================
# 12. Exceptions tests
# ===========================================================================


class TestExceptions:

    def test_module_loader_error_module_id(self):
        exc = ModuleLoaderError("msg", module_id="x.y")
        assert exc.module_id == "x.y"
        assert str(exc) == "msg"

    def test_discovery_error_module_path(self):
        exc = ModuleDiscoveryError("msg", module_path="/some/path", module_id="a")
        assert exc.module_path == "/some/path"
        assert exc.module_id == "a"

    def test_dependency_resolution_error_missing(self):
        exc = DependencyResolutionError("msg", module_id="a", missing=["b", "c"])
        assert exc.missing == ["b", "c"]
        assert exc.module_id == "a"

    def test_circular_dependency_error_cycle(self):
        exc = CircularDependencyError("msg", module_id="x", cycle=["x", "y", "x"])
        assert exc.cycle == ["x", "y", "x"]
        assert exc.module_id == "x"

    def test_module_validation_error_field(self):
        exc = ModuleValidationError("bad", module_id="m", field="version", invalid_value="bad")
        assert exc.field == "version"
        assert exc.invalid_value == "bad"

    def test_exception_hierarchy(self):
        assert issubclass(ModuleDiscoveryError, ModuleLoaderError)
        assert issubclass(DependencyResolutionError, ModuleLoaderError)
        assert issubclass(CircularDependencyError, DependencyResolutionError)
        assert issubclass(ModuleValidationError, ModuleLoaderError)

    def test_circular_inherits_from_dependency_resolution(self):
        exc = CircularDependencyError("cycle!")
        assert isinstance(exc, DependencyResolutionError)
        assert isinstance(exc, ModuleLoaderError)


# ===========================================================================
# 13. discover_all convenience method
# ===========================================================================


class TestDiscoverAll:

    def test_discover_all_registers_multiple(self):
        loader = ModuleLoader()
        manifests = [
            (make_manifest("da.a"), "p.a"),
            (make_manifest("da.b"), "p.b"),
            (make_manifest("da.c"), "p.c"),
        ]
        descriptors = loader.discover_all(manifests)
        assert len(descriptors) == 3
        ids = {d.id for d in descriptors}
        assert ids == {"da.a", "da.b", "da.c"}


# ===========================================================================
# 14. DependencyGraph edge cases
# ===========================================================================


class TestDependencyGraphEdgeCases:

    def test_iterate_over_graph(self):
        manifests = [make_manifest(mid) for mid in ("p", "q", "r")]
        g = DependencyGraph(manifests)
        assert set(g) == {"p", "q", "r"}

    def test_len_of_graph(self):
        g = DependencyGraph([make_manifest("solo")])
        assert len(g) == 1

    def test_kahn_sort_stable_for_independent_nodes(self):
        """Independent nodes should appear in alphabetical order (sorted seed)."""
        manifests = [make_manifest(mid) for mid in ("z", "a", "m")]
        g = DependencyGraph(manifests)
        order = g.topological_order()
        assert order == sorted(order)

    def test_validate_then_topological_order(self):
        manifests = [
            make_manifest("root"),
            make_manifest("child", dependencies=("root",)),
        ]
        g = DependencyGraph(manifests)
        g.validate()
        order = g.topological_order()
        assert order.index("root") < order.index("child")