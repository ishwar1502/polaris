# tests/test_contract_framework.py
"""
Smoke-test suite for the POLARIS v5 Contract Framework v1.

These tests validate that the public contract surface behaves correctly
under normal and exceptional conditions.  They are NOT toy examples —
they exercise the real production code paths.
"""

from __future__ import annotations

import threading
from typing import override

import pytest

from core.contracts.capability import (
    Capability,
    CapabilityAlreadyRegisteredError,
    CapabilityNotFoundError,
    CapabilityRegistry,
)
from core.contracts.health import (
    HealthCheckResult,
    HealthReport,
    HealthStatus,
)
from core.contracts.lifecycle import LifecycleError, LifecycleState
from core.contracts.subsystem import (
    DependencyError,
    RegistrationError,
    RuntimeErrorBase,
    SubsystemContract,
    SubsystemMetadata,
)
from core.registry.registry import SubsystemRegistry
from core.types.identifiers import (
    make_capability_id,
    make_subsystem_id,
    make_version,
    parse_capability_id,
)


# ---------------------------------------------------------------------------
# Concrete subsystem fixture
# ---------------------------------------------------------------------------


def _make_metadata(
    name: str = "test.subsystem.alpha",
    version: str = "1.0.0",
    description: str = "Test subsystem.",
    dependencies: frozenset | None = None,
    capabilities: tuple = (),
) -> SubsystemMetadata:
    return SubsystemMetadata(
        id=make_subsystem_id(name),
        name="Alpha",
        version=make_version(version),
        description=description,
        dependencies=dependencies or frozenset(),
        capabilities=capabilities,
    )


class _ConcreteSubsystem(SubsystemContract):
    """Minimal concrete subsystem for testing."""

    def __init__(self, metadata: SubsystemMetadata) -> None:
        super().__init__(metadata)
        self.init_called = False
        self.start_called = False
        self.pause_called = False
        self.resume_called = False
        self.stop_called = False
        self._fail_on: str | None = None

    def fail_on(self, method: str) -> "_ConcreteSubsystem":
        self._fail_on = method
        return self

    def _do_initialize(self) -> None:
        if self._fail_on == "initialize":
            raise RuntimeError("Simulated initialize failure.")
        self.init_called = True

    def _do_start(self) -> None:
        if self._fail_on == "start":
            raise RuntimeError("Simulated start failure.")
        self.start_called = True

    def _do_pause(self) -> None:
        if self._fail_on == "pause":
            raise RuntimeError("Simulated pause failure.")
        self.pause_called = True

    def _do_resume(self) -> None:
        if self._fail_on == "resume":
            raise RuntimeError("Simulated resume failure.")
        self.resume_called = True

    def _do_stop(self) -> None:
        if self._fail_on == "stop":
            raise RuntimeError("Simulated stop failure.")
        self.stop_called = True

    def health(self) -> HealthReport:
        return HealthReport.healthy(
            "All good.",
            checks=[HealthCheckResult(name="self-check", passed=True)],
        )


def _make_subsystem(name: str = "test.subsystem.alpha", **kw) -> _ConcreteSubsystem:
    meta = _make_metadata(name=name, **kw)
    return _ConcreteSubsystem(meta)


# ---------------------------------------------------------------------------
# Identifier tests
# ---------------------------------------------------------------------------


class TestIdentifiers:
    def test_valid_subsystem_id(self) -> None:
        sid = make_subsystem_id("polaris.core.memory")
        assert sid == "polaris.core.memory"

    def test_invalid_subsystem_id_single_segment(self) -> None:
        with pytest.raises(ValueError):
            make_subsystem_id("polaris")

    def test_invalid_subsystem_id_uppercase(self) -> None:
        with pytest.raises(ValueError):
            make_subsystem_id("POLARIS.core")

    def test_valid_capability_id(self) -> None:
        sid = make_subsystem_id("polaris.core.memory")
        cid = make_capability_id(sid, "vector-search")
        assert cid == "polaris.core.memory/vector-search"

    def test_parse_capability_id(self) -> None:
        sid = make_subsystem_id("polaris.core.memory")
        cid = make_capability_id(sid, "vector-search")
        parsed_sid, slug = parse_capability_id(cid)
        assert parsed_sid == "polaris.core.memory"
        assert slug == "vector-search"

    def test_valid_version(self) -> None:
        v = make_version("1.2.3-alpha.1+build.42")
        assert v == "1.2.3-alpha.1+build.42"

    def test_invalid_version(self) -> None:
        with pytest.raises(ValueError):
            make_version("not-a-version")


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_full_happy_path(self) -> None:
        s = _make_subsystem()
        assert s.state is LifecycleState.CREATED
        s.initialize()
        assert s.state is LifecycleState.INITIALIZED
        s.start()
        assert s.state is LifecycleState.RUNNING
        s.pause()
        assert s.state is LifecycleState.PAUSED
        s.resume()
        assert s.state is LifecycleState.RUNNING
        s.stop()
        assert s.state is LifecycleState.STOPPED

    def test_start_before_initialize_raises(self) -> None:
        s = _make_subsystem()
        with pytest.raises(LifecycleError) as exc_info:
            s.start()
        assert exc_info.value.current_state is LifecycleState.CREATED

    def test_pause_before_running_raises(self) -> None:
        s = _make_subsystem()
        s.initialize()
        with pytest.raises(LifecycleError):
            s.pause()

    def test_resume_before_pause_raises(self) -> None:
        s = _make_subsystem()
        s.initialize()
        s.start()
        with pytest.raises(LifecycleError):
            s.resume()

    def test_stop_twice_raises(self) -> None:
        s = _make_subsystem()
        s.initialize()
        s.start()
        s.stop()
        with pytest.raises(LifecycleError):
            s.stop()

    def test_initialize_failure_transitions_to_failed(self) -> None:
        s = _make_subsystem().fail_on("initialize")
        with pytest.raises(RuntimeError):
            s.initialize()
        assert s.state is LifecycleState.FAILED

    def test_start_failure_transitions_to_failed(self) -> None:
        s = _make_subsystem().fail_on("start")
        s.initialize()
        with pytest.raises(RuntimeError):
            s.start()
        assert s.state is LifecycleState.FAILED

    def test_lifecycle_history_recorded(self) -> None:
        s = _make_subsystem()
        s.initialize()
        s.start()
        history = s.lifecycle_history
        states = [t.to_state for t in history]
        assert LifecycleState.INITIALIZED in states
        assert LifecycleState.STARTING in states
        assert LifecycleState.RUNNING in states


# ---------------------------------------------------------------------------
# Health tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_healthy_factory(self) -> None:
        r = HealthReport.healthy()
        assert r.status is HealthStatus.HEALTHY
        assert r.is_operational

    def test_failed_factory(self) -> None:
        r = HealthReport.failed("disk error")
        assert r.status is HealthStatus.FAILED
        assert not r.is_operational

    def test_severity_ordering(self) -> None:
        assert HealthStatus.FAILED.is_worse_than(HealthStatus.UNHEALTHY)
        assert HealthStatus.UNHEALTHY.is_worse_than(HealthStatus.DEGRADED)
        assert HealthStatus.DEGRADED.is_worse_than(HealthStatus.HEALTHY)
        assert HealthStatus.HEALTHY.is_better_than(HealthStatus.DEGRADED)

    def test_failed_checks_filter(self) -> None:
        r = HealthReport(
            status=HealthStatus.DEGRADED,
            message="partial failure",
            checks=[
                HealthCheckResult(name="a", passed=True),
                HealthCheckResult(name="b", passed=False, message="timeout"),
            ],
        )
        assert len(r.failed_checks) == 1
        assert r.failed_checks[0].name == "b"

    def test_to_dict_round_trip(self) -> None:
        r = HealthReport.healthy("ok", checks=[HealthCheckResult(name="ping", passed=True)])
        d = r.to_dict()
        assert d["status"] == "HEALTHY"
        assert len(d["checks"]) == 1

    def test_timestamp_must_be_timezone_aware(self) -> None:
        from datetime import datetime
        with pytest.raises(ValueError):
            HealthReport(
                status=HealthStatus.HEALTHY,
                message="x",
                timestamp=datetime(2024, 1, 1),  # naive — no tzinfo
            )


# ---------------------------------------------------------------------------
# Capability tests
# ---------------------------------------------------------------------------


class TestCapabilityRegistry:
    def _make_cap(self, slug: str = "vector-search") -> Capability:
        sid = make_subsystem_id("test.caps.owner")
        return Capability.create(
            owner=sid,
            slug=slug,
            name="Vector Search",
            description="ANN vector search capability.",
            version="1.0.0",
            tags={"search", "embedding"},
        )

    def test_register_and_get(self) -> None:
        reg = CapabilityRegistry()
        cap = self._make_cap()
        reg.register(cap)
        assert reg.get(cap.id) == cap

    def test_register_duplicate_raises(self) -> None:
        reg = CapabilityRegistry()
        cap = self._make_cap()
        reg.register(cap)
        with pytest.raises(CapabilityAlreadyRegisteredError):
            reg.register(cap)

    def test_unregister(self) -> None:
        reg = CapabilityRegistry()
        cap = self._make_cap()
        reg.register(cap)
        removed = reg.unregister(cap.id)
        assert removed == cap
        assert not reg.has(cap.id)

    def test_get_missing_raises(self) -> None:
        reg = CapabilityRegistry()
        cid = make_capability_id(make_subsystem_id("test.caps.owner"), "missing")
        with pytest.raises(CapabilityNotFoundError):
            reg.get(cid)

    def test_list_by_tag(self) -> None:
        reg = CapabilityRegistry()
        reg.register(self._make_cap("search-a"))
        reg.register(self._make_cap("search-b"))
        results = reg.list_by_tag("embedding")
        assert len(results) == 2

    def test_thread_safety(self) -> None:
        reg = CapabilityRegistry()
        errors: list[Exception] = []
        caps = [
            Capability.create(
                owner=make_subsystem_id("test.caps.owner"),
                slug=f"cap-{i}",
                name=f"Cap {i}",
                description="Test.",
                version="1.0.0",
            )
            for i in range(50)
        ]

        def _register(cap: Capability) -> None:
            try:
                reg.register(cap)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_register, args=(c,)) for c in caps]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(reg) == 50


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestSubsystemRegistry:
    def test_register_and_get(self) -> None:
        reg = SubsystemRegistry()
        s = _make_subsystem()
        reg.register(s)
        assert reg.get(s.metadata.id) is s

    def test_register_duplicate_raises(self) -> None:
        reg = SubsystemRegistry()
        s = _make_subsystem()
        reg.register(s)
        s2 = _make_subsystem()  # same id
        with pytest.raises(RegistrationError):
            reg.register(s2)

    def test_unregister_stopped(self) -> None:
        reg = SubsystemRegistry()
        s = _make_subsystem()
        reg.register(s)
        s.initialize()
        s.start()
        s.stop()
        reg.unregister(s.metadata.id)
        assert not reg.has(s.metadata.id)

    def test_unregister_running_without_force_raises(self) -> None:
        reg = SubsystemRegistry()
        s = _make_subsystem()
        reg.register(s)
        s.initialize()
        s.start()
        with pytest.raises(LifecycleError):
            reg.unregister(s.metadata.id)

    def test_unregister_running_with_force_stop(self) -> None:
        reg = SubsystemRegistry()
        s = _make_subsystem()
        reg.register(s)
        s.initialize()
        s.start()
        reg.unregister(s.metadata.id, force_stop=True)
        assert not reg.has(s.metadata.id)

    def test_verify_dependencies_satisfied(self) -> None:
        reg = SubsystemRegistry()
        dep = _make_subsystem("test.dep.alpha")
        reg.register(dep)
        dep.initialize()
        dep.start()

        child = _make_subsystem(
            "test.child.alpha",
            dependencies=frozenset({make_subsystem_id("test.dep.alpha")}),
        )
        reg.register(child)
        reg.verify_dependencies(child.metadata.id)  # must not raise

    def test_verify_dependencies_missing_raises(self) -> None:
        reg = SubsystemRegistry()
        child = _make_subsystem(
            "test.child.beta",
            dependencies=frozenset({make_subsystem_id("test.dep.missing")}),
        )
        reg.register(child)
        with pytest.raises(DependencyError) as exc_info:
            reg.verify_dependencies(child.metadata.id)
        assert make_subsystem_id("test.dep.missing") in exc_info.value.missing

    def test_verify_dependencies_not_running_raises(self) -> None:
        reg = SubsystemRegistry()
        dep = _make_subsystem("test.dep.beta")
        reg.register(dep)
        # dep is CREATED, not RUNNING

        child = _make_subsystem(
            "test.child.gamma",
            dependencies=frozenset({make_subsystem_id("test.dep.beta")}),
        )
        reg.register(child)
        with pytest.raises(DependencyError):
            reg.verify_dependencies(child.metadata.id)

    def test_build_dependency_order(self) -> None:
        reg = SubsystemRegistry()
        a_id = make_subsystem_id("test.order.alpha")
        b_id = make_subsystem_id("test.order.beta")
        c_id = make_subsystem_id("test.order.gamma")

        a = _make_subsystem("test.order.alpha")
        b = _make_subsystem("test.order.beta", dependencies=frozenset({a_id}))
        c = _make_subsystem("test.order.gamma", dependencies=frozenset({b_id}))

        for s in (a, b, c):
            reg.register(s)

        order = reg.build_dependency_order()
        assert order.index(a_id) < order.index(b_id)
        assert order.index(b_id) < order.index(c_id)

    def test_collect_health(self) -> None:
        reg = SubsystemRegistry()
        for i in range(3):
            reg.register(_make_subsystem(f"test.health.sub{i}"))
        reports = reg.collect_health()
        assert len(reports) == 3
        for report in reports.values():
            assert isinstance(report, HealthReport)

    def test_aggregate_health_all_healthy(self) -> None:
        reg = SubsystemRegistry()
        reg.register(_make_subsystem("test.agg.alpha"))
        reg.register(_make_subsystem("test.agg.beta"))
        assert reg.aggregate_health() is HealthStatus.HEALTHY

    def test_list_by_state(self) -> None:
        reg = SubsystemRegistry()
        s1 = _make_subsystem("test.state.alpha")
        s2 = _make_subsystem("test.state.beta")
        reg.register(s1)
        reg.register(s2)
        s1.initialize()
        s1.start()
        running = reg.list_by_state(LifecycleState.RUNNING)
        created = reg.list_by_state(LifecycleState.CREATED)
        assert s1 in running
        assert s2 in created

    def test_thread_safe_concurrent_registration(self) -> None:
        reg = SubsystemRegistry()
        subsystems = [_make_subsystem(f"test.concurrent.sub{i}") for i in range(20)]
        errors: list[Exception] = []

        def _register(s: SubsystemContract) -> None:
            try:
                reg.register(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_register, args=(s,)) for s in subsystems]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(reg) == 20

    def test_contains_operator(self) -> None:
        reg = SubsystemRegistry()
        s = _make_subsystem()
        reg.register(s)
        assert s.metadata.id in reg
        assert make_subsystem_id("test.nonexistent.sub") not in reg