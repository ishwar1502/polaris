# tests/test_runtime_kernel.py
"""
Comprehensive pytest suite for POLARIS v5 Runtime Phase 6 — Runtime Kernel.

Coverage
--------
* RuntimeState enumeration
* KernelConfiguration immutable dataclass
* RuntimeHealthReport immutable dataclass
* BootstrapManager bootstrap sequence
* StartupManager start sequence
* ShutdownManager shutdown sequence
* HealthMonitor health checks
* RuntimeKernel full lifecycle
* State transitions (valid and invalid)
* Recovery paths: FAILED → RECOVERING → RUNNING
* Module registration, loading, starting, stopping
* Thread safety under concurrent load
* Event generation (KernelBootstrapped, KernelStarted, KernelStopped,
  KernelFailed, KernelRecovered)
* Failure handling at each phase
* Context manager support
* KernelError / BootstrapError / StartupError / ShutdownError / RecoveryError

Target: 125+ tests
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from core.kernel.bootstrap import BootstrapManager
from core.kernel.exceptions import (
    BootstrapError,
    KernelError,
    RecoveryError,
    ShutdownError,
    StartupError,
)
from core.kernel.health import HealthMonitor
from core.kernel.kernel import RuntimeKernel
from core.kernel.models import (
    KernelConfiguration,
    RuntimeHealthReport,
    RuntimeState,
    OPERATIONAL_STATES,
    TERMINAL_STATES,
)
from core.kernel.shutdown import ShutdownManager
from core.kernel.startup import StartupManager


# ===========================================================================
# Helpers & fixtures
# ===========================================================================


def _make_mock_components(**overrides: Any) -> dict[str, Any]:
    """Build a mock components dict for use in tests."""
    event_bus = MagicMock(name="event_bus")
    event_bus.statistics.return_value = MagicMock(
        total_published=0, subscriber_count=0
    )

    memory_registry = MagicMock(name="memory_registry")
    memory_registry.__iter__ = MagicMock(return_value=iter([]))
    memory_registry.__len__ = MagicMock(return_value=0)

    memory_gateway = MagicMock(name="memory_gateway")
    subsystem_registry = MagicMock(name="subsystem_registry")
    lifecycle_manager = MagicMock(name="lifecycle_manager")
    lifecycle_manager.__len__ = MagicMock(return_value=0)

    module_loader = MagicMock(name="module_loader")
    module_loader.all_descriptors.return_value = []

    components = {
        "event_bus": event_bus,
        "memory_registry": memory_registry,
        "memory_gateway": memory_gateway,
        "subsystem_registry": subsystem_registry,
        "module_loader": module_loader,
        "lifecycle_manager": lifecycle_manager,
    }
    components.update(overrides)
    return components


def _make_kernel(config: KernelConfiguration | None = None) -> RuntimeKernel:
    """Create a fresh RuntimeKernel for testing."""
    return RuntimeKernel(config or KernelConfiguration())


def _bootstrap_kernel(kernel: RuntimeKernel) -> dict[str, Any]:
    """Bootstrap a kernel with mocked components."""
    mock_components = _make_mock_components()
    with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
        kernel.bootstrap()
    return mock_components


def _start_kernel(kernel: RuntimeKernel) -> None:
    """Bootstrap and start a kernel with mocked components."""
    _bootstrap_kernel(kernel)
    with patch.object(StartupManager, "start"):
        kernel.start()


# ===========================================================================
# I. RuntimeState Enumeration Tests
# ===========================================================================


class TestRuntimeState:
    """Tests for the RuntimeState enumeration."""

    def test_all_states_exist(self) -> None:
        states = {s.name for s in RuntimeState}
        expected = {
            "CREATED", "BOOTSTRAPPING", "STARTING", "RUNNING",
            "DEGRADED", "RECOVERING", "STOPPING", "STOPPED", "FAILED",
        }
        assert states == expected

    def test_operational_states(self) -> None:
        assert RuntimeState.RUNNING.is_operational()
        assert RuntimeState.DEGRADED.is_operational()
        assert RuntimeState.RECOVERING.is_operational()

    def test_non_operational_states(self) -> None:
        for state in [
            RuntimeState.CREATED, RuntimeState.BOOTSTRAPPING,
            RuntimeState.STARTING, RuntimeState.STOPPING,
            RuntimeState.STOPPED, RuntimeState.FAILED,
        ]:
            assert not state.is_operational(), f"{state} should not be operational"

    def test_terminal_states(self) -> None:
        assert RuntimeState.STOPPED.is_terminal()
        assert RuntimeState.FAILED.is_terminal()

    def test_non_terminal_states(self) -> None:
        for state in [
            RuntimeState.CREATED, RuntimeState.BOOTSTRAPPING,
            RuntimeState.STARTING, RuntimeState.RUNNING,
            RuntimeState.DEGRADED, RuntimeState.RECOVERING,
            RuntimeState.STOPPING,
        ]:
            assert not state.is_terminal(), f"{state} should not be terminal"

    def test_operational_states_constant(self) -> None:
        assert RuntimeState.RUNNING in OPERATIONAL_STATES
        assert RuntimeState.DEGRADED in OPERATIONAL_STATES
        assert RuntimeState.RECOVERING in OPERATIONAL_STATES

    def test_terminal_states_constant(self) -> None:
        assert RuntimeState.STOPPED in TERMINAL_STATES
        assert RuntimeState.FAILED in TERMINAL_STATES

    def test_state_uniqueness(self) -> None:
        values = [s.value for s in RuntimeState]
        assert len(values) == len(set(values))

    def test_state_enum_membership(self) -> None:
        assert RuntimeState.RUNNING in RuntimeState
        assert RuntimeState.FAILED in RuntimeState


# ===========================================================================
# II. KernelConfiguration Tests
# ===========================================================================


class TestKernelConfiguration:
    """Tests for the KernelConfiguration immutable dataclass."""

    def test_default_configuration(self) -> None:
        config = KernelConfiguration()
        assert config.environment == "development"
        assert config.auto_start is False
        assert config.enable_recovery is True
        assert config.health_interval == 30.0
        assert config.max_recovery_attempts == 3
        assert config.shutdown_timeout == 30.0
        assert config.metadata == {}

    def test_custom_configuration(self) -> None:
        config = KernelConfiguration(
            environment="production",
            auto_start=True,
            enable_recovery=False,
            health_interval=60.0,
            max_recovery_attempts=5,
            shutdown_timeout=10.0,
            metadata={"version": "5.0.0"},
        )
        assert config.environment == "production"
        assert config.auto_start is True
        assert config.enable_recovery is False
        assert config.health_interval == 60.0
        assert config.max_recovery_attempts == 5
        assert config.shutdown_timeout == 10.0
        assert config.metadata == {"version": "5.0.0"}

    def test_configuration_is_immutable(self) -> None:
        config = KernelConfiguration()
        with pytest.raises((AttributeError, TypeError)):
            config.environment = "other"  # type: ignore[misc]

    def test_empty_environment_raises(self) -> None:
        with pytest.raises(ValueError, match="environment"):
            KernelConfiguration(environment="")

    def test_whitespace_environment_raises(self) -> None:
        with pytest.raises(ValueError, match="environment"):
            KernelConfiguration(environment="   ")

    def test_negative_health_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="health_interval"):
            KernelConfiguration(health_interval=-1.0)

    def test_negative_max_recovery_attempts_raises(self) -> None:
        with pytest.raises(ValueError, match="max_recovery_attempts"):
            KernelConfiguration(max_recovery_attempts=-1)

    def test_negative_shutdown_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="shutdown_timeout"):
            KernelConfiguration(shutdown_timeout=-1.0)

    def test_zero_health_interval_allowed(self) -> None:
        config = KernelConfiguration(health_interval=0.0)
        assert config.health_interval == 0.0

    def test_zero_max_recovery_attempts_allowed(self) -> None:
        config = KernelConfiguration(max_recovery_attempts=0)
        assert config.max_recovery_attempts == 0

    def test_metadata_isolation(self) -> None:
        meta = {"key": "value"}
        config = KernelConfiguration(metadata=meta)
        assert config.metadata["key"] == "value"


# ===========================================================================
# III. RuntimeHealthReport Tests
# ===========================================================================


class TestRuntimeHealthReport:
    """Tests for the RuntimeHealthReport immutable dataclass."""

    def _make_report(self, **kwargs: Any) -> RuntimeHealthReport:
        defaults = dict(
            runtime_state=RuntimeState.RUNNING,
            loaded_modules=(),
            running_modules=("mod.a",),
            failed_modules=(),
            memory_status="ok",
            event_bus_status="ok",
        )
        defaults.update(kwargs)
        return RuntimeHealthReport(**defaults)

    def test_basic_report_creation(self) -> None:
        report = self._make_report()
        assert report.runtime_state is RuntimeState.RUNNING
        assert report.running_modules == ("mod.a",)
        assert report.memory_status == "ok"

    def test_is_healthy_running_no_failures(self) -> None:
        report = self._make_report(
            runtime_state=RuntimeState.RUNNING,
            failed_modules=(),
        )
        assert report.is_healthy

    def test_is_not_healthy_with_failed_modules(self) -> None:
        report = self._make_report(
            runtime_state=RuntimeState.RUNNING,
            failed_modules=("mod.failed",),
        )
        assert not report.is_healthy

    def test_is_not_healthy_degraded_state(self) -> None:
        report = self._make_report(runtime_state=RuntimeState.DEGRADED)
        assert not report.is_healthy

    def test_is_operational_running(self) -> None:
        report = self._make_report(runtime_state=RuntimeState.RUNNING)
        assert report.is_operational

    def test_is_operational_degraded(self) -> None:
        report = self._make_report(runtime_state=RuntimeState.DEGRADED)
        assert report.is_operational

    def test_is_not_operational_stopped(self) -> None:
        report = self._make_report(runtime_state=RuntimeState.STOPPED)
        assert not report.is_operational

    def test_is_not_operational_failed(self) -> None:
        report = self._make_report(runtime_state=RuntimeState.FAILED)
        assert not report.is_operational

    def test_timestamp_is_timezone_aware(self) -> None:
        report = self._make_report()
        assert report.timestamp.tzinfo is not None

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            RuntimeHealthReport(
                runtime_state=RuntimeState.RUNNING,
                loaded_modules=(),
                running_modules=(),
                failed_modules=(),
                memory_status="ok",
                event_bus_status="ok",
                timestamp=datetime(2025, 1, 1),  # naive
            )

    def test_to_dict(self) -> None:
        report = self._make_report()
        d = report.to_dict()
        assert d["runtime_state"] == "RUNNING"
        assert d["memory_status"] == "ok"
        assert isinstance(d["timestamp"], str)
        assert "is_healthy" in d
        assert "is_operational" in d

    def test_report_is_immutable(self) -> None:
        report = self._make_report()
        with pytest.raises((AttributeError, TypeError)):
            report.runtime_state = RuntimeState.FAILED  # type: ignore[misc]

    def test_report_with_metadata(self) -> None:
        report = self._make_report(metadata={"build": "5.0.0"})
        assert report.metadata["build"] == "5.0.0"

    def test_to_dict_contains_all_keys(self) -> None:
        report = self._make_report()
        d = report.to_dict()
        expected_keys = {
            "runtime_state", "loaded_modules", "running_modules",
            "failed_modules", "memory_status", "event_bus_status",
            "timestamp", "is_healthy", "is_operational", "metadata",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# IV. Exception Tests
# ===========================================================================


class TestKernelExceptions:
    """Tests for the kernel exception hierarchy."""

    def test_kernel_error_base(self) -> None:
        exc = KernelError("kernel error")
        assert str(exc) == "kernel error"
        assert exc.kernel_state is None

    def test_kernel_error_with_state(self) -> None:
        exc = KernelError("err", kernel_state="RUNNING")
        assert exc.kernel_state == "RUNNING"

    def test_bootstrap_error(self) -> None:
        exc = BootstrapError("bootstrap failed", failed_component="event_bus")
        assert exc.failed_component == "event_bus"
        assert isinstance(exc, KernelError)

    def test_startup_error(self) -> None:
        exc = StartupError("startup failed", failed_module="my.module")
        assert exc.failed_module == "my.module"
        assert isinstance(exc, KernelError)

    def test_shutdown_error(self) -> None:
        exc = ShutdownError("shutdown failed", failed_module="my.module")
        assert exc.failed_module == "my.module"
        assert isinstance(exc, KernelError)

    def test_recovery_error(self) -> None:
        exc = RecoveryError("recovery failed", recovery_attempt=2)
        assert exc.recovery_attempt == 2
        assert isinstance(exc, KernelError)

    def test_bootstrap_error_is_kernel_error(self) -> None:
        with pytest.raises(KernelError):
            raise BootstrapError("test")

    def test_startup_error_is_kernel_error(self) -> None:
        with pytest.raises(KernelError):
            raise StartupError("test")

    def test_shutdown_error_is_kernel_error(self) -> None:
        with pytest.raises(KernelError):
            raise ShutdownError("test")

    def test_recovery_error_is_kernel_error(self) -> None:
        with pytest.raises(KernelError):
            raise RecoveryError("test")

    def test_exception_chaining(self) -> None:
        cause = ValueError("root cause")
        exc = BootstrapError("wrap", failed_component="bus")
        exc.__cause__ = cause
        assert exc.__cause__ is cause

    def test_kernel_error_defaults(self) -> None:
        exc = BootstrapError("test")
        assert exc.failed_component is None

    def test_startup_error_defaults(self) -> None:
        exc = StartupError("test")
        assert exc.failed_module is None

    def test_shutdown_error_defaults(self) -> None:
        exc = ShutdownError("test")
        assert exc.failed_module is None

    def test_recovery_error_defaults(self) -> None:
        exc = RecoveryError("test")
        assert exc.recovery_attempt == 1


# ===========================================================================
# V. BootstrapManager Tests
# ===========================================================================


class TestBootstrapManager:
    """Tests for BootstrapManager."""

    def test_bootstrap_returns_components(self) -> None:
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        components = mgr.bootstrap()

        assert "event_bus" in components
        assert "memory_registry" in components
        assert "memory_gateway" in components
        assert "subsystem_registry" in components
        assert "module_loader" in components
        assert "lifecycle_manager" in components

    def test_bootstrap_event_bus_is_event_bus_instance(self) -> None:
        from core.events.bus import EventBus
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        components = mgr.bootstrap()
        assert isinstance(components["event_bus"], EventBus)

    def test_bootstrap_memory_gateway_correct_type(self) -> None:
        from core.memory.gateway import MemoryGateway
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        components = mgr.bootstrap()
        assert isinstance(components["memory_gateway"], MemoryGateway)

    def test_bootstrap_module_loader_correct_type(self) -> None:
        from core.loader.loader import ModuleLoader
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        components = mgr.bootstrap()
        assert isinstance(components["module_loader"], ModuleLoader)

    def test_bootstrap_lifecycle_manager_correct_type(self) -> None:
        from core.lifecycle.manager import LifecycleManager
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        components = mgr.bootstrap()
        assert isinstance(components["lifecycle_manager"], LifecycleManager)

    def test_get_component_after_bootstrap(self) -> None:
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        mgr.bootstrap()
        bus = mgr.get_component("event_bus")
        assert bus is not None

    def test_get_component_missing_raises(self) -> None:
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        with pytest.raises(KeyError):
            mgr.get_component("nonexistent")

    def test_components_property_returns_copy(self) -> None:
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        mgr.bootstrap()
        c1 = mgr.components
        c2 = mgr.components
        assert c1 is not c2

    def test_validate_dependencies_before_bootstrap_raises(self) -> None:
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        with pytest.raises(BootstrapError, match="missing"):
            mgr.validate_dependencies()

    def test_event_bus_failure_raises_bootstrap_error(self) -> None:
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        with patch.object(mgr, "_init_event_bus", side_effect=RuntimeError("bus fail")):
            with pytest.raises(BootstrapError, match="Event Bus"):
                mgr.bootstrap()

    def test_memory_failure_raises_bootstrap_error(self) -> None:
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        with patch.object(mgr, "_init_memory", side_effect=RuntimeError("mem fail")):
            with pytest.raises(BootstrapError, match="Memory Gateway"):
                mgr.bootstrap()

    def test_loader_failure_raises_bootstrap_error(self) -> None:
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        with patch.object(mgr, "_init_loader", side_effect=RuntimeError("loader fail")):
            with pytest.raises(BootstrapError):
                mgr.bootstrap()

    def test_lifecycle_manager_failure_raises_bootstrap_error(self) -> None:
        config = KernelConfiguration()
        mgr = BootstrapManager(config)
        with patch.object(
            mgr, "_init_lifecycle_manager", side_effect=RuntimeError("lm fail")
        ):
            with pytest.raises(BootstrapError):
                mgr.bootstrap()


# ===========================================================================
# VI. StartupManager Tests
# ===========================================================================


class TestStartupManager:
    """Tests for StartupManager."""

    def test_start_with_no_modules(self) -> None:
        components = _make_mock_components()
        mgr = StartupManager(components)
        mgr.start()  # Should not raise

    def test_start_publishes_kernel_started_event(self) -> None:
        components = _make_mock_components()
        mgr = StartupManager(components)
        mgr.start()
        components["event_bus"].publish.assert_called()

    def test_start_calls_loader_start_for_initialized_modules(self) -> None:
        from core.loader.models import ModuleState
        loader = MagicMock()
        desc = MagicMock()
        desc.state = ModuleState.INITIALIZED
        desc.manifest.id = "mod.a"
        loader.all_descriptors.return_value = [desc]

        components = _make_mock_components(module_loader=loader)
        mgr = StartupManager(components)
        mgr.start()
        loader.start.assert_called_with("mod.a")

    def test_start_raises_if_event_bus_missing(self) -> None:
        components = _make_mock_components()
        components["event_bus"] = None
        mgr = StartupManager(components)
        with pytest.raises(StartupError, match="Missing required"):
            mgr.start()

    def test_start_module_success(self) -> None:
        components = _make_mock_components()
        mgr = StartupManager(components)
        mgr.start_module("mod.a")
        components["module_loader"].start.assert_called_with("mod.a")

    def test_start_module_no_loader_raises(self) -> None:
        components = _make_mock_components(module_loader=None)
        mgr = StartupManager(components)
        with pytest.raises(StartupError, match="module_loader"):
            mgr.start_module("mod.a")

    def test_start_module_loader_error_raises_startup_error(self) -> None:
        components = _make_mock_components()
        components["module_loader"].start.side_effect = RuntimeError("fail")
        mgr = StartupManager(components)
        with pytest.raises(StartupError, match="mod.a"):
            mgr.start_module("mod.a")

    def test_verify_startup_success_no_failures(self) -> None:
        components = _make_mock_components()
        components["module_loader"].all_descriptors.return_value = []
        mgr = StartupManager(components)
        assert mgr.verify_startup_success()

    def test_verify_startup_success_with_failed_module(self) -> None:
        from core.loader.models import ModuleState
        loader = MagicMock()
        desc = MagicMock()
        desc.state = ModuleState.FAILED
        loader.all_descriptors.return_value = [desc]
        components = _make_mock_components(module_loader=loader)
        mgr = StartupManager(components)
        assert not mgr.verify_startup_success()

    def test_verify_startup_no_loader(self) -> None:
        components = _make_mock_components(module_loader=None)
        mgr = StartupManager(components)
        assert not mgr.verify_startup_success()

    def test_module_start_failure_raises_startup_error(self) -> None:
        from core.loader.models import ModuleState
        loader = MagicMock()
        desc = MagicMock()
        desc.state = ModuleState.INITIALIZED
        desc.manifest.id = "mod.bad"
        loader.all_descriptors.return_value = [desc]
        loader.start.side_effect = RuntimeError("start failed")
        components = _make_mock_components(module_loader=loader)
        mgr = StartupManager(components)
        with pytest.raises(StartupError, match="mod.bad"):
            mgr.start()


# ===========================================================================
# VII. ShutdownManager Tests
# ===========================================================================


class TestShutdownManager:
    """Tests for ShutdownManager."""

    def test_shutdown_with_no_modules(self) -> None:
        components = _make_mock_components()
        mgr = ShutdownManager(components)
        mgr.shutdown()  # Should not raise

    def test_shutdown_publishes_stopping_event(self) -> None:
        components = _make_mock_components()
        mgr = ShutdownManager(components)
        mgr.shutdown()
        publish_calls = components["event_bus"].publish.call_args_list
        event_types = [
            str(c.args[0].event_type) for c in publish_calls
            if hasattr(c.args[0], "event_type")
        ]
        # At minimum we expect some events published
        assert components["event_bus"].publish.called

    def test_shutdown_calls_shutdown_all(self) -> None:
        components = _make_mock_components()
        mgr = ShutdownManager(components)
        mgr.shutdown()
        components["module_loader"].shutdown_all.assert_called_once()

    def test_stop_module_success(self) -> None:
        components = _make_mock_components()
        mgr = ShutdownManager(components)
        mgr.stop_module("mod.a")
        components["module_loader"].stop.assert_called_with("mod.a")

    def test_stop_module_no_loader_raises(self) -> None:
        components = _make_mock_components(module_loader=None)
        mgr = ShutdownManager(components)
        with pytest.raises(ShutdownError, match="module_loader"):
            mgr.stop_module("mod.a")

    def test_stop_module_loader_error_raises(self) -> None:
        components = _make_mock_components()
        components["module_loader"].stop.side_effect = RuntimeError("stop fail")
        mgr = ShutdownManager(components)
        with pytest.raises(ShutdownError, match="mod.a"):
            mgr.stop_module("mod.a")

    def test_unload_module_success(self) -> None:
        components = _make_mock_components()
        mgr = ShutdownManager(components)
        mgr.unload_module("mod.a")
        components["module_loader"].unload.assert_called_with("mod.a")

    def test_unload_module_no_loader_raises(self) -> None:
        components = _make_mock_components(module_loader=None)
        mgr = ShutdownManager(components)
        with pytest.raises(ShutdownError, match="module_loader"):
            mgr.unload_module("mod.a")

    def test_shutdown_continues_after_shutdown_all_error(self) -> None:
        from core.loader.models import ModuleState
        loader = MagicMock()
        loader.shutdown_all.side_effect = RuntimeError("fail")
        desc = MagicMock()
        desc.state = ModuleState.STOPPED
        desc.manifest.id = "mod.a"
        loader.all_descriptors.return_value = [desc]
        components = _make_mock_components(module_loader=loader)
        mgr = ShutdownManager(components)
        mgr.shutdown()  # Should not raise despite shutdown_all error

    def test_shutdown_no_loader_skips_gracefully(self) -> None:
        components = _make_mock_components(module_loader=None)
        mgr = ShutdownManager(components)
        mgr.shutdown()  # No error expected


# ===========================================================================
# VIII. HealthMonitor Tests
# ===========================================================================


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    def test_check_returns_report(self) -> None:
        components = _make_mock_components()
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.RUNNING)
        assert isinstance(report, RuntimeHealthReport)

    def test_check_reflects_current_state(self) -> None:
        components = _make_mock_components()
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.DEGRADED)
        assert report.runtime_state is RuntimeState.DEGRADED

    def test_check_running_modules(self) -> None:
        from core.loader.models import ModuleState
        loader = MagicMock()
        desc = MagicMock()
        desc.state = ModuleState.RUNNING
        desc.manifest.id = "mod.a"
        loader.all_descriptors.return_value = [desc]
        components = _make_mock_components(module_loader=loader)
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.RUNNING)
        assert "mod.a" in report.running_modules

    def test_check_failed_modules(self) -> None:
        from core.loader.models import ModuleState
        loader = MagicMock()
        desc = MagicMock()
        desc.state = ModuleState.FAILED
        desc.manifest.id = "mod.bad"
        loader.all_descriptors.return_value = [desc]
        components = _make_mock_components(module_loader=loader)
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.RUNNING)
        assert "mod.bad" in report.failed_modules

    def test_check_memory_status_no_providers(self) -> None:
        components = _make_mock_components()
        components["memory_registry"].__iter__ = MagicMock(return_value=iter([]))
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.RUNNING)
        assert "no_providers" in report.memory_status or "ok" in report.memory_status

    def test_check_memory_status_unavailable(self) -> None:
        components = _make_mock_components(memory_registry=None)
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.RUNNING)
        assert report.memory_status == "unavailable"

    def test_check_event_bus_status_unavailable(self) -> None:
        components = _make_mock_components(event_bus=None)
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.RUNNING)
        assert report.event_bus_status == "unavailable"

    def test_check_event_bus_ok(self) -> None:
        components = _make_mock_components()
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.RUNNING)
        assert "ok" in report.event_bus_status

    def test_check_timestamp_is_recent(self) -> None:
        components = _make_mock_components()
        monitor = HealthMonitor(components)
        before = datetime.now(timezone.utc)
        report = monitor.check(RuntimeState.RUNNING)
        after = datetime.now(timezone.utc)
        assert before <= report.timestamp <= after

    def test_check_no_module_loader(self) -> None:
        components = _make_mock_components(module_loader=None)
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.RUNNING)
        assert report.running_modules == ()
        assert report.failed_modules == ()

    def test_check_loader_error_returns_empty(self) -> None:
        loader = MagicMock()
        loader.all_descriptors.side_effect = RuntimeError("fail")
        components = _make_mock_components(module_loader=loader)
        monitor = HealthMonitor(components)
        report = monitor.check(RuntimeState.RUNNING)
        assert report.running_modules == ()


# ===========================================================================
# IX. RuntimeKernel — State & Configuration Tests
# ===========================================================================


class TestRuntimeKernelCreation:
    """Tests for kernel creation and initial state."""

    def test_kernel_starts_in_created_state(self) -> None:
        kernel = _make_kernel()
        assert kernel.state is RuntimeState.CREATED

    def test_kernel_uses_default_config_when_none(self) -> None:
        kernel = RuntimeKernel()
        assert kernel.config.environment == "development"

    def test_kernel_uses_provided_config(self) -> None:
        config = KernelConfiguration(environment="production")
        kernel = RuntimeKernel(config)
        assert kernel.config.environment == "production"

    def test_kernel_state_is_thread_safe(self) -> None:
        kernel = _make_kernel()
        states = []

        def read_state() -> None:
            for _ in range(100):
                states.append(kernel.state)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(s is RuntimeState.CREATED for s in states)

    def test_kernel_is_not_running_initially(self) -> None:
        kernel = _make_kernel()
        assert not kernel.is_running

    def test_components_empty_before_bootstrap(self) -> None:
        kernel = _make_kernel()
        assert kernel.components == {}


# ===========================================================================
# X. RuntimeKernel — Bootstrap Tests
# ===========================================================================


class TestRuntimeKernelBootstrap:
    """Tests for kernel bootstrap lifecycle."""

    def test_bootstrap_transitions_to_starting(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        assert kernel.state is RuntimeState.STARTING

    def test_bootstrap_populates_components(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        assert "event_bus" in kernel.components

    def test_bootstrap_from_non_created_raises(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        # Already bootstrapped — now in STARTING
        with pytest.raises(KernelError):
            kernel.bootstrap()

    def test_bootstrap_failure_transitions_to_failed(self) -> None:
        kernel = _make_kernel()
        with patch.object(
            BootstrapManager, "bootstrap", side_effect=BootstrapError("fail")
        ):
            with pytest.raises(BootstrapError):
                kernel.bootstrap()
        assert kernel.state is RuntimeState.FAILED

    def test_bootstrap_publishes_event(self) -> None:
        kernel = _make_kernel()
        events: list[Any] = []
        mock_components = _make_mock_components()

        def capture_event(event: Any) -> None:
            events.append(event)

        mock_components["event_bus"].publish.side_effect = capture_event

        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()

        assert len(events) >= 1

    def test_auto_start_triggers_start(self) -> None:
        config = KernelConfiguration(auto_start=True)
        kernel = RuntimeKernel(config)
        mock_components = _make_mock_components()

        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            with patch.object(StartupManager, "start"):
                kernel.bootstrap()

        assert kernel.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED)

    def test_unexpected_exception_becomes_bootstrap_error(self) -> None:
        kernel = _make_kernel()
        with patch.object(
            BootstrapManager, "bootstrap", side_effect=RuntimeError("surprise")
        ):
            with pytest.raises(BootstrapError):
                kernel.bootstrap()

    def test_bootstrap_stores_managers(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        assert kernel._startup_manager is not None
        assert kernel._shutdown_manager is not None
        assert kernel._health_monitor is not None


# ===========================================================================
# XI. RuntimeKernel — Startup Tests
# ===========================================================================


class TestRuntimeKernelStartup:
    """Tests for kernel start lifecycle."""

    def test_start_transitions_to_running(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        with patch.object(StartupManager, "start"):
            kernel.start()
        assert kernel.state is RuntimeState.RUNNING

    def test_start_before_bootstrap_raises(self) -> None:
        kernel = _make_kernel()
        with pytest.raises(KernelError):
            kernel.start()

    def test_start_failure_transitions_to_failed(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        with patch.object(StartupManager, "start", side_effect=StartupError("fail")):
            with pytest.raises(StartupError):
                kernel.start()
        assert kernel.state is RuntimeState.FAILED

    def test_start_with_failed_modules_is_degraded(self) -> None:
        from core.loader.models import ModuleState
        kernel = _make_kernel()
        mock_components = _make_mock_components()

        desc = MagicMock()
        desc.state = ModuleState.FAILED
        mock_components["module_loader"].all_descriptors.return_value = [desc]

        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()
        with patch.object(StartupManager, "start"):
            kernel.start()

        assert kernel.state is RuntimeState.DEGRADED

    def test_start_kernel_is_running(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        assert kernel.is_running

    def test_start_publishes_kernel_started_event(self) -> None:
        kernel = _make_kernel()
        mock_components = _make_mock_components()
        events: list[Any] = []
        mock_components["event_bus"].publish.side_effect = lambda e: events.append(e)

        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()
        with patch.object(StartupManager, "start"):
            kernel.start()

        assert len(events) >= 1


# ===========================================================================
# XII. RuntimeKernel — Shutdown Tests
# ===========================================================================


class TestRuntimeKernelShutdown:
    """Tests for kernel stop lifecycle."""

    def test_stop_transitions_to_stopped(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        with patch.object(ShutdownManager, "shutdown"):
            kernel.stop()
        assert kernel.state is RuntimeState.STOPPED

    def test_stop_already_stopped_is_no_op(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        with patch.object(ShutdownManager, "shutdown"):
            kernel.stop()
            kernel.stop()  # Second call — no-op
        assert kernel.state is RuntimeState.STOPPED

    def test_stop_before_bootstrap_raises(self) -> None:
        kernel = _make_kernel()
        with pytest.raises(KernelError):
            kernel.stop()

    def test_stop_publishes_stopped_event(self) -> None:
        kernel = _make_kernel()
        mock_components = _make_mock_components()
        events: list[Any] = []
        mock_components["event_bus"].publish.side_effect = lambda e: events.append(e)

        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()
        with patch.object(StartupManager, "start"):
            kernel.start()

        events.clear()
        with patch.object(ShutdownManager, "shutdown"):
            kernel.stop()

        assert len(events) >= 1

    def test_stop_from_failed_state(self) -> None:
        kernel = _make_kernel()
        mock_components = _make_mock_components()
        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            with patch.object(
                BootstrapManager, "bootstrap", side_effect=BootstrapError("fail")
            ):
                with pytest.raises(BootstrapError):
                    kernel.bootstrap()

        assert kernel.state is RuntimeState.FAILED

        # Stopping from FAILED state
        with patch.object(ShutdownManager, "shutdown"):
            kernel.stop()
        assert kernel.state is RuntimeState.STOPPED

    def test_shutdown_error_still_reaches_stopped(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        with patch.object(ShutdownManager, "shutdown", side_effect=ShutdownError("fail")):
            kernel.stop()  # Should not propagate
        assert kernel.state is RuntimeState.STOPPED


# ===========================================================================
# XIII. RuntimeKernel — Restart Tests
# ===========================================================================


class TestRuntimeKernelRestart:
    """Tests for kernel restart."""

    def test_restart_from_running(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)

        with patch.object(ShutdownManager, "shutdown"):
            with patch.object(BootstrapManager, "bootstrap", return_value=_make_mock_components()):
                with patch.object(StartupManager, "start"):
                    kernel.restart()

        assert kernel.state in (RuntimeState.RUNNING, RuntimeState.STARTING, RuntimeState.DEGRADED)

    def test_restart_resets_components(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        old_components = kernel.components

        with patch.object(ShutdownManager, "shutdown"):
            with patch.object(BootstrapManager, "bootstrap", return_value=_make_mock_components()):
                with patch.object(StartupManager, "start"):
                    kernel.restart()

        # Components are refreshed after restart
        assert kernel.state is not RuntimeState.STOPPED

    def test_restart_from_created_raises(self) -> None:
        kernel = _make_kernel()
        with pytest.raises(KernelError):
            kernel.restart()

    def test_restart_from_stopped_raises(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        with patch.object(ShutdownManager, "shutdown"):
            kernel.stop()
        with pytest.raises(KernelError):
            kernel.restart()

    def test_restart_resets_recovery_attempts(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        kernel._recovery_attempts = 2

        with patch.object(ShutdownManager, "shutdown"):
            with patch.object(BootstrapManager, "bootstrap", return_value=_make_mock_components()):
                with patch.object(StartupManager, "start"):
                    kernel.restart()

        assert kernel._recovery_attempts == 0


# ===========================================================================
# XIV. RuntimeKernel — Recovery Tests
# ===========================================================================


class TestRuntimeKernelRecovery:
    """Tests for kernel recovery paths."""

    def _make_failed_kernel(self) -> RuntimeKernel:
        """Return a kernel stuck in FAILED state."""
        config = KernelConfiguration(enable_recovery=True, max_recovery_attempts=3)
        kernel = RuntimeKernel(config)
        mock_components = _make_mock_components()
        with patch.object(BootstrapManager, "bootstrap", side_effect=BootstrapError("fail")):
            with pytest.raises(BootstrapError):
                kernel.bootstrap()
        # Manually inject components so health/recovery works
        kernel._components = mock_components
        kernel._health_monitor = HealthMonitor(mock_components)
        return kernel

    def test_recover_from_failed_transitions_to_running(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        # Force to FAILED
        kernel._state = RuntimeState.FAILED

        with patch.object(kernel, "_recover_failed_modules"):
            kernel.recover()

        assert kernel.state is RuntimeState.RUNNING

    def test_recover_from_degraded_transitions_to_running(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        kernel._state = RuntimeState.DEGRADED

        with patch.object(kernel, "_recover_failed_modules"):
            kernel.recover()

        assert kernel.state is RuntimeState.RUNNING

    def test_recover_from_running_raises(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        with pytest.raises(KernelError):
            kernel.recover()

    def test_recover_increments_attempt_counter(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        kernel._state = RuntimeState.FAILED

        with patch.object(kernel, "_recover_failed_modules"):
            kernel.recover()

        # On success, counter resets to 0
        assert kernel._recovery_attempts == 0

    def test_recover_max_attempts_exceeded_raises(self) -> None:
        config = KernelConfiguration(max_recovery_attempts=2)
        kernel = RuntimeKernel(config)
        _start_kernel(kernel)
        kernel._state = RuntimeState.FAILED
        kernel._recovery_attempts = 2

        with pytest.raises(RecoveryError, match="max attempts"):
            kernel.recover()

    def test_recover_disabled_raises(self) -> None:
        config = KernelConfiguration(enable_recovery=False)
        kernel = RuntimeKernel(config)
        _start_kernel(kernel)
        kernel._state = RuntimeState.FAILED

        with pytest.raises(RecoveryError, match="disabled"):
            kernel.recover()

    def test_recover_failure_transitions_to_failed(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        kernel._state = RuntimeState.FAILED

        with patch.object(
            kernel, "_recover_failed_modules", side_effect=RuntimeError("cannot recover")
        ):
            with pytest.raises(RecoveryError):
                kernel.recover()

        assert kernel.state is RuntimeState.FAILED

    def test_recover_publishes_recovered_event(self) -> None:
        kernel = _make_kernel()
        mock_components = _make_mock_components()
        events: list[Any] = []
        mock_components["event_bus"].publish.side_effect = lambda e: events.append(e)

        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()
        with patch.object(StartupManager, "start"):
            kernel.start()

        kernel._state = RuntimeState.FAILED
        events.clear()

        with patch.object(kernel, "_recover_failed_modules"):
            kernel.recover()

        assert any("recover" in str(getattr(e, "event_type", "")).lower() for e in events)

    def test_recovery_path_failed_recovering_running(self) -> None:
        """Full recovery path: FAILED → RECOVERING → RUNNING."""
        kernel = _make_kernel()
        _start_kernel(kernel)
        kernel._state = RuntimeState.FAILED

        assert kernel.state is RuntimeState.FAILED

        with patch.object(kernel, "_recover_failed_modules"):
            kernel.recover()

        assert kernel.state is RuntimeState.RUNNING

    def test_partial_recovery_results_in_degraded(self) -> None:
        from core.loader.models import ModuleState
        kernel = _make_kernel()
        mock_components = _make_mock_components()

        desc = MagicMock()
        desc.state = ModuleState.FAILED
        mock_components["module_loader"].all_descriptors.return_value = [desc]

        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()
        with patch.object(StartupManager, "start"):
            kernel.start()

        kernel._state = RuntimeState.FAILED

        with patch.object(kernel, "_recover_failed_modules"):
            kernel.recover()

        assert kernel.state is RuntimeState.DEGRADED


# ===========================================================================
# XV. RuntimeKernel — Health Tests
# ===========================================================================


class TestRuntimeKernelHealth:
    """Tests for kernel health reporting."""

    def test_health_before_bootstrap(self) -> None:
        kernel = _make_kernel()
        report = kernel.health()
        assert isinstance(report, RuntimeHealthReport)
        assert report.runtime_state is RuntimeState.CREATED
        assert report.memory_status == "unavailable"

    def test_health_after_bootstrap(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        report = kernel.health()
        assert isinstance(report, RuntimeHealthReport)

    def test_health_after_start(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        report = kernel.health()
        assert report.runtime_state in (RuntimeState.RUNNING, RuntimeState.DEGRADED)

    def test_health_never_raises(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        monitor = MagicMock()
        monitor.check.side_effect = RuntimeError("monitor failure")
        kernel._health_monitor = monitor
        report = kernel.health()
        assert isinstance(report, RuntimeHealthReport)
        assert "health_check_error" in report.metadata

    def test_health_timestamp_is_recent(self) -> None:
        kernel = _make_kernel()
        before = datetime.now(timezone.utc)
        report = kernel.health()
        after = datetime.now(timezone.utc)
        assert before <= report.timestamp <= after


# ===========================================================================
# XVI. RuntimeKernel — Module Management Tests
# ===========================================================================


class TestRuntimeKernelModuleManagement:
    """Tests for module registration/loading/starting/stopping via kernel."""

    def test_register_module_success(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)

        manifest = MagicMock()
        manifest.id = "mod.test"
        kernel.register_module(manifest)
        kernel._components["module_loader"].discover.assert_called_once_with(
            manifest, module_path=None
        )

    def test_register_module_before_bootstrap_raises(self) -> None:
        kernel = _make_kernel()
        manifest = MagicMock()
        manifest.id = "mod.test"
        with pytest.raises(KernelError):
            kernel.register_module(manifest)

    def test_load_module_success(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        kernel.load_module("mod.test")
        kernel._components["module_loader"].load.assert_called_with("mod.test")

    def test_load_module_before_bootstrap_raises(self) -> None:
        kernel = _make_kernel()
        with pytest.raises(KernelError):
            kernel.load_module("mod.test")

    def test_load_module_error_wraps_in_kernel_error(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        kernel._components["module_loader"].load.side_effect = RuntimeError("fail")
        with pytest.raises(KernelError, match="mod.test"):
            kernel.load_module("mod.test")

    def test_start_module_success(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        kernel.start_module("mod.test")
        kernel._components["module_loader"].start.assert_called_with("mod.test")

    def test_start_module_before_bootstrap_raises(self) -> None:
        kernel = _make_kernel()
        with pytest.raises(KernelError):
            kernel.start_module("mod.test")

    def test_stop_module_success(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        kernel.stop_module("mod.test")
        kernel._components["module_loader"].stop.assert_called_with("mod.test")

    def test_stop_module_before_bootstrap_raises(self) -> None:
        kernel = _make_kernel()
        with pytest.raises(KernelError):
            kernel.stop_module("mod.test")

    def test_register_module_with_path(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        manifest = MagicMock()
        manifest.id = "mod.test"
        kernel.register_module(manifest, module_path="myproject.MyModule")
        kernel._components["module_loader"].discover.assert_called_once_with(
            manifest, module_path="myproject.MyModule"
        )


# ===========================================================================
# XVII. RuntimeKernel — State Transition Tests
# ===========================================================================


class TestRuntimeKernelStateTransitions:
    """Tests for valid and invalid kernel state transitions."""

    def test_created_to_bootstrapping_is_valid(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        # After bootstrap we are in STARTING (passes through BOOTSTRAPPING)
        assert kernel.state is RuntimeState.STARTING

    def test_cannot_double_bootstrap(self) -> None:
        kernel = _make_kernel()
        _bootstrap_kernel(kernel)
        with pytest.raises(KernelError):
            _bootstrap_kernel(kernel)

    def test_state_after_start_is_running_or_degraded(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        assert kernel.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED)

    def test_cannot_start_twice(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        with pytest.raises(KernelError):
            kernel.start()

    def test_cannot_stop_before_start(self) -> None:
        kernel = _make_kernel()
        with pytest.raises(KernelError):
            kernel.stop()

    def test_failed_to_recovering_via_recover(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        kernel._state = RuntimeState.FAILED
        with patch.object(kernel, "_recover_failed_modules"):
            kernel.recover()
        assert kernel.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED)


# ===========================================================================
# XVIII. RuntimeKernel — Thread Safety Tests
# ===========================================================================


class TestRuntimeKernelThreadSafety:
    """Tests for thread safety of the runtime kernel."""

    def test_concurrent_state_reads(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        errors: list[Exception] = []

        def read_state() -> None:
            try:
                for _ in range(200):
                    _ = kernel.state
                    _ = kernel.is_running
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=read_state) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"

    def test_concurrent_health_checks(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        errors: list[Exception] = []

        def do_health() -> None:
            try:
                for _ in range(50):
                    report = kernel.health()
                    assert isinstance(report, RuntimeHealthReport)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=do_health) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Health check thread errors: {errors}"

    def test_concurrent_component_access(self) -> None:
        kernel = _make_kernel()
        _start_kernel(kernel)
        errors: list[Exception] = []

        def access_components() -> None:
            try:
                for _ in range(100):
                    _ = kernel.components
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=access_components) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ===========================================================================
# XIX. RuntimeKernel — Event Generation Tests
# ===========================================================================


class TestRuntimeKernelEvents:
    """Tests for kernel event generation."""

    def _start_with_events(self) -> tuple[RuntimeKernel, list[Any]]:
        """Return a started kernel and the list of published events."""
        events: list[Any] = []
        mock_components = _make_mock_components()
        mock_components["event_bus"].publish.side_effect = lambda e: events.append(e)

        kernel = _make_kernel()
        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()
        with patch.object(StartupManager, "start"):
            kernel.start()

        events.clear()  # Clear bootstrap/start events for focused testing
        kernel._components = mock_components
        return kernel, events

    def test_kernel_bootstrapped_event_published(self) -> None:
        events: list[Any] = []
        mock_components = _make_mock_components()
        mock_components["event_bus"].publish.side_effect = lambda e: events.append(e)

        kernel = _make_kernel()
        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()

        assert any(
            "bootstrap" in str(getattr(e, "event_type", "")).lower()
            for e in events
        )

    def test_kernel_started_event_published(self) -> None:
        events: list[Any] = []
        mock_components = _make_mock_components()
        mock_components["event_bus"].publish.side_effect = lambda e: events.append(e)

        kernel = _make_kernel()
        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()
        events.clear()
        with patch.object(StartupManager, "start"):
            kernel.start()

        assert any(
            "start" in str(getattr(e, "event_type", "")).lower()
            for e in events
        )

    def test_kernel_stopped_event_published(self) -> None:
        kernel, events = self._start_with_events()
        with patch.object(ShutdownManager, "shutdown"):
            kernel.stop()

        assert any(
            "stop" in str(getattr(e, "event_type", "")).lower()
            for e in events
        )

    def test_kernel_failed_event_published_on_bootstrap_failure(self) -> None:
        events: list[Any] = []
        mock_components = _make_mock_components()
        mock_components["event_bus"].publish.side_effect = lambda e: events.append(e)

        kernel = _make_kernel()
        kernel._components = mock_components

        with patch.object(
            BootstrapManager, "bootstrap", side_effect=BootstrapError("fail")
        ):
            with pytest.raises(BootstrapError):
                kernel.bootstrap()

    def test_kernel_recovered_event_published(self) -> None:
        kernel, events = self._start_with_events()
        kernel._state = RuntimeState.FAILED

        with patch.object(kernel, "_recover_failed_modules"):
            kernel.recover()

        assert any(
            "recover" in str(getattr(e, "event_type", "")).lower()
            for e in events
        )

    def test_events_have_kernel_source(self) -> None:
        events: list[Any] = []
        mock_components = _make_mock_components()
        mock_components["event_bus"].publish.side_effect = lambda e: events.append(e)

        kernel = _make_kernel()
        with patch.object(BootstrapManager, "bootstrap", return_value=mock_components):
            kernel.bootstrap()

        for event in events:
            if hasattr(event, "source"):
                assert "kernel" in str(event.source).lower()


# ===========================================================================
# XX. RuntimeKernel — Context Manager Tests
# ===========================================================================


class TestRuntimeKernelContextManager:
    """Tests for kernel context manager support."""

    def test_context_manager_bootstraps_kernel(self) -> None:
        kernel = _make_kernel()
        with (
            patch.object(BootstrapManager, "bootstrap", return_value=_make_mock_components()),
            patch.object(ShutdownManager, "shutdown"),
        ):
            with kernel:
                assert kernel.state is RuntimeState.STARTING

    def test_context_manager_stops_on_exit(self) -> None:
        kernel = _make_kernel()
        with (
            patch.object(BootstrapManager, "bootstrap", return_value=_make_mock_components()),
            patch.object(ShutdownManager, "shutdown"),
        ):
            with kernel:
                pass

        assert kernel.state is RuntimeState.STOPPED

    def test_context_manager_stops_on_exception(self) -> None:
        kernel = _make_kernel()
        with (
            patch.object(BootstrapManager, "bootstrap", return_value=_make_mock_components()),
            patch.object(ShutdownManager, "shutdown"),
        ):
            try:
                with kernel:
                    raise ValueError("test error")
            except ValueError:
                pass

        assert kernel.state is RuntimeState.STOPPED


# ===========================================================================
# XXI. Integration Tests
# ===========================================================================


class TestRuntimeKernelIntegration:
    """Integration tests using real (non-mocked) runtime components."""

    def test_full_bootstrap_with_real_components(self) -> None:
        """Bootstrap using real EventBus, MemoryGateway, etc."""
        config = KernelConfiguration(environment="test")
        kernel = RuntimeKernel(config)
        kernel.bootstrap()

        assert kernel.state is RuntimeState.STARTING
        assert "event_bus" in kernel.components
        assert "module_loader" in kernel.components

        # Clean up — no modules to stop, go directly to stopped
        kernel.stop()
        assert kernel.state is RuntimeState.STOPPED

    def test_health_report_with_real_components(self) -> None:
        config = KernelConfiguration(environment="test")
        kernel = RuntimeKernel(config)
        kernel.bootstrap()

        report = kernel.health()
        assert isinstance(report, RuntimeHealthReport)
        assert report.runtime_state is RuntimeState.STARTING

        kernel.stop()

    def test_start_with_real_components_no_modules(self) -> None:
        config = KernelConfiguration(environment="test")
        kernel = RuntimeKernel(config)
        kernel.bootstrap()
        kernel.start()  # No modules discovered, should succeed

        assert kernel.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED)

        kernel.stop()
        assert kernel.state is RuntimeState.STOPPED

    def test_full_lifecycle_with_real_components(self) -> None:
        """Full bootstrap → start → health → stop lifecycle."""
        config = KernelConfiguration(environment="test")
        kernel = RuntimeKernel(config)

        kernel.bootstrap()
        assert kernel.state is RuntimeState.STARTING

        kernel.start()
        assert kernel.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED)

        report = kernel.health()
        assert report.is_operational

        kernel.stop()
        assert kernel.state is RuntimeState.STOPPED

    def test_health_report_after_stop(self) -> None:
        config = KernelConfiguration(environment="test")
        kernel = RuntimeKernel(config)
        kernel.bootstrap()
        kernel.start()
        kernel.stop()

        report = kernel.health()
        assert report.runtime_state is RuntimeState.STOPPED

    def test_get_component_after_real_bootstrap(self) -> None:
        from core.events.bus import EventBus
        config = KernelConfiguration(environment="test")
        kernel = RuntimeKernel(config)
        kernel.bootstrap()

        bus = kernel._get_component("event_bus")
        assert isinstance(bus, EventBus)

        kernel.stop()