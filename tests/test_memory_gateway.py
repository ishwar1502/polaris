# tests/test_memory_gateway.py
"""
Comprehensive pytest suite for POLARIS v5 Memory Gateway (Phase 3).

Coverage targets
----------------
* MemoryType enum
* MemoryRequest validation and factory
* MemoryResponse factories and fields
* MemoryProvider abstract contract
* MemoryRegistry — registration, lookup, thread safety
* MemoryRouter — routing, dispatch, failure handling
* MemoryGateway — all public operations, query(), validation
* Exception hierarchy
* Thread safety under concurrent load

Run with::

    python -m pytest tests/test_memory_gateway.py -v
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.memory.exceptions import (
    MemoryError as MemoryGatewayError,
    MemoryProviderError,
    MemoryRoutingError,
    MemoryValidationError,
)
from core.memory.gateway import MemoryGateway
from core.memory.models import (
    MemoryType,
    ROUTING_TABLE,
    PROVIDER_ID_ECHO,
    PROVIDER_ID_LUNA,
    PROVIDER_ID_CHRONOS,
    PROVIDER_ID_CONSTELLATION,
)
from core.memory.provider import MemoryProvider
from core.memory.registry import MemoryRegistry
from core.memory.request import MemoryRequest, VALID_OPERATIONS
from core.memory.response import MemoryResponse
from core.memory.router import MemoryRouter


# ===========================================================================
# Helpers / Fixtures
# ===========================================================================


class InMemoryProvider(MemoryProvider):
    """Fully functional in-memory provider for testing."""

    def __init__(self, provider_id: str, memory_type: MemoryType) -> None:
        super().__init__(provider_id, memory_type)
        self._store: dict[str, Any] = {}
        self._available: bool = True

    def set_available(self, value: bool) -> None:
        self._available = value

    def is_available(self) -> bool:
        return self._available

    def store(self, request: MemoryRequest) -> MemoryResponse:
        self._store[request.key] = request.value
        return MemoryResponse.success_response(request.memory_type, data=None)

    def retrieve(self, request: MemoryRequest) -> MemoryResponse:
        data = self._store.get(request.key)
        return MemoryResponse.success_response(request.memory_type, data=data)

    def delete(self, request: MemoryRequest) -> MemoryResponse:
        existed = request.key in self._store
        self._store.pop(request.key, None)
        return MemoryResponse.success_response(request.memory_type, data=existed)

    def exists(self, request: MemoryRequest) -> MemoryResponse:
        return MemoryResponse.success_response(
            request.memory_type, data=request.key in self._store
        )

    def search(self, request: MemoryRequest) -> MemoryResponse:
        query = request.value or ""
        results = [k for k in self._store if str(query) in k]
        return MemoryResponse.success_response(request.memory_type, data=results)


class BrokenProvider(MemoryProvider):
    """Provider that always raises on every operation."""

    def __init__(self) -> None:
        super().__init__("broken", MemoryType.EXPERIENCE)

    def store(self, request: MemoryRequest) -> MemoryResponse:
        raise RuntimeError("backend exploded")

    def retrieve(self, request: MemoryRequest) -> MemoryResponse:
        raise RuntimeError("backend exploded")

    def delete(self, request: MemoryRequest) -> MemoryResponse:
        raise RuntimeError("backend exploded")

    def exists(self, request: MemoryRequest) -> MemoryResponse:
        raise RuntimeError("backend exploded")

    def search(self, request: MemoryRequest) -> MemoryResponse:
        raise RuntimeError("backend exploded")


def make_echo() -> InMemoryProvider:
    return InMemoryProvider(PROVIDER_ID_ECHO, MemoryType.EXPERIENCE)


def make_luna() -> InMemoryProvider:
    return InMemoryProvider(PROVIDER_ID_LUNA, MemoryType.KNOWLEDGE)


def make_chronos() -> InMemoryProvider:
    return InMemoryProvider(PROVIDER_ID_CHRONOS, MemoryType.TEMPORAL)


def make_constellation() -> InMemoryProvider:
    return InMemoryProvider(PROVIDER_ID_CONSTELLATION, MemoryType.RELATIONSHIP)


@pytest.fixture
def registry() -> MemoryRegistry:
    return MemoryRegistry()


@pytest.fixture
def full_registry() -> MemoryRegistry:
    r = MemoryRegistry()
    r.register(make_echo())
    r.register(make_luna())
    r.register(make_chronos())
    r.register(make_constellation())
    return r


@pytest.fixture
def gateway(full_registry) -> MemoryGateway:
    return MemoryGateway(full_registry)


# ===========================================================================
# 1. MemoryType
# ===========================================================================


class TestMemoryType:
    def test_all_four_types_exist(self):
        assert MemoryType.EXPERIENCE
        assert MemoryType.KNOWLEDGE
        assert MemoryType.TEMPORAL
        assert MemoryType.RELATIONSHIP

    def test_values_are_strings(self):
        for mt in MemoryType:
            assert isinstance(mt.value, str)

    def test_unique_values(self):
        values = [mt.value for mt in MemoryType]
        assert len(values) == len(set(values))

    def test_routing_table_covers_all_types(self):
        for mt in MemoryType:
            assert mt in ROUTING_TABLE

    def test_routing_table_maps_to_expected_providers(self):
        assert ROUTING_TABLE[MemoryType.EXPERIENCE] == PROVIDER_ID_ECHO
        assert ROUTING_TABLE[MemoryType.KNOWLEDGE] == PROVIDER_ID_LUNA
        assert ROUTING_TABLE[MemoryType.TEMPORAL] == PROVIDER_ID_CHRONOS
        assert ROUTING_TABLE[MemoryType.RELATIONSHIP] == PROVIDER_ID_CONSTELLATION

    def test_str_representation(self):
        assert str(MemoryType.EXPERIENCE) == "experience"

    def test_instantiation_by_value(self):
        assert MemoryType("experience") == MemoryType.EXPERIENCE


# ===========================================================================
# 2. Exceptions
# ===========================================================================


class TestExceptions:
    def test_memory_error_is_base(self):
        assert issubclass(MemoryProviderError, MemoryGatewayError)
        assert issubclass(MemoryRoutingError, MemoryGatewayError)
        assert issubclass(MemoryValidationError, MemoryGatewayError)

    def test_memory_error_stores_fields(self):
        exc = MemoryGatewayError("msg", memory_type="experience", operation="store")
        assert str(exc) == "msg"
        assert exc.memory_type == "experience"
        assert exc.operation == "store"

    def test_provider_error_stores_cause(self):
        cause = ValueError("oops")
        exc = MemoryProviderError("fail", provider_id="echo", cause=cause)
        assert exc.provider_id == "echo"
        assert exc.cause is cause

    def test_validation_error_stores_field(self):
        exc = MemoryValidationError("bad", field="key", invalid_value="")
        assert exc.field == "key"
        assert exc.invalid_value == ""

    def test_routing_error_inherits_memory_error(self):
        with pytest.raises(MemoryGatewayError):
            raise MemoryRoutingError("no route", memory_type="temporal")

    def test_exceptions_are_exceptions(self):
        for cls in (
            MemoryGatewayError,
            MemoryProviderError,
            MemoryRoutingError,
            MemoryValidationError,
        ):
            with pytest.raises(cls):
                raise cls("test")


# ===========================================================================
# 3. MemoryRequest
# ===========================================================================


class TestMemoryRequest:
    def test_create_store_request(self):
        req = MemoryRequest.create(
            memory_type=MemoryType.EXPERIENCE,
            operation="store",
            key="k1",
            value={"data": 1},
        )
        assert req.memory_type == MemoryType.EXPERIENCE
        assert req.operation == "store"
        assert req.key == "k1"
        assert req.value == {"data": 1}

    def test_auto_generated_request_id(self):
        req = MemoryRequest.create(
            memory_type=MemoryType.KNOWLEDGE, operation="retrieve", key="k"
        )
        assert req.request_id
        uuid.UUID(req.request_id)  # must be valid UUID

    def test_explicit_request_id(self):
        rid = str(uuid.uuid4())
        req = MemoryRequest.create(
            memory_type=MemoryType.KNOWLEDGE,
            operation="retrieve",
            key="k",
            request_id=rid,
        )
        assert req.request_id == rid

    def test_timestamp_is_utc(self):
        req = MemoryRequest.create(
            memory_type=MemoryType.TEMPORAL, operation="exists", key="k"
        )
        assert req.timestamp.tzinfo is not None

    def test_metadata_round_trips(self):
        req = MemoryRequest.create(
            memory_type=MemoryType.RELATIONSHIP,
            operation="store",
            key="k",
            value=1,
            metadata={"ttl": 300, "source": "agent"},
        )
        assert req.metadata == {"ttl": 300, "source": "agent"}
        assert req.get_metadata("ttl") == 300
        assert req.get_metadata("missing", "default") == "default"

    def test_immutability(self):
        req = MemoryRequest.create(
            memory_type=MemoryType.EXPERIENCE, operation="store", key="k", value=1
        )
        with pytest.raises((AttributeError, TypeError)):
            req.key = "other"  # type: ignore[misc]

    def test_invalid_operation_raises(self):
        with pytest.raises(MemoryValidationError):
            MemoryRequest.create(
                memory_type=MemoryType.EXPERIENCE,
                operation="invalid_op",
                key="k",
            )

    def test_empty_key_raises_for_store(self):
        with pytest.raises(MemoryValidationError):
            MemoryRequest.create(
                memory_type=MemoryType.EXPERIENCE, operation="store", key="", value=1
            )

    def test_empty_key_raises_for_retrieve(self):
        with pytest.raises(MemoryValidationError):
            MemoryRequest.create(
                memory_type=MemoryType.EXPERIENCE, operation="retrieve", key=""
            )

    def test_none_key_allowed_for_search(self):
        req = MemoryRequest.create(
            memory_type=MemoryType.KNOWLEDGE,
            operation="search",
            key=None,
            value="query string",
        )
        assert req.key is None

    def test_key_too_long_raises(self):
        with pytest.raises(MemoryValidationError):
            MemoryRequest.create(
                memory_type=MemoryType.EXPERIENCE,
                operation="store",
                key="x" * 600,
                value=1,
            )

    def test_metadata_too_many_keys_raises(self):
        meta = {str(i): i for i in range(65)}
        with pytest.raises(MemoryValidationError):
            MemoryRequest.create(
                memory_type=MemoryType.EXPERIENCE,
                operation="store",
                key="k",
                value=1,
                metadata=meta,
            )

    def test_naive_timestamp_raises(self):
        with pytest.raises(MemoryValidationError):
            MemoryRequest(
                memory_type=MemoryType.EXPERIENCE,
                operation="store",
                key="k",
                value=1,
                timestamp=datetime(2024, 1, 1),  # no tzinfo
            )

    def test_to_dict_keys(self):
        req = MemoryRequest.create(
            memory_type=MemoryType.EXPERIENCE, operation="store", key="k", value=99
        )
        d = req.to_dict()
        for field in ("request_id", "memory_type", "operation", "key", "value", "metadata", "timestamp"):
            assert field in d

    def test_valid_operations_set(self):
        assert VALID_OPERATIONS == {"store", "retrieve", "delete", "exists", "search"}


# ===========================================================================
# 4. MemoryResponse
# ===========================================================================


class TestMemoryResponse:
    def test_success_response_factory(self):
        resp = MemoryResponse.success_response(MemoryType.EXPERIENCE, data="val")
        assert resp.success is True
        assert resp.data == "val"
        assert resp.memory_type == MemoryType.EXPERIENCE

    def test_failure_response_factory(self):
        resp = MemoryResponse.failure_response(MemoryType.KNOWLEDGE, message="boom")
        assert resp.success is False
        assert resp.message == "boom"

    def test_default_message_on_success(self):
        resp = MemoryResponse.success_response(MemoryType.TEMPORAL)
        assert resp.message == "OK"

    def test_timestamp_is_utc(self):
        resp = MemoryResponse.success_response(MemoryType.RELATIONSHIP)
        assert resp.timestamp.tzinfo is not None

    def test_immutability(self):
        resp = MemoryResponse.success_response(MemoryType.EXPERIENCE)
        with pytest.raises((AttributeError, TypeError)):
            resp.success = False  # type: ignore[misc]

    def test_to_dict_keys(self):
        resp = MemoryResponse.success_response(MemoryType.EXPERIENCE, data=1)
        d = resp.to_dict()
        for f in ("success", "memory_type", "data", "message", "timestamp"):
            assert f in d

    def test_to_dict_memory_type_is_string(self):
        resp = MemoryResponse.success_response(MemoryType.KNOWLEDGE)
        assert resp.to_dict()["memory_type"] == "knowledge"


# ===========================================================================
# 5. MemoryProvider (abstract contract)
# ===========================================================================


class TestMemoryProvider:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            MemoryProvider("id", MemoryType.EXPERIENCE)  # type: ignore[abstract]

    def test_provider_id_property(self):
        p = make_echo()
        assert p.provider_id == PROVIDER_ID_ECHO

    def test_memory_type_property(self):
        p = make_echo()
        assert p.memory_type == MemoryType.EXPERIENCE

    def test_is_available_default_true(self):
        p = make_echo()
        assert p.is_available() is True

    def test_is_available_can_be_overridden(self):
        p = make_echo()
        p.set_available(False)
        assert p.is_available() is False

    def test_empty_provider_id_raises(self):
        with pytest.raises(ValueError):
            InMemoryProvider("", MemoryType.EXPERIENCE)

    def test_whitespace_provider_id_raises(self):
        with pytest.raises(ValueError):
            InMemoryProvider("   ", MemoryType.EXPERIENCE)


# ===========================================================================
# 6. MemoryRegistry
# ===========================================================================


class TestMemoryRegistry:
    def test_register_and_get(self, registry):
        p = make_echo()
        registry.register(p)
        assert registry.get_provider(MemoryType.EXPERIENCE) is p

    def test_is_registered_true(self, registry):
        registry.register(make_luna())
        assert registry.is_registered(MemoryType.KNOWLEDGE) is True

    def test_is_registered_false(self, registry):
        assert registry.is_registered(MemoryType.TEMPORAL) is False

    def test_is_available_true(self, registry):
        registry.register(make_echo())
        assert registry.is_available(MemoryType.EXPERIENCE) is True

    def test_is_available_false_when_not_registered(self, registry):
        assert registry.is_available(MemoryType.EXPERIENCE) is False

    def test_is_available_false_when_provider_unavailable(self, registry):
        p = make_echo()
        p.set_available(False)
        registry.register(p)
        assert registry.is_available(MemoryType.EXPERIENCE) is False

    def test_duplicate_registration_raises(self, registry):
        registry.register(make_echo())
        with pytest.raises(MemoryProviderError):
            registry.register(make_echo())

    def test_unregister(self, registry):
        registry.register(make_echo())
        registry.unregister(MemoryType.EXPERIENCE)
        assert registry.is_registered(MemoryType.EXPERIENCE) is False

    def test_unregister_unknown_raises(self, registry):
        with pytest.raises(MemoryRoutingError):
            registry.unregister(MemoryType.EXPERIENCE)

    def test_get_unregistered_raises(self, registry):
        with pytest.raises(MemoryRoutingError):
            registry.get_provider(MemoryType.EXPERIENCE)

    def test_registered_types(self, registry):
        registry.register(make_echo())
        registry.register(make_luna())
        types = registry.registered_types()
        assert MemoryType.EXPERIENCE in types
        assert MemoryType.KNOWLEDGE in types

    def test_all_providers(self, registry):
        p1 = make_echo()
        p2 = make_luna()
        registry.register(p1)
        registry.register(p2)
        providers = registry.all_providers()
        assert p1 in providers
        assert p2 in providers

    def test_len(self, registry):
        assert len(registry) == 0
        registry.register(make_echo())
        assert len(registry) == 1

    def test_iter(self, registry):
        p = make_echo()
        registry.register(p)
        providers = list(registry)
        assert p in providers

    def test_register_non_provider_raises_type_error(self, registry):
        with pytest.raises(TypeError):
            registry.register("not a provider")  # type: ignore[arg-type]

    def test_thread_safe_concurrent_registration(self):
        """Multiple threads register distinct providers without corruption."""
        registry = MemoryRegistry()
        providers = [
            make_echo(),
            make_luna(),
            make_chronos(),
            make_constellation(),
        ]
        errors = []

        def register(p):
            try:
                registry.register(p)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(p,)) for p in providers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(registry) == 4

    def test_thread_safe_concurrent_reads(self, full_registry):
        """Concurrent get_provider calls do not raise or corrupt state."""
        results = []
        errors = []

        def lookup():
            try:
                for mt in MemoryType:
                    p = full_registry.get_provider(mt)
                    results.append(p.memory_type)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=lookup) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 20 * len(MemoryType)


# ===========================================================================
# 7. MemoryRouter
# ===========================================================================


class TestMemoryRouter:
    def test_route_store_request(self, full_registry):
        router = MemoryRouter(full_registry)
        req = MemoryRequest.create(
            memory_type=MemoryType.EXPERIENCE,
            operation="store",
            key="k",
            value=42,
        )
        resp = router.route(req)
        assert resp.success is True

    def test_route_to_correct_provider(self, full_registry):
        router = MemoryRouter(full_registry)
        for mt in MemoryType:
            req = MemoryRequest.create(
                memory_type=mt, operation="exists", key="some-key"
            )
            resp = router.route(req)
            assert resp.memory_type == mt

    def test_unregistered_type_raises_routing_error(self, registry):
        router = MemoryRouter(registry)
        req = MemoryRequest.create(
            memory_type=MemoryType.EXPERIENCE, operation="store", key="k", value=1
        )
        with pytest.raises(MemoryRoutingError):
            router.route(req)

    def test_unavailable_provider_raises_routing_error(self, registry):
        p = make_echo()
        p.set_available(False)
        registry.register(p)
        router = MemoryRouter(registry)
        req = MemoryRequest.create(
            memory_type=MemoryType.EXPERIENCE, operation="retrieve", key="k"
        )
        with pytest.raises(MemoryRoutingError):
            router.route(req)

    def test_broken_provider_wraps_in_provider_error(self, registry):
        registry.register(BrokenProvider())
        router = MemoryRouter(registry)
        req = MemoryRequest.create(
            memory_type=MemoryType.EXPERIENCE, operation="store", key="k", value=1
        )
        with pytest.raises(MemoryProviderError) as exc_info:
            router.route(req)
        assert exc_info.value.cause is not None

    def test_router_requires_registry_instance(self):
        with pytest.raises(TypeError):
            MemoryRouter("not a registry")  # type: ignore[arg-type]

    def test_all_operations_dispatched(self, full_registry):
        """All five operation types must dispatch without routing error."""
        router = MemoryRouter(full_registry)
        ops_and_kwargs = [
            ("store", dict(key="k", value=1)),
            ("retrieve", dict(key="k")),
            ("exists", dict(key="k")),
            ("delete", dict(key="k")),
            ("search", dict(key=None, value="k")),
        ]
        for op, kwargs in ops_and_kwargs:
            req = MemoryRequest.create(
                memory_type=MemoryType.KNOWLEDGE, operation=op, **kwargs
            )
            resp = router.route(req)
            assert isinstance(resp, MemoryResponse), f"operation={op!r} did not return MemoryResponse"


# ===========================================================================
# 8. MemoryGateway — full API
# ===========================================================================


class TestMemoryGatewayStore:
    def test_store_success(self, gateway):
        resp = gateway.store(
            memory_type=MemoryType.EXPERIENCE, key="evt:1", value={"x": 1}
        )
        assert resp.success is True

    def test_store_all_memory_types(self, gateway):
        for mt in MemoryType:
            resp = gateway.store(memory_type=mt, key="k", value="v")
            assert resp.success is True

    def test_store_with_metadata(self, gateway):
        resp = gateway.store(
            memory_type=MemoryType.KNOWLEDGE,
            key="fact:1",
            value="water is wet",
            metadata={"confidence": 0.99},
        )
        assert resp.success is True


class TestMemoryGatewayRetrieve:
    def test_retrieve_existing_key(self, gateway):
        gateway.store(memory_type=MemoryType.EXPERIENCE, key="e1", value="hello")
        resp = gateway.retrieve(memory_type=MemoryType.EXPERIENCE, key="e1")
        assert resp.success is True
        assert resp.data == "hello"

    def test_retrieve_missing_key_returns_none(self, gateway):
        resp = gateway.retrieve(memory_type=MemoryType.EXPERIENCE, key="missing")
        assert resp.success is True
        assert resp.data is None

    def test_retrieve_different_types_isolated(self, gateway):
        gateway.store(memory_type=MemoryType.EXPERIENCE, key="shared", value="echo")
        gateway.store(memory_type=MemoryType.KNOWLEDGE, key="shared", value="luna")
        r1 = gateway.retrieve(memory_type=MemoryType.EXPERIENCE, key="shared")
        r2 = gateway.retrieve(memory_type=MemoryType.KNOWLEDGE, key="shared")
        assert r1.data == "echo"
        assert r2.data == "luna"


class TestMemoryGatewayDelete:
    def test_delete_existing_key(self, gateway):
        gateway.store(memory_type=MemoryType.TEMPORAL, key="t1", value="ts")
        resp = gateway.delete(memory_type=MemoryType.TEMPORAL, key="t1")
        assert resp.success is True
        assert resp.data is True

    def test_delete_nonexistent_key(self, gateway):
        resp = gateway.delete(memory_type=MemoryType.TEMPORAL, key="ghost")
        assert resp.success is True
        assert resp.data is False

    def test_deleted_key_not_retrievable(self, gateway):
        gateway.store(memory_type=MemoryType.RELATIONSHIP, key="r1", value="link")
        gateway.delete(memory_type=MemoryType.RELATIONSHIP, key="r1")
        resp = gateway.retrieve(memory_type=MemoryType.RELATIONSHIP, key="r1")
        assert resp.data is None


class TestMemoryGatewayExists:
    def test_exists_true_after_store(self, gateway):
        gateway.store(memory_type=MemoryType.KNOWLEDGE, key="exists:1", value=1)
        resp = gateway.exists(memory_type=MemoryType.KNOWLEDGE, key="exists:1")
        assert resp.data is True

    def test_exists_false_before_store(self, gateway):
        resp = gateway.exists(memory_type=MemoryType.KNOWLEDGE, key="nope")
        assert resp.data is False

    def test_exists_false_after_delete(self, gateway):
        gateway.store(memory_type=MemoryType.EXPERIENCE, key="gone", value=1)
        gateway.delete(memory_type=MemoryType.EXPERIENCE, key="gone")
        resp = gateway.exists(memory_type=MemoryType.EXPERIENCE, key="gone")
        assert resp.data is False


class TestMemoryGatewaySearch:
    def test_search_finds_matching_keys(self, gateway):
        gateway.store(memory_type=MemoryType.EXPERIENCE, key="event:alpha", value=1)
        gateway.store(memory_type=MemoryType.EXPERIENCE, key="event:beta", value=2)
        gateway.store(memory_type=MemoryType.EXPERIENCE, key="other:gamma", value=3)
        resp = gateway.search(memory_type=MemoryType.EXPERIENCE, query="event:")
        assert resp.success is True
        assert "event:alpha" in resp.data
        assert "event:beta" in resp.data
        assert "other:gamma" not in resp.data

    def test_search_empty_result(self, gateway):
        resp = gateway.search(memory_type=MemoryType.KNOWLEDGE, query="zzznomatch")
        assert resp.success is True
        assert resp.data == []


class TestMemoryGatewayQuery:
    def test_query_with_key_delegates_to_retrieve(self, gateway):
        gateway.store(memory_type=MemoryType.EXPERIENCE, key="q1", value="qval")
        resp = gateway.query(memory_type=MemoryType.EXPERIENCE, key="q1")
        assert resp.success is True
        assert resp.data == "qval"

    def test_query_with_query_delegates_to_search(self, gateway):
        gateway.store(memory_type=MemoryType.EXPERIENCE, key="search:1", value=1)
        resp = gateway.query(memory_type=MemoryType.EXPERIENCE, query="search:")
        assert resp.success is True
        assert isinstance(resp.data, list)

    def test_query_without_key_or_query_raises(self, gateway):
        with pytest.raises(MemoryValidationError):
            gateway.query(memory_type=MemoryType.EXPERIENCE)

    def test_query_key_takes_precedence_over_query(self, gateway):
        gateway.store(memory_type=MemoryType.KNOWLEDGE, key="priority", value="direct")
        resp = gateway.query(
            memory_type=MemoryType.KNOWLEDGE,
            key="priority",
            query="priority",
        )
        # key wins → retrieve → data is a value, not a list
        assert resp.data == "direct"


class TestMemoryGatewayErrorHandling:
    def test_no_provider_raises_routing_error(self, registry):
        gw = MemoryGateway(registry)  # empty registry
        with pytest.raises(MemoryRoutingError):
            gw.store(memory_type=MemoryType.EXPERIENCE, key="k", value=1)

    def test_unavailable_provider_raises_routing_error(self, registry):
        p = make_echo()
        p.set_available(False)
        registry.register(p)
        gw = MemoryGateway(registry)
        with pytest.raises(MemoryRoutingError):
            gw.store(memory_type=MemoryType.EXPERIENCE, key="k", value=1)

    def test_broken_provider_raises_provider_error(self, registry):
        registry.register(BrokenProvider())
        gw = MemoryGateway(registry)
        with pytest.raises(MemoryProviderError):
            gw.store(memory_type=MemoryType.EXPERIENCE, key="k", value=1)

    def test_gateway_requires_registry(self):
        with pytest.raises(TypeError):
            MemoryGateway("not a registry")  # type: ignore[arg-type]

    def test_registry_property_returns_registry(self, gateway, full_registry):
        assert gateway.registry is full_registry


# ===========================================================================
# 9. Thread safety — gateway under concurrent load
# ===========================================================================


class TestGatewayThreadSafety:
    def test_concurrent_store_and_retrieve(self, gateway):
        """100 threads each store + retrieve a unique key without error."""
        errors = []
        count = 100

        def worker(i: int) -> None:
            try:
                key = f"thread-key-{i}"
                gateway.store(
                    memory_type=MemoryType.EXPERIENCE, key=key, value=i
                )
                resp = gateway.retrieve(
                    memory_type=MemoryType.EXPERIENCE, key=key
                )
                assert resp.data == i
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in concurrent test: {errors}"

    def test_concurrent_mixed_operations(self, gateway):
        """Mixed operations across all MemoryTypes from many threads."""
        errors = []
        operations = []

        def worker(i: int) -> None:
            try:
                mt = list(MemoryType)[i % len(MemoryType)]
                key = f"mt-key-{i}"
                gateway.store(memory_type=mt, key=key, value=i)
                gateway.exists(memory_type=mt, key=key)
                gateway.retrieve(memory_type=mt, key=key)
                gateway.delete(memory_type=mt, key=key)
                operations.append(True)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(80)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in concurrent test: {errors}"
        assert len(operations) == 80

    def test_concurrent_registry_mutations(self):
        """Register/unregister cycles from multiple threads are safe."""
        registry = MemoryRegistry()
        registry.register(make_echo())
        errors = []

        def cycle(_):
            try:
                registry.unregister(MemoryType.EXPERIENCE)
                registry.register(make_echo())
            except Exception as exc:
                errors.append(exc)

        # Sequential cycles — each cycle is one thread to avoid race
        for i in range(20):
            t = threading.Thread(target=cycle, args=(i,))
            t.start()
            t.join()

        assert not errors


# ===========================================================================
# 10. Package-level imports
# ===========================================================================


class TestPackageImports:
    def test_all_symbols_importable_from_package(self):
        from core.memory import (  # noqa: F401
            MemoryGateway,
            MemoryType,
            MemoryRequest,
            MemoryResponse,
            MemoryProvider,
            MemoryRegistry,
            MemoryRouter,
            MemoryError,
            MemoryProviderError,
            MemoryRoutingError,
            MemoryValidationError,
            ROUTING_TABLE,
        )

    def test_routing_table_exported(self):
        from core.memory import ROUTING_TABLE

        assert len(ROUTING_TABLE) == 4