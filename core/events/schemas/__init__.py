# core/events/schemas/__init__.py
"""
Event schema definitions for well-known POLARIS v5 system events.

This package provides canonical :class:`~core.events.event.EventType`
constants and typed payload dataclasses for every system-level event emitted
by the POLARIS runtime itself.  Subsystems should define their own schema
modules here as the system grows.
"""

from __future__ import annotations

from core.events.schemas.system import (
    SubsystemHealthChangedPayload,
    SubsystemLifecyclePayload,
    SubsystemRegisteredPayload,
    SubsystemUnregisteredPayload,
    SystemEvents,
)

__all__ = [
    "SubsystemHealthChangedPayload",
    "SubsystemLifecyclePayload",
    "SubsystemRegisteredPayload",
    "SubsystemUnregisteredPayload",
    "SystemEvents",
]