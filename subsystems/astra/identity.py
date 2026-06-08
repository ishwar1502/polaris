# subsystems/astra/identity.py
"""
ASTRA v5 Identity Engine.

The Identity Engine is responsible for maintaining the user's stable
self-model.  It owns the :class:`~subsystems.astra.models.IdentityProfile`
and enforces the rule that short-term events do not equal identity change.

All mutations are thread-safe via an internal :class:`threading.RLock`.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from subsystems.astra.exceptions import IdentityNotFoundError, IdentityValidationError
from subsystems.astra.models import IdentityProfile

_logger = logging.getLogger(__name__)


class IdentityEngine:
    """Manages the user's stable core identity model.

    The engine maintains a single :class:`~subsystems.astra.models.IdentityProfile`
    and provides validated, thread-safe access to it.  Direct field writes
    are never permitted; all updates flow through :meth:`update_identity` to
    ensure versioning and validation are applied.

    Parameters
    ----------
    initial_profile:
        Optional pre-built profile to seed the engine.  If not provided,
        the engine starts with no profile; :meth:`update_identity` must be
        called before any :meth:`get_identity` call succeeds.
    """

    def __init__(self, initial_profile: IdentityProfile | None = None) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._profile: IdentityProfile | None = initial_profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_identity(self) -> IdentityProfile:
        """Return the current identity profile.

        Returns
        -------
        IdentityProfile
            A reference to the current profile.

        Raises
        ------
        IdentityNotFoundError
            If no identity profile has been created yet.
        """
        with self._lock:
            if self._profile is None:
                raise IdentityNotFoundError()
            return self._profile

    def has_identity(self) -> bool:
        """Return True if an identity profile exists."""
        with self._lock:
            return self._profile is not None

    def update_identity(self, updates: dict[str, Any]) -> IdentityProfile:
        """Apply validated updates to the identity profile.

        If no profile exists, one is created from the *updates* dict.
        The ``name`` field is required when creating a new profile.

        Parameters
        ----------
        updates:
            Dictionary of field-name → new-value pairs.  Accepted fields:
            ``name``, ``background``, ``education``, ``career_direction``,
            ``interests``, ``core_identity_tags``, ``metadata``.

        Returns
        -------
        IdentityProfile
            The updated profile.

        Raises
        ------
        IdentityValidationError
            If *updates* contains unknown fields or invalid values.
        """
        allowed_fields = {
            "name",
            "background",
            "education",
            "career_direction",
            "interests",
            "core_identity_tags",
            "metadata",
        }
        unknown = set(updates) - allowed_fields
        if unknown:
            raise IdentityValidationError(
                f"Unknown identity fields: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed_fields)}.",
                field=next(iter(unknown)),
            )

        with self._lock:
            if self._profile is None:
                # Creating a new profile
                if "name" not in updates or not str(updates["name"]).strip():
                    raise IdentityValidationError(
                        "'name' is required when creating a new identity profile.",
                        field="name",
                    )
                self._profile = IdentityProfile(name=str(updates["name"]))
                # Apply remaining fields
                remaining = {k: v for k, v in updates.items() if k != "name"}
                self._apply_updates(remaining)
            else:
                self._apply_updates(updates)

            _logger.debug(
                "Identity updated: fields=%s version=%d",
                list(updates.keys()),
                self._profile.version,
            )
            return self._profile

    def get_changed_fields(self, updates: dict[str, Any]) -> list[str]:
        """Return the list of fields that would change given *updates*.

        Useful for building event payloads before committing an update.

        Parameters
        ----------
        updates:
            Proposed update dictionary.

        Returns
        -------
        list[str]
            Fields whose values differ from the current profile.
        """
        with self._lock:
            if self._profile is None:
                return list(updates.keys())
            changed = []
            for field, new_val in updates.items():
                current = getattr(self._profile, field, None)
                if current != new_val:
                    changed.append(field)
            return changed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_updates(self, updates: dict[str, Any]) -> None:
        """Apply *updates* to the existing profile (lock must be held)."""
        assert self._profile is not None
        profile = self._profile

        if "name" in updates:
            val = str(updates["name"]).strip()
            if not val:
                raise IdentityValidationError(
                    "Identity name cannot be empty.", field="name"
                )
            profile.name = val

        if "background" in updates:
            profile.background = str(updates["background"])

        if "education" in updates:
            profile.education = str(updates["education"])

        if "career_direction" in updates:
            profile.career_direction = str(updates["career_direction"])

        if "interests" in updates:
            val = updates["interests"]
            if not isinstance(val, list):
                raise IdentityValidationError(
                    "'interests' must be a list of strings.", field="interests"
                )
            profile.interests = [str(i) for i in val]

        if "core_identity_tags" in updates:
            val = updates["core_identity_tags"]
            if not isinstance(val, list):
                raise IdentityValidationError(
                    "'core_identity_tags' must be a list of strings.",
                    field="core_identity_tags",
                )
            profile.core_identity_tags = [str(t) for t in val]

        if "metadata" in updates:
            val = updates["metadata"]
            if not isinstance(val, dict):
                raise IdentityValidationError(
                    "'metadata' must be a dictionary.", field="metadata"
                )
            profile.metadata.update(val)

        profile.bump_version()