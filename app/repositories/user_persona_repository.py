from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from supabase.client import AsyncClient  # type: ignore[import]
else:  # pragma: no cover - allow import without supabase installed
    AsyncClient = Any  # type: ignore[misc]


logger = logging.getLogger(__name__)


class UserPersonaProvider(Protocol):
    async def get_persona(self, user_id: str | None) -> str | None:
        raise NotImplementedError

    async def get_persona_by_type(self, user_id: str | None, persona_type: str) -> str | None:
        raise NotImplementedError

    async def save_persona(
        self,
        user_id: str,
        persona_text: str,
        persona_type: str = "default",
    ) -> None:
        raise NotImplementedError


class NullUserPersonaProvider(UserPersonaProvider):
    async def get_persona(self, user_id: str | None) -> str | None:
        return None

    async def get_persona_by_type(self, user_id: str | None, persona_type: str) -> str | None:
        return None

    async def save_persona(
        self,
        user_id: str,
        persona_text: str,
        persona_type: str = "default",
    ) -> None:
        return None


class SupabaseUserPersonaRepository(UserPersonaProvider):
    _DEFAULT_TABLE = "user_personas"

    # type: ignore[misc]
    def __init__(self, client: AsyncClient, table_name: str | None = None) -> None:
        self._client = client
        self._table = table_name or self._DEFAULT_TABLE

    async def get_persona(self, user_id: str | None) -> str | None:
        if not user_id:
            return None
        try:
            response = (
                await self._client.table(self._table)
                .select("persona_text")
                .eq("user_id", user_id)
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - network layer guard
            logger.warning(
                "Failed to fetch persona for user %s: %s", user_id, exc)
            return None

        error = getattr(response, "error", None)
        if error:
            logger.warning(
                "Supabase persona query returned error for user %s: %s",
                user_id,
                error,
            )
            return None

        records = response.data or []
        if not records:
            return None

        persona = records[0].get("persona_text")
        if isinstance(persona, str) and persona.strip():
            return persona.strip()
        return None

    async def get_persona_by_type(self, user_id: str | None, persona_type: str) -> str | None:
        if not user_id:
            return None
        try:
            response = (
                await self._client.table(self._table)
                .select("persona_text")
                .eq("user_id", user_id)
                .eq("persona_type", persona_type)
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - network layer guard
            logger.warning(
                "Failed to fetch persona for user %s type %s: %s", user_id, persona_type, exc)
            return None

        error = getattr(response, "error", None)
        if error:
            logger.warning(
                "Database error fetching persona for user %s type %s: %s",
                user_id, persona_type, error)
            return None

        records = response.data or []
        if not records:
            return None

        persona = records[0].get("persona_text")
        if isinstance(persona, str) and persona.strip():
            return persona.strip()
        return None

    async def save_persona(
        self,
        user_id: str,
        persona_text: str,
        persona_type: str = "default",
    ) -> None:
        payload = {
            "user_id": user_id,
            "persona_type": persona_type,
            "persona_text": persona_text,
        }
        try:
            existing = (
                await self._client.table(self._table)
                .select("id")
                .eq("user_id", user_id)
                .eq("persona_type", persona_type)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - network layer guard
            logger.warning(
                "Failed to query persona before save for user %s type %s: %s",
                user_id,
                persona_type,
                exc,
            )
            return

        error = getattr(existing, "error", None)
        if error:
            logger.warning(
                "Supabase error while checking persona existence for user %s type %s: %s",
                user_id,
                persona_type,
                error,
            )
            return

        records = existing.data or []
        if records:
            record_id = records[0].get("id")
            query = self._client.table(self._table).update(
                {"persona_text": persona_text}
            )
            if record_id is not None:
                query = query.eq("id", record_id)
            else:  # fallback to user/type match if id missing
                query = query.eq("user_id", user_id).eq(
                    "persona_type", persona_type
                )
            await query.execute()
            return

        await self._client.table(self._table).insert(payload).execute()
