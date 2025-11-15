from __future__ import annotations

from typing import Mapping

from app.repositories.user_persona_repository import UserPersonaProvider


class MockUserPersonaRepository(UserPersonaProvider):
    """
    Very simple mock persona provider.

    You can seed it with a dict mapping user_id -> persona_text.
    Missing users return None.
    """

    def __init__(self, personas: Mapping[str, str] | None = None) -> None:
        self._personas = {k: v for k, v in (personas or {}).items()}

    async def get_persona(self, user_id: str | None) -> str | None:
        if not user_id:
            return None
        persona = self._personas.get(user_id)
        if persona:
            return persona.strip()
        return None

    async def get_persona_by_type(
        self, user_id: str | None, persona_type: str
    ) -> str | None:
        # Mock traktuje wszystkie persony identycznie niezależnie od typu.
        return await self.get_persona(user_id)

    async def save_persona(
        self,
        user_id: str,
        persona_text: str,
        persona_type: str = "default",
    ) -> None:
        self._personas[user_id] = persona_text
