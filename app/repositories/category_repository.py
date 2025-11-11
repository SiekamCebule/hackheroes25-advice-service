from __future__ import annotations

from typing import Any, Protocol, Sequence


class AdviceCategoryRepository(Protocol):
    async def get_all(self) -> Sequence[str]:
        raise NotImplementedError

    async def contains(self, category: str) -> bool:
        raise NotImplementedError


class SupabaseAdviceCategoryRepository(AdviceCategoryRepository):
    def __init__(self, client: Any) -> None:
        self._client = client

    async def get_all(self) -> Sequence[str]:
        raise NotImplementedError(
            "SupabaseAdviceCategoryRepository.get_all must be implemented with Supabase queries."
        )

    async def contains(self, category: str) -> bool:
        raise NotImplementedError(
            "SupabaseAdviceCategoryRepository.contains must be implemented with Supabase queries."
        )


class StaticAdviceCategoryRepository(AdviceCategoryRepository):
    def __init__(self, categories: Sequence[str]) -> None:
        self._categories = tuple({category.lower() for category in categories})

    async def get_all(self) -> Sequence[str]:
        return self._categories

    async def contains(self, category: str) -> bool:
        return category.lower() in self._categories
