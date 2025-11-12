from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence

if TYPE_CHECKING:  # pragma: no cover - optional dependency hint
    from supabase.client import AsyncClient  # type: ignore[import]
else:  # pragma: no cover - allow import without supabase installed
    AsyncClient = Any  # type: ignore


class AdviceCategoryRepository(Protocol):
    async def get_all(self) -> Sequence[str]:
        raise NotImplementedError

    async def contains(self, category: str) -> bool:
        raise NotImplementedError


class SupabaseAdviceCategoryRepository(AdviceCategoryRepository):
    _TABLE_NAME = "advice_categories"

    def __init__(self, client: AsyncClient) -> None:  # type: ignore[misc]
        self._client = client

    async def get_all(self) -> Sequence[str]:
        response = (
            self._client.table(self._TABLE_NAME)
            .select("name")
            .order("name")
        )
        response = await response.execute()
        self._raise_on_error(response)
        names = []
        for row in response.data or []:
            name = row.get("name")
            if isinstance(name, str):
                names.append(name)
        return tuple(names)

    async def contains(self, category: str) -> bool:
        normalized = category.lower()
        names = await self.get_all()
        return normalized in {name.lower() for name in names}

    @staticmethod
    def _raise_on_error(response: Any) -> None:
        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"Supabase query failed: {error}")


class StaticAdviceCategoryRepository(AdviceCategoryRepository):
    def __init__(self, categories: Sequence[str]) -> None:
        self._categories = tuple({category.lower() for category in categories})

    async def get_all(self) -> Sequence[str]:
        return self._categories

    async def contains(self, category: str) -> bool:
        return category.lower() in self._categories
