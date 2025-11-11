from __future__ import annotations

from typing import Any, Protocol, Sequence

from app.models.advice import Advice, AdviceKind


class AdviceRepository(Protocol):
    async def get_all(self) -> Sequence[Advice]:
        raise NotImplementedError

    async def get_by_kind(self, kind: AdviceKind) -> Sequence[Advice]:
        raise NotImplementedError

    async def get_by_kind_and_containing_any_category(
        self,
        kind: AdviceKind,
        categories: Sequence[str],
    ) -> Sequence[Advice]:
        raise NotImplementedError


class SupabaseAdviceRepository(AdviceRepository):
    def __init__(self, client: Any) -> None:
        self._client = client

    async def get_all(self) -> Sequence[Advice]:
        raise NotImplementedError(
            "SupabaseAdviceRepository.get_all must be implemented with Supabase queries."
        )

    async def get_by_kind(self, kind: AdviceKind) -> Sequence[Advice]:
        raise NotImplementedError(
            "SupabaseAdviceRepository.get_by_kind must be implemented with Supabase queries."
        )

    async def get_by_kind_and_containing_any_category(
        self,
        kind: AdviceKind,
        categories: Sequence[str],
    ) -> Sequence[Advice]:
        raise NotImplementedError(
            "SupabaseAdviceRepository.get_by_kind_and_containing_any_category must be "
            "implemented with Supabase queries."
        )


class InMemoryAdviceRepository(AdviceRepository):
    def __init__(self, advice_items: Sequence[Advice]) -> None:
        self._advice_items = tuple(advice_items)

    async def get_all(self) -> Sequence[Advice]:
        return self._advice_items

    async def get_by_kind(self, kind: AdviceKind) -> Sequence[Advice]:
        return tuple(item for item in self._advice_items if item.kind == kind)

    async def get_by_kind_and_containing_any_category(
        self,
        kind: AdviceKind,
        categories: Sequence[str],
    ) -> Sequence[Advice]:
        category_set = {category.lower() for category in categories}
        return tuple(
            item
            for item in self._advice_items
            if item.kind == kind
            and any(category.lower() in category_set for category in item.categories)
        )
