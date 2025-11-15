from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, cast

from app.models.advice import Advice, AdviceKind

if TYPE_CHECKING:  # pragma: no cover - optional dependency hint
    from supabase.client import AsyncClient  # type: ignore[import]
else:  # pragma: no cover - allow import without supabase installed
    AsyncClient = Any  # type: ignore

AdviceRow = dict[str, Any]
CategoryLinkRow = dict[str, Any]
QueryBuilder = Any


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
    _TABLE_NAME = "advices"

    def __init__(self, client: AsyncClient) -> None:  # type: ignore[misc]
        self._client = client

    async def get_all(self) -> Sequence[Advice]:
        return await self._fetch_advices()

    async def get_by_kind(self, kind: AdviceKind) -> Sequence[Advice]:
        return await self._fetch_advices(lambda query: query.eq("kind", kind.value))

    async def get_by_kind_and_containing_any_category(
        self,
        kind: AdviceKind,
        categories: Sequence[str],
    ) -> Sequence[Advice]:
        category_names = tuple({category.strip() for category in categories})
        if not category_names:
            return ()

        return await self._fetch_advices(
            lambda query: query.eq("kind", kind.value).filter(
                "advice_category_links.category.name",
                "in",
                self._build_supabase_in_filter(category_names),
            ),
            inner_join_categories=True,
        )

    async def _fetch_advices(
        self,
        apply_filters: Callable[[QueryBuilder], QueryBuilder] | None = None,
        inner_join_categories: bool = False,
    ) -> Sequence[Advice]:
        category_select = (
            "advice_category_links:advice_category_links"
            f"{'!inner' if inner_join_categories else ''}"
            "(category:advice_categories(name))"
        )

        base_query = self._client.table(self._TABLE_NAME).select(
            ",".join(
                [
                    "id",
                    "name",
                    "kind",
                    "description",
                    "link",
                    "image_url",
                    "author",
                    "embedding",
                    category_select,
                ]
            )
        )

        query = apply_filters(base_query) if apply_filters else base_query
        response = await query.execute()
        self._raise_on_error(response)
        # Supabase client stubs type `data` as a JSON-like union; at runtime we know
        # it is a list of dicts for this query, so we narrow the type here.
        raw_rows = getattr(response, "data", None) or []
        rows = cast(Sequence[AdviceRow], raw_rows)
        return tuple(self._map_advice(row) for row in rows)

    @staticmethod
    def _build_supabase_in_filter(values: Sequence[Any]) -> str:
        serialized_values = []
        for value in values:
            if isinstance(value, str):
                escaped = value.replace('"', r'\"')
                serialized_values.append(f'"{escaped}"')
            else:
                serialized_values.append(str(value))
        serialized = ",".join(serialized_values)
        return f"({serialized})"

    @staticmethod
    def _raise_on_error(response: Any) -> None:
        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"Supabase query failed: {error}")

    def _map_advice(self, row: AdviceRow) -> Advice:
        kind_value = row.get("kind")
        try:
            kind = AdviceKind(kind_value)
        except ValueError as err:
            raise RuntimeError(
                f"Unknown advice kind received from Supabase: {kind_value}") from err

        return Advice(
            id=row.get("id"),
            name=row.get("name", ""),
            kind=kind,
            description=row.get("description") or "",
            link_url=row.get("link"),
            image_url=row.get("image_url"),
            author=row.get("author"),
            categories=self._extract_categories(row),
            embedding=row.get("embedding"),
        )

    @staticmethod
    def _extract_categories(row: AdviceRow) -> Sequence[str]:
        links: Sequence[CategoryLinkRow] = row.get(
            "advice_category_links") or ()
        categories: list[str] = []
        for link in links:
            category = link.get("category") or {}
            name = category.get("name")
            if isinstance(name, str):
                categories.append(name)
        return tuple(categories)


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


class EmbeddingUpdatableAdviceRepository(SupabaseAdviceRepository):
    """
    Extension of SupabaseAdviceRepository that exposes an explicit method
    for updating cached advice embeddings in the database.
    """

    async def update_embedding(self, advice_id: int, embedding: Sequence[float]) -> None:
        await (
            self._client.table(self._TABLE_NAME)
            .update({"embedding": list(embedding)})
            .eq("id", advice_id)
            .execute()
        )
